from typing import Optional
from collections import defaultdict
from tqdm import tqdm
import time
from datetime import timedelta
import os
import torch
import torch.nn as nn
import numpy as np
from ..models.inn import INN
from ..models.cfm import CFM, CFMwithTransformer
from ..models.transfermer import Transfermer
from ..models.transfusion import TransfusionAR, TransfusionParallel
from ..models.multiplicity import Classifier
from .preprocessing import build_preprocessing, MultiplicityPreproc, PreprocChain
from ..processes.base import Process, ProcessData
from .documenter import Documenter
from .balanced_sampler import BalancedSampler


class Model:
    """
    Class for training, evaluating, loading and saving models for density estimation or
    importance sampling.
    """

    def __init__(
        self,
        params: dict,
        verbose: bool,
        device: torch.device,
        model_path: str,
        dims_in: tuple[int, ...],
        dims_c: tuple[int, ...],
        state_dict_attrs: list[str],
    ):
        """
        Initializes the training class.

        Args:
            params: Dictionary with run parameters
            verbose: print
            device: pytorch device
            model_path: path to save trained models and checkpoints in
            input_data: tensors with training, validation and test input data
            cond_data: tensors with training, validation and test condition data
            state_dict_attrs: list of attribute whose state dicts will be stored
        """
        self.params = params
        self.device = device
        self.model_path = model_path
        self.verbose = verbose
        self.is_classifier = False
        model = params.get("model", "INN")
        try:
            self.model = eval(model)(dims_in[0], dims_c[0], params)
        except NameError:
            raise NameError("model not recognised. Use exact class name")
        if model == "Classifier":
            self.is_classifier = True
        self.model.to(device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params}")
        #if hasattr(torch, "compile"):
        #    print("  Compiling model")
        #    self.mode = torch.compile(self.model)
        self.state_dict_attrs = [*state_dict_attrs, "model", "optimizer"]
        self.losses = defaultdict(list)

    def init_data_loaders(
        self,
        input_data: tuple[torch.Tensor, ...],
        cond_data: tuple[torch.Tensor, ...],
    ):
        input_train, input_val, input_test = input_data
        cond_train, cond_val, cond_test = cond_data
        self.n_train_samples = len(input_train)
        bs = self.params.get("batch_size")
        bs_sample = self.params.get("batch_size_sample", 4*bs)
        balanced_sampling = self.params.get("balanced_sampling")
        if balanced_sampling is not None:
            assert len(input_train.shape) == 3
            n_one_hot = input_train.shape[1] - self.params["min_n_particles"] + 1
            train_loader_kwargs = {
                "sampler": BalancedSampler(cond_train[:, 0, -n_one_hot:], balanced_sampling)
            }
            val_loader_kwargs = {
                "sampler": BalancedSampler(cond_val[:, 0, -n_one_hot:], balanced_sampling)
            }
        else:
            train_loader_kwargs = {"shuffle": True}
            val_loader_kwargs = {"shuffle": False}

        self.train_loader = self.get_loader(
            input_train, cond_train, batch_size=bs, drop_last=True, **train_loader_kwargs
        )
        self.val_loader = self.get_loader(
            input_val, cond_val, batch_size=bs_sample, drop_last=True, **val_loader_kwargs
        )
        self.test_loader = self.get_loader(
            input_test, cond_test, batch_size=bs_sample, shuffle=False, drop_last=False
        )
        format_dim = lambda shape: shape[0] if len(shape) == 1 else (*shape,)
        print(f"  Input dimension: {format_dim(input_train.shape[1:])}")
        print(f"  Condition dimension: {format_dim(cond_train.shape[1:])}")

    def progress(self, iterable, **kwargs):
        """
        Shows a progress bar if verbose training is enabled

        Args:
            iterable: iterable object
            kwargs: keyword arguments passed on to tqdm if verbose
        Returns:
            Unchanged iterable if not verbose, otherwise wrapped by tqdm
        """
        if self.verbose:
            return tqdm(iterable, **kwargs)
        else:
            return iterable

    def print(self, text):
        """
        Chooses print function depending on verbosity setting

        Args:
            text: String to be printed
        """
        if self.verbose:
            tqdm.write(text)
        else:
            print(text, flush=True)

    def get_loader(
        self,
        input_data: torch.Tensor,
        cond_data: torch.Tensor,
        **loader_kwargs,
    ) -> torch.utils.data.DataLoader:
        """
        Constructs a DataSet and DataLoader for the given input and condition data

        Args:
            input_data: Tensor with input data, shape (n_events, dims_in)
            cond_data: Tensor with condition data, shape (n_events, dims_c)
            kwargs: keyword arguments passed to the data loader
        Returns:
            Constructed DataLoader
        """
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(input_data.float(), cond_data.float()),
            **loader_kwargs,
        )

    def init_optimizer(self):
        """
        Initialize optimizer and learning rate scheduling
        """
        optimizer = {
            "adam": torch.optim.Adam,
            "radam": torch.optim.RAdam,
        }[self.params.get("optimizer", "adam")]
        self.optimizer = optimizer(
            self.model.parameters(),
            lr=self.params.get("lr", 0.0002),
            betas=self.params.get("betas", [0.9, 0.999]),
            eps=self.params.get("eps", 1e-6),
            weight_decay=self.params.get("weight_decay", 0.0),
        )

        self.lr_sched_mode = self.params.get("lr_scheduler", "one_cycle_lr")
        if self.lr_sched_mode == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.params["lr_decay_epochs"],
                gamma=self.params["lr_decay_factor"],
            )
        elif self.lr_sched_mode == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                self.params.get("max_lr", self.params["lr"] * 10),
                epochs=self.params["epochs"],
                steps_per_epoch=len(self.train_loader),
            )
        elif self.lr_sched_mode == "cosine_annealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max= self.params["epochs"] * len(self.train_loader)
            )
        else:
            raise ValueError(f"Unknown LR scheduler {self.lr_sched_mode}")

    def begin_epoch(self):
        """
        Overload this function to perform some task at the beginning of each epoch.
        """
        pass

    def train(self):
        """
        Main training loop
        """
        best_val_loss = 1e20
        checkpoint_interval = self.params.get("checkpoint_interval")
        checkpoint_overwrite = self.params.get("checkpoint_overwrite", True)
        use_ema = self.params.get("use_ema", False)

        start_time = time.time()
        self.model.train()
        for epoch in self.progress(
            range(self.params["epochs"]), desc="  Epoch", leave=False, position=0
        ):
            self.begin_epoch()
            self.model.train()
            epoch_train_losses = defaultdict(int)
            loss_scale = 1 / len(self.train_loader)
            for xs, cs in self.progress(
                self.train_loader, desc="  Batch", leave=False, position=1
            ):
                self.optimizer.zero_grad()
                loss, loss_terms = self.model.batch_loss(
                    xs, cs, 1 / self.n_train_samples
                )
                loss.backward()
                self.optimizer.step()
                if self.lr_sched_mode == "one_cycle" or self.lr_sched_mode == "cosine_annealing":
                    self.scheduler.step()
                for name, loss in loss_terms.items():
                    epoch_train_losses[name] += loss * loss_scale
                if use_ema:
                    self.model.ema.update()
            if self.lr_sched_mode == "step":
                self.scheduler.step()

            for name, loss in epoch_train_losses.items():
                self.losses[f"train_{name}"].append(loss)
            for name, loss in self.dataset_loss(self.val_loader).items():
                self.losses[f"val_{name}"].append(loss)
            if epoch < 20:
                last_20_val_losses = self.losses["val_loss"]
            else:
                last_20_val_losses = self.losses["val_loss"][-20:]
            self.losses["val_loss_movingAvg"].append(torch.tensor(last_20_val_losses).mean().item())

            self.losses["lr"].append(self.optimizer.param_groups[0]["lr"])

            if self.losses["val_loss"][-1] < best_val_loss:
                best_val_loss = self.losses["val_loss"][-1]
                self.save("best")
            if (
                checkpoint_interval is not None
                and (epoch + 1) % checkpoint_interval == 0
            ):
                self.save("final" if checkpoint_overwrite else f"epoch_{epoch}")

            self.print(
                f"  Epoch {epoch}: "
                + ", ".join(
                    [
                        f"{name} = {loss[-1]:{'.2e' if name == 'lr' else '.6f'}}"
                        for name, loss in self.losses.items()
                    ]
                )
                + f", time = {timedelta(seconds=round(time.time() - start_time))} seconds"
            )

        self.save("final")
        time_diff = timedelta(seconds=round(time.time() - start_time))
        print(f"  Training completed after {time_diff}")

    def dataset_loss(self, loader: torch.utils.data.DataLoader) -> dict:
        """
        Computes the losses (without gradients) for the given data loader

        Args:
            loader: data loader
        Returns:
            Dictionary with loss terms averaged over all samples
        """
        self.model.eval()
        n_total = 0
        total_losses = defaultdict(list)
        with torch.no_grad():
            for xs, cs in self.progress(loader, desc="  Batch", leave=False):
                n_samples = xs.shape[0]
                n_total += n_samples
                _, losses = self.model.batch_loss(
                    xs, cs, kl_scale=1 / self.n_train_samples
                )
                for name, loss in losses.items():
                    total_losses[name].append(loss * n_samples)
        return {name: sum(losses) / n_total for name, losses in total_losses.items()}

    def predict_loglikelihood(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Predict the log likelihood for each event from the given dataloader

        Returns:
            tensor with log likelihoods, shape (n_events, )
        """
        self.model.eval()
        bayesian_samples = self.params["bayesian_samples"] if self.model.bayesian else 1
        with torch.no_grad():
            all_loglikelihoods = []
            for i in range(bayesian_samples):
                logp_batches = []
                if self.model.bayesian:
                    self.model.reset_random_state()
                for xs, cs in self.progress(
                    loader,
                    desc="  Generating",
                    leave=False,
                    initial=i * len(loader),
                    total=bayesian_samples * len(loader),
                ):
                    logp_batches.append(self.model.log_prob(xs, cs))
                all_loglikelihoods.append(torch.cat(logp_batches, dim=0))
            all_loglikelihoods = torch.stack(all_loglikelihoods, dim=0)
            if self.model.bayesian:
                return all_loglikelihoods
            else:
                return all_loglikelihoods[0]

    def predict(self, loader=None) -> torch.Tensor:
        """
        Predict one sample for each event from the test data

        Returns:
            tensor with samples, shape (n_events, dims_in)
        """
        self.model.eval()
        bayesian_samples = self.params.get("bayesian_samples", 20) if self.model.bayesian else 1
        if loader is None:
            loader = self.test_loader
        with torch.no_grad():
            all_samples = []
            for i in range(bayesian_samples):
                t0 = time.time()
                data_batches = []
                if self.model.bayesian:
                    self.model.reset_random_state()
                for xs, cs in self.progress(
                    loader,
                    desc="  Generating",
                    leave=False,
                    initial=i * len(loader),
                    total=bayesian_samples * len(loader),
                ):
                    while True:
                        try:
                            data_batches.append(self.model.sample(cs)[0])
                            break
                        except AssertionError:
                            print("Batch failed, repeating")
                all_samples.append(torch.cat(data_batches, dim=0))
                if self.model.bayesian:
                    print(f"Finished bayesian sample {i} in {time.time() - t0}", flush=True)
            all_samples = torch.stack(all_samples, dim=0)
            if self.model.bayesian:
                return all_samples
            else:
                return all_samples[0]

    def predict_distribution(self, loader=None) -> torch.Tensor:
        """
        Predict multiple samples for a part of the test dataset

        Returns:
            tensor with samples, shape (n_events, n_samples, dims_in)
        """
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        bayesian_samples = self.params.get("bayesian_samples", 20) if self.model.bayesian else 1
        max_batches = min(len(loader), self.params.get("max_dist_batches", 2))
        samples_per_event = self.params.get("dist_samples_per_event", 60)
        with torch.no_grad():
            all_samples = []
            for j in range(bayesian_samples):
                offset = j * max_batches * samples_per_event
                if self.model.bayesian:
                    self.model.reset_random_state()
                for i, (xs, cs) in enumerate(loader):
                    if i == max_batches:
                        break
                    data_batches = []
                    for _ in self.progress(
                        range(samples_per_event),
                        desc="  Generating",
                        leave=False,
                        initial=offset + i * samples_per_event,
                        total=bayesian_samples * max_batches * samples_per_event,
                    ):
                        while True:
                            try:
                                data_batches.append(self.model.sample(cs)[0])
                                break
                            except AssertionError:
                                print("Batch failed, repeating")
                    all_samples.append(torch.stack(data_batches, dim=1))
            all_samples = torch.cat(all_samples, dim=0)
            if self.model.bayesian:
                return all_samples.reshape(
                    bayesian_samples,
                    len(all_samples) // bayesian_samples,
                    *all_samples.shape[1:],
                )
            else:
                return all_samples

    def predict_classes(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            mult_true = []
            mult_predict = []
            for xs, cs in self.progress(self.test_loader, desc="  Predicting", leave=False):
                mult_true.append(xs)
                mult_predict.append(self.model.probs(cs))
        return torch.cat(mult_true, dim=0), torch.cat(mult_predict, dim=0)

    def save(self, name: str):
        """
        Saves the model, preprocessing, optimizer and losses.

        Args:
            name: File name for the model (without path and extension)
        """
        file = os.path.join(self.model_path, f"{name}.pth")
        torch.save(
            {
                **{
                    attr: getattr(self, attr).state_dict()
                    for attr in self.state_dict_attrs
                },
                "losses": self.losses,
            },
            file,
        )

    def load(self, name: str):
        """
        Loads the model, preprocessing, optimizer and losses.

        Args:
            name: File name for the model (without path and extension)
        """
        file = os.path.join(self.model_path, f"{name}.pth")
        state_dicts = torch.load(file, map_location=self.device)
        for attr in self.state_dict_attrs:
            try:
                getattr(self, attr).load_state_dict(state_dicts[attr])
            except AttributeError:
                pass
        self.losses = state_dicts["losses"]


class TransferFunction(Model):
    def __init__(
        self,
        params: dict,
        verbose: bool,
        device: torch.device,
        model_path: str,
        process: Process
    ):
        params["mass_mask"] = [m is None for m in process.reco_masses()]
        self.reco_pp = build_preprocessing(params["reco_preprocessing"], process.reco_masses())
        self.hard_pp = build_preprocessing(params["hard_preprocessing"], process.hard_masses())
        self.hard_pp.to(device)
        self.reco_pp.to(device)
        if "alpha_preprocessing" in params:
            self.alpha_pp = build_preprocessing(params["alpha_preprocessing"])
            self.alpha_pp.to(device)
        else:
            self.alpha_pp = None
        if "multiplicity_model" in params:
            self.multiplicity_model = load_model(
                params["multiplicity_model"], process, device
            )
            assert (
                isinstance(self.multiplicity_model.reco_pp, PreprocChain) and
                isinstance(self.multiplicity_model.reco_pp.trafos[0], MultiplicityPreproc)
            )

            self.multiplicity_pp = self.multiplicity_model.reco_pp
            self.multiplicity_classes = self.multiplicity_pp.trafos[0].n_classes
        else:
            self.multiplicity_model = None
            self.multiplicity_classes = 0

        super().__init__(
            params,
            verbose,
            device,
            model_path,
            self.reco_pp.output_shape,
            (
                *self.hard_pp.output_shape[:-1],
                self.hard_pp.output_shape[-1]
                + (self.alpha_pp.output_shape[-1] if self.alpha_pp is not None else 0)
                + self.multiplicity_classes,
            ),
            state_dict_attrs=["reco_pp", "hard_pp"]
            + ([] if self.alpha_pp is None else ["alpha_pp"]),
        )

    def prepare_condition(
        self, x_hard: torch.Tensor, x_reco: torch.Tensor, alpha: torch.Tensor
    ):
        features = [self.hard_pp(x_hard)]
        if len(features[0].shape) == 3:
            unsqueeze = lambda f: f[:,None,:].expand(-1, features[0].shape[1], -1)
        else:
            unsqueeze = lambda f: f
        if self.alpha_pp is not None:
            features.append(unsqueeze(self.alpha_pp(alpha)))
        if self.multiplicity_model is not None:
            features.append(unsqueeze(self.multiplicity_pp(x_reco)))
        return torch.cat(features, dim=-1)

    def init_data_loaders(self, data: tuple[ProcessData, ...]):
        data_train, _, _ = data
        self.reco_pp.init_normalization(data_train.x_reco)
        input_data = tuple(self.reco_pp(subset.x_reco) for subset in data)
        self.hard_pp.init_normalization(data_train.x_hard)
        if self.alpha_pp is not None:
            self.alpha_pp.init_normalization(data_train.alpha)
        cond_data = tuple(
            self.prepare_condition(subset.x_hard, subset.x_reco, subset.alpha)
            for subset in data
        )
        super().init_data_loaders(input_data, cond_data)

    def predict(self, loader=None) -> torch.Tensor:
        """
        Predict one sample for each event from the given loader

        Returns:
            tensor with samples, shape (n_events, dims_in)
        """
        samples = super().predict(loader)
        if self.model.bayesian:
            samples_pp = [self.reco_pp(sample, rev=True, jac=False) for sample in samples]
            samples_pp = torch.stack(samples_pp, dim=0)
        else:
            samples_pp = self.reco_pp(samples, rev=True, jac=False)
        return samples_pp

    def predict_distribution(self, loader=None) -> torch.Tensor:
        """
        Predict multiple samples for a part of the test dataset

        Returns:
            tensor with samples, shape (n_events, n_samples, dims_in)
        """
        samples = super().predict_distribution(loader)
        if self.model.bayesian:
            samples_pp = [
                self.reco_pp(
                    sample.reshape(-1, *sample.shape[2:]),
                    rev=True,
                    jac=False,
                    batch_size=1000,
                ) for sample in samples
            ]
            samples_pp = [
                sample.reshape(*samples[0].shape[:2], *sample.shape[1:])
                for sample in samples_pp
            ]
            samples_pp = torch.stack(samples_pp, dim=0)
        else:
            samples_pp = self.reco_pp(
                samples.reshape(-1, *samples.shape[2:]),
                rev=True,
                jac=False,
                batch_size=1000,
            )
            samples_pp = samples_pp.reshape(*samples.shape[:2], *samples_pp.shape[1:])
        return samples_pp

    def estimate_transfer(
        self,
        x_reco: torch.Tensor,
        x_hard: torch.Tensor,
        alpha: torch.Tensor = None,
        event_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns the estimated transfer function value for the given events.

        Args:
            x_reco: Reco-level momenta, shape (n_events, n_reco_particles, 4)
            x_hard: Hard-scattering momenta, shape (n_events, n_hard_particles, 4)
            alpha: Theory parameters, shape (n_events, n_parameters)
            event_type: Type of the event, e.g. LO or NLO, as a one-hot encoded tensor,
                        shape (n_events, n_types), optional
        Returns:
            Tensor with estimated values of the transfer function, shape (n_events, )
        """
        self.model.eval()
        with torch.no_grad():
            c = self.prepare_condition(x_hard, x_reco, alpha)
            x, pp_jac = self.reco_pp(x_reco, jac=True)
            log_prob = self.model.log_prob(x.float(), c.float())
            if self.multiplicity_model is not None:
                mult_prob = self.multiplicity_model.estimate_transfer(
                    x_reco, x_hard, alpha, event_type
                ).double()
            else:
                mult_prob = 0
            return torch.exp(log_prob.double() - pp_jac.double() + mult_prob)

    def sample_events(
        self,
        x_hard: torch.Tensor,
        alpha: torch.Tensor,
        event_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Samples one reco-level event for each given hard-scattering level event

        Args:
            x_hard: Hard-scattering momenta, shape (n_events, n_hard_particles, 4)
            alpha: Theory parameters, shape (n_events, n_parameters)
            event_type: Type of the event, e.g. LO or NLO, as a one-hot encoded tensor,
                        shape (n_events, n_types), optional
        Returns:
            Tensor with reco-level momenta, shape (n_events, n_reco_particles, 4)
        """
        with torch.no_grad():
            if self.multiplicity_model is not None:
                x_reco = self.multiplicity_model.sample_events(x_hard, alpha, event_type)
            else:
                x_reco = None
            c = self.prepare_condition(x_hard, x_reco, alpha)
            x, _ = self.model.sample(c.float())
            return self.reco_pp(x, rev=True)


class ImportanceSampler(Model):
    def __init__(
        self,
        params: dict,
        verbose: bool,
        device: torch.device,
        model_path: str,
        process: Process
    ):
        self.process = process
        self.hard_pp = build_preprocessing(params["hard_preprocessing"], process.hard_masses())
        self.reco_pp = build_preprocessing(params["reco_preprocessing"], process.reco_masses())
        self.alpha_pp = build_preprocessing(params["alpha_preprocessing"])
        self.hard_pp.to(device)
        self.reco_pp.to(device)
        self.alpha_pp.to(device)
        self.latent_dimension = self.hard_pp.output_shape[0] #TODO: do this properly!!!!!

        super().__init__(
            params,
            verbose,
            device,
            model_path,
            self.hard_pp.output_shape,
            (
                *self.reco_pp.output_shape[:-1],
                self.reco_pp.output_shape[-1] + self.alpha_pp.output_shape[-1],
            ),
            state_dict_attrs=["hard_pp", "reco_pp", "alpha_pp"],
        )

    def init_data_loaders(self, data: tuple[ProcessData, ...]):
        sample_transfer_function = self.params.get("sample_transfer_function")
        if sample_transfer_function is None:
            self.transfer_function = None
            reco_data = [subset.x_reco for subset in data]
        else:
            self.transfer_function = load_model(sample_transfer_function, self.process, self.device)
            self.transfer_function.model.eval()
            reco_data = [self.regenerate_reco_data(subset) for subset in data]
            if self.transfer_function.model.bayesian:
                for layer in self.transfer_function.model.bayesian_layers:
                    layer.map = True
                print("Using bayesian transfer function with MAP to generate data")

        self.data_train, _, _ = data
        self.hard_pp.init_normalization(self.data_train.x_hard)
        input_data = tuple(self.hard_pp(subset.x_hard) for subset in data)
        self.reco_pp.init_normalization(self.data_train.x_reco)
        self.alpha_pp.init_normalization(self.data_train.alpha)
        cond_data = tuple(
            torch.cat((self.reco_pp(x_reco), self.alpha_pp(subset.alpha)), dim=1)
            for x_reco, subset in zip(reco_data, data)
        )
        super().init_data_loaders(input_data, cond_data)

    def regenerate_reco_data(self, data: ProcessData):
        batch_size = self.params["batch_size"]
        batch_size_sample = self.params.get("batch_size_sample", 8*batch_size)
        x_reco_new = []
        for x_hard, alpha in self.progress(
            zip(
                data.x_hard.split(batch_size_sample, dim=0),
                data.alpha.split(batch_size_sample, dim=0),
            ),
            desc="  Sampling TF",
            leave=False,
            total=(len(data.x_hard) + batch_size_sample - 1) // batch_size_sample,
        ):
            while True:
                try:
                    x_reco_batch = self.transfer_function.sample_events(x_hard, alpha)
                    if x_reco_batch.isnan().any():
                        print("Generated nan, repeating batch")
                    else:
                        break
                except AssertionError:
                    print("Batch failed")

            x_reco_new.append(x_reco_batch)
        return torch.cat(x_reco_new, dim=0)

    def begin_epoch(self):
        if self.transfer_function is not None:
            x_reco_new = self.regenerate_reco_data(self.data_train)
            input_train = self.hard_pp(self.data_train.x_hard)
            cond_train = torch.cat(
                (self.reco_pp(x_reco_new), self.alpha_pp(self.data_train.alpha)), dim=1
            )
            self.train_loader = self.get_loader(
                input_train, cond_train, batch_size=self.params.get("batch_size"), shuffle=True, drop_last=True
            )

    def predict(self, loader=None) -> torch.Tensor:
        """
        Predict one sample for each event in the test dataset

        Returns:
            tensor with samples, shape (n_events, dims_in)
        """
        samples = super().predict(loader)
        samples_pp = self.hard_pp(
            samples.reshape(-1, samples.shape[-1]), rev=True, jac=False, batch_size=1000
        )
        return samples_pp.reshape(*samples.shape[:-1], *samples_pp.shape[1:])

    def predict_distribution(self, loader=None) -> torch.Tensor:
        """
        Predict multiple samples for a part of the test dataset

        Returns:
            tensor with samples, shape (n_events, n_samples, dims_in)
        """
        samples = super().predict_distribution(loader)
        samples_pp = self.hard_pp(
            samples.reshape(-1, *samples.shape[2:]),
            rev=True,
            jac=False,
            batch_size=1000,
        )
        return samples_pp.reshape(*samples.shape[:2], *samples_pp.shape[1:])

    def sample_events(
        self,
        n_samples: int,
        x_reco: torch.Tensor,
        alpha: torch.Tensor,
        event_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples hard-scattering level events for the given reco-level events.

        Args:
            n_samples: number of hard-scattering events to sample for each reco-level event
            x_reco: Reco-level momenta, shape (n_events or 1, n_reco_particles, 4)
            alpha: Theory parameters, shape (n_events or 1, n_parameters)
            event_type: Type of the event, e.g. LO or NLO, as a one-hot encoded tensor,
                        shape (n_events, n_types), optional
        Returns:
            Tensor with hard-scattering momenta, shape (n_samples, n_events, n_hard_particles, 4)
            Tensor with phase space densities, shape (n_samples, n_events)
        """
        with torch.no_grad():
            n_events = max(len(x_reco), len(alpha))
            c = torch.cat((
                self.reco_pp(x_reco).expand(n_events, -1),
                self.alpha_pp(alpha).expand(n_events, -1)
            ), dim=1).float()
            x, log_prob = self.model.sample_with_probs(c)
            x_hard, jac = self.hard_pp(x, rev=True, jac=True)
            return x_hard.reshape(n_samples, -1, *x_hard.shape[1:]), torch.exp(
                log_prob - jac
            )

    def transform_hypercube(
        self,
        r: torch.Tensor,
        x_reco: torch.Tensor,
        alpha: torch.Tensor,
        event_type: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples hard-scattering level events for the given reco-level events.

        Args:
            r: points on the the unit hypercube, shape (n_samples, dims_in)
            x_reco: Reco-level momenta, shape (n_samples, n_reco_particles, 4)
            alpha: Theory parameters, shape (n_samples, n_parameters)
            event_type: Type of the event, e.g. LO or NLO, as a one-hot encoded tensor,
                        shape (n_samples, n_types), optional
        Returns:
            Tensor with hard-scattering momenta, shape (n_samples, n_hard_particles, 4)
            Tensor with jacobians, shape (n_samples, )
        """
        with torch.no_grad():
            c = torch.cat((self.reco_pp(x_reco), self.alpha_pp(alpha)), dim=1).float()
            x, model_jac = self.model.transform_hypercube(r, c)
            x_hard, pp_jac = self.hard_pp(x, rev=True, jac=True)
            return x_hard, torch.exp(model_jac + pp_jac)


class Efficiency(Model):
    def __init__(
        self,
        params: dict,
        verbose: bool,
        device: torch.device,
        model_path: str,
        process: Process
    ):
        self.hard_pp = build_preprocessing(params["hard_preprocessing"], process.hard_masses())
        self.hard_pp.to(device)
        if "alpha_preprocessing" in params:
            self.alpha_pp = build_preprocessing(params["alpha_preprocessing"])
            self.alpha_pp.to(device)
        else:
            self.alpha_pp = None

        super().__init__(
            params,
            verbose,
            device,
            model_path,
            (1, ),
            (
                *self.hard_pp.output_shape[:-1],
                self.hard_pp.output_shape[-1]
                + (self.alpha_pp.output_shape[-1] if self.alpha_pp is not None else 0)
            ),
            state_dict_attrs=["hard_pp"]
            + ([] if self.alpha_pp is None else ["alpha_pp"]),
        )

    def prepare_condition(
        self, x_hard: torch.Tensor, alpha: torch.Tensor
    ):
        features = [self.hard_pp(x_hard)]
        if len(features[0].shape) == 3:
            unsqueeze = lambda f: f[:,None,:].expand(-1, features[0].shape[1], -1)
        else:
            unsqueeze = lambda f: f
        if self.alpha_pp is not None:
            features.append(unsqueeze(self.alpha_pp(alpha)))
        return torch.cat(features, dim=-1)

    def init_data_loaders(self, data: tuple[ProcessData, ...]):
        data_train, _, _ = data
        input_data = tuple(subset.accepted[:,None] for subset in data)
        self.hard_pp.init_normalization(data_train.x_hard)
        if self.alpha_pp is not None:
            self.alpha_pp.init_normalization(data_train.alpha)
        cond_data = tuple(
            self.prepare_condition(subset.x_hard, subset.alpha)
            for subset in data
        )
        super().init_data_loaders(input_data, cond_data)

    def estimate_efficiency(
        self,
        x_hard: torch.Tensor,
        alpha: torch.Tensor = None,
        event_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns the estimated efficiency for the given events.

        Args:
            x_hard: Hard-scattering momenta, shape (n_events, n_hard_particles, 4)
            alpha: Theory parameters, shape (n_events, n_parameters)
            event_type: Type of the event, e.g. LO or NLO, as a one-hot encoded tensor,
                        shape (n_events, n_types), optional
        Returns:
            Tensor with estimated values of the transfer function, shape (n_events, )
        """
        with torch.no_grad():
            c = self.prepare_condition(x_hard, alpha)
            return self.model.probs(c.float())


def load_model(run_name: str, process: Process, device: torch.device) -> Model:
    doc, params = Documenter.from_saved_run(run_name, read_only=True)
    model_path = doc.get_file("model", False)
    if params["type"] == "transfer_function":
        model = TransferFunction(params, False, device, model_path, process)
    elif params["type"] == "importance_sampler":
        model = ImportanceSampler(params, False, device, model_path, process)
    elif params["type"] == "efficiency":
        model = Efficiency(params, False, device, model_path, process)
    else:
        raise ValueError(f"Unknown model type '{params['type']}'")
    model.load("final")
    return model
