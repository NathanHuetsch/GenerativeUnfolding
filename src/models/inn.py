from typing import Type, Callable, Union
import math
import torch
import torch.nn as nn
import numpy as np
import FrEIA.framework as ff
import FrEIA.modules as fm

from .spline_blocks import RationalQuadraticSplineBlock
from .layers import VBLinear


class Subnet(nn.Module):
    """
    Standard MLP or bayesian network to be used as a trainable subnet in INNs
    """

    def __init__(
        self,
        num_layers: int,
        size_in: int,
        size_out: int,
        internal_size: int,
        dropout: float = 0.0,
        layer_class: Type = nn.Linear,
        layer_args: dict = {},
    ):
        """
        Constructs the subnet.

        Args:
            num_layers: number of layers
            size_in: input size of the subnet
            size: output size of the subnet
            internal_size: hidden size of the subnet
            dropout: dropout chance of the subnet
            layer_class: class to construct the linear layers
            layer_args: keyword arguments to pass to the linear layer
        """
        super().__init__()
        if num_layers < 1:
            raise (ValueError("Subnet size has to be 1 or greater"))
        self.layer_list = []
        for n in range(num_layers):
            input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in
            if n == num_layers - 1:
                output_dim = size_out

            self.layer_list.append(layer_class(input_dim, output_dim, **layer_args))

            if n < num_layers - 1:
                if dropout > 0:
                    self.layer_list.append(nn.Dropout(p=dropout))
                self.layer_list.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layer_list)

        for name, param in self.layer_list[-1].named_parameters():
            if "logsig2_w" not in name:
                param.data *= 0.02

    def forward(self, x):
        return self.layers(x)


class INN(nn.Module):
    """
    Class implementing a standard conditional INN
    """

    def __init__(self, dims_in: int, dims_c: int, params: dict):
        """
        Initializes and builds the conditional INN

        Args:
            dims_in: dimension of input
            dims_c: dimension of condition
            params: dictionary with architecture/hyperparameters
        """
        super().__init__()
        self.params = params
        self.dims_in = dims_in
        self.dims_c = dims_c
        self.bayesian = params.get("bayesian", False)
        self.bayesian_transfer = False
        if self.bayesian:
            self.bayesian_samples = params.get("bayesian_samples", 20)
            self.bayesian_layers = []
        self.latent_space = self.params.get("latent_space", "gaussian")
        self.build_inn()

    def get_constructor_func(self) -> Callable[[int, int], nn.Module]:
        """
        Returns a function that constructs a subnetwork with the given parameters

        Returns:
            Function that returns a subnet with input and output size as parameters
        """
        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in self.params:
            layer_args["prior_prec"] = self.params["prior_prec"]
        if "std_init" in self.params:
            layer_args["std_init"] = self.params["std_init"]

        def func(x_in: int, x_out: int) -> nn.Module:
            subnet = Subnet(
                self.params.get("layers_per_block", 3),
                x_in,
                x_out,
                internal_size=self.params.get("internal_size"),
                dropout=self.params.get("dropout", 0.0),
                layer_class=layer_class,
                layer_args=layer_args,
            )
            if self.bayesian:
                self.bayesian_layers.extend(
                    layer for layer in subnet.layer_list if isinstance(layer, VBLinear)
                )
            return subnet

        return func

    def get_coupling_block(self) -> tuple[Type, dict]:
        """
        Returns the class and keyword arguments for different coupling block types
        """
        constructor_fct = self.get_constructor_func()
        permute_soft = self.params.get("permute_soft", False)
        coupling_type = self.params.get("coupling_type", "affine")

        if coupling_type == "affine":
            if self.latent_space == "uniform":
                raise ValueError("Affine couplings only support gaussian latent space")
            CouplingBlock = fm.AllInOneBlock
            block_kwargs = {
                "affine_clamping": self.params.get("clamping", 5.0),
                "subnet_constructor": constructor_fct,
                "global_affine_init": 0.92,
                "permute_soft": permute_soft,
            }
        elif coupling_type == "rational_quadratic":
            if self.latent_space == "gaussian":
                upper_bound = self.params.get("bounds", 10)
                lower_bound = -upper_bound
            elif self.latent_space == "uniform":
                lower_bound = 0
                upper_bound = 1
                if permute_soft:
                    raise ValueError(
                        "Soft permutations not supported for uniform latent space"
                    )

            CouplingBlock = RationalQuadraticSplineBlock
            bounds = self.params.get("spline_bounds", 10)
            block_kwargs = {
                "num_bins": self.params.get("num_bins", 10),
                "subnet_constructor": constructor_fct,
                "left": lower_bound,
                "right": upper_bound,
                "bottom": lower_bound,
                "top": upper_bound,
                "permute_soft": permute_soft,
            }
        else:
            raise ValueError(f"Unknown coupling block type {coupling_type}")

        return CouplingBlock, block_kwargs

    def build_inn(self):
        """
        Construct the INN
        """
        self.inn = ff.SequenceINN(self.dims_in)
        CouplingBlock, block_kwargs = self.get_coupling_block()
        for i in range(self.params.get("n_blocks", 10)):
            self.inn.append(
                CouplingBlock, cond=0, cond_shape=(self.dims_c,), **block_kwargs
            )

    def latent_log_prob(self, z: torch.Tensor) -> Union[torch.Tensor, float]:
        """
        Returns the log probability for a tensor in latent space

        Args:
            z: latent space tensor, shape (n_events, dims_in)
        Returns:
            log probabilities, shape (n_events, )
        """
        if self.latent_space == "gaussian":
            return -(z**2 / 2 + 0.5 * math.log(2 * math.pi)).sum(dim=1)
        elif self.latent_space == "uniform":
            return 0.0

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            log probabilities, shape (n_events, ) if not bayesian
                               shape (1+self.bayesian_samples, n_events) if bayesian
        """
        if not self.bayesian_transfer:
            z, jac = self.inn(x, (c,))
            return self.latent_log_prob(z) + jac

        else:
            log_probs = []
            for layer in self.bayesian_layers:
                layer.map = True
            z_map, jac_map = self.inn(x, (c,))
            log_probs.append(self.latent_log_prob(z_map) + jac_map)

            for layer in self.bayesian_layers:
                layer.map = False

            for random_state in self.random_states:
                self.import_random_state(random_state)
                z, jac = self.inn(x, (c,))
                log_probs.append(self.latent_log_prob(z) + jac)

            return torch.stack(log_probs, dim=0)

    def sample(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
            log_prob: log probabilites, shape (n_events, )
        """
        z = torch.randn((c.shape[0], self.dims_in), dtype=c.dtype, device=c.device)
        x, jac = self.inn(z, (c,), rev=True)
        return x, self.latent_log_prob(z) - jac

    def sample_with_probs(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition
        For INNs this is equivalent to normal sampling
        """
        return self.sample(c)

    def transform_hypercube(
        self, r: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes values and jacobians for the given condition and numbers on the unit
        hypercube

        Args:
            r: points on the the unit hypercube, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
            jac: jacobians, shape (n_events, )
        """
        if self.latent_space == "gaussian":
            z = torch.erfinv(2 * r - 1) * math.sqrt(2)
        elif self.latent_space == "uniform":
            z = r
        x, jac = self.inn(z, (c,), rev=True)
        return x, -self.latent_log_prob(z) + jac

    def kl(self) -> torch.Tensor:
        """
        Compute the KL divergence between weight prior and posterior

        Returns:
            Scalar tensor with KL divergence
        """
        assert self.bayesian
        return sum(layer.kl() for layer in self.bayesian_layers)

    def batch_loss(
        self, x: torch.Tensor, c: torch.Tensor, kl_scale: float = 0.0
    ) -> tuple[torch.Tensor, dict]:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
            kl_scale: factor in front of KL loss term, default 0
        Returns:
            loss: batch loss
            loss_terms: dictionary with loss contributions
        """
        inn_loss = -self.log_prob(x, c).mean() / self.dims_in
        if self.bayesian:
            kl_loss = kl_scale * self.kl() / self.dims_in
            loss = inn_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "likeli_loss": inn_loss.item(),
                "kl_loss": kl_loss.item(),
            }
        else:
            loss = inn_loss
            loss_terms = {
                "loss": loss.item(),
            }
        return loss, loss_terms

    def reset_random_state(self):
        """
        Resets the random state of the Bayesian layers
        """
        assert self.bayesian
        for layer in self.bayesian_layers:
            layer.reset_random()

    def sample_random_state(self) -> list[np.ndarray]:
        """
        Sample new random states for the Bayesian layers and return them as a list

        Returns:
            List of numpy arrays with random states
        """
        assert self.bayesian
        return [layer.sample_random_state() for layer in self.bayesian_layers]

    def import_random_state(self, states: list[np.ndarray]):
        """
        Import a list of random states into the Bayesian layers

        Args:
            states: List of numpy arrays with random states
        """
        assert self.bayesian
        for layer, s in zip(self.bayesian_layers, states):
            layer.import_random_state(s)

    def generate_random_state(self):
        """
        Generate and save a set of random states for repeated use
        """
        assert self.bayesian
        self.random_states = [self.sample_random_state() for i in range(self.bayesian_samples)]
