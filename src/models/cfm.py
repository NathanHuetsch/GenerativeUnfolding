from typing import Type, Callable, Union, Optional
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from .layers import *


class CFM(nn.Module):
    """
    Class implementing a conditional CFM model
    """

    def __init__(self, params: dict):
        """
        Initializes and builds the conditional CFM

        Args:
            dims_in: dimension of input
            dims_c: dimension of condition
            params: dictionary with architecture/hyperparameters
        """
        super().__init__()
        self.params = params
        self.dims_in = params["dims_in"]
        self.dims_c = params["dims_c"]
        self.bayesian = params.get("bayesian", False)
        self.bayesian_samples = params.get("bayesian_samples", 20)
        self.bayesian_layers = []
        self.bayesian_factor = params.get("bayesian_factor", 1)


        t_noise_scale = params.get("t_noise_scale", 0)
        minimum_noise_scale = params.get("minimum_noise_scale", 0)
        self.trajectory = LinearTrajectory(t_noise_scale=t_noise_scale,
                                           minimum_noise_scale=minimum_noise_scale)

        self.build_embeddings()
        self.build_net()
        self.build_distributions()
        self.build_solver()

        self.l2_regularization = self.params.get("l2_regularization", False)
        if self.l2_regularization:
            self.l2_factor = self.params.get("l2_factor", 1.e-4)

        self.loss_fct = nn.MSELoss()

        self.use_ema_sample = self.params.get("use_ema_sample", True)

    def build_net(self):
        """
        Construct the Network
        """
        # Build the main network
        network = self.params["network_params"].get("network_class", Subnet)
        self.net = eval(network)(params=self.params, conditional=self.dims_c > 0)
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"        Network: {network} with {n_params} parameters")

        if self.bayesian:
            if isinstance(self.net, Subnet):
                self.bayesian_layers.extend(
                    layer for layer in self.net.layer_list if isinstance(layer, VBLinear)
                )
            else:
                #for block in self.net.blocks:
                #    self.bayesian_layers.extend(
                #        layer for layer in block.layer_list if isinstance(layer, VBLinear)
                #    )
                raise NotImplementedError("Bayesian only implemented for Subnet atm")
            print(f"        Bayesian set to True, Bayesian layers: ", len(self.bayesian_layers))

    def build_embeddings(self):

        params = self.params["network_params"]

        if self.dims_c > 0:
            embed_c = params.get("embed_c", False)
            if embed_c:
                self.embed_c_dim = params.get("embed_c_dim", self.dims_c)
                # Build the c embedding network if used
                self.c_embedding = EmbeddingNet(
                    params=params.get("embed_c_params"),
                    size_in= self.dims_c,
                    size_out=self.embed_c_dim
                )
                n_params = sum(p.numel() for p in self.c_embedding.parameters() if p.requires_grad)
                print(f"        c_embedding: Subnet with {n_params} parameters to {self.embed_c_dim} dimensions")
            else:
                self.embed_c_dim = self.dims_c
                params["embed_c_dim"] = self.dims_c
                self.c_embedding = nn.Identity()
                print(f"        c_embedding: None")

        embed_x = params.get("embed_x", False)
        if embed_x:
            self.embed_x_dim = params.get("embed_x_dim", self.dims_in)
            self.x_embedding = EmbeddingNet(
                params=params.get("embed_x_params"),
                size_in=self.dims_in,
                size_out=self.embed_x_dim
            )
            n_params = sum(p.numel() for p in self.x_embedding.parameters() if p.requires_grad)
            print(f"        x_embedding: Subnet with {n_params} parameters to {self.embed_x_dim} dimensions")
        else:
            self.embed_x_dim = self.dims_in
            params["embed_x_dim"] = self.dims_in
            self.x_embedding = nn.Identity()
            print(f"        x_embedding: None")

        embed_t = params.get("embed_t", False)
        if embed_t:
            self.embed_t_dim = params.get("embed_t_dim", self.dims_in)
            embed_t_mode = params.get("embed_t_mode", "linear")
            if embed_t_mode == "gfprojection":
                self.t_embedding = nn.Sequential(
                    GaussianFourierProjection(embed_dim=self.embed_t_dim),
                    nn.Linear(self.embed_t_dim, self.embed_t_dim))
            elif embed_t_mode == "sinusoidal":
                self.t_embedding = nn.Sequential(
                    sinusoidal_t_embedding(embed_dim=self.embed_t_dim),
                    nn.Linear(self.embed_t_dim, self.embed_t_dim))
            else:
                self.t_embedding = nn.Linear(1, self.embed_t_dim)
            n_params = sum(p.numel() for p in self.t_embedding.parameters() if p.requires_grad)
            print(f"        t_embedding: Mode {embed_t_mode} with {n_params} parameters to {self.embed_t_dim} dimensions")
        else:
            self.embed_t_dim = 1
            params["embed_t_dim"] = 1
            self.t_embedding = nn.Identity()
            print(f"        t_embedding: None")

    def build_distributions(self):

        self.beta_dist = self.params.get("beta_dist", False)
        if self.beta_dist:
            self.beta_dist_a = self.params.get("beta_dist_a", 1)
            self.beta_dist_b = self.params.get("beta_dist_b", 0.7)
            self.t_dist = torch.distributions.beta.Beta(concentration1=float(self.beta_dist_a),
                                                        concentration0=float(self.beta_dist_b))
            print(f"        t distribution: beta distribution with params {self.beta_dist_a, self.beta_dist_b}")
        else:
            self.t_dist = torch.distributions.uniform.Uniform(low=0, high=1)
            self.mod_t = self.params.get("mod_t", False)
            print(f"        t distribution: uniform")

        self.latent_space = self.params.get("latent_space", "gaussian")
        if self.latent_space == "gaussian":
            self.latent_dist = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(self.dims_in), torch.eye(self.dims_in))
            print(f"        latent space: gaussian")
        elif self.latent_space == "uniform":
            uniform_bounds = self.params.get("uniform_bounds", [0., 1.])
            self.latent_dist = torch.distributions.uniform.Uniform(
                torch.full((self.dims_in, ), uniform_bounds[0]), torch.full((self.dims_in, ), uniform_bounds[1]))
            print(f"        latent space: uniform with bounds {uniform_bounds}")
        elif self.latent_space == "mixture":
            self.uniform_channels = self.params.get("uniform_channels")
            self.normal_channels = [i for i in range(self.dims_in) if i not in self.uniform_channels]
            self.latent_dist = MixtureDistribution(normal_channels=self.normal_channels,
                                                   uniform_channels=self.uniform_channels)
            print(f"        latent space: mixture with uniform channels {self.uniform_channels}")

    def build_solver(self):

        self.solver_type = self.params.get("solver", "ODE")
        if self.solver_type == "ODE":
            self.solver = ODEsolver(params=self.params.get("solver_params", {}))
        else:
            self.solver = SDEsolver(params=self.params.get("solver_params", {}))

    def sample(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
        """
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        try:
            if self.use_ema_sample:
                net = self.ema
            else:
                net = self.net
        except:
            print("Loading Ema failed")
            net=self.net

        with torch.no_grad():
            c = self.c_embedding(c)

        # Wrap the network such that the ODE solver can call it
            if self.solver_type == "ODE":
                def net_wrapper(t, x_t):
                    x_t = self.x_embedding(x_t)
                    t = self.t_embedding(t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device))
                    v = net(t, x_t, c)
                    return v
            else:
                net_wrapper = SDE_wrapper(self, c)

            # Sample from the latent distribution
            x_0 = self.latent_dist.sample((batch_size, )).to(device, dtype=dtype)
            # Solve the ODE from t=1 to t=0 from the sampled initial condition
            x_t = self.solver(net_wrapper, x_0, reverse=False)
        return x_t[-1]

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
        # Sample a target for the diffusion trajectory
        x_0 = self.latent_dist.sample((x.size(0), )).to(x.device, dtype=x.dtype)

        # Sample a time step
        if self.mod_t:
            t = torch.rand((1), dtype=x.dtype, device=x.device)
            t = (t + torch.arange(x.size(0)).unsqueeze(-1) / x.size(0)) % 1
        else:
            t = torch.rand((x.size(0), 1), dtype=x.dtype, device=x.device)

        # Calculate x_t, x_t_dot along the trajectory
        x_t, x_t_dot = self.trajectory(x_0, x, t)
        # Predict the velocity
        t = self.t_embedding(t)
        c = self.c_embedding(c)
        x_t = self.x_embedding(x_t)
        v_pred = self.net(t, x_t, c)
        # Calculate the loss
        cfm_loss = self.loss_fct(v_pred, x_t_dot)

        if self.bayesian:
            kl_loss = self.bayesian_factor * kl_scale * self.kl() / self.dims_in
            loss = cfm_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "mse": cfm_loss.item(),
                "kl": kl_loss.item(),
            }
        elif self.l2_regularization:
            regularization_loss = self.l2_factor * torch.norm(v_pred)
            loss = cfm_loss + regularization_loss
            loss_terms = {
                "loss": loss.item(),
                "mse": cfm_loss.item(),
                "l2": regularization_loss.item()
            }
        else:
            loss = cfm_loss
            loss_terms = {
                "loss": loss.item(),
            }
        return loss, loss_terms

    def kl(self) -> torch.Tensor:
        """
        Compute the KL divergence between weight prior and posterior

        Returns:
            Scalar tensor with KL divergence
        """
        assert self.bayesian
        return sum(layer.kl() for layer in self.bayesian_layers)

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
        self.random_states = [self.sample_random_state() for i in range(self.bayesian_samples)]


class CFMwithTransformer(CFM):

    def __init__(self, params: dict):
        super().__init__(params)
        self.reco_jets = self.params["process_params"].get("reco_jets", 4)
        print(f"    Using reco_jets {self.reco_jets}")

    def build_net(self):
        self.dim_embedding = self.params["dim_embedding"]
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=self.params["n_head"],
            num_encoder_layers=self.params["n_encoder_layers"],
            num_decoder_layers=self.params["n_decoder_layers"],
            dim_feedforward=self.params["dim_feedforward"],
            dropout=self.params.get("dropout", 0.0),
            batch_first=True,
        )
        n_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        print(f"        Network: Transformer with {n_params} parameters")

        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in self.params:
            layer_args["prior_prec"] = self.params["prior_prec"]
        if "std_init" in self.params:
            layer_args["std_init"] = self.params["std_init"]
        self.net = layer_class(self.dim_embedding+self.embed_t_dim, 1, **layer_args)
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"        Network: Final layer with {n_params} parameters")

        if self.bayesian:
            self.bayesian_layers.append(self.net)
            print(f"        Bayesian set to True, Bayesian layers: ", len(self.bayesian_layers))



    def compute_embedding(
        self, p: torch.Tensor, n_components: int, t: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(n_components, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        if t is None:
            p = p.unsqueeze(-1)
        else:
            p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1).expand(t.shape[0], p.shape[1], 1)], dim=-1)
        n_rest = self.dim_embedding - n_components - p.shape[-1]
        assert n_rest >= 0
        zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
        return torch.cat((p, one_hot, zeros), dim=-1)

    def compute_padding_mask(
        self, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        mask = torch.isnan(p)
        #print(mask[:, :23].float().mean(), mask[:, 23:27].float().mean(), mask[:, 27:].float().mean())
        return mask

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
        # Sample a target for the diffusion trajectory
        x_1 = torch.randn(x.size(), dtype=x.dtype, device=x.device)
        # Sample a time step
        t = torch.rand((x.size(0), 1), dtype=x.dtype, device=x.device)
        # Calculate x_t along the trajectory
        x_t = (1 - t) * x + t * x_1
        # Calculate the derivative of x_t, i.e. the conditional velocity
        x_t_dot = -x + x_1
        # Predict the velocity
        #if self.embed_t:
        #    t = self.t_embedding(t)
        if self.reco_jets > 4:
            jet_mask = self.compute_padding_mask(c)
            c_fixed = torch.where(jet_mask, 0, c)
            embedding = self.transformer(
                src=self.compute_embedding(
                    c_fixed,
                    n_components=self.dims_c
                ),
                tgt=self.compute_embedding(
                    x_t,
                    n_components=self.dims_in,
                    t=t
                ))#,src_key_padding_mask=jet_mask)
        else:
            embedding = self.transformer(
                src=self.compute_embedding(
                    c,
                    n_components=self.dims_c
                ),
                tgt=self.compute_embedding(
                    x_t,
                    n_components=self.dims_in,
                    t=t
                )
            )

        t = self.t_embedding(t)
        v_pred = self.net(torch.cat([t.unsqueeze(1).repeat(1, x_t.size(1), 1), embedding], dim=-1)).squeeze()
        #v_pred = self.net(embedding).squeeze()
        # Calculate the loss
        cfm_loss = self.loss_fct(v_pred, x_t_dot)

        if self.bayesian:
            kl_loss = kl_scale * self.kl() / self.dims_in
            loss = cfm_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "mse": cfm_loss.item(),
                "kl": kl_loss.item(),
            }
        else:
            loss = cfm_loss
            loss_terms = {
                "loss": loss.item(),
            }
        return loss, loss_terms

    def sample(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
        """
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        # Wrap the network such that the ODE solver can call it
        def net_wrapper(t, x_t):
            t = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            #if self.embed_t:
            #    t = self.t_embedding(t)
            embedding = self.transformer.decoder(
                self.compute_embedding(
                    x_t,
                    n_components=self.dims_in,
                    t=t
                ),
                memory
            )
            t = self.t_embedding(t)
            v = self.net(torch.cat([t.unsqueeze(1).repeat(1, x_t.size(1), 1), embedding], dim=-1)).squeeze()
            # v_pred = self.net(embedding).squeeze()
            #v = self.net(torch.cat([t.unsqueeze(-1).repeat(1, x_t.size(1), 1), embedding], dim=-1)).squeeze()
            return v

        with torch.no_grad():
            if self.reco_jets > 4:
                jet_mask = self.compute_padding_mask(c)
                c_fixed = torch.where(jet_mask, 0, c)
                memory = self.transformer.encoder(self.compute_embedding(
                        c_fixed,
                        n_components=self.dims_c,
                    ))#,src_key_padding_mask=jet_mask)
            else:
                memory = self.transformer.encoder(self.compute_embedding(
                    c,
                    n_components=self.dims_c
                ))
            # Sample from the latent distribution
            x_1 = torch.randn((batch_size, self.dims_in), dtype=dtype, device=device)
            # Solve the ODE from t=1 to t=0 from the sampled initial condition
            x_t = self.solver(net_wrapper, x_1, reverse=True)
        return x_t[-1]


class TransfusionAR(CFM):

    def __init__(self, params: dict):
        super().__init__(params)

    def build_net(self):
        self.dim_embedding = self.params["dim_embedding"]
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=self.params["n_head"],
            num_encoder_layers=self.params["n_encoder_layers"],
            num_decoder_layers=self.params["n_decoder_layers"],
            dim_feedforward=self.params["dim_feedforward"],
            dropout=self.params.get("dropout", 0.0),
            batch_first=True,
        )
        n_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        print(f"        Network: Transformer with {n_params} parameters")

        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in self.params:
            layer_args["prior_prec"] = self.params["prior_prec"]
        if "std_init" in self.params:
            layer_args["std_init"] = self.params["std_init"]

        self.params["network_params"]["embed_x_dim"] = 1
        self.params["network_params"]["embed_c_dim"] = self.dim_embedding
        self.params["network_params"]["final_output_dim"] = 1

        self.net = Subnet(self.params, conditional=True)
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"        Network: Final layer with {n_params} parameters")

        if self.bayesian:
            self.bayesian_layers.append(self.net)
            print(f"        Bayesian set to True, Bayesian factor: {self.bayesian_factor}, Bayesian layers: ", len(self.bayesian_layers))

    def compute_embedding(
        self, p: torch.Tensor, n_components: int, t: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(n_components, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        if t is None:
            p = p.unsqueeze(-1)
        else:
            p = torch.cat([p.unsqueeze(-1), t.unsqueeze(-1).expand(t.shape[0], p.shape[1], 1)], dim=-1)
        n_rest = self.dim_embedding - n_components - p.shape[-1]
        assert n_rest >= 0
        zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
        return torch.cat((p, one_hot, zeros), dim=-1)

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
        # Sample a target for the diffusion trajectory
        x_1 = torch.randn(x.size(), dtype=x.dtype, device=x.device)
        # Sample a time step
        t = torch.rand((x.size(0), 1), dtype=x.dtype, device=x.device)
        # Calculate x_t along the trajectory
        x_t = (1 - t) * x + t * x_1
        # Calculate the derivative of x_t, i.e. the conditional velocity
        x_t_dot = -x + x_1
        # Predict the velocity
        #if self.embed_t:
        #    t = self.t_embedding(t)
        xp = nn.functional.pad(x[:, :-1], (1, 0))
        embedding = self.transformer(
            src=self.compute_embedding(
                c,
                n_components=self.dims_c,
            ),
            tgt=self.compute_embedding(
                xp,
                n_components=self.dims_in + 1,
            ),
            tgt_mask=torch.ones(
                (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
            ).triu(diagonal=1),
        )

        t = self.t_embedding(t)
        v_pred = self.net(t.unsqueeze(1).repeat(1, x_t.size(1), 1), x_t.unsqueeze(-1), embedding).squeeze()

        # Calculate the loss
        cfm_loss = self.loss_fct(v_pred, x_t_dot)

        if self.bayesian:
            kl_loss = kl_scale * self.kl() / self.dims_in
            loss = cfm_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "mse": cfm_loss.item(),
                "kl": kl_loss.item(),
            }
        else:
            loss = cfm_loss
            loss_terms = {
                "loss": loss.item(),
            }
        return loss, loss_terms

    def sample(self, c: torch.Tensor) -> torch.Tensor:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
        """
        x = torch.zeros((c.shape[0], 1), device=c.device, dtype=c.dtype)
        c_emb = self.compute_embedding(
            c, n_components=self.dims_c)
        for i in range(self.dims_in):
            embedding = self.transformer(
                src=c_emb,
                tgt=self.compute_embedding(
                    x,
                    n_components=self.dims_in + 1),
                tgt_mask=torch.ones(
                    (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1),
            )
            x_new = self.sample_dimension(embedding[:, -1:, :])
            x = torch.cat((x, x_new), dim=1)
        return x[:, 1:]

    def sample_dimension(self, c):

        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        def net_wrapper(t, x_t):
            t_torch = t * torch.ones((x_t.size(0), 1), dtype=dtype, device=device)
            t_torch = self.t_embedding(t_torch)
            v_pred = self.net(t_torch.unsqueeze(1), x_t.unsqueeze(-1), c).reshape((x_t.size(0), 1))
            return v_pred

        with torch.no_grad():
            # Sample from the latent distribution
            x_0 = self.latent_dist.sample((batch_size, 1)).to(device, dtype=dtype)[..., 0]
            # Solve the ODE from t=1 to t=0 from the sampled initial condition
            x_t = self.solver(net_wrapper, x_0, reverse=False)
        return x_t[-1]
