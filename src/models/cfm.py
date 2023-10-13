from typing import Type, Callable, Union, Optional
import time
import math
import torch
import torch.nn as nn
import numpy as np
from .vblinear import VBLinear
from torchdiffeq import odeint
from .layers import GaussianFourierProjection, autograd_trace, hutch_trace, SinCos_embedding, Subnet, Resnet
from .optimal_transport import OTPlanSampler


class CFM(nn.Module):
    """
    Class implementing a conditional CFM model
    """

    def __init__(self, dims_in: int, dims_c: int, params: dict):
        """
        Initializes and builds the conditional CFM

        Args:
            dims_in: dimension of input
            dims_c: dimension of condition
            params: dictionary with architecture/hyperparameters
        """
        super().__init__()
        self.params = params
        self.dims_in = dims_in
        self.dims_c = dims_c
        # TODO: Bayesian CFM not implemented yet
        self.bayesian = params.get("bayesian", False)
        self.bayesian_samples = params.get("bayesian_samples", 20)
        self.bayesian_layers = []
        self.bayesian_factor = params.get("bayesian_factor", 1)
        self.bayesian_transfer = False
        self.latent_space = self.params.get("latent_space", "gaussian")

        # Parameters for the ODE solver. Default settings should work fine
        # Details on the solver under https://github.com/rtqichen/torchdiffeq
        self.hutch = self.params.get("hutch", False)
        self.method = self.params.get("method", "dopri5")
        self.rtol = self.params.get("rtol", 1.e-3)
        self.atol = self.params.get("atol", 1.e-5)
        self.step_size = self.params.get("step_size", 0.025)

        if self.method == "dopri5":
            print(f"Using ODE method {self.method}, atol {self.atol}, rtol {self.rtol}, hutch {self.hutch}", flush=True)
        else:
            print(f"Using ODE method {self.method}, step_size {self.step_size}, hutch {self.hutch}", flush=True)

        self.use_OT = self.params.get("use_OT", False)
        if self.use_OT:
            ot_method = self.params.get("OT_method", "exact")
            self.OT_sampler = OTPlanSampler(method=ot_method)
            print(f"Using OT with method {ot_method}")

        self.embed_c = self.params.get("embed_c", False)
        if self.embed_c:
            self.embed_c_dim = self.params.get("embed_c_dim", self.dims_in)
        else:
            self.embed_c_dim = self.dims_c

        self.embed_x = self.params.get("embed_x", False)
        if self.embed_x:
            self.embed_x_frequencies = self.params.get("embed_x_frequencies", 10)
            self.embed_x_dim = self.embed_x_frequencies*2*self.dims_in
            self.embed_c_dim = self.embed_x_frequencies*2*self.dims_c
        else:
            self.embed_x_dim = self.dims_in
            self.embed_x_frequencies = None
        self.build_net()

        self.l2_regularization = self.params.get("l2_regularization", False)
        if self.l2_regularization:
            self.l2_factor = self.params.get("l2_factor", 1.e-4)

        self.loss_fct = nn.MSELoss()

        self.beta_dist = self.params.get("beta_dist", False)
        if self.beta_dist:
            self.beta_dist_a = self.params.get("beta_dist_a", 1)
            self.beta_dist_b = self.params.get("beta_dist_b", 0.7)
            self.dist = torch.distributions.beta.Beta(concentration1=float(self.beta_dist_a),
                                                      concentration0=float(self.beta_dist_b))
            print(f"Using beta distribution to sample t with params {self.beta_dist_a, self.beta_dist_b}")
        else:
            self.dist = torch.distributions.uniform.Uniform(low=0, high=1)
            print(f"Using uniform distribution to sample t")

        self.mod_t = self.params.get("mod_t", False)

        self.t_min = self.params.get("t_min", 0.)
        self.t_max = self.params.get("t_max", 0.)

    def build_net(self):
        """
        Construct the Network
        """
        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in self.params:
            layer_args["prior_prec"] = self.params["prior_prec"]
        if "std_init" in self.params:
            layer_args["std_init"] = self.params["std_init"]

        # Prepare embedding on times t
        # Use SinCos frequency embedding or GaussianFourierProjection if wanted. Add one linear layer if wanted
        self.frequency_embed_t = self.params.get("frequency_embed_t", False)
        self.gfprojection_embed_t = self.params.get("gfprojection_embed_t", False)
        self.linear_embed_t = self.params.get("linear_embed_t", False)
        if self.frequency_embed_t:
            self.embed_t_dim = self.params.get("embed_t_dim", self.dims_in)
            if self.linear_embed_t:
                self.t_embed = nn.Sequential(SinCos_embedding(n_frequencies=int(self.embed_t_dim / 2), sigmoid=False),
                                             nn.Linear(self.embed_t_dim, self.embed_t_dim))
            else:
                self.t_embed = SinCos_embedding(n_frequencies=int(self.embed_t_dim / 2), sigmoid=False)
        elif self.gfprojection_embed_t:
            self.embed_t_dim = self.params.get("embed_t_dim", self.dims_in)
            if self.linear_embed_t:
                self.t_embed = nn.Sequential(
                    GaussianFourierProjection(embed_dim=self.embed_t_dim),
                    nn.Linear(self.embed_t_dim, self.embed_t_dim))
            else:
                self.t_embed = GaussianFourierProjection(embed_dim=self.embed_t_dim)
        else:
            self.t_embed = nn.Identity()
            self.embed_t_dim = 1

        # Build the main network
        if self.params.get("n_blocks", 1) == 1:
            self.net = Subnet(
                self.params.get("layers_per_block", 3),
                size_in = self.embed_x_dim + self.embed_t_dim + self.embed_c_dim,
                size_out = self.dims_in,
                internal_size=self.params.get("internal_size"),
                dropout=self.params.get("dropout", 0.0),
                activation=self.params.get("activation", nn.SiLU),
                layer_class=layer_class,
                layer_args=layer_args,
            )
        else:
            self.net = Resnet(
                self.params.get("layers_per_block", 3),
                embed_t_dim = self.embed_t_dim,
                embed_x_dim = self.embed_x_dim,
                embed_c_dim = self.embed_c_dim,
                size_out = self.dims_in,
                internal_size=self.params.get("internal_size"),
                dropout=self.params.get("dropout", 0.0),
                activation=self.params.get("activation", nn.SiLU),
                layer_class=layer_class,
                layer_args=layer_args,
                n_blocks=self.params.get("n_blocks", 1),
                condition_mode=self.params.get("condition_mode", "concat")
            )

        # Build the c embedding network if used
        if self.embed_c:
            self.c_embedding = Subnet(
            self.params.get("layers_per_block_c", 3),
            size_in = self.dims_c,
            size_out=self.embed_c_dim,
            internal_size=self.params.get("internal_size_c", self.params.get("internal_size")),
            dropout=self.params.get("dropout_c", 0.0),
            activation=self.params.get("activation_c", nn.SiLU),
            layer_class=nn.Linear
            )

        if self.bayesian:
            if isinstance(self.net, Subnet):
                self.bayesian_layers.extend(
                    layer for layer in self.net.layer_list if isinstance(layer, VBLinear)
                )
            else:
                for block in self.net.blocks:
                    self.bayesian_layers.extend(
                        layer for layer in block.layer_list if isinstance(layer, VBLinear)
                    )

            print("Bayesian layers: ", len(self.bayesian_layers))

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

        batch_size = x.size(0)
        dtype = x.dtype
        device = x.device

        # Wrap the network such that the ODE solver can call it
        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                # Prepare the network inputs
                x_t = state[0].detach().requires_grad_(True)
                t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device).requires_grad_(False)
                t_torch = self.t_embed(t_torch)
                # Predict v
                v = self.net(torch.cat([t_torch, x_t, c], dim=-1))
                # Calculate the jacobian trace
                if self.hutch:
                    dlogp_dt = -hutch_trace(v, x_t).view(-1, 1)
                else:
                    dlogp_dt = -autograd_trace(v, x_t).view(-1, 1)
            return v.detach(), dlogp_dt.detach()

        # Set initial conditions for the ODE
        logp_diff_1 = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        states = (x, logp_diff_1)
        if self.embed_c:
            c = self.c_embedding(c)
        if not self.bayesian_transfer:
            # Solve the ODE from t=0 to t=1 from data to noise
            x_t, logp_diff_t = odeint(
                net_wrapper,
                states,
                torch.tensor([self.t_min, self.t_max], dtype=dtype, device=device),
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
                options=dict(step_size=self.step_size)
                )
            # Extract the latent space points and the jacobians
            x_1 = x_t[-1].detach()
            jac = logp_diff_t[-1].detach()
            return self.latent_log_prob(x_1).squeeze() - jac.squeeze()

        else:
            log_probs = []
            for layer in self.bayesian_layers:
                layer.map = True
            x_t, logp_diff_t = odeint(
                net_wrapper,
                states,
                torch.tensor([0, 1], dtype=dtype, device=device),
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
                options=dict(step_size=self.step_size)
            )
            x_1_map = x_t[-1].detach()
            jac_map = logp_diff_t[-1].detach()
            log_probs.append(self.latent_log_prob(x_1_map) - jac_map.squeeze())

            for layer in self.bayesian_layers:
                layer.map = False

            for random_state in self.random_states:
                self.import_random_state(random_state)
                x_t, logp_diff_t = odeint(
                    net_wrapper,
                    states,
                    torch.tensor([0, 1], dtype=dtype, device=device),
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.method,
                    options=dict(step_size=self.step_size)
                )
                x_1 = x_t[-1].detach()
                jac = logp_diff_t[-1].detach()
                log_probs.append(self.latent_log_prob(x_1) - jac.squeeze())

            return torch.stack(log_probs, dim=0)

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
            t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            t_torch = self.t_embed(t_torch).reshape(batch_size, -1)
            v = self.net(torch.cat([t_torch, x_t, c], dim=-1))
            return v

        with torch.no_grad():
            if self.embed_c:
                c = self.c_embedding(c)
            # Sample from the latent distribution
            x_1 = torch.randn((batch_size, self.dims_in), dtype=dtype, device=device)
            # Solve the ODE from t=1 to t=0 from the sampled initial condition
            x_t = odeint(
                net_wrapper,
                x_1,
                torch.tensor([1, 0], dtype=dtype, device=device),
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
                options=dict(step_size=self.step_size)
            )
        # return the generated sample. This function does not calculate jacobians and just returns a 0 instead
        return x_t[-1], torch.Tensor([0])

    def sample_with_probs(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
            log_prob: log probabilites, shape (n_events, )
        """
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        # Wrap the network such that the ODE solver can call it
        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                x_t = state[0].requires_grad_(True)
                t_torch = (t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)).requires_grad_(False)
                t_torch = self.t_embed(t_torch)
                # Predict v
                v = self.net(torch.cat([t_torch, x_t, c], dim=-1))
                if self.hutch:
                    dlogp_dt = -hutch_trace(v, x_t).view(-1, 1)
                else:
                    dlogp_dt = -autograd_trace(v, x_t).view(-1, 1)
            return v.detach(), dlogp_dt.detach()

        # Sample from the latent distribution and prepare initial state of ODE solver
        x_1 = torch.randn((batch_size, self.dims_in), dtype=dtype, device=device)
        logp_diff_1 = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        states = (x_1, logp_diff_1)
        c.requires_grad_(False)
        if self.embed_c:
            c = self.c_embedding(c)
        # Solve the ODE from t=1 to t=0 from the sampled initial condition
        x_t, logp_diff_t = odeint(
            net_wrapper,
            states,
            torch.tensor([1, 0], dtype=dtype, device=device),
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
            options=dict(step_size=self.step_size)
        )
        # Extract the generated samples and the jacobians
        x_0 = x_t[-1].detach()
        jac = logp_diff_t[-1].detach()
        return x_0, self.latent_log_prob(x_1).squeeze() + jac.squeeze()

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
        # Sample a target for the diffusion trajectory
        x_1 = torch.randn(x.size(), dtype=x.dtype, device=x.device)

        if self.use_OT:
            with torch.no_grad():
                x, x_1 = self.OT_sampler.sample_plan(x, x_1)
                x, x_1, c = x.repeat(20, 1), x_1.repeat(20, 1), c.repeat(20, 1)

        # Sample a time step

        if self.mod_t:
            t = torch.rand((1), dtype=x.dtype, device=x.device)
            t = (t + torch.arange(x.size(0)).unsqueeze(-1) / x.size(0)) % 1
        else:
            t = torch.rand((x.size(0), 1), dtype=x.dtype, device=x.device)

        #t = 1 - self.dist.sample((x.size(0), 1)).to(x.device)
        # Calculate x_t along the trajectory
        x_t = (1 - t) * x + t * x_1
        # Calculate the derivative of x_t, i.e. the conditional velocity
        x_t_dot = -x + x_1
        # Predict the velocity
        t = self.t_embed(t).reshape(x.size(0), -1)
        if self.embed_c:
            c = self.c_embedding(c)
        v_pred = self.net(torch.cat([t, x_t, c], dim=-1))
        # Calculate the loss
        cfm_loss = self.loss_fct(v_pred, x_t_dot)

        if self.bayesian:
            kl_loss = self.bayesian_factor * kl_scale * self.kl() / self.dims_in
            loss = cfm_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "likeli_loss": cfm_loss.item(),
                "kl_loss": kl_loss.item(),
            }
        elif self.l2_regularization:
            regularization_loss = self.l2_factor * torch.norm(v_pred)
            loss = cfm_loss + regularization_loss
            loss_terms = {
                "loss": loss.item(),
                "cfm_loss": cfm_loss.item(),
                "regularization_loss": regularization_loss.item()
            }
        else:
            loss = cfm_loss
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
        self.random_states = [self.sample_random_state() for i in range(self.bayesian_samples)]


class CFMwithTransformer(CFM):

    def __init__(self, dims_in: int, dims_c: int, params: dict):
        super().__init__(dims_in, dims_c, params)
        self.bayesian_layers = []
        # Build the cfm MLP
        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in self.params:
            layer_args["prior_prec"] = self.params["prior_prec"]
        if "std_init" in self.params:
            layer_args["std_init"] = self.params["std_init"]
        self.dim_embedding = params["dim_embedding"]
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=self.params["n_head"],
            num_encoder_layers=self.params["n_encoder_layers"],
            num_decoder_layers=self.params["n_decoder_layers"],
            dim_feedforward=self.params["dim_feedforward"],
            dropout=self.params.get("dropout", 0.0),
            # activation=params.get("activation", "relu"),
            batch_first=True,
        )

        single_layer = self.params.get("single_layer", False)
        if single_layer:
            self.net = layer_class(self.dim_embedding+1, 1)
            if self.bayesian:
                self.bayesian_layers.append(self.net)
        else:
            self.net = Subnet(
                self.params.get("layers_per_block", 3),
                size_in=self.dim_embedding+1,
                size_out=1,
                internal_size=self.params.get("internal_size"),
                dropout=self.params.get("dropout", 0.0),
                activation=self.params.get("activation", nn.SiLU),
                layer_class=layer_class,
                layer_args=layer_args,
            )
            if self.bayesian:
                self.bayesian_layers.extend(
                    layer for layer in self.net.layer_list if isinstance(layer, VBLinear)
                )

        print("Bayesian layers ",len(self.bayesian_layers))

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
        #print(p.shape)
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
        embedding = self.transformer(
            src=self.compute_embedding(
                c,
                n_components=self.dims_c
            ),
            tgt=self.compute_embedding(
                x_t,
                n_components=self.dims_in,
                t = t
            )
        )
        #v_pred = self.net(torch.cat([t, embedding.reshape((embedding.size(0), -1))], dim=-1))
        v_pred = self.net(torch.cat([t.unsqueeze(-1).repeat(1, x_t.size(1), 1), embedding], dim=-1)).squeeze()
        #v_pred = self.net(embedding).squeeze()
        # Calculate the loss
        cfm_loss = self.loss_fct(v_pred, x_t_dot)

        if self.bayesian:
            kl_loss = kl_scale * self.kl() / self.dims_in
            loss = cfm_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "likeli_loss": cfm_loss.item(),
                "kl_loss": kl_loss.item(),
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
            t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            #if self.embed_t:
            #    t_torch = self.t_embedding(t_torch)
            embedding = self.transformer.decoder(
                self.compute_embedding(
                    x_t,
                    n_components=self.dims_in,
                    t=t_torch
                ),
                memory
            )
            v = self.net(torch.cat([t_torch.unsqueeze(-1).repeat(1, x_t.size(1), 1), embedding], dim=-1)).squeeze()
            return v

        with torch.no_grad():
            memory = self.transformer.encoder(self.compute_embedding(
                    c,
                    n_components=self.dims_c
                ))
            # Sample from the latent distribution
            x_1 = torch.randn((batch_size, self.dims_in), dtype=dtype, device=device)
            # Solve the ODE from t=1 to t=0 from the sampled initial condition

            x_t = odeint(
                net_wrapper,
                x_1,
                torch.tensor([1, 0], dtype=dtype, device=device),
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
                options=dict(step_size=self.step_size)
            )
        # return the generated sample. This function does not calculate jacobians and just returns a 0 instead
        return x_t[-1], torch.Tensor([0])

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

        batch_size = x.size(0)
        dtype = x.dtype
        device = x.device

        # Wrap the network such that the ODE solver can call it
        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                # Prepare the network inputs
                x_t = state[0].detach().requires_grad_(True)
                t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device).requires_grad_(False)
                embedding = self.transformer.decoder(
                    self.compute_embedding(
                        x_t,
                        n_components=self.dims_in,
                        t=t_torch
                    ),
                    memory
                )
                # Predict v
                v = self.net(torch.cat([t_torch.unsqueeze(-1).repeat(1, x_t.size(1), 1), embedding], dim=-1)).squeeze()
                # Calculate the jacobian trace
                if self.hutch:
                    dlogp_dt = -hutch_trace(v, x_t).view(-1, 1)
                else:
                    dlogp_dt = -autograd_trace(v, x_t).view(-1, 1)
            return v.detach(), dlogp_dt.detach()

        # Set initial conditions for the ODE
        logp_diff_1 = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        states = (x, logp_diff_1)

        memory = self.transformer.encoder(self.compute_embedding(
            c,
            n_components=self.dims_c
        )).requires_grad_(False)

        if not self.bayesian_transfer:
            # Solve the ODE from t=0 to t=1 from data to noise
            x_t, logp_diff_t = odeint(
                net_wrapper,
                states,
                torch.tensor([self.t_min, self.t_max], dtype=dtype, device=device),
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
                options=dict(step_size=self.step_size)
                )
            # Extract the latent space points and the jacobians
            x_1 = x_t[-1].detach()
            jac = logp_diff_t[-1].detach()
            return self.latent_log_prob(x_1).squeeze() - jac.squeeze()

        else:
            log_probs = []
            for layer in self.bayesian_layers:
                layer.map = True
            x_t, logp_diff_t = odeint(
                net_wrapper,
                states,
                torch.tensor([0, 1], dtype=dtype, device=device),
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
                options=dict(step_size=self.step_size)
            )
            x_1_map = x_t[-1].detach()
            jac_map = logp_diff_t[-1].detach()
            log_probs.append(self.latent_log_prob(x_1_map) - jac_map.squeeze())

            for layer in self.bayesian_layers:
                layer.map = False

            for random_state in self.random_states:
                self.import_random_state(random_state)
                x_t, logp_diff_t = odeint(
                    net_wrapper,
                    states,
                    torch.tensor([0, 1], dtype=dtype, device=device),
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.method,
                    options=dict(step_size=self.step_size)
                )
                x_1 = x_t[-1].detach()
                jac = logp_diff_t[-1].detach()
                log_probs.append(self.latent_log_prob(x_1) - jac.squeeze())

            return torch.stack(log_probs, dim=0)
