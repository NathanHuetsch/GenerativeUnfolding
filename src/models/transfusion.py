import math
from typing import Type, Callable, Union, Optional
import torch
import torch.nn as nn
from torchdiffeq import odeint
from .layers import *
import numpy as np

L2PI = 0.5 * math.log(2 * math.pi)


class TransfusionAR(nn.Module):
    def __init__(self, n_particles_in: int, n_particles_c: int, params: dict):
        super().__init__()
        self.params = params
        self.n_particles_in = n_particles_in
        self.n_particles_c = n_particles_c
        self.min_n_particles = params.get("min_n_particles", n_particles_in)

        self.bayesian = params.get("bayesian", False)
        self.bayesian_samples = params.get("bayesian_samples", 20)
        self.bayesian_layers = []
        self.bayesian_factor = params.get("bayesian_factor", 1)
        self.bayesian_transfer = False

        self.latent_space = self.params.get("latent_space", "gaussian")
        self.dim_embedding = params["dim_embedding"]
        self.n_one_hot = n_particles_in - self.min_n_particles
        self.pt_eta_phi = params.get("pt_eta_phi", False)
        self.eta_cut = params.get("eta_cut", False)

        self.hutch = self.params.get("hutch", False)
        self.method = self.params.get("method", "dopri5")
        self.rtol = self.params.get("rtol", 1.e-3)
        self.atol = self.params.get("atol", 1.e-5)
        self.step_size = self.params.get("step_size", 0.01)
        print(f"Using ODE method {self.method}, step_size {self.step_size}, hutch {self.hutch}", flush=True)

        # Build the transformer
        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=params["n_head"],
            num_encoder_layers=params["n_encoder_layers"],
            num_decoder_layers=params["n_decoder_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params.get("dropout_transformer", 0.0),
            #activation=params.get("activation", "relu"),
            batch_first=True,
        )
        self.mass_mask = nn.Parameter(torch.tensor([params["mass_mask"]]), requires_grad=False)

        # Prepare embedding on inputs x
        # Use SinCos frequency embedding if wanted. Add one linear layer if wanted
        self.frequency_embed_x = self.params.get("frequency_embed_x", False)
        self.linear_embed_x = self.params.get("linear_embed_x", False)
        if self.frequency_embed_x:
            assert self.dim_embedding%8 == 0, "Required for frequency embed"
            n_frequencies = int(self.dim_embedding/8)
            if self.linear_embed_x:
                self.x_embed = nn.Sequential(SinCos_embedding(n_frequencies=n_frequencies),
                                               nn.Linear(self.dim_embedding, self.dim_embedding))
            else:
                self.x_embed = SinCos_embedding(n_frequencies=n_frequencies)
        else:
            self.x_embed = None

        # Prepare embedding on conditions c
        # Use SinCos frequency embedding if wanted. Add one linear layer if wanted
        self.frequency_embed_c = self.params.get("frequency_embed_c", self.frequency_embed_x)
        self.linear_embed_c = self.params.get("linear_embed_c", self.linear_embed_x)
        if self.frequency_embed_c:
            assert self.dim_embedding % 8 == 0, "Required for frequency embed"
            n_frequencies = int(self.dim_embedding / 8)
            if self.linear_embed_x:
                self.c_embed = nn.Sequential(SinCos_embedding(n_frequencies=n_frequencies),
                                               nn.Linear(self.dim_embedding, self.dim_embedding))
            else:
                self.c_embed = SinCos_embedding(n_frequencies=n_frequencies)
        else:
            self.c_embed = None

        self.positional_encoding = PositionalEncoding(d_model=self.dim_embedding,
                                                      max_len=max(n_particles_in, n_particles_c)+1,
                                                      dropout=0.0)

        # Prepare embedding on times t
        # Use SinCos frequency embedding or GaussianFourierProjection if wanted. Add one linear layer if wanted
        self.frequency_embed_t = self.params.get("frequency_embed_t", False)
        self.gfprojection_embed_t = self.params.get("gfprojection_embed_t", False)
        self.linear_embed_t = self.params.get("frequency_embed_t", False)
        if self.frequency_embed_t:
            self.embed_t_dim = self.params.get("embed_t_dim", 64)
            if self.linear_embed_t:
                self.t_embed = nn.Sequential(SinCos_embedding(n_frequencies=int(self.embed_t_dim/2), sigmoid=False),
                                             nn.Linear(self.embed_t_dim, self.embed_t_dim))
            else:
                self.t_embed = SinCos_embedding(n_frequencies=int(self.embed_t_dim/2), sigmoid=False)
        elif self.gfprojection_embed_t:
            self.embed_t_dim = self.params.get("embed_t_dim", 64)
            if self.linear_embed_t:
                self.t_embed = nn.Sequential(
                    GaussianFourierProjection(embed_dim=self.embed_t_dim),
                    nn.Linear(self.embed_t_dim, self.embed_t_dim))
            else:
                self.t_embed = GaussianFourierProjection(embed_dim=self.embed_t_dim)
        else:
            self.t_embed = None
            self.embed_t_dim = 1

        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in self.params:
            layer_args["prior_prec"] = self.params["prior_prec"]
        if "std_init" in self.params:
            layer_args["std_init"] = self.params["std_init"]

        single_layer = self.params.get("single_layer", False)

        if not single_layer:
            # Build the cfm MLP
            self.net = Subnet(num_layers=params.get("layers_per_block", 8),
                              size_in=self.dim_embedding + self.embed_t_dim + 4,
                              size_out=4,
                              internal_size=params.get("internal_size", 512),
                              dropout=params.get("dropout", 0.0),
                              activation=params.get("activation", nn.SiLU),
                              layer_class=layer_class,
                              layer_args=layer_args,
                              )

            if self.bayesian:
                self.bayesian_layers.extend(
                    layer for layer in self.net.layer_list if isinstance(layer, VBLinear)
                )
        else:
            self.net = layer_class(self.dim_embedding + self.embed_t_dim + 4, 4)
            if self.bayesian:
                self.bayesian_layers.append(
                    self.net
                )

        self.individual_nets = self.params.get("individual_nets", False)
        if self.individual_nets:
            if not single_layer:
                self.net_3d = Subnet(num_layers=params.get("layers_per_block", 8),
                                  size_in=self.dim_embedding + self.embed_t_dim + 3,
                                  size_out=3,
                                  internal_size=params.get("internal_size", 512),
                                  dropout=params.get("dropout", 0.0),
                                  activation=params.get("activation", nn.SiLU),
                                  layer_class=layer_class,
                                  layer_args=layer_args,
                                  )

                if self.bayesian:
                    self.bayesian_layers.extend(
                        layer for layer in self.net_3d.layer_list if isinstance(layer, VBLinear)
                    )
            else:
                self.net_3d = layer_class(self.dim_embedding + self.embed_t_dim + 3, 3)
                if self.bayesian:
                    self.bayesian_layers.append(
                        self.net_3d
                    )

        self.loss_fct = torch.nn.MSELoss()

    def compute_embedding(
        self, p: torch.Tensor,
            n_particles: int,
            embedding_net: Optional[nn.Module]
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(n_particles, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        if embedding_net is None:
            n_rest = self.dim_embedding - n_particles - p.shape[-1]
            assert n_rest >= 0
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            return torch.cat((p, one_hot, zeros), dim=2)
        else:
            return self.positional_encoding(embedding_net(p))

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            log probabilities, shape (n_events, )
        """
        batch_size = x.size(0)
        n_particles = x.size(1)
        dtype = x.dtype
        device = x.device

        xp = nn.functional.pad(x, (0, 0, 1, 0))
        embedding = self.transformer(
            src=self.compute_embedding(
                c,
                n_particles=self.n_particles_c,
                embedding_net=self.c_embed,
            ),
            tgt=self.compute_embedding(
                xp,
                n_particles=self.n_particles_in + 1,
                embedding_net=self.x_embed,
            ),
            tgt_mask=torch.ones(
                (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
            ).triu(diagonal=1),
        ).detach()

        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                # Prepare the network inputs
                x_t = state[..., :-1].requires_grad_(True)
                t_torch = t * torch.ones((batch_size, 1), dtype=dtype, device=device).requires_grad_(False)
                if self.t_embed is not None:
                    t_torch = self.t_embed(t_torch)

                temp_logp = torch.zeros((batch_size, n_particles, 1), dtype=dtype, device=device)
                temp_v = torch.zeros((batch_size, n_particles, 4), dtype=dtype, device=device)

                for i in range(n_particles):
                    e_p = embedding[:, i, :].requires_grad_(False)
                    if i < 2:
                        x_p = x_t[:, i, :-1].requires_grad_(True)
                        v = self.net_3d(torch.cat([t_torch, x_p, e_p], dim=-1))
                        if self.hutch:
                            dlogp_dt = -hutch_trace2(v, x_p).view(-1, 1)
                        else:
                            dlogp_dt = -autograd_trace(v, x_p).view(-1, 1)
                        v = torch.cat([v, torch.zeros((batch_size, 1), dtype=dtype, device=device)], dim=-1)
                    else:
                        x_p = x_t[:, i].requires_grad_(True)
                        v = self.net(torch.cat([t_torch, x_p, e_p], dim=-1))
                        if self.hutch:
                            dlogp_dt = -hutch_trace2(v, x_p).view(-1, 1)
                        else:
                            dlogp_dt = -autograd_trace(v, x_p).view(-1, 1)

                    temp_logp[:, i] = dlogp_dt.detach()
                    temp_v[:, i] = v.detach()
            return torch.cat([temp_v.detach(), temp_logp.detach()], dim=-1)

        # Set initial conditions for the ODE
        logp_diff_1 = torch.zeros((batch_size, n_particles, 1), dtype=dtype, device=device)
        states = torch.cat([x, logp_diff_1], dim=-1)
        # Solve the ODE from t=0 to t=1 from data to noise
        solution = odeint(
            net_wrapper,
            states,
            torch.tensor([0, 1], dtype=dtype, device=device),
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
            options=dict(step_size=self.step_size)
            )
        # Extract the latent space points and the jacobians
        x_1 = solution[-1, :, :, :-1].detach()
        jac = solution[-1, :, :, -1].detach()
        return (self.latent_log_prob(x_1).squeeze() - jac.squeeze()).sum(dim=-1)

    def log_prob2(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            log probabilities, shape (n_events, )
        """
        batch_size = x.size(0)
        n_particles = x.size(1)
        dtype = x.dtype
        device = x.device

        xp = nn.functional.pad(x, (0, 0, 1, 0))
        embedding = self.transformer(
            src=self.compute_embedding(
                c,
                n_particles=self.n_particles_c,
                embedding_net=self.c_embed,
            ),
            tgt=self.compute_embedding(
                xp,
                n_particles=self.n_particles_in + 1,
                embedding_net=self.x_embed,
            ),
            tgt_mask=torch.ones(
                (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
            ).triu(diagonal=1),
        ).detach().requires_grad_(False)

        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                # Prepare the network inputs
                x_t = state[0].requires_grad_(True)
                t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device).requires_grad_(False)
                if self.t_embed is not None:
                    t_torch = self.t_embed(t_torch)
                # Predict v
                v = net(torch.cat([t_torch, x_t, c], dim=-1))
                # Calculate the jacobian trace
                if self.hutch:
                    dlogp_dt = -hutch_trace(v, x_t).view(-1, 1)
                else:
                    dlogp_dt = -autograd_trace(v, x_t, drop_last=not mass and not self.individual_nets).view(-1, 1)
            return v.detach(), dlogp_dt.detach()

        # Set initial conditions for the ODE
        logp_diff_1 = torch.zeros((batch_size, 1), dtype=dtype, device=device)

        if not self.bayesian_transfer:
            prob = 0
            for p in range(n_particles):
                mass = self.mass_mask[0, p]
                if self.individual_nets and not mass:
                    net = self.net_3d
                    states = (x[:, p, :-1], logp_diff_1)
                else:
                    net = self.net
                    states = (x[:, p, :], logp_diff_1)
                c = embedding[:, p, :]

                # Solve the ODE from t=0 to t=1 from data to noise
                x_1_p, jac_p = odeint(
                    net_wrapper,
                    states,
                    torch.tensor([0, 1], dtype=dtype, device=device),
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.method,
                    options=dict(step_size=self.step_size)
                    )
            # Extract the latent space points and the jacobians
                prob += -jac_p[-1].detach().squeeze()
                if mass:
                    prob += self.latent_log_prob(x_1_p[-1].detach())
                else:
                    prob += self.latent_log_prob(x_1_p[-1][:, :3].detach())
            return prob

        else:
            log_probs = []
            for layer in self.bayesian_layers:
                layer.map = True
            prob = 0
            for p in range(n_particles):
                mass = self.mass_mask[0, p]
                if self.individual_nets and not mass:
                    net = self.net_3d
                    states = (x[:, p, :-1], logp_diff_1)
                else:
                    net = self.net
                    states = (x[:, p, :], logp_diff_1)
                c = embedding[:, p, :]

                # Solve the ODE from t=0 to t=1 from data to noise
                x_1_p, jac_p = odeint(
                    net_wrapper,
                    states,
                    torch.tensor([0, 1], dtype=dtype, device=device),
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.method,
                    options=dict(step_size=self.step_size)
                )
                # Extract the latent space points and the jacobians
                prob += -jac_p[-1].detach().squeeze()
                if mass:
                    prob += self.latent_log_prob(x_1_p[-1].detach())
                else:
                    prob += self.latent_log_prob(x_1_p[-1][:, :3].detach())
            log_probs.append(prob)

            for layer in self.bayesian_layers:
                layer.map = False

            for random_state in self.random_states:
                self.import_random_state(random_state)
                prob = 0
                for p in range(n_particles):
                    mass = self.mass_mask[0, p]
                    if self.individual_nets and not mass:
                        net = self.net_3d
                        states = (x[:, p, :-1], logp_diff_1)
                    else:
                        net = self.net
                        states = (x[:, p, :], logp_diff_1)
                    c = embedding[:, p, :]

                    # Solve the ODE from t=0 to t=1 from data to noise
                    x_1_p, jac_p = odeint(
                        net_wrapper,
                        states,
                        torch.tensor([0, 1], dtype=dtype, device=device),
                        atol=self.atol,
                        rtol=self.rtol,
                        method=self.method,
                        options=dict(step_size=self.step_size)
                    )
                    # Extract the latent space points and the jacobians
                    prob += -jac_p[-1].detach().squeeze()
                    if mass:
                        prob += self.latent_log_prob(x_1_p[-1].detach())
                    else:
                        prob += self.latent_log_prob(x_1_p[-1][:, :3].detach())
                log_probs.append(prob)

            return torch.stack(log_probs, dim=0)

    def latent_log_prob(self, z: torch.Tensor) -> Union[torch.Tensor, float]:
        """
        Returns the log probability for a tensor in latent space

        Args:
            z: latent space tensor, shape (n_events, dims_in)
        Returns:
            log probabilities, shape (n_events, )
        """
        if self.latent_space == "gaussian":
            return -(z**2 / 2 + 0.5 * math.log(2 * math.pi)).sum(dim=-1)
        elif self.latent_space == "uniform":
            return 0.0

    def sample(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
        """
        x = torch.zeros((c.shape[0], 1, 4), device=c.device, dtype=c.dtype)
        one_hot_mult = c[:, 0, -self.n_one_hot:].bool()
        c_emb = self.compute_embedding(
            c, n_particles=self.n_particles_c, embedding_net=self.c_embed
        )
        jac = 0
        for i in range(self.n_particles_in):
            if i >= self.min_n_particles:
                mask = one_hot_mult[:,i - self.min_n_particles:].any(dim=1)
                x_masked = x[mask]
                c_masked = c_emb[mask]
            else:
                x_masked = x
                c_masked = c_emb
            embedding = self.transformer(
                src=c_masked,
                tgt=self.compute_embedding(
                    x_masked,
                    n_particles=self.n_particles_in + 1,
                    embedding_net=self.x_embed,
                ),
                tgt_mask=torch.ones(
                    (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1),
            )
            x_new, jac_new = self.sample_particle(
                embedding[:, -1:, :], self.mass_mask[:, i : i + 1]
            )
            if i >= self.min_n_particles:
                jac[mask] += jac_new
                x = torch.cat((x, torch.zeros_like(x[:,:1,:])), dim=1)
                x[mask,-1:] = x_new
            else:
                jac += jac_new
                x = torch.cat((x, x_new), dim=1)

        return x[:, 1:], jac

    def sample_particle(
        self, c: torch.Tensor, mass_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        if self.individual_nets and not mass_mask:
            net = self.net_3d
            x_1 = torch.randn((batch_size, 3), device=device, dtype=dtype)
        else:
            net = self.net
            x_1 = torch.randn((batch_size, 4), device=device, dtype=dtype)

        # NN wrapper to pass into ODE solver
        def net_wrapper(t, x_t):
            t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            if self.t_embed is not None:
                t_torch = self.t_embed(t_torch)
            v = net(torch.cat([t_torch.reshape(batch_size, -1), x_t, c.squeeze()], dim=-1))
            return v

        # Solve ODE from t=1 to t=0
        with torch.no_grad():
            x_t = odeint(
                net_wrapper,
                x_1,
                torch.tensor([1, 0], dtype=dtype, device=device),
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
                options=dict(step_size=self.step_size)
            )
        # Extract generated samples and mask out masses if not needed
        x_0 = x_t[-1]

        if self.individual_nets and not mass_mask:
            x_0 = torch.cat([x_0,
                             torch.zeros((x_0.size(0), 1), device=device, dtype=dtype)], dim=-1)
        else:
            zero = torch.tensor(0., dtype=dtype, device=device)
            x_0[:, -1] = torch.where(mass_mask, x_0[:, -1], zero)

        jac = torch.zeros(c.shape[:-1], device=device, dtype=dtype)
        return x_0.unsqueeze(1), jac

    def sample_with_probs(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
        """
        x = torch.zeros((c.shape[0], 1, 4), device=c.device, dtype=c.dtype)
        c_emb = self.transformer.encoder(self.compute_embedding(
            c, n_particles=self.n_particles_c, embedding_net=self.c_embed
        ))
        jac = 0
        for i in range(self.n_particles_in):
            embedding = self.transformer.decoder(
                tgt=self.compute_embedding(
                    x,
                    n_particles=self.n_particles_in + 1,
                    embedding_net=self.x_embed,
                ),
                memory=c_emb,
                tgt_mask=torch.ones(
                    (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1),
            )
            x_new, jac_new = self.sample_particle_with_probs(
                embedding[:, -1:, :], self.mass_mask[:, i : i + 1]
            )
            jac += jac_new
            x = torch.cat((x, x_new), dim=1)
        return x[:, 1:], jac

    def sample_particle_with_probs(
        self, c: torch.Tensor, mass_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        if self.individual_nets and not mass_mask:
            net = self.net_3d
            x_1 = torch.randn((batch_size, 3), device=device, dtype=dtype)
        else:
            net = self.net
            x_1 = torch.randn((batch_size, 4), device=device, dtype=dtype)

        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                x_t = state[0].requires_grad_(True)
                t_torch = (t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)).requires_grad_(False)
                if self.t_embed is not None:
                    t_torch = self.t_embed(t_torch)
                # Predict v
                v = net(torch.cat([t_torch.reshape(batch_size, -1), x_t, c.squeeze()], dim=-1))
                if self.hutch:
                    dlogp_dt = -hutch_trace(v, x_t).view(-1, 1)
                else:
                    dlogp_dt = -autograd_trace(v, x_t).view(-1, 1)
            return v.detach(), dlogp_dt.detach()

        logp_diff_1 = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        states = (x_1, logp_diff_1)
        c.requires_grad_(False)

        # Solve ODE from t=1 to t=0
        x_t, logp_diff_t = odeint(
            net_wrapper,
            states,
            torch.tensor([1, 0], dtype=dtype, device=device),
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
            options=dict(step_size=self.step_size)
        )
        # Extract generated samples and mask out masses if not needed
        x_0 = x_t[-1].detach()
        jac = logp_diff_t[-1].detach()

        if self.individual_nets and not mass_mask:
            x_0 = torch.cat([x_0,
                             torch.zeros((x_0.size(0), 1), device=device, dtype=dtype)], dim=-1)
        else:
            zero = torch.tensor(0., dtype=dtype, device=device)
            x_0[:, -1] = torch.where(mass_mask, x_0[:, -1], zero)

        return x_0.unsqueeze(1), jac

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
        if self.min_n_particles < self.n_particles_in:
            one_hot_mult = c[:, 0, -self.n_one_hot:]
            multiplicity_mask = torch.cat((
                torch.ones_like(x[:,:-self.n_one_hot,0], dtype=torch.bool),
                (1 - one_hot_mult.cumsum(dim=1) + one_hot_mult).bool()
            ), dim=1)
        else:
            multiplicity_mask = torch.ones_like(x[:,:,0], dtype=torch.bool)

        xp = nn.functional.pad(x[:, :-1], (0, 0, 1, 0))
        embedding = self.transformer(
            src=self.compute_embedding(
                c,
                n_particles=self.n_particles_c,
                embedding_net=self.c_embed,
            ),
            tgt=self.compute_embedding(
                xp,
                n_particles=self.n_particles_in + 1,
                embedding_net=self.x_embed,
            ),
            tgt_mask=torch.ones(
                (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
            ).triu(diagonal=1),
        )

        # Sample noise variables
        x_1 = torch.randn(x.size(), dtype=x.dtype, device=x.device)
        # Sample time steps
        t = torch.rand((x.size(0), x.size(1), 1), dtype=x.dtype, device=x.device)
        # Calculate point and derivative on trajectory
        x_t = (1 - t) * x + t * x_1
        x_t_dot = -x + x_1

        # Predict v
        if self.t_embed is not None:
            t = self.t_embed(t)

        if self.individual_nets:
            v_pred3 = self.net_3d(torch.cat([t[:, :2, :],
                                             x_t[:, :2, :-1],
                                             embedding[:, :2, :]], dim=-1))
            v_pred4 = self.net(torch.cat([t[:, 2:, :],
                                             x_t[:, 2:, :],
                                             embedding[:, 2:, :]], dim=-1))

            mse = torch.cat([((v_pred3 - x_t_dot[:, :2, :-1])**2).sum(dim=-1),
                             ((v_pred4 - x_t_dot[:, 2:, :])**2).sum(dim=-1)], dim=1)
        else:
            v_pred = self.net(torch.cat([t, x_t, embedding], dim=-1))
            # Mask out masses if not needed
            v_pred[:, :, -1] = v_pred[:, :, -1] * self.mass_mask
            x_t_dot[:, :, -1] = x_t_dot[:, :, -1] * self.mass_mask

            mse = ((v_pred - x_t_dot)**2).sum(dim=-1)
        # Calculate loss, Mask out according to particle multiplicity
        zero = torch.tensor(0., dtype=x.dtype, device=x.device)
        cfm_loss = torch.where(multiplicity_mask, mse, zero).mean()

        if self.bayesian:
            kl_loss = self.bayesian_factor * kl_scale * self.kl() / (4 * x.shape[1] - self.mass_mask.sum())
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

    def kl(self) -> torch.Tensor:
        """
        Compute the KL divergence between weight prior and posterior

        Returns:
            Scalar tensor with KL divergence
        """
        assert self.bayesian
        return sum(layer.kl() for layer in self.bayesian_layers)


class TransfusionParallel(TransfusionAR):

    def __init__(self, n_particles_in: int, n_particles_c: int, params: dict):
        super().__init__(n_particles_in, n_particles_c, params)

        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in self.params:
            layer_args["prior_prec"] = self.params["prior_prec"]
        if "std_init" in self.params:
            layer_args["std_init"] = self.params["std_init"]

        single_layer = self.params.get("single_layer", False)
        self.bayesian_layers = []
        if not single_layer:
            # Build the cfm MLP
            self.net = Subnet(num_layers=params.get("layers_per_block", 8),
                              size_in=self.dim_embedding + self.embed_t_dim,
                              size_out=4,
                              internal_size=params.get("internal_size", 512),
                              dropout=params.get("dropout", 0.0),
                              activation=params.get("activation", nn.SiLU),
                              layer_class=layer_class,
                              layer_args=layer_args,
                              )

            if self.bayesian:
                self.bayesian_layers.extend(
                    layer for layer in self.net.layer_list if isinstance(layer, VBLinear)
                )
        else:
            self.net = layer_class(self.dim_embedding + self.embed_t_dim, 4)
            if self.bayesian:
                self.bayesian_layers.append(
                    self.net
                )

        self.individual_nets = self.params.get("individual_nets", False)
        if self.individual_nets:
            if not single_layer:
                self.net_3d = Subnet(num_layers=params.get("layers_per_block", 8),
                                  size_in=self.dim_embedding + self.embed_t_dim,
                                  size_out=3,
                                  internal_size=params.get("internal_size", 512),
                                  dropout=params.get("dropout", 0.0),
                                  activation=params.get("activation", nn.SiLU),
                                  layer_class=layer_class,
                                  layer_args=layer_args,
                                  )

                if self.bayesian:
                    self.bayesian_layers.extend(
                        layer for layer in self.net_3d.layer_list if isinstance(layer, VBLinear)
                    )
            else:
                self.net_3d = layer_class(self.dim_embedding + self.embed_t_dim, 3)
                if self.bayesian:
                    self.bayesian_layers.append(
                        self.net_3d
                    )

    def sample(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
        """

        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        # NN wrapper to pass into ODE solver
        def net_wrapper(t, x_t):
            t_torch = t * torch.ones_like(x_t[:, :, [0]], dtype=dtype, device=device)
            if self.t_embed is not None:
                t_torch = self.t_embed(t_torch)
            embedding = self.transformer.decoder(
                self.compute_embedding(torch.cat([x_t, t_torch], dim=-1), n_particles=self.n_particles_in, embedding_net=self.x_embed),
                c_embedding)
            if not self.individual_nets:
                v = self.net(torch.cat([t_torch, embedding], dim=-1))
            else:
                v3 = self.net_3d(torch.cat([t_torch[:, :2], embedding[:, :2]], dim=-1))
                v3 = torch.cat([v3, torch.zeros((v3.size(0), 2, 1), device=device, dtype=dtype)], dim=-1)
                v4 = self.net(torch.cat([t_torch[:, 2:], embedding[:, 2:]], dim=-1))
                v = torch.cat([v3, v4], dim=1)
            return v

        # Calculate the time-independent encoding of the condition
        c_embedding = self.transformer.encoder(self.compute_embedding(
            c, n_particles=self.n_particles_c, embedding_net=self.c_embed
        ))

        # Sample from the latent distribution
        x_1 = torch.randn((batch_size, self.n_particles_in, 4), device=device, dtype=dtype)

        # Solve ODE from t=1 to t=0
        with torch.no_grad():
            x_t = odeint(
                net_wrapper,
                x_1,
                torch.tensor([1, 0], dtype=dtype, device=device),
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
                options=dict(step_size=self.step_size)
            )

        # Extract generated samples and mask out masses if not needed
        x_0 = x_t[-1]
        x_0[:, :, -1] = x_0[:, :, -1] * self.mass_mask
        jac = torch.zeros(c.shape[:-1], device=device, dtype=dtype)  # TODO jacobian
        return x_0, jac

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

        multiplicity_mask = torch.ones_like(x[:, :, 0], dtype=torch.bool)

        # Sample noise variables
        x_1 = torch.randn(x.size(), dtype=x.dtype, device=x.device)
        # Sample time steps
        t = torch.rand((x.size(0), x.size(1), 1), dtype=x.dtype, device=x.device)
        # Calculate point and derivative on trajectory
        x_t = (1 - t) * x + t * x_1
        x_t_dot = -x + x_1

        if self.t_embed is not None:
            t = self.t_embed(t)

        embedding = self.transformer(
            src=self.compute_embedding(
                c,
                n_particles=self.n_particles_c,
                embedding_net=self.c_embed,
            ),
            tgt=self.compute_embedding(
                torch.cat([x_t, t], dim=-1),
                n_particles=self.n_particles_in,
                embedding_net=self.x_embed,
            )
        )

        if self.individual_nets:
            v_pred3 = self.net_3d(torch.cat([t[:, :2, :],
                                             embedding[:, :2, :]], dim=-1))
            v_pred4 = self.net(torch.cat([t[:, 2:, :],
                                             embedding[:, 2:, :]], dim=-1))

            mse = torch.cat([((v_pred3 - x_t_dot[:, :2, :-1])**2).sum(dim=-1),
                             ((v_pred4 - x_t_dot[:, 2:, :])**2).sum(dim=-1)], dim=1)
        else:
            v_pred = self.net(torch.cat([t, embedding], dim=-1))
            # Mask out masses if not needed
            v_pred[:, :, -1] = v_pred[:, :, -1] * self.mass_mask
            x_t_dot[:, :, -1] = x_t_dot[:, :, -1] * self.mass_mask

            mse = ((v_pred - x_t_dot)**2).sum(dim=-1)
        # Calculate loss, Mask out according to particle multiplicity
        zero = torch.tensor(0., dtype=x.dtype, device=x.device)
        cfm_loss = torch.where(multiplicity_mask, mse, zero).mean()

        if self.bayesian:
            kl_loss = self.bayesian_factor * kl_scale * self.kl() / (4 * x.shape[1] - self.mass_mask.sum())
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

