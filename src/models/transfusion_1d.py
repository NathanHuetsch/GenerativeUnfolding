import math
from typing import Type, Callable, Union, Optional
import torch
import torch.nn as nn
from torchdiffeq import odeint
from .layers import *
import numpy as np

L2PI = 0.5 * math.log(2 * math.pi)


class TransfusionAR(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.dims_in = params["dims_in"]
        self.dims_c = params["dims_c"]
        self.dim_embedding = params["dim_embedding"]

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

        self.bayesian = params.get("bayesian", False)
        self.bayesian_samples = params.get("bayesian_samples", 20)
        self.bayesian_layers = []
        self.bayesian_factor = params.get("bayesian_factor", 1)
        self.bayesian_transfer = False

        self.latent_space = self.params.get("latent_space", "gaussian")
        self.pt_eta_phi = params.get("pt_eta_phi", False)

        self.hutch = self.params.get("hutch", False)
        self.method = self.params.get("method", "dopri5")
        self.rtol = self.params.get("rtol", 1.e-3)
        self.atol = self.params.get("atol", 1.e-5)
        self.step_size = self.params.get("step_size", 0.01)
        print(f"Using ODE method {self.method}, step_size {self.step_size}, hutch {self.hutch}", flush=True)


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

        self.net = Subnet(self.params, conditional=True)

        if self.bayesian:
            self.bayesian_layers.extend(
                layer for layer in self.net.layer_list if isinstance(layer, VBLinear)
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
