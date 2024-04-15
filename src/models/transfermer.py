import math
from typing import Callable, Optional
import torch
import torch.nn as nn
import numpy as np
from .inn import Subnet
from .splines import unconstrained_rational_quadratic_spline
from .layers import VBLinear

L2PI = 0.5 * math.log(2 * math.pi)


def sin_cos_embedding(
    x: torch.Tensor, n_frequencies: int, sigmoid_dims: list = []
) -> torch.Tensor:
    x_pp = x.clone()
    x_pp[:,:,sigmoid_dims] = 2*nn.functional.sigmoid(x[:,:,sigmoid_dims]) - 1
    ret = []
    for i in range(n_frequencies):
        ret.append(torch.sin(math.pi * 2**i * x_pp))
        ret.append(torch.cos(math.pi * 2**i * x_pp))
    return torch.cat(ret, dim=-1)


class ParticleINN(nn.Module):
    def __init__(
        self,
        dims_c: int,
        subnet_constructor: Callable[[int, int], nn.Module],
        num_bins: int,
        pt_eta_phi: bool,
        eta_cut: bool,
    ):
        super().__init__()
        self.pt_eta_phi = pt_eta_phi
        self.eta_cut = eta_cut
        self.net_1 = subnet_constructor(dims_c + 2, 3 * num_bins + 1)
        self.net_2 = subnet_constructor(dims_c + 2, 3 * num_bins + 1)
        self.net_3 = subnet_constructor(dims_c + 2, 3 * num_bins + 1)
        self.net_m = subnet_constructor(dims_c + 3, 3 * num_bins + 1)
        self.bound = 10
        self.coupling = (
            lambda x, c, rev, bound=self.bound, periodic=False:
            unconstrained_rational_quadratic_spline(
                x,
                c,
                rev,
                num_bins=num_bins,
                left=-bound,
                right=bound,
                bottom=-bound,
                top=bound,
                min_bin_width=1e-5,
                min_bin_height=1e-5,
                min_derivative=1e-5,
                periodic=periodic,
                sum_jacobian=False,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mass_mask: torch.Tensor
    ):
        """
        Args:
            x: shape (n_batch, n_particles, 4)
            c: shape (n_batch, n_particles, dims_c)
            mass_mask: shape (n_batch, n_particles)
        Return:
            Log probability, shape (n_batch, )
        """
        z_m, j_m = self.coupling(
            x[..., -1], self.net_m(torch.cat((x[..., :3], c), dim=-1)), rev=False
        )
        z_3, j_3 = self.coupling(
            x[..., 2],
            self.net_3(torch.cat((x[..., :2], c), dim=-1)),
            rev=False,
            periodic=self.pt_eta_phi,
            bound=1 if self.pt_eta_phi else self.bound,
        )
        z_2, j_2 = self.coupling(
            x[..., 1],
            self.net_2(torch.cat((x[..., 0:1], z_3[..., None], c), dim=-1)),
            rev=False,
            bound=1 if self.pt_eta_phi and self.eta_cut else self.bound,
        )
        z_1, j_1 = self.coupling(
            x[..., 0],
            self.net_1(torch.cat((z_2[..., None], z_3[..., None], c), dim=-1)),
            rev=False,
        )
        zero = torch.tensor(0., dtype=x.dtype, device=x.device)
        prob_1 = -L2PI - z_1**2 / 2 + j_1
        if self.pt_eta_phi and self.eta_cut:
            prob_2 = -2 * L2PI + j_2
        else:
            prob_2 = -L2PI - z_2**2 / 2 + j_2
        if self.pt_eta_phi:
            prob_3 = -2 * L2PI + j_3
        else:
            prob_3 = -L2PI - z_3**2 / 2 + j_3
        prob_m = torch.where(mass_mask, -L2PI - z_m**2 / 2 + j_m, zero)
        return (prob_1 + prob_2 + prob_3 + prob_m).sum(dim=1)

    def sample(
        self, c: torch.Tensor, mass_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn((*c.shape[:-1], 4), device=c.device, dtype=c.dtype)
        if self.pt_eta_phi:
            if self.eta_cut:
                z[..., 1] = 2*torch.rand(c.shape[:-1], device=c.device, dtype=c.dtype)-1
            z[..., 2] = 2*torch.rand(c.shape[:-1], device=c.device, dtype=c.dtype)-1
        x_1, j_1 = self.coupling(
            z[..., 0], self.net_1(torch.cat((z[..., 1:3], c), dim=-1)), rev=True
        )
        x_2, j_2 = self.coupling(
            z[..., 1],
            self.net_2(torch.cat((x_1[..., None], z[..., 2:3], c), dim=-1)),
            rev=True,
            bound=1 if self.pt_eta_phi and self.eta_cut else self.bound,
        )
        x_3, j_3 = self.coupling(
            z[..., 2],
            self.net_3(torch.cat((x_1[..., None], x_2[..., None], c), dim=-1)),
            rev=True,
            periodic=self.pt_eta_phi,
            bound=1 if self.pt_eta_phi else self.bound,
        )
        x_m, j_m = self.coupling(
            z[..., 3],
            self.net_m(
                torch.cat((x_1[..., None], x_2[..., None], x_3[..., None], c), dim=-1)
            ),
            rev=True,
        )
        zero = torch.tensor(0., dtype=z.dtype, device=z.device)
        x_m = torch.where(mass_mask, x_m, zero)
        jac = torch.zeros(c.shape[:-1], device=c.device, dtype=c.dtype)  # TODO jacobian
        return torch.stack((x_1, x_2, x_3, x_m), dim=-1), jac


class Transfermer(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        n_particles_in=6
        n_particles_c=6
        self.n_particles_in = n_particles_in
        self.n_particles_c = n_particles_c
        self.dim_embedding = params["dim_embedding"]
        self.sin_cos_embedding = params.get("sin_cos_embedding")
        self.min_n_particles = params.get("min_n_particles", n_particles_in)
        if self.n_particles_in > self.min_n_particles:
            self.n_one_hot = n_particles_in - self.min_n_particles + 1
        else:
            self.n_one_hot = 0
        self.pt_eta_phi = params.get("pt_eta_phi", False)
        self.eta_cut = params.get("eta_cut", False)

        self.bayesian = params.get("bayesian", False)
        self.bayesian_transfer = False
        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in params:
            layer_args["prior_prec"] = params["prior_prec"]
        if "std_init" in params:
            layer_args["std_init"] = params["std_init"]

        subnet_constructor = lambda x_in, x_out: Subnet(
            params.get("layers_per_block", 3),
            x_in,
            x_out,
            internal_size=params.get("internal_size"),
            dropout=params.get("dropout", 0.0),
            layer_class=layer_class,
            layer_args=layer_args,
        )
        self.particle_inn = ParticleINN(
            dims_c=self.dim_embedding,
            subnet_constructor=subnet_constructor,
            num_bins=params["num_bins"],
            pt_eta_phi=self.pt_eta_phi,
            eta_cut=self.eta_cut,
        )

        if self.bayesian:
            self.bayesian_samples = params.get("bayesian_samples", 20)
            self.bayesian_layers = []
            self.bayesian_layers.extend(
                layer for layer in self.particle_inn.net_1.layer_list if isinstance(layer, VBLinear)
            )
            self.bayesian_layers.extend(
                layer for layer in self.particle_inn.net_2.layer_list if isinstance(layer, VBLinear)
            )
            self.bayesian_layers.extend(
                layer for layer in self.particle_inn.net_3.layer_list if isinstance(layer, VBLinear)
            )
            self.bayesian_layers.extend(
                layer for layer in self.particle_inn.net_m.layer_list if isinstance(layer, VBLinear)
            )

        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=params["n_head"],
            num_encoder_layers=params["n_encoder_layers"],
            num_decoder_layers=params["n_decoder_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params.get("dropout", 0.0),
            activation=params.get("activation", "relu"),
            batch_first=True,
        )
        self.mass_mask = nn.Parameter(torch.tensor([[False]*6]), requires_grad=False)
        if params.get("embedding_nets", False):
            embedding_factor = (2 * self.sin_cos_embedding if self.sin_cos_embedding else 1)
            self.encoder_embedding_net = nn.Linear(
                4 * embedding_factor + n_particles_c + self.n_one_hot, self.dim_embedding
            )
            self.decoder_embedding_net = nn.Linear(
                4 * embedding_factor + n_particles_in + 1, self.dim_embedding
            )
        else:
            self.encoder_embedding_net = None
            self.decoder_embedding_net = None
        if self.sin_cos_embedding is not None:
            if not self.pt_eta_phi:
                self.sigmoid_dims = [0,1,2,3]
            elif self.eta_cut:
                self.sigmoid_dims = [0,3]
            else:
                self.sigmoid_dims = [0,1,3]

    def compute_embedding(
        self, p: torch.Tensor, n_particles: int, embedding_net: Optional[nn.Module]
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(n_particles, device=p.device, dtype=p.dtype)[
            None, : p.shape[1], :
        ].expand(p.shape[0], -1, -1)
        if self.sin_cos_embedding is not None:
            pp = sin_cos_embedding(p, self.sin_cos_embedding, self.sigmoid_dims)
        else:
            pp = p

        if embedding_net is None:
            n_rest = self.dim_embedding - n_particles - pp.shape[-1]
            assert n_rest >= 0
            zeros = torch.zeros((*pp.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            return torch.cat((pp, one_hot, zeros), dim=2)
        else:
            return embedding_net(torch.cat((pp, one_hot), dim=2))

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            log probabilities, shape (n_events, )
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
                embedding_net=self.encoder_embedding_net,
            ),
            tgt=self.compute_embedding(
                xp,
                n_particles=self.n_particles_in + 1,
                embedding_net=self.decoder_embedding_net,
            ),
            tgt_mask=torch.ones(
                (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
            ).triu(diagonal=1),
        )
        return self.particle_inn(x, embedding, self.mass_mask)

    def sample(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
            log_prob: log probabilites, shape (n_events, )
        """
        x = torch.zeros((c.shape[0], 1, 4), device=c.device, dtype=c.dtype)
        one_hot_mult = c[:, 0, -self.n_one_hot:].bool()
        c_emb = self.compute_embedding(
            c, n_particles=self.n_particles_c, embedding_net=self.encoder_embedding_net
        )
        jac = 0
        for i in range(self.n_particles_in):
            if i >= self.min_n_particles:
                mask = one_hot_mult[:,i - self.min_n_particles + 1:].any(dim=1)
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
                    embedding_net=self.decoder_embedding_net,
                ),
                tgt_mask=torch.ones(
                    (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1),
            )
            x_new, jac_new = self.particle_inn.sample(
                embedding[:, -1:, :], self.mass_mask[:, i : i + 1]
            )
            if i >= self.min_n_particles:
                jac[mask] += jac_new
                x = torch.cat((x, torch.zeros_like(x[:,:1,:])), dim=1)
                x[mask,-1:] = x_new
            else:
                jac += jac_new
                x = torch.cat((x, x_new), dim=1)
        return x[:, 1:]

    def sample_with_probs(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition
        For INNs this is equivalent to normal sampling
        """
        return self.sample(c)

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
        inn_loss = -self.log_prob(x, c).mean() / (4 * x.shape[1] - self.mass_mask.sum())

        if self.bayesian:
            kl_loss = kl_scale * self.kl() / (4 * x.shape[1] - self.mass_mask.sum())
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
        assert self.bayesian
        self.random_states = [self.sample_random_state() for i in range(self.bayesian_samples)]
