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
    x: torch.Tensor, n_frequencies: int
) -> torch.Tensor:
    x_pp = 2*nn.functional.sigmoid(x) - 1
    ret = []
    for i in range(n_frequencies):
        ret.append(torch.sin(math.pi * 2**i * x_pp))
        ret.append(torch.cos(math.pi * 2**i * x_pp))
    return torch.cat(ret, dim=-1)


class Flow(nn.Module):
    def __init__(
        self,
        dims_c: int,
        subnet_constructor: Callable[[int, int], nn.Module],
        num_bins: int
    ):
        super().__init__()
        self.net = subnet_constructor(dims_c, 3 * num_bins + 1)
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
        c: torch.Tensor
    ):
        """
        Args:
            x: shape (n_batch, n_particles, 4)
            c: shape (n_batch, n_particles, dims_c)
            mass_mask: shape (n_batch, n_particles)
            multiplicity_mask: shape (n_batch, n_particles)
        Return:
            Log probability, shape (n_batch, )
        """
        z, jac = self.coupling(
            x[..., -1], self.net(c), rev=False
        )
        prob = -L2PI - z**2 / 2 + jac
        return prob.sum(dim=1)

    def sample(
        self, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(c.shape[:-1], device=c.device, dtype=c.dtype)
        x, jac = self.coupling(z, self.net(c), rev=True)
        return x, jac


class Transfermer(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.dims_in = params["dims_in"]
        self.dims_c = params["dims_c"]
        self.dim_embedding = params["dim_embedding"]
        self.sin_cos_embedding = params.get("sin_cos_embedding")

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
        n_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        print(f"        Network: Transformer with {n_params} parameters")

        self.bayesian = params.get("bayesian", False)
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

        self.flow = Flow(
            dims_c=self.dim_embedding,
            subnet_constructor=subnet_constructor,
            num_bins=params["num_bins"]
        )
        n_params = sum(p.numel() for p in self.flow.parameters() if p.requires_grad)
        print(f"        Network: Flow with {n_params} parameters")

        if params.get("embedding_nets", False):
            embedding_factor = (2 * self.sin_cos_embedding if self.sin_cos_embedding else 1)
            self.encoder_embedding_net = nn.Linear(
                4 * embedding_factor + self.dims_c + self.n_one_hot, self.dim_embedding
            )
            self.decoder_embedding_net = nn.Linear(
                4 * embedding_factor + self.dims_in + 1, self.dim_embedding
            )
        else:
            self.encoder_embedding_net = None
            self.decoder_embedding_net = None

        if self.bayesian:
            self.bayesian_samples = params.get("bayesian_samples", 20)
            self.bayesian_layers = []
            self.bayesian_layers.extend(
                layer for layer in self.flow.net.layer_list if isinstance(layer, VBLinear)
            )
            print(f"        Bayesian set to True, Bayesian layers: ", len(self.bayesian_layers))

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
            pp = sin_cos_embedding(p, self.sin_cos_embedding)
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

        xp = nn.functional.pad(x[:, :-1], (0, 0, 1, 0))
        embedding = self.transformer(
            src=self.compute_embedding(
                c,
                n_particles=self.dims_c,
                embedding_net=self.encoder_embedding_net,
            ),
            tgt=self.compute_embedding(
                xp,
                n_particles=self.dims_in + 1,
                embedding_net=self.decoder_embedding_net,
            ),
            tgt_mask=torch.ones(
                (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
            ).triu(diagonal=1),
        )
        return self.flow(x, embedding)

    def sample(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
            log_prob: log probabilites, shape (n_events, )
        """
        x = torch.zeros((c.shape[0], 1), device=c.device, dtype=c.dtype)
        c_emb = self.compute_embedding(
            c.unsqueeze(-1), n_particles=self.dims_c, embedding_net=self.encoder_embedding_net
        )
        jac = 0
        for i in range(self.dims_in):
            embedding = self.transformer(
                src=c_emb,
                tgt=self.compute_embedding(
                    x.unsqueeze(-1),
                    n_particles=self.dims_in + 1,
                    embedding_net=self.decoder_embedding_net,
                ),
                tgt_mask=torch.ones(
                    (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1),
            )
            x_new, jac_new = self.flow.sample(embedding[:, -1:, :])
            jac += jac_new
            x = torch.cat((x, x_new), dim=1)
        return x[:, 1:], jac

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
        inn_loss = -self.log_prob(x.unsqueeze(-1), c.unsqueeze(-1)).mean() / x.shape[1]

        if self.bayesian:
            kl_loss = kl_scale * self.kl() / x.shape[1]
            loss = inn_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "nll": inn_loss.item(),
                "kl": kl_loss.item(),
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
