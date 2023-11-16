from typing import Type, Callable, Union, Optional
import time
import math
import torch
import torch.nn as nn
import numpy as np
from .layers import *
from collections import namedtuple
from math import sqrt, prod
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

SurrogateOutput = namedtuple("SurrogateOutput", ["z", "x1", "nll", "surrogate", "regularizations"])
Transform = Callable[[torch.Tensor], torch.Tensor]


class FreeFormFlow(nn.Module):
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
        self.latent_space = self.params.get("latent_space", "gaussian")

        self.beta = params.get("beta", 1.e-4)

        network_params = self.params.get("network_params")
        self.encoder = EmbeddingNet(
            params=network_params,
            size_in=self.dims_in + self.dims_c,
            size_out=self.dims_in
        )

        self.decoder = EmbeddingNet(
            params=network_params,
            size_in=self.dims_in + self.dims_c,
            size_out=self.dims_in
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
        batch_size = x.size(0)
        dtype = x.dtype
        device = x.device
        pass


    def sample(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device
        z = torch.randn((batch_size, self.dims_in), dtype=dtype, device=device)
        x = self.decoder(torch.cat([z, c], dim=-1))

        return x, torch.Tensor([0])

    def sample_with_probs(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        pass

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

        nll_loss, mse_loss = fff_loss(x, c, self.encoder, self.decoder, self.beta)

        if self.bayesian:
            kl_loss = self.bayesian_factor * kl_scale * self.kl() / self.dims_in
            loss = nll_loss + mse_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "nll": nll_loss.item(),
                "mse": mse_loss.item(),
                "kl": kl_loss.item(),
            }

        else:
            loss = nll_loss + mse_loss
            loss_terms = {
                "loss": loss.item(),
                "nll": nll_loss.item(),
                "mse": mse_loss.item()
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


def sample_v(x: torch.Tensor, hutchinson_samples: int):
    """
    Sample a random vector v of shape (*x.shape, hutchinson_samples)
    with scaled orthonormal columns.

    The reference data is used for shape, device and dtype.

    :param x: Reference data.
    :param hutchinson_samples: Number of Hutchinson samples to draw.
    :return:
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])
    if hutchinson_samples > total_dim:
        raise ValueError(f"Too many Hutchinson samples: got {hutchinson_samples}, expected <= {total_dim}")
    v = torch.randn(batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype)
    q = torch.linalg.qr(v).Q.reshape(*x.shape, hutchinson_samples)
    return q * sqrt(total_dim)


def nll_surrogate(x: torch.Tensor,
                  c: torch.Tensor,
                  encode: Transform,
                  decode: Transform,
                  hutchinson_samples: int = 1) -> SurrogateOutput:
    """
    Compute the per-sample surrogate for the negative log-likelihood and the volume change estimator.
    The gradient of the surrogate is the gradient of the actual negative log-likelihood.

    :param x: Input data. Shape: (batch_size, ...)
    :param c: Condition data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    :param decode: Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    :param hutchinson_samples: Number of Hutchinson samples to use for the volume change estimator.
    :return: Per-sample loss. Shape: (batch_size,)
    """
    with torch.set_grad_enabled(True):
        x.requires_grad_()
        c.requires_grad_(False)
        z = encode(torch.cat([x, c], dim=-1))
        surrogate = 0
        vs = sample_v(z, hutchinson_samples)
        for k in range(hutchinson_samples):
            v = vs[..., k]
            with dual_level():
                dual_z = make_dual(z, v)
                dual_x1 = decode(torch.cat([dual_z, c], dim=-1))
                x1, v1 = unpack_dual(dual_x1)
            v2, = grad(z, x, v, create_graph=True)
            surrogate += sum_except_batch(v2 * v1.detach()) / hutchinson_samples
    nll = sum_except_batch((z ** 2)) / 2 - surrogate
    return SurrogateOutput(z, x1, nll, surrogate, {})


def fff_loss(x: torch.Tensor,
             c: torch.Tensor,
             encode: Transform,
             decode: Transform,
             beta,
             hutchinson_samples: int = 1) -> [torch.Tensor, torch.Tensor]:
    """
    Compute the per-sample FFF/FIF loss:
    $$
    \mathcal{L} = \beta ||x - decode(encode(x))||^2 + ||encode(x)||^2 // 2 - \sum_{k=1}^K v_k^T f'(x) stop_grad(g'(z)) v_k
    $$
    where $E[v_k^T v_k] = 1$, and $ f'(x) $ and $ g'(z) $ are the Jacobians of `encode` and `decode`.

    :param x: Input data. Shape: (batch_size, ...)
    :param c: Condition data. Shape: (batch_size, ...)
    :param encode: Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    :param decode: Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    :param beta: Weight of the mean squared error.
    :param hutchinson_samples: Number of Hutchinson samples to use for the volume change estimator.
    :return: Batch-averaged loss terms nll_loss, mse_loss*beta
    """
    surrogate = nll_surrogate(x, c, encode, decode, hutchinson_samples)
    mse = torch.sum((x - surrogate.x1) ** 2, dim=tuple(range(1, len(x.shape))))
    return surrogate.nll.mean(), mse.mean()*beta


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Sum over all dimensions except the first.
    :param x: Input tensor.
    :return: Sum over all dimensions except the first.
    """
    return torch.sum(x.reshape(x.shape[0], -1), dim=1)