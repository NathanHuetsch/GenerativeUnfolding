"""
Implementation of distributions for sampling and for importance sampling
"""

from typing import Union, Tuple, Optional
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .base import Mapping, InverseMapping
from ..distributions import typechecks


class Sigmoid(Mapping):
    """Map reals to unit hypercube"""

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        temperature=1.0,
        epsilon=1e-8,
        learn_temperature=False,
    ):
        super().__init__(dims_in, dims_c)
        self.epsilon = epsilon
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor(temperature))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))

    def _forward(self, x, condition, **kwargs):
        # Note: the condition is ignored.
        del condition
        x = self.temperature * x
        z = torch.sigmoid(x)
        logdet = torch.sum(
            torch.log(self.temperature) - F.softplus(-x) - F.softplus(x),
            dim = 1
        )
        return z, logdet

    def _inverse(self, z, condition, **kwargs):
        # Note: the condition is ignored.
        del condition
        z = torch.clamp(z, self.epsilon, 1 - self.epsilon)
        x = (1 / self.temperature) * (torch.log(z) - torch.log1p(-z))
        logdet = -torch.sum(
            torch.log(self.temperature)
            - F.softplus(-self.temperature * x)
            - F.softplus(self.temperature * x),
            dim = 1
        )
        return x, logdet

    def _log_det(self, x_or_z, condition=None, inverse=False, **kwargs):
        # Note: the condition is ignored.
        del condition
        if inverse:
            z = torch.clamp(x_or_z, self.epsilon, 1 - self.epsilon)
            x = (1 / self.temperature) * (torch.log(z) - torch.log1p(-z))
            log_det = -torch.sum(
                torch.log(self.temperature)
                - F.softplus(-self.temperature * x)
                - F.softplus(self.temperature * x),
                dim = 1
            )
            return log_det
        else:
            x = self.temperature * x_or_z
            log_det = torch.sum(
                torch.log(self.temperature) - F.softplus(-x) - F.softplus(x),
                dim = 1
            )
            return log_det


class Logit(InverseMapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        temperature=1.0,
        epsilon=1e-8,
        learn_temperature=False,
    ):
        super().__init__(
            dims_in,
            dims_c,
            Sigmoid(dims_in, dims_c, temperature, epsilon, learn_temperature),
        )


class CauchyCDF(Mapping):
    """Multi-dimensional Cauchy CDF. Its jacobian determinant
    coincides with a multivariate and diagonal Cauchy distribution."""

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        mean: Union[torch.Tensor, float] = 0.0,
        gamma: Union[torch.Tensor, float] = 1.0,
    ):
        """
        Args:
            means (List[torch.Tensor]): peak locations.
            gammas (List[float]): scale parameters (FWHM)
        """
        super().__init__(dims_in, dims_c)

        # Define mean
        if isinstance(mean, (int, float)):
            self.register_buffer("mean", torch.full((1, dims_in), mean))
        elif isinstance(mean, torch.Tensor):
            typechecks.check_shape(mean, (1, dims_in))
            self.register_buffer("mean", mean)
        else:
            raise TypeError("Unexpected type of mean")

        # Define log_std
        if isinstance(gamma, (int, float)):
            self.register_buffer("gamma", torch.full((1, dims_in), gamma))
        elif isinstance(gamma, torch.Tensor):
            typechecks.check_shape(gamma, (1, dims_in))
            self.register_buffer("gamma", gamma)
        else:
            raise TypeError("Unexpected type of gamma")

        self.register_buffer("pi", torch.tensor(np.pi))

    def _forward(self, x, condition, **kwargs):
        """The forward pass of the mapping"""
        # Note: the condition is ignored.
        del condition

        z = 1 / self.pi * torch.atan((x - self.mean) / self.gamma) + 0.5
        logdet = self.log_det(x)
        return z, logdet

    def _inverse(self, z, condition, **kwargs):
        """The inverse pass of the mapping"""
        # Note: the condition is ignored.
        del condition

        x = self.mean + self.gamma * torch.tan(self.pi * (z - 0.5))
        logdet = self.log_det(x, inverse=True)
        return x, logdet

    def _log_det(self, x_or_z, condition, inverse=False, **kwargs):
        # Note: condition is ingored
        del condition
        log_func = (
            torch.log(self.gamma)
            - torch.log(self.pi)
            - torch.log((x_or_z - self.mean) ** 2 + self.gamma**2)
        )
        return (-1) ** inverse * torch.sum(log_func, dim=1)


class CauchyCDFInverse(InverseMapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        mean: Union[torch.Tensor, float] = 0.0,
        gamma: Union[torch.Tensor, float] = 1.0,
    ):
        super().__init__(dims_in, dims_c, CauchyCDF(dims_in, dims_c, mean, gamma))


class NormalCDF(Mapping):
    """Multi-dimensional NormalCDF. Its jacobian determinant of
    the forward pass coincides with a multivariate
    diagonal Normal distribution.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        mean: Union[torch.Tensor, float] = 0.0,
        log_std: Union[torch.Tensor, float] = 0.0,
    ):
        """
        Args:
            mean: float, location of the peak
            std: float, standard deviation
        """
        super().__init__(dims_in, dims_c)

        # Define mean
        if isinstance(mean, (int, float)):
            self.register_buffer("mean", torch.full((1, dims_in), mean))
        elif isinstance(mean, torch.Tensor):
            typechecks.check_shape(mean, (1, dims_in))
            self.register_buffer("mean", mean)
        else:
            raise TypeError("Unexpected type of mean")

        # Define log_std
        if isinstance(log_std, (int, float)):
            self.register_buffer("log_std", torch.full((1, dims_in), log_std))
        elif isinstance(log_std, torch.Tensor):
            typechecks.check_shape(log_std, (1, dims_in))
            self.register_buffer("log_std", log_std)
        else:
            raise TypeError("Unexpected type of log_std")

        self.register_buffer("pi", torch.tensor(np.pi))

    def _forward(self, x, condition, **kwargs):
        """In this 1-dimensional case
        the forward pass of this mapping coincides
        with cumulative distribution function (cdf)
        of the normal distribution.
        """
        # Note: the condition is ignored.
        del condition

        z = torch.special.ndtr((x - self.mean) * torch.exp(-self.log_std))
        log_det = self.log_det(x, inverse=False)
        return z, log_det

    def _inverse(self, z, condition, **kwargs):
        """In this 1-dimensional case
        the inverse pass of the mapping (distribution)
        coincides with the quantile function.
        """
        # Note: the condition is ignored.
        del condition

        x = torch.special.ndtri(z) * torch.exp(self.log_std) + self.mean
        log_det = self.log_det(x, inverse=True)
        return x, log_det

    def _log_det(self, x_or_z, condition, inverse=False):
        # Note: the condition is ignored.
        del condition
        log_norm = 0.5 * torch.log(2 * self.pi) - self.log_std
        log_inner = -0.5 * torch.exp(-2 * self.log_std) * (x_or_z - self.mean) ** 2
        return (-1) ** inverse * torch.sum(log_inner - log_norm, dim=1)


class NormalCDFInverse(InverseMapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        mean: Union[torch.Tensor, float] = 0.0,
        log_std: Union[torch.Tensor, float] = 0.0,
    ):
        super().__init__(dims_in, dims_c, NormalCDF(dims_in, dims_c, mean, log_std))
