"""
Implementation of distributions for sampling and for importance sampling
"""


import numpy as np
import torch
from torch import nn
from typing import Tuple, Callable, Union, Optional

from .base import Distribution
from . import typechecks


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, dims_in: int, dims_c: Optional[int]):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
            ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
        """
        super().__init__(dims_in, dims_c)
        self.register_buffer(
            "log_norm",
            torch.tensor(0.5 * np.log(2 * np.pi), dtype=torch.get_default_dtype()),
        )

    def _log_prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        log_inner = -0.5 * x**2
        return torch.sum(log_inner - self.log_norm, dim=1)

    def _sample(self, num_samples, condition):
        del condition
        return torch.randn((num_samples, self.dims_in), device=self.log_norm.device)


class Normal(Distribution):
    """A multivariate diagonal Normal with given mean and log_std."""

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        mean: Union[torch.Tensor, float] = 0.0,
        log_std: Union[torch.Tensor, float] = 0.0,
    ):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
                ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
            mean (float, optional): location of the peak. Defaults to 0.
            log_std (float, optional): log of standard deviation. Defaults to 0.
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

        # Define the norm
        self.register_buffer(
            "log_norm",
            torch.tensor(0.5 * np.log(2 * np.pi), dtype=torch.get_default_dtype()),
        )

    def _log_prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        log_norm = self.log_norm + self.log_std
        log_inner = -0.5 * torch.exp(-2 * self.log_std) * ((x - self.mean) ** 2)
        return torch.sum(log_inner - log_norm, dim=1)

    def _sample(self, num_samples, condition):
        del condition
        eps = torch.randn((num_samples, self.dims_in), device=self.log_norm.device)
        return torch.exp(self.log_std) * eps + self.mean


class DiagonalNormal(Distribution):
    """A diagonal multivariate Normal with trainable mean and log_std."""

    def __init__(self, dims_in: int, dims_c: Optional[int]):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
            ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
        """
        super().__init__(dims_in, dims_c)

        # trainable mean and log_std
        self.mean = nn.Parameter(torch.zeros((1, dims_in)))
        self.log_std = nn.Parameter(torch.zeros((1, dims_in)))
        # Define the norm
        self.register_buffer(
            "log_norm",
            torch.tensor(0.5 * np.log(2 * np.pi), dtype=torch.get_default_dtype()),
        )

    def _log_prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        log_norm = self.log_norm + self.log_std
        log_inner = -0.5 * torch.exp(-2 * self.log_std) * ((x - self.mean) ** 2)
        return torch.sum(log_inner - log_norm, dim=1)

    def _sample(self, num_samples, condition):
        del condition
        eps = torch.randn((num_samples, self.dims_in), device=self.log_norm.device)
        return torch.exp(self.log_std) * eps + self.mean


class ConditionalMeanNormal(Distribution):
    """A multivariate Normal with conditional mean and fixed std."""

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        log_std: Union[torch.Tensor, float] = 0.0,
        embedding_net: Callable = None,
    ):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
            ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
            log_std: float or Tensor, log of standard deviation. Defaults to 0.
            embedding_net: callable or None, embedded the condition to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__(dims_in, dims_c)

        # Allow for an encoding net
        if embedding_net is None:
            self.embedding_net = lambda x: x
        else:
            self.embedding_net = embedding_net

        # Define log_std
        if isinstance(log_std, (int, float)):
            self.register_buffer("log_std", torch.full((1, dims_in), log_std))
        elif isinstance(log_std, torch.Tensor):
            typechecks.check_shape(log_std, (1, dims_in))
            self.register_buffer("log_std", log_std)
        else:
            raise TypeError("Unexpected type of log_std")

        # Define the norm
        self.register_buffer(
            "log_norm",
            torch.tensor(0.5 * np.log(2 * np.pi), dtype=torch.get_default_dtype()),
        )

    def _compute_mean(self, condition):
        """Compute the mean from the condition."""
        if condition is None:
            raise ValueError("Condition can't be None.")

        mean = self.embedding_net(condition)
        typechecks.check_dim_shape(mean, (self.dims_in,))
        return mean

    def _log_prob(self, x, condition):
        # compute parameters
        mean = self._compute_mean(condition)

        log_norm = self.log_norm + self.log_std
        log_inner = -0.5 * torch.exp(-2 * self.log_std) * ((x - mean) ** 2)
        return torch.sum(log_inner - log_norm, dim=1)

    def _sample(self, num_samples, condition):
        if condition is None:
            raise ValueError("Condition can't be None.")
        else:
            # compute parameters
            mean = self._compute_mean(condition)
            log_std = self.log_std

            # generate samples
            eps = torch.randn((num_samples, self.dims_in), device=self.log_norm.device)
            return torch.exp(log_std) * eps + mean


class ConditionalDiagonalNormal(Distribution):
    """A diagonal multivariate Normal with conditional mean and log_std.."""

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        embedding_net: Callable = None
    ):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
            ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
            embedding_net: callable or None, encodes the condition to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__(dims_in, dims_c)

        # Allow for an encoding net
        if embedding_net is None:
            self.embedding_net = lambda x: x
        else:
            self.embedding_net = embedding_net
            
        # Define the norm
        self.register_buffer(
            "log_norm",
            torch.tensor(0.5 * np.log(2 * np.pi), dtype=torch.get_default_dtype()),
        )

    def _compute_params(self, condition):
        """Compute the means and log_stds from the condition."""
        if condition is None:
            raise ValueError("Condition can't be None.")

        params = self.embedding_net(condition)
        if params.shape[-1] % 2 != 0:
            raise ValueError(
                "The embedding net must return a tensor which last dimension is even."
            )
        if params.shape[0] != condition.shape[0]:
            raise ValueError(
                "The batch dimension of the parameters is inconsistent with the input."
            )

        mean, log_std = torch.split(params, 2, dim=-1)
        return mean, log_std

    def _log_prob(self, x, condition):
        # compute parameters
        mean, log_std = self._compute_params(condition)
        typechecks.check_shape(mean, x.shape)
        typechecks.check_shape(log_std, x.shape)

        log_norm = self.log_norm + self.log_std
        log_inner = -0.5 * torch.exp(-2 * log_std) * ((x - mean) ** 2)
        return torch.sum(log_inner - log_norm, dim=1)

    def _sample(self, num_samples, condition):
        if condition is None:
            raise ValueError("Condition can't be None.")
        else:
            # compute parameters
            mean, log_std = self._compute_params(condition)

            # generate samples
            eps = torch.randn((num_samples, self.dims_in), device=self.log_norm.device)
            return torch.exp(log_std) * eps + mean
