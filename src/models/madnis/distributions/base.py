"""Implementation of distribution class."""

import torch
from typing import Tuple, Optional
from torch import nn

from . import typechecks

from ..mappings.base import Mapping

class Distribution(nn.Module):
    """Base class for all distribution objects."""

    def __init__(self, dims_in: int, dims_c: Optional[int]):
        super().__init__()
        self.dims_in = dims_in
        self.dims_c = dims_c

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None):
        """Forward method just calls the prob function"""
        return self.prob(x, condition)

    def _check_shape(self, x, condition):
        if len(x.shape) != 2 or x.shape[1] != self.dims_in:
            raise ValueError("Wrong input shape")
        if self.dims_c is None:
            if condition is not None:
                raise ValueError("Distribution does not expect condition")
        else:
            if condition is None:
                raise ValueError("Distribution expects condition")
            if len(condition.shape) != 2 or condition.shape[1] != self.dims_c:
                raise ValueError("Wrong condition shape")

    def log_prob(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """Calculate log probability of the distribution.

        Args:
            x: Tensor, shape (batch_size, ...).
            condition (optional): None or Tensor, shape (batch_size, ...).
                Must have the same number or rows as input.
                If None, the condition is ignored.

        Returns:
            log_prob: Tensor of shape (batch_size,), the log probability of the inputs.
        """
        self._check_shape(x, condition)
        return self._call_log_prob(x, condition)

    def _call_log_prob(self, x, condition):
        """Wrapper around _log_prob"""
        if hasattr(self, "_log_prob"):
            return self._log_prob(x, condition)
        if hasattr(self, "_prob"):
            return torch.log(self._prob(x, condition))
        raise NotImplementedError("log_prob is not implemented")

    def prob(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """Calculate probability of the distribution.

        Args:
            x: Tensor, shape (batch_size, ...).
            condition: None or Tensor, shape (batch_size, ...).
                Must have the same number or rows as input.
                If None, the condition is ignored.

        Returns:
            prob: Tensor of shape (batch_size,), the probability of the inputs.
        """
        self._check_shape(x, condition)
        return self._call_prob(x, condition)

    def _call_prob(self, x, condition):
        """Wrapper around _prob."""
        if hasattr(self, "_prob"):
            return self._prob(x, condition)
        if hasattr(self, "_log_prob"):
            return torch.exp(self._log_prob(x, condition))
        raise NotImplementedError("prob is not implemented")

    def sample(
        self,
        num_samples: int,
        condition: torch.Tensor = None,
        batch_size: int = None,
    ) -> torch.Tensor:
        """Generates samples from the distribution. Samples can be generated in batches.
        Args:
            num_samples: int, number of samples to generate.
            condition: Tensor or None, conditioning variables of shape (condition_size, ...).
                If None, the condition is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.
        Returns:
            samples: Tensor with shape (num_samples, ...) if condition is None, or
            (condition_size, num_samples, ...) if condition is given.
        """
        if not typechecks.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if batch_size is None:
            return self._sample(num_samples, condition)

        else:
            if not typechecks.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, condition) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, condition))

            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, condition):
        raise NotImplementedError("sampling is not implemented")

    def sample_and_log_prob(
        self,
        num_samples: int,
        condition: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates samples from the distribution together with with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            condition: Tensor or None, conditioning variables of shape (condition_size, ...).
                If None, the condition is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape (num_samples, ...).
                * A Tensor containing the log probabilities of the samples, with shape
                  (num_samples,)
        """
        samples = self.sample(num_samples, condition)
        log_prob = self.log_prob(samples, condition)
        return samples, log_prob

    def sample_and_prob(
        self,
        num_samples: int,
        condition: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates samples from the distribution together with with their probability.

        Args:
            num_samples: int, number of samples to generate.
            condition: Tensor or None, conditioning variables of shape (condition_size, ...).
                If None, the condition is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape (num_samples, ...).
                * A Tensor containing the probabilities of the samples, with shape
                  (num_samples,).
        """
        samples = self.sample(num_samples, condition)
        prob = self.prob(samples, condition)
        return samples, prob

    def apply_mapping(self, mapping: Mapping):
        """Constructs a new Distribution by combining this Distribution with a Mapping.

        Args:
            mapping: Mapping, the Mapping to apply to the Distribution

        Returns:
            A MappedDistribution that combines self and mapping
        """
        return MappedDistribution(self.dims_in, self.dims_c, self, mapping)


class MappedDistribution(Distribution):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        base_dist: Distribution,
        mapping: Mapping
    ):
        super().__init__(dims_in, dims_c)
        self.base_dist = base_dist
        self.mapping = mapping

        if dims_in != base_dist.dims_in:
            raise ValueError("Input dimension incompatible with base distribution")
        if dims_in != mapping.dims_in:
            raise ValueError("Input dimension incompatible with mapping")
        if base_dist.dims_c is not None and dims_c != base_dist.dims_c:
            raise ValueError("Conditional dimension incompatible with base distribution")
        if mapping.dims_c is not None and dims_c != mapping.dims_c:
            raise ValueError("Conditional dimension incompatible with mapping")

    def _prob(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        z, log_det = self.mapping.forward(x, condition)
        return self.base_dist.prob(z, condition) * torch.exp(log_det)

    def _log_prob(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        z, log_det = self.mapping.forward(x, condition)
        base_prob = self.base_dist.log_prob(z, condition)
        return base_prob + log_det

    def _sample(self, num_samples: int, condition: torch.Tensor = None) -> torch.Tensor:
        z = self.base_dist.sample(num_samples, condition)
        x, _ = self.mapping.inverse(z, condition)
        return x

    def sample_and_log_prob(
        self,
        num_samples: int,
        condition: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, log_prob, _ = self.sample_and_log_prob_latent(num_samples, condition)
        return x, log_prob

    def sample_and_log_prob_latent(
        self,
        num_samples: int,
        condition: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, log_prob = self.base_dist.sample_and_log_prob(num_samples, condition)
        x, log_det = self.mapping.inverse(z, condition)
        return x, log_prob - log_det, z

    def sample_and_prob(
        self,
        num_samples: int,
        condition: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z, prob = self.base_dist.sample_and_prob(num_samples, condition)
        x, log_det = self.mapping.inverse(z, condition)
        return x, prob * torch.exp(-log_det)

    def apply_mapping(self, mapping: Mapping):
        dims_c = mapping.dims_c if self.dims_c is None else self.dims_c
        return MappedDistribution(
            self.dims_in,
            dims_c,
            self.base_dist,
            self.mapping.apply_mapping(mapping)
        )
