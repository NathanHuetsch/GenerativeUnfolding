"""
Implementation of distributions for sampling and for importance sampling
"""

import torch
from torch import nn
from typing import Tuple, Optional

from .base import Distribution


class StandardUniform(Distribution):
    """A multivariate Uniform with boundaries (0,1)."""

    def __init__(self, dims_in: int, dims_c: Optional[int]):
        super().__init__(dims_in, dims_c)
        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("one", torch.ones(1))

    def _log_prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        lb = x.ge(self.zero).type(self.zero.dtype).mean()
        ub = x.le(self.one).type(self.one.dtype).mean()
        return torch.log(lb * ub)

    def _sample(self, num_samples, condition):
        del condition
        return torch.rand((num_samples, self.dims_in), device=self.zero.device)
