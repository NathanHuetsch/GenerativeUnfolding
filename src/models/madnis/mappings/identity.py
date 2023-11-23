"""
Implementation of distributions for sampling and for importance sampling
"""

from typing import Tuple
import numpy as np
import torch

from .base import Mapping

class Identity(Mapping):
    """Identity mapping"""

    def _forward(self, x, condition, **kwargs):
        # Note: the condition is ignored.
        del condition
        return x, torch.zeros(x.shape[:1], dtype=x.dtype, device=x.device)

    def _inverse(self, z, condition, **kwargs):
        # Note: the condition is ignored.
        del condition
        return z, torch.zeros(z.shape[:1], dtype=z.dtype, device=z.device)

    def _det(self, x_or_z, condition=None, inverse=False, **kwargs):
        # Note: the condition is ignored.
        del condition
        return torch.ones(x_or_z.shape[:1], dtype=x_or_z.dtype, device=x_or_z.device)
    
    def _log_det(self, x_or_z, condition=None, inverse=False, **kwargs):
        # Note: the condition is ignored.
        del condition
        return torch.zeros(x_or_z.shape[:1], dtype=x_or_z.dtype, device=x_or_z.device)
