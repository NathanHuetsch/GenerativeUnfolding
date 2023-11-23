""" This module implements utility functions for splines.

These utility functions are used in many different spline types. Having them
all in one location allows for transparency in the code. Some of the common functions
include the ability to ensure that the inputs are in the correct range, to shift inputs
from an arbitrary range to be between zero and 1, etc.

"""

import torch

# pylint: disable=invalid-name, too-many-arguments
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1
    
def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)
