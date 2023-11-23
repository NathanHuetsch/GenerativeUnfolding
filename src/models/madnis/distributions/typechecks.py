"""Functions that check types."""

import torch
from typing import Tuple, Union


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False


def check_shape(
    input: torch.Tensor,
    target_shape: Union[Tuple[int], torch.Size],
):
    if input.shape != target_shape:
        raise ValueError(
            f"Expected input of shape {input.shape}, but got {target_shape}"
        )


def check_dim_shape(
    input: torch.Tensor,
    target_shape: Union[Tuple[int], torch.Size],
):
    if input.shape[1:] != target_shape:
        raise ValueError(
            f"Expected input of shape {input.shape[1:]}, but got {target_shape}"
        )


def check_batch_shape(
    input: torch.Tensor,
    batch_shape: int,
):
    if input.shape[0] != batch_shape:
        raise ValueError(
            f"Expected {input.shape[0]} number of elements, but got {batch_shape}"
        )
