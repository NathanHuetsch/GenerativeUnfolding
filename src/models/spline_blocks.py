import warnings
from typing import Callable, Iterable
import math

from scipy.stats import special_ortho_group
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import FrEIA.modules as fm

from .splines import unconstrained_rational_quadratic_spline


class RationalQuadraticSplineBlock(fm.InvertibleModule):
    """
    Implementation of rational quadratic spline coupling blocks
    (https://arxiv.org/pdf/1906.04032.pdf) as a FrEIA invertible module,
    based on the implementation in https://github.com/bayesiains/nflows
    """

    DEFAULT_MIN_BIN_WIDTH = 1e-3
    DEFAULT_MIN_BIN_HEIGHT = 1e-3
    DEFAULT_MIN_DERIVATIVE = 1e-3

    def __init__(
        self,
        dims_in: Iterable[tuple[int]],
        dims_c=Iterable[tuple[int]],
        subnet_constructor: Callable = None,
        num_bins: int = 10,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
        permute_soft: bool = False,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE,
    ):
        """
        Initializes the RQS coupling block

        Args:
            dims_in: shapes of the inputs
            dims_c: shapes of the conditions
            subnet_constructor: function that constructs the coupling block subnet
            num_bins: number of spline bins
            left: lower input bound (forward)
            right: upper input bound (forward)
            bottom: lower input bound (inverse)
            top: upper input bound (inverse)
            permute_soft: if True, insert rotations matrix instead of permutation after
                          the coupling block
            min_bin_width: minimal spline bin width
            min_bin_height: minimal spline bin height
            min_derivative: minimal derivative at bin boundary
        """
        super().__init__(dims_in, dims_c)
        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))
        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(
                dims_in[0][1:]
            ), f"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]
        self.num_bins = num_bins
        if self.DEFAULT_MIN_BIN_WIDTH * self.num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if self.DEFAULT_MIN_BIN_HEIGHT * self.num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")
        self.in_channels = channels
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels, channels))
            for i, j in enumerate(np.random.permutation(channels)):
                w[i, j] = 1.0

        self.w_perm = nn.Parameter(
            torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)),
            requires_grad=False,
        )
        self.w_perm_inv = nn.Parameter(
            torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)),
            requires_grad=False,
        )

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )
        self.subnet = subnet_constructor(
            self.splits[0] + self.condition_channels,
            (3 * self.num_bins + 1) * self.splits[1],
        )

    def forward(
        self,
        x: Iterable[torch.Tensor],
        c: Iterable[torch.Tensor] = [],
        rev: bool = False,
        jac: bool = True,
    ) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        """
        Computes the coupling transformation

        Args:
            x: Input tensors
            c: Condition tensors
            rev: If True, compute inverse transformation
            jac: Not used, Jacobian is always computed

        Returns:
            Output tensors and log jacobian determinants
        """

        (x,) = x

        if rev:
            x = nnf.linear(x, self.w_perm_inv)

        x1, x2 = torch.split(x, self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], dim=1)
        else:
            x1c = x1

        theta = self.subnet(x1c).reshape(
            x1c.shape[0], self.splits[1], 3 * self.num_bins + 1
        )
        x2, log_jac_det = unconstrained_rational_quadratic_spline(
            x2,
            theta,
            rev=rev,
            num_bins=self.num_bins,
            left=self.left,
            right=self.right,
            top=self.top,
            bottom=self.bottom,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        x_out = torch.cat((x1, x2), dim=1)

        if not rev:
            x_out = nnf.linear(x_out, self.w_perm)

        return (x_out,), log_jac_det

    def output_dims(self, input_dims: list[tuple[int]]) -> list[tuple[int]]:
        """
        Defines the output shapes of the coupling block

        Args:
            input_dims: Shapes of the inputs

        Returns:
            Shape of the outputs
        """
        return input_dims
