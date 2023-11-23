""" RQS Coupling Blocks """

from typing import Dict, Callable, Optional
import torch

from .base import SplineCouplingBlock
from ..functional import splines


class LinearSplineBlock(SplineCouplingBlock):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        num_bins: int = 10,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
    ):

        super().__init__(
            dims_in,
            dims_c,
            subnet_meta,
            subnet_constructor,
            condition_mask,
            splitting_mask,
            num_bins,
            left,
            right,
            bottom,
            top,
        )

    def _output_dim_multiplier(self):
        return self.num_bins

    def _elementwise_function(self, x: torch.Tensor, a: torch.Tensor, rev: bool):
        y, ldj_elementwise = splines.unconstrained_linear_spline(
            x,
            a,
            inverse=rev,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
        )

        ldj = ldj_elementwise.sum(-1)

        return y, ldj


class QuadraticSplineBlock(SplineCouplingBlock):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        num_bins: int = 10,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
    ):

        super().__init__(
            dims_in,
            dims_c,
            subnet_meta,
            subnet_constructor,
            condition_mask,
            splitting_mask,
            num_bins,
            left,
            right,
            bottom,
            top,
        )

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 1

    def _elementwise_function(self, x: torch.Tensor, a: torch.Tensor, rev: bool):
        # split into different contributions
        unnormalized_widths = a[..., : self.num_bins]
        unnormalized_heights = a[..., self.num_bins :]

        y, ldj_elementwise = splines.unconstrained_quadratic_spline(
            x,
            unnormalized_widths,
            unnormalized_heights,
            inverse=rev,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
        )

        ldj = ldj_elementwise.sum(-1)

        return y, ldj


class RationalQuadraticSplineBlock(SplineCouplingBlock):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        num_bins: int = 10,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
        **spline_kwargs
    ):

        super().__init__(
            dims_in,
            dims_c,
            subnet_meta,
            subnet_constructor,
            condition_mask,
            splitting_mask,
            num_bins,
            left,
            right,
            bottom,
            top,
        )
        self.spline_kwargs = spline_kwargs

    def _output_dim_multiplier(self):
        return 3 * self.num_bins + 1

    def _elementwise_function(self, x, a, rev=False):
        # split into different contributions
        unnormalized_widths = a[..., : self.num_bins]
        unnormalized_heights = a[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = a[..., 2 * self.num_bins :]

        y, ldj_elementwise = splines.unconstrained_rational_quadratic_spline(
            x,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=rev,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
            **self.spline_kwargs
        )

        ldj = ldj_elementwise.sum(-1)

        return y, ldj


class CubicSplineBlock(SplineCouplingBlock):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        num_bins: int = 10,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
    ):

        super().__init__(
            dims_in,
            dims_c,
            subnet_meta,
            subnet_constructor,
            condition_mask,
            splitting_mask,
            num_bins,
            left,
            right,
            bottom,
            top,
        )

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 2

    def _elementwise_function(self, x, a, rev=False):
        # split into different contributions
        unnormalized_widths = a[..., : self.num_bins]
        unnormalized_heights = a[..., self.num_bins : 2 * self.num_bins]
        unnorm_derivatives_left = a[..., 2 * self.num_bins : 2 * self.num_bins + 1]
        unnorm_derivatives_right = a[..., 2 * self.num_bins + 1 :]

        y, ldj_elementwise = splines.unconstrained_cubic_spline(
            x,
            unnormalized_widths,
            unnormalized_heights,
            unnorm_derivatives_left,
            unnorm_derivatives_right,
            inverse=rev,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
        )

        ldj = ldj_elementwise.sum(-1)

        return y, ldj
