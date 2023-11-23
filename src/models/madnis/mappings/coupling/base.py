""" Coupling Blocks """

from typing import Dict, Callable, Union, Optional
import torch

from ..base import ConditionalMapping


# pylint: disable=C0103, R1729, E1120, E1124, W0221
class CouplingBlock(ConditionalMapping):
    """Base class to implement various coupling schemes.  It takes care of
    checking the dimensions, conditions, clamping mechanism, etc.
    Each child class only has to implement the _coupling method
    for the actual coupling operation.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        clamp: float = 0.0,
        clamp_activation: Union[str, Callable] = (lambda u: u),
    ):
        """
        Additional args in docstring of base class.
        Args:
            condition_mask: boolean mask that defines which features of the condition
                are used in this coupling block.
            splitting_mask: boolean mask that defines how the input tensor is split
                into two halves. Allows for a flexible splitting for each block.
            subnet_meta:
                meta defining the structure of the subnet like number of layers, units,
                activation functions etc.
            subnet_constructor: function or class, with signature
                constructor(dims_in, dims_out, *kwargs).  The result should be a torch
                nn.Module, that takes dims_in input channels, and dims_out output
                channels.
            clamp: Soft clamping for the multiplicative component. The
                amplification or attenuation of each input dimension can be at most
                exp(±clamp).
            clamp_activation: Function to perform the clamping. String values
                "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
                object can be passed. TANH behaves like the original realNVP paper.
                A custom function should take tensors and map -inf to -1 and +inf to +1.
        """
        super().__init__(dims_in, dims_c, condition_mask)

        if splitting_mask is None:
            ones = torch.ones(dims_in // 2, dtype=torch.bool)
            zeros = torch.zeros(dims_in - dims_in // 2, dtype=torch.bool)
            splitting_mask = torch.cat((ones, zeros))
        elif splitting_mask.shape != (dims_in,):
            raise ValueError(f"Splitting mask must have shape (dims_in,)")
        elif splitting_mask.dtype != torch.bool:
            raise ValueError(f"Splitting mask must be boolean tensor")
        self.register_buffer("splitting_mask", splitting_mask)

        self.clamp = clamp
        self.conditional = self.dims_c > 0

        self.split_len1 = torch.sum(self.splitting_mask).item()
        self.split_len2 = self.dims_in - self.split_len1
        self.cond_len = torch.sum(self.condition_mask).item()

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = lambda u: 0.636 * torch.atan(u)
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = lambda u: 2.0 * (torch.sigmoid(u) - 0.5)
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )

        # prefill with meta
        self.make_subnet = lambda n_in, n_out: subnet_constructor(subnet_meta, n_in, n_out)

    def _forward(self, x: torch.Tensor, condition: torch.Tensor, **kwargs):
        """
        Perform a forward pass
        through this layer operator.
        Args:
            x: input data (array-like of one or more tensors)
                of the form: ``x = [input_tensor_1]``
            condition: conditioning data (array-like of none or more tensors)
                of the form: ``x = [cond_tensor_1, cond_tensor_2, ...] ``
        """

        # notation:
        # x1, x2: two halves of the input
        # x_out: output
        # *_c: variable with condition concatenated
        # j: Jacobian of the coupling operation

        x1, x2 = x[:, self.splitting_mask], x[:, ~self.splitting_mask]

        if self.conditional:
            c = condition[:, self.condition_mask]

        # active half
        x1_c = torch.cat([x1, c], -1) if self.conditional else x1
        y2, j = self._coupling(x2, x1_c, **kwargs)

        # Combine to full output vector
        x_out = torch.ones_like(x)
        x_out[:, self.splitting_mask] = x1
        x_out[:, ~self.splitting_mask] = y2

        return x_out, j

    def _inverse(self, z: torch.Tensor, condition: torch.Tensor, **kwargs):
        """
        Perform a backward pass
        through this layer operator.
        Args:
            z: input data (array-like of one or more tensors)
                of the form: ``z = [input_tensor_1]``
            condition (torch.Tensor): conditioning data.
        """

        # notation:
        # z1, z2: two halves of the input
        # z_out: output
        # *_c: variable with condition concatenated
        # j: Jacobian of the coupling operation

        z1, z2 = z[:, self.splitting_mask], z[:, ~self.splitting_mask]

        if self.conditional:
            c = condition[:, self.condition_mask]

        z1_c = torch.cat([z1, c], -1) if self.conditional else z1
        y2, j = self._coupling(z2, z1_c, rev=True, **kwargs)

        # Combine to full output vector
        z_out = torch.ones_like(z)
        z_out[:, self.splitting_mask] = z1
        z_out[:, ~self.splitting_mask] = y2

        return z_out, j

    def _coupling(self, x2: torch.Tensor, u1: torch.Tensor, rev: bool = False, **kwargs):
        """The coupling operation in a half-sided coupling block.
        Args:
          x2 (Tensor): the 'active' half being transformed.
          u1 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y2 (Tensor): same shape as x1, the transformed 'active' half.
          j (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        """
        raise NotImplementedError()


class SplineCouplingBlock(CouplingBlock):
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
            clamp=0.0,
            clamp_activation=(lambda u: u),
        )

        # Definitions for spline
        self.num_bins = num_bins
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

        self.subnet = self.make_subnet(
            self.split_len1 + self.cond_len,
            self._output_dim_multiplier() * self.split_len2,
        )

    def _output_dim_multiplier(self):
        """Needs to be provided by all subclasses."""
        return NotImplementedError()

    def _elementwise_function(self, x: torch.Tensor, a: torch.Tensor, rev: bool):
        """Needs to be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _elementwise_function(...) method"
        )

    def _coupling(self, x2: torch.Tensor, u1: torch.Tensor, rev: bool = False, **kwargs):
        a1 = self.subnet(u1, **kwargs).reshape(
            u1.shape[0], self.split_len2, self._output_dim_multiplier()
        )
        # check format
        assert a1.shape[-1] == self._output_dim_multiplier()

        y2, j = self._elementwise_function(x2, a1, rev=rev)

        return y2, j


class TwoSidedCouplingBlock(CouplingBlock):
    """Base class to implement coupling blocks with two-sided operations.
    Each child class only has to implement the _coupling1 and _coupling2 methods
    for the left and right coupling operations.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        clamp: float = 0.0,
        clamp_activation: Union[str, Callable] = (lambda u: u),
    ):
        """
        Args in docstring of base class.
        """
        super().__init__(
            dims_in,
            dims_c,
            subnet_meta,
            subnet_constructor,
            condition_mask,
            splitting_mask,
            clamp,
            clamp_activation,
        )

    def _forward(self, x: torch.Tensor, condition: torch.Tensor, **kwargs):
        """
        Perform a forward pass
        through this layer operator.
        Args:
            x: input data (array-like of one or more tensors)
                of the form: ``x = [input_tensor_1]``
            condition: conditioning data (array-like of none or more tensors)
                of the form: ``x = [cond_tensor_1, cond_tensor_2, ...] ``
        """

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations
        x1, x2 = x[:, self.splitting_mask], x[:, ~self.splitting_mask]

        if self.conditional:
            c = condition[:, self.condition_mask]

        x2_c = torch.cat([x2, c], -1) if self.conditional else x2
        y1, j1 = self._coupling1(x1, x2_c)

        y1_c = torch.cat([y1, c], -1) if self.conditional else y1
        y2, j2 = self._coupling2(x2, y1_c)

        # Combine to full output vector
        x_out = torch.ones_like(x)
        x_out[:, self.splitting_mask] = y1
        x_out[:, ~self.splitting_mask] = y2

        return x_out, j1 + j2

    def _inverse(self, z: torch.Tensor, condition: torch.Tensor, **kwargs):
        """
        Perform a backward pass
        through this layer operator.
        Args:
            z: input data (array-like of one or more tensors)
                of the form: ``z = [input_tensor_1]``
            condition (torch.Tensor): conditioning data.
        """

        # notation:
        # z1, z2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations

        z1, z2 = z[:, self.splitting_mask], z[:, ~self.splitting_mask]

        if self.conditional:
            c = condition[:, self.condition_mask]

        z1_c = torch.cat([z1, c], -1) if self.conditional else z1
        y2, j2 = self._coupling2(z2, z1_c, rev=True)

        y2_c = torch.cat([y2, c], -1) if self.conditional else y2
        y1, j1 = self._coupling1(z1, y2_c, rev=True)

        # Combine to full output vector
        z_out = torch.ones_like(z)
        z_out[:, self.splitting_mask] = y1
        z_out[:, ~self.splitting_mask] = y2

        return z_out, j1 + j2

    def _coupling1(self, x1: torch.Tensor, u2: torch.Tensor, rev: bool = False):
        """The first/left coupling operation in a two-sided coupling block.
        Args:
          x1 (Tensor): the 'active' half being transformed.
          u2 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y1 (Tensor): same shape as x1, the transformed 'active' half.
          j1 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        """
        raise NotImplementedError()

    def _coupling2(self, x2: torch.Tensor, u1: torch.Tensor, rev: bool = False):
        """The second/right coupling operation in a two-sided coupling block.
        Args:
          x2 (Tensor): the 'active' half being transformed.
          u1 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y2 (Tensor): same shape as x1, the transformed 'active' half.
          j2 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        """
        raise NotImplementedError()
