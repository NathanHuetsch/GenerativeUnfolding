""" Coupling Blocks """

from typing import Dict, Callable, Union, Optional
import torch

from .base import CouplingBlock, TwoSidedCouplingBlock


class NICECouplingBlock(TwoSidedCouplingBlock):
    """Coupling Block following the NICE (Dinh et al, 2015) design.
    The inputs are split in two halves. For 2D, 3D, 4D inputs, the split is
    performed along the channel (first) dimension. Then, residual coefficients are
    predicted by two subnetworks that are added to each half in turn.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
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

        self.F = self.make_subnet(self.split_len2 + self.cond_len, self.split_len1)
        self.G = self.make_subnet(self.split_len1 + self.cond_len, self.split_len2)

    def _coupling1(self, x1, u2, rev=False, **kwargs):
        if rev:
            return x1 - self.F(u2, **kwargs), 0.0
        return x1 + self.F(u2, **kwargs), 0.0

    def _coupling2(self, x2, u1, rev=False, **kwargs):
        if rev:
            return x2 - self.G(u1, **kwargs), 0.0
        return x2 + self.G(u1, **kwargs), 0.0


class RNVPCouplingBlock(TwoSidedCouplingBlock):
    """Coupling Block following the RealNVP design (Dinh et al, 2017) with some
    minor differences. The inputs are split in two halves. For 2D, 3D, 4D
    inputs, the split is performed along the channel dimension. Two affine
    coupling operations are performed in turn on both halves of the input.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):
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

        self.net_s1 = self.make_subnet(self.split_len1 + self.cond_len, self.split_len2)
        self.net_t1 = self.make_subnet(self.split_len1 + self.cond_len, self.split_len2)
        self.net_s2 = self.make_subnet(self.split_len2 + self.cond_len, self.split_len1)
        self.net_t2 = self.make_subnet(self.split_len2 + self.cond_len, self.split_len1)

    def _coupling1(self, x1, u2, rev=False, **kwargs):
        s2, t2 = self.net_s2(u2, **kwargs), self.net_t2(u2, **kwargs)
        s2 = self.clamp * self.f_clamp(s2)
        j1 = s2.sum(-1)

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, -j1

        y1 = torch.exp(s2) * x1 + t2
        return y1, j1

    def _coupling2(self, x2, u1, rev=False, **kwargs):
        s1, t1 = self.net_s1(u1, **kwargs), self.net_t1(u1, **kwargs)
        s1 = self.clamp * self.f_clamp(s1)
        j2 = s1.sum(-1)

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, -j2

        y2 = torch.exp(s1) * x2 + t1
        return y2, j2


class GLOWCouplingBlock(TwoSidedCouplingBlock):
    """Coupling Block following the GLOW design. Note, this is only the coupling
    part itself, and does not include ActNorm, invertible 1x1 convolutions, etc.
    See AllInOneBlock for a block combining these functions at once.
    The only difference to the RNVPCouplingBlock coupling blocks
    is that it uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate
    subnetworks. This reduces computational cost and speeds up learning.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):

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

        self.F1 = self.make_subnet(self.split_len1 + self.cond_len, self.split_len2 * 2)
        self.F2 = self.make_subnet(self.split_len2 + self.cond_len, self.split_len1 * 2)

    def _coupling1(self, x1, u2, rev=False, **kwargs):
        a2 = self.F2(u2, **kwargs)
        s2, t2 = a2[:, : self.split_len1], a2[:, self.split_len1 :]
        s2 = self.clamp * self.f_clamp(s2)
        j1 = s2.sum(-1)

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, -j1

        y1 = torch.exp(s2) * x1 + t2
        return y1, j1

    def _coupling2(self, x2, u1, rev=False, **kwargs):
        a1 = self.F1(u1, **kwargs)
        s1, t1 = a1[:, : self.split_len2], a1[:, self.split_len2 :]
        s1 = self.clamp * self.f_clamp(s1)
        j2 = s1.sum(-1)

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, -j2

        y2 = torch.exp(s1) * x2 + t1
        return y2, j2


class GINCouplingBlock(TwoSidedCouplingBlock):
    """Coupling Block following the GIN design. The difference from
    GLOWCouplingBlock (and other affine coupling blocks) is that the Jacobian
    determinant is constrained to be 1.  This constrains the block to be
    volume-preserving. Volume preservation is achieved by subtracting the mean
    of the output of the s subnetwork from itself.  While volume preserving, GIN
    is still more powerful than NICE, as GIN is not volume preserving within
    each dimension.
    Note: this implementation differs slightly from the originally published
    implementation, which scales the final component of the s subnetwork so the
    sum of the outputs of s is zero. There was no difference found between the
    implementations in practice, but subtracting the mean guarantees that all
    outputs of s are at most ±exp(clamp), which might be more stable in certain
    cases.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):

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

        self.F1 = self.make_subnet(self.split_len1 + self.cond_len, self.split_len2 * 2)
        self.F2 = self.make_subnet(self.split_len2 + self.cond_len, self.split_len1 * 2)

    def _coupling1(self, x1, u2, rev=False, **kwargs):
        a2 = self.F2(u2, **kwargs)
        s2, t2 = a2[:, : self.split_len1], a2[:, self.split_len1 :]
        s2 = self.clamp * self.f_clamp(s2)
        s2 -= s2.mean(1, keepdim=True)

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, 0.0

        y1 = torch.exp(s2) * x1 + t2
        return y1, 0.0

    def _coupling2(self, x2, u1, rev=False, **kwargs):
        a1 = self.F1(u1, **kwargs)
        s1, t1 = a1[:, : self.split_len2], a1[:, self.split_len2 :]
        s1 = self.clamp * self.f_clamp(s1)
        s1 -= s1.mean(1, keepdim=True)

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, 0.0

        y2 = torch.exp(s1) * x2 + t1
        return y2, 0.0


class AffineCoupling(CouplingBlock):
    """Half of a coupling block following the GLOWCouplingBlock design. This
    means only one affine transformation on half the inputs. In the case where
    random permutations or orthogonal transforms are used after every block,
    this is not a restriction and simplifies the design.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):

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

        self.F = self.make_subnet(self.split_len1 + self.cond_len, 2 * self.split_len2)

    def _coupling(self, x2: torch.Tensor, u1: torch.Tensor, rev: bool = False, **kwargs):
        a1 = self.F(u1, **kwargs)
        a1 *= 0.1  # factor stabilizes initialisation
        s, t = a1[:, : self.split_len2], a1[:, self.split_len2 :]
        s = self.clamp * self.f_clamp(s)
        j = s.sum(-1)

        if rev:
            y2 = (x2 - t) * torch.exp(-s)
            return y2, -j

        y2 = x2 * torch.exp(s) + t
        return y2, j


class ConditionalAffineTransform(CouplingBlock):
    """Similar to the conditioning layers from SPADE (Park et al, 2019): Perform
    an affine transformation on the whole input, where the affine coefficients
    are predicted from only the condition.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        subnet_meta: Dict,
        subnet_constructor: Callable,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):

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

        if not self.conditional:
            raise ValueError("ConditionalAffineTransform must have a condition")

        self.subnet = self.make_subnet(self.cond_len, 2 * self.dims_in)

    def _forward(self, x: torch.Tensor, condition: torch.Tensor, **kwargs):
        c = condition[:, self.condition_mask]
        a = self.subnet(c, **kwargs)
        s, t = a[:, : self.dims_in], a[:, self.dims_in :]
        s = self.clamp * self.f_clamp(s)
        j = s.sum(-1)
        y = torch.exp(s) * x + t

        return y, j

    def _inverse(self, z: torch.Tensor, condition: torch.Tensor, **kwargs):
        c = condition[:, self.condition_mask]
        a = self.subnet(c, **kwargs)
        s, t = a[:, : self.dims_in], a[:, self.dims_in :]
        s = self.clamp * self.f_clamp(s)
        j = s.sum(-1)
        y = (z - t) * torch.exp(-s)

        return y, -j
