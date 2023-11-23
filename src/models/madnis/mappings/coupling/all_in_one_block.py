"""All in one coupling Block"""

import warnings
from typing import Dict, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import special_ortho_group

from .base import CouplingBlock


class AllInOneBlock(CouplingBlock):
    """Module combining the most common operations in a normalizing flow or
    similar model. It combines affine coupling, permutation, and
    global affine transformation ('ActNorm'). It can also be used as
    GIN coupling block and use an inverted pre-permutation.
    The affine transformation includes a soft clamping mechanism,
    first used in Real-NVP.
    """

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        splitting_mask: Optional[torch.Tensor] = None,
        subnet_meta: Dict = None,
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        gin_block: bool = False,
        global_affine_init: float = 1.0,
        global_affine_type: str = "SOFTPLUS",
        permute_soft: bool = False,
        learned_householder_permutation: int = 0,
        reverse_permutation: bool = False,
    ):
        """
        Args:
            gin_block:
                Turn the block into a GIN block from Sorrenson et al, 2019.
                Makes it so that the coupling operations as a whole is
                volume preserving.
            global_affine_init:
                Initial value for the global affine scaling.
            global_affine_type:
                ``'SIGMOID'``, ``'SOFTPLUS'``, or ``'EXP'``. Defines the activation
                to be used on the beta for the global affine scaling.
            permute_soft:
                bool, whether to sample the permutation matrix `R` from `SO(N)`,
                or to use hard permutations instead. Note, ``permute_soft=True``
                is very slow when working with >512 dimensions.
            learned_householder_permutation:
                Int, if >0, turn on the matrix :math:`V` above, that represents
                multiple learned householder reflections. Slow if large number.
                Dubious whether it actually helps network performance.
            reverse_permutation:
                Reverse the permutation before the block, as introduced by Putzky
                et al, 2019. Turns on the :math:`R^{-1}` pre-multiplication above.
        """

        super().__init__(
            dims_in,
            dims_c,
            condition_mask,
            splitting_mask,
            subnet_meta,
            subnet_constructor,
            clamp,
            clamp_activation=(lambda u: u),
        )

        self.GIN = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder = learned_householder_permutation

        if permute_soft and dims_in > 512:
            warnings.warn(
                (
                    "Soft permutation will take a very long time to initialize "
                    f"with {dims_in} feature channels. Consider using hard permutation instead."
                )
            )

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # global_scale_activation(global_scale) = global_affine_init
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        if global_affine_type == "SIGMOID":
            global_scale = 2.0 - np.log(10.0 / global_affine_init - 1.0)
            self.global_scale_activation = self._sigmoid_global_scale_activation
        elif global_affine_type == "SOFTPLUS":
            global_scale = 2.0 * np.log(np.exp(0.5 * 10.0 * global_affine_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.global_scale_activation = self._softplus_global_scale_activation
        elif global_affine_type == "EXP":
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = self._exp_global_scale_activation
        else:
            raise ValueError(
                'Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"'
            )

        self.global_scale = nn.Parameter(
            torch.ones(1, self.dims_in) * float(global_scale)
        )
        self.global_offset = nn.Parameter(torch.zeros(1, self.dims_in))

        if permute_soft:
            w = special_ortho_group.rvs(self.dims_in)
        else:
            w = np.zeros((self.dims_in, self.dims_in))
            for i, j in enumerate(np.random.permutation(self.dims_in)):
                w[i, j] = 1.0

        if self.householder:
            # instead of just the permutation matrix w, the learned housholder
            # permutation keeps track of reflection vectors vk, in addition to a
            # random initial permutation w_0.
            self.vk_householder = nn.Parameter(
                0.2 * torch.randn(self.householder, self.dims_in), requires_grad=True
            )
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
        else:
            self.w_perm = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
            self.w_perm_inv = nn.Parameter(torch.FloatTensor(w.T), requires_grad=False)

        self.F = self.make_subnet(self.split_len1 + self.cond_len, 2 * self.split_len2)
        self.last_jac = None

    def _sigmoid_global_scale_activation(self, a: torch.Tensor):
        return 10 * torch.sigmoid(a - 2.0)

    def _softplus_global_scale_activation(self, a: torch.Tensor):
        return 0.1 * self.softplus(a)

    def _exp_global_scale_activation(self, a: torch.Tensor):
        return torch.exp(a)

    def _construct_householder_permutation(self):
        """Computes a permutation matrix from the reflection vectors that are
        learned internally as nn.Parameters."""
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(
                w,
                torch.eye(self.dims_in).to(w.device)
                - 2 * torch.ger(vk, vk) / torch.dot(vk, vk),
            )

        return w

    def _permute(self, x: torch.Tensor, rev: bool = False):
        """Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation."""
        if self.GIN:
            scale = 1.0
            perm_log_jac = 0.0
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = torch.sum(torch.log(scale))

        if rev:
            return (x @ self.w_perm_inv - self.global_offset) / scale, perm_log_jac
        else:
            x_scaled = x * scale + self.global_offset
            return x_scaled @ self.w_perm, perm_log_jac

    def _pre_permute(self, x: torch.Tensor, rev: bool = False):
        """Permutes before the coupling block, only used if
        reverse_permutation is set"""
        if rev:
            return x @ self.w_perm
        else:
            return x @ self.w_perm_inv

    def _affine(self, x: torch.Tensor, a: torch.Tensor, rev: bool = False):
        """Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the affine coupling operation.
        Returns both the transformed inputs and the LogJacDet."""

        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        a *= 0.1
        s, t = a[:, : self.split_len2], a[:, self.split_len2 :]

        sub_jac = self.clamp * torch.tanh(s)
        if self.GIN:
            sub_jac -= sub_jac.mean(1, keepdim=True)

        if not rev:
            return x * torch.exp(sub_jac) + t, sub_jac.sum(-1)
        else:
            return (x - t) * torch.exp(-sub_jac), -sub_jac.sum(-1)

    def _forward(self, x: torch.Tensor, condition: torch.Tensor, **kwargs):
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if self.reverse_pre_permute:
            x = self._pre_permute(x, rev=False)

        x1, x2 = x[:, self.splitting_mask], x[:, ~self.splitting_mask]

        if self.conditional:
            c = condition[:, self.condition_mask]

        x1c = torch.cat([x1, c], -1) if self.conditional else x1
        a1 = self.F(x1c, **kwargs)
        y2, j2 = self._affine(x2, a1)

        log_jac_det = j2

        # Combine to full output vector
        x_out = torch.ones_like(x)
        x_out[:, self.splitting_mask] = x1
        x_out[:, ~self.splitting_mask] = y2

        # Act Norm
        x_out, global_scaling_jac = self._permute(x_out, rev=False)

        # add the global scaling Jacobian to the total.
        log_jac_det += global_scaling_jac

        return x_out, log_jac_det

    def _inverse(self, x: torch.Tensor, condition: torch.Tensor, **kwargs):
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        # Act Norm
        x, global_scaling_jac = self._permute(x, rev=True)

        x1, x2 = x[:, self.splitting_mask], x[:, ~self.splitting_mask]

        if self.conditional:
            c = condition[:, self.condition_mask]

        x1c = torch.cat([x1, c], -1) if self.conditional else x1
        a1 = self.F(x1c, **kwargs)
        y2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2

        # Combine to full output vector
        x_out = torch.ones_like(x)
        x_out[:, self.splitting_mask] = x1
        x_out[:, ~self.splitting_mask] = y2

        if self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        log_jac_det += (-1) * global_scaling_jac

        return x_out, log_jac_det
