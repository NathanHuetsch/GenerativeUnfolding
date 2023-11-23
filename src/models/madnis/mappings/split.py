from typing import List, Optional
import torch
import torch.nn as nn

from ..mappings.base import Mapping, ConditionalMapping

class ConditionalSplit(ConditionalMapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        mappings: List[Mapping],
        condition_type: str = "one_hot",
        condition_mask: Optional[torch.Tensor] = None
    ):
        super().__init__(dims_in, dims_c, condition_mask)

        self.mappings = nn.ModuleList(mappings)
        for mapping in mappings:
            if mapping.dims_in != dims_in:
                raise ValueError("Incompatible input dimensions")
            if mapping.dims_c != dims_c:
                raise ValueError("Incompatible condition dimensions")

        active_dims_c = self.condition_mask.sum()
        if condition_type == "one_hot":
            if active_dims_c != len(mappings):
                raise ValueError(
                    "Number of active condition dimensions must match number of mappings"
                )
        elif condition_type == "index":
            if active_dims_c != 1:
                raise ValueError(
                    "Only one active condition dimension allowed"
                )
        else:
            raise ValueError("Unknown condition type")
        self.condition_type = condition_type

    def _eval_mappings(self, x_or_z: torch.Tensor, condition: torch.Tensor, inverse: bool, **kwargs):
        masked_condition = condition[:, self.condition_mask]
        if self.condition_type == "one_hot":
            channels = torch.argmax(masked_condition, dim=1)
        elif self.condition_type == "index":
            channels = masked_condition[:,0]

        y = torch.zeros_like(x_or_z)
        log_det = torch.zeros(x_or_z.shape[0], dtype=x_or_z.dtype, device=x_or_z.device)
        for i, mapping in enumerate(self.mappings):
            mask = channels == i
            if inverse:
                y[mask], log_det[mask] = mapping.inverse(x_or_z[mask], condition[mask], **kwargs)
            else:
                y[mask], log_det[mask] = mapping.forward(x_or_z[mask], condition[mask], **kwargs)
        return y, log_det

    def _forward(self, x: torch.Tensor, condition: torch.Tensor, **kwargs):
        return self._eval_mappings(x, condition, inverse=False, **kwargs)

    def _inverse(self, z: torch.Tensor, condition: torch.Tensor, **kwargs):
        return self._eval_mappings(z, condition, inverse=True, **kwargs)

    def _log_det(
        self,
        x_or_z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        inverse: bool = False,
        **kwargs
    ):
        _, log_det = self._eval_mappings(x_or_z, condition, inverse, **kwargs)
        return log_det


class ConditionalPseudoSplit(ConditionalMapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        mapping: Mapping,
        n_channels: int,
        channel_map: torch.Tensor,
        condition_perms: list[torch.Tensor],
        condition_type: str = "one_hot",
        condition_mask: Optional[torch.Tensor] = None
    ):
        super().__init__(dims_in, dims_c, condition_mask)

        active_dims_c = self.condition_mask.sum()
        if condition_type == "one_hot":
            if active_dims_c != n_channels:
                raise ValueError(
                    "Number of active condition dimensions must match number of mappings"
                )
        elif condition_type == "index":
            if active_dims_c != 1:
                raise ValueError(
                    "Only one active condition dimension allowed"
                )
        else:
            raise ValueError("Unknown condition type")
        self.condition_type = condition_type
        self.mapping = mapping
        self.n_channels_mapped = len(channel_map.unique())
        self.channel_map = channel_map
        self.condition_perms = condition_perms

    def _eval_mappings(self, x_or_z: torch.Tensor, condition: torch.Tensor, inverse: bool, **kwargs):
        masked_condition = condition[:, self.condition_mask]
        if self.condition_type == "one_hot":
            channels = torch.argmax(masked_condition, dim=1)
        elif self.condition_type == "index":
            channels = masked_condition[:,0]
        channels = self.channel_map[channels]
        section_sizes = torch.bincount(channels, minlength=self.n_channels_mapped).tolist()
        channel_sort = torch.argsort(channels)
        inv_channel_sort = torch.argsort(channel_sort)
        permuted_condition = condition[channel_sort]
        for pc, perm in zip(permuted_condition.split(section_sizes), self.condition_perms):
            pc[:] = pc[:, perm]

        if inverse:
            y, log_det = self.mapping.inverse(
                x_or_z[channel_sort], permuted_condition, section_sizes=section_sizes, **kwargs
            )
        else:
            y, log_det = self.mapping.forward(
                x_or_z[channel_sort], permuted_condition, section_sizes=section_sizes, **kwargs
            )
        return y[inv_channel_sort], log_det[inv_channel_sort]

    def _forward(self, x: torch.Tensor, condition: torch.Tensor, **kwargs):
        return self._eval_mappings(x, condition, inverse=False, **kwargs)

    def _inverse(self, z: torch.Tensor, condition: torch.Tensor, **kwargs):
        return self._eval_mappings(z, condition, inverse=True, **kwargs)

    def _log_det(
        self,
        x_or_z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        inverse: bool = False,
        **kwargs
    ):
        _, log_det = self._eval_mappings(x_or_z, condition, inverse, **kwargs)
        return log_det


class DimensionalSplit(Mapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        mappings: List[Mapping]
    ):
        super().__init__(dims_in, dims_c)
        self.mappings = nn.ModuleList(mappings)
        self.all_dims_in = [m.dims_in for m in mappings]
        if dims_in != sum(self.all_dims_in):
            raise ValueError("dims_in for mappings must add up to total dims_in")
        if any(m.dims_c != dims_c for m in mappings):
            raise ValueError("dims_c must be the same for all mappings")

    def _eval_mappings(self, x: torch.Tensor, condition: torch.Tensor, inverse: bool, **kwargs):
        xs = torch.split(x, self.all_dims_in, dim=1)
        zs = []
        log_jac = 0.
        for mapping, xi in zip(self.mappings, xs):
            if inverse:
                zi, log_jac_i = mapping.inverse(xi, condition, **kwargs)
            else:
                zi, log_jac_i = mapping.forward(xi, condition, **kwargs)
            zs.append(zi)
            log_jac += log_jac_i
        return torch.cat(zs, dim=1), log_jac

    def _forward(self, x: torch.Tensor, condition: torch.Tensor, **kwargs):
        return self._eval_mappings(x, condition, inverse=False, **kwargs)

    def _inverse(self, z: torch.Tensor, condition: torch.Tensor, **kwargs):
        return self._eval_mappings(z, condition, inverse=True, **kwargs)

    def _log_det(
        self,
        x_or_z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        inverse: bool = False,
        **kwargs
    ):
        _, log_det = self._eval_mappings(x_or_z, condition, inverse, **kwargs)
        return log_det
