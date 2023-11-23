from typing import Optional
import torch

from .base import Mapping


class LinearMapping(Mapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        m: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None
    ):
        super().__init__(dims_in, dims_c)
        if m is None:
            m = torch.eye(dims_in)
        elif m.shape != (dims_in, dims_in):
            raise ValueError("m needs to have shape (dims_in, dims_in)")
        self.register_buffer("m", m.t().float())
        self.register_buffer("m_inv", m.t().float().inverse())

        if b is None:
            b = torch.zeros(dims_in)
        elif b.shape != (dims_in, ):
            raise ValueError("b needs to have shape (dims_b)")
        self.register_buffer("b", b.float())

        self.register_buffer("log_det_m", torch.slogdet(m)[1])

    def _forward(self, x: torch.Tensor, condition: Optional[torch.Tensor], **kwargs):
        del condition
        log_det = self.log_det_m.expand(x.shape[0])
        return x @ self.m + self.b, log_det

    def _inverse(self, z: torch.Tensor, condition: Optional[torch.Tensor], **kwargs):
        del condition
        log_det = self.log_det_m.expand(z.shape[0])
        return (z - self.b) @ self.m_inv, -log_det

    def _log_det(self, x_or_z, condition: Optional[torch.Tensor] = None, inverse=False, **kwargs):
        del condition
        return (-1)**inverse * self.log_det_m.expand(x_or_z.shape[0])
