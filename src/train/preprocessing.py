from typing import Tuple, Optional, Union, Iterable
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from math import pi, gamma

# Define Metric
MINKOWSKI = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0]))


class PreprocTrafo(nn.Module):
    """
    Base class for a preprocessing transformation. It allows for different input and
    output shapes and both non-invertible as well as invertible transformations with or
    without known Jacobian
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        invertible: bool,
        has_jacobian: bool,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.invertible = invertible
        self.has_jacobian = has_jacobian

    def forward(
        self,
        x: torch.Tensor,
        rev: bool = False,
        jac: bool = False,
        return_jac: Optional[bool] = None,
        batch_size: int = 100000,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if rev and not self.invertible:
            raise ValueError("Tried to call inverse of non-invertible transformation")
        if jac and not self.has_jacobian:
            raise ValueError(
                "Tried to get jacobian from transformation without jacobian"
            )
        input_shape, output_shape = (
            (self.output_shape, self.input_shape)
            if rev
            else (self.input_shape, self.output_shape)
        )
        if x.shape[1:] != input_shape:
            raise ValueError(
                f"Wrong input shape. Expected {input_shape}, "
                + f"got {tuple(x.shape[1:])}"
            )

        ybs = []
        jbs = []
        for xb in x.split(batch_size, dim=0):
            yb, jb = self.transform(xb, rev, jac)
            ybs.append(yb)
            jbs.append(jb)
        y, j = torch.cat(ybs, dim=0), torch.cat(jbs, dim=0)

        if not jac:
            j = torch.zeros(x.shape[:1], dtype=x.dtype, device=x.device)
        if y.shape[1:] != output_shape:
            raise ValueError(
                f"Wrong output shape. Expected {output_shape}, "
                + f"got {tuple(y.shape[1:])}"
            )
        return (y, j) if return_jac or (return_jac is None and jac) else y


class PreprocChain(PreprocTrafo):
    def __init__(
        self,
        trafos: Iterable[PreprocTrafo],
        normalize: bool = True,
        n_dim: int = None,
        unit_hypercube: bool = False,
        erf_norm: bool = False
    ):
        if any(
            tp.output_shape != tn.input_shape
            for i, (tp, tn) in enumerate(zip(trafos[:-1], trafos[1:]))
        ):
            raise ValueError(
                f"Output shape {trafos[0].output_shape} of transformation {0} not "
                + f"equal to input shape {trafos[1].input_shape} of transformation {1}"
            )

        trafos.append(NormalizationPreproc((n_dim,)))
        if erf_norm:
            trafos.append(ErfPreproc((n_dim,)))


        super().__init__(
            trafos[0].input_shape,
            trafos[-1].output_shape,
            all(t.invertible for t in trafos),
            all(t.has_jacobian for t in trafos),
        )
        self.trafos = nn.ModuleList(trafos)
        self.normalize = normalize
        self.unit_hypercube = unit_hypercube
        self.erf_norm = erf_norm

    def init_normalization(self, x: torch.Tensor, batch_size: int = 100000):
        if not self.normalize:
            return
        xbs = []
        for xb in x.split(batch_size, dim=0):
            for t in self.trafos[:-1]:
                xb = t(xb)
            xbs.append(xb)
        x = torch.cat(xbs, dim=0)
        norm_dims = tuple(range(len(x.shape)-1))
        x_mean = x.mean(dim=norm_dims, keepdims=True)
        x_std = x.std(dim=norm_dims, keepdims=True)
        if self.unit_hypercube:
            mins = torch.min(x-x.mean(dim=0), dim=0)[0]
            maxs = torch.max(x-x.mean(dim=0), dim=0)[0]
            x_std[:, 4] = 1.001*torch.where(maxs[4] > torch.abs(mins[4]), maxs[4], torch.abs(mins[4])).unsqueeze(0)
        self.mean = x_mean
        self.std = x_std
        if self.erf_norm:
            self.trafos[-2].set_norm(x_mean[0].expand(x.shape[1:]), x_std[0].expand(x.shape[1:]))
        else:
            self.trafos[-1].set_norm(x_mean[0].expand(x.shape[1:]), x_std[0].expand(x.shape[1:]))

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        j_all = torch.zeros(x.shape[:1], dtype=x.dtype, device=x.device)
        for t in reversed(self.trafos) if rev else self.trafos:
            x, j = t(x, rev=rev, jac=jac, return_jac=True)
            j_all = j_all + j
        return x, j_all


class NormalizationPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...]
    ):
        super().__init__(
            input_shape=shape, output_shape=shape, invertible=True, has_jacobian=True
        )
        self.mean = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.std = nn.Parameter(torch.ones(shape), requires_grad=False)
        self.jac = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor):
        with torch.no_grad():
            self.mean.data.copy_(mean)
            self.std.data.copy_(std)
            self.jac.data.copy_(std.log().sum())

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rev:
            z, jac = x * self.std + self.mean, self.jac
        else:
            z, jac = (x - self.mean) / self.std, -self.jac

        return z, jac.expand(x.shape[0])


class UniformNoisePreprocessing(PreprocTrafo):
    def __init__(self, shape: Tuple[int, ...], channels):

        super().__init__(
            input_shape=shape, output_shape=shape, invertible=True, has_jacobian=True
        )
        self.channels = channels

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rev:
            z, jac = x, torch.tensor([0])
            z[:, self.channels] = torch.round(z[:, self.channels])
        else:
            z, jac = x, torch.tensor([0])
            noise = torch.rand_like(z[:, self.channels])-0.5
            z[:, self.channels] = z[:, self.channels] + noise
        return z, jac.expand(x.shape[0])


class ErfPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...]
    ):
        super().__init__(
            input_shape=shape, output_shape=shape, invertible=True, has_jacobian=True
        )
        self.mean = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.std = nn.Parameter(torch.ones(shape), requires_grad=False)
        self.jac = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor):
        with torch.no_grad():
            self.mean.data.copy_(mean)
            self.std.data.copy_(std)
            self.jac.data.copy_(std.log().sum())

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rev:
            z, jac = x, self.jac,
            z[:, 4], jac = torch.erf(z[:, 4]), self.jac
        else:
            z, jac = x, -self.jac
            #print("before erfinv:", z[:, 4].max(), z[:, 4].min())
            z[:, 4], jac = torch.erfinv(z[:, 4]), -self.jac # torch.erfinv(torch.clamp(z[:, 4], min=-0.999, max=0.999)), -self.jac
            #print("after erfinv:", z[:, 4].max(), z[:, 4].min())

        return z, jac.expand(x.shape[0])  # TODO: fix jacobians


def build_preprocessing(params: dict, n_dim: int) -> PreprocChain:
    """
    Builds a preprocessing chain with the given parameters

    Args:
        params: dictionary with preprocessing parameters
    Returns:
        Preprocessing chain
    """
    normalize = True
    unit_hypercube = params.get("unit_hypercube", False)
    erf_norm = params.get("erf_norm", False)
    uniform_noise_channels = params.get("uniform_noise_channels", [])

    trafos = []
    if len(uniform_noise_channels) != 0:
        trafos.append(UniformNoisePreprocessing(shape=(n_dim,), channels=uniform_noise_channels))

    return PreprocChain(
        trafos,
        normalize=normalize,
        n_dim=n_dim,
        unit_hypercube=unit_hypercube,
        erf_norm=erf_norm
    )
