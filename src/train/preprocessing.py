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
        invertible: bool
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.invertible = invertible

    def forward(
        self,
        x: torch.Tensor,
        rev: bool = False,
        batch_size: int = 100000,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if rev and not self.invertible:
            raise ValueError("Tried to call inverse of non-invertible transformation")
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
        for xb in x.split(batch_size, dim=0):
            yb = self.transform(xb, rev)
            ybs.append(yb)
        y = torch.cat(ybs, dim=0)

        if y.shape[1:] != output_shape:
            raise ValueError(
                f"Wrong output shape. Expected {output_shape}, "
                + f"got {tuple(y.shape[1:])}"
            )
        return y


class PreprocChain(PreprocTrafo):
    def __init__(
        self,
        trafos: Iterable[PreprocTrafo],
        normalize: bool = True,
        n_dim: int = None,
        erf_norm_channels=[]
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
        if len(erf_norm_channels) != 0:
            trafos.append(ErfPreproc((n_dim,), channels=erf_norm_channels))

        super().__init__(
            trafos[0].input_shape,
            trafos[-1].output_shape,
            all(t.invertible for t in trafos)
        )
        self.trafos = nn.ModuleList(trafos)
        self.normalize = normalize
        self.erf_norm = len(erf_norm_channels) != 0

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
        #if self.unit_hypercube:
        #    mins = torch.min(x-x.mean(dim=0), dim=0)[0]
        #    maxs = torch.max(x-x.mean(dim=0), dim=0)[0]
        #    x_std = torch.where(maxs > torch.abs(mins), maxs, torch.abs(mins)).unsqueeze(0)
        #else:
        x_std = x.std(dim=norm_dims, keepdims=True)
        self.mean = x_mean
        self.std = x_std
        if self.erf_norm:
            self.trafos[-2].set_norm(x_mean[0].expand(x.shape[1:]), x_std[0].expand(x.shape[1:]))
        else:
            self.trafos[-1].set_norm(x_mean[0].expand(x.shape[1:]), x_std[0].expand(x.shape[1:]))

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        for t in reversed(self.trafos) if rev else self.trafos:
            x = t(x, rev=rev)
        return x


class NormalizationPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...]
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.mean = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.std = nn.Parameter(torch.ones(shape), requires_grad=False)

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor):
        with torch.no_grad():
            self.mean.data.copy_(mean)
            self.std.data.copy_(std)

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x * self.std + self.mean
        else:
            z = (x - self.mean) / self.std
        return z


class UniformNoisePreprocessing(PreprocTrafo):
    def __init__(self, shape: Tuple[int, ...], channels):

        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.channels = channels

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x
            z[:, self.channels] = torch.round(z[:, self.channels])
        else:
            z = x
            noise = torch.rand_like(z[:, self.channels])-0.5
            z[:, self.channels] = z[:, self.channels] + noise
        return z


class ErfPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...], channels
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.channels = channels

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x
            z[:, self.channels] = torch.erf(z[:, self.channels]) + .001
        else:
            z = x
            z[:, self.channels] = torch.erfinv(z[:, self.channels] - .001)
        return z


class CubicRootPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...], channels
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.channels = channels

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x
            z[:, self.channels] = z[:, self.channels] ** 3
        else:
            z = x
            z[:, self.channels] = np.cbrt(z[:, self.channels])# ** (1./3.)
        return z


class LogPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...], channels
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.channels = channels

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x
            z[:, self.channels] = z[:, self.channels].exp()
            if 3 in self.channels:
                z[:, 3] = -1 * z[:, 3]
        else:
            z = x
            if 3 in self.channels:
                z[:, 3] = -1 * z[:, 3]
            z[:, self.channels] = (z[:, self.channels]).log()
        return z


class SpecialPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...]
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x
            z4 = z[:, 4]
            #z4 = z4*self.std2 + self.mean2
            z4 = torch.erf(z4)
            z4 = z4*self.factor
            z4 = z4+self.shift
            z4 = z4.exp()
            z4 = torch.where(z4 < 0.1, 0, z4)
            z[:, 4] = z4
        else:
            z = x
            z4 = z[:, 4]
            noise = torch.rand(size=z4.shape)/1000. * 3 + 0.097
            z4 = torch.where(z4 < 0.1, noise, z4)
            z4 = z4.log()
            self.shift = (z4.max() + z4.min())/2.
            z4 = z4-self.shift
            self.factor = max(z4.max(), -1 * z4.min())*1.001
            z4 = z4/self.factor
            z4 = torch.erfinv(z4)
            z[:, 4] = z4
        return z


def build_preprocessing(params: dict, n_dim: int) -> PreprocChain:
    """
    Builds a preprocessing chain with the given parameters

    Args:
        params: dictionary with preprocessing parameters
    Returns:
        Preprocessing chain
    """
    normalize = True

    uniform_noise_channels = params.get("uniform_noise_channels", [])
    erf_norm_channels = params.get("erf_norm_channels", [])
    cubic_root_channels = params.get("cubic_root_channels", [])
    log_channels = params.get("log_channels", [])

    special_preproc = params.get("special_preproc", False)

    trafos = []
    if len(uniform_noise_channels) != 0:
        trafos.append(UniformNoisePreprocessing(shape=(n_dim,), channels=uniform_noise_channels))

    if len(cubic_root_channels) != 0:
        trafos.append(CubicRootPreproc(shape=(n_dim,), channels=cubic_root_channels))

    if len(log_channels) != 0:
        trafos.append(LogPreproc(shape=(n_dim,), channels=log_channels))

    if special_preproc:
        trafos.append(SpecialPreproc(shape=(n_dim,)))

    return PreprocChain(
        trafos,
        normalize=normalize,
        n_dim=n_dim,
        erf_norm_channels=erf_norm_channels
    )
