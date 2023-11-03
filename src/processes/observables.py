from types import SimpleNamespace
from typing import Optional, Callable
from dataclasses import dataclass
import math
import torch


@dataclass
class Observable:
    """
    Data class for an observable used for plotting
    Args:
        compute: Function that computes the observable value for the given momenta
        tex_label: Observable name in LaTeX for labels in plots
        bins: function that returns tensor with bin boundaries for given observable data
        xscale: X axis scale, "linear" (default) or "log", optional
        yscale: Y axis scale, "linear" (default) or "log", optional
        unit: Unit of the observable or None, if dimensionless, optional
    """

    compute: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    tex_label: str
    bins: Callable[[torch.Tensor], torch.Tensor]
    xscale: str = "linear"
    yscale: str = "linear"
    unit: Optional[str] = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["compute"] = None
        d["bins"] = None
        return d

def get_quantile_bins(data, n_bins=50, lower=0.0, upper=0.0):
    return torch.linspace(
        torch.nanquantile(data, lower), torch.nanquantile(data, 1 - upper), n_bins + 1
    )

def get_hardcoded_bins(data, n_bins, lower, upper):
    return torch.linspace(
        lower, upper, n_bins + 1
    )

def get_integer_bins(data, lower, upper):
    return torch.arange(
        lower - 0.5, upper + 0.5
    )

def nanify(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.where(
        p[...,0] != 0., obs, torch.tensor(float("nan"), device=p.device, dtype=p.dtype)
    )

def round(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.where(
        p[...,0] != 0., torch.round(obs), torch.tensor(float("nan"), device=p.device, dtype=p.dtype)
    )

def return_obs(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return obs

def compute_pt(p: torch.Tensor) -> torch.Tensor:
    return nanify(p, torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2))


def compute_phi(p: torch.Tensor) -> torch.Tensor:
    return nanify(p, torch.arctan2(p[..., 2], p[..., 1]))


def compute_eta(p: torch.Tensor) -> torch.Tensor:
    return nanify(p, torch.arctanh(
        p[..., 3] / torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2 + p[..., 3] ** 2)
    ))


def compute_m(p: torch.Tensor) -> torch.Tensor:
    return nanify(p, torch.sqrt(torch.clamp(
        p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2, min=0
    )))


def momenta_to_observables(
    n_bins: int = 50,
) -> list[Observable]:
    observables = []
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 0]),
            tex_label=r"\text{Jet mass}",
            unit="GeV",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=0, upper=60
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 1]),
            tex_label=r"\text{Jet multiplicity}",
            bins=lambda obs: get_integer_bins(
                obs, lower=0, upper=60
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 2]),
            tex_label=r"\text{Jet width}",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=0., upper=0.7
            ),
            yscale="log",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 3]),
            tex_label=r"\text{N-subjettiness ratio}",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=0., upper=1.2
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 4]),
            tex_label=r"\text{Groomed mass}",
            unit="GeV",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=-14, upper=-2
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 5]),
            tex_label=r"\text{Groomed momentum fraction}",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=-0.05, upper=0.55
            ),
            yscale="log",
        )
    )

    return observables
