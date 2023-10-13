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
        compute: Function that computes the observable value for the given momenta and
                 event types
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

def nanify(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.where(
        p[...,0] != 0., obs, torch.tensor(float("nan"), device=p.device, dtype=p.dtype)
    )

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
    particle_names: list[str],
    delta_pairs: list[tuple[int, int]],
    hard_scattering: bool,
    off_shell: list[bool],
    n_bins: int = 50,
    sqrt_s: float = 13000.0,
) -> list[Observable]:
    observables = []
    for i, name in enumerate(particle_names):
        observables.append(
            Observable(
                compute=lambda p, _, i=i: nanify(p[..., i, :], p[..., i, 0]),
                tex_label=f"E_{{{name}}}",
                unit="GeV",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-4, upper=1e-4
                ),
                yscale="log",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, _, i=i: nanify(p[..., i, :], p[..., i, 1]),
                tex_label=f"p_{{x,{name}}}",
                unit="GeV",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-4, upper=1e-4
                ),
                yscale="log",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, _, i=i: nanify(p[..., i, :], p[..., i, 2]),
                tex_label=f"p_{{y,{name}}}",
                unit="GeV",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-4, upper=1e-4
                ),
                yscale="log",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, _, i=i: nanify(p[..., i, :], p[..., i, 3]),
                tex_label=f"p_{{z,{name}}}",
                unit="GeV",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-4, upper=1e-4
                ),
                yscale="log",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, _, i=i: compute_pt(p[..., i, :]),
                tex_label=f"p_{{T,{name}}}",
                unit="GeV",
                bins=lambda obs: get_quantile_bins(obs, n_bins=n_bins, upper=1e-4),
                yscale="log",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, _, i=i: compute_phi(p[..., i, :]),
                tex_label=f"\\phi_{{{name}}}",
                unit=None,
                bins=lambda obs: torch.linspace(-math.pi, math.pi, n_bins + 1),
            )
        )
        observables.append(
            Observable(
                compute=lambda p, _, i=i: compute_eta(p[..., i, :]),
                tex_label=f"\\eta_{{{name}}}",
                unit=None,
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-4, upper=1e-4
                ),
            )
        )
        if off_shell[i]:
            observables.append(
                Observable(
                    compute=lambda p, _, i=i: compute_m(p[..., i, :]),
                    tex_label=f"M_{{{name}}}",
                    unit="GeV",
                    bins=lambda obs: get_quantile_bins(obs, n_bins=n_bins, upper=1e-4),
                )
            )

    for i, j in delta_pairs:
        name_i, name_j = particle_names[i], particle_names[j]
        observables.append(
            Observable(
                compute=lambda p, _, i=i, j=j: (
                    compute_phi(p[..., i, :]) - compute_phi(p[..., j, :]) + math.pi
                )
                % (2 * math.pi)
                - math.pi,
                tex_label=f"\\Delta \\phi_{{{name_i},{name_j}}}",
                unit=None,
                bins=lambda obs: torch.linspace(-math.pi, math.pi, n_bins + 1),
            )
        )
        observables.append(
            Observable(
                compute=lambda p, _, i=i, j=j: compute_eta(p[..., i, :])
                - compute_eta(p[..., j, :]),
                tex_label=f"\\Delta \\eta_{{{name_i},{name_j}}}",
                unit=None,
                bins=lambda obs: torch.linspace(-6, 6, n_bins + 1),
            )
        )
        observables.append(
            Observable(
                compute=lambda p, _, i=i, j=j: torch.sqrt(
                    (
                        (
                            compute_phi(p[..., i, :])
                            - compute_phi(p[..., j, :])
                            + math.pi
                        )
                        % (2 * math.pi)
                        - math.pi
                    )
                    ** 2
                    + (compute_eta(p[..., i, :]) - compute_eta(p[..., j, :])) ** 2
                ),
                tex_label=f"\\Delta R_{{{name_i},{name_j}}}",
                unit=None,
                bins=lambda obs: torch.linspace(0, 10, n_bins + 1),
            )
        )

    if hard_scattering:
        observables.append(
            Observable(
                compute=lambda p, _: (p[..., 0].sum(dim=-1) + p[..., 3].sum(dim=-1))
                / sqrt_s,
                tex_label="x_1",
                unit=None,
                bins=lambda obs: torch.linspace(0, 1, n_bins + 1),
                yscale="log",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, _: (p[..., 0].sum(dim=-1) - p[..., 3].sum(dim=-1))
                / sqrt_s,
                tex_label="x_2",
                unit=None,
                bins=lambda obs: torch.linspace(0, 1, n_bins + 1),
                yscale="log",
            )
        )

    return observables
