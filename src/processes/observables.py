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


def compute_invariant_mass(p, particles) -> torch.Tensor:

    px_sum = 0
    py_sum = 0
    pz_sum = 0
    e_sum = 0
    for particle in particles:

        m = p[..., particle, 0]
        pT = p[..., particle, 1]
        eta = p[..., particle, 2]
        phi = p[..., particle, 3]

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        e = torch.sqrt(m ** 2 + px ** 2 + py ** 2 + pz ** 2)

        px_sum += px
        py_sum += py
        pz_sum += pz
        e_sum += e

    m = torch.sqrt(torch.clamp(
        (e_sum)**2 - (px_sum)**2 - (py_sum)**2 - (pz_sum)**2, min=0
    ))
    return m


def compute_e(p) -> torch.Tensor:

    m = p[..., 0]
    pT = p[..., 1]
    eta = p[..., 2]
    phi = p[..., 3]

    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    pz = pT * torch.sinh(eta)
    e = torch.sqrt(m ** 2 + px ** 2 + py ** 2 + pz ** 2)
    return e


def compute_deltaR(p, ind_1, ind_2) -> torch.Tensor:

    eta_1 = p[..., ind_1, 2]
    phi_1 = p[..., ind_1, 3]
    eta_2 = p[..., ind_2, 2]
    phi_2 = p[..., ind_2, 3]
    deltaR_2 = (eta_1-eta_2)**2 + (phi_1-phi_2)**2
    return torch.sqrt(deltaR_2)


def compute_px(p) -> torch.Tensor:
    pT = p[..., 1]
    phi = p[..., 3]
    return pT * torch.cos(phi)


def compute_py(p) -> torch.Tensor:
    pT = p[..., 1]
    phi = p[..., 3]
    return pT * torch.sin(phi)


def compute_pz(p) -> torch.Tensor:
    pT = p[..., 1]
    eta = p[..., 2]
    return pT * torch.sinh(eta)


def ZJets_Observables(
    n_bins: int = 50,
) -> list[Observable]:
    observables = []
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 0]),
            tex_label=r"\text{Jet mass } m",
            unit="GeV",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=1, upper=60
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 1]),
            tex_label=r"\text{Jet width } w",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=0., upper=0.6
            ),
            yscale="log",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: round(p[..., :], p[..., 2]),
            tex_label=r"\text{Jet multiplicity } N",
            bins=lambda obs: get_integer_bins(
                obs, lower=4, upper=60
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 3]),
            tex_label=r"\text{Groomed mass }\log \rho",
            unit="GeV",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=-14, upper=-2
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 4]),
            tex_label=r"\text{Groomed momentum fraction }z_g",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=0.05, upper=0.55
            ),
            yscale="log",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: nanify(p[..., :], p[..., 5]),
            tex_label=r"\text{N-subjettiness ratio } \tau_{21}",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=0.1, upper=1.1
            ),
            yscale="linear",
        )
    )

    return observables


def TTBar_Observables(
    n_bins: int = 50,
) -> list[Observable]:
    particle_names = [r"t_l", r"b_l", r"W_l", r"l", r"\nu", r"t_h", r"b_h", r"W_h", r"q_1", r"q_2"]
    observables = []
    for i, name in enumerate(particle_names):
        if i in [0, 1, 2, 4, 5, 6, 7, 9]:
            b = lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-2, upper=1e-2
                )
        elif i == 8:
            b = lambda obs: get_hardcoded_bins(
                obs, n_bins=500, lower=-0.2, upper=1.7
            )
        else:
            b = lambda obs: get_hardcoded_bins(
                obs, n_bins=500, lower=-0.1, upper=0.3
            )
        observables.append(
            Observable(
                compute=lambda p, i=i: return_obs(p[...,i, :], p[..., i, 0]),
                tex_label=f"m_{{{name}}} ",
                unit="GeV",
                bins=b,
                yscale="linear",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, i=i: return_obs(p[..., i, :], p[..., i, 1]),
                tex_label=f"p_{{T,{name}}} ",
                unit="GeV",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=0, upper=1e-2
                ),
                yscale="linear",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, i=i: return_obs(p[..., i, :], p[..., i, 2]),
                tex_label=f"\eta_{{{name}}} ",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-4, upper=1e-4
                ),
                yscale="linear",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, i=i: return_obs(p[..., i, :], p[..., i, 3]),
                tex_label=f"\phi_{{{name}}} ",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-4, upper=1e-4
                ),
                yscale="linear",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, i=i: compute_e(p[..., i, :]),
                tex_label=f"E_{{{name}}} ",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=0, upper=1e-2
                ),
                yscale="linear",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, i=i: compute_px(p[..., i, :]),
                tex_label=f"p_{{x, {name}}} ",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-3, upper=1e-3
                ),
                yscale="linear",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, i=i: compute_py(p[..., i, :]),
                tex_label=f"p_{{y, {name}}} ",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-3, upper=1e-3
                ),
                yscale="linear",
            )
        )
        observables.append(
            Observable(
                compute=lambda p, i=i: compute_pz(p[..., i, :]),
                tex_label=f"p_{{z, {name}}} ",
                bins=lambda obs: get_quantile_bins(
                    obs, n_bins=n_bins, lower=1e-3, upper=1e-3
                ),
                yscale="linear",
            )
        )

    observables.append(
        Observable(
            compute=lambda p: compute_invariant_mass(p, [3, 4]),
            tex_label=f"M_{{W, l}} ",
            unit="GeV",
            bins=lambda obs: get_quantile_bins(
                obs, n_bins=n_bins, lower=1e-2, upper=1e-2
            ),
            yscale="linear",
        )
    )

    observables.append(
        Observable(
            compute=lambda p: compute_invariant_mass(p, [8, 9]),
            tex_label=f"M_{{W, h}} ",
            unit="GeV",
            bins=lambda obs: get_quantile_bins(
                obs, n_bins=n_bins, lower=1e-2, upper=1e-2
            ),
            yscale="linear",
        )
    )

    observables.append(
        Observable(
            compute=lambda p: compute_invariant_mass(p, [1, 3, 4]),
            tex_label=f"M_{{t, l}} ",
            unit="GeV",
            bins=lambda obs: get_quantile_bins(
                obs, n_bins=n_bins, lower=1e-2, upper=1e-2
            ),
            yscale="linear",
        )
    )

    observables.append(
        Observable(
            compute=lambda p: compute_invariant_mass(p, [6, 8, 9]),
            tex_label=f"M_{{t, h}} ",
            unit="GeV",
            bins=lambda obs: get_quantile_bins(
                obs, n_bins=n_bins, lower=1e-2, upper=1e-2
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: compute_e(p[..., 0, :]) - compute_e(p[..., 2, :]),
            tex_label=f"E_{{t_l}} - E_{{W_l}}",
            unit="GeV",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=-50, upper=500
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: compute_e(p[..., 5, :]) - compute_e(p[..., 7, :]),
            tex_label=f"E_{{t_h}} - E_{{W_h}}",
            unit="GeV",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=-50, upper=500
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: compute_e(p[..., 2, :]) - compute_e(p[..., 3, :]),
            tex_label=f"E_{{W_l}} - E_{{l}}",
            unit="GeV",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=-50, upper=500
            ),
            yscale="linear",
        )
    )
    observables.append(
        Observable(
            compute=lambda p: compute_e(p[..., 7, :]) - compute_e(p[..., 8, :]),
            tex_label=f"E_{{W_h}} - E_{{q_1}}",
            unit="GeV",
            bins=lambda obs: get_hardcoded_bins(
                obs, n_bins=n_bins, lower=-50, upper=500
            ),
            yscale="linear",
        )
    )

    for i in range(5):
        for j in range(i+1, 5):
            observables.append(
                Observable(
                    compute=lambda p, i=i, j=j: compute_deltaR(p, ind_1=i, ind_2=j),
                    tex_label=f"\Delta R_{{{particle_names[i]}, {particle_names[j]}}}",
                    bins=lambda obs: get_quantile_bins(
                        obs, n_bins=n_bins, lower=1e-2, upper=1e-2
                    ),
                    yscale="linear",
                )
            )
    for i in range(5, 10):
        for j in range(i+1, 10):
            observables.append(
                Observable(
                    compute=lambda p, i=i, j=j: compute_deltaR(p, ind_1=i, ind_2=j),
                    tex_label=f"\Delta R_{{{particle_names[i]}, {particle_names[j]}}}",
                    bins=lambda obs: get_quantile_bins(
                        obs, n_bins=n_bins, lower=1e-2, upper=1e-2
                    ),
                    yscale="linear",
                )
            )




    return observables
