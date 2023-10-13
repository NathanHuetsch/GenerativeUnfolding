import numpy as np
from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from math import pi
import sys


ITER = 100
XTOL = 2e-12
RTOL = 4 * torch.finfo(float).eps


def map_fourvector_rambo(xs: torch.Tensor) -> torch.Tensor:
    """Transform unit hypercube points into into four-vectors."""
    cos = 2.0 * xs[:, :, 0] - 1.0
    phi = 2.0 * pi * xs[:, :, 1]

    q = torch.zeros_like(xs)
    q[:, :, 0] = -torch.log(xs[:, :, 2] * xs[:, :, 3])
    q[:, :, 1] = q[:, :, 0] * torch.sqrt(1 - cos**2) * np.cos(phi)
    q[:, :, 2] = q[:, :, 0] * torch.sqrt(1 - cos**2) * np.sin(phi)
    q[:, :, 3] = q[:, :, 0] * cos

    return q


def two_body_decay_factor(
    M_i_minus_1: torch.Tensor,
    M_i: torch.Tensor,
    m_i_minus_1: torch.Tensor,
) -> torch.Tensor:
    """Gives two-body decay factor from recursive n-body phase space"""
    return (
        1.0
        / (8 * M_i_minus_1**2)
        * torch.sqrt(
            (M_i_minus_1**2 - (M_i + m_i_minus_1) ** 2)
            * (M_i_minus_1**2 - (M_i - m_i_minus_1) ** 2)
        )
    )


def boost(Q: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Boost momentum q with 4-vector Q"""
    p = torch.empty_like(q)
    rsQ = torch.sqrt(Q[:, 0] ** 2 - Q[:, 1] ** 2 - Q[:, 2] ** 2 - Q[:, 3] ** 2)
    p[:, 0] = torch.einsum("ki,ki->k", Q, q) / rsQ
    c1 = (q[:, 0] + p[:, 0]) / (rsQ + Q[:, 0])
    p[:, 1:] = q[:, 1:] + c1[:, None] * Q[:, 1:]

    return p


def boost_z(q: torch.Tensor, eta: torch.Tensor, inverse: bool = False):
    """Boost momentum q along z-axis with given radidity"""
    p = torch.empty_like(q)
    sign = -1.0 if inverse else 1.0
    p[:, :, 0] = q[:, :, 0] * torch.cosh(eta) + sign * q[:, :, 3] * torch.sinh(eta)
    p[:, :, 1] = q[:, :, 1]
    p[:, :, 2] = q[:, :, 2]
    p[:, :, 3] = q[:, :, 3] * torch.cosh(eta) + sign * q[:, :, 0] * torch.sinh(eta)

    return p


def massless_propogator(
    r_or_s: torch.Tensor,
    s_min: torch.Tensor,
    s_max: torch.Tensor,
    nu: float = 0.95,
    m2_eps: float = -1e-3,
    inverse: bool = True,
):
    if s_min == 0 and nu > 1.0:
        m2_eps = torch.tensor(-1.0)
    elif s_min == 0:
        m2_eps = torch.tensor(m2_eps)
    else:
        m2_eps = torch.tensor(0.0)

    if nu == 1.0:
        if inverse:
            s = (
                torch.exp(
                    r_or_s * torch.log(s_max - m2_eps)
                    + (1 - r_or_s) * torch.log(s_min - m2_eps)
                )
                + m2_eps
            )
            logdet = -torch.log(
                (torch.log(s_max - m2_eps) - torch.log(s_min - m2_eps)) * (s - m2_eps)
            )
            return s, -logdet
        else:
            r = (torch.log(r_or_s - m2_eps) - torch.log(s_min - m2_eps)) / (
                torch.log(s_max - m2_eps) - torch.log(s_min - m2_eps)
            )
            logdet = -torch.log(
                (torch.log(s_max - m2_eps) - torch.log(s_min - m2_eps))
                * (r_or_s - m2_eps)
            )
            return r, logdet

    # if nu is not equal to 1
    if inverse:
        s = (
            r_or_s * (s_max - m2_eps) ** (1 - nu)
            + (1 - r_or_s) * (s_min - m2_eps) ** (1 - nu)
        ) ** (1 / (1 - nu)) + m2_eps
        logdet = torch.log(
            (1 - nu)
            / (
                (s - m2_eps) ** nu
                * ((s_max - m2_eps) ** (1 - nu) - (s_min - m2_eps) ** (1 - nu))
            )
        )
        return s, -logdet
    else:
        r = ((r_or_s - m2_eps) ** (1 - nu) - (s_min - m2_eps) ** (1 - nu)) / (
            (s_max - m2_eps) ** (1 - nu) - (s_min - m2_eps) ** (1 - nu)
        )
        logdet = torch.log(
            (1 - nu)
            / (
                (r_or_s - m2_eps) ** nu
                * ((s_max - m2_eps) ** (1 - nu) - (s_min - m2_eps) ** (1 - nu))
            )
        )
        return r, logdet


def rambo_func(
    x: Union[float, torch.Tensor],
    nparticles: int,
    xs: torch.Tensor,
    diff: bool = False,
) -> torch.Tensor:
    if isinstance(x, float):
        x = x * torch.ones_like(xs[:, 0 : nparticles - 2])
    elif isinstance(x, torch.Tensor):
        assert x.shape[1] == nparticles - 2
    else:
        raise ValueError("x is not valid input")

    i = torch.arange(2, nparticles)[None, :]
    f = (
        (nparticles + 1 - i) * x ** (2 * (nparticles - i))
        - (nparticles - i) * x ** (2 * (nparticles + 1 - i))
        - xs[:, 0 : nparticles - 2]
    )
    if diff:
        df = (nparticles + 1 - i) * (2 * (nparticles - i)) * x ** (
            2 * (nparticles - i) - 1
        ) - (nparticles - i) * (2 * (nparticles + 1 - i)) * x ** (
            2 * (nparticles + 1 - i) - 1
        )
        return df
    return f


def mass_func(
    x: Union[float, torch.Tensor],
    p: torch.Tensor,
    m: torch.Tensor,
    e_cm: torch.Tensor,
    diff: bool = False,
) -> torch.Tensor:
    if isinstance(x, float):
        x = x * torch.ones(m.shape[0], 1)
    elif isinstance(x, torch.Tensor):
        assert x.dim() == 1
    else:
        raise ValueError("x is not valid input")

    root = torch.sqrt(x[:, None] ** 2 * p[:, :, 0] ** 2 + m**2)
    f = torch.sum(root, dim=-1) - e_cm[:, 0]
    if diff:
        return torch.sum(x[:, None] * p[:, :, 0] ** 2 / root, dim=-1)
    return f


def newton(
    f: Callable,
    df: Callable,
    a: float,
    b: float,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = ITER,
    epsilon=1e-8,
):
    if torch.any(f(a) * f(b) > 0):
        raise ValueError(f"None or no unique root in given intervall [{a},{b}]")
    
    # Define lower/upper boundaries as tensor
    xa = a * torch.ones_like(f(x0))
    xb = b * torch.ones_like(f(x0))

    # initilize guess
    if x0 is None:
        x0 = (xa + xb) / 2

    for _ in range(max_iter):
        if torch.any(df(x0) < epsilon):
            raise ValueError("Derivative is too small")

        # do newtons-step
        x1 = x0 - f(x0) / df(x0)

        # check if within given intervall
        higher = x1 > xb
        lower = x1 < xa
        if torch.any(higher):
            x1[higher] = (xb[higher] + x0[higher]) / 2
        if torch.any(lower):
            x1[lower] = (xa[lower] + x0[lower]) / 2

        if torch.allclose(x1, x0, atol=XTOL, rtol=RTOL):
            return x1
        
        # Adjust brackets
        low = f(x1) * f(xa) > 0
        xa[low] = x1[low]
        xb[~low] = x1[~low]

        x0 = x1

    print(f"not converged")
    return x0
