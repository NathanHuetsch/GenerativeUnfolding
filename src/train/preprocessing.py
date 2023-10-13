from typing import Tuple, Optional, Union, Iterable
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from math import pi, gamma
from .utils import (
    boost,
    boost_z,
    two_body_decay_factor,
    massless_propogator,
    newton,
    mass_func,
    rambo_func,
)

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
        norm_keep_zeros: bool = False,
        norm_mask: Optional[torch.Tensor] = None,
        individual_norms: bool = True,
    ):
        if any(
            tp.output_shape != tn.input_shape
            for i, (tp, tn) in enumerate(zip(trafos[:-1], trafos[1:]))
        ):
            raise ValueError(
                f"Output shape {trafos[0].output_shape} of transformation {0} not "
                + f"equal to input shape {trafos[1].input_shape} of transformation {1}"
            )
        if normalize:
            trafos.append(NormalizationPreproc(trafos[-1].output_shape, norm_keep_zeros, norm_mask))
        super().__init__(
            trafos[0].input_shape,
            trafos[-1].output_shape,
            all(t.invertible for t in trafos),
            all(t.has_jacobian for t in trafos),
        )
        self.trafos = nn.ModuleList(trafos)
        self.normalize = normalize
        self.norm_keep_zeros = norm_keep_zeros
        self.individual_norms = individual_norms

    def init_normalization(self, x: torch.Tensor, batch_size: int = 100000):
        if not self.normalize:
            return
        xbs = []
        for xb in x.split(batch_size, dim=0):
            for t in self.trafos[:-1]:
                xb = t(xb)
            xbs.append(xb)
        x = torch.cat(xbs, dim=0)
        norm_dims = 0 if self.individual_norms else tuple(range(len(x.shape)-1))
        if self.norm_keep_zeros:
            x_count = torch.sum(x != 0.0, dim=norm_dims, keepdims=True)
            x_mean = x.sum(dim=norm_dims, keepdims=True) / x_count
            x_std = torch.sqrt(torch.sum((x - x_mean) ** 2, dim=norm_dims, keepdims=True) / x_count)
        else:
            x_mean = x.mean(dim=norm_dims, keepdims=True)
            x_std = x.std(dim=norm_dims, keepdims=True)
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
        shape: Tuple[int, ...],
        keep_zeros: bool = False,
        norm_mask: Optional[torch.Tensor] = None
    ):
        super().__init__(
            input_shape=shape, output_shape=shape, invertible=True, has_jacobian=True
        )
        self.mean = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.std = nn.Parameter(torch.ones(shape), requires_grad=False)
        self.jac = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.keep_zeros = keep_zeros
        if norm_mask is None:
            self.norm_mask = nn.Parameter(torch.ones(shape, dtype=torch.bool), requires_grad=False)
        else:
            self.norm_mask = nn.Parameter(norm_mask, requires_grad=False)

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor):
        with torch.no_grad():
            self.mean.data.copy_(torch.where(self.norm_mask, mean, 0.))
            self.std.data.copy_(torch.where(self.norm_mask, std, 1.))
            self.jac.data.copy_(std[self.norm_mask].log().sum())

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rev:
            z, jac = x * self.std + self.mean, self.jac
        else:
            z, jac = (x - self.mean) / self.std, -self.jac

        if self.keep_zeros:
            z = torch.where(x != 0.0, z, 0.0)

        return z, jac.expand(x.shape[0])  # TODO: fix jacobians


class ScalePreproc(PreprocTrafo):
    def __init__(self, factors):
        ftensor = torch.tensor(factors)
        super().__init__(
            input_shape=ftensor.shape,
            output_shape=ftensor.shape,
            invertible=True,
            has_jacobian=True,
        )
        self.factors = nn.Parameter(ftensor, requires_grad=False)
        self.jac = nn.Parameter(ftensor.log().sum(), requires_grad=False)

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rev:
            z, jac = x * self.factors, self.jac
        else:
            z, jac = x / self.factors, -self.jac
        return z, jac.expand(x.shape[0])


class MaskPreproc(PreprocTrafo):
    def __init__(self, input_shape, mask):
        self.mask = torch.tensor(mask)
        if tuple(self.mask.shape) != (np.prod(input_shape),):
            raise ValueError(
                f"Mask shape {self.mask.shape} incompatible with input shape {input_shape}"
            )
        output_shape = (self.mask.shape[0],)
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            invertible=False,
            has_jacobian=False,
        )

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = x.reshape((x.shape[0], -1))[:, self.mask]
        return y, torch.zeros(x.shape[:1], dtype=x.dtype, device=x.device)


class NetworkPreproc(PreprocTrafo):
    def __init__(self, network, input_shape, output_shape, drop_half=False):
        super().__init__(
            input_shape, output_shape, invertible=False, has_jacobian=False
        )
        self.network = network
        self.drop_half = drop_half

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            y = self.network(x)
            if self.drop_half:
                y = y[:, : y.shape[1] // 2]
            return y, torch.zeros(x.shape[:1], dtype=x.dtype, device=x.device)


class MultiplicityPreproc(PreprocTrafo):
    def __init__(self, n_total: int, n_min: int):
        self.n_min = n_min
        self.n_total = n_total
        self.n_classes = n_total - n_min + 1
        super().__init__(
            input_shape=(n_total, 4),
            output_shape=(self.n_classes, ),
            invertible=True,
            has_jacobian=True,
        )

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rev:
            y = torch.cat((
                torch.ones((x.shape[0], self.n_min-1), dtype=x.dtype, device=x.device),
                (1 - x.cumsum(dim=1) + x)
            ), dim=1)[:,:,None].expand(-1,-1,4)
            return y, torch.zeros(x.shape[:1], dtype=x.dtype, device=x.device)
        else:
            multiplicity = torch.count_nonzero(x[:,:self.n_total,0], dim=1)
            return (
                F.one_hot(multiplicity - self.n_min, num_classes=self.n_classes),
                torch.zeros(x.shape[:1], dtype=x.dtype, device=x.device)
            )


class TransfermerPreproc(PreprocTrafo):
    def __init__(
        self,
        masses: list[Optional[float]],
        pt_eta_phi: bool = False,
        eta_cut: Optional[float] = None,
        full_momenta: bool = False
    ):
        n_particles = len(masses)
        self.pt_eta_phi = pt_eta_phi
        self.masses = [None] * n_particles if full_momenta else masses
        self.norm_mask = torch.tensor([
            [True, not pt_eta_phi or eta_cut is None, not pt_eta_phi, mass is None]
            for mass in self.masses
        ])
        self.eta_cut = eta_cut
        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(n_particles, 4),
            invertible=True,
            has_jacobian=True,
        )

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_jac = torch.zeros(x.shape[:1], dtype=x.dtype, device=x.device)
        if rev:
            zero_mask = x[:,:,0] == 0.
            x = x.clone()
            zero = torch.tensor(0., dtype=x.dtype, device=x.device)
            for i, mass in enumerate(self.masses):
                if mass is not None:
                    x[:, i, -1] = torch.where(x[:, i, 0] != 0, mass, zero)
                else:
                    x[:, i, -1] = torch.where(x[:, i, -1] != 0, x[:, i, -1].exp(), 0)
            if self.pt_eta_phi:
                pt, eta, phi = x[:,:,0].exp(), x[:,:,1], x[:,:,2] * pi
                if self.eta_cut is not None:
                    #eta = torch.tanh(eta) / 0.9999 * self.eta_cut
                    eta = eta * self.eta_cut
                px = pt * phi.cos()
                py = pt * phi.sin()
                pz = pt * eta.sinh()
                x[:,:,0], x[:,:,1], x[:,:,2] = px, py, pz
            y = torch.cat(
                (torch.sum(x**2, dim=2, keepdims=True).sqrt(), x[:, :, :-1]),
                dim=2,
            )
            y[zero_mask] = 0.
            return y, log_jac
        else:
            if self.pt_eta_phi:
                e, px, py, pz = x[:,:,0], x[:,:,1], x[:,:,2], x[:,:,3]
                phi = torch.arctan2(py, px) / pi
                log_pt = torch.sqrt(px**2 + py**2).log()
                p2 = px**2 + py**2 + pz**2
                limit = 1 - 1e-6
                eta = torch.arctanh(torch.clamp(pz / torch.sqrt(p2), min=-limit, max=limit))
                if self.eta_cut is not None:
                    #eta = torch.arctanh(eta / self.eta_cut * 0.9999)
                    eta = torch.clamp(eta / self.eta_cut, min=-1, max=1)
                log_m = 0.5 * torch.clamp(e**2 - p2, min=1e-6).log()
                for i, mass in enumerate(self.masses):
                    if mass is not None:
                        log_m[:, i] = 0.
                zero_mask = e == 0.
                log_pt[zero_mask] = 0.
                eta[zero_mask] = 0.
                phi[zero_mask] = 0.
                log_m[zero_mask] = 0.
                return torch.stack((log_pt, eta, phi, log_m), dim=2), log_jac
            log_m = 0.5*torch.clamp(
                x[:, :, 0:1] ** 2 - x[:, :, 1:2] ** 2 - x[:, :, 2:3] ** 2 - x[:, :, 3:4] ** 2,
                min=1e-6,
            ).log()
            for i, mass in enumerate(self.masses):
                if mass is not None:
                    log_m[:, i, :] = 0.
            return (torch.cat((x[:, :, 1:], log_m), dim=2), log_jac)


class MomentumPreproc(PreprocTrafo):
    def __init__(
        self,
        masses=[],
        full_momenta=False,
        m_pt_phi_eta=False,
        pt_conserved=False,
        mppp = False
    ):
        n_particles = len(masses)
        self.m_pt_phi_eta = m_pt_phi_eta
        self.mppp = mppp
        if full_momenta:
            self.onshell = []
            self.masses = []
        else:
            self.onshell = [i for i, m in enumerate(masses) if m is not None]
            self.masses = [m for m in masses if m is not None]
        self.pt_conserved = pt_conserved
        output_dim = 4 * n_particles - 2 * int(pt_conserved) - len(self.onshell)
        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(output_dim,),
            invertible=True,
            has_jacobian=True,
        )

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rev:
            return self.to_four_momenta(x)
        else:
            return self.from_four_momenta(x)

    def from_four_momenta(self, p):
        if self.m_pt_phi_eta:
            e, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
            m = torch.sqrt(torch.clip(e**2 - px**2 - py**2 - pz**2, 0, None))
            pt = torch.sqrt(px**2 + py**2)
            phi = torch.atan2(py, px)
            eps = 1e-7
            phi_mod = torch.arctanh(torch.clip(phi / pi, -1 + eps, 1 - eps))
            pabs = torch.sqrt(px**2 + py**2 + pz**2)
            eta = 0.5 * (
                torch.log(torch.clip(torch.abs(pabs + pz), 1e-15, None))
                - torch.log(torch.clip(torch.abs(pabs - pz), 1e-15, None))
            )
            data = torch.stack([m, pt, phi_mod, eta], dim=-1)
        elif self.mppp:
            e, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
            m = torch.sqrt(torch.clamp(e ** 2 - px ** 2 - py ** 2 - pz ** 2, min=1e-6))
            data = torch.stack([m.log(), px, py, pz], dim=-1)
        else:
            data = p

        data_flat = data.reshape(*data.shape[:-2], -1)
        log_jac = torch.zeros(p.shape[:1], dtype=p.dtype, device=p.device)
        mask = torch.full((data_flat.shape[-1],), True)
        for i in range(data.shape[-2]):
            if i in self.onshell:
                mask[4 * i] = False
            elif self.m_pt_phi_eta:
                log_jac += torch.log(e[:, i] / m[:, i])
            elif self.mppp:
                log_jac += torch.log(e[:, i] / m[:, i]**2)
        if self.pt_conserved:
            mask[-2] = False
            mask[-3] = False
        if self.m_pt_phi_eta:
            log_jac -= torch.sum(torch.log(pabs), dim=-1)
            pt_count = pt.shape[-1] - int(self.pt_conserved)
            log_jac -= torch.sum(
                torch.log(
                    pt[..., :pt_count] * (pi**2 - phi[..., :pt_count] ** 2) / pi
                ),
                dim=-1,
            )

        return data_flat[:, mask], log_jac

    def to_four_momenta(self, data):
        n_momenta = (
            data.shape[-1] + len(self.onshell) + 2 * int(self.pt_conserved)
        ) // 4
        idx = 0
        p = torch.zeros((*data.shape[:-1], n_momenta, 4), device=data.device)
        log_jac = torch.zeros(data.shape[:1], dtype=data.dtype, device=data.device)
        if self.m_pt_phi_eta:
            for i in range(n_momenta):
                if self.pt_conserved and i == n_momenta - 1:
                    if i in self.onshell:
                        m = self.masses[self.onshell.index(i)]
                        eta = data[..., idx]
                    else:
                        m, eta = data[..., idx], data[..., idx + 1]
                    px = -torch.sum(p[..., :-1, 1], dim=-1)
                    py = -torch.sum(p[..., :-1, 2], dim=-1)
                    pt = torch.sqrt(px**2 + py**2)
                else:
                    if i in self.onshell:
                        m = self.masses[self.onshell.index(i)]
                        pt, phi_mod, eta = (
                            data[..., idx],
                            data[..., idx + 1],
                            data[..., idx + 2],
                        )
                        idx += 3
                    else:
                        m, pt, phi_mod, eta = (
                            data[..., idx],
                            data[..., idx + 1],
                            data[..., idx + 2],
                            data[..., idx + 3],
                        )
                        idx += 4
                    phi = pi * torch.tanh(phi_mod)
                    px = pt * torch.cos(phi)
                    py = pt * torch.sin(phi)
                    log_jac += torch.log(pt * (pi**2 - phi**2) / pi)
                pz = pt * torch.sinh(eta)
                pabs2 = px**2 + py**2 + pz**2
                e = torch.sqrt(m**2 + pabs2)
                p[..., i, :] = torch.stack([e, px, py, pz], dim=-1)
                log_jac += 0.5 * torch.log(pabs2)
                if i not in self.onshell:
                    log_jac -= torch.log(e / m)

        elif self.mppp:
            for i in range(n_momenta):
                if self.pt_conserved and i == n_momenta - 1:
                    if i in self.onshell:
                        m = self.masses[self.onshell.index(i)]
                        pz = data[..., idx]
                    else:
                        m, pz = data[:, idx], data[..., idx + 1]
                    px = -torch.sum(p[..., :-1, 1], dim=-1)
                    py = -torch.sum(p[..., :-1, 2], dim=-1)
                else:
                    if i in self.onshell:
                        m, px, py, pz = (
                            None,
                            data[..., idx],
                            data[..., idx + 1],
                            data[..., idx + 2],
                        )
                        idx += 3
                    else:
                        m, px, py, pz = (
                            data[..., idx].exp(),
                            data[..., idx + 1],
                            data[..., idx + 2],
                            data[..., idx + 3],
                        )
                        idx += 4
                if m is None:
                    m = self.masses[self.onshell.index(i)]
                e = torch.sqrt(m**2 + px**2 + py**2 + pz**2)
                p[..., i, :] = torch.stack([e, px, py, pz], dim=-1)
                if i not in self.onshell:
                    log_jac -= torch.log(e / m**2)

        else:
            for i in range(n_momenta):
                if self.pt_conserved and i == n_momenta - 1:
                    if i in self.onshell:
                        e, pz = None, data[..., idx]
                    else:
                        e, pz = data[:, idx], data[..., idx + 1]
                    px = -torch.sum(p[..., :-1, 1], dim=-1)
                    py = -torch.sum(p[..., :-1, 2], dim=-1)
                else:
                    if i in self.onshell:
                        e, px, py, pz = (
                            None,
                            data[..., idx],
                            data[..., idx + 1],
                            data[..., idx + 2],
                        )
                        idx += 3
                    else:
                        e, px, py, pz = (
                            data[..., idx],
                            data[..., idx + 1],
                            data[..., idx + 2],
                            data[..., idx + 3],
                        )
                        idx += 4
                if e is None:
                    m = self.masses[self.onshell.index(i)]
                    e = torch.sqrt(m**2 + px**2 + py**2 + pz**2)
                p[..., i, :] = torch.stack([e, px, py, pz], dim=-1)

        return p, log_jac


class ThreePartonTransformation(PreprocTrafo):
    def __init__(self, mt=173.2, mh=125.0, sqrt_s=13000):
        super().__init__(
            input_shape=(3, 4), output_shape=(7,), invertible=True, has_jacobian=True
        )
        self.mt = mt
        self.mh = mh
        self.sqrt_s = sqrt_s
        self.s = sqrt_s**2
        self.shmn = (mt + mh) ** 2
        self.shrng = self.s - self.shmn
        self.mtHmn = mt + mh

    def boost_had(self, xa, xb, p):
        xa, xb = xa[:, None], xb[:, None]
        sqrtxaxb = 2 * torch.sqrt(xa * xb)
        g = (xa + xb) / sqrtxaxb
        bg = (xb - xa) / sqrtxaxb
        return torch.stack(
            (
                g * p[..., 0] - bg * p[..., 3],
                p[..., 1],
                p[..., 2],
                g * p[..., 3] - bg * p[..., 0],
            ),
            dim=-1,
        )

    def boost_cms(self, pr, p, inv):
        pr = pr[:, None, :]
        mr = torch.sqrt(
            pr[..., 0] ** 2 - pr[..., 1] ** 2 - pr[..., 2] ** 2 - pr[..., 3] ** 2
        )
        ga = pr[..., 0] / mr
        vp_pr_p = torch.sum(pr[..., 1:] * p[..., 1:], dim=-1)

        return torch.cat(
            (
                ga[..., None] * p[..., :1] - inv * (vp_pr_p / mr)[..., None],
                -inv * pr[..., 1:] * p[..., :1] / mr[:, None]
                + p[..., 1:]
                + ((ga - 1.0) * vp_pr_p / torch.sum(pr[..., 1:] * pr[..., 1:], dim=-1))[
                    ..., None
                ]
                * pr[..., 1:],
            ),
            dim=-1,
        )

    def transform(
        self, x: torch.Tensor, rev: bool, jac: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not rev:
            p = x.double()
            e_tot = p[:, 0, 0] + p[:, 1, 0] + p[:, 2, 0]
            pz_tot = p[:, 0, 3] + p[:, 1, 3] + p[:, 2, 3]
            xac = (e_tot + pz_tot) / self.sqrt_s
            xbc = (e_tot - pz_tot) / self.sqrt_s
            p_had = self.boost_had(xbc, xac, p)
            ptpHh = p_had[:, 0] + p_had[:, 1]
            mtHc = torch.sqrt(
                ptpHh[..., 0] ** 2
                - ptpHh[..., 1] ** 2
                - ptpHh[..., 2] ** 2
                - ptpHh[..., 3] ** 2
            )
            p_cms = self.boost_cms(ptpHh, p_had[:, :1], 1)
            sh = xac * xbc * self.s
            sh_sqrt = sh.sqrt()
            sp = xac * self.s
            mtHrng = sh_sqrt - self.mtHmn
            ptsabs = torch.sum(p_cms[:, 0, 1:] ** 2, dim=-1).sqrt()
            pjhabs = torch.sum(p_had[:, 2, 1:] ** 2, dim=-1).sqrt()

            z_sig = torch.stack(
                (
                    (xac * xbc * self.s - self.shmn) / (self.s - self.shmn),
                    xac * self.s * (1.0 - xbc) / (self.s - xac * xbc * self.s),
                    (mtHc - self.mt - self.mh)
                    / (torch.sqrt(xac * xbc * self.s) - self.mt - self.mh),
                    (torch.atan2(p_cms[:, 0, 2], p_cms[:, 0, 1]) + pi) / (2.0 * pi),
                    (p_cms[:, 0, 3] / ptsabs + 1.0) / 2.0,
                    (torch.atan2(p_had[:, 2, 2], p_had[:, 2, 1]) + pi) / (2.0 * pi),
                    (p_had[:, 2, 3] / pjhabs + 1.0) / 2.0,
                ),
                dim=-1,
            )
            z = torch.logit(z_sig)
        else:
            x = torch.sigmoid(x.double())
            z_sig = x
            sh = self.shmn + self.shrng * x[:, 0]
            sh_sqrt = sh.sqrt()
            sp = sh + (self.s - sh) * x[:, 1]
            xa = sp / self.s
            xb = sh / sp
            mtHrng = sh_sqrt - self.mtHmn
            mtH = self.mtHmn + mtHrng * x[:, 2]
            ptsabs = (
                torch.sqrt(
                    (mtH * mtH - self.shmn)
                    * (mtH * mtH - (self.mt - self.mh) * (self.mt - self.mh))
                )
                / 2.0
                / mtH
            )
            pjhabs = (sh - mtH * mtH) / 2.0 / sh_sqrt
            phts = -pi + 2.0 * pi * x[:, 3]
            costhts = -1.0 + 2.0 * x[:, 4]
            phjh = -pi + 2.0 * pi * x[:, 5]
            costhjh = -1.0 + 2.0 * x[:, 6]

            pts = torch.stack(
                (
                    torch.sqrt(ptsabs * ptsabs + self.mt * self.mt),
                    ptsabs * torch.sqrt(1.0 - costhts * costhts) * torch.cos(phts),
                    ptsabs * torch.sqrt(1.0 - costhts * costhts) * torch.sin(phts),
                    ptsabs * costhts,
                ),
                dim=-1,
            )
            pHs = torch.stack(
                (
                    torch.sqrt(ptsabs * ptsabs + self.mh * self.mh),
                    -pts[:, 1],
                    -pts[:, 2],
                    -pts[:, 3],
                ),
                dim=-1,
            )
            pjh = torch.stack(
                (
                    pjhabs,
                    pjhabs * torch.sqrt(1.0 - costhjh * costhjh) * torch.cos(phjh),
                    pjhabs * torch.sqrt(1.0 - costhjh * costhjh) * torch.sin(phjh),
                    pjhabs * costhjh,
                ),
                dim=-1,
            )
            ptpHh = torch.stack(
                (
                    torch.sqrt(pjhabs * pjhabs + mtH * mtH),
                    -pjh[:, 1],
                    -pjh[:, 2],
                    -pjh[:, 3],
                ),
                dim=-1,
            )

            ptHh = self.boost_cms(ptpHh, torch.stack((pts, pHs), dim=1), -1)
            p = self.boost_had(xa, xb, torch.cat((ptHh, pjh[:, None, :]), dim=1))

        jac = 8 * pi**2 * self.shrng * ptsabs * pjhabs * (self.s - sh) * mtHrng / (
            sp * sh_sqrt
        ) * p[:, 0, 0] * p[:, 1, 0] * p[:, 2, 0] + (z_sig * (1 - z_sig)).log().sum(
            dim=1
        )
        if not rev:
            return z.float(), -jac.log().float()
        else:
            return p.float(), jac.log().float()


class Mahambo(PreprocTrafo):
    """Massive hadronic Rambo on diet (Mahambo) algorithm as preprocessing

    Adopted from
        [1] Rambo [Comput. Phys. Commun. 40 (1986) 359-373]
        [2] Rambo on diet [1308.2922]

    Note, that there is an error in the algorithm of [2], which has been fixed.
    For details see
        - "How to GAN - Novel simulation methods for the LHC", PhD thesis
        (10.11588/heidok.00029154)

    There is also an error in the derivation of the massive weight in [1].
    For details see appendix of [23XX.xxxx] (our paper).

    """

    def __init__(
        self,
        e_had: float,
        n_particles: int,
        masses: list = None,
        nu: float = 0.95,
        normal_latent: bool = True,
        epsilon=1e-8,
    ):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            n_particles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(3 * n_particles - 2,),
            invertible=True,
            has_jacobian=True,
        )

        # self.e_had = e_had
        self.n_particles = n_particles
        self.normal_latent = normal_latent
        self.epsilon = epsilon

        # Make sure the list is as long as the number of
        # finals state particles if given
        if masses is not None:
            self.masses = torch.tensor(masses)
            assert len(self.masses) == self.n_particles
        else:
            self.masses = masses

        # Define min energy due to masses (no cuts etc)
        e_min = torch.sum(self.masses) if self.masses is not None else torch.tensor(0.0)
        # self.tau_min = (e_min / self.e_had) ** 2
        self.s_max = e_had**2
        self.s_min = e_min**2

        # Define parameter for lumi integration
        self.nu = nu

    def transform(
        self,
        x_or_p: torch.Tensor,
        rev: bool,
        jac: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rev:
            p, logjac = self.rand_to_momenta(x_or_p)
            return p, logjac

        x, logjac = self.momenta_to_rand(x_or_p)
        return x, logjac

    def _rap_to_rand(
        self,
        s: torch.Tensor,
        r2_or_rap: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.log(s / self.s_max)
        if inverse:
            rap = (r2_or_rap - 0.5) * scale
            return rap, scale.abs().log().squeeze()

        r2 = r2_or_rap / scale + 0.5
        return r2, -scale.abs().log().squeeze()

    def _energy_to_rand(
        self,
        r1_or_s: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inverse:
            s, logdet = massless_propogator(
                r1_or_s, self.s_min, self.s_max, self.nu, inverse=True
            )
            return s, logdet.squeeze()

        r1, logdet = massless_propogator(
            r1_or_s, self.s_min, self.s_max, self.nu, inverse=False
        )
        return r1, logdet.squeeze()

    def _lumi_to_rand(
        self,
        r1_or_s: torch.Tensor,
        r2_or_rap: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inverse:
            s, ldj1 = self._energy_to_rand(r1_or_s, inverse=True)
            rap, ldj2 = self._rap_to_rand(s, r2_or_rap, inverse=True)
            return s, rap, ldj1 + ldj2

        r1, ldj1 = self._energy_to_rand(r1_or_s)
        r2, ldj2 = self._rap_to_rand(r1_or_s, r2_or_rap)
        return r1, r2, ldj1 + ldj2

    def _get_rapidity_and_energy(
        self, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = q[:, :, 0] ** 2 - q[:, :, 1] ** 2 - q[:, :, 2] ** 2 - q[:, :, 3] ** 2
        rapidity = torch.arctanh(q[:, :, 3] / q[:, :, 0])
        return s, rapidity

    def rand_to_momenta(self, rs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps random numbers onto momenta

        Args:
            rs (torch.Tensor): random numbers with shape (batch, 3 * n_particles - 2)

        Returns:
            p (torch.Tensor): momenta, shape (batch, 4 * n_particles)
            logdet (torch.Tensor): log determinant of mapping, shape (batch,)
        """
        # Map onto unit-hypercube
        # Map onto gaussian random numbers
        if self.normal_latent:
            xs = torch.sigmoid(rs)
            ld_sigmoid = -torch.sum(F.softplus(-rs) + F.softplus(rs), dim=1)
        else:
            xs = rs
            ld_sigmoid = torch.zeros(xs.shape[:1], dtype=xs.dtype, device=xs.device)

        # extract parts
        xs, r1, r2 = xs[:, 2:], xs[:, [0]], xs[:, [1]]

        # get partonic energies and boost variables
        s, rapidity, ld_lumi = self._lumi_to_rand(r1, r2, inverse=True)
        e_cm = torch.sqrt(s)

        # prepare momenta
        p = torch.empty((xs.shape[0], self.n_particles, 4))
        k = torch.empty((xs.shape[0], self.n_particles, 4))

        # q = np.empty((xs.shape[0], 4))
        M = torch.zeros((xs.shape[0], self.n_particles))
        u = torch.empty((xs.shape[0], self.n_particles - 2))

        Q = e_cm * torch.tile(torch.tensor([1, 0, 0, 0]), (xs.shape[0], 1))
        Q_prev = torch.empty((xs.shape[0], 4))
        M[:, 0] = e_cm[:, 0]

        # Solve equation numerically for all directly
        func = lambda x: rambo_func(x, self.n_particles, xs)
        df = lambda x: rambo_func(x, self.n_particles, xs, diff=True)
        guess = 0.5 * torch.ones((xs.shape[0], self.n_particles - 2))
        u[:, :] = newton(func, df, 0.0, 1.0, guess)

        for i in range(2, self.n_particles + 1):
            Q_prev[:, :] = Q[:, :]
            if i != self.n_particles:
                M[:, i - 1] = torch.prod(u[:, : i - 1], dim=1) * e_cm[:, 0]

            cos_theta = 2 * xs[:, self.n_particles - 6 + 2 * i] - 1
            phi = 2 * pi * xs[:, self.n_particles - 5 + 2 * i]
            q = 4 * M[:, i - 2] * two_body_decay_factor(M[:, i - 2], M[:, i - 1], 0)

            p[:, i - 2, 0] = q
            p[:, i - 2, 1] = q * torch.cos(phi) * torch.sqrt(1 - cos_theta**2)
            p[:, i - 2, 2] = q * torch.sin(phi) * torch.sqrt(1 - cos_theta**2)
            p[:, i - 2, 3] = q * cos_theta
            Q[:, 0] = torch.sqrt(q**2 + M[:, i - 1] ** 2)
            Q[:, 1:] = -p[:, i - 2, 1:]
            p[:, i - 2] = boost(Q_prev, p[:, i - 2])
            Q = boost(Q_prev, Q)

        p[:, self.n_particles - 1] = Q

        if self.masses is not None:
            # Define masses
            m = torch.tile(self.masses, (xs.shape[0], 1))

            # solve for xi in massive case, see Ref. [1]
            xi = torch.empty((xs.shape[0], 1, 1))
            func = lambda x: mass_func(x, p, m, e_cm)
            df = lambda x: mass_func(x, p, m, e_cm, diff=True)
            guess = 0.5 * torch.ones((xs.shape[0],))
            xi[:, 0, 0] = newton(func, df, 0.0, 1.0, guess)

            # Make them massive
            k[:, :, 0] = torch.sqrt(m**2 + xi[:, :, 0] ** 2 * p[:, :, 0] ** 2)
            k[:, :, 1:] = xi * p[:, :, 1:]

            # Get jacobian and then boost into lab frame
            jac = self.weight(e_cm[:, 0], k, p, xi)
            k = boost_z(k, rapidity, inverse=False)

            return k, jac.log() + ld_lumi + ld_sigmoid

        # Get jacobian and then boost into lab frame
        jac_rambo = self.weight(e_cm[:, 0], p, p)
        p = boost_z(p, rapidity, inverse=False)

        return p, jac_rambo.log() + ld_lumi + ld_sigmoid

    def momenta_to_rand(self, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps momenta onto random numbers.

        Args:
            k (torch.Tensor): momenta, shape (batch, n_particles, 4)

        Returns:
            xs (torch.Tensor): random numbers, shape (batch, 3 * n_particles - 2)
            logdet (torch.Tensor): log determinant of mapping, shape (batch,)
        """
        M = torch.empty(k.shape[0])
        M_prev = torch.empty(k.shape[0])
        Q = torch.empty((k.shape[0], 4))
        r = torch.empty((k.shape[0], 3 * self.n_particles - 4))

        # Boost into partonic CM frame and get x1 and x2
        q = torch.sum(k, dim=1, keepdim=True)
        s, rapidity = self._get_rapidity_and_energy(q)
        e_cm = torch.sqrt(s)
        k = boost_z(k, rapidity, inverse=True)

        # Make momenta massless before going back to random numbers
        p = torch.empty((k.shape[0], self.n_particles, 4))
        xi = torch.empty((k.shape[0], 1, 1))
        if self.masses is not None:
            # Define masses
            m = torch.tile(self.masses, (k.shape[0], 1))
            # solve for xi in massive case, see Ref. [1], here analytic result possible!
            xi[:, 0, 0] = (
                torch.sum(torch.sqrt(k[:, :, 0] ** 2 - m**2), dim=-1) / e_cm[:, 0]
            )
            # Make them massive
            p[:, :, 0] = torch.sqrt(k[:, :, 0] ** 2 - m**2) / xi[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:] / xi

        else:
            xi = None
            p[:, :, 0] = k[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:]

        # Get jacobian
        # (its the inverse! so return with minus later for logdet!)
        jac = self.weight(e_cm[:, 0], k, p, xi)

        Q[:] = p[:, -1]

        for i in range(self.n_particles, 1, -1):
            M_prev[:] = M[:]
            P = p[:, i - 2 :].sum(dim=1)
            M = torch.sqrt(P[:, 0] ** 2 - P[:, 1] ** 2 - P[:, 2] ** 2 - P[:, 3] ** 2)

            if i != self.n_particles:
                u = M_prev / M
                r[:, i - 2] = (self.n_particles + 1 - i) * u ** (
                    2 * (self.n_particles - i)
                ) - (self.n_particles - i) * u ** (2 * (self.n_particles + 1 - i))

            Q += p[:, i - 2]
            p_prime = boost(torch.einsum("ij,ki->kj", MINKOWSKI, Q), p[:, i - 2])
            r[:, self.n_particles - 6 + 2 * i] = 0.5 * (
                p_prime[:, 3] / torch.sqrt(torch.sum(p_prime[:, 1:] ** 2, dim=1)) + 1
            )
            phi = torch.arctan2(p_prime[:, 2], p_prime[:, 1])
            r[:, self.n_particles - 5 + 2 * i] = phi / (2 * pi) + (phi < 0)

        # get additional random numbers for the pdfs
        r1, r2, ld_lumi = self._lumi_to_rand(s, rapidity)
        r = torch.concat([r1, r2, r], dim=-1)

        # Map onto gaussian random numbers
        if self.normal_latent:
            r_normal = torch.logit(r)
            ld_logit = torch.sum(F.softplus(-r_normal) + F.softplus(r_normal), dim=1)
            return r_normal, -jac.log() + ld_lumi + ld_logit

        return r, -jac.log() + ld_lumi

    def weight(
        self,
        e_cm: torch.Tensor,
        k: torch.Tensor,
        p: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            e_cm (torch.Tensor): center-of-mass energy of partonic frame with shape (batch,)
            k (torch.Tensor): massive momenta in shape (batch, particles, 4)
            p (torch.Tensor): massless momenta in shape (batch, particles, 4)
            xi (torch.Tensor, Optional): shift variable with shape (batch,)

        Returns:
            torch.Tensor: weight of sampler
        """
        # get volume for massless particles
        w0 = (
            (pi / 2.0) ** (self.n_particles - 1)
            * e_cm ** (2 * self.n_particles - 4)
            / (gamma(self.n_particles) * gamma(self.n_particles - 1))
        )

        if xi is not None:
            xi = xi[:, 0, 0]
            # get correction factor for massive ones
            ks = torch.sqrt(k[:, :, 1] ** 2 + k[:, :, 2] ** 2 + k[:, :, 3] ** 2)
            ps = torch.sqrt(p[:, :, 1] ** 2 + p[:, :, 2] ** 2 + p[:, :, 3] ** 2)
            k0 = k[:, :, 0]
            p0 = p[:, :, 0]
            w_M = (
                xi ** (3 * self.n_particles - 3)
                * torch.prod(p0 / k0, dim=1)
                * torch.sum(ps**2 / p0, dim=1)
                / torch.sum(ks**2 / k0, dim=1)
            )
            return w0 * w_M

        return w0


def build_preprocessing(params: dict, masses: Optional[list[Optional[float]]] = None) -> PreprocChain:
    """
    Builds a preprocessing chain with the given parameters

    Args:
        params: dictionary with preprocessing parameters
    Returns:
        Preprocessing chain
    """
    pp_type = params.get("type", "momentum")
    normalize = True
    norm_mask = None
    individual_norms = params.get("individual_norms", True)
    args = params.get("args", {})
    if pp_type == "momentum":
        chain = [MomentumPreproc(masses = masses, **args)]
    elif pp_type == "multiplicity":
        chain = [MultiplicityPreproc(n_total=len(masses), **args)]
        normalize = False
    elif pp_type == "transfermer":
        chain = [TransfermerPreproc(masses = masses, **args)]
        norm_mask = chain[0].norm_mask
    elif pp_type == "three_parton":
        chain = [ThreePartonTransformation(**args)]
    elif pp_type == "mahambo":
        chain = [Mahambo(**args)]
        if not args.get("gaussian_latent", True):
            normalize = False
    elif pp_type == "scale":
        chain = [ScalePreproc(**args)]
    else:
        raise ValueError(f"Unknown preprocessing {pp_type}")

    mask_pp = params.get("mask")
    if mask_pp is not None:
        chain.append(MaskPreproc(chain[-1].output_shape, mask_pp))

    return PreprocChain(
        chain,
        normalize=normalize,
        norm_keep_zeros=params.get("keep_zeros", False),
        norm_mask = norm_mask,
        individual_norms = individual_norms,
    )
