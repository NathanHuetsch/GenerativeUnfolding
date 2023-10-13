import torch
import math
import numpy as np
import lhapdf
from .matrix_element import MatrixElement

DOWN = 1
UP = 2
STRANGE = 3
CHARM = 4
BOTTOM = 5
TOP = 6

ADOWN = -1
AUP = -2
ASTRANGE = -3
ACHARM = -4
ABOTTOM = -5
ATOP = -6

muF = 173.2
sqrts = 13000
s = sqrts**2


class CrossSection:
    def __init__(self):
        self.lo = MatrixElement()
        self.pdf = lhapdf.mkPDF("NNPDF23_nlo_as_0119_qed", 0)
        self.xmax = self.pdf.xMax

    def phase_space_dependence(self, momenta: torch.Tensor):
        p_top, p_higgs, p_jet = momenta[:, 0], momenta[:, 1], momenta[:, 2]
        e_total = momenta[:,:,0].sum(dim=1)
        pz_total = momenta[:,:,3].sum(dim=1)
        zeros = torch.zeros_like(e_total)

        pval1 = e_total + pz_total
        pval2 = e_total - pz_total
        xa = torch.clamp(pval1 / sqrts, max=self.xmax)
        xb = torch.clamp(pval2 / sqrts, max=self.xmax)
        ss = xa * xb * s
        p1 = torch.stack((pval1 / 2, zeros, zeros, pval1 / 2,), dim=1)
        p2 = torch.stack((pval2 / 2, zeros, zeros, -pval2 / 2,), dim=1)

        jact = (
            (2 * math.pi) ** (-5.0)
            / (8 * s)
            / (p_top[:,0] * p_higgs[:,0] * p_jet[:,0] * ss)
        )

        # calculate pdfs
        q = np.full(xa.shape, muF)
        fu1, fc1, fdbar1, fsbar1, fb1 = torch.split(torch.tensor(self.pdf.xfxQ(
            [UP, CHARM, ADOWN, ASTRANGE, BOTTOM], xa.cpu().numpy(), q
        ), device=xa.device, dtype=xa.dtype) / xa[:, None], 1, dim=1)
        fu2, fc2, fdbar2, fsbar2, fb2 = torch.split(torch.tensor(self.pdf.xfxQ(
            [UP, CHARM, ADOWN, ASTRANGE, BOTTOM], xb.cpu().numpy(), q
        ), device=xa.device, dtype=xa.dtype) / xb[:, None], 1, dim=1)

        res = self.lo.LO_bu_tdh_t_channel_vec(p1, p2, p_top, p_jet, p_higgs)
        return jact[:,None] * (
            (fb1 * fu2 + fb1 * fc2)
            * self.lo.LO_bu_tdh_t_channel_vec(p1, p2, p_top, p_jet, p_higgs)
            + (fu1 * fb2 + fc1 * fb2)
            * self.lo.LO_bu_tdh_t_channel_vec(p2, p1, p_top, p_jet, p_higgs)
            + (fb1 * fdbar2 + fb1 * fsbar2)
            * self.lo.LO_bdbar_tubarh_t_channel_vec(p1, p2, p_top, p_jet, p_higgs)
            + (fdbar1 * fb2 + fsbar1 * fb2)
            * self.lo.LO_bdbar_tubarh_t_channel_vec(p2, p1, p_top, p_jet, p_higgs)
        )

    def alpha_dependence(self, alpha: torch.Tensor):
        angle = np.pi / 180. * alpha
        sina = angle.sin()
        cosa = angle.cos()
        ones = torch.ones_like(sina)
        return torch.stack((ones, sina, cosa, sina*cosa, sina*sina), dim=-1)
