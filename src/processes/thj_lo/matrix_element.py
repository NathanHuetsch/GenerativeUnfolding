import torch
import math


class MatrixElement:
    """Everything for the LO matrix element"""

    def __init__(self):
        self.aEW = 0.007546771113978883
        self.sw2 = 0.2222465330928909
        self.vev = 246.2184768185336
        self.ktt = 1.0
        self.att = 2.0 / 3.0
        self.ytsign = 1.0
        self.mt = 173.2
        self.mtsq = self.mt**2
        self.mz = 91.188
        self.mw = self.mz * math.sqrt(1.0 - self.sw2)
        self.mwsq = self.mw**2
        self.mh = 125.0
        self.mhsq = self.mh**2

    def dotp(self, p1, p2):
        return (
            p1[:, 0] * p2[:, 0]
            - p1[:, 1] * p2[:, 1]
            - p1[:, 2] * p2[:, 2]
            - p1[:, 3] * p2[:, 3]
        )

    def imag_neg_sign(self, p):
        return torch.where(p[:, 0] < 0, -1.0j, 1.0)

    def s(self, p1, p2):
        pp = p1[:, 0] - p1[:, 1]
        qp = p2[:, 0] - p2[:, 1]
        pt = p1[:, 2] + 1.0j * p1[:, 3]
        qt = p2[:, 2] + 1.0j * p2[:, 3]
        sval = (
            self.imag_neg_sign(p1)
            * self.imag_neg_sign(p2)
            / torch.sqrt(torch.abs(qp * pp))
            * (pp * qt - qp * pt)
        )
        return sval

    def t(self, p1, p2):
        pp = p1[:, 0] - p1[:, 1]
        qp = p2[:, 0] - p2[:, 1]
        ptc = p1[:, 2] - 1.0j * p1[:, 3]
        qtc = p2[:, 2] - 1.0j * p2[:, 3]
        tval = (
            self.imag_neg_sign(p1)
            * self.imag_neg_sign(p2)
            / torch.sqrt(torch.abs(qp * pp))
            * (ptc * qp - qtc * pp)
        )

        return tval

    def qdef(self, p3, p1):
        scalar = self.mtsq / (2.0 * self.dotp(p1, p3))
        q1 = scalar[:,None] * p1
        q2 = p3 - q1
        return q1, q2

    def LO_iMup(self, p1, p2, p3, p4, p5):
        p1p2 = self.dotp(p1, p2)
        p1p3 = self.dotp(p1, p3)
        p1p4 = self.dotp(p1, p4)
        p2p3 = self.dotp(p2, p3)
        p2p4 = self.dotp(p2, p4)
        p3p4 = self.dotp(p3, p4)

        C = -1.0j * math.pi * self.aEW / self.sw2 / self.vev
        Ct = (
            torch.tensor([0,0,1], dtype=p1.dtype, device=p1.device).reshape(3,1)
            * self.ktt
            * self.ytsign
            * self.mt
            / (
                self.mwsq * self.mtsq
                - 4.0 * p1p2 * p2p4
                - 2.0 * p1p2 * self.mwsq
                + 4.0 * p1p4 * p2p4
                + 2.0 * p1p4 * self.mwsq
                + 2.0 * p2p4 * self.mtsq
                + 2.0 * p2p4 * self.mwsq
                + 4.0 * p2p4 * p2p4
            )
        )
        Cw = (
            torch.tensor([1,0,0], dtype=p1.dtype, device=p1.device).reshape(3,1)
            * -2.0
            * self.mwsq
            / (
                -self.mwsq * self.mtsq
                + self.mwsq * self.mwsq
                + 4.0 * p1p3 * p2p4
                + 2.0 * p1p3 * self.mwsq
                - 2.0 * p2p4 * self.mtsq
                + 2.0 * p2p4 * self.mwsq
            )
        )
        q1, q2 = self.qdef(p3, p1)
        iMup = -(
            (
                self.s(p1, p2)
                * (
                    Ct * self.mt * self.s(p1, p2) * self.t(p1, q2) * self.t(p2, p4)
                    - 2.0 * (Cw - Ct * self.mt) * p1p3 * self.t(p4, q2)
                )
            )
            / p1p3
        )
        return C * iMup.T

    def LO_iMdown(self, p1, p2, p3, p4, p5):
        p1p2 = self.dotp(p1, p2)
        p1p3 = self.dotp(p1, p3)
        p1p4 = self.dotp(p1, p4)
        p2p3 = self.dotp(p2, p3)
        p2p4 = self.dotp(p2, p4)
        p3p4 = self.dotp(p3, p4)

        C = -1.0j * math.pi * self.aEW / self.sw2 / (self.vev)
        Ct = (
            torch.tensor([0,0,1], dtype=p1.dtype, device=p1.device).reshape(3,1)
            * self.ktt
            * self.ytsign
            * self.mt
            / (
                self.mwsq * self.mtsq
                - 4.0 * p1p2 * p2p4
                - 2.0 * p1p2 * self.mwsq
                + 4.0 * p1p4 * p2p4
                + 2.0 * p1p4 * self.mwsq
                + 2.0 * p2p4 * self.mtsq
                + 2.0 * p2p4 * self.mwsq
                + 4.0 * p2p4 * p2p4
            )
        )
        Cw = (
            torch.tensor([1,0,0], dtype=p1.dtype, device=p1.device).reshape(3,1)
            * -2.0
            * self.mwsq
            / (
                -self.mwsq * self.mtsq
                + self.mwsq * self.mwsq
                + 4.0 * p1p3 * p2p4
                + 2.0 * p1p3 * self.mwsq
                - 2.0 * p2p4 * self.mtsq
                + 2.0 * p2p4 * self.mwsq
            )
        )
        q1, q2 = self.qdef(p3, p1)
        iMdown = -(
            self.s(p1, p2)
            * (
                (
                    2.0 * Ct * self.mwsq * (self.mtsq + 2.0 * p1p3)
                    + Cw * self.mt * (self.mtsq - 2.0 * (self.mwsq + p1p3))
                )
                * self.s(p1, q2)
                * self.t(p1, p4)
                + 4.0 * Ct * self.mwsq * p1p3 * self.s(p2, q2) * self.t(p2, p4)
            )
            + 2.0
            * Cw
            * self.mt
            * p1p3
            * self.s(p1, q2)
            * self.s(p2, q2)
            * self.t(p4, q2)
        ) / (2.0 * self.mwsq * p1p3)
        return C * iMdown.T

    def LO_Anomalous_iMup(self, p1, p2, p3, p4, p5):
        p1p2 = self.dotp(p1, p2)
        p1p3 = self.dotp(p1, p3)
        p1p4 = self.dotp(p1, p4)
        p2p3 = self.dotp(p2, p3)
        p2p4 = self.dotp(p2, p4)
        p3p4 = self.dotp(p3, p4)

        C = -1.0j * math.pi * self.aEW / self.sw2 / (self.vev)
        Ct = (
            1.0j
            * torch.tensor([0,1,0], dtype=p1.dtype, device=p1.device).reshape(3,1)
            * self.att
            * self.ytsign
            * self.mt
            / (
                self.mwsq * self.mtsq
                - 4.0 * p1p2 * p2p4
                - 2.0 * p1p2 * self.mwsq
                + 4.0 * p1p4 * p2p4
                + 2.0 * p1p4 * self.mwsq
                + 2.0 * p2p4 * self.mtsq
                + 2.0 * p2p4 * self.mwsq
                + 4.0 * p2p4 * p2p4
            )
        )
        q1, q2 = self.qdef(p3, p1)
        iMup = (
            Ct
            * self.mt
            * self.s(p1, p2)
            * (
                (self.s(p1, p2) * self.t(p1, q2) * self.t(p2, p4)) / p1p3
                - 2.0 * self.t(p4, q2)
            )
        )
        return C * iMup.T

    def LO_Anomalous_iMdown(self, p1, p2, p3, p4, p5):
        p1p2 = self.dotp(p1, p2)
        p1p3 = self.dotp(p1, p3)
        p1p4 = self.dotp(p1, p4)
        p2p3 = self.dotp(p2, p3)
        p2p4 = self.dotp(p2, p4)
        p3p4 = self.dotp(p3, p4)

        C = -1.0j * math.pi * self.aEW / self.sw2 / (self.vev)
        q1, q2 = self.qdef(p3, p1)
        Ct = (
            1.0j
            * torch.tensor([0,1,0], dtype=p1.dtype, device=p1.device).reshape(3,1)
            * self.att
            * self.ytsign
            * self.mt
            / (
                self.mwsq * self.mtsq
                - 4.0 * p1p2 * p2p4
                - 2.0 * p1p2 * self.mwsq
                + 4.0 * p1p4 * p2p4
                + 2.0 * p1p4 * self.mwsq
                + 2.0 * p2p4 * self.mtsq
                + 2.0 * p2p4 * self.mwsq
                + 4.0 * p2p4 * p2p4
            )
        )
        iMdown = (
            Ct
            * self.s(p1, p2)
            * (
                -((self.mtsq - 2.0 * p1p3) * self.s(p1, q2) * self.t(p1, p4))
                + 2.0 * p1p3 * self.s(p2, q2) * self.t(p2, p4)
            )
        ) / p1p3
        return C * iMdown.T

    def alpha_vec_square(self, x: torch.Tensor):
        a, b, c = x[:,0], x[:,1], x[:,2]
        a_bar, b_bar, c_bar = a.conj(), b.conj(), c.conj()
        return torch.stack((
            a * a_bar + c * c_bar,
            2 * a * b_bar,
            2 * a * c_bar,
            2 * b * c_bar,
            b * b_bar - c * c_bar,
        ), axis=1).real

    def LO_bu_tdh_t_channel_vec(self, p1, p2, p3, p4, p5):
        up = self.LO_iMup(p1, p2, p3, p4, p5) + self.LO_Anomalous_iMup(
            p1, p2, p3, p4, p5
        )
        down = self.LO_iMdown(p1, p2, p3, p4, p5) + self.LO_Anomalous_iMdown(
            p1, p2, p3, p4, p5
        )
        return self.alpha_vec_square(up) + self.alpha_vec_square(down)

    def LO_bdbar_tubarh_t_channel_vec(self, p1, p2, p3, p4, p5):
        q1 = p1
        q2 = -p4
        q3 = p3
        q4 = -p2
        q5 = p5
        up = self.LO_iMup(q1, q2, q3, q4, q5) + self.LO_Anomalous_iMup(
            q1, q2, q3, q4, q5
        )
        down = self.LO_iMdown(q1, q2, q3, q4, q5) + self.LO_Anomalous_iMdown(
            q1, q2, q3, q4, q5
        )
        return self.alpha_vec_square(up) + self.alpha_vec_square(down)

    def LO_dbaru_tbbarh_s_channel_vec(self, p1, p2, p3, p4, p5):
        q1 = -p4
        q2 = p2
        q3 = p3
        q4 = -p1
        q5 = p5
        up = self.LO_iMup(q1, q2, q3, q4, q5) + self.LO_Anomalous_iMup(
            q1, q2, q3, q4, q5
        )
        down = self.LO_iMdown(q1, q2, q3, q4, q5) + self.LO_Anomalous_iMdown(
            q1, q2, q3, q4, q5
        )
        return self.alpha_vec_square(up) + self.alpha_vec_square(down)
