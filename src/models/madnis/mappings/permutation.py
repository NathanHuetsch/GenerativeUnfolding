""" Random Permutation """

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import special_ortho_group

from .base import Mapping
from .linear import LinearMapping


# pylint: disable=C0103
class Permutation(Mapping):
    """Base class for simple permutations
    Permutes along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in: int, dims_c: Optional[int], permutation: torch.Tensor):
        """
        Additional args in docstring of base class base.InvertibleModule.
        Args:
            only one of the two needed:
            - permutation: a permutation of range(dims_in)
            - permutation_matrix: a (dims_in x dims_in) matrix describing the permutation
        """
        super().__init__(dims_in, dims_c)
        if permutation.shape != (dims_in, ):
            raise ValueError("Permutation must have shape (dims_in, )")
        self.register_buffer("permutation", permutation)
        self.register_buffer("permutation_inv", permutation.argsort())

    def _forward(self, x: torch.Tensor, condition: Optional[torch.Tensor], **kwargs):
        del condition
        return (
            x[:, self.permutation],
            torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        )

    def _inverse(self, z: torch.Tensor, condition: Optional[torch.Tensor], **kwargs):
        del condition
        return (
            z[:, self.permutation_inv],
            torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        )

    def _log_det(self, x_or_z, condition: Optional[torch.Tensor] = None, inverse=False):
        del condition
        return torch.zeros(x_or_z.shape[0], dtype=x_or_z.dtype, device=x_or_z.device)


# pylint: disable=C0103
class PermuteExchange(Permutation):
    """Constructs a permutation that just exchanges the sets A and B of the Coupling Layer.
    Permutes along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in: int, dims_c: Optional[int]):
        permutation = torch.cat((
            torch.arange(dims_in // 2, dims_in),
            torch.arange(0, dims_in // 2)
        ))
        super().__init__(dims_in, dims_c, permutation)


# pylint: disable=C0103
class PermuteRandom(Permutation):
    """Constructs a random permutation, that stays fixed during training.
    Permutes along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in: int, dims_c: Optional[int]):
        super().__init__(dims_in, dims_c, torch.randperm(dims_in))

# pylint: disable=C0103
class PermuteSoft(LinearMapping):
    """Constructs a soft permutation, that stays fixed during training.
    Perfoms a rotation along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in: int, dims_c: Optional[int], seed: Optional[int] = None):
        """
        Additional args in docstring of base class base.InvertibleModule.
        Args:
          seed: Int seed for the permutation (numpy is used for RNG). If seed is
            None, do not reseed RNG.
        """
        m = torch.tensor(special_ortho_group.rvs(dims_in, random_state=seed))
        super().__init__(dims_in, dims_c, m)


# pylint: disable=C0103
class PermuteSoftLearn(Mapping):
    """Constructs a soft permutation, that is learnable in training.
    Perfoms a rotation along the first (channel-) dimension for multi-dimenional tensors.
    Rotations are parametrized by their Euler angles.
    Formulas are based on "Generalization of Euler Angles to N‐Dimensional Orthogonal Matrices"
    Journal of Mathematical Physics 13, 528 (1972); https://doi.org/10.1063/1.1666011
    David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
    Algorithm inspired by
    https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/utils/mathematics.py
    """

    def __init__(self, dims_in, dims_c=None):
        """
        Additional args in docstring of base class base.InvertibleModule.
        Args:
        """
        super().__init__(dims_in, dims_c)

        # initialize k*(k-1)/2 angles. The majority is in [-pi/2, pi/2], but some are in [-pi, pi]
        # trainable parameters are unbounded, so we have to ensure the boundaries ourselves

        # number of all angles
        num_all = dims_in * (dims_in - 1) // 2
        # which indices are in the larger domain:
        indices_full = list(num_all - 1 - np.cumsum(np.arange(dims_in-1)))
        # build masks
        mask_full = np.zeros(num_all, dtype=bool)
        mask_full[indices_full] = True
        self.indices_red = np.arange(num_all)[~mask_full]
        # number of angles in reduced domain
        num_reduced = num_all - len(indices_full)
        # initialize trainable parameters such that they cover final angle space more or less
        # uniformly. Found empirically based on subsequent transformations.
        init_all = torch.zeros(num_all)
        init_all[indices_full] = torch.rand(len(indices_full)) * 2. - 1.
        init_all[self.indices_red] = torch.randn(len(self.indices_red)) * 1.5

        self.perm_ang = nn.Parameter(init_all)

    def _forward(self, x: torch.Tensor, condition: Optional[torch.Tensor], **kwargs):
        return (
            x @ self._translate_to_matrix(),
            torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        )

    def _inverse(self, z: torch.Tensor, condition: Optional[torch.Tensor], **kwargs):
        return (
            z @ self._translate_to_matrix().t(),
            torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        )

    def _translate_to_matrix(self):
        """ translates the trainable parameters to angles in the right domain and then to
            the rotation matrix
        """
        perm_ang = self.perm_ang.clone()
        # ensure that it stays in reduced domain:
        perm_ang[self.indices_red] = torch.sigmoid(perm_ang[self.indices_red]) - 0.5
        return self._gea_orthogonal_from_angles(perm_ang * np.pi)

    def _gea_orthogonal_from_angles(self, angles_list):
        """
        Generalized Euler Angles
        Return the orthogonal matrix from its generalized angles
        Formulas are based on "Generalization of Euler Angles to N-Dimensional Orthogonal Matrices"
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011
        Algorithm adapted from numpy version at
        https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/utils/mathematics.py
        :param angles_list: List of angles, for a k-dimensional space the total number
                        of angles is k*(k-1)/2
        """

        b = torch.eye(2, dtype=angles_list.dtype, device=angles_list.device)
        tmp = angles_list

        # For SO(k) there are k*(k-1)/2 angles that are grouped in k-1 sets
        # { (k-1 angles), (k-2 angles), ... , (1 angle)}
        for i in range(1, self.dims_in):
            angles = nn.functional.pad(tmp[-i:], (0, 1), "constant", np.pi/2)
            tmp = tmp[:-i]
            ma = self._gea_matrix_a(angles)  # matrix i+1 x i+1
            b = ma.t() @ b
            # We skip doing making a larger matrix for the last iteration
            if i < self.dims_in-1:
                c = nn.functional.pad(b, (0, 1, 0, 1), "constant")
                eye_i2 = torch.eye(i+2, dtype=angles_list.dtype, device=angles_list.device) 
                eye_i1 = torch.eye(i+1, dtype=angles_list.dtype, device=angles_list.device) 
                corr = eye_i2 - nn.functional.pad(eye_i1, (0, 1, 0, 1), "constant")
                b = c + corr
        return b

    def _gea_matrix_a(self, angles):
        """
        Generalized Euler Angles
        Return the parametric angles described on Eqs. 15-19 from the paper:
        Formulas are based on "Generalization of Euler Angles to N-Dimensional Orthogonal Matrices"
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011
        Algorithm adapted from numpy version at
        https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/utils/mathematics.py
        """
        n = len(angles)
        matrix_a = torch.eye(n, dtype=angles.dtype, device=angles.device)
        # Region I, eq. 16:
        cos_angles = torch.cos(angles)
        matrix_a = matrix_a * cos_angles
        # Region II, eq. 17 tan:
        tan_vec = torch.tan(angles)
        # Region II, eq. 17 cos:
        cos_vec = torch.cumprod(cos_angles, dim=0)
        # Region II, eq. 17 all:
        matrix_a[:,-1] += tan_vec * cos_vec
        # Region III, eq. 18 tan:
        region_iii_tan = - tan_vec[:,None] * tan_vec[None,:]
        # Region III, eq. 18, cos:
        shifted_cos = torch.nn.functional.pad(cos_vec[:-1], (1,0), "constant", 1)
        region_iii_cos = cos_vec[:,None] / shifted_cos[None,:]
        matrix_a += torch.tril(region_iii_tan * region_iii_cos, diagonal=-1)

        return matrix_a
