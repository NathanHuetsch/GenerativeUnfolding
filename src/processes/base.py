from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch

from .observables import Observable


@dataclass
class ProcessData:
    """
    Data class for training/validation/testing data for a process.
    Args:
        x_hard: Hard-scattering momenta, shape (n_events, n_hard_particles, 4)
        x_reco: Reco-level momenta, shape (n_events, n_reco_particles, 4)
        alpha: Theory parameters, shape (n_events, n_parameters)
        event_type: Type of the event, e.g. LO or NLO, as a one-hot encoded tensor,
                    shape (n_events, n_types), optional
    """

    x_hard: torch.Tensor
    x_reco: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    event_type: Optional[torch.Tensor] = None
    accepted: Optional[torch.Tensor] = None


class Process(ABC):
    @abstractmethod
    def __init__(self, params: dict, device: torch.device):
        """
        Loads the data and initializes the process object.

        Args:
            params: Parameters for the specific process
        """
        pass

    @abstractmethod
    def get_data(self, subset: str) -> ProcessData:
        """
        Returns data from the specified subset of the dataset.

        Args:
            subset: Which part of the data, e.g. "train", "val", "test"
        Returns:
            ProcessData object containing the data
        """
        pass

    @abstractmethod
    def diff_cross_section(
        self,
        x_hard: torch.Tensor,
        alpha: torch.Tensor,
        event_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the differential cross section for the given hard-scattering momenta and
        theory parameters.

        Args:
            x_hard: Hard-scattering momenta, shape (..., n_particles, 4)
            alpha: Theory parameters, shape (..., n_parameters)
            event_type: Type of the event, e.g. LO or NLO, as a one-hot encoded tensor,
                        shape (..., n_types), optional
        Returns:
            Tensor with differential cross sections, shape (...)
        """
        pass

    def dcs_phase_space_factors(
        self, x_hard: torch.Tensor, event_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Only if the differential cross section factorizes into phase-space dependent and
        parameter dependent parts. Computes the phase-space dependent factors.

        Args:
            x_hard: Hard-scattering momenta, shape (..., n_particles, 4)
            event_type: Type of the event, e.g. LO or NLO, as a one-hot encoded tensor,
                        shape (..., n_types), optional
        Returns:
            Tensor with phase-space dependent factors, shape (..., n_factors)
        """
        raise NotImplementedError()

    def dcs_alpha_factors(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Only if the differential cross section factorizes into phase-space dependent and
        parameter dependent parts. Computes the parameter dependent factors.

        Args:
            alpha: Theory parameters, shape (..., n_parameters)
        Returns:
            Tensor with parameter dependent factors, shape (..., n_factors)
        """
        raise NotImplementedError()

    @abstractmethod
    def fiducial_cross_section(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Computes the fiducial cross section for the given theory parameters

        Args:
            alpha: Theory parameters, shape (n_points, )
        Returns:
            Tensor with fiducial cross sections, shape (n_points, )
        """
        pass

    @abstractmethod
    def hard_observables(self) -> list[Observable]:
        """
        Returns observables at the hard-scattering level for this process.

        Returns:
            List of observables
        """
        pass

    @abstractmethod
    def hard_masses(self) -> list[Optional[float]]:
        """
        Returns masses or None (if off-shell) for the hard-scattering level particles

        Returns:
            List of masses or None
        """
        pass

    @abstractmethod
    def reco_masses(self) -> list[Optional[float]]:
        """
        Returns masses or None (if off-shell) for the reco-level particles

        Returns:
            List of masses or None
        """
        pass

    @abstractmethod
    def reco_observables(self) -> list[Observable]:
        """
        Returns observables at the reconstruction level for this process.

        Returns:
            List of observables
        """
        pass
