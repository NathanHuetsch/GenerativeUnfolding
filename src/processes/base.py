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
