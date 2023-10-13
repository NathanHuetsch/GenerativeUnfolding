from typing import Optional
import pandas as pd
import torch
import math

from ..base import Process, ProcessData, Observable
from ..observables import Observable, momenta_to_observables


class LoThjProcess(Process):
    def __init__(self, params: dict, device: torch.device):
        self.params = params
        self.analysis_as_test = params.get("analysis_as_test", False)
        self.final_state = params["final_state"]
        self.data = {}
        self.device = device

    def load_data(self, subset: str):
        """
        Load training, validation, testing and analysis data from the specified h5 files
        if it is not loaded yet

        Args:
            subset: Which part of the data, e.g. "train", "val", "test", "analysis"
        """
        if subset in self.data:
            return

        data_type = self.params.get("data_type", "paired")
        raw_data = pd.read_hdf(self.params[
            "analysis_file" if subset == "analysis" else "training_file"
        ]).to_numpy()
        n_events = len(raw_data)
        alpha = torch.tensor(raw_data[:, :1], dtype=torch.float32, device=self.device)
        x_hard = torch.tensor(
            raw_data[:, 1:13].reshape(-1, 3, 4), dtype=torch.float32, device=self.device
        )
        if data_type == "paired":
            x_reco = torch.tensor(
                raw_data[:, 13:].reshape(n_events, -1, 4),
                dtype=torch.float32,
                device=self.device
            )
            max_reco_momenta = self.params.get("max_reco_momenta")
            if max_reco_momenta is not None:
                x_reco = x_reco[:, :max_reco_momenta]

            min_reco_momenta = self.params.get("min_reco_momenta")
            if min_reco_momenta is not None:
                mask = (x_reco[:, :, 0] != 0.).bool().sum(dim=1) >= min_reco_momenta
                x_reco = x_reco[mask]
                x_hard = x_hard[mask]
                alpha = alpha[mask]
                n_events = len(x_reco)

            if subset == "analysis":
                self.data["analysis"] = ProcessData(x_hard, x_reco, alpha)
            else:
                for subs in ["train", "test", "val"]:
                    low, high = self.params[f"{subs}_slice"]
                    data_slice = slice(int(n_events * low), int(n_events * high))
                    self.data[subs] = ProcessData(
                        x_hard[data_slice], x_reco[data_slice], alpha[data_slice],
                    )
        elif data_type == "efficiency":
            accepted = torch.tensor(raw_data[:,13], dtype=torch.float32, device=self.device)
            for subs in ["train", "test", "val"]:
                low, high = self.params[f"{subs}_slice"]
                data_slice = slice(int(n_events * low), int(n_events * high))
                self.data[subs] = ProcessData(
                    x_hard[data_slice], accepted=accepted[data_slice],
                )

    def get_data(self, subset: str) -> ProcessData:
        """
        Returns data from the specified subset of the dataset.

        Args:
            subset: Which part of the data, e.g. "train", "val", "test", "analysis"
        Returns:
            ProcessData object containing the data
        """
        if subset in ["train", "val", "test", "analysis"]:
            if self.analysis_as_test and subset == "test":
                subset = "analysis"
            self.load_data(subset)
            return self.data[subset]
        else:
            raise ValueError(f"Unknown subset '{subset}'")

    def hard_masses(self) -> list[Optional[float]]:
        """
        Returns masses or None (if off-shell) for the hard-scattering level particles

        Returns:
            List of masses or None
        """
        return [173.2, 125.0, 0.]

    def reco_masses(self) -> list[Optional[float]]:
        """
        Returns masses or None (if off-shell) for the reco-level particles

        Returns:
            List of masses or None
        """
        if self.final_state == "leptonic":
            return [0., 0., 0., None, None] + [None]*(self.params.get("max_reco_momenta", 5) - 5)
        else:
            return [0., 0., None, None, None, None] + [None]*(self.params.get("max_reco_momenta", 6) - 6)

    def hard_observables(self) -> list[Observable]:
        """
        Returns observables at the hard-scattering level for this process.

        Returns:
            List of observables
        """
        return momenta_to_observables(
            particle_names=["t", "h", "j"],
            delta_pairs=[(0, 1), (1, 2), (0, 2)],
            hard_scattering=True,
            off_shell=[False, False, False]
        )

    def reco_observables(self) -> list[Observable]:
        """
        Returns observables at the reconstruction level for this process.

        Returns:
            List of observables
        """
        reco_masses = self.reco_masses()
        if self.final_state == "leptonic":
            n_extra_jets = len(reco_masses) - 5
            particle_names = (
                [r"\gamma_1", r"\gamma_2", r"\mu", r"j_b", r"j_1"]
                + [f"j_{{{i}}}" for i in range(2, 2 + n_extra_jets)]
            )
            delta_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
        else:
            n_extra_jets = len(reco_masses) - 6
            particle_names = (
                [r"\gamma_1", r"\gamma_2", r"j_b", r"j_1", r"j_2", r"j_3"]
                + [f"j_{{{i}}}" for i in range(4, 4 + n_extra_jets)]
            )
            delta_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

        return momenta_to_observables(
            particle_names=particle_names,
            delta_pairs=delta_pairs,
            hard_scattering=False,
            off_shell=[m is None for m in reco_masses]
        )
