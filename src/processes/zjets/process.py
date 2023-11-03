from typing import Optional
import torch
import numpy as np
import warnings

from ..base import Process, ProcessData, Observable
from ..observables import Observable, momenta_to_observables


class ZjetsProcess(Process):
    def __init__(self, params: dict, device: torch.device):
        self.params = params
        self.analysis_as_test = params.get("analysis_as_test", False)
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

        path = self.params["analysis_file" if subset == "analysis" else "training_file"]
        data = np.load(path, allow_pickle=True)["arr_0"].item()
        mask = ((data["sim_widths"] != 0) * (data["gen_widths"] != 0) * (data["sim_sdms"] > 0) * (data["gen_sdms"] > 0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            jet_masses_hard = data["gen_jets"][:, -1]
            jet_multiplicities_hard = data["gen_mults"]
            jet_widths_hard = data["gen_widths"]
            jet_Nsubjettinessratio_hard = data["gen_tau2s"] / data["gen_widths"]
            jet_lnp_hard = 2*np.log(data["gen_sdms"] / data["gen_jets"][:, 0])
            jet_zgs_hard = data["gen_zgs"]

            jet_masses_reco = data["sim_jets"][:, -1]
            jet_multiplicities_reco = data["sim_mults"]
            jet_widths_reco = data["sim_widths"]
            jet_Nsubjettinessratio_reco = data["sim_tau2s"] / data["sim_widths"]
            jet_lnp_reco = 2*np.log(data["sim_sdms"] / data["sim_jets"][:, 0])
            jet_zgs_reco = data["sim_zgs"]

        x_hard = np.stack([jet_masses_hard,
                          jet_multiplicities_hard,
                          jet_widths_hard,
                          jet_Nsubjettinessratio_hard,
                          jet_lnp_hard,
                          jet_zgs_hard], axis=1)[mask]
        x_hard = torch.tensor(x_hard, dtype=torch.float32, device=self.device)

        x_reco = np.stack([jet_masses_reco,
                          jet_multiplicities_reco,
                          jet_widths_reco,
                          jet_Nsubjettinessratio_reco,
                          jet_lnp_reco,
                          jet_zgs_reco], axis=1)[mask]
        x_reco = torch.tensor(x_reco, dtype=torch.float32, device=self.device)

        if subset == "analysis":
            self.data["analysis"] = ProcessData(x_hard, x_reco)
        else:
            n_events = len(x_hard)
            for subs in ["train", "test", "val"]:
                low, high = self.params[f"{subs}_slice"]
                data_slice = slice(int(n_events * low), int(n_events * high))
                self.data[subs] = ProcessData(
                    x_hard[data_slice], x_reco[data_slice]
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
        pass

    def reco_masses(self) -> list[Optional[float]]:
        """
        Returns masses or None (if off-shell) for the reco-level particles

        Returns:
            List of masses or None
        """
        pass

    def hard_observables(self) -> list[Observable]:
        """
        Returns observables at the hard-scattering level for this process.

        Returns:
            List of observables
        """
        return momenta_to_observables()

    def reco_observables(self) -> list[Observable]:
        """
        Returns observables at the reconstruction level for this process.

        Returns:
            List of observables
        """
        return momenta_to_observables()
