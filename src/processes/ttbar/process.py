from typing import Optional
import torch
import numpy as np
import warnings
import h5py

from ..base import Process, ProcessData, Observable
from ..observables import Observable, TTBar_Observables


class TTBarGenerative(Process):
    def __init__(self, params: dict, device: torch.device):
        self.params = params
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

        f = h5py.File(path, "r")
        x_hard = f["parton"][:]
        x_reco = f["reco"][:]
        f.close()

        #max_reco_jets = self.params.get("max_reco_jets", 4)
        #x_reco = x_reco[:, :max_reco_jets+2]
        # hard gives t, b, wl_c, wl_l, wl_nu, tbar, bbar, wh_c, wh_q1, wh_q2
        x_hard = torch.tensor(x_hard, dtype=torch.float32, device=self.device)[:, [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]]

        enforce_2_bjets = self.params.get("enforce_2_bjets", False)
        if enforce_2_bjets:
            nu_l = x_reco[:, :2]
            jets = x_reco[:, 2:]

            # get b_jet mask
            b_mask = jets[:, :, 0] == 1
            event_mask = b_mask.sum(1) >= 2
            # set all non-b jets to zero
            b_masked_jets = jets*np.expand_dims(b_mask, -1)
            # sort jets by pT
            indices = np.flip(np.argsort((b_masked_jets[:, :, 2])), -1)
            sorted_b_jets = b_masked_jets[np.arange(b_masked_jets.shape[0])[:, None], indices, :]
            # extract leading 2 b-jets
            leading_b_jets = sorted_b_jets[:, :2]

            not_b_masked_jets = jets*np.expand_dims(~b_mask, -1)
            indices = np.flip(np.argsort((not_b_masked_jets[:, :, 2])), -1)
            sorted_not_b_jets = not_b_masked_jets[np.arange(not_b_masked_jets.shape[0])[:, None], indices, :]
            # extract leading 2 b-jets
            leading_not_b_jets = sorted_not_b_jets[:, :2]
            x_reco = torch.tensor(np.concatenate([nu_l, leading_b_jets, leading_not_b_jets], axis=1), dtype=torch.float32, device=self.device)

            x_reco = x_reco[event_mask]
            x_hard = x_hard[event_mask]

            non_b_event_mask = x_reco[:, -1, 2] == 0
            x_reco = x_reco[~non_b_event_mask]
            x_hard = x_hard[~non_b_event_mask]

        else:
            x_reco = torch.tensor(x_reco, dtype=torch.float32, device=self.device)[:, :6]

        drop_particle_type = self.params.get("drop_particle_type", True)
        if drop_particle_type:
            x_reco = x_reco[:, :, 1:]
        else:
            raise NotImplementedError



        #x_hard = x_hard[:, [1,2,3,4,5,6,7,9,10,11,12,13,14,15]]
        #x_reco = x_reco[:, [1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]

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
            self.load_data(subset)
            return self.data[subset]
        else:
            raise ValueError(f"Unknown subset '{subset}'")

    def hard_observables(self) -> list[Observable]:
        """
        Returns observables at the hard-scattering level for this process.

        Returns:
            List of observables
        """
        return TTBar_Observables()

    def reco_observables(self) -> list[Observable]:
        """
        Returns observables at the reconstruction level for this process.

        Returns:
            List of observables
        """
        return TTBar_Observables()

    def hard_masses(self) -> list[Optional[float]]:
        """
        Returns masses or None (if off-shell) for the hard-scattering level particles

        Returns:
            List of masses or None
        """
        return [4.7, None, 4.7, None]

    def reco_masses(self) -> list[Optional[float]]:
        """
        Returns masses or None (if off-shell) for the reco-level particles

        Returns:
            List of masses or None
        """
        return [0., 1., None, None, None, None] + [None]*(self.params.get("max_reco_jets", 4) - 4)
