from typing import Optional
import torch
import numpy as np
import warnings
import h5py

from ..base import Process, ProcessData, Observable
from ..observables import Observable, TTBar_Observables
from ...train.utils import *

class TTBarGenerative_v2(Process):
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

        path = self.params["training_file" if subset == "train" else "test_file"]

        f = h5py.File(path, "r")

        W_l = jet2cartesian(f["parton"][:, 1]) + jet2cartesian(f["parton"][:, 2])
        t_l = cartesian2jet(W_l + jet2cartesian(f["parton"][:, 0]))
        W_h = jet2cartesian(f["parton"][:, 4]) + jet2cartesian(f["parton"][:, 5])
        t_h = cartesian2jet(W_h + jet2cartesian(f["parton"][:, 3]))

        x_hard = np.stack([t_l,
                           f["parton"][:, 0],
                           cartesian2jet(W_l),
                           f["parton"][:, 2],
                           f["parton"][:, 1],
                           t_h,
                           f["parton"][:, 3],
                           cartesian2jet(W_h),
                           f["parton"][:, 4],
                           f["parton"][:, 5]], axis=1)

        x_reco = f["reco"][:]
        f.close()

        x_hard = torch.tensor(x_hard, dtype=torch.float32, device=self.device)

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
            reco_jets = self.params.get("reco_jets", 4)
            n_reco_particles = reco_jets + 2
            x_reco = torch.tensor(x_reco, dtype=torch.float32, device=self.device)[:, :n_reco_particles]
            print(x_reco[x_reco == -999].shape, x_reco[x_reco == -999].mean())
            x_reco[x_reco == -999] = torch.nan


        #drop_particle_type = self.params.get("drop_particle_type", True)
        #if drop_particle_type:
        #    x_reco = x_reco[:, :, 1:]
        #else:
        #    raise NotImplementedError

        if subset == "test":
            self.data["test"] = ProcessData(x_hard, x_reco)
        else:
            n_events = len(x_hard)
            train_slice = slice(int(n_events * 0), int(n_events * 0.95))
            val_slice = slice(int(n_events * 0.95), int(n_events * 1))
            self.data["train"] = ProcessData(x_hard[train_slice], x_reco[train_slice])
            self.data["val"] = ProcessData(x_hard[val_slice], x_reco[val_slice])

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
