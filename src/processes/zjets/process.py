from typing import Optional
import torch
import numpy as np
import warnings

from ..base import Process, ProcessData, Observable
from ..observables import Observable, momenta_to_observables


class ZJetsGenerative(Process):
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

        if self.params.get("high_statistics", True):
            print("Loading high statistics file")
            if subset == "analysis":
                path = self.params["analysis_file"]
                data = np.load(path, allow_pickle=True)["arr_0"].item()
            else:
                path = "data/OmniFold_Big.pkl"
                with open(path, 'rb') as pickle_file:
                    data = np.load(pickle_file, allow_pickle=True)
        else:
            path = self.params["analysis_file" if subset == "analysis" else "training_file"]
            data = np.load(path, allow_pickle=True)["arr_0"].item()

        if self.params.get("loader", "theirs") == "ours":
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
                               jet_widths_hard,
                               jet_multiplicities_hard,
                               jet_lnp_hard,
                               jet_zgs_hard,
                               jet_Nsubjettinessratio_hard], axis=1)[mask]
            x_hard = torch.tensor(x_hard, dtype=torch.float32, device=self.device)

            x_reco = np.stack([jet_masses_reco,
                               jet_widths_reco,
                               jet_multiplicities_reco,
                               jet_lnp_reco,
                               jet_zgs_reco,
                               jet_Nsubjettinessratio_reco], axis=1)[mask]
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

        else:
            feature_names = ['widths', 'mults', 'sdms', 'zgs', 'tau2s']
            gen_features = [data['gen_jets'][:, 3]]
            sim_features = [data['sim_jets'][:, 3]]

            for feature in feature_names:
                gen_features.append(data['gen_' + feature])
                sim_features.append(data['sim_' + feature])

            gen_features = np.stack(gen_features, -1)
            sim_features = np.stack(sim_features, -1)
            # ln rho
            gen_features[:, 3] = 2 * np.ma.log(
                np.ma.divide(gen_features[:, 3], data['gen_jets'][:, 0]).filled(0)).filled(0)
            sim_features[:, 3] = 2 * np.ma.log(
                np.ma.divide(sim_features[:, 3], data['sim_jets'][:, 0]).filled(0)).filled(0)
            # tau2
            gen_features[:, 5] = gen_features[:, 5] / (10 ** -50 + gen_features[:, 1])
            sim_features[:, 5] = sim_features[:, 5] / (10 ** -50 + sim_features[:, 1])

            x_hard = torch.tensor(gen_features, dtype=torch.float32, device=self.device)
            x_reco = torch.tensor(sim_features, dtype=torch.float32, device=self.device)

            if subset == "analysis":
                self.data["analysis"] = ProcessData(x_hard, x_reco)
            else:
                n_events = len(x_hard)
                for subs in ["train", "test", "val"]:
                    low, high = self.params[f"{subs}_slice"]
                    data_slice = slice(int(n_events * low), int(n_events * high))
                    #slices = {"train": slice(0, 950000),
                    #          "val": slice(950000, 1000000),
                    #          "test": slice(1000000, 1600000)}
                    #data_slice = slices[subs]
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
        return momenta_to_observables()

    def reco_observables(self) -> list[Observable]:
        """
        Returns observables at the reconstruction level for this process.

        Returns:
            List of observables
        """
        return momenta_to_observables()


class ZJetsOmnifold(ZJetsGenerative):

    def __init__(self, params: dict, device: torch.device):
        super(ZJetsOmnifold, self).__init__(params, device)

    def load_data(self, subset: str):
        """
        Load training, validation, testing and analysis data from the specified h5 files
        if it is not loaded yet

        Args:
            subset: Which part of the data, e.g. "train", "val", "test", "analysis"
        """
        if subset == "analysis":
            raise ValueError("Analysis file not implemented for Omnifold")

        if subset in self.data:
            return

        training_path = self.params["training_file"]
        anaylsis_path = self.params["analysis_file"]
        x_hard = []
        x_reco = []
        for path in [training_path, anaylsis_path]:
            data = np.load(path, allow_pickle=True)["arr_0"].item()
            if self.params.get("loader", "theirs") == "ours":
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

                x_hard_subset = np.stack([jet_masses_hard,
                                   jet_widths_hard,
                                   jet_multiplicities_hard,
                                   jet_lnp_hard,
                                   jet_zgs_hard,
                                   jet_Nsubjettinessratio_hard], axis=1)[mask]

                x_reco_subset = np.stack([jet_masses_reco,
                                   jet_widths_reco,
                                   jet_multiplicities_reco,
                                   jet_lnp_reco,
                                   jet_zgs_reco,
                                   jet_Nsubjettinessratio_reco], axis=1)[mask]
            else:
                feature_names = ['widths', 'mults', 'sdms', 'zgs', 'tau2s']
                gen_features = [data['gen_jets'][:, 3]]
                sim_features = [data['sim_jets'][:, 3]]

                for feature in feature_names:
                    gen_features.append(data['gen_' + feature])
                    sim_features.append(data['sim_' + feature])

                gen_features = np.stack(gen_features, -1)
                sim_features = np.stack(sim_features, -1)
                # ln rho
                gen_features[:, 3] = 2 * np.ma.log(
                    np.ma.divide(gen_features[:, 3], data['gen_jets'][:, 0]).filled(0)).filled(0)
                sim_features[:, 3] = 2 * np.ma.log(
                    np.ma.divide(sim_features[:, 3], data['sim_jets'][:, 0]).filled(0)).filled(0)
                # tau2
                gen_features[:, 5] = gen_features[:, 5] / (10 ** -50 + gen_features[:, 1])
                sim_features[:, 5] = sim_features[:, 5] / (10 ** -50 + sim_features[:, 1])

                x_hard_subset = torch.tensor(gen_features, dtype=torch.float32, device=self.device)
                x_reco_subset = torch.tensor(sim_features, dtype=torch.float32, device=self.device)

            x_hard.append(torch.tensor(x_hard_subset))
            x_reco.append(torch.tensor(x_reco_subset))
        # using numpy to seed a fixed permutation for reproducibility of src plot
        # torch.manual_seed() is not convenient because set_rng_state(prev_state) is not available on cuda
        rng = np.random.RandomState(0) # anything generated with this rng is reproducible

        if self.params.get("pythia_only", False):
            print("\n Using only Pythia data - training on mapping reco (50%) onto reco (50%)\n")
            x_hard = x_hard[0] # keep only pythia
            x_reco = x_reco[0] # keep only pythia
            label = [torch.ones((int(0.5*len(x_hard)), 1)), torch.zeros((len(x_hard) - int(0.5*len(x_hard)), 1))]
            label = torch.cat(label).to(torch.float32).to(self.device)
            x_hard = x_hard.to(torch.float32).to(self.device)
            x_reco = x_reco.to(torch.float32).to(self.device)


            if self.params.get("add_noise", False):
                scale = self.params.get("noise_std_factor", 1e-3)
                print(f"Adding noise to Pythia 1 with factor {scale:.1e} * std")
                
                x_reco_std = torch.std(x_reco[label.bool().squeeze(1)], dim=0)
                x_hard_std = torch.std(x_hard[label.bool().squeeze(1)], dim=0)
                
                debug = True
                if debug:
                    print("RECO means and stds before noise:", torch.mean(x_reco[label.bool().squeeze(1)], dim=0), x_reco_std)
                    print("HARD means and stds before noise:", torch.mean(x_hard[label.bool().squeeze(1)], dim=0), x_hard_std)
                    pass
                
                noise_reco = torch.randn_like(x_reco) * torch.std(x_reco, dim=0) * scale
                x_reco[label.bool().squeeze(1)] += noise_reco[label.bool().squeeze(1)]
                noise_hard = torch.randn_like(x_hard) * torch.std(x_hard, dim=0) * scale
                x_hard[label.bool().squeeze(1)] += noise_hard[label.bool().squeeze(1)]

                if debug:
                    print("\nRECO means and stds after noise:", torch.mean(x_reco[label.bool().squeeze(1)], dim=0), torch.std(x_reco[label.bool().squeeze(1)], dim=0))
                    print("HARD means and stds after noise:", torch.mean(x_hard[label.bool().squeeze(1)], dim=0), torch.std(x_hard[label.bool().squeeze(1)], dim=0))
                    pass


        else:    
            label = [torch.ones((len(x_hard[0]), 1)), torch.zeros((len(x_hard[1]), 1))]
            label = torch.cat(label).to(torch.float32).to(self.device)
            x_hard = torch.cat(x_hard).to(torch.float32).to(self.device)
            x_reco = torch.cat(x_reco).to(torch.float32).to(self.device)

        n_events = len(x_hard)
        assert len(label) == n_events
        assert len(x_reco) == n_events

        # using numpy to seed a fixed permutation for reproducibility of src plot
        # torch.manual_seed() is not convenient because set_rng_state(prev_state) is not available on cuda
        permutation = torch.as_tensor(rng.permutation(n_events))
        x_hard = x_hard[permutation]
        x_reco = x_reco[permutation]
        label = label[permutation]

        for subs in ["train", "test", "val"]:
            low, high = self.params[f"{subs}_slice"]
            print(f"{subs}_slice: {low} - {high}")
            data_slice = slice(int(n_events * low), int(n_events * high))
            self.data[subs] = ProcessData(
                x_hard=x_hard[data_slice],
                x_reco=x_reco[data_slice],
                label=label[data_slice]
            )

    def get_data(self, subset: str) -> ProcessData:
        """
        Returns data from the specified subset of the dataset.

        Args:
            subset: Which part of the data, e.g. "train", "val", "test", "analysis"
        Returns:
            ProcessData object containing the data
        """
        if subset in ["train", "val", "test"]:
            self.load_data(subset)
            return self.data[subset]
        else:
            raise ValueError(f"Unknown subset '{subset}'")