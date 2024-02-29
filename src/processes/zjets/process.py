from typing import Optional
import torch
import h5py
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

    
        if subset == "train" or subset == "val":
            path = self.params["training_file"]
        elif subset == "test":
            path = self.params["test_file"]
        elif subset == "analysis":
            path = self.params["analysis_file"]
        else:
            raise ValueError(f"Unknown subset {subset}")
        
        print(f"File path: {path} for subset: {subset}")
        
        if "h5" in path:
            print(f"Using {path} as h5 file.")
            data= h5py.File(path, "r")
            x_hard = torch.tensor(np.array(data["hard"]), dtype=torch.float32, device=self.device)
            x_reco = torch.tensor(np.array(data["reco"]), dtype=torch.float32, device=self.device)
        else:
            data = np.load(path, allow_pickle=True)["arr_0"].item()
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
        elif subset == "train" or subset == "val" or subset == "test":
            n_events = len(x_hard)
            low, high = self.params[f"{subset}_slice"]
            data_slice = slice(int(n_events * low), int(n_events * high))
            self.data[subset] = ProcessData(
                x_hard[data_slice], x_reco[data_slice]
                )
        else:
            raise ValueError(f"Unknown subset {subset}")


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
        test_path = self.params["test_file"]
        anaylsis_path = self.params["analysis_file"]
        x_hard = []
        x_reco = []


        for path in [training_path, anaylsis_path, test_path]:
            if "h5" in path:
                print(f"Using {path} as h5 file.")
                data= h5py.File(path, "r")
                x_hard.append(torch.tensor(np.array(data["hard"])))
                x_reco.append(torch.tensor(np.array(data["reco"])))
            else:
                data = np.load(path, allow_pickle=True)["arr_0"].item()
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

                x_hard.append(torch.tensor(gen_features))
                x_reco.append(torch.tensor(sim_features))

        # using numpy to seed a fixed permutation for reproducibility of src plot
        # torch.manual_seed() is not convenient because set_rng_state(prev_state) is not available on cuda
        rng = np.random.RandomState(0) # anything generated with this rng is reproducible

        if self.params.get("pythia_only", False):
            print("\n Using only Pythia data - training on mapping reco (50%) onto reco (50%)\n")
            x_hard = torch.cat((x_hard[0], x_hard[2]), dim = 0)
            x_reco = torch.cat((x_reco[0], x_reco[2]), dim = 0)

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

                n_events = len(x_hard)
                assert len(label) == n_events
                assert len(x_reco) == n_events

                permutation = torch.as_tensor(rng.permutation(n_events))
                x_hard = x_hard[permutation]
                x_reco = x_reco[permutation]
                label = label[permutation]

                for subs in ["train", "val", "test"]:
                    low, high = self.params[f"{subs}_slice"]
                    print(f"{subs}_slice: {low} - {high}")
                    data_slice = slice(int(n_events * low), int(n_events * high))
                    self.data[subs] = ProcessData(
                        x_hard[data_slice],
                        x_reco[data_slice],
                        label[data_slice]
                    )
        else:
            if "Pythia26" not in self.params["training_file"]:
                train_max_len = min(len(x_hard[0]), len(x_hard[1])) # I want to make sure that the training set is balanced
                x_hard_pythia_train = x_hard[0][:train_max_len]
                x_hard_herwig_train = x_hard[1][:train_max_len]
                x_reco_pythia_train = x_reco[0][:train_max_len]
                x_reco_herwig_train = x_reco[1][:train_max_len]
                x_hard_herwig_test = x_hard[1][train_max_len:] # I also want to make sure that there is no overlap between the training and testing set
                x_reco_herwig_test = x_reco[1][train_max_len:]
                x_hard_pythia_test = x_hard[2][:len(x_hard_herwig_test)] # I also want to make sure that there is no overlap between the training and testing set
                x_reco_pythia_test = x_reco[2][:len(x_hard_herwig_test)]

                print(f"Using {train_max_len} events (each) for training Pythia vs. Herwig (before slicing)")
                label_train = [torch.ones((len(x_hard_pythia_train), 1)), torch.zeros((len(x_hard_herwig_train), 1))]
                label_train = torch.cat(label_train).to(torch.float32).to(self.device)
                
                label_test = [torch.zeros((len(x_hard_herwig_test), 1)), torch.ones((len(x_hard_pythia_test), 1))]
                label_test = torch.cat(label_test).to(torch.float32).to(self.device)
                
                x_hard_train = torch.cat((x_hard_pythia_train, x_hard_herwig_train)).to(torch.float32).to(self.device)
                x_reco_train = torch.cat((x_reco_pythia_train, x_reco_herwig_train)).to(torch.float32).to(self.device)

                x_hard_test = torch.cat((x_hard_herwig_test, x_hard_pythia_test)).to(torch.float32).to(self.device)
                x_reco_test = torch.cat((x_reco_herwig_test, x_reco_pythia_test)).to(torch.float32).to(self.device)

                n_events = len(x_hard_train)
                assert len(label_train) == n_events
                assert len(x_reco_train) == n_events
                n_events_test = len(x_hard_test)
                assert len(label_test) == n_events_test
                assert len(x_reco_test) == n_events_test

                print(f"Using {len(x_hard_herwig_test)} (H) and {len(x_hard_pythia_test)} (P) for test (before slicing)")
                
                permutation_train = torch.as_tensor(rng.permutation(n_events))
                x_hard_train = x_hard_train[permutation_train]
                x_reco_train = x_reco_train[permutation_train]
                label_train = label_train[permutation_train]

                train_slice = slice(int(n_events * self.params["train_slice"][0]), int(n_events * self.params["train_slice"][1]))
                val_slice = slice(int(n_events * self.params["val_slice"][0]), int(n_events * self.params["val_slice"][1]))
                test_slice = slice(int(n_events_test * self.params["test_slice"][0]), int(n_events_test * self.params["test_slice"][1]))
                print(f"Actual number of Pythia events in training set: {len(x_hard_train[train_slice][label_train[train_slice].bool().squeeze(1)])}")
                print(f"Actual number of Herwig events in training set: {len(x_hard_train[train_slice][~label_train[train_slice].bool().squeeze(1)])}")
                print(f"Actual number of Pythia events in val set: {len(x_hard_train[val_slice][label_train[val_slice].bool().squeeze(1)])}")
                print(f"Actual number of Herwig events in val set: {len(x_hard_train[val_slice][~label_train[val_slice].bool().squeeze(1)])}")
                print(f"Actual number of Pythia events in test set: {len(x_hard_test[label_test.bool().squeeze(1)])}")
                print(f"Actual number of Herwig events in test set: {len(x_hard_test[~label_test.bool().squeeze(1)])}")
                
                self.data["train"] = ProcessData(
                            x_hard_train[train_slice],#[:, [0, 1, 2, 3, 4, 5]],
                            x_reco_train[train_slice],#[:, [0, 1, 2, 3, 4, 5]],
                            label_train[train_slice]
                        )
                self.data["val"] = ProcessData(
                            x_hard_train[val_slice],#[:, [0, 1, 2, 3, 4, 5]],
                            x_reco_train[val_slice],#[:, [0, 1, 2, 3, 4, 5]],
                            label_train[val_slice]
                        )
                
                permutation_test = torch.as_tensor(rng.permutation(n_events_test))
                x_hard_test = x_hard_test[permutation_test]
                x_reco_test = x_reco_test[permutation_test]
                label_test = label_test[permutation_test]
                self.data["test"] = ProcessData(
                                x_hard_test[test_slice],#[:, [0, 1, 2, 3, 4, 5]],
                                x_reco_test[test_slice],#[:, [0, 1, 2, 3, 4, 5]],
                                label_test[test_slice]
                            )
            else:
                x_hard_pythia = x_hard[0]
                x_reco_pythia = x_reco[0]
                x_hard_herwig = x_hard[1]
                x_reco_herwig = x_reco[1]

                label = [torch.ones((len(x_hard_pythia), 1)), torch.zeros((len(x_hard_herwig), 1))]
                label = torch.cat(label).to(torch.float32).to(self.device)
                
                x_hard = torch.cat((x_hard_pythia, x_hard_herwig)).to(torch.float32).to(self.device)
                x_reco = torch.cat((x_reco_pythia, x_reco_herwig)).to(torch.float32).to(self.device)

                n_events = len(x_hard)
                assert len(label) == n_events
                assert len(x_reco) == n_events

                permutation = torch.as_tensor(rng.permutation(n_events))

                x_hard = x_hard[permutation]
                x_reco = x_reco[permutation]
                label = label[permutation]
                for subs in ["train", "val", "test"]:
                    low, high = self.params[f"{subs}_slice"]
                    print(f"{subs}_slice: {low} - {high}")
                    data_slice = slice(int(n_events * low), int(n_events * high))
                    self.data[subs] = ProcessData(
                        x_hard[data_slice],
                        x_reco[data_slice],
                        label[data_slice]
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