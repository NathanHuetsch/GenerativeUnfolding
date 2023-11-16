import torch
import torch.nn as nn
import numpy as np

from .inn import Subnet
from .layers import VBLinear


class Classifier(nn.Module):
    def __init__(self, params: dict):
        """
        Initializes and builds a conditional classifier for the multiplicity

        Args:
            dims_in: dimension of input
            dims_c: dimension of condition
            params: dictionary with architecture/hyperparameters
        """
        super().__init__()
        self.params = params
        self.dims_in = 1
        self.dims_c = params["dims_c"]
        self.bayesian = params.get("bayesian", False)
        self.bayesian_layers = []
        self.build_classifier()
        if self.bayesian:
            print(f"        Bayesian set to True, Bayesian layers: ", len(self.bayesian_layers))

    def build_classifier(self):
        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in self.params:
            layer_args["prior_prec"] = self.params["prior_prec"]
        if "std_init" in self.params:
            layer_args["std_init"] = self.params["std_init"]
        self.model = Subnet(
            self.params["layers_per_block"],
            self.dims_c,
            self.dims_in,
            internal_size = self.params["internal_size"],
            dropout = self.params["dropout"],
            layer_class=layer_class,
            layer_args=layer_args,
        )
        if self.dims_in > 1:
            self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        else:
            self.cross_entropy = nn.BCEWithLogitsLoss(reduction="none")
        if self.bayesian is not None:
            self.bayesian_layers.extend(
                layer for layer in self.model.layer_list if isinstance(layer, VBLinear)
            )

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            log probabilities, shape (n_events, )
        """
        return -self.cross_entropy(self.model(c), x)

    def probs(self, c: torch.Tensor) -> torch.Tensor:
        """
        Get all class probabilities

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            probabilities, shape (n_events, )
        """
        if self.dims_in > 1:
            return nn.functional.softmax(self.model(c), dim=1)
        else:
            return torch.sigmoid(self.model(c))

    def batch_loss(
        self, x: torch.Tensor, c: torch.Tensor, kl_scale: float = 0.0
    ) -> tuple[torch.Tensor, dict]:
        """
        Evaluate the log probability

        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
            kl_scale: factor in front of KL loss term, default 0
        Returns:
            loss: batch loss
            loss_terms: dictionary with loss contributions
        """
        classifier_loss = -self.log_prob(x, c).mean()
        if self.bayesian:
            kl_loss = kl_scale * self.kl() / self.dims_in
            loss = classifier_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "bce": classifier_loss.item(),
                "kl": kl_loss.item(),
            }
        else:
            loss = classifier_loss
            loss_terms = {
                "loss": loss.item(),
            }
        return loss, loss_terms

    def kl(self) -> torch.Tensor:
        """
        Compute the KL divergence between weight prior and posterior

        Returns:
            Scalar tensor with KL divergence
        """
        assert self.bayesian
        return sum(layer.kl() for layer in self.bayesian_layers)

    def reset_random_state(self):
        """
        Resets the random state of the Bayesian layers
        """
        assert self.bayesian
        for layer in self.bayesian_layers:
            layer.reset_random()

    def sample_random_state(self) -> list[np.ndarray]:
        """
        Sample new random states for the Bayesian layers and return them as a list

        Returns:
            List of numpy arrays with random states
        """
        assert self.bayesian
        return [layer.sample_random_state() for layer in self.bayesian_layers]

    def import_random_state(self, states: list[np.ndarray]):
        """
        Import a list of random states into the Bayesian layers

        Args:
            states: List of numpy arrays with random states
        """
        assert self.bayesian
        for layer, s in zip(self.bayesian_layers, states):
            layer.import_random_state(s)

    def generate_random_state(self):
        """
        Generate and save a set of random states for repeated use
        """
        assert self.bayesian
        self.random_states = [self.sample_random_state() for i in range(self.bayesian_samples)]
