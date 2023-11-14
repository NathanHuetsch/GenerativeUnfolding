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

    def sample(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and log probabilities for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
            log_prob: log probabilites, shape (n_events, )
        """
        logits = self.model(c)
        unnormed_probs = logits.exp()
        log_probs = logits - unnormed_probs.sum(dim=1, keepdims=True).log()
        classes = torch.multinomial(unnormed_probs, num_samples=1, replacement=True)[:,0]
        x = nn.functional.one_hot(classes, num_classes=self.dims_in)
        log_prob = log_probs[:, classes]
        return x, log_prob

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
        loss = -self.log_prob(x, c).mean()
        return loss, {"loss": loss.item()}
