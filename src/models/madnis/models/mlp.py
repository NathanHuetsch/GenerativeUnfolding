""" MLP subnetwork """

from typing import Union, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Creates a dense subnetwork
    which can be used within the invertible modules.
    """

    def __init__(
        self,
        meta: Dict,
        features_in: int,
        features_out: int,
        pass_inputs: bool = False
    ):
        """
        Args:
          meta:
            Dictionary with defining parameters
            to construct the network.
          features_in:
            Number of input features.
          features_out:
            Number of output features.
          pass_inputs:
            If True, a tuple is expected as input to forward and only the first tensor
            is used as input to the network
        """
        super().__init__()

        # which activation
        if isinstance(meta["activation"], str):
            try:
                activation = {
                    "relu": nn.ReLU,
                    "elu": nn.ELU,
                    "leakyrelu": nn.LeakyReLU,
                    "tanh": nn.Tanh
                }[meta["activation"]]
            except KeyError:
                raise ValueError(f'Unknown activation "{meta["activation"]}"')
        else:
            activation = meta["activation"]

        layer_constructor = meta.get("layer_constructor", nn.Linear)

        # Define the layers
        input_dim = features_in
        layers = []
        for i in range(meta["layers"] - 1):
            layers.append(layer_constructor(
                input_dim,
                meta["units"]
            ))
            layers.append(activation())
            input_dim = meta["units"]
        layers.append(layer_constructor(
            input_dim,
            features_out
        ))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.layers = nn.Sequential(*layers)
        self.pass_inputs = pass_inputs

    def forward(self, x):
        if self.pass_inputs:
            x, *rest = x
            return self.layers(x), *rest
        else:
            return self.layers(x)


class StackedMLP(nn.Module):
    """
    Creates a dense subnetwork
    which can be used within the invertible modules.
    """

    def __init__(
        self,
        meta: Dict,
        features_in: list[int],
        features_out: int,
        n_channels: int,
    ):
        """
        Args:
          meta:
            Dictionary with defining parameters
            to construct the network.
          features_in:
            Number of input features.
          features_out:
            Number of output features.
          n_channels:
            Number of channels
        """
        super().__init__()

        # which activation
        if isinstance(meta["activation"], str):
            try:
                self.activation = {
                    "relu": F.relu,
                    "elu": F.elu,
                    "leakyrelu": F.leaky_relu,
                    "tanh": F.tanh
                }[meta["activation"]]
            except KeyError:
                raise ValueError(f'Unknown activation "{meta["activation"]}"')
        else:
            self.activation = meta["activation"]()

        input_dim = features_in
        layer_dims = []
        for i in range(meta["layers"] - 1):
            layer_dims.append((input_dim, meta["units"]))
            input_dim = meta["units"]
        layer_dims.append((input_dim, features_out))

        self.weights = nn.ParameterList([
            torch.empty((n_channels, n_out, n_in)) for n_in, n_out in layer_dims
        ])
        self.biases = nn.ParameterList([
            torch.empty((n_channels, n_out)) for n_in, n_out in layer_dims
        ])
        self.n_channels = n_channels
        self.n_layers = len(layer_dims)
        self.reset_parameters()

    def reset_parameters(self):
        for ws, bs in zip(self.weights[:-1], self.biases[:-1]):
            for w, b in zip(ws, bs):
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(b, -bound, bound)
        nn.init.zeros_(self.weights[-1])
        nn.init.zeros_(self.biases[-1])

    def forward(self, x: torch.Tensor, section_sizes: list[int]):
        ys = []
        for i, xs in enumerate(x.split(section_sizes, dim=0)):
            for w, b in zip(self.weights[:-1], self.biases[:-1]):
                xs = self.activation(F.linear(xs, w[i], b[i]))
            ys.append(F.linear(xs, self.weights[-1][i], self.biases[-1][i]))
        return torch.cat(ys, dim=0)
