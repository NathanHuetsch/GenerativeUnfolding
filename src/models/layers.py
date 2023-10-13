from typing import Type, Callable, Union
import math
import torch
import torch.nn as nn
import numpy as np
from .vblinear import VBLinear
from torch.autograd import grad
from torchdiffeq import odeint
import time
import os
import pickle
import torch
import math



# Calculates trace of network jacobian brute force
def autograd_trace(x_out, x_in, drop_last=False):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    if drop_last:
        for i in range(x_out.shape[1]-1):
            trJ += grad(x_out[:, i].sum(), x_in,
                        retain_graph=True)[0].contiguous()[:, i].contiguous().detach()
    else:
        for i in range(x_out.shape[1]):
            trJ += grad(x_out[:, i].sum(), x_in,
                        retain_graph=True)[0].contiguous()[:, i].contiguous().detach()
    return trJ.contiguous()

def autograd_trace_transfusion(x_out, x_in):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_out.shape[1]):
        trJ += grad(x_out[:, i].sum(), x_in, create_graph=True)[0].contiguous()[:, i].contiguous().detach()
    return trJ.sum(dim=-1)


# Calculates hutchinson estimator of trace of network jacobian
def hutch_trace2(x_out, x_in):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    noise = torch.randint_like(x_in, low=0, high=2).float() * 2 - 1.
    jvp = grad(x_out, x_in, noise)[0].detach()
    return torch.einsum('bi,bi->b', jvp, noise)


# Calculates hutchinson estimator of trace of network jacobian
def hutch_trace(x_out, x_in):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    noise = torch.randint_like(x_in, low=0, high=2).float() * 2 - 1.
    x_out_noise = torch.sum(x_out * noise)
    gradient = grad(x_out_noise, x_in)[0].detach()
    return torch.sum(gradient * noise, dim=1)


# Method to encode t for diffusion models.
# TODO: Test transformer frequency encoding
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)


# Sine activation function as described in SIREN paper. Tested once, network didnt learn anything.
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class SinCos_embedding(nn.Module):

    def __init__(self, n_frequencies: int, sigmoid=True):
        super().__init__()
        self.arg = nn.Parameter(2 * math.pi * 2**torch.arange(n_frequencies), requires_grad=False)
        self.sigmoid = sigmoid

    def forward(self, x):
        if self.sigmoid:
            x_pp = 2*nn.functional.sigmoid(x)-1
        else:
            x_pp = x
        frequencies = (x_pp.unsqueeze(-1)*self.arg).reshape(x_pp.size(0), x_pp.size(1), -1)
        return torch.cat([torch.sin(frequencies), torch.cos(frequencies)], dim=-1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Subnet(nn.Module):
    """
    Standard MLP or bayesian network to be used as a trainable subnet in INNs
    """

    def __init__(
        self,
        num_layers: int,
        size_in: int,
        size_out: int,
        internal_size: int,
        dropout: float = 0.0,
        layer_class: Type = nn.Linear,
        activation: Type = nn.SiLU,
        layer_args: dict = {},
    ):
        """
        Constructs the subnet.

        Args:
            num_layers: number of layers
            size_in: input size of the subnet
            size: output size of the subnet
            internal_size: hidden size of the subnet
            dropout: dropout chance of the subnet
            layer_class: class to construct the linear layers
            layer_args: keyword arguments to pass to the linear layer
            sin_cos_embedding: if integer>1 use sin-cos frequency embeddings
        """
        super().__init__()

        if isinstance(activation, str):
            activation = getattr(nn, activation)

        if num_layers < 1:
            raise (ValueError("Subnet size has to be 1 or greater"))
        self.layer_list = []
        for n in range(num_layers):
            input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in
            if n == num_layers - 1:
                output_dim = size_out

            self.layer_list.append(layer_class(input_dim, output_dim, **layer_args))

            if n < num_layers - 1:
                if dropout > 0:
                    self.layer_list.append(nn.Dropout(p=dropout))
                self.layer_list.append(activation())

        self.layers = nn.Sequential(*self.layer_list)

        for name, param in self.layer_list[-1].named_parameters():
            if "logsig2_w" not in name:
                param.data *= 0.02

    def forward(self, net_in):
        return self.layers(net_in)


class Resnet(nn.Module):

    def __init__(
        self,
        num_layers: int,
        embed_t_dim: int,
        embed_x_dim: int,
        embed_c_dim: int,
        size_out: int,
        internal_size: int,
        dropout: float = 0.0,
        layer_class: Type = nn.Linear,
        activation: Type = nn.SiLU,
        layer_args: dict = {},
        n_blocks: int = 1,
        bottleneck_dim: int = None,
        condition_mode: str = "concat"
    ):
        super().__init__()
        self.embed_t_dim = embed_t_dim
        self.embed_x_dim = embed_x_dim
        self.embed_c_dim = embed_c_dim
        self.condition_mode = condition_mode

        if bottleneck_dim is None:
            bottleneck_dim = embed_x_dim

        blocks = []
        for i in range(n_blocks):
            if self.condition_mode == "concat":
                size_in = embed_t_dim + embed_x_dim + embed_c_dim
                input_dim, output_dim = embed_t_dim + bottleneck_dim + embed_c_dim, bottleneck_dim
                if i == 0:
                    input_dim = size_in
            elif self.condition_mode == "add":
                assert embed_t_dim == embed_x_dim
                assert embed_c_dim == embed_x_dim
                input_dim, output_dim = embed_x_dim, embed_x_dim

            if i == n_blocks-1:
                output_dim = size_out
            blocks.append(Subnet(num_layers,
                                 input_dim,
                                 output_dim,
                                 internal_size,
                                 dropout,
                                 layer_class,
                                 activation,
                                 layer_args))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, net_in):

        t = net_in[:, :self.embed_t_dim]
        x = net_in[:, self.embed_t_dim:self.embed_t_dim + self.embed_x_dim]
        c = net_in[:, -self.embed_c_dim:]
        if self.condition_mode == "concat":
            for block in self.blocks:
                x = x + block(torch.cat([t.unsqueeze(-1), x, c], dim=1))
        elif self.condition_mode == "add":
            x = self.blocks[0](t+x+c)
            for block in self.blocks[1:]:
                x = x + block(t + x + c)
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResNetDense(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=1):
        super(ResNetDense, self).__init__()

        self.residual = nn.Linear(input_size, hidden_size)

        layers = []
        for _ in range(nlayers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                torch.nn.LeakyReLU(),
                # nn.Dropout(0.1)
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.residual(x)
        layer = self.layers(x)
        return residual + layer


class DenseNet(nn.Module):
    def __init__(self, noise_levels, x_dim=2,
                 hidden_dim=32, time_embed_dim=16,
                 nresnet=4, cond=False):
        super(DenseNet, self).__init__()

        self.cond = cond  # condition model over reco level inputs
        self.noise_levels = noise_levels

        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.x_dim = x_dim

        self.time_dense = nn.Sequential(
            torch.nn.Linear(self.time_embed_dim, self.hidden_dim),
            nn.LeakyReLU(),
            # torch.nn.Linear(self.hidden_dim, self.hidden_dim),

        )

        self.inputs_dense = nn.Sequential(
            nn.Linear(self.x_dim if not self.cond else 2 * self.x_dim, self.hidden_dim),
            nn.LeakyReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.residual = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.resnet_layers = nn.ModuleList([
            ResNetDense(self.hidden_dim, self.hidden_dim) for _ in range(nresnet)
        ])

        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * self.hidden_dim, self.x_dim)
        )

    def forward(self, inputs, t, cond=None):

        #t = self.noise_levels[steps].detach()
        assert t.dim() == 1 and t.shape[0] == inputs.shape[0]

        embed = self.time_dense(timestep_embedding(t, self.time_embed_dim))
        if self.cond:
            x = torch.cat([inputs, cond], dim=1)
        else:
            x = inputs

        inputs_dense = self.inputs_dense(x)
        residual = self.residual(inputs_dense + embed)
        x = residual
        for layer in self.resnet_layers:
            x = layer(x)

        output = self.final_layer(x)
        output = output + inputs
        return output