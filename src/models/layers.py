import warnings
from typing import Type
import torch.nn as nn
import numpy as np
from torch.autograd import grad
import torch
import math
from torchdiffeq import odeint



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


class sinusoidal_t_embedding(nn.Module):

    def __init__(self, embed_dim, max_period=10000):
        super(sinusoidal_t_embedding, self).__init__()
        half = embed_dim // 2
        self.freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )[None]

    def forward(self, t):
        args = t.float() * self.freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


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
            params,
            conditional=True
    ):
        super().__init__()


        network_params = params.get("network_params")
        embed_x_dim = network_params.get("embed_x_dim", params["dims_in"])
        embed_c_dim = network_params.get("embed_c_dim", params["dims_c"])
        embed_t_dim = network_params.get("embed_t_dim", 1)

        self.conditional = conditional

        num_layers = network_params.get("n_layers", 3)
        internal_size = network_params.get("internal_size", 3)
        dropout = network_params.get("dropout", 0.0)
        activation = network_params.get("activation", nn.SiLU)
        if isinstance(activation, str):
            activation = getattr(nn, activation)

        bayesian = params.get("bayesian")
        if bayesian:
            layer_class = VBLinear
            layer_args = {"prior_prec": params.get("prior_prec", 1.0),
                          "std_init": params.get("std_init", -9)}
        else:
            layer_class = nn.Linear
            layer_args = {}

        layer_list = []
        for n in range(num_layers):
            input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = embed_t_dim+embed_x_dim+embed_c_dim
            if n == num_layers - 1:
                output_dim = params["dims_in"]

            layer_list.append(layer_class(input_dim, output_dim, **layer_args))

            if n < num_layers - 1:
                if dropout > 0:
                    layer_list.append(nn.Dropout(p=dropout))
                layer_list.append(activation())

        self.layers = nn.Sequential(*layer_list)

        for name, param in layer_list[-1].named_parameters():
            if "logsig2_w" not in name:
                param.data *= 0.02


    def forward(self, t, x, c=None):
        if self.conditional:
            return self.layers(torch.cat([t, x, c], dim=-1))
        else:
            return self.layers(torch.cat([t, x], dim=-1))


class EmbeddingNet(nn.Module):
    """
    Standard MLP or bayesian network to be used as a trainable subnet in INNs
    """

    def __init__(
            self,
            params,
            size_in,
            size_out
    ):
        super().__init__()

        num_layers = params.get("n_layers", 3)
        internal_size = params.get("internal_size", 3)
        dropout = params.get("dropout", 0.0)
        activation = params.get("activation", nn.SiLU)

        if isinstance(activation, str):
            activation = getattr(nn, activation)

        layer_list = []
        for n in range(num_layers):
            input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in
            if n == num_layers - 1:
                output_dim = size_out

            layer_list.append(nn.Linear(input_dim, output_dim))

            if n < num_layers - 1:
                if dropout > 0:
                    layer_list.append(nn.Dropout(p=dropout))
                layer_list.append(activation())

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


class FFFnet(nn.Module):
    """
    Standard MLP or bayesian network to be used as a trainable subnet in INNs
    """

    def __init__(
            self,
            params,
            size_in,
            size_out
    ):
        super().__init__()

        num_layers = params.get("n_layers", 3)
        internal_size = params.get("internal_size", 3)
        dropout = params.get("dropout", 0.0)
        activation = params.get("activation", nn.SiLU)

        if isinstance(activation, str):
            activation = getattr(nn, activation)

        layer_list = []
        for n in range(num_layers):
            input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in
            if n == num_layers - 1:
                output_dim = size_out

            layer_list.append(nn.Linear(input_dim, output_dim))

            if n < num_layers - 1:
                if dropout > 0:
                    layer_list.append(nn.Dropout(p=dropout))
                layer_list.append(activation())

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x, c):
        return self.layers(torch.cat([x, c], dim=-1))
        

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






class ResNetDense(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=1):
        super(ResNetDense, self).__init__()

        self.residual = nn.Linear(input_size, hidden_size)

        layers = []
        for _ in range(nlayers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                torch.nn.LeakyReLU()
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.residual(x)
        layer = self.layers(x)
        return residual + layer


class DenseNet(nn.Module):

    def __init__(
            self,
            params,
            conditional=True
    ):
        super().__init__()

        network_params = params.get("network_params")
        embed_x_dim = network_params.get("embed_x_dim", params["dims_in"])
        embed_c_dim = network_params.get("embed_c_dim", params["dims_c"])
        embed_t_dim = network_params.get("embed_t_dim", 1)

        internal_size = network_params.get("internal_size", 32)
        n_blocks = network_params.get("n_blocks", 4)

        self.conditional = conditional  # condition model over reco level inputs

        self.time_dense = nn.Sequential(
            torch.nn.Linear(self.time_embed_dim, self.hidden_dim),
            nn.LeakyReLU(),
            # torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.inputs_dense = nn.Sequential(
            nn.Linear(embed_x_dim+embed_c_dim if self.conditional else embed_x_dim, internal_size),
            nn.LeakyReLU(),
            # nn.Linear(self.internal_size, self.internal_size),
        )

        self.residual = nn.Sequential(
            nn.Linear(internal_size, internal_size),
            nn.LeakyReLU(),
            nn.Linear(internal_size, internal_size),
        )

        self.resnet_layers = nn.ModuleList([
            ResNetDense(internal_size, internal_size) for _ in range(n_blocks)
        ])

        self.final_layer = nn.Sequential(
            nn.Linear(internal_size, 2 * internal_size),
            nn.LeakyReLU(),
            nn.Linear(2 * internal_size, params["dims_in"])
        )

    def forward(self, t, x, c=None):
        assert t.shape[-1] == x.shape[-1], "Fix embed dimensions"

        if self.conditional:
            x = torch.cat([x, c], dim=1)

        inputs_dense = self.inputs_dense(x)
        x = self.residual(inputs_dense + t)
        for layer in self.resnet_layers:
            x = layer(x)

        output = self.final_layer(x)
        output = output + x
        return output


class VBLinear(nn.Module):
    """
    Bayesian linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_prec: float = 1.0,
        enable_map: bool = False,
        std_init: float = -9,
    ):
        """
        Constructs the Bayesian linear layer
        Args:
            in_features: Number of input dimensions
            out_features: Number of input dimensions
            prior_prec: Standard deviation of the Gaussian prior
            enable_map: If True, does not sample from posterior during evaluation
                        (maximum-a-posteriori)
            std_init: Logarithm of the initial standard deviation of the weights
        """
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = enable_map
        self.prior_prec = prior_prec
        self.random = None
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.std_init = std_init
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of the layer.
        """
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(self.std_init, 0.001)
        self.bias.data.zero_()

    def reset_random(self):
        """
        Reset the random weights. New weights will be sampled the next time, forward is
        called in evaluation mode.
        """
        self.random = None

    def sample_random_state(self) -> np.ndarray:
        """
        Sample a random state and return it as a numpy array

        Returns:
            Tensor with sampled weights
        """
        return torch.randn_like(self.logsig2_w).detach().cpu().numpy()

    def import_random_state(self, state: np.ndarray):
        """
        Replace the random state

        Args:
            state: Numpy array with random numbers
        """
        self.random = torch.tensor(
            state, device=self.logsig2_w.device, dtype=self.logsig2_w.dtype
        )

    def kl(self) -> torch.Tensor:
        """
        KL divergence between posterior and prior.

        Returns:
            KL divergence
        """
        logsig2_w = self.logsig2_w.clamp(-11, 11)
        kl = (
            0.5
            * (
                self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                - logsig2_w
                - 1
                - np.log(self.prior_prec)
            ).sum()
        )
        return kl

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bayesian linear layer. In training mode, use the local
        reparameterization trick.
        Args:
            input: Input tensor
        Returns:
            Output tensor
        """
        if self.training:
            # local reparameterization trick is more efficient and leads to
            # an estimate of the gradient with smaller variance.
            # https://arxiv.org/pdf/1506.02557.pdf
            mu_out = nn.functional.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            s2_w = logsig2_w.exp()
            var_out = nn.functional.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:
            if self.map:
                return nn.functional.linear(input, self.mu_w, self.bias)

            logsig2_w = self.logsig2_w.clamp(-11, 11)
            if self.random is None:
                self.random = torch.randn_like(self.logsig2_w)
            s2_w = logsig2_w.exp()
            weight = self.mu_w + s2_w.sqrt() * self.random
            return nn.functional.linear(input, weight, self.bias) + 1e-8

    def __repr__(self) -> str:
        """
        Returns a string representation of the layer.
        Returns:
            String representation of the layer
        """
        return f"{self.__class__.__name__} ({self.n_in}) -> ({self.n_out})"


class ODEsolver(nn.Module):

    def __init__(self, params):
        # Parameters for the ODE solver. Default settings should work fine
        # Details on the solver under https://github.com/rtqichen/torchdiffeq

        super(ODEsolver, self).__init__()
        self.method = params.get("method", "dopri5")
        self.rtol = params.get("rtol", 1.e-3)
        self.atol = params.get("atol", 1.e-6)
        self.step_size = params.get("step_size", 1.e-2)
        t_min = float(params.get("t_min", 0.))
        t_max = float(params.get("t_max", 1.))
        self.time_intervall_forward = torch.tensor([t_min, t_max])
        self.time_intervall_backward = torch.tensor([t_max, t_min])
        
        if self.method == "dopri5":
            print(f"ODE solver: {self.method}, atol {self.atol}, rtol {self.rtol}, t {[t_min, t_max]}", flush=True)
        else:
            print(f"ODE solver: {self.method}, step_size {self.step_size}, t {[t_min, t_max]}", flush=True)

    def forward(self, function, x_initial, reverse=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                x_t = odeint(
                    func=function,
                    y0=x_initial,
                    t=self.time_intervall_backward if reverse else self.time_intervall_forward,
                    rtol=self.rtol,
                    atol=self.atol,
                    method=self.method,
                    options=dict(step_size=self.step_size)
                )
            except AssertionError:
                warnings.warn(f"Integration with {self.method} failed, trying with RK4")
                x_t = odeint(
                    func=function,
                    y0=x_initial,
                    t=self.time_intervall_backward if reverse else self.time_intervall_forward,
                    method="rk4",
                    options=dict(step_size=self.step_size)
                )
            return x_t


class MixtureDistribution(torch.distributions.distribution.Distribution):

    def __init__(self, normal_channels, uniform_channels):
        super().__init__(validate_args=False)
        self.dim = len(normal_channels) + len(uniform_channels)
        self.uniform_channels = uniform_channels
        self.normal_channels = normal_channels

        self.normal = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(len(self.normal_channels)), torch.eye(len(self.normal_channels))
        )
        self.uniform = torch.distributions.uniform.Uniform(
                torch.zeros(len(self.uniform_channels)), torch.ones(len(self.uniform_channels)))

    def sample(self, n):
        samples = torch.zeros((n[0], self.dim))
        samples[:, self.uniform_channels] = self.uniform.sample(n)
        samples[:, self.normal_channels] = self.normal.sample(n)
        return samples
