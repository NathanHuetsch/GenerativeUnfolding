from typing import Type, Callable, Union, Optional
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from .layers import *
from .cfm import CFM


class DirectDiffusion(nn.Module):
    """
    Class implementing a conditional CFM model
    """

    def __init__(self, params: dict):
        """
        Initializes and builds the conditional CFM

        Args:
            dims_in: dimension of input
            dims_c: dimension of condition
            params: dictionary with architecture/hyperparameters
        """
        super().__init__()
        self.params = params
        self.dims_in = params["dims_in"]
        #self.dims_c = params["dims_c"]
        self.dims_c = 0
        self.bayesian = params.get("bayesian", False)
        self.bayesian_samples = params.get("bayesian_samples", 20)
        self.bayesian_layers = []
        self.bayesian_factor = params.get("bayesian_factor", 1)

        self.ODEsolver = ODEsolver(params=params.get("ODE_params", {}))

        self.build_net()
        self.build_distributions()

        self.l2_regularization = self.params.get("l2_regularization", False)
        if self.l2_regularization:
            self.l2_factor = self.params.get("l2_factor", 1.e-4)

        self.loss_fct = nn.MSELoss()

        self.add_noise = self.params.get("add_noise", False)
        if self.add_noise:
            self.add_noise_scale = self.params.get("add_noise_scale", 1.e-4)
            print(f"        add noise: True with scale {self.add_noise_scale}")

    def build_net(self):
        """
        Construct the Network
        """
        self.build_embeddings()
        # Build the main network
        network = self.params["network_params"].get("network_class", Subnet)
        self.net = eval(network)(params=self.params, conditional=self.dims_c > 0)
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"        Network: {network} with {n_params} parameters")

        if self.bayesian:
            if isinstance(self.net, Subnet):
                self.bayesian_layers.extend(
                    layer for layer in self.net.layer_list if isinstance(layer, VBLinear)
                )
            else:
                #for block in self.net.blocks:
                #    self.bayesian_layers.extend(
                #        layer for layer in block.layer_list if isinstance(layer, VBLinear)
                #    )
                raise ValueError("Bayesian only implemented for Subnet atm")
            print(f"        Bayesian set to True, Bayesian layers: ", len(self.bayesian_layers))

    def build_embeddings(self):

        params = self.params["network_params"]
        embed_x = params.get("embed_x", False)
        if embed_x:
            self.embed_x_dim = params.get("embed_x_dim", self.dims_in)
            self.x_embedding = EmbeddingNet(
                params=params.get("embed_x_params"),
                size_in=self.dims_in,
                size_out=self.embed_x_dim
            )
            n_params = sum(p.numel() for p in self.x_embedding.parameters() if p.requires_grad)
            print(f"        x_embedding: Subnet with {n_params} parameters to {self.embed_x_dim} dimensions")
        else:
            self.embed_x_dim = self.dims_in
            params["embed_x_dim"] = self.dims_in
            self.x_embedding = nn.Identity()
            print(f"        x_embedding: None")

        embed_t = params.get("embed_t", False)
        if embed_t:
            self.embed_t_dim = params.get("embed_t_dim", self.dims_in)
            embed_t_mode = params.get("embed_t_mode", "linear")
            if embed_t_mode == "gfprojection":
                self.t_embedding = nn.Sequential(
                    GaussianFourierProjection(embed_dim=self.embed_t_dim),
                    nn.Linear(self.embed_t_dim, self.embed_t_dim))
            elif embed_t_mode == "sinusoidal":
                self.t_embedding = nn.Sequential(
                    sinusoidal_t_embedding(embed_dim=self.embed_t_dim),
                    nn.Linear(self.embed_t_dim, self.embed_t_dim))
            else:
                self.t_embedding = nn.Linear(1, self.embed_t_dim)
            n_params = sum(p.numel() for p in self.t_embedding.parameters() if p.requires_grad)
            print(f"        t_embedding: Mode {embed_t_mode} with {n_params} parameters to {self.embed_t_dim} dimensions")
        else:
            self.embed_t_dim = 1
            params["embed_t_dim"] = 1
            self.t_embedding = nn.Identity()
            print(f"        t_embedding: None")

    def build_distributions(self):

        self.beta_dist = self.params.get("beta_dist", False)
        if self.beta_dist:
            self.beta_dist_a = self.params.get("beta_dist_a", 1)
            self.beta_dist_b = self.params.get("beta_dist_b", 0.7)
            self.t_dist = torch.distributions.beta.Beta(concentration1=float(self.beta_dist_a),
                                                        concentration0=float(self.beta_dist_b))
            print(f"        t distribution: beta distribution with params {self.beta_dist_a, self.beta_dist_b}")
        else:
            self.t_dist = torch.distributions.uniform.Uniform(low=0, high=1)
            self.mod_t = self.params.get("mod_t", False)
            print(f"        t distribution: uniform")

    def sample(self, x_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples for the given condition

        Args:
            c: condition tensor, shape (n_events, dims_c)
        Returns:
            x: generated samples, shape (n_events, dims_in)
        """
        dtype = x_1.dtype
        device = x_1.device

        # Wrap the network such that the ODE solver can call it
        def net_wrapper(t, x_t):
            x_t = self.x_embedding(x_t)
            t = self.t_embedding(t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device))
            v = self.net(t, x_t)
            return v

        if self.add_noise:
            x_1 = x_1 + self.add_noise_scale*torch.randn_like(x_1, device=device, dtype=dtype)

        with torch.no_grad():
            # Solve the ODE from t=1 to t=0 from the sampled initial condition
            x_t = self.ODEsolver(net_wrapper, x_1, reverse=True)
        # return the generated sample. This function does not calculate jacobians and just returns a 0 instead
        return x_t[-1], torch.Tensor([0])

    def batch_loss(
        self, x_0: torch.Tensor, x_1: torch.Tensor, kl_scale: float = 0.0
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

        # Sample a time step
        if self.mod_t:
            t = torch.rand((1), dtype=x_0.dtype, device=x_0.device)
            t = (t + torch.arange(x_0.size(0)).unsqueeze(-1) / x_0.size(0)) % 1
        else:
            t = torch.rand((x_0.size(0), 1), dtype=x_0.dtype, device=x_0.device)

        # Calculate x_t along the trajectory
        x_t = (1 - t) * x_0 + t * x_1
        if self.add_noise:
            x_t = x_t + torch.randn_like(x_t, device=x_0.device, dtype=x_0.dtype) * self.add_noise_scale
        # Calculate the derivative of x_t, i.e. the conditional velocity
        x_t_dot = -x_0 + x_1
        # Predict the velocity
        t = self.t_embedding(t)
        x_t = self.x_embedding(x_t)
        v_pred = self.net(t, x_t)
        # Calculate the loss
        cfm_loss = self.loss_fct(v_pred, x_t_dot)

        if self.bayesian:
            kl_loss = self.bayesian_factor * kl_scale * self.kl() / self.dims_in
            loss = cfm_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "likeli_loss": cfm_loss.item(),
                "kl_loss": kl_loss.item(),
            }
        elif self.l2_regularization:
            regularization_loss = self.l2_factor * torch.norm(v_pred)
            loss = cfm_loss + regularization_loss
            loss_terms = {
                "loss": loss.item(),
                "cfm_loss": cfm_loss.item(),
                "regularization_loss": regularization_loss.item()
            }
        else:
            loss = cfm_loss
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
        self.random_states = [self.sample_random_state() for i in range(self.bayesian_samples)]
