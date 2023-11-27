from typing import Type, Callable, Union, Optional
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from .layers import *
from .cfm import CFM


class DirectDiffusion(CFM):
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
        super(CFM, self).__init__()
        self.params = params
        self.give_x1 = self.params.get("give_x1", False)
        if self.give_x1:
            print(f"        Using give_x1")
        self.dims_in = params["dims_in"]
        self.dims_c = 0
        self.bayesian = params.get("bayesian", False)
        self.bayesian_samples = params.get("bayesian_samples", 20)
        self.bayesian_layers = []
        self.bayesian_factor = params.get("bayesian_factor", 1)

        t_noise_scale = params.get("t_noise_scale", 0)
        minimum_noise_scale = params.get("minimum_noise_scale", 0)
        self.trajectory = LinearTrajectory(t_noise_scale=t_noise_scale,
                                           minimum_noise_scale=minimum_noise_scale)

        self.build_embeddings()
        self.build_net()
        self.build_distributions()
        self.build_solver()

        self.l2_regularization = self.params.get("l2_regularization", False)
        if self.l2_regularization:
            self.l2_factor = self.params.get("l2_factor", 1.e-4)

        self.loss_fct = nn.MSELoss()

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

        with torch.no_grad():
            # Solve the ODE from t=1 to t=0 from the sampled initial condition
            if self.give_x1:
                x_1_embedded = self.x_embedding(x_1)

            # Wrap the network such that the ODE solver can call it
            if self.solver_type == "ODE":
                def net_wrapper(t, x_t):
                    x_t = self.x_embedding(x_t)
                    t = self.t_embedding(t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device))
                    if self.give_x1:
                        v = self.net(x_1_embedded, x_t)
                    else:
                        v = self.net(t, x_t)
                    return v
            else:
                net_wrapper = SDE_wrapper(self, condition=None)

            x_t = self.solver(net_wrapper, x_1, reverse=True)

        # return the generated samples
        return x_t[-1]

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

        # Calculate x_t along the trajectory and the derivative x_t_dot
        x_t, x_t_dot = self.trajectory(x_0, x_1, t)
        # Predict the velocity
        t = self.t_embedding(t)
        x_t = self.x_embedding(x_t)
        if self.give_x1:
            x_1 = self.x_embedding(x_1)
            v_pred = self.net(x_1, x_t)
        else:
            v_pred = self.net(t, x_t)
        # Calculate the loss
        cfm_loss = self.loss_fct(v_pred, x_t_dot)

        if self.bayesian:
            kl_loss = self.bayesian_factor * kl_scale * self.kl() / self.dims_in
            loss = cfm_loss + kl_loss
            loss_terms = {
                "loss": loss.item(),
                "mse": cfm_loss.item(),
                "kl": kl_loss.item(),
            }
        elif self.l2_regularization:
            regularization_loss = self.l2_factor * torch.norm(v_pred)
            loss = cfm_loss + regularization_loss
            loss_terms = {
                "loss": loss.item(),
                "mse": cfm_loss.item(),
                "l2": regularization_loss.item()
            }
        else:
            loss = cfm_loss
            loss_terms = {
                "loss": loss.item(),
            }
        return loss, loss_terms
