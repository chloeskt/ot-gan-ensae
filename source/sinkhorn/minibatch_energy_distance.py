import torch
import torch.nn as nn

from .sinkhorn_algorithm import (
    sinkhorn_algorithm,
    pairwise_cosine_distance,
    new_sinkhorn_algorithm,
)
from ..models import Critic


class MinibatchEnergyDistance(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    @staticmethod
    def forward(
        x: torch.Tensor,
        x_prime: torch.Tensor,
        y: torch.Tensor,
        y_prime: torch.Tensor,
        critic: Critic,
        eps_regularization: float,
        nb_sinkhorn_iterations: int,
        device: str,
    ) -> torch.Tensor:
        term1 = sinkhorn_algorithm(
            x, y, critic, eps_regularization, nb_sinkhorn_iterations, device
        )
        term2 = sinkhorn_algorithm(
            x, y_prime, critic, eps_regularization, nb_sinkhorn_iterations, device
        )
        term3 = sinkhorn_algorithm(
            x_prime, y, critic, eps_regularization, nb_sinkhorn_iterations, device
        )
        term4 = sinkhorn_algorithm(
            x_prime, y_prime, critic, eps_regularization, nb_sinkhorn_iterations, device
        )
        term5 = sinkhorn_algorithm(
            x, x_prime, critic, eps_regularization, nb_sinkhorn_iterations, device
        )
        term6 = sinkhorn_algorithm(
            y, y_prime, critic, eps_regularization, nb_sinkhorn_iterations, device
        )

        return term1 + term2 + term3 + term4 - 2 * term5 - 2 * term6


class NewMinibatchEnergyDistance(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    @staticmethod
    def forward(
        x: torch.Tensor,
        x_prime: torch.Tensor,
        y: torch.Tensor,
        y_prime: torch.Tensor,
        critic: Critic,
        eps_regularization: float,
        nb_sinkhorn_iterations: int,
        device: str,
    ):
        # Compute critic
        x = critic(x)
        x_prime = critic(x_prime)
        y = critic(y)
        y_prime = critic(y_prime)

        C_11 = pairwise_cosine_distance(x, y).double()
        C_12 = pairwise_cosine_distance(x, y_prime).double()
        C_21 = pairwise_cosine_distance(x_prime, y).double()
        C_22 = pairwise_cosine_distance(x_prime, y_prime).double()
        C_rr = pairwise_cosine_distance(x, x_prime).double()
        C_ff = pairwise_cosine_distance(y, y_prime).double()

        MC_11, MC_12, MC_21, MC_22, MC_rr, MC_ff = new_sinkhorn_algorithm(
            C_11,
            C_12,
            C_21,
            C_22,
            C_rr,
            C_ff,
            eps_regularization=eps_regularization,
            nb_sinkhorn_iterations=nb_sinkhorn_iterations,
            device=device,
        )

        W_11 = torch.sum(MC_11 * C_11)
        W_12 = torch.sum(MC_12 * C_12)
        W_21 = torch.sum(MC_21 * C_21)
        W_22 = torch.sum(MC_22 * C_22)
        W_rr = torch.sum(MC_rr * C_rr)
        W_ff = torch.sum(MC_ff * C_ff)

        return W_11 + W_12 + W_21 + W_22 - 2 * W_rr - 2 * W_ff
