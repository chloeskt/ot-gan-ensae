import torch
import torch.nn as nn

from .sinkhorn_algorithm import sinkhorn_algorithm
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
