import torch
import torch.nn as nn

from .sinkhorn_algorithm import sinkhorn_algorithm, pairwise_cosine_distance, sinkhorn
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
        # # Compute critic
        # x = critic(x)
        # x_prime = critic(x_prime)
        # y = critic(y)
        # y_prime = critic(y_prime)
        #
        # C_11 = pairwise_cosine_distance(x, y).double()
        # C_12 = pairwise_cosine_distance(x, y_prime).double()
        # C_21 = pairwise_cosine_distance(x_prime, y).double()
        # C_22 = pairwise_cosine_distance(x_prime, y_prime).double()
        # C_rr = pairwise_cosine_distance(x, x_prime).double()
        # C_ff = pairwise_cosine_distance(y, y_prime).double()
        #
        # MC_11, MC_12, MC_21, MC_22, MC_rr, MC_ff = new_sinkhorn_algorithm(
        #     C_11,
        #     C_12,
        #     C_21,
        #     C_22,
        #     C_rr,
        #     C_ff,
        #     eps_regularization=eps_regularization,
        #     nb_sinkhorn_iterations=nb_sinkhorn_iterations,
        #     device=device,
        # )
        #
        # W_11 = torch.sum(MC_11 * C_11)
        # W_12 = torch.sum(MC_12 * C_12)
        # W_21 = torch.sum(MC_21 * C_21)
        # W_22 = torch.sum(MC_22 * C_22)
        # W_rr = torch.sum(MC_rr * C_rr)
        # W_ff = torch.sum(MC_ff * C_ff)
        #
        # return W_11 + W_12 + W_21 + W_22 - 2 * W_rr - 2 * W_ff

        x = critic(x)
        x_prime = critic(x_prime)
        y = critic(y)
        y_prime = critic(y_prime)

        # Computing all matrices of costs
        batch_size = 200
        costs = torch.zeros((4, 4, batch_size, batch_size)).cuda()

        costs[0, 1] = pairwise_cosine_distance(x, x_prime)
        costs[0, 2] = pairwise_cosine_distance(x, y)
        costs[0, 3] = pairwise_cosine_distance(x, y_prime)
        costs[1, 2] = pairwise_cosine_distance(x_prime, y)
        costs[1, 3] = pairwise_cosine_distance(x_prime, y_prime)
        costs[2, 3] = pairwise_cosine_distance(y, y_prime)

        # Computing optimal plans for all costs

        a = (torch.ones(batch_size) / batch_size).cuda()
        b = (torch.ones(batch_size) / batch_size).cuda()

        plans = torch.zeros((4, 4, batch_size, batch_size)).cuda()
        plans[0, 1] = sinkhorn(a, b, costs[0, 1])
        plans[0, 2] = sinkhorn(a, b, costs[0, 2])
        plans[0, 3] = sinkhorn(a, b, costs[0, 3])
        plans[1, 2] = sinkhorn(a, b, costs[1, 2])
        plans[1, 3] = sinkhorn(a, b, costs[1, 3])
        plans[2, 3] = sinkhorn(a, b, costs[2, 3])

        # Computing losses

        losses = torch.zeros((4, 4)).cuda()

        losses[0, 1] = torch.sum(plans[0, 1] * costs[0, 1])
        losses[0, 2] = torch.sum(plans[0, 2] * costs[0, 2])
        losses[0, 3] = torch.sum(plans[0, 3] * costs[0, 3])
        losses[1, 2] = torch.sum(plans[1, 2] * costs[1, 2])
        losses[1, 3] = torch.sum(plans[1, 3] * costs[1, 3])
        losses[2, 3] = torch.sum(plans[2, 3] * costs[2, 3])

        loss = (
            losses[0, 2]
            + losses[0, 3]
            + losses[1, 2]
            + losses[1, 3]
            - 2 * losses[0, 1]
            - 2 * losses[2, 3]
        )

        return loss
