import torch

from source import Critic


def pairwise_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = x / torch.pow(torch.sum(torch.square(x), dim=1), 1 / 2).unsqueeze(1)
    y_norm = y / torch.pow(torch.sum(torch.square(y), dim=1), 1 / 2).unsqueeze(1)
    return 1 - x_norm @ y_norm.T


def sinkhorn_algorithm(
    x: torch.Tensor,
    y: torch.Tensor,
    critic: Critic,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    device: str,
):
    """Computes a distance between X and Y"""
    # feed to critic: maps images to learned latent space
    x = critic(x)
    y = critic(y)

    # compute transport cost matrix C
    c = pairwise_cosine_distance(x, y)

    # compute kernel K
    k = torch.exp(-c / eps_regularization).to(device)

    # parameters and variables
    n = x.shape[0]
    a = None
    b = torch.ones(n).to(device)
    ones = torch.ones(n).to(device) / n

    # sinkhorn
    for iteration in range(nb_sinkhorn_iterations):
        a = ones / torch.matmul(k, b)
        b = ones / torch.matmul(k, a)

    return torch.dot(torch.matmul(k * c, b), a)
