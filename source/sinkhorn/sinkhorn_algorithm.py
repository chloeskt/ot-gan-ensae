import logging

import torch

from ..models import OTGANCritic


def pairwise_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # normalization has already been done at the end of the OTGANCritic network
    return 1 - x @ y.T


def sinkhorn_algorithm(
    x: torch.Tensor,
    y: torch.Tensor,
    critic: OTGANCritic,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    device: str,
) -> float:
    """Sinkhorn algorithm: https://arxiv.org/abs/1306.0895"""

    logger = logging.getLogger(__name__)

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
    with torch.no_grad():
        for iteration in range(nb_sinkhorn_iterations):
            a = ones / (torch.matmul(k, b) + 1e-8)
            b = ones / (torch.matmul(k.T, a) + 1e-8)

    result = torch.dot(torch.matmul(k * c, b), a)
    if torch.isnan(result):
        print()
        logger.debug("ISSUE WITH SINKHORN ALGORITHM")
        logger.debug("Kernel K", k)
        logger.debug("a", a)
        logger.debug("b", b)
        logger.debug("c", c)

    return result
