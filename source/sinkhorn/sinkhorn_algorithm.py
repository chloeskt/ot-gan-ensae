import logging
from typing import Optional

import torch

from ..models import Critic


def pairwise_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # normalization has already been done at the end of the Critic network
    return 1 - x @ y.T


def sinkhorn_algorithm(
    x: torch.Tensor,
    y: torch.Tensor,
    critic: Critic,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    device: str,
):
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


def new_sinkhorn_algorithm(
    *c: torch.Tensor,
    eps_regularization: float,
    nb_sinkhorn_iterations: int,
    device: str,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    double: bool = True,
    delta: float = 0.05
):
    c = torch.stack(c)

    a = (
        torch.ones(c.shape[0], c.shape[-1], 1, device=device) / c.shape[-1]
        if a is None
        else a
    )
    b = (
        torch.ones(c.shape[0], c.shape[-1], 1, device=device) / c.shape[-1]
        if b is None
        else b
    )

    K = torch.exp(-c / eps_regularization).to(device)
    v = torch.ones((b.shape[0], b.shape[1], 1)).to(device)

    if double:
        a = a.double()
        b = b.double()
        K = K.double()
        v = v.double()

    previous_u = None
    for _ in range(nb_sinkhorn_iterations):
        u = a / (K @ v)
        v = b / (K.permute(0, 2, 1) @ u)

        if (previous_u is not None) and (delta > 0):
            distance = torch.mean(((u - previous_u) ** 2), 1)
            rel_distance = distance / torch.mean((u**2), 1)
            if rel_distance.max() < delta:
                break

        previous_u = u

    return (
        torch.diag_embed(u.reshape(-1, u.shape[1]))
        @ K
        @ torch.diag_embed(v.reshape(-1, v.shape[1]))
    )


def sinkhorn(a, b, C, reg=1, max_iters=100):
    u = torch.ones_like(a).cuda()
    v = torch.ones_like(b).cuda()

    with torch.no_grad():
        K = torch.exp(-C / reg)
        for i in range(max_iters):
            u = a / (torch.matmul(K, v) + 1e-8)
            v = b / (torch.matmul(K.T, u) + 1e-8)

    M = torch.matmul(torch.diag(u), torch.matmul(K, torch.diag(v)))

    return M
