import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_noise(
    batch_size: int, latent_dim: int, device: str, latent_type: str = "uniform"
) -> torch.Tensor:
    # Generate fake data
    if latent_type == "uniform":
        z = 2 * torch.rand(batch_size * batch_size, latent_dim).to(device) - 1
    elif latent_type == "gaussian":
        z = torch.randn(batch_size * batch_size, latent_dim).to(device)
    else:
        raise NotImplementedError
    return z


def generate_images_with_generator(
    generator: nn.Module,
    batch_size: int,
    latent_dim: int,
    latent_type: str,
    device: str,
    img_size: int = 32,
) -> np.array:
    z = generate_noise(
        batch_size=batch_size,
        latent_dim=latent_dim,
        latent_type=latent_type,
        device=device,
    )
    output = generator(z).detach().cpu()
    output = output.view(-1, 1, img_size, img_size)
    if output.shape[1] == 1:
        output = output.squeeze(dim=1)
    output = np.clip(output.numpy(), 0, 1)

    return output
