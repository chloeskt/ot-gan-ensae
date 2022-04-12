from typing import Optional

import torch
import torch.nn as nn

from .utils import Reshape


class Generator(nn.Module):
    """
    Generator architecture, inspired from https://arxiv.org/abs/1803.05573

    Summary of the architecture:
        - linear
        - GLU
        - reshape
        - 2 times 2x2 nearest-neighbor upsampling operation
        - conv: 5x5 kernel, stride 1
        - GLU
        - 2 times 2x2 nearest-neighbor upsampling operation
        - conv: 5x5 kernel, stride 1
        - GLU
        - 2 times 2x2 nearest-neighbor upsampling operation
        - conv: 5x5 kernel, stride 1
        - GLU
        - conv 5x5 stride 1
        - Tanh
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int = 1) -> None:
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim  # 1024
        self.latent_dim = latent_dim  # 100
        self.output_dim = output_dim  # 1
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32 * self.hidden_dim),  # _ x (32 * hidden_dim)
            nn.GLU(dim=1),  # _ x (32 * hidden_dim // 2)
            Reshape(-1, hidden_dim, 4, 4),  # _ x hidden_dim x 4 x 4
            nn.Upsample(scale_factor=2, mode="nearest"),  # _ x hidden_dim x 8 x 8
            nn.Conv2d(
                self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2, stride=1
            ),  # _ x hidden_dim x 8 x 8
            nn.GLU(dim=1),  # _ x hidden_dim // 2 x 8 x 8
            nn.Upsample(
                scale_factor=2, mode="nearest"
            ),  # _ x hidden_dim // 2 x 16 x 16
            nn.Conv2d(
                hidden_dim // 2, hidden_dim // 2, kernel_size=5, padding=2, stride=1
            ),  # _ x hidden_dim // 2 x 16 x 16
            nn.GLU(dim=1),  # _ x hidden_dim // 4 x 16 x 16
            nn.Upsample(
                scale_factor=2, mode="nearest"
            ),  # _ x hidden_dim // 4 x 32 x 32
            nn.Conv2d(
                hidden_dim // 4, hidden_dim // 4, kernel_size=5, padding=2, stride=1
            ),  # _ x hidden_dim // 4 x 32 x 32
            nn.GLU(dim=1),  # _ x hidden_dim // 8 x 32 x 32
            nn.Conv2d(
                hidden_dim // 8, self.output_dim, kernel_size=5, padding=2, stride=1
            ),  # _ x output_dim x 32 x 32
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

