import torch
import torch.nn as nn

from .activations import CReLU
from .utils import Reshape, L2Normalize


class Critic(nn.Module):
    """
    Critic/Discriminator architecture, inspired from https://arxiv.org/abs/1803.05573

    Summary of the architecture:
        - conv kernel 5x5 stride 1
        - CReLU
        - conv kernel 5x5 stride 2
        - CReLU
        - conv kernel 5x5 stride 2
        - CReLU
        - conv kernel 5x5 stride 2
        - CReLU
        - reshape
        - L2 normalize
    """

    def __init__(
        self, output_dim: int, hidden_dim: int, kernel_size: int, input_dim: int = 1
    ) -> None:
        nn.Module.__init__(self)
        self.output_dim = output_dim  # 32768
        self.hidden_dim = hidden_dim  # 256
        self.input_dim = input_dim  # 1
        self.kernel_size = kernel_size  # 5

        if self.kernel_size == 3:
            self.padding = 1
        else:
            self.padding = 2

        self.model = nn.Sequential(
            nn.Conv2d(
                self.input_dim,
                self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=1,
            ),  # _ x hidden_dim x 32 x 32
            CReLU(),  # _ x hidden_dim x 32 x 32
            nn.Conv2d(
                2 * self.hidden_dim,
                2 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=2,
            ),
            # _ x 2 * hidden_dim x 16 x 16
            CReLU(),  # _ x 2 * hidden_dim x 16 x 16
            nn.Conv2d(
                4 * self.hidden_dim,
                4 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=2,
            ),
            # _ x 4 * hidden_dim x 8 x 8
            CReLU(),  # _ x 4 * hidden_dim x 8 x 8
            nn.Conv2d(
                8 * self.hidden_dim,
                8 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=2,
            ),
            # _ x 8 * hidden_dim x 4 x 4
            CReLU(),  # _ x 8 * hidden_dim x 4 x 4
            Reshape(-1, self.output_dim),  # _ x output_dim
            L2Normalize(),  # _ x output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
