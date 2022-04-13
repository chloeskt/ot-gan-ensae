import torch
import torch.nn as nn


class DCGANGenerator(nn.Module):
    """
    Basic Critic/Discriminator architecture
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_shape):
        super(DCGANGenerator, self).__init__()


    def forward(self, x):

        return x  # Shape : (n_batch, 1, 32, 32)
