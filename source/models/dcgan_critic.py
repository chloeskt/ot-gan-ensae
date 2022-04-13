import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGANCritic(nn.Module):
    """
    Basic Critic/Discriminator architecture
    """

    def __init__(self, input_size, hidden_dim: int):
        super(DCGANCritic, self).__init__()


    def forward(self, x):

        return x  # Shape : (n_batch, 1)
