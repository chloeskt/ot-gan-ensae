import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import normal_init

class DCGANCritic(nn.Module):
    """
    Basic DC GAN Critic/Discriminator architecture
    """

    def __init__(self,hidden_dim: int, nc=1):
        super(DCGANCritic, self).__init__()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(hidden_dim * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)  # Shape : (n_batch, 1)
