import torch
import torch.nn as nn
from .utils import normal_init

class DCGANGenerator(nn.Module):
    """
    Basic Critic/Discriminator architecture
    """
    def __init__(self, nc=1, latent_dim=100, hidden_dim=64):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        return self.model(x)  # Shape : (n_batch, 1, 32, 32)
