import torch
import torch.nn as nn
from .utils import normal_init

class DCGANGenerator(nn.Module):
    """
    Basic Critic/Discriminator architecture
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_shape):
        super(DCGANGenerator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(latent_dim, hidden_dim, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(hidden_dim)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(hidden_dim//2)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(hidden_dim//4)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(hidden_dim//8)
        self.deconv5 = nn.ConvTranspose2d(hidden_dim//8, 1, 4, 2, 1)

        self.activ = nn.ReLU()

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = x.view(-1, self.z_dim, 1, 1)
        x = self.activ(self.deconv1_bn(self.deconv1(x)))
        x = self.activ(self.deconv2_bn(self.deconv2(x)))
        x = self.activ(self.deconv3_bn(self.deconv3(x)))
        x = self.activ(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x # Shape : (n_batch, 1, 64, 64)
