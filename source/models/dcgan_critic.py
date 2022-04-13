import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import normal_init

class DCGANCritic(nn.Module):
    """
    Basic Critic/Discriminator architecture
    """

    def __init__(self, hidden_dim: int):
        super(DCGANCritic, self).__init__()

        self.conv1 = nn.Conv2d(1, hidden_dim, 4, 2, 1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(hidden_dim*2)
        self.conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(hidden_dim*4)
        self.conv4 = nn.Conv2d(hidden_dim*4, hidden_dim*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(hidden_dim*8)
        self.conv5 = nn.Conv2d(hidden_dim*8, 1, 4, 1, 0)

        self.activ = nn.LeakyReLU(negative_slope=0.2)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.activ(self.conv1(x))
        x = self.activ(self.conv2_bn(self.conv2(x)))
        x = self.activ(self.conv3_bn(self.conv3(x)))
        x = self.activ(self.conv4_bn(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))
        x = x.view((-1, 1))

        return x # Shape : (n_batch, 1)
