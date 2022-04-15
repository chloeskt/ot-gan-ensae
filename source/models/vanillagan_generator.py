import torch
import torch.nn as nn


class VanillaGANGenerator(nn.Module):
    """
    Basic Critic/Discriminator architecture
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_shape):
        super(VanillaGANGenerator, self).__init__()

        self.layer1 = nn.Linear(latent_dim, hidden_dim)
        self.layer2 = nn.Linear(self.layer1.out_features, hidden_dim * 2)
        self.layer3 = nn.Linear(self.layer2.out_features, hidden_dim * 4)
        self.layer4 = nn.Linear(
            self.layer3.out_features,
            output_shape[0] * output_shape[1] * output_shape[2],
        )

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.output_shape = output_shape

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = x.view((-1,) + self.output_shape)

        return x  # Shape : (n_batch, 1, 32, 32)
