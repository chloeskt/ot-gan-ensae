import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaGANCritic(nn.Module):
    """
    Basic Critic/Discriminator architecture
    """

    def __init__(self, input_size, hidden_dim: int):
        super(VanillaGANCritic, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(self.layer1.out_features, hidden_dim // 2)
        self.layer3 = nn.Linear(self.layer2.out_features, hidden_dim // 4)
        self.layer4 = nn.Linear(self.layer3.out_features, 1)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.act(self.layer1(x))
        x = F.dropout(x, 0.3)
        x = self.act(self.layer2(x))
        x = F.dropout(x, 0.3)
        x = self.act(self.layer3(x))
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(self.layer4(x))

        return x  # Shape : (n_batch, 1)
