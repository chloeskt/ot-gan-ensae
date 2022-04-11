import torch
import torch.nn as nn
import torch.nn.functional as F


class CReLU(nn.Module):
    """
    Concatenated Rectified Linear Unit (CReLU).
    Taken from https://github.com/pytorch/pytorch/issues/1327
    """

    def __init__(self):
        nn.Module.__init__(self)

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((F.relu(x), F.relu(-x)), 1)
