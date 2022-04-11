import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *args):
        nn.Module.__init__(self)
        self.args = args

    def forward(self, x):
        return x.view(*self.args)


class L2Normalize(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    @staticmethod
    def forward(x):
        return F.normalize(x, dim=1, p=2)
