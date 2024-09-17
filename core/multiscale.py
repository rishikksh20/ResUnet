import torch
import torch.nn as nn
from utils.utils import weights_init
from .discriminator import Discriminator



class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc1 = Discriminator()
        self.disc2 = Discriminator()
        self.disc3 = Discriminator()

        self.apply(weights_init)

    def forward(self, x, start):
        results = []
        results.append(self.disc1(x[:, : , 0:20, start: start + 40]))
        results.append(self.disc2(x[:, :, 20:40, start: start + 40]))
        results.append(self.disc3(x[:, :, 40:80, start: start + 40]))
        return results