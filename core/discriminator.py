import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential( nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                            nn.LeakyReLU()
                                          )
        self.out = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        x = self.discriminator(x)
        return self.out(x)