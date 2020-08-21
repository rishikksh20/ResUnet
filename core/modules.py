import torch.nn as nn

class ResidualConv(nn.Module):

  def __init__(self, input_dim, output_dim, stride, padding):
    super(ResidualConv, self).__init__()

    self.conv_block = nn.Sequential(nn.BatchNorm2d(input_dim),
                                    nn.ReLU(),
                                    nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
                                    nn.BatchNorm2d(output_dim),
                                    nn.ReLU(),
                                    nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
                                    )
    self.conv_skip = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
                                   nn.BatchNorm2d(output_dim))

  def forward(self, x):

    return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):

  def __init__(self, input_dim, output_dim, kernel, stride):
    super(Upsample, self).__init__()

    self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride)

  def forward(self, x):
    return self.upsample(x)