import torch.nn as nn
import torch
from core.modules import ResidualConv, ASPP, AttentionBlock, Upsample_, Squeeze_Excite_Block

class ResUnetPlusPlus(nn.Module):

    def __init__(self, channel):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(nn.Conv2d(channel, 32, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                         )
        self.input_skip = nn.Sequential(nn.Conv2d(channel, 32, kernel_size=3, padding=1))

        self.squeeze_excite1 = Squeeze_Excite_Block(32)

        self.residual_conv1 = ResidualConv(32, 64, 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(64)

        self.residual_conv2 = ResidualConv(64, 128, 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(128)

        self.residual_conv3 = ResidualConv(128, 256, 2, 1)

        self.aspp_bridge = ASPP(256, 512)

        self.attn1 = AttentionBlock(128, 512, 512)
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(512 + 128, 256, 1, 1)

        self.attn2 = AttentionBlock(64, 256, 256)
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(256 + 64, 128, 1, 1)

        self.attn3 = AttentionBlock(32, 128, 128)
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(128 + 32, 64, 1, 1)

        self.aspp_out = ASPP(64, 32)

        self.output_layer = nn.Sequential(nn.Conv2d(32, 1, 1),
                                          nn.Sigmoid())

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out