import torch
from core.res_unet import ResUnet, ResidualConv, Upsample

def test_resunet():
    img = torch.ones(1, 3, 224, 224)
    resunet = ResUnet(3)
    assert resunet(img).shape == torch.Size([1, 1, 224, 224]) 
    
    
def test_residual_conv():
    x = torch.ones(1, 64, 224, 224)
    res_conv = ResidualConv(64, 128, 2, 1) 
    assert res_conv(x).shape == torch.Size([1, 128, 112, 112]) 
    

def test_upsample():
    x = torch.ones(1, 512, 28, 28)
    upsample = Upsample(512, 512, 2, 2)
    assert upsample(x).shape == torch.Size([1, 512, 56, 56])
    