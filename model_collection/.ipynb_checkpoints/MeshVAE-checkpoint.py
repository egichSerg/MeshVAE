import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint_sequential
from pathlib import Path

from model_collection.decoder3d import UNetDecoder
from model_collection.resnet3d import resnext101_32x8d


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x


class Reparametrize(nn.Module):
    def __init(self):
        super(Reparametrize, self).__init__()

    def forward(self, x):
        # reparametrize
        return x


class MeshVAE(nn.Module):
    def __init__(self):
        super(MeshVAE, self).__init__()

        self.encoder = resnext101_32x8d(num_classes=10, resnet_in_channels=40)
        self.encoder.load_state_dict(torch.load(f=Path('models') / 'ResNet' / '05' / 'best.pt'))
        self.encoder.fc = Linear()

        self.reparametrization = Reparametrize()
        
        self.decoder = UNetDecoder(num_classes=4, start_filts=2048)

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.reparametrization(x)
        x = self.decoder(x)
        return x


    def get_modules(self):
       return [module for k, module in self.encoder._modules.items()] + \
        [self.reparametrization] + [self.decoder._modules['workpool']] + \
        [i for i in self.decoder._modules['up_convs']] + [self.decoder._modules['conv_final']]


    def get_renders(self, indexes):
        pass