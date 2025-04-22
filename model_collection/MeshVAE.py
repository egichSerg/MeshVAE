import torch
import torch.nn as nn

import numpy as np

from torch.utils.checkpoint import checkpoint_sequential
from pathlib import Path

from model_collection.decoder3d import UNetDecoder
from model_collection.resnet3d import resnext101_32x8d
from model_collection.harmonic_embedding import HarmonicEmbedding


from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
FoVPerspectiveCameras, VolumeRenderer,
NDCGridRaysampler, EmissionAbsorptionRaymarcher
)

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x


class Reparametrize(nn.Module):
    def __init__(self, channels=64, features=2048):
        super(Reparametrize, self).__init__()

        self.bn = nn.BatchNorm3d(num_features=features)
        self.fc_mu = nn.Linear(in_features=channels, out_features=channels, bias=False)
        self.fc_logvar = nn.Linear(in_features=channels, out_features=channels, bias=False)

    def forward(self, x):
        # get params
        x = self.bn(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # reparametrize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu, logvar, mu + eps * std)


class VAELoss(nn.Module):
    def __init__(self, loss_fn):
        super(VAELoss, self).__init__()
        self.criterion = loss_fn

    # question: how is the loss function using the mu and variance?
    def forward(self, mu, log_var, render_imgs, render_sils, target_imgs, target_sils):
        """gives the batch normalized Variational Error."""

        # print('VAE Loss:', 'mu', mu)
        # print('VAE Loss:', 'log_var', log_var)
        
        batch_size = target_imgs.shape[0]
        
        sil_err = self.criterion(render_sils, target_sils).abs().mean()
        color_err = self.criterion(render_imgs, target_imgs).abs().mean()
        
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KL_divergence = torch.sum(KLD_element).mul_(-0.5)
        

        loss = color_err + sil_err + KL_divergence
        
        
        return (loss / batch_size).mean()


class MeshVAE(nn.Module):
    def __init__(self):
        super(MeshVAE, self).__init__()

        self.encoder = resnext101_32x8d(num_classes=10, resnet_in_channels=40)
        self.encoder.load_state_dict(torch.load(f=Path('models') / 'ResNet' / '05' / 'best.pt'))
        self.encoder.harmonic_embedding = Linear()
        self.encoder.fc = Linear()

        self.reparametrize = Reparametrize(channels=8)

        n_harmonic_functions = 60
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)
        
        self.decoder = UNetDecoder(num_classes=2, start_filts=2048) # densities and black-white color

        self.render_size = 144
        self.volume_extent_world = 3.0

    
    def forward(self, x):
        x = self.encoder(x)
        mu, logvar, x = self.reparametrize(x)
        x = self.harmonic_embedding(x)
        x = self.decoder(x)
        return (mu, logvar, x)


    def get_modules(self):
       return [module for k, module in self.encoder._modules.items()] + \
        [self.reparametrize] + [self.harmonic_embedding] + [self.decoder._modules['workpool']] + \
        [i for i in self.decoder._modules['up_convs']] + [self.decoder._modules['conv_final']]


    def get_encoder_modules(self):
       return [module for k, module in self.encoder._modules.items()] + [self.reparametrize]

    
    def get_decoder_modules(self):
       return [self.harmonic_embedding] + [self.decoder._modules['workpool']] + \
        [i for i in self.decoder._modules['up_convs']] + [self.decoder._modules['conv_final']]


    def create_renderer(self):
        raysampler = NDCGridRaysampler(
            image_width = self.render_size,
            image_height = self.render_size,
            n_pts_per_ray = 150,
            min_depth = 0.1,
            max_depth = self.volume_extent_world,
        )

        raymarcher = EmissionAbsorptionRaymarcher()
        return VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)
    

    def get_renders(self, x, cameras):
        renders = []
        for i in range(x.shape[0]):
            batch_size = cameras.R.shape[0]
            renderer = self.create_renderer()
            
            triplane = x[i]
            # TODO: check shapes
            densities = triplane[0].unsqueeze(0)
            colors = triplane[1:]

            voxel_size = self.volume_extent_world / self.render_size
            volumes = Volumes(
                densities = densities.unsqueeze(0).expand(
                    batch_size, *densities.shape
                ),
                features = colors.unsqueeze(0).expand(
                    batch_size, *colors.shape
                ),
                voxel_size = voxel_size
            )
            # rendered_images, rendered_silhouttes = renderer(cameras=cameras, volumes=volumes)[0].split([1, 1], dim = -1)
            render = renderer(cameras=cameras, volumes=volumes)[0]
            renders.append(render)
            
            del renderer
            torch.cuda.empty_cache()

        return torch.stack(renders)


class MeshAE(nn.Module):
    def __init__(self):
        super(MeshAE, self).__init__()

        self.encoder = resnext101_32x8d(num_classes=10, resnet_in_channels=40)
        self.encoder.fc = Linear()

        self.reparametrize = Linear()

        n_harmonic_functions = 60
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)
        
        self.decoder = UNetDecoder(num_classes=2, start_filts=2048) # densities and black-white color

        self.render_size = 144
        self.volume_extent_world = 3.0

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.reparametrize(x)
        x = self.harmonic_embedding(x)
        x = self.decoder(x)
        return x


    def get_modules(self):
       return [module for k, module in self.encoder._modules.items()] + \
        [self.reparametrize] + [self.harmonic_embedding] + [self.decoder._modules['workpool']] + \
        [i for i in self.decoder._modules['up_convs']] + [self.decoder._modules['conv_final']]


    def create_renderer(self):
        raysampler = NDCGridRaysampler(
            image_width = self.render_size,
            image_height = self.render_size,
            n_pts_per_ray = 150,
            min_depth = 0.1,
            max_depth = self.volume_extent_world,
        )

        raymarcher = EmissionAbsorptionRaymarcher()
        return VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)
    

    def get_renders(self, x, cameras):
        renders = []
        for i in range(x.shape[0]):
            batch_size = cameras.R.shape[0]
            renderer = self.create_renderer()
            
            triplane = x[i]
            # TODO: check shapes
            densities = triplane[0].unsqueeze(0)
            colors = triplane[1:]

            voxel_size = self.volume_extent_world / self.render_size
            volumes = Volumes(
                densities = densities.unsqueeze(0).expand(
                    batch_size, *densities.shape
                ),
                features = colors.unsqueeze(0).expand(
                    batch_size, *colors.shape
                ),
                voxel_size = voxel_size
            )
            # rendered_images, rendered_silhouttes = renderer(cameras=cameras, volumes=volumes)[0].split([1, 1], dim = -1)
            render = renderer(cameras=cameras, volumes=volumes)[0]
            renders.append(render)
            
            del renderer
            torch.cuda.empty_cache()

        return torch.stack(renders)