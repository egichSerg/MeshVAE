import torch
import torch.nn as nn

from pathlib import Path

from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
FoVPerspectiveCameras, VolumeRenderer,
NDCGridRaysampler, EmissionAbsorptionRaymarcher
)

from model_collection.VAE_parts.encoder import Encoder
from model_collection.VAE_parts.reparametrize import Reparametrize
from model_collection.VAE_parts.decoder import Decoder


class MeshVAE(nn.Module):
    def __init__(self, hidden_channels=64, latent_channels=512):
        super(MeshVAE, self).__init__()

        self.encoder = Encoder(channels=40, hidden_channels=hidden_channels)
        self.reparametrize = Reparametrize(in_channels=hidden_channels*16, latent_channels=latent_channels)
        self.decoder = Decoder(channels=2, hidden_channels=hidden_channels, latent_channels=latent_channels)
        
        self.render_size = 64
        self.volume_extent_world = 3.0

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar, x = self.reparametrize(x)
        x = x.unsqueeze(2)
        x = self.decoder(x)
        return (mu, logvar, x)


    def get_encoder_modules(self):
       return self.encoder.get_modules() + [self.reparametrize]

    
    def get_decoder_modules(self):
       return self.decoder.get_modules()


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
            render = renderer(cameras=cameras, volumes=volumes)[0]
            renders.append(render)
            
            del renderer
            torch.cuda.empty_cache()

        return torch.stack(renders)