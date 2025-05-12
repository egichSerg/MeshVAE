import torch
import torch.nn as nn

class Reparametrize(nn.Module):
    def __init__(self, in_channels=1024, latent_channels=512):
        super(Reparametrize, self).__init__()

        self.conv_mu = nn.Conv2d(in_channels, latent_channels, 4, 1)
        self.conv_log_var = nn.Conv2d(in_channels, latent_channels, 4, 1)

    def forward(self, x):
        x = x.squeeze()
        mu = self.conv_mu(x)
        logvar = self.conv_log_var(x)

        # reparametrize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu, logvar, mu + eps * std)