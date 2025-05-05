import torch
import torch.nn as nn

class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv3d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm3d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv3d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm3d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv3d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, hidden_channels=64, latent_channels=512):
        super(Decoder, self).__init__()
        self.conv_t_up = nn.ConvTranspose3d(latent_channels, hidden_channels * 16, 4, 1)
        self.res_up_block1 = ResUp(hidden_channels * 16, hidden_channels * 8)
        self.res_up_block2 = ResUp(hidden_channels * 8, hidden_channels * 4)
        self.res_up_block3 = ResUp(hidden_channels * 4, hidden_channels * 2)
        self.res_up_block4 = ResUp(hidden_channels * 2, hidden_channels)
        self.conv_out = nn.Conv3d(hidden_channels, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_t_up(x))  # 4
        x = self.res_up_block1(x)  # 8
        x = self.res_up_block2(x)  # 16
        x = self.res_up_block3(x)  # 32
        x = self.res_up_block4(x)  # 64
        x = torch.sigmoid(self.conv_out(x))

        return x

    def get_modules(self):
        return [self.conv_t_up, self.act_fnc,
               self.res_up_block1, self.res_up_block2,
               self.res_up_block3, self.res_up_block4,
               self.conv_out, nn.Sigmoid()]