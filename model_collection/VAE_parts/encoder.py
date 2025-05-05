import torch.nn as nn

class ResDown(nn.Module):
    """
    Residual downsampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm3d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv3d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm3d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv3d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class Encoder(nn.Module):
    def __init__(self, channels, hidden_channels=64):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv3d(channels, hidden_channels, 7, 1, 3)
        self.res_down_block1 = ResDown(hidden_channels, 2 * hidden_channels)
        self.res_down_block2 = ResDown(2 * hidden_channels, 4 * hidden_channels)
        self.res_down_block3 = ResDown(4 * hidden_channels, 8 * hidden_channels)
        self.res_down_block4 = ResDown(8 * hidden_channels, 16 * hidden_channels)
        
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_in(x))
        x = self.res_down_block1(x)  # 32
        x = self.res_down_block2(x)  # 16
        x = self.res_down_block3(x)  # 8
        x = self.res_down_block4(x)  # 4
        return x

    def get_modules(self):
        return [self.conv_in, self.act_fnc,
               self.res_down_block1, self.res_down_block2,
               self.res_down_block3, self.res_down_block4]