import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):    
    """3x3 convolution"""
    return nn.Conv3d( in_channels, out_channels, kernel_size=3, stride=stride,
        padding=padding, bias=bias, groups=groups)

def conv1x1(in_channels, out_channels, groups=1, kernel_size=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     groups=groups, stride=1)

def upconv2x2(in_channels, out_channels):
    scale_factor = 2
    return nn.Sequential(
        nn.Upsample(mode='trilinear', scale_factor=scale_factor),
        conv1x1(in_channels, out_channels))

# upconv
class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = upconv2x2(self.in_channels, self.out_channels)
        self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, x):
        x = self.upconv(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# class
class UNetDecoder(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64):
        super(UNetDecoder, self).__init__()

        self.workpool = nn.AdaptiveAvgPool3d(9)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.up_convs = []

        outs = self.start_filts
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.up_convs = nn.ModuleList(self.up_convs)
        
        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    
    def render_volumes(self):
        pass

    def forward(self, x):
        x = self.workpool(x)
        for i, module in enumerate(self.up_convs):
            x = module(x)
        
        x = self.conv_final(x)
        return x