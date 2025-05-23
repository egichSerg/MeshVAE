import torch.nn as nn
import torch

from pathlib import Path

class VGG19(nn.Module):
    """
     Simplified version of the VGG19 "feature" block
     This module's only job is to return the "feature loss" for the inputs
    """

    def __init__(self, load_weights, weights_path, channel_in=3, width=64, num_classes=10):
        super(VGG19, self).__init__()
        
        self.conv1 = nn.Conv3d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv3d(width, width, 3, 1, 1)

        self.conv3 = nn.Conv3d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv3d(2 * width, 2 * width, 3, 1, 1)

        self.conv5 = nn.Conv3d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv3d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv3d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv3d(4 * width, 4 * width, 3, 1, 1)

        self.conv9 = nn.Conv3d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv3d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv3d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv3d(8 * width, 8 * width, 3, 1, 1)

        self.conv13 = nn.Conv3d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv3d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv3d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv3d(8 * width, 8 * width, 3, 1, 1)

        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        if load_weights:
            self.classifier = nn.Linear(139264, 5)
            self.load_params_(num_classes, weights_path)
        else:
            self.classifier = nn.Linear(139264, num_classes)
            self.load_weights()

    def load_params_(self, num_classes, weights_path):
        # path = Path('./model_collection/weights') / 'VGG19-ShapeNet10-mvcnn.pt'
        path = weights_path
        state_dict = torch.load(path)
        for ((name, source_param), target_param) in zip(state_dict.items(), self.parameters()):
            target_param.data = source_param.data
            target_param.requires_grad = False

        if num_classes != 10:
            self.classifier = nn.Linear(139264, num_classes)

    def load_weights(self):
        # Download and load Pytorch's pre-trained weights
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        for ((name, source_param), target_param) in zip(state_dict.items(), self.parameters()):
            param_name, _, param_type = name.split('.')
            if param_name == 'classifier':
                continue
            if param_type == 'weight':
                target_param.data = torch.stack([source_param.data] , dim=2)
            elif param_type == 'bias':
                target_param.data = source_param.data

    def feature_loss(self, x):
        return (x[:x.shape[0] // 2] - x[x.shape[0] // 2:]).pow(2).mean()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.relu(x))
        x = self.mp(self.relu(x))  # 64x64

        x = self.conv3(x)
        x = self.conv4(self.relu(x))
        x = self.mp(self.relu(x))  # 32x32

        x = self.conv5(x)
        x = self.conv6(self.relu(x))
        x = self.conv7(self.relu(x))
        x = self.conv8(self.relu(x))
        x = self.mp(self.relu(x))  # 16x16

        x = self.conv9(x)
        x = self.conv10(self.relu(x))
        x = self.conv11(self.relu(x))
        x = self.conv12(self.relu(x))
        x = self.mp(self.relu(x))  # 8x8

        x = self.conv13(x)
        x = self.conv14(self.relu(x))
        x = self.conv15(self.relu(x))
        x = self.conv16(self.relu(x))
        
        x = self.flatten(self.relu(x))
        x = self.classifier(x)
        return x

    
    def perc_loss(self, x):
        """
        :param x: Expects x to be the target and source to concatenated on dimension 0
        :return: Feature loss
        """
        x = self.conv1(x)
        loss = self.feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 64x64

        x = self.conv3(x)
        loss += self.feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 32x32

        x = self.conv5(x)
        loss += self.feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 16x16

        x = self.conv9(x)
        loss += self.feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 8x8

        x = self.conv13(x)
        loss += self.feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += self.feature_loss(x)

        return loss/16