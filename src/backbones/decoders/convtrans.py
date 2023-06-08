import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class ConvTransHead(nn.Module):
    def __init__(
        self,
        channels=768,
        num_classes=14,
        norm_name="instance",
    ):
        super(ConvTransHead, self).__init__()

        self.decoder1 = nn.ConvTranspose2d(
            channels, channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder2 = nn.ConvTranspose2d(
            channels, channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder3 = nn.ConvTranspose2d(
            channels, channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder4 = nn.ConvTranspose2d(
            channels, num_classes, kernel_size=(2, 2), stride=(2, 2)
        )
        self.norm0 = get_norm_layer(name=norm_name, spatial_dims=2, channels=channels)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=2, channels=channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=2, channels=channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=2, channels=channels)

    def forward(self, inputs):
        x = inputs[-1]
        x = F.relu(self.norm0(x))
        x = F.relu(self.norm1(self.decoder1(x)))
        x = F.relu(self.norm2(self.decoder2(x)))
        x = F.relu(self.norm3(self.decoder3(x)))
        x = self.decoder4(x)
        return x
