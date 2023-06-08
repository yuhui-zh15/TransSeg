import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class SetrPupHead(nn.Module):
    def __init__(
        self,
        channels=768,
        num_classes=14,
        norm_name="instance",
    ):
        super(SetrPupHead, self).__init__()

        self.decoder0 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.decoder1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.decoder2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.decoder3 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.decoder4 = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)
        self.norm0 = get_norm_layer(name=norm_name, spatial_dims=2, channels=channels)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=2, channels=channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=2, channels=channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=2, channels=channels)

    def forward(self, inputs):
        x = inputs[-1]

        x = self.decoder0(x)
        x = self.norm0(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear")

        x = self.decoder1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear")

        x = self.decoder2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear")

        x = self.decoder3(x)
        x = self.norm3(x)
        x = F.relu(x, inplace=True)
        x = self.decoder4(x)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear")
        return x
