# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""U-Net parts.

Inspired by https://github.com/milesial/Pytorch-UNet/tree/master .
"""

from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DoubleConv(nn.Module):
    """Basic block for U-Nets."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        kernel_size: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        act_layer: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            norm_layer(mid_channels),
            act_layer(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=False,
            ),
            norm_layer(out_channels),
            act_layer(),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        act_layer: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        act_layer: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    @override
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        pad_h = x2.size()[2] - x1.size()[2]
        pad_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 output convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
