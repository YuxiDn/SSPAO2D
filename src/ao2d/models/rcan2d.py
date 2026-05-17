from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import output_activation


class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class RCAB2D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, bn: bool = False, reduction: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)]
        if bn:
            layers.append(nn.BatchNorm2d(channels))
        layers += [nn.ReLU(inplace=True), nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)]
        if bn:
            layers.append(nn.BatchNorm2d(channels))
        self.body = nn.Sequential(*layers)
        self.ca = ChannelAttention2D(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ca(self.body(x))


class ResidualGroup2D(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 5, bn: bool = False, reduction: int = 8, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(RCAB2D(channels, bn=bn, reduction=reduction) for _ in range(num_blocks)))
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.blocks(x)) + self.residual_scale * x


class RCAN2D(nn.Module):
    """2-D RCAN baseline converted from SSPAO's RCAN3D."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_groups: int = 5,
        num_blocks: int = 5,
        reduction: int = 8,
        bn: bool = False,
        residual_scale: float = 1.0,
        final_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.groups = nn.Sequential(
            *(ResidualGroup2D(num_features, num_blocks, bn, reduction, residual_scale) for _ in range(num_groups))
        )
        self.body = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)
        self.activation = output_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        body = self.body(self.groups(feat)) + feat
        return self.activation(self.tail(body))

