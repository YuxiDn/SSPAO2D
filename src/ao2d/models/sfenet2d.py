from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DoubleConv2D, Down2D, Up2D, fft_amplitude_2d, output_activation
from .rcan2d import RCAN2D


class ResUNet2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64, depth: int = 3) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(depth + 1)]
        self.in_conv = DoubleConv2D(in_channels, channels[0], norm="instance")
        self.downs = nn.ModuleList(Down2D(channels[i], channels[i + 1], norm="instance") for i in range(depth))
        self.ups = nn.ModuleList(
            Up2D(channels[depth - i], channels[depth - i - 1], channels[depth - i - 1], norm="instance")
            for i in range(depth)
        )
        self.out = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for i, up in enumerate(self.ups):
            x = up(x, skips[-i - 2])
        return self.out(x)


class SFENet2D(nn.Module):
    """2-D spatial-frequency enhancement network converted from SSPAO's SFENet."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 32,
        encoder_channels: int = 64,
        num_groups: int = 3,
        num_blocks: int = 3,
        reduction: int = 16,
        fft_branch: bool = True,
        unet_depth: int = 3,
        final_activation: str = "relu",
    ) -> None:
        super().__init__()
        half = encoder_channels // 2 if fft_branch else encoder_channels
        self.fft_branch = fft_branch
        self.spatial_encoder = RCAN2D(
            in_channels,
            half,
            num_features=num_features,
            num_groups=num_groups,
            num_blocks=num_blocks,
            reduction=reduction,
            final_activation="none",
        )
        if fft_branch:
            self.frequency_encoder = RCAN2D(
                in_channels,
                half,
                num_features=num_features,
                num_groups=num_groups,
                num_blocks=num_blocks,
                reduction=reduction,
                final_activation="none",
            )
        self.decoder = ResUNet2D(encoder_channels, out_channels, base_channels=max(16, encoder_channels), depth=unet_depth)
        self.activation = output_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_encoder(x)
        if self.fft_branch:
            freq = self.frequency_encoder(fft_amplitude_2d(x))
            encoded = torch.cat([spatial, freq], dim=1)
        else:
            encoded = spatial
        return self.activation(self.decoder(encoded))

