from __future__ import annotations

import torch.nn as nn

import torch

from .blocks import DoubleConv2D, Down2D, Up2D, output_activation
from .rcan2d import RCAN2D


class ResUNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 3,
        num_pixel_stack_layer: int = 0,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(depth + 1)]
        self.in_conv = DoubleConv2D(in_channels, channels[0], norm="instance")
        self.downs = nn.ModuleList(Down2D(channels[i], channels[i + 1], norm="instance") for i in range(depth))
        self.ups = nn.ModuleList(
            Up2D(channels[depth - i], channels[depth - i - 1], channels[depth - i - 1], norm="instance")
            for i in range(depth)
        )
        if num_pixel_stack_layer < 0:
            raise ValueError("num_pixel_stack_layer must be non-negative")
        self.pixel_stack = nn.Sequential(
            *(
                nn.Sequential(
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(channels[0] * 4, channels[0], 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_pixel_stack_layer)
            )
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
        x = self.pixel_stack(x)
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
        fft: bool = True,
        fft_shift: bool = False,
        fft_forward: bool = True,
        unet_depth: int = 3,
        num_pixel_stack_layer: int = 0,
        final_activation: str = "relu",
    ) -> None:
        super().__init__()
        if encoder_channels % 2 != 0 and fft_branch:
            raise ValueError("encoder_channels must be even when fft_branch is enabled")
        half = encoder_channels // 2 if fft_branch else encoder_channels
        self.fft_branch = fft_branch
        self.fft = fft
        self.fft_shift = fft_shift
        self.fft_forward = fft_forward
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
        self.decoder = ResUNet2D(
            encoder_channels,
            out_channels,
            base_channels=max(16, encoder_channels),
            depth=unet_depth,
            num_pixel_stack_layer=num_pixel_stack_layer,
        )
        self.activation = output_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_encoder(x)
        if self.fft_branch:
            if self.fft_forward:
                freq = self._frequency_transform(x)
                freq = self.frequency_encoder(freq)
            else:
                freq = self.frequency_encoder(x)
                freq = self._frequency_transform(freq)
            encoded = torch.cat([spatial, freq], dim=1)
        else:
            encoded = spatial
        return self.activation(self.decoder(encoded))

    def _frequency_transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fft:
            return x
        amp = torch.abs(torch.fft.fft2(x.float(), dim=(-2, -1))).clamp_min(1e-8)
        amp = torch.log10(1 + amp)
        if self.fft_shift:
            amp = torch.fft.fftshift(amp, dim=(-2, -1))
        mean = amp.mean(dim=(-2, -1), keepdim=True)
        std = amp.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return (amp - mean) / std
