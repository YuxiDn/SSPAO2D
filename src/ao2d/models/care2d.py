from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DoubleConv2D, Down2D, Up2D, output_activation


class CARE2D(nn.Module):
    """2-D CARE baseline: encoder-decoder U-Net from SSPAO's CARE3D."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        norm: str = "batch",
        final_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**i) for i in range(depth + 1)]
        self.latent_channels = channels[-1]
        self.in_conv = DoubleConv2D(in_channels, channels[0], norm=norm)
        self.downs = nn.ModuleList(Down2D(channels[i], channels[i + 1], norm=norm) for i in range(depth))
        self.ups = nn.ModuleList(
            Up2D(channels[depth - i], channels[depth - i - 1], channels[depth - i - 1], norm=norm)
            for i in range(depth)
        )
        self.out_conv = nn.Conv2d(channels[0], out_channels, 1)
        self.activation = output_activation(final_activation)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.in_conv(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        return x, skips

    def decode(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for i, up in enumerate(self.ups):
            x = up(x, skips[-i - 2])
        return self.activation(self.out_conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, skips = self.encode(x)
        return self.decode(latent, skips)

