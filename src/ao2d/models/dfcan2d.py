from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import fft_amplitude_2d, output_activation


class FCALayer2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.spatial = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = fft_amplitude_2d(x)
        return x * self.gate(self.spatial(freq))


class FCAB2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.fca = FCALayer2D(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fca(self.body(x))


class DFCANGroup2D(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 4, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(*(FCAB2D(channels, reduction) for _ in range(num_blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DFCAN2D(nn.Module):
    """2-D DFCAN baseline with Fourier channel attention."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_groups: int = 4,
        num_blocks: int = 4,
        reduction: int = 16,
        final_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.entry = nn.Sequential(nn.Conv2d(in_channels, num_features, 3, padding=1), nn.GELU())
        self.groups = nn.Sequential(*(DFCANGroup2D(num_features, num_blocks, reduction) for _ in range(num_groups)))
        self.exit = nn.Conv2d(num_features, out_channels, 3, padding=1)
        self.activation = output_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.exit(self.groups(self.entry(x))))

