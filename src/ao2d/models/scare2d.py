from __future__ import annotations

import torch
import torch.nn as nn

from .care2d import CARE2D


class ZernikeRegression2D(nn.Module):
    """Global latent-feature regressor for OPD Zernike coefficients in micrometers."""

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        last = in_channels
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), nn.LeakyReLU(0.1, inplace=True)]
            last = hidden
        layers.append(nn.Linear(last, out_channels))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SCARE2D(CARE2D):
    """2-D SCARE: CARE-style restoration plus a Zernike coefficient branch."""

    def __init__(
        self,
        *args,
        zernike_modes: int = 13,
        zernike_hidden: int = 128,
        zernike_depth: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.zernike_head = ZernikeRegression2D(
            self.latent_channels,
            zernike_modes,
            hidden=zernike_hidden,
            depth=zernike_depth,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, skips = self.encode(x)
        restored = self.decode(latent, skips)
        coeff = self.zernike_head(latent)
        return restored, coeff

