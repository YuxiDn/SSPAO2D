from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .care2d import CARE2D
from .rcan2d import RCAN2D


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


class ZernikeResidualBlock2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        residual = residual * self.attn(residual)
        return self.act(x + residual)


class ZernikeResNetRegression2D(nn.Module):
    """Residual convolutional regressor for fused spatial/frequency/gradient features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 128,
        depth: int = 3,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.blocks = nn.Sequential(*(ZernikeResidualBlock2D(hidden, reduction=reduction) for _ in range(depth)))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, out_channels),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class SobelGradient2D(nn.Module):
    """Fixed Sobel gradient magnitude transform with per-sample normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        self.channels = channels
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1) / 8.0)
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1) / 8.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        grad_x = F.conv2d(x32, self.kernel_x, padding=1, groups=self.channels)
        grad_y = F.conv2d(x32, self.kernel_y, padding=1, groups=self.channels)
        grad = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
        mean = grad.mean(dim=(-2, -1), keepdim=True)
        std = grad.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return (grad - mean) / std


class ZernikeMultiBranch2D(nn.Module):
    """SFE-style Zernike branch: spatial latent plus frequency and gradient RCAN features."""

    def __init__(
        self,
        in_channels: int,
        spatial_channels: int,
        out_channels: int,
        branch_channels: int = 64,
        num_features: int = 32,
        num_groups: int = 3,
        num_blocks: int = 3,
        reduction: int = 16,
        fft_branch: bool = True,
        gradient_branch: bool = True,
        fft: bool = True,
        fft_shift: bool = False,
        zernike_hidden: int = 128,
        zernike_depth: int = 3,
    ) -> None:
        super().__init__()
        if branch_channels < 1:
            raise ValueError("branch_channels must be positive")
        self.fft_branch = fft_branch
        self.gradient_branch = gradient_branch
        self.fft = fft
        self.fft_shift = fft_shift

        fused_channels = spatial_channels
        if fft_branch:
            self.frequency_encoder = RCAN2D(
                in_channels,
                branch_channels,
                num_features=num_features,
                num_groups=num_groups,
                num_blocks=num_blocks,
                reduction=reduction,
                final_activation="none",
            )
            fused_channels += branch_channels
        if gradient_branch:
            self.gradient_transform = SobelGradient2D(in_channels)
            self.gradient_encoder = RCAN2D(
                in_channels,
                branch_channels,
                num_features=num_features,
                num_groups=num_groups,
                num_blocks=num_blocks,
                reduction=reduction,
                final_activation="none",
            )
            fused_channels += branch_channels

        self.regressor = ZernikeResNetRegression2D(
            fused_channels,
            out_channels,
            hidden=zernike_hidden,
            depth=zernike_depth,
            reduction=max(1, reduction),
        )

    def forward(self, x: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        features = [spatial]
        target_size = spatial.shape[-2:]
        if self.fft_branch:
            freq = self.frequency_encoder(self._frequency_transform(x))
            features.append(F.adaptive_avg_pool2d(freq, target_size))
        if self.gradient_branch:
            grad = self.gradient_encoder(self.gradient_transform(x))
            features.append(F.adaptive_avg_pool2d(grad, target_size))
        return self.regressor(torch.cat(features, dim=1))

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


class SCARE2D(CARE2D):
    """2-D SCARE: CARE restoration plus multi-branch Zernike coefficient regression."""

    def __init__(
        self,
        *args,
        zernike_modes: int = 13,
        zernike_hidden: int = 128,
        zernike_depth: int = 3,
        zernike_branch_channels: int = 64,
        zernike_num_features: int = 32,
        zernike_num_groups: int = 3,
        zernike_num_blocks: int = 3,
        zernike_reduction: int = 16,
        zernike_fft_branch: bool = True,
        zernike_gradient_branch: bool = True,
        zernike_fft: bool = True,
        zernike_fft_shift: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        in_channels = int(args[0] if args else kwargs.get("in_channels", 1))
        self.zernike_head = ZernikeMultiBranch2D(
            in_channels,
            self.latent_channels,
            zernike_modes,
            branch_channels=zernike_branch_channels,
            num_features=zernike_num_features,
            num_groups=zernike_num_groups,
            num_blocks=zernike_num_blocks,
            reduction=zernike_reduction,
            fft_branch=zernike_fft_branch,
            gradient_branch=zernike_gradient_branch,
            fft=zernike_fft,
            fft_shift=zernike_fft_shift,
            zernike_hidden=zernike_hidden,
            zernike_depth=zernike_depth,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, skips = self.encode(x)
        restored = self.decode(latent, skips)
        coeff = self.zernike_head(x, latent)
        return restored, coeff
