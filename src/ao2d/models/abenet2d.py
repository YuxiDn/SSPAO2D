from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import output_activation
from .scare2d import SobelGradient2D, ZernikeResNetRegression2D
from .sfenet2d import ResUNet2D


class ResidualConvBlock2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class BranchEncoder2D(nn.Module):
    """Conv + residual blocks used by image, gradient, and frequency inputs."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 2) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(ResidualConvBlock2D(out_channels) for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class LogFFTAmplitude2D(nn.Module):
    """Per-sample normalized log FFT amplitude transform."""

    def __init__(self, fft: bool = True, fft_shift: bool = False) -> None:
        super().__init__()
        self.fft = fft
        self.fft_shift = fft_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fft:
            return x
        amp = torch.abs(torch.fft.fft2(x.float(), dim=(-2, -1))).clamp_min(1e-8)
        amp = torch.log10(1 + amp)
        if self.fft_shift:
            amp = torch.fft.fftshift(amp, dim=(-2, -1))
        mean = amp.mean(dim=(-2, -1), keepdim=True)
        std = amp.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return (amp - mean) / std


class ABEFusionNet2D(nn.Module):
    """SFE/PICNet-style model for aberrated input.

    The input aberrated image is encoded by image, Sobel-gradient, and log-FFT
    branches. Their features are concatenated, fused with a 1x1 convolution,
    and decoded by two heads: a ResUNet object head and a ResNet Zernike head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        zernike_modes: int = 13,
        branch_channels: int = 32,
        fusion_channels: int = 96,
        branch_depth: int = 2,
        obj_base_channels: int = 64,
        obj_depth: int = 3,
        zernike_hidden: int = 128,
        zernike_depth: int = 3,
        zernike_reduction: int = 8,
        fft: bool = True,
        fft_shift: bool = False,
        num_pixel_stack_layer: int = 0,
        final_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if branch_channels < 1:
            raise ValueError("branch_channels must be positive")
        if fusion_channels < 1:
            raise ValueError("fusion_channels must be positive")

        self.gradient_transform = SobelGradient2D(in_channels)
        self.frequency_transform = LogFFTAmplitude2D(fft=fft, fft_shift=fft_shift)
        self.image_branch = BranchEncoder2D(in_channels, branch_channels, depth=branch_depth)
        self.gradient_branch = BranchEncoder2D(in_channels, branch_channels, depth=branch_depth)
        self.frequency_branch = BranchEncoder2D(in_channels, branch_channels, depth=branch_depth)
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 3, fusion_channels, 1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.object_head = ResUNet2D(
            fusion_channels,
            out_channels,
            base_channels=obj_base_channels,
            depth=obj_depth,
            num_pixel_stack_layer=num_pixel_stack_layer,
        )
        self.zernike_head = ZernikeResNetRegression2D(
            fusion_channels,
            zernike_modes,
            hidden=zernike_hidden,
            depth=zernike_depth,
            reduction=zernike_reduction,
        )
        self.activation = output_activation(final_activation)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.image_branch(x)
        gradient = self.gradient_branch(self.gradient_transform(x))
        frequency = self.frequency_branch(self.frequency_transform(x))
        fused = self.fusion(torch.cat([image, gradient, frequency], dim=1))
        obj = self.activation(self.object_head(fused))
        zernike = self.zernike_head(fused)
        return obj, zernike
