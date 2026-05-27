from __future__ import annotations

import torch
import torch.nn as nn

from .abenet2d import BranchEncoder2D, LogFFTAmplitude2D
from .blocks import output_activation
from .scare2d import SobelGradient2D, ZernikeResNetRegression2D
from .sfenet2d import ResUNet2D


class ABEFusionNetV2D(nn.Module):
    """ABE fusion model with separate object and Zernike fused features.

    The object head uses image and Sobel-gradient features only. The Zernike
    head uses image, Sobel-gradient, and log-FFT frequency features.
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
        image_gate_obj_init: float | None = 0.3,
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
        self.image_gate_obj = (
            nn.Parameter(torch.tensor(float(image_gate_obj_init))) if image_gate_obj_init is not None else None
        )
        self.obj_fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 2, fusion_channels, 1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.zernike_fusion = nn.Sequential(
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

        obj_image = image if self.image_gate_obj is None else self.image_gate_obj * image
        obj_fused = self.obj_fusion(torch.cat([obj_image, gradient], dim=1))
        zernike_fused = self.zernike_fusion(torch.cat([image, gradient, frequency], dim=1))

        obj = self.activation(self.object_head(obj_fused))
        zernike = self.zernike_head(zernike_fused)
        return obj, zernike
