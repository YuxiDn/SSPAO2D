from __future__ import annotations

import torch
import torch.nn as nn

from .abenet2d import BranchEncoder2D, LogFFTAmplitude2D
from .blocks import output_activation
from .scare2d import SobelGradient2D, ZernikeResNetRegression2D
from .sfenet2d import ResUNet2D


class ABESplitNet2D(nn.Module):
    """ABE model with PICNet-style separated object and aberration generators.

    The object generator has its own image and Sobel-gradient feature branches.
    The aberration generator has separate image, Sobel-gradient, and log-FFT
    branches. No learnable feature encoders are shared between the two paths.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        zernike_modes: int = 13,
        branch_channels: int = 32,
        fusion_channels: int = 96,
        branch_depth: int = 2,
        obj_branch_channels: int | None = None,
        abe_branch_channels: int | None = None,
        obj_fusion_channels: int | None = None,
        abe_fusion_channels: int | None = None,
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
        obj_branch_channels = branch_channels if obj_branch_channels is None else obj_branch_channels
        abe_branch_channels = branch_channels if abe_branch_channels is None else abe_branch_channels
        obj_fusion_channels = fusion_channels if obj_fusion_channels is None else obj_fusion_channels
        abe_fusion_channels = fusion_channels if abe_fusion_channels is None else abe_fusion_channels
        for name, value in (
            ("obj_branch_channels", obj_branch_channels),
            ("abe_branch_channels", abe_branch_channels),
            ("obj_fusion_channels", obj_fusion_channels),
            ("abe_fusion_channels", abe_fusion_channels),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")

        self.gradient_transform = SobelGradient2D(in_channels)
        self.frequency_transform = LogFFTAmplitude2D(fft=fft, fft_shift=fft_shift)

        self.obj_image_branch = BranchEncoder2D(in_channels, obj_branch_channels, depth=branch_depth)
        self.obj_gradient_branch = BranchEncoder2D(in_channels, obj_branch_channels, depth=branch_depth)
        self.obj_fusion = nn.Sequential(
            nn.Conv2d(obj_branch_channels * 2, obj_fusion_channels, 1, bias=False),
            nn.BatchNorm2d(obj_fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.object_head = ResUNet2D(
            obj_fusion_channels,
            out_channels,
            base_channels=obj_base_channels,
            depth=obj_depth,
            num_pixel_stack_layer=num_pixel_stack_layer,
        )

        self.abe_image_branch = BranchEncoder2D(in_channels, abe_branch_channels, depth=branch_depth)
        self.abe_gradient_branch = BranchEncoder2D(in_channels, abe_branch_channels, depth=branch_depth)
        self.abe_frequency_branch = BranchEncoder2D(in_channels, abe_branch_channels, depth=branch_depth)
        self.abe_fusion = nn.Sequential(
            nn.Conv2d(abe_branch_channels * 3, abe_fusion_channels, 1, bias=False),
            nn.BatchNorm2d(abe_fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.aberration_head = ZernikeResNetRegression2D(
            abe_fusion_channels,
            zernike_modes,
            hidden=zernike_hidden,
            depth=zernike_depth,
            reduction=zernike_reduction,
        )
        self.activation = output_activation(final_activation)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gradient = self.gradient_transform(x)

        obj_image = self.obj_image_branch(x)
        obj_gradient = self.obj_gradient_branch(gradient)
        obj_fused = self.obj_fusion(torch.cat([obj_image, obj_gradient], dim=1))

        abe_image = self.abe_image_branch(x)
        abe_gradient = self.abe_gradient_branch(gradient)
        abe_frequency = self.abe_frequency_branch(self.frequency_transform(x))
        abe_fused = self.abe_fusion(torch.cat([abe_image, abe_gradient, abe_frequency], dim=1))

        obj = self.activation(self.object_head(obj_fused))
        zernike = self.aberration_head(abe_fused)
        return obj, zernike
