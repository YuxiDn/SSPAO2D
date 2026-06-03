from __future__ import annotations

import torch
import torch.nn as nn

from ao2d.optics import AO2DConfig

from .abenet2d import (
    BranchEncoder2D,
    LogFFTAmplitude2D,
    LogFFTAmplitudePhase2D,
    forward_aberration_head_2d,
    make_aberration_head_2d,
)
from .blocks import output_activation
from .scare2d import SobelGradient2D
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
        zernike_indices: tuple[int, ...] | None = None,
        aberration_head_type: str = "direct",
        delta_phi_pair_count: int = 512,
        delta_phi_pupil_grid_size: int = 32,
        delta_phi_ridge: float = 1e-4,
        delta_phi_max_opd: float | None = 0.75,
        template_image_size: tuple[int, int] | None = None,
        template_optics_config: AO2DConfig | None = None,
        template_epsilon_um: float | tuple[float, ...] = 0.05,
        template_max_amp_um: float | tuple[float, ...] | None = None,
        template_max_amp_base_um: float = 0.12,
        template_encoder_channels: int = 32,
        template_encoder_blocks: int = 3,
        template_encoder_kernel_size: int = 5,
        template_encoder_type: str = "depthwise",
        template_eta: float = 5.0,
        template_alpha: float = 10.0,
        template_tau_init: float = 0.3,
        template_confidence_init: float = 1.4,
        template_fft_shift: bool = True,
        template_input_center: bool = True,
        template_input_phase_mask_percentile: float = 72.0,
        template_input_feature_channels: int = 5,
        template_otf_mtf_threshold: float = 0.03,
        fft: bool = True,
        fft_shift: bool = False,
        fft_phase_features: bool = False,
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
        self.frequency_transform = (
            LogFFTAmplitudePhase2D(fft=fft, fft_shift=fft_shift)
            if fft_phase_features
            else LogFFTAmplitude2D(fft=fft, fft_shift=fft_shift)
        )
        frequency_channels = in_channels * 3 if fft and fft_phase_features else in_channels
        self.image_branch = BranchEncoder2D(in_channels, branch_channels, depth=branch_depth)
        self.gradient_branch = BranchEncoder2D(in_channels, branch_channels, depth=branch_depth)
        self.frequency_branch = BranchEncoder2D(frequency_channels, branch_channels, depth=branch_depth)
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
        self.zernike_head = make_aberration_head_2d(
            fusion_channels,
            zernike_modes,
            zernike_hidden,
            zernike_depth,
            zernike_reduction,
            head_type=aberration_head_type,
            zernike_indices=zernike_indices,
            delta_phi_pair_count=delta_phi_pair_count,
            delta_phi_pupil_grid_size=delta_phi_pupil_grid_size,
            delta_phi_ridge=delta_phi_ridge,
            delta_phi_max_opd=delta_phi_max_opd,
            template_image_size=template_image_size,
            template_optics_config=template_optics_config,
            template_epsilon_um=template_epsilon_um,
            template_max_amp_um=template_max_amp_um,
            template_max_amp_base_um=template_max_amp_base_um,
            template_encoder_channels=template_encoder_channels,
            template_encoder_blocks=template_encoder_blocks,
            template_encoder_kernel_size=template_encoder_kernel_size,
            template_encoder_type=template_encoder_type,
            template_eta=template_eta,
            template_alpha=template_alpha,
            template_tau_init=template_tau_init,
            template_confidence_init=template_confidence_init,
            template_fft_shift=template_fft_shift,
            template_input_center=template_input_center,
            template_input_phase_mask_percentile=template_input_phase_mask_percentile,
            template_input_feature_channels=template_input_feature_channels,
            template_otf_mtf_threshold=template_otf_mtf_threshold,
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
        zernike = forward_aberration_head_2d(self.zernike_head, zernike_fused, x)
        return obj, zernike
