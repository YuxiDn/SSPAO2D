from __future__ import annotations

from typing import Any

import torch.nn as nn

from . import CARE2D, DFCAN2D, PICNet2D, RCAN2D, SCARE2D, SFENet2D


def make_model(config: dict[str, Any]) -> nn.Module:
    name = str(config.get("name", config.get("model", "scare2d"))).lower()
    common = dict(
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        final_activation=str(config.get("final_activation", "sigmoid")),
    )
    if name in {"care", "care2d", "unet", "unet2d"}:
        return CARE2D(
            **common,
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            norm=str(config.get("norm", "batch")),
        )
    if name in {"scare", "scare2d"}:
        return SCARE2D(
            **common,
            base_channels=int(config.get("base_channels", 32)),
            depth=int(config.get("depth", 3)),
            norm=str(config.get("norm", "batch")),
            zernike_modes=int(config.get("zernike_modes", len(config.get("zernike_indices", list(range(3, 16)))))),
            zernike_hidden=int(config.get("zernike_hidden", 128)),
            zernike_depth=int(config.get("zernike_depth", 3)),
            zernike_branch_channels=int(config.get("zernike_branch_channels", 64)),
            zernike_num_features=int(config.get("zernike_num_features", config.get("num_features", 32))),
            zernike_num_groups=int(config.get("zernike_num_groups", config.get("num_groups", 3))),
            zernike_num_blocks=int(config.get("zernike_num_blocks", config.get("num_blocks", 3))),
            zernike_reduction=int(config.get("zernike_reduction", config.get("reduction", 16))),
            zernike_fft_branch=bool(config.get("zernike_fft_branch", True)),
            zernike_gradient_branch=bool(config.get("zernike_gradient_branch", True)),
            zernike_fft=bool(config.get("zernike_fft", config.get("fft", True))),
            zernike_fft_shift=bool(config.get("zernike_fft_shift", config.get("fft_shift", False))),
        )
    if name in {"rcan", "rcan2d"}:
        return RCAN2D(
            **common,
            num_features=int(config.get("num_features", config.get("base_channels", 64))),
            num_groups=int(config.get("num_groups", config.get("num_residual_groups", 5))),
            num_blocks=int(config.get("num_blocks", config.get("num_residual_blocks", 5))),
            reduction=int(config.get("reduction", 8)),
            bn=bool(config.get("bn", False)),
            residual_scale=float(config.get("residual_scale", 1.0)),
        )
    if name in {"dfcan", "dfcan2d"}:
        return DFCAN2D(
            **common,
            num_features=int(config.get("num_features", 64)),
            num_groups=int(config.get("num_groups", config.get("num_residual_groups", 4))),
            num_blocks=int(config.get("num_blocks", config.get("num_residual_blocks", 4))),
            reduction=int(config.get("reduction", 16)),
        )
    if name in {"sfenet", "sfenet2d", "sfe"}:
        return SFENet2D(
            **common,
            num_features=int(config.get("num_features", 32)),
            encoder_channels=int(config.get("encoder_channels", 64)),
            num_groups=int(config.get("num_groups", config.get("num_rg", 3))),
            num_blocks=int(config.get("num_blocks", config.get("num_rcab", 3))),
            reduction=int(config.get("reduction", 16)),
            fft_branch=bool(config.get("fft_branch", config.get("fft_brunch", True))),
            fft=bool(config.get("fft", True)),
            fft_shift=bool(config.get("fft_shift", False)),
            fft_forward=bool(config.get("fft_forward", True)),
            unet_depth=int(config.get("unet_depth", config.get("num_down_up", 3))),
            num_pixel_stack_layer=int(config.get("num_pixel_stack_layer", 0)),
        )
    if name in {"picnet", "picnet2d"}:
        return PICNet2D(
            **common,
            zernike_modes=int(config.get("zernike_modes", len(config.get("zernike_indices", list(range(3, 16)))))),
        )
    raise ValueError(f"Unknown model name: {name}")
