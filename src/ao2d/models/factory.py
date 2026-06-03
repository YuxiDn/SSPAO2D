from __future__ import annotations

from typing import Any

import torch.nn as nn

from ao2d.optics import AO2DConfig

from . import (
    ABESplitNet2D,
    ABEFusionNet2D,
    ABEFusionNetV2D,
    CARE2D,
    DFCAN2D,
    PICNet2D,
    RCAN2D,
    RLN2D,
    SCARE2D,
    SFENet2D,
)


def _as_tuple(values: Any) -> tuple | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        return tuple(values)
    return (values,)


def _make_template_optics_config(config: dict[str, Any]) -> AO2DConfig:
    template_cfg = config.get("template_attention", {})
    optics = template_cfg.get("optics", config.get("template_optics", config.get("optics", {})))
    return AO2DConfig(
        pixel_size=float(optics.get("pixel_size", 0.300)),
        na=float(optics.get("na", 1.05)),
        lambda_emission=float(optics.get("lambda_emission", optics.get("wavelength", 1.000))),
        lambda_excitation=float(optics.get("lambda_excitation", 0.808)),
        mode=str(optics.get("mode", "widefield")),
        pinhole_au=float(optics.get("pinhole_au", 1.0)),
        lightsheet_fwhm=float(optics.get("lightsheet_fwhm", 1.2)),
        normalize=str(optics.get("normalize", "sum")),
    )


def _template_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    template_cfg = config.get("template_attention", {})
    image_size = template_cfg.get(
        "image_size",
        config.get("template_image_size", config.get("image_size", config.get("patch_size"))),
    )
    epsilon = template_cfg.get("epsilon_um", config.get("template_epsilon_um", 0.05))
    max_amp = template_cfg.get("max_amp_um", config.get("template_max_amp_um"))
    signed_indices = template_cfg.get(
        "signed_zernike_indices",
        config.get("template_signed_zernike_indices", None),
    )
    return dict(
        template_image_size=None if image_size is None else tuple(int(v) for v in image_size),
        template_optics_config=_make_template_optics_config(config),
        template_epsilon_um=_as_tuple(epsilon) if isinstance(epsilon, (list, tuple)) else float(epsilon),
        template_max_amp_um=(
            None
            if max_amp is None
            else (_as_tuple(max_amp) if isinstance(max_amp, (list, tuple)) else float(max_amp))
        ),
        template_max_amp_base_um=float(template_cfg.get("max_amp_base_um", config.get("template_max_amp_base_um", 0.12))),
        template_encoder_channels=int(template_cfg.get("encoder_channels", config.get("template_encoder_channels", 32))),
        template_encoder_blocks=int(template_cfg.get("encoder_blocks", config.get("template_encoder_blocks", 3))),
        template_encoder_kernel_size=int(
            template_cfg.get("encoder_depthwise_kernel", config.get("template_encoder_kernel_size", 5))
        ),
        template_encoder_type=str(template_cfg.get("encoder_type", config.get("template_encoder_type", "depthwise"))),
        template_alpha=float(template_cfg.get("alpha", config.get("template_alpha", 10.0))),
        template_tau_init=float(template_cfg.get("tau_init", config.get("template_tau_init", 0.3))),
        template_confidence_init=float(
            template_cfg.get("confidence_init", config.get("template_confidence_init", 1.4))
        ),
        template_fft_shift=bool(template_cfg.get("fft_shift", config.get("template_fft_shift", config.get("fft_shift", True)))),
        template_input_center=bool(template_cfg.get("input_center", config.get("template_input_center", True))),
        template_input_phase_mask_percentile=float(
            template_cfg.get("input_phase_mask_percentile", config.get("template_input_phase_mask_percentile", 72.0))
        ),
        template_otf_mtf_threshold=float(
            template_cfg.get("otf_mtf_threshold", config.get("template_otf_mtf_threshold", 0.03))
        ),
        template_signed_zernike_indices=None if signed_indices is None else tuple(int(v) for v in signed_indices),
    )


def make_model(config: dict[str, Any]) -> nn.Module:
    name = str(config.get("name", config.get("model", "scare2d"))).lower()
    common = dict(
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        final_activation=config.get("final_activation", "sigmoid"),
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
    if name in {"rln", "rln2d", "richardson_lucy_net", "richardson-lucy-net"}:
        return RLN2D(
            in_channels=int(config.get("in_channels", 1)),
            out_channels=int(config.get("out_channels", 1)),
            num_features=int(config.get("num_features", config.get("base_channels", 8))),
            num_iterations=int(config.get("num_iterations", config.get("iterations", 1))),
            kernel_size=int(config.get("kernel_size", 3)),
            norm=str(config.get("norm", "batch")),
            eps=float(config.get("eps", 1e-4)),
            ratio_clip=(
                None
                if config.get("ratio_clip", 10.0) is None
                else float(config.get("ratio_clip", config.get("clip_ratio", 10.0)))
            ),
            residual=bool(config.get("residual", False)),
            final_activation=config.get("final_activation", "softplus"),
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
    if name in {"abenet", "abenet2d", "abe_fusion", "abefusionnet2d"}:
        return ABEFusionNet2D(
            **common,
            zernike_modes=int(config.get("zernike_modes", len(config.get("zernike_indices", list(range(3, 16)))))),
            branch_channels=int(config.get("branch_channels", 32)),
            fusion_channels=int(config.get("fusion_channels", 96)),
            branch_depth=int(config.get("branch_depth", 2)),
            obj_base_channels=int(config.get("obj_base_channels", config.get("base_channels", 64))),
            obj_depth=int(config.get("obj_depth", config.get("unet_depth", config.get("depth", 3)))),
            zernike_hidden=int(config.get("zernike_hidden", 128)),
            zernike_depth=int(config.get("zernike_depth", 3)),
            zernike_reduction=int(config.get("zernike_reduction", config.get("reduction", 8))),
            zernike_indices=tuple(config["zernike_indices"]) if "zernike_indices" in config else None,
            aberration_head_type=str(config.get("aberration_head_type", "direct")),
            delta_phi_pair_count=int(config.get("delta_phi_pair_count", 512)),
            delta_phi_pupil_grid_size=int(config.get("delta_phi_pupil_grid_size", 32)),
            delta_phi_ridge=float(config.get("delta_phi_ridge", 1e-4)),
            delta_phi_max_opd=config.get("delta_phi_max_opd", 0.75),
            fft=bool(config.get("fft", True)),
            fft_shift=bool(config.get("fft_shift", False)),
            fft_phase_features=bool(config.get("fft_phase_features", False)),
            num_pixel_stack_layer=int(config.get("num_pixel_stack_layer", 0)),
            **_template_kwargs(config),
        )
    if name in {"abenetv2", "abenetv2d", "abenet2dv2", "abe_fusion_v2", "abefusionnetv2d"}:
        return ABEFusionNetV2D(
            **common,
            zernike_modes=int(config.get("zernike_modes", len(config.get("zernike_indices", list(range(3, 16)))))),
            branch_channels=int(config.get("branch_channels", 32)),
            fusion_channels=int(config.get("fusion_channels", 96)),
            branch_depth=int(config.get("branch_depth", 2)),
            obj_base_channels=int(config.get("obj_base_channels", config.get("base_channels", 64))),
            obj_depth=int(config.get("obj_depth", config.get("unet_depth", config.get("depth", 3)))),
            zernike_hidden=int(config.get("zernike_hidden", 128)),
            zernike_depth=int(config.get("zernike_depth", 3)),
            zernike_reduction=int(config.get("zernike_reduction", config.get("reduction", 8))),
            zernike_indices=tuple(config["zernike_indices"]) if "zernike_indices" in config else None,
            aberration_head_type=str(config.get("aberration_head_type", "direct")),
            delta_phi_pair_count=int(config.get("delta_phi_pair_count", 512)),
            delta_phi_pupil_grid_size=int(config.get("delta_phi_pupil_grid_size", 32)),
            delta_phi_ridge=float(config.get("delta_phi_ridge", 1e-4)),
            delta_phi_max_opd=config.get("delta_phi_max_opd", 0.75),
            fft=bool(config.get("fft", True)),
            fft_shift=bool(config.get("fft_shift", False)),
            fft_phase_features=bool(config.get("fft_phase_features", False)),
            num_pixel_stack_layer=int(config.get("num_pixel_stack_layer", 0)),
            image_gate_obj_init=config.get("image_gate_obj_init", 0.3),
            **_template_kwargs(config),
        )
    if name in {"abesplit", "abesplit2d", "abenet_split", "abenet_split2d", "abefusionnetsplit2d"}:
        return ABESplitNet2D(
            **common,
            zernike_modes=int(config.get("zernike_modes", len(config.get("zernike_indices", list(range(3, 16)))))),
            branch_channels=int(config.get("branch_channels", 32)),
            fusion_channels=int(config.get("fusion_channels", 96)),
            branch_depth=int(config.get("branch_depth", 2)),
            obj_branch_channels=(
                int(config["obj_branch_channels"]) if "obj_branch_channels" in config else None
            ),
            abe_branch_channels=(
                int(config["abe_branch_channels"]) if "abe_branch_channels" in config else None
            ),
            obj_fusion_channels=(
                int(config["obj_fusion_channels"]) if "obj_fusion_channels" in config else None
            ),
            abe_fusion_channels=(
                int(config["abe_fusion_channels"]) if "abe_fusion_channels" in config else None
            ),
            obj_base_channels=int(config.get("obj_base_channels", config.get("base_channels", 64))),
            obj_depth=int(config.get("obj_depth", config.get("unet_depth", config.get("depth", 3)))),
            zernike_hidden=int(config.get("zernike_hidden", 128)),
            zernike_depth=int(config.get("zernike_depth", 3)),
            zernike_reduction=int(config.get("zernike_reduction", config.get("reduction", 8))),
            zernike_indices=tuple(config["zernike_indices"]) if "zernike_indices" in config else None,
            aberration_head_type=str(config.get("aberration_head_type", "direct")),
            delta_phi_pair_count=int(config.get("delta_phi_pair_count", 512)),
            delta_phi_pupil_grid_size=int(config.get("delta_phi_pupil_grid_size", 32)),
            delta_phi_ridge=float(config.get("delta_phi_ridge", 1e-4)),
            delta_phi_max_opd=config.get("delta_phi_max_opd", 0.75),
            fft=bool(config.get("fft", True)),
            fft_shift=bool(config.get("fft_shift", False)),
            fft_phase_features=bool(config.get("fft_phase_features", False)),
            num_pixel_stack_layer=int(config.get("num_pixel_stack_layer", 0)),
            **_template_kwargs(config),
        )
    raise ValueError(f"Unknown model name: {name}")
