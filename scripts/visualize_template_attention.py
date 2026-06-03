#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao2d.data.dataset import _estimate_scale_from_input
from ao2d.data.io import IMAGE_EXTENSIONS, load_image
from ao2d.data.paths import resolve_path
from ao2d.models.factory import make_model
from ao2d.optics import (
    convolve_fft2,
    generate_psf2d_from_zernike,
    pupil_coordinates_2d,
    zernike_index_to_nm,
    zernike_wavefront,
)

from pretrain_template_attention import (
    get_template_head,
    make_dataset,
    make_optics_config,
    prepare_model_config,
    read_config,
    target_otf_features,
)


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _robust_limits(values: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [100.0 - percentile, percentile])
    if abs(float(hi - lo)) < 1e-12:
        center = float(np.mean(finite))
        return center - 1.0, center + 1.0
    return float(lo), float(hi)


def _symmetric_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return float(max(np.percentile(np.abs(finite), percentile), 1e-8))


def _show_image(ax, image: np.ndarray, title: str, cmap: str = "magma", symmetric: bool = False) -> None:
    if symmetric:
        limit = _symmetric_limit(image)
        im = ax.imshow(image, cmap=cmap, vmin=-limit, vmax=limit)
    else:
        vmin, vmax = _robust_limits(image)
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _save_grid(
    path: Path,
    images: list[np.ndarray],
    titles: list[str],
    ncols: int,
    cmap: str = "coolwarm",
    symmetric: bool = True,
    figsize_scale: float = 2.25,
) -> None:
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_scale * ncols, figsize_scale * nrows), squeeze=False)
    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            ax.axis("off")
            continue
        _show_image(ax, images[idx], titles[idx], cmap=cmap, symmetric=symmetric)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_heatmap(path: Path, matrix: np.ndarray, title: str, xlabels: list[str], ylabels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _object_id_from_name(name: str) -> str | None:
    match = re.search(r"(obj_\d+)", name)
    return match.group(1) if match else None


def _find_original_object(sample_path: Path, config: dict, data_root: Path, split: str) -> Path | None:
    object_id = _object_id_from_name(sample_path.stem)
    if object_id is None:
        return None
    data_cfg = config.get("data", {})
    candidates: list[Path] = []
    split_cfg = data_cfg.get(split, {})
    for key in ("object_dir", "obj_dir"):
        if key in split_cfg:
            candidates.append(resolve_path(split_cfg[key], data_root))
    if split == "val":
        candidates.append(data_root / "validation" / "OBJ")
    elif split == "test":
        candidates.append(data_root / "Test" / "OBJ")
    candidates.append(resolve_path(data_cfg.get("train", {}).get("object_dir", "OBJ"), data_root))
    candidates.append(data_root / "OBJ")

    seen: set[Path] = set()
    for root in candidates:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for suffix in IMAGE_EXTENSIONS:
            direct = root / f"{object_id}{suffix}"
            if direct.exists():
                return direct
        matches = sorted(p for p in root.iterdir() if p.is_file() and p.stem.startswith(object_id))
        if matches:
            return matches[0]
    return None


def _load_scaled_image(path: Path, config: dict) -> np.ndarray:
    raw = load_image(path).astype(np.float32, copy=False)
    scale = _estimate_scale_from_input(
        raw,
        method=str(config.get("data", {}).get("input_scale_method", "percentile")),
        percentile=float(config.get("data", {}).get("input_scale_percentile", 99.9)),
    )
    return np.maximum(raw, 0) / scale


def _crop_like_sample(image: np.ndarray, sample_hw: tuple[int, int]) -> np.ndarray:
    if image.shape[-2:] == sample_hw:
        return image
    h, w = sample_hw
    y0 = max(0, (image.shape[-2] - h) // 2)
    x0 = max(0, (image.shape[-1] - w) // 2)
    return image[y0 : y0 + h, x0 : x0 + w]


def _phase_products(
    coefficients: torch.Tensor,
    image_size: tuple[int, int],
    zernike_indices: tuple[int, ...],
    config: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    optics = make_optics_config(config)
    rho, theta, pupil_mask = pupil_coordinates_2d(
        image_size,
        optics.pixel_size,
        optics.lambda_emission,
        optics.na,
        device=device,
        dtype=torch.float32,
    )
    wavefront = zernike_wavefront(zernike_indices, coefficients[None].float().to(device), rho, theta)[0]
    phase = 2.0 * torch.pi / optics.lambda_emission * wavefront
    pupil = pupil_mask.to(phase.dtype)
    phase_in_pupil = phase * pupil
    phase_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(phase_in_pupil)))
    psf = generate_psf2d_from_zernike(image_size, zernike_indices, coefficients[None].float().to(device), optics)[0]
    otf = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(psf), s=image_size))
    return {
        "rho": rho,
        "pupil_mask": pupil_mask,
        "wavefront": wavefront * pupil,
        "phase": phase_in_pupil,
        "cos_phase": torch.cos(phase) * pupil,
        "sin_phase": torch.sin(phase) * pupil,
        "phase_fft_log_amp": torch.log1p(torch.abs(phase_fft)),
        "phase_fft_cos": phase_fft.real / torch.abs(phase_fft).clamp_min(1e-8),
        "phase_fft_sin": phase_fft.imag / torch.abs(phase_fft).clamp_min(1e-8),
        "psf": psf,
        "otf_log_amp": torch.log1p(torch.abs(otf)),
        "otf_cos": otf.real / torch.abs(otf).clamp_min(1e-8),
        "otf_sin": otf.imag / torch.abs(otf).clamp_min(1e-8),
    }


def _input_fft_features(image: torch.Tensor, fft_shift: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spectrum = torch.fft.fft2(image.float(), dim=(-2, -1))
    if fft_shift:
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    log_amp = torch.log1p(torch.abs(spectrum))
    denom = torch.abs(spectrum).clamp_min(1e-8)
    cos_phase = spectrum.real / denom
    sin_phase = spectrum.imag / denom
    return log_amp, cos_phase, sin_phase


def _fft_feature_stack_from_spectrum(spectrum: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    amp = torch.log1p(torch.abs(spectrum))
    denom = torch.abs(spectrum).clamp_min(eps)
    cos_phase = spectrum.real / denom
    sin_phase = spectrum.imag / denom
    return torch.stack([amp, cos_phase.to(amp.dtype), sin_phase.to(amp.dtype)], dim=-3)


def _zscore_l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = tuple(range(-3, 0))
    x = (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True).clamp_min(eps)
    norm = torch.linalg.vector_norm(x.flatten(-3), dim=-1, keepdim=True).clamp_min(eps)
    return x / norm[..., None, None]


def _mode_labels(zernike_indices: tuple[int, ...]) -> list[str]:
    n_order, m_order = zernike_index_to_nm(torch.as_tensor(zernike_indices, dtype=torch.int64))
    return [f"j={j} n={int(n)} m={int(m)}" for j, n, m in zip(zernike_indices, n_order, m_order, strict=True)]


def _load_model_head(
    config: dict,
    checkpoint_path: Path | None,
    device: torch.device,
):
    model = make_model(config["model"]).to(device)
    head = get_template_head(model)
    if checkpoint_path is None:
        return model, head, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint["_skipped_keys"] = []
    state = checkpoint.get("model", checkpoint)
    if "model" in checkpoint:
        checkpoint["_skipped_keys"].extend(_load_compatible_state_dict(model, state))
    elif "head" in checkpoint:
        checkpoint["_skipped_keys"].extend(_load_compatible_state_dict(head, checkpoint["head"]))
    else:
        checkpoint["_skipped_keys"].extend(_load_compatible_state_dict(model, state))
    if "head" in checkpoint:
        checkpoint["_skipped_keys"].extend(_load_compatible_state_dict(head, checkpoint["head"]))
    model.eval()
    return model, head, checkpoint


def _load_compatible_state_dict(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    current = module.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current:
            skipped.append(key)
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible[key] = value
    module.load_state_dict(compatible, strict=False)
    return skipped


def save_sample_overview(
    output_dir: Path,
    sample_name: str,
    object_image: np.ndarray | None,
    aberrated_image: np.ndarray,
    coefficients: np.ndarray,
    zernike_indices: tuple[int, ...],
    phase_data: dict[str, torch.Tensor],
) -> None:
    x_axis = np.arange(len(zernike_indices))
    fig, axes = plt.subplots(4, 3, figsize=(13, 15))
    axes = axes.reshape(4, 3)
    if object_image is not None:
        _show_image(axes[0, 0], object_image, "Original OBJ", cmap="gray")
    else:
        axes[0, 0].axis("off")
        axes[0, 0].set_title("Original OBJ not found", fontsize=10)
    _show_image(axes[0, 1], aberrated_image, "Aberrated input", cmap="gray")
    axes[0, 2].bar(x_axis, coefficients, color="#2563eb")
    axes[0, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 2].set_title("Zernike coefficients", fontsize=10)
    axes[0, 2].set_xlabel("Zernike index")
    axes[0, 2].set_ylabel("OPD (um)")
    axes[0, 2].set_xticks(x_axis)
    axes[0, 2].set_xticklabels([str(v) for v in zernike_indices], rotation=45, ha="right")
    axes[0, 2].grid(axis="y", linestyle="--", alpha=0.35)

    panels = [
        ("Wavefront OPD (um)", "wavefront", "coolwarm", True),
        ("Phase (rad)", "phase", "coolwarm", True),
        ("cos(phase)", "cos_phase", "coolwarm", True),
        ("Phase FFT log(1+|Y|)", "phase_fft_log_amp", "magma", False),
        ("Phase FFT cos(angle Y)", "phase_fft_cos", "coolwarm", True),
        ("Phase FFT sin(angle Y)", "phase_fft_sin", "coolwarm", True),
        ("PSF from coefficients", "psf", "magma", False),
        ("OTF log(1+|Y|)", "otf_log_amp", "magma", False),
        ("OTF cos(angle Y)", "otf_cos", "coolwarm", True),
    ]
    for ax, (title, key, cmap, symmetric) in zip(axes.flat[3:], panels, strict=True):
        _show_image(ax, _as_numpy(phase_data[key]), title, cmap=cmap, symmetric=symmetric)

    fig.suptitle(sample_name, y=0.997)
    fig.tight_layout()
    fig.savefig(output_dir / "sample_overview.png", dpi=180)
    plt.close(fig)


def save_input_fft_figure(output_dir: Path, image: torch.Tensor, fft_shift: bool) -> None:
    log_amp, cos_phase, sin_phase = _input_fft_features(image, fft_shift)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _show_image(axes[0], _as_numpy(log_amp[0, 0]), "Input FFT log(1+|Y|)", cmap="magma", symmetric=False)
    _show_image(axes[1], _as_numpy(cos_phase[0, 0]), "Input FFT cos(angle Y)", cmap="coolwarm", symmetric=True)
    _show_image(axes[2], _as_numpy(sin_phase[0, 0]), "Input FFT sin(angle Y)", cmap="coolwarm", symmetric=True)
    fig.tight_layout()
    fig.savefig(output_dir / "input_fft_features.png", dpi=180)
    plt.close(fig)


def save_attention_input_feature_figure(output_dir: Path, head, image: torch.Tensor) -> None:
    with torch.no_grad():
        features = head._input_features(image)[0].detach().cpu().numpy()
    titles = ["zscored log amp", "masked cos(angle)", "masked sin(angle)"]
    if features.shape[0] >= 5:
        titles.extend(["reliability mask", "frequency radius"])
    fig, axes = plt.subplots(1, features.shape[0], figsize=(4.0 * features.shape[0], 4), squeeze=False)
    for idx, ax in enumerate(axes[0]):
        _show_image(
            ax,
            features[idx],
            f"Attention input: {titles[idx] if idx < len(titles) else f'ch{idx}'}",
            cmap="magma" if idx in {0, 3, 4} else "coolwarm",
            symmetric=idx in {0, 1, 2},
        )
    fig.tight_layout()
    fig.savefig(output_dir / "attention_input_features.png", dpi=180)
    plt.close(fig)


def save_otf_target_figure(
    output_dir: Path,
    coefficients: torch.Tensor,
    image_size: tuple[int, int],
    zernike_indices: tuple[int, ...],
    config: dict,
    fft_shift: bool,
) -> None:
    optics = make_optics_config(config)
    target = target_otf_features(
        coefficients[None],
        image_size,
        zernike_indices,
        optics,
        fft_shift,
        1e-8,
    )[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Target OTF log amp", "Target OTF relative cos", "Target OTF relative sin"]
    for idx, ax in enumerate(axes):
        _show_image(ax, _as_numpy(target[idx]), titles[idx], cmap="coolwarm", symmetric=True)
    fig.tight_layout()
    fig.savefig(output_dir / "target_otf_features_normalized.png", dpi=180)
    plt.close(fig)


def save_mode_physics_atlas(
    output_dir: Path,
    image_size: tuple[int, int],
    zernike_indices: tuple[int, ...],
    config: dict,
    device: torch.device,
) -> None:
    optics = make_optics_config(config)
    template_cfg = config.get("model", {}).get("template_attention", {})
    epsilon_cfg = template_cfg.get("epsilon_um", 0.05)
    epsilon = torch.as_tensor(epsilon_cfg, dtype=torch.float32, device=device)
    if epsilon.ndim == 0:
        epsilon = epsilon.repeat(len(zernike_indices))
    if epsilon.numel() != len(zernike_indices):
        epsilon = torch.full((len(zernike_indices),), 0.05, dtype=torch.float32, device=device)

    rho, theta, pupil_mask = pupil_coordinates_2d(
        image_size,
        optics.pixel_size,
        optics.lambda_emission,
        optics.na,
        device=device,
        dtype=torch.float32,
    )
    zero = torch.zeros((1, len(zernike_indices)), device=device, dtype=torch.float32)
    psf0 = generate_psf2d_from_zernike(image_size, zernike_indices, zero, optics)[0]
    otf0 = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(psf0), s=image_size))

    rows = len(zernike_indices)
    fig, axes = plt.subplots(rows, 7, figsize=(19, 2.5 * rows), squeeze=False)
    labels = _mode_labels(zernike_indices)
    summary = []
    for mode_idx, label in enumerate(labels):
        coeff_plus = torch.zeros((1, len(zernike_indices)), device=device, dtype=torch.float32)
        coeff_minus = torch.zeros_like(coeff_plus)
        coeff_plus[0, mode_idx] = epsilon[mode_idx]
        coeff_minus[0, mode_idx] = -epsilon[mode_idx]

        mode_opd = zernike_wavefront(zernike_indices, coeff_plus, rho, theta)[0] / epsilon[mode_idx].clamp_min(1e-8)
        phase_plus = 2.0 * torch.pi / optics.lambda_emission * zernike_wavefront(zernike_indices, coeff_plus, rho, theta)[0]
        pupil = pupil_mask.to(phase_plus.dtype)
        psf_plus = generate_psf2d_from_zernike(image_size, zernike_indices, coeff_plus, optics)[0]
        psf_minus = generate_psf2d_from_zernike(image_size, zernike_indices, coeff_minus, optics)[0]
        otf_plus = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(psf_plus), s=image_size))
        otf_minus = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(psf_minus), s=image_size))

        log_amp_der = (torch.log1p(torch.abs(otf_plus)) - torch.log1p(torch.abs(otf_minus))) / (
            2.0 * epsilon[mode_idx].clamp_min(1e-8)
        )
        rel_plus = otf_plus * torch.conj(otf0)
        rel_minus = otf_minus * torch.conj(otf0)
        rel_cos_plus = rel_plus.real / torch.abs(rel_plus).clamp_min(1e-8)
        rel_cos_minus = rel_minus.real / torch.abs(rel_minus).clamp_min(1e-8)
        rel_sin_plus = rel_plus.imag / torch.abs(rel_plus).clamp_min(1e-8)
        rel_sin_minus = rel_minus.imag / torch.abs(rel_minus).clamp_min(1e-8)
        rel_cos_der = (rel_cos_plus - rel_cos_minus) / (2.0 * epsilon[mode_idx].clamp_min(1e-8))
        rel_sin_der = (rel_sin_plus - rel_sin_minus) / (2.0 * epsilon[mode_idx].clamp_min(1e-8))

        panels = [
            ("Zernike OPD / eps", mode_opd * pupil, "coolwarm", True),
            ("Phase +eps", phase_plus * pupil, "coolwarm", True),
            ("cos(phase)", torch.cos(phase_plus) * pupil, "coolwarm", True),
            ("PSF(+eps)-PSF0", psf_plus - psf0, "coolwarm", True),
            ("d log(1+|OTF|)/dc", log_amp_der, "coolwarm", True),
            ("d rel cos / dc", rel_cos_der, "coolwarm", True),
            ("d rel sin / dc", rel_sin_der, "coolwarm", True),
        ]
        for col, (title, values, cmap, symmetric) in enumerate(panels):
            ax = axes[mode_idx, col]
            _show_image(ax, _as_numpy(values), title if mode_idx == 0 else "", cmap=cmap, symmetric=symmetric)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)

        summary.append(
            {
                "zernike_index": zernike_indices[mode_idx],
                "epsilon_um": float(epsilon[mode_idx].detach().cpu()),
                "psf_delta_norm": float(torch.linalg.vector_norm((psf_plus - psf0).flatten()).detach().cpu()),
                "log_amp_derivative_norm": float(torch.linalg.vector_norm(log_amp_der.flatten()).detach().cpu()),
                "rel_cos_derivative_norm": float(torch.linalg.vector_norm(rel_cos_der.flatten()).detach().cpu()),
                "rel_sin_derivative_norm": float(torch.linalg.vector_norm(rel_sin_der.flatten()).detach().cpu()),
            }
        )

    fig.suptitle("Per-mode physics representations used to build OTF template attention", y=0.999)
    fig.tight_layout()
    fig.savefig(output_dir / "mode_physics_atlas.png", dpi=180)
    plt.close(fig)

    keys = summary[0].keys()
    np.savez_compressed(
        output_dir / "mode_physics_summary.npz",
        **{key: np.asarray([item[key] for item in summary]) for key in keys},
    )


def save_input_feature_factorization(
    output_dir: Path,
    object_image: np.ndarray | None,
    aberrated_image: np.ndarray,
    coefficients: torch.Tensor,
    image_size: tuple[int, int],
    zernike_indices: tuple[int, ...],
    config: dict,
    fft_shift: bool,
    device: torch.device,
) -> None:
    if object_image is None:
        return
    optics = make_optics_config(config)
    obj = torch.from_numpy(np.ascontiguousarray(object_image)).to(device=device, dtype=torch.float32)
    abe = torch.from_numpy(np.ascontiguousarray(aberrated_image)).to(device=device, dtype=torch.float32)
    psf = generate_psf2d_from_zernike(image_size, zernike_indices, coefficients[None].float().to(device), optics)[0]

    obj_fft = torch.fft.fft2(obj)
    abe_fft = torch.fft.fft2(abe)
    otf = torch.fft.fft2(torch.fft.ifftshift(psf), s=image_size)
    if fft_shift:
        obj_fft = torch.fft.fftshift(obj_fft)
        abe_fft = torch.fft.fftshift(abe_fft)
        otf = torch.fft.fftshift(otf)

    obj_feat = _fft_feature_stack_from_spectrum(obj_fft)
    abe_feat = _fft_feature_stack_from_spectrum(abe_fft)
    otf_feat = _fft_feature_stack_from_spectrum(otf)
    rel_obs = abe_fft * torch.conj(obj_fft)
    rel_obs_feat = _fft_feature_stack_from_spectrum(rel_obs)

    normalized = [_zscore_l2(x) for x in (obj_feat, abe_feat, otf_feat, rel_obs_feat)]
    names = ["OBJ FFT", "Aberrated FFT input", "Known OTF", "Aberrated * conj(OBJ)"]
    cos_to_otf = [float((x * normalized[2]).sum().detach().cpu()) for x in normalized]
    channel_titles = ["log(1+|Y|)", "cos(angle Y)", "sin(angle Y)"]

    fig, axes = plt.subplots(4, 3, figsize=(13, 14), squeeze=False)
    for row, (name, feat) in enumerate(zip(names, (obj_feat, abe_feat, otf_feat, rel_obs_feat), strict=True)):
        for col, channel_title in enumerate(channel_titles):
            _show_image(
                axes[row, col],
                _as_numpy(feat[col]),
                f"{name}: {channel_title}\ncos-to-known-OTF={cos_to_otf[row]:.3f}" if col == 0 else channel_title,
                cmap="magma" if col == 0 else "coolwarm",
                symmetric=col != 0,
            )
    fig.suptitle("Input feature factorization: FFT(input) mixes object structure with the aberration OTF", y=0.997)
    fig.tight_layout()
    fig.savefig(output_dir / "input_feature_factorization.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "input_feature_factorization.npz",
        cosine_to_known_otf=np.asarray(cos_to_otf),
        names=np.asarray(names),
    )


def save_per_mode_attention_input_effects(
    output_dir: Path,
    head,
    object_image: np.ndarray | None,
    image_size: tuple[int, int],
    zernike_indices: tuple[int, ...],
    config: dict,
    device: torch.device,
) -> None:
    if object_image is None:
        return

    optics = make_optics_config(config)
    template_cfg = config.get("model", {}).get("template_attention", {})
    epsilon_cfg = template_cfg.get("epsilon_um", 0.05)
    epsilon = torch.as_tensor(epsilon_cfg, dtype=torch.float32, device=device)
    if epsilon.ndim == 0:
        epsilon = epsilon.repeat(len(zernike_indices))
    if epsilon.numel() != len(zernike_indices):
        epsilon = torch.full((len(zernike_indices),), 0.05, dtype=torch.float32, device=device)

    obj = torch.from_numpy(np.ascontiguousarray(object_image)).to(device=device, dtype=torch.float32)[None, None]
    zero_coeff = torch.zeros((1, len(zernike_indices)), device=device, dtype=torch.float32)
    psf0 = generate_psf2d_from_zernike(image_size, zernike_indices, zero_coeff, optics)
    no_abe = convolve_fft2(obj, psf0)
    with torch.no_grad():
        no_features = head._input_features(no_abe)

    labels = _mode_labels(zernike_indices)
    channel_names = ["log amp", "masked cos", "masked sin", "reliability", "freq radius"]
    derivative_rows: list[list[np.ndarray]] = []
    plus_delta_rows: list[list[np.ndarray]] = []
    minus_delta_rows: list[list[np.ndarray]] = []
    summary = []
    for mode_idx, label in enumerate(labels):
        coeff_plus = torch.zeros((1, len(zernike_indices)), device=device, dtype=torch.float32)
        coeff_minus = torch.zeros_like(coeff_plus)
        coeff_plus[0, mode_idx] = epsilon[mode_idx]
        coeff_minus[0, mode_idx] = -epsilon[mode_idx]

        psf_plus = generate_psf2d_from_zernike(image_size, zernike_indices, coeff_plus, optics)
        psf_minus = generate_psf2d_from_zernike(image_size, zernike_indices, coeff_minus, optics)
        abe_plus = convolve_fft2(obj, psf_plus)
        abe_minus = convolve_fft2(obj, psf_minus)

        with torch.no_grad():
            plus_features = head._input_features(abe_plus)
            minus_features = head._input_features(abe_minus)
        signed_derivative = (plus_features - minus_features)[0] / (2.0 * epsilon[mode_idx].clamp_min(1e-8))
        plus_delta = (plus_features - no_features)[0]
        minus_delta = (minus_features - no_features)[0]
        n_channels = min(5, signed_derivative.shape[0])
        derivative_rows.append([_as_numpy(signed_derivative[channel]) for channel in range(n_channels)])
        plus_delta_rows.append([_as_numpy(plus_delta[channel]) for channel in range(n_channels)])
        minus_delta_rows.append([_as_numpy(minus_delta[channel]) for channel in range(n_channels)])
        summary.append(
            {
                "zernike_index": zernike_indices[mode_idx],
                "epsilon_um": float(epsilon[mode_idx].detach().cpu()),
                **{
                    f"{channel_names[channel].replace(' ', '_')}_derivative_norm": float(
                        torch.linalg.vector_norm(signed_derivative[channel].flatten()).detach().cpu()
                    )
                    for channel in range(min(5, signed_derivative.shape[0]))
                },
                **{
                    f"{channel_names[channel].replace(' ', '_')}_plus_delta_norm": float(
                        torch.linalg.vector_norm(plus_delta[channel].flatten()).detach().cpu()
                    )
                    for channel in range(min(5, plus_delta.shape[0]))
                },
                **{
                    f"{channel_names[channel].replace(' ', '_')}_minus_delta_norm": float(
                        torch.linalg.vector_norm(minus_delta[channel].flatten()).detach().cpu()
                    )
                    for channel in range(min(5, minus_delta.shape[0]))
                },
            }
        )

    ncols = min(5, len(channel_names))
    for rows, filename, title in (
        (
            derivative_rows,
            "mode_attention_input_effects.png",
            "Per-mode effect on the actual 5-channel attention input: d feature / d coefficient",
        ),
        (
            plus_delta_rows,
            "mode_attention_input_plus_delta.png",
            "Per-mode +epsilon change in the actual 5-channel attention input: feature(+eps) - feature(0)",
        ),
        (
            minus_delta_rows,
            "mode_attention_input_minus_delta.png",
            "Per-mode -epsilon change in the actual 5-channel attention input: feature(-eps) - feature(0)",
        ),
    ):
        fig, axes = plt.subplots(len(labels), ncols, figsize=(3.2 * ncols, 2.35 * len(labels)), squeeze=False)
        for row_idx, label in enumerate(labels):
            for col_idx in range(ncols):
                ax = axes[row_idx, col_idx]
                _show_image(
                    ax,
                    rows[row_idx][col_idx],
                    channel_names[col_idx] if row_idx == 0 else "",
                    cmap="coolwarm",
                    symmetric=True,
                )
                if col_idx == 0:
                    ax.set_ylabel(label, fontsize=8)
        fig.suptitle(title, y=0.999)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    keys = summary[0].keys()
    np.savez_compressed(
        output_dir / "mode_attention_input_effects_summary.npz",
        **{key: np.asarray([item[key] for item in summary]) for key in keys},
    )

    norm_keys = [key for key in keys if key.endswith("_derivative_norm")]
    plus_norm_keys = [key for key in keys if key.endswith("_plus_delta_norm")]
    minus_norm_keys = [key for key in keys if key.endswith("_minus_delta_norm")]
    for selected_keys, filename, title, ylabel, suffix in (
        (
            norm_keys,
            "mode_attention_input_effect_norms.png",
            "Per-mode derivative norm on each attention input channel",
            "||d feature / d coefficient||",
            "_derivative_norm",
        ),
        (
            plus_norm_keys,
            "mode_attention_input_plus_delta_norms.png",
            "Per-mode +epsilon delta norm on each attention input channel",
            "||feature(+eps) - feature(0)||",
            "_plus_delta_norm",
        ),
        (
            minus_norm_keys,
            "mode_attention_input_minus_delta_norms.png",
            "Per-mode -epsilon delta norm on each attention input channel",
            "||feature(-eps) - feature(0)||",
            "_minus_delta_norm",
        ),
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        x_axis = np.arange(len(zernike_indices))
        width = 0.8 / max(1, len(selected_keys))
        for idx, key in enumerate(selected_keys):
            values = np.asarray([item[key] for item in summary], dtype=float)
            ax.bar(
                x_axis + (idx - (len(selected_keys) - 1) / 2) * width,
                values,
                width=width,
                label=key.replace(suffix, ""),
            )
        ax.set_title(title)
        ax.set_xlabel("Zernike index")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_axis)
        ax.set_xticklabels([str(v) for v in zernike_indices])
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def save_template_bank_figures(output_dir: Path, head, image: torch.Tensor, zernike_indices: tuple[int, ...]) -> None:
    head.template_bank.ensure(image)
    templates = head.template_bank.templates.detach().cpu()
    derivative = head.template_bank.derivative_templates.detach().cpu()
    labels = _mode_labels(zernike_indices)

    for channel, name in enumerate(("log_amp", "relative_cos", "relative_sin")):
        _save_grid(
            output_dir / f"template_positive_{name}.png",
            [templates[k, 0, channel].numpy() for k in range(templates.shape[0])],
            labels,
            ncols=min(5, templates.shape[0]),
            cmap="coolwarm",
            symmetric=True,
        )
        _save_grid(
            output_dir / f"template_derivative_{name}.png",
            [derivative[k, channel].numpy() for k in range(derivative.shape[0])],
            labels,
            ncols=min(5, derivative.shape[0]),
            cmap="coolwarm",
            symmetric=True,
        )


def save_template_diagnostics(output_dir: Path, head, image: torch.Tensor, zernike_indices: tuple[int, ...]) -> None:
    head.template_bank.ensure(image)
    templates = head.template_bank.templates.detach().cpu()
    derivative = head.template_bank.derivative_templates.detach().cpu()
    short_labels = [str(v) for v in zernike_indices]

    derivative_flat = torch.nn.functional.normalize(derivative.flatten(1), dim=1)
    derivative_similarity = derivative_flat @ derivative_flat.T
    signed_delta = templates[:, 0] - templates[:, 1]
    signed_delta_flat = torch.nn.functional.normalize(signed_delta.flatten(1), dim=1)
    signed_delta_similarity = signed_delta_flat @ signed_delta_flat.T
    pos_flat = torch.nn.functional.normalize(templates[:, 0].flatten(1), dim=1)
    neg_flat = torch.nn.functional.normalize(templates[:, 1].flatten(1), dim=1)
    pos_neg_cos = (pos_flat * neg_flat).sum(dim=1)
    signed_energy = torch.linalg.vector_norm(signed_delta.flatten(1), dim=1)
    template_energy = torch.linalg.vector_norm(templates.flatten(2), dim=2).mean(dim=1).clamp_min(1e-8)
    signed_fraction = signed_energy / template_energy

    _save_heatmap(
        output_dir / "template_derivative_mode_similarity.png",
        derivative_similarity.numpy(),
        "Derivative template cosine similarity",
        short_labels,
        short_labels,
    )
    _save_heatmap(
        output_dir / "template_signed_delta_mode_similarity.png",
        signed_delta_similarity.numpy(),
        "Positive-negative delta cosine similarity",
        short_labels,
        short_labels,
    )

    x_axis = np.arange(len(zernike_indices))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].bar(x_axis, pos_neg_cos.numpy(), color="#64748b")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axhline(0.95, color="#dc2626", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("cos(+eps, -eps)")
    axes[0].set_title("Sign separability: values near 1 mean positive/negative templates look almost identical")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(x_axis, signed_fraction.numpy(), color="#2563eb")
    axes[1].set_ylabel("||+ - -|| / mean ||template||")
    axes[1].set_xlabel("Zernike index")
    axes[1].set_xticks(x_axis)
    axes[1].set_xticklabels([str(v) for v in zernike_indices])
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "template_sign_separability.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "template_diagnostics.npz",
        zernike_indices=np.asarray(zernike_indices),
        derivative_similarity=derivative_similarity.numpy(),
        signed_delta_similarity=signed_delta_similarity.numpy(),
        positive_negative_cosine=pos_neg_cos.numpy(),
        signed_fraction=signed_fraction.numpy(),
    )


def save_attention_score_summary(
    output_dir: Path,
    coefficients: np.ndarray,
    zernike_indices: tuple[int, ...],
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    der_scores: np.ndarray,
    prediction: np.ndarray | None,
    feature_label: str,
) -> None:
    x_axis = np.arange(len(zernike_indices))
    signed_scores = pos_scores - neg_scores
    coeff_scale = max(float(np.max(np.abs(coefficients))), 1e-8)
    score_scale = max(float(np.max(np.abs(der_scores))), float(np.max(np.abs(signed_scores))), 1e-8)
    norm_coeff = coefficients / coeff_scale
    norm_der = der_scores / score_scale
    norm_signed = signed_scores / score_scale

    rows = 4 if prediction is not None else 3
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3.0 * rows), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].bar(x_axis, coefficients, color="#111827", label="ground truth")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("coeff (um)")
    axes[0].set_title("Ground-truth Zernike coefficients")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    if prediction is not None:
        axes[1].bar(x_axis - 0.18, coefficients, width=0.36, color="#111827", label="ground truth")
        axes[1].bar(x_axis + 0.18, prediction, width=0.36, color="#2563eb", label="prediction")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel("coeff (um)")
        axes[1].set_title("Checkpoint prediction")
        axes[1].legend(loc="upper right")
        axes[1].grid(axis="y", linestyle="--", alpha=0.35)
        score_ax = axes[2]
        compare_ax = axes[3]
    else:
        score_ax = axes[1]
        compare_ax = axes[2]

    width = 0.26
    score_ax.bar(x_axis - width, pos_scores, width=width, color="#64748b", label="positive template")
    score_ax.bar(x_axis, neg_scores, width=width, color="#94a3b8", label="negative template")
    score_ax.bar(x_axis + width, der_scores, width=width, color="#dc2626", label="derivative")
    score_ax.axhline(0.0, color="black", linewidth=0.8)
    score_ax.set_ylabel("score")
    score_ax.set_title(f"Mode scores from {feature_label}; flat bars mean weak mode selectivity")
    score_ax.legend(loc="upper right", ncol=3)
    score_ax.grid(axis="y", linestyle="--", alpha=0.35)

    compare_ax.plot(x_axis, norm_coeff, marker="o", color="#111827", label="target coeff normalized")
    compare_ax.plot(x_axis, norm_der, marker="s", color="#dc2626", label="derivative score normalized")
    compare_ax.plot(x_axis, norm_signed, marker="^", color="#2563eb", label="pos-neg score normalized")
    compare_ax.axhline(0.0, color="black", linewidth=0.8)
    compare_ax.set_ylabel("normalized")
    compare_ax.set_xlabel("Zernike index")
    compare_ax.set_title("Useful attention should peak on the same modes as the target coefficients")
    compare_ax.set_xticks(x_axis)
    compare_ax.set_xticklabels([str(v) for v in zernike_indices])
    compare_ax.legend(loc="upper right")
    compare_ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "attention_score_summary.png", dpi=180)
    plt.close(fig)


def save_mode_correlation_figures(
    output_dir: Path,
    head,
    image: torch.Tensor,
    coefficients: torch.Tensor,
    zernike_indices: tuple[int, ...],
    config: dict,
    checkpoint_loaded: bool,
) -> None:
    head.eval()
    with torch.no_grad():
        if checkpoint_loaded:
            features = head.encode_template_features(image)
            feature_label = "encoded"
        else:
            feature_label = "target_otf"
            features = target_otf_features(
                coefficients[None].to(image.device),
                tuple(int(v) for v in image.shape[-2:]),
                zernike_indices,
                make_optics_config(config),
                bool(getattr(head, "fft_shift", False)),
                float(getattr(head, "eps", 1e-8)),
            ).to(device=image.device, dtype=image.dtype)
        templates = head.template_bank.ensure(image).to(device=image.device, dtype=features.dtype)
        derivative = head.template_bank.derivative_templates.to(device=image.device, dtype=features.dtype)
        pos_maps = (features[:, None] * templates[None, :, 0]).sum(dim=2)[0]
        neg_maps = (features[:, None] * templates[None, :, 1]).sum(dim=2)[0]
        der_maps = (features[:, None] * derivative[None]).sum(dim=2)[0]
        pos_scores = pos_maps.flatten(1).sum(dim=1)
        neg_scores = neg_maps.flatten(1).sum(dim=1)
        der_scores = der_maps.flatten(1).sum(dim=1)
        pred = scores = presence = sign = None
        if checkpoint_loaded:
            pred, scores, presence, sign = head._template_coefficients(image)

    labels = _mode_labels(zernike_indices)
    labels = [
        f"{label}\npos={float(pos_scores[k]):.3f} neg={float(neg_scores[k]):.3f} d={float(der_scores[k]):.3f}"
        for k, label in enumerate(labels)
    ]
    ncols = min(5, len(zernike_indices))
    _save_grid(
        output_dir / f"mode_correlation_{feature_label}_positive.png",
        [pos_maps[k].detach().cpu().numpy() for k in range(pos_maps.shape[0])],
        labels,
        ncols=ncols,
        cmap="coolwarm",
        symmetric=True,
    )
    _save_grid(
        output_dir / f"mode_correlation_{feature_label}_negative.png",
        [neg_maps[k].detach().cpu().numpy() for k in range(neg_maps.shape[0])],
        labels,
        ncols=ncols,
        cmap="coolwarm",
        symmetric=True,
    )
    _save_grid(
        output_dir / f"mode_correlation_{feature_label}_derivative.png",
        [der_maps[k].detach().cpu().numpy() for k in range(der_maps.shape[0])],
        labels,
        ncols=ncols,
        cmap="coolwarm",
        symmetric=True,
    )
    coeff_np = _as_numpy(coefficients)
    pos_np = _as_numpy(pos_scores)
    neg_np = _as_numpy(neg_scores)
    der_np = _as_numpy(der_scores)
    pred_np = None if pred is None else _as_numpy(pred[0])
    save_attention_score_summary(output_dir, coeff_np, zernike_indices, pos_np, neg_np, der_np, pred_np, feature_label)

    payload = {
        "zernike_indices": np.asarray(zernike_indices),
        "coefficients": coeff_np,
        "positive_map_scores": pos_np,
        "negative_map_scores": neg_np,
        "derivative_map_scores": der_np,
        "feature_label": np.asarray(feature_label),
    }
    if pred is not None and scores is not None and presence is not None and sign is not None:
        payload.update(
            prediction=pred_np,
            scores=_as_numpy(scores[0]),
            presence=_as_numpy(presence[0]),
            sign=_as_numpy(sign[0]),
        )
    np.savez_compressed(output_dir / "attention_mode_scores.npz", **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize OTF template-attention Zernike inputs and correlations.")
    parser.add_argument("-c", "--config", required=True, help="Template-attention pretrain config JSON.")
    parser.add_argument("--checkpoint", default=None, help="Optional trained checkpoint, e.g. best.pt or last.pt.")
    parser.add_argument("--data-root", default=None, help="Override data root from config.")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="Dataset split to sample.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index in the selected split.")
    parser.add_argument("--output", default=None, help="Output directory for visualization PNG/NPZ files.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    config["model"] = prepare_model_config(config)
    data_root = Path(args.data_root or config.get("data", {}).get("root", ".")).expanduser()
    dataset = make_dataset(config, args.split, data_root)
    sample = dataset[int(args.sample_index) % len(dataset)]
    image = sample["input"][None]
    coefficients = sample["coeff"].float()
    image_size = tuple(int(v) for v in image.shape[-2:])
    zernike_indices = tuple(int(v) for v in config["model"].get("zernike_indices", list(range(3, 16))))

    output_dir = Path(args.output or Path(config.get("output_dir", "outputs")) / "template_attention_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    _, head, checkpoint = _load_model_head(
        config,
        None if args.checkpoint is None else Path(args.checkpoint).expanduser(),
        device,
    )
    skipped_keys = [] if checkpoint is None else list(checkpoint.get("_skipped_keys", []))
    encoder_checkpoint_loaded = checkpoint is not None and not any("encoder" in key for key in skipped_keys)
    image = image.to(device)
    coefficients = coefficients.to(device)

    input_path = Path(str(sample["input_path"]))
    object_path = _find_original_object(input_path, config, data_root, args.split)
    object_image = None
    if object_path is not None:
        object_image = _crop_like_sample(_load_scaled_image(object_path, config), image_size)
    aberrated_image = _as_numpy(sample["input"][0])
    phase_data = _phase_products(coefficients, image_size, zernike_indices, config, device)

    save_sample_overview(
        output_dir,
        input_path.name,
        object_image,
        aberrated_image,
        _as_numpy(coefficients),
        zernike_indices,
        phase_data,
    )
    save_input_fft_figure(output_dir, image, bool(getattr(head, "fft_shift", False)))
    save_attention_input_feature_figure(output_dir, head, image)
    save_otf_target_figure(
        output_dir,
        coefficients,
        image_size,
        zernike_indices,
        config,
        bool(getattr(head, "fft_shift", False)),
    )
    save_mode_physics_atlas(output_dir, image_size, zernike_indices, config, device)
    save_input_feature_factorization(
        output_dir,
        object_image,
        aberrated_image,
        coefficients,
        image_size,
        zernike_indices,
        config,
        bool(getattr(head, "fft_shift", False)),
        device,
    )
    save_per_mode_attention_input_effects(output_dir, head, object_image, image_size, zernike_indices, config, device)
    save_template_bank_figures(output_dir, head, image, zernike_indices)
    save_template_diagnostics(output_dir, head, image, zernike_indices)
    save_mode_correlation_figures(
        output_dir,
        head,
        image,
        coefficients,
        zernike_indices,
        config,
        checkpoint_loaded=encoder_checkpoint_loaded,
    )

    metadata = {
        "input_path": str(input_path),
        "coeff_path": str(sample["coeff_path"]),
        "object_path": None if object_path is None else str(object_path),
        "checkpoint": args.checkpoint,
        "encoder_checkpoint_loaded": encoder_checkpoint_loaded,
        "skipped_checkpoint_keys": skipped_keys,
        "zernike_indices": list(zernike_indices),
        "output_dir": str(output_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved template-attention visualizations to {output_dir}")


if __name__ == "__main__":
    main()
