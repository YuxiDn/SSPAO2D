#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao2d.data.dataset import _center_crop_single, _estimate_scale_from_input, _random_crop_single, _to_tensor
from ao2d.data.io import IMAGE_EXTENSIONS, load_image
from ao2d.data.paths import resolve_path
from ao2d.models.factory import make_model
from ao2d.optics import AO2DConfig, generate_psf2d_from_zernike, random_zernike_coefficients
from ao2d.training.forward_model import AO2DForwardModel


def read_config(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return json.load(f)


def valid_image_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))


def load_zernike_coefficients(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            for key in ("coefficients_um", "coefficients", "zernike_coefficients"):
                if key in data.files:
                    return np.asarray(data[key], dtype=np.float32).reshape(-1)
        raise ValueError(f"No coefficient array found in {path}")
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if path.suffix.lower() == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise RuntimeError("scipy is required to load .mat Zernike labels") from exc
        data = loadmat(path)
        for key in ("coefficients", "coefficients_um", "zernike_coefficients"):
            if key in data:
                return np.asarray(data[key], dtype=np.float32).reshape(-1)
        raise ValueError(f"No coefficient array found in {path}")
    raise ValueError(f"Unsupported Zernike label format: {path}")


def matched_zernike_path(input_path: Path, zernike_dir: Path) -> Path | None:
    for suffix in (".mat", ".npz", ".npy"):
        path = zernike_dir / f"{input_path.stem}_zernike{suffix}"
        if path.exists():
            return path
    return None


@dataclass(frozen=True)
class ZernikeRecord:
    image: Path
    coeff: Path
    object_id: str


class TemplateAttentionPretrainDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        zernike_dir: str | Path,
        patch_size: tuple[int, int] | None = None,
        crop_mode: str = "random",
        samples_per_epoch: int | None = None,
        input_scale_method: str = "percentile",
        input_scale_percentile: float = 99.9,
    ) -> None:
        image_dir = Path(image_dir)
        zernike_dir = Path(zernike_dir)
        records: list[ZernikeRecord] = []
        for image_path in valid_image_files(image_dir):
            coeff_path = matched_zernike_path(image_path, zernike_dir)
            if coeff_path is not None:
                match = re.search(r"(obj_\d+)", image_path.stem)
                object_id = match.group(1) if match is not None else image_path.stem
                records.append(ZernikeRecord(image_path, coeff_path, object_id))
        if not records:
            raise ValueError(f"No image/Zernike pairs found in {image_dir} and {zernike_dir}")
        crop_mode = crop_mode.lower()
        if crop_mode not in {"random", "center", "none"}:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")
        self.records = records
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        self.samples_per_epoch = samples_per_epoch
        self.input_scale_method = input_scale_method
        self.input_scale_percentile = float(input_scale_percentile)

    def __len__(self) -> int:
        return int(self.samples_per_epoch or len(self.records))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index % len(self.records)]
        raw = load_image(record.image).astype(np.float32, copy=False)
        scale = _estimate_scale_from_input(raw, method=self.input_scale_method, percentile=self.input_scale_percentile)
        image = np.maximum(raw, 0) / scale
        if self.crop_mode == "center":
            image = _center_crop_single(image, self.patch_size)
        elif self.crop_mode == "random":
            image = _random_crop_single(image, self.patch_size)
        coeff = load_zernike_coefficients(record.coeff)
        return {
            "input": _to_tensor(image),
            "coeff": torch.from_numpy(np.ascontiguousarray(coeff)).float(),
            "object_id": record.object_id,
            "input_path": str(record.image),
            "coeff_path": str(record.coeff),
        }


class CleanObjectDataset(Dataset):
    def __init__(
        self,
        object_dir: str | Path,
        patch_size: tuple[int, int] | None = None,
        crop_mode: str = "random",
        samples_per_epoch: int | None = None,
        input_scale_method: str = "percentile",
        input_scale_percentile: float = 99.9,
    ) -> None:
        self.files = valid_image_files(object_dir)
        if not self.files:
            raise ValueError(f"No object images found in {object_dir}")
        crop_mode = crop_mode.lower()
        if crop_mode not in {"random", "center", "none"}:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        self.samples_per_epoch = samples_per_epoch
        self.input_scale_method = input_scale_method
        self.input_scale_percentile = float(input_scale_percentile)

    def __len__(self) -> int:
        return int(self.samples_per_epoch or len(self.files))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index % len(self.files)]
        raw = load_image(path).astype(np.float32, copy=False)
        scale = _estimate_scale_from_input(raw, method=self.input_scale_method, percentile=self.input_scale_percentile)
        image = np.maximum(raw, 0) / scale
        if self.crop_mode == "center":
            image = _center_crop_single(image, self.patch_size)
        elif self.crop_mode == "random":
            image = _random_crop_single(image, self.patch_size)
        match = re.search(r"(obj_\d+)", path.stem)
        object_id = match.group(1) if match is not None else path.stem
        return {
            "object": _to_tensor(image),
            "object_id": object_id,
            "object_path": str(path),
        }


class SameObjectBatchSampler(BatchSampler):
    """Yield batches containing different aberrations of one object id."""

    def __init__(
        self,
        dataset: TemplateAttentionPretrainDataset,
        batch_size: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive")
        groups: dict[str, list[int]] = {}
        for idx, record in enumerate(dataset.records):
            groups.setdefault(record.object_id, []).append(idx)
        if not groups:
            raise ValueError("SameObjectBatchSampler received no object groups")
        self.groups = groups
        self.object_ids = sorted(groups)
        self.batch_size = int(batch_size)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.batches_per_epoch):
            object_id = rng.choice(self.object_ids)
            indices = self.groups[object_id]
            if len(indices) >= self.batch_size:
                yield rng.sample(indices, self.batch_size)
            else:
                yield [rng.choice(indices) for _ in range(self.batch_size)]

    def __len__(self) -> int:
        return self.batches_per_epoch


def prepare_model_config(config: dict) -> dict:
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("aberration_head_type", "template_attention")
    model_cfg.setdefault("template_image_size", config.get("data", {}).get("patch_size", [256, 256]))
    model_cfg.setdefault("template_optics", config.get("optics", {}))
    if "zernike_indices" not in model_cfg and "zernike_indices" in config.get("optics", {}):
        model_cfg["zernike_indices"] = config["optics"]["zernike_indices"]
        model_cfg["zernike_modes"] = len(model_cfg["zernike_indices"])
    return model_cfg


def get_template_head(model):
    head = getattr(model, "zernike_head", None)
    if head is None:
        head = getattr(model, "aberration_head", None)
    if head is None or not hasattr(head, "_template_coefficients"):
        raise TypeError("Model must use aberration_head_type='template_attention'")
    return head


def set_trainable_template_params(head, train_thresholds: bool = True) -> list[torch.nn.Parameter]:
    for param in head.parameters():
        param.requires_grad = False
    for param in head.encoder.parameters():
        param.requires_grad = True
    amplitude_head = getattr(head, "amplitude_head", None)
    if amplitude_head is not None:
        for param in amplitude_head.parameters():
            param.requires_grad = True
    if train_thresholds:
        for name in ("raw_eta", "raw_alpha", "raw_response_scale", "tau"):
            param = getattr(head, name)
            param.requires_grad = True
    return [param for param in head.parameters() if param.requires_grad]


def estimate_max_amp(dataset: TemplateAttentionPretrainDataset, quantile: float, scale: float) -> torch.Tensor:
    coeffs = [load_zernike_coefficients(record.coeff) for record in dataset.records]
    stacked = np.stack(coeffs, axis=0)
    return torch.from_numpy((np.quantile(np.abs(stacked), quantile, axis=0) * scale).astype(np.float32))


def coefficient_metrics(pred: torch.Tensor, target: torch.Tensor, presence: torch.Tensor, threshold: float) -> dict[str, float]:
    error = pred - target
    mae_per_mode = error.abs().mean(dim=0)
    active = target.abs() > threshold
    if active.any():
        sign_acc = ((torch.sign(pred[active]) == torch.sign(target[active])).float().mean()).item()
    else:
        sign_acc = float("nan")
    pred_active = presence > 0.5
    presence_acc = (pred_active == active).float().mean().item()
    return {
        "mae": float(error.abs().mean().detach()),
        "rmse": float(torch.sqrt(error.square().mean()).detach()),
        "max_mode_mae": float(mae_per_mode.max().detach()),
        "sign_acc": float(sign_acc),
        "presence_acc": float(presence_acc),
        "presence_mean": float(presence.mean().detach()),
    }


def normalize_otf_feature_target(x: torch.Tensor, eps: float) -> torch.Tensor:
    dims = tuple(range(-3, 0))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True).clamp_min(eps)
    x = (x - mean) / std
    norm = torch.linalg.vector_norm(x.flatten(-3), dim=-1, keepdim=True).clamp_min(eps)
    return x / norm[..., None, None]


def target_otf_features(
    coefficients: torch.Tensor,
    image_size: tuple[int, int],
    zernike_indices: tuple[int, ...],
    optics_config: AO2DConfig,
    fft_shift: bool,
    eps: float,
    otf_mtf_threshold: float = 0.03,
) -> torch.Tensor:
    with torch.no_grad():
        zero = torch.zeros_like(coefficients)
        psf = generate_psf2d_from_zernike(image_size, zernike_indices, coefficients, optics_config)
        psf0 = generate_psf2d_from_zernike(image_size, zernike_indices, zero, optics_config)
        otf = torch.fft.fft2(torch.fft.ifftshift(psf, dim=(-2, -1)), s=image_size)
        otf0 = torch.fft.fft2(torch.fft.ifftshift(psf0, dim=(-2, -1)), s=image_size)
        if fft_shift:
            otf = torch.fft.fftshift(otf, dim=(-2, -1))
            otf0 = torch.fft.fftshift(otf0, dim=(-2, -1))
        log_amp = torch.log1p(torch.abs(otf)) - torch.log1p(torch.abs(otf0))
        relative_phase = otf * torch.conj(otf0)
        phase_den = torch.abs(relative_phase).clamp_min(eps)
        mtf_support = (
            (torch.abs(otf) > float(otf_mtf_threshold))
            & (torch.abs(otf0) > float(otf_mtf_threshold))
        ).to(dtype=coefficients.dtype)
        phase_cos = relative_phase.real / phase_den * mtf_support
        phase_sin = relative_phase.imag / phase_den * mtf_support
        target = torch.stack([log_amp, phase_cos, phase_sin], dim=1).real.to(dtype=coefficients.dtype)
        return normalize_otf_feature_target(target, eps)


def make_optics_config(config: dict) -> AO2DConfig:
    optics = config.get("optics", {})
    return AO2DConfig(
        pixel_size=float(optics.get("pixel_size", 0.300)),
        na=float(optics.get("na", 1.05)),
        lambda_emission=float(optics.get("lambda_emission", optics.get("wavelength", 1.000))),
        lambda_excitation=float(optics.get("lambda_excitation", 0.808)),
        mode=str(optics.get("mode", "widefield")),
        pinhole_au=float(optics.get("pinhole_au", 1.0)),
        lightsheet_fwhm=float(optics.get("lightsheet_fwhm", 1.2)),
    )


def random_coefficients_batch(batch_size: int, zernike_indices: tuple[int, ...], device, config: dict) -> torch.Tensor:
    training = config.get("training", {})
    rms_min = float(training.get("zernike_rms_min", 0.02))
    rms_max = float(training.get("zernike_rms_max", 0.30))
    wavelength = float(config.get("optics", {}).get("lambda_emission", config.get("optics", {}).get("wavelength", 1.0)))
    spectrum = str(training.get("zernike_spectrum", "loworder"))
    include_defocus = bool(training.get("zernike_include_defocus", True))
    coeffs = []
    for _ in range(batch_size):
        rms = float(torch.empty((), device=device).uniform_(rms_min, rms_max).item())
        coeffs.append(
            random_zernike_coefficients(
                zernike_indices,
                rms_waves=rms,
                wavelength=wavelength,
                spectrum=spectrum,
                include_defocus=include_defocus,
                device=device,
            )
        )
    return torch.stack(coeffs, dim=0)


def run_epoch(head, loader, optimizer, device, train: bool, config: dict) -> dict[str, float]:
    head.train(train)
    weights = config.get("loss", {})
    coeff_l1_weight = float(weights.get("coeff_l1_weight", 1.0))
    coeff_mse_weight = float(weights.get("coeff_mse_weight", 0.0))
    presence_weight = float(weights.get("presence_bce_weight", 0.0))
    presence_threshold = float(weights.get("presence_threshold_um", 0.02))
    totals: dict[str, float] = {}
    total_steps = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
            image = batch["input"].to(device, non_blocking=True)
            target = batch["coeff"].to(device, non_blocking=True)
            pred, _, presence, _ = head._template_coefficients(image)
            loss = coeff_l1_weight * F.l1_loss(pred, target)
            if coeff_mse_weight > 0:
                loss = loss + coeff_mse_weight * F.mse_loss(pred, target)
            if presence_weight > 0:
                presence_target = (target.abs() > presence_threshold).to(dtype=presence.dtype)
                loss = loss + presence_weight * F.binary_cross_entropy(presence, presence_target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad], float(config.get("grad_clip", 1.0)))
                optimizer.step()
            metrics = coefficient_metrics(pred.detach(), target.detach(), presence.detach(), presence_threshold)
            metrics["loss"] = float(loss.detach())
            metrics["eta"] = float(torch.nn.functional.softplus(head.raw_eta).detach())
            metrics["alpha"] = float(torch.nn.functional.softplus(head.raw_alpha).detach())
            metrics["response_scale_mean"] = float(torch.nn.functional.softplus(head.raw_response_scale).mean().detach())
            metrics["tau_mean"] = float(head.tau.mean().detach())
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            total_steps += 1
    return {key: value / max(1, total_steps) for key, value in totals.items()}


def run_synthetic_same_object_epoch(
    head,
    loader,
    forward_model: AO2DForwardModel,
    optimizer,
    device,
    train: bool,
    config: dict,
    coeff_batch_size: int,
) -> dict[str, float]:
    head.train(train)
    weights = config.get("loss", {})
    coeff_l1_weight = float(weights.get("coeff_l1_weight", 1.0))
    coeff_mse_weight = float(weights.get("coeff_mse_weight", 0.0))
    presence_weight = float(weights.get("presence_bce_weight", 0.0))
    otf_feature_weight = float(weights.get("otf_feature_l1_weight", 0.0))
    otf_feature_cosine_weight = float(weights.get("otf_feature_cosine_weight", 0.0))
    presence_threshold = float(weights.get("presence_threshold_um", 0.02))
    zernike_indices = tuple(int(v) for v in config["model"].get("zernike_indices", list(range(3, 16))))
    use_reference_features = bool(config.get("training", {}).get("use_object_reference_features", False))
    optics_config = make_optics_config(config)
    image_size = tuple(int(v) for v in forward_model.image_size)
    totals: dict[str, float] = {}
    total_steps = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
            obj = batch["object"].to(device, non_blocking=True)
            if obj.shape[0] != 1:
                obj = obj[:1]
            target = random_coefficients_batch(coeff_batch_size, zernike_indices, device, config).to(dtype=obj.dtype)
            obj_batch = obj.expand(coeff_batch_size, -1, -1, -1).contiguous()
            with torch.no_grad():
                aberrated = forward_model(obj_batch, target).detach()
                reference = None
                if use_reference_features:
                    zero_target = torch.zeros_like(target)
                    reference = forward_model(obj_batch, zero_target).detach()
            pred, _, presence, _, encoded = head._template_coefficients(
                aberrated,
                reference_image=reference,
                return_features=True,
            )
            loss = coeff_l1_weight * F.l1_loss(pred, target)
            if coeff_mse_weight > 0:
                loss = loss + coeff_mse_weight * F.mse_loss(pred, target)
            if presence_weight > 0:
                presence_target = (target.abs() > presence_threshold).to(dtype=presence.dtype)
                loss = loss + presence_weight * F.binary_cross_entropy(presence, presence_target)
            otf_loss = torch.zeros((), device=device, dtype=loss.dtype)
            if otf_feature_weight > 0 or otf_feature_cosine_weight > 0:
                target_features = target_otf_features(
                    target,
                    image_size,
                    zernike_indices,
                    optics_config,
                    bool(getattr(head, "fft_shift", False)),
                    float(getattr(head, "eps", 1e-8)),
                    float(getattr(head, "otf_mtf_threshold", 0.03)),
                ).to(dtype=encoded.dtype)
                if otf_feature_weight > 0:
                    otf_loss = otf_loss + otf_feature_weight * F.l1_loss(encoded, target_features)
                if otf_feature_cosine_weight > 0:
                    cosine = (encoded * target_features).flatten(1).sum(dim=1)
                    otf_loss = otf_loss + otf_feature_cosine_weight * (1.0 - cosine).mean()
                loss = loss + otf_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in head.parameters() if p.requires_grad],
                    float(config.get("grad_clip", 1.0)),
                )
                optimizer.step()
            metrics = coefficient_metrics(pred.detach(), target.detach(), presence.detach(), presence_threshold)
            metrics["loss"] = float(loss.detach())
            metrics["eta"] = float(torch.nn.functional.softplus(head.raw_eta).detach())
            metrics["alpha"] = float(torch.nn.functional.softplus(head.raw_alpha).detach())
            metrics["response_scale_mean"] = float(torch.nn.functional.softplus(head.raw_response_scale).mean().detach())
            metrics["tau_mean"] = float(head.tau.mean().detach())
            metrics["otf_loss"] = float(otf_loss.detach())
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            total_steps += 1
    return {key: value / max(1, total_steps) for key, value in totals.items()}


def save_validation_coefficient_figures(
    head,
    dataset: TemplateAttentionPretrainDataset,
    output_dir: Path,
    device,
    epoch: int,
    config: dict,
) -> None:
    vis_cfg = config.get("visualization", {})
    count = int(vis_cfg.get("random_val_samples", 4))
    interval = int(vis_cfg.get("interval", 1))
    if count <= 0 or interval <= 0 or epoch % interval != 0:
        return

    save_dir = output_dir / "validation_coefficients"
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(vis_cfg.get("seed", 20260602)) + epoch)
    indices = rng.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    zernike_indices = tuple(int(v) for v in config["model"].get("zernike_indices", list(range(3, 16))))
    x_axis = np.arange(len(zernike_indices))

    was_training = head.training
    head.eval()
    with torch.no_grad():
        for out_idx, dataset_idx in enumerate(indices, start=1):
            sample = dataset[int(dataset_idx)]
            image = sample["input"][None].to(device)
            target = sample["coeff"].detach().cpu().numpy()
            pred, scores, presence, sign = head._template_coefficients(image)
            pred_np = pred[0].detach().cpu().numpy()
            presence_np = presence[0].detach().cpu().numpy()
            sign_np = sign[0].detach().cpu().numpy()
            score_np = scores[0].detach().cpu().numpy()
            error_np = pred_np - target

            fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
            width = 0.38
            axes[0].bar(x_axis - width / 2, target, width=width, label="ground truth", color="#374151")
            axes[0].bar(x_axis + width / 2, pred_np, width=width, label="template prediction", color="#2563eb")
            axes[0].axhline(0.0, color="black", linewidth=0.8)
            axes[0].set_ylabel("coefficient (um)")
            axes[0].set_title(
                f"Epoch {epoch} sample {out_idx}: {Path(str(sample['input_path'])).name}\n"
                f"MAE={np.mean(np.abs(error_np)):.4f} um"
            )
            axes[0].legend(loc="upper right")
            axes[0].grid(axis="y", linestyle="--", alpha=0.35)

            axes[1].bar(x_axis, error_np, width=0.55, color="#dc2626", label="prediction - ground truth")
            axes[1].plot(x_axis, presence_np, marker="o", linewidth=1.2, color="#059669", label="presence")
            axes[1].plot(x_axis, sign_np, marker="x", linewidth=1.2, color="#9333ea", label="sign")
            axes[1].axhline(0.0, color="black", linewidth=0.8)
            axes[1].set_ylabel("error / gate")
            axes[1].set_xlabel("Zernike index")
            axes[1].grid(axis="y", linestyle="--", alpha=0.35)
            axes[1].legend(loc="upper right", ncol=3, fontsize=8)

            axes[1].set_xticks(x_axis)
            axes[1].set_xticklabels([str(v) for v in zernike_indices])
            fig.tight_layout()
            fig.savefig(save_dir / f"epoch_{epoch:04d}_sample_{out_idx:02d}.png", dpi=180)
            plt.close(fig)

            np.savez_compressed(
                save_dir / f"epoch_{epoch:04d}_sample_{out_idx:02d}.npz",
                zernike_indices=np.asarray(zernike_indices),
                target=target,
                prediction=pred_np,
                error=error_np,
                presence=presence_np,
                sign=sign_np,
                scores=score_np,
                input_path=str(sample["input_path"]),
                coeff_path=str(sample["coeff_path"]),
            )
    head.train(was_training)


def save_same_object_batch_figure(
    head,
    dataset: TemplateAttentionPretrainDataset,
    output_dir: Path,
    device,
    epoch: int,
    config: dict,
) -> None:
    vis_cfg = config.get("visualization", {})
    count = int(vis_cfg.get("same_object_batch_samples", 0))
    interval = int(vis_cfg.get("interval", 1))
    if count <= 0 or interval <= 0 or epoch % interval != 0:
        return

    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(dataset.records):
        groups.setdefault(record.object_id, []).append(idx)
    groups = {key: value for key, value in groups.items() if value}
    if not groups:
        return

    rng = np.random.default_rng(int(vis_cfg.get("seed", 20260602)) + 100_000 + epoch)
    object_id = str(rng.choice(sorted(groups)))
    group_indices = groups[object_id]
    sampled = rng.choice(group_indices, size=min(count, len(group_indices)), replace=len(group_indices) < count)

    images = []
    targets = []
    input_paths = []
    coeff_paths = []
    for dataset_idx in sampled:
        sample = dataset[int(dataset_idx)]
        images.append(sample["input"])
        targets.append(sample["coeff"])
        input_paths.append(str(sample["input_path"]))
        coeff_paths.append(str(sample["coeff_path"]))

    image_batch = torch.stack(images, dim=0).to(device)
    target_batch = torch.stack(targets, dim=0).to(device)
    zernike_indices = tuple(int(v) for v in config["model"].get("zernike_indices", list(range(3, 16))))

    save_dir = output_dir / "validation_same_object_batches"
    save_dir.mkdir(parents=True, exist_ok=True)

    was_training = head.training
    head.eval()
    with torch.no_grad():
        pred, _, presence, sign = head._template_coefficients(image_batch)
    head.train(was_training)

    target_np = target_batch.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    error_np = pred_np - target_np
    presence_np = presence.detach().cpu().numpy()
    sign_np = sign.detach().cpu().numpy()
    vmax = float(max(np.abs(target_np).max(), np.abs(pred_np).max(), 1e-6))
    err_vmax = float(max(np.abs(error_np).max(), 1e-6))

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    panels = [
        ("ground truth coefficients (um)", target_np, vmax),
        ("template prediction (um)", pred_np, vmax),
        ("prediction error (um)", error_np, err_vmax),
        ("presence gate", presence_np, 1.0),
    ]
    for ax, (title, values, limit) in zip(axes, panels, strict=True):
        im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        if title == "presence gate":
            im.set_clim(0.0, 1.0)
            im.set_cmap("viridis")
        ax.set_ylabel("sample")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    axes[-1].set_xticks(np.arange(len(zernike_indices)))
    axes[-1].set_xticklabels([str(v) for v in zernike_indices])
    axes[-1].set_xlabel("Zernike index")
    fig.suptitle(
        f"Epoch {epoch}: same-object validation batch {object_id}, "
        f"MAE={np.mean(np.abs(error_np)):.4f} um",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(save_dir / f"epoch_{epoch:04d}_{object_id}.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        save_dir / f"epoch_{epoch:04d}_{object_id}.npz",
        zernike_indices=np.asarray(zernike_indices),
        object_id=object_id,
        target=target_np,
        prediction=pred_np,
        error=error_np,
        presence=presence_np,
        sign=sign_np,
        input_paths=np.asarray(input_paths),
        coeff_paths=np.asarray(coeff_paths),
    )


def make_dataset(config: dict, split: str, data_root: Path) -> TemplateAttentionPretrainDataset:
    data_cfg = config["data"][split]
    patch_size = config["data"].get("patch_size")
    if patch_size is not None:
        patch_size = tuple(int(v) for v in patch_size)
    image_dir = resolve_path(data_cfg.get("image_dir", "abe"), data_root)
    zernike_dir = resolve_path(data_cfg.get("zernike_dir", "Zernike"), data_root)
    return TemplateAttentionPretrainDataset(
        image_dir,
        zernike_dir,
        patch_size=patch_size,
        crop_mode=str(data_cfg.get("crop_mode", "random" if split == "train" else "center")),
        samples_per_epoch=data_cfg.get("samples_per_epoch"),
        input_scale_method=str(config["data"].get("input_scale_method", "percentile")),
        input_scale_percentile=float(config["data"].get("input_scale_percentile", 99.9)),
    )


def make_object_dataset(config: dict, data_root: Path) -> CleanObjectDataset:
    data_cfg = config["data"]["train"]
    patch_size = config["data"].get("patch_size")
    if patch_size is not None:
        patch_size = tuple(int(v) for v in patch_size)
    object_dir = resolve_path(data_cfg.get("object_dir", "OBJ"), data_root)
    return CleanObjectDataset(
        object_dir,
        patch_size=patch_size,
        crop_mode=str(data_cfg.get("object_crop_mode", data_cfg.get("crop_mode", "random"))),
        samples_per_epoch=int(config["training"].get("synthetic_batches_per_epoch", data_cfg.get("samples_per_epoch", 250))),
        input_scale_method=str(config["data"].get("input_scale_method", "percentile")),
        input_scale_percentile=float(config["data"].get("input_scale_percentile", 99.9)),
    )


def batch_size_for_epoch(config: dict, epoch: int) -> int:
    training = config["training"]
    schedule = training.get("batch_size_schedule")
    if not schedule:
        return int(training.get("batch_size", 8))
    for item in schedule:
        until_epoch = int(item.get("until_epoch", item.get("end_epoch", epoch)))
        if epoch <= until_epoch:
            return int(item["batch_size"])
    return int(schedule[-1]["batch_size"])


def make_loader(
    dataset: TemplateAttentionPretrainDataset,
    config: dict,
    split: str,
    batch_size: int,
    epoch: int,
) -> DataLoader:
    training = config["training"]
    num_workers = int(training.get("num_workers", 4))
    if split == "train" and bool(training.get("same_object_batches", False)):
        batches_per_epoch = math.ceil(len(dataset) / batch_size)
        sampler = SameObjectBatchSampler(
            dataset,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            seed=int(training.get("seed", 20260602)) + epoch,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )


def make_object_loader(dataset: CleanObjectDataset, config: dict, epoch: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the OTF template attention encoder on Zernike labels.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    data_root = Path(args.data_root or config.get("data", {}).get("root", ".")).expanduser()
    output_dir = Path(args.output or config.get("output_dir", "outputs/template_attention_pretrain"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config["model"] = prepare_model_config(config)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthetic_train = bool(config["training"].get("synthetic_same_object_batches", False))
    train_set = make_object_dataset(config, data_root) if synthetic_train else make_dataset(config, "train", data_root)
    val_set = make_dataset(config, "val", data_root)
    val_batch_size = int(config["training"].get("val_batch_size", config["training"].get("batch_size", 8)))
    val_loader = make_loader(val_set, config, "val", val_batch_size, epoch=0)

    model = make_model(config["model"]).to(device)
    head = get_template_head(model)
    if bool(config["training"].get("auto_max_amp", True)):
        q = float(config["training"].get("max_amp_quantile", 0.95))
        scale = float(config["training"].get("max_amp_scale", 1.1))
        max_amp_source = val_set if synthetic_train else train_set
        max_amp = estimate_max_amp(max_amp_source, q, scale).to(device)
        head.max_amp_um = max_amp
        print(f"Using max_amp_um from q={q}: {max_amp.detach().cpu().numpy().round(4).tolist()}")

    forward_model = None
    if synthetic_train:
        image_size = tuple(int(v) for v in config["data"].get("patch_size", [256, 256]))
        zernike_indices = tuple(int(v) for v in config["model"].get("zernike_indices", list(range(3, 16))))
        forward_model = AO2DForwardModel(image_size, zernike_indices, make_optics_config(config)).to(device)

    trainable = set_trainable_template_params(head, bool(config["training"].get("train_thresholds", True)))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"].get("lr", config["training"].get("initial_lr", 1e-4))),
        weight_decay=float(config["training"].get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["training"].get("epochs", 50))),
        eta_min=float(config["training"].get("min_lr", 5e-6)),
    )

    metrics_path = output_dir / "metrics.csv"
    best_mae = float("inf")
    with metrics_path.open("w", newline="") as f:
        writer = None
        for epoch in range(1, int(config["training"].get("epochs", 50)) + 1):
            train_batch_size = batch_size_for_epoch(config, epoch)
            if synthetic_train:
                train_loader = make_object_loader(train_set, config, epoch)
                train_metrics = run_synthetic_same_object_epoch(
                    head,
                    train_loader,
                    forward_model,
                    optimizer,
                    device,
                    True,
                    config,
                    train_batch_size,
                )
            else:
                train_loader = make_loader(train_set, config, "train", train_batch_size, epoch)
                train_metrics = run_epoch(head, train_loader, optimizer, device, True, config)
            val_metrics = run_epoch(head, val_loader, optimizer, device, False, config)
            scheduler.step()
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "batch_size": train_batch_size}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(
                f"epoch={epoch:03d} "
                f"train_mae={train_metrics['mae']:.5f} val_mae={val_metrics['mae']:.5f} "
                f"val_sign={val_metrics['sign_acc']:.3f} val_presence={val_metrics['presence_acc']:.3f}"
            )
            save_validation_coefficient_figures(head, val_set, output_dir, device, epoch, config)
            save_same_object_batch_figure(head, val_set, output_dir, device, epoch, config)
            checkpoint = {
                "epoch": epoch,
                "config": config,
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
