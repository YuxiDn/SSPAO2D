#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao2d.data import AO2DSelfDataset, build_dataloader, get_data_root, resolve_path
from ao2d.data.io import load_image
from ao2d.models.factory import make_model
from ao2d.models.picnet2d import Discriminator2D
from ao2d.optics import AO2DConfig
from ao2d.optics import pupil_coordinates_2d, random_zernike_coefficients, zernike_wavefront
from ao2d.training.epoch_metrics import read_metrics_xlsx, write_metrics_xlsx
from ao2d.training import (
    AO2DForwardModel,
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    get_current_lr,
    grad_norm,
    make_sampler,
    psnr,
    reduce_metrics,
    set_sampler_epoch,
    setup_distributed,
    step_scheduler,
    ssim,
    total_variation_2d,
    unwrap_ddp,
    wrap_ddp,
)


def adversarial_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def adversarial_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -0.5 * fake_logits.mean()


def feature_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gradx_pred = torch.cat(
        [prediction[:, :, :, 1:] - prediction[:, :, :, :-1], prediction[:, :, :, :1] - prediction[:, :, :, -1:]],
        dim=3,
    )
    grady_pred = torch.cat(
        [prediction[:, :, 1:, :] - prediction[:, :, :-1, :], prediction[:, :, :1, :] - prediction[:, :, -1:, :]],
        dim=2,
    )
    gradx_target = torch.cat(
        [target[:, :, :, 1:] - target[:, :, :, :-1], target[:, :, :, :1] - target[:, :, :, -1:]],
        dim=3,
    )
    grady_target = torch.cat(
        [target[:, :, 1:, :] - target[:, :, :-1, :], target[:, :, :1, :] - target[:, :, -1:, :]],
        dim=2,
    )
    return (torch.abs(gradx_pred - gradx_target) + torch.abs(grady_pred - grady_target)).mean()


def zernike_phase(
    coefficients: torch.Tensor,
    zernike_indices: tuple[int, ...],
    image_size: tuple[int, int],
    optics_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    wavelength = float(optics_config.get("lambda_emission", optics_config.get("wavelength", 1.0)))
    rho, theta, pupil_mask = pupil_coordinates_2d(
        image_size,
        float(optics_config.get("pixel_size", 0.300)),
        wavelength,
        float(optics_config.get("na", 1.05)),
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    wavefront = zernike_wavefront(zernike_indices, coefficients, rho, theta)
    phase = 2.0 * torch.pi / wavelength * wavefront
    return phase, pupil_mask


def phase_consistency_losses(
    pred_coeff: torch.Tensor,
    target_coeff: torch.Tensor,
    zernike_indices: tuple[int, ...],
    image_size: tuple[int, int],
    optics_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_phase, pupil_mask = zernike_phase(pred_coeff, zernike_indices, image_size, optics_config)
    target_phase, _ = zernike_phase(target_coeff, zernike_indices, image_size, optics_config)
    pupil_mask = pupil_mask.to(device=pred_phase.device)

    phase_delta = pred_phase - target_phase
    phase_loss = (1.0 - torch.cos(phase_delta))[:, pupil_mask].mean()

    pred_dx = pred_phase[..., :, 1:] - pred_phase[..., :, :-1]
    target_dx = target_phase[..., :, 1:] - target_phase[..., :, :-1]
    pred_dy = pred_phase[..., 1:, :] - pred_phase[..., :-1, :]
    target_dy = target_phase[..., 1:, :] - target_phase[..., :-1, :]
    mask_dx = pupil_mask[:, 1:] & pupil_mask[:, :-1]
    mask_dy = pupil_mask[1:, :] & pupil_mask[:-1, :]
    phase_grad_loss = F.l1_loss(pred_dx[:, mask_dx], target_dx[:, mask_dx]) + F.l1_loss(
        pred_dy[:, mask_dy], target_dy[:, mask_dy]
    )
    return phase_loss, phase_grad_loss


def set_requires_grad(module, requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def maybe_no_sync(module, sync_grad: bool):
    if sync_grad or not hasattr(module, "no_sync"):
        return nullcontext()
    return module.no_sync()


def read_config(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return json.load(f)


def adversarial_weight(config: dict) -> float:
    training = config["training"]
    return float(training.get("adversarial_weight", training.get("adv_coefficient", 0.0)))


def make_optics_config(config: dict) -> AO2DConfig:
    optics = config.get("optics", {})
    return AO2DConfig(
        pixel_size=float(optics.get("pixel_size", 0.300)),
        na=float(optics.get("na", 1.05)),
        lambda_emission=float(optics.get("lambda_emission", 1.000)),
        lambda_excitation=float(optics.get("lambda_excitation", 0.808)),
        mode=str(optics.get("mode", "widefield")),
        pinhole_au=float(optics.get("pinhole_au", 1.0)),
        lightsheet_fwhm=float(optics.get("lightsheet_fwhm", 1.2)),
    )


def make_self_dataset(config: dict, split: str, data_root: Path | None, augment_default: bool = False) -> AO2DSelfDataset:
    data_cfg = config["data"]
    split_cfg = data_cfg[split]
    augment = bool(split_cfg.get("augment", augment_default))
    return AO2DSelfDataset(
        resolve_path(split_cfg["image_dir"], data_root),
        patch_size=tuple(data_cfg.get("patch_size", [256, 256])),
        augment=augment,
        samples_per_epoch=split_cfg.get("samples_per_epoch"),
        normalization_mode=str(split_cfg.get("normalization_mode", data_cfg.get("normalization_mode", "input_scale"))),
        input_scale_method=str(split_cfg.get("input_scale_method", data_cfg.get("input_scale_method", "percentile"))),
        input_scale_percentile=float(split_cfg.get("input_scale_percentile", data_cfg.get("input_scale_percentile", 99.9))),
        crop_mode=str(split_cfg.get("crop_mode", "random" if augment else "center")),
    )


def metric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(key)
        if raw in {None, ""}:
            continue
        values.append(float(raw))
    return values


def training_float(training: dict, key: str, aliases: tuple[str, ...] = (), default: float = 0.0) -> float:
    for candidate in (key, *aliases):
        if candidate in training:
            return float(training[candidate])
    return float(default)


def add_metric_aliases(rows: list[dict[str, object]]) -> None:
    aliases = {
        "cycle_aberrated_image_feature_loss": ("cycle_aberrated_loss",),
        "cycle_aberrated_image_l1_loss": ("cycle_aberrated_l1_loss",),
        "zernike_coeff_l1_loss": ("cycle_aberration_loss",),
    }
    for row in rows:
        for new_key, old_keys in aliases.items():
            if new_key in row:
                continue
            for old_key in old_keys:
                if old_key in row:
                    row[new_key] = row[old_key]
                    break


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    if file_exists:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            existing_fields = list(reader.fieldnames or [])
        if existing_fields != fieldnames:
            merged_fields = existing_fields + [name for name in fieldnames if name not in existing_fields]
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fields)
                writer.writeheader()
                for existing_row in rows:
                    writer.writerow({name: existing_row.get(name, "") for name in merged_fields})
            fieldnames = merged_fields
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_metrics_row(path: Path, rows: list[dict[str, object]], row: dict[str, object]) -> None:
    rows.append(row)
    append_csv_row(path.with_suffix(".csv"), list(row.keys()), row)
    write_metrics_xlsx(path.with_suffix(".xlsx"), rows)


def spaced_indices(total: int, limit: int) -> list[int]:
    if limit <= 0 or total <= 0:
        return []
    if limit >= total:
        return list(range(total))
    return np.linspace(0, total - 1, num=limit, dtype=int).tolist()


def center_crop_array(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = x.shape[-2:]
    th, tw = shape
    if h == th and w == tw:
        return x
    top = max(0, (h - th) // 2)
    left = max(0, (w - tw) // 2)
    return x[top : top + th, left : left + tw]


def rmse_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def pcc_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    pred_std = float(pred_flat.std())
    target_std = float(target_flat.std())
    if pred_std < 1e-12 or target_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred_flat, target_flat)[0, 1])


def load_zernike_coefficients(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            for key in ("coefficients_um", "coefficients", "zernike_coefficients"):
                if key in data.files:
                    return np.asarray(data[key], dtype=np.float32).reshape(-1)
        return None
    if path.suffix.lower() == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError:
            return None
        data = loadmat(path)
        for key in ("coefficients", "coefficients_um", "zernike_coefficients"):
            if key in data:
                return np.asarray(data[key], dtype=np.float32).reshape(-1)
    return None


def infer_validation_paths(config: dict, data_root: Path | None) -> tuple[Path | None, Path | None]:
    data_cfg = config["data"]
    val_cfg = data_cfg.get("val", {})
    if data_root is None:
        return None, None
    object_dir = val_cfg.get("object_dir")
    zernike_dir = val_cfg.get("zernike_dir")
    if object_dir is None:
        object_dir = str(Path(val_cfg.get("image_dir", "validation/abe")).parent / "OBJ")
    if zernike_dir is None:
        zernike_dir = str(Path(val_cfg.get("image_dir", "validation/abe")).parent / "Zernike")
    return resolve_path(object_dir, data_root), resolve_path(zernike_dir, data_root)


def matched_object_path(input_path: Path, object_dir: Path | None) -> Path | None:
    if object_dir is None:
        return None
    match = re.search(r"(obj_\d+)", input_path.stem)
    if match is None:
        return None
    for suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        path = object_dir / f"{match.group(1)}{suffix}"
        if path.exists():
            return path
    return None


def matched_zernike_path(input_path: Path, zernike_dir: Path | None) -> Path | None:
    if zernike_dir is None:
        return None
    for suffix in (".mat", ".npz", ".npy"):
        path = zernike_dir / f"{input_path.stem}_zernike{suffix}"
        if path.exists():
            return path
    return None


def save_result_figure(
    save_path: Path,
    measured: np.ndarray,
    target: np.ndarray | None,
    restored: np.ndarray,
    true_coeff: np.ndarray | None,
    pred_coeff: np.ndarray,
    metrics: dict[str, float],
    display_limits: dict[str, tuple[float | None, float | None]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def add_colorbar(fig, ax, im) -> None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.03)
        fig.colorbar(im, cax=cax)

    intensity_vmin, intensity_vmax = display_limits.get("intensity", (None, None))
    object_vmin, object_vmax = display_limits.get("object", (None, None))
    coeff_vmin, coeff_vmax = display_limits.get("zernike", (None, None))
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.8])

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(measured, cmap="gray", vmin=intensity_vmin, vmax=intensity_vmax)
    ax1.set_title("Aberrated")
    ax1.axis("off")
    add_colorbar(fig, ax1, im1)

    ax2 = fig.add_subplot(gs[0, 1])
    if target is not None:
        im2 = ax2.imshow(target, cmap="hot", vmin=object_vmin, vmax=object_vmax)
        ax2.set_title("Object (GT)")
        add_colorbar(fig, ax2, im2)
    else:
        ax2.text(0.5, 0.5, "No GT object", ha="center", va="center")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    title = "Object (Pred)"
    if metrics:
        title += (
            f"\nPSNR: {metrics.get('psnr', float('nan')):.2f}  "
            f"SSIM: {metrics.get('ssim', float('nan')):.2f}\n"
            f"RMSE: {metrics.get('rmse', float('nan')):.4f}  "
            f"PCC: {metrics.get('pcc', float('nan')):.2f}"
        )
    im3 = ax3.imshow(restored, cmap="hot", vmin=object_vmin, vmax=object_vmax)
    ax3.set_title(title)
    ax3.axis("off")
    add_colorbar(fig, ax3, im3)

    ax4 = fig.add_subplot(gs[1, :])
    x = np.arange(pred_coeff.size)
    if true_coeff is not None and true_coeff.size == pred_coeff.size:
        ax4.bar(x - 0.18, true_coeff, width=0.36, label="Ground truth")
        ax4.bar(x + 0.18, pred_coeff, width=0.36, label="SCARE2D")
    else:
        ax4.bar(x, pred_coeff, width=0.5, label="SCARE2D")
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(i) for i in range(3, 3 + pred_coeff.size)])
    ax4.set_xlabel("Zernike polynomials")
    ax4.set_ylabel("Coefficients")
    if coeff_vmin is not None or coeff_vmax is not None:
        ax4.set_ylim(coeff_vmin, coeff_vmax)
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", frameon=False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_validation_figures(
    model,
    dataset,
    forward_model,
    output_dir: Path,
    config: dict,
    data_root: Path | None,
    device,
) -> None:
    if dataset is None:
        return
    val_cfg = config.get("data", {}).get("val", {})
    limit = int(val_cfg.get("save_limit", config.get("training", {}).get("val_save_limit", 8)))
    indices = spaced_indices(len(dataset), limit)
    if not indices:
        return
    object_dir, zernike_dir = infer_validation_paths(config, data_root)
    display_limits = {
        "intensity": tuple(config.get("display", {}).get("intensity", [0.0, 1.0])),
        "object": tuple(config.get("display", {}).get("object", [0.0, 1.0])),
        "zernike": tuple(config.get("display", {}).get("zernike", [None, None])),
    }
    eval_model = unwrap_ddp(model)
    was_training = eval_model.training
    eval_model.eval()
    with torch.no_grad():
        for out_idx, dataset_idx in enumerate(indices, start=1):
            sample = dataset[dataset_idx]
            x = sample["input"][None].to(device)
            restored, pred_coeff = eval_model(x)
            measured = x[0, 0].detach().cpu().numpy()
            restored_np = restored[0, 0].detach().cpu().numpy()
            pred_coeff_np = pred_coeff[0].detach().cpu().numpy()

            input_path = Path(str(sample.get("input_path", "")))
            object_path = matched_object_path(input_path, object_dir)
            target_np = None
            metrics = {}
            if object_path is not None:
                scale = float(sample.get("input_scale", torch.tensor(1.0)))
                target_np = np.maximum(load_image(object_path).astype(np.float32, copy=False), 0) / max(
                    scale, float(np.finfo(np.float32).eps)
                )
                target_np = center_crop_array(target_np, restored_np.shape)
                metrics = {
                    "psnr": float(psnr(torch.from_numpy(target_np)[None, None], torch.from_numpy(restored_np)[None, None])),
                    "ssim": float(ssim(torch.from_numpy(target_np)[None, None], torch.from_numpy(restored_np)[None, None])),
                    "rmse": rmse_np(restored_np, target_np),
                    "pcc": pcc_np(restored_np, target_np),
                }

            coeff_path = matched_zernike_path(input_path, zernike_dir)
            true_coeff = load_zernike_coefficients(coeff_path) if coeff_path is not None else None
            save_result_figure(
                output_dir / f"test{out_idx:04d}.png",
                measured,
                target_np,
                restored_np,
                true_coeff,
                pred_coeff_np,
                metrics,
                display_limits,
            )
    eval_model.train(was_training)


def checkpoint_state(
    epoch: int,
    model,
    discriminator,
    optimizer_G,
    optimizer_D,
    scheduler_G,
    scheduler_D,
    config: dict,
    val_metrics: dict,
) -> dict:
    return {
        "epoch": epoch,
        "model": unwrap_ddp(model).state_dict(),
        "discriminator": unwrap_ddp(discriminator).state_dict() if discriminator is not None else None,
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict() if optimizer_D is not None else None,
        "scheduler_G": scheduler_G.state_dict() if scheduler_G is not None else None,
        "scheduler_D": scheduler_D.state_dict() if scheduler_D is not None else None,
        "config": config,
        "val": val_metrics,
    }


def iteration_checkpoint_state(
    iteration: int,
    model,
    discriminator,
    optimizer_G,
    optimizer_D,
    scheduler_G,
    scheduler_D,
    config: dict,
    loss_history: dict[str, list[float]],
    metrics: dict,
) -> dict:
    model_state = unwrap_ddp(model).state_dict()
    discriminator_state = unwrap_ddp(discriminator).state_dict() if discriminator is not None else None
    return {
        "iteration": iteration,
        "epoch": iteration,
        "model": model_state,
        "model_state_dict": model_state,
        "discriminator": discriminator_state,
        "discriminator_state_dict": discriminator_state,
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict() if optimizer_D is not None else None,
        "scheduler_G": scheduler_G.state_dict() if scheduler_G is not None else None,
        "scheduler_D": scheduler_D.state_dict() if scheduler_D is not None else None,
        "loss": loss_history,
        "metrics": metrics,
        "config": config,
        "val": metrics,
    }


def resume_training(
    resume: str,
    output_dir: Path,
    device,
    model,
    discriminator,
    optimizer_G,
    optimizer_D,
    scheduler_G,
    scheduler_D,
) -> tuple[Path, int, float]:
    resume_path = output_dir / "last.pt" if resume == "auto" else Path(resume)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    unwrap_ddp(model).load_state_dict(checkpoint.get("model", checkpoint.get("model_state_dict")))
    if discriminator is not None and checkpoint.get("discriminator") is not None:
        unwrap_ddp(discriminator).load_state_dict(checkpoint["discriminator"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G"])
    if optimizer_D is not None and checkpoint.get("optimizer_D") is not None:
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])
    if scheduler_G is not None and checkpoint.get("scheduler_G") is not None:
        scheduler_G.load_state_dict(checkpoint["scheduler_G"])
    if scheduler_D is not None and checkpoint.get("scheduler_D") is not None:
        scheduler_D.load_state_dict(checkpoint["scheduler_D"])

    start_epoch = int(checkpoint.get("iteration", checkpoint.get("epoch", 0))) + 1
    best = float(checkpoint.get("val", {}).get("loss", float("inf")))
    return resume_path, start_epoch, best


def next_image_batch(loader_iter, loader, device):
    if loader is None:
        return None, loader_iter
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch["input"].to(device, non_blocking=True), loader_iter


def random_coefficients_batch(batch_size: int, zernike_indices: tuple[int, ...], device, config: dict) -> torch.Tensor:
    optics = config.get("optics", {})
    training = config["training"]
    rms_min = float(training.get("zernike_rms_min", training.get("random_zernike_rms_min", 0.02)))
    rms_max = float(training.get("zernike_rms_max", training.get("random_zernike_rms_max", 0.30)))
    wavelength = float(optics.get("lambda_emission", optics.get("wavelength", 1.0)))
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


def loss_weights(config: dict) -> dict[str, float]:
    training = config["training"]
    return {
        "tv": float(training.get("tv_weight", 1e-5)),
        "coeff_l2": float(training.get("coeff_l2", 1e-4)),
        "cycle_aberrated_image_feature": training_float(
            training,
            "cycle_aberrated_image_feature_weight",
            ("cycle_aberrated_weight", "intensity_coefficient"),
        ),
        "cycle_aberrated_image_l1": training_float(
            training,
            "cycle_aberrated_image_l1_weight",
            ("cycle_aberrated_l1_weight", "intensity_l1_coefficient", "abeloss_l1_weight"),
            default=1.0,
        ),
        "cycle_object": float(training.get("cycle_object_weight", training.get("pha_coefficient", 0.0))),
        "cycle_object_feature": float(
            training.get("cycle_object_feature_weight", training.get("feature_pha_coefficient", 0.0))
        ),
        "zernike_coeff_l1": training_float(
            training,
            "zernike_coeff_l1_weight",
            ("cycle_aberration_weight", "aberration_coefficient"),
        ),
        "phase": float(training.get("phase_loss_weight", 0.0)),
        "phase_grad": float(training.get("phase_grad_loss_weight", 0.0)),
        "identity": float(training.get("identity_weight", training.get("identity_coefficient", 0.0))),
        "random_reaberration_object": training_float(
            training,
            "random_reaberration_object_weight",
            ("reaberration_object_weight", "random_object_consistency_weight"),
        ),
        "random_reaberration_feature": training_float(
            training,
            "random_reaberration_feature_weight",
            ("reaberration_feature_weight", "random_object_feature_consistency_weight"),
        ),
        "random_reaberration_coeff": training_float(
            training,
            "random_reaberration_coeff_weight",
            ("reaberration_coeff_weight", "random_zernike_consistency_weight"),
        ),
        "adversarial": adversarial_weight(config),
    }


def train_iteration(
    model,
    forward_model,
    batch,
    optimizer_G,
    device,
    config,
    discriminator=None,
    optimizer_D=None,
    object_loader=None,
    object_iter=None,
    identity_loader=None,
    identity_iter=None,
    zero_grad: bool = True,
    step_optimizer: bool = True,
    sync_grad: bool = True,
    loss_scale: float = 1.0,
) -> tuple[dict[str, float], object, object]:
    model.train(True)
    if discriminator is not None:
        discriminator.train(True)
    weights = loss_weights(config)
    zernike_indices = tuple(config.get("optics", {}).get("zernike_indices", list(range(3, 16))))
    x = batch["input"].to(device, non_blocking=True)

    use_object_cycle = object_loader is not None and (
        weights["cycle_object"] > 0 or weights["cycle_object_feature"] > 0 or weights["zernike_coeff_l1"] > 0
    )
    use_identity = identity_loader is not None and weights["identity"] > 0
    real_obj = None
    target_coeff = None
    clean_obj = None
    model_inputs = [x]
    split_sizes = [x.shape[0]]

    if use_object_cycle:
        real_obj, object_iter = next_image_batch(object_iter, object_loader, device)
        target_coeff = random_coefficients_batch(real_obj.shape[0], zernike_indices, device, config).to(dtype=real_obj.dtype)
        generated_aberrated = forward_model(real_obj, target_coeff).float()
        model_inputs.append(generated_aberrated)
        split_sizes.append(generated_aberrated.shape[0])

    if use_identity:
        clean_obj, identity_iter = next_image_batch(identity_iter, identity_loader, device)
        model_inputs.append(clean_obj)
        split_sizes.append(clean_obj.shape[0])

    output = model(torch.cat(model_inputs, dim=0))
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError("Self-supervised AO training requires a model that returns (restored, zernike_coeff), such as scare2d or picnet2d.")
    restored_all, coeff_all = output
    restored_parts = torch.split(restored_all, split_sizes, dim=0)
    coeff_parts = torch.split(coeff_all, split_sizes, dim=0)
    restored = restored_parts[0]
    coeff = coeff_parts[0]
    estimated = forward_model(restored, coeff)

    loss_cycle_aberrated_image_feature = weights["cycle_aberrated_image_feature"] * feature_l1_loss(
        torch.sqrt(estimated.clamp_min(1e-8)),
        torch.sqrt(x.clamp_min(1e-8)),
    )
    loss_cycle_aberrated_image_l1 = weights["cycle_aberrated_image_l1"] * F.l1_loss(estimated, x)
    loss_tv = weights["tv"] * total_variation_2d(restored)
    loss_coeff_l2 = weights["coeff_l2"] * torch.mean(coeff**2)
    loss = loss_cycle_aberrated_image_feature + loss_cycle_aberrated_image_l1 + loss_tv + loss_coeff_l2
    loss_cycle_object = x.new_zeros(())
    loss_zernike_coeff_l1 = x.new_zeros(())
    loss_phase = x.new_zeros(())
    loss_phase_grad = x.new_zeros(())
    loss_identity = x.new_zeros(())
    loss_random_reaberration_object = x.new_zeros(())
    loss_random_reaberration_coeff = x.new_zeros(())
    loss_adv = x.new_zeros(())
    loss_adv_weighted = x.new_zeros(())
    loss_D = x.new_zeros(())

    part_idx = 1
    if use_object_cycle and real_obj is not None and target_coeff is not None:
        restored_obj = restored_parts[part_idx]
        pred_coeff = coeff_parts[part_idx]
        part_idx += 1
        loss_cycle_object = (
            weights["cycle_object"] * F.l1_loss(restored_obj, real_obj)
            + weights["cycle_object_feature"] * feature_l1_loss(restored_obj, real_obj)
        )
        loss_zernike_coeff_l1 = weights["zernike_coeff_l1"] * F.l1_loss(pred_coeff, target_coeff)
        if weights["phase"] > 0 or weights["phase_grad"] > 0:
            raw_phase_loss, raw_phase_grad_loss = phase_consistency_losses(
                pred_coeff,
                target_coeff,
                zernike_indices,
                tuple(config["data"].get("patch_size", [256, 256])),
                config.get("optics", {}),
            )
            loss_phase = weights["phase"] * raw_phase_loss
            loss_phase_grad = weights["phase_grad"] * raw_phase_grad_loss
        loss = loss + loss_cycle_object + loss_zernike_coeff_l1 + loss_phase + loss_phase_grad

    if use_identity and clean_obj is not None:
        identity_restored = restored_parts[part_idx]
        zero_coeff = torch.zeros(clean_obj.shape[0], len(zernike_indices), device=device, dtype=clean_obj.dtype)
        identity_estimated = forward_model(identity_restored, zero_coeff).float()
        loss_identity = weights["identity"] * F.l1_loss(identity_estimated, clean_obj)
        loss = loss + loss_identity

    if (
        weights["random_reaberration_object"] > 0
        or weights["random_reaberration_feature"] > 0
        or weights["random_reaberration_coeff"] > 0
    ):
        restored_ref = restored.detach()
        random_coeff = random_coefficients_batch(restored_ref.shape[0], zernike_indices, device, config).to(
            dtype=restored_ref.dtype
        )
        random_aberrated = forward_model(restored_ref, random_coeff).float().detach()
        random_restored, random_pred_coeff = model(random_aberrated)
        loss_random_reaberration_object = (
            weights["random_reaberration_object"] * F.l1_loss(random_restored, restored_ref)
            + weights["random_reaberration_feature"] * feature_l1_loss(random_restored, restored_ref)
        )
        loss_random_reaberration_coeff = weights["random_reaberration_coeff"] * F.l1_loss(
            random_pred_coeff,
            random_coeff,
        )
        loss = loss + loss_random_reaberration_object + loss_random_reaberration_coeff

    if discriminator is not None and optimizer_D is not None and weights["adversarial"] > 0:
        if real_obj is None:
            real_obj, object_iter = next_image_batch(object_iter, object_loader, device)
        set_requires_grad(discriminator, True)
        if zero_grad:
            optimizer_D.zero_grad(set_to_none=True)
        real_logits, fake_logits = torch.chunk(discriminator(torch.cat([real_obj, restored.detach()], dim=0)), 2, dim=0)
        loss_D = adversarial_d_loss(real_logits, fake_logits)
        with maybe_no_sync(discriminator, sync_grad):
            (loss_D / loss_scale).backward()
        if step_optimizer:
            optimizer_D.step()

        set_requires_grad(discriminator, False)
        fake_logits_G = discriminator(restored)
        loss_adv = adversarial_g_loss(fake_logits_G)
        loss_adv_weighted = weights["adversarial"] * loss_adv
        loss = loss + loss_adv_weighted

    if zero_grad:
        optimizer_G.zero_grad(set_to_none=True)
    with maybe_no_sync(model, sync_grad):
        (loss / loss_scale).backward()
    grad = grad_norm(model.parameters()) if step_optimizer else 0.0
    if step_optimizer:
        optimizer_G.step()
    set_requires_grad(discriminator, True)

    metrics = {
        "loss": float(loss.detach()),
        "loss_adv": float(loss_adv_weighted.detach()),
        "loss_D": float(loss_D.detach()),
        "loss_cycle_aberrated_image_feature": float(loss_cycle_aberrated_image_feature.detach()),
        "loss_cycle_aberrated_image_l1": float(loss_cycle_aberrated_image_l1.detach()),
        "loss_cycle_object": float(loss_cycle_object.detach()),
        "loss_zernike_coeff_l1": float(loss_zernike_coeff_l1.detach()),
        "loss_phase": float(loss_phase.detach()),
        "loss_phase_grad": float(loss_phase_grad.detach()),
        "loss_identity": float(loss_identity.detach()),
        "loss_random_reaberration_object": float(loss_random_reaberration_object.detach()),
        "loss_random_reaberration_coeff": float(loss_random_reaberration_coeff.detach()),
        "cycle_psnr": float(psnr(x, estimated).detach()),
        "cycle_ssim": float(ssim(x, estimated).detach()),
        "grad_norm": float(grad),
    }
    return metrics, object_iter, identity_iter


def average_metric_sums(metric_sums: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(1, count) for key, value in metric_sums.items()}


def add_metric_sums(metric_sums: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)


def run_epoch(
    model,
    forward_model,
    loader,
    optimizer_G,
    device,
    config,
    train: bool,
    show_progress: bool,
    discriminator=None,
    optimizer_D=None,
    object_loader=None,
    identity_loader=None,
):
    model.train(train)
    if discriminator is not None:
        discriminator.train(train)
    totals = {"loss": 0.0, "cycle_psnr": 0.0, "cycle_ssim": 0.0}
    if train:
        totals["grad_norm"] = 0.0
    if train:
        totals["loss_cycle_aberrated_image_feature"] = 0.0
        totals["loss_cycle_aberrated_image_l1"] = 0.0
        totals["loss_cycle_object"] = 0.0
        totals["loss_zernike_coeff_l1"] = 0.0
        totals["loss_phase"] = 0.0
        totals["loss_phase_grad"] = 0.0
        totals["loss_identity"] = 0.0
    if train and discriminator is not None:
        totals["loss_adv"] = 0.0
        totals["loss_D"] = 0.0
    if not train:
        totals["object_psnr"] = 0.0
        totals["object_ssim"] = 0.0
        totals["object_rmse"] = 0.0
        totals["object_pcc"] = 0.0
        totals["zernike_mae"] = 0.0
        totals["zernike_rmse"] = 0.0
        totals["object_count"] = 0.0
        totals["zernike_count"] = 0.0
    tv_weight = float(config["training"].get("tv_weight", 1e-5))
    coeff_l2 = float(config["training"].get("coeff_l2", 1e-4))
    training = config["training"]
    cycle_aberrated_image_feature_weight = training_float(
        training,
        "cycle_aberrated_image_feature_weight",
        ("cycle_aberrated_weight", "intensity_coefficient"),
    )
    cycle_aberrated_image_l1_weight = training_float(
        training,
        "cycle_aberrated_image_l1_weight",
        ("cycle_aberrated_l1_weight", "intensity_l1_coefficient", "abeloss_l1_weight"),
        default=1.0,
    )
    cycle_object_weight = float(config["training"].get("cycle_object_weight", config["training"].get("pha_coefficient", 0.0)))
    cycle_object_feature_weight = float(
        config["training"].get("cycle_object_feature_weight", config["training"].get("feature_pha_coefficient", 0.0))
    )
    zernike_coeff_l1_weight = training_float(
        training,
        "zernike_coeff_l1_weight",
        ("cycle_aberration_weight", "aberration_coefficient"),
    )
    phase_loss_weight = float(training.get("phase_loss_weight", 0.0))
    phase_grad_loss_weight = float(training.get("phase_grad_loss_weight", 0.0))
    identity_weight = float(config["training"].get("identity_weight", config["training"].get("identity_coefficient", 0.0)))
    adv_weight = adversarial_weight(config)
    zernike_indices = tuple(config.get("optics", {}).get("zernike_indices", list(range(3, 16))))
    object_iter = iter(object_loader) if object_loader is not None else None
    identity_iter = iter(identity_loader) if identity_loader is not None else None
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="train" if train else "val", leave=False, disable=not show_progress):
            x = batch["input"].to(device, non_blocking=True)
            output = model(x)
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("Self-supervised AO training requires a model that returns (restored, zernike_coeff), such as scare2d or picnet2d.")
            restored, coeff = output
            estimated = forward_model(restored, coeff)
            loss_cycle_aberrated_image_feature = cycle_aberrated_image_feature_weight * feature_l1_loss(
                torch.sqrt(estimated.clamp_min(1e-8)),
                torch.sqrt(x.clamp_min(1e-8)),
            )
            loss_cycle_aberrated_image_l1 = cycle_aberrated_image_l1_weight * F.l1_loss(estimated, x)
            loss_tv = tv_weight * total_variation_2d(restored)
            loss_coeff_l2 = coeff_l2 * torch.mean(coeff**2)
            loss = loss_cycle_aberrated_image_feature + loss_cycle_aberrated_image_l1 + loss_tv + loss_coeff_l2
            loss_cycle_object = x.new_zeros(())
            loss_zernike_coeff_l1 = x.new_zeros(())
            loss_phase = x.new_zeros(())
            loss_phase_grad = x.new_zeros(())
            loss_identity = x.new_zeros(())

            if train:
                if object_loader is not None and (
                    cycle_object_weight > 0
                    or cycle_object_feature_weight > 0
                    or zernike_coeff_l1_weight > 0
                    or phase_loss_weight > 0
                    or phase_grad_loss_weight > 0
                ):
                    real_obj, object_iter = next_image_batch(object_iter, object_loader, device)
                    target_coeff = random_coefficients_batch(real_obj.shape[0], zernike_indices, device, config).to(dtype=real_obj.dtype)
                    generated_aberrated = forward_model(real_obj, target_coeff).float()
                    restored_obj, pred_coeff = model(generated_aberrated)
                    loss_cycle_object = (
                        cycle_object_weight * F.l1_loss(restored_obj, real_obj)
                        + cycle_object_feature_weight * feature_l1_loss(restored_obj, real_obj)
                    )
                    loss_zernike_coeff_l1 = zernike_coeff_l1_weight * F.l1_loss(pred_coeff, target_coeff)
                    if phase_loss_weight > 0 or phase_grad_loss_weight > 0:
                        raw_phase_loss, raw_phase_grad_loss = phase_consistency_losses(
                            pred_coeff,
                            target_coeff,
                            zernike_indices,
                            tuple(config["data"].get("patch_size", [256, 256])),
                            config.get("optics", {}),
                        )
                        loss_phase = phase_loss_weight * raw_phase_loss
                        loss_phase_grad = phase_grad_loss_weight * raw_phase_grad_loss
                    loss = loss + loss_cycle_object + loss_zernike_coeff_l1 + loss_phase + loss_phase_grad

                if identity_loader is not None and identity_weight > 0:
                    clean_obj, identity_iter = next_image_batch(identity_iter, identity_loader, device)
                    identity_restored, _ = model(clean_obj)
                    zero_coeff = torch.zeros(
                        clean_obj.shape[0],
                        len(zernike_indices),
                        device=device,
                        dtype=clean_obj.dtype,
                    )
                    identity_estimated = forward_model(identity_restored, zero_coeff).float()
                    loss_identity = identity_weight * F.l1_loss(identity_estimated, clean_obj)
                    loss = loss + loss_identity

                if discriminator is not None and optimizer_D is not None and adv_weight > 0:
                    real_obj, object_iter = next_image_batch(object_iter, object_loader, device)
                    set_requires_grad(discriminator, True)
                    optimizer_D.zero_grad(set_to_none=True)
                    real_logits = discriminator(real_obj)
                    fake_logits = discriminator(restored.detach())
                    loss_D = adversarial_d_loss(real_logits, fake_logits)
                    loss_D.backward()
                    optimizer_D.step()

                    set_requires_grad(discriminator, False)
                    fake_logits_G = discriminator(restored)
                    loss_adv = adversarial_g_loss(fake_logits_G)
                    loss = loss + adv_weight * loss_adv
                    totals["loss_adv"] += float(loss_adv.detach())
                    totals["loss_D"] += float(loss_D.detach())

                optimizer_G.zero_grad(set_to_none=True)
                loss.backward()
                totals["grad_norm"] += grad_norm(model.parameters())
                optimizer_G.step()
                set_requires_grad(discriminator, True)
                totals["loss_cycle_aberrated_image_feature"] += float(loss_cycle_aberrated_image_feature.detach())
                totals["loss_cycle_aberrated_image_l1"] += float(loss_cycle_aberrated_image_l1.detach())
                totals["loss_cycle_object"] += float(loss_cycle_object.detach())
                totals["loss_zernike_coeff_l1"] += float(loss_zernike_coeff_l1.detach())
                totals["loss_phase"] += float(loss_phase.detach())
                totals["loss_phase_grad"] += float(loss_phase_grad.detach())
                totals["loss_identity"] += float(loss_identity.detach())
            totals["loss"] += float(loss.detach())
            totals["cycle_psnr"] += float(psnr(x, estimated).detach())
            totals["cycle_ssim"] += float(ssim(x, estimated).detach())

            if not train:
                data_root = get_data_root(config, None)
                object_dir, zernike_dir = infer_validation_paths(config, data_root)
                input_paths = batch.get("input_path", [])
                if isinstance(input_paths, str):
                    input_paths = [input_paths]
                input_scales = batch.get("input_scale")
                restored_cpu = restored.detach().cpu()
                coeff_cpu = coeff.detach().cpu()
                for sample_idx, input_path in enumerate(input_paths):
                    object_path = matched_object_path(Path(str(input_path)), object_dir)
                    if object_path is not None:
                        scale = (
                            float(input_scales[sample_idx])
                            if input_scales is not None
                            else 1.0
                        )
                        target_np = np.maximum(load_image(object_path).astype(np.float32, copy=False), 0) / max(
                            scale, float(np.finfo(np.float32).eps)
                        )
                        restored_np = restored_cpu[sample_idx, 0].numpy()
                        target_np = center_crop_array(target_np, restored_np.shape)
                        totals["object_psnr"] += float(
                            psnr(torch.from_numpy(target_np)[None, None], torch.from_numpy(restored_np)[None, None])
                        )
                        totals["object_ssim"] += float(
                            ssim(torch.from_numpy(target_np)[None, None], torch.from_numpy(restored_np)[None, None])
                        )
                        totals["object_rmse"] += rmse_np(restored_np, target_np)
                        totals["object_pcc"] += pcc_np(restored_np, target_np)
                        totals["object_count"] += 1.0

                    coeff_path = matched_zernike_path(Path(str(input_path)), zernike_dir)
                    true_coeff = load_zernike_coefficients(coeff_path) if coeff_path is not None else None
                    if true_coeff is not None:
                        pred_coeff = coeff_cpu[sample_idx].numpy()
                        n = min(pred_coeff.shape[0], true_coeff.shape[0])
                        diff = pred_coeff[:n] - true_coeff[:n]
                        totals["zernike_mae"] += float(np.mean(np.abs(diff)))
                        totals["zernike_rmse"] += float(np.sqrt(np.mean(diff**2)))
                        totals["zernike_count"] += 1.0
    averaged = {k: v / max(1, len(loader)) for k, v in totals.items()}
    if not train:
        object_count = max(1.0, totals.get("object_count", 0.0))
        zernike_count = max(1.0, totals.get("zernike_count", 0.0))
        for key in ("object_psnr", "object_ssim", "object_rmse", "object_pcc"):
            if key in totals:
                averaged[key] = totals[key] / object_count
        for key in ("zernike_mae", "zernike_rmse"):
            if key in totals:
                averaged[key] = totals[key] / zernike_count
        averaged["object_count"] = totals.get("object_count", 0.0)
        averaged["zernike_count"] = totals.get("zernike_count", 0.0)
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a self-supervised 2-D SCARE AO model.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--data-root", default=None, help="Dataset root. Overrides data.root/data.data_root and AO2D_DATA_ROOT.")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from. Use 'auto' to load last.pt from the output directory.")
    parser.add_argument(
        "--accumulate-steps",
        type=int,
        default=None,
        help="Number of micro-batches to accumulate before one optimizer step. Overrides training.accumulate_steps.",
    )
    args = parser.parse_args()

    ctx = setup_distributed()
    config = read_config(args.config)
    if args.accumulate_steps is not None:
        config.setdefault("training", {})["accumulate_steps"] = args.accumulate_steps
    data_root = get_data_root(config, args.data_root)
    output_dir = Path(args.output or config.get("output_dir", "outputs/self_supervised"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.is_main:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    device = ctx.device
    model = make_model(config["model"]).to(device)
    model = wrap_ddp(model, ctx)
    optimizer_G = build_optimizer(model.parameters(), config["training"], prefix="G")
    iterations = int(config["training"].get("iterations", 300000))
    chk_iter = int(config["training"].get("chk_iter", config["training"].get("checkpoint_interval", 500)))
    model_chk_iter = int(config["training"].get("model_chk_iter", chk_iter))
    accumulate_steps = max(1, int(config["training"].get("accumulate_steps", 1)))
    scheduler_G = build_scheduler(optimizer_G, config["training"], total_epochs=max(1, iterations // max(1, chk_iter)), prefix="G")
    train_data_cfg = config["data"].get("train", {})
    use_discriminator = adversarial_weight(config) > 0 and "object_dir" in train_data_cfg
    discriminator = Discriminator2D(in_channels=int(config["model"].get("out_channels", 1))).to(device) if use_discriminator else None
    if discriminator is not None:
        discriminator = wrap_ddp(discriminator, ctx)
        optimizer_D = build_optimizer(discriminator.parameters(), config["training"], prefix="D")
        scheduler_D = build_scheduler(optimizer_D, config["training"], total_epochs=max(1, iterations // max(1, chk_iter)), prefix="D")
    else:
        optimizer_D = None
        scheduler_D = None

    image_size = tuple(config["data"].get("patch_size", [256, 256]))
    zernike_indices = tuple(config.get("optics", {}).get("zernike_indices", list(range(3, 16))))
    forward_model = AO2DForwardModel(image_size, zernike_indices, make_optics_config(config)).to(device)

    train_set = make_self_dataset(config, "train", data_root, augment_default=True)
    val_set = make_self_dataset(config, "val", data_root) if "val" in config["data"] else None
    train_sampler = make_sampler(train_set, ctx, shuffle=True)
    val_sampler = make_sampler(val_set, ctx, shuffle=False)
    training = config["training"]
    needs_object_cycle = (
        training_float(training, "cycle_object_weight", ("pha_coefficient",)) > 0
        or training_float(training, "cycle_object_feature_weight", ("feature_pha_coefficient",)) > 0
        or training_float(training, "zernike_coeff_l1_weight", ("cycle_aberration_weight", "aberration_coefficient")) > 0
        or training_float(training, "phase_loss_weight") > 0
        or training_float(training, "phase_grad_loss_weight") > 0
    )
    self_norm_kwargs = dict(
        normalization_mode=str(train_data_cfg.get("normalization_mode", config["data"].get("normalization_mode", "input_scale"))),
        input_scale_method=str(train_data_cfg.get("input_scale_method", config["data"].get("input_scale_method", "percentile"))),
        input_scale_percentile=float(
            train_data_cfg.get("input_scale_percentile", config["data"].get("input_scale_percentile", 99.9))
        ),
    )
    object_set = (
        AO2DSelfDataset(
            resolve_path(train_data_cfg["object_dir"], data_root),
            patch_size=tuple(config["data"].get("patch_size", [256, 256])),
            augment=bool(train_data_cfg.get("object_augment", train_data_cfg.get("augment", True))),
            samples_per_epoch=train_data_cfg.get("object_samples_per_epoch", train_data_cfg.get("samples_per_epoch")),
            **self_norm_kwargs,
        )
        if "object_dir" in train_data_cfg and (use_discriminator or needs_object_cycle)
        else None
    )
    object_sampler = make_sampler(object_set, ctx, shuffle=True)
    identity_dir = train_data_cfg.get("identity_dir", train_data_cfg.get("noabe_dir"))
    identity_weight = float(config["training"].get("identity_weight", config["training"].get("identity_coefficient", 0.0)))
    identity_set = (
        AO2DSelfDataset(
            resolve_path(identity_dir, data_root),
            patch_size=tuple(config["data"].get("patch_size", [256, 256])),
            augment=bool(train_data_cfg.get("identity_augment", train_data_cfg.get("augment", True))),
            samples_per_epoch=train_data_cfg.get("identity_samples_per_epoch", train_data_cfg.get("samples_per_epoch")),
            **self_norm_kwargs,
        )
        if identity_dir and identity_weight > 0
        else None
    )
    identity_sampler = make_sampler(identity_set, ctx, shuffle=True)
    train_loader = build_dataloader(
        train_set,
        int(config["training"].get("batch_size", 4)),
        train_sampler is None,
        int(config["training"].get("num_workers", 4)),
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = (
        build_dataloader(
            val_set,
            int(config["training"].get("batch_size", 4)),
            False,
            int(config["training"].get("num_workers", 4)),
            sampler=val_sampler,
            drop_last=False,
        )
        if val_set
        else None
    )
    object_loader = (
        build_dataloader(
            object_set,
            int(config["training"].get("batch_size", 4)),
            object_sampler is None,
            int(config["training"].get("num_workers", 4)),
            sampler=object_sampler,
            drop_last=True,
        )
        if object_set
        else None
    )
    identity_loader = (
        build_dataloader(
            identity_set,
            int(config["training"].get("batch_size", 4)),
            identity_sampler is None,
            int(config["training"].get("num_workers", 4)),
            sampler=identity_sampler,
            drop_last=True,
        )
        if identity_set
        else None
    )

    metrics_dir = output_dir / "metrics"
    generated_dir = output_dir / "generated"
    train_metrics_path = metrics_dir / "train_metrics"
    val_metrics_path = metrics_dir / "val_metrics"
    train_rows = read_metrics_xlsx(train_metrics_path.with_suffix(".xlsx")) if args.resume else []
    val_rows = read_metrics_xlsx(val_metrics_path.with_suffix(".xlsx")) if args.resume else []
    add_metric_aliases(train_rows)
    best = min(metric_values(val_rows, "loss"), default=float("inf"))
    start_iteration = 1
    loss_history = {
        "G_adversarial_loss": [],
        "D_adversarial_loss": [],
        "G_cycle_aberrated_image_feature_loss": [],
        "G_cycle_aberrated_image_l1_loss": [],
        "G_cycle_object_loss": [],
        "G_zernike_coeff_l1_loss": [],
        "G_phase_loss": [],
        "G_phase_grad_loss": [],
        "G_identity_clean_loss": [],
        "G_random_reaberration_object_loss": [],
        "G_random_reaberration_coeff_loss": [],
    }

    if args.resume:
        resume_path, start_iteration, checkpoint_best = resume_training(
            args.resume,
            output_dir,
            device,
            model,
            discriminator,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D,
        )
        best = min(best, checkpoint_best)
        if ctx.is_main:
            print(
                f"resumed from {resume_path} at iteration={start_iteration - 1:06d} "
                f"lr_G={get_current_lr(optimizer_G):.6g}"
                + (f" lr_D={get_current_lr(optimizer_D):.6g}" if optimizer_D is not None else "")
            )

    try:
        train_iter = iter(train_loader)
        object_iter = iter(object_loader) if object_loader is not None else None
        identity_iter = iter(identity_loader) if identity_loader is not None else None
        metric_sums: dict[str, float] = {}
        interval_start = time.time()
        sampler_epoch = start_iteration
        set_sampler_epoch(train_sampler, sampler_epoch)
        set_sampler_epoch(object_sampler, sampler_epoch)
        set_sampler_epoch(identity_sampler, sampler_epoch)

        for iteration in tqdm(
            range(start_iteration, iterations + 1),
            initial=start_iteration - 1,
            total=iterations,
            desc="Training",
            disable=not ctx.is_main,
        ):
            micro_metric_sums: dict[str, float] = {}
            for micro_step in range(accumulate_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    sampler_epoch += 1
                    set_sampler_epoch(train_sampler, sampler_epoch)
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                is_first_micro = micro_step == 0
                is_last_micro = micro_step == accumulate_steps - 1
                micro_metrics, object_iter, identity_iter = train_iteration(
                    model,
                    forward_model,
                    batch,
                    optimizer_G,
                    device,
                    config,
                    discriminator=discriminator,
                    optimizer_D=optimizer_D,
                    object_loader=object_loader,
                    object_iter=object_iter,
                    identity_loader=identity_loader,
                    identity_iter=identity_iter,
                    zero_grad=is_first_micro,
                    step_optimizer=is_last_micro,
                    sync_grad=is_last_micro,
                    loss_scale=float(accumulate_steps),
                )
                add_metric_sums(micro_metric_sums, micro_metrics)

            metrics = average_metric_sums(micro_metric_sums, accumulate_steps)
            metrics = reduce_metrics(metrics, ctx)
            add_metric_sums(metric_sums, metrics)

            if iteration % chk_iter != 0:
                continue

            train_metrics = average_metric_sums(metric_sums, chk_iter)
            step_scheduler(scheduler_G, train_metrics["loss"])
            step_scheduler(scheduler_D, train_metrics["loss"])
            lr = get_current_lr(optimizer_G)
            lr_D = get_current_lr(optimizer_D) if optimizer_D is not None else None
            elapsed = time.time() - interval_start

            loss_history["G_adversarial_loss"].append(round(train_metrics.get("loss_adv", 0.0), 6))
            loss_history["D_adversarial_loss"].append(round(train_metrics.get("loss_D", 0.0), 6))
            loss_history["G_cycle_aberrated_image_feature_loss"].append(
                round(train_metrics.get("loss_cycle_aberrated_image_feature", 0.0), 6)
            )
            loss_history["G_cycle_aberrated_image_l1_loss"].append(
                round(train_metrics.get("loss_cycle_aberrated_image_l1", 0.0), 6)
            )
            loss_history["G_cycle_object_loss"].append(round(train_metrics.get("loss_cycle_object", 0.0), 6))
            loss_history["G_zernike_coeff_l1_loss"].append(round(train_metrics.get("loss_zernike_coeff_l1", 0.0), 6))
            loss_history["G_phase_loss"].append(round(train_metrics.get("loss_phase", 0.0), 6))
            loss_history["G_phase_grad_loss"].append(round(train_metrics.get("loss_phase_grad", 0.0), 6))
            loss_history["G_identity_clean_loss"].append(round(train_metrics.get("loss_identity", 0.0), 6))
            loss_history["G_random_reaberration_object_loss"].append(
                round(train_metrics.get("loss_random_reaberration_object", 0.0), 6)
            )
            loss_history["G_random_reaberration_coeff_loss"].append(
                round(train_metrics.get("loss_random_reaberration_coeff", 0.0), 6)
            )

            val_metrics = None
            if val_loader is not None and iteration % model_chk_iter == 0:
                val_metrics = run_epoch(
                    model,
                    forward_model,
                    val_loader,
                    optimizer_G,
                    device,
                    config,
                    train=False,
                    show_progress=ctx.is_main,
                )
                val_metrics = reduce_metrics(val_metrics, ctx)
                best = min(best, val_metrics["loss"])

            if ctx.is_main:
                lr_text = f"lr_G={lr:.6g}" + (f" lr_D={lr_D:.6g}" if lr_D is not None else "")
                print(f"iteration={iteration:06d}/{iterations} {lr_text} train={train_metrics}" + (f" val={val_metrics}" if val_metrics is not None else ""))
                train_row = {
                    "iteration": iteration,
                    "epoch": round(iteration / chk_iter, 6),
                    "time_sec": round(elapsed, 4),
                    "G_adversarial_loss": train_metrics.get("loss_adv"),
                    "D_adversarial_loss": train_metrics.get("loss_D"),
                    "cycle_aberrated_image_feature_loss": train_metrics.get("loss_cycle_aberrated_image_feature"),
                    "cycle_aberrated_image_l1_loss": train_metrics.get("loss_cycle_aberrated_image_l1"),
                    "cycle_object_loss": train_metrics.get("loss_cycle_object"),
                    "zernike_coeff_l1_loss": train_metrics.get("loss_zernike_coeff_l1"),
                    "phase_loss": train_metrics.get("loss_phase"),
                    "phase_grad_loss": train_metrics.get("loss_phase_grad"),
                    "identity_clean_loss": train_metrics.get("loss_identity"),
                    "random_reaberration_object_loss": train_metrics.get("loss_random_reaberration_object"),
                    "random_reaberration_coeff_loss": train_metrics.get("loss_random_reaberration_coeff"),
                    "total_loss": train_metrics.get("loss"),
                    "cycle_psnr": train_metrics.get("cycle_psnr"),
                    "cycle_ssim": train_metrics.get("cycle_ssim"),
                    "grad_norm": train_metrics.get("grad_norm"),
                    "lr_G": lr,
                    "lr_D": lr_D,
                }
                append_metrics_row(train_metrics_path, train_rows, train_row)

                checkpoint_metrics = val_metrics or train_metrics
                ckpt = iteration_checkpoint_state(
                    iteration,
                    model,
                    discriminator,
                    optimizer_G,
                    optimizer_D,
                    scheduler_G,
                    scheduler_D,
                    config,
                    loss_history,
                    checkpoint_metrics,
                )
                iteration_dir = generated_dir / f"iterations_{iteration}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt, iteration_dir / "model.pth")
                if iteration % model_chk_iter == 0:
                    save_validation_figures(
                        model,
                        val_set,
                        forward_model,
                        iteration_dir,
                        config,
                        data_root,
                        device,
                    )
                torch.save(ckpt, output_dir / "last.pt")
                if val_metrics is not None:
                    val_row = {
                        "iteration": iteration,
                        "epoch": round(iteration / chk_iter, 6),
                        "num_samples": len(val_loader.dataset),
                        "loss": val_metrics.get("loss"),
                        "cycle_psnr": val_metrics.get("cycle_psnr"),
                        "cycle_ssim": val_metrics.get("cycle_ssim"),
                        "object_psnr": val_metrics.get("object_psnr"),
                        "object_ssim": val_metrics.get("object_ssim"),
                        "object_rmse": val_metrics.get("object_rmse"),
                        "object_pcc": val_metrics.get("object_pcc"),
                        "zernike_mae": val_metrics.get("zernike_mae"),
                        "zernike_rmse": val_metrics.get("zernike_rmse"),
                        "object_count": val_metrics.get("object_count"),
                        "zernike_count": val_metrics.get("zernike_count"),
                    }
                    append_metrics_row(val_metrics_path, val_rows, val_row)
                    if val_metrics["loss"] <= best:
                        torch.save(ckpt, output_dir / "best.pt")

            metric_sums = {}
            interval_start = time.time()
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
