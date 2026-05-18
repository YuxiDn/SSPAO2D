#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao2d.data import AO2DSelfDataset, build_dataloader, get_data_root, resolve_path
from ao2d.models.picnet2d import AberrationGenerator2D, OBJGenerator2D
from ao2d.optics import AO2DConfig, random_zernike_coefficients
from ao2d.training.epoch_metrics import write_metrics_xlsx
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
    set_optimizer_lr,
    setup_distributed,
    step_scheduler,
    ssim,
    total_variation_2d,
    unwrap_ddp,
    wrap_ddp,
)


def read_config(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return json.load(f)


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


def sample_coefficients(batch_size: int, config: dict, device: torch.device) -> torch.Tensor:
    optics = config.get("optics", {})
    training = config.get("training", {})
    zernike_indices = tuple(optics.get("zernike_indices", list(range(3, 16))))
    rms_range = training.get("synthetic_rms_waves_range", [0.02, 0.30])
    coeffs = []
    for _ in range(batch_size):
        rms = random.uniform(float(rms_range[0]), float(rms_range[1]))
        coeffs.append(
            random_zernike_coefficients(
                zernike_indices,
                rms_waves=rms,
                wavelength=float(optics.get("lambda_emission", 1.000)),
                device=device,
            )
        )
    return torch.stack(coeffs, dim=0)


def safe_intensity(x: torch.Tensor, use_sqrt: bool) -> torch.Tensor:
    if not use_sqrt:
        return x
    return torch.sqrt(torch.clamp(x, min=1e-8))


def make_models(config: dict, device: torch.device) -> tuple[OBJGenerator2D, AberrationGenerator2D]:
    model_cfg = config.get("model", {})
    zernike_modes = len(config.get("optics", {}).get("zernike_indices", list(range(3, 16))))
    obj_net = OBJGenerator2D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        final_activation=str(model_cfg.get("final_activation", "sigmoid")),
    )
    coeff_net = AberrationGenerator2D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("zernike_modes", zernike_modes)),
        base_channels=int(model_cfg.get("aberration_base_channels", 32)),
    )
    return obj_net.to(device), coeff_net.to(device)


def train_stage1_epoch(obj_net, coeff_net, forward_model, obj_loader, optimizer, device, config, show_progress: bool):
    obj_net.train()
    coeff_net.train()
    weights = config["training"]
    use_sqrt = bool(weights.get("sqrt_intensity_loss", False))
    totals = {"loss": 0.0, "loss_cycle": 0.0, "loss_object": 0.0, "loss_coeff": 0.0, "grad_norm": 0.0}

    for batch in tqdm(obj_loader, desc="stage1", leave=False, disable=not show_progress):
        obj = batch["input"].to(device, non_blocking=True)
        coeff_gt = sample_coefficients(obj.shape[0], config, device)
        synth_abe = forward_model(obj, coeff_gt).detach()

        pred_obj = obj_net(synth_abe)
        pred_coeff = coeff_net(synth_abe)
        synth_hat = forward_model(pred_obj, pred_coeff)

        loss_cycle = F.l1_loss(safe_intensity(synth_hat, use_sqrt), safe_intensity(synth_abe, use_sqrt))
        loss_object = F.l1_loss(pred_obj, obj)
        loss_coeff = F.mse_loss(pred_coeff, coeff_gt)
        loss = (
            float(weights.get("w_cycle_stage1", 1.0)) * loss_cycle
            + float(weights.get("w_object_stage1", 1.0)) * loss_object
            + float(weights.get("w_coeff_stage1", 0.5)) * loss_coeff
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        totals["grad_norm"] += grad_norm(list(obj_net.parameters()) + list(coeff_net.parameters()))
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["loss_cycle"] += float(loss_cycle.detach())
        totals["loss_object"] += float(loss_object.detach())
        totals["loss_coeff"] += float(loss_coeff.detach())

    return {k: v / max(1, len(obj_loader)) for k, v in totals.items()}


def train_stage2_epoch(obj_net, coeff_net, forward_model, real_loader, obj_loader, optimizer, device, config, show_progress: bool):
    obj_net.train()
    coeff_net.train()
    obj_iter = iter(obj_loader)
    weights = config["training"]
    use_sqrt = bool(weights.get("sqrt_intensity_loss", False))
    tv_weight = float(weights.get("tv_weight", 1e-5))
    coeff_l2 = float(weights.get("coeff_l2", 1e-4))
    totals = {
        "loss": 0.0,
        "loss_real_cycle": 0.0,
        "loss_synth_cycle": 0.0,
        "loss_synth_object": 0.0,
        "loss_synth_coeff": 0.0,
        "grad_norm": 0.0,
    }

    for real_batch in tqdm(real_loader, desc="stage2", leave=False, disable=not show_progress):
        real_abe = real_batch["input"].to(device, non_blocking=True)
        try:
            obj_batch = next(obj_iter)
        except StopIteration:
            obj_iter = iter(obj_loader)
            obj_batch = next(obj_iter)
        obj = obj_batch["input"].to(device, non_blocking=True)

        pred_real_obj = obj_net(real_abe)
        pred_real_coeff = coeff_net(real_abe)
        real_hat = forward_model(pred_real_obj, pred_real_coeff)
        loss_real_cycle = F.l1_loss(safe_intensity(real_hat, use_sqrt), safe_intensity(real_abe, use_sqrt))

        coeff_gt = sample_coefficients(obj.shape[0], config, device)
        synth_abe = forward_model(obj, coeff_gt).detach()
        pred_synth_obj = obj_net(synth_abe)
        pred_synth_coeff = coeff_net(synth_abe)
        synth_hat = forward_model(pred_synth_obj, pred_synth_coeff)
        loss_synth_cycle = F.l1_loss(safe_intensity(synth_hat, use_sqrt), safe_intensity(synth_abe, use_sqrt))
        loss_synth_object = F.l1_loss(pred_synth_obj, obj)
        loss_synth_coeff = F.mse_loss(pred_synth_coeff, coeff_gt)

        loss = (
            float(weights.get("w_real_cycle_stage2", 1.0)) * loss_real_cycle
            + float(weights.get("w_synth_cycle_stage2", 1.0)) * loss_synth_cycle
            + float(weights.get("w_object_stage2", 0.5)) * loss_synth_object
            + float(weights.get("w_coeff_stage2", 0.05)) * loss_synth_coeff
            + tv_weight * total_variation_2d(pred_real_obj)
            + coeff_l2 * torch.mean(pred_real_coeff**2)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        totals["grad_norm"] += grad_norm(list(obj_net.parameters()) + list(coeff_net.parameters()))
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["loss_real_cycle"] += float(loss_real_cycle.detach())
        totals["loss_synth_cycle"] += float(loss_synth_cycle.detach())
        totals["loss_synth_object"] += float(loss_synth_object.detach())
        totals["loss_synth_coeff"] += float(loss_synth_coeff.detach())

    return {k: v / max(1, len(real_loader)) for k, v in totals.items()}


def validate(obj_net, coeff_net, forward_model, loader, device, show_progress: bool):
    if loader is None:
        return None
    obj_net.eval()
    coeff_net.eval()
    totals = {"loss": 0.0, "cycle_psnr": 0.0, "cycle_ssim": 0.0}
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False, disable=not show_progress):
            x = batch["input"].to(device, non_blocking=True)
            pred_obj = obj_net(x)
            pred_coeff = coeff_net(x)
            x_hat = forward_model(pred_obj, pred_coeff)
            loss = F.l1_loss(x_hat, x)
            totals["loss"] += float(loss)
            totals["cycle_psnr"] += float(psnr(x, x_hat))
            totals["cycle_ssim"] += float(ssim(x, x_hat))
    return {k: v / max(1, len(loader)) for k, v in totals.items()}


def save_checkpoint(path: Path, epoch: int, obj_net, coeff_net, optimizer, scheduler, config: dict, metrics: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "object_generator": unwrap_ddp(obj_net).state_dict(),
            "aberration_generator": unwrap_ddp(coeff_net).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def metrics_row(stage: int, epoch: int, lr: float, train_metrics: dict, val_metrics: dict | None = None) -> dict[str, object]:
    row = {"stage": stage, "epoch": epoch, "lr": lr}
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    if val_metrics is not None:
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
    return row


def load_stage1(path: str | Path, obj_net, coeff_net, optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    obj_net.load_state_dict(ckpt["object_generator"])
    coeff_net.load_state_dict(ckpt["aberration_generator"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage 2-D PICNet-style AO training.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--stage", choices=["stage1", "stage2", "both"], default="both")
    parser.add_argument("--resume_stage1", default=None)
    parser.add_argument("--data-root", default=None, help="Dataset root. Overrides data.root/data.data_root and AO2D_DATA_ROOT.")
    args = parser.parse_args()

    ctx = setup_distributed()
    config = read_config(args.config)
    data_root = get_data_root(config, args.data_root)
    output_dir = Path(args.output or config.get("output_dir", "outputs/two_stage"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.is_main:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    device = ctx.device
    obj_net, coeff_net = make_models(config, device)
    if args.resume_stage1:
        start_epoch = load_stage1(args.resume_stage1, obj_net, coeff_net)
        if ctx.is_main:
            print(f"Loaded stage-1 checkpoint from epoch {start_epoch}: {args.resume_stage1}")
    obj_net = wrap_ddp(obj_net, ctx)
    coeff_net = wrap_ddp(coeff_net, ctx)
    params = list(obj_net.parameters()) + list(coeff_net.parameters())
    optimizer = build_optimizer(params, config["training"], prefix="stage1")

    patch_size = tuple(config["data"].get("patch_size", [256, 256]))
    zernike_indices = tuple(config.get("optics", {}).get("zernike_indices", list(range(3, 16))))
    forward_model = AO2DForwardModel(patch_size, zernike_indices, make_optics_config(config)).to(device)

    obj_set = AO2DSelfDataset(
        resolve_path(config["data"]["train"]["object_dir"], data_root),
        patch_size=patch_size,
        augment=bool(config["data"]["train"].get("augment", True)),
        samples_per_epoch=config["data"]["train"].get("object_samples_per_epoch"),
    )
    real_set = AO2DSelfDataset(
        resolve_path(config["data"]["train"]["aberrated_dir"], data_root),
        patch_size=patch_size,
        augment=bool(config["data"]["train"].get("augment", True)),
        samples_per_epoch=config["data"]["train"].get("aberrated_samples_per_epoch"),
    )
    val_set = (
        AO2DSelfDataset(
            resolve_path(config["data"]["val"]["aberrated_dir"], data_root),
            patch_size=patch_size,
            augment=False,
            samples_per_epoch=config["data"]["val"].get("samples_per_epoch"),
        )
        if "val" in config["data"]
        else None
    )

    batch_size = int(config["training"].get("batch_size", 4))
    num_workers = int(config["training"].get("num_workers", 4))
    obj_sampler = make_sampler(obj_set, ctx, shuffle=True)
    real_sampler = make_sampler(real_set, ctx, shuffle=True)
    val_sampler = make_sampler(val_set, ctx, shuffle=False)
    obj_loader = build_dataloader(obj_set, batch_size, obj_sampler is None, num_workers, sampler=obj_sampler, drop_last=True)
    real_loader = build_dataloader(real_set, batch_size, real_sampler is None, num_workers, sampler=real_sampler, drop_last=True)
    val_loader = build_dataloader(val_set, batch_size, False, num_workers, sampler=val_sampler, drop_last=False) if val_set else None

    best = float("inf")
    epoch_rows = []
    metrics_path = output_dir / "metrics.xlsx"
    try:
        if args.stage in {"stage1", "both"}:
            epochs_stage1 = int(config["training"].get("epochs_stage1", 50))
            scheduler_stage1 = build_scheduler(optimizer, config["training"], total_epochs=epochs_stage1, prefix="stage1")
            for epoch in range(1, epochs_stage1 + 1):
                set_sampler_epoch(obj_sampler, epoch)
                metrics = train_stage1_epoch(obj_net, coeff_net, forward_model, obj_loader, optimizer, device, config, show_progress=ctx.is_main)
                metrics = reduce_metrics(metrics, ctx)
                step_scheduler(scheduler_stage1, metrics["loss"])
                lr = get_current_lr(optimizer)
                if ctx.is_main:
                    print(f"stage=1 epoch={epoch:04d} lr={lr:.6g} train={metrics}")
                    epoch_rows.append(metrics_row(1, epoch, lr, metrics))
                    write_metrics_xlsx(metrics_path, epoch_rows)
                    save_checkpoint(output_dir / "stage1_last.pt", epoch, obj_net, coeff_net, optimizer, scheduler_stage1, config, metrics)
                    if metrics["loss"] < best:
                        best = metrics["loss"]
                        save_checkpoint(output_dir / "stage1_best.pt", epoch, obj_net, coeff_net, optimizer, scheduler_stage1, config, metrics)

        if args.stage in {"stage2", "both"}:
            stage1_lr = float(config["training"].get("lr_stage1", config["training"].get("initial_learning_rate", config["training"].get("lr", 1e-4))))
            stage2_lr = float(config["training"].get("lr_stage2", stage1_lr * float(config["training"].get("stage2_lr_scale", 0.1))))
            set_optimizer_lr(optimizer, stage2_lr)
            best = float("inf")
            epochs_stage2 = int(config["training"].get("epochs_stage2", 100))
            scheduler_stage2 = build_scheduler(optimizer, config["training"], total_epochs=epochs_stage2, prefix="stage2")
            for epoch in range(1, epochs_stage2 + 1):
                set_sampler_epoch(obj_sampler, epoch)
                set_sampler_epoch(real_sampler, epoch)
                set_sampler_epoch(val_sampler, epoch)
                train_metrics = train_stage2_epoch(obj_net, coeff_net, forward_model, real_loader, obj_loader, optimizer, device, config, show_progress=ctx.is_main)
                val_metrics = validate(obj_net, coeff_net, forward_model, val_loader, device, show_progress=ctx.is_main) or train_metrics
                train_metrics = reduce_metrics(train_metrics, ctx)
                val_metrics = reduce_metrics(val_metrics, ctx)
                step_scheduler(scheduler_stage2, val_metrics["loss"])
                lr = get_current_lr(optimizer)
                if ctx.is_main:
                    print(f"stage=2 epoch={epoch:04d} lr={lr:.6g} train={train_metrics} val={val_metrics}")
                    epoch_rows.append(metrics_row(2, epoch, lr, train_metrics, val_metrics))
                    write_metrics_xlsx(metrics_path, epoch_rows)
                    save_checkpoint(output_dir / "stage2_last.pt", epoch, obj_net, coeff_net, optimizer, scheduler_stage2, config, val_metrics)
                    if val_metrics["loss"] < best:
                        best = val_metrics["loss"]
                        save_checkpoint(output_dir / "stage2_best.pt", epoch, obj_net, coeff_net, optimizer, scheduler_stage2, config, val_metrics)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
