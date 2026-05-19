#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao2d.data import AO2DSelfDataset, build_dataloader, get_data_root, resolve_path
from ao2d.models.factory import make_model
from ao2d.models.picnet2d import Discriminator2D
from ao2d.optics import AO2DConfig
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
    setup_distributed,
    step_scheduler,
    ssim,
    total_variation_2d,
    unwrap_ddp,
    wrap_ddp,
)


def adversarial_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


def adversarial_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(-fake_logits).mean()


def set_requires_grad(module, requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(requires_grad)


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


def next_object_batch(object_iter, object_loader, device):
    if object_loader is None:
        return None, object_iter
    try:
        batch = next(object_iter)
    except StopIteration:
        object_iter = iter(object_loader)
        batch = next(object_iter)
    return batch["input"].to(device, non_blocking=True), object_iter


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
):
    model.train(train)
    if discriminator is not None:
        discriminator.train(train)
    totals = {"loss": 0.0, "cycle_psnr": 0.0, "cycle_ssim": 0.0}
    if train:
        totals["grad_norm"] = 0.0
    if train and discriminator is not None:
        totals["loss_adv"] = 0.0
        totals["loss_D"] = 0.0
    tv_weight = float(config["training"].get("tv_weight", 1e-5))
    coeff_l2 = float(config["training"].get("coeff_l2", 1e-4))
    adv_weight = float(config["training"].get("adversarial_weight", config["training"].get("adv_coefficient", 0.0)))
    object_iter = iter(object_loader) if object_loader is not None else None
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="train" if train else "val", leave=False, disable=not show_progress):
            x = batch["input"].to(device, non_blocking=True)
            output = model(x)
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("Self-supervised AO training requires a model that returns (restored, zernike_coeff), such as scare2d or picnet2d.")
            restored, coeff = output
            estimated = forward_model(restored, coeff)
            loss = F.l1_loss(estimated, x) + tv_weight * total_variation_2d(restored) + coeff_l2 * torch.mean(coeff**2)
            if train:
                if discriminator is not None and optimizer_D is not None and adv_weight > 0:
                    real_obj, object_iter = next_object_batch(object_iter, object_loader, device)
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
            totals["loss"] += float(loss.detach())
            totals["cycle_psnr"] += float(psnr(x, estimated).detach())
            totals["cycle_ssim"] += float(ssim(x, estimated).detach())
    return {k: v / max(1, len(loader)) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a self-supervised 2-D SCARE AO model.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--data-root", default=None, help="Dataset root. Overrides data.root/data.data_root and AO2D_DATA_ROOT.")
    args = parser.parse_args()

    ctx = setup_distributed()
    config = read_config(args.config)
    data_root = get_data_root(config, args.data_root)
    output_dir = Path(args.output or config.get("output_dir", "outputs/self_supervised"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.is_main:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    device = ctx.device
    model = make_model(config["model"]).to(device)
    model = wrap_ddp(model, ctx)
    optimizer_G = build_optimizer(model.parameters(), config["training"], prefix="G")
    epochs = int(config["training"].get("epochs", 100))
    scheduler_G = build_scheduler(optimizer_G, config["training"], total_epochs=epochs, prefix="G")
    use_discriminator = (
        str(config.get("model", {}).get("name", "")).lower() in {"picnet", "picnet2d"}
        and float(config["training"].get("adversarial_weight", config["training"].get("adv_coefficient", 0.0))) > 0
        and "object_dir" in config["data"].get("train", {})
    )
    discriminator = Discriminator2D(in_channels=int(config["model"].get("out_channels", 1))).to(device) if use_discriminator else None
    if discriminator is not None:
        discriminator = wrap_ddp(discriminator, ctx)
        optimizer_D = build_optimizer(discriminator.parameters(), config["training"], prefix="D")
        scheduler_D = build_scheduler(optimizer_D, config["training"], total_epochs=epochs, prefix="D")
    else:
        optimizer_D = None
        scheduler_D = None

    image_size = tuple(config["data"].get("patch_size", [256, 256]))
    zernike_indices = tuple(config.get("optics", {}).get("zernike_indices", list(range(3, 16))))
    forward_model = AO2DForwardModel(image_size, zernike_indices, make_optics_config(config)).to(device)

    train_set = AO2DSelfDataset(
        resolve_path(config["data"]["train"]["image_dir"], data_root),
        patch_size=tuple(config["data"].get("patch_size", [256, 256])),
        augment=bool(config["data"]["train"].get("augment", True)),
        samples_per_epoch=config["data"]["train"].get("samples_per_epoch"),
    )
    val_set = AO2DSelfDataset(
        resolve_path(config["data"]["val"]["image_dir"], data_root),
        patch_size=tuple(config["data"].get("patch_size", [256, 256])),
        augment=False,
        samples_per_epoch=config["data"]["val"].get("samples_per_epoch"),
    ) if "val" in config["data"] else None
    train_sampler = make_sampler(train_set, ctx, shuffle=True)
    val_sampler = make_sampler(val_set, ctx, shuffle=False)
    object_set = (
        AO2DSelfDataset(
            resolve_path(config["data"]["train"]["object_dir"], data_root),
            patch_size=tuple(config["data"].get("patch_size", [256, 256])),
            augment=bool(config["data"]["train"].get("augment", True)),
            samples_per_epoch=config["data"]["train"].get("object_samples_per_epoch", config["data"]["train"].get("samples_per_epoch")),
        )
        if use_discriminator
        else None
    )
    object_sampler = make_sampler(object_set, ctx, shuffle=True)
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

    best = float("inf")
    epoch_rows = []
    metrics_path = output_dir / "metrics.xlsx"
    try:
        for epoch in range(1, epochs + 1):
            set_sampler_epoch(train_sampler, epoch)
            set_sampler_epoch(val_sampler, epoch)
            set_sampler_epoch(object_sampler, epoch)
            train_metrics = run_epoch(
                model,
                forward_model,
                train_loader,
                optimizer_G,
                device,
                config,
                train=True,
                show_progress=ctx.is_main,
                discriminator=discriminator,
                optimizer_D=optimizer_D,
                object_loader=object_loader,
            )
            val_metrics = run_epoch(model, forward_model, val_loader, optimizer_G, device, config, train=False, show_progress=ctx.is_main) if val_loader else train_metrics
            train_metrics = reduce_metrics(train_metrics, ctx)
            val_metrics = reduce_metrics(val_metrics, ctx)
            step_scheduler(scheduler_G, val_metrics["loss"])
            step_scheduler(scheduler_D, val_metrics["loss"])
            lr = get_current_lr(optimizer_G)
            lr_D = get_current_lr(optimizer_D) if optimizer_D is not None else None
            if ctx.is_main:
                lr_text = f"lr_G={lr:.6g}" + (f" lr_D={lr_D:.6g}" if lr_D is not None else "")
                print(f"epoch={epoch:04d} {lr_text} train={train_metrics} val={val_metrics}")
                epoch_rows.append({
                    "epoch": epoch,
                    "lr_G": lr,
                    "lr_D": lr_D,
                    "train_loss": train_metrics.get("loss"),
                    "train_loss_adv": train_metrics.get("loss_adv"),
                    "train_loss_D": train_metrics.get("loss_D"),
                    "train_cycle_psnr": train_metrics.get("cycle_psnr"),
                    "train_cycle_ssim": train_metrics.get("cycle_ssim"),
                    "train_grad_norm": train_metrics.get("grad_norm"),
                    "val_loss": val_metrics.get("loss"),
                    "val_cycle_psnr": val_metrics.get("cycle_psnr"),
                    "val_cycle_ssim": val_metrics.get("cycle_ssim"),
                })
                write_metrics_xlsx(metrics_path, epoch_rows)
                ckpt = {
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
                torch.save(ckpt, output_dir / "last.pt")
                if val_metrics["loss"] < best:
                    best = val_metrics["loss"]
                    torch.save(ckpt, output_dir / "best.pt")
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
