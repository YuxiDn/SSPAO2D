#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao2d.data import AO2DSelfDataset, build_dataloader, get_data_root, resolve_path
from ao2d.models.factory import make_model
from ao2d.optics import AO2DConfig
from ao2d.training import (
    AO2DForwardModel,
    cleanup_distributed,
    make_sampler,
    psnr,
    reduce_metrics,
    set_sampler_epoch,
    setup_distributed,
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


def run_epoch(model, forward_model, loader, optimizer, device, config, train: bool, show_progress: bool):
    model.train(train)
    totals = {"loss": 0.0, "cycle_psnr": 0.0, "cycle_ssim": 0.0}
    tv_weight = float(config["training"].get("tv_weight", 1e-5))
    coeff_l2 = float(config["training"].get("coeff_l2", 1e-4))
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
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
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
    optimizer = AdamW(model.parameters(), lr=float(config["training"].get("lr", 1e-4)), weight_decay=float(config["training"].get("weight_decay", 1e-4)))

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

    best = float("inf")
    try:
        for epoch in range(1, int(config["training"].get("epochs", 100)) + 1):
            set_sampler_epoch(train_sampler, epoch)
            set_sampler_epoch(val_sampler, epoch)
            train_metrics = run_epoch(model, forward_model, train_loader, optimizer, device, config, train=True, show_progress=ctx.is_main)
            val_metrics = run_epoch(model, forward_model, val_loader, optimizer, device, config, train=False, show_progress=ctx.is_main) if val_loader else train_metrics
            train_metrics = reduce_metrics(train_metrics, ctx)
            val_metrics = reduce_metrics(val_metrics, ctx)
            if ctx.is_main:
                print(f"epoch={epoch:04d} train={train_metrics} val={val_metrics}")
                ckpt = {"epoch": epoch, "model": unwrap_ddp(model).state_dict(), "optimizer": optimizer.state_dict(), "config": config, "val": val_metrics}
                torch.save(ckpt, output_dir / "last.pt")
                if val_metrics["loss"] < best:
                    best = val_metrics["loss"]
                    torch.save(ckpt, output_dir / "best.pt")
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
