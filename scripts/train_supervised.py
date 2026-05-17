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

from ao2d.data import AO2DPairDataset, build_dataloader, get_data_root, resolve_path
from ao2d.models.factory import make_model
from ao2d.training import (
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    get_current_lr,
    make_sampler,
    psnr,
    reduce_metrics,
    set_sampler_epoch,
    setup_distributed,
    step_scheduler,
    unwrap_ddp,
    wrap_ddp,
    ssim,
)


def read_config(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return json.load(f)


def make_dataset(config: dict, split: str, data_root: Path | None = None):
    data_cfg = config["data"][split]
    common = dict(
        patch_size=tuple(config["data"].get("patch_size", [256, 256])),
        augment=bool(data_cfg.get("augment", split == "train")),
        samples_per_epoch=data_cfg.get("samples_per_epoch"),
    )
    if "manifest" in data_cfg:
        manifest = resolve_path(data_cfg["manifest"], data_root)
        return AO2DPairDataset.from_manifest(manifest, data_root=data_root, **common)
    return AO2DPairDataset.from_dirs(
        resolve_path(data_cfg["aberrated_dir"], data_root),
        resolve_path(data_cfg["target_dir"], data_root),
        **common,
    )


def run_epoch(model, loader, optimizer, device, train: bool, show_progress: bool):
    model.train(train)
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="train" if train else "val", leave=False, disable=not show_progress):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            loss = F.l1_loss(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["psnr"] += float(psnr(y, pred).detach())
            totals["ssim"] += float(ssim(y, pred).detach())
    return {k: v / max(1, len(loader)) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised 2-D AO restoration model.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--data-root", default=None, help="Dataset root. Overrides data.root/data.data_root and AO2D_DATA_ROOT.")
    args = parser.parse_args()

    ctx = setup_distributed()
    config = read_config(args.config)
    data_root = get_data_root(config, args.data_root)
    output_dir = Path(args.output or config.get("output_dir", "outputs/supervised"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.is_main:
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    device = ctx.device
    model = make_model(config["model"]).to(device)
    model = wrap_ddp(model, ctx)
    optimizer = build_optimizer(model.parameters(), config["training"])
    epochs = int(config["training"].get("epochs", 100))
    scheduler = build_scheduler(optimizer, config["training"], total_epochs=epochs)

    train_set = make_dataset(config, "train", data_root)
    val_set = make_dataset(config, "val", data_root) if "val" in config["data"] else None
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
        for epoch in range(1, epochs + 1):
            set_sampler_epoch(train_sampler, epoch)
            set_sampler_epoch(val_sampler, epoch)
            train_metrics = run_epoch(model, train_loader, optimizer, device, train=True, show_progress=ctx.is_main)
            val_metrics = run_epoch(model, val_loader, optimizer, device, train=False, show_progress=ctx.is_main) if val_loader else train_metrics
            train_metrics = reduce_metrics(train_metrics, ctx)
            val_metrics = reduce_metrics(val_metrics, ctx)
            step_scheduler(scheduler, val_metrics["loss"])
            lr = get_current_lr(optimizer)
            if ctx.is_main:
                print(f"epoch={epoch:04d} lr={lr:.6g} train={train_metrics} val={val_metrics}")

                ckpt = {
                    "epoch": epoch,
                    "model": unwrap_ddp(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
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
