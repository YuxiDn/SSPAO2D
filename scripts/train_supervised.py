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
from ao2d.training.epoch_metrics import read_metrics_xlsx, write_metrics_xlsx
from ao2d.training import (
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
        normalization_mode=str(data_cfg.get("normalization_mode", config["data"].get("normalization_mode", "input_scale"))),
        input_scale_method=str(data_cfg.get("input_scale_method", config["data"].get("input_scale_method", "percentile"))),
        input_scale_percentile=float(
            data_cfg.get("input_scale_percentile", config["data"].get("input_scale_percentile", 99.9))
        ),
    )
    if "normalize_percentile" in config["data"]:
        common["normalize_percentile"] = tuple(config["data"]["normalize_percentile"])
    if "normalize_percentile" in data_cfg:
        common["normalize_percentile"] = tuple(data_cfg["normalize_percentile"])
    for key in ("min_foreground_fraction", "foreground_threshold", "max_patch_tries"):
        if key in data_cfg:
            common[key] = data_cfg[key]
    if "manifest" in data_cfg:
        manifest = resolve_path(data_cfg["manifest"], data_root)
        return AO2DPairDataset.from_manifest(manifest, data_root=data_root, **common)
    return AO2DPairDataset.from_dirs(
        resolve_path(data_cfg["aberrated_dir"], data_root),
        resolve_path(data_cfg["target_dir"], data_root),
        **common,
    )


def supervised_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    name = loss_name.lower()
    if name in {"l1", "mae"}:
        return F.l1_loss(pred, target)
    if name in {"mse", "l2"}:
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported supervised loss: {loss_name}")


def run_epoch(model, loader, optimizer, device, train: bool, show_progress: bool, loss_name: str = "l1"):
    model.train(train)
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "pred_min": 0.0, "pred_max": 0.0, "pred_mean": 0.0}
    if train:
        totals["grad_norm"] = 0.0
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="train" if train else "val", leave=False, disable=not show_progress):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            loss = supervised_loss(pred, y, loss_name)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                totals["grad_norm"] += grad_norm(model.parameters())
                optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["psnr"] += float(psnr(y, pred).detach())
            totals["ssim"] += float(ssim(y, pred).detach())
            pred_detached = pred.detach()
            totals["pred_min"] += float(pred_detached.amin())
            totals["pred_max"] += float(pred_detached.amax())
            totals["pred_mean"] += float(pred_detached.mean())
    return {k: v / max(1, len(loader)) for k, v in totals.items()}


def load_checkpoint_model(path: str | Path, model, device, load_optimizer: bool = False, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location=device)
    unwrap_ddp(model).load_state_dict(ckpt["model"])
    if load_optimizer and optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if load_optimizer and scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised 2-D AO restoration model.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--data-root", default=None, help="Dataset root. Overrides data.root/data.data_root and AO2D_DATA_ROOT.")
    parser.add_argument("--resume", default=None, help="Checkpoint to load model weights from before training.")
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Also restore optimizer and scheduler state from --resume. By default a fresh optimizer/scheduler is used.",
    )
    parser.add_argument(
        "--reset-lr",
        type=float,
        default=None,
        help="Override all optimizer parameter groups to this learning rate after loading a checkpoint.",
    )
    parser.add_argument(
        "--continue-epoch-numbers",
        action="store_true",
        help="Start printed/checkpoint epoch numbers after the checkpoint epoch instead of from 1.",
    )
    parser.add_argument(
        "--append-metrics",
        action="store_true",
        help="Load existing metrics.xlsx from the output directory and append new epochs to it.",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate data.test once after training and write test_metrics.json. Test metrics are not used for checkpoint selection.",
    )
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
    resume_epoch = 0
    if args.resume:
        resume_epoch = load_checkpoint_model(args.resume, model, device, args.resume_optimizer, optimizer, scheduler)
        if args.reset_lr is not None:
            set_optimizer_lr(optimizer, args.reset_lr)
        if ctx.is_main:
            restored = "model, optimizer, and scheduler" if args.resume_optimizer else "model"
            print(f"Loaded {restored} from epoch {resume_epoch}: {args.resume}")

    train_set = make_dataset(config, "train", data_root)
    val_set = make_dataset(config, "val", data_root) if "val" in config["data"] else None
    test_set = make_dataset(config, "test", data_root) if args.eval_test and "test" in config["data"] else None
    train_sampler = make_sampler(train_set, ctx, shuffle=True)
    val_sampler = make_sampler(val_set, ctx, shuffle=False)
    test_sampler = make_sampler(test_set, ctx, shuffle=False)
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
    test_loader = (
        build_dataloader(
            test_set,
            int(config["training"].get("batch_size", 4)),
            False,
            int(config["training"].get("num_workers", 4)),
            sampler=test_sampler,
            drop_last=False,
        )
        if test_set
        else None
    )

    metrics_path = output_dir / "metrics.xlsx"
    epoch_rows = read_metrics_xlsx(metrics_path) if args.append_metrics else []
    best = min(
        (float(row["val_loss"]) for row in epoch_rows if row.get("val_loss") not in {None, ""}),
        default=float("inf"),
    )
    epoch_offset = resume_epoch if args.continue_epoch_numbers else 0
    try:
        for local_epoch in range(1, epochs + 1):
            epoch = epoch_offset + local_epoch
            set_sampler_epoch(train_sampler, epoch)
            set_sampler_epoch(val_sampler, epoch)
            loss_name = str(config["training"].get("loss", config["training"].get("loss_function", "l1")))
            train_metrics = run_epoch(model, train_loader, optimizer, device, train=True, show_progress=ctx.is_main, loss_name=loss_name)
            val_metrics = run_epoch(model, val_loader, optimizer, device, train=False, show_progress=ctx.is_main, loss_name=loss_name) if val_loader else train_metrics
            train_metrics = reduce_metrics(train_metrics, ctx)
            val_metrics = reduce_metrics(val_metrics, ctx)
            step_scheduler(scheduler, val_metrics["loss"])
            lr = get_current_lr(optimizer)
            if ctx.is_main:
                print(f"epoch={epoch:04d} lr={lr:.6g} train={train_metrics} val={val_metrics}")
                epoch_rows.append({
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": train_metrics.get("loss"),
                    "train_psnr": train_metrics.get("psnr"),
                    "train_ssim": train_metrics.get("ssim"),
                    "train_grad_norm": train_metrics.get("grad_norm"),
                    "train_pred_min": train_metrics.get("pred_min"),
                    "train_pred_max": train_metrics.get("pred_max"),
                    "train_pred_mean": train_metrics.get("pred_mean"),
                    "val_loss": val_metrics.get("loss"),
                    "val_psnr": val_metrics.get("psnr"),
                    "val_ssim": val_metrics.get("ssim"),
                    "val_pred_min": val_metrics.get("pred_min"),
                    "val_pred_max": val_metrics.get("pred_max"),
                    "val_pred_mean": val_metrics.get("pred_mean"),
                })
                write_metrics_xlsx(metrics_path, epoch_rows)

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
        if test_loader is not None:
            set_sampler_epoch(test_sampler, epoch_offset + epochs + 1)
            loss_name = str(config["training"].get("loss", config["training"].get("loss_function", "l1")))
            test_metrics = run_epoch(model, test_loader, optimizer, device, train=False, show_progress=ctx.is_main, loss_name=loss_name)
            test_metrics = reduce_metrics(test_metrics, ctx)
            if ctx.is_main:
                print(f"test={test_metrics}")
                (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
