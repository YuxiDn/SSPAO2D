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

from ao2d.data import AO2DPairDataset, build_dataloader
from ao2d.models.factory import make_model
from ao2d.training import psnr, ssim


def read_config(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return json.load(f)


def make_dataset(config: dict, split: str):
    data_cfg = config["data"][split]
    common = dict(
        patch_size=tuple(config["data"].get("patch_size", [256, 256])),
        augment=bool(data_cfg.get("augment", split == "train")),
        samples_per_epoch=data_cfg.get("samples_per_epoch"),
    )
    if "manifest" in data_cfg:
        return AO2DPairDataset.from_manifest(data_cfg["manifest"], **common)
    return AO2DPairDataset.from_dirs(data_cfg["aberrated_dir"], data_cfg["target_dir"], **common)


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
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
    args = parser.parse_args()

    config = read_config(args.config)
    output_dir = Path(args.output or config.get("output_dir", "outputs/supervised"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config["model"]).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["training"].get("lr", 1e-4)), weight_decay=float(config["training"].get("weight_decay", 1e-4)))

    train_set = make_dataset(config, "train")
    val_set = make_dataset(config, "val") if "val" in config["data"] else None
    train_loader = build_dataloader(train_set, int(config["training"].get("batch_size", 4)), True, int(config["training"].get("num_workers", 4)))
    val_loader = build_dataloader(val_set, int(config["training"].get("batch_size", 4)), False, int(config["training"].get("num_workers", 4))) if val_set else None

    best = float("inf")
    epochs = int(config["training"].get("epochs", 100))
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False) if val_loader else train_metrics
        print(f"epoch={epoch:04d} train={train_metrics} val={val_metrics}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val": val_metrics,
        }
        torch.save(ckpt, output_dir / "last.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(ckpt, output_dir / "best.pt")


if __name__ == "__main__":
    main()
