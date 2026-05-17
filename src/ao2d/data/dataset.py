from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .io import IMAGE_EXTENSIONS, load_image, normalize01
from .paths import infer_manifest_data_root, resolve_manifest_record_path


def _valid_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))


def _augment_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.random.randint(0, 4)
    x = np.rot90(x, k=k)
    y = np.rot90(y, k=k)
    if np.random.rand() < 0.5:
        x = np.flip(x, axis=-1)
        y = np.flip(y, axis=-1)
    if np.random.rand() < 0.5:
        x = np.flip(x, axis=-2)
        y = np.flip(y, axis=-2)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _augment_single(x: np.ndarray) -> np.ndarray:
    k = np.random.randint(0, 4)
    x = np.rot90(x, k=k)
    if np.random.rand() < 0.5:
        x = np.flip(x, axis=-1)
    if np.random.rand() < 0.5:
        x = np.flip(x, axis=-2)
    return np.ascontiguousarray(x)


def _random_crop_pair(x: np.ndarray, y: np.ndarray, patch_size: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    if patch_size is None:
        return x, y
    ph, pw = patch_size
    h, w = x.shape
    if h < ph or w < pw:
        raise ValueError(f"Patch size {patch_size} is larger than image shape {x.shape}")
    top = random.randint(0, h - ph)
    left = random.randint(0, w - pw)
    return x[top : top + ph, left : left + pw], y[top : top + ph, left : left + pw]


def _random_crop_single(x: np.ndarray, patch_size: tuple[int, int] | None) -> np.ndarray:
    if patch_size is None:
        return x
    ph, pw = patch_size
    h, w = x.shape
    if h < ph or w < pw:
        raise ValueError(f"Patch size {patch_size} is larger than image shape {x.shape}")
    top = random.randint(0, h - ph)
    left = random.randint(0, w - pw)
    return x[top : top + ph, left : left + pw]


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x))[None].float()


@dataclass(frozen=True)
class PairRecord:
    aberrated: Path
    target: Path
    coeff: Path | None = None


class AO2DPairDataset(Dataset):
    """Paired 2-D aberrated/no-aberration dataset for supervised training."""

    def __init__(
        self,
        records: list[PairRecord],
        patch_size: tuple[int, int] | None = None,
        augment: bool = True,
        samples_per_epoch: int | None = None,
        normalize_percentile: tuple[float, float] | None = (0.1, 99.9),
    ) -> None:
        if not records:
            raise ValueError("AO2DPairDataset received no records.")
        self.records = records
        self.patch_size = patch_size
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        self.normalize_percentile = normalize_percentile

    @classmethod
    def from_dirs(
        cls,
        aberrated_dir: str | Path,
        target_dir: str | Path,
        **kwargs,
    ) -> "AO2DPairDataset":
        target_by_name = {p.name: p for p in _valid_files(target_dir)}
        records = []
        for src in _valid_files(aberrated_dir):
            candidates = [
                src.name,
                src.name.replace("abe", "no_abe"),
                src.name.replace("Aberrated", "GT"),
                src.name.replace("aberrated", "target"),
            ]
            target = next((target_by_name[name] for name in candidates if name in target_by_name), None)
            if target is not None:
                records.append(PairRecord(src, target))
        return cls(records, **kwargs)

    @classmethod
    def from_manifest(cls, manifest: str | Path, data_root: str | Path | None = None, **kwargs) -> "AO2DPairDataset":
        records: list[PairRecord] = []
        manifest = Path(manifest)
        root = infer_manifest_data_root(manifest, data_root)
        with Path(manifest).open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get("abe_path") or row.get("aberrated") or row.get("input")
                tgt = row.get("no_abe_path") or row.get("target") or row.get("gt")
                coeff = row.get("zernike_path") or row.get("coeff_path") or None
                if not src or not tgt:
                    continue
                records.append(
                    PairRecord(
                        resolve_manifest_record_path(src, root),
                        resolve_manifest_record_path(tgt, root),
                        resolve_manifest_record_path(coeff, root) if coeff else None,
                    )
                )
        return cls(records, **kwargs)

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index % len(self.records)]
        x = normalize01(load_image(record.aberrated), self.normalize_percentile)
        y = normalize01(load_image(record.target), self.normalize_percentile)
        x, y = _random_crop_pair(x, y, self.patch_size)
        if self.augment:
            x, y = _augment_pair(x, y)
        return {
            "input": _to_tensor(x),
            "target": _to_tensor(y),
            "input_path": str(record.aberrated),
            "target_path": str(record.target),
        }


class AO2DSelfDataset(Dataset):
    """Single-image dataset for self-supervised AO consistency training."""

    def __init__(
        self,
        image_dir: str | Path,
        patch_size: tuple[int, int] | None = None,
        augment: bool = True,
        samples_per_epoch: int | None = None,
        normalize_percentile: tuple[float, float] | None = (0.1, 99.9),
    ) -> None:
        self.files = _valid_files(image_dir)
        if not self.files:
            raise ValueError(f"No images found in {image_dir}")
        self.patch_size = patch_size
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        self.normalize_percentile = normalize_percentile

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index % len(self.files)]
        x = normalize01(load_image(path), self.normalize_percentile)
        x = _random_crop_single(x, self.patch_size)
        if self.augment:
            x = _augment_single(x)
        return {"input": _to_tensor(x), "input_path": str(path)}


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    sampler: Sampler | None = None,
    drop_last: bool | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle if drop_last is None else drop_last,
    )
