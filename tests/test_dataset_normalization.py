from __future__ import annotations

import csv

import numpy as np
import torch

from ao2d.data.dataset import AO2DPairDataset, AO2DSelfDataset, _normalize_pair_by_input_scale


def test_normalize_pair_uses_input_scale_and_allows_target_above_one():
    x = np.array([[0, 1], [2, 4]], dtype=np.float32)
    y = np.array([[0, 2], [4, 8]], dtype=np.float32)

    x_norm, y_norm, scale = _normalize_pair_by_input_scale(x, y, method="max")

    assert scale == 4
    assert np.max(x_norm) == 1
    assert np.max(y_norm) == 2


def test_pair_dataset_input_scale_ignores_manifest_scale(tmp_path):
    abe = np.array([[0, 1], [2, 4]], dtype=np.float32)
    no_abe = np.array([[0, 2], [4, 8]], dtype=np.float32)
    np.save(tmp_path / "abe.npy", abe)
    np.save(tmp_path / "no_abe.npy", no_abe)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "abe_path",
                "no_abe_path",
                "training_normalization_scale",
                "output_intensity_units",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "abe_path": "abe.npy",
                "no_abe_path": "no_abe.npy",
                "training_normalization_scale": "100",
                "output_intensity_units": "uint16_raw",
            }
        )

    dataset = AO2DPairDataset.from_manifest(
        manifest,
        patch_size=None,
        augment=False,
        normalization_mode="input_scale",
        input_scale_method="max",
    )
    assert dataset.records[0].training_normalization_scale == 100
    assert dataset.records[0].output_intensity_units == "uint16_raw"

    sample = dataset[0]
    assert torch.isclose(sample["input_scale"], torch.tensor(4.0))
    assert torch.isclose(sample["input"].amax(), torch.tensor(1.0))
    assert torch.isclose(sample["target"].amax(), torch.tensor(2.0))


def test_pair_dataset_manifest_scale_is_explicit_mode(tmp_path):
    abe = np.array([[0, 1], [2, 4]], dtype=np.float32)
    no_abe = np.array([[0, 2], [4, 8]], dtype=np.float32)
    np.save(tmp_path / "abe.npy", abe)
    np.save(tmp_path / "no_abe.npy", no_abe)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "abe_path,no_abe_path,training_normalization_scale\n"
        "abe.npy,no_abe.npy,8\n"
    )

    dataset = AO2DPairDataset.from_manifest(
        manifest,
        patch_size=None,
        augment=False,
        normalization_mode="manifest_scale",
    )
    sample = dataset[0]
    assert torch.isclose(sample["input_scale"], torch.tensor(8.0))
    assert torch.isclose(sample["input"].amax(), torch.tensor(0.5))
    assert torch.isclose(sample["target"].amax(), torch.tensor(1.0))


def test_self_dataset_uses_single_input_scale(tmp_path):
    image = np.array([[0, 2], [4, 8]], dtype=np.float32)
    np.save(tmp_path / "image.npy", image)

    dataset = AO2DSelfDataset(
        tmp_path,
        patch_size=None,
        augment=False,
        normalization_mode="input_scale",
        input_scale_method="max",
    )

    sample = dataset[0]
    assert torch.isclose(sample["input_scale"], torch.tensor(8.0))
    assert torch.isclose(sample["input"].amax(), torch.tensor(1.0))
