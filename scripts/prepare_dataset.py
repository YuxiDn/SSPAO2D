#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao2d.data.io import IMAGE_EXTENSIONS, load_image, normalize01, save_image
from ao2d.optics import AO2DConfig, convolve_fft2, generate_psf2d_from_zernike, random_zernike_coefficients


def gaussian_filter_np(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    kernel = torch.einsum("i,j->ij", g, g)[None, None]
    tensor = torch.from_numpy(image.astype(np.float32))[None, None]
    out = torch.nn.functional.conv2d(tensor, kernel, padding=radius)
    return out.squeeze().numpy()


def synthetic_phantom(image_size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    h, w = image_size
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.integers(18, 42)):
        cy = rng.uniform(0, h)
        cx = rng.uniform(0, w)
        sy = rng.uniform(1.5, 8.0)
        sx = rng.uniform(1.5, 8.0)
        amp = rng.uniform(0.3, 1.0)
        img += amp * np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    for _ in range(rng.integers(6, 14)):
        cy = rng.uniform(0, h)
        cx = rng.uniform(0, w)
        angle = rng.uniform(0, np.pi)
        min_dim = min(h, w)
        length = rng.uniform(max(8, min_dim * 0.25), max(9, min_dim * 0.9))
        radius = rng.uniform(0.8, 2.2)
        xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        img += rng.uniform(0.25, 0.8) * np.exp(-0.5 * (yr / radius) ** 2) * (np.abs(xr) < length / 2)
    img = gaussian_filter_np(img, sigma=float(rng.uniform(0.2, 0.8)))
    img += rng.normal(0, 0.01, size=img.shape).astype(np.float32)
    return normalize01(img)


def add_noise(img: np.ndarray, rng: np.random.Generator, gaussian_std: float, photon_peak: float) -> np.ndarray:
    noisy = rng.poisson(np.clip(img, 0, 1) * photon_peak).astype(np.float32) / photon_peak
    noisy += rng.normal(0, gaussian_std, size=img.shape).astype(np.float32)
    return normalize01(np.maximum(noisy, 0))


def list_source_images(source_dir: str | None) -> list[Path]:
    if source_dir is None:
        return []
    root = Path(source_dir)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or organize a 2-D AO microscopy dataset.")
    parser.add_argument("--source_dir", default=None, help="Optional clean/object image directory.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_objects", type=int, default=30)
    parser.add_argument("--aberrations_per_object", type=int, default=3)
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--pixel_size", type=float, default=0.300)
    parser.add_argument("--na", type=float, default=1.05)
    parser.add_argument("--lambda_emission", type=float, default=1.000)
    parser.add_argument("--lambda_excitation", type=float, default=0.808)
    parser.add_argument("--rms_min", type=float, default=0.02)
    parser.add_argument("--rms_max", type=float, default=0.30)
    parser.add_argument("--zernike_indices", type=int, nargs="+", default=list(range(3, 16)))
    parser.add_argument("--gaussian_noise_std", type=float, default=0.005)
    parser.add_argument("--photon_peak", type=float, default=3000.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output = Path(args.output_root)
    dirs = {
        "object": output / "OBJ",
        "target": output / "No_abe",
        "aberrated": output / "abe",
        "psf_target": output / "PSF" / "No_abe",
        "psf_aberrated": output / "PSF" / "abe",
        "zernike": output / "Zernike",
        "metadata": output / "metadata",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def manifest_path(path: Path) -> str:
        return path.relative_to(output).as_posix()

    rng = np.random.default_rng(args.seed)
    torch_gen = torch.Generator().manual_seed(args.seed)
    image_size = tuple(args.image_size)
    optics = AO2DConfig(
        pixel_size=args.pixel_size,
        na=args.na,
        lambda_emission=args.lambda_emission,
        lambda_excitation=args.lambda_excitation,
        mode="widefield",
    )
    source_files = list_source_images(args.source_dir)
    zero_coeff = torch.zeros(len(args.zernike_indices))
    psf_clean = generate_psf2d_from_zernike(image_size, args.zernike_indices, zero_coeff, optics)[0]

    rows = []
    for obj_idx in tqdm(range(args.num_objects), desc="objects"):
        if source_files:
            obj = normalize01(load_image(source_files[obj_idx % len(source_files)]))
            if obj.shape != image_size:
                raise ValueError(f"{source_files[obj_idx % len(source_files)]} has shape {obj.shape}, expected {image_size}")
        else:
            obj = synthetic_phantom(image_size, rng)
        obj_id = f"obj_{obj_idx + 1:06d}"
        obj_path = dirs["object"] / f"{obj_id}.tif"
        target_path = dirs["target"] / f"{obj_id}.tif"
        save_image(obj_path, obj)

        target = convolve_fft2(torch.from_numpy(obj).float(), psf_clean).squeeze().numpy()
        target = add_noise(target, rng, args.gaussian_noise_std, args.photon_peak)
        save_image(target_path, target)
        np.savez_compressed(dirs["psf_target"] / f"{obj_id}.npz", psf=psf_clean.numpy(), zernike_indices=np.array(args.zernike_indices), coefficients_um=zero_coeff.numpy())

        for abe_idx in range(args.aberrations_per_object):
            rms = rng.uniform(args.rms_min, args.rms_max)
            coeff = random_zernike_coefficients(
                args.zernike_indices,
                rms_waves=float(rms),
                wavelength=args.lambda_emission,
                generator=torch_gen,
            )
            psf = generate_psf2d_from_zernike(image_size, args.zernike_indices, coeff, optics)[0]
            aberrated = convolve_fft2(torch.from_numpy(obj).float(), psf).squeeze().numpy()
            aberrated = add_noise(aberrated, rng, args.gaussian_noise_std, args.photon_peak)
            abe_id = f"{obj_id}_abe_{abe_idx + 1:02d}_rms_{rms:.3f}"
            abe_path = dirs["aberrated"] / f"{abe_id}.tif"
            psf_path = dirs["psf_aberrated"] / f"{abe_id}.npz"
            zernike_path = dirs["zernike"] / f"{abe_id}_zernike.npz"
            save_image(abe_path, aberrated)
            np.savez_compressed(psf_path, psf=psf.numpy(), zernike_indices=np.array(args.zernike_indices), coefficients_um=coeff.numpy())
            np.savez_compressed(zernike_path, zernike_indices=np.array(args.zernike_indices), coefficients_um=coeff.numpy(), rms_waves=rms)
            rows.append(
                {
                    "object_id": obj_id,
                    "aberration_id": abe_id,
                    "object_path": manifest_path(obj_path),
                    "no_abe_path": manifest_path(target_path),
                    "abe_path": manifest_path(abe_path),
                    "psf_no_abe_path": manifest_path(dirs["psf_target"] / f"{obj_id}.npz"),
                    "psf_abe_path": manifest_path(psf_path),
                    "zernike_path": manifest_path(zernike_path),
                    "rms_waves": f"{rms:.8g}",
                    "pixel_size_um": args.pixel_size,
                    "na": args.na,
                    "lambda_em_um": args.lambda_emission,
                    "lambda_ex_um": args.lambda_excitation,
                }
            )

    manifest = dirs["metadata"] / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
