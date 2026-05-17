#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao2d.data.io import IMAGE_EXTENSIONS, load_image, normalize01, save_image
from ao2d.models.factory import make_model
from ao2d.models.picnet2d import AberrationGenerator2D, OBJGenerator2D


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2-D AO restoration inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input image or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "object_generator" in ckpt and "aberration_generator" in ckpt:
        model_cfg = config.get("model", {})
        zernike_modes = len(config.get("optics", {}).get("zernike_indices", list(range(3, 16))))
        object_generator = OBJGenerator2D(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            final_activation=str(model_cfg.get("final_activation", "sigmoid")),
        ).to(device)
        aberration_generator = AberrationGenerator2D(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("zernike_modes", zernike_modes)),
            base_channels=int(model_cfg.get("aberration_base_channels", 32)),
        ).to(device)
        object_generator.load_state_dict(ckpt["object_generator"])
        aberration_generator.load_state_dict(ckpt["aberration_generator"])
        object_generator.eval()
        aberration_generator.eval()
        model = None
    else:
        model = make_model(config["model"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        object_generator = None
        aberration_generator = None

    input_path = Path(args.input)
    files = [input_path] if input_path.is_file() else sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for path in tqdm(files, desc="test"):
            img = normalize01(load_image(path), (0.1, 99.9))
            x = torch.from_numpy(np.ascontiguousarray(img))[None, None].float().to(device)
            if model is None:
                out = (object_generator(x), aberration_generator(x))
            else:
                out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            restored = pred.squeeze().detach().cpu().numpy()
            save_image(output_dir / f"{path.stem}_restored.tif", restored)
            if isinstance(out, tuple):
                coeff = out[1].squeeze().detach().cpu().numpy()
                np.savetxt(output_dir / f"{path.stem}_zernike_coeff_um.txt", coeff[None], fmt="%.8g")


if __name__ == "__main__":
    main()
