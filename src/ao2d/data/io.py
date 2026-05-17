from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy", ".npz"}


def load_image(path: str | Path, npz_key: str | None = None) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if npz_key is None:
            if len(data.files) != 1:
                raise ValueError(f"{path} contains multiple arrays; pass npz_key.")
            npz_key = data.files[0]
        arr = data[npz_key]
    elif suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        if tifffile is not None:
            arr = tifffile.imread(path)
        elif Image is not None:
            arr = np.asarray(Image.open(path))
        else:
            raise ImportError("Reading image files requires tifffile or Pillow.")
    else:
        raise ValueError(f"Unsupported image file: {path}")

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {arr.shape} from {path}")
    return np.ascontiguousarray(arr)


def normalize01(arr: np.ndarray, percentile: tuple[float, float] | None = None) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if percentile is not None:
        lo, hi = np.percentile(arr, percentile)
    else:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    arr = np.clip(arr, lo, hi)
    denom = max(float(hi - lo), np.finfo(np.float32).eps)
    return ((arr - lo) / denom).astype(np.float32)


def save_image(path: str | Path, image: np.ndarray, dtype: str = "uint16") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = normalize01(np.asarray(image))
    if dtype == "uint16":
        arr = np.round(img * 65535).astype(np.uint16)
    elif dtype == "float32":
        arr = img.astype(np.float32)
    else:
        raise ValueError(f"Unsupported output dtype: {dtype}")
    if tifffile is not None:
        tifffile.imwrite(path, arr)
    elif Image is not None:
        Image.fromarray(arr).save(path)
    else:
        raise ImportError("Writing image files requires tifffile or Pillow.")
