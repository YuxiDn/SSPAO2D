from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "AO2D_DATA_ROOT"
DATA_ANCHORS = {
    "abe",
    "No_abe",
    "no_abe",
    "OBJ",
    "objects",
    "metadata",
    "validation",
    "psf",
    "psf_aberrated",
    "psf_no_abe",
    "zernike",
}


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def get_data_root(config: dict | None = None, override: str | Path | None = None) -> Path | None:
    if override:
        return expand_path(override)
    data_cfg = (config or {}).get("data", {})
    root = data_cfg.get("root") or data_cfg.get("data_root") or os.environ.get(DATA_ROOT_ENV)
    return expand_path(root) if root else None


def resolve_path(path: str | Path, data_root: str | Path | None = None) -> Path:
    path = expand_path(path)
    if path.is_absolute():
        return path
    root = expand_path(data_root) if data_root else None
    return root / path if root else path


def infer_manifest_data_root(manifest: str | Path, data_root: str | Path | None = None) -> Path:
    if data_root:
        return expand_path(data_root)
    manifest = expand_path(manifest)
    if manifest.parent.name == "metadata":
        return manifest.parent.parent
    return manifest.parent


def resolve_manifest_record_path(path: str | Path, manifest_root: str | Path) -> Path:
    raw = expand_path(path)
    root = expand_path(manifest_root)
    if not raw.is_absolute():
        return root / raw
    if raw.exists():
        return raw
    parts = raw.parts
    for index, part in enumerate(parts):
        if part in DATA_ANCHORS:
            remapped = root.joinpath(*parts[index:])
            if remapped.exists():
                return remapped
    return raw
