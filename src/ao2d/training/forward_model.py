from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ao2d.optics import AO2DConfig, convolve_fft2, generate_psf2d_from_zernike


class AO2DForwardModel(nn.Module):
    """Differentiable image -> aberrated image forward model."""

    def __init__(self, image_size: tuple[int, int], zernike_indices: Iterable[int], config: AO2DConfig) -> None:
        super().__init__()
        self.image_size = tuple(int(v) for v in image_size)
        self.zernike_indices = tuple(int(v) for v in zernike_indices)
        self.config = config

    def forward(self, object_or_restored: torch.Tensor, coefficients: torch.Tensor | None = None) -> torch.Tensor:
        if coefficients is None:
            batch_size = object_or_restored.shape[0] if object_or_restored.ndim >= 3 else 1
            coefficients = torch.zeros(
                (batch_size, len(self.zernike_indices)),
                dtype=object_or_restored.dtype,
                device=object_or_restored.device,
            )
        psf = generate_psf2d_from_zernike(
            self.image_size,
            self.zernike_indices,
            coefficients.to(dtype=object_or_restored.dtype, device=object_or_restored.device),
            self.config,
        )
        return convolve_fft2(object_or_restored, psf)
