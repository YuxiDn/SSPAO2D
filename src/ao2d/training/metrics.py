from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(target: torch.Tensor, pred: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    return (10.0 * torch.log10(max_val**2 / (mse + 1e-12))).mean()


def _gaussian_kernel_2d(size: int, sigma: float, device, dtype) -> torch.Tensor:
    center = size // 2
    x = torch.arange(-center, size - center, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    kernel = torch.einsum("i,j->ij", g, g)
    return (kernel / kernel.sum())[None, None]


def ssim(target: torch.Tensor, pred: torch.Tensor, max_val: float = 1.0, size: int = 9, sigma: float = 1.5) -> torch.Tensor:
    if target.shape != pred.shape:
        raise ValueError(f"SSIM expects matching shapes, got {target.shape} and {pred.shape}")
    kernel = _gaussian_kernel_2d(size, sigma, target.device, target.dtype)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    def filt(x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        y = F.conv2d(x.reshape(b * c, 1, *x.shape[-2:]), kernel, padding=size // 2)
        return y.reshape(b, c, *x.shape[-2:])

    ux = filt(target)
    uy = filt(pred)
    uxx = filt(target * target)
    uyy = filt(pred * pred)
    uxy = filt(target * pred)
    numerator = (2 * ux * uy + c1) * (2 * (uxy - ux * uy) + c2)
    denominator = (ux**2 + uy**2 + c1) * (uxx - ux**2 + uyy - uy**2 + c2)
    return (numerator / (denominator + 1e-8)).mean()

