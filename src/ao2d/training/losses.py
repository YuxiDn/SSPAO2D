from __future__ import annotations

import torch


def total_variation_2d(x: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
    dx = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()
    return dx + dy


def relative_std_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True)
    return torch.mean(1.0 / (std / (mean.abs() + eps) + eps))

