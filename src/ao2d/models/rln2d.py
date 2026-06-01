from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import _norm_layer, output_activation


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma**2))
    return kernel / kernel.amax().clamp_min(1e-8)


def _init_gaussian_bank(conv: nn.Conv2d, sigma_min: float = 0.6, sigma_max: float = 1.8) -> None:
    if conv.kernel_size[0] != conv.kernel_size[1]:
        return
    kernel_size = conv.kernel_size[0]
    sigmas = torch.linspace(sigma_min, sigma_max, conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        for out_idx, sigma in enumerate(sigmas):
            kernel = _gaussian_kernel_2d(kernel_size, float(sigma), conv.weight.device, conv.weight.dtype)
            conv.weight[out_idx].copy_(kernel.expand_as(conv.weight[out_idx]))
        conv.weight.mul_(1.0 / max(1, conv.in_channels * kernel_size * kernel_size))
        if conv.bias is not None:
            conv.bias.zero_()


class _PositiveConvBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: str = "batch",
        gaussian_init: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if gaussian_init:
            _init_gaussian_bank(self.conv)
        self.norm = _norm_layer(norm)(out_channels)
        self.act = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class RLN2D(nn.Module):
    """2-D Richardson-Lucy-Net baseline.

    This is a PyTorch 2-D adaptation of the public TensorFlow 1.x RLN code:
    a positive initial estimate is refined by learned Richardson-Lucy style
    ratio and multiplicative update blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 8,
        num_iterations: int = 1,
        kernel_size: int = 3,
        norm: str = "batch",
        eps: float = 1e-4,
        ratio_clip: float | None = 10.0,
        residual: bool = False,
        final_activation: str | None = "softplus",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial size.")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1.")
        self.eps = eps
        self.ratio_clip = ratio_clip
        self.residual = residual

        self.input_projection = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.estimator = nn.Sequential(
            _PositiveConvBlock2D(in_channels, num_features, kernel_size, norm=norm, gaussian_init=True),
            _PositiveConvBlock2D(num_features, num_features, kernel_size, norm=norm, gaussian_init=True),
            nn.Conv2d(num_features, out_channels, 1),
            nn.Softplus(beta=1.0),
        )

        self.forward_blurs = nn.ModuleList(
            nn.Conv2d(out_channels, in_channels, kernel_size, padding=kernel_size // 2)
            for _ in range(num_iterations)
        )
        self.backprojections = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            for _ in range(num_iterations)
        )
        self.refiners = nn.ModuleList(
            nn.Sequential(
                _PositiveConvBlock2D(out_channels * 2, num_features, kernel_size, norm=norm),
                _PositiveConvBlock2D(num_features, num_features, kernel_size, norm=norm),
                nn.Conv2d(num_features, out_channels, kernel_size, padding=kernel_size // 2),
                nn.Softplus(beta=1.0),
            )
            for _ in range(num_iterations)
        )
        for conv in [*self.forward_blurs, *self.backprojections]:
            _init_gaussian_bank(conv)
        self.activation = output_activation(final_activation)

    def _safe_ratio(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        ratio = numerator.clamp_min(self.eps) / denominator.clamp_min(self.eps)
        ratio = torch.nan_to_num(ratio, nan=0.0, posinf=1.0 / self.eps, neginf=0.0)
        if self.ratio_clip is not None:
            ratio = ratio.clamp(max=self.ratio_clip)
        return ratio

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        anchor = self.input_projection(x)
        estimate = self.estimator(x).clamp_min(self.eps)
        aux = estimate

        for forward_blur, backprojection, refiner in zip(self.forward_blurs, self.backprojections, self.refiners):
            blurred = F.softplus(forward_blur(estimate), beta=1.0).clamp_min(self.eps)
            ratio = self._safe_ratio(x, blurred)
            correction = F.softplus(backprojection(ratio), beta=1.0).clamp_min(self.eps)
            estimate = refiner(torch.cat([estimate * correction, anchor], dim=1))

        if self.residual:
            estimate = estimate + anchor
        estimate = self.activation(estimate)
        if return_aux:
            return estimate, aux
        return estimate
