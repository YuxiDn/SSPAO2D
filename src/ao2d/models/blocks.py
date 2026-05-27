from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def output_activation(name: str | None) -> nn.Module:
    name = (name or "none").lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "softplus":
        return nn.Softplus(beta=1.0)
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name in {"none", "linear"}:
        return nn.Identity()
    raise ValueError(f"Unsupported output activation: {name}")


class DoubleConv2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch", residual: bool = True) -> None:
        super().__init__()
        norm_layer = _norm_layer(norm)
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if residual or in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class Down2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch") -> None:
        super().__init__()
        self.body = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(in_ch, out_ch, norm=norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Up2D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "batch") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(in_ch + skip_ch, out_ch, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = match_size(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class ResidualLayer2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _group_norm(in_ch),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _group_norm(out_ch),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DownResidual2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            _group_norm(in_ch),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            _group_norm(out_ch),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2)
        self.residual = ResidualLayer2D(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.main(x) + self.skip(x)
        return self.residual(fused) + fused


class UpResidual2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            _group_norm(in_ch),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            _group_norm(out_ch),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.residual = ResidualLayer2D(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.main(x) + self.skip(x)
        return self.residual(fused) + fused


def fft_amplitude_2d(x: torch.Tensor, gamma: float = 0.8, shift: bool = True) -> torch.Tensor:
    x32 = x.float()
    amp = torch.abs(torch.fft.fftn(x32, dim=(-2, -1))).clamp_min(1e-8).pow(gamma)
    if shift:
        amp = torch.fft.fftshift(amp, dim=(-2, -1))
    amp = torch.log10(1 + amp)
    mean = amp.mean(dim=(-2, -1), keepdim=True)
    std = amp.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return (amp - mean) / std


def match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dy = target.shape[-2] - x.shape[-2]
    dx = target.shape[-1] - x.shape[-1]
    if dy or dx:
        x = F.pad(x, [0, max(dx, 0), 0, max(dy, 0)])
        x = x[..., : target.shape[-2], : target.shape[-1]]
    return x


def _norm_layer(name: str):
    name = (name or "none").lower()
    if name == "batch":
        return nn.BatchNorm2d
    if name == "instance":
        return nn.InstanceNorm2d
    if name in {"none", "identity"}:
        return lambda _: nn.Identity()
    raise ValueError(f"Unsupported normalization: {name}")


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)
