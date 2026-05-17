from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DownResidual2D, ResidualLayer2D, UpResidual2D, match_size, output_activation


class OBJGenerator2D(nn.Module):
    """PICNet object/restoration generator converted from SSPAO's 3-D OBJ_Generator."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 48, 96, 128),
        final_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.dropout = nn.Dropout(p=0.02)
        self.down1 = DownResidual2D(in_channels, c1)
        self.down2 = DownResidual2D(c1, c2)
        self.down3 = DownResidual2D(c2, c3)
        self.down4 = DownResidual2D(c3, c4)
        self.down5 = DownResidual2D(c4, c5)
        self.up1 = UpResidual2D(c5, c4 * 2)
        self.up2 = UpResidual2D(c4 * 2 + c4, c4)
        self.up3 = UpResidual2D(c4 + c3, c3)
        self.up4 = UpResidual2D(c3 + c2, c2)
        self.up5 = UpResidual2D(c2 + c1, c2)
        self.refine = ResidualLayer2D(c2, c1)
        self.out = nn.Conv2d(c1, out_channels, 1)
        self.activation = output_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.dropout(self.down1(x))
        x2 = self.dropout(self.down2(x1))
        x3 = self.dropout(self.down3(x2))
        x4 = self.dropout(self.down4(x3))
        x5 = self.dropout(self.down5(x4))

        y = match_size(self.dropout(self.up1(x5)), x4)
        y = torch.cat([x4, y], dim=1)
        y = match_size(self.dropout(self.up2(y)), x3)
        y = torch.cat([x3, y], dim=1)
        y = match_size(self.dropout(self.up3(y)), x2)
        y = torch.cat([x2, y], dim=1)
        y = match_size(self.dropout(self.up4(y)), x1)
        y = torch.cat([x1, y], dim=1)
        y = self.up5(y)
        y = self.refine(y)
        return self.activation(self.out(y))


class AberrationGenerator2D(nn.Module):
    """PICNet aberration/Zernike regressor. Uses a compact 2-D CNN instead of 3-D ResNet34."""

    def __init__(self, in_channels: int = 1, out_channels: int = 13, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            _res_stage(base_channels, base_channels, 2),
            _res_stage(base_channels, base_channels * 2, 2, stride=2),
            _res_stage(base_channels * 2, base_channels * 4, 2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, out_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Discriminator2D(nn.Module):
    """Simple PatchGAN-style discriminator for optional GAN training."""

    def __init__(self, in_channels: int = 1, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _disc_block(in_channels, hidden),
            _disc_block(hidden, hidden * 2),
            _disc_block(hidden * 2, hidden * 4),
            _disc_block(hidden * 4, hidden * 8),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 8, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class PICNet2D(nn.Module):
    """Convenience wrapper returning restored image and predicted Zernike coefficients."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        zernike_modes: int = 13,
        final_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.object_generator = OBJGenerator2D(in_channels, out_channels, final_activation=final_activation)
        self.aberration_generator = AberrationGenerator2D(in_channels, zernike_modes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.object_generator(x), self.aberration_generator(x)


class _BasicResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.skip(x))


def _res_stage(in_ch: int, out_ch: int, blocks: int, stride: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = [_BasicResidual(in_ch, out_ch, stride)]
    layers += [_BasicResidual(out_ch, out_ch) for _ in range(blocks - 1)]
    return nn.Sequential(*layers)


def _disc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=False),
    )

