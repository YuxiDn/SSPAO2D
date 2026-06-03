from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ao2d.optics import zernike_index_to_nm, zernike_mode


def _pupil_points(grid_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    axis = torch.linspace(-1.0, 1.0, grid_size, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    mask = torch.hypot(xx, yy) <= 1.0
    return torch.stack([xx[mask], yy[mask]], dim=-1)


def _sample_pair_indices(num_points: int, pair_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    if num_points < 2:
        raise ValueError("pupil grid must contain at least two points")

    idx = torch.linspace(0, num_points - 1, pair_count).round().to(torch.long)
    offset = max(1, num_points // 3)
    idx2 = (idx * 37 + offset) % num_points
    same = idx2 == idx
    idx2[same] = (idx2[same] + 1) % num_points
    return idx, idx2


def _sample_pair_points(points: torch.Tensor, pair_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx1, idx2 = _sample_pair_indices(points.shape[0], pair_count)
    return points[idx1], points[idx2]


def _zernike_matrix(indices: Iterable[int], points: torch.Tensor) -> torch.Tensor:
    x = points[:, 0]
    y = points[:, 1]
    rho = torch.hypot(x, y)
    theta = torch.atan2(y, x)
    n_list, m_list = zernike_index_to_nm(torch.as_tensor(tuple(indices), dtype=torch.int64))
    modes = [
        zernike_mode(int(n), int(m), rho, theta)
        for n, m in zip(n_list.tolist(), m_list.tolist(), strict=True)
    ]
    return torch.stack(modes, dim=-1)


class ZernikeDifferenceProjection(nn.Module):
    """Fixed least-squares layer mapping pairwise OPD differences to Zernike coefficients."""

    def __init__(
        self,
        zernike_indices: Iterable[int],
        pair_count: int = 512,
        pupil_grid_size: int = 32,
        ridge: float = 1e-4,
    ) -> None:
        super().__init__()
        indices = tuple(int(v) for v in zernike_indices)
        if not indices:
            raise ValueError("zernike_indices must not be empty")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")

        points = _pupil_points(pupil_grid_size)
        pair_index1, pair_index2 = _sample_pair_indices(points.shape[0], pair_count)
        p1 = points[pair_index1]
        p2 = points[pair_index2]
        z1 = _zernike_matrix(indices, p1)
        z2 = _zernike_matrix(indices, p2)
        difference_basis = z1 - z2

        row_norm = difference_basis.norm(dim=1)
        keep = row_norm > 1e-7
        difference_basis = difference_basis[keep]
        pair_index1 = pair_index1[keep]
        pair_index2 = pair_index2[keep]
        if difference_basis.shape[0] < len(indices):
            raise ValueError("not enough informative pair samples for the requested Zernike modes")

        gram = difference_basis.T @ difference_basis
        gram = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
        projection = torch.linalg.solve(gram, difference_basis.T)

        self.zernike_indices = indices
        self.pupil_grid_size = int(pupil_grid_size)
        self.register_buffer("pupil_mask", torch.hypot(
            *torch.meshgrid(
                torch.linspace(-1.0, 1.0, pupil_grid_size),
                torch.linspace(-1.0, 1.0, pupil_grid_size),
                indexing="ij",
            )
        ) <= 1.0)
        self.register_buffer("pair_index1", pair_index1)
        self.register_buffer("pair_index2", pair_index2)
        self.register_buffer("difference_basis", difference_basis)
        self.register_buffer("projection", projection)

    @property
    def pair_count(self) -> int:
        return int(self.difference_basis.shape[0])

    def forward(self, delta_phi: torch.Tensor) -> torch.Tensor:
        """Project ``delta_phi`` samples with shape ``[B, M]`` to coefficients ``[B, K]``."""

        if delta_phi.shape[-1] != self.pair_count:
            raise ValueError(f"delta_phi last dimension must be {self.pair_count}")
        return delta_phi @ self.projection.T.to(dtype=delta_phi.dtype, device=delta_phi.device)

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Reconstruct projected pairwise differences from coefficients."""

        return coefficients @ self.difference_basis.T.to(dtype=coefficients.dtype, device=coefficients.device)

    def phase_to_difference(self, phase: torch.Tensor) -> torch.Tensor:
        """Sample pairwise OPD differences from a pupil phase map ``[B, P, P]``."""

        if phase.shape[-2:] != (self.pupil_grid_size, self.pupil_grid_size):
            raise ValueError(f"phase map must have spatial shape {(self.pupil_grid_size, self.pupil_grid_size)}")
        mask = self.pupil_mask.to(device=phase.device)
        phase_vec = phase[..., mask]
        index1 = self.pair_index1.to(device=phase.device)
        index2 = self.pair_index2.to(device=phase.device)
        return phase_vec[..., index1] - phase_vec[..., index2]


class _DeltaPhiResidualBlock2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        residual = residual * self.attn(residual)
        return self.act(x + residual)


class DeltaPhiZernikeProjectionHead2D(nn.Module):
    """Predict pairwise OPD differences, then project them to Zernike coefficients."""

    def __init__(
        self,
        in_channels: int,
        zernike_indices: Iterable[int],
        hidden: int = 128,
        depth: int = 3,
        reduction: int = 8,
        pair_count: int = 512,
        pupil_grid_size: int = 32,
        ridge: float = 1e-4,
        max_delta_opd: float | None = 0.75,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.projection = ZernikeDifferenceProjection(
            zernike_indices,
            pair_count=pair_count,
            pupil_grid_size=pupil_grid_size,
            ridge=ridge,
        )
        self.max_delta_opd = max_delta_opd
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.blocks = nn.Sequential(*(_DeltaPhiResidualBlock2D(hidden, reduction=reduction) for _ in range(depth)))
        self.delta_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, self.projection.pair_count),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        delta = self.delta_head(x)
        if self.max_delta_opd is not None:
            delta = float(self.max_delta_opd) * torch.tanh(delta)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.forward_delta(x)
        return self.projection(delta)


class PupilPhaseZernikeProjectionHead2D(nn.Module):
    """Decode a pupil OPD map, sample pairwise differences, and project to Zernike coefficients."""

    def __init__(
        self,
        in_channels: int,
        zernike_indices: Iterable[int],
        hidden: int = 128,
        depth: int = 3,
        reduction: int = 8,
        pair_count: int = 512,
        pupil_grid_size: int = 32,
        ridge: float = 1e-4,
        max_phase_opd: float | None = 0.75,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.projection = ZernikeDifferenceProjection(
            zernike_indices,
            pair_count=pair_count,
            pupil_grid_size=pupil_grid_size,
            ridge=ridge,
        )
        self.max_phase_opd = max_phase_opd
        self.last_phase: torch.Tensor | None = None
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.blocks = nn.Sequential(*(_DeltaPhiResidualBlock2D(hidden, reduction=reduction) for _ in range(depth)))
        self.phase_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, pupil_grid_size * pupil_grid_size),
        )
        nn.init.xavier_uniform_(self.phase_head[-1].weight, gain=1e-3)
        nn.init.zeros_(self.phase_head[-1].bias)

    def forward_phase(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        phase = self.phase_head(x)
        phase = phase.view(x.shape[0], self.projection.pupil_grid_size, self.projection.pupil_grid_size)
        if self.max_phase_opd is not None:
            phase = float(self.max_phase_opd) * torch.tanh(phase)

        mask = self.projection.pupil_mask.to(device=phase.device, dtype=phase.dtype)
        phase = phase * mask
        piston = phase.sum(dim=(-2, -1), keepdim=True) / mask.sum().clamp_min(torch.finfo(phase.dtype).eps)
        return (phase - piston) * mask

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        phase = self.forward_phase(x)
        self.last_phase = phase
        return self.projection.phase_to_difference(phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.forward_delta(x)
        return self.projection(delta)
