from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ao2d.optics import AO2DConfig, generate_psf2d_from_zernike, pupil_coordinates_2d, zernike_index_to_nm

from .scare2d import ZernikeResNetRegression2D


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _softplus_inverse(value: float) -> float:
    x = torch.as_tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(torch.clamp(x, min=1e-6))))


def _zscore_template(x: torch.Tensor, eps: float) -> torch.Tensor:
    dims = tuple(range(-3, 0))
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _l2_normalize_template(x: torch.Tensor, eps: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x.flatten(-3), dim=-1, keepdim=True).clamp_min(eps)
    return x / norm[..., None, None]


def _centered_fft2(image: torch.Tensor, fft_shift: bool, subtract_mean: bool = True) -> torch.Tensor:
    x = image.float()
    if subtract_mean:
        x = x - x.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fft2(x, dim=(-2, -1))
    if fft_shift:
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    return spectrum


def _quantile_mask(score: torch.Tensor, percentile: float, eps: float) -> torch.Tensor:
    percentile = float(percentile)
    if percentile <= 0:
        return torch.ones_like(score, dtype=score.dtype)
    if percentile >= 100:
        return torch.zeros_like(score, dtype=score.dtype)
    flat = score.flatten(-2)
    threshold = torch.quantile(flat, percentile / 100.0, dim=-1, keepdim=True)
    return (flat > threshold.clamp_min(eps)).view_as(score).to(dtype=score.dtype)


class DepthwiseSeparableResidualBlock2D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        groups = _group_count(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class LightweightOTFEncoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        out_channels: int = 3,
        blocks: int = 3,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if blocks < 0:
            raise ValueError("blocks must be non-negative")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(DepthwiseSeparableResidualBlock2D(hidden_channels, kernel_size=kernel_size) for _ in range(blocks))
        )
        self.proj = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.blocks(self.stem(x)))


class OTFUNetConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class LightweightOTFUNetEncoder2D(nn.Module):
    """Compact UNet encoder for OTF amplitude/phase feature maps."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        out_channels: int = 3,
        depth: int = 2,
        channel_multiplier_cap: int = 4,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if depth < 1:
            raise ValueError("depth must be positive for LightweightOTFUNetEncoder2D")
        cap_channels = hidden_channels * max(1, int(channel_multiplier_cap))
        channels = [hidden_channels]
        for level in range(depth):
            channels.append(min(hidden_channels * (2 ** (level + 1)), cap_channels))

        self.input_block = OTFUNetConvBlock2D(in_channels, channels[0])
        self.down_blocks = nn.ModuleList(
            OTFUNetConvBlock2D(channels[level], channels[level + 1]) for level in range(depth)
        )
        self.up_blocks = nn.ModuleList(
            OTFUNetConvBlock2D(channels[level + 1] + channels[level], channels[level])
            for level in range(depth - 1, -1, -1)
        )
        self.proj = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.input_block(x)
        skips.append(x)
        for down_block in self.down_blocks:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
            x = down_block(x)
            skips.append(x)

        for up_block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up_block(torch.cat([x, skip], dim=1))
        return self.proj(x)


class ZernikeOTFTemplateBank2D(nn.Module):
    """Fixed OTF response templates for signed single-mode Zernike perturbations."""

    def __init__(
        self,
        image_size: Sequence[int] | None,
        zernike_indices: Iterable[int],
        optics_config: AO2DConfig = AO2DConfig(),
        epsilon_um: float | Sequence[float] = 0.05,
        fft_shift: bool = True,
        otf_mtf_threshold: float = 0.03,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        indices = tuple(int(v) for v in zernike_indices)
        if not indices:
            raise ValueError("zernike_indices must not be empty")
        self.image_size = None if image_size is None else tuple(int(v) for v in image_size)
        self.zernike_indices = indices
        self.optics_config = optics_config
        self.fft_shift = bool(fft_shift)
        self.otf_mtf_threshold = float(otf_mtf_threshold)
        self.eps = float(eps)

        epsilon = torch.as_tensor(epsilon_um, dtype=torch.float32)
        if epsilon.ndim == 0:
            epsilon = epsilon.repeat(len(indices))
        if epsilon.numel() != len(indices):
            raise ValueError("epsilon_um must be a scalar or have one value per Zernike mode")
        self.register_buffer("epsilon_um", epsilon, persistent=True)
        self.register_buffer("templates", torch.empty(0), persistent=False)
        self.register_buffer("derivative_templates", torch.empty(0), persistent=False)
        self.register_buffer("rho", torch.empty(0), persistent=False)
        self.register_buffer("pupil_mask", torch.empty(0), persistent=False)
        if self.image_size is not None:
            self.rebuild(self.image_size)

    @property
    def num_modes(self) -> int:
        return len(self.zernike_indices)

    def rebuild(self, image_size: Sequence[int], device=None, dtype=torch.float32) -> None:
        image_size = tuple(int(v) for v in image_size)
        device = device if device is not None else self.epsilon_um.device
        dtype = dtype if dtype is not None else self.epsilon_um.dtype

        coeff = torch.zeros((self.num_modes * 2, self.num_modes), device=device, dtype=torch.float32)
        epsilon = self.epsilon_um.to(device=device, dtype=torch.float32)
        for mode_idx in range(self.num_modes):
            coeff[2 * mode_idx, mode_idx] = epsilon[mode_idx]
            coeff[2 * mode_idx + 1, mode_idx] = -epsilon[mode_idx]

        zero = torch.zeros((1, self.num_modes), device=device, dtype=torch.float32)
        psf = generate_psf2d_from_zernike(image_size, self.zernike_indices, coeff, self.optics_config)
        psf0 = generate_psf2d_from_zernike(image_size, self.zernike_indices, zero, self.optics_config)

        otf = torch.fft.fft2(torch.fft.ifftshift(psf, dim=(-2, -1)), s=image_size)
        otf0 = torch.fft.fft2(torch.fft.ifftshift(psf0, dim=(-2, -1)), s=image_size)
        if self.fft_shift:
            otf = torch.fft.fftshift(otf, dim=(-2, -1))
            otf0 = torch.fft.fftshift(otf0, dim=(-2, -1))

        log_amp = torch.log1p(torch.abs(otf)) - torch.log1p(torch.abs(otf0))
        relative_phase = otf * torch.conj(otf0)
        phase_den = torch.abs(relative_phase).clamp_min(self.eps)
        mtf_support = (
            (torch.abs(otf) > self.otf_mtf_threshold)
            & (torch.abs(otf0) > self.otf_mtf_threshold)
        ).to(dtype=log_amp.dtype)
        phase_cos = relative_phase.real / phase_den * mtf_support
        phase_sin = relative_phase.imag / phase_den * mtf_support

        raw_templates = torch.stack([log_amp, phase_cos, phase_sin], dim=1)
        raw_templates = raw_templates.view(self.num_modes, 2, 3, *image_size).real.to(dtype=dtype)

        templates = _zscore_template(raw_templates, self.eps)
        templates = _l2_normalize_template(templates, self.eps)

        derivative_templates = (raw_templates[:, 0] - raw_templates[:, 1]) / (
            2.0 * epsilon.to(device=device, dtype=dtype)[:, None, None, None]
        )
        derivative_templates = _zscore_template(derivative_templates, self.eps)
        derivative_templates = _l2_normalize_template(derivative_templates, self.eps)

        rho, _, pupil_mask = pupil_coordinates_2d(
            image_size,
            self.optics_config.pixel_size,
            self.optics_config.lambda_emission,
            self.optics_config.na,
            device=device,
            dtype=dtype,
        )
        if not self.fft_shift:
            rho = torch.fft.ifftshift(rho, dim=(-2, -1))
            pupil_mask = torch.fft.ifftshift(pupil_mask, dim=(-2, -1))

        self.image_size = image_size
        self.templates = templates
        self.derivative_templates = derivative_templates
        self.rho = rho.to(dtype=dtype)
        self.pupil_mask = pupil_mask.to(dtype=dtype)

    def ensure(self, image: torch.Tensor) -> torch.Tensor:
        image_size = tuple(int(v) for v in image.shape[-2:])
        needs_rebuild = self.templates.numel() == 0 or self.image_size != image_size
        if needs_rebuild:
            self.rebuild(image_size, device=image.device, dtype=image.dtype)
        elif self.templates.device != image.device or self.templates.dtype != image.dtype:
            self.templates = self.templates.to(device=image.device, dtype=image.dtype)
            self.derivative_templates = self.derivative_templates.to(device=image.device, dtype=image.dtype)
            self.rho = self.rho.to(device=image.device, dtype=image.dtype)
            self.pupil_mask = self.pupil_mask.to(device=image.device, dtype=image.dtype)
        return self.templates

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.ensure(image)


class OTFTemplateAttentionHead2D(nn.Module):
    """OTF template attention head with per-mode confidence fusion."""

    def __init__(
        self,
        in_channels: int,
        zernike_indices: Iterable[int],
        image_size: Sequence[int] | None = None,
        optics_config: AO2DConfig = AO2DConfig(),
        hidden: int = 128,
        depth: int = 3,
        reduction: int = 8,
        epsilon_um: float | Sequence[float] = 0.05,
        max_amp_um: float | Sequence[float] | None = None,
        max_amp_base_um: float = 0.12,
        encoder_channels: int = 32,
        encoder_blocks: int = 3,
        encoder_kernel_size: int = 5,
        encoder_type: str = "depthwise",
        eta: float = 5.0,
        alpha: float = 10.0,
        tau_init: float = 0.3,
        confidence_init: float = 1.4,
        fft_shift: bool = True,
        input_center: bool = True,
        input_phase_mask_percentile: float = 72.0,
        input_feature_channels: int = 5,
        otf_mtf_threshold: float = 0.03,
        eps: float = 1e-8,
        use_amplitude_head: bool = True,
        response_scale_init: float = 50.0,
    ) -> None:
        super().__init__()
        indices = tuple(int(v) for v in zernike_indices)
        if not indices:
            raise ValueError("zernike_indices must not be empty")
        self.zernike_indices = indices
        self.eps = float(eps)
        self.fft_shift = bool(fft_shift)
        self.input_center = bool(input_center)
        self.input_phase_mask_percentile = float(input_phase_mask_percentile)
        self.input_feature_channels = int(input_feature_channels)
        self.otf_mtf_threshold = float(otf_mtf_threshold)
        if self.input_feature_channels not in {3, 5}:
            raise ValueError("input_feature_channels must be 3 or 5")
        self.base_head = ZernikeResNetRegression2D(
            in_channels,
            len(indices),
            hidden=hidden,
            depth=depth,
            reduction=reduction,
        )
        self.template_bank = ZernikeOTFTemplateBank2D(
            image_size,
            indices,
            optics_config=optics_config,
            epsilon_um=epsilon_um,
            fft_shift=fft_shift,
            otf_mtf_threshold=otf_mtf_threshold,
            eps=eps,
        )
        encoder_type = str(encoder_type).lower().replace("-", "_")
        self.encoder_type = encoder_type
        if encoder_type in {"depthwise", "depthwise_separable", "separable", "lightweight"}:
            self.encoder = LightweightOTFEncoder2D(
                in_channels=self.input_feature_channels,
                hidden_channels=encoder_channels,
                out_channels=3,
                blocks=encoder_blocks,
                kernel_size=encoder_kernel_size,
            )
        elif encoder_type in {"unet", "u_net", "lightweight_unet"}:
            self.encoder = LightweightOTFUNetEncoder2D(
                in_channels=self.input_feature_channels,
                hidden_channels=encoder_channels,
                out_channels=3,
                depth=max(1, encoder_blocks),
            )
        else:
            raise ValueError(f"Unsupported template encoder_type: {encoder_type}")

        if max_amp_um is None:
            n_order, _ = zernike_index_to_nm(torch.as_tensor(indices, dtype=torch.int64))
            max_amp = float(max_amp_base_um) / torch.clamp(n_order.to(torch.float32), min=1.0)
        else:
            max_amp = torch.as_tensor(max_amp_um, dtype=torch.float32)
            if max_amp.ndim == 0:
                max_amp = max_amp.repeat(len(indices))
            if max_amp.numel() != len(indices):
                raise ValueError("max_amp_um must be a scalar or have one value per Zernike mode")
        self.register_buffer("max_amp_um", max_amp.to(torch.float32), persistent=True)

        self.raw_eta = nn.Parameter(torch.tensor(_softplus_inverse(eta), dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor(_softplus_inverse(alpha), dtype=torch.float32))
        self.raw_response_scale = nn.Parameter(
            torch.full((len(indices),), _softplus_inverse(response_scale_init), dtype=torch.float32)
        )
        self.tau = nn.Parameter(torch.full((len(indices),), float(tau_init), dtype=torch.float32))
        self.use_amplitude_head = bool(use_amplitude_head)
        self.amplitude_head = nn.Sequential(
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.amplitude_head[-1].weight)
        nn.init.constant_(self.amplitude_head[-1].bias, 0.0)
        self.confidence_head = nn.Sequential(
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.constant_(self.confidence_head[-1].bias, float(confidence_init))

        self.last_scores: torch.Tensor | None = None
        self.last_presence: torch.Tensor | None = None
        self.last_sign: torch.Tensor | None = None
        self.last_confidence: torch.Tensor | None = None
        self.last_a_base: torch.Tensor | None = None
        self.last_a_template: torch.Tensor | None = None

    def _input_features(self, image: torch.Tensor, reference_image: torch.Tensor | None = None) -> torch.Tensor:
        spectrum = _centered_fft2(image, self.fft_shift, subtract_mean=self.input_center)
        reference_spectrum = None
        if reference_image is not None:
            reference_spectrum = _centered_fft2(reference_image, self.fft_shift, subtract_mean=self.input_center)

        if reference_spectrum is None:
            magnitude_score = torch.abs(spectrum)
            amp = torch.log1p(magnitude_score)
            phase_source = spectrum
        else:
            magnitude_score = torch.abs(spectrum) * torch.abs(reference_spectrum)
            amp = torch.log1p(torch.abs(spectrum)) - torch.log1p(torch.abs(reference_spectrum))
            phase_source = spectrum * torch.conj(reference_spectrum)
        phase_mask = _quantile_mask(magnitude_score, self.input_phase_mask_percentile, self.eps)
        reliability = torch.log1p(magnitude_score)
        reliability = reliability / reliability.flatten(-2).amax(dim=-1, keepdim=True).clamp_min(self.eps)[..., None]
        reliability = reliability * phase_mask
        amp = (amp - amp.mean(dim=(-2, -1), keepdim=True)) / amp.std(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        denom = torch.abs(phase_source).clamp_min(self.eps)
        cos_phase = phase_source.real / denom * phase_mask
        sin_phase = phase_source.imag / denom * phase_mask

        self.template_bank.ensure(image)
        if self.input_feature_channels == 3:
            return torch.cat([amp, cos_phase.to(amp.dtype), sin_phase.to(amp.dtype)], dim=1)
        rho = self.template_bank.rho.to(device=image.device, dtype=amp.dtype)
        rho = rho.clamp(max=1.0).expand(amp.shape[0], amp.shape[1], -1, -1)
        return torch.cat(
            [
                amp,
                cos_phase.to(amp.dtype),
                sin_phase.to(amp.dtype),
                reliability.to(amp.dtype),
                rho,
            ],
            dim=1,
        )

    def encode_template_features(
        self,
        image: torch.Tensor,
        reference_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encoder(self._input_features(image, reference_image=reference_image)).to(dtype=image.dtype)
        features = _zscore_template(features, self.eps)
        return _l2_normalize_template(features, self.eps)

    def _coefficients_from_encoded_features(
        self,
        image: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        templates = self.template_bank.ensure(image)
        derivative_templates = self.template_bank.derivative_templates.to(dtype=features.dtype, device=features.device)
        scores = torch.einsum("bchw,kqchw->bkq", features, templates.to(dtype=features.dtype, device=features.device))
        derivative_score = torch.einsum("bchw,kchw->bk", features, derivative_templates)
        scores_pos = scores[..., 0]
        scores_neg = scores[..., 1]
        signed_score = scores_pos - scores_neg
        best_score = torch.maximum(scores_pos, scores_neg)

        eta = torch.nn.functional.softplus(self.raw_eta)
        alpha = torch.nn.functional.softplus(self.raw_alpha)
        response_scale = torch.nn.functional.softplus(self.raw_response_scale).to(
            device=image.device,
            dtype=derivative_score.dtype,
        )
        sign = torch.tanh(response_scale[None, :] * derivative_score)
        presence = torch.sigmoid(alpha * (best_score - self.tau.to(device=image.device, dtype=best_score.dtype)))
        max_amp = self.max_amp_um.to(device=image.device, dtype=presence.dtype)
        a_template = max_amp[None, :] * sign
        return a_template, scores, presence, sign

    def _template_coefficients(
        self,
        image: torch.Tensor,
        reference_image: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode_template_features(image, reference_image=reference_image)
        outputs = self._coefficients_from_encoded_features(image, features)
        if return_features:
            return (*outputs, features)
        return outputs

    def forward_with_image(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        a_base = self.base_head(features)
        a_template, scores, presence, sign = self._template_coefficients(image)

        scores_pos = scores[..., 0]
        scores_neg = scores[..., 1]
        best_score = torch.maximum(scores_pos, scores_neg)
        signed_score = scores_pos - scores_neg
        confidence_inputs = torch.stack(
            [
                scores_pos,
                scores_neg,
                best_score,
                torch.abs(signed_score),
                torch.abs(a_template),
                torch.abs(a_template - a_base),
            ],
            dim=-1,
        )
        confidence = torch.sigmoid(self.confidence_head(confidence_inputs).squeeze(-1))
        a_final = confidence * a_template + (1.0 - confidence) * a_base

        self.last_scores = scores.detach()
        self.last_presence = presence.detach()
        self.last_sign = sign.detach()
        self.last_confidence = confidence.detach()
        self.last_a_base = a_base.detach()
        self.last_a_template = a_template.detach()
        return a_final

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("OTFTemplateAttentionHead2D requires forward_with_image(features, image).")
