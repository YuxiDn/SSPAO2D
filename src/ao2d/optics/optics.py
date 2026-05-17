from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F


NormalizeMode = Literal["sum", "max"]
MicroscopyMode = Literal["widefield", "confocal", "twophoton", "lightsheet"]


@dataclass(frozen=True)
class AO2DConfig:
    """Optical parameters in micrometers, matching the MATLAB 2-D AO functions."""

    pixel_size: float = 0.065
    na: float = 1.0
    lambda_emission: float = 0.525
    lambda_excitation: float = 0.488
    mode: MicroscopyMode = "widefield"
    pinhole_au: float = 1.0
    lightsheet_fwhm: float = 1.2
    normalize: NormalizeMode = "sum"


def _as_device_dtype_tensor(values, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


def zernike_index_to_nm(indices: Iterable[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert OSA/ANSI single index p to (n, m), identical to the MATLAB helper."""

    p = torch.as_tensor(indices, dtype=torch.float64)
    d = torch.sqrt(9.0 + 8.0 * p)
    n = torch.ceil((d - 3.0) / 2.0)
    m = 2.0 * p - n * (n + 2.0)
    return n.to(torch.int64), m.to(torch.int64)


def zernike_mode(n: int, m: int, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Normalized OSA/ANSI Zernike mode used by the MATLAB implementation."""

    radial = torch.zeros_like(rho)
    m_abs = abs(int(m))
    for s in range((int(n) - m_abs) // 2 + 1):
        coefficient = (
            (-1.0) ** s
            * torch.exp(torch.lgamma(torch.tensor(float(n - s + 1), device=rho.device, dtype=rho.dtype)))
            / (
                torch.exp(torch.lgamma(torch.tensor(float(s + 1), device=rho.device, dtype=rho.dtype)))
                * torch.exp(torch.lgamma(torch.tensor(float((n + m_abs) // 2 - s + 1), device=rho.device, dtype=rho.dtype)))
                * torch.exp(torch.lgamma(torch.tensor(float((n - m_abs) // 2 - s + 1), device=rho.device, dtype=rho.dtype)))
            )
        )
        radial = radial + coefficient * rho ** (int(n) - 2 * s)

    normalization = ((1.0 + float(m != 0)) * (int(n) + 1.0)) ** 0.5
    if m > 0:
        return normalization * radial * torch.cos(m_abs * theta)
    if m < 0:
        return normalization * radial * torch.sin(m_abs * theta)
    return normalization * radial


def zernike_wavefront(
    zernike_indices: Iterable[int] | torch.Tensor,
    coefficients: torch.Tensor,
    rho: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Build OPD wavefront in micrometers from OSA/ANSI Zernike coefficients."""

    indices = torch.as_tensor(list(zernike_indices), device=rho.device, dtype=torch.int64)
    coeff = coefficients.to(device=rho.device, dtype=rho.dtype)
    if coeff.ndim == 1:
        coeff = coeff.unsqueeze(0)
    if coeff.shape[-1] != indices.numel():
        raise ValueError("coefficients must have the same length as zernike_indices")

    n_list, m_list = zernike_index_to_nm(indices.cpu())
    wavefront = torch.zeros((coeff.shape[0], *rho.shape), device=rho.device, dtype=rho.dtype)
    for k, (n, m) in enumerate(zip(n_list.tolist(), m_list.tolist(), strict=True)):
        wavefront = wavefront + coeff[:, k, None, None] * zernike_mode(n, m, rho, theta)
    return wavefront


def pupil_coordinates_2d(
    image_size: tuple[int, int],
    pixel_size: float,
    wavelength: float,
    na: float,
    device=None,
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rho, theta and pupil mask on MATLAB's centered frequency grid."""

    if wavelength > 20:
        raise ValueError("Wavelengths must be in micrometers, e.g. 0.525 rather than 525 nm.")
    if pixel_size > 5:
        raise ValueError("Pixel size must be in micrometers, e.g. 0.300 rather than 300 nm.")

    ny, nx = int(image_size[0]), int(image_size[1])
    fx = torch.arange(-nx // 2, -nx // 2 + nx, device=device, dtype=dtype) / (nx * pixel_size)
    fy = torch.arange(-ny // 2, -ny // 2 + ny, device=device, dtype=dtype) / (ny * pixel_size)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
    cutoff = na / wavelength
    theta = torch.atan2(fy_grid / cutoff, fx_grid / cutoff)
    rho = torch.hypot(fx_grid / cutoff, fy_grid / cutoff)
    pupil_mask = rho <= 1.0
    return rho, theta, pupil_mask


def normalize_psf(psf: torch.Tensor, mode: NormalizeMode = "sum") -> torch.Tensor:
    psf = torch.clamp(psf.real, min=0)
    if mode == "sum":
        denom = psf.flatten(-2).sum(dim=-1).clamp_min(torch.finfo(psf.dtype).eps)
        return psf / denom[..., None, None]
    if mode == "max":
        denom = psf.flatten(-2).amax(dim=-1).clamp_min(torch.finfo(psf.dtype).eps)
        return psf / denom[..., None, None]
    raise ValueError(f"Unsupported PSF normalization: {mode}")


def _scalar_psf2d(
    image_size: tuple[int, int],
    zernike_indices: Iterable[int],
    coefficients: torch.Tensor,
    pixel_size: float,
    wavelength: float,
    na: float,
) -> torch.Tensor:
    rho, theta, pupil_mask = pupil_coordinates_2d(
        image_size, pixel_size, wavelength, na, device=coefficients.device, dtype=coefficients.dtype
    )
    wavefront = zernike_wavefront(zernike_indices, coefficients, rho, theta)
    phase = 2.0 * torch.pi / wavelength * wavefront
    pupil = pupil_mask.to(coefficients.dtype).unsqueeze(0) * torch.exp(1j * phase.to(torch.complex64))
    field = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(pupil, dim=(-2, -1))), dim=(-2, -1))
    return normalize_psf(torch.abs(field) ** 2, "sum")


def _centered_pixel_grid(image_size: tuple[int, int], device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    ny, nx = image_size
    y = torch.arange(1, ny + 1, device=device, dtype=dtype) - ny // 2 - 1
    x = torch.arange(1, nx + 1, device=device, dtype=dtype) - nx // 2 - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return yy, xx


def _pinhole_mask_2d(
    image_size: tuple[int, int],
    pixel_size: float,
    lambda_emission: float,
    na: float,
    pinhole_au: float,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    airy_radius = 0.61 * lambda_emission / na
    radius_pixels = max(0.5, pinhole_au * airy_radius / pixel_size)
    yy, xx = _centered_pixel_grid(image_size, device=device, dtype=dtype)
    mask = ((xx**2 + yy**2) <= radius_pixels**2).to(dtype)
    return mask / mask.sum().clamp_min(torch.finfo(dtype).eps)


def generate_psf2d_from_zernike(
    image_size: tuple[int, int],
    zernike_indices: Iterable[int] = tuple(range(3, 16)),
    coefficients: torch.Tensor | None = None,
    config: AO2DConfig = AO2DConfig(),
) -> torch.Tensor:
    """Generate a 2-D microscopy PSF from Zernike OPD coefficients in micrometers."""

    indices = tuple(int(v) for v in zernike_indices)
    if coefficients is None:
        coefficients = torch.zeros((1, len(indices)), dtype=torch.float32)
    coefficients = _as_device_dtype_tensor(coefficients, dtype=torch.float32)
    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(0)

    psf_em = _scalar_psf2d(
        image_size, indices, coefficients, config.pixel_size, config.lambda_emission, config.na
    )

    if config.mode == "widefield":
        psf = psf_em
    elif config.mode == "twophoton":
        excitation_coeff = coefficients * config.lambda_excitation / config.lambda_emission
        psf_ex = _scalar_psf2d(
            image_size, indices, excitation_coeff, config.pixel_size, config.lambda_excitation, config.na
        )
        psf = psf_ex**2
    elif config.mode == "confocal":
        excitation_coeff = coefficients * config.lambda_excitation / config.lambda_emission
        psf_ex = _scalar_psf2d(
            image_size, indices, excitation_coeff, config.pixel_size, config.lambda_excitation, config.na
        )
        pinhole = _pinhole_mask_2d(
            image_size,
            config.pixel_size,
            config.lambda_emission,
            config.na,
            config.pinhole_au,
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        filtered = F.conv2d(
            psf_em[:, None],
            pinhole[None, None],
            padding=(image_size[0] // 2, image_size[1] // 2),
        )[:, :, : image_size[0], : image_size[1]].squeeze(1)
        psf = psf_ex * filtered
    elif config.mode == "lightsheet":
        sigma = config.lightsheet_fwhm / (2.0 * (2.0 * torch.log(torch.tensor(2.0))).sqrt().item())
        yy, _ = _centered_pixel_grid(image_size, device=coefficients.device, dtype=coefficients.dtype)
        sheet = torch.exp(-0.5 * ((yy * config.pixel_size) / sigma) ** 2)
        psf = psf_em * sheet
    else:
        raise ValueError(f"Unsupported microscopy mode: {config.mode}")

    return normalize_psf(psf, config.normalize)


def random_zernike_coefficients(
    zernike_indices: Iterable[int] = tuple(range(3, 16)),
    rms_waves: float = 0.18,
    wavelength: float = 0.525,
    spectrum: Literal["flat", "loworder"] = "loworder",
    include_defocus: bool = True,
    generator: torch.Generator | None = None,
    device=None,
) -> torch.Tensor:
    """Generate random OPD coefficients in micrometers, matching MATLAB defaults."""

    indices = torch.as_tensor(tuple(zernike_indices), dtype=torch.int64, device=device)
    coeff = torch.randn(indices.shape, generator=generator, device=device)
    coeff[(indices == 0) | (indices == 1) | (indices == 2)] = 0
    if not include_defocus:
        coeff[indices == 4] = 0
    if spectrum == "loworder":
        n_order, _ = zernike_index_to_nm(indices.cpu())
        coeff = coeff / torch.clamp(n_order.to(device=device, dtype=coeff.dtype), min=1)
    elif spectrum != "flat":
        raise ValueError(f"Unsupported coefficient spectrum: {spectrum}")
    current_rms = torch.sqrt(torch.mean(coeff**2))
    target_rms = rms_waves * wavelength
    if current_rms > 0:
        coeff = coeff / current_rms * target_rms
    return coeff.to(torch.float32)


def convolve_fft2(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Circular FFT convolution with ifftshift(psf), matching the MATLAB dataset code."""

    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(1)
    if psf.ndim == 2:
        psf = psf.unsqueeze(0)
    if psf.ndim == 3:
        psf = psf.unsqueeze(1)

    otf = torch.fft.fft2(torch.fft.ifftshift(psf, dim=(-2, -1)), s=image.shape[-2:])
    out = torch.real(torch.fft.ifft2(torch.fft.fft2(image) * otf))
    out = torch.clamp(out, min=0)
    low = out.amin(dim=(-2, -1), keepdim=True)
    high = out.amax(dim=(-2, -1), keepdim=True)
    out = (out - low) / (high - low).clamp_min(torch.finfo(out.dtype).eps)
    return out

