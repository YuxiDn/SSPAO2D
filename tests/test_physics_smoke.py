import torch

from ao2d.optics import AO2DConfig, convolve_fft2, generate_psf2d_from_zernike, random_zernike_coefficients
from ao2d.training import AO2DForwardModel


def test_psf_and_convolution_smoke():
    image_size = (32, 32)
    indices = tuple(range(3, 16))
    coeff = random_zernike_coefficients(indices, rms_waves=0.05, wavelength=0.525)
    psf = generate_psf2d_from_zernike(image_size, indices, coeff, AO2DConfig())
    assert psf.shape == (1, 32, 32)
    assert torch.isclose(psf.sum(), torch.tensor(1.0), atol=1e-5)
    img = torch.zeros(32, 32)
    img[16, 16] = 1
    out = convolve_fft2(img, psf[0])
    assert out.shape == (1, 1, 32, 32)
    assert torch.isfinite(out).all()


def test_convolution_preserves_amplitude_for_sum_normalized_psf():
    img = torch.full((8, 8), 3.0)
    psf = torch.ones(3, 3) / 9.0

    out = convolve_fft2(img, psf)

    assert torch.allclose(out, torch.full((1, 1, 8, 8), 3.0), atol=1e-6)


def test_none_zernike_coefficients_mean_no_aberration():
    image_size = (32, 32)
    indices = tuple(range(3, 16))
    zero_coeff = torch.zeros(len(indices))
    config = AO2DConfig()

    psf_none = generate_psf2d_from_zernike(image_size, indices, None, config)
    psf_zero = generate_psf2d_from_zernike(image_size, indices, zero_coeff, config)
    assert torch.allclose(psf_none, psf_zero)

    image = torch.rand(2, 1, *image_size)
    forward_model = AO2DForwardModel(image_size, indices, config)
    no_aberration = torch.zeros(2, len(indices))
    assert torch.allclose(forward_model(image, None), forward_model(image, no_aberration))
