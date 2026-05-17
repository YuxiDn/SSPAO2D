import torch

from ao2d.optics import AO2DConfig, convolve_fft2, generate_psf2d_from_zernike, random_zernike_coefficients


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
