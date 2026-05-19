import torch

from ao2d.models import SFENet2D
from ao2d.models.factory import make_model


def test_sfenet2d_fft_modes_keep_shape_without_pixel_stack():
    x = torch.rand(1, 1, 16, 16)
    model = SFENet2D(
        in_channels=1,
        out_channels=1,
        num_features=4,
        encoder_channels=8,
        num_groups=1,
        num_blocks=1,
        fft_branch=True,
        fft=True,
        fft_shift=True,
        fft_forward=False,
        unet_depth=1,
        num_pixel_stack_layer=0,
        final_activation="none",
    )

    assert model(x).shape == x.shape


def test_sfenet2d_pixel_stack_reduces_spatial_size():
    x = torch.rand(1, 1, 16, 16)
    model = SFENet2D(
        in_channels=1,
        out_channels=1,
        num_features=4,
        encoder_channels=8,
        num_groups=1,
        num_blocks=1,
        fft_branch=False,
        unet_depth=1,
        num_pixel_stack_layer=1,
        final_activation="none",
    )

    assert model(x).shape == (1, 1, 8, 8)


def test_factory_passes_sfenet2d_extended_options():
    model = make_model(
        {
            "name": "sfenet2d",
            "in_channels": 1,
            "out_channels": 1,
            "num_features": 4,
            "encoder_channels": 8,
            "num_rg": 1,
            "num_rcab": 1,
            "fft_brunch": True,
            "fft": False,
            "fft_shift": True,
            "fft_forward": False,
            "num_down_up": 1,
            "num_pixel_stack_layer": 0,
            "final_activation": "none",
        }
    )

    assert isinstance(model, SFENet2D)
    assert model.fft is False
    assert model.fft_shift is True
    assert model.fft_forward is False
