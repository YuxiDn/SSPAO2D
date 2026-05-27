import torch

from ao2d.models import ABEFusionNet2D
from ao2d.models.factory import make_model


def test_abenet2d_returns_object_and_zernike_outputs():
    x = torch.rand(2, 1, 32, 32)
    model = ABEFusionNet2D(
        in_channels=1,
        out_channels=1,
        zernike_modes=13,
        branch_channels=4,
        fusion_channels=12,
        branch_depth=1,
        obj_base_channels=8,
        obj_depth=1,
        zernike_hidden=16,
        zernike_depth=1,
        fft=True,
        fft_shift=True,
        final_activation="none",
    )

    obj, zernike = model(x)

    assert obj.shape == x.shape
    assert zernike.shape == (2, 13)


def test_factory_builds_abenet2d_with_extended_options():
    model = make_model(
        {
            "name": "abenet2d",
            "in_channels": 1,
            "out_channels": 1,
            "zernike_modes": 5,
            "branch_channels": 4,
            "fusion_channels": 12,
            "branch_depth": 1,
            "obj_base_channels": 8,
            "obj_depth": 1,
            "zernike_hidden": 16,
            "zernike_depth": 1,
            "fft": False,
            "fft_shift": True,
            "final_activation": "none",
        }
    )

    assert isinstance(model, ABEFusionNet2D)
    assert model.frequency_transform.fft is False
    assert model.frequency_transform.fft_shift is True
