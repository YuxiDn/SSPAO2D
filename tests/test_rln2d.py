import pytest
import torch

from ao2d.models import RLN2D
from ao2d.models.factory import make_model


def test_rln2d_keeps_2d_shape():
    x = torch.rand(2, 1, 17, 19)
    model = RLN2D(
        in_channels=1,
        out_channels=1,
        num_features=4,
        num_iterations=2,
        norm="none",
        final_activation="none",
    )

    assert model(x).shape == x.shape


def test_rln2d_factory_aliases():
    model = make_model(
        {
            "name": "richardson_lucy_net",
            "in_channels": 1,
            "out_channels": 1,
            "num_features": 4,
            "num_iterations": 1,
            "norm": "none",
            "final_activation": "none",
        }
    )

    assert isinstance(model, RLN2D)


def test_rln2d_rejects_even_kernel_size():
    with pytest.raises(ValueError, match="kernel_size"):
        RLN2D(kernel_size=4)
