import torch

from ao2d.models import (
    ABESplitNet2D,
    ABEFusionNet2D,
    DeltaPhiZernikeProjectionHead2D,
    OTFTemplateAttentionHead2D,
    PupilPhaseZernikeProjectionHead2D,
)
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


def test_abesplit2d_uses_separate_generators_without_obj_frequency_branch():
    x = torch.rand(2, 1, 32, 32)
    model = ABESplitNet2D(
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
    assert not hasattr(model, "obj_frequency_branch")
    assert model.obj_image_branch is not model.abe_image_branch
    assert model.obj_gradient_branch is not model.abe_gradient_branch


def test_factory_builds_abesplit2d_with_split_options():
    model = make_model(
        {
            "name": "abesplit2d",
            "in_channels": 1,
            "out_channels": 1,
            "zernike_modes": 5,
            "branch_channels": 4,
            "fusion_channels": 12,
            "branch_depth": 1,
            "obj_branch_channels": 3,
            "abe_branch_channels": 5,
            "obj_fusion_channels": 6,
            "abe_fusion_channels": 10,
            "obj_base_channels": 8,
            "obj_depth": 1,
            "zernike_hidden": 16,
            "zernike_depth": 1,
            "fft": False,
            "fft_shift": True,
            "final_activation": "none",
        }
    )

    assert isinstance(model, ABESplitNet2D)
    assert model.obj_fusion[0].in_channels == 6
    assert model.abe_fusion[0].in_channels == 15
    assert model.frequency_transform.fft is False
    assert model.frequency_transform.fft_shift is True


def test_delta_phi_projection_head_returns_zernike_coefficients():
    x = torch.rand(2, 12, 8, 8)
    head = DeltaPhiZernikeProjectionHead2D(
        in_channels=12,
        zernike_indices=tuple(range(3, 8)),
        hidden=8,
        depth=1,
        pair_count=32,
        pupil_grid_size=12,
    )

    coeff = head(x)

    assert coeff.shape == (2, 5)
    assert torch.isfinite(coeff).all()


def test_factory_builds_abesplit2d_with_delta_phi_projection_head():
    model = make_model(
        {
            "name": "abesplit2d",
            "in_channels": 1,
            "out_channels": 1,
            "zernike_modes": 5,
            "zernike_indices": [3, 4, 5, 6, 7],
            "branch_channels": 4,
            "fusion_channels": 12,
            "branch_depth": 1,
            "obj_base_channels": 8,
            "obj_depth": 1,
            "zernike_hidden": 16,
            "zernike_depth": 1,
            "aberration_head_type": "delta_phi_projection",
            "delta_phi_pair_count": 32,
            "delta_phi_pupil_grid_size": 12,
            "fft": False,
            "final_activation": "none",
        }
    )
    x = torch.rand(2, 1, 32, 32)

    obj, zernike = model(x)

    assert obj.shape == x.shape
    assert zernike.shape == (2, 5)


def test_pupil_phase_projection_head_returns_zernike_coefficients():
    x = torch.rand(2, 12, 8, 8)
    head = PupilPhaseZernikeProjectionHead2D(
        in_channels=12,
        zernike_indices=tuple(range(3, 8)),
        hidden=8,
        depth=1,
        pair_count=32,
        pupil_grid_size=12,
    )

    coeff = head(x)
    phase = head.forward_phase(x)
    delta = head.forward_delta(x)

    assert coeff.shape == (2, 5)
    assert phase.shape == (2, 12, 12)
    assert delta.shape == (2, head.projection.pair_count)
    assert torch.isfinite(coeff).all()


def test_factory_builds_abesplit2d_with_fft_phase_and_pupil_phase_projection_head():
    model = make_model(
        {
            "name": "abesplit2d",
            "in_channels": 1,
            "out_channels": 1,
            "zernike_modes": 5,
            "zernike_indices": [3, 4, 5, 6, 7],
            "branch_channels": 4,
            "fusion_channels": 12,
            "branch_depth": 1,
            "obj_base_channels": 8,
            "obj_depth": 1,
            "zernike_hidden": 16,
            "zernike_depth": 1,
            "aberration_head_type": "pupil_phase_projection",
            "delta_phi_pair_count": 32,
            "delta_phi_pupil_grid_size": 12,
            "fft": True,
            "fft_phase_features": True,
            "final_activation": "none",
        }
    )
    x = torch.rand(2, 1, 32, 32)

    obj, zernike = model(x)

    assert obj.shape == x.shape
    assert zernike.shape == (2, 5)


def test_factory_builds_abesplit2d_with_template_attention_head():
    model = make_model(
        {
            "name": "abesplit2d",
            "in_channels": 1,
            "out_channels": 1,
            "zernike_modes": 5,
            "zernike_indices": [3, 4, 5, 6, 7],
            "branch_channels": 4,
            "fusion_channels": 12,
            "branch_depth": 1,
            "obj_base_channels": 8,
            "obj_depth": 1,
            "zernike_hidden": 16,
            "zernike_depth": 1,
            "zernike_reduction": 4,
            "aberration_head_type": "template_attention",
            "template_image_size": [24, 24],
            "template_optics": {
                "pixel_size": 0.3,
                "na": 1.05,
                "lambda_emission": 1.0,
                "lambda_excitation": 0.808,
                "mode": "widefield",
            },
            "template_attention": {
                "epsilon_um": 0.05,
                "encoder_channels": 8,
                "encoder_blocks": 1,
                "eta": 5.0,
                "alpha": 10.0,
                "tau_init": 0.3,
                "confidence_init": 1.4,
                "signed_zernike_indices": [6, 7],
            },
            "fft": True,
            "fft_phase_features": True,
            "final_activation": "none",
        }
    )
    x = torch.rand(2, 1, 24, 24)

    obj, zernike = model(x)
    head = model.aberration_head

    assert isinstance(head, OTFTemplateAttentionHead2D)
    assert obj.shape == x.shape
    assert zernike.shape == (2, 5)
    assert torch.isfinite(zernike).all()
    assert head.template_bank.templates.shape == (5, 2, 3, 24, 24)
    assert head.template_signed_mode_mask.tolist() == [False, False, False, True, True]
    assert head.last_confidence is not None
    assert head.last_confidence.shape == (2, 5)
    assert torch.all((head.last_confidence >= 0) & (head.last_confidence <= 1))
