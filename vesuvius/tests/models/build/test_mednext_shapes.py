from __future__ import annotations

import torch

from vesuvius.models.build.mednext import MedNeXtV1, create_mednext_v1, get_mednext_v1_config


def test_create_mednext_v1_small_forward_shape_3d():
    model = create_mednext_v1(1, 3, "S", kernel_size=3, deep_supervision=False)
    out = model(torch.randn(2, 1, 32, 32, 32))

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3, 32, 32, 32)


def test_mednext_v1_deep_supervision_returns_multiscale_outputs():
    model = create_mednext_v1(1, 2, "B", kernel_size=3, deep_supervision=True)
    out = model(torch.randn(1, 1, 32, 32, 32))

    assert isinstance(out, list)
    assert len(out) == 5
    assert out[0].shape == (1, 2, 32, 32, 32)
    assert out[1].shape == (1, 2, 16, 16, 16)
    assert out[2].shape == (1, 2, 8, 8, 8)
    assert out[3].shape == (1, 2, 4, 4, 4)
    assert out[4].shape == (1, 2, 2, 2, 2)


def test_mednext_v1_outside_block_checkpointing_smoke_forward():
    cfg = get_mednext_v1_config("M")
    model = MedNeXtV1(
        in_channels=1,
        n_channels=cfg["n_channels"],
        n_classes=2,
        exp_r=cfg["exp_r"],
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        checkpoint_style="outside_block",
        block_counts=cfg["block_counts"],
        norm_type="group",
        grn=False,
    )
    out = model(torch.randn(1, 1, 32, 32, 32))

    assert out.shape == (1, 2, 32, 32, 32)
