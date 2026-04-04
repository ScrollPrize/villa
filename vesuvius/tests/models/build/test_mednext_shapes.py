from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.build.mednext import MedNeXtV1, create_mednext_v1, get_mednext_v1_config


def _make_mgr(
    patch_size=(32, 32, 32),
    *,
    targets=None,
    separate_decoders=True,
    enable_deep_supervision=False,
    architecture_type="mednext_v1",
    mednext_model_id="B",
):
    return SimpleNamespace(
        targets=targets or {"surface": {"out_channels": 2, "activation": "none"}},
        train_patch_size=patch_size,
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        model_name="mednext_test",
        enable_deep_supervision=enable_deep_supervision,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "architecture_type": architecture_type,
            "mednext_model_id": mednext_model_id,
            "mednext_kernel_size": 3,
            "separate_decoders": separate_decoders,
        },
    )


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


def test_network_from_config_mednext_v1_forward_shape():
    model = NetworkFromConfig(_make_mgr())
    out = model(torch.randn(2, 1, 32, 32, 32))

    assert set(out.keys()) == {"surface"}
    assert out["surface"].shape == (2, 2, 32, 32, 32)
    assert model.final_config["architecture_type"] == "mednext_v1"
    assert model.final_config["mednext_model_id"] == "B"


def test_network_from_config_mednext_v1_shared_decoder_multi_target():
    mgr = _make_mgr(
        targets={
            "surface": {"out_channels": 2, "activation": "none"},
            "mask": {"out_channels": 1, "activation": "none"},
        },
        separate_decoders=False,
    )
    mgr.targets = {
        "surface": {"out_channels": 2, "activation": "none"},
        "tissue": {"out_channels": 1, "activation": "none"},
    }
    model = NetworkFromConfig(mgr)
    out = model(torch.randn(1, 1, 32, 32, 32))

    assert set(out.keys()) == {"surface", "tissue"}
    assert out["surface"].shape == (1, 2, 32, 32, 32)
    assert out["tissue"].shape == (1, 1, 32, 32, 32)


def test_network_from_config_mednext_v1_deep_supervision_returns_multiscale_logits():
    model = NetworkFromConfig(_make_mgr(enable_deep_supervision=True))
    out = model(torch.randn(1, 1, 32, 32, 32))

    assert isinstance(out["surface"], list)
    assert len(out["surface"]) == 5
    assert out["surface"][0].shape == (1, 2, 32, 32, 32)


def test_network_from_config_mednext_requires_explicit_model_id():
    mgr = _make_mgr(mednext_model_id="")
    del mgr.model_config["mednext_model_id"]

    with pytest.raises(ValueError, match="mednext_model_id"):
        NetworkFromConfig(mgr)
