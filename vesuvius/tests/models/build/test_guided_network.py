from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.build.guidance import TokenBook3D
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone


def _tiny_dinovol_model_config() -> dict:
    return {
        "model_type": "v2",
        "input_channels": 1,
        "global_crops_size": [16, 16, 16],
        "local_crops_size": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "embed_dim": 48,
        "depth": 2,
        "num_heads": 4,
        "num_reg_tokens": 2,
        "mlp_ratio": 2.0,
        "drop_path_rate": 0.0,
        "qkv_fused": True,
    }


def _write_local_guide_checkpoint(path: Path) -> None:
    backbone = build_dinovol_2_backbone(_tiny_dinovol_model_config())
    teacher_state = {f"backbone.{key}": value.cpu() for key, value in backbone.state_dict().items()}
    checkpoint = {
        "config": {
            "model": _tiny_dinovol_model_config(),
            "dataset": {
                "global_crop_size": [16, 16, 16],
                "local_crop_size": [16, 16, 16],
            },
        },
        "teacher": teacher_state,
    }
    torch.save(checkpoint, path)


def _make_mgr(
    checkpoint_path: Path,
    *,
    basic_encoder_block: str,
    guide_tokenbook_tokens: int | None = None,
    guide_compile_policy: str | None = None,
    guide_fusion_stage: str | None = None,
) -> SimpleNamespace:
    guided_config = {}
    if guide_tokenbook_tokens is not None:
        guided_config["guide_tokenbook_tokens"] = int(guide_tokenbook_tokens)
    if guide_compile_policy is not None:
        guided_config["guide_compile_policy"] = str(guide_compile_policy)
    if guide_fusion_stage is not None:
        guided_config["guide_fusion_stage"] = str(guide_fusion_stage)
    return SimpleNamespace(
        targets={"ink": {"out_channels": 1, "activation": "none"}},
        train_patch_size=(16, 16, 16),
        train_batch_size=2,
        in_channels=1,
        autoconfigure=False,
        model_name="guided_test",
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "features_per_stage": [8, 16, 32],
            "n_stages": 3,
            "n_blocks_per_stage": [1, 1, 1],
            "n_conv_per_stage_decoder": [1, 1],
            "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "basic_encoder_block": basic_encoder_block,
            "basic_decoder_block": "ConvBlock",
            "bottleneck_block": "BasicBlockD",
            "separate_decoders": True,
            "guide_backbone": str(checkpoint_path),
            "guide_freeze": True,
            "guide_tokenbook_sample_rate": 1.0,
            "input_shape": [16, 16, 16],
            **guided_config,
        },
    )


def test_tokenbook3d_returns_unit_interval_mask():
    module = TokenBook3D(n_tokens=8, embed_dim=16)
    x = torch.randn(2, 16, 2, 2, 2)

    guide = module(x)

    assert guide.shape == (2, 1, 2, 2, 2)
    assert float(guide.min().detach()) >= 0.0
    assert float(guide.max().detach()) <= 1.0


@pytest.mark.parametrize("basic_encoder_block", ["ConvBlock", "BasicBlockD"])
def test_guided_network_forward_shapes_and_aux_outputs(tmp_path: Path, basic_encoder_block: str):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(checkpoint_path, basic_encoder_block=basic_encoder_block)
    model = NetworkFromConfig(mgr)
    x = torch.randn(2, 1, 16, 16, 16)

    outputs = model(x)
    outputs_with_aux, aux = model(x, return_aux=True)

    assert set(outputs.keys()) == {"ink"}
    assert outputs["ink"].shape == (2, 1, 16, 16, 16)
    assert set(outputs_with_aux.keys()) == {"ink"}
    assert aux["guide_mask"].shape == (2, 1, 2, 2, 2)


@pytest.mark.parametrize("basic_encoder_block", ["ConvBlock", "BasicBlockD"])
def test_feature_encoder_guidance_returns_encoder_stage_aux_outputs(tmp_path: Path, basic_encoder_block: str):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block=basic_encoder_block,
        guide_fusion_stage="feature_encoder",
    )
    model = NetworkFromConfig(mgr)

    outputs, aux = model(torch.randn(2, 1, 16, 16, 16), return_aux=True)

    assert outputs["ink"].shape == (2, 1, 16, 16, 16)
    assert list(aux.keys()) == ["enc_0", "enc_1", "enc_2"]
    assert aux["enc_0"].shape == (2, 1, 16, 16, 16)
    assert aux["enc_1"].shape == (2, 1, 8, 8, 8)
    assert aux["enc_2"].shape == (2, 1, 4, 4, 4)
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"
    assert model.final_config["guide_stage_keys"] == ["enc_0", "enc_1", "enc_2"]
    assert len(model.guide_stage_tokenbooks) == 3


def test_guided_network_backprop_updates_tokenbook_but_not_frozen_guide_backbone(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(checkpoint_path, basic_encoder_block="ConvBlock")
    model = NetworkFromConfig(mgr)
    model.train()

    x = torch.randn(2, 1, 16, 16, 16)
    target = torch.randn(2, 1, 16, 16, 16)
    outputs, aux = model(x, return_aux=True)
    loss = torch.nn.functional.mse_loss(outputs["ink"], target) + aux["guide_mask"].mean()
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.guide_tokenbook.parameters())
    assert all(parameter.grad is None for parameter in model.guide_backbone.parameters())


def test_feature_encoder_guidance_backprop_updates_all_stage_tokenbooks(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_encoder",
    )
    model = NetworkFromConfig(mgr)
    model.train()

    x = torch.randn(2, 1, 16, 16, 16)
    target = torch.randn(2, 1, 16, 16, 16)
    outputs, aux = model(x, return_aux=True)
    loss = torch.nn.functional.mse_loss(outputs["ink"], target)
    loss = loss + sum(stage_aux.mean() for stage_aux in aux.values())
    loss.backward()

    assert set(aux.keys()) == {"enc_0", "enc_1", "enc_2"}
    assert all(
        any(parameter.grad is not None for parameter in tokenbook.parameters())
        for tokenbook in model.guide_stage_tokenbooks.values()
    )
    assert all(parameter.grad is None for parameter in model.guide_backbone.parameters())


def test_guided_network_supports_reduced_tokenbook_prototype_count(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_tokenbook_tokens=3,
    )
    model = NetworkFromConfig(mgr)
    outputs, aux = model(torch.randn(1, 1, 16, 16, 16), return_aux=True)

    assert model.guide_tokenbook.book.shape[0] == 3
    assert model.final_config["guide_tokenbook_tokens"] == 3
    assert outputs["ink"].shape == (1, 1, 16, 16, 16)
    assert aux["guide_mask"].shape == (1, 1, 2, 2, 2)


def test_feature_encoder_guidance_supports_reduced_tokenbook_prototype_count(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_tokenbook_tokens=3,
        guide_fusion_stage="feature_encoder",
    )
    model = NetworkFromConfig(mgr)
    outputs, aux = model(torch.randn(1, 1, 16, 16, 16), return_aux=True)

    assert model.final_config["guide_tokenbook_tokens"] == 3
    assert all(tokenbook.book.shape[0] == 3 for tokenbook in model.guide_stage_tokenbooks.values())
    assert outputs["ink"].shape == (1, 1, 16, 16, 16)
    assert set(aux.keys()) == {"enc_0", "enc_1", "enc_2"}


@pytest.mark.parametrize("guide_compile_policy", ["off", "backbone_only", "tokenbook_only"])
def test_guided_network_records_guide_compile_policy(tmp_path: Path, guide_compile_policy: str):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_compile_policy=guide_compile_policy,
    )
    model = NetworkFromConfig(mgr)

    assert model.guide_compile_policy == guide_compile_policy
    assert model.final_config["guide_compile_policy"] == guide_compile_policy


def test_feature_encoder_guidance_records_exact_fusion_stage(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_fusion_stage="feature_encoder",
    )
    model = NetworkFromConfig(mgr)

    assert model.guide_fusion_stage == "feature_encoder"
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"


def test_guided_network_rejects_invalid_guide_compile_policy(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)
    mgr = _make_mgr(
        checkpoint_path,
        basic_encoder_block="ConvBlock",
        guide_compile_policy="bad_policy",
    )

    with pytest.raises(ValueError, match="guide_compile_policy"):
        NetworkFromConfig(mgr)


def test_guided_network_rejects_primus_architecture(tmp_path: Path):
    checkpoint_path = tmp_path / "guide_backbone.pt"
    _write_local_guide_checkpoint(checkpoint_path)

    mgr = SimpleNamespace(
        targets={"ink": {"out_channels": 1, "activation": "none"}},
        train_patch_size=(16, 16, 16),
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        model_name="guided_test",
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        model_config={
            "architecture_type": "primus_s",
            "input_shape": [16, 16, 16],
            "patch_embed_size": [8, 8, 8],
            "guide_backbone": str(checkpoint_path),
        },
    )

    with pytest.raises(ValueError, match="not supported with primus"):
        NetworkFromConfig(mgr)
