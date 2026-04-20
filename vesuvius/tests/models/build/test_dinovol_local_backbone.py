from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vesuvius.models.build.pretrained_backbones.dinov2 import build_dinov2_backbone
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


def _write_checkpoint(path: Path, *, include_config: bool, config: dict | None = None) -> None:
    backbone = build_dinovol_2_backbone(_tiny_dinovol_model_config())
    teacher_state = {f"backbone.{key}": value.cpu() for key, value in backbone.state_dict().items()}
    checkpoint = {"teacher": teacher_state}
    if include_config:
        checkpoint["config"] = config
    torch.save(checkpoint, path)


def test_build_dinov2_backbone_from_local_checkpoint_with_embedded_config(tmp_path: Path):
    checkpoint_path = tmp_path / "tiny_dinovol.pt"
    config = {
        "model": _tiny_dinovol_model_config(),
        "dataset": {
            "global_crop_size": [16, 16, 16],
            "local_crop_size": [16, 16, 16],
        },
    }
    _write_checkpoint(checkpoint_path, include_config=True, config=config)

    backbone = build_dinov2_backbone(
        str(checkpoint_path),
        input_channels=1,
        input_shape=(16, 16, 16),
    )

    x = torch.randn(1, 1, 16, 16, 16)
    features = backbone(x)

    assert backbone.patch_embed_size == (8, 8, 8)
    assert len(features) == 1
    assert features[0].shape == (1, 48, 2, 2, 2)


def test_build_dinov2_backbone_from_local_checkpoint_with_sidecar_config(tmp_path: Path):
    checkpoint_path = tmp_path / "tiny_dinovol_no_config.pt"
    sidecar_path = tmp_path / "guide_backbone.json"
    config = {
        "model": _tiny_dinovol_model_config(),
        "dataset": {
            "global_crop_size": [16, 16, 16],
            "local_crop_size": [16, 16, 16],
        },
    }
    _write_checkpoint(checkpoint_path, include_config=False)
    sidecar_path.write_text(json.dumps(config), encoding="utf-8")

    backbone = build_dinov2_backbone(
        str(checkpoint_path),
        input_channels=1,
        input_shape=(16, 16, 16),
        config_path=str(sidecar_path),
    )

    x = torch.randn(1, 1, 16, 16, 16)
    features = backbone(x)

    assert features[0].shape == (1, 48, 2, 2, 2)


def test_build_dinov2_backbone_local_checkpoint_requires_config(tmp_path: Path):
    checkpoint_path = tmp_path / "missing_config.pt"
    _write_checkpoint(checkpoint_path, include_config=False)

    with pytest.raises(ValueError, match="does not contain an embedded config"):
        build_dinov2_backbone(
            str(checkpoint_path),
            input_channels=1,
            input_shape=(16, 16, 16),
        )
