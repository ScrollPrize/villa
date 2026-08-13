from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.models.configuration.config_manager import ConfigManager


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


@pytest.fixture(scope="session")
def tiny_dinovol_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("checkpoints") / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(path)
    return path


def _load_mgr(rel_path: str, *, checkpoint: Path | None = None) -> ConfigManager:
    cfg = Path(rel_path)
    mgr = ConfigManager(verbose=False)
    mgr.load_config(str(cfg))
    mgr.spacing = (1.0, 1.0, 1.0)
    mgr.autoconfigure = False
    mgr.train_batch_size = 1
    if checkpoint is not None:
        ckpt_str = str(checkpoint)
        if "guide_backbone" in mgr.model_config and mgr.model_config["guide_backbone"] is not None:
            mgr.model_config["guide_backbone"] = ckpt_str
        if "pretrained_backbone" in mgr.model_config and mgr.model_config["pretrained_backbone"] is not None:
            mgr.model_config["pretrained_backbone"] = ckpt_str
    return mgr


def test_ps256_medial_config_loads_and_builds_without_guide_path():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is False
    assert model.final_config["guide_backbone"] is None


def test_ps256_dicece_config_loads_and_builds_without_guide_path():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_dicece.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is False
    assert model.final_config["guide_backbone"] is None


def test_ps256_guided_medial_config_loads_and_builds_with_guide_path(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_medial.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_backbone"] is not None
    assert model.final_config["guide_tokenbook_tokens"] == 256


def test_ps256_guided_dicece_config_loads_and_builds_with_guide_path(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_dicece.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_backbone"] is not None
    assert model.final_config["guide_tokenbook_tokens"] == 256


def test_ps128_guided_feature_encoder_config_loads_and_builds_with_encoder_guidance(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_guided_feature_encoder_ink.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (128, 128, 128)
    assert list(mgr.targets.keys()) == ["ink"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"
    assert model.final_config["guide_stage_keys"] == ["enc_0", "enc_1", "enc_2", "enc_3", "enc_4"]
    assert model.final_config["guide_feature_gate_alpha"] == 0.9
    assert model.final_config["guide_tokenbook_prototype_weighting"] == "token_mlp"


def test_ps256_guided_feature_encoder_medial_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_feature_encoder_medial.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"
    assert model.final_config["guide_tokenbook_tokens"] == 256
    assert model.final_config["guide_feature_gate_alpha"] == 0.9
    assert model.final_config["guide_tokenbook_prototype_weighting"] == "token_mlp"


def test_ps256_guided_feature_encoder_dicece_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_feature_encoder_dicece.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"
    assert model.final_config["guide_tokenbook_tokens"] == 256
    assert model.final_config["guide_feature_gate_alpha"] == 0.9
    assert model.final_config["guide_tokenbook_prototype_weighting"] == "token_mlp"


def test_ps128_guided_feature_skip_concat_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_guided_feature_skip_concat_ink.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (128, 128, 128)
    assert list(mgr.targets.keys()) == ["ink"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_skip_concat"
    assert model.final_config["guide_stage_keys"] == ["enc_0", "enc_1", "enc_2", "enc_3", "enc_4"]
    assert model.final_config["guide_skip_concat_projector_channels"] == [8, 16, 32, 64, 80]
    assert model.final_config["guide_feature_gate_alpha"] is None
    assert model.final_config["guide_tokenbook_tokens"] is None
    assert model.final_config["guide_tokenbook_prototype_weighting"] is None


def test_ps256_guided_feature_skip_concat_medial_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_feature_skip_concat_medial.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_skip_concat"
    assert model.final_config["guide_skip_concat_projector_channels"] == [8, 16, 32, 64, 80, 80, 80]
    assert model.final_config["guide_feature_gate_alpha"] is None
    assert model.final_config["guide_tokenbook_tokens"] is None
    assert model.final_config["guide_tokenbook_prototype_weighting"] is None


def test_ps256_guided_feature_skip_concat_dicece_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_feature_skip_concat_dicece.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_skip_concat"
    assert model.final_config["guide_skip_concat_projector_channels"] == [8, 16, 32, 64, 80, 80, 80]
    assert model.final_config["guide_feature_gate_alpha"] is None
    assert model.final_config["guide_tokenbook_tokens"] is None
    assert model.final_config["guide_tokenbook_prototype_weighting"] is None


def test_ps128_guided_direct_segmentation_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_guided_direct_segmentation_ink.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (128, 128, 128)
    assert list(mgr.targets.keys()) == ["ink"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "direct_segmentation"
    assert model.final_config["guide_direct_output_mode"] == "two_channel_logits"


def test_ps256_guided_direct_segmentation_medial_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_direct_segmentation_medial.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "direct_segmentation"
    assert model.final_config["guide_direct_output_mode"] == "two_channel_logits"


def test_ps256_guided_direct_segmentation_dicece_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_direct_segmentation_dicece.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "direct_segmentation"
    assert model.final_config["guide_direct_output_mode"] == "two_channel_logits"


def test_ps128_pretrained_dino_pixelshuffle_medial_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_pretrained_dino_pixelshuffle_medial.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (128, 128, 128)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.final_config["pretrained_backbone"] is not None
    assert model.final_config["pretrained_decoder_type"] == "pixelshuffle_conv"
    assert model.final_config["freeze_encoder"] is True


def test_ps256_pretrained_dino_pixelshuffle_medial_config_loads_and_builds(tiny_dinovol_checkpoint):
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_pretrained_dino_pixelshuffle_medial.yaml", checkpoint=tiny_dinovol_checkpoint)
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.final_config["pretrained_backbone"] is not None
    assert model.final_config["pretrained_decoder_type"] == "pixelshuffle_conv"
    assert model.final_config["freeze_encoder"] is True


def test_ps128_mednext_v1_medial_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_mednext_v1_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (128, 128, 128)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.final_config["architecture_type"] == "mednext_v1"
    assert model.final_config["mednext_model_id"] == "B"


def test_ps256_mednext_v1_medial_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_mednext_v1_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.final_config["architecture_type"] == "mednext_v1"
    assert model.final_config["mednext_model_id"] == "B"


def test_ps128_mednext_v2_l_medial_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_mednext_v2_l_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (128, 128, 128)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.final_config["architecture_type"] == "mednext_v2"
    assert model.final_config["mednext_model_id"] == "L"
    assert model.final_config["mednext_grn"] is True


def test_ps192_mednext_v2_l_medial_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps192_mednext_v2_l_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (192, 192, 192)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.final_config["architecture_type"] == "mednext_v2"
    assert model.final_config["mednext_model_id"] == "L"


def test_ps128_mednext_v2_b_medial_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_mednext_v2_b_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (128, 128, 128)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.final_config["architecture_type"] == "mednext_v2"
    assert model.final_config["mednext_model_id"] == "B"


def test_ps128_mednext_v2_l_width2_medial_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_mednext_v2_l_width2_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (128, 128, 128)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.final_config["architecture_type"] == "mednext_v2"
    assert model.final_config["mednext_model_id"] == "L"
    assert model.final_config["mednext_width_factor"] == 2
