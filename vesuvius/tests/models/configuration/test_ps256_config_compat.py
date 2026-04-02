from __future__ import annotations

from pathlib import Path

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.configuration.config_manager import ConfigManager


def _load_mgr(rel_path: str) -> ConfigManager:
    cfg = Path(rel_path)
    mgr = ConfigManager(verbose=False)
    mgr.load_config(str(cfg))
    mgr.spacing = (1.0, 1.0, 1.0)
    mgr.autoconfigure = False
    mgr.train_batch_size = 1
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


def test_ps256_guided_medial_config_loads_and_builds_with_guide_path():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_backbone"] is not None
    assert model.final_config["guide_tokenbook_tokens"] == 256


def test_ps256_guided_dicece_config_loads_and_builds_with_guide_path():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_dicece.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_backbone"] is not None
    assert model.final_config["guide_tokenbook_tokens"] == 256


def test_ps128_guided_feature_encoder_config_loads_and_builds_with_encoder_guidance():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_guided_feature_encoder_ink.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (128, 128, 128)
    assert list(mgr.targets.keys()) == ["ink"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"
    assert model.final_config["guide_stage_keys"] == ["enc_0", "enc_1", "enc_2", "enc_3", "enc_4"]
    assert model.final_config["guide_feature_gate_alpha"] == 0.9
    assert model.final_config["guide_tokenbook_prototype_weighting"] == "token_mlp"


def test_ps256_guided_feature_encoder_medial_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_feature_encoder_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"
    assert model.final_config["guide_tokenbook_tokens"] == 256
    assert model.final_config["guide_feature_gate_alpha"] == 0.9
    assert model.final_config["guide_tokenbook_prototype_weighting"] == "token_mlp"


def test_ps256_guided_feature_encoder_dicece_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_feature_encoder_dicece.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_encoder"
    assert model.final_config["guide_tokenbook_tokens"] == 256
    assert model.final_config["guide_feature_gate_alpha"] == 0.9
    assert model.final_config["guide_tokenbook_prototype_weighting"] == "token_mlp"


def test_ps128_guided_feature_skip_concat_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps128_guided_feature_skip_concat_ink.yaml")
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


def test_ps256_guided_feature_skip_concat_medial_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_feature_skip_concat_medial.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_skip_concat"
    assert model.final_config["guide_skip_concat_projector_channels"] == [8, 16, 32, 64, 80, 80, 80]
    assert model.final_config["guide_feature_gate_alpha"] is None
    assert model.final_config["guide_tokenbook_tokens"] is None
    assert model.final_config["guide_tokenbook_prototype_weighting"] is None


def test_ps256_guided_feature_skip_concat_dicece_config_loads_and_builds():
    mgr = _load_mgr("src/vesuvius/models/configuration/single_task/ps256_guided_feature_skip_concat_dicece.yaml")
    model = NetworkFromConfig(mgr)

    assert mgr.train_patch_size == (256, 256, 256)
    assert list(mgr.targets.keys()) == ["surface"]
    assert model.guide_enabled is True
    assert model.final_config["guide_fusion_stage"] == "feature_skip_concat"
    assert model.final_config["guide_skip_concat_projector_channels"] == [8, 16, 32, 64, 80, 80, 80]
    assert model.final_config["guide_feature_gate_alpha"] is None
    assert model.final_config["guide_tokenbook_tokens"] is None
    assert model.final_config["guide_tokenbook_prototype_weighting"] is None
