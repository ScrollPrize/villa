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
