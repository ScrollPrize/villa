import json
from pathlib import Path

import pytest

from flatten_spiral_checkpoint import (
    MODEL_CONFIG_KEYS,
    _checkpoint_config,
    _resolve_lasagna,
    _resolve_umbilicus,
)


def test_legacy_checkpoint_model_config_gets_current_aliases():
    legacy = {key: index for index, key in enumerate(MODEL_CONFIG_KEYS)}
    config = _checkpoint_config({"cfg": legacy})
    for key in MODEL_CONFIG_KEYS:
        assert config[f"model_{key}"] == legacy[key]


def test_current_checkpoint_model_config_is_preserved():
    current = {
        f"model_{key}": index for index, key in enumerate(MODEL_CONFIG_KEYS)
    }
    config = _checkpoint_config({"cfg": current})
    assert config == current


def test_resolve_umbilicus_prefers_explicit_path(tmp_path):
    path = tmp_path / "umbilicus.json"
    path.write_text(json.dumps({"control_points": []}))
    checkpoint = tmp_path / "checkpoint.ckpt"
    assert _resolve_umbilicus(checkpoint, path) == path.resolve()


def test_resolve_umbilicus_finds_checkpoint_ancestor(tmp_path):
    path = tmp_path / "umbilicus.json"
    path.write_text(json.dumps({"control_points": []}))
    checkpoint = tmp_path / "spiral_output" / "run" / "checkpoint.ckpt"
    assert _resolve_umbilicus(checkpoint, None) == path.resolve()


def test_resolve_lasagna_requires_service_and_config(tmp_path):
    (tmp_path / "fit_service.py").write_text("")
    config = tmp_path / "configs" / "flatten_fast_nofilter.json"
    config.parent.mkdir()
    config.write_text("{}")
    assert _resolve_lasagna(tmp_path) == (
        (tmp_path / "fit_service.py").resolve(),
        config.resolve(),
    )


def test_checkpoint_config_reports_missing_fields():
    with pytest.raises(ValueError, match="missing model configuration"):
        _checkpoint_config({"cfg": {}})
