"""Regression tests for vesuvius.find_patches CLI (issue #1525)."""
import sys

import numpy as np
import pytest
import zarr

from vesuvius.models.preprocessing.patches.cli import main

CONFIG = """\
tr_setup:
  model_name: test
tr_config:
  patch_size: [32, 32, 32]
dataset_config:
  targets:
    ink:
      out_channels: 1
"""


def _write_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG)
    return config_path


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["vesuvius.find_patches", *argv])
    main()


def test_exits_nonzero_when_no_volumes_found(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, ["--config", str(config_path)])
    assert excinfo.value.code == 1


def test_errors_on_missing_input_directory(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        _run(
            monkeypatch,
            ["--config", str(config_path), "-i", str(tmp_path / "missing")],
        )
    assert excinfo.value.code == 2
    # Assert on the message, not just the code: on pre-fix code the same
    # invocation also exits 2, but as "unrecognized arguments: -i" — and a
    # future regression that drops the exists() check while keeping -i would
    # change the failure mode without changing the number.
    assert "Input directory does not exist" in capsys.readouterr().err


def test_input_overrides_config_data_path(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)

    dataset = tmp_path / "dataset"
    rng = np.random.default_rng(0)
    image = zarr.open(str(dataset / "images" / "vol.zarr"), mode="w", shape=(64, 64, 64), dtype="uint8")
    image[:] = rng.integers(0, 255, (64, 64, 64), dtype=np.uint8)
    label = zarr.open(str(dataset / "labels" / "vol_ink.zarr"), mode="w", shape=(64, 64, 64), dtype="uint8")
    label[:] = 1

    _run(monkeypatch, ["--config", str(config_path), "-i", str(dataset)])

    assert (dataset / ".patches_cache").exists()
