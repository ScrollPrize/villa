"""vesuvius.train must not let argparse defaults override tr_config (issue #1489).

``--max-epoch`` etc. carried non-None argparse defaults (1000 / 250 / 50 / 1),
and ``update_config_from_args`` could not tell "user passed 1000" from
"default 1000", so ``tr_config: {max_epoch: 50, max_steps_per_epoch: 150}``
still trained for ``Epoch 1/1000`` with 250 steps. ``--batch-size`` and
``--patch-size`` already used ``default=None``; the schedule flags now do too,
and the old defaults apply only when neither the flag nor the YAML sets a value.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.training.cli import build_parser
from vesuvius.models.utilities.cli_utils import CLI_SCHEDULE_DEFAULTS, update_config_from_args

BASE_CONFIG = {
    "tr_setup": {"model_name": "schedule_test"},
    "tr_config": {"patch_size": [64, 64, 64], "enable_deep_supervision": False},
    "model_config": {"architecture_type": "mednext_v1", "mednext_model_id": "B"},
    "dataset_config": {
        "normalization_scheme": "zscore",
        "skip_patch_validation": True,
        "targets": {"surface": {"activation": "none", "losses": [{"name": "SoftDiceLoss", "weight": 1.0}]}},
    },
}


def _resolve(tmp_path: Path, tr_config_extra: dict, argv_extra: list[str]) -> ConfigManager:
    config = yaml.safe_load(yaml.safe_dump(BASE_CONFIG))
    config["tr_config"].update(tr_config_extra)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    args = build_parser().parse_args(
        ["--config", str(config_path), "--input", str(data_dir), "--format", "zarr",
         "--output", str(tmp_path / "out"), *argv_extra]
    )
    mgr = ConfigManager(verbose=False)
    mgr.load_config(str(config_path))
    update_config_from_args(mgr, args)
    return mgr


def test_schedule_flags_default_to_none():
    args = build_parser().parse_args(["--config", "x.yaml"])
    assert args.max_epoch is None
    assert args.max_steps_per_epoch is None
    assert args.max_val_steps_per_epoch is None
    assert args.val_every_n is None
    assert args.early_stopping_patience is None


def test_yaml_schedule_survives_without_flags(tmp_path):
    mgr = _resolve(
        tmp_path,
        {"max_epoch": 50, "max_steps_per_epoch": 150, "max_val_steps_per_epoch": 12, "val_every_n": 3},
        [],
    )
    assert (mgr.max_epoch, mgr.max_steps_per_epoch, mgr.max_val_steps_per_epoch, mgr.val_every_n) == (50, 150, 12, 3)
    assert mgr.tr_configs["max_epoch"] == 50
    assert mgr.tr_configs["max_steps_per_epoch"] == 150


def test_explicit_flags_override_yaml(tmp_path):
    mgr = _resolve(
        tmp_path,
        {"max_epoch": 50, "max_steps_per_epoch": 150, "max_val_steps_per_epoch": 12, "val_every_n": 3},
        ["--max-epoch", "7", "--max-steps-per-epoch", "9", "--max-val-steps-per-epoch", "4", "--val-every-n", "2"],
    )
    assert (mgr.max_epoch, mgr.max_steps_per_epoch, mgr.max_val_steps_per_epoch, mgr.val_every_n) == (7, 9, 4, 2)


def test_historical_defaults_apply_when_neither_is_set(tmp_path):
    mgr = _resolve(tmp_path, {}, [])
    assert mgr.max_epoch == CLI_SCHEDULE_DEFAULTS["max_epoch"] == 1000
    assert mgr.max_steps_per_epoch == CLI_SCHEDULE_DEFAULTS["max_steps_per_epoch"] == 250
    assert mgr.max_val_steps_per_epoch == CLI_SCHEDULE_DEFAULTS["max_val_steps_per_epoch"] == 50
    assert mgr.val_every_n == 1


def test_yaml_early_stopping_patience_survives(tmp_path):
    mgr = _resolve(tmp_path, {"early_stopping_patience": 15}, [])
    assert mgr.early_stopping_patience == 15
    mgr = _resolve(tmp_path, {"early_stopping_patience": 15}, ["--early-stopping-patience", "0"])
    assert mgr.early_stopping_patience == 0


def test_full_epoch_still_clears_step_limits(tmp_path):
    mgr = _resolve(tmp_path, {"max_steps_per_epoch": 150}, ["--full-epoch"])
    assert mgr.max_steps_per_epoch is None
    assert mgr.max_val_steps_per_epoch is None


def test_val_every_n_below_one_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="--val-every-n must be >= 1"):
        _resolve(tmp_path, {}, ["--val-every-n", "0"])
    # the YAML value is live now, so it gets the same check
    with pytest.raises(ValueError, match="tr_config.val_every_n must be >= 1"):
        _resolve(tmp_path, {"val_every_n": 0}, [])
