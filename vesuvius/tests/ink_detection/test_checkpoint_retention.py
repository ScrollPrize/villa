"""keep_last_checkpoints: config parsing and periodic checkpoint pruning."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping
from vesuvius.ink_detection.training.train import _run_training, stage_training_request

from .helpers import config_mapping
from .test_training_numerics import (
    _SyntheticTrainingDataset,
    _SyntheticTrainingLoss,
    _SyntheticTrainingModel,
    _training_mapping,
)


def _parse(**extra) -> TrainingConfig:
    authored = config_mapping(depth=3, side=8)
    authored.update({"mode": "flat", "num_iterations": 6, "out_dir": "/tmp/ink-output", "seed": 17})
    authored.update(extra)
    return TrainingConfig.from_mapping(resolve_training_mapping(authored))


def _touch(directory, *names):
    for name in names:
        (directory / name).write_bytes(b"x")


def test_keep_last_checkpoints_defaults_to_keeping_everything():
    assert _parse().keep_last_checkpoints is None
    assert _parse(keep_last_checkpoints=None).keep_last_checkpoints is None


def test_keep_last_checkpoints_accepts_a_positive_integer():
    assert _parse(keep_last_checkpoints=2).keep_last_checkpoints == 2


@pytest.mark.parametrize(
    ("value", "error"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError), ("2", TypeError)],
)
def test_keep_last_checkpoints_rejects_invalid_values(value, error):
    with pytest.raises(error):
        _parse(keep_last_checkpoints=value)


def test_training_prunes_only_its_own_periodic_checkpoints(tmp_path, monkeypatch):
    import vesuvius.ink_detection.data.dataset as dataset_module
    import vesuvius.ink_detection.models.model as model_module
    import vesuvius.ink_detection.training.losses as losses_module
    import vesuvius.ink_detection.training.optimizers as optimizers_module
    import vesuvius.ink_detection.training.samplers as samplers_module

    authored = _training_mapping()
    authored.update(
        num_iterations=8, out_dir=str(tmp_path), mixed_precision="no", dataloader_workers=0,
        val_every=99, save_every=2, log_every=99, save_iterations=[4], keep_last_checkpoints=1,
    )
    authored["model_config"]["pretrained_backbone"] = "synthetic"
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(authored), encoding="utf-8")
    # Earlier runs' files: one this run overwrites, a newer one and another config's milestone.
    _touch(tmp_path, "ckpt_000002.pth", "ckpt_000030.pth", "ckpt_000005.pth")

    monkeypatch.setattr(dataset_module, "InkDataset", _SyntheticTrainingDataset)
    monkeypatch.setattr(model_module, "make_model", lambda config: _SyntheticTrainingModel())
    monkeypatch.setattr(losses_module, "create_loss", lambda config: _SyntheticTrainingLoss())
    monkeypatch.setattr(
        optimizers_module,
        "create_training_optimizer",
        lambda model, config: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(
        samplers_module,
        "build_sampling_policy",
        lambda patches, config, batch_size: SimpleNamespace(
            batch_sampler=None, shuffle=False, sampler=None, generator=None, audit={}
        ),
    )

    assert _run_training(stage_training_request(config_path)) == 0

    assert sorted(path.name for path in tmp_path.glob("ckpt_*.pth")) == [
        "ckpt_000002.pth",
        "ckpt_000004.pth",
        "ckpt_000005.pth",
        "ckpt_000008.pth",
        "ckpt_000030.pth",
    ]
