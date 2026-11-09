"""keep_last_checkpoints: config parsing and periodic checkpoint pruning."""

from __future__ import annotations

import pytest

from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping
from vesuvius.ink_detection.training.train import prune_periodic_checkpoints

from .helpers import config_mapping


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


def test_prune_keeps_the_newest_periodic_checkpoints(tmp_path):
    _touch(tmp_path, "ckpt_000500.pth", "ckpt_001000.pth", "ckpt_001500.pth", "ckpt_002000.pth")

    removed = prune_periodic_checkpoints(tmp_path, keep=2)

    assert sorted(path.name for path in removed) == ["ckpt_000500.pth", "ckpt_001000.pth"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ckpt_001500.pth", "ckpt_002000.pth"]


def test_prune_orders_by_iteration_not_by_name_width(tmp_path):
    _touch(tmp_path, "ckpt_999000.pth", "ckpt_1000000.pth")

    prune_periodic_checkpoints(tmp_path, keep=1)

    assert [path.name for path in tmp_path.iterdir()] == ["ckpt_1000000.pth"]


def test_prune_never_deletes_milestones_best_or_other_files(tmp_path):
    _touch(
        tmp_path,
        "ckpt_000500.pth",
        "ckpt_001000.pth",
        "ckpt_001500.pth",
        "best_val_loss.pth",
        "ckpt_final.pth",
        "notes.txt",
    )

    prune_periodic_checkpoints(tmp_path, keep=1, save_iterations=(500,))

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "best_val_loss.pth",
        "ckpt_000500.pth",
        "ckpt_001500.pth",
        "ckpt_final.pth",
        "notes.txt",
    ]


def test_prune_is_a_no_op_when_under_the_limit(tmp_path):
    _touch(tmp_path, "ckpt_000500.pth")

    assert prune_periodic_checkpoints(tmp_path, keep=3) == []
    assert [path.name for path in tmp_path.iterdir()] == ["ckpt_000500.pth"]
