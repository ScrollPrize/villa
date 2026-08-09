import pytest
import json

import torch

from koine_machines.training.train import (
    _append_jsonl,
    _benchmark_summary,
    _disable_z_projection_for_normal_pooled_3d,
    _full_3d_dilation_distances_for_level,
    _masked_unsmoothed_bce_with_logits,
)


def test_benchmark_summary_reports_global_throughput_and_memory():
    summary = _benchmark_summary(
        elapsed_seconds=4.0,
        measured_steps=10,
        batch_size=8,
        world_size=2,
        data_wait_seconds=1.0,
        peak_allocated_bytes=123,
        peak_reserved_bytes=456,
    )

    assert summary['steps_per_second'] == 2.5
    assert summary['examples'] == 160
    assert summary['examples_per_second'] == 40.0
    assert summary['data_wait_seconds_per_step'] == 0.1
    assert summary['peak_allocated_bytes'] == 123
    assert summary['peak_reserved_bytes'] == 456


def test_append_jsonl_persists_machine_readable_validation_records(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _append_jsonl(path, {"step": 4, "val_loss": 0.5})
    _append_jsonl(path, {"step": 8, "val_loss": 0.4})

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"step": 4, "val_loss": 0.5},
        {"step": 8, "val_loss": 0.4},
    ]


def test_masked_unsmoothed_bce_uses_hard_targets_and_ignores_invalid_pixels():
    logits = torch.tensor([[[[4.0, -4.0, -20.0]]]])
    targets = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    ignore_mask = torch.tensor([[[[0.0, 0.0, 1.0]]]])

    actual = _masked_unsmoothed_bce_with_logits(logits, targets, ignore_mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[..., :2],
        targets[..., :2],
    )

    torch.testing.assert_close(actual, expected)


def test_full_3d_dilation_distances_scale_with_native_volume_level():
    config = {
        "full_3d": {
            "label_dilation_distance": 4.0,
            "supervision_dilation_distance": 8.0,
        },
        "datasets": [{"volume_scale": "2"}, {"volume_scale": 2}],
    }

    assert _full_3d_dilation_distances_for_level(config) == (1.0, 2.0)


def test_full_3d_dilation_distances_disabled_without_reading_datasets():
    assert _full_3d_dilation_distances_for_level({"full_3d": {}}) == (0.0, 0.0)


def test_full_3d_dilation_distances_reject_mixed_volume_scales():
    config = {
        "full_3d": {"label_dilation_distance": 4.0},
        "datasets": [{"volume_scale": "0"}, {"volume_scale": "2"}],
    }

    with pytest.raises(ValueError, match="single volume_scale"):
        _full_3d_dilation_distances_for_level(config)


def test_disable_z_projection_for_normal_pooled_3d_forces_projection_off():
    config = {
        "mode": "normal_pooled_3d",
        "model_config": {"z_projection_mode": "logsumexp"},
        "targets": {
            "ink": {
                "z_projection_mode": "mean",
                "z_projection": {
                    "mode": "learned_mlp",
                    "z_projection_mlp_hidden": 64,
                },
            },
            "aux": {
                "z_projection_mode": "max",
            },
        },
    }

    _disable_z_projection_for_normal_pooled_3d(config)

    assert config["model_config"]["z_projection_mode"] == "none"
    assert config["targets"]["ink"]["z_projection_mode"] == "none"
    assert config["targets"]["ink"]["z_projection"]["mode"] == "none"
    assert config["targets"]["aux"]["z_projection_mode"] == "none"
