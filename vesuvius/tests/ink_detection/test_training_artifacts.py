"""Training preview, JSONL, cadence, and import-purity tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import tifffile
import torch

from vesuvius.ink_detection.training.train import (
    append_validation_metrics,
    benchmark_summary,
    build_training_checkpoint_payload,
)

from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping

from .test_model_foundation import _config_mapping
from vesuvius.ink_detection.training.visualization import (
    PreviewAccumulator,
    central_full_3d_preview,
    to_uint8_image,
    to_uint8_label,
    to_uint8_probability,
)


class _LocalAccelerator:
    is_main_process = True

    @staticmethod
    def gather_for_metrics(value):
        return value

    @staticmethod
    def get_state_dict(model):
        return model.state_dict()


def test_preview_literal_encoding_width_and_lzw_readback(tmp_path):
    assert to_uint8_image([[2.0, 4.0]]).tolist() == [[0, 255]]
    assert to_uint8_label([[0.0, 1.0]], [[0.0, 1.0]]).tolist() == [
        [0, 127]
    ]
    assert to_uint8_probability([[0.0, 0.5, 1.0]]).tolist() == [
        [0, 128, 255]
    ]
    batch = {"image": torch.tensor([[[[[2.0, 4.0], [6.0, 8.0]]]]])}
    accumulator = PreviewAccumulator(
        accelerator=_LocalAccelerator(),
        get_model_input=lambda value: value["image"],
    )
    accumulator.add_batch(
        batch,
        torch.zeros(1, 1, 2, 2),
        torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]),
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
    )
    output_path = tmp_path / "preview.tif"
    accumulator.save(output_path)

    observed = tifffile.imread(output_path)
    assert observed.dtype == np.uint8
    assert observed.shape == (2, 14)
    assert observed[0].tolist() == [
        0,
        85,
        0,
        0,
        0,
        0,
        0,
        127,
        0,
        0,
        0,
        0,
        128,
        128,
    ]


def test_full_3d_preview_selects_central_z_and_first_image_channel():
    batch = {
        "image": torch.arange(10.0).reshape(1, 1, 5, 1, 2),
        "supervision_mask": torch.ones(1, 1, 5, 1, 2),
        "surface_mask": torch.full((1, 1, 5, 1, 2), 99.0),
    }
    values = torch.arange(10.0).reshape(1, 1, 5, 1, 2)

    preview_batch, predictions, targets, ignore = central_full_3d_preview(
        batch, values, values + 20, values + 40
    )

    torch.testing.assert_close(
        preview_batch["image"], torch.tensor([[[[[4.0, 5.0]]]]])
    )
    torch.testing.assert_close(predictions, torch.tensor([[[[4.0, 5.0]]]]))
    torch.testing.assert_close(targets, torch.tensor([[[[24.0, 25.0]]]]))
    torch.testing.assert_close(ignore, torch.tensor([[[[44.0, 45.0]]]]))


def test_validation_jsonl_has_exact_schema_sorted_bytes_and_appends(tmp_path):
    record = {
        "step": 3,
        "val_loss": 1.25,
        "val_bce_unsmoothed": 0.5,
        "val_balanced_accuracy": 0.75,
        "val_tp": 2.0,
        "val_fp": 1.0,
        "val_fn": 3.0,
        "val_tn": 4.0,
        "val_batches": 2,
        "learning_rate": 0.01,
    }
    path = tmp_path / "validation_metrics.jsonl"

    append_validation_metrics(path, record)
    append_validation_metrics(path, record)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == json.dumps(record, sort_keys=True)
    assert json.loads(lines[0]) == record
    assert set(json.loads(lines[0])) == {
        "step",
        "val_loss",
        "val_bce_unsmoothed",
        "val_balanced_accuracy",
        "val_tp",
        "val_fp",
        "val_fn",
        "val_tn",
        "val_batches",
        "learning_rate",
    }
    assert "ema" not in lines[0]


def test_benchmark_summary_uses_global_examples_and_cpu_zero_peaks():
    summary = benchmark_summary(
        elapsed_seconds=2.0,
        measured_steps=4,
        batch_size=3,
        world_size=2,
        data_wait_seconds=0.5,
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
    )

    assert summary == {
        "elapsed_seconds": 2.0,
        "measured_steps": 4,
        "batch_size_per_process": 3,
        "world_size": 2,
        "examples": 24,
        "steps_per_second": 2.0,
        "examples_per_second": 12.0,
        "data_wait_seconds": 0.5,
        "data_wait_seconds_per_step": 0.125,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
    }


def test_checkpoint_payload_uses_canonical_mapping_and_optional_ema_state():
    authored = _config_mapping(depth=1, side=4)
    authored.update(
        {
            "num_iterations": 6,
            "out_dir": "/tmp/ink-output",
            "seed": 9,
            "wandb_run_id": "run-123",
            "ema": {"enabled": True, "save_in_checkpoint": True},
            "scheduler": {"name": "cosine_annealing"},
        }
    )
    config = TrainingConfig.from_mapping(resolve_training_mapping(authored))
    model = torch.nn.Linear(1, 1)
    ema_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    payload = build_training_checkpoint_payload(
        accelerator=_LocalAccelerator(),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=2,
        optimizer_step=3,
        ema_model=ema_model,
        wandb_module=None,
        validation_metrics={"val_loss": 0.25},
    )

    assert list(payload) == [
        "model",
        "optimizer",
        "lr_scheduler",
        "config",
        "step",
        "wandb_run_id",
        "validation_metrics",
        "ema_model",
        "ema_optimizer_step",
    ]
    assert payload["step"] == 2
    assert payload["wandb_run_id"] == "run-123"
    assert payload["ema_optimizer_step"] == 3
    assert payload["validation_metrics"] == {"val_loss": 0.25}
    assert "learning_rate" not in payload["config"]
    assert payload["config"]["ema"]["enabled"] is True


def test_train_help_is_import_pure_without_accelerate_or_wandb():
    source_root = Path(__file__).parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    script = (
        "import sys; import vesuvius; baseline = set(sys.modules); "
        "import vesuvius.ink_detection.training.train as train; "
        "added = set(sys.modules) - baseline; "
        "assert 'accelerate' not in sys.modules; "
        "assert 'wandb' not in sys.modules; "
        "assert not any(n == 'cupy' or n.startswith('cupy.') for n in added); "
        "assert not any(n == 'cucim' or n.startswith('cucim.') for n in added); "
        "train.main(['-h'])"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "config_path" in completed.stdout
