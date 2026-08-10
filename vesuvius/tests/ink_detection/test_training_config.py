"""Ordered training configuration and frozen-view tests."""

from __future__ import annotations

import json

import pytest

from vesuvius.ink_detection.config import (
    TrainingConfig,
    resolve_training_mapping,
)
from vesuvius.ink_detection.train import stage_training_request

from .test_model_foundation import _config_mapping


def _training_mapping(*, mode: str = "flat") -> dict:
    authored = _config_mapping(depth=3, side=8)
    authored.update(
        {
            "mode": mode,
            "num_iterations": 6,
            "out_dir": "/tmp/ink-output",
            "seed": 17,
        }
    )
    return authored


def test_raw_relative_checkpoint_is_selected_before_canonical_mutation(tmp_path):
    authored = _training_mapping(mode="full_3d")
    authored.pop("in_channels")
    authored["targets"]["ink"].pop("out_channels")
    authored["targets"]["ink"].pop("activation")
    authored["checkpoint"] = "weights/start.pth"
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(authored), encoding="utf-8")
    selected = []

    def loader(path):
        selected.append(path)
        still_raw = json.loads(config_path.read_text(encoding="utf-8"))
        assert "ema" not in still_raw
        assert "crop_size" not in still_raw
        return {"model": {}}

    request = stage_training_request(config_path, checkpoint_loader=loader)
    canonical = request.config.to_mapping()

    assert selected == [tmp_path / "weights/start.pth"]
    assert request.checkpoint_path == tmp_path / "weights/start.pth"
    assert canonical["in_channels"] == 1
    assert canonical["model_config"]["z_projection_mode"] == "none"
    assert canonical["targets"]["ink"]["out_channels"] == 1
    assert canonical["targets"]["ink"]["activation"] == "none"


def test_canonical_mutations_are_complete_without_local_default_materialization():
    authored = _training_mapping()
    canonical = resolve_training_mapping(authored)
    training = TrainingConfig.from_mapping(canonical)

    assert list(canonical["ema"]) == [
        "enabled",
        "decay",
        "start_step",
        "update_every_steps",
        "validate",
        "save_in_checkpoint",
    ]
    assert canonical["ema"] == {
        "enabled": False,
        "decay": 0.999,
        "start_step": 0,
        "update_every_steps": 1,
        "validate": False,
        "save_in_checkpoint": False,
    }
    assert canonical["volume_auth_json"] is None
    assert canonical["crop_size"] == [3, 8, 8]
    assert canonical["patch_size"] == [3, 8, 8]
    assert canonical["stitch_factor"] == 1
    assert canonical["use_stitched_forward"] is False
    assert canonical["stitched_gradient_checkpointing"] is True
    for local_only in (
        "learning_rate",
        "grad_acc_steps",
        "max_steps",
        "val_every",
        "save_every",
        "dataloader_workers",
        "benchmark",
        "scheduler",
    ):
        assert local_only not in training.to_mapping()
    assert training.optimizer.learning_rate == 0.01
    assert training.dataloader_workers == 0
    assert training.ink.data.dataloader_workers == 8


def test_dinov2_nested_values_win_during_training_resolution():
    authored = _training_mapping()
    authored["model_type"] = "dinov2"
    authored["pretrained_backbone"] = "/top.pth"
    authored["pretrained_decoder_type"] = "top"
    authored["model_config"]["pretrained_backbone"] = "/nested.pth"
    authored["model_config"]["pretrained_decoder_type"] = "nested"

    canonical = resolve_training_mapping(authored)

    assert canonical["model_type"] == "vesuvius_unet"
    assert canonical["model_config"]["pretrained_backbone"] == "/nested.pth"
    assert canonical["model_config"]["pretrained_decoder_type"] == "nested"


def test_stitching_scales_only_loader_yx_and_native_forces_factor_one():
    flat = _training_mapping()
    flat["stitch_factor"] = 2
    flat_training = TrainingConfig.from_authored_mapping(flat)
    native = _training_mapping(mode="full_3d_single_wrap")
    native.pop("in_channels")
    native["stitch_factor"] = 2
    native_training = TrainingConfig.from_authored_mapping(native)

    assert flat_training.model_crop_size == (3, 8, 8)
    assert flat_training.loader_patch_size == (3, 16, 16)
    assert flat_training.use_stitched_forward is True
    assert native_training.loader_patch_size == (3, 8, 8)
    assert native_training.stitch_factor == 1
    assert native_training.use_stitched_forward is False
    assert native_training.ink.model.in_channels == 2


def test_training_seed_remains_required_instead_of_using_data_default():
    authored = _training_mapping()
    authored.pop("seed")

    with pytest.raises(KeyError, match="seed"):
        TrainingConfig.from_authored_mapping(authored)


def test_diffusers_warmup_is_top_level_not_nested_scheduler_value():
    authored = _training_mapping()
    authored["warmup_steps"] = 23
    authored["scheduler"] = {
        "name": "diffusers_cosine_warmup",
        "warmup_steps": 999,
    }

    training = TrainingConfig.from_authored_mapping(authored)

    assert training.scheduler.warmup_steps == 23
    assert training.to_mapping()["scheduler"]["warmup_steps"] == 999


def test_inactive_optimizer_and_scheduler_values_are_not_interpreted():
    authored = _training_mapping()
    authored.update(
        optimizer="sgd",
        optimizer_betas="unused-by-sgd",
        encoder_lr_mult=None,
        scheduler={
            "name": "diffusers_cosine_warmup",
            "t_max": None,
            "total_steps": None,
            "pct_start": None,
            "final_div_factor": None,
        },
    )

    training = TrainingConfig.from_authored_mapping(authored)

    assert training.optimizer.betas == "unused-by-sgd"
    assert training.optimizer.encoder_lr_mult is None
    assert training.scheduler.name == "diffusers_cosine_warmup"


def test_pin_memory_distinguishes_absence_from_explicit_null():
    absent = TrainingConfig.from_authored_mapping(_training_mapping())
    explicit = _training_mapping()
    explicit["pin_memory"] = None

    assert absent.pin_memory is None
    assert TrainingConfig.from_authored_mapping(explicit).pin_memory is False
