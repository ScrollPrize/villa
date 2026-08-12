"""Ordered training configuration and frozen-view tests."""

from __future__ import annotations

import json

import pytest

import vesuvius.ink_detection.config as config_module
import vesuvius.ink_detection.training.train as train_module
from vesuvius.ink_detection.config import (
    InkDataConfig,
    TrainingConfig,
    resolve_training_mapping,
)
from vesuvius.ink_detection.data.dataset import InkDataset
from vesuvius.ink_detection.training.train import (
    stage_training_request,
    training_dataset_config,
)

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


def _parse_training(authored: dict) -> TrainingConfig:
    return TrainingConfig.from_mapping(resolve_training_mapping(authored))


def test_relative_checkpoint_resolves_against_the_config_directory(
    tmp_path, monkeypatch
):
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
        return {"model": {}}

    monkeypatch.setattr(train_module, "load_checkpoint", loader)
    request = stage_training_request(config_path)
    canonical = request.config.to_mapping()

    assert selected == [tmp_path / "weights/start.pth"]
    assert request.checkpoint_path == tmp_path / "weights/start.pth"
    assert canonical["in_channels"] == 1
    assert canonical["model_config"]["z_projection_mode"] == "none"
    assert canonical["targets"]["ink"]["out_channels"] == 1
    assert canonical["targets"]["ink"]["activation"] == "none"


def test_weights_only_flows_through_the_typed_checkpoint_config(
    tmp_path, monkeypatch
):
    authored = _training_mapping()
    authored["checkpoint"] = "weights/start.pth"
    authored["weights_only"] = True
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(authored), encoding="utf-8")
    monkeypatch.setattr(
        train_module, "load_checkpoint", lambda path: {"model": {}}
    )

    request = stage_training_request(config_path)

    assert request.config.ink.checkpoint.weights_only is True


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("out_channels", 2, "out_channels must be 1"),
        ("activation", "sigmoid", "activation must be 'none'"),
    ],
)
def test_training_rejects_contradicted_forced_target_settings(key, value, message):
    authored = _training_mapping()
    authored["targets"]["ink"][key] = value

    with pytest.raises(ValueError, match=message):
        resolve_training_mapping(authored)


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
    assert "volume_auth_json" not in canonical
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
    flat_training = _parse_training(flat)
    native = _training_mapping(mode="full_3d_single_wrap")
    native.pop("in_channels")
    native["stitch_factor"] = 2
    native_training = _parse_training(native)

    assert flat_training.model_crop_size == (3, 8, 8)
    assert flat_training.loader_patch_size == (3, 16, 16)
    assert flat_training.use_stitched_forward is True
    assert native_training.loader_patch_size == (3, 8, 8)
    assert native_training.use_stitched_forward is False
    assert native_training.ink.model.in_channels == 2


def test_training_seed_remains_required_instead_of_using_data_default():
    authored = _training_mapping()
    authored.pop("seed")

    with pytest.raises(KeyError, match="seed"):
        _parse_training(authored)


def test_diffusers_warmup_is_top_level_not_nested_scheduler_value():
    authored = _training_mapping()
    authored["warmup_steps"] = 23
    authored["scheduler"] = {
        "name": "diffusers_cosine_warmup",
        "warmup_steps": 999,
    }

    training = _parse_training(authored)

    assert training.scheduler.warmup_steps == 23
    assert training.to_mapping()["scheduler"]["warmup_steps"] == 999


def test_optimizer_values_are_typed_even_when_inactive():
    authored = _training_mapping()
    authored.update(
        optimizer="sgd",
        optimizer_betas=["0.8", "0.95"],
        optimizer_momentum="0.75",
        optimizer_nesterov=0,
        encoder_lr_mult="0.25",
        scheduler={
            "name": "diffusers_cosine_warmup",
            "t_max": None,
            "total_steps": None,
            "pct_start": None,
            "final_div_factor": None,
        },
    )

    training = _parse_training(authored)

    assert training.optimizer.betas == (0.8, 0.95)
    assert training.optimizer.momentum == 0.75
    assert training.optimizer.nesterov is False
    assert training.optimizer.encoder_lr_mult == 0.25
    assert training.scheduler.name == "diffusers_cosine_warmup"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("optimizer_betas", "not-a-pair", "optimizer_betas"),
        ("optimizer_betas", [0.9], "optimizer_betas"),
        ("optimizer_momentum", None, "float"),
        ("encoder_lr_mult", None, "float"),
    ],
)
def test_optimizer_values_fail_at_the_config_boundary(key, value, message):
    authored = _training_mapping()
    authored[key] = value

    with pytest.raises((TypeError, ValueError), match=message):
        _parse_training(authored)


def test_pin_memory_distinguishes_absence_from_explicit_null():
    absent = _parse_training(_training_mapping())
    explicit = _training_mapping()
    explicit["pin_memory"] = None

    assert absent.pin_memory is None
    assert _parse_training(explicit).pin_memory is False


@pytest.mark.parametrize("targets", [None, {}, {"other": {}}])
def test_training_target_presence_is_validated_before_mutation(targets):
    authored = _training_mapping(mode="full_3d")
    authored["targets"] = targets

    with pytest.raises(
        ValueError,
        match="targets must be a non-empty object containing 'ink'",
    ):
        resolve_training_mapping(authored)


def test_training_target_values_fail_factually_before_native_mutation():
    authored = _training_mapping(mode="full_3d")
    authored["targets"]["ink"] = "not-an-object"

    with pytest.raises(TypeError, match="targets.ink must be an object"):
        resolve_training_mapping(authored)


def test_training_model_canonicalization_runs_once(monkeypatch):
    calls = []
    original = config_module._canonical_model_mapping

    def counted(authored):
        calls.append(authored)
        return original(authored)

    monkeypatch.setattr(config_module, "_canonical_model_mapping", counted)

    _parse_training(_training_mapping())

    assert len(calls) == 1


def test_typed_runtime_fields_preserve_reference_precedence():
    top_level = _training_mapping()
    top_level["model_config"]["pretrained_backbone"] = "/weights.pth"
    top_level["freeze_encoder"] = True
    top_level["model_config"]["spacing"] = [2, 3, 4]
    top_level["wandb_run_id"] = 123
    top_level["wandb_resume"] = 1

    training = _parse_training(top_level)

    assert training.ink.model.pretrained_backbone == "/weights.pth"
    assert training.ink.model.freeze_encoder is True
    assert training.ink.model.spacing == (2.0, 3.0, 4.0)
    assert training.wandb_run_id == "123"
    assert training.wandb_resume is True

    nested = _training_mapping()
    nested["model_config"]["pretrained_backbone"] = "/weights.pth"
    nested["model_config"]["freeze_encoder"] = True
    assert _parse_training(nested).ink.model.freeze_encoder is True

    no_backbone = _training_mapping()
    no_backbone["freeze_encoder"] = True
    no_backbone["model_config"]["freeze_encoder"] = True
    assert _parse_training(no_backbone).ink.model.freeze_encoder is False

    falsey = _training_mapping()
    falsey["model_config"]["pretrained_backbone"] = False
    falsey["wandb_run_id"] = 0
    falsey_training = _parse_training(falsey)
    assert falsey_training.ink.model.pretrained_backbone is None
    assert falsey_training.wandb_run_id is None


def test_data_and_training_defaults_remain_deliberately_forked():
    authored = _training_mapping()
    data_authored = dict(authored)
    data_authored.pop("out_dir")
    data_authored.pop("dataloader_workers", None)
    data_authored.pop("seed")

    data = InkDataConfig.from_mapping(data_authored)

    assert (str(data.out_dir), data.dataloader_workers, data.seed) == (".", 8, 0)
    assert _parse_training(authored).dataloader_workers == 0
    for required in ("out_dir", "seed"):
        missing = _training_mapping()
        missing.pop(required)
        with pytest.raises(KeyError, match=required):
            _parse_training(missing)


def test_staged_training_rejects_a_non_object_at_the_config_boundary(tmp_path):
    config_path = tmp_path / "training.json"
    config_path.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="ink training config must be an object"):
        stage_training_request(config_path)


def test_training_config_constructs_a_synthetic_dataset(tmp_path):
    authored = _training_mapping()
    authored["out_dir"] = str(tmp_path / "output")
    authored["datasets"][0]["segments_path"] = str(tmp_path / "segments")
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(authored), encoding="utf-8")

    request = stage_training_request(config_path)
    data_config = training_dataset_config(request.config)
    dataset = InkDataset(
        data_config,
        do_augmentations=False,
        patches=[],
    )

    assert dataset.config.patch_size == request.config.loader_patch_size
    assert dataset.config.seed == request.config.seed
    assert len(dataset) == 0
