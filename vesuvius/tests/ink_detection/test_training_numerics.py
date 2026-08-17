"""Optimizer, supervision, stitching, dilation, and EMA numerical tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping
from vesuvius.ink_detection.training.deep_supervision import (
    DeepSupervisionWrapper,
    build_deep_supervision_targets,
    deep_supervision_weights,
)
import vesuvius.ink_detection.training.dilation as dilation_module
import vesuvius.ink_detection.training.train as train_module
from vesuvius.ink_detection.training.dilation import (
    apply_label_dilation,
    dilate_label_batch_with_cucim,
    resolve_dilation_distances,
)
from vesuvius.ink_detection.training.optimizers import (
    create_training_optimizer,
    plan_optimizer_target,
)
from vesuvius.ink_detection.training.stitching import run_model_forward
from vesuvius.ink_detection.training.train import (
    create_training_scheduler,
    initialize_training_model,
    stage_training_request,
)

from .test_model_foundation import _config_mapping


def _training_mapping() -> dict:
    authored = _config_mapping(depth=1, side=4)
    authored.update(
        {
            "num_iterations": 6,
            "out_dir": "/tmp/ink-output",
            "seed": 9,
            "scheduler": {"name": "cosine_annealing"},
        }
    )
    return authored


def _training_config(authored: dict) -> TrainingConfig:
    return TrainingConfig.from_mapping(resolve_training_mapping(authored))


class _EncoderDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_encoder = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)


def test_optimizer_group_freeze_and_multiplier_order_are_literal():
    authored = _training_mapping()
    authored["model_config"]["pretrained_backbone"] = "/weights.pth"
    authored["encoder_lr_mult"] = 0.25
    authored["learning_rate"] = 0.01
    config = _training_config(authored)
    model = _EncoderDecoder()

    target = plan_optimizer_target(model, config)

    assert isinstance(target, list)
    groups = target
    assert [id(value) for value in groups[0]["params"]] == [
        id(model.decoder.weight)
    ]
    assert [id(value) for value in groups[1]["params"]] == [
        id(model.shared_encoder.weight)
    ]
    assert groups[1]["lr"] == 0.0025
    optimizer = create_training_optimizer(model, config)
    assert [id(value) for value in optimizer.param_groups[0]["params"]] == [
        id(model.decoder.weight)
    ]
    assert [id(value) for value in optimizer.param_groups[1]["params"]] == [
        id(model.shared_encoder.weight)
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [0.01, 0.0025]

    frozen_mapping = _training_mapping()
    frozen_mapping["model_config"]["pretrained_backbone"] = "/weights.pth"
    frozen_mapping["freeze_encoder"] = True
    frozen_model = _EncoderDecoder()
    frozen = plan_optimizer_target(
        frozen_model, _training_config(frozen_mapping)
    )
    assert [id(value) for value in frozen[0]["params"]] == [
        id(frozen_model.decoder.weight)
    ]
    assert frozen_model.shared_encoder.weight.requires_grad is False


def test_default_sgd_one_step_matches_hand_calculated_nesterov_update():
    config = _training_config(_training_mapping())
    model = _EncoderDecoder()
    with torch.no_grad():
        model.shared_encoder.weight.fill_(1.0)
    optimizer = create_training_optimizer(model, config)
    model.shared_encoder.weight.grad = torch.tensor([[2.0]])

    optimizer.step()

    expected = 1.0 - 0.01 * (2.00003 + 0.99 * 2.00003)
    assert optimizer.defaults["momentum"] == 0.99
    assert optimizer.defaults["nesterov"] is True
    assert optimizer.defaults["weight_decay"] == 3e-5
    torch.testing.assert_close(
        model.shared_encoder.weight, torch.tensor([[expected]])
    )


def test_cosine_and_one_cycle_use_configured_constructor_values():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    cosine_mapping = _training_mapping()
    cosine_mapping["scheduler"] = {
        "name": "cosine_annealing",
        "t_max": 4,
        "eta_min": 0.001,
    }
    cosine = create_training_scheduler(
        optimizer, _training_config(cosine_mapping)
    )
    assert cosine.T_max == 4
    assert cosine.eta_min == 0.001

    one_cycle_mapping = _training_mapping()
    one_cycle_mapping["scheduler"] = {
        "name": "one_cycle",
        "max_lr": 0.1,
        "total_steps": 6,
        "pct_start": 0.3,
        "final_div_factor": 1e4,
    }
    one_cycle_optimizer = torch.optim.SGD([nn.Parameter(torch.tensor(1.0))], lr=0.01)
    one_cycle = create_training_scheduler(
        one_cycle_optimizer,
        _training_config(one_cycle_mapping),
    )
    assert one_cycle.total_steps == 6
    assert one_cycle._schedule_phases[0]["end_step"] == pytest.approx(0.8)


def test_fresh_initialization_overwrites_zero_attention_and_pretrained_skips():
    fresh_config = _training_config(_training_mapping())
    fresh = nn.Conv3d(1, 1, kernel_size=1)
    nn.init.zeros_(fresh.weight)

    initialize_training_model(fresh, fresh_config)
    assert torch.count_nonzero(fresh.weight).item() == 1

    pretrained_mapping = _training_mapping()
    pretrained_mapping["model_config"]["pretrained_backbone"] = "/weights.pth"
    pretrained = nn.Conv3d(1, 1, kernel_size=1)
    nn.init.zeros_(pretrained.weight)
    initialize_training_model(pretrained, _training_config(pretrained_mapping))
    assert torch.count_nonzero(pretrained.weight).item() == 0


class _ScalarLoss(nn.Module):
    def forward(self, prediction, target):
        del target
        return prediction.reshape(())


def test_deep_supervision_weights_loss_and_single_stage_zero_edge():
    weights = deep_supervision_weights(3)
    wrapper = DeepSupervisionWrapper(_ScalarLoss(), weights)

    result = wrapper(
        [torch.tensor(2.0), torch.tensor(4.0), torch.tensor(100.0)],
        [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)],
    )
    single = DeepSupervisionWrapper(
        _ScalarLoss(), deep_supervision_weights(1)
    )([torch.tensor(99.0, requires_grad=True)], [torch.tensor(0.0)])

    assert weights == pytest.approx([2 / 3, 1 / 3, 0])
    assert result.item() == pytest.approx(8 / 3)
    assert single.item() == 0.0
    assert single.requires_grad is False


def test_deep_supervision_checkerboard_uses_nearest_per_level():
    checkerboard = torch.tensor(
        [[[[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]]]
    )
    outputs = (
        torch.empty(1, 1, 4, 4),
        torch.empty(1, 1, 2, 2),
    )

    pyramid = build_deep_supervision_targets(checkerboard, outputs)

    assert isinstance(pyramid, tuple)
    torch.testing.assert_close(pyramid[0], checkerboard)
    torch.testing.assert_close(
        pyramid[1], torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    )


class _TilePyramid(nn.Module):
    def forward(self, image_BCZYX):
        tile_B1YX = image_BCZYX[:, :1, 0]
        local = tile_B1YX[..., :1, :1]
        full = tile_B1YX + local
        return {"ink": (full, full[..., ::2, ::2])}


def test_stitched_tuple_is_mosaicked_exactly_at_each_output_scale():
    image = torch.tensor(
        [[[[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]]]]]
    )

    result = run_model_forward(
        _TilePyramid(),
        image,
        (1, 2, 2),
        stitched=True,
        use_gradient_checkpointing=False,
    )

    assert isinstance(result, tuple)
    torch.testing.assert_close(result[0], image[:, :, 0] * 2)
    torch.testing.assert_close(
        result[1], torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    )
    with pytest.raises(ValueError, match="depth"):
        run_model_forward(_TilePyramid(), image, (2, 2, 2))
    with pytest.raises(ValueError, match="divide exactly"):
        run_model_forward(_TilePyramid(), image[..., :3, :], (1, 2, 2))


def _fake_dilator(labels, valid, distance):
    del valid, distance
    return F.max_pool3d(labels, kernel_size=3, stride=1, padding=1)


def test_dilation_union_excludes_new_ink_from_background(monkeypatch):
    labels = torch.zeros(1, 1, 1, 1, 5)
    labels[..., 2] = 1
    supervision = torch.zeros_like(labels)
    supervision[..., 0] = 1

    monkeypatch.setattr(
        dilation_module, "dilate_label_batch_with_cucim", _fake_dilator
    )
    output = apply_label_dilation(
        {"inklabels": labels, "supervision_mask": supervision},
        1.0,
        1.0,
    )

    torch.testing.assert_close(
        output["inklabels"],
        torch.tensor([[[[[0.0, 1.0, 1.0, 1.0, 0.0]]]]]),
    )
    torch.testing.assert_close(
        output["supervision_mask"],
        torch.tensor([[[[[1.0, 1.0, 1.0, 1.0, 0.0]]]]]),
    )


def test_zero_dilation_is_import_free_and_positive_cpu_fails_factually():
    labels = torch.zeros(1, 1, 2, 2, 2)
    valid = torch.ones_like(labels)

    assert dilate_label_batch_with_cucim(labels, valid, 0) is labels
    with pytest.raises(RuntimeError, match="CUDA.*CuPy.*cuCIM"):
        dilate_label_batch_with_cucim(labels, valid, 1)


def test_dilation_level_scaling_and_mixed_positive_levels():
    authored = _training_mapping()
    authored["mode"] = "full_3d"
    authored["full_3d"] = {
        "label_dilation_distance": 8,
        "supervision_dilation_distance": 4,
    }
    authored["datasets"][0]["volume_scale"] = 2
    config = _training_config(authored)
    assert resolve_dilation_distances(config) == (2.0, 1.0)

    authored["datasets"].append(
        {"segments_path": "/tmp/other", "volume_scale": 1}
    )
    with pytest.raises(ValueError, match="single volume_scale"):
        resolve_dilation_distances(
            _training_config(authored)
        )


def test_flat_mode_never_enters_positive_full_3d_dilation_configuration():
    authored = _training_mapping()
    authored["full_3d"] = {
        "label_dilation_distance": 8,
        "supervision_dilation_distance": 4,
    }

    assert resolve_dilation_distances(
        _training_config(authored)
    ) == (0.0, 0.0)


class _SyntheticTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, config, *, do_augmentations, patches=None, segments=None):
        del config, do_augmentations
        self.training_patches = [0, 1]
        self.validation_patches = []
        self.segments = []
        self._patches = self.training_patches if patches is None else list(patches)
        if segments is not None:
            self.segments = list(segments)

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        return {
            "image": torch.ones(1, 1, 4, 4),
            "inklabels": torch.zeros(1, 1, 4, 4),
            "supervision_mask": torch.ones(1, 1, 4, 4),
        }


class _SyntheticTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        self.counter.add_(1)
        return {"ink": image[:, :1, 0] * self.value}


class _SyntheticTrainingLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((prediction - target[:, :1]) ** 2)


class _RecordingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}


def test_real_training_entry_preserves_prepare_sync_and_ema_numerics(
    tmp_path: Path, monkeypatch
):
    import accelerate
    import vesuvius.ink_detection.data.dataset as dataset_module
    import vesuvius.ink_detection.models.model as model_module
    import vesuvius.ink_detection.training.losses as losses_module
    import vesuvius.ink_detection.training.optimizers as optimizers_module
    import vesuvius.ink_detection.training.samplers as samplers_module

    authored = _training_mapping()
    authored.update(
        {
            "num_iterations": 2,
            "out_dir": str(tmp_path / "output"),
            "grad_acc_steps": 2,
            "mixed_precision": "no",
            "dataloader_workers": 0,
            "pin_memory": False,
            "val_every": 99,
            "save_every": 2,
            "log_every": 99,
            "ema": {
                "enabled": True,
                "decay": 0.5,
                "start_step": 1,
                "update_every_steps": 1,
                "validate": False,
                "save_in_checkpoint": True,
            },
        }
    )
    authored["model_config"]["pretrained_backbone"] = "synthetic"
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(authored), encoding="utf-8")

    model = _SyntheticTrainingModel()
    scheduler = _RecordingScheduler()
    prepared_objects: list[tuple[object, ...]] = []
    original_prepare = accelerate.Accelerator.prepare

    def recording_prepare(accelerator, *objects):
        prepared_objects.append(objects)
        return original_prepare(accelerator, *objects)

    monkeypatch.setattr(accelerate.Accelerator, "prepare", recording_prepare)
    monkeypatch.setattr(dataset_module, "InkDataset", _SyntheticTrainingDataset)
    monkeypatch.setattr(model_module, "make_model", lambda config: model)
    monkeypatch.setattr(
        losses_module, "create_loss", lambda config: _SyntheticTrainingLoss()
    )
    monkeypatch.setattr(
        optimizers_module,
        "create_training_optimizer",
        lambda model, config: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    monkeypatch.setattr(
        samplers_module,
        "build_sampling_policy",
        lambda patches, config, batch_size: SimpleNamespace(
            batch_sampler=None,
            shuffle=False,
            sampler=None,
            generator=None,
            audit={},
        ),
    )
    monkeypatch.setattr(
        train_module, "create_training_scheduler", lambda optimizer, config: scheduler
    )

    assert train_module._run_training(stage_training_request(config_path)) == 0

    assert len(prepared_objects) == 1
    assert len(prepared_objects[0]) == 4
    assert scheduler not in prepared_objects[0]
    assert scheduler.steps == 1
    checkpoint = torch.load(
        tmp_path / "output" / "ckpt_000002.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["ema_optimizer_step"] == 1
    torch.testing.assert_close(checkpoint["model"]["value"], torch.tensor(0.8))
    torch.testing.assert_close(checkpoint["ema_model"]["value"], torch.tensor(0.9))
    assert checkpoint["model"]["counter"].item() == 2
    assert checkpoint["ema_model"]["counter"].item() == 2
