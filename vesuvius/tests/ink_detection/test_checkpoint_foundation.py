"""Checkpoint payload, DDP-key, restoration, and EMA-selection witnesses."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from vesuvius.ink_detection.models.checkpoint import (
    config_from_checkpoint,
    load_checkpoint,
    load_model_state,
    resolve_checkpoint_path,
    restore_training_state,
    select_inference_weights,
)

from .test_model_foundation import _config_mapping


def _training_payload(model_state):
    return {
        "model": model_state,
        "optimizer": {},
        "lr_scheduler": {},
        "step": 0,
    }


class _WrappedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = nn.Linear(2, 1)


def test_ddp_prefix_is_stripped_for_an_unwrapped_destination():
    source = _WrappedLinear()
    with torch.no_grad():
        source.module.weight.fill_(3.0)
        source.module.bias.fill_(-2.0)
    destination = nn.Linear(2, 1)

    load_model_state(destination, source.state_dict())

    torch.testing.assert_close(destination.weight, torch.full_like(destination.weight, 3.0))
    torch.testing.assert_close(destination.bias, torch.full_like(destination.bias, -2.0))


def test_ddp_prefix_is_added_for_a_wrapped_destination():
    source = nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(5.0)
        source.bias.fill_(7.0)
    destination = _WrappedLinear()

    load_model_state(destination, source.state_dict())

    torch.testing.assert_close(
        destination.module.weight,
        torch.full_like(destination.module.weight, 5.0),
    )
    torch.testing.assert_close(
        destination.module.bias,
        torch.full_like(destination.module.bias, 7.0),
    )


@pytest.mark.parametrize(
    "model_state",
    [
        {"module.weight": torch.zeros((1, 2))},
        {
            "module.weight": torch.zeros((1, 2)),
            "module.bias": torch.zeros(1),
            "module.unexpected": torch.zeros(1),
        },
        {
            "module.weight": torch.zeros((1, 3)),
            "module.bias": torch.zeros(1),
        },
    ],
    ids=["missing", "unexpected", "wrong-shaped"],
)
def test_ddp_compatibility_never_relaxes_strict_state_loading(model_state):
    with pytest.raises(RuntimeError):
        load_model_state(nn.Linear(2, 1), model_state)


def test_weights_only_requires_only_model_at_restore_use_point():
    source = nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(11.0)
        source.bias.fill_(13.0)
    destination = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(destination.parameters(), lr=0.7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    next_step, ema_step = restore_training_state(
        destination,
        optimizer,
        scheduler,
        {"model": source.state_dict()},
        "weights.pth",
        load_weights_only=True,
    )

    assert (next_step, ema_step) == (0, 0)
    assert optimizer.param_groups[0]["lr"] == 0.7
    assert scheduler.last_epoch == 0
    torch.testing.assert_close(destination.weight, source.weight)
    torch.testing.assert_close(destination.bias, source.bias)

    with pytest.raises(ValueError, match="missing 'model'"):
        restore_training_state(
            destination,
            optimizer,
            scheduler,
            {},
            "weights.pth",
            load_weights_only=True,
        )


def test_malformed_model_state_fails_naturally_in_load_state_dict():
    destination = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(destination.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    with pytest.raises(RuntimeError, match="Missing key"):
        restore_training_state(
            destination,
            optimizer,
            scheduler,
            {"model": {}},
            "weights.pth",
            load_weights_only=True,
        )


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("optimizer", "optimizer state"),
        ("lr_scheduler", "lr_scheduler state"),
        ("step", "missing step"),
    ],
)
def test_full_restore_requires_training_state_at_its_use_point(missing, message):
    source = nn.Linear(2, 1)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.2)
    source_scheduler = torch.optim.lr_scheduler.StepLR(
        source_optimizer,
        step_size=1,
    )
    payload = _training_payload(source.state_dict())
    payload["optimizer"] = source_optimizer.state_dict()
    payload["lr_scheduler"] = source_scheduler.state_dict()
    payload.pop(missing)
    destination = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(destination.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    with pytest.raises(KeyError, match=message):
        restore_training_state(
            destination,
            optimizer,
            scheduler,
            payload,
            "run.pth",
        )


def test_full_restore_without_config_or_wandb_loads_all_training_state():
    torch.manual_seed(17)
    source = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.zero_grad()
    source(torch.tensor([[2.0, -1.0]])).sum().backward()
    optimizer.step()
    scheduler.step()
    ema_source = deepcopy(source)
    with torch.no_grad():
        for parameter in ema_source.parameters():
            parameter.add_(4.0)

    payload = _training_payload(source.state_dict())
    payload["optimizer"] = optimizer.state_dict()
    payload["lr_scheduler"] = scheduler.state_dict()
    payload["step"] = 7
    payload["ema_model"] = ema_source.state_dict()
    payload["ema_optimizer_step"] = 19
    assert "config" not in payload
    assert "wandb_run_id" not in payload

    destination = nn.Linear(2, 1)
    destination_optimizer = torch.optim.SGD(
        destination.parameters(), lr=0.01, momentum=0.9
    )
    destination_scheduler = torch.optim.lr_scheduler.StepLR(
        destination_optimizer,
        step_size=3,
        gamma=0.1,
    )
    ema_destination = nn.Linear(2, 1)

    next_step, ema_step = restore_training_state(
        destination,
        destination_optimizer,
        destination_scheduler,
        payload,
        "run.pth",
        ema_model=ema_destination,
    )

    assert (next_step, ema_step) == (8, 19)
    assert destination_optimizer.param_groups[0]["lr"] == 0.1
    assert destination_scheduler.last_epoch == 1
    source_momentum = next(iter(optimizer.state.values()))["momentum_buffer"]
    destination_momentum = next(iter(destination_optimizer.state.values()))[
        "momentum_buffer"
    ]
    torch.testing.assert_close(destination_momentum, source_momentum)
    for actual, expected in zip(destination.parameters(), source.parameters()):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(ema_destination.parameters(), ema_source.parameters()):
        torch.testing.assert_close(actual, expected)


def test_full_restore_defaults_missing_ema_optimizer_step_to_zero():
    source = nn.Linear(2, 1)
    ema_source = deepcopy(source)
    with torch.no_grad():
        ema_source.weight.fill_(23.0)
        ema_source.bias.fill_(-29.0)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    payload = _training_payload(source.state_dict())
    payload["optimizer"] = optimizer.state_dict()
    payload["lr_scheduler"] = scheduler.state_dict()
    payload["step"] = 7
    payload["ema_model"] = ema_source.state_dict()

    destination = nn.Linear(2, 1)
    destination_optimizer = torch.optim.SGD(destination.parameters(), lr=0.1)
    destination_scheduler = torch.optim.lr_scheduler.StepLR(
        destination_optimizer,
        step_size=3,
    )
    ema_destination = nn.Linear(2, 1)

    next_step, ema_step = restore_training_state(
        destination,
        destination_optimizer,
        destination_scheduler,
        payload,
        "run.pth",
        ema_model=ema_destination,
    )

    assert (next_step, ema_step) == (8, 0)
    torch.testing.assert_close(ema_destination.weight, ema_source.weight)
    torch.testing.assert_close(ema_destination.bias, ema_source.bias)


def test_full_restore_skips_none_ema_model_state():
    source = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    payload = _training_payload(source.state_dict())
    payload["optimizer"] = optimizer.state_dict()
    payload["lr_scheduler"] = scheduler.state_dict()
    payload["step"] = 4
    payload["ema_model"] = None
    ema_destination = nn.Linear(2, 1)
    ema_before = deepcopy(ema_destination.state_dict())
    destination = nn.Linear(2, 1)
    destination_optimizer = torch.optim.SGD(destination.parameters(), lr=0.1)
    destination_scheduler = torch.optim.lr_scheduler.StepLR(
        destination_optimizer,
        step_size=1,
    )

    next_step, ema_step = restore_training_state(
        destination,
        destination_optimizer,
        destination_scheduler,
        payload,
        "run.pth",
        ema_model=ema_destination,
    )

    assert (next_step, ema_step) == (5, 0)
    for name, tensor in ema_destination.state_dict().items():
        torch.testing.assert_close(tensor, ema_before[name])


def test_relative_checkpoint_uses_lexical_symlinked_config_directory(tmp_path):
    real_directory = tmp_path / "real"
    alias_directory = tmp_path / "alias"
    real_directory.mkdir()
    alias_directory.mkdir()
    real_config = real_directory / "config.json"
    real_config.write_text("{}")
    linked_config = alias_directory / "config.json"
    linked_config.symlink_to(real_config)

    resolved = resolve_checkpoint_path("checkpoint.pth", linked_config)

    assert resolved == alias_directory / "checkpoint.pth"
    assert resolved != real_directory / "checkpoint.pth"


def test_checkpoint_file_loading_does_not_interpret_payload(tmp_path):
    state = nn.Linear(2, 1).state_dict()
    payload = {
        "state_dict": state,
        "config": _config_mapping(),
        "ema_model": None,
    }
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(payload, checkpoint_path)

    loaded = load_checkpoint(checkpoint_path)
    config = config_from_checkpoint(loaded, source=checkpoint_path)

    assert config.to_mapping() == payload["config"]
    assert config.model.model_type == "vesuvius_unet"
    assert loaded["state_dict"].keys() == state.keys()


def test_config_is_required_only_when_reconstructing_config():
    with pytest.raises(KeyError, match="config"):
        config_from_checkpoint({"model": nn.Linear(2, 1).state_dict()})


def test_inference_ema_preference_and_fallback_are_explicit():
    model_state = {"weight": torch.tensor([1.0])}
    ema_state = {"weight": torch.tensor([2.0])}
    payload = {"model": model_state, "ema_model": ema_state}

    selected_name, selected_state = select_inference_weights(payload)
    assert selected_name == "ema_model"
    assert selected_state is ema_state

    selected_name, selected_state = select_inference_weights(
        {"model": model_state}
    )
    assert selected_name == "model"
    assert selected_state is model_state
