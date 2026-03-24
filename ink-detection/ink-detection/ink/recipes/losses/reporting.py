from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class TrainLossOutput:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]


def _to_tensor(value, *, device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _coerce_per_sample(values: torch.Tensor, *, batch_size: int, name: str) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    if values.ndim == 0:
        return values.expand(batch_size).clone()

    if values.ndim > 1 and int(values.shape[0]) == batch_size:
        return values.reshape(batch_size, -1).mean(dim=1)

    values = values.reshape(-1)
    if int(values.numel()) == 1 and batch_size > 1:
        return values.expand(batch_size).clone()
    if int(values.numel()) != batch_size:
        raise ValueError(
            f"{name} must return scalar or {batch_size} per-sample values; got shape {tuple(values.shape)}"
        )
    return values


def _call_with_valid_mask(fn, logits, targets, *, valid_mask=None):
    try:
        return fn(logits, targets, valid_mask=valid_mask)
    except TypeError:
        return fn(logits, targets)


def _call_value_method(loss, method_name: str, logits, targets, *, valid_mask=None) -> torch.Tensor | None:
    method = getattr(loss, str(method_name), None)
    if not callable(method):
        return None
    return _to_tensor(_call_with_valid_mask(method, logits, targets, valid_mask=valid_mask), device=logits.device)


def _normalize_components(raw_components: Mapping[str, Any], *, device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in raw_components.items():
        out[str(key)] = _to_tensor(value, device=device)
    return out


def _parse_train_output(raw_output, *, device) -> TrainLossOutput:
    if isinstance(raw_output, Mapping):
        if "loss" not in raw_output:
            raise TypeError("loss.training_outputs(...) must include a 'loss' key")
        loss_tensor = _to_tensor(raw_output["loss"], device=device)
        components: dict[str, torch.Tensor] = {}

        nested_components = raw_output.get("components")
        if nested_components is not None:
            if not isinstance(nested_components, Mapping):
                raise TypeError("loss.training_outputs(...)[\"components\"] must be a mapping")
            components.update(_normalize_components(nested_components, device=device))

        for key, value in raw_output.items():
            key = str(key)
            if key in {"loss", "components"}:
                continue
            if isinstance(value, Mapping):
                raise TypeError("loss.training_outputs(...) nested mappings must live under 'components'")
            components[key] = _to_tensor(value, device=device)

        return TrainLossOutput(loss=loss_tensor, components=components)

    return TrainLossOutput(loss=_to_tensor(raw_output, device=device), components={})


def resolve_train_output(loss, logits, targets, *, valid_mask=None) -> TrainLossOutput:
    train_output_method = getattr(loss, "training_outputs", None)
    if callable(train_output_method):
        raw_output = _call_with_valid_mask(train_output_method, logits, targets, valid_mask=valid_mask)
        return _parse_train_output(raw_output, device=logits.device)

    if not callable(loss):
        raise TypeError("loss recipe must be callable")

    raw_loss = _call_with_valid_mask(loss, logits, targets, valid_mask=valid_mask)
    return _parse_train_output(raw_loss, device=logits.device)


def loss_values(loss, logits, targets, *, valid_mask=None) -> torch.Tensor:
    values = _call_value_method(loss, "loss_values", logits, targets, valid_mask=valid_mask)
    if values is None:
        values = resolve_train_output(loss, logits, targets, valid_mask=valid_mask).loss
    return _coerce_per_sample(values, batch_size=int(targets.shape[0]), name="loss_values")


def train_components(loss, logits, targets, *, valid_mask=None) -> dict[str, torch.Tensor]:
    output = resolve_train_output(loss, logits, targets, valid_mask=valid_mask)
    components: dict[str, torch.Tensor] = {}
    for key, value in output.components.items():
        tensor = _to_tensor(value, device=logits.device)
        if tensor.ndim > 0:
            tensor = tensor.reshape(-1).mean()
        components[str(key)] = tensor.detach()
    return components
