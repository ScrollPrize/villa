from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from ink.core.run_fs import load_checkpoint


def state_dict(owner, *, name: str):
    """Call state_dict() with a clearer error when the object is not checkpointable."""
    state_dict_fn = getattr(owner, "state_dict", None)
    if not callable(state_dict_fn):
        raise TypeError(f"{name} must define state_dict()")
    return state_dict_fn()


def resolve_checkpoint_extra_state(extra_state, *, epoch: int) -> dict[str, Any] | None:
    """Resolve static or callable extra checkpoint payloads for the current epoch."""
    if extra_state is None:
        return None
    resolved = extra_state(int(epoch)) if callable(extra_state) else extra_state
    if resolved is None:
        return None
    return dict(resolved)


def resolve_checkpoint_path(path: str | None) -> Path | None:
    """Expand a checkpoint path and resolve relative paths from the current working directory."""
    if path is None:
        return None
    resolved = Path(str(path)).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _strip_module_prefix(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload and all(str(key).startswith("module.") for key in payload.keys()):
        return {str(key)[len("module."):]: value for key, value in payload.items()}
    return dict(payload)


def _extract_model_state(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Accept a few common checkpoint layouts and normalize DDP-prefixed keys."""
    for key in ("model", "state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return _strip_module_prefix(dict(candidate))
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return _strip_module_prefix(dict(payload))
    raise ValueError(f"Unsupported checkpoint format for {source}")


def _extract_optimizer_state(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("optimizer", "optimizer_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return None


def _extract_epoch(payload: Mapping[str, Any], *, source: str) -> int | None:
    epoch = payload.get("epoch")
    if epoch is None:
        return None
    if isinstance(epoch, torch.Tensor):
        if int(epoch.numel()) != 1:
            raise ValueError(f"checkpoint epoch tensor must be scalar for {source}")
        epoch = epoch.detach().cpu().item()
    try:
        return int(epoch)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"checkpoint epoch must be an integer for {source}") from exc


def _load_state_dict(owner, payload: Mapping[str, Any], *, name: str, source: str) -> None:
    load_state_dict = getattr(owner, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TypeError(f"{name} must define load_state_dict(...) to restore checkpoint {source}")
    load_state_dict(dict(payload))


def apply_init_checkpoint(model, *, ckpt_path: Path) -> None:
    """Load model weights only from an initialization checkpoint."""
    source = f"init_ckpt_path={str(ckpt_path)!r}"
    payload = load_checkpoint(ckpt_path)
    _load_state_dict(
        model,
        _extract_model_state(payload, source=source),
        name="experiment.model",
        source=source,
    )


def apply_resume_checkpoint(model, optimizer, *, ckpt_path: Path) -> int:
    """Restore model and optimizer state and return the next epoch index."""
    source = f"resume_ckpt_path={str(ckpt_path)!r}"
    payload = load_checkpoint(ckpt_path)
    optimizer_state = _extract_optimizer_state(payload)
    if optimizer_state is None:
        raise ValueError(f"checkpoint missing optimizer state for {source}")

    _load_state_dict(
        model,
        _extract_model_state(payload, source=source),
        name="experiment.model",
        source=source,
    )
    _load_state_dict(
        optimizer,
        optimizer_state,
        name="optimizer",
        source=source,
    )

    epoch = _extract_epoch(payload, source=source)
    if epoch is None:
        return 0
    return int(epoch) + 1


def save_last_checkpoint(
    run_fs,
    *,
    model,
    optimizer,
    epoch: int,
    checkpoint_extra_state: Mapping[str, Any] | Callable[[int], Mapping[str, Any] | None] | None = None,
) -> None:
    run_fs.save_last(
        model_state=state_dict(model, name="experiment.model"),
        optimizer_state=state_dict(optimizer, name="optimizer"),
        epoch=epoch,
        extra_state=resolve_checkpoint_extra_state(checkpoint_extra_state, epoch=epoch),
    )


def save_epoch_checkpoint(
    run_fs,
    *,
    model,
    optimizer,
    epoch: int,
    checkpoint_extra_state: Mapping[str, Any] | Callable[[int], Mapping[str, Any] | None] | None = None,
) -> None:
    run_fs.save_epoch(
        model_state=state_dict(model, name="experiment.model"),
        optimizer_state=state_dict(optimizer, name="optimizer"),
        epoch=epoch,
        extra_state=resolve_checkpoint_extra_state(checkpoint_extra_state, epoch=epoch),
    )


def maybe_save_best_checkpoint(
    run_fs,
    key: str,
    report,
    *,
    model,
    optimizer,
    epoch: int,
    checkpoint_extra_state: Mapping[str, Any] | Callable[[int], Mapping[str, Any] | None] | None = None,
    higher_is_better: bool = True,
) -> None:
    run_fs.maybe_save_best(
        key,
        report,
        model_state=state_dict(model, name="experiment.model"),
        optimizer_state=state_dict(optimizer, name="optimizer"),
        epoch=epoch,
        extra_state=resolve_checkpoint_extra_state(checkpoint_extra_state, epoch=epoch),
        higher_is_better=bool(higher_is_better),
    )
