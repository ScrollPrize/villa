from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ink.core.run_fs import to_plain


@dataclass(frozen=True)
class WandbLogger:
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    run_name: str | None = None
    mode: str | None = None
    dir: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "project", None if self.project is None else str(self.project))
        object.__setattr__(self, "entity", None if self.entity is None else str(self.entity))
        object.__setattr__(self, "group", None if self.group is None else str(self.group))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in tuple(self.tags)))
        object.__setattr__(self, "run_name", None if self.run_name is None else str(self.run_name))
        object.__setattr__(self, "mode", None if self.mode is None else str(self.mode))
        object.__setattr__(self, "dir", None if self.dir is None else str(self.dir))


def _to_wandb_config_value(value: Any) -> Any:
    value = to_plain(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_wandb_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wandb_config_value(item) for item in value]
    return repr(value)


def _to_metric_scalar(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if int(value.numel()) != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _wandb_scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in dict(metrics).items():
        scalar = _to_metric_scalar(value)
        if scalar is None:
            continue
        payload[str(key)] = scalar
    return payload


@dataclass
class WandbSession:
    run: Any

    def log_train_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        payload = {"trainer/epoch": int(epoch), **_wandb_scalar_metrics(metrics)}
        self.run.log(payload, step=int(epoch))

    def log_eval_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        payload = {"trainer/epoch": int(epoch), **_wandb_scalar_metrics(metrics)}
        self.run.log(payload, step=int(epoch))

    def finish(self) -> None:
        finish = getattr(self.run, "finish", None)
        if callable(finish):
            finish()


def init_wandb_session(experiment, *, run_fs=None) -> WandbSession | None:
    runtime = experiment.runtime
    config = getattr(runtime, "wandb", None)
    if config is None:
        return None
    assert isinstance(config, WandbLogger)
    if not config.enabled:
        return None

    project = config.project
    if project is None or not str(project).strip():
        raise ValueError("runtime.wandb.project must be set when runtime.wandb.enabled is true")

    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "W&B logging is enabled but the 'wandb' package is not installed"
        ) from exc

    run_name = config.run_name
    if run_name is None and run_fs is not None:
        run_name = str(run_fs.run_dir.name)
    if run_name is None:
        run_name = str(experiment.name)

    init_kwargs = {
        "project": str(project),
        "name": str(run_name),
        "config": _to_wandb_config_value(experiment),
    }
    if config.entity is not None:
        init_kwargs["entity"] = str(config.entity)
    if config.group is not None:
        init_kwargs["group"] = str(config.group)
    if config.mode is not None:
        init_kwargs["mode"] = str(config.mode)
    if config.dir is not None:
        init_kwargs["dir"] = str(config.dir)
    if config.tags:
        init_kwargs["tags"] = [str(tag) for tag in config.tags]

    run = wandb.init(**init_kwargs)
    if run is None:
        return None

    if run_fs is not None:
        summary = getattr(run, "summary", None)
        if summary is not None:
            for key, value in run_fs.local_paths().items():
                summary[f"local/{str(key)}"] = str(value)

    return WandbSession(run=run)
