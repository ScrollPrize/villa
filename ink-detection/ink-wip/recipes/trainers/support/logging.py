from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ink.core.run_fs import to_plain
from ink.core.types import EvalReport


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
    media_downsample: int = 1
    log_eval_by_group: bool = False
    log_eval_by_segment: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "project", None if self.project is None else str(self.project))
        object.__setattr__(self, "entity", None if self.entity is None else str(self.entity))
        object.__setattr__(self, "group", None if self.group is None else str(self.group))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in tuple(self.tags)))
        object.__setattr__(self, "run_name", None if self.run_name is None else str(self.run_name))
        object.__setattr__(self, "mode", None if self.mode is None else str(self.mode))
        object.__setattr__(self, "dir", None if self.dir is None else str(self.dir))
        media_downsample = int(self.media_downsample)
        if media_downsample < 1:
            raise ValueError(f"wandb media_downsample must be >= 1, got {self.media_downsample!r}")
        object.__setattr__(self, "media_downsample", media_downsample)
        object.__setattr__(self, "log_eval_by_group", bool(self.log_eval_by_group))
        object.__setattr__(self, "log_eval_by_segment", bool(self.log_eval_by_segment))


def _to_wandb_config_value(value: Any) -> Any:
    """Normalize experiment config values into W&B-friendly scalars and containers."""
    value = to_plain(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_wandb_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wandb_config_value(item) for item in value]
    return repr(value)


def _to_logged_scalar(value: Any) -> float | None:
    """Best-effort conversion of one logged scalar value."""
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


def _wandb_scalar_payload(values: Mapping[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in dict(values).items():
        scalar = _to_logged_scalar(value)
        if scalar is None:
            continue
        payload[str(key)] = scalar
    return payload


def _prefixed_scalar_payload(values: Mapping[str, Any], *, prefix: str) -> dict[str, float]:
    payload = _wandb_scalar_payload(values)
    return {f"{str(prefix)}{str(key)}": float(value) for key, value in payload.items()}


def _eval_report_scalar_payload(
    report: EvalReport,
    *,
    include_by_group: bool,
    include_by_segment: bool,
) -> dict[str, float]:
    payload = _wandb_scalar_payload(report.summary)
    if include_by_group:
        for group, metrics in dict(report.by_group).items():
            payload.update(
                _prefixed_scalar_payload(
                    metrics,
                    prefix=f"group/{str(group)}/",
                )
            )
    if include_by_segment:
        for segment, metrics in dict(report.by_segment).items():
            payload.update(
                _prefixed_scalar_payload(
                    metrics,
                    prefix=f"segment/{str(segment)}/",
                )
            )
    return payload


@dataclass
class WandbSession:
    run: Any
    image_factory: Any = None
    media_downsample: int = 1
    log_eval_by_group: bool = False
    log_eval_by_segment: bool = False

    def log_train_epoch(self, epoch: int, components: Mapping[str, Any]) -> None:
        payload = {"trainer/epoch": int(epoch), **_wandb_scalar_payload(components)}
        self.run.log(payload, step=int(epoch))

    def log_eval_epoch(self, epoch: int, report: EvalReport) -> None:
        if not isinstance(report, EvalReport):
            raise TypeError("WandbSession.log_eval_epoch requires EvalReport")
        scalars = _eval_report_scalar_payload(
            report,
            include_by_group=bool(self.log_eval_by_group),
            include_by_segment=bool(self.log_eval_by_segment),
        )
        payload = {"trainer/epoch": int(epoch), **scalars}
        self.run.log(payload, step=int(epoch))

    def log_images(self, epoch: int, images: Mapping[str, Mapping[str, Any]]) -> None:
        if not images:
            return
        payload: dict[str, Any] = {"trainer/epoch": int(epoch)}
        for key, item in dict(images).items():
            image = item.get("image")
            if image is None:
                continue
            caption = item.get("caption")
            if callable(self.image_factory):
                payload[str(key)] = self.image_factory(image, caption=caption)
            else:
                payload[str(key)] = image
        if len(payload) > 1:
            self.run.log(payload, step=int(epoch))

    def finish(self) -> None:
        finish = getattr(self.run, "finish", None)
        if callable(finish):
            finish()


def init_wandb_session(experiment, *, run_fs=None) -> WandbSession | None:
    """Create a W&B run only when logging is explicitly enabled and configured."""
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

    return WandbSession(
        run=run,
        image_factory=getattr(wandb, "Image", None),
        media_downsample=int(config.media_downsample),
        log_eval_by_group=bool(config.log_eval_by_group),
        log_eval_by_segment=bool(config.log_eval_by_segment),
    )
