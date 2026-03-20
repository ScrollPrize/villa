"""Run filesystem helpers for storing experiment artifacts and checkpoints."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from ink.core.types import EvalReport


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return to_plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def _dump_yaml_text(payload: Any) -> str:
    try:
        import yaml  # type: ignore
    except ImportError:
        return json.dumps(payload, indent=2, sort_keys=False) + "\n"
    return yaml.safe_dump(payload, sort_keys=False)


def _write_yaml(path: Path, payload: Any) -> None:
    path.write_text(_dump_yaml_text(to_plain(payload)), encoding="utf-8")


def _checkpoint_payload(
    *,
    model_state: Any,
    optimizer_state: Any = None,
    epoch: int | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {"model": model_state}
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if extra_state:
        payload.update(dict(extra_state))
    return payload


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        with path.open("wb") as handle:
            pickle.dump(dict(payload), handle)
        return
    torch.save(dict(payload), path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import torch  # type: ignore
    except ImportError:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"checkpoint at {str(path)!r} must be a mapping, got {type(payload).__name__}"
        )
    return dict(payload)


class RunFS:
    def __init__(self, run_dir: str | Path, experiment: Any):
        self.run_dir = Path(run_dir)
        self.eval_dir = self.run_dir / "eval"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.history_path = self.run_dir / "history.jsonl"
        self.summary_path = self.run_dir / "summary.yaml"
        self.experiment_path = self.run_dir / "experiment.yaml"
        self.best_metric: float | None = None

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        _write_yaml(self.experiment_path, experiment)

    def local_paths(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "checkpoints_dir": str(self.ckpt_dir),
            "eval_dir": str(self.eval_dir),
            "history_path": str(self.history_path),
            "summary_path": str(self.summary_path),
            "artifacts_dir": str(self.artifacts_dir),
        }

    def append_history(self, event: Mapping[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_plain(dict(event)), sort_keys=False) + "\n")

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        _write_yaml(self.summary_path, dict(summary))

    def log_train_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        self.append_history(
            {
                "epoch": int(epoch),
                "split": "train",
                "metrics": dict(metrics),
            }
        )

    def log_eval_epoch(self, epoch: int, report: EvalReport) -> None:
        report_summary = dict(report.summary)
        payload = {
            "epoch": int(epoch),
            "split": "val",
            "summary": report_summary,
        }
        self.append_history(payload)
        self.write_summary(payload)
        _write_yaml(self.eval_dir / "latest.yaml", report_summary)

    def save_last(
        self,
        *,
        model_state: Any,
        optimizer_state: Any = None,
        epoch: int | None = None,
        extra_state: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.ckpt_dir / "last.pt"
        _save_checkpoint(
            path,
            _checkpoint_payload(
                model_state=model_state,
                optimizer_state=optimizer_state,
                epoch=epoch,
                extra_state=extra_state,
            ),
        )
        return path

    def maybe_save_best(
        self,
        key: str,
        report: EvalReport,
        *,
        model_state: Any,
        optimizer_state: Any = None,
        epoch: int | None = None,
        extra_state: Mapping[str, Any] | None = None,
        higher_is_better: bool = True,
    ) -> Path | None:
        report_summary = dict(report.summary)
        metric = float(report_summary[key])
        should_replace = self.best_metric is None
        if self.best_metric is not None:
            if higher_is_better:
                should_replace = metric > self.best_metric
            else:
                should_replace = metric < self.best_metric
        if not should_replace:
            return None

        self.best_metric = metric
        _write_yaml(self.eval_dir / "best.yaml", report_summary)
        path = self.ckpt_dir / "best.pt"
        _save_checkpoint(
            path,
            _checkpoint_payload(
                model_state=model_state,
                optimizer_state=optimizer_state,
                epoch=epoch,
                extra_state=extra_state,
            ),
        )
        return path
