from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from vesuvius.neural_tracing.fiber_trace.dataset import FiberTraceBatchBuilder
from vesuvius.neural_tracing.fiber_trace.losses import compute_fiber_trace_loss
from vesuvius.neural_tracing.fiber_trace.model import build_fiber_trace_model


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config JSON must be an object, got {type(config).__name__}")
    config.setdefault("_config_dir", str(config_path.parent))
    return config


def _sanitize_run_name(value: Any) -> str:
    name = str(value or "fiber_trace").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._-") or "fiber_trace"


def _reject_legacy_checkpoint_path(config: dict[str, Any]) -> None:
    if "checkpoint_path" in config:
        raise ValueError(
            "checkpoint_path was replaced by run_path/run_name; snapshots are "
            "written to <run_path>/<run_name>_<datestr>/snapshots/current.pt "
            "and best.pt"
        )


def _resolve_run_layout(config: dict[str, Any]) -> tuple[Path, Path]:
    _reject_legacy_checkpoint_path(config)
    run_path = Path(str(config.get("run_path", "runs/fiber_trace")))
    run_name = _sanitize_run_name(config.get("run_name", "fiber_trace"))
    date_str = str(
        config.get("run_datestr") or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = run_path / f"{run_name}_{date_str}"
    snapshot_dir = run_dir / "snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, snapshot_dir


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return f"<array shape={tuple(shape)} dtype={dtype}>"
    return repr(value)


def _config_json_text(config: dict[str, Any]) -> str:
    return json.dumps(_json_safe(config), indent=2, sort_keys=True)


def _make_summary_writer(log_dir: Path, *, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires the tensorboard package. "
            "Install tensorboard or set tensorboard_enabled=false."
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))


def _make_test_config(config: dict[str, Any]) -> dict[str, Any] | None:
    if config.get("_test_array_records"):
        test_config = dict(config)
        test_config["_array_records"] = config["_test_array_records"]
        return test_config

    if config.get("test_datasets"):
        test_config = dict(config)
        test_config["datasets"] = config["test_datasets"]
        test_config.pop("_array_records", None)
        return test_config

    datasets = config.get("datasets")
    if not isinstance(datasets, list):
        return None

    test_datasets: list[dict[str, Any]] = []
    for dataset_raw in datasets:
        dataset = dict(dataset_raw)
        test_paths = dataset.get("test_fiber_paths")
        test_glob = dataset.get("test_fiber_glob")
        if not test_paths and not test_glob:
            continue

        dataset.pop("fiber_paths", None)
        dataset.pop("fiber_glob", None)
        if test_paths:
            dataset["fiber_paths"] = test_paths
        if test_glob:
            dataset["fiber_glob"] = test_glob
        test_datasets.append(dataset)

    if not test_datasets:
        return None

    test_config = dict(config)
    test_config["datasets"] = test_datasets
    test_config.pop("_array_records", None)
    return test_config


def _loss_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = dict(config.get("loss", {}))
    return {
        "temperature": float(loss_cfg.get("temperature", 0.1)),
        "contrastive_weight": float(loss_cfg.get("contrastive_weight", 1.0)),
        "fw_weight": float(loss_cfg.get("fw_weight", 1.0)),
        "up_weight": float(loss_cfg.get("up_weight", 1.0)),
        "max_contrastive_samples": int(loss_cfg.get("max_contrastive_samples", 4096)),
    }


def _compute_losses(
    model: torch.nn.Module,
    batch_builder: FiberTraceBatchBuilder,
    device: torch.device,
    loss_kwargs: dict[str, Any],
):
    batch = batch_builder.sample_batch().to(device)
    outputs = model(batch.volume, batch.cond_fw_xyz, batch.cond_up_xyz)
    return compute_fiber_trace_loss(outputs, batch, **loss_kwargs)


def _loss_scalars(losses) -> dict[str, float]:
    return {
        "total": float(losses.total.detach().cpu()),
        "contrastive": float(losses.contrastive.detach().cpu()),
        "fw": float(losses.fw.detach().cpu()),
        "up": float(losses.up.detach().cpu()),
    }


def _log_scalars(writer: Any, prefix: str, scalars: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for name, value in scalars.items():
        writer.add_scalar(f"{prefix}/{name}", value, step)


def _save_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    steps: int,
    metric_name: str,
    metric_value: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": _json_safe(config),
            "step": int(step),
            "steps": int(steps),
            "metric_name": str(metric_name),
            "metric_value": float(metric_value),
        },
        path,
    )


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if "seed" in config:
        torch.manual_seed(int(config["seed"]))
    _reject_legacy_checkpoint_path(config)

    batch_builder = FiberTraceBatchBuilder(config)
    test_config = _make_test_config(config)
    test_batch_builder = (
        FiberTraceBatchBuilder(test_config) if test_config is not None else None
    )
    run_dir, snapshot_dir = _resolve_run_layout(config)
    current_snapshot = snapshot_dir / "current.pt"
    best_snapshot = snapshot_dir / "best.pt"
    writer = _make_summary_writer(
        run_dir, enabled=bool(config.get("tensorboard_enabled", True))
    )

    model = build_fiber_trace_model(config).to(device)
    opt_cfg = dict(config.get("optimizer", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", opt_cfg.get("lr", 1e-3))),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )

    loss_kwargs = _loss_kwargs(config)
    steps = int(config.get("num_steps", config.get("steps", 1)))
    log_every = max(1, int(config.get("log_every", 100)))
    best_metric = float("inf")
    best_metric_name = (
        "test/total" if test_batch_builder is not None else "train/total"
    )

    try:
        if writer is not None:
            writer.add_text(
                "config/json", f"```json\n{_config_json_text(config)}\n```", 0
            )
            writer.flush()

        model.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = _compute_losses(model, batch_builder, device, loss_kwargs)
            losses.total.backward()
            optimizer.step()

            should_log = step % log_every == 0 or step == steps
            if not should_log:
                continue

            train_scalars = _loss_scalars(losses)
            test_scalars = None
            if test_batch_builder is not None:
                model.eval()
                with torch.no_grad():
                    test_scalars = _loss_scalars(
                        _compute_losses(
                            model, test_batch_builder, device, loss_kwargs
                        )
                    )
                model.train()

            _log_scalars(writer, "train", train_scalars, step)
            if test_scalars is not None:
                _log_scalars(writer, "test", test_scalars, step)
            if writer is not None:
                writer.flush()

            metric_value = (
                test_scalars["total"]
                if test_scalars is not None
                else train_scalars["total"]
            )
            _save_snapshot(
                current_snapshot,
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                steps=steps,
                metric_name=best_metric_name,
                metric_value=metric_value,
            )
            if metric_value < best_metric:
                best_metric = metric_value
                _save_snapshot(
                    best_snapshot,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    steps=steps,
                    metric_name=best_metric_name,
                    metric_value=metric_value,
                )

            message = (
                f"step={step} train_total={train_scalars['total']:.6f} "
                f"train_contrastive={train_scalars['contrastive']:.6f} "
                f"train_fw={train_scalars['fw']:.6f} "
                f"train_up={train_scalars['up']:.6f}"
            )
            if test_scalars is not None:
                message += (
                    f" test_total={test_scalars['total']:.6f} "
                    f"test_contrastive={test_scalars['contrastive']:.6f} "
                    f"test_fw={test_scalars['fw']:.6f} "
                    f"test_up={test_scalars['up']:.6f}"
                )
            print(message, flush=True)
    finally:
        if writer is not None:
            writer.close()

    return {
        "run_dir": str(run_dir),
        "snapshot_dir": str(snapshot_dir),
        "current_snapshot": str(current_snapshot),
        "best_snapshot": str(best_snapshot),
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the fiber tracing MVP model.")
    parser.add_argument("config", help="Path to a fiber trace training config JSON.")
    args = parser.parse_args(argv)
    run_training(_load_config(args.config))


if __name__ == "__main__":
    main()
