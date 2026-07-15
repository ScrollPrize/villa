from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DBatch,
    FiberTrace3DLoader,
    load_config,
)
from vesuvius.neural_tracing.fiber_trace_3d.model import (
    build_fiber_trace_3d_model,
    direction_output,
    presence_output,
)


def _load_raw_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    config.setdefault("_config_dir", str(config_path.parent))
    return config


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sanitize_run_name(value: Any) -> str:
    name = str(value or "fiber_trace_3d").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._-") or "fiber_trace_3d"


def _resolve_run_layout(config: dict[str, Any]) -> tuple[Path, Path]:
    training = dict(config.get("training", {}))
    run_path = Path(str(training.get("run_path", config.get("run_path", "runs/fiber_trace_3d"))))
    run_name = _sanitize_run_name(training.get("run_name", config.get("run_name", "fiber_trace_3d")))
    date_str = str(training.get("run_datestr") or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = run_path / f"{run_name}_{date_str}"
    snapshot_dir = run_dir / "snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, snapshot_dir


def _make_summary_writer(log_dir: Path, *, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires tensorboard; install it or set "
            "training.tensorboard_enabled=false"
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))


def _device_from_training(training: dict[str, Any]) -> torch.device:
    raw = str(training.get("device", "auto"))
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=value.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (value * mask_f).sum() / denom


def compute_losses(
    output: torch.Tensor,
    batch: FiberTrace3DBatch,
    *,
    direction_weight: float,
    presence_weight: float,
) -> dict[str, torch.Tensor]:
    pred_dir = direction_output(output)
    pred_presence = presence_output(output)
    direction_mask = batch.direction_mask.expand_as(pred_dir)
    direction_error = (pred_dir - batch.direction_target) ** 2
    direction_error = direction_error * batch.direction_weight
    direction_loss = _masked_mean(direction_error, direction_mask)

    presence_mask = batch.presence_mask.expand_as(pred_presence)
    presence_bce = F.binary_cross_entropy(
        pred_presence.clamp(1.0e-6, 1.0 - 1.0e-6),
        batch.presence_target,
        reduction="none",
    )
    pos = (batch.presence_target > 0.5) & presence_mask
    neg = (batch.presence_target <= 0.5) & presence_mask
    if bool(pos.any()) and bool(neg.any()):
        presence_loss = 0.5 * _masked_mean(presence_bce, pos) + 0.5 * _masked_mean(
            presence_bce, neg
        )
    else:
        presence_loss = _masked_mean(presence_bce, presence_mask)
    total = float(direction_weight) * direction_loss + float(presence_weight) * presence_loss
    return {
        "total": total,
        "direction": direction_loss,
        "presence": presence_loss,
    }


def _save_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    metric: float | None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "config": _json_safe(config),
            "metric": metric,
        },
        path,
    )


def _load_snapshot(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str = "cpu",
) -> int:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0)) if isinstance(payload, dict) else 0


@torch.no_grad()
def evaluate_dense_loss(
    model: torch.nn.Module,
    loader: FiberTrace3DLoader,
    *,
    device: torch.device,
    start_sample_index: int,
    sample_count: int,
    micro_batch_size: int,
    direction_weight: float,
    presence_weight: float,
) -> dict[str, float]:
    model.eval()
    total_rows: list[dict[str, float]] = []
    consumed = 0
    while consumed < sample_count:
        batch = loader.load_batch(
            start_sample_index + consumed,
            sample_mode="random",
            device=device,
        )
        take = min(int(batch.volume.shape[0]), sample_count - consumed)
        if take < int(batch.volume.shape[0]):
            batch = _slice_batch(batch, 0, take)
        rows = _forward_loss_microbatched(
            model,
            batch,
            micro_batch_size=micro_batch_size,
            direction_weight=direction_weight,
            presence_weight=presence_weight,
            backward=False,
        )
        total_rows.append(rows)
        consumed += take
    model.train()
    if not total_rows:
        return {"total": math.inf, "direction": math.inf, "presence": math.inf}
    return {
        key: float(sum(row[key] for row in total_rows) / len(total_rows))
        for key in total_rows[0]
    }


def _slice_batch(batch: FiberTrace3DBatch, start: int, stop: int) -> FiberTrace3DBatch:
    return FiberTrace3DBatch(
        volume=batch.volume[start:stop],
        valid_mask=batch.valid_mask[start:stop],
        direction_target=batch.direction_target[start:stop],
        direction_weight=batch.direction_weight[start:stop],
        direction_mask=batch.direction_mask[start:stop],
        presence_target=batch.presence_target[start:stop],
        presence_mask=batch.presence_mask[start:stop],
        cp_local_zyx=batch.cp_local_zyx[start:stop],
        crop_origin_zyx=batch.crop_origin_zyx[start:stop],
        sample_indices=batch.sample_indices[start:stop],
        record_indices=batch.record_indices[start:stop],
        control_point_indices=batch.control_point_indices[start:stop],
        fiber_paths=batch.fiber_paths[start:stop],
    )


def _forward_loss_microbatched(
    model: torch.nn.Module,
    batch: FiberTrace3DBatch,
    *,
    micro_batch_size: int,
    direction_weight: float,
    presence_weight: float,
    backward: bool,
) -> dict[str, float]:
    batch_size = int(batch.volume.shape[0])
    micro = max(1, int(micro_batch_size))
    totals = {"total": 0.0, "direction": 0.0, "presence": 0.0}
    for start in range(0, batch_size, micro):
        stop = min(start + micro, batch_size)
        sub = _slice_batch(batch, start, stop)
        output = model(sub.volume)
        losses = compute_losses(
            output,
            sub,
            direction_weight=direction_weight,
            presence_weight=presence_weight,
        )
        scale = float(stop - start) / float(batch_size)
        if backward:
            (losses["total"] * scale).backward()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * scale
    return totals


def run_training(config_path: str | Path) -> None:
    raw_config = _load_raw_config(config_path)
    loader_config = load_config(config_path)
    training = dict(raw_config.get("training", {}))
    device = _device_from_training(training)
    loader = FiberTrace3DLoader(loader_config)
    test_loader = None
    if raw_config.get("test_datasets"):
        test_raw = dict(raw_config)
        test_raw["datasets"] = raw_config["test_datasets"]
        tmp_path = Path("/tmp") / f"fiber_trace_3d_test_{int(time.time() * 1000)}.json"
        tmp_path.write_text(json.dumps(_json_safe(test_raw)), encoding="utf-8")
        try:
            test_loader = FiberTrace3DLoader(load_config(tmp_path))
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    model = build_fiber_trace_3d_model(raw_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1.0e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    resume = training.get("resume") or raw_config.get("resume")
    start_step = 0
    if resume:
        start_step = _load_snapshot(resume, model=model, optimizer=optimizer, map_location=device)

    run_dir, snapshot_dir = _resolve_run_layout(raw_config)
    writer = _make_summary_writer(
        run_dir,
        enabled=bool(training.get("tensorboard_enabled", True)),
    )
    if writer is not None:
        writer.add_text("config/json", json.dumps(_json_safe(raw_config), indent=2, sort_keys=True), 0)

    max_steps = int(training.get("max_steps", 1))
    if max_steps <= 0:
        max_steps = max(1, math.ceil(loader.sample_count / max(loader.config.batch_size, 1)))
    micro_batch_size = int(training.get("model_micro_batch_size", training.get("micro_batch_size", loader.config.batch_size)))
    scalar_interval = int(training.get("scalar_log_interval", 100))
    checkpoint_interval = int(training.get("checkpoint_interval", 100))
    test_interval = int(training.get("test_interval", 0))
    test_control_points = int(training.get("test_control_points", loader.config.batch_size))
    if test_control_points <= 0:
        test_control_points = test_loader.sample_count if test_loader is not None else loader.sample_count
    direction_weight = float(training.get("direction_weight", 1.0))
    presence_weight = float(training.get("presence_weight", 1.0))
    best_metric = math.inf

    print(
        "fiber_trace_3d train: "
        f"samples={loader.sample_count} batch_size={loader.config.batch_size} "
        f"micro_batch_size={micro_batch_size} device={device} run_dir={run_dir}",
        flush=True,
    )

    for step in range(start_step + 1, max_steps + 1):
        load_start = time.perf_counter()
        sample_index = (step - 1) * loader.config.batch_size
        batch = loader.load_batch(sample_index, sample_mode="random", device=device)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        optimizer.zero_grad(set_to_none=True)
        fw_start = time.perf_counter()
        losses = _forward_loss_microbatched(
            model,
            batch,
            micro_batch_size=micro_batch_size,
            direction_weight=direction_weight,
            presence_weight=presence_weight,
            backward=True,
        )
        optimizer.step()
        step_ms = (time.perf_counter() - fw_start) * 1000.0

        if step <= 100 or step % scalar_interval == 0:
            print(
                f"step={step} loss_total={losses['total']:.6f} "
                f"loss_direction={losses['direction']:.6f} "
                f"loss_presence={losses['presence']:.6f} "
                f"load_ms={load_ms:.1f} fw_bw_step_ms={step_ms:.1f}",
                flush=True,
            )
        if writer is not None and (step == 1 or step % scalar_interval == 0):
            writer.add_scalar("train/loss_total", losses["total"], step)
            writer.add_scalar("train/loss_direction", losses["direction"], step)
            writer.add_scalar("train/loss_presence", losses["presence"], step)
            writer.add_scalar("timing/load_ms", load_ms, step)
            writer.add_scalar("timing/fw_bw_step_ms", step_ms, step)

        metric = losses["total"]
        if test_loader is not None and test_interval > 0 and step % test_interval == 0:
            test_losses = evaluate_dense_loss(
                model,
                test_loader,
                device=device,
                start_sample_index=int(training.get("test_start_sample_index", 0)),
                sample_count=test_control_points,
                micro_batch_size=micro_batch_size,
                direction_weight=direction_weight,
                presence_weight=presence_weight,
            )
            metric = test_losses["total"]
            print(
                f"test step={step} loss_total={test_losses['total']:.6f} "
                f"loss_direction={test_losses['direction']:.6f} "
                f"loss_presence={test_losses['presence']:.6f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("test/loss_total", test_losses["total"], step)
                writer.add_scalar("test/loss_direction", test_losses["direction"], step)
                writer.add_scalar("test/loss_presence", test_losses["presence"], step)

        if step % checkpoint_interval == 0 or step == max_steps:
            _save_snapshot(
                snapshot_dir / "current.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config=raw_config,
                metric=metric,
            )
        if metric < best_metric:
            best_metric = float(metric)
            _save_snapshot(
                snapshot_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config=raw_config,
                metric=best_metric,
            )
    if writer is not None:
        writer.flush()
        writer.close()


def run_benchmark(config_path: str | Path, *, load_only: bool, batches: int) -> None:
    raw_config = _load_raw_config(config_path)
    loader = FiberTrace3DLoader(load_config(config_path))
    training = dict(raw_config.get("training", {}))
    device = _device_from_training(training)
    model = build_fiber_trace_3d_model(raw_config).to(device)
    model.eval()
    micro_batch_size = int(training.get("model_micro_batch_size", loader.config.batch_size))
    direction_weight = float(training.get("direction_weight", 1.0))
    presence_weight = float(training.get("presence_weight", 1.0))
    print("batch patches total_ms load_ms fw_ms")
    for batch_index in range(1, int(batches) + 1):
        start = time.perf_counter()
        batch = loader.load_batch(
            (batch_index - 1) * loader.config.batch_size,
            sample_mode="random",
            device=device,
        )
        load_ms = (time.perf_counter() - start) * 1000.0
        fw_ms = 0.0
        if not load_only:
            fw_start = time.perf_counter()
            with torch.no_grad():
                _forward_loss_microbatched(
                    model,
                    batch,
                    micro_batch_size=micro_batch_size,
                    direction_weight=direction_weight,
                    presence_weight=presence_weight,
                    backward=False,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fw_ms = (time.perf_counter() - fw_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"{batch_index:5d} {loader.config.batch_size:7d} "
            f"{total_ms:8.2f} {load_ms:8.2f} {fw_ms:8.2f}",
            flush=True,
        )


def run_prefetch(config_path: str | Path, *, prefetch_steps: int, workers: int | None) -> None:
    loader = FiberTrace3DLoader(load_config(config_path))
    if int(prefetch_steps) == 0:
        sample_count = loader.sample_count
    else:
        sample_count = int(prefetch_steps) * int(loader.config.batch_size)
    summary = loader.prefetch(0, sample_count, workers=workers)
    print("fiber_trace_3d prefetch summary: " + json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--prefetch-steps", type=int, default=1)
    parser.add_argument("--prefetch-workers", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-batches", type=int, default=10)
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()
    if args.prefetch:
        run_prefetch(
            args.config,
            prefetch_steps=int(args.prefetch_steps),
            workers=args.prefetch_workers,
        )
    elif args.benchmark:
        run_benchmark(
            args.config,
            load_only=bool(args.load_only),
            batches=int(args.benchmark_batches),
        )
    else:
        run_training(args.config)


if __name__ == "__main__":
    main()
