from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace_2d.augmentation import overlay_line_coords_rgb, resolve_torch_device
from vesuvius.neural_tracing.fiber_trace_2d.direction import (
    DirectionSupervision,
    build_direction_supervision,
    cp_neighborhood_yx,
    decode_lasagna_direction_xy,
    direction_mse_loss,
    line_cp_and_tangent_xy,
)
from vesuvius.neural_tracing.fiber_trace_2d.loader import FiberStrip2DBatch, FiberStrip2DLoader, SamplerFactory, load_config
from vesuvius.neural_tracing.fiber_trace_2d.model import (
    FiberStripDirectionModelConfig,
    FiberStripDirectionNet,
)


@dataclass(frozen=True)
class FiberStripTrainingConfig:
    run_path: str = "runs/fiber_trace_2d"
    run_name: str = "fiber_strip_direction"
    max_steps: int = 1000
    learning_rate: float = 1.0e-3
    scalar_log_interval: int = 100
    tensorboard_image_interval: int = 1000
    checkpoint_interval: int = 1000
    train_control_points_per_step: int = 4
    device: str = "auto"
    tensorboard_enabled: bool = True
    model_hidden_channels: int = 32
    model_depth: int = 5


def _training_config_from_raw(raw: dict[str, Any]) -> FiberStripTrainingConfig:
    train = raw.get("training", {})
    if train is None:
        train = {}
    if not isinstance(train, dict):
        raise ValueError("'training' must be a JSON object when provided")

    def get(name: str, default: Any) -> Any:
        return train.get(name, raw.get(f"train_{name}", default))

    config = FiberStripTrainingConfig(
        run_path=str(get("run_path", "runs/fiber_trace_2d")),
        run_name=str(get("run_name", "fiber_strip_direction")),
        max_steps=int(get("max_steps", 1000)),
        learning_rate=float(get("learning_rate", 1.0e-3)),
        scalar_log_interval=max(1, int(get("scalar_log_interval", 100))),
        tensorboard_image_interval=max(1, int(get("tensorboard_image_interval", 1000))),
        checkpoint_interval=max(1, int(get("checkpoint_interval", 1000))),
        train_control_points_per_step=max(1, int(get("control_points_per_step", get("control_points", 4)))),
        device=str(get("device", "auto")),
        tensorboard_enabled=bool(get("tensorboard_enabled", True)),
        model_hidden_channels=max(1, int(get("model_hidden_channels", 32))),
        model_depth=max(1, int(get("model_depth", 5))),
    )
    if config.max_steps <= 0:
        raise ValueError("training.max_steps must be > 0")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be positive and finite")
    return config


def _load_raw_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return raw


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


def _make_run_dir(config: FiberStripTrainingConfig) -> Path:
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.run_path).expanduser() / f"{config.run_name}_{date}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def _make_summary_writer(run_dir: Path, *, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires the tensorboard package. "
            "Install tensorboard or set training.tensorboard_enabled=false."
        ) from exc
    return SummaryWriter(log_dir=str(run_dir))


def _flatten_batch(batch: FiberStrip2DBatch) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(batch.images, dtype=np.float32)
    valid = np.asarray(batch.valid_mask, dtype=bool)
    if images.ndim != 5 or images.shape[2] != 1:
        raise ValueError("batch.images must have shape B,Z,1,H,W")
    b, z, _, h, w = images.shape
    return images.reshape(b * z, 1, h, w), valid.reshape(b * z, h, w)


def _prepare_images(images_np: np.ndarray, valid_np: np.ndarray, *, device: torch.device) -> torch.Tensor:
    images = torch.as_tensor(images_np, dtype=torch.float32, device=device)
    valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device).unsqueeze(1)
    counts = valid.sum(dim=(2, 3), keepdim=True).clamp_min(1)
    masked = torch.where(valid, images, torch.zeros_like(images))
    mean = masked.sum(dim=(2, 3), keepdim=True) / counts
    var = torch.where(valid, (images - mean) ** 2, torch.zeros_like(images)).sum(dim=(2, 3), keepdim=True) / counts
    std = torch.sqrt(var.clamp_min(1.0e-6))
    return torch.where(valid, (images - mean) / std, torch.zeros_like(images))


def _compute_batch_loss(
    model: FiberStripDirectionNet,
    batch: FiberStrip2DBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, DirectionSupervision]:
    images_np, valid_np = _flatten_batch(batch)
    images = _prepare_images(images_np, valid_np, device=device)
    supervision = build_direction_supervision(batch.samples, valid_np, device=device)
    outputs = model(images)
    loss = direction_mse_loss(outputs, supervision)
    return loss, outputs, supervision


def _cache_scalars(cache_stats: Any | None) -> dict[str, float]:
    if cache_stats is None:
        return {}
    return {
        "cache/hits": float(getattr(cache_stats, "cache_hits", 0)),
        "cache/downloads": float(getattr(cache_stats, "downloads", 0)),
        "cache/misses": float(getattr(cache_stats, "missing", 0)) + float(getattr(cache_stats, "negative_hits", 0)),
        "cache/hit_mib": float(getattr(cache_stats, "cache_hit_bytes", 0)) / (1024.0 * 1024.0),
        "cache/download_mib": float(getattr(cache_stats, "download_bytes", 0)) / (1024.0 * 1024.0),
        "cache/hit_ms": float(getattr(cache_stats, "cache_hit_ms", 0.0)),
        "cache/download_ms": float(getattr(cache_stats, "download_ms", 0.0)),
    }


def _to_u8_image(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if bool(valid.any()):
        values = arr[valid]
        lo = float(values.min())
        hi = float(values.max())
        scale = 255.0 / max(hi - lo, 1.0e-6)
        out[valid] = np.clip((arr[valid] - lo) * scale, 0.0, 255.0).astype(np.uint8)
    return out


def _draw_supervision_and_direction(
    rgb: np.ndarray,
    *,
    sample_line_xy: np.ndarray,
    cp_xy: np.ndarray,
    prediction_xy: np.ndarray | None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    for y, x in cp_neighborhood_yx(cp_xy, (pil.height, pil.width)).tolist():
        draw.rectangle((x - 1, y - 1, x + 1, y + 1), outline=(255, 255, 0, 220))
    x0, y0 = float(cp_xy[0]), float(cp_xy[1])
    draw.line((x0 - 4, y0, x0 - 2, y0), fill=(0, 255, 255, 240), width=1)
    draw.line((x0 + 2, y0, x0 + 4, y0), fill=(0, 255, 255, 240), width=1)
    draw.line((x0, y0 - 4, x0, y0 - 2), fill=(0, 255, 255, 240), width=1)
    draw.line((x0, y0 + 2, x0, y0 + 4), fill=(0, 255, 255, 240), width=1)
    if prediction_xy is not None and np.isfinite(prediction_xy).all():
        direction = np.asarray(prediction_xy, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1.0e-6:
            direction = direction / norm
            length = 8.0
            draw.line(
                (
                    x0 - direction[0] * length,
                    y0 - direction[1] * length,
                    x0 + direction[0] * length,
                    y0 + direction[1] * length,
                ),
                fill=(0, 255, 0, 240),
                width=1,
            )
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8)


def _make_training_visualization(
    batch: FiberStrip2DBatch,
    outputs: torch.Tensor,
    *,
    max_patches: int = 8,
) -> np.ndarray:
    flat_images, flat_valid = _flatten_batch(batch)
    outputs_cpu = outputs.detach().cpu()
    patch_count = min(int(flat_images.shape[0]), int(max_patches))
    cells: list[np.ndarray] = []
    for patch_index in range(patch_count):
        sample = batch.samples[patch_index]
        image_u8 = _to_u8_image(flat_images[patch_index, 0], flat_valid[patch_index])
        rgb = overlay_line_coords_rgb(image_u8, sample.line_xy, opacity=0.5, thickness=1)
        cp_tangent = line_cp_and_tangent_xy(sample.line_xy, getattr(sample, "control_point_xy", None))
        prediction_xy = None
        if cp_tangent is not None:
            cp_xy, _ = cp_tangent
            center = np.rint(cp_xy).astype(np.int64)
            y = int(np.clip(center[1], 0, flat_valid.shape[1] - 1))
            x = int(np.clip(center[0], 0, flat_valid.shape[2] - 1))
            encoded = outputs_cpu[patch_index, :, y, x]
            prediction_xy = decode_lasagna_direction_xy(encoded).cpu().numpy()
            rgb = _draw_supervision_and_direction(
                rgb,
                sample_line_xy=sample.line_xy,
                cp_xy=cp_xy,
                prediction_xy=prediction_xy,
            )
        cells.append(rgb)
    if not cells:
        return np.zeros((3, 1, 1), dtype=np.uint8)
    h, w = cells[0].shape[:2]
    cols = min(4, len(cells))
    rows = int(math.ceil(len(cells) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        row = i // cols
        col = i % cols
        sheet[row * h : (row + 1) * h, col * w : (col + 1) * w] = cell
    return np.transpose(sheet, (2, 0, 1))


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model: FiberStripDirectionNet,
    optimizer: torch.optim.Optimizer,
    loss: float,
    raw_config: dict[str, Any],
) -> None:
    torch.save(
        {
            "step": int(step),
            "loss": float(loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": _json_safe(raw_config),
        },
        path,
    )


def run_training(
    config_path: str | Path,
    *,
    sampler_factory: SamplerFactory | None = None,
) -> Path:
    raw_config = _load_raw_config(config_path)
    training = _training_config_from_raw(raw_config)
    loader_config = load_config(config_path)
    loader = FiberStrip2DLoader(loader_config, sampler_factory=sampler_factory)
    expected_patches = int(training.train_control_points_per_step) * int(loader_config.strip_z_offset_count)
    if expected_patches != 64:
        print(
            "fiber_trace_2d train: patch batch is "
            f"{expected_patches}, expected 64 for the default 4 control points x 16 strip offsets",
            flush=True,
        )

    device = resolve_torch_device(training.device)
    model = FiberStripDirectionNet(
        FiberStripDirectionModelConfig(
            in_channels=1,
            hidden_channels=training.model_hidden_channels,
            depth=training.model_depth,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)

    run_dir = _make_run_dir(training)
    snapshots = run_dir / "snapshots"
    writer = _make_summary_writer(run_dir, enabled=training.tensorboard_enabled)
    if writer is not None:
        writer.add_text("config/json", json.dumps(_json_safe(raw_config), indent=2, sort_keys=True), 0)
    print(f"fiber_trace_2d train run_dir={run_dir}", flush=True)

    best_loss = float("inf")
    last_loss = float("nan")
    try:
        for step in range(1, training.max_steps + 1):
            start_sample_index = (step - 1) * int(training.train_control_points_per_step)
            t0 = time.perf_counter()
            batch = loader.load_batch(start_sample_index, batch_size=training.train_control_points_per_step)
            load_ms = (time.perf_counter() - t0) * 1000.0

            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, outputs, supervision = _compute_batch_loss(model, batch, device=device)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            last_loss = loss_value

            if writer is not None and (step == 1 or step % training.scalar_log_interval == 0):
                writer.add_scalar("train/loss_direction", loss_value, step)
                writer.add_scalar("train/supervision_samples", int(supervision.target.shape[0]), step)
                writer.add_scalar("timing/load_ms", load_ms, step)
                for key, value in _cache_scalars(batch.cache_stats).items():
                    writer.add_scalar(key, value, step)
            if writer is not None and (step == 1 or step % training.tensorboard_image_interval == 0):
                writer.add_image("train/batch_direction_overlay", _make_training_visualization(batch, outputs), step)
            if step == 1 or step % training.checkpoint_interval == 0 or step == training.max_steps:
                _save_checkpoint(
                    snapshots / "current.pt",
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    loss=loss_value,
                    raw_config=raw_config,
                )
            if loss_value < best_loss:
                best_loss = loss_value
                _save_checkpoint(
                    snapshots / "best.pt",
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    loss=loss_value,
                    raw_config=raw_config,
                )
            if step == 1 or step % training.scalar_log_interval == 0:
                print(
                    f"step={step} loss_direction={loss_value:.6f} "
                    f"supervision_samples={int(supervision.target.shape[0])} load_ms={load_ms:.1f}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
    print(f"fiber_trace_2d train complete step={training.max_steps} loss_direction={last_loss:.6f}", flush=True)
    return run_dir


def prefetch_training(
    config_path: str | Path,
    *,
    prefetch_steps: int | None = None,
    prefetch_start_step: int = 1,
    sampler_factory: SamplerFactory | None = None,
) -> dict[str, Any]:
    raw_config = _load_raw_config(config_path)
    training = _training_config_from_raw(raw_config)
    if int(prefetch_start_step) <= 0:
        raise ValueError("--prefetch-start-step must be >= 1")
    if prefetch_steps is not None and int(prefetch_steps) < 0:
        raise ValueError("--prefetch-steps must be >= 0")

    effective_steps = training.max_steps if prefetch_steps is None or int(prefetch_steps) == 0 else int(prefetch_steps)
    start_sample_index = (int(prefetch_start_step) - 1) * int(training.train_control_points_per_step)
    sample_count = int(effective_steps) * int(training.train_control_points_per_step)
    loader_config = load_config(config_path)
    loader = FiberStrip2DLoader(loader_config, sampler_factory=sampler_factory)
    print(
        "fiber_trace_2d prefetch "
        f"start_step={int(prefetch_start_step)} steps={int(effective_steps)} "
        f"control_points_per_step={int(training.train_control_points_per_step)} "
        f"start_sample_index={start_sample_index} samples={sample_count}",
        flush=True,
    )
    summary = loader.prefetch(start_sample_index, sample_count)
    print(
        "fiber_trace_2d prefetch complete "
        f"generated={int(summary.get('generated', 0))} missing={int(summary.get('missing', 0))} "
        f"downloaded={int(summary.get('downloaded', 0))} errors={int(summary.get('errors', 0))}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train V0 2D fiber-strip direction model")
    parser.add_argument("config", help="Path to fiber_trace_2d JSON config")
    parser.add_argument("--prefetch", action="store_true", help="Prefetch training chunks and exit without training")
    parser.add_argument(
        "--prefetch-steps",
        type=int,
        default=None,
        help="Training steps to prefetch; 0 or omitted means all configured training.max_steps",
    )
    parser.add_argument(
        "--prefetch-start-step",
        type=int,
        default=1,
        help="1-based training step whose deterministic sample range starts prefetching",
    )
    args = parser.parse_args(argv)
    if args.prefetch:
        try:
            prefetch_training(
                args.config,
                prefetch_steps=args.prefetch_steps,
                prefetch_start_step=args.prefetch_start_step,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return
    run_training(args.config)


if __name__ == "__main__":
    main()
