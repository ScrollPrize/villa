from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace_2d.augmentation import (
    limit_augmentation_rows,
    overlay_line_coords_rgb,
    random_combined_augmentation,
    resolve_torch_device,
)
from vesuvius.neural_tracing.fiber_trace_2d.direction import (
    decode_lasagna_direction_xy,
    line_cp_and_tangent_xy,
)
from vesuvius.neural_tracing.fiber_trace_2d.loader import FiberStrip2DLoader, load_config
from vesuvius.neural_tracing.fiber_trace_2d.model import (
    FiberStripDirectionModelConfig,
    FiberStripDirectionNet,
)


def _to_u8_image(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[valid] = np.clip(arr[valid], 0.0, 255.0).astype(np.uint8)
    return out


def _write_jpg(path: Path, image_u8: np.ndarray) -> None:
    from PIL import Image

    image = np.asarray(image_u8, dtype=np.uint8)
    mode = "RGB" if image.ndim == 3 and image.shape[-1] == 3 else "L"
    Image.fromarray(image, mode=mode).save(path, quality=95)


def _draw_cp_crosshair(image: np.ndarray, cp_xy: np.ndarray | None) -> np.ndarray:
    from PIL import Image, ImageDraw

    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    point = np.asarray(cp_xy, dtype=np.float32) if cp_xy is not None else np.asarray([np.nan, np.nan])
    if point.shape != (2,) or not bool(np.isfinite(point).all()):
        return arr
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    height, width = arr.shape[:2]
    if x < 0 or x >= width or y < 0 or y >= height:
        return arr
    pil = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(pil, mode="RGBA")
    arm = 5
    gap = 1
    color = (32, 255, 255, 255)
    segments = [
        ((x, y - arm), (x, y - gap)),
        ((x, y + gap), (x, y + arm)),
    ]
    for start, end in segments:
        draw.line((start, end), fill=color, width=1)
    return np.asarray(pil, dtype=np.uint8)


def _overlay_polyline_rgb(
    image: np.ndarray,
    line_xy: np.ndarray,
    *,
    color_rgba: tuple[int, int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    coords = np.asarray(line_xy, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        return arr
    finite = np.isfinite(coords).all(axis=1)
    coords = coords[finite]
    if coords.shape[0] == 0:
        return arr
    pil = Image.fromarray(arr, mode="RGB")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    points = [(float(x), float(y)) for x, y in coords.tolist()]
    if len(points) == 1:
        x, y = points[0]
        r = max(1, int(thickness))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color_rgba)
    else:
        draw.line(points, fill=color_rgba, width=max(1, int(thickness)), joint="curve")
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8)


def _draw_label_band(image: np.ndarray, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    text = str(label)
    font = ImageFont.load_default()
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe, mode="RGBA")
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)
    pad_x = 3
    pad_y = 2
    band_h = int(text_h + 2 * pad_y)
    cell = np.zeros((arr.shape[0] + band_h, arr.shape[1], 3), dtype=np.uint8)
    cell[band_h:, :, :] = arr
    pil = Image.fromarray(cell, mode="RGB")
    draw = ImageDraw.Draw(pil, mode="RGBA")
    draw.text((pad_x, pad_y), text[: max(1, arr.shape[1])], fill=(255, 255, 255, 255), font=font)
    return np.asarray(pil, dtype=np.uint8)


def _write_contact_sheet(
    path: Path, images: list[np.ndarray], *, columns: int = 8, labels: list[str] | None = None
) -> None:
    if not images:
        return
    if labels is not None and len(labels) != len(images):
        raise ValueError("labels length must match images length")
    prepared = [
        _draw_label_band(image, labels[i]) if labels is not None else np.asarray(image, dtype=np.uint8)
        for i, image in enumerate(images)
    ]
    h = max(int(image.shape[0]) for image in prepared)
    w = max(int(image.shape[1]) for image in prepared)
    channels = () if prepared[0].ndim == 2 else (prepared[0].shape[2],)
    cols = max(1, min(columns, len(images)))
    rows = (len(images) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, *channels), dtype=np.uint8)
    for i, image in enumerate(prepared):
        row = i // cols
        col = i % cols
        image_h, image_w = image.shape[:2]
        sheet[row * h : row * h + image_h, col * w : col * w + image_w] = image
    _write_jpg(path, sheet)


def _inside_trace_margin(point_xy: np.ndarray, shape_hw: tuple[int, int], margin: float) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    height, width = (int(v) for v in shape_hw)
    m = max(0.0, float(margin))
    return m <= x <= (float(width - 1) - m) and m <= y <= (float(height - 1) - m)


def _bilinear_direction_sample(
    direction_xy: np.ndarray,
    point_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    field = np.asarray(direction_xy, dtype=np.float32)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("direction_xy must have shape H,W,2")
    point = np.asarray(point_xy, dtype=np.float32)
    if point.shape != (2,) or not bool(np.isfinite(point).all()):
        return None
    height, width = int(field.shape[0]), int(field.shape[1])
    x = float(point[0])
    y = float(point[1])
    if x < 0.0 or y < 0.0 or x > float(width - 1) or y > float(height - 1):
        return None
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (height, width):
            raise ValueError("valid_mask must match direction field shape")
        if not (bool(valid[y0, x0]) and bool(valid[y0, x1]) and bool(valid[y1, x0]) and bool(valid[y1, x1])):
            return None
    tx = np.float32(x - float(x0))
    ty = np.float32(y - float(y0))
    top = field[y0, x0] * (1.0 - tx) + field[y0, x1] * tx
    bottom = field[y1, x0] * (1.0 - tx) + field[y1, x1] * tx
    direction = top * (1.0 - ty) + bottom * ty
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        return None
    return (direction / norm).astype(np.float32)


def _trace_direction_line(
    direction_xy: np.ndarray,
    cp_xy: np.ndarray,
    tangent_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
    max_steps: int | None = None,
) -> np.ndarray:
    field = np.asarray(direction_xy, dtype=np.float32)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("direction_xy must have shape H,W,2")
    cp = np.asarray(cp_xy, dtype=np.float32)
    tangent = np.asarray(tangent_xy, dtype=np.float32)
    if cp.shape != (2,) or tangent.shape != (2,):
        raise ValueError("cp_xy and tangent_xy must have shape (2,)")
    tangent_norm = float(np.linalg.norm(tangent))
    if not np.isfinite(tangent_norm) or tangent_norm <= 1.0e-6:
        raise ValueError("tangent_xy must be finite and non-zero")
    step = max(float(step_px), 1.0e-3)
    unit_tangent = (tangent / tangent_norm).astype(np.float32)
    shape_hw = (int(field.shape[0]), int(field.shape[1]))
    if max_steps is None:
        max_steps = int(np.ceil(np.hypot(*shape_hw) / step)) + 2

    def one_side(initial: np.ndarray) -> list[np.ndarray]:
        points = [cp.astype(np.float32)]
        previous = initial.astype(np.float32)
        current = cp.astype(np.float32)
        for _ in range(int(max_steps)):
            if not _inside_trace_margin(current, shape_hw, rf_margin_px):
                break
            sampled = _bilinear_direction_sample(field, current, valid_mask=valid_mask)
            if sampled is None:
                break
            if float(np.dot(sampled, previous)) < 0.0:
                sampled = -sampled
            next_point = current + sampled * np.float32(step)
            if not _inside_trace_margin(next_point, shape_hw, rf_margin_px):
                break
            points.append(next_point.astype(np.float32))
            current = next_point.astype(np.float32)
            previous = sampled.astype(np.float32)
        return points

    backward = one_side(-unit_tangent)
    forward = one_side(unit_tangent)
    combined = list(reversed(backward[1:])) + forward
    return np.stack(combined, axis=0).astype(np.float32)


def _prepare_model_image(image: np.ndarray, valid_mask: np.ndarray, *, device: torch.device) -> torch.Tensor:
    image_t = torch.as_tensor(np.asarray(image, dtype=np.float32), dtype=torch.float32, device=device).view(
        1, 1, *np.asarray(image).shape
    )
    valid_t = torch.as_tensor(np.asarray(valid_mask, dtype=bool), dtype=torch.bool, device=device).view(
        1, 1, *np.asarray(valid_mask).shape
    )
    counts = valid_t.sum(dim=(2, 3), keepdim=True).clamp_min(1)
    masked = torch.where(valid_t, image_t, torch.zeros_like(image_t))
    mean = masked.sum(dim=(2, 3), keepdim=True) / counts
    var = torch.where(valid_t, (image_t - mean) ** 2, torch.zeros_like(image_t)).sum(dim=(2, 3), keepdim=True) / counts
    std = torch.sqrt(var.clamp_min(1.0e-6))
    return torch.where(valid_t, (image_t - mean) / std, torch.zeros_like(image_t))


def _model_config_from_checkpoint(checkpoint: dict, loader: FiberStrip2DLoader) -> FiberStripDirectionModelConfig:
    raw_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    training = raw_config.get("training", {}) if isinstance(raw_config, dict) else {}
    if not isinstance(training, dict):
        training = {}
    return FiberStripDirectionModelConfig(
        in_channels=1,
        hidden_channels=max(1, int(training.get("model_hidden_channels", 32))),
        depth=max(1, int(training.get("model_depth", 5))),
    )


def _load_direction_model(
    checkpoint_path: str | Path,
    loader: FiberStrip2DLoader,
    *,
    device: torch.device,
) -> tuple[FiberStripDirectionNet, dict]:
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a fiber_trace_2d training checkpoint")
    model = FiberStripDirectionNet(_model_config_from_checkpoint(checkpoint, loader)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _export_line_trace_vis(
    loader: FiberStrip2DLoader,
    sample_index: int,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    step_px: float,
    rf_margin_px: float | None,
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(loader.config.augment.device)
    sample, image, valid_mask = loader.build_center_strip_patch(sample_index)
    cp_tangent = line_cp_and_tangent_xy(sample.line_xy, sample.control_point_xy)
    if cp_tangent is None:
        raise ValueError("selected sample has no usable transformed CP/tangent line")
    cp_xy, tangent_xy = cp_tangent
    model, checkpoint = _load_direction_model(checkpoint_path, loader, device=device)
    # The model has one 3x3 convolution per configured depth, giving radius=depth.
    configured_depth = max(1, int(_model_config_from_checkpoint(checkpoint, loader).depth))
    margin = float(configured_depth if rf_margin_px is None else rf_margin_px)
    with torch.no_grad():
        input_image = _prepare_model_image(image, valid_mask, device=device)
        encoded = model(input_image)[0].permute(1, 2, 0)
        direction_xy = decode_lasagna_direction_xy(encoded, bins=720).detach().cpu().numpy()
    traced_line = _trace_direction_line(
        direction_xy,
        cp_xy,
        tangent_xy,
        valid_mask=valid_mask,
        step_px=step_px,
        rf_margin_px=margin,
    )
    image_u8 = _to_u8_image(image, valid_mask)
    overlay = overlay_line_coords_rgb(image_u8, sample.line_xy, opacity=0.5, thickness=1)
    overlay = _overlay_polyline_rgb(overlay, traced_line, color_rgba=(0, 255, 0, 230), thickness=1)
    overlay = _draw_cp_crosshair(overlay, cp_xy)
    _write_jpg(out / "line_trace_vis.jpg", overlay)
    summary = [
        f"sample_index={sample_index}",
        f"checkpoint={Path(checkpoint_path).expanduser().resolve()}",
        f"checkpoint_step={checkpoint.get('step', 'unknown')}",
        f"fiber_path={sample.fiber_path}",
        f"control_point_index={sample.control_point_index}",
        f"strip_z_offset={sample.strip_z_offset}",
        f"cp_xy=({cp_xy[0]:.3f}, {cp_xy[1]:.3f})",
        f"tangent_xy=({tangent_xy[0]:.6f}, {tangent_xy[1]:.6f})",
        f"step_px={float(step_px):.3f}",
        f"rf_margin_px={margin:.3f}",
        f"trace_points={int(traced_line.shape[0])}",
    ]
    (out / "line_trace_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"exported line_trace_vis.jpg and line_trace_summary.txt to {out}")


def _offset_label(offset: float) -> str:
    value = float(offset)
    if value.is_integer():
        return f"{int(value):+04d}"
    return f"{value:+07.3f}".replace(".", "p")


def _batch_control_points(batch) -> list[tuple[int, np.ndarray]]:
    points: list[tuple[int, np.ndarray]] = []
    offsets_per_sample = int(len(batch.strip_z_offsets))
    for batch_index in range(batch.images.shape[0]):
        sample = batch.samples[batch_index * offsets_per_sample]
        points.append((int(sample.control_point_index), np.asarray(sample.control_point_xyz, dtype=np.float64)))
    return points


class _Timer:
    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "_Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        return None


def _add_timing(total: dict[str, float], key: str, value_ms: float) -> None:
    total[key] = total.get(key, 0.0) + float(value_ms)


def _print_timing_table(rows: list[tuple[str, dict[str, float]]], totals: dict[str, float]) -> None:
    keys = [
        "total",
        "loader_total",
        "descriptor",
        "line_window",
        "lasagna_normals",
        "strip_coords",
        "coord_augmentation",
        "volume_sample",
        "value_augmentation",
        "line_coords",
        "to_u8",
        "overlay",
    ]
    header = "name".ljust(16) + "".join(key[:12].rjust(13) for key in keys)
    print("fiber_trace_2d augment-vis timings in ms")
    print(header)
    for name, timing in rows:
        print(name.ljust(16) + "".join(f"{timing.get(key, 0.0):13.1f}" for key in keys))
    print("total".ljust(16) + "".join(f"{totals.get(key, 0.0):13.1f}" for key in keys))
    volume_stats = {
        key.removeprefix("volume_stat_"): value
        for key, value in totals.items()
        if key.startswith("volume_stat_")
    }
    if volume_stats:
        print(
            "fiber_trace_2d volume sampler stats: "
            + " ".join(f"{key}={value:.0f}" for key, value in sorted(volume_stats.items()))
        )


def _image_stats(image: np.ndarray, valid_mask: np.ndarray, image_u8: np.ndarray) -> dict[str, float]:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    out: dict[str, float] = {
        "valid": float(np.count_nonzero(valid)),
        "u8_nonzero": float(np.count_nonzero(np.asarray(image_u8))),
    }
    if bool(valid.any()):
        values = arr[valid]
        out["min"] = float(values.min())
        out["max"] = float(values.max())
        out["mean"] = float(values.mean())
    else:
        out["min"] = 0.0
        out["max"] = 0.0
        out["mean"] = 0.0
    return out


def _print_image_stats(rows: list[tuple[str, dict[str, float]]]) -> None:
    print("fiber_trace_2d augment-vis image stats")
    print("name".ljust(16) + f"{'valid':>10}{'min':>12}{'max':>12}{'mean':>12}{'u8_nz':>10}")
    for name, stats in rows:
        print(
            name.ljust(16)
            + f"{stats.get('valid', 0.0):10.0f}"
            + f"{stats.get('min', 0.0):12.3f}"
            + f"{stats.get('max', 0.0):12.3f}"
            + f"{stats.get('mean', 0.0):12.3f}"
            + f"{stats.get('u8_nonzero', 0.0):10.0f}"
        )


def _export_batch(batch, output_dir: str | Path) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    image_files: list[str] = []
    mask_files: list[str] = []
    contact_images: list[np.ndarray] = []
    for batch_index in range(batch.images.shape[0]):
        for offset_index, offset in enumerate(batch.strip_z_offsets.tolist()):
            stem = f"sample_{batch_index:03d}_offset_{_offset_label(offset)}"
            image_name = f"{stem}.jpg"
            mask_name = f"{stem}_valid.jpg"
            image = batch.images[batch_index, offset_index, 0]
            valid_mask = batch.valid_mask[batch_index, offset_index].astype(bool)
            image_u8 = _to_u8_image(image, valid_mask)
            mask_u8 = (valid_mask.astype(np.uint8) * 255)
            _write_jpg(out / image_name, image_u8)
            _write_jpg(out / mask_name, mask_u8)

            contact_images.append(image_u8)
            image_files.append(image_name)
            mask_files.append(mask_name)

    _write_contact_sheet(out / "contact_sheet.jpg", contact_images)
    summary_lines = [
        f"images_shape={list(batch.images.shape)}",
        f"coords_zyx_shape={list(batch.coords_zyx.shape)}",
        f"valid_mask_shape={list(batch.valid_mask.shape)}",
        f"strip_z_offsets={batch.strip_z_offsets.tolist()}",
        f"record_indices={batch.record_indices.tolist()}",
        f"control_point_indices={batch.control_point_indices.tolist()}",
    ]
    for i, path in enumerate(batch.fiber_paths):
        summary_lines.append(f"sample_{i}_fiber_path={path}")
    for i, (cp_index, cp_xyz) in enumerate(_batch_control_points(batch)):
        cp_zyx = cp_xyz[[2, 1, 0]]
        summary_lines.append(
            f"sample_{i}_cp_index={cp_index} "
            f"cp_xyz=({cp_xyz[0]:.3f}, {cp_xyz[1]:.3f}, {cp_xyz[2]:.3f}) "
            f"cp_zyx=({cp_zyx[0]:.3f}, {cp_zyx[1]:.3f}, {cp_zyx[2]:.3f})"
        )
    (out / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(
        f"exported {len(image_files)} strip jpgs, {len(mask_files)} strip mask jpgs, "
        f"contact sheet, and summary.txt to {out}"
    )


def _export_augment_contact_sheet(loader: FiberStrip2DLoader, sample_index: int, output_dir: str | Path) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(loader.config.augment.device)

    lower_entries, upper_entries = limit_augmentation_rows(loader.config.augment, sample_index)
    combined_entries = [
        (f"combined_{i:02d}", random_combined_augmentation(loader.config.augment, sample_index, i))
        for i in range(len(lower_entries))
    ]
    entries = lower_entries + upper_entries + combined_entries

    contact_images: list[np.ndarray] = []
    contact_labels: list[str] = []
    first_sample = None
    timing_rows: list[tuple[str, dict[str, float]]] = []
    timing_totals: dict[str, float] = {}
    image_stat_rows: list[tuple[str, dict[str, float]]] = []
    source_profile: dict[str, float] = {}
    with _Timer() as source_timer:
        source = loader.build_augmented_center_strip_source(
            sample_index,
            device=device,
            profile=source_profile,
        )
    summary_lines = [
        f"sample_index={sample_index}",
        f"device={device}",
        "layout=row 1: lower limits; row 2: upper limits; row 3: random combined training-style augmentations",
    ]
    for entry_index, (name, params) in enumerate(entries):
        timing: dict[str, float] = dict(source_profile) if entry_index == 0 else {}
        with _Timer() as total_timer:
            with _Timer() as loader_timer:
                sample, aug_image, aug_valid, line_xy = loader.build_augmented_center_strip_patch(
                    sample_index,
                    params,
                    device=device,
                    profile=timing,
                    source=source,
                )
            timing["loader_total"] = loader_timer.elapsed_ms
            with _Timer() as to_u8_timer:
                image_u8 = _to_u8_image(aug_image, aug_valid)
            timing["to_u8"] = to_u8_timer.elapsed_ms
            image_stat_rows.append((name, _image_stats(aug_image, aug_valid, image_u8)))
            with _Timer() as overlay_timer:
                overlay = overlay_line_coords_rgb(image_u8, line_xy, opacity=0.5, thickness=1)
                overlay = _draw_cp_crosshair(overlay, sample.control_point_xy)
            timing["overlay"] = overlay_timer.elapsed_ms
        timing["total"] = total_timer.elapsed_ms
        if entry_index == 0:
            timing["loader_total"] += source_timer.elapsed_ms
            timing["total"] += source_timer.elapsed_ms
        timing_rows.append((name, timing))
        for key, value in timing.items():
            _add_timing(timing_totals, key, value)
        if first_sample is None:
            first_sample = sample
        contact_images.append(overlay)
        contact_labels.append(name)
        cp_xy = np.asarray(sample.control_point_xy, dtype=np.float32)
        summary_lines.append(f"{name}: {params} cp_xy=({cp_xy[0]:.3f}, {cp_xy[1]:.3f})")
    if first_sample is not None:
        summary_lines.insert(2, f"fiber_path={first_sample.fiber_path}")
        summary_lines.insert(3, f"control_point_index={first_sample.control_point_index}")
        summary_lines.insert(4, f"strip_z_offset={first_sample.strip_z_offset}")

    with _Timer() as write_timer:
        _write_contact_sheet(
            out / "augment_contact_sheet.jpg",
            contact_images,
            columns=len(lower_entries),
            labels=contact_labels,
        )
    timing_totals["write_contact_sheet"] = write_timer.elapsed_ms
    with _Timer() as summary_timer:
        (out / "augment_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    timing_totals["write_summary"] = summary_timer.elapsed_ms
    _print_timing_table(timing_rows, timing_totals)
    _print_image_stats(image_stat_rows)
    print(
        "fiber_trace_2d output timings: "
        f"write_contact_sheet={timing_totals['write_contact_sheet']:.1f}ms "
        f"write_summary={timing_totals['write_summary']:.1f}ms"
    )
    print(f"exported augment_contact_sheet.jpg and augment_summary.txt to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load or prefetch 2D fiber-strip batches.")
    parser.add_argument("config", help="Path to Vesuvius-style JSON loader config")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--prefetch-samples", type=int, default=None)
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--augment-vis", action="store_true")
    parser.add_argument("--line-trace-vis", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--line-trace-step", type=float, default=4.0)
    parser.add_argument("--line-trace-rf-margin", type=float, default=None)
    args = parser.parse_args()

    with _Timer() as config_timer:
        config = load_config(args.config)
    with _Timer() as loader_timer:
        loader = FiberStrip2DLoader(config)
    print(
        "fiber_trace_2d startup timings: "
        f"load_config={config_timer.elapsed_ms:.1f}ms "
        f"construct_loader={loader_timer.elapsed_ms:.1f}ms"
    )
    batch_size = config.batch_size if args.batch_size is None else args.batch_size

    if args.prefetch:
        sample_count = batch_size if args.prefetch_samples is None else args.prefetch_samples
        summary = loader.prefetch(args.sample_index, sample_count)
        print(
            "prefetch summary: "
            f"generated={summary['generated']} missing={summary['missing']} "
            f"downloaded={summary['downloaded']} bytes={summary['bytes']} "
            f"errors={summary['errors']} workers={summary['workers']}"
        )
        return

    if args.augment_vis:
        if args.export_dir is None:
            raise SystemExit("--augment-vis requires --export-dir")
        _export_augment_contact_sheet(loader, args.sample_index, args.export_dir)
        return

    if args.line_trace_vis:
        if args.export_dir is None:
            raise SystemExit("--line-trace-vis requires --export-dir")
        if args.checkpoint is None:
            raise SystemExit("--line-trace-vis requires --checkpoint")
        _export_line_trace_vis(
            loader,
            args.sample_index,
            args.export_dir,
            checkpoint_path=args.checkpoint,
            step_px=args.line_trace_step,
            rf_margin_px=args.line_trace_rf_margin,
        )
        return

    if not args.skip_prefetch:
        loader.prefetch(args.sample_index, batch_size)

    batch = loader.load_batch(args.sample_index, batch_size=batch_size)
    print(f"images shape={batch.images.shape} dtype={batch.images.dtype}")
    print(f"coords_zyx shape={batch.coords_zyx.shape} dtype={batch.coords_zyx.dtype}")
    print(f"valid_mask shape={batch.valid_mask.shape} dtype={batch.valid_mask.dtype}")
    print(f"strip_z_offsets={batch.strip_z_offsets.tolist()}")
    print(f"record_indices={batch.record_indices.tolist()}")
    print(f"control_point_indices={batch.control_point_indices.tolist()}")
    for i, path in enumerate(batch.fiber_paths):
        print(f"sample {i}: fiber_path={path}")
    for i, (cp_index, cp_xyz) in enumerate(_batch_control_points(batch)):
        cp_zyx = cp_xyz[[2, 1, 0]]
        print(
            f"sample {i}: cp_index={cp_index} "
            f"cp_xyz=({cp_xyz[0]:.3f}, {cp_xyz[1]:.3f}, {cp_xyz[2]:.3f}) "
            f"cp_zyx=({cp_zyx[0]:.3f}, {cp_zyx[1]:.3f}, {cp_zyx[2]:.3f})"
        )
    stats = batch.cache_stats
    if stats is not None:
        print(
            "cache: "
            f"hits={getattr(stats, 'cache_hits', 0)} "
            f"downloads={getattr(stats, 'downloads', 0)} "
            f"missing={getattr(stats, 'missing', 0)}"
        )
    if args.export_dir:
        _export_batch(batch, args.export_dir)


if __name__ == "__main__":
    main()
