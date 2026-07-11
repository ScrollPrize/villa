from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace_2d.augmentation import (
    FiberStripAugmentParams,
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


def _direction_field_overlay_rgb(
    image: np.ndarray,
    valid_mask: np.ndarray,
    direction_xy: np.ndarray,
    *,
    scale: int = 2,
    stride: int = 2,
    color_rgba: tuple[int, int, int, int] = (32, 255, 255, 220),
) -> tuple[np.ndarray, int]:
    from PIL import Image, ImageDraw

    base = np.asarray(image, dtype=np.uint8)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=-1)
    valid = np.asarray(valid_mask, dtype=bool)
    field = np.asarray(direction_xy, dtype=np.float32)
    if valid.shape != base.shape[:2]:
        raise ValueError("valid_mask must match image shape")
    if field.shape != (*base.shape[:2], 2):
        raise ValueError("direction_xy must have shape H,W,2 matching image")
    display_scale = max(1, int(scale))
    sample_stride = max(1, int(stride))
    scaled = np.repeat(np.repeat(base, display_scale, axis=0), display_scale, axis=1)
    pil = Image.fromarray(scaled, mode="RGB")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    half_len = max(1.0, float(display_scale * sample_stride) * 0.375)
    drawn = 0
    height, width = valid.shape
    for y in range(0, height, sample_stride):
        for x in range(0, width, sample_stride):
            if not bool(valid[y, x]):
                continue
            direction = field[y, x]
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm <= 1.0e-6:
                continue
            unit = direction / np.float32(norm)
            cx = (float(x) + 0.5) * float(display_scale)
            cy = (float(y) + 0.5) * float(display_scale)
            dx = float(unit[0]) * half_len
            dy = float(unit[1]) * half_len
            draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=color_rgba, width=1)
            drawn += 1
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8), drawn


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


def _trace_margin_reason(point_xy: np.ndarray, shape_hw: tuple[int, int], margin: float) -> str:
    x, y = float(point_xy[0]), float(point_xy[1])
    height, width = (int(v) for v in shape_hw)
    m = max(0.0, float(margin))
    outside_x = x < m or x > (float(width - 1) - m)
    outside_y = y < m or y > (float(height - 1) - m)
    if outside_x and outside_y:
        return "rf_margin_xy"
    if outside_x:
        return "rf_margin_x"
    if outside_y:
        return "rf_margin_y"
    return "rf_margin"


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


@dataclass(frozen=True)
class _TtaDirectionField:
    name: str
    direction_xy: np.ndarray
    valid_mask: np.ndarray
    source_xy_grid: np.ndarray
    reference_to_tta_xy_grid: np.ndarray


@dataclass(frozen=True)
class _Trace2CpResult:
    score: float
    raw_y_error_px: float
    trace_y_at_target_x: float
    target_x: float
    reached_target_column: bool
    reason: str


@dataclass(frozen=True)
class _Trace2CpDirectionResult:
    trace_xy: np.ndarray
    result: _Trace2CpResult


@dataclass(frozen=True)
class _Trace2CpBidirectionalResult:
    forward: _Trace2CpDirectionResult
    reverse: _Trace2CpDirectionResult

    @property
    def score(self) -> float:
        return 0.5 * (float(self.forward.result.score) + float(self.reverse.result.score))

    @property
    def raw_y_error_px(self) -> float:
        return 0.5 * (
            float(self.forward.result.raw_y_error_px) + float(self.reverse.result.raw_y_error_px)
        )


def _geometric_only_params(params: FiberStripAugmentParams) -> FiberStripAugmentParams:
    return FiberStripAugmentParams(
        shift_x=params.shift_x,
        shift_y=params.shift_y,
        rotation_degrees=params.rotation_degrees,
        shear_x=params.shear_x,
        shear_y=params.shear_y,
        scale=params.scale,
        smooth_offset=params.smooth_offset,
        smooth_offset_stride=params.smooth_offset_stride,
        smooth_offset_seed=params.smooth_offset_seed,
        flip_x=params.flip_x,
        flip_y=params.flip_y,
        noise_seed=params.noise_seed,
    )


def _trace2cp_tta_params(params: FiberStripAugmentParams) -> FiberStripAugmentParams:
    return FiberStripAugmentParams(
        shift_x=params.shift_x,
        shift_y=0.0,
        rotation_degrees=params.rotation_degrees,
        shear_x=params.shear_x,
        shear_y=params.shear_y,
        scale=1.0,
        smooth_offset=params.smooth_offset,
        smooth_offset_stride=params.smooth_offset_stride,
        smooth_offset_seed=params.smooth_offset_seed,
        flip_x=params.flip_x,
        flip_y=params.flip_y,
        noise_seed=params.noise_seed,
    )


def _orient_direction_to_previous(direction_xy: np.ndarray, previous_xy: np.ndarray) -> np.ndarray | None:
    direction = np.asarray(direction_xy, dtype=np.float32)
    previous = np.asarray(previous_xy, dtype=np.float32)
    if direction.shape != (2,) or previous.shape != (2,):
        raise ValueError("direction_xy and previous_xy must have shape (2,)")
    norm = float(np.linalg.norm(direction))
    previous_norm = float(np.linalg.norm(previous))
    if not np.isfinite(norm) or norm <= 1.0e-6 or not np.isfinite(previous_norm) or previous_norm <= 1.0e-6:
        return None
    unit = (direction / norm).astype(np.float32)
    previous_unit = (previous / previous_norm).astype(np.float32)
    if float(np.dot(unit, previous_unit)) < 0.0:
        unit = -unit
    if float(np.dot(unit, previous_unit)) < 0.0:
        return None
    return unit.astype(np.float32)


def _reference_point_to_tta(reference_to_tta_xy_grid: np.ndarray, point_xy: np.ndarray) -> np.ndarray | None:
    grid = np.asarray(reference_to_tta_xy_grid, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)
    if grid.ndim != 3 or grid.shape[2] != 2 or point.shape != (2,):
        raise ValueError("reference_to_tta_xy_grid must have shape H,W,2 and point_xy must have shape (2,)")
    height, width = grid.shape[:2]
    x = float(point[0])
    y = float(point[1])
    if not np.isfinite(x) or not np.isfinite(y) or x < 0.0 or y < 0.0 or x > float(width - 1) or y > float(height - 1):
        return None
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    corners = grid[[y0, y0, y1, y1], [x0, x1, x0, x1]]
    if not bool(np.isfinite(corners).all()):
        return None
    tx = np.float32(x - float(x0))
    ty = np.float32(y - float(y0))
    top = grid[y0, x0] * (1.0 - tx) + grid[y0, x1] * tx
    bottom = grid[y1, x0] * (1.0 - tx) + grid[y1, x1] * tx
    mapped = top * (1.0 - ty) + bottom * ty
    if not bool(np.isfinite(mapped).all()):
        return None
    return mapped.astype(np.float32)


def _source_grid_direction_to_reference(source_xy_grid: np.ndarray, point_xy: np.ndarray, direction_xy: np.ndarray) -> np.ndarray | None:
    source = np.asarray(source_xy_grid, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)
    direction = np.asarray(direction_xy, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 2 or point.shape != (2,) or direction.shape != (2,):
        raise ValueError("source_xy_grid must have shape H,W,2 and point/direction must have shape (2,)")
    height, width = source.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    x0 = max(0, x - 1)
    x1 = min(width - 1, x + 1)
    y0 = max(0, y - 1)
    y1 = min(height - 1, y + 1)
    if x0 == x1 or y0 == y1:
        return None
    local = source[[y, y, y0, y1], [x0, x1, x, x]]
    if not bool(np.isfinite(local).all()):
        return None
    dsource_dx = (source[y, x1] - source[y, x0]) / np.float32(x1 - x0)
    dsource_dy = (source[y1, x] - source[y0, x]) / np.float32(y1 - y0)
    transformed = dsource_dx * direction[0] + dsource_dy * direction[1]
    norm = float(np.linalg.norm(transformed))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        return None
    return (transformed / norm).astype(np.float32)


def _source_grid_point_to_reference(source_xy_grid: np.ndarray, point_xy: np.ndarray) -> np.ndarray | None:
    source = np.asarray(source_xy_grid, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 2 or point.shape != (2,):
        raise ValueError("source_xy_grid must have shape H,W,2 and point_xy must have shape (2,)")
    height, width = source.shape[:2]
    x = float(point[0])
    y = float(point[1])
    if not np.isfinite(x) or not np.isfinite(y) or x < 0.0 or y < 0.0 or x > float(width - 1) or y > float(height - 1):
        return None
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    if not bool(np.isfinite(source[[y0, y0, y1, y1], [x0, x1, x0, x1]]).all()):
        return None
    tx = np.float32(x - float(x0))
    ty = np.float32(y - float(y0))
    top = source[y0, x0] * (1.0 - tx) + source[y0, x1] * tx
    bottom = source[y1, x0] * (1.0 - tx) + source[y1, x1] * tx
    return (top * (1.0 - ty) + bottom * ty).astype(np.float32)


def _reference_direction_to_source_grid_direction(
    source_xy_grid: np.ndarray,
    point_xy: np.ndarray,
    reference_direction_xy: np.ndarray,
) -> np.ndarray | None:
    source = np.asarray(source_xy_grid, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)
    direction = np.asarray(reference_direction_xy, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 2 or point.shape != (2,) or direction.shape != (2,):
        raise ValueError("source_xy_grid must have shape H,W,2 and point/direction must have shape (2,)")
    height, width = source.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    x0 = max(0, x - 1)
    x1 = min(width - 1, x + 1)
    y0 = max(0, y - 1)
    y1 = min(height - 1, y + 1)
    if x0 == x1 or y0 == y1:
        return None
    local = source[[y, y, y0, y1], [x0, x1, x, x]]
    if not bool(np.isfinite(local).all()):
        return None
    dsource_dx = (source[y, x1] - source[y, x0]) / np.float32(x1 - x0)
    dsource_dy = (source[y1, x] - source[y0, x]) / np.float32(y1 - y0)
    jacobian = np.stack([dsource_dx, dsource_dy], axis=1).astype(np.float32)
    transformed = np.linalg.pinv(jacobian).astype(np.float32) @ direction
    norm = float(np.linalg.norm(transformed))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        return None
    return (transformed / norm).astype(np.float32)


def _source_grid_points_to_reference(source_xy_grid: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape N,2")
    mapped = [_source_grid_point_to_reference(source_xy_grid, point) for point in points]
    finite = [point for point in mapped if point is not None]
    if not finite:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(finite, axis=0).astype(np.float32)


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
        hidden_channels=max(1, int(training.get("model_hidden_channels", 64))),
        depth=max(1, int(training.get("model_depth", 10))),
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


def _predict_direction_field(
    model: FiberStripDirectionNet,
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        input_image = _prepare_model_image(image, valid_mask, device=device)
        encoded = model(input_image)[0].permute(1, 2, 0)
        return decode_lasagna_direction_xy(encoded).detach().cpu().numpy().astype(np.float32)


def _identity_source_xy_grid(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = (int(v) for v in shape_hw)
    yy, xx = np.indices((height, width), dtype=np.float32)
    return np.stack([xx, yy], axis=-1).astype(np.float32)


def _sample_tta_direction_in_reference(
    fields: list[_TtaDirectionField],
    point_xy: np.ndarray,
    previous_xy: np.ndarray,
) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for field in fields:
        tta_point = _reference_point_to_tta(field.reference_to_tta_xy_grid, point_xy)
        if tta_point is None:
            continue
        sampled = _bilinear_direction_sample(field.direction_xy, tta_point, valid_mask=field.valid_mask)
        if sampled is None:
            continue
        sampled_ref = _source_grid_direction_to_reference(field.source_xy_grid, tta_point, sampled)
        if sampled_ref is None:
            continue
        oriented = _orient_direction_to_previous(sampled_ref, previous_xy)
        if oriented is not None:
            candidates.append(oriented)
    if not candidates:
        return None
    median = np.median(np.stack(candidates, axis=0), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(median))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        return None
    return (median / norm).astype(np.float32)


def _trace_median_tta_direction_line(
    fields: list[_TtaDirectionField],
    cp_xy: np.ndarray,
    tangent_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
    max_steps: int | None = None,
) -> np.ndarray:
    if not fields:
        raise ValueError("fields must contain at least the reference direction field")
    cp = np.asarray(cp_xy, dtype=np.float32)
    tangent = np.asarray(tangent_xy, dtype=np.float32)
    if cp.shape != (2,) or tangent.shape != (2,):
        raise ValueError("cp_xy and tangent_xy must have shape (2,)")
    tangent_norm = float(np.linalg.norm(tangent))
    if not np.isfinite(tangent_norm) or tangent_norm <= 1.0e-6:
        raise ValueError("tangent_xy must be finite and non-zero")
    step = max(float(step_px), 1.0e-3)
    unit_tangent = (tangent / tangent_norm).astype(np.float32)
    if max_steps is None:
        max_steps = int(np.ceil(np.hypot(*shape_hw) / step)) + 2

    def one_side(initial: np.ndarray) -> list[np.ndarray]:
        points = [cp.astype(np.float32)]
        previous = initial.astype(np.float32)
        current = cp.astype(np.float32)
        for _ in range(int(max_steps)):
            if not _inside_trace_margin(current, shape_hw, rf_margin_px):
                break
            sampled = _sample_tta_direction_in_reference(fields, current, previous)
            if sampled is None:
                break
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


def _trace_median_tta_direction_line_to_target(
    fields: list[_TtaDirectionField],
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
    max_steps: int | None = None,
) -> tuple[np.ndarray, str]:
    if not fields:
        raise ValueError("fields must contain at least the reference direction field")
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("start_xy and target_xy must have shape (2,)")
    delta = target - start
    delta_norm = float(np.linalg.norm(delta))
    if not np.isfinite(delta_norm) or delta_norm <= 1.0e-6:
        raise ValueError("trace2cp start and target points must be distinct")
    step = max(float(step_px), 1.0e-3)
    if max_steps is None:
        max_steps = int(np.ceil(np.hypot(*shape_hw) / step)) + 2
    current = start.astype(np.float32)
    previous = (delta / delta_norm).astype(np.float32)
    points = [current.copy()]
    target_x = float(target[0])
    for _ in range(int(max_steps)):
        if not _inside_trace_margin(current, shape_hw, rf_margin_px):
            return np.stack(points, axis=0).astype(np.float32), _trace_margin_reason(
                current, shape_hw, rf_margin_px
            )
        sampled = _sample_tta_direction_in_reference(fields, current, previous)
        if sampled is None:
            return np.stack(points, axis=0).astype(np.float32), "invalid_direction"
        next_point = current + sampled * np.float32(step)
        previous_x = float(current[0])
        next_x = float(next_point[0])
        crosses_target = (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0
        if crosses_target:
            points.append(next_point.astype(np.float32))
            return np.stack(points, axis=0).astype(np.float32), "target_column"
        if not _inside_trace_margin(next_point, shape_hw, rf_margin_px):
            return np.stack(points, axis=0).astype(np.float32), _trace_margin_reason(
                next_point, shape_hw, rf_margin_px
            )
        points.append(next_point.astype(np.float32))
        current = next_point.astype(np.float32)
        previous = sampled.astype(np.float32)
    return np.stack(points, axis=0).astype(np.float32), "max_steps"


def _trace_direction_line_to_target(
    direction_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
    max_steps: int | None = None,
) -> tuple[np.ndarray, str]:
    field = np.asarray(direction_xy, dtype=np.float32)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("direction_xy must have shape H,W,2")
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("start_xy and target_xy must have shape (2,)")
    target_delta = target - start
    target_norm = float(np.linalg.norm(target_delta))
    if not np.isfinite(target_norm) or target_norm <= 1.0e-6:
        raise ValueError("trace2cp start and target points must be distinct")
    step = max(float(step_px), 1.0e-3)
    shape_hw = (int(field.shape[0]), int(field.shape[1]))
    if max_steps is None:
        max_steps = int(np.ceil(np.hypot(*shape_hw) / step)) + 2
    target_x = float(target[0])
    previous = (target_delta / target_norm).astype(np.float32)
    current = start.astype(np.float32)
    points = [current.copy()]

    for _ in range(int(max_steps)):
        if not _inside_trace_margin(current, shape_hw, rf_margin_px):
            return np.stack(points, axis=0).astype(np.float32), _trace_margin_reason(
                current, shape_hw, rf_margin_px
            )
        sampled = _bilinear_direction_sample(field, current, valid_mask=valid_mask)
        if sampled is None:
            return np.stack(points, axis=0).astype(np.float32), "invalid_direction"
        if float(np.dot(sampled, previous)) < 0.0:
            sampled = -sampled
        next_point = current + sampled * np.float32(step)
        previous_x = float(current[0])
        next_x = float(next_point[0])
        crosses_target = (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0
        if crosses_target:
            points.append(next_point.astype(np.float32))
            return np.stack(points, axis=0).astype(np.float32), "target_column"
        if not _inside_trace_margin(next_point, shape_hw, rf_margin_px):
            return np.stack(points, axis=0).astype(np.float32), _trace_margin_reason(
                next_point, shape_hw, rf_margin_px
            )
        points.append(next_point.astype(np.float32))
        current = next_point.astype(np.float32)
        previous = sampled.astype(np.float32)
    return np.stack(points, axis=0).astype(np.float32), "max_steps"


def _trace_y_at_x(trace_xy: np.ndarray, target_x: float) -> float | None:
    trace = np.asarray(trace_xy, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 2 or trace.shape[0] == 0:
        raise ValueError("trace_xy must have shape N,2")
    x_target = float(target_x)
    for p0, p1 in zip(trace[:-1], trace[1:]):
        x0 = float(p0[0])
        x1 = float(p1[0])
        if (x0 - x_target) == 0.0:
            return float(p0[1])
        if (x0 - x_target) * (x1 - x_target) <= 0.0 and x0 != x1:
            alpha = (x_target - x0) / (x1 - x0)
            return float(p0[1] + np.float32(alpha) * (p1[1] - p0[1]))
    if float(trace[-1, 0]) == x_target:
        return float(trace[-1, 1])
    return None


def _score_trace2cp(
    trace_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    rf_margin_px: float,
    termination_reason: str,
) -> _Trace2CpResult:
    target = np.asarray(target_xy, dtype=np.float32)
    if target.shape != (2,):
        raise ValueError("target_xy must have shape (2,)")
    height, _ = (int(v) for v in shape_hw)
    margin = max(0.0, float(rf_margin_px))
    top = margin
    bottom = float(height - 1) - margin
    denom = min(float(target[1]) - top, bottom - float(target[1]))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(
            "trace2cp target has no usable vertical edge-distance denominator: "
            f"target_y={float(target[1]):.3f} height={height} rf_margin_px={margin:.3f}"
        )
    y_at_target = _trace_y_at_x(trace_xy, float(target[0]))
    if y_at_target is None:
        return _Trace2CpResult(
            score=1.0,
            raw_y_error_px=denom,
            trace_y_at_target_x=float("nan"),
            target_x=float(target[0]),
            reached_target_column=False,
            reason=termination_reason,
        )
    raw_error = abs(float(y_at_target) - float(target[1]))
    return _Trace2CpResult(
        score=float(np.clip(raw_error / denom, 0.0, 1.0)),
        raw_y_error_px=float(raw_error),
        trace_y_at_target_x=float(y_at_target),
        target_x=float(target[0]),
        reached_target_column=True,
        reason=termination_reason,
    )


def _trace_score_trace2cp_direction(
    direction_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> _Trace2CpDirectionResult:
    traced_line, reason = _trace_direction_line_to_target(
        direction_xy,
        start_xy,
        target_xy,
        valid_mask=valid_mask,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    result = _score_trace2cp(
        traced_line,
        target_xy,
        shape_hw=(int(direction_xy.shape[0]), int(direction_xy.shape[1])),
        rf_margin_px=rf_margin_px,
        termination_reason=reason,
    )
    return _Trace2CpDirectionResult(trace_xy=traced_line, result=result)


def _trace_score_trace2cp_bidirectional(
    direction_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> _Trace2CpBidirectionalResult:
    forward = _trace_score_trace2cp_direction(
        direction_xy,
        start_xy,
        target_xy,
        valid_mask=valid_mask,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    reverse = _trace_score_trace2cp_direction(
        direction_xy,
        target_xy,
        start_xy,
        valid_mask=valid_mask,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    return _Trace2CpBidirectionalResult(forward=forward, reverse=reverse)


def _trace_score_trace2cp_median_tta_direction(
    fields: list[_TtaDirectionField],
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> _Trace2CpDirectionResult:
    traced_line, reason = _trace_median_tta_direction_line_to_target(
        fields,
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    result = _score_trace2cp(
        traced_line,
        target_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
        termination_reason=reason,
    )
    return _Trace2CpDirectionResult(trace_xy=traced_line, result=result)


def _trace_score_trace2cp_median_tta_bidirectional(
    fields: list[_TtaDirectionField],
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> _Trace2CpBidirectionalResult:
    forward = _trace_score_trace2cp_median_tta_direction(
        fields,
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    reverse = _trace_score_trace2cp_median_tta_direction(
        fields,
        target_xy,
        start_xy,
        shape_hw=shape_hw,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    return _Trace2CpBidirectionalResult(forward=forward, reverse=reverse)


def _export_line_trace_vis(
    loader: FiberStrip2DLoader,
    sample_index: int,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    step_px: float,
    rf_margin_px: float | None,
    med_tta: bool = False,
    line_trace_tta_count: int = 100,
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
    configured_depth = max(1, int(_model_config_from_checkpoint(checkpoint, loader).depth))
    default_margin = configured_depth
    margin = float(default_margin if rf_margin_px is None else rf_margin_px)
    direction_xy = _predict_direction_field(model, image, valid_mask, device=device)
    traced_line = _trace_direction_line(
        direction_xy,
        cp_xy,
        tangent_xy,
        valid_mask=valid_mask,
        step_px=step_px,
        rf_margin_px=margin,
    )
    image_u8 = _to_u8_image(image, valid_mask)
    single_overlay = overlay_line_coords_rgb(image_u8, sample.line_xy, opacity=0.5, thickness=1)
    single_overlay = _overlay_polyline_rgb(single_overlay, traced_line, color_rgba=(0, 255, 0, 230), thickness=1)
    single_overlay = _draw_cp_crosshair(single_overlay, cp_xy)

    flock_overlay = overlay_line_coords_rgb(image_u8, sample.line_xy, opacity=0.35, thickness=1)
    flock_overlay = _overlay_polyline_rgb(flock_overlay, traced_line, color_rgba=(0, 255, 0, 220), thickness=1)
    tta_rows: list[str] = []
    random_tta_fields: list[_TtaDirectionField] = []
    tta_count = max(0, int(line_trace_tta_count))
    for variant_index in range(tta_count):
        params = _geometric_only_params(random_combined_augmentation(loader.config.augment, sample_index, variant_index))
        tta_name = f"random_{variant_index:03d}"
        try:
            tta_patch = loader.build_center_tta_patch_from_sample(
                sample,
                params,
                rf_margin_px=margin,
                device=device,
            )
            tta_image = tta_patch.image
            tta_valid = tta_patch.valid_mask
            source_xy_grid = tta_patch.source_xy_grid
            reference_to_tta_xy_grid = tta_patch.reference_to_tta_xy_grid
            tta_sample = tta_patch.sample
            tta_cp_tangent = line_cp_and_tangent_xy(tta_sample.line_xy, tta_sample.control_point_xy)
            if tta_cp_tangent is None:
                raise ValueError("could not derive transformed CP/tangent in TTA patch")
            tta_cp, tta_tangent = tta_cp_tangent
            if _reference_point_to_tta(reference_to_tta_xy_grid, cp_xy) is None:
                raise ValueError("could not invert CP into TTA patch")
            tta_direction = _predict_direction_field(model, tta_image, tta_valid, device=device)
            field = _TtaDirectionField(
                name=tta_name,
                direction_xy=tta_direction,
                valid_mask=tta_valid,
                source_xy_grid=source_xy_grid,
                reference_to_tta_xy_grid=reference_to_tta_xy_grid,
            )
            random_tta_fields.append(field)
            tta_trace = _trace_direction_line(
                tta_direction,
                tta_cp,
                tta_tangent,
                valid_mask=tta_valid,
                step_px=step_px,
                rf_margin_px=margin,
            )
            trace_back = _source_grid_points_to_reference(source_xy_grid, tta_trace)
            flock_overlay = _overlay_polyline_rgb(
                flock_overlay,
                trace_back,
                color_rgba=(64, 224, 255, 120),
                thickness=1,
            )
            tta_rows.append(f"{tta_name}: points={int(trace_back.shape[0])}")
        except (ValueError, np.linalg.LinAlgError) as exc:
            tta_rows.append(f"{tta_name}: skipped={exc}")
    flock_overlay = _draw_cp_crosshair(flock_overlay, cp_xy)

    overlays = [single_overlay, flock_overlay]
    med_trace: np.ndarray | None = None
    med_tta_fields_count = 0
    if med_tta:
        med_fields = [
            _TtaDirectionField(
                name="reference",
                direction_xy=direction_xy,
                valid_mask=np.asarray(valid_mask, dtype=bool),
                source_xy_grid=_identity_source_xy_grid(image.shape),
                reference_to_tta_xy_grid=_identity_source_xy_grid(image.shape),
            ),
        ]
        med_fields.extend(random_tta_fields)
        med_tta_fields_count = len(med_fields)
        med_trace = _trace_median_tta_direction_line(
            med_fields,
            cp_xy,
            tangent_xy,
            shape_hw=image.shape,
            step_px=step_px,
            rf_margin_px=margin,
        )
        med_overlay = overlay_line_coords_rgb(image_u8, sample.line_xy, opacity=0.35, thickness=1)
        med_overlay = _overlay_polyline_rgb(med_overlay, med_trace, color_rgba=(255, 220, 64, 230), thickness=1)
        med_overlay = _draw_cp_crosshair(med_overlay, cp_xy)
        overlays.append(med_overlay)

    combined = np.zeros((image_u8.shape[0], image_u8.shape[1] * len(overlays), 3), dtype=np.uint8)
    combined[:, : image_u8.shape[1], :] = single_overlay
    for index, overlay in enumerate(overlays[1:], start=1):
        x0 = image_u8.shape[1] * index
        combined[:, x0 : x0 + image_u8.shape[1], :] = overlay
    _write_jpg(out / "line_trace_vis.jpg", combined)
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
        f"med_tta={bool(med_tta)}",
        f"line_trace_tta_count={int(line_trace_tta_count)}",
        f"med_tta_fields={int(med_tta_fields_count)}",
        f"med_tta_trace_points={int(med_trace.shape[0]) if med_trace is not None else 0}",
        "tta_traces:",
        *tta_rows,
    ]
    (out / "line_trace_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"exported line_trace_vis.jpg and line_trace_summary.txt to {out}")


def _draw_trace2cp_overlay(
    image_u8: np.ndarray,
    *,
    line_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    bidirectional_result: _Trace2CpBidirectionalResult,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    overlay = overlay_line_coords_rgb(image_u8, line_xy, opacity=0.35, thickness=1)
    overlay = _overlay_polyline_rgb(
        overlay,
        bidirectional_result.forward.trace_xy,
        color_rgba=(0, 255, 0, 230),
        thickness=1,
    )
    overlay = _overlay_polyline_rgb(
        overlay,
        bidirectional_result.reverse.trace_xy,
        color_rgba=(255, 64, 220, 230),
        thickness=1,
    )
    arr = np.asarray(overlay, dtype=np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(pil, mode="RGBA")
    height, width = arr.shape[:2]

    sx, sy = (float(v) for v in np.asarray(start_xy, dtype=np.float32))
    tx, ty = (float(v) for v in np.asarray(target_xy, dtype=np.float32))
    if np.isfinite([sx, sy]).all():
        draw.line((sx, 0.0, sx, float(height - 1)), fill=(32, 255, 255, 100), width=1)
        draw.ellipse((sx - 2.0, sy - 2.0, sx + 2.0, sy + 2.0), outline=(32, 255, 255, 255), width=1)
    if np.isfinite([tx, ty]).all():
        draw.line((tx, 0.0, tx, float(height - 1)), fill=(255, 220, 64, 130), width=1)
        gap = 2.0
        arm = 6.0
        draw.line((tx - arm, ty, tx - gap, ty), fill=(255, 220, 64, 255), width=1)
        draw.line((tx + gap, ty, tx + arm, ty), fill=(255, 220, 64, 255), width=1)
        draw.line((tx, ty - arm, tx, ty - gap), fill=(255, 220, 64, 255), width=1)
        draw.line((tx, ty + gap, tx, ty + arm), fill=(255, 220, 64, 255), width=1)

    forward = bidirectional_result.forward.result
    reverse = bidirectional_result.reverse.result
    label = (
        f"score={bidirectional_result.score:.4f} "
        f"f={forward.score:.4f} r={reverse.score:.4f} "
        f"hit={int(forward.reached_target_column)}/{int(reverse.reached_target_column)}"
    )
    return _draw_label_band(np.asarray(pil, dtype=np.uint8), label)


def _base_corners_xy(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = (int(v) for v in shape_hw)
    return np.asarray(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )


def _draw_trace2cp_tta_slice(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    base_corners_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    label: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    arr = _to_u8_image(image, valid_mask)
    rgb = np.repeat(arr[..., None], 3, axis=-1)
    pil = Image.fromarray(rgb, mode="RGB")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    corners = np.asarray(base_corners_xy, dtype=np.float32)
    if corners.shape == (4, 2) and bool(np.isfinite(corners).all()):
        points = [(float(x), float(y)) for x, y in corners.tolist()]
        draw.line(points + [points[0]], fill=(255, 220, 64, 230), width=1)
    for point_xy, color in (
        (start_xy, (32, 255, 255, 255)),
        (target_xy, (255, 64, 220, 255)),
    ):
        point = np.asarray(point_xy, dtype=np.float32)
        if point.shape == (2,) and bool(np.isfinite(point).all()):
            x = float(point[0])
            y = float(point[1])
            draw.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), outline=color, width=1)
    return _draw_label_band(
        np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8),
        label,
    )


def _write_trace2cp_tta_debug_images(
    output_dir: Path,
    entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Path:
    tta_dir = output_dir / "trace2cp_tta"
    tta_dir.mkdir(parents=True, exist_ok=True)
    debug_images: list[np.ndarray] = []
    labels: list[str] = []
    for name, image, valid_mask, corners_xy, start_xy, target_xy in entries:
        label = f"{name} shape={int(image.shape[0])}x{int(image.shape[1])}"
        drawn = _draw_trace2cp_tta_slice(
            image,
            valid_mask,
            base_corners_xy=corners_xy,
            start_xy=start_xy,
            target_xy=target_xy,
            label=label,
        )
        _write_jpg(tta_dir / f"{name}.jpg", drawn)
        debug_images.append(drawn)
        labels.append(name)
    _write_contact_sheet(tta_dir / "contact_sheet.jpg", debug_images, columns=4, labels=labels)
    return tta_dir


def _export_trace2cp_vis(
    loader: FiberStrip2DLoader,
    sample_index: int,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    step_px: float,
    rf_margin_px: float | None,
    target_offset: int,
    target_cp_index: int | None,
    med_tta: bool = False,
    line_trace_tta_count: int = 100,
    vis_tta: bool = False,
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(loader.config.augment.device)
    model, checkpoint = _load_direction_model(checkpoint_path, loader, device=device)
    configured_depth = max(1, int(_model_config_from_checkpoint(checkpoint, loader).depth))
    default_margin = configured_depth
    margin = float(default_margin if rf_margin_px is None else rf_margin_px)
    sample, image, valid_mask = loader.build_trace2cp_segment_patch(
        sample_index,
        target_control_point_index=target_cp_index,
        target_offset=target_offset,
        rf_margin_px=margin,
        device=device,
    )
    direction_xy = _predict_direction_field(model, image, valid_mask, device=device)
    start_xy = np.asarray(sample.start_control_point_xy, dtype=np.float32)
    target_xy = np.asarray(sample.target_control_point_xy, dtype=np.float32)
    selected_result = _trace_score_trace2cp_bidirectional(
        direction_xy,
        start_xy,
        target_xy,
        valid_mask=valid_mask,
        step_px=step_px,
        rf_margin_px=margin,
    )
    tta_rows: list[str] = []
    tta_count = max(0, int(line_trace_tta_count)) if med_tta else 0
    selected_mode = "base"
    med_fields_count = 0
    tta_debug_entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    tta_debug_dir: Path | None = None
    if med_tta:
        med_fields = [
            _TtaDirectionField(
                name="reference",
                direction_xy=direction_xy,
                valid_mask=np.asarray(valid_mask, dtype=bool),
                source_xy_grid=_identity_source_xy_grid(image.shape),
                reference_to_tta_xy_grid=_identity_source_xy_grid(image.shape),
            ),
        ]
        if vis_tta:
            tta_debug_entries.append(
                (
                    "reference",
                    image,
                    valid_mask,
                    _base_corners_xy(image.shape),
                    start_xy,
                    target_xy,
                )
            )
        for variant_index in range(tta_count):
            params = _trace2cp_tta_params(
                random_combined_augmentation(loader.config.augment, sample_index, variant_index)
            )
            tta_name = f"random_{variant_index:03d}"
            try:
                tta_patch = loader.build_trace2cp_tta_patch_from_sample(
                    sample,
                    params,
                    rf_margin_px=margin,
                    device=device,
                )
                tta_image = tta_patch.image
                tta_valid = tta_patch.valid_mask
                source_xy_grid = tta_patch.source_xy_grid
                reference_to_tta_xy_grid = tta_patch.reference_to_tta_xy_grid
                tta_sample = tta_patch.sample
                tta_start_xy = np.asarray(tta_sample.start_control_point_xy, dtype=np.float32)
                tta_target_xy = np.asarray(tta_sample.target_control_point_xy, dtype=np.float32)
                tta_start = _reference_point_to_tta(reference_to_tta_xy_grid, start_xy)
                tta_target = _reference_point_to_tta(reference_to_tta_xy_grid, target_xy)
                if tta_start is None or tta_target is None:
                    raise ValueError("could not map start/target CP into TTA patch")
                med_fields.append(
                    _TtaDirectionField(
                        name=tta_name,
                        direction_xy=_predict_direction_field(model, tta_image, tta_valid, device=device),
                        valid_mask=tta_valid,
                        source_xy_grid=source_xy_grid,
                        reference_to_tta_xy_grid=reference_to_tta_xy_grid,
                    )
                )
                if vis_tta:
                    tta_debug_entries.append(
                        (
                            tta_name,
                            tta_image,
                            tta_valid,
                            tta_patch.base_corners_xy,
                            tta_start_xy,
                            tta_target_xy,
                        )
                    )
                tta_rows.append(
                    f"{tta_name}: field=ok shape_hw=({int(tta_image.shape[0])}, {int(tta_image.shape[1])})"
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                tta_rows.append(f"{tta_name}: skipped={exc}")
        med_fields_count = len(med_fields)
        selected_result = _trace_score_trace2cp_median_tta_bidirectional(
            med_fields,
            start_xy,
            target_xy,
            shape_hw=image.shape,
            step_px=step_px,
            rf_margin_px=margin,
        )
        selected_mode = "med_tta"
        if vis_tta and tta_debug_entries:
            tta_debug_dir = _write_trace2cp_tta_debug_images(out, tta_debug_entries)

    image_u8 = _to_u8_image(image, valid_mask)
    overlay = _draw_trace2cp_overlay(
        image_u8,
        line_xy=sample.line_xy,
        start_xy=start_xy,
        target_xy=target_xy,
        bidirectional_result=selected_result,
    )
    _write_jpg(out / "trace2cp_vis.jpg", overlay)
    forward = selected_result.forward.result
    reverse = selected_result.reverse.result
    trace_points = int(selected_result.forward.trace_xy.shape[0]) + int(
        selected_result.reverse.trace_xy.shape[0]
    )
    summary = [
        f"sample_index={sample_index}",
        f"checkpoint={Path(checkpoint_path).expanduser().resolve()}",
        f"checkpoint_step={checkpoint.get('step', 'unknown')}",
        f"fiber_path={sample.fiber_path}",
        f"start_control_point_index={sample.start_control_point_index}",
        f"target_control_point_index={sample.target_control_point_index}",
        f"strip_z_offset={sample.strip_z_offset}",
        f"image_shape_hw=({int(image.shape[0])}, {int(image.shape[1])})",
        f"start_cp_xy=({start_xy[0]:.3f}, {start_xy[1]:.3f})",
        f"target_cp_xy=({target_xy[0]:.3f}, {target_xy[1]:.3f})",
        f"step_px={float(step_px):.3f}",
        f"rf_margin_px={margin:.3f}",
        f"trace_mode={selected_mode}",
        f"trace_points={trace_points}",
        f"trace2cp_score={selected_result.score:.8f}",
        f"raw_y_error_px={selected_result.raw_y_error_px:.6f}",
        f"forward_trace_points={int(selected_result.forward.trace_xy.shape[0])}",
        f"forward_trace2cp_score={forward.score:.8f}",
        f"forward_raw_y_error_px={forward.raw_y_error_px:.6f}",
        f"forward_target_x={forward.target_x:.6f}",
        f"forward_trace_y_at_target_x={forward.trace_y_at_target_x:.6f}",
        f"forward_reached_target_column={forward.reached_target_column}",
        f"forward_termination_reason={forward.reason}",
        f"reverse_trace_points={int(selected_result.reverse.trace_xy.shape[0])}",
        f"reverse_trace2cp_score={reverse.score:.8f}",
        f"reverse_raw_y_error_px={reverse.raw_y_error_px:.6f}",
        f"reverse_target_x={reverse.target_x:.6f}",
        f"reverse_trace_y_at_target_x={reverse.trace_y_at_target_x:.6f}",
        f"reverse_reached_target_column={reverse.reached_target_column}",
        f"reverse_termination_reason={reverse.reason}",
        f"med_tta={bool(med_tta)}",
        f"vis_tta={bool(vis_tta)}",
        f"trace2cp_tta_debug_dir={str(tta_debug_dir) if tta_debug_dir is not None else ''}",
        f"line_trace_tta_count={tta_count}",
        f"med_tta_fields={med_fields_count}",
        "tta_fields:",
        *tta_rows,
    ]
    (out / "trace2cp_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(
        "trace2cp "
        f"sample_index={sample_index} fiber_path={sample.fiber_path} "
        f"start_cp={sample.start_control_point_index} target_cp={sample.target_control_point_index} "
        f"mode={selected_mode} score={selected_result.score:.8f} "
        f"raw_y_error_px={selected_result.raw_y_error_px:.6f} "
        f"forward_score={forward.score:.8f} reverse_score={reverse.score:.8f} "
        f"forward_reached={forward.reached_target_column} reverse_reached={reverse.reached_target_column} "
        f"forward_reason={forward.reason} reverse_reason={reverse.reason}"
    )
    print(f"exported trace2cp_vis.jpg and trace2cp_summary.txt to {out}")


def _export_dir_vis(
    loader: FiberStrip2DLoader,
    sample_index: int,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(loader.config.augment.device)
    sample, image, valid_mask = loader.build_center_strip_patch(sample_index)
    model, checkpoint = _load_direction_model(checkpoint_path, loader, device=device)
    direction_xy = _predict_direction_field(model, image, valid_mask, device=device)
    image_u8 = _to_u8_image(image, valid_mask)
    overlay, drawn = _direction_field_overlay_rgb(
        image_u8,
        valid_mask,
        direction_xy,
        scale=2,
        stride=2,
    )
    _write_jpg(out / "dir_vis.jpg", overlay)
    summary = [
        f"sample_index={sample_index}",
        f"checkpoint={Path(checkpoint_path).expanduser().resolve()}",
        f"checkpoint_step={checkpoint.get('step', 'unknown')}",
        f"fiber_path={sample.fiber_path}",
        f"control_point_index={sample.control_point_index}",
        f"strip_z_offset={sample.strip_z_offset}",
        f"image_shape_hw=({int(image.shape[0])}, {int(image.shape[1])})",
        "display_scale=2",
        "direction_stride=2",
        f"drawn_directions={drawn}",
    ]
    (out / "dir_vis_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"exported dir_vis.jpg and dir_vis_summary.txt to {out}")


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


class _NullTimer:
    elapsed_ms = 0.0

    def __enter__(self) -> "_NullTimer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _timer_if(enabled: bool) -> _Timer | _NullTimer:
    return _Timer() if enabled else _NullTimer()


def _add_timing(total: dict[str, float], key: str, value_ms: float) -> None:
    total[key] = total.get(key, 0.0) + float(value_ms)


def _timing_totals_for_rows(rows: list[tuple[str, dict[str, float]]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for _, timing in rows:
        for key, value in timing.items():
            _add_timing(totals, key, value)
    return totals


def _print_timing_table(
    rows: list[tuple[str, dict[str, float]]],
    totals: dict[str, float],
    *,
    title: str = "fiber_trace_2d augment-vis timings in ms",
) -> None:
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
    print(title)
    print(header)
    for name, timing in rows:
        print(name.ljust(16) + "".join(f"{timing.get(key, 0.0):13.1f}" for key in keys))
    print("total".ljust(16) + "".join(f"{totals.get(key, 0.0):13.1f}" for key in keys))
    row_count = len(rows)
    if row_count > 0:
        print("avg/patch".ljust(16) + "".join(f"{totals.get(key, 0.0) / row_count:13.1f}" for key in keys))
    warm_rows = rows[1:]
    warm_count = len(warm_rows)
    if warm_count > 0:
        warm_totals = _timing_totals_for_rows(warm_rows)
        print("total/no-first".ljust(16) + "".join(f"{warm_totals.get(key, 0.0):13.1f}" for key in keys))
        print(
            "avg/no-first".ljust(16)
            + "".join(f"{warm_totals.get(key, 0.0) / warm_count:13.1f}" for key in keys)
        )
    volume_stats = {
        key.removeprefix("volume_stat_"): value
        for key, value in totals.items()
        if key.startswith("volume_stat_")
    }
    if volume_stats:
        print(
            f"{title} volume sampler stats: "
            + " ".join(f"{key}={value:.0f}" for key, value in sorted(volume_stats.items()))
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


def _run_augment_contact_sheet_pass(
    loader: FiberStrip2DLoader,
    sample_index: int,
    entries: list[tuple[str, FiberStripAugmentParams]],
    *,
    device: torch.device,
    source,
    source_profile: dict[str, float] | None,
    source_elapsed_ms: float,
    collect_images: bool,
    profile: bool,
) -> tuple[list[np.ndarray], list[str], list[str], object | None, list[tuple[str, dict[str, float]]], dict[str, float]]:
    contact_images: list[np.ndarray] = []
    contact_labels: list[str] = []
    summary_lines: list[str] = []
    first_sample = None
    timing_rows: list[tuple[str, dict[str, float]]] = []
    timing_totals: dict[str, float] = {}
    for entry_index, (name, params) in enumerate(entries):
        timing: dict[str, float] | None
        timing = dict(source_profile or {}) if profile and entry_index == 0 else ({} if profile else None)
        with _timer_if(profile) as total_timer:
            with _timer_if(profile) as loader_timer:
                sample, aug_image, aug_valid, line_xy = loader.build_augmented_center_strip_patch(
                    sample_index,
                    params,
                    device=device,
                    profile=timing,
                    source=source,
                )
            if timing is not None:
                timing["loader_total"] = loader_timer.elapsed_ms
            with _timer_if(profile) as to_u8_timer:
                image_u8 = _to_u8_image(aug_image, aug_valid)
            if timing is not None:
                timing["to_u8"] = to_u8_timer.elapsed_ms
            with _timer_if(profile) as overlay_timer:
                overlay = overlay_line_coords_rgb(image_u8, line_xy, opacity=0.5, thickness=1)
                overlay = _draw_cp_crosshair(overlay, sample.control_point_xy)
            if timing is not None:
                timing["overlay"] = overlay_timer.elapsed_ms
        if timing is not None:
            timing["total"] = total_timer.elapsed_ms
            if entry_index == 0:
                timing["loader_total"] += source_elapsed_ms
                timing["total"] += source_elapsed_ms
            timing_rows.append((name, timing))
            for key, value in timing.items():
                _add_timing(timing_totals, key, value)
        if first_sample is None:
            first_sample = sample
        if collect_images:
            contact_images.append(overlay)
            contact_labels.append(name)
            cp_xy = np.asarray(sample.control_point_xy, dtype=np.float32)
            summary_lines.append(f"{name}: {params} cp_xy=({cp_xy[0]:.3f}, {cp_xy[1]:.3f})")
    return contact_images, contact_labels, summary_lines, first_sample, timing_rows, timing_totals


def _export_augment_contact_sheet(
    loader: FiberStrip2DLoader, sample_index: int, output_dir: str | Path, *, profile: bool = False
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(loader.config.augment.device)

    lower_entries, upper_entries = limit_augmentation_rows(loader.config.augment, sample_index)
    combined_entries = [
        (f"combined_{i:02d}", random_combined_augmentation(loader.config.augment, sample_index, i))
        for i in range(len(lower_entries))
    ]
    entries = lower_entries + upper_entries + combined_entries

    source_profile: dict[str, float] | None = {} if profile else None
    # Build the deterministic sample-order cache outside augment-vis profiling.
    # The first random descriptor lookup sorts the whole CP order for this pass;
    # that cost is global state setup, not per-sample patch construction.
    loader.descriptor_for_sample_index(sample_index)
    with _timer_if(profile) as source_timer:
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
    contact_images, contact_labels, entry_summary_lines, first_sample, timing_rows, timing_totals = (
        _run_augment_contact_sheet_pass(
            loader,
            sample_index,
            entries,
            device=device,
            source=source,
            source_profile=source_profile,
            source_elapsed_ms=source_timer.elapsed_ms if profile else 0.0,
            collect_images=True,
            profile=profile,
        )
    )
    summary_lines.extend(entry_summary_lines)
    warm_timing_rows: list[tuple[str, dict[str, float]]] = []
    warm_timing_totals: dict[str, float] = {}
    if profile:
        _, _, _, _, warm_timing_rows, warm_timing_totals = _run_augment_contact_sheet_pass(
            loader,
            sample_index,
            entries,
            device=device,
            source=source,
            source_profile=None,
            source_elapsed_ms=0.0,
            collect_images=False,
            profile=True,
        )
    if first_sample is not None:
        summary_lines.insert(2, f"fiber_path={first_sample.fiber_path}")
        summary_lines.insert(3, f"control_point_index={first_sample.control_point_index}")
        summary_lines.insert(4, f"strip_z_offset={first_sample.strip_z_offset}")

    with _timer_if(profile) as write_timer:
        _write_contact_sheet(
            out / "augment_contact_sheet.jpg",
            contact_images,
            columns=len(lower_entries),
            labels=contact_labels,
        )
    with _timer_if(profile) as summary_timer:
        (out / "augment_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if profile:
        timing_totals["write_contact_sheet"] = write_timer.elapsed_ms
        timing_totals["write_summary"] = summary_timer.elapsed_ms
        _print_timing_table(
            timing_rows,
            timing_totals,
            title="fiber_trace_2d augment-vis timings pass=1 cold-ish in ms",
        )
        _print_timing_table(
            warm_timing_rows,
            warm_timing_totals,
            title="fiber_trace_2d augment-vis timings pass=2 warm in ms",
        )
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
    parser.add_argument(
        "--augment-profile",
        action="store_true",
        help="When used with --augment-vis, print two augment timing passes to expose cold and warm costs",
    )
    parser.add_argument("--line-trace-vis", action="store_true")
    parser.add_argument("--trace2cp-vis", action="store_true")
    parser.add_argument("--med-tta", action="store_true")
    parser.add_argument("--vis-tta", action="store_true")
    parser.add_argument("--dir-vis", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--line-trace-step", type=float, default=4.0)
    parser.add_argument("--line-trace-tta-count", type=int, default=100)
    parser.add_argument("--med-tta-count", type=int, default=None)
    parser.add_argument("--line-trace-rf-margin", type=float, default=None)
    parser.add_argument("--trace2cp-target-offset", type=int, default=1)
    parser.add_argument("--trace2cp-target-cp-index", type=int, default=None)
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
        _export_augment_contact_sheet(loader, args.sample_index, args.export_dir, profile=bool(args.augment_profile))
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
            med_tta=args.med_tta,
            line_trace_tta_count=(
                args.line_trace_tta_count if args.med_tta_count is None else args.med_tta_count
            ),
        )
        return

    if args.trace2cp_vis:
        if args.export_dir is None:
            raise SystemExit("--trace2cp-vis requires --export-dir")
        if args.checkpoint is None:
            raise SystemExit("--trace2cp-vis requires --checkpoint")
        _export_trace2cp_vis(
            loader,
            args.sample_index,
            args.export_dir,
            checkpoint_path=args.checkpoint,
            step_px=args.line_trace_step,
            rf_margin_px=args.line_trace_rf_margin,
            target_offset=args.trace2cp_target_offset,
            target_cp_index=args.trace2cp_target_cp_index,
            med_tta=args.med_tta,
            line_trace_tta_count=(
                args.line_trace_tta_count if args.med_tta_count is None else args.med_tta_count
            ),
            vis_tta=args.vis_tta,
        )
        return

    if args.dir_vis:
        if args.export_dir is None:
            raise SystemExit("--dir-vis requires --export-dir")
        if args.checkpoint is None:
            raise SystemExit("--dir-vis requires --checkpoint")
        _export_dir_vis(
            loader,
            args.sample_index,
            args.export_dir,
            checkpoint_path=args.checkpoint,
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
