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
from vesuvius.neural_tracing.fiber_trace_2d.loader import (
    FiberStrip2DLoader,
    FiberStripSegmentSample,
    load_config,
)
from vesuvius.neural_tracing.fiber_trace_2d.model import (
    FiberStripDirectionModelConfig,
    FiberStripDirectionNet,
)
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import control_point_line_index


def _to_u8_image(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[valid] = np.clip(arr[valid], 0.0, 255.0).astype(np.uint8)
    return out


def _to_u8_display_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    out = np.zeros(arr.shape, dtype=np.uint8)
    finite = np.isfinite(arr)
    out[finite] = np.clip(arr[finite], 0.0, 255.0).astype(np.uint8)
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
    scale: int = 4,
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
    aa_scale = 4
    overlay = Image.new("RGBA", (pil.size[0] * aa_scale, pil.size[1] * aa_scale), (0, 0, 0, 0))
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
            draw.line(
                (
                    (cx - dx) * aa_scale,
                    (cy - dy) * aa_scale,
                    (cx + dx) * aa_scale,
                    (cy + dy) * aa_scale,
                ),
                fill=color_rgba,
                width=aa_scale,
            )
            drawn += 1
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    overlay = overlay.resize(pil.size, resample=resample)
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8), drawn


@dataclass(frozen=True)
class _DirVisImageAugment:
    name: str
    image: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class _DirVisPrediction:
    augment: _DirVisImageAugment
    direction_xy: np.ndarray


def _dir_vis_image_space_augmentations(image: np.ndarray, valid_mask: np.ndarray) -> list[_DirVisImageAugment]:
    arr = np.asarray(image)
    valid = np.asarray(valid_mask, dtype=bool)
    if arr.shape[:2] != valid.shape:
        raise ValueError("valid_mask must match image height/width")

    transforms = [
        ("identity", lambda value: value),
        ("flip_x", lambda value: np.flip(value, axis=1)),
        ("flip_y", lambda value: np.flip(value, axis=0)),
        ("rot90", lambda value: np.rot90(value, k=1, axes=(0, 1))),
        ("rot180", lambda value: np.rot90(value, k=2, axes=(0, 1))),
        ("rot270", lambda value: np.rot90(value, k=3, axes=(0, 1))),
    ]
    return [
        _DirVisImageAugment(
            name=name,
            image=np.ascontiguousarray(transform(arr)),
            valid_mask=np.ascontiguousarray(transform(valid), dtype=bool),
        )
        for name, transform in transforms
    ]


def _render_dir_vis_panel(augment: _DirVisImageAugment, direction_xy: np.ndarray) -> tuple[np.ndarray, int]:
    image_u8 = _to_u8_display_image(augment.image)
    return _direction_field_overlay_rgb(
        image_u8,
        augment.valid_mask,
        direction_xy,
        scale=4,
        stride=2,
    )


def _render_dir_vis_raw_panel(image: np.ndarray, *, scale: int = 4) -> np.ndarray:
    image_u8 = _to_u8_display_image(image)
    if image_u8.ndim == 2:
        image_u8 = np.repeat(image_u8[..., None], 3, axis=-1)
    display_scale = max(1, int(scale))
    return np.repeat(np.repeat(image_u8, display_scale, axis=0), display_scale, axis=1)


def _center_crop_2d(array: np.ndarray, side: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError("array must be 2D")
    crop_side = min(max(1, int(side)), int(arr.shape[0]), int(arr.shape[1]))
    y0 = (int(arr.shape[0]) - crop_side) // 2
    x0 = (int(arr.shape[1]) - crop_side) // 2
    return arr[y0 : y0 + crop_side, x0 : x0 + crop_side]


def _paste_unaugmented_center_patch(
    base_image: np.ndarray,
    base_valid_mask: np.ndarray,
    target: _DirVisImageAugment,
    *,
    paste_side: int,
) -> _DirVisImageAugment:
    image = np.asarray(target.image).copy()
    valid = np.asarray(target.valid_mask, dtype=bool).copy()
    base = np.asarray(base_image)
    base_valid = np.asarray(base_valid_mask, dtype=bool)
    if image.ndim != 2 or valid.shape != image.shape or base.ndim != 2 or base_valid.shape != base.shape:
        raise ValueError("base and target images/masks must be matching 2D arrays")
    patch = _center_crop_2d(base, paste_side)
    patch_valid = _center_crop_2d(base_valid, paste_side)
    side = int(patch.shape[0])
    y0 = (int(image.shape[0]) - side) // 2
    x0 = (int(image.shape[1]) - side) // 2
    image[y0 : y0 + side, x0 : x0 + side] = patch
    valid[y0 : y0 + side, x0 : x0 + side] = patch_valid
    return _DirVisImageAugment(
        name=f"paste_{target.name}",
        image=np.ascontiguousarray(image),
        valid_mask=np.ascontiguousarray(valid, dtype=bool),
    )


def _dir_vis_half_image_paste_side(image_shape_hw: tuple[int, int]) -> int:
    if len(image_shape_hw) != 2:
        raise ValueError("image_shape_hw must contain exactly height and width")
    height, width = int(image_shape_hw[0]), int(image_shape_hw[1])
    side = min(height, width)
    if side <= 0:
        raise ValueError(f"dir-vis image must be non-empty, got image_shape_hw=({height}, {width})")
    return max(1, (side + 1) // 2)


def _direction_model_receptive_field_diameter(config: FiberStripDirectionModelConfig) -> tuple[int, int]:
    radius = 1 + 2 * max(1, int(config.depth))
    return radius, 2 * radius + 1


def _draw_label_band(image: np.ndarray, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    text = str(label)
    font = ImageFont.load_default()
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe, mode="RGBA")
    pad_x = 3
    pad_y = 2
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
        text_y = pad_y - int(bbox[1])
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)
        text_y = pad_y
    band_h = int(text_h + 2 * pad_y)
    cell = np.zeros((arr.shape[0] + band_h, arr.shape[1], 3), dtype=np.uint8)
    cell[band_h:, :, :] = arr
    pil = Image.fromarray(cell, mode="RGB")
    draw = ImageDraw.Draw(pil, mode="RGBA")
    draw.text((pad_x, text_y), text[: max(1, arr.shape[1])], fill=(255, 255, 255, 255), font=font)
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


def _labeled_panel_strip_rgb(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    if not images:
        raise ValueError("images must not be empty")
    if len(images) != len(labels):
        raise ValueError("labels length must match images length")
    prepared: list[np.ndarray] = []
    for image in images:
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("images must be grayscale or RGB")
        prepared.append(arr)

    font = ImageFont.load_default()
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe, mode="RGBA")
    sample_text = max((str(label) for label in labels), key=len)
    pad_x = 3
    pad_y = 2
    try:
        bbox = draw.textbbox((0, 0), sample_text, font=font)
        text_h = int(bbox[3] - bbox[1])
        text_y = pad_y - int(bbox[1])
    except AttributeError:
        _, text_h = draw.textsize(sample_text, font=font)
        text_y = pad_y
    band_h = int(text_h + 2 * pad_y)
    total_w = sum(int(image.shape[1]) for image in prepared)
    max_h = max(int(image.shape[0]) for image in prepared)
    sheet = np.zeros((band_h + max_h, total_w, 3), dtype=np.uint8)
    x = 0
    for image, label in zip(prepared, labels):
        image_h, image_w = image.shape[:2]
        sheet[band_h : band_h + image_h, x : x + image_w] = image
        x += image_w

    pil = Image.fromarray(sheet, mode="RGB")
    draw = ImageDraw.Draw(pil, mode="RGBA")
    x = 0
    for image, label in zip(prepared, labels):
        draw.text((x + pad_x, text_y), str(label)[: max(1, image.shape[1])], fill=(255, 255, 255, 255), font=font)
        x += int(image.shape[1])
    return np.asarray(pil, dtype=np.uint8)


def _write_labeled_panel_strip(path: Path, images: list[np.ndarray], labels: list[str]) -> None:
    _write_jpg(path, _labeled_panel_strip_rgb(images, labels))


def _labeled_panel_grid_rgb(rows: list[list[np.ndarray]], labels: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    if not rows or any(not row for row in rows):
        raise ValueError("rows must not be empty")
    columns = len(rows[0])
    if len(labels) != columns:
        raise ValueError("labels length must match column count")
    if any(len(row) != columns for row in rows):
        raise ValueError("all rows must have the same column count")

    prepared: list[list[np.ndarray]] = []
    shape_hw: tuple[int, int] | None = None
    for row in rows:
        prepared_row: list[np.ndarray] = []
        for image in row:
            arr = np.asarray(image, dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError("images must be grayscale or RGB")
            current_shape = (int(arr.shape[0]), int(arr.shape[1]))
            if shape_hw is None:
                shape_hw = current_shape
            elif current_shape != shape_hw:
                raise ValueError("all grid panels must have the same size")
            prepared_row.append(arr)
        prepared.append(prepared_row)
    assert shape_hw is not None
    panel_h, panel_w = shape_hw

    font = ImageFont.load_default()
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe, mode="RGBA")
    sample_text = max((str(label) for label in labels), key=len)
    pad_x = 3
    pad_y = 2
    try:
        bbox = draw.textbbox((0, 0), sample_text, font=font)
        text_h = int(bbox[3] - bbox[1])
        text_y = pad_y - int(bbox[1])
    except AttributeError:
        _, text_h = draw.textsize(sample_text, font=font)
        text_y = pad_y
    band_h = int(text_h + 2 * pad_y)
    sheet = np.zeros((band_h + len(prepared) * panel_h, columns * panel_w, 3), dtype=np.uint8)
    for row_index, row in enumerate(prepared):
        y0 = band_h + row_index * panel_h
        for col_index, image in enumerate(row):
            x0 = col_index * panel_w
            sheet[y0 : y0 + panel_h, x0 : x0 + panel_w] = image

    pil = Image.fromarray(sheet, mode="RGB")
    draw = ImageDraw.Draw(pil, mode="RGBA")
    for col_index, label in enumerate(labels):
        draw.text(
            (col_index * panel_w + pad_x, text_y),
            str(label)[: max(1, panel_w)],
            fill=(255, 255, 255, 255),
            font=font,
        )
    return np.asarray(pil, dtype=np.uint8)


def _write_labeled_panel_grid(path: Path, rows: list[list[np.ndarray]], labels: list[str]) -> None:
    _write_jpg(path, _labeled_panel_grid_rgb(rows, labels))


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
class _Trace2CpRefinementResult:
    score: float
    raw_y_error_px: float
    considered_y_error_px: float
    center_penalty: float
    denominator_px: float
    closest_x: float
    forward_y_at_closest_x: float
    reverse_y_at_closest_x: float
    closest_midpoint_xy: np.ndarray
    reached_overlap: bool
    reason: str
    partial_forward_xy: np.ndarray
    partial_reverse_xy: np.ndarray
    fused_dense_xy: np.ndarray
    fused_resampled_xy: np.ndarray
    optimized_xy: np.ndarray


@dataclass(frozen=True)
class _Trace2CpBidirectionalResult:
    forward: _Trace2CpDirectionResult
    reverse: _Trace2CpDirectionResult
    refinement: _Trace2CpRefinementResult

    @property
    def score(self) -> float:
        return float(self.refinement.score)

    @property
    def raw_y_error_px(self) -> float:
        return float(self.refinement.raw_y_error_px)

    @property
    def considered_y_error_px(self) -> float:
        return float(self.refinement.considered_y_error_px)

    @property
    def endpoint_score(self) -> float:
        return 0.5 * (float(self.forward.result.score) + float(self.reverse.result.score))

    @property
    def endpoint_raw_y_error_px(self) -> float:
        return 0.5 * (
            float(self.forward.result.raw_y_error_px) + float(self.reverse.result.raw_y_error_px)
        )


@dataclass(frozen=True)
class _Trace2CpPairEvaluation:
    sample_index: int
    sample: FiberStripSegmentSample
    image: np.ndarray
    valid_mask: np.ndarray
    base_result: _Trace2CpBidirectionalResult
    selected_result: _Trace2CpBidirectionalResult
    selected_mode: str
    tta_count: int
    med_fields_count: int
    tta_rows: tuple[str, ...]
    tta_debug_entries: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class _Trace2CpFiberPairPlacement:
    evaluation: _Trace2CpPairEvaluation
    x_shift: float
    y_shift: float


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


def _build_dir_vis_center_patch(
    loader: FiberStrip2DLoader,
    sample_index: int,
    *,
    device: torch.device,
) -> tuple[object, np.ndarray, np.ndarray]:
    sample, image, valid_mask = loader.build_center_strip_patch(sample_index, device=device)
    image_arr = np.asarray(image, dtype=np.float32)
    valid_arr = np.asarray(valid_mask, dtype=bool)
    if image_arr.ndim != 2 or valid_arr.shape != image_arr.shape:
        raise ValueError("dir-vis center patch image and valid mask must be matching 2D arrays")
    height, width = image_arr.shape
    side = min(int(height), int(width))
    if side <= 0:
        raise ValueError(f"dir-vis center patch must be non-empty, got image_shape_hw=({height}, {width})")
    y0 = (int(height) - side) // 2
    x0 = (int(width) - side) // 2
    return (
        sample,
        np.ascontiguousarray(image_arr[y0 : y0 + side, x0 : x0 + side], dtype=np.float32),
        np.ascontiguousarray(valid_arr[y0 : y0 + side, x0 : x0 + side], dtype=bool),
    )


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


def _ordered_x_values_between(start_x: float, end_x: float) -> np.ndarray:
    start = float(start_x)
    end = float(end_x)
    if not np.isfinite(start) or not np.isfinite(end):
        return np.zeros((0,), dtype=np.float32)
    if abs(end - start) <= 1.0e-6:
        return np.asarray([start], dtype=np.float32)
    lo = min(start, end)
    hi = max(start, end)
    columns = np.arange(np.ceil(lo), np.floor(hi) + 1.0, dtype=np.float32)
    values = np.concatenate(
        [
            np.asarray([lo, hi], dtype=np.float32),
            columns.astype(np.float32, copy=False),
        ],
        axis=0,
    )
    values = np.unique(values[(values >= lo - 1.0e-6) & (values <= hi + 1.0e-6)]).astype(np.float32)
    if start > end:
        values = values[::-1].copy()
    return values


def _resample_trace_at_x_values(trace_xy: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace_xy, dtype=np.float32)
    values = np.asarray(x_values, dtype=np.float32).reshape(-1)
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError("trace_xy must have shape N,2")
    points: list[list[float]] = []
    for x in values.tolist():
        y = _trace_y_at_x(trace, float(x))
        if y is not None and np.isfinite(y):
            points.append([float(x), float(y)])
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _resample_trace_between_x(trace_xy: np.ndarray, start_x: float, end_x: float) -> np.ndarray:
    return _resample_trace_at_x_values(trace_xy, _ordered_x_values_between(float(start_x), float(end_x)))


def _usable_trace2cp_vertical_span(shape_hw: tuple[int, int], rf_margin_px: float) -> tuple[float, float, float]:
    height, _ = (int(v) for v in shape_hw)
    margin = max(0.0, float(rf_margin_px))
    top = margin
    bottom = float(height - 1) - margin
    span = bottom - top
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(
            "trace2cp has no usable vertical scoring span: "
            f"height={height} rf_margin_px={margin:.3f}"
        )
    return top, bottom, span


def _trace2cp_overlap_x_values(
    forward_trace_xy: np.ndarray,
    reverse_trace_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
) -> np.ndarray:
    forward = np.asarray(forward_trace_xy, dtype=np.float32)
    reverse = np.asarray(reverse_trace_xy, dtype=np.float32)
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if forward.ndim != 2 or forward.shape[1] != 2 or reverse.ndim != 2 or reverse.shape[1] != 2:
        raise ValueError("forward/reverse traces must have shape N,2")
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("start_xy and target_xy must have shape (2,)")
    finite_forward = forward[np.isfinite(forward).all(axis=1)]
    finite_reverse = reverse[np.isfinite(reverse).all(axis=1)]
    if finite_forward.shape[0] == 0 or finite_reverse.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    cp_lo = min(float(start[0]), float(target[0]))
    cp_hi = max(float(start[0]), float(target[0]))
    overlap_lo = max(float(np.min(finite_forward[:, 0])), float(np.min(finite_reverse[:, 0])), cp_lo)
    overlap_hi = min(float(np.max(finite_forward[:, 0])), float(np.max(finite_reverse[:, 0])), cp_hi)
    if overlap_hi < overlap_lo:
        return np.zeros((0,), dtype=np.float32)
    values = _ordered_x_values_between(overlap_lo, overlap_hi)
    if float(start[0]) > float(target[0]):
        values = values[::-1].copy()
    return values


def _trace2cp_center_penalty(x: float, start_x: float, target_x: float) -> float:
    half_span = 0.5 * abs(float(target_x) - float(start_x))
    if not np.isfinite(half_span) or half_span <= 1.0e-6:
        return 1.0
    center_x = 0.5 * (float(start_x) + float(target_x))
    normalized = abs(float(x) - center_x) / half_span
    return float(1.0 + np.clip(normalized, 0.0, 1.0))


def _closest_trace2cp_approach(
    forward_trace_xy: np.ndarray,
    reverse_trace_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    rf_margin_px: float,
) -> tuple[float, float, float, float, float, float, float, np.ndarray, str, bool]:
    _, _, denominator = _usable_trace2cp_vertical_span(shape_hw, rf_margin_px)
    x_values = _trace2cp_overlap_x_values(forward_trace_xy, reverse_trace_xy, start_xy, target_xy)
    if x_values.size == 0:
        midpoint = np.asarray([np.nan, np.nan], dtype=np.float32)
        return (
            1.0,
            denominator,
            denominator,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            midpoint,
            "no_trace_overlap",
            False,
        )

    forward_rows: list[tuple[float, float]] = []
    reverse_rows: list[tuple[float, float]] = []
    gaps: list[float] = []
    penalties: list[float] = []
    considered_gaps: list[float] = []
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    for x in x_values.tolist():
        forward_y = _trace_y_at_x(forward_trace_xy, float(x))
        reverse_y = _trace_y_at_x(reverse_trace_xy, float(x))
        if forward_y is None or reverse_y is None:
            continue
        if not np.isfinite(forward_y) or not np.isfinite(reverse_y):
            continue
        gap = abs(float(forward_y) - float(reverse_y))
        penalty = _trace2cp_center_penalty(float(x), float(start[0]), float(target[0]))
        forward_rows.append((float(x), float(forward_y)))
        reverse_rows.append((float(x), float(reverse_y)))
        gaps.append(gap)
        penalties.append(penalty)
        considered_gaps.append(gap * penalty)
    if not gaps:
        midpoint = np.asarray([np.nan, np.nan], dtype=np.float32)
        return (
            1.0,
            denominator,
            denominator,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            midpoint,
            "no_trace_overlap",
            False,
        )

    considered_arr = np.asarray(considered_gaps, dtype=np.float32)
    min_considered = float(np.min(considered_arr))
    midpoint_x = 0.5 * (float(start[0]) + float(target[0]))
    candidate_indices = np.flatnonzero(np.isclose(considered_arr, min_considered, rtol=1.0e-6, atol=1.0e-6))
    if candidate_indices.size > 1:
        candidate_x = np.asarray([forward_rows[int(index)][0] for index in candidate_indices], dtype=np.float32)
        closest_index = int(candidate_indices[int(np.argmin(np.abs(candidate_x - np.float32(midpoint_x))))])
    else:
        closest_index = int(candidate_indices[0])
    closest_x, forward_y = forward_rows[closest_index]
    _, reverse_y = reverse_rows[closest_index]
    gap = float(gaps[closest_index])
    penalty = float(penalties[closest_index])
    considered_gap = float(considered_gaps[closest_index])
    midpoint = np.asarray([closest_x, 0.5 * (forward_y + reverse_y)], dtype=np.float32)
    score = float(np.clip(considered_gap / denominator, 0.0, 1.0))
    return score, gap, considered_gap, penalty, closest_x, forward_y, reverse_y, midpoint, "closest_approach", True


def _vertical_warp_trace_to_meet(
    trace_xy: np.ndarray,
    *,
    anchor_xy: np.ndarray,
    meet_x: float,
    meet_y: float,
    source_meet_y: float,
) -> np.ndarray:
    trace = np.asarray(trace_xy, dtype=np.float32)
    anchor = np.asarray(anchor_xy, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError("trace_xy must have shape N,2")
    if anchor.shape != (2,):
        raise ValueError("anchor_xy must have shape (2,)")
    if trace.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    warped = trace.copy()
    denom = abs(float(meet_x) - float(anchor[0]))
    if denom <= 1.0e-6:
        blend = np.ones((trace.shape[0],), dtype=np.float32)
    else:
        blend = np.clip(np.abs(trace[:, 0] - np.float32(anchor[0])) / np.float32(denom), 0.0, 1.0)
    warped[:, 1] = warped[:, 1] + blend * np.float32(float(meet_y) - float(source_meet_y))
    warped[0] = anchor.astype(np.float32, copy=False)
    warped[-1, 0] = np.float32(meet_x)
    warped[-1, 1] = np.float32(meet_y)
    return warped.astype(np.float32, copy=False)


def _resample_polyline_by_arclength(line_xy: np.ndarray, *, step_px: float) -> np.ndarray:
    line = np.asarray(line_xy, dtype=np.float32)
    if line.ndim != 2 or line.shape[1] != 2:
        raise ValueError("line_xy must have shape N,2")
    finite = np.isfinite(line).all(axis=1)
    line = line[finite]
    if line.shape[0] <= 1:
        return line.astype(np.float32, copy=True)
    delta = line[1:] - line[:-1]
    seg_len = np.linalg.norm(delta, axis=1)
    keep = np.concatenate([[True], seg_len > 1.0e-6])
    line = line[keep]
    if line.shape[0] <= 1:
        return line.astype(np.float32, copy=True)
    delta = line[1:] - line[:-1]
    seg_len = np.linalg.norm(delta, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len, dtype=np.float64)]).astype(np.float64)
    total = float(cumulative[-1])
    if total <= 1.0e-6:
        return line[:1].astype(np.float32, copy=True)
    step = max(float(step_px), 1.0e-3)
    distances = np.arange(0.0, total, step, dtype=np.float64)
    if distances.size == 0 or abs(float(distances[-1]) - total) > 1.0e-6:
        distances = np.concatenate([distances, np.asarray([total], dtype=np.float64)], axis=0)
    xs = np.interp(distances, cumulative, line[:, 0].astype(np.float64))
    ys = np.interp(distances, cumulative, line[:, 1].astype(np.float64))
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _sample_direction_field_torch(
    direction_xy: torch.Tensor,
    points_xy: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if direction_xy.ndim != 3 or int(direction_xy.shape[2]) != 2:
        raise ValueError("direction_xy must have shape H,W,2")
    if points_xy.ndim != 2 or int(points_xy.shape[1]) != 2:
        raise ValueError("points_xy must have shape N,2")
    height = int(direction_xy.shape[0])
    width = int(direction_xy.shape[1])
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    inside = (x >= 0.0) & (y >= 0.0) & (x <= float(width - 1)) & (y <= float(height - 1))
    x0 = torch.floor(x).to(dtype=torch.long).clamp(0, max(0, width - 1))
    y0 = torch.floor(y).to(dtype=torch.long).clamp(0, max(0, height - 1))
    x1 = (x0 + 1).clamp(0, max(0, width - 1))
    y1 = (y0 + 1).clamp(0, max(0, height - 1))
    tx = (x - x0.to(dtype=torch.float32)).clamp(0.0, 1.0).view(-1, 1)
    ty = (y - y0.to(dtype=torch.float32)).clamp(0.0, 1.0).view(-1, 1)
    top = direction_xy[y0, x0] * (1.0 - tx) + direction_xy[y0, x1] * tx
    bottom = direction_xy[y1, x0] * (1.0 - tx) + direction_xy[y1, x1] * tx
    sampled = top * (1.0 - ty) + bottom * ty
    raw_norm = torch.linalg.vector_norm(sampled, dim=1)
    sampled = sampled / raw_norm.clamp_min(1.0e-12).view(-1, 1)
    valid = inside & torch.isfinite(sampled).all(dim=1) & (raw_norm > 1.0e-6)
    if valid_mask is not None:
        valid = (
            valid
            & valid_mask[y0, x0]
            & valid_mask[y0, x1]
            & valid_mask[y1, x0]
            & valid_mask[y1, x1]
        )
    return sampled, valid


def _optimize_trace2cp_line(
    initial_line_xy: np.ndarray,
    direction_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None,
    step_px: float,
    iterations: int = 80,
) -> np.ndarray:
    line = np.asarray(initial_line_xy, dtype=np.float32)
    field = np.asarray(direction_xy, dtype=np.float32)
    if line.ndim != 2 or line.shape[1] != 2:
        raise ValueError("initial_line_xy must have shape N,2")
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("direction_xy must have shape H,W,2")
    if line.shape[0] <= 2:
        return line.astype(np.float32, copy=True)

    device = torch.device("cpu")
    field_t = torch.as_tensor(field, dtype=torch.float32, device=device)
    valid_t = None
    if valid_mask is not None:
        valid_arr = np.asarray(valid_mask, dtype=bool)
        if valid_arr.shape != field.shape[:2]:
            raise ValueError("valid_mask must match direction field shape")
        valid_t = torch.as_tensor(valid_arr, dtype=torch.bool, device=device)
    base = torch.as_tensor(line, dtype=torch.float32, device=device)
    x_fixed = base[:, 0].detach()
    y_initial = base[:, 1].detach()
    y_param = torch.nn.Parameter(y_initial[1:-1].clone())
    optimizer = torch.optim.Adam([y_param], lr=0.05)
    target_step = max(float(step_px), 1.0e-3)

    for _ in range(max(1, int(iterations))):
        optimizer.zero_grad(set_to_none=True)
        y = torch.cat([y_initial[:1], y_param, y_initial[-1:]], dim=0)
        points = torch.stack([x_fixed, y], dim=1)
        segments = points[1:] - points[:-1]
        lengths = torch.linalg.vector_norm(segments, dim=1).clamp_min(1.0e-6)
        segment_unit = segments / lengths.view(-1, 1)
        midpoints = 0.5 * (points[1:] + points[:-1])
        sampled, valid = _sample_direction_field_torch(field_t, midpoints, valid_t)
        if not bool(valid.any()):
            return line.astype(np.float32, copy=True)
        direction_loss = 1.0 - torch.abs(torch.sum(segment_unit[valid] * sampled[valid], dim=1)).clamp(0.0, 1.0)
        step_loss = ((lengths - lengths.mean().detach()) / target_step).pow(2)
        anchor_loss = ((y - y_initial) / target_step).pow(2)
        loss = direction_loss.mean() + 0.05 * step_loss.mean() + 0.01 * anchor_loss.mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y = torch.cat([y_initial[:1], y_param, y_initial[-1:]], dim=0)
        optimized = torch.stack([x_fixed, y], dim=1).detach().cpu().numpy().astype(np.float32)
    return optimized


def _trace2cp_refinement_from_traces(
    forward_trace_xy: np.ndarray,
    reverse_trace_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    direction_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    shape_hw: tuple[int, int],
    step_px: float,
    rf_margin_px: float,
) -> _Trace2CpRefinementResult:
    score, gap, considered_gap, penalty, closest_x, forward_y, reverse_y, midpoint, reason, reached = (
        _closest_trace2cp_approach(
            forward_trace_xy,
            reverse_trace_xy,
            start_xy,
            target_xy,
            shape_hw=shape_hw,
            rf_margin_px=rf_margin_px,
        )
    )
    _, _, denominator = _usable_trace2cp_vertical_span(shape_hw, rf_margin_px)
    empty = np.zeros((0, 2), dtype=np.float32)
    if not reached:
        return _Trace2CpRefinementResult(
            score=score,
            raw_y_error_px=gap,
            considered_y_error_px=considered_gap,
            center_penalty=penalty,
            denominator_px=denominator,
            closest_x=closest_x,
            forward_y_at_closest_x=forward_y,
            reverse_y_at_closest_x=reverse_y,
            closest_midpoint_xy=midpoint,
            reached_overlap=False,
            reason=reason,
            partial_forward_xy=empty,
            partial_reverse_xy=empty,
            fused_dense_xy=empty,
            fused_resampled_xy=empty,
            optimized_xy=empty,
        )

    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    partial_forward = _resample_trace_between_x(forward_trace_xy, float(start[0]), closest_x)
    partial_reverse = _resample_trace_between_x(reverse_trace_xy, float(target[0]), closest_x)
    warped_forward = _vertical_warp_trace_to_meet(
        partial_forward,
        anchor_xy=start,
        meet_x=closest_x,
        meet_y=float(midpoint[1]),
        source_meet_y=forward_y,
    )
    warped_reverse = _vertical_warp_trace_to_meet(
        partial_reverse,
        anchor_xy=target,
        meet_x=closest_x,
        meet_y=float(midpoint[1]),
        source_meet_y=reverse_y,
    )
    reverse_meet_to_target = warped_reverse[::-1].copy()
    if warped_forward.shape[0] > 0 and reverse_meet_to_target.shape[0] > 0:
        fused_dense = np.concatenate([warped_forward, reverse_meet_to_target[1:]], axis=0)
    else:
        fused_dense = np.zeros((0, 2), dtype=np.float32)
    fused_resampled = _resample_polyline_by_arclength(fused_dense, step_px=step_px)
    optimized = _optimize_trace2cp_line(
        fused_resampled,
        direction_xy,
        valid_mask=valid_mask,
        step_px=step_px,
    )
    return _Trace2CpRefinementResult(
        score=score,
        raw_y_error_px=gap,
        considered_y_error_px=considered_gap,
        center_penalty=penalty,
        denominator_px=denominator,
        closest_x=closest_x,
        forward_y_at_closest_x=forward_y,
        reverse_y_at_closest_x=reverse_y,
        closest_midpoint_xy=midpoint,
        reached_overlap=True,
        reason=reason,
        partial_forward_xy=partial_forward,
        partial_reverse_xy=partial_reverse,
        fused_dense_xy=fused_dense.astype(np.float32, copy=False),
        fused_resampled_xy=fused_resampled.astype(np.float32, copy=False),
        optimized_xy=optimized.astype(np.float32, copy=False),
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
    refinement = _trace2cp_refinement_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        direction_xy=direction_xy,
        valid_mask=valid_mask,
        shape_hw=(int(direction_xy.shape[0]), int(direction_xy.shape[1])),
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    return _Trace2CpBidirectionalResult(forward=forward, reverse=reverse, refinement=refinement)


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
    reference = fields[0]
    refinement = _trace2cp_refinement_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        direction_xy=reference.direction_xy,
        valid_mask=reference.valid_mask,
        shape_hw=shape_hw,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    return _Trace2CpBidirectionalResult(forward=forward, reverse=reverse, refinement=refinement)


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
                raise ValueError("could not map CP into TTA patch")
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
    result_label: str = "trace2cp",
    reference_result: _Trace2CpBidirectionalResult | None = None,
    reference_label: str = "reference",
) -> np.ndarray:
    from PIL import Image, ImageDraw

    def draw_row(
        polylines: list[tuple[np.ndarray, tuple[int, int, int, int], int]],
        *,
        label: str,
        meeting_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        overlay = overlay_line_coords_rgb(image_u8, line_xy, opacity=0.25, thickness=1)
        for points_xy, color, thickness in polylines:
            overlay = _overlay_polyline_rgb(overlay, points_xy, color_rgba=color, thickness=thickness)
        arr = np.asarray(overlay, dtype=np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(pil, mode="RGBA")
        height, _ = arr.shape[:2]

        sx, sy = (float(v) for v in np.asarray(start_xy, dtype=np.float32))
        tx, ty = (float(v) for v in np.asarray(target_xy, dtype=np.float32))
        if np.isfinite([sx, sy]).all():
            draw.line((sx, 0.0, sx, float(height - 1)), fill=(32, 255, 255, 80), width=1)
            draw.ellipse((sx - 2.0, sy - 2.0, sx + 2.0, sy + 2.0), outline=(32, 255, 255, 255), width=1)
        if np.isfinite([tx, ty]).all():
            draw.line((tx, 0.0, tx, float(height - 1)), fill=(255, 220, 64, 95), width=1)
            gap = 2.0
            arm = 6.0
            draw.line((tx - arm, ty, tx - gap, ty), fill=(255, 220, 64, 255), width=1)
            draw.line((tx + gap, ty, tx + arm, ty), fill=(255, 220, 64, 255), width=1)
            draw.line((tx, ty - arm, tx, ty - gap), fill=(255, 220, 64, 255), width=1)
            draw.line((tx, ty + gap, tx, ty + arm), fill=(255, 220, 64, 255), width=1)
        if meeting_xy is not None:
            meet = np.asarray(meeting_xy, dtype=np.float32)
            if meet.shape == (2,) and bool(np.isfinite(meet).all()):
                mx, my = (float(v) for v in meet)
                draw.ellipse((mx - 2.0, my - 2.0, mx + 2.0, my + 2.0), outline=(255, 255, 255, 255), width=1)
        return _draw_label_band(np.asarray(pil, dtype=np.uint8), label)

    def result_stack(result: _Trace2CpBidirectionalResult, label_prefix: str) -> np.ndarray:
        refinement = result.refinement
        rows = [
            draw_row(
                [
                    (result.forward.trace_xy, (0, 255, 0, 230), 1),
                    (result.reverse.trace_xy, (255, 64, 220, 230), 1),
                ],
                label=(
                    f"{label_prefix} traces score={result.score:.4f} "
                    f"gap={refinement.raw_y_error_px:.2f}px "
                    f"considered={refinement.considered_y_error_px:.2f}px "
                    f"endpoint={result.endpoint_score:.4f}"
                ),
                meeting_xy=refinement.closest_midpoint_xy,
            ),
            draw_row(
                [
                    (refinement.partial_forward_xy, (0, 255, 0, 230), 1),
                    (refinement.partial_reverse_xy, (255, 64, 220, 230), 1),
                ],
                label=(
                    f"{label_prefix} partial closest_x={refinement.closest_x:.2f} "
                    f"f_y={refinement.forward_y_at_closest_x:.2f} "
                    f"r_y={refinement.reverse_y_at_closest_x:.2f}"
                ),
                meeting_xy=refinement.closest_midpoint_xy,
            ),
            draw_row(
                [(refinement.fused_resampled_xy, (255, 220, 64, 240), 1)],
                label=f"{label_prefix} fused points={int(refinement.fused_resampled_xy.shape[0])}",
                meeting_xy=refinement.closest_midpoint_xy,
            ),
            draw_row(
                [(refinement.optimized_xy, (64, 224, 255, 240), 1)],
                label=f"{label_prefix} optimized points={int(refinement.optimized_xy.shape[0])} reason={refinement.reason}",
                meeting_xy=refinement.closest_midpoint_xy,
            ),
        ]
        width = max(int(row.shape[1]) for row in rows)
        total_h = sum(int(row.shape[0]) for row in rows)
        stack = np.zeros((total_h, width, 3), dtype=np.uint8)
        y0 = 0
        for row in rows:
            h, w = row.shape[:2]
            stack[y0 : y0 + h, :w] = row
            y0 += h
        return stack

    stacks = [result_stack(bidirectional_result, result_label)]
    if reference_result is not None:
        stacks.append(result_stack(reference_result, reference_label))
    width = sum(int(stack.shape[1]) for stack in stacks)
    height = max(int(stack.shape[0]) for stack in stacks)
    combined = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = 0
    for stack in stacks:
        h, w = stack.shape[:2]
        combined[:h, x0 : x0 + w] = stack
        x0 += w
    return combined


def _trace2cp_fiber_pair_cp_indices(cp_count: int, target_offset: int) -> tuple[tuple[int, int], ...]:
    count = int(cp_count)
    offset = int(target_offset)
    if offset == 0:
        raise ValueError("trace2cp fiber target offset must be non-zero")
    if count <= abs(offset):
        return ()
    if offset > 0:
        return tuple((index, index + offset) for index in range(0, count - offset))
    return tuple((index, index + offset) for index in range(-offset, count))


def _translated_trace2cp_points(points_xy: np.ndarray, x_shift: float, y_shift: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    translated = points.copy()
    translated[:, 0] += np.float32(x_shift)
    translated[:, 1] += np.float32(y_shift)
    return translated


def _draw_trace2cp_fiber_overlay(
    evaluations: list[_Trace2CpPairEvaluation],
    *,
    control_point_x: np.ndarray,
    label: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    if not evaluations:
        raise ValueError("no Trace2CP pair evaluations to draw")
    cp_x = np.asarray(control_point_x, dtype=np.float32)
    if cp_x.ndim != 1 or cp_x.shape[0] == 0:
        raise ValueError("control_point_x must be a non-empty 1D array")

    placements: list[_Trace2CpFiberPairPlacement] = []
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    for evaluation in evaluations:
        sample = evaluation.sample
        start_cp = int(sample.start_control_point_index)
        if start_cp < 0 or start_cp >= cp_x.shape[0]:
            raise ValueError(f"start control point {start_cp} is outside control_point_x")
        start_xy = np.asarray(sample.start_control_point_xy, dtype=np.float32)
        if start_xy.shape != (2,) or not bool(np.isfinite(start_xy).all()):
            raise ValueError("Trace2CP sample has invalid start_control_point_xy")
        image = np.asarray(evaluation.image)
        if image.ndim != 2:
            raise ValueError("Trace2CP fiber visualization expects 2D pair images")
        height, width = image.shape[:2]
        x_shift = float(cp_x[start_cp]) - float(start_xy[0])
        y_shift = -float(start_xy[1])
        placements.append(_Trace2CpFiberPairPlacement(evaluation=evaluation, x_shift=x_shift, y_shift=y_shift))
        min_x = min(min_x, x_shift)
        min_y = min(min_y, y_shift)
        max_x = max(max_x, x_shift + float(width))
        max_y = max(max_y, y_shift + float(height))

    pad = 8
    global_x_shift = float(pad) - float(np.floor(min_x))
    global_y_shift = float(pad) - float(np.floor(min_y))
    canvas_width = max(1, int(np.ceil(max_x + global_x_shift)) + pad)
    canvas_height = max(1, int(np.ceil(max_y + global_y_shift)) + pad)
    accum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)

    for placement in placements:
        evaluation = placement.evaluation
        image_u8 = _to_u8_image(evaluation.image, evaluation.valid_mask)
        rgb = np.repeat(image_u8[..., None], 3, axis=-1).astype(np.float32)
        valid = np.asarray(evaluation.valid_mask, dtype=bool)
        x0 = int(round(placement.x_shift + global_x_shift))
        y0 = int(round(placement.y_shift + global_y_shift))
        h, w = image_u8.shape[:2]
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(canvas_width, x0 + w)
        dst_y1 = min(canvas_height, y0 + h)
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            continue
        src_x0 = dst_x0 - x0
        src_y0 = dst_y0 - y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        local_valid = valid[src_y0:src_y1, src_x0:src_x1]
        if not bool(local_valid.any()):
            continue
        local_weight = local_valid[..., None].astype(np.float32)
        accum[dst_y0:dst_y1, dst_x0:dst_x1] += rgb[src_y0:src_y1, src_x0:src_x1] * local_weight
        weight[dst_y0:dst_y1, dst_x0:dst_x1] += local_weight

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    covered = weight[..., 0] > 0.0
    canvas[covered] = np.clip(accum[covered] / weight[covered], 0.0, 255.0).astype(np.uint8)
    pil = Image.fromarray(canvas, mode="RGB")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    font = ImageFont.load_default()

    def draw_points(points_xy: np.ndarray, color: tuple[int, int, int, int], width: int = 1) -> None:
        coords = np.asarray(points_xy, dtype=np.float32)
        finite = coords.ndim == 2 and coords.shape[1] == 2
        if not finite:
            return
        coords = coords[np.isfinite(coords).all(axis=1)]
        if coords.shape[0] == 0:
            return
        points = [(float(x), float(y)) for x, y in coords.tolist()]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 1.0, y - 1.0, x + 1.0, y + 1.0), outline=color, width=max(1, width))
        else:
            draw.line(points, fill=color, width=max(1, width), joint="curve")

    for placement in placements:
        evaluation = placement.evaluation
        result = evaluation.selected_result
        x_shift = placement.x_shift + global_x_shift
        y_shift = placement.y_shift + global_y_shift
        draw_points(
            _translated_trace2cp_points(evaluation.sample.line_xy, x_shift, y_shift),
            (210, 210, 210, 95),
            width=1,
        )
        draw_points(
            _translated_trace2cp_points(result.forward.trace_xy, x_shift, y_shift),
            (0, 255, 0, 230),
            width=1,
        )
        draw_points(
            _translated_trace2cp_points(result.reverse.trace_xy, x_shift, y_shift),
            (255, 64, 220, 230),
            width=1,
        )
        draw_points(
            _translated_trace2cp_points(result.refinement.optimized_xy, x_shift, y_shift),
            (64, 224, 255, 240),
            width=1,
        )
        for point_xy, color in (
            (evaluation.sample.start_control_point_xy, (32, 255, 255, 255)),
            (evaluation.sample.target_control_point_xy, (255, 220, 64, 255)),
        ):
            point = _translated_trace2cp_points(np.asarray(point_xy, dtype=np.float32).reshape(1, 2), x_shift, y_shift)
            if point.shape == (1, 2) and bool(np.isfinite(point).all()):
                x, y = (float(v) for v in point[0])
                draw.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), outline=color, width=1)
        midpoint_x = 0.5 * (
            float(cp_x[int(evaluation.sample.start_control_point_index)])
            + float(cp_x[int(evaluation.sample.target_control_point_index)])
        ) + global_x_shift
        draw.text(
            (midpoint_x - 10.0, 2.0),
            f"{evaluation.selected_result.score:.3f}",
            fill=(255, 255, 255, 220),
            font=font,
        )

    drawn = np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8)
    return _draw_label_band(drawn, label)


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


def _evaluate_trace2cp_pair(
    loader: FiberStrip2DLoader,
    model: torch.nn.Module,
    sample_index: int,
    *,
    device: torch.device,
    step_px: float,
    rf_margin_px: float,
    target_offset: int,
    target_cp_index: int | None,
    med_tta: bool,
    line_trace_tta_count: int,
    vis_tta: bool,
    sample_mode: str = "random",
) -> _Trace2CpPairEvaluation:
    sample, image, valid_mask = loader.build_trace2cp_segment_patch(
        sample_index,
        target_control_point_index=target_cp_index,
        target_offset=target_offset,
        rf_margin_px=rf_margin_px,
        device=device,
        sample_mode=sample_mode,
    )
    direction_xy = _predict_direction_field(model, image, valid_mask, device=device)
    start_xy = np.asarray(sample.start_control_point_xy, dtype=np.float32)
    target_xy = np.asarray(sample.target_control_point_xy, dtype=np.float32)
    base_result = _trace_score_trace2cp_bidirectional(
        direction_xy,
        start_xy,
        target_xy,
        valid_mask=valid_mask,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    selected_result = base_result
    tta_rows: list[str] = []
    tta_count = max(0, int(line_trace_tta_count)) if med_tta else 0
    selected_mode = "base"
    med_fields_count = 0
    tta_debug_entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
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
                    rf_margin_px=rf_margin_px,
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
            rf_margin_px=rf_margin_px,
        )
        selected_mode = "med_tta"
    return _Trace2CpPairEvaluation(
        sample_index=int(sample_index),
        sample=sample,
        image=image,
        valid_mask=valid_mask,
        base_result=base_result,
        selected_result=selected_result,
        selected_mode=selected_mode,
        tta_count=tta_count,
        med_fields_count=med_fields_count,
        tta_rows=tuple(tta_rows),
        tta_debug_entries=tuple(tta_debug_entries),
    )


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
    evaluation = _evaluate_trace2cp_pair(
        loader,
        model,
        sample_index,
        device=device,
        step_px=step_px,
        rf_margin_px=margin,
        target_offset=target_offset,
        target_cp_index=target_cp_index,
        med_tta=med_tta,
        line_trace_tta_count=line_trace_tta_count,
        vis_tta=vis_tta,
    )
    sample = evaluation.sample
    image = evaluation.image
    valid_mask = evaluation.valid_mask
    start_xy = np.asarray(sample.start_control_point_xy, dtype=np.float32)
    target_xy = np.asarray(sample.target_control_point_xy, dtype=np.float32)
    base_result = evaluation.base_result
    selected_result = evaluation.selected_result
    selected_mode = evaluation.selected_mode
    tta_count = evaluation.tta_count
    med_fields_count = evaluation.med_fields_count
    tta_rows = list(evaluation.tta_rows)
    tta_debug_dir: Path | None = None
    if vis_tta and evaluation.tta_debug_entries:
        tta_debug_dir = _write_trace2cp_tta_debug_images(out, list(evaluation.tta_debug_entries))

    image_u8 = _to_u8_image(image, valid_mask)
    overlay = _draw_trace2cp_overlay(
        image_u8,
        line_xy=sample.line_xy,
        start_xy=start_xy,
        target_xy=target_xy,
        bidirectional_result=selected_result,
        result_label=selected_mode,
        reference_result=base_result if selected_result is not base_result else None,
        reference_label="reference",
    )
    _write_jpg(out / "trace2cp_vis.jpg", overlay)
    forward = selected_result.forward.result
    reverse = selected_result.reverse.result
    refinement = selected_result.refinement
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
        f"actual_y_error_px={selected_result.raw_y_error_px:.6f}",
        f"considered_y_error_px={selected_result.considered_y_error_px:.6f}",
        f"center_penalty={refinement.center_penalty:.6f}",
        f"score_semantics=center_penalized_minimum_vertical_trace_to_trace_separation",
        f"trace2cp_denominator_px={refinement.denominator_px:.6f}",
        f"closest_x={refinement.closest_x:.6f}",
        f"closest_forward_y={refinement.forward_y_at_closest_x:.6f}",
        f"closest_reverse_y={refinement.reverse_y_at_closest_x:.6f}",
        f"closest_midpoint_xy=({refinement.closest_midpoint_xy[0]:.6f}, {refinement.closest_midpoint_xy[1]:.6f})",
        f"closest_reached_overlap={refinement.reached_overlap}",
        f"closest_reason={refinement.reason}",
        f"fused_points={int(refinement.fused_resampled_xy.shape[0])}",
        f"optimized_points={int(refinement.optimized_xy.shape[0])}",
        f"endpoint_trace2cp_score={selected_result.endpoint_score:.8f}",
        f"endpoint_raw_y_error_px={selected_result.endpoint_raw_y_error_px:.6f}",
        f"reference_trace2cp_score={base_result.score:.8f}",
        f"reference_actual_y_error_px={base_result.raw_y_error_px:.6f}",
        f"reference_considered_y_error_px={base_result.considered_y_error_px:.6f}",
        f"reference_center_penalty={base_result.refinement.center_penalty:.6f}",
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
        f"actual_y_error_px={selected_result.raw_y_error_px:.6f} "
        f"considered_y_error_px={selected_result.considered_y_error_px:.6f} "
        f"center_penalty={refinement.center_penalty:.6f} "
        f"reference_score={base_result.score:.8f} "
        f"endpoint_score={selected_result.endpoint_score:.8f} "
        f"closest_x={refinement.closest_x:.6f} "
        f"forward_endpoint_score={forward.score:.8f} reverse_endpoint_score={reverse.score:.8f} "
        f"forward_reached={forward.reached_target_column} reverse_reached={reverse.reached_target_column} "
        f"forward_reason={forward.reason} reverse_reason={reverse.reason}"
    )
    print(f"exported trace2cp_vis.jpg and trace2cp_summary.txt to {out}")


def _trace2cp_control_point_x_positions(loader: FiberStrip2DLoader, flat_indices: tuple[int, ...]) -> np.ndarray:
    if not flat_indices:
        raise ValueError("fiber has no control points")
    record, _, _ = loader.descriptor_for_sample_index(int(flat_indices[0]), sample_mode="flat")
    cumulative = FiberStrip2DLoader._line_arc_lengths(np.asarray(record.fiber.line_points_xyz, dtype=np.float64))
    positions: list[float] = []
    for cp_index in range(len(flat_indices)):
        line_index = control_point_line_index(record.fiber, cp_index)
        positions.append(float(cumulative[line_index]) / float(record.volume_spacing_base))
    values = np.asarray(positions, dtype=np.float32)
    return values - np.float32(float(np.nanmin(values)))


def _export_trace2cp_fiber_vis(
    loader: FiberStrip2DLoader,
    fiber_json: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    step_px: float,
    rf_margin_px: float | None,
    target_offset: int,
    med_tta: bool = False,
    line_trace_tta_count: int = 100,
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(loader.config.augment.device)
    model, checkpoint = _load_direction_model(checkpoint_path, loader, device=device)
    configured_depth = max(1, int(_model_config_from_checkpoint(checkpoint, loader).depth))
    default_margin = configured_depth
    margin = float(default_margin if rf_margin_px is None else rf_margin_px)
    flat_indices = loader.flat_sample_indices_for_fiber_json(fiber_json)
    pair_indices = _trace2cp_fiber_pair_cp_indices(len(flat_indices), int(target_offset))
    if not pair_indices:
        raise ValueError(
            "fiber has no in-range Trace2CP pairs for target offset "
            f"{int(target_offset)}: fiber_json='{fiber_json}' cp_count={len(flat_indices)}"
        )

    evaluations: list[_Trace2CpPairEvaluation] = []
    for start_cp_index, target_cp_index in pair_indices:
        evaluation = _evaluate_trace2cp_pair(
            loader,
            model,
            int(flat_indices[start_cp_index]),
            device=device,
            step_px=step_px,
            rf_margin_px=margin,
            target_offset=target_offset,
            target_cp_index=int(target_cp_index),
            med_tta=med_tta,
            line_trace_tta_count=line_trace_tta_count,
            vis_tta=False,
            sample_mode="flat",
        )
        evaluations.append(evaluation)
        print(
            "trace2cp fiber_pair "
            f"fiber_path={evaluation.sample.fiber_path} "
            f"start_cp={evaluation.sample.start_control_point_index} "
            f"target_cp={evaluation.sample.target_control_point_index} "
            f"mode={evaluation.selected_mode} score={evaluation.selected_result.score:.8f} "
            f"actual_y_error_px={evaluation.selected_result.raw_y_error_px:.6f} "
            f"considered_y_error_px={evaluation.selected_result.considered_y_error_px:.6f}",
            flush=True,
        )

    cp_x = _trace2cp_control_point_x_positions(loader, flat_indices)
    scores = np.asarray([evaluation.selected_result.score for evaluation in evaluations], dtype=np.float64)
    actual_errors = np.asarray(
        [evaluation.selected_result.raw_y_error_px for evaluation in evaluations], dtype=np.float64
    )
    fiber_path_display = evaluations[0].sample.fiber_path or str(fiber_json)
    mode = "med_tta" if med_tta else "base"
    label = (
        f"trace2cp fiber pairs={len(evaluations)} mode={mode} "
        f"mean={float(np.mean(scores)):.4f} max={float(np.max(scores)):.4f} "
        f"mean_actual_px={float(np.mean(actual_errors)):.2f}"
    )
    overlay = _draw_trace2cp_fiber_overlay(
        evaluations,
        control_point_x=cp_x,
        label=label,
    )
    _write_jpg(out / "trace2cp_fiber_vis.jpg", overlay)

    rows = [
        "start_cp target_cp score actual_y_error_px considered_y_error_px center_penalty "
        "reference_score endpoint_score closest_x forward_reason reverse_reason"
    ]
    for evaluation in evaluations:
        result = evaluation.selected_result
        refinement = result.refinement
        rows.append(
            f"{evaluation.sample.start_control_point_index} "
            f"{evaluation.sample.target_control_point_index} "
            f"{result.score:.8f} "
            f"{result.raw_y_error_px:.6f} "
            f"{result.considered_y_error_px:.6f} "
            f"{refinement.center_penalty:.6f} "
            f"{evaluation.base_result.score:.8f} "
            f"{result.endpoint_score:.8f} "
            f"{refinement.closest_x:.6f} "
            f"{result.forward.result.reason} "
            f"{result.reverse.result.reason}"
        )
    summary = [
        f"fiber_json={fiber_path_display}",
        f"checkpoint={Path(checkpoint_path).expanduser().resolve()}",
        f"checkpoint_step={checkpoint.get('step', 'unknown')}",
        f"target_offset={int(target_offset)}",
        f"cp_count={len(flat_indices)}",
        f"pair_count={len(evaluations)}",
        f"step_px={float(step_px):.3f}",
        f"rf_margin_px={margin:.3f}",
        f"trace_mode={mode}",
        f"med_tta={bool(med_tta)}",
        f"line_trace_tta_count={max(0, int(line_trace_tta_count)) if med_tta else 0}",
        f"score_mean={float(np.mean(scores)):.8f}",
        f"score_max={float(np.max(scores)):.8f}",
        f"score_min={float(np.min(scores)):.8f}",
        f"actual_y_error_mean_px={float(np.mean(actual_errors)):.6f}",
        f"image_shape_hw=({int(overlay.shape[0])}, {int(overlay.shape[1])})",
        "",
        *rows,
    ]
    (out / "trace2cp_fiber_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(
        "trace2cp fiber "
        f"fiber_json={fiber_path_display} "
        f"pairs={len(evaluations)} mode={mode} mean_score={float(np.mean(scores)):.8f} "
        f"max_score={float(np.max(scores)):.8f} mean_actual_y_error_px={float(np.mean(actual_errors)):.6f}"
    )
    print(f"exported trace2cp_fiber_vis.jpg and trace2cp_fiber_summary.txt to {out}")


def _export_dir_vis(
    loader: FiberStrip2DLoader,
    sample_index: int,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    dbg_dirs: bool = False,
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(loader.config.augment.device)
    sample, image, valid_mask = _build_dir_vis_center_patch(loader, sample_index, device=device)
    model, checkpoint = _load_direction_model(checkpoint_path, loader, device=device)
    model_config = _model_config_from_checkpoint(checkpoint, loader)
    rf_radius_px, rf_diameter_px = _direction_model_receptive_field_diameter(model_config)
    dbg_paste_side_px = _dir_vis_half_image_paste_side(image.shape)
    augmented_inputs = _dir_vis_image_space_augmentations(image, valid_mask)
    predictions: list[_DirVisPrediction] = []
    for augment in augmented_inputs:
        predictions.append(
            _DirVisPrediction(
                augment=augment,
                direction_xy=_predict_direction_field(model, augment.image, augment.valid_mask, device=device),
            )
        )
    rendered: list[tuple[str, np.ndarray, int]] = []
    for prediction in predictions:
        panel, drawn = _render_dir_vis_panel(prediction.augment, prediction.direction_xy)
        rendered.append((prediction.augment.name, panel, drawn))
    labels = [name for name, _, _ in rendered]
    panels = [panel for _, panel, _ in rendered]
    drawn_by_augmentation = [(name, int(drawn)) for name, _, drawn in rendered]
    panel_rows = [panels]
    if dbg_dirs:
        pasted_row: list[np.ndarray] = [_render_dir_vis_raw_panel(image)]
        for augment in augmented_inputs[1:]:
            pasted = _paste_unaugmented_center_patch(
                image,
                valid_mask,
                augment,
                paste_side=dbg_paste_side_px,
            )
            direction_xy = _predict_direction_field(model, pasted.image, pasted.valid_mask, device=device)
            panel, drawn = _render_dir_vis_panel(pasted, direction_xy)
            pasted_row.append(panel)
            drawn_by_augmentation.append((pasted.name, int(drawn)))
        panel_rows.append(pasted_row)
        _write_labeled_panel_grid(out / "dir_vis.jpg", panel_rows, labels)
    else:
        _write_labeled_panel_strip(out / "dir_vis.jpg", panels, labels)
    drawn_total = sum(drawn for _, drawn in drawn_by_augmentation)
    augmentation_names = [name for name, _ in drawn_by_augmentation]
    summary = [
        f"sample_index={sample_index}",
        f"checkpoint={Path(checkpoint_path).expanduser().resolve()}",
        f"checkpoint_step={checkpoint.get('step', 'unknown')}",
        f"fiber_path={sample.fiber_path}",
        f"control_point_index={sample.control_point_index}",
        f"strip_z_offset={sample.strip_z_offset}",
        f"image_shape_hw=({int(image.shape[0])}, {int(image.shape[1])})",
        "display_scale=4",
        "display_patch_upsampling=nearest_neighbor",
        "model_inference_resolution=native",
        f"model_receptive_field_radius_px={rf_radius_px}",
        f"model_receptive_field_diameter_px={rf_diameter_px}",
        f"dbg_dirs_paste_side_px={dbg_paste_side_px if dbg_dirs else 0}",
        "direction_stride=2",
        "direction_cell_px=8",
        "direction_segment_px=6",
        "direction_augmentations=" + ",".join(augmentation_names),
        "direction_panels_show_augmented_images=true",
        f"dbg_dirs={bool(dbg_dirs)}",
        f"direction_panel_rows={len(panel_rows)}",
        "direction_row_1=augmented_inputs_with_direction_arrows",
        "direction_row_2=raw_patch_then_augmented_context_with_unaugmented_half_image_center"
        if dbg_dirs
        else "direction_row_2=none",
        "direction_panel_layout=" + ("same_size_grid" if dbg_dirs else "natural_size_horizontal_strip"),
        f"direction_panels_per_row={len(panels)}",
        f"drawn_directions={drawn_total}",
        *[f"drawn_directions_{name}={drawn}" for name, drawn in drawn_by_augmentation],
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
    parser.add_argument(
        "--dbg-dirs",
        action="store_true",
        help="When used with --dir-vis, add a second row of center-pasted direction debug variants",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--line-trace-step", type=float, default=4.0)
    parser.add_argument("--line-trace-tta-count", type=int, default=100)
    parser.add_argument("--med-tta-count", type=int, default=None)
    parser.add_argument("--line-trace-rf-margin", type=float, default=None)
    parser.add_argument("--trace2cp-target-offset", type=int, default=1)
    parser.add_argument("--trace2cp-target-cp-index", type=int, default=None)
    parser.add_argument("--fiber-json", default=None)
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
        if args.fiber_json is not None:
            if args.trace2cp_target_cp_index is not None:
                raise SystemExit("--trace2cp-vis --fiber-json cannot be combined with --trace2cp-target-cp-index")
            if args.vis_tta:
                raise SystemExit("--trace2cp-vis --fiber-json does not support --vis-tta")
            _export_trace2cp_fiber_vis(
                loader,
                args.fiber_json,
                args.export_dir,
                checkpoint_path=args.checkpoint,
                step_px=args.line_trace_step,
                rf_margin_px=args.line_trace_rf_margin,
                target_offset=args.trace2cp_target_offset,
                med_tta=args.med_tta,
                line_trace_tta_count=(
                    args.line_trace_tta_count if args.med_tta_count is None else args.med_tta_count
                ),
            )
            return
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
            dbg_dirs=bool(args.dbg_dirs),
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
