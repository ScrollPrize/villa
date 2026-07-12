from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
    direction_output,
    embedding_output,
    presence_output,
)
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import control_point_line_index


def _path_arg_for_config(path: str | Path) -> str:
    path_s = str(path)
    if path_s.startswith(("s3://", "http://", "https://")):
        return path_s
    return str(Path(path_s).expanduser().resolve())


def _config_for_trace2cp_fiber_json(config, fiber_json: str | Path):
    if len(config.datasets) != 1:
        raise ValueError(
            "--trace2cp-vis --fiber-json requires a config with exactly one dataset entry "
            f"so the volume/manifest context is unambiguous; got {len(config.datasets)}"
        )
    dataset = dict(config.datasets[0])
    dataset["fiber_paths"] = [_path_arg_for_config(fiber_json)]
    dataset.pop("fiber_glob", None)
    return replace(config, datasets=(dataset,))


def _to_u8_image(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[valid] = np.clip(arr[valid], 0.0, 255.0).astype(np.uint8)
    return out


def _trace2cp_z_corrected_image_u8(
    *,
    plane_cache: _Trace2CpZPlaneCache,
    trace_xyz: np.ndarray,
    fallback_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, int, np.ndarray]:
    height, width = (int(v) for v in fallback_shape_hw)
    output = np.zeros((height, width), dtype=np.uint8)
    layers_by_column = np.full((width,), -10_000, dtype=np.int32)
    missing = 0
    trace = np.asarray(trace_xyz, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 3 or trace.shape[0] == 0:
        return output, width, layers_by_column
    layer_images = {
        int(layer): _to_u8_image(prediction.image, prediction.valid_mask)
        for layer, prediction in plane_cache.layers.items()
    }
    for x in range(width):
        point = _trace_xyz_at_x(trace, float(x))
        if point is None or not bool(np.isfinite(point).all()):
            missing += 1
            continue
        z_voxels = float(point[2])
        layer = int(round(z_voxels / float(plane_cache.z_step_voxels)))
        layers_by_column[x] = layer
        image_u8 = layer_images.get(layer)
        if image_u8 is None:
            missing += 1
            continue
        if image_u8.shape != output.shape:
            missing += 1
            continue
        output[:, x] = image_u8[:, x]
    return output, missing, layers_by_column


def _trace2cp_refinement_fused_xyz(refinement: _Trace2CpRefinementResult) -> np.ndarray:
    xy = np.asarray(refinement.fused_resampled_xy, dtype=np.float32)
    z = refinement.fused_resampled_z
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0 or z is None:
        return np.zeros((0, 3), dtype=np.float32)
    z_arr = np.asarray(z, dtype=np.float32).reshape(-1)
    if int(z_arr.shape[0]) != int(xy.shape[0]):
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate([xy, z_arr[:, None]], axis=1).astype(np.float32, copy=False)


def _z_layer_columns_to_rgb(
    layers_by_column: np.ndarray,
    *,
    height: int,
    max_layer: int,
) -> np.ndarray:
    layers = np.asarray(layers_by_column, dtype=np.int32).reshape(-1)
    width = int(layers.shape[0])
    out = np.zeros((max(1, int(height)), width, 3), dtype=np.uint8)
    bound = max(1, int(max_layer))
    valid = layers > -9999
    if not bool(np.any(valid)):
        return out
    normalized = np.clip(layers.astype(np.float32) / float(bound), -1.0, 1.0)
    red = np.clip(normalized, 0.0, 1.0)
    blue = np.clip(-normalized, 0.0, 1.0)
    green = 1.0 - np.abs(normalized)
    colors = np.stack([red, green, blue], axis=1)
    colors = (colors * 255.0).clip(0.0, 255.0).astype(np.uint8)
    out[:, valid, :] = colors[valid][None, :, :]
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


def _bilinear_embedding_sample(
    embedding_chw: np.ndarray,
    point_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    field = np.asarray(embedding_chw, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError("embedding_chw must have shape C,H,W")
    channels, height, width = (int(v) for v in field.shape)
    if channels <= 0:
        raise ValueError("embedding_chw must contain at least one channel")
    point = np.asarray(point_xy, dtype=np.float32)
    if point.shape != (2,) or not bool(np.isfinite(point).all()):
        return None
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
            raise ValueError("valid_mask must match embedding field shape")
        if not (bool(valid[y0, x0]) and bool(valid[y0, x1]) and bool(valid[y1, x0]) and bool(valid[y1, x1])):
            return None
    tx = np.float32(x - float(x0))
    ty = np.float32(y - float(y0))
    top = field[:, y0, x0] * (1.0 - tx) + field[:, y0, x1] * tx
    bottom = field[:, y1, x0] * (1.0 - tx) + field[:, y1, x1] * tx
    embedding = top * (1.0 - ty) + bottom * ty
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        return None
    return (embedding / norm).astype(np.float32)


def _bilinear_scalar_sample(
    field_hw: np.ndarray,
    point_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> float | None:
    field = np.asarray(field_hw, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError("field_hw must have shape H,W")
    height, width = (int(v) for v in field.shape)
    point = np.asarray(point_xy, dtype=np.float32)
    if point.shape != (2,) or not bool(np.isfinite(point).all()):
        return None
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
            raise ValueError("valid_mask must match field shape")
        if not (bool(valid[y0, x0]) and bool(valid[y0, x1]) and bool(valid[y1, x0]) and bool(valid[y1, x1])):
            return None
    tx = np.float32(x - float(x0))
    ty = np.float32(y - float(y0))
    top = field[y0, x0] * (1.0 - tx) + field[y0, x1] * tx
    bottom = field[y1, x0] * (1.0 - tx) + field[y1, x1] * tx
    value = float(top * (1.0 - ty) + bottom * ty)
    if not np.isfinite(value):
        return None
    return value


def _embedding_cosine_loss(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32).reshape(-1)
    rhs = np.asarray(b, dtype=np.float32).reshape(-1)
    if lhs.shape != rhs.shape or lhs.shape[0] == 0:
        raise ValueError("embedding vectors must have the same non-empty shape")
    if not bool(np.isfinite(lhs).all()) or not bool(np.isfinite(rhs).all()):
        return float("inf")
    return float(1.0 - np.clip(float(np.dot(lhs, rhs)), -1.0, 1.0))


def _trace2cp_unit_or_none(vector_xy: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector_xy, dtype=np.float32)
    if vector.shape != (2,):
        raise ValueError("vector_xy must have shape (2,)")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        return None
    return (vector / norm).astype(np.float32)


def _trace2cp_descriptor_across_axis(along_xy: np.ndarray) -> np.ndarray:
    along = np.asarray(along_xy, dtype=np.float32)
    if along.shape != (2,):
        raise ValueError("along_xy must have shape (2,)")
    across = np.asarray([-along[1], along[0]], dtype=np.float32)
    # Stable across-axis convention: reversing the fiber direction should only
    # horizontally flip the descriptor, not rotate the local descriptor frame.
    if float(across[1]) < -1.0e-6 or (abs(float(across[1])) <= 1.0e-6 and float(across[0]) < 0.0):
        across = -across
    norm = float(np.linalg.norm(across))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        raise ValueError("could not build descriptor across axis")
    return (across / norm).astype(np.float32)


def _trace2cp_image_descriptor_spatial_weights(config: _Trace2CpImageScoringConfig) -> np.ndarray:
    along = max(1, int(config.patch_along))
    across = max(1, int(config.patch_across))
    x = np.arange(along, dtype=np.float32) - np.float32((along - 1) * 0.5)
    y = np.arange(across, dtype=np.float32) - np.float32((across - 1) * 0.5)
    sigma_x = max(float(along) * 0.25, 1.0)
    sigma_y = max(float(across) * 0.25, 1.0)
    weights = np.exp(
        -0.5
        * (
            (y[:, None] / np.float32(sigma_y)) ** 2
            + (x[None, :] / np.float32(sigma_x)) ** 2
        )
    ).astype(np.float32)
    weights /= np.float32(max(float(np.max(weights)), 1.0e-12))
    return weights


def _trace2cp_image_descriptor_blur_kernel(radius: int) -> np.ndarray:
    r = max(0, int(radius))
    if r <= 0:
        return np.asarray([1.0], dtype=np.float32)
    x = np.arange(-r, r + 1, dtype=np.float32)
    sigma = max(float(r) / 2.0, 1.0)
    kernel = np.exp(-0.5 * (x / np.float32(sigma)) ** 2).astype(np.float32)
    kernel /= np.float32(max(float(np.sum(kernel)), 1.0e-12))
    return kernel


def _trace2cp_convolve_same_length(row: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.float32).reshape(-1)
    filt = np.asarray(kernel, dtype=np.float32).reshape(-1)
    if filt.size <= 1:
        return values.astype(np.float32, copy=True)
    full = np.convolve(values, filt, mode="full").astype(np.float32)
    start = int((filt.size - 1) // 2)
    return full[start : start + values.size].astype(np.float32, copy=False)


def _trace2cp_bilinear_image_grid(
    image_hw: np.ndarray,
    valid_mask: np.ndarray,
    xy_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image_hw, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    grid = np.asarray(xy_grid, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image_hw must have shape H,W")
    if valid.shape != image.shape:
        raise ValueError("valid_mask must match image shape")
    if grid.ndim != 3 or grid.shape[2] != 2:
        raise ValueError("xy_grid must have shape H,W,2")
    height, width = image.shape
    x = grid[..., 0]
    y = grid[..., 1]
    in_bounds = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (y >= 0.0)
        & (x <= np.float32(width - 1))
        & (y <= np.float32(height - 1))
    )
    x_safe = np.clip(np.where(in_bounds, x, 0.0), 0.0, float(width - 1))
    y_safe = np.clip(np.where(in_bounds, y, 0.0), 0.0, float(height - 1))
    x0 = np.floor(x_safe).astype(np.int64)
    y0 = np.floor(y_safe).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    corners_valid = valid[y0, x0] & valid[y0, x1] & valid[y1, x0] & valid[y1, x1]
    sample_valid = in_bounds & corners_valid
    tx = (x_safe - x0.astype(np.float32)).astype(np.float32)
    ty = (y_safe - y0.astype(np.float32)).astype(np.float32)
    top = image[y0, x0] * (1.0 - tx) + image[y0, x1] * tx
    bottom = image[y1, x0] * (1.0 - tx) + image[y1, x1] * tx
    values = (top * (1.0 - ty) + bottom * ty).astype(np.float32)
    values[~sample_valid] = 0.0
    return values, sample_valid.astype(bool)


def _trace2cp_blur_descriptor_along_axis(
    values: np.ndarray,
    valid: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.float32)
    if vals.shape != mask.shape:
        raise ValueError("descriptor values and valid mask must have matching shapes")
    kernel = _trace2cp_image_descriptor_blur_kernel(radius)
    if kernel.size <= 1:
        return vals.astype(np.float32, copy=True), mask > 0.5
    blurred = np.zeros_like(vals, dtype=np.float32)
    support = np.zeros_like(vals, dtype=np.float32)
    for row in range(int(vals.shape[0])):
        numerator = _trace2cp_convolve_same_length(vals[row] * mask[row], kernel)
        denominator = _trace2cp_convolve_same_length(mask[row], kernel)
        row_valid = denominator > 1.0e-6
        blurred[row, row_valid] = numerator[row_valid] / denominator[row_valid]
        support[row] = denominator
    return blurred, support > 1.0e-6


def _trace2cp_image_descriptor(
    image_hw: np.ndarray,
    valid_mask: np.ndarray,
    point_xy: np.ndarray,
    along_xy: np.ndarray,
    config: _Trace2CpImageScoringConfig | None = None,
) -> _Trace2CpImageDescriptor | None:
    cfg = config or _Trace2CpImageScoringConfig()
    along = _trace2cp_unit_or_none(along_xy)
    if along is None:
        return None
    point = np.asarray(point_xy, dtype=np.float32)
    if point.shape != (2,) or not bool(np.isfinite(point).all()):
        return None
    patch_along = max(1, int(cfg.patch_along))
    patch_across = max(1, int(cfg.patch_across))
    along_offsets = np.arange(patch_along, dtype=np.float32) - np.float32((patch_along - 1) * 0.5)
    across_offsets = np.arange(patch_across, dtype=np.float32) - np.float32((patch_across - 1) * 0.5)
    across = _trace2cp_descriptor_across_axis(along)
    xy = (
        point[None, None, :]
        + across_offsets[:, None, None] * across[None, None, :]
        + along_offsets[None, :, None] * along[None, None, :]
    ).astype(np.float32)
    sampled, sample_valid = _trace2cp_bilinear_image_grid(image_hw, valid_mask, xy)
    valid_fraction = float(np.mean(sample_valid)) if sample_valid.size else 0.0
    if valid_fraction < float(cfg.min_valid_fraction):
        return None
    blurred, blurred_valid = _trace2cp_blur_descriptor_along_axis(sampled, sample_valid, int(cfg.blur_radius))
    spatial = _trace2cp_image_descriptor_spatial_weights(cfg)
    weights = spatial * blurred_valid.astype(np.float32)
    if float(np.sum(weights)) <= 1.0e-6:
        return None
    values = (blurred / np.float32(255.0)).astype(np.float32)
    return _Trace2CpImageDescriptor(values=values, weights=weights.astype(np.float32))


def _trace2cp_image_descriptor_loss(
    candidate: _Trace2CpImageDescriptor,
    reference: _Trace2CpImageDescriptor,
) -> float:
    cand_values = np.asarray(candidate.values, dtype=np.float32)
    cand_weights = np.asarray(candidate.weights, dtype=np.float32)
    ref_values = np.asarray(reference.values, dtype=np.float32)
    ref_weights = np.asarray(reference.weights, dtype=np.float32)
    if (
        cand_values.shape != ref_values.shape
        or cand_weights.shape != cand_values.shape
        or ref_weights.shape != ref_values.shape
    ):
        raise ValueError("image descriptors must have matching value/weight shapes")
    losses: list[float] = []
    for values, weights in (
        (ref_values, ref_weights),
        (ref_values[:, ::-1], ref_weights[:, ::-1]),
    ):
        combined_weights = cand_weights * weights
        denom = float(np.sum(combined_weights))
        if denom <= 1.0e-6:
            losses.append(float("inf"))
            continue
        diff = cand_values - values
        losses.append(float(np.sum(combined_weights * diff * diff) / denom))
    return float(min(losses))


def _normalized_embedding_field(embedding_chw: np.ndarray) -> np.ndarray:
    field = _require_trace2cp_embedding_field(embedding_chw)
    norm = np.linalg.norm(field, axis=0, keepdims=True)
    finite = np.isfinite(field).all(axis=0, keepdims=True) & np.isfinite(norm) & (norm > 1.0e-6)
    return np.where(finite, field / np.clip(norm, 1.0e-12, None), 0.0).astype(np.float32)


def _embedding_similarity_map(
    normalized_embedding_chw: np.ndarray,
    reference_embedding: np.ndarray,
    *,
    valid_mask: np.ndarray | None,
    normalize_reference: bool = True,
) -> np.ndarray:
    field = _require_trace2cp_embedding_field(normalized_embedding_chw)
    ref = np.asarray(reference_embedding, dtype=np.float32).reshape(-1)
    if int(ref.shape[0]) != int(field.shape[0]):
        raise ValueError("reference embedding channel count must match embedding field")
    if not bool(np.isfinite(ref).all()):
        raise ValueError("reference embedding must be finite")
    if normalize_reference:
        ref_norm = float(np.linalg.norm(ref))
        if not np.isfinite(ref_norm) or ref_norm <= 1.0e-6:
            raise ValueError("reference embedding must be non-zero")
        ref = (ref / np.float32(ref_norm)).astype(np.float32)
    similarity = np.sum(field * ref[:, None, None], axis=0, dtype=np.float32)
    similarity = np.clip(similarity, -1.0, 1.0).astype(np.float32)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != similarity.shape:
            raise ValueError("valid_mask must match embedding spatial shape")
        similarity = np.where(valid, similarity, np.nan).astype(np.float32)
    return similarity


def _trace_progress_similarity_map(
    normalized_embedding_chw: np.ndarray,
    trace_xy: np.ndarray,
    *,
    valid_mask: np.ndarray,
    step_px: float,
) -> np.ndarray | None:
    field = _require_trace2cp_embedding_field(normalized_embedding_chw)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != tuple(int(v) for v in field.shape[1:]):
        raise ValueError("valid_mask must match embedding spatial shape")
    points = np.asarray(trace_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("trace_xy must have shape N,2")
    height = int(field.shape[1])
    width = int(field.shape[2])
    radius = max(1, int(math.ceil(max(float(step_px), 1.0e-3) * 0.5)))
    painted = np.full((height, width), np.nan, dtype=np.float32)
    wrote = False
    for index in range(1, int(points.shape[0])):
        point = points[index]
        previous_point = points[index - 1]
        if (
            point.shape != (2,)
            or previous_point.shape != (2,)
            or not bool(np.isfinite(point).all())
            or not bool(np.isfinite(previous_point).all())
        ):
            continue
        reference = _bilinear_embedding_sample(field, previous_point, valid_mask=valid)
        if reference is None:
            continue
        center_x = int(round(float(point[0])))
        if center_x < 0 or center_x >= width:
            continue
        x0 = max(0, center_x - radius)
        x1 = min(width, center_x + radius + 1)
        band = np.sum(field[:, :, x0:x1] * reference[:, None, None], axis=0, dtype=np.float32)
        band = np.clip(band, -1.0, 1.0).astype(np.float32)
        painted[:, x0:x1] = np.where(valid[:, x0:x1], band, np.nan).astype(np.float32)
        wrote = True
    return painted if wrote else None


def _trace2cp_similarity_debug(
    embedding_chw: np.ndarray | None,
    valid_mask: np.ndarray,
    *,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    forward_trace_xy: np.ndarray,
    reverse_trace_xy: np.ndarray,
    step_px: float = 1.0,
    fiber_embeddings: np.ndarray | None = None,
) -> _Trace2CpSimilarityDebug | None:
    if embedding_chw is None:
        return None
    embedding = _require_trace2cp_embedding_field(embedding_chw)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != tuple(int(v) for v in embedding.shape[1:]):
        raise ValueError("valid_mask must match embedding spatial shape")
    start_embedding = _bilinear_embedding_sample(embedding, start_xy, valid_mask=valid)
    target_embedding = _bilinear_embedding_sample(embedding, target_xy, valid_mask=valid)
    if start_embedding is None or target_embedding is None:
        return None

    normalized = _normalized_embedding_field(embedding)
    global_similarity: np.ndarray | None = None
    global_bank_size = 0
    if fiber_embeddings is not None:
        bank = np.asarray(fiber_embeddings, dtype=np.float32)
        if bank.ndim == 2 and bank.shape[0] > 0 and bank.shape[1] == embedding.shape[0]:
            bank_norm = np.linalg.norm(bank, axis=1, keepdims=True)
            bank_valid = np.isfinite(bank).all(axis=1, keepdims=True) & np.isfinite(bank_norm) & (bank_norm > 1.0e-6)
            normalized_bank = np.where(bank_valid, bank / np.clip(bank_norm, 1.0e-12, None), 0.0).astype(np.float32)
            normalized_bank = normalized_bank[bank_valid[:, 0]]
            global_bank_size = int(normalized_bank.shape[0])
            if global_bank_size > 0:
                mean_reference = np.mean(normalized_bank, axis=0, dtype=np.float32)
                global_similarity = _embedding_similarity_map(
                    normalized,
                    mean_reference,
                    valid_mask=valid,
                    normalize_reference=False,
                )

    return _Trace2CpSimilarityDebug(
        start_cp_similarity=_embedding_similarity_map(normalized, start_embedding, valid_mask=valid),
        target_cp_similarity=_embedding_similarity_map(normalized, target_embedding, valid_mask=valid),
        global_similarity=global_similarity,
        forward_last_similarity=_trace_progress_similarity_map(
            normalized,
            forward_trace_xy,
            valid_mask=valid,
            step_px=step_px,
        ),
        reverse_last_similarity=_trace_progress_similarity_map(
            normalized,
            reverse_trace_xy,
            valid_mask=valid,
            step_px=step_px,
        ),
        global_bank_size=global_bank_size,
    )


def _trace2cp_candidate_angles_degrees(max_degrees: float, step_degrees: float) -> np.ndarray:
    maximum = float(max_degrees)
    step = float(step_degrees)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError("candidate max degrees must be finite and >= 0")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("candidate step degrees must be finite and > 0")
    count = int(np.floor(maximum / step + 1.0e-6))
    values = (np.arange(-count, count + 1, dtype=np.float32) * np.float32(step)).astype(np.float32)
    if values.size == 0:
        values = np.asarray([0.0], dtype=np.float32)
    return values[np.abs(values) <= np.float32(maximum + 1.0e-5)].astype(np.float32, copy=False)


def _trace2cp_candidate_fan_directions(
    oriented_direction_xy: np.ndarray,
    *,
    max_degrees: float,
    step_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(oriented_direction_xy, dtype=np.float32)
    if direction.shape != (2,):
        raise ValueError("oriented_direction_xy must have shape (2,)")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-6:
        raise ValueError("oriented_direction_xy must be finite and non-zero")
    base = (direction / norm).astype(np.float32)
    angles = _trace2cp_candidate_angles_degrees(max_degrees, step_degrees)
    radians = np.deg2rad(angles.astype(np.float64)).astype(np.float32)
    cos = np.cos(radians).astype(np.float32)
    sin = np.sin(radians).astype(np.float32)
    x = base[0] * cos - base[1] * sin
    y = base[0] * sin + base[1] * cos
    vectors = np.stack([x, y], axis=1).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(1.0e-12).astype(np.float32)
    return angles, vectors


def _trace2cp_target_trace_max_steps(
    shape_hw: tuple[int, int],
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    step_px: float,
) -> int:
    height, width = (int(v) for v in shape_hw)
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("start_xy and target_xy must have shape (2,)")
    step = max(float(step_px), 1.0e-3)
    dx = abs(float(target[0]) - float(start[0]))
    dy = abs(float(target[1]) - float(start[1]))
    base_distance = max(float(np.hypot(height, width)), float(dx), float(dx + 4.0 * dy))
    # Trace2CP should stop by reaching the target x-column or a visible
    # validity/margin failure. The budget is deliberately generous so curved
    # traces do not silently stop short and get scored by the missing-column
    # fallback metric.
    return int(np.ceil((base_distance / step) * 8.0)) + 256


def _trace2cp_target_column_point(
    current_xy: np.ndarray,
    next_xy: np.ndarray,
    target_x: float,
) -> np.ndarray:
    current = np.asarray(current_xy, dtype=np.float32)
    nxt = np.asarray(next_xy, dtype=np.float32)
    if current.shape != (2,) or nxt.shape != (2,):
        raise ValueError("current_xy and next_xy must have shape (2,)")
    x0 = float(current[0])
    x1 = float(nxt[0])
    if abs(x1 - x0) <= 1.0e-6:
        return nxt.astype(np.float32, copy=True)
    t = np.float32(np.clip((float(target_x) - x0) / (x1 - x0), 0.0, 1.0))
    point = current + (nxt - current) * t
    point[0] = np.float32(target_x)
    return point.astype(np.float32)


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
class _Trace2CpPredictedFields:
    direction_xy: np.ndarray
    embedding_chw: np.ndarray | None
    presence_hw: np.ndarray | None = None


@dataclass(frozen=True)
class _Trace2CpCombinedWeights:
    direction: float = 1.0
    last: float = 1.0
    enclosing: float = 1.0
    fiber: float = 1.0
    image: float = 1.0
    presence: float = 0.0


@dataclass(frozen=True)
class _Trace2CpImageScoringConfig:
    patch_along: int = 16
    patch_across: int = 8
    blur_radius: int = 5
    min_valid_fraction: float = 0.75


@dataclass(frozen=True)
class _Trace2CpImageDescriptor:
    values: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class _Trace2CpCombinedTraceStats:
    steps: int
    invalid_candidates: int
    direction_loss_sum: float
    last_loss_sum: float
    enclosing_loss_sum: float
    fiber_loss_sum: float
    total_loss_sum: float
    reason: str
    image_loss_sum: float = 0.0
    presence_loss_sum: float = 0.0

    def mean(self, name: str) -> float:
        count = max(1, int(self.steps))
        return float(getattr(self, f"{name}_loss_sum") / count)


@dataclass(frozen=True)
class _Trace2CpCombinedSummary:
    forward: _Trace2CpCombinedTraceStats
    reverse: _Trace2CpCombinedTraceStats
    candidate_angles_degrees: np.ndarray
    fiber_bank_size: int
    fiber_bank_skipped: int

    @property
    def steps(self) -> int:
        return int(self.forward.steps) + int(self.reverse.steps)

    @property
    def invalid_candidates(self) -> int:
        return int(self.forward.invalid_candidates) + int(self.reverse.invalid_candidates)

    def mean(self, name: str) -> float:
        steps = max(1, int(self.steps))
        total = float(getattr(self.forward, f"{name}_loss_sum")) + float(
            getattr(self.reverse, f"{name}_loss_sum")
        )
        return total / steps


@dataclass(frozen=True)
class _Trace2CpZLayerPrediction:
    layer: int
    z_voxels: float
    sample: FiberStripSegmentSample
    image: np.ndarray
    valid_mask: np.ndarray
    fields: "_Trace2CpPredictedFields"


@dataclass(frozen=True)
class _Trace2CpZSearchConfig:
    enabled: bool = False
    step_voxels: float = 2.0
    max_layer: int = 4


@dataclass(frozen=True)
class _Trace2CpZTraceDebug:
    forward_trace_xyz: np.ndarray
    reverse_trace_xyz: np.ndarray
    fused_trace_xyz: np.ndarray
    forward_z_image: np.ndarray
    reverse_z_image: np.ndarray
    fused_z_image: np.ndarray
    forward_missing_columns: int
    reverse_missing_columns: int
    fused_missing_columns: int
    forward_layer_columns: np.ndarray
    reverse_layer_columns: np.ndarray
    fused_layer_columns: np.ndarray
    layers: tuple[int, ...]
    z_step_voxels: float
    max_layer: int

    @property
    def layer_min(self) -> int:
        return min(self.layers) if self.layers else 0

    @property
    def layer_max(self) -> int:
        return max(self.layers) if self.layers else 0

    def z_stats(self, trace_xyz: np.ndarray) -> tuple[float, float, float]:
        trace = np.asarray(trace_xyz, dtype=np.float32)
        if trace.ndim != 2 or trace.shape[1] != 3 or trace.shape[0] == 0:
            return 0.0, 0.0, 0.0
        values = trace[:, 2]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 0.0, 0.0
        return float(np.min(finite)), float(np.max(finite)), float(np.mean(np.abs(finite)))


@dataclass(frozen=True)
class _Trace2CpEmbeddingBank:
    embeddings: np.ndarray
    skipped: int
    rows: tuple[str, ...]


@dataclass(frozen=True)
class _Trace2CpSimilarityDebug:
    start_cp_similarity: np.ndarray
    target_cp_similarity: np.ndarray
    global_similarity: np.ndarray | None
    forward_last_similarity: np.ndarray | None
    reverse_last_similarity: np.ndarray | None
    global_bank_size: int


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
    partial_forward_z: np.ndarray | None = None
    partial_reverse_z: np.ndarray | None = None
    fused_dense_z: np.ndarray | None = None
    fused_resampled_z: np.ndarray | None = None
    optimized_z: np.ndarray | None = None
    closest_forward_z_voxels: float = 0.0
    closest_reverse_z_voxels: float = 0.0
    closest_midpoint_z_voxels: float = 0.0


@dataclass(frozen=True)
class _Trace2CpMetricResult:
    error: float
    raw_y_error_px: float
    horizontal_span_px: float
    max_y_error_px: float
    forward_target_x: float
    reverse_target_x: float
    forward_y_at_target_x: float
    reverse_y_at_target_x: float
    reached_target_columns: bool
    closest_x: float
    forward_y_at_closest_x: float
    reverse_y_at_closest_x: float
    closest_midpoint_xy: np.ndarray
    reached_overlap: bool
    reason: str


@dataclass(frozen=True)
class _Trace2CpBidirectionalResult:
    forward: _Trace2CpDirectionResult
    reverse: _Trace2CpDirectionResult
    refinement: _Trace2CpRefinementResult
    metric: _Trace2CpMetricResult

    @property
    def score(self) -> float:
        return float(self.refinement.score)

    @property
    def metric_error(self) -> float:
        return float(self.metric.error)

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
    combined_summary: _Trace2CpCombinedSummary | None
    tta_count: int
    med_fields_count: int
    tta_rows: tuple[str, ...]
    tta_debug_entries: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]
    similarity_debug: _Trace2CpSimilarityDebug | None = None
    z_search_debug: _Trace2CpZTraceDebug | None = None
    presence_debug: np.ndarray | None = None


@dataclass(frozen=True)
class _Trace2CpFiberPairPlacement:
    evaluation: _Trace2CpPairEvaluation
    x_scale: float
    x_offset: float
    y_shift: float


@dataclass(frozen=True)
class _Trace2CpSkippedPair:
    start_cp_index: int
    target_cp_index: int
    reason: str


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
        presence_channels=1 if bool(training.get("presence_enabled", False)) else 0,
        embedding_channels=max(0, int(training.get("contrastive_embedding_channels", 0))),
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
    fields = _predict_trace2cp_fields(model, image, valid_mask, device=device)
    return fields.direction_xy


def _predict_trace2cp_fields(
    model: FiberStripDirectionNet,
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    device: torch.device,
) -> _Trace2CpPredictedFields:
    with torch.no_grad():
        input_image = _prepare_model_image(image, valid_mask, device=device)
        output = model(input_image)
        encoded = direction_output(output)[0].permute(1, 2, 0)
        direction_xy = decode_lasagna_direction_xy(encoded).detach().cpu().numpy().astype(np.float32)
        presence_channels = int(getattr(model, "presence_channels", 0))
        presence_hw: np.ndarray | None
        if presence_channels <= 0:
            presence_hw = None
        else:
            presence_hw = presence_output(output, presence_channels=presence_channels)[0, 0].detach().cpu().numpy().astype(np.float32)
        embeddings = embedding_output(output, presence_channels=presence_channels)
        embedding_chw: np.ndarray | None
        if int(embeddings.shape[1]) <= 0:
            embedding_chw = None
        else:
            embedding_chw = embeddings[0].detach().cpu().numpy().astype(np.float32)
        return _Trace2CpPredictedFields(
            direction_xy=direction_xy,
            embedding_chw=embedding_chw,
            presence_hw=presence_hw,
        )


class _Trace2CpZPlaneCache:
    def __init__(
        self,
        *,
        loader: FiberStrip2DLoader,
        model: torch.nn.Module,
        sample_index: int,
        target_cp_index: int | None,
        target_offset: int,
        rf_margin_px: float,
        sample_mode: str,
        row_axis_alignment_line_index: int | None,
        row_axis_alignment_xyz: np.ndarray | None,
        device: torch.device,
        segment_source: Any,
        center_layer: _Trace2CpZLayerPrediction,
        z_step_voxels: float,
        max_layer: int,
    ) -> None:
        step = float(z_step_voxels)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("trace2cp z step must be finite and > 0")
        bound = int(max_layer)
        if bound < 1:
            raise ValueError("trace2cp z max layer must be >= 1")
        self.loader = loader
        self.model = model
        self.sample_index = int(sample_index)
        self.target_cp_index = target_cp_index
        self.target_offset = int(target_offset)
        self.rf_margin_px = float(rf_margin_px)
        self.sample_mode = sample_mode
        self.row_axis_alignment_line_index = row_axis_alignment_line_index
        self.row_axis_alignment_xyz = (
            None
            if row_axis_alignment_xyz is None
            else np.asarray(row_axis_alignment_xyz, dtype=np.float32).copy()
        )
        self.device = device
        self.segment_source = segment_source
        self.z_step_voxels = step
        self.max_layer = bound
        self.center_offset = float(center_layer.sample.strip_z_offset)
        self.layers: dict[int, _Trace2CpZLayerPrediction] = {0: center_layer}

    def has_layer(self, layer: int) -> bool:
        return int(layer) in self.layers

    def get(self, layer: int) -> _Trace2CpZLayerPrediction:
        layer_i = int(layer)
        if abs(layer_i) > self.max_layer:
            raise ValueError(
                f"trace2cp z layer {layer_i} outside configured bound +/-{self.max_layer}"
            )
        existing = self.layers.get(layer_i)
        if existing is not None:
            return existing
        offset = self.center_offset + float(layer_i) * self.z_step_voxels
        sample, image, valid_mask = self.loader.sample_trace2cp_segment_source(
            self.segment_source,
            strip_z_offset=offset,
        )
        fields = _predict_trace2cp_fields(self.model, image, valid_mask, device=self.device)
        prediction = _Trace2CpZLayerPrediction(
            layer=layer_i,
            z_voxels=float(layer_i) * self.z_step_voxels,
            sample=sample,
            image=np.asarray(image, dtype=np.float32),
            valid_mask=np.asarray(valid_mask, dtype=bool),
            fields=fields,
        )
        self.layers[layer_i] = prediction
        return prediction

    def ensure_neighbors(self, layer: int) -> None:
        center = int(layer)
        for candidate in (center - 1, center, center + 1):
            if abs(candidate) <= self.max_layer:
                self.get(candidate)

    def layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(int(v) for v in self.layers))


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
        max_steps = _trace2cp_target_trace_max_steps(shape_hw, start, target, step)
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
            points.append(_trace2cp_target_column_point(current, next_point, target_x))
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
        max_steps = _trace2cp_target_trace_max_steps(shape_hw, start, target, step)
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
            points.append(_trace2cp_target_column_point(current, next_point, target_x))
            return np.stack(points, axis=0).astype(np.float32), "target_column"
        if not _inside_trace_margin(next_point, shape_hw, rf_margin_px):
            return np.stack(points, axis=0).astype(np.float32), _trace_margin_reason(
                next_point, shape_hw, rf_margin_px
            )
        points.append(next_point.astype(np.float32))
        current = next_point.astype(np.float32)
        previous = sampled.astype(np.float32)
    return np.stack(points, axis=0).astype(np.float32), "max_steps"


def _require_trace2cp_embedding_field(embedding_chw: np.ndarray | None) -> np.ndarray:
    if embedding_chw is None:
        raise ValueError("trace2cp combined tracing requires model embedding channels")
    emb = np.asarray(embedding_chw, dtype=np.float32)
    if emb.ndim != 3 or int(emb.shape[0]) <= 0:
        raise ValueError("trace2cp combined tracing requires model embedding channels")
    return emb


def _require_trace2cp_presence_field(presence_hw: np.ndarray | None) -> np.ndarray:
    if presence_hw is None:
        raise ValueError("trace2cp combined presence scoring requires a model presence channel")
    presence = np.asarray(presence_hw, dtype=np.float32)
    if presence.ndim != 2:
        raise ValueError("trace2cp combined presence scoring requires a 2D presence field")
    return presence


def _trace2cp_presence_loss(
    presence_hw: np.ndarray | None,
    point_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None,
) -> float | None:
    presence = _require_trace2cp_presence_field(presence_hw)
    sampled = _bilinear_scalar_sample(presence, point_xy, valid_mask=valid_mask)
    if sampled is None:
        return None
    return float(1.0 - np.clip(float(sampled), 0.0, 1.0))


def _sample_trace2cp_combined_direction(
    *,
    direction_xy: np.ndarray | None,
    tta_fields: list[_TtaDirectionField] | None,
    current_xy: np.ndarray,
    previous_xy: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray | None:
    if tta_fields is not None:
        return _sample_tta_direction_in_reference(tta_fields, current_xy, previous_xy)
    if direction_xy is None:
        raise ValueError("direction_xy is required when tta_fields are not provided")
    sampled = _bilinear_direction_sample(direction_xy, current_xy, valid_mask=valid_mask)
    if sampled is None:
        return None
    return _orient_direction_to_previous(sampled, previous_xy)


def _trace_combined_direction_line_to_target(
    *,
    direction_xy: np.ndarray | None,
    tta_fields: list[_TtaDirectionField] | None,
    embedding_chw: np.ndarray | None,
    valid_mask: np.ndarray | None,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    start_embedding: np.ndarray | None,
    target_embedding: np.ndarray | None,
    fiber_embeddings: np.ndarray | None,
    weights: _Trace2CpCombinedWeights,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    presence_hw: np.ndarray | None = None,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
    max_steps: int | None = None,
) -> tuple[np.ndarray, str, _Trace2CpCombinedTraceStats]:
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("start_xy and target_xy must have shape (2,)")
    target_delta = target - start
    target_norm = float(np.linalg.norm(target_delta))
    if not np.isfinite(target_norm) or target_norm <= 1.0e-6:
        raise ValueError("trace2cp start and target points must be distinct")

    embedding_required = any(
        float(value) != 0.0 for value in (weights.last, weights.enclosing, weights.fiber)
    )
    presence_required = float(weights.presence) != 0.0
    embedding: np.ndarray | None = None
    if embedding_required:
        embedding = _require_trace2cp_embedding_field(embedding_chw)
        shape_hw = (int(embedding.shape[1]), int(embedding.shape[2]))
    elif direction_xy is not None:
        direction_arr = np.asarray(direction_xy, dtype=np.float32)
        if direction_arr.ndim != 3 or direction_arr.shape[2] != 2:
            raise ValueError("direction_xy must have shape H,W,2")
        shape_hw = (int(direction_arr.shape[0]), int(direction_arr.shape[1]))
    elif presence_required:
        presence_arr = _require_trace2cp_presence_field(presence_hw)
        shape_hw = (int(presence_arr.shape[0]), int(presence_arr.shape[1]))
    elif tta_fields:
        shape_hw = tuple(int(v) for v in np.asarray(tta_fields[0].direction_xy).shape[:2])
    elif embedding_chw is not None:
        fallback_embedding = _require_trace2cp_embedding_field(embedding_chw)
        shape_hw = (int(fallback_embedding.shape[1]), int(fallback_embedding.shape[2]))
    else:
        raise ValueError("trace2cp combined direction tracing requires a direction, TTA, presence, or embedding field")
    if direction_xy is not None:
        direction = np.asarray(direction_xy, dtype=np.float32)
        if direction.ndim != 3 or direction.shape[2] != 2:
            raise ValueError("direction_xy must have shape H,W,2")
        if tuple(int(v) for v in direction.shape[:2]) != tuple(shape_hw):
            raise ValueError("direction_xy and embedding_chw spatial shapes must match")
    if valid_mask is not None and tuple(int(v) for v in np.asarray(valid_mask).shape) != tuple(shape_hw):
        raise ValueError("valid_mask shape must match embedding field shape")
    presence: np.ndarray | None = None
    if presence_required:
        presence = _require_trace2cp_presence_field(presence_hw)
        if tuple(int(v) for v in presence.shape) != tuple(shape_hw):
            raise ValueError("presence_hw spatial shape must match trace2cp fields")

    if embedding_required:
        assert embedding is not None
        bank = np.asarray(
            np.zeros((0, int(embedding.shape[0])), dtype=np.float32)
            if fiber_embeddings is None
            else fiber_embeddings,
            dtype=np.float32,
        )
        if bank.size == 0:
            bank = bank.reshape(0, int(embedding.shape[0]))
        if bank.ndim != 2 or int(bank.shape[1]) != int(embedding.shape[0]):
            raise ValueError("fiber_embeddings must have shape N,C matching embedding_chw")
        if float(weights.fiber) != 0.0 and int(bank.shape[0]) == 0:
            raise ValueError("trace2cp combined fiber weight is non-zero but the fiber CP embedding bank is empty")
        if start_embedding is None or target_embedding is None:
            raise ValueError("trace2cp embedding scoring requires start and target embeddings")
        start_emb = np.asarray(start_embedding, dtype=np.float32).reshape(-1)
        target_emb = np.asarray(target_embedding, dtype=np.float32).reshape(-1)
        if start_emb.shape[0] != int(embedding.shape[0]) or target_emb.shape[0] != int(embedding.shape[0]):
            raise ValueError("start/target embeddings must match embedding_chw channel count")
    else:
        bank = np.zeros((0, 0), dtype=np.float32)
        start_emb = None
        target_emb = None

    step = max(float(step_px), 1.0e-3)
    if max_steps is None:
        max_steps = _trace2cp_target_trace_max_steps(shape_hw, start, target, step)
    current = start.astype(np.float32)
    previous_direction = (target_delta / target_norm).astype(np.float32)
    previous_embedding = None if start_emb is None else start_emb.astype(np.float32)
    points = [current.copy()]
    target_x = float(target[0])

    steps = 0
    invalid_candidates = 0
    direction_loss_sum = 0.0
    last_loss_sum = 0.0
    enclosing_loss_sum = 0.0
    fiber_loss_sum = 0.0
    total_loss_sum = 0.0
    presence_loss_sum = 0.0

    def finish(reason: str) -> tuple[np.ndarray, str, _Trace2CpCombinedTraceStats]:
        return (
            np.stack(points, axis=0).astype(np.float32),
            reason,
            _Trace2CpCombinedTraceStats(
                steps=int(steps),
                invalid_candidates=int(invalid_candidates),
                direction_loss_sum=float(direction_loss_sum),
                last_loss_sum=float(last_loss_sum),
                enclosing_loss_sum=float(enclosing_loss_sum),
                fiber_loss_sum=float(fiber_loss_sum),
                total_loss_sum=float(total_loss_sum),
                reason=reason,
                presence_loss_sum=float(presence_loss_sum),
            ),
        )

    for _ in range(int(max_steps)):
        if not _inside_trace_margin(current, shape_hw, rf_margin_px):
            return finish(_trace_margin_reason(current, shape_hw, rf_margin_px))
        oriented = _sample_trace2cp_combined_direction(
            direction_xy=direction_xy,
            tta_fields=tta_fields,
            current_xy=current,
            previous_xy=previous_direction,
            valid_mask=valid_mask,
        )
        if oriented is None:
            return finish("invalid_direction")
        angles, candidate_vectors = _trace2cp_candidate_fan_directions(
            oriented,
            max_degrees=candidate_max_degrees,
            step_degrees=candidate_step_degrees,
        )
        scored: list[tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray | None, float, float, float, float, float]] = []
        margin_reason: str | None = None
        for order, (angle, candidate_unit) in enumerate(zip(angles.tolist(), candidate_vectors)):
            next_point = current + candidate_unit.astype(np.float32) * np.float32(step)
            previous_x = float(current[0])
            next_x = float(next_point[0])
            crosses_target = (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0
            terminal_point = (
                _trace2cp_target_column_point(current, next_point, target_x)
                if crosses_target
                else next_point.astype(np.float32)
            )
            if not crosses_target and not _inside_trace_margin(next_point, shape_hw, rf_margin_px):
                invalid_candidates += 1
                if margin_reason is None:
                    margin_reason = _trace_margin_reason(next_point, shape_hw, rf_margin_px)
                continue
            if crosses_target:
                current_direction_loss = float(
                    1.0 - np.clip(float(np.dot(candidate_unit, oriented)), -1.0, 1.0)
                )
                presence_loss = 0.0
                if presence_required:
                    sampled_presence_loss = _trace2cp_presence_loss(
                        presence,
                        terminal_point,
                        valid_mask=valid_mask,
                    )
                    if sampled_presence_loss is None:
                        invalid_candidates += 1
                        continue
                    presence_loss = float(sampled_presence_loss)
                total = (
                    float(weights.direction) * current_direction_loss
                    + float(weights.presence) * presence_loss
                )
                scored.append(
                    (
                        total,
                        abs(float(angle)),
                        int(order),
                        terminal_point.astype(np.float32),
                        candidate_unit.astype(np.float32),
                        None if previous_embedding is None else previous_embedding.astype(np.float32),
                        current_direction_loss,
                        0.0,
                        0.0,
                        0.0,
                        presence_loss,
                    )
                )
                continue
            candidate_embedding: np.ndarray | None = None
            if embedding_required:
                assert embedding is not None
                candidate_embedding = _bilinear_embedding_sample(embedding, next_point, valid_mask=valid_mask)
                if candidate_embedding is None:
                    invalid_candidates += 1
                    continue
            candidate_oriented = _sample_trace2cp_combined_direction(
                direction_xy=direction_xy,
                tta_fields=tta_fields,
                current_xy=next_point,
                previous_xy=candidate_unit,
                valid_mask=valid_mask,
            )
            if candidate_oriented is None:
                invalid_candidates += 1
                continue
            current_direction_loss = float(
                1.0 - np.clip(float(np.dot(candidate_unit, oriented)), -1.0, 1.0)
            )
            candidate_direction_loss = float(
                1.0 - np.clip(float(np.dot(candidate_unit, candidate_oriented)), -1.0, 1.0)
            )
            direction_loss = 0.5 * (current_direction_loss + candidate_direction_loss)
            if embedding_required:
                assert candidate_embedding is not None
                assert previous_embedding is not None
                assert start_emb is not None and target_emb is not None
                last_loss = _embedding_cosine_loss(candidate_embedding, previous_embedding)
                enclosing_loss = 0.5 * (
                    _embedding_cosine_loss(candidate_embedding, start_emb)
                    + _embedding_cosine_loss(candidate_embedding, target_emb)
                )
                if int(bank.shape[0]) == 0:
                    fiber_loss = 0.0
                else:
                    fiber_similarity = np.clip(bank @ candidate_embedding.reshape(-1, 1), -1.0, 1.0)
                    fiber_loss = float(np.mean(1.0 - fiber_similarity))
            else:
                last_loss = 0.0
                enclosing_loss = 0.0
                fiber_loss = 0.0
            presence_loss = 0.0
            if presence_required:
                sampled_presence_loss = _trace2cp_presence_loss(
                    presence,
                    next_point,
                    valid_mask=valid_mask,
                )
                if sampled_presence_loss is None:
                    invalid_candidates += 1
                    continue
                presence_loss = float(sampled_presence_loss)
            total = (
                float(weights.direction) * direction_loss
                + float(weights.last) * last_loss
                + float(weights.enclosing) * enclosing_loss
                + float(weights.fiber) * fiber_loss
                + float(weights.presence) * presence_loss
            )
            if np.isfinite(total):
                scored.append(
                    (
                        total,
                        abs(float(angle)),
                        int(order),
                        next_point.astype(np.float32),
                        candidate_unit.astype(np.float32),
                        None if candidate_embedding is None else candidate_embedding.astype(np.float32),
                        direction_loss,
                        last_loss,
                        enclosing_loss,
                        fiber_loss,
                        presence_loss,
                    )
                )
            else:
                invalid_candidates += 1
        if not scored:
            return finish(margin_reason or "invalid_candidate")
        scored.sort(key=lambda row: (row[0], row[1], row[2]))
        (
            selected_total,
            _selected_abs_angle,
            _selected_order,
            next_point,
            selected_direction,
            selected_embedding,
            selected_direction_loss,
            selected_last_loss,
            selected_enclosing_loss,
            selected_fiber_loss,
            selected_presence_loss,
        ) = scored[0]
        points.append(next_point.astype(np.float32))
        steps += 1
        direction_loss_sum += float(selected_direction_loss)
        last_loss_sum += float(selected_last_loss)
        enclosing_loss_sum += float(selected_enclosing_loss)
        fiber_loss_sum += float(selected_fiber_loss)
        presence_loss_sum += float(selected_presence_loss)
        total_loss_sum += float(selected_total)
        previous_x = float(current[0])
        next_x = float(next_point[0])
        if (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0:
            return finish("target_column")
        current = next_point.astype(np.float32)
        previous_direction = selected_direction.astype(np.float32)
        previous_embedding = None if selected_embedding is None else selected_embedding.astype(np.float32)
    return finish("max_steps")


def _trace_combined_image_line_to_target(
    *,
    direction_xy: np.ndarray | None,
    tta_fields: list[_TtaDirectionField] | None,
    image_hw: np.ndarray,
    valid_mask: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    weights: _Trace2CpCombinedWeights,
    image_config: _Trace2CpImageScoringConfig | None,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    presence_hw: np.ndarray | None = None,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
    max_steps: int | None = None,
) -> tuple[np.ndarray, str, _Trace2CpCombinedTraceStats]:
    image = np.asarray(image_hw, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if image.ndim != 2:
        raise ValueError("image_hw must have shape H,W")
    if valid.shape != image.shape:
        raise ValueError("valid_mask must match image shape")
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("start_xy and target_xy must have shape (2,)")
    target_delta = target - start
    target_norm = float(np.linalg.norm(target_delta))
    if not np.isfinite(target_norm) or target_norm <= 1.0e-6:
        raise ValueError("trace2cp start and target points must be distinct")

    shape_hw = (int(image.shape[0]), int(image.shape[1]))
    if direction_xy is not None:
        direction = np.asarray(direction_xy, dtype=np.float32)
        if direction.ndim != 3 or direction.shape[2] != 2:
            raise ValueError("direction_xy must have shape H,W,2")
        if tuple(int(v) for v in direction.shape[:2]) != shape_hw:
            raise ValueError("direction_xy and image spatial shapes must match")
    presence_required = float(weights.presence) != 0.0
    presence: np.ndarray | None = None
    if presence_required:
        presence = _require_trace2cp_presence_field(presence_hw)
        if tuple(int(v) for v in presence.shape) != shape_hw:
            raise ValueError("presence_hw spatial shape must match image spatial shape")
    cfg = image_config or _Trace2CpImageScoringConfig()
    step = max(float(step_px), 1.0e-3)
    if max_steps is None:
        max_steps = _trace2cp_target_trace_max_steps(shape_hw, start, target, step)
    current = start.astype(np.float32)
    previous_direction = (target_delta / target_norm).astype(np.float32)
    start_descriptor = _trace2cp_image_descriptor(image, valid, start, previous_direction, cfg)
    target_descriptor = _trace2cp_image_descriptor(image, valid, target, previous_direction, cfg)
    if start_descriptor is None:
        raise ValueError("trace2cp image combined tracing could not sample start CP image descriptor")
    if target_descriptor is None:
        raise ValueError("trace2cp image combined tracing could not sample target CP image descriptor")
    previous_descriptor = start_descriptor
    points = [current.copy()]
    target_x = float(target[0])

    steps = 0
    invalid_candidates = 0
    direction_loss_sum = 0.0
    last_loss_sum = 0.0
    enclosing_loss_sum = 0.0
    fiber_loss_sum = 0.0
    total_loss_sum = 0.0
    image_loss_sum = 0.0
    presence_loss_sum = 0.0

    def finish(reason: str) -> tuple[np.ndarray, str, _Trace2CpCombinedTraceStats]:
        return (
            np.stack(points, axis=0).astype(np.float32),
            reason,
            _Trace2CpCombinedTraceStats(
                steps=int(steps),
                invalid_candidates=int(invalid_candidates),
                direction_loss_sum=float(direction_loss_sum),
                last_loss_sum=float(last_loss_sum),
                enclosing_loss_sum=float(enclosing_loss_sum),
                fiber_loss_sum=float(fiber_loss_sum),
                total_loss_sum=float(total_loss_sum),
                reason=reason,
                image_loss_sum=float(image_loss_sum),
                presence_loss_sum=float(presence_loss_sum),
            ),
        )

    for _ in range(int(max_steps)):
        if not _inside_trace_margin(current, shape_hw, rf_margin_px):
            return finish(_trace_margin_reason(current, shape_hw, rf_margin_px))
        oriented = _sample_trace2cp_combined_direction(
            direction_xy=direction_xy,
            tta_fields=tta_fields,
            current_xy=current,
            previous_xy=previous_direction,
            valid_mask=valid,
        )
        if oriented is None:
            return finish("invalid_direction")
        angles, candidate_vectors = _trace2cp_candidate_fan_directions(
            oriented,
            max_degrees=candidate_max_degrees,
            step_degrees=candidate_step_degrees,
        )
        scored: list[
            tuple[
                float,
                float,
                int,
                np.ndarray,
                np.ndarray,
                _Trace2CpImageDescriptor,
                float,
                float,
                float,
                float,
                float,
            ]
        ] = []
        margin_reason: str | None = None
        for order, (angle, candidate_unit) in enumerate(zip(angles.tolist(), candidate_vectors)):
            next_point = current + candidate_unit.astype(np.float32) * np.float32(step)
            previous_x = float(current[0])
            next_x = float(next_point[0])
            crosses_target = (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0
            terminal_point = (
                _trace2cp_target_column_point(current, next_point, target_x)
                if crosses_target
                else next_point.astype(np.float32)
            )
            if not crosses_target and not _inside_trace_margin(next_point, shape_hw, rf_margin_px):
                invalid_candidates += 1
                if margin_reason is None:
                    margin_reason = _trace_margin_reason(next_point, shape_hw, rf_margin_px)
                continue
            if crosses_target:
                current_direction_loss = float(
                    1.0 - np.clip(float(np.dot(candidate_unit, oriented)), -1.0, 1.0)
                )
                presence_loss = 0.0
                if presence_required:
                    sampled_presence_loss = _trace2cp_presence_loss(
                        presence,
                        terminal_point,
                        valid_mask=valid,
                    )
                    if sampled_presence_loss is None:
                        invalid_candidates += 1
                        continue
                    presence_loss = float(sampled_presence_loss)
                total = (
                    float(weights.direction) * current_direction_loss
                    + float(weights.presence) * presence_loss
                )
                scored.append(
                    (
                        total,
                        abs(float(angle)),
                        int(order),
                        terminal_point.astype(np.float32),
                        candidate_unit.astype(np.float32),
                        previous_descriptor,
                        current_direction_loss,
                        0.0,
                        0.0,
                        0.0,
                        presence_loss,
                    )
                )
                continue
            candidate_descriptor = _trace2cp_image_descriptor(image, valid, next_point, candidate_unit, cfg)
            if candidate_descriptor is None:
                invalid_candidates += 1
                continue
            candidate_oriented = _sample_trace2cp_combined_direction(
                direction_xy=direction_xy,
                tta_fields=tta_fields,
                current_xy=next_point,
                previous_xy=candidate_unit,
                valid_mask=valid,
            )
            if candidate_oriented is None:
                invalid_candidates += 1
                continue
            current_direction_loss = float(
                1.0 - np.clip(float(np.dot(candidate_unit, oriented)), -1.0, 1.0)
            )
            candidate_direction_loss = float(
                1.0 - np.clip(float(np.dot(candidate_unit, candidate_oriented)), -1.0, 1.0)
            )
            direction_loss = 0.5 * (current_direction_loss + candidate_direction_loss)
            last_loss = _trace2cp_image_descriptor_loss(candidate_descriptor, previous_descriptor)
            start_loss = _trace2cp_image_descriptor_loss(candidate_descriptor, start_descriptor)
            target_loss = _trace2cp_image_descriptor_loss(candidate_descriptor, target_descriptor)
            enclosing_loss = 0.5 * (start_loss + target_loss)
            image_loss = (last_loss + start_loss + target_loss) / 3.0
            presence_loss = 0.0
            if presence_required:
                sampled_presence_loss = _trace2cp_presence_loss(
                    presence,
                    next_point,
                    valid_mask=valid,
                )
                if sampled_presence_loss is None:
                    invalid_candidates += 1
                    continue
                presence_loss = float(sampled_presence_loss)
            total = (
                float(weights.direction) * direction_loss
                + float(weights.image) * image_loss
                + float(weights.presence) * presence_loss
            )
            if np.isfinite(total):
                scored.append(
                    (
                        total,
                        abs(float(angle)),
                        int(order),
                        next_point.astype(np.float32),
                        candidate_unit.astype(np.float32),
                        candidate_descriptor,
                        direction_loss,
                        last_loss,
                        enclosing_loss,
                        image_loss,
                        presence_loss,
                    )
                )
            else:
                invalid_candidates += 1
        if not scored:
            return finish(margin_reason or "invalid_candidate")
        scored.sort(key=lambda row: (row[0], row[1], row[2]))
        (
            selected_total,
            _selected_abs_angle,
            _selected_order,
            next_point,
            selected_direction,
            selected_descriptor,
            selected_direction_loss,
            selected_last_loss,
            selected_enclosing_loss,
            selected_image_loss,
            selected_presence_loss,
        ) = scored[0]
        points.append(next_point.astype(np.float32))
        steps += 1
        direction_loss_sum += float(selected_direction_loss)
        last_loss_sum += float(selected_last_loss)
        enclosing_loss_sum += float(selected_enclosing_loss)
        image_loss_sum += float(selected_image_loss)
        presence_loss_sum += float(selected_presence_loss)
        total_loss_sum += float(selected_total)
        previous_x = float(current[0])
        next_x = float(next_point[0])
        if (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0:
            return finish("target_column")
        current = next_point.astype(np.float32)
        previous_direction = selected_direction.astype(np.float32)
        previous_descriptor = selected_descriptor
    return finish("max_steps")


def _trace_combined_direction_line_to_target_z(
    *,
    plane_cache: _Trace2CpZPlaneCache,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    start_embedding: np.ndarray | None,
    target_embedding: np.ndarray | None,
    fiber_embeddings: np.ndarray | None,
    weights: _Trace2CpCombinedWeights,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
    max_steps: int | None = None,
) -> tuple[np.ndarray, str, _Trace2CpCombinedTraceStats]:
    center = plane_cache.get(0)
    embedding_required = any(
        float(value) != 0.0 for value in (weights.last, weights.enclosing, weights.fiber)
    )
    presence_required = float(weights.presence) != 0.0
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("start_xy and target_xy must have shape (2,)")
    target_delta = target - start
    target_norm = float(np.linalg.norm(target_delta))
    if not np.isfinite(target_norm) or target_norm <= 1.0e-6:
        raise ValueError("trace2cp start and target points must be distinct")

    shape_hw = tuple(int(v) for v in np.asarray(center.fields.direction_xy).shape[:2])
    if len(shape_hw) != 2:
        raise ValueError("trace2cp z-search direction field must have spatial shape H,W")
    center_embedding: np.ndarray | None = None
    if embedding_required:
        center_embedding = _require_trace2cp_embedding_field(center.fields.embedding_chw)
        if tuple(int(v) for v in center_embedding.shape[1:]) != shape_hw:
            raise ValueError("center embedding spatial shape must match z-search direction field")
        bank = np.asarray(
            np.zeros((0, int(center_embedding.shape[0])), dtype=np.float32)
            if fiber_embeddings is None
            else fiber_embeddings,
            dtype=np.float32,
        )
        if bank.size == 0:
            bank = bank.reshape(0, int(center_embedding.shape[0]))
        if bank.ndim != 2 or int(bank.shape[1]) != int(center_embedding.shape[0]):
            raise ValueError("fiber_embeddings must have shape N,C matching embedding_chw")
        if float(weights.fiber) != 0.0 and int(bank.shape[0]) == 0:
            raise ValueError("trace2cp combined fiber weight is non-zero but the fiber CP embedding bank is empty")
        if start_embedding is None or target_embedding is None:
            raise ValueError("trace2cp z-search embedding scoring requires start and target embeddings")
        start_emb = np.asarray(start_embedding, dtype=np.float32).reshape(-1)
        target_emb = np.asarray(target_embedding, dtype=np.float32).reshape(-1)
        if (
            start_emb.shape[0] != int(center_embedding.shape[0])
            or target_emb.shape[0] != int(center_embedding.shape[0])
        ):
            raise ValueError("start/target embeddings must match embedding_chw channel count")
    else:
        bank = np.zeros((0, 0), dtype=np.float32)
        start_emb = None
        target_emb = None
    if presence_required:
        center_presence = _require_trace2cp_presence_field(center.fields.presence_hw)
        if tuple(int(v) for v in center_presence.shape) != shape_hw:
            raise ValueError("center presence spatial shape must match z-search direction field")

    step = max(float(step_px), 1.0e-3)
    if max_steps is None:
        max_steps = _trace2cp_target_trace_max_steps(shape_hw, start, target, step)
    current = start.astype(np.float32)
    current_layer = 0
    previous_direction = (target_delta / target_norm).astype(np.float32)
    previous_embedding = None if start_emb is None else start_emb.astype(np.float32)
    points = [np.asarray([current[0], current[1], 0.0], dtype=np.float32)]
    target_x = float(target[0])

    steps = 0
    invalid_candidates = 0
    direction_loss_sum = 0.0
    last_loss_sum = 0.0
    enclosing_loss_sum = 0.0
    fiber_loss_sum = 0.0
    total_loss_sum = 0.0
    presence_loss_sum = 0.0

    def finish(reason: str) -> tuple[np.ndarray, str, _Trace2CpCombinedTraceStats]:
        return (
            np.stack(points, axis=0).astype(np.float32),
            reason,
            _Trace2CpCombinedTraceStats(
                steps=int(steps),
                invalid_candidates=int(invalid_candidates),
                direction_loss_sum=float(direction_loss_sum),
                last_loss_sum=float(last_loss_sum),
                enclosing_loss_sum=float(enclosing_loss_sum),
                fiber_loss_sum=float(fiber_loss_sum),
                total_loss_sum=float(total_loss_sum),
                reason=reason,
                presence_loss_sum=float(presence_loss_sum),
            ),
        )

    for _ in range(int(max_steps)):
        if not _inside_trace_margin(current, shape_hw, rf_margin_px):
            return finish(_trace_margin_reason(current, shape_hw, rf_margin_px))
        plane_cache.ensure_neighbors(current_layer)
        current_prediction = plane_cache.get(current_layer)
        oriented = _sample_trace2cp_combined_direction(
            direction_xy=current_prediction.fields.direction_xy,
            tta_fields=None,
            current_xy=current,
            previous_xy=previous_direction,
            valid_mask=current_prediction.valid_mask,
        )
        if oriented is None:
            return finish("invalid_direction")
        angles, candidate_vectors = _trace2cp_candidate_fan_directions(
            oriented,
            max_degrees=candidate_max_degrees,
            step_degrees=candidate_step_degrees,
        )
        scored: list[
            tuple[
                float,
                float,
                int,
                int,
                np.ndarray,
                np.ndarray,
                np.ndarray | None,
                float,
                float,
                float,
                float,
                float,
                float,
            ]
        ] = []
        margin_reason: str | None = None
        for order, (angle, candidate_unit) in enumerate(zip(angles.tolist(), candidate_vectors)):
            next_point = current + candidate_unit.astype(np.float32) * np.float32(step)
            previous_x = float(current[0])
            next_x = float(next_point[0])
            crosses_target = (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0
            terminal_point = (
                _trace2cp_target_column_point(current, next_point, target_x)
                if crosses_target
                else next_point.astype(np.float32)
            )
            if not crosses_target and not _inside_trace_margin(next_point, shape_hw, rf_margin_px):
                invalid_candidates += 1
                if margin_reason is None:
                    margin_reason = _trace_margin_reason(next_point, shape_hw, rf_margin_px)
                continue
            for layer_delta in (-1, 0, 1):
                candidate_layer = int(current_layer + layer_delta)
                if abs(candidate_layer) > int(plane_cache.max_layer):
                    invalid_candidates += 1
                    continue
                if crosses_target:
                    current_direction_loss = float(
                        1.0 - np.clip(float(np.dot(candidate_unit, oriented)), -1.0, 1.0)
                    )
                    presence_loss = 0.0
                    if presence_required:
                        candidate_prediction = plane_cache.get(candidate_layer)
                        sampled_presence_loss = _trace2cp_presence_loss(
                            candidate_prediction.fields.presence_hw,
                            terminal_point,
                            valid_mask=candidate_prediction.valid_mask,
                        )
                        if sampled_presence_loss is None:
                            invalid_candidates += 1
                            continue
                        presence_loss = float(sampled_presence_loss)
                    total = (
                        float(weights.direction) * current_direction_loss
                        + float(weights.presence) * presence_loss
                    )
                    scored.append(
                        (
                            total,
                            abs(float(angle)),
                            abs(int(layer_delta)),
                            int(order),
                            terminal_point.astype(np.float32),
                            candidate_unit.astype(np.float32),
                            None if previous_embedding is None else previous_embedding.astype(np.float32),
                            float(candidate_layer),
                            current_direction_loss,
                            0.0,
                            0.0,
                            0.0,
                            presence_loss,
                        )
                    )
                    continue
                candidate_prediction = plane_cache.get(candidate_layer)
                candidate_embedding: np.ndarray | None = None
                if embedding_required:
                    embedding = _require_trace2cp_embedding_field(candidate_prediction.fields.embedding_chw)
                    if tuple(int(v) for v in embedding.shape[1:]) != shape_hw:
                        raise ValueError("candidate embedding spatial shape must match z-search direction field")
                    candidate_embedding = _bilinear_embedding_sample(
                        embedding,
                        next_point,
                        valid_mask=candidate_prediction.valid_mask,
                    )
                    if candidate_embedding is None:
                        invalid_candidates += 1
                        continue
                candidate_oriented = _sample_trace2cp_combined_direction(
                    direction_xy=candidate_prediction.fields.direction_xy,
                    tta_fields=None,
                    current_xy=next_point,
                    previous_xy=candidate_unit,
                    valid_mask=candidate_prediction.valid_mask,
                )
                if candidate_oriented is None:
                    invalid_candidates += 1
                    continue
                current_direction_loss = float(
                    1.0 - np.clip(float(np.dot(candidate_unit, oriented)), -1.0, 1.0)
                )
                candidate_direction_loss = float(
                    1.0 - np.clip(float(np.dot(candidate_unit, candidate_oriented)), -1.0, 1.0)
                )
                direction_loss = 0.5 * (current_direction_loss + candidate_direction_loss)
                if embedding_required:
                    assert candidate_embedding is not None
                    assert previous_embedding is not None
                    assert start_emb is not None and target_emb is not None
                    last_loss = _embedding_cosine_loss(candidate_embedding, previous_embedding)
                    enclosing_loss = 0.5 * (
                        _embedding_cosine_loss(candidate_embedding, start_emb)
                        + _embedding_cosine_loss(candidate_embedding, target_emb)
                    )
                    if int(bank.shape[0]) == 0:
                        fiber_loss = 0.0
                    else:
                        fiber_similarity = np.clip(bank @ candidate_embedding.reshape(-1, 1), -1.0, 1.0)
                        fiber_loss = float(np.mean(1.0 - fiber_similarity))
                else:
                    last_loss = 0.0
                    enclosing_loss = 0.0
                    fiber_loss = 0.0
                presence_loss = 0.0
                if presence_required:
                    sampled_presence_loss = _trace2cp_presence_loss(
                        candidate_prediction.fields.presence_hw,
                        next_point,
                        valid_mask=candidate_prediction.valid_mask,
                    )
                    if sampled_presence_loss is None:
                        invalid_candidates += 1
                        continue
                    presence_loss = float(sampled_presence_loss)
                total = (
                    float(weights.direction) * direction_loss
                    + float(weights.last) * last_loss
                    + float(weights.enclosing) * enclosing_loss
                    + float(weights.fiber) * fiber_loss
                    + float(weights.presence) * presence_loss
                )
                if np.isfinite(total):
                    scored.append(
                        (
                            total,
                            abs(float(angle)),
                            abs(int(layer_delta)),
                            int(order),
                            next_point.astype(np.float32),
                            candidate_unit.astype(np.float32),
                            None if candidate_embedding is None else candidate_embedding.astype(np.float32),
                            float(candidate_layer),
                            direction_loss,
                            last_loss,
                            enclosing_loss,
                            fiber_loss,
                            presence_loss,
                        )
                    )
                else:
                    invalid_candidates += 1
        if not scored:
            return finish(margin_reason or "invalid_candidate")
        scored.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        (
            selected_total,
            _selected_abs_angle,
            _selected_abs_layer_delta,
            _selected_order,
            next_point,
            selected_direction,
            selected_embedding,
            selected_layer_float,
            selected_direction_loss,
            selected_last_loss,
            selected_enclosing_loss,
            selected_fiber_loss,
            selected_presence_loss,
        ) = scored[0]
        selected_layer = int(selected_layer_float)
        selected_z = float(selected_layer) * float(plane_cache.z_step_voxels)
        points.append(np.asarray([next_point[0], next_point[1], selected_z], dtype=np.float32))
        steps += 1
        direction_loss_sum += float(selected_direction_loss)
        last_loss_sum += float(selected_last_loss)
        enclosing_loss_sum += float(selected_enclosing_loss)
        fiber_loss_sum += float(selected_fiber_loss)
        presence_loss_sum += float(selected_presence_loss)
        total_loss_sum += float(selected_total)
        previous_x = float(current[0])
        next_x = float(next_point[0])
        if (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0:
            return finish("target_column")
        current = next_point.astype(np.float32)
        current_layer = selected_layer
        previous_direction = selected_direction.astype(np.float32)
        previous_embedding = None if selected_embedding is None else selected_embedding.astype(np.float32)
    return finish("max_steps")


def _trace_combined_image_line_to_target_z(
    *,
    plane_cache: _Trace2CpZPlaneCache,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    weights: _Trace2CpCombinedWeights,
    image_config: _Trace2CpImageScoringConfig | None,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
    max_steps: int | None = None,
) -> tuple[np.ndarray, str, _Trace2CpCombinedTraceStats]:
    center = plane_cache.get(0)
    center_image = np.asarray(center.image, dtype=np.float32)
    center_valid = np.asarray(center.valid_mask, dtype=bool)
    if center_image.ndim != 2 or center_valid.shape != center_image.shape:
        raise ValueError("trace2cp z-search image mode requires center image and valid mask")
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("start_xy and target_xy must have shape (2,)")
    target_delta = target - start
    target_norm = float(np.linalg.norm(target_delta))
    if not np.isfinite(target_norm) or target_norm <= 1.0e-6:
        raise ValueError("trace2cp start and target points must be distinct")

    cfg = image_config or _Trace2CpImageScoringConfig()
    shape_hw = (int(center_image.shape[0]), int(center_image.shape[1]))
    presence_required = float(weights.presence) != 0.0
    if presence_required:
        center_presence = _require_trace2cp_presence_field(center.fields.presence_hw)
        if tuple(int(v) for v in center_presence.shape) != shape_hw:
            raise ValueError("center presence spatial shape must match z-search image")
    step = max(float(step_px), 1.0e-3)
    if max_steps is None:
        max_steps = _trace2cp_target_trace_max_steps(shape_hw, start, target, step)
    current = start.astype(np.float32)
    current_layer = 0
    previous_direction = (target_delta / target_norm).astype(np.float32)
    start_descriptor = _trace2cp_image_descriptor(center_image, center_valid, start, previous_direction, cfg)
    target_descriptor = _trace2cp_image_descriptor(center_image, center_valid, target, previous_direction, cfg)
    if start_descriptor is None:
        raise ValueError("trace2cp image z-search could not sample start CP image descriptor")
    if target_descriptor is None:
        raise ValueError("trace2cp image z-search could not sample target CP image descriptor")
    previous_descriptor = start_descriptor
    points = [np.asarray([current[0], current[1], 0.0], dtype=np.float32)]
    target_x = float(target[0])

    steps = 0
    invalid_candidates = 0
    direction_loss_sum = 0.0
    last_loss_sum = 0.0
    enclosing_loss_sum = 0.0
    fiber_loss_sum = 0.0
    total_loss_sum = 0.0
    image_loss_sum = 0.0
    presence_loss_sum = 0.0

    def finish(reason: str) -> tuple[np.ndarray, str, _Trace2CpCombinedTraceStats]:
        return (
            np.stack(points, axis=0).astype(np.float32),
            reason,
            _Trace2CpCombinedTraceStats(
                steps=int(steps),
                invalid_candidates=int(invalid_candidates),
                direction_loss_sum=float(direction_loss_sum),
                last_loss_sum=float(last_loss_sum),
                enclosing_loss_sum=float(enclosing_loss_sum),
                fiber_loss_sum=float(fiber_loss_sum),
                total_loss_sum=float(total_loss_sum),
                reason=reason,
                image_loss_sum=float(image_loss_sum),
                presence_loss_sum=float(presence_loss_sum),
            ),
        )

    for _ in range(int(max_steps)):
        if not _inside_trace_margin(current, shape_hw, rf_margin_px):
            return finish(_trace_margin_reason(current, shape_hw, rf_margin_px))
        plane_cache.ensure_neighbors(current_layer)
        current_prediction = plane_cache.get(current_layer)
        oriented = _sample_trace2cp_combined_direction(
            direction_xy=current_prediction.fields.direction_xy,
            tta_fields=None,
            current_xy=current,
            previous_xy=previous_direction,
            valid_mask=current_prediction.valid_mask,
        )
        if oriented is None:
            return finish("invalid_direction")
        angles, candidate_vectors = _trace2cp_candidate_fan_directions(
            oriented,
            max_degrees=candidate_max_degrees,
            step_degrees=candidate_step_degrees,
        )
        scored: list[
            tuple[
                float,
                float,
                int,
                int,
                np.ndarray,
                np.ndarray,
                _Trace2CpImageDescriptor,
                float,
                float,
                float,
                float,
                float,
                float,
            ]
        ] = []
        margin_reason: str | None = None
        for order, (angle, candidate_unit) in enumerate(zip(angles.tolist(), candidate_vectors)):
            next_point = current + candidate_unit.astype(np.float32) * np.float32(step)
            previous_x = float(current[0])
            next_x = float(next_point[0])
            crosses_target = (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0
            terminal_point = (
                _trace2cp_target_column_point(current, next_point, target_x)
                if crosses_target
                else next_point.astype(np.float32)
            )
            if not crosses_target and not _inside_trace_margin(next_point, shape_hw, rf_margin_px):
                invalid_candidates += 1
                if margin_reason is None:
                    margin_reason = _trace_margin_reason(next_point, shape_hw, rf_margin_px)
                continue
            for layer_delta in (-1, 0, 1):
                candidate_layer = int(current_layer + layer_delta)
                if abs(candidate_layer) > int(plane_cache.max_layer):
                    invalid_candidates += 1
                    continue
                if crosses_target:
                    current_direction_loss = float(
                        1.0 - np.clip(float(np.dot(candidate_unit, oriented)), -1.0, 1.0)
                    )
                    presence_loss = 0.0
                    if presence_required:
                        candidate_prediction = plane_cache.get(candidate_layer)
                        sampled_presence_loss = _trace2cp_presence_loss(
                            candidate_prediction.fields.presence_hw,
                            terminal_point,
                            valid_mask=candidate_prediction.valid_mask,
                        )
                        if sampled_presence_loss is None:
                            invalid_candidates += 1
                            continue
                        presence_loss = float(sampled_presence_loss)
                    total = (
                        float(weights.direction) * current_direction_loss
                        + float(weights.presence) * presence_loss
                    )
                    scored.append(
                        (
                            total,
                            abs(float(angle)),
                            abs(int(layer_delta)),
                            int(order),
                            terminal_point.astype(np.float32),
                            candidate_unit.astype(np.float32),
                            previous_descriptor,
                            float(candidate_layer),
                            current_direction_loss,
                            0.0,
                            0.0,
                            0.0,
                            presence_loss,
                        )
                    )
                    continue
                candidate_prediction = plane_cache.get(candidate_layer)
                candidate_image = np.asarray(candidate_prediction.image, dtype=np.float32)
                candidate_valid = np.asarray(candidate_prediction.valid_mask, dtype=bool)
                candidate_descriptor = _trace2cp_image_descriptor(
                    candidate_image,
                    candidate_valid,
                    next_point,
                    candidate_unit,
                    cfg,
                )
                if candidate_descriptor is None:
                    invalid_candidates += 1
                    continue
                candidate_oriented = _sample_trace2cp_combined_direction(
                    direction_xy=candidate_prediction.fields.direction_xy,
                    tta_fields=None,
                    current_xy=next_point,
                    previous_xy=candidate_unit,
                    valid_mask=candidate_prediction.valid_mask,
                )
                if candidate_oriented is None:
                    invalid_candidates += 1
                    continue
                current_direction_loss = float(
                    1.0 - np.clip(float(np.dot(candidate_unit, oriented)), -1.0, 1.0)
                )
                candidate_direction_loss = float(
                    1.0 - np.clip(float(np.dot(candidate_unit, candidate_oriented)), -1.0, 1.0)
                )
                direction_loss = 0.5 * (current_direction_loss + candidate_direction_loss)
                last_loss = _trace2cp_image_descriptor_loss(candidate_descriptor, previous_descriptor)
                start_loss = _trace2cp_image_descriptor_loss(candidate_descriptor, start_descriptor)
                target_loss = _trace2cp_image_descriptor_loss(candidate_descriptor, target_descriptor)
                enclosing_loss = 0.5 * (start_loss + target_loss)
                image_loss = (last_loss + start_loss + target_loss) / 3.0
                presence_loss = 0.0
                if presence_required:
                    sampled_presence_loss = _trace2cp_presence_loss(
                        candidate_prediction.fields.presence_hw,
                        next_point,
                        valid_mask=candidate_prediction.valid_mask,
                    )
                    if sampled_presence_loss is None:
                        invalid_candidates += 1
                        continue
                    presence_loss = float(sampled_presence_loss)
                total = (
                    float(weights.direction) * direction_loss
                    + float(weights.image) * image_loss
                    + float(weights.presence) * presence_loss
                )
                if np.isfinite(total):
                    scored.append(
                        (
                            total,
                            abs(float(angle)),
                            abs(int(layer_delta)),
                            int(order),
                            next_point.astype(np.float32),
                            candidate_unit.astype(np.float32),
                            candidate_descriptor,
                            float(candidate_layer),
                            direction_loss,
                            last_loss,
                            enclosing_loss,
                            image_loss,
                            presence_loss,
                        )
                    )
                else:
                    invalid_candidates += 1
        if not scored:
            return finish(margin_reason or "invalid_candidate")
        scored.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
        (
            selected_total,
            _selected_abs_angle,
            _selected_abs_layer_delta,
            _selected_order,
            next_point,
            selected_direction,
            selected_descriptor,
            selected_layer_float,
            selected_direction_loss,
            selected_last_loss,
            selected_enclosing_loss,
            selected_image_loss,
            selected_presence_loss,
        ) = scored[0]
        selected_layer = int(selected_layer_float)
        selected_z = float(selected_layer) * float(plane_cache.z_step_voxels)
        points.append(np.asarray([next_point[0], next_point[1], selected_z], dtype=np.float32))
        steps += 1
        direction_loss_sum += float(selected_direction_loss)
        last_loss_sum += float(selected_last_loss)
        enclosing_loss_sum += float(selected_enclosing_loss)
        image_loss_sum += float(selected_image_loss)
        presence_loss_sum += float(selected_presence_loss)
        total_loss_sum += float(selected_total)
        previous_x = float(current[0])
        next_x = float(next_point[0])
        if (previous_x - target_x) == 0.0 or (previous_x - target_x) * (next_x - target_x) <= 0.0:
            return finish("target_column")
        current = next_point.astype(np.float32)
        current_layer = selected_layer
        previous_direction = selected_direction.astype(np.float32)
        previous_descriptor = selected_descriptor
    return finish("max_steps")


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


def _trace_xyz_at_x(trace_xyz: np.ndarray, target_x: float) -> np.ndarray | None:
    trace = np.asarray(trace_xyz, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 3 or trace.shape[0] == 0:
        raise ValueError("trace_xyz must have shape N,3")
    x_target = float(target_x)
    for p0, p1 in zip(trace[:-1], trace[1:]):
        x0 = float(p0[0])
        x1 = float(p1[0])
        if (x0 - x_target) == 0.0:
            return p0.astype(np.float32, copy=True)
        if (x0 - x_target) * (x1 - x_target) <= 0.0 and x0 != x1:
            alpha = (x_target - x0) / (x1 - x0)
            return (p0 + np.float32(alpha) * (p1 - p0)).astype(np.float32)
    if float(trace[-1, 0]) == x_target:
        return trace[-1].astype(np.float32, copy=True)
    return None


def _resample_trace_xyz_at_x_values(trace_xyz: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace_xyz, dtype=np.float32)
    values = np.asarray(x_values, dtype=np.float32).reshape(-1)
    if trace.ndim != 2 or trace.shape[1] != 3:
        raise ValueError("trace_xyz must have shape N,3")
    points: list[np.ndarray] = []
    for x in values.tolist():
        point = _trace_xyz_at_x(trace, float(x))
        if point is not None and bool(np.isfinite(point).all()):
            points.append(point.astype(np.float32, copy=False))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(points, axis=0).astype(np.float32)


def _resample_trace_xyz_between_x(trace_xyz: np.ndarray, start_x: float, end_x: float) -> np.ndarray:
    return _resample_trace_xyz_at_x_values(trace_xyz, _ordered_x_values_between(float(start_x), float(end_x)))


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


def _closest_trace2cp_approach_z(
    forward_trace_xyz: np.ndarray,
    reverse_trace_xyz: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    rf_margin_px: float,
) -> tuple[float, float, float, float, float, float, float, float, float, np.ndarray, float, str, bool]:
    _, _, denominator = _usable_trace2cp_vertical_span(shape_hw, rf_margin_px)
    forward_xy = np.asarray(forward_trace_xyz, dtype=np.float32)[:, :2]
    reverse_xy = np.asarray(reverse_trace_xyz, dtype=np.float32)[:, :2]
    x_values = _trace2cp_overlap_x_values(forward_xy, reverse_xy, start_xy, target_xy)
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
            float("nan"),
            float("nan"),
            midpoint,
            float("nan"),
            "no_trace_overlap",
            False,
        )

    rows: list[tuple[float, float, float, float, float]] = []
    gaps: list[float] = []
    penalties: list[float] = []
    considered_gaps: list[float] = []
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    for x in x_values.tolist():
        forward_point = _trace_xyz_at_x(forward_trace_xyz, float(x))
        reverse_point = _trace_xyz_at_x(reverse_trace_xyz, float(x))
        if forward_point is None or reverse_point is None:
            continue
        if not bool(np.isfinite(forward_point).all()) or not bool(np.isfinite(reverse_point).all()):
            continue
        y_gap = abs(float(forward_point[1]) - float(reverse_point[1]))
        z_gap = abs(float(forward_point[2]) - float(reverse_point[2]))
        gap = y_gap + z_gap
        penalty = _trace2cp_center_penalty(float(x), float(start[0]), float(target[0]))
        rows.append(
            (
                float(x),
                float(forward_point[1]),
                float(reverse_point[1]),
                float(forward_point[2]),
                float(reverse_point[2]),
            )
        )
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
            float("nan"),
            float("nan"),
            midpoint,
            float("nan"),
            "no_trace_overlap",
            False,
        )

    considered_arr = np.asarray(considered_gaps, dtype=np.float32)
    min_considered = float(np.min(considered_arr))
    midpoint_x = 0.5 * (float(start[0]) + float(target[0]))
    candidate_indices = np.flatnonzero(np.isclose(considered_arr, min_considered, rtol=1.0e-6, atol=1.0e-6))
    if candidate_indices.size > 1:
        candidate_x = np.asarray([rows[int(index)][0] for index in candidate_indices], dtype=np.float32)
        closest_index = int(candidate_indices[int(np.argmin(np.abs(candidate_x - np.float32(midpoint_x))))])
    else:
        closest_index = int(candidate_indices[0])
    closest_x, forward_y, reverse_y, forward_z, reverse_z = rows[closest_index]
    gap = float(gaps[closest_index])
    penalty = float(penalties[closest_index])
    considered_gap = float(considered_gaps[closest_index])
    midpoint = np.asarray([closest_x, 0.5 * (forward_y + reverse_y)], dtype=np.float32)
    midpoint_z = 0.5 * (float(forward_z) + float(reverse_z))
    score = float(np.clip(considered_gap / denominator, 0.0, 1.0))
    return (
        score,
        gap,
        considered_gap,
        penalty,
        closest_x,
        forward_y,
        reverse_y,
        forward_z,
        reverse_z,
        midpoint,
        midpoint_z,
        "closest_approach_z",
        True,
    )


def _trace2cp_horizontal_span_px(start_xy: np.ndarray, target_xy: np.ndarray) -> float:
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    if start.shape != (2,) or target.shape != (2,):
        raise ValueError("start_xy and target_xy must have shape (2,)")
    span = abs(float(target[0]) - float(start[0]))
    if not np.isfinite(span) or span <= 1.0e-6:
        raise ValueError(
            "trace2cp metric requires distinct horizontal CP positions: "
            f"start_x={float(start[0]):.6f} target_x={float(target[0]):.6f}"
        )
    return span


def _trace2cp_default_max_y_error_px(
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    rf_margin_px: float,
) -> float:
    top, bottom, span = _usable_trace2cp_vertical_span(shape_hw, rf_margin_px)
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    center_y = 0.5 * (float(start[1]) + float(target[1]))
    if not np.isfinite(center_y):
        return span
    center_y = float(np.clip(center_y, top, bottom))
    edge_distance = min(center_y - top, bottom - center_y)
    if not np.isfinite(edge_distance) or edge_distance <= 0.0:
        return span
    return float(edge_distance)


def _trace2cp_midpoint_xy(start_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    return (0.5 * (start + target)).astype(np.float32, copy=False)


def _trace2cp_metric_from_traces(
    forward_trace_xy: np.ndarray,
    reverse_trace_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    rf_margin_px: float,
) -> _Trace2CpMetricResult:
    horizontal_span = _trace2cp_horizontal_span_px(start_xy, target_xy)
    max_y_error = _trace2cp_default_max_y_error_px(
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
    )
    start = np.asarray(start_xy, dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    forward_target_x = float(target[0])
    reverse_target_x = float(start[0])
    forward_y = _trace_y_at_x(forward_trace_xy, forward_target_x)
    reverse_y = _trace_y_at_x(reverse_trace_xy, reverse_target_x)
    forward_reached = forward_y is not None and np.isfinite(forward_y)
    reverse_reached = reverse_y is not None and np.isfinite(reverse_y)

    forward_error = (
        abs(float(forward_y) - float(target[1])) if forward_reached else float(max_y_error)
    )
    reverse_error = (
        abs(float(reverse_y) - float(start[1])) if reverse_reached else float(max_y_error)
    )
    raw_y_error = 0.5 * (float(forward_error) + float(reverse_error))
    metric_gap = 0.5 * (
        min(float(forward_error), float(max_y_error))
        + min(float(reverse_error), float(max_y_error))
    )
    reached_target_columns = bool(forward_reached and reverse_reached)
    if reached_target_columns:
        reason = "target_columns"
    elif forward_reached or reverse_reached:
        reason = "partial_target_columns"
    else:
        reason = "missing_target_columns"

    endpoint_mid_x = 0.5 * (float(start[0]) + float(target[0]))
    endpoint_ys = [
        float(y)
        for y, reached in ((forward_y, forward_reached), (reverse_y, reverse_reached))
        if reached
    ]
    if endpoint_ys:
        endpoint_mid_y = float(np.mean(np.asarray(endpoint_ys, dtype=np.float32)))
    else:
        endpoint_mid_y = float(_trace2cp_midpoint_xy(start, target)[1])
    endpoint_midpoint = np.asarray([endpoint_mid_x, endpoint_mid_y], dtype=np.float32)
    return _Trace2CpMetricResult(
        error=float(metric_gap / horizontal_span),
        raw_y_error_px=float(raw_y_error),
        horizontal_span_px=float(horizontal_span),
        max_y_error_px=float(max_y_error),
        forward_target_x=float(forward_target_x),
        reverse_target_x=float(reverse_target_x),
        forward_y_at_target_x=float(forward_y) if forward_reached else float("nan"),
        reverse_y_at_target_x=float(reverse_y) if reverse_reached else float("nan"),
        reached_target_columns=reached_target_columns,
        closest_x=float(endpoint_mid_x),
        forward_y_at_closest_x=float(forward_y) if forward_reached else float("nan"),
        reverse_y_at_closest_x=float(reverse_y) if reverse_reached else float("nan"),
        closest_midpoint_xy=endpoint_midpoint,
        reached_overlap=reached_target_columns,
        reason=reason,
    )


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


def _vertical_z_warp_trace_to_meet(
    trace_xyz: np.ndarray,
    *,
    anchor_xyz: np.ndarray,
    meet_x: float,
    meet_y: float,
    meet_z: float,
    source_meet_y: float,
    source_meet_z: float,
) -> np.ndarray:
    trace = np.asarray(trace_xyz, dtype=np.float32)
    anchor = np.asarray(anchor_xyz, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 3:
        raise ValueError("trace_xyz must have shape N,3")
    if anchor.shape != (3,):
        raise ValueError("anchor_xyz must have shape (3,)")
    if trace.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    warped = trace.copy()
    denom = abs(float(meet_x) - float(anchor[0]))
    if denom <= 1.0e-6:
        blend = np.ones((trace.shape[0],), dtype=np.float32)
    else:
        blend = np.clip(np.abs(trace[:, 0] - np.float32(anchor[0])) / np.float32(denom), 0.0, 1.0)
    warped[:, 1] = warped[:, 1] + blend * np.float32(float(meet_y) - float(source_meet_y))
    warped[:, 2] = warped[:, 2] + blend * np.float32(float(meet_z) - float(source_meet_z))
    warped[0] = anchor.astype(np.float32, copy=False)
    warped[-1, 0] = np.float32(meet_x)
    warped[-1, 1] = np.float32(meet_y)
    warped[-1, 2] = np.float32(meet_z)
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


def _resample_polyline_xyz_by_arclength(line_xyz: np.ndarray, *, step_px: float) -> np.ndarray:
    line = np.asarray(line_xyz, dtype=np.float32)
    if line.ndim != 2 or line.shape[1] != 3:
        raise ValueError("line_xyz must have shape N,3")
    finite = np.isfinite(line).all(axis=1)
    line = line[finite]
    if line.shape[0] <= 1:
        return line.astype(np.float32, copy=True)
    delta_xy = line[1:, :2] - line[:-1, :2]
    seg_len = np.linalg.norm(delta_xy, axis=1)
    keep = np.concatenate([[True], seg_len > 1.0e-6])
    line = line[keep]
    if line.shape[0] <= 1:
        return line.astype(np.float32, copy=True)
    delta_xy = line[1:, :2] - line[:-1, :2]
    seg_len = np.linalg.norm(delta_xy, axis=1)
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
    zs = np.interp(distances, cumulative, line[:, 2].astype(np.float64))
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


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


def _trace2cp_refinement_from_traces_z(
    forward_trace_xyz: np.ndarray,
    reverse_trace_xyz: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    step_px: float,
    rf_margin_px: float,
) -> _Trace2CpRefinementResult:
    (
        score,
        gap,
        considered_gap,
        penalty,
        closest_x,
        forward_y,
        reverse_y,
        forward_z,
        reverse_z,
        midpoint,
        midpoint_z,
        reason,
        reached,
    ) = _closest_trace2cp_approach_z(
        forward_trace_xyz,
        reverse_trace_xyz,
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
    )
    _, _, denominator = _usable_trace2cp_vertical_span(shape_hw, rf_margin_px)
    empty_xy = np.zeros((0, 2), dtype=np.float32)
    empty_z = np.zeros((0,), dtype=np.float32)
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
            partial_forward_xy=empty_xy,
            partial_reverse_xy=empty_xy,
            fused_dense_xy=empty_xy,
            fused_resampled_xy=empty_xy,
            optimized_xy=empty_xy,
            partial_forward_z=empty_z,
            partial_reverse_z=empty_z,
            fused_dense_z=empty_z,
            fused_resampled_z=empty_z,
            optimized_z=empty_z,
            closest_forward_z_voxels=forward_z,
            closest_reverse_z_voxels=reverse_z,
            closest_midpoint_z_voxels=midpoint_z,
        )

    start = np.asarray([float(start_xy[0]), float(start_xy[1]), 0.0], dtype=np.float32)
    target = np.asarray([float(target_xy[0]), float(target_xy[1]), 0.0], dtype=np.float32)
    partial_forward = _resample_trace_xyz_between_x(forward_trace_xyz, float(start[0]), closest_x)
    partial_reverse = _resample_trace_xyz_between_x(reverse_trace_xyz, float(target[0]), closest_x)
    warped_forward = _vertical_z_warp_trace_to_meet(
        partial_forward,
        anchor_xyz=start,
        meet_x=closest_x,
        meet_y=float(midpoint[1]),
        meet_z=float(midpoint_z),
        source_meet_y=forward_y,
        source_meet_z=forward_z,
    )
    warped_reverse = _vertical_z_warp_trace_to_meet(
        partial_reverse,
        anchor_xyz=target,
        meet_x=closest_x,
        meet_y=float(midpoint[1]),
        meet_z=float(midpoint_z),
        source_meet_y=reverse_y,
        source_meet_z=reverse_z,
    )
    reverse_meet_to_target = warped_reverse[::-1].copy()
    if warped_forward.shape[0] > 0 and reverse_meet_to_target.shape[0] > 0:
        fused_dense_xyz = np.concatenate([warped_forward, reverse_meet_to_target[1:]], axis=0)
    else:
        fused_dense_xyz = np.zeros((0, 3), dtype=np.float32)
    fused_resampled_xyz = _resample_polyline_xyz_by_arclength(fused_dense_xyz, step_px=step_px)
    optimized_xyz = fused_resampled_xyz
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
        reason="z_search_refinement_skips_y_optimizer",
        partial_forward_xy=partial_forward[:, :2].astype(np.float32, copy=False),
        partial_reverse_xy=partial_reverse[:, :2].astype(np.float32, copy=False),
        fused_dense_xy=fused_dense_xyz[:, :2].astype(np.float32, copy=False),
        fused_resampled_xy=fused_resampled_xyz[:, :2].astype(np.float32, copy=False),
        optimized_xy=optimized_xyz[:, :2].astype(np.float32, copy=False),
        partial_forward_z=partial_forward[:, 2].astype(np.float32, copy=False),
        partial_reverse_z=partial_reverse[:, 2].astype(np.float32, copy=False),
        fused_dense_z=fused_dense_xyz[:, 2].astype(np.float32, copy=False),
        fused_resampled_z=fused_resampled_xyz[:, 2].astype(np.float32, copy=False),
        optimized_z=optimized_xyz[:, 2].astype(np.float32, copy=False),
        closest_forward_z_voxels=forward_z,
        closest_reverse_z_voxels=reverse_z,
        closest_midpoint_z_voxels=midpoint_z,
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
    metric = _trace2cp_metric_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        shape_hw=(int(direction_xy.shape[0]), int(direction_xy.shape[1])),
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
    return _Trace2CpBidirectionalResult(forward=forward, reverse=reverse, refinement=refinement, metric=metric)


def _trace2cp_metric_bidirectional(
    direction_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> _Trace2CpMetricResult:
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
    return _trace2cp_metric_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        shape_hw=(int(direction_xy.shape[0]), int(direction_xy.shape[1])),
        rf_margin_px=rf_margin_px,
    )


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
    metric = _trace2cp_metric_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
    )
    return _Trace2CpBidirectionalResult(forward=forward, reverse=reverse, refinement=refinement, metric=metric)


def _trace_score_trace2cp_combined_direction(
    *,
    direction_xy: np.ndarray | None,
    tta_fields: list[_TtaDirectionField] | None,
    embedding_chw: np.ndarray | None,
    presence_hw: np.ndarray | None,
    valid_mask: np.ndarray | None,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    start_embedding: np.ndarray | None,
    target_embedding: np.ndarray | None,
    fiber_embeddings: np.ndarray | None,
    weights: _Trace2CpCombinedWeights,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> tuple[_Trace2CpDirectionResult, _Trace2CpCombinedTraceStats]:
    if direction_xy is not None:
        shape_hw = tuple(int(v) for v in np.asarray(direction_xy).shape[:2])
    elif presence_hw is not None:
        shape_hw = tuple(int(v) for v in np.asarray(presence_hw).shape[:2])
    elif embedding_chw is not None:
        emb = np.asarray(embedding_chw)
        shape_hw = (int(emb.shape[1]), int(emb.shape[2]))
    elif tta_fields:
        shape_hw = tuple(int(v) for v in np.asarray(tta_fields[0].direction_xy).shape[:2])
    else:
        raise ValueError("trace2cp combined direction scoring requires a spatial prediction field")
    traced_line, reason, stats = _trace_combined_direction_line_to_target(
        direction_xy=direction_xy,
        tta_fields=tta_fields,
        embedding_chw=embedding_chw,
        presence_hw=presence_hw,
        valid_mask=valid_mask,
        start_xy=start_xy,
        target_xy=target_xy,
        start_embedding=start_embedding,
        target_embedding=target_embedding,
        fiber_embeddings=fiber_embeddings,
        weights=weights,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
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
    return _Trace2CpDirectionResult(trace_xy=traced_line, result=result), stats


def _trace_score_trace2cp_combined_bidirectional(
    *,
    direction_xy: np.ndarray | None,
    tta_fields: list[_TtaDirectionField] | None,
    embedding_chw: np.ndarray | None,
    presence_hw: np.ndarray | None,
    valid_mask: np.ndarray | None,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    fiber_embeddings: np.ndarray | None,
    weights: _Trace2CpCombinedWeights,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    fiber_bank_skipped: int = 0,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> tuple[_Trace2CpBidirectionalResult, _Trace2CpCombinedSummary]:
    embedding_required = any(
        float(value) != 0.0 for value in (weights.last, weights.enclosing, weights.fiber)
    )
    embedding: np.ndarray | None = None
    start_embedding: np.ndarray | None = None
    target_embedding: np.ndarray | None = None
    if embedding_required:
        embedding = _require_trace2cp_embedding_field(embedding_chw)
        start_embedding = _bilinear_embedding_sample(embedding, start_xy, valid_mask=valid_mask)
        target_embedding = _bilinear_embedding_sample(embedding, target_xy, valid_mask=valid_mask)
        if start_embedding is None:
            raise ValueError("trace2cp combined tracing could not sample start CP embedding")
        if target_embedding is None:
            raise ValueError("trace2cp combined tracing could not sample target CP embedding")
        shape_hw = (int(embedding.shape[1]), int(embedding.shape[2]))
    elif direction_xy is not None:
        shape_hw = tuple(int(v) for v in np.asarray(direction_xy).shape[:2])
    elif presence_hw is not None:
        shape_hw = tuple(int(v) for v in np.asarray(presence_hw).shape[:2])
    elif tta_fields:
        shape_hw = tuple(int(v) for v in np.asarray(tta_fields[0].direction_xy).shape[:2])
    else:
        raise ValueError("trace2cp combined scoring requires a direction, TTA, presence, or embedding field")
    forward, forward_stats = _trace_score_trace2cp_combined_direction(
        direction_xy=direction_xy,
        tta_fields=tta_fields,
        embedding_chw=embedding,
        presence_hw=presence_hw,
        valid_mask=valid_mask,
        start_xy=start_xy,
        target_xy=target_xy,
        start_embedding=start_embedding,
        target_embedding=target_embedding,
        fiber_embeddings=fiber_embeddings,
        weights=weights,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    reverse, reverse_stats = _trace_score_trace2cp_combined_direction(
        direction_xy=direction_xy,
        tta_fields=tta_fields,
        embedding_chw=embedding,
        presence_hw=presence_hw,
        valid_mask=valid_mask,
        start_xy=target_xy,
        target_xy=start_xy,
        start_embedding=target_embedding,
        target_embedding=start_embedding,
        fiber_embeddings=fiber_embeddings,
        weights=weights,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    metric = _trace2cp_metric_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
    )
    reference_direction = direction_xy if direction_xy is not None else tta_fields[0].direction_xy if tta_fields else None
    reference_valid = valid_mask if valid_mask is not None else tta_fields[0].valid_mask if tta_fields else None
    if reference_direction is None:
        raise ValueError("trace2cp combined refinement requires a reference direction field")
    refinement = _trace2cp_refinement_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        direction_xy=reference_direction,
        valid_mask=reference_valid,
        shape_hw=shape_hw,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    summary = _Trace2CpCombinedSummary(
        forward=forward_stats,
        reverse=reverse_stats,
        candidate_angles_degrees=_trace2cp_candidate_angles_degrees(candidate_max_degrees, candidate_step_degrees),
        fiber_bank_size=int(np.asarray(fiber_embeddings).shape[0]) if fiber_embeddings is not None and np.asarray(fiber_embeddings).ndim == 2 else 0,
        fiber_bank_skipped=int(fiber_bank_skipped),
    )
    return (
        _Trace2CpBidirectionalResult(forward=forward, reverse=reverse, refinement=refinement, metric=metric),
        summary,
    )


def _trace_score_trace2cp_image_combined_direction(
    *,
    direction_xy: np.ndarray | None,
    tta_fields: list[_TtaDirectionField] | None,
    image_hw: np.ndarray,
    valid_mask: np.ndarray,
    presence_hw: np.ndarray | None,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    weights: _Trace2CpCombinedWeights,
    image_config: _Trace2CpImageScoringConfig | None,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> tuple[_Trace2CpDirectionResult, _Trace2CpCombinedTraceStats]:
    traced_line, reason, stats = _trace_combined_image_line_to_target(
        direction_xy=direction_xy,
        tta_fields=tta_fields,
        image_hw=image_hw,
        valid_mask=valid_mask,
        presence_hw=presence_hw,
        start_xy=start_xy,
        target_xy=target_xy,
        weights=weights,
        image_config=image_config,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    result = _score_trace2cp(
        traced_line,
        target_xy,
        shape_hw=tuple(int(v) for v in np.asarray(image_hw).shape),
        rf_margin_px=rf_margin_px,
        termination_reason=reason,
    )
    return _Trace2CpDirectionResult(trace_xy=traced_line, result=result), stats


def _trace_score_trace2cp_image_combined_bidirectional(
    *,
    direction_xy: np.ndarray | None,
    tta_fields: list[_TtaDirectionField] | None,
    image_hw: np.ndarray,
    valid_mask: np.ndarray,
    presence_hw: np.ndarray | None,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    weights: _Trace2CpCombinedWeights,
    image_config: _Trace2CpImageScoringConfig | None,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> tuple[_Trace2CpBidirectionalResult, _Trace2CpCombinedSummary]:
    image = np.asarray(image_hw, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if image.ndim != 2 or valid.shape != image.shape:
        raise ValueError("trace2cp image combined requires image and valid mask with matching H,W shapes")
    forward, forward_stats = _trace_score_trace2cp_image_combined_direction(
        direction_xy=direction_xy,
        tta_fields=tta_fields,
        image_hw=image,
        valid_mask=valid,
        presence_hw=presence_hw,
        start_xy=start_xy,
        target_xy=target_xy,
        weights=weights,
        image_config=image_config,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    reverse, reverse_stats = _trace_score_trace2cp_image_combined_direction(
        direction_xy=direction_xy,
        tta_fields=tta_fields,
        image_hw=image,
        valid_mask=valid,
        presence_hw=presence_hw,
        start_xy=target_xy,
        target_xy=start_xy,
        weights=weights,
        image_config=image_config,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    metric = _trace2cp_metric_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        shape_hw=(int(image.shape[0]), int(image.shape[1])),
        rf_margin_px=rf_margin_px,
    )
    reference_direction = direction_xy if direction_xy is not None else tta_fields[0].direction_xy if tta_fields else None
    reference_valid = valid_mask if valid_mask is not None else tta_fields[0].valid_mask if tta_fields else None
    if reference_direction is None:
        raise ValueError("trace2cp image combined refinement requires a reference direction field")
    refinement = _trace2cp_refinement_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        direction_xy=reference_direction,
        valid_mask=reference_valid,
        shape_hw=(int(image.shape[0]), int(image.shape[1])),
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    summary = _Trace2CpCombinedSummary(
        forward=forward_stats,
        reverse=reverse_stats,
        candidate_angles_degrees=_trace2cp_candidate_angles_degrees(candidate_max_degrees, candidate_step_degrees),
        fiber_bank_size=0,
        fiber_bank_skipped=0,
    )
    return (
        _Trace2CpBidirectionalResult(forward=forward, reverse=reverse, refinement=refinement, metric=metric),
        summary,
    )


def _trace_score_trace2cp_combined_z_bidirectional(
    *,
    plane_cache: _Trace2CpZPlaneCache,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    fiber_embeddings: np.ndarray | None,
    weights: _Trace2CpCombinedWeights,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    fiber_bank_skipped: int = 0,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> tuple[_Trace2CpBidirectionalResult, _Trace2CpCombinedSummary, np.ndarray, np.ndarray]:
    center = plane_cache.get(0)
    shape_hw = tuple(int(v) for v in np.asarray(center.fields.direction_xy).shape[:2])
    embedding_required = any(
        float(value) != 0.0 for value in (weights.last, weights.enclosing, weights.fiber)
    )
    start_embedding: np.ndarray | None = None
    target_embedding: np.ndarray | None = None
    if embedding_required:
        embedding = _require_trace2cp_embedding_field(center.fields.embedding_chw)
        start_embedding = _bilinear_embedding_sample(embedding, start_xy, valid_mask=center.valid_mask)
        target_embedding = _bilinear_embedding_sample(embedding, target_xy, valid_mask=center.valid_mask)
        if start_embedding is None:
            raise ValueError("trace2cp z-search could not sample start CP embedding")
        if target_embedding is None:
            raise ValueError("trace2cp z-search could not sample target CP embedding")
        shape_hw = (int(embedding.shape[1]), int(embedding.shape[2]))
    plane_cache.ensure_neighbors(0)
    forward_xyz, forward_reason, forward_stats = _trace_combined_direction_line_to_target_z(
        plane_cache=plane_cache,
        start_xy=start_xy,
        target_xy=target_xy,
        start_embedding=start_embedding,
        target_embedding=target_embedding,
        fiber_embeddings=fiber_embeddings,
        weights=weights,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    reverse_xyz, reverse_reason, reverse_stats = _trace_combined_direction_line_to_target_z(
        plane_cache=plane_cache,
        start_xy=target_xy,
        target_xy=start_xy,
        start_embedding=target_embedding,
        target_embedding=start_embedding,
        fiber_embeddings=fiber_embeddings,
        weights=weights,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    forward_xy = forward_xyz[:, :2].astype(np.float32, copy=False)
    reverse_xy = reverse_xyz[:, :2].astype(np.float32, copy=False)
    forward_result = _score_trace2cp(
        forward_xy,
        target_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
        termination_reason=forward_reason,
    )
    reverse_result = _score_trace2cp(
        reverse_xy,
        start_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
        termination_reason=reverse_reason,
    )
    forward = _Trace2CpDirectionResult(trace_xy=forward_xy, result=forward_result)
    reverse = _Trace2CpDirectionResult(trace_xy=reverse_xy, result=reverse_result)
    metric = _trace2cp_metric_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
    )
    refinement = _trace2cp_refinement_from_traces_z(
        forward_xyz,
        reverse_xyz,
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    summary = _Trace2CpCombinedSummary(
        forward=forward_stats,
        reverse=reverse_stats,
        candidate_angles_degrees=_trace2cp_candidate_angles_degrees(candidate_max_degrees, candidate_step_degrees),
        fiber_bank_size=int(np.asarray(fiber_embeddings).shape[0]) if fiber_embeddings is not None and np.asarray(fiber_embeddings).ndim == 2 else 0,
        fiber_bank_skipped=int(fiber_bank_skipped),
    )
    return (
        _Trace2CpBidirectionalResult(forward=forward, reverse=reverse, refinement=refinement, metric=metric),
        summary,
        forward_xyz.astype(np.float32, copy=False),
        reverse_xyz.astype(np.float32, copy=False),
    )


def _trace_score_trace2cp_image_combined_z_bidirectional(
    *,
    plane_cache: _Trace2CpZPlaneCache,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    weights: _Trace2CpCombinedWeights,
    image_config: _Trace2CpImageScoringConfig | None,
    candidate_max_degrees: float,
    candidate_step_degrees: float,
    step_px: float = 1.0,
    rf_margin_px: float = 5.0,
) -> tuple[_Trace2CpBidirectionalResult, _Trace2CpCombinedSummary, np.ndarray, np.ndarray]:
    center = plane_cache.get(0)
    center_image = np.asarray(center.image, dtype=np.float32)
    if center_image.ndim != 2:
        raise ValueError("trace2cp image z-search requires a center image")
    plane_cache.ensure_neighbors(0)
    forward_xyz, forward_reason, forward_stats = _trace_combined_image_line_to_target_z(
        plane_cache=plane_cache,
        start_xy=start_xy,
        target_xy=target_xy,
        weights=weights,
        image_config=image_config,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    reverse_xyz, reverse_reason, reverse_stats = _trace_combined_image_line_to_target_z(
        plane_cache=plane_cache,
        start_xy=target_xy,
        target_xy=start_xy,
        weights=weights,
        image_config=image_config,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    forward_xy = forward_xyz[:, :2].astype(np.float32, copy=False)
    reverse_xy = reverse_xyz[:, :2].astype(np.float32, copy=False)
    shape_hw = (int(center_image.shape[0]), int(center_image.shape[1]))
    forward_result = _score_trace2cp(
        forward_xy,
        target_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
        termination_reason=forward_reason,
    )
    reverse_result = _score_trace2cp(
        reverse_xy,
        start_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
        termination_reason=reverse_reason,
    )
    forward = _Trace2CpDirectionResult(trace_xy=forward_xy, result=forward_result)
    reverse = _Trace2CpDirectionResult(trace_xy=reverse_xy, result=reverse_result)
    metric = _trace2cp_metric_from_traces(
        forward.trace_xy,
        reverse.trace_xy,
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        rf_margin_px=rf_margin_px,
    )
    refinement = _trace2cp_refinement_from_traces_z(
        forward_xyz,
        reverse_xyz,
        start_xy,
        target_xy,
        shape_hw=shape_hw,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    summary = _Trace2CpCombinedSummary(
        forward=forward_stats,
        reverse=reverse_stats,
        candidate_angles_degrees=_trace2cp_candidate_angles_degrees(candidate_max_degrees, candidate_step_degrees),
        fiber_bank_size=0,
        fiber_bank_skipped=0,
    )
    return (
        _Trace2CpBidirectionalResult(forward=forward, reverse=reverse, refinement=refinement, metric=metric),
        summary,
        forward_xyz.astype(np.float32, copy=False),
        reverse_xyz.astype(np.float32, copy=False),
    )


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


def _similarity_map_to_u8(similarity: np.ndarray | None, valid_mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid.shape, dtype=np.uint8)
    if similarity is None:
        return out
    values = np.asarray(similarity, dtype=np.float32)
    if values.shape != valid.shape:
        raise ValueError("similarity map must match valid mask shape")
    mask = valid & np.isfinite(values)
    out[mask] = np.clip((values[mask] + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    return out


def _presence_map_to_u8(presence: np.ndarray | None, valid_mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)
    out = np.zeros(valid.shape, dtype=np.uint8)
    if presence is None:
        return out
    values = np.asarray(presence, dtype=np.float32)
    if values.shape != valid.shape:
        raise ValueError("presence map must match valid mask shape")
    mask = valid & np.isfinite(values)
    out[mask] = np.clip(values[mask] * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def _draw_trace2cp_presence_panel(
    presence: np.ndarray | None,
    valid_mask: np.ndarray,
    *,
    line_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    bidirectional_result: _Trace2CpBidirectionalResult,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    panel = np.repeat(_presence_map_to_u8(presence, valid_mask)[..., None], 3, axis=-1)
    panel = overlay_line_coords_rgb(panel, line_xy, opacity=0.22, thickness=1)
    panel = _overlay_polyline_rgb(
        panel,
        bidirectional_result.forward.trace_xy,
        color_rgba=(0, 255, 0, 230),
        thickness=1,
    )
    panel = _overlay_polyline_rgb(
        panel,
        bidirectional_result.reverse.trace_xy,
        color_rgba=(255, 64, 220, 230),
        thickness=1,
    )
    pil = Image.fromarray(panel, mode="RGB")
    draw = ImageDraw.Draw(pil, mode="RGBA")
    for point_xy, color in (
        (start_xy, (32, 255, 255, 255)),
        (target_xy, (255, 220, 64, 255)),
    ):
        point = np.asarray(point_xy, dtype=np.float32)
        if point.shape == (2,) and bool(np.isfinite(point).all()):
            x, y = (float(v) for v in point)
            draw.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), outline=color, width=1)
    return _draw_label_band(np.asarray(pil, dtype=np.uint8), "presence 0..1")


def _draw_trace2cp_similarity_panel(
    similarity: np.ndarray | None,
    valid_mask: np.ndarray,
    *,
    line_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    trace_xy: np.ndarray | None,
    label: str,
    trace_color: tuple[int, int, int, int] = (255, 255, 255, 220),
) -> np.ndarray:
    from PIL import Image, ImageDraw

    panel = np.repeat(_similarity_map_to_u8(similarity, valid_mask)[..., None], 3, axis=-1)
    panel = overlay_line_coords_rgb(panel, line_xy, opacity=0.22, thickness=1)
    if trace_xy is not None:
        panel = _overlay_polyline_rgb(panel, trace_xy, color_rgba=trace_color, thickness=1)
    pil = Image.fromarray(panel, mode="RGB")
    draw = ImageDraw.Draw(pil, mode="RGBA")
    for point_xy, color in (
        (start_xy, (32, 255, 255, 255)),
        (target_xy, (255, 220, 64, 255)),
    ):
        point = np.asarray(point_xy, dtype=np.float32)
        if point.shape == (2,) and bool(np.isfinite(point).all()):
            x, y = (float(v) for v in point)
            draw.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), outline=color, width=1)
    return _draw_label_band(np.asarray(pil, dtype=np.uint8), label)


def _draw_trace2cp_similarity_debug_column(
    debug: _Trace2CpSimilarityDebug,
    *,
    valid_mask: np.ndarray,
    line_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    bidirectional_result: _Trace2CpBidirectionalResult,
) -> np.ndarray:
    panels = [
        _draw_trace2cp_similarity_panel(
            debug.start_cp_similarity,
            valid_mask,
            line_xy=line_xy,
            start_xy=start_xy,
            target_xy=target_xy,
            trace_xy=None,
            label="emb sim start_cp",
        ),
        _draw_trace2cp_similarity_panel(
            debug.target_cp_similarity,
            valid_mask,
            line_xy=line_xy,
            start_xy=start_xy,
            target_xy=target_xy,
            trace_xy=None,
            label="emb sim target_cp",
        ),
        _draw_trace2cp_similarity_panel(
            debug.global_similarity,
            valid_mask,
            line_xy=line_xy,
            start_xy=start_xy,
            target_xy=target_xy,
            trace_xy=None,
            label=f"emb sim fiber_global n={int(debug.global_bank_size)}",
        ),
        _draw_trace2cp_similarity_panel(
            debug.forward_last_similarity,
            valid_mask,
            line_xy=line_xy,
            start_xy=start_xy,
            target_xy=target_xy,
            trace_xy=bidirectional_result.forward.trace_xy,
            label="emb sim forward_last",
            trace_color=(0, 255, 0, 230),
        ),
        _draw_trace2cp_similarity_panel(
            debug.reverse_last_similarity,
            valid_mask,
            line_xy=line_xy,
            start_xy=start_xy,
            target_xy=target_xy,
            trace_xy=bidirectional_result.reverse.trace_xy,
            label="emb sim reverse_last",
            trace_color=(255, 64, 220, 230),
        ),
    ]
    width = max(int(panel.shape[1]) for panel in panels)
    total_h = sum(int(panel.shape[0]) for panel in panels)
    stack = np.zeros((total_h, width, 3), dtype=np.uint8)
    y0 = 0
    for panel in panels:
        h, w = panel.shape[:2]
        stack[y0 : y0 + h, :w] = panel
        y0 += h
    return stack


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
    similarity_debug: _Trace2CpSimilarityDebug | None = None,
    z_search_debug: _Trace2CpZTraceDebug | None = None,
    presence_debug: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
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
                    f"{label_prefix} traces metric={result.metric.error:.4f} "
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

    def z_stack(debug: _Trace2CpZTraceDebug) -> np.ndarray:
        def layer_counts_text(layers_by_column: np.ndarray) -> str:
            layers = np.asarray(layers_by_column, dtype=np.int32).reshape(-1)
            valid_layers = layers[layers > -9999]
            if valid_layers.size == 0:
                return "layers=none"
            values, counts = np.unique(valid_layers, return_counts=True)
            return "layers=" + ",".join(f"{int(v)}:{int(c)}" for v, c in zip(values.tolist(), counts.tolist()))

        def draw_z_row(
            image: np.ndarray,
            trace_xy: np.ndarray,
            *,
            label: str,
            color: tuple[int, int, int, int],
        ) -> np.ndarray:
            overlay = overlay_line_coords_rgb(image, line_xy, opacity=0.25, thickness=1)
            overlay = _overlay_polyline_rgb(overlay, trace_xy, color_rgba=color, thickness=1)
            return _draw_label_band(np.asarray(overlay, dtype=np.uint8), label)

        f_min, f_max, f_mean_abs = debug.z_stats(debug.forward_trace_xyz)
        r_min, r_max, r_mean_abs = debug.z_stats(debug.reverse_trace_xyz)
        fused_min, fused_max, fused_mean_abs = debug.z_stats(debug.fused_trace_xyz)
        layer_map = _z_layer_columns_to_rgb(
            debug.fused_layer_columns,
            height=int(debug.fused_z_image.shape[0]),
            max_layer=int(debug.max_layer),
        )
        rows = [
            draw_z_row(
                debug.forward_z_image,
                debug.forward_trace_xyz[:, :2],
                label=(
                    f"z forward layers={debug.layer_min}..{debug.layer_max} "
                    f"z={f_min:.1f}..{f_max:.1f} mean_abs={f_mean_abs:.1f} "
                    f"missing_cols={debug.forward_missing_columns} "
                    f"{layer_counts_text(debug.forward_layer_columns)}"
                ),
                color=(0, 255, 0, 230),
            ),
            draw_z_row(
                debug.reverse_z_image,
                debug.reverse_trace_xyz[:, :2],
                label=(
                    f"z reverse layers={debug.layer_min}..{debug.layer_max} "
                    f"z={r_min:.1f}..{r_max:.1f} mean_abs={r_mean_abs:.1f} "
                    f"missing_cols={debug.reverse_missing_columns} "
                    f"{layer_counts_text(debug.reverse_layer_columns)}"
                ),
                color=(255, 64, 220, 230),
            ),
            draw_z_row(
                debug.fused_z_image,
                debug.fused_trace_xyz[:, :2],
                label=(
                    f"z fused corrected z={fused_min:.1f}..{fused_max:.1f} "
                    f"mean_abs={fused_mean_abs:.1f} missing_cols={debug.fused_missing_columns} "
                    f"{layer_counts_text(debug.fused_layer_columns)}"
                ),
                color=(255, 220, 64, 230),
            ),
            _draw_label_band(
                layer_map,
                label=(
                    "z fused layer map "
                    "blue=negative green=center red=positive "
                    f"{layer_counts_text(debug.fused_layer_columns)}"
                ),
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
    if z_search_debug is not None:
        stacks.append(z_stack(z_search_debug))
    if presence_debug is not None:
        presence_valid = (
            np.ones(np.asarray(image_u8).shape[:2], dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        )
        stacks.append(
            _draw_trace2cp_presence_panel(
                presence_debug,
                presence_valid,
                line_xy=line_xy,
                start_xy=start_xy,
                target_xy=target_xy,
                bidirectional_result=bidirectional_result,
            )
        )
    if similarity_debug is not None:
        sim_valid = (
            np.ones(np.asarray(image_u8).shape[:2], dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool)
        )
        stacks.append(
            _draw_trace2cp_similarity_debug_column(
                similarity_debug,
                valid_mask=sim_valid,
                line_xy=line_xy,
                start_xy=start_xy,
                target_xy=target_xy,
                bidirectional_result=bidirectional_result,
            )
        )
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


def _trace2cp_skip_reason(exc: BaseException) -> str:
    return " ".join(str(exc).split())


def _mapped_trace2cp_points(
    points_xy: np.ndarray,
    *,
    x_scale: float,
    x_offset: float,
    y_shift: float,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    mapped = points.copy()
    mapped[:, 0] = np.float32(x_offset) + np.float32(x_scale) * mapped[:, 0]
    mapped[:, 1] += np.float32(y_shift)
    return mapped


def _trace2cp_unit_or_zero(values: np.ndarray) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32).reshape(-1)
    if vec.shape != (3,):
        return np.zeros(3, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float32)
    return (vec / np.float32(norm)).astype(np.float32, copy=False)


def _fmt_vec(values: np.ndarray, *, precision: int = 4) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return "(" + ",".join(f"{float(v):.{precision}f}" for v in arr.tolist()) + ")"


def _trace2cp_pair_debug_row(
    evaluation: _Trace2CpPairEvaluation,
    *,
    start_line_index: int,
    target_line_index: int,
    alignment_line_index: int | None,
    alignment_reference_xyz: np.ndarray | None,
) -> str:
    sample = evaluation.sample
    start_xy = np.asarray(sample.start_control_point_xy, dtype=np.float32)
    target_xy = np.asarray(sample.target_control_point_xy, dtype=np.float32)
    dxy = target_xy - start_xy
    start_xyz = np.asarray(sample.start_control_point_xyz, dtype=np.float32)
    target_xyz = np.asarray(sample.target_control_point_xyz, dtype=np.float32)
    delta_xyz = target_xyz - start_xyz
    frame = sample.frame
    if frame is None:
        tangent = np.zeros(3, dtype=np.float32)
        side = np.zeros(3, dtype=np.float32)
        normal = np.zeros(3, dtype=np.float32)
    else:
        tangent = _trace2cp_unit_or_zero(np.asarray(frame.tangent_xyz, dtype=np.float32))
        side = _trace2cp_unit_or_zero(np.asarray(frame.side_xyz, dtype=np.float32))
        normal = _trace2cp_unit_or_zero(np.asarray(frame.mesh_normal_xyz, dtype=np.float32))
    start_axis = _trace2cp_unit_or_zero(np.asarray(sample.start_row_axis_xyz, dtype=np.float32))
    target_axis = _trace2cp_unit_or_zero(np.asarray(sample.target_row_axis_xyz, dtype=np.float32))
    frame_dot_tsn = np.asarray(
        [
            float(np.dot(delta_xyz, tangent)),
            float(np.dot(delta_xyz, side)),
            float(np.dot(delta_xyz, normal)),
        ],
        dtype=np.float32,
    )
    alignment_dot = np.nan
    if alignment_line_index is not None and alignment_reference_xyz is not None:
        ref = _trace2cp_unit_or_zero(np.asarray(alignment_reference_xyz, dtype=np.float32))
        if int(alignment_line_index) == int(start_line_index):
            alignment_dot = float(np.dot(start_axis, ref))
        elif int(alignment_line_index) == int(target_line_index):
            alignment_dot = float(np.dot(target_axis, ref))
    return (
        "trace2cp fiber_pair_vectors "
        f"start_cp={int(sample.start_control_point_index)} "
        f"target_cp={int(sample.target_control_point_index)} "
        f"start_line={int(start_line_index)} target_line={int(target_line_index)} "
        f"start_xy={_fmt_vec(start_xy, precision=3)} "
        f"target_xy={_fmt_vec(target_xy, precision=3)} "
        f"dxy={_fmt_vec(dxy, precision=3)} "
        f"start_axis_xyz={_fmt_vec(start_axis, precision=5)} "
        f"target_axis_xyz={_fmt_vec(target_axis, precision=5)} "
        f"frame_tangent_xyz={_fmt_vec(tangent, precision=5)} "
        f"frame_side_xyz={_fmt_vec(side, precision=5)} "
        f"frame_normal_xyz={_fmt_vec(normal, precision=5)} "
        f"cp_delta_xyz={_fmt_vec(delta_xyz, precision=3)} "
        f"cp_delta_dot_tsn={_fmt_vec(frame_dot_tsn, precision=3)} "
        f"alignment_line={'' if alignment_line_index is None else int(alignment_line_index)} "
        f"alignment_dot={alignment_dot:.6f}"
    )


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
        target_cp = int(sample.target_control_point_index)
        if start_cp < 0 or start_cp >= cp_x.shape[0]:
            raise ValueError(f"start control point {start_cp} is outside control_point_x")
        if target_cp < 0 or target_cp >= cp_x.shape[0]:
            raise ValueError(f"target control point {target_cp} is outside control_point_x")
        start_xy = np.asarray(sample.start_control_point_xy, dtype=np.float32)
        target_xy = np.asarray(sample.target_control_point_xy, dtype=np.float32)
        if (
            start_xy.shape != (2,)
            or target_xy.shape != (2,)
            or not bool(np.isfinite(start_xy).all())
            or not bool(np.isfinite(target_xy).all())
        ):
            raise ValueError("Trace2CP sample has invalid start/target control point xy")
        image = np.asarray(evaluation.image)
        if image.ndim != 2:
            raise ValueError("Trace2CP fiber visualization expects 2D pair images")
        height, width = image.shape[:2]
        local_dx = float(target_xy[0] - start_xy[0])
        global_dx = float(cp_x[target_cp] - cp_x[start_cp])
        if abs(local_dx) <= 1.0e-6:
            x_scale = 1.0
        else:
            x_scale = global_dx / local_dx
        x_offset = float(cp_x[start_cp]) - x_scale * float(start_xy[0])
        y_shift = -float(start_xy[1])
        placements.append(
            _Trace2CpFiberPairPlacement(
                evaluation=evaluation,
                x_scale=x_scale,
                x_offset=x_offset,
                y_shift=y_shift,
            )
        )
        corner_x = x_offset + x_scale * np.asarray([0.0, float(width - 1)], dtype=np.float32)
        min_x = min(min_x, float(np.min(corner_x)))
        min_y = min(min_y, y_shift)
        max_x = max(max_x, float(np.max(corner_x)))
        max_y = max(max_y, y_shift + float(height - 1))

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
        rgb = np.repeat(image_u8[..., None], 3, axis=-1).astype(np.uint8)
        valid = np.asarray(evaluation.valid_mask, dtype=bool)
        if not bool(valid.any()):
            continue
        height, width = image_u8.shape[:2]
        x0 = global_x_shift + placement.x_offset
        x1 = global_x_shift + placement.x_offset + placement.x_scale * float(width - 1)
        if placement.x_scale < 0.0:
            rgb = np.flip(rgb, axis=1)
            valid = np.flip(valid, axis=1)
        dst_x0 = int(round(min(x0, x1)))
        dst_y0 = int(round(global_y_shift + placement.y_shift))
        src_y0 = max(0, -dst_y0)
        src_x0 = max(0, -dst_x0)
        dst_y = max(0, dst_y0)
        dst_x = max(0, dst_x0)
        copy_h = min(int(rgb.shape[0]) - src_y0, canvas_height - dst_y)
        copy_w = min(int(rgb.shape[1]) - src_x0, canvas_width - dst_x)
        if copy_h <= 0 or copy_w <= 0:
            continue
        valid_crop = valid[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
        rgb_crop = rgb[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w].astype(np.float32)
        if not bool(valid_crop.any()):
            continue
        target_slice = np.s_[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w]
        accum[target_slice] += rgb_crop * valid_crop[..., None].astype(np.float32)
        weight[target_slice] += valid_crop[..., None].astype(np.float32)

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    covered = weight[..., 0] > 0.0
    canvas[covered] = np.clip(accum[covered] / weight[covered], 0.0, 255.0).astype(np.uint8)

    def compose_presence_canvas() -> np.ndarray | None:
        if not any(placement.evaluation.presence_debug is not None for placement in placements):
            return None
        presence_accum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        presence_weight = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)
        for placement in placements:
            evaluation = placement.evaluation
            presence = evaluation.presence_debug
            if presence is None:
                continue
            valid = np.asarray(evaluation.valid_mask, dtype=bool)
            if not bool(valid.any()):
                continue
            presence_u8 = _presence_map_to_u8(presence, valid)
            rgb = np.repeat(presence_u8[..., None], 3, axis=-1).astype(np.uint8)
            _height, width = presence_u8.shape[:2]
            x0 = global_x_shift + placement.x_offset
            x1 = global_x_shift + placement.x_offset + placement.x_scale * float(width - 1)
            if placement.x_scale < 0.0:
                rgb = np.flip(rgb, axis=1)
                valid = np.flip(valid, axis=1)
            dst_x0 = int(round(min(x0, x1)))
            dst_y0 = int(round(global_y_shift + placement.y_shift))
            src_y0 = max(0, -dst_y0)
            src_x0 = max(0, -dst_x0)
            dst_y = max(0, dst_y0)
            dst_x = max(0, dst_x0)
            copy_h = min(int(rgb.shape[0]) - src_y0, canvas_height - dst_y)
            copy_w = min(int(rgb.shape[1]) - src_x0, canvas_width - dst_x)
            if copy_h <= 0 or copy_w <= 0:
                continue
            valid_crop = valid[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
            rgb_crop = rgb[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w].astype(np.float32)
            if not bool(valid_crop.any()):
                continue
            target_slice = np.s_[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w]
            presence_accum[target_slice] += rgb_crop * valid_crop[..., None].astype(np.float32)
            presence_weight[target_slice] += valid_crop[..., None].astype(np.float32)
        presence_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        presence_covered = presence_weight[..., 0] > 0.0
        if not bool(presence_covered.any()):
            return None
        presence_canvas[presence_covered] = np.clip(
            presence_accum[presence_covered] / presence_weight[presence_covered],
            0.0,
            255.0,
        ).astype(np.uint8)
        return presence_canvas

    presence_canvas = compose_presence_canvas()
    font = ImageFont.load_default()

    def map_points(points_xy: np.ndarray, placement: _Trace2CpFiberPairPlacement) -> np.ndarray:
        return _mapped_trace2cp_points(
            points_xy,
            x_scale=placement.x_scale,
            x_offset=placement.x_offset + global_x_shift,
            y_shift=placement.y_shift + global_y_shift,
        )

    def draw_points(
        draw: ImageDraw.ImageDraw,
        points_xy: np.ndarray,
        color: tuple[int, int, int, int],
        width: int = 1,
    ) -> None:
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

    def draw_cp_markers(draw: ImageDraw.ImageDraw, placement: _Trace2CpFiberPairPlacement) -> None:
        evaluation = placement.evaluation
        for point_xy, color in (
            (evaluation.sample.start_control_point_xy, (32, 255, 255, 255)),
            (evaluation.sample.target_control_point_xy, (255, 220, 64, 255)),
        ):
            point = map_points(np.asarray(point_xy, dtype=np.float32).reshape(1, 2), placement)
            if point.shape == (1, 2) and bool(np.isfinite(point).all()):
                x, y = (float(v) for v in point[0])
                draw.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), outline=color, width=1)

    def draw_row(row_label: str, kind: str, *, base_canvas: np.ndarray | None = None) -> np.ndarray:
        row_canvas = canvas if base_canvas is None else base_canvas
        pil = Image.fromarray(row_canvas, mode="RGB")
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, mode="RGBA")
        for placement in placements:
            evaluation = placement.evaluation
            result = evaluation.selected_result
            refinement = result.refinement
            draw_points(draw, map_points(evaluation.sample.line_xy, placement), (210, 210, 210, 95), width=1)
            if kind == "full":
                draw_points(draw, map_points(result.forward.trace_xy, placement), (0, 255, 0, 230), width=1)
                draw_points(draw, map_points(result.reverse.trace_xy, placement), (255, 64, 220, 230), width=1)
                midpoint_x = 0.5 * (
                    float(cp_x[int(evaluation.sample.start_control_point_index)])
                    + float(cp_x[int(evaluation.sample.target_control_point_index)])
                ) + global_x_shift
                draw.text(
                    (midpoint_x - 10.0, 2.0),
                    f"{result.metric.error:.3f}",
                    fill=(255, 255, 255, 220),
                    font=font,
                )
            elif kind == "partial":
                draw_points(draw, map_points(refinement.partial_forward_xy, placement), (0, 255, 0, 230), width=1)
                draw_points(draw, map_points(refinement.partial_reverse_xy, placement), (255, 64, 220, 230), width=1)
                point = map_points(refinement.closest_midpoint_xy.reshape(1, 2), placement)
                if point.shape == (1, 2) and bool(np.isfinite(point).all()):
                    x, y = (float(v) for v in point[0])
                    draw.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), outline=(255, 255, 255, 255), width=1)
            elif kind == "fused":
                draw_points(draw, map_points(refinement.fused_resampled_xy, placement), (255, 220, 64, 240), width=1)
            elif kind == "optimized":
                draw_points(draw, map_points(refinement.optimized_xy, placement), (64, 224, 255, 240), width=1)
            else:
                raise ValueError(f"unknown Trace2CP fiber row kind {kind!r}")
            draw_cp_markers(draw, placement)
        drawn = np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8)
        return _draw_label_band(drawn, row_label)

    rows = [
        draw_row(f"{label} full traces", "full"),
        draw_row("partial closest-approach traces", "partial"),
        draw_row("fused CP-to-CP line", "fused"),
        draw_row("optimized CP-to-CP line", "optimized"),
    ]
    if presence_canvas is not None:
        rows.append(draw_row("presence 0..1 full traces", "full", base_canvas=presence_canvas))
    width = max(int(row.shape[1]) for row in rows)
    height = sum(int(row.shape[0]) for row in rows)
    stacked = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = 0
    for row in rows:
        h, w = row.shape[:2]
        stacked[y0 : y0 + h, :w] = row
        y0 += h
    return stacked


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


def _build_trace2cp_embedding_bank(
    loader: FiberStrip2DLoader,
    model: torch.nn.Module,
    flat_indices: tuple[int, ...],
    *,
    device: torch.device,
    rf_margin_px: float,
) -> _Trace2CpEmbeddingBank:
    embeddings: list[np.ndarray] = []
    rows: list[str] = []
    skipped = 0
    cp_count = len(flat_indices)
    if cp_count <= 1:
        return _Trace2CpEmbeddingBank(
            embeddings=np.zeros((0, 0), dtype=np.float32),
            skipped=cp_count,
            rows=("embedding_bank_skipped=not_enough_control_points",),
        )
    for cp_index, flat_index in enumerate(flat_indices):
        target_cp = cp_index + 1 if cp_index + 1 < cp_count else cp_index - 1
        try:
            sample, image, valid_mask = loader.build_trace2cp_segment_patch(
                int(flat_index),
                target_control_point_index=int(target_cp),
                rf_margin_px=rf_margin_px,
                device=device,
                sample_mode="flat",
            )
            fields = _predict_trace2cp_fields(model, image, valid_mask, device=device)
            embedding = _require_trace2cp_embedding_field(fields.embedding_chw)
            cp_embedding = _bilinear_embedding_sample(
                embedding,
                np.asarray(sample.start_control_point_xy, dtype=np.float32),
                valid_mask=valid_mask,
            )
            if cp_embedding is None:
                raise ValueError("could not sample CP embedding")
            embeddings.append(cp_embedding.astype(np.float32))
            rows.append(
                f"embedding_bank_cp={int(cp_index)} flat_index={int(flat_index)} "
                f"target_cp={int(target_cp)} status=ok"
            )
        except ValueError as exc:
            skipped += 1
            rows.append(
                f"embedding_bank_cp={int(cp_index)} flat_index={int(flat_index)} "
                f"target_cp={int(target_cp)} status=skipped reason={_trace2cp_skip_reason(exc)}"
            )
    if embeddings:
        bank = np.stack(embeddings, axis=0).astype(np.float32)
    else:
        bank = np.zeros((0, 0), dtype=np.float32)
    return _Trace2CpEmbeddingBank(embeddings=bank, skipped=int(skipped), rows=tuple(rows))


def _raise_trace2cp_max_steps(
    result: _Trace2CpBidirectionalResult,
    *,
    mode: str,
    sample: FiberStripSegmentSample,
    image_shape_hw: tuple[int, int],
    step_px: float,
    rf_margin_px: float,
) -> None:
    failures: list[str] = []
    if result.forward.result.reason == "max_steps":
        failures.append(
            f"forward points={int(result.forward.trace_xy.shape[0])} "
            f"target_x={float(result.forward.result.target_x):.3f}"
        )
    if result.reverse.result.reason == "max_steps":
        failures.append(
            f"reverse points={int(result.reverse.trace_xy.shape[0])} "
            f"target_x={float(result.reverse.result.target_x):.3f}"
        )
    if not failures:
        return
    height, width = (int(v) for v in image_shape_hw)
    raise ValueError(
        "Trace2CP trace exhausted max_steps before reaching the opposite CP "
        "x-column; this is not a valid metric result. "
        f"mode={mode} fiber_path={sample.fiber_path} "
        f"start_cp={sample.start_control_point_index} "
        f"target_cp={sample.target_control_point_index} "
        f"image_shape_hw=({height},{width}) step_px={float(step_px):.3f} "
        f"rf_margin_px={float(rf_margin_px):.3f} "
        f"failures={'; '.join(failures)}"
    )


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
    combined: bool = False,
    combined_weights: _Trace2CpCombinedWeights | None = None,
    combined_fiber_bank: _Trace2CpEmbeddingBank | None = None,
    combined_mode: str = "embedding",
    image_scoring: _Trace2CpImageScoringConfig | None = None,
    candidate_max_degrees: float = 25.0,
    candidate_step_degrees: float = 1.0,
    z_search: _Trace2CpZSearchConfig | None = None,
    sample_mode: str = "random",
    row_axis_alignment_line_index: int | None = None,
    row_axis_alignment_xyz: np.ndarray | None = None,
    build_similarity_debug: bool = True,
) -> _Trace2CpPairEvaluation:
    segment_source = loader.build_trace2cp_segment_source(
        sample_index,
        target_control_point_index=target_cp_index,
        target_offset=target_offset,
        rf_margin_px=rf_margin_px,
        row_axis_alignment_line_index=row_axis_alignment_line_index,
        row_axis_alignment_xyz=row_axis_alignment_xyz,
        device=device,
        sample_mode=sample_mode,
    )
    sample, image, valid_mask = loader.sample_trace2cp_segment_source(segment_source)
    fields = _predict_trace2cp_fields(model, image, valid_mask, device=device)
    direction_xy = fields.direction_xy
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
    med_fields: list[_TtaDirectionField] | None = None
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
    combined_summary: _Trace2CpCombinedSummary | None = None
    debug_fiber_embeddings: np.ndarray | None = None
    z_search_debug: _Trace2CpZTraceDebug | None = None
    presence_debug: np.ndarray | None = None
    z_config = z_search or _Trace2CpZSearchConfig(enabled=False)
    if combined:
        weights = combined_weights or _Trace2CpCombinedWeights()
        if float(weights.presence) != 0.0:
            presence_debug = fields.presence_hw
        mode = str(combined_mode).strip().lower()
        if mode not in {"direction", "embedding", "image"}:
            raise ValueError(f"unsupported trace2cp combined mode: {combined_mode!r}")
        bank = combined_fiber_bank
        if mode == "embedding":
            bank = bank or _Trace2CpEmbeddingBank(
                embeddings=np.zeros((0, 0), dtype=np.float32),
                skipped=0,
                rows=(),
            )
            debug_fiber_embeddings = bank.embeddings
        if z_config.enabled:
            if med_tta:
                raise ValueError("trace2cp z-search does not support --med-tta")
            center_prediction = _Trace2CpZLayerPrediction(
                layer=0,
                z_voxels=0.0,
                sample=sample,
                image=np.asarray(image, dtype=np.float32),
                valid_mask=np.asarray(valid_mask, dtype=bool),
                fields=fields,
            )
            plane_cache = _Trace2CpZPlaneCache(
                loader=loader,
                model=model,
                sample_index=sample_index,
                target_cp_index=target_cp_index,
                target_offset=target_offset,
                rf_margin_px=rf_margin_px,
                sample_mode=sample_mode,
                row_axis_alignment_line_index=row_axis_alignment_line_index,
                row_axis_alignment_xyz=row_axis_alignment_xyz,
                device=device,
                segment_source=segment_source,
                center_layer=center_prediction,
                z_step_voxels=float(z_config.step_voxels),
                max_layer=int(z_config.max_layer),
            )
            if mode == "image":
                selected_result, combined_summary, forward_xyz, reverse_xyz = _trace_score_trace2cp_image_combined_z_bidirectional(
                    plane_cache=plane_cache,
                    start_xy=start_xy,
                    target_xy=target_xy,
                    weights=weights,
                    image_config=image_scoring,
                    candidate_max_degrees=float(candidate_max_degrees),
                    candidate_step_degrees=float(candidate_step_degrees),
                    step_px=step_px,
                    rf_margin_px=rf_margin_px,
                )
            elif mode == "embedding":
                assert bank is not None
                selected_result, combined_summary, forward_xyz, reverse_xyz = _trace_score_trace2cp_combined_z_bidirectional(
                    plane_cache=plane_cache,
                    start_xy=start_xy,
                    target_xy=target_xy,
                    fiber_embeddings=bank.embeddings,
                    weights=weights,
                    candidate_max_degrees=float(candidate_max_degrees),
                    candidate_step_degrees=float(candidate_step_degrees),
                    fiber_bank_skipped=int(bank.skipped),
                    step_px=step_px,
                    rf_margin_px=rf_margin_px,
                )
            else:
                selected_result, combined_summary, forward_xyz, reverse_xyz = _trace_score_trace2cp_combined_z_bidirectional(
                    plane_cache=plane_cache,
                    start_xy=start_xy,
                    target_xy=target_xy,
                    fiber_embeddings=None,
                    weights=weights,
                    candidate_max_degrees=float(candidate_max_degrees),
                    candidate_step_degrees=float(candidate_step_degrees),
                    fiber_bank_skipped=0,
                    step_px=step_px,
                    rf_margin_px=rf_margin_px,
                )
            forward_z_image, forward_missing, forward_layer_columns = _trace2cp_z_corrected_image_u8(
                plane_cache=plane_cache,
                trace_xyz=forward_xyz,
                fallback_shape_hw=image.shape,
            )
            reverse_z_image, reverse_missing, reverse_layer_columns = _trace2cp_z_corrected_image_u8(
                plane_cache=plane_cache,
                trace_xyz=reverse_xyz,
                fallback_shape_hw=image.shape,
            )
            fused_xyz = _trace2cp_refinement_fused_xyz(selected_result.refinement)
            fused_z_image, fused_missing, fused_layer_columns = _trace2cp_z_corrected_image_u8(
                plane_cache=plane_cache,
                trace_xyz=fused_xyz,
                fallback_shape_hw=image.shape,
            )
            z_search_debug = _Trace2CpZTraceDebug(
                forward_trace_xyz=forward_xyz,
                reverse_trace_xyz=reverse_xyz,
                fused_trace_xyz=fused_xyz,
                forward_z_image=forward_z_image,
                reverse_z_image=reverse_z_image,
                fused_z_image=fused_z_image,
                forward_missing_columns=int(forward_missing),
                reverse_missing_columns=int(reverse_missing),
                fused_missing_columns=int(fused_missing),
                forward_layer_columns=forward_layer_columns,
                reverse_layer_columns=reverse_layer_columns,
                fused_layer_columns=fused_layer_columns,
                layers=plane_cache.layer_indices(),
                z_step_voxels=float(z_config.step_voxels),
                max_layer=int(z_config.max_layer),
            )
            selected_mode = {
                "direction": "combined_direction_z",
                "embedding": "combined_embedding_z",
                "image": "combined_image_z",
            }[mode]
        else:
            if mode == "image":
                selected_result, combined_summary = _trace_score_trace2cp_image_combined_bidirectional(
                    direction_xy=None if med_tta else direction_xy,
                    tta_fields=med_fields if med_tta else None,
                    image_hw=image,
                    valid_mask=valid_mask,
                    presence_hw=fields.presence_hw,
                    start_xy=start_xy,
                    target_xy=target_xy,
                    weights=weights,
                    image_config=image_scoring,
                    candidate_max_degrees=float(candidate_max_degrees),
                    candidate_step_degrees=float(candidate_step_degrees),
                    step_px=step_px,
                    rf_margin_px=rf_margin_px,
                )
                selected_mode = "combined_image_med_tta" if med_tta else "combined_image"
            else:
                embedding_for_trace = None
                if mode == "embedding":
                    assert bank is not None
                    embedding_for_trace = _require_trace2cp_embedding_field(fields.embedding_chw)
                selected_result, combined_summary = _trace_score_trace2cp_combined_bidirectional(
                    direction_xy=None if med_tta else direction_xy,
                    tta_fields=med_fields if med_tta else None,
                    embedding_chw=embedding_for_trace,
                    presence_hw=fields.presence_hw,
                    valid_mask=valid_mask,
                    start_xy=start_xy,
                    target_xy=target_xy,
                    fiber_embeddings=None if bank is None else bank.embeddings,
                    weights=weights,
                    candidate_max_degrees=float(candidate_max_degrees),
                    candidate_step_degrees=float(candidate_step_degrees),
                    fiber_bank_skipped=0 if bank is None else int(bank.skipped),
                    step_px=step_px,
                    rf_margin_px=rf_margin_px,
                )
                if mode == "embedding":
                    selected_mode = "combined_embedding_med_tta" if med_tta else "combined_embedding"
                else:
                    selected_mode = "combined_direction_med_tta" if med_tta else "combined_direction"
    _raise_trace2cp_max_steps(
        base_result,
        mode="reference",
        sample=sample,
        image_shape_hw=tuple(int(v) for v in image.shape),
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    if selected_result is not base_result:
        _raise_trace2cp_max_steps(
            selected_result,
            mode=selected_mode,
            sample=sample,
            image_shape_hw=tuple(int(v) for v in image.shape),
            step_px=step_px,
            rf_margin_px=rf_margin_px,
        )
    similarity_debug = (
        _trace2cp_similarity_debug(
            fields.embedding_chw,
            valid_mask,
            start_xy=start_xy,
            target_xy=target_xy,
            forward_trace_xy=selected_result.forward.trace_xy,
            reverse_trace_xy=selected_result.reverse.trace_xy,
            step_px=step_px,
            fiber_embeddings=debug_fiber_embeddings,
        )
        if build_similarity_debug
        else None
    )
    return _Trace2CpPairEvaluation(
        sample_index=int(sample_index),
        sample=sample,
        image=image,
        valid_mask=valid_mask,
        base_result=base_result,
        selected_result=selected_result,
        selected_mode=selected_mode,
        combined_summary=combined_summary,
        tta_count=tta_count,
        med_fields_count=med_fields_count,
        tta_rows=tuple(tta_rows),
        tta_debug_entries=tuple(tta_debug_entries),
        similarity_debug=similarity_debug,
        z_search_debug=z_search_debug,
        presence_debug=presence_debug,
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
    combined: bool = False,
    combined_weights: _Trace2CpCombinedWeights | None = None,
    combined_mode: str = "embedding",
    image_scoring: _Trace2CpImageScoringConfig | None = None,
    candidate_max_degrees: float = 25.0,
    candidate_step_degrees: float = 1.0,
    z_search: _Trace2CpZSearchConfig | None = None,
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(loader.config.augment.device)
    model, checkpoint = _load_direction_model(checkpoint_path, loader, device=device)
    configured_depth = max(1, int(_model_config_from_checkpoint(checkpoint, loader).depth))
    default_margin = configured_depth
    margin = float(default_margin if rf_margin_px is None else rf_margin_px)
    weights = combined_weights or _Trace2CpCombinedWeights()
    mode = str(combined_mode).strip().lower()
    if mode not in {"direction", "embedding", "image"}:
        raise ValueError(f"unsupported trace2cp combined mode: {combined_mode!r}")
    embedding_bank: _Trace2CpEmbeddingBank | None = None
    if combined and mode == "embedding":
        flat_indices = _trace2cp_flat_indices_for_sample(loader, sample_index, sample_mode="random")
        embedding_bank = _build_trace2cp_embedding_bank(
            loader,
            model,
            flat_indices,
            device=device,
            rf_margin_px=margin,
        )
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
        combined=combined,
        combined_weights=weights,
        combined_fiber_bank=embedding_bank,
        combined_mode=mode,
        image_scoring=image_scoring,
        candidate_max_degrees=candidate_max_degrees,
        candidate_step_degrees=candidate_step_degrees,
        z_search=z_search,
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
    result_label = selected_mode
    if evaluation.combined_summary is not None:
        result_label = (
            f"{selected_mode} total={evaluation.combined_summary.mean('total'):.4f} "
            f"dir={evaluation.combined_summary.mean('direction'):.4f} "
            f"last={evaluation.combined_summary.mean('last'):.4f} "
            f"enc={evaluation.combined_summary.mean('enclosing'):.4f} "
            f"fib={evaluation.combined_summary.mean('fiber'):.4f} "
            f"img={evaluation.combined_summary.mean('image'):.4f} "
            f"pres={evaluation.combined_summary.mean('presence'):.4f}"
        )
    overlay = _draw_trace2cp_overlay(
        image_u8,
        line_xy=sample.line_xy,
        start_xy=start_xy,
        target_xy=target_xy,
        bidirectional_result=selected_result,
        result_label=result_label,
        reference_result=base_result if selected_result is not base_result else None,
        reference_label="reference",
        similarity_debug=evaluation.similarity_debug,
        z_search_debug=evaluation.z_search_debug,
        presence_debug=evaluation.presence_debug,
        valid_mask=valid_mask,
    )
    _write_jpg(out / "trace2cp_vis.jpg", overlay)
    forward = selected_result.forward.result
    reverse = selected_result.reverse.result
    refinement = selected_result.refinement
    metric = selected_result.metric
    reference_metric = base_result.metric
    trace_points = int(selected_result.forward.trace_xy.shape[0]) + int(
        selected_result.reverse.trace_xy.shape[0]
    )
    z_debug = evaluation.z_search_debug
    if z_debug is None:
        z_summary_lines = ["trace2cp_z_search=false"]
    else:
        f_min, f_max, f_mean_abs = z_debug.z_stats(z_debug.forward_trace_xyz)
        r_min, r_max, r_mean_abs = z_debug.z_stats(z_debug.reverse_trace_xyz)
        fused_min, fused_max, fused_mean_abs = z_debug.z_stats(z_debug.fused_trace_xyz)
        z_summary_lines = [
            "trace2cp_z_search=true",
            f"trace2cp_z_step_voxels={float(z_debug.z_step_voxels):.6f}",
            f"trace2cp_z_max_layer={int(z_debug.max_layer)}",
            f"trace2cp_z_layers={','.join(str(int(v)) for v in z_debug.layers)}",
            f"trace2cp_z_layer_min={int(z_debug.layer_min)}",
            f"trace2cp_z_layer_max={int(z_debug.layer_max)}",
            f"trace2cp_z_forward_min_voxels={f_min:.6f}",
            f"trace2cp_z_forward_max_voxels={f_max:.6f}",
            f"trace2cp_z_forward_mean_abs_voxels={f_mean_abs:.6f}",
            f"trace2cp_z_reverse_min_voxels={r_min:.6f}",
            f"trace2cp_z_reverse_max_voxels={r_max:.6f}",
            f"trace2cp_z_reverse_mean_abs_voxels={r_mean_abs:.6f}",
            f"trace2cp_z_fused_min_voxels={fused_min:.6f}",
            f"trace2cp_z_fused_max_voxels={fused_max:.6f}",
            f"trace2cp_z_fused_mean_abs_voxels={fused_mean_abs:.6f}",
            f"trace2cp_z_forward_missing_columns={int(z_debug.forward_missing_columns)}",
            f"trace2cp_z_reverse_missing_columns={int(z_debug.reverse_missing_columns)}",
            f"trace2cp_z_fused_missing_columns={int(z_debug.fused_missing_columns)}",
        ]
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
        f"trace2cp_error={metric.error:.8f}",
        f"trace2cp_metric_error={metric.error:.8f}",
        f"trace2cp_metric_raw_y_error_px={metric.raw_y_error_px:.6f}",
        f"trace2cp_metric_horizontal_span_px={metric.horizontal_span_px:.6f}",
        f"trace2cp_metric_max_y_error_px={metric.max_y_error_px:.6f}",
        f"trace2cp_metric_forward_target_x={metric.forward_target_x:.6f}",
        f"trace2cp_metric_reverse_target_x={metric.reverse_target_x:.6f}",
        f"trace2cp_metric_forward_y_at_target_x={metric.forward_y_at_target_x:.6f}",
        f"trace2cp_metric_reverse_y_at_target_x={metric.reverse_y_at_target_x:.6f}",
        f"trace2cp_metric_reached_target_columns={metric.reached_target_columns}",
        f"trace2cp_metric_reason={metric.reason}",
        f"trace2cp_similarity_debug={evaluation.similarity_debug is not None}",
        f"trace2cp_similarity_global_bank_size={int(evaluation.similarity_debug.global_bank_size) if evaluation.similarity_debug is not None else 0}",
        f"trace2cp_presence_debug={evaluation.presence_debug is not None}",
        f"trace2cp_refine_score={selected_result.score:.8f}",
        f"actual_y_error_px={selected_result.raw_y_error_px:.6f}",
        f"considered_y_error_px={selected_result.considered_y_error_px:.6f}",
        f"center_penalty={refinement.center_penalty:.6f}",
        f"metric_semantics=target_column_y_error_per_horizontal_cp_span",
        f"refine_score_semantics=center_penalized_minimum_vertical_trace_to_trace_separation",
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
        f"reference_trace2cp_metric_error={reference_metric.error:.8f}",
        f"reference_trace2cp_metric_raw_y_error_px={reference_metric.raw_y_error_px:.6f}",
        f"reference_trace2cp_refine_score={base_result.score:.8f}",
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
        *z_summary_lines,
        f"trace2cp_tta_debug_dir={str(tta_debug_dir) if tta_debug_dir is not None else ''}",
        f"line_trace_tta_count={tta_count}",
        f"med_tta_fields={med_fields_count}",
        *_trace2cp_combined_summary_lines(
            evaluation.combined_summary,
            weights=weights,
            mode=mode,
            image_scoring=image_scoring,
            bank_rows=embedding_bank.rows if embedding_bank is not None else (),
        ),
        "tta_fields:",
        *tta_rows,
    ]
    (out / "trace2cp_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"trace2cp_error={metric.error:.8f}")
    print(
        "trace2cp details "
        f"sample_index={sample_index} fiber_path={sample.fiber_path} "
        f"start_cp={sample.start_control_point_index} target_cp={sample.target_control_point_index} "
        f"mode={selected_mode} "
        f"metric_raw_y_error_px={metric.raw_y_error_px:.6f} "
        f"metric_horizontal_span_px={metric.horizontal_span_px:.6f} "
        f"reached_target_columns={metric.reached_target_columns} "
        f"metric_reason={metric.reason} "
        f"actual_y_error_px={selected_result.raw_y_error_px:.6f} "
        f"considered_y_error_px={selected_result.considered_y_error_px:.6f} "
        f"center_penalty={refinement.center_penalty:.6f} "
        f"reference_error={reference_metric.error:.8f} "
        f"refine_score={selected_result.score:.8f} "
        f"reference_refine_score={base_result.score:.8f} "
        f"endpoint_score={selected_result.endpoint_score:.8f} "
        f"closest_x={refinement.closest_x:.6f} "
        f"forward_endpoint_score={forward.score:.8f} reverse_endpoint_score={reverse.score:.8f} "
        f"forward_reached={forward.reached_target_column} reverse_reached={reverse.reached_target_column} "
        f"forward_reason={forward.reason} reverse_reason={reverse.reason}"
    )
    if evaluation.combined_summary is not None:
        combined_summary = evaluation.combined_summary
        print(
            "trace2cp combined "
            f"reference_error={reference_metric.error:.8f} "
            f"candidate_count={int(combined_summary.candidate_angles_degrees.size)} "
            f"steps={int(combined_summary.steps)} "
            f"direction_loss={combined_summary.mean('direction'):.8f} "
            f"last_loss={combined_summary.mean('last'):.8f} "
            f"enclosing_loss={combined_summary.mean('enclosing'):.8f} "
            f"fiber_loss={combined_summary.mean('fiber'):.8f} "
            f"image_loss={combined_summary.mean('image'):.8f} "
            f"presence_loss={combined_summary.mean('presence'):.8f} "
            f"total_loss={combined_summary.mean('total'):.8f}"
        )
    if z_debug is not None:
        f_min, f_max, f_mean_abs = z_debug.z_stats(z_debug.forward_trace_xyz)
        r_min, r_max, r_mean_abs = z_debug.z_stats(z_debug.reverse_trace_xyz)
        fused_min, fused_max, fused_mean_abs = z_debug.z_stats(z_debug.fused_trace_xyz)
        print(
            "trace2cp z_search "
            f"enabled=true z_step_voxels={float(z_debug.z_step_voxels):.3f} "
            f"layers={int(z_debug.layer_min)}..{int(z_debug.layer_max)} "
            f"forward_z={f_min:.3f}..{f_max:.3f} mean_abs={f_mean_abs:.3f} "
            f"reverse_z={r_min:.3f}..{r_max:.3f} mean_abs={r_mean_abs:.3f} "
            f"fused_z={fused_min:.3f}..{fused_max:.3f} mean_abs={fused_mean_abs:.3f} "
            f"missing_columns={int(z_debug.forward_missing_columns)}/"
            f"{int(z_debug.reverse_missing_columns)}/{int(z_debug.fused_missing_columns)}"
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


def _trace2cp_flat_indices_for_sample(
    loader: FiberStrip2DLoader,
    sample_index: int,
    *,
    sample_mode: str,
) -> tuple[int, ...]:
    record, record_index, _ = loader.descriptor_for_sample_index(int(sample_index), sample_mode=sample_mode)
    if record.fiber.path is not None:
        return loader.flat_sample_indices_for_fiber_json(str(record.fiber.path))
    flat_start = int(loader._record_flat_offsets[int(record_index)])
    return tuple(flat_start + index for index in range(int(record.fiber.control_points_xyz.shape[0])))


def _trace2cp_control_point_line_indices(loader: FiberStrip2DLoader, flat_indices: tuple[int, ...]) -> np.ndarray:
    if not flat_indices:
        raise ValueError("fiber has no control points")
    record, _, _ = loader.descriptor_for_sample_index(int(flat_indices[0]), sample_mode="flat")
    return np.asarray(
        [control_point_line_index(record.fiber, cp_index) for cp_index in range(len(flat_indices))],
        dtype=np.int64,
    )


def _trace2cp_combined_summary_lines(
    summary: _Trace2CpCombinedSummary | None,
    *,
    weights: _Trace2CpCombinedWeights,
    mode: str = "embedding",
    image_scoring: _Trace2CpImageScoringConfig | None = None,
    bank_rows: tuple[str, ...] = (),
) -> list[str]:
    if summary is None:
        return ["trace2cp_combined=false"]
    angles = np.asarray(summary.candidate_angles_degrees, dtype=np.float32)
    angle_min = float(np.min(angles)) if angles.size else 0.0
    angle_max = float(np.max(angles)) if angles.size else 0.0
    angle_step = float(abs(angles[1] - angles[0])) if angles.size > 1 else 0.0
    cfg = image_scoring or _Trace2CpImageScoringConfig()
    rows = [
        "trace2cp_combined=true",
        f"trace2cp_combined_mode={str(mode).strip().lower()}",
        f"trace2cp_combined_candidate_count={int(angles.size)}",
        f"trace2cp_combined_candidate_min_deg={angle_min:.6f}",
        f"trace2cp_combined_candidate_max_deg={angle_max:.6f}",
        f"trace2cp_combined_candidate_step_deg={angle_step:.6f}",
        f"trace2cp_combined_weight_direction={float(weights.direction):.6f}",
        f"trace2cp_combined_weight_last={float(weights.last):.6f}",
        f"trace2cp_combined_weight_enclosing={float(weights.enclosing):.6f}",
        f"trace2cp_combined_weight_fiber={float(weights.fiber):.6f}",
        f"trace2cp_combined_weight_image={float(weights.image):.6f}",
        f"trace2cp_combined_weight_presence={float(weights.presence):.6f}",
        f"trace2cp_combined_steps={int(summary.steps)}",
        f"trace2cp_combined_invalid_candidates={int(summary.invalid_candidates)}",
        f"trace2cp_combined_fiber_bank_size={int(summary.fiber_bank_size)}",
        f"trace2cp_combined_fiber_bank_skipped={int(summary.fiber_bank_skipped)}",
        f"trace2cp_combined_direction_loss_mean={summary.mean('direction'):.8f}",
        f"trace2cp_combined_last_loss_mean={summary.mean('last'):.8f}",
        f"trace2cp_combined_enclosing_loss_mean={summary.mean('enclosing'):.8f}",
        f"trace2cp_combined_fiber_loss_mean={summary.mean('fiber'):.8f}",
        f"trace2cp_combined_image_loss_mean={summary.mean('image'):.8f}",
        f"trace2cp_combined_presence_loss_mean={summary.mean('presence'):.8f}",
        f"trace2cp_combined_total_loss_mean={summary.mean('total'):.8f}",
        f"trace2cp_combined_forward_reason={summary.forward.reason}",
        f"trace2cp_combined_reverse_reason={summary.reverse.reason}",
    ]
    if str(mode).strip().lower() == "image":
        rows.extend(
            [
                f"trace2cp_combined_image_patch_along={int(cfg.patch_along)}",
                f"trace2cp_combined_image_patch_across={int(cfg.patch_across)}",
                f"trace2cp_combined_image_blur_radius={int(cfg.blur_radius)}",
            ]
        )
    if str(mode).strip().lower() == "embedding":
        rows.extend(["trace2cp_combined_embedding_bank:", *bank_rows])
    return rows


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
    combined: bool = False,
    combined_weights: _Trace2CpCombinedWeights | None = None,
    combined_mode: str = "embedding",
    image_scoring: _Trace2CpImageScoringConfig | None = None,
    candidate_max_degrees: float = 25.0,
    candidate_step_degrees: float = 1.0,
    z_search: _Trace2CpZSearchConfig | None = None,
) -> None:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(loader.config.augment.device)
    model, checkpoint = _load_direction_model(checkpoint_path, loader, device=device)
    configured_depth = max(1, int(_model_config_from_checkpoint(checkpoint, loader).depth))
    default_margin = configured_depth
    margin = float(default_margin if rf_margin_px is None else rf_margin_px)
    flat_indices = loader.flat_sample_indices_for_fiber_json(fiber_json)
    weights = combined_weights or _Trace2CpCombinedWeights()
    mode = str(combined_mode).strip().lower()
    if mode not in {"direction", "embedding", "image"}:
        raise ValueError(f"unsupported trace2cp combined mode: {combined_mode!r}")
    embedding_bank: _Trace2CpEmbeddingBank | None = None
    if combined and mode == "embedding":
        embedding_bank = _build_trace2cp_embedding_bank(
            loader,
            model,
            flat_indices,
            device=device,
            rf_margin_px=margin,
        )
    pair_indices = _trace2cp_fiber_pair_cp_indices(len(flat_indices), int(target_offset))
    if not pair_indices:
        raise ValueError(
            "fiber has no in-range Trace2CP pairs for target offset "
            f"{int(target_offset)}: fiber_json='{fiber_json}' cp_count={len(flat_indices)}"
        )

    evaluations: list[_Trace2CpPairEvaluation] = []
    skipped_pairs: list[_Trace2CpSkippedPair] = []
    cp_line_indices = _trace2cp_control_point_line_indices(loader, flat_indices)
    row_axis_refs_by_line_index: dict[int, np.ndarray] = {}
    debug_rows: list[str] = []
    for start_cp_index, target_cp_index in pair_indices:
        alignment_line_index: int | None = None
        alignment_axis: np.ndarray | None = None
        for cp_index in (int(start_cp_index), int(target_cp_index)):
            line_index = int(cp_line_indices[cp_index])
            existing = row_axis_refs_by_line_index.get(line_index)
            if existing is not None:
                alignment_line_index = line_index
                alignment_axis = existing
                break
        try:
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
                combined=combined,
                combined_weights=weights,
                combined_fiber_bank=embedding_bank,
                combined_mode=mode,
                image_scoring=image_scoring,
                candidate_max_degrees=candidate_max_degrees,
                candidate_step_degrees=candidate_step_degrees,
                z_search=z_search,
                sample_mode="flat",
                row_axis_alignment_line_index=alignment_line_index,
                row_axis_alignment_xyz=alignment_axis,
                build_similarity_debug=False,
            )
        except ValueError as exc:
            reason = _trace2cp_skip_reason(exc)
            skipped_pairs.append(
                _Trace2CpSkippedPair(
                    start_cp_index=int(start_cp_index),
                    target_cp_index=int(target_cp_index),
                    reason=reason,
                )
            )
            print(
                "trace2cp fiber_pair_skip "
                f"fiber_json={fiber_json} start_cp={int(start_cp_index)} "
                f"target_cp={int(target_cp_index)} reason={reason}",
                flush=True,
            )
            continue
        evaluations.append(evaluation)
        start_line_index = int(cp_line_indices[int(start_cp_index)])
        target_line_index = int(cp_line_indices[int(target_cp_index)])
        start_axis = _trace2cp_unit_or_zero(np.asarray(evaluation.sample.start_row_axis_xyz, dtype=np.float32))
        target_axis = _trace2cp_unit_or_zero(np.asarray(evaluation.sample.target_row_axis_xyz, dtype=np.float32))
        if bool(np.any(start_axis)):
            row_axis_refs_by_line_index[start_line_index] = start_axis
        if bool(np.any(target_axis)):
            row_axis_refs_by_line_index[target_line_index] = target_axis
        debug_row = _trace2cp_pair_debug_row(
            evaluation,
            start_line_index=start_line_index,
            target_line_index=target_line_index,
            alignment_line_index=alignment_line_index,
            alignment_reference_xyz=alignment_axis,
        )
        debug_rows.append(debug_row)
        print(debug_row, flush=True)
        print(
            "trace2cp fiber_pair "
            f"fiber_path={evaluation.sample.fiber_path} "
            f"start_cp={evaluation.sample.start_control_point_index} "
            f"target_cp={evaluation.sample.target_control_point_index} "
            f"mode={evaluation.selected_mode} trace2cp_error={evaluation.selected_result.metric.error:.8f} "
            f"metric_raw_y_error_px={evaluation.selected_result.metric.raw_y_error_px:.6f} "
            f"metric_horizontal_span_px={evaluation.selected_result.metric.horizontal_span_px:.6f} "
            f"refine_score={evaluation.selected_result.score:.8f} "
            f"actual_y_error_px={evaluation.selected_result.raw_y_error_px:.6f} "
            f"considered_y_error_px={evaluation.selected_result.considered_y_error_px:.6f}",
            flush=True,
        )
    if not evaluations:
        first_skip = skipped_pairs[0].reason if skipped_pairs else ""
        raise ValueError(
            "no valid Trace2CP pairs for whole-fiber visualization: "
            f"fiber_json='{fiber_json}' requested_pairs={len(pair_indices)} "
            f"skipped_pairs={len(skipped_pairs)} first_skip='{first_skip}'"
        )

    cp_x = _trace2cp_control_point_x_positions(loader, flat_indices)
    metric_errors = np.asarray([evaluation.selected_result.metric.error for evaluation in evaluations], dtype=np.float64)
    refine_scores = np.asarray([evaluation.selected_result.score for evaluation in evaluations], dtype=np.float64)
    actual_errors = np.asarray(
        [evaluation.selected_result.metric.raw_y_error_px for evaluation in evaluations], dtype=np.float64
    )
    fiber_path_display = evaluations[0].sample.fiber_path or str(fiber_json)
    mode = evaluations[0].selected_mode
    combined_label = ""
    first_combined = next((evaluation.combined_summary for evaluation in evaluations if evaluation.combined_summary is not None), None)
    if first_combined is not None:
        combined_label = (
            f" combined_total={first_combined.mean('total'):.4f}"
            f" combined_image={first_combined.mean('image'):.4f}"
            f" combined_presence={first_combined.mean('presence'):.4f}"
        )
    z_debugs = [evaluation.z_search_debug for evaluation in evaluations if evaluation.z_search_debug is not None]
    z_label = ""
    if z_debugs:
        layer_min = min(debug.layer_min for debug in z_debugs)
        layer_max = max(debug.layer_max for debug in z_debugs)
        z_label = f" z_layers={layer_min}..{layer_max}"
    label = (
        f"trace2cp fiber pairs={len(evaluations)} mode={mode} "
        f"skipped={len(skipped_pairs)} "
        f"trace2cp_error_mean={float(np.mean(metric_errors)):.4f} "
        f"trace2cp_error_max={float(np.max(metric_errors)):.4f} "
        f"mean_metric_raw_px={float(np.mean(actual_errors)):.2f}"
        f"{combined_label}"
        f"{z_label}"
    )
    overlay = _draw_trace2cp_fiber_overlay(
        evaluations,
        control_point_x=cp_x,
        label=label,
    )
    _write_jpg(out / "trace2cp_fiber_vis.jpg", overlay)
    (out / "trace2cp_fiber_debug.txt").write_text("\n".join(debug_rows) + "\n", encoding="utf-8")

    rows = [
        "start_cp target_cp trace2cp_error metric_raw_y_error_px metric_horizontal_span_px "
        "metric_max_y_error_px metric_reason refine_score actual_y_error_px considered_y_error_px "
        "center_penalty reference_metric_error reference_refine_score endpoint_score closest_x "
        "combined_total_loss_mean forward_reason reverse_reason"
    ]
    for evaluation in evaluations:
        result = evaluation.selected_result
        refinement = result.refinement
        metric = result.metric
        combined_total = (
            evaluation.combined_summary.mean("total")
            if evaluation.combined_summary is not None
            else float("nan")
        )
        rows.append(
            f"{evaluation.sample.start_control_point_index} "
            f"{evaluation.sample.target_control_point_index} "
            f"{metric.error:.8f} "
            f"{metric.raw_y_error_px:.6f} "
            f"{metric.horizontal_span_px:.6f} "
            f"{metric.max_y_error_px:.6f} "
            f"{metric.reason} "
            f"{result.score:.8f} "
            f"{result.raw_y_error_px:.6f} "
            f"{result.considered_y_error_px:.6f} "
            f"{refinement.center_penalty:.6f} "
            f"{evaluation.base_result.metric.error:.8f} "
            f"{evaluation.base_result.score:.8f} "
            f"{result.endpoint_score:.8f} "
            f"{refinement.closest_x:.6f} "
            f"{combined_total:.8f} "
            f"{result.forward.result.reason} "
            f"{result.reverse.result.reason}"
        )
    combined_summaries = [evaluation.combined_summary for evaluation in evaluations if evaluation.combined_summary is not None]
    if combined_summaries:
        combined_steps = sum(int(summary.steps) for summary in combined_summaries)
        combined_invalid = sum(int(summary.invalid_candidates) for summary in combined_summaries)
        combined_component_lines = [
            "trace2cp_combined=true",
            f"trace2cp_combined_pair_count={len(combined_summaries)}",
            f"trace2cp_combined_steps={combined_steps}",
            f"trace2cp_combined_invalid_candidates={combined_invalid}",
            f"trace2cp_combined_fiber_bank_size={int(combined_summaries[0].fiber_bank_size)}",
            f"trace2cp_combined_fiber_bank_skipped={int(combined_summaries[0].fiber_bank_skipped)}",
        ]
        for component in ("direction", "last", "enclosing", "fiber", "image", "presence", "total"):
            weighted_sum = sum(float(summary.mean(component)) * max(1, int(summary.steps)) for summary in combined_summaries)
            denom = max(1, sum(max(1, int(summary.steps)) for summary in combined_summaries))
            combined_component_lines.append(f"trace2cp_combined_{component}_loss_mean={weighted_sum / denom:.8f}")
        combined_component_lines.extend(
            _trace2cp_combined_summary_lines(
                combined_summaries[0],
                weights=weights,
                mode=mode,
                image_scoring=image_scoring,
                bank_rows=embedding_bank.rows if embedding_bank is not None else (),
            )
        )
    else:
        combined_component_lines = ["trace2cp_combined=false"]
    if z_debugs:
        z_forward_mean_abs = float(np.mean([debug.z_stats(debug.forward_trace_xyz)[2] for debug in z_debugs]))
        z_reverse_mean_abs = float(np.mean([debug.z_stats(debug.reverse_trace_xyz)[2] for debug in z_debugs]))
        z_fused_mean_abs = float(np.mean([debug.z_stats(debug.fused_trace_xyz)[2] for debug in z_debugs]))
        z_component_lines = [
            "trace2cp_z_search=true",
            f"trace2cp_z_pair_count={len(z_debugs)}",
            f"trace2cp_z_step_voxels={float(z_debugs[0].z_step_voxels):.6f}",
            f"trace2cp_z_max_layer={int(z_debugs[0].max_layer)}",
            f"trace2cp_z_layer_min={min(debug.layer_min for debug in z_debugs)}",
            f"trace2cp_z_layer_max={max(debug.layer_max for debug in z_debugs)}",
            f"trace2cp_z_forward_mean_abs_voxels={z_forward_mean_abs:.6f}",
            f"trace2cp_z_reverse_mean_abs_voxels={z_reverse_mean_abs:.6f}",
            f"trace2cp_z_fused_mean_abs_voxels={z_fused_mean_abs:.6f}",
            f"trace2cp_z_forward_missing_columns={sum(int(debug.forward_missing_columns) for debug in z_debugs)}",
            f"trace2cp_z_reverse_missing_columns={sum(int(debug.reverse_missing_columns) for debug in z_debugs)}",
            f"trace2cp_z_fused_missing_columns={sum(int(debug.fused_missing_columns) for debug in z_debugs)}",
        ]
    else:
        z_component_lines = ["trace2cp_z_search=false"]
    summary = [
        f"fiber_json={fiber_path_display}",
        f"checkpoint={Path(checkpoint_path).expanduser().resolve()}",
        f"checkpoint_step={checkpoint.get('step', 'unknown')}",
        f"target_offset={int(target_offset)}",
        f"cp_count={len(flat_indices)}",
        f"requested_pair_count={len(pair_indices)}",
        f"valid_pair_count={len(evaluations)}",
        f"skipped_pair_count={len(skipped_pairs)}",
        f"step_px={float(step_px):.3f}",
        f"rf_margin_px={margin:.3f}",
        f"trace_mode={mode}",
        f"med_tta={bool(med_tta)}",
        f"line_trace_tta_count={max(0, int(line_trace_tta_count)) if med_tta else 0}",
        *combined_component_lines,
        *z_component_lines,
        f"trace2cp_error_mean={float(np.mean(metric_errors)):.8f}",
        f"trace2cp_error_max={float(np.max(metric_errors)):.8f}",
        f"trace2cp_error_min={float(np.min(metric_errors)):.8f}",
        f"trace2cp_metric_error_mean={float(np.mean(metric_errors)):.8f}",
        f"trace2cp_metric_error_max={float(np.max(metric_errors)):.8f}",
        f"trace2cp_metric_error_min={float(np.min(metric_errors)):.8f}",
        f"trace2cp_metric_raw_y_error_mean_px={float(np.mean(actual_errors)):.6f}",
        f"refine_score_mean={float(np.mean(refine_scores)):.8f}",
        f"refine_score_max={float(np.max(refine_scores)):.8f}",
        f"refine_score_min={float(np.min(refine_scores)):.8f}",
        f"image_shape_hw=({int(overlay.shape[0])}, {int(overlay.shape[1])})",
        f"debug_path={out / 'trace2cp_fiber_debug.txt'}",
        "",
        *rows,
        "",
        "skipped_pairs:",
        *(
            f"{pair.start_cp_index} {pair.target_cp_index} {pair.reason}"
            for pair in skipped_pairs
        ),
    ]
    (out / "trace2cp_fiber_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"trace2cp_error_mean={float(np.mean(metric_errors)):.8f}")
    print(
        "trace2cp fiber "
        f"fiber_json={fiber_path_display} "
        f"pairs={len(evaluations)} mode={mode} "
        f"trace2cp_error_mean={float(np.mean(metric_errors)):.8f} "
        f"trace2cp_error_max={float(np.max(metric_errors)):.8f} "
        f"metric_raw_y_error_mean_px={float(np.mean(actual_errors)):.6f} "
        f"refine_score_mean={float(np.mean(refine_scores)):.8f} "
        f"skipped_pairs={len(skipped_pairs)}"
    )
    if combined_summaries:
        denom = max(1, sum(max(1, int(summary.steps)) for summary in combined_summaries))
        total_mean = sum(float(summary.mean("total")) * max(1, int(summary.steps)) for summary in combined_summaries) / denom
        image_mean = sum(float(summary.mean("image")) * max(1, int(summary.steps)) for summary in combined_summaries) / denom
        presence_mean = sum(float(summary.mean("presence")) * max(1, int(summary.steps)) for summary in combined_summaries) / denom
        print(
            "trace2cp combined fiber "
            f"trace2cp_error_mean={float(np.mean(metric_errors)):.8f} "
            f"pairs={len(combined_summaries)} "
            f"candidate_count={int(combined_summaries[0].candidate_angles_degrees.size)} "
            f"fiber_bank_size={int(combined_summaries[0].fiber_bank_size)} "
            f"image_loss_mean={image_mean:.8f} "
            f"presence_loss_mean={presence_mean:.8f} "
            f"total_loss_mean={total_mean:.8f}"
        )
    if z_debugs:
        print(
            "trace2cp z_search fiber "
            f"pairs={len(z_debugs)} "
            f"z_step_voxels={float(z_debugs[0].z_step_voxels):.3f} "
            f"layers={min(debug.layer_min for debug in z_debugs)}..{max(debug.layer_max for debug in z_debugs)} "
            f"missing_columns={sum(int(debug.forward_missing_columns) for debug in z_debugs)}/"
            f"{sum(int(debug.reverse_missing_columns) for debug in z_debugs)}/"
            f"{sum(int(debug.fused_missing_columns) for debug in z_debugs)}"
        )
    print(f"exported trace2cp_fiber_vis.jpg, trace2cp_fiber_summary.txt, and trace2cp_fiber_debug.txt to {out}")


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
    parser.add_argument("--trace2cp-combined", action="store_true")
    parser.add_argument("--trace2cp-z-search", action="store_true")
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
    parser.add_argument("--trace2cp-candidate-max-deg", type=float, default=25.0)
    parser.add_argument("--trace2cp-candidate-step-deg", type=float, default=1.0)
    parser.add_argument("--trace2cp-combined-mode", choices=("direction", "embedding", "image"), default="direction")
    parser.add_argument("--trace2cp-use-embedding", action="store_true")
    parser.add_argument("--trace2cp-use-image", action="store_true")
    parser.add_argument("--trace2cp-use-presence", action="store_true")
    parser.add_argument("--trace2cp-image-patch-along", type=int, default=16)
    parser.add_argument("--trace2cp-image-patch-across", type=int, default=8)
    parser.add_argument("--trace2cp-image-blur-radius", type=int, default=5)
    parser.add_argument("--trace2cp-z-step-voxels", type=float, default=1.0)
    parser.add_argument("--trace2cp-z-max-layer", type=int, default=4)
    parser.add_argument("--trace2cp-combined-direction-weight", type=float, default=1.0)
    parser.add_argument("--trace2cp-combined-last-weight", type=float, default=1.0)
    parser.add_argument("--trace2cp-combined-enclosing-weight", type=float, default=1.0)
    parser.add_argument("--trace2cp-combined-fiber-weight", type=float, default=1.0)
    parser.add_argument("--trace2cp-combined-image-weight", type=float, default=1.0)
    parser.add_argument("--trace2cp-combined-presence-weight", type=float, default=1.0)
    parser.add_argument("--fiber-json", default=None)
    args = parser.parse_args()

    with _Timer() as config_timer:
        config = load_config(args.config)
        if args.trace2cp_vis and args.fiber_json is not None:
            config = _config_for_trace2cp_fiber_json(config, args.fiber_json)
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
        requested_similarity_modes = int(bool(args.trace2cp_use_embedding)) + int(bool(args.trace2cp_use_image))
        if requested_similarity_modes > 1:
            raise SystemExit("--trace2cp-use-embedding and --trace2cp-use-image are mutually exclusive")
        trace2cp_mode = str(args.trace2cp_combined_mode).strip().lower()
        if args.trace2cp_use_embedding:
            if trace2cp_mode not in {"direction", "embedding"}:
                raise SystemExit("--trace2cp-use-embedding conflicts with --trace2cp-combined-mode image")
            trace2cp_mode = "embedding"
        if args.trace2cp_use_image:
            if trace2cp_mode not in {"direction", "image"}:
                raise SystemExit("--trace2cp-use-image conflicts with --trace2cp-combined-mode embedding")
            trace2cp_mode = "image"
        combined_enabled = bool(
            args.trace2cp_combined
            or args.trace2cp_use_embedding
            or args.trace2cp_use_image
            or args.trace2cp_use_presence
            or trace2cp_mode != "direction"
        )
        combined_weights = _Trace2CpCombinedWeights(
            direction=float(args.trace2cp_combined_direction_weight),
            last=float(args.trace2cp_combined_last_weight) if trace2cp_mode == "embedding" else 0.0,
            enclosing=float(args.trace2cp_combined_enclosing_weight) if trace2cp_mode == "embedding" else 0.0,
            fiber=float(args.trace2cp_combined_fiber_weight) if trace2cp_mode == "embedding" else 0.0,
            image=float(args.trace2cp_combined_image_weight) if trace2cp_mode == "image" else 0.0,
            presence=float(args.trace2cp_combined_presence_weight) if args.trace2cp_use_presence else 0.0,
        )
        image_scoring = _Trace2CpImageScoringConfig(
            patch_along=int(args.trace2cp_image_patch_along),
            patch_across=int(args.trace2cp_image_patch_across),
            blur_radius=int(args.trace2cp_image_blur_radius),
        )
        if args.trace2cp_z_search and not combined_enabled:
            raise SystemExit("--trace2cp-z-search requires --trace2cp-combined or an enabled combined scoring term")
        if args.trace2cp_z_search and args.med_tta:
            raise SystemExit("--trace2cp-z-search does not currently support --med-tta")
        z_search = _Trace2CpZSearchConfig(
            enabled=bool(args.trace2cp_z_search),
            step_voxels=float(args.trace2cp_z_step_voxels),
            max_layer=int(args.trace2cp_z_max_layer),
        )
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
                combined=combined_enabled,
                combined_weights=combined_weights,
                combined_mode=trace2cp_mode,
                image_scoring=image_scoring,
                candidate_max_degrees=args.trace2cp_candidate_max_deg,
                candidate_step_degrees=args.trace2cp_candidate_step_deg,
                z_search=z_search,
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
            combined=combined_enabled,
            combined_weights=combined_weights,
            combined_mode=trace2cp_mode,
            image_scoring=image_scoring,
            candidate_max_degrees=args.trace2cp_candidate_max_deg,
            candidate_step_degrees=args.trace2cp_candidate_step_deg,
            z_search=z_search,
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
