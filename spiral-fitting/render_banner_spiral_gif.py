#!/usr/bin/env python3
"""Render a z-slice GIF morphing an ideal spiral onto fitted TIFXYZ windings."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import s3fs
import zarr
from PIL import Image, ImageDraw, ImageFilter
from scipy.optimize import least_squares


ZARR_URL = (
    "s3://vesuvius-challenge-open-data/PHercParis4/volumes/"
    "20260411134726-2.400um-0.2m-78keV-masked.zarr/"
)
Z_MESH_LEVEL = 14000.0
ZARR_LEVEL = 3
MESH_ZARR_LEVEL = 2
FRAME_COUNT = 42
MAX_SIZE = 1400
MARGIN = 140

WINDING_STEP = 25
LINE_COLOR = (
    np.asarray((255, 0, 0), dtype=np.float32),
    np.asarray((255, 0, 0), dtype=np.float32),
)
LINE_WIDTH = 7.0
LINE_ALPHA = 255
GLOW_ALPHA = 170
START_SPIRAL_SCALE = 1.4
SHRINK_FRACTION = 0.45
CONTINUOUS_START_SPIRAL = True


@dataclass
class WindingCurve:
    winding: int
    target_yx_mesh: np.ndarray
    ideal_yx_mesh: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the fixed banner spiral GIF recipe at z=14000.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("/Users/sean/Desktop/banner_segments"),
    )
    parser.add_argument(
        "--background-cache",
        type=Path,
        default=Path("outputs/banner_spiral_z14000_level3_crop.npy"),
    )
    parser.add_argument(
        "--background-cache-preview",
        type=Path,
        default=Path("outputs/banner_spiral_z14000_level3_crop.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/banner_spiral_z14000.gif"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("outputs/banner_spiral_z14000_final.png"),
    )
    return parser.parse_args()


def read_tif(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32)


def extract_z_curve(path: Path, z0: float) -> np.ndarray:
    z = read_tif(path / "z.tif")
    y = read_tif(path / "y.tif")
    x = read_tif(path / "x.tif")

    points: list[tuple[float, float]] = []
    for col_idx in range(z.shape[1]):
        zc = z[:, col_idx]
        yc = y[:, col_idx]
        xc = x[:, col_idx]
        valid = (zc != -1) & (yc != -1) & (xc != -1)
        if valid.sum() < 2:
            continue
        zv = zc[valid]
        yv = yc[valid]
        xv = xc[valid]
        order = np.argsort(zv)
        zv = zv[order]
        if z0 < zv[0] or z0 > zv[-1]:
            continue
        yv = yv[order]
        xv = xv[order]
        idx = int(np.searchsorted(zv, z0))
        if idx == 0:
            points.append((float(yv[0]), float(xv[0])))
        elif idx >= len(zv):
            points.append((float(yv[-1]), float(xv[-1])))
        else:
            z_a, z_b = float(zv[idx - 1]), float(zv[idx])
            if abs(z_b - z_a) < 1e-6:
                t = 0.0
            else:
                t = (z0 - z_a) / (z_b - z_a)
            points.append(
                (
                    float(yv[idx - 1] + (yv[idx] - yv[idx - 1]) * t),
                    float(xv[idx - 1] + (xv[idx] - xv[idx - 1]) * t),
                )
            )
    return np.asarray(points, dtype=np.float32)


def load_curves(mesh_dir: Path, z0: float) -> list[WindingCurve]:
    curves: list[WindingCurve] = []
    for path in sorted(mesh_dir.glob("w*_spliced_baseline_tracks_m7_ds2_surf")):
        match = re.match(r"w(\d+)_", path.name)
        if not match:
            continue
        winding = int(match.group(1))
        target = extract_z_curve(path, z0)
        if len(target) >= 2:
            curves.append(WindingCurve(winding=winding, target_yx_mesh=target))
    if not curves:
        raise RuntimeError(f"no spliced mesh curves intersect z={z0:g}")
    return curves


def filter_winding_step(curves: list[WindingCurve], winding_step: int) -> list[WindingCurve]:
    if winding_step <= 1:
        return curves
    first = curves[0].winding
    filtered = [curve for curve in curves if (curve.winding - first) % winding_step == 0]
    if len(filtered) < 2:
        raise RuntimeError(f"winding step {winding_step} left fewer than two curves")
    return filtered


def fit_ideal_spiral(curves: list[WindingCurve]) -> tuple[float, float, float, float, int]:
    yx = []
    windings = []
    for curve in curves:
        step = max(1, len(curve.target_yx_mesh) // 120)
        pts = curve.target_yx_mesh[::step]
        yx.append(pts)
        windings.append(np.full(len(pts), curve.winding, dtype=np.float32))
    points = np.concatenate(yx, axis=0)
    winding_idx = np.concatenate(windings, axis=0)

    inner = np.concatenate(
        [curve.target_yx_mesh for curve in curves if curve.winding <= curves[0].winding + 3],
        axis=0,
    )
    starts = [
        (float(np.median(points[:, 0])), float(np.median(points[:, 1]))),
        (float(inner[:, 0].mean()), float(inner[:, 1].mean())),
        (4096.0, 4096.0),
    ]

    def residual(params: np.ndarray, sense: int) -> np.ndarray:
        cy, cx, log_dr, phase = params
        dr = math.exp(log_dr)
        dy = points[:, 0] - cy
        dx = points[:, 1] - cx
        radius = np.hypot(dy, dx)
        alpha = np.arctan2(dy, dx)
        theta = np.mod(sense * (alpha - phase), 2 * np.pi)
        expected = (winding_idx + theta / (2 * np.pi)) * dr
        return (radius - expected) / 25.0

    best = None
    for sense in (1, -1):
        for cy, cx in starts:
            for dr0 in (18.0, 22.0, 26.0, 30.0):
                for phase0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    x0 = np.asarray([cy, cx, math.log(dr0), phase0], dtype=np.float64)
                    result = least_squares(
                        residual,
                        x0,
                        args=(sense,),
                        loss="soft_l1",
                        f_scale=2.0,
                        max_nfev=700,
                    )
                    score = float(np.mean(np.abs(residual(result.x, sense))))
                    if best is None or score < best[0]:
                        best = (score, result.x, sense)
    assert best is not None
    _, params, sense = best
    cy, cx, log_dr, phase = params
    dr = math.exp(float(log_dr))
    phase = float(np.mod(phase, 2 * np.pi))

    for curve in curves:
        dy = curve.target_yx_mesh[:, 0] - cy
        dx = curve.target_yx_mesh[:, 1] - cx
        alpha = np.arctan2(dy, dx)
        theta = np.mod(sense * (alpha - phase), 2 * np.pi)
        ideal_alpha = phase + sense * theta
        radius = (curve.winding + theta / (2 * np.pi)) * dr
        curve.ideal_yx_mesh = np.stack(
            [
                cy + np.sin(ideal_alpha) * radius,
                cx + np.cos(ideal_alpha) * radius,
            ],
            axis=1,
        ).astype(np.float32)
    return float(cy), float(cx), dr, phase, sense


def open_s3_zarr(url: str, level: int) -> zarr.Array:
    fs = s3fs.S3FileSystem(anon=True)
    group = zarr.open_group(s3fs.S3Map(root=url, s3=fs, check=False), mode="r")
    return group[str(level)]


def load_background(
    array: zarr.Array | None,
    z0: float,
    crop_yx: tuple[int, int, int, int],
    mesh_to_zarr_divisor: float,
    cache_path: Path | None,
    preview_path: Path | None,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    if array is None:
        raise RuntimeError("background cache is missing and no zarr array was opened")

    z_level = int(round(z0 / mesh_to_zarr_divisor))
    y0, y1, x0, x1 = crop_yx
    bg = np.asarray(array[z_level, y0:y1, x0:x1], dtype=np.uint8)
    if bg.max() <= bg.min():
        return np.full(bg.shape, 20, dtype=np.uint8)
    lo, hi = np.percentile(bg[bg > 0] if np.any(bg > 0) else bg, [1, 99.7])
    if hi <= lo:
        hi = lo + 1
    bg = np.clip((bg.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
    bg = bg.astype(np.uint8)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, bg)
        metadata = {
            "z_mesh_level": z0,
            "zarr_level": int(round(z0 / mesh_to_zarr_divisor)),
            "crop_yx": list(crop_yx),
            "shape": list(bg.shape),
            "normalization_percentiles": [float(lo), float(hi)],
        }
        cache_path.with_suffix(cache_path.suffix + ".json").write_text(json.dumps(metadata, indent=2))
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(bg).save(preview_path)

    return bg


def ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def order_by_spiral_phase(
    yx: np.ndarray,
    center_yx: np.ndarray,
    spiral_phase: float,
    spiral_sense: int,
) -> np.ndarray:
    rel = yx - center_yx[None, :]
    alpha = np.arctan2(rel[:, 0], rel[:, 1])
    theta = np.mod(float(spiral_sense) * (alpha - spiral_phase), 2.0 * np.pi)
    return yx[np.argsort(theta)]


def draw_frame(
    bg_small: Image.Image,
    curves: list[WindingCurve],
    t: float,
    crop_yx: tuple[int, int, int, int],
    scale: float,
    mesh_to_zarr_divisor: float,
    spiral_center_yx: tuple[float, float],
    spiral_phase: float,
    spiral_sense: int,
) -> Image.Image:
    frame = bg_small.convert("RGB")
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    line_layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    line_draw = ImageDraw.Draw(line_layer)
    y0, _, x0, _ = crop_yx
    shrink_fraction = float(np.clip(SHRINK_FRACTION, 0.0, 0.95))
    if shrink_fraction > 0.0 and t < shrink_fraction:
        shrink_t = ease(t / shrink_fraction)
        deform_t = 0.0
        spiral_scale = START_SPIRAL_SCALE * (1.0 - shrink_t) + shrink_t
    elif shrink_fraction > 0.0:
        deform_t = ease((t - shrink_fraction) / (1.0 - shrink_fraction))
        spiral_scale = 1.0
    else:
        deform_t = ease(t)
        spiral_scale = START_SPIRAL_SCALE * (1.0 - deform_t) + deform_t

    color_a, color_b = LINE_COLOR
    color = tuple(np.clip(color_a * (1 - deform_t) + color_b * deform_t, 0, 255).astype(np.uint8).tolist())
    line_alpha = int(np.clip(LINE_ALPHA, 0, 255))
    glow_alpha = int(np.clip(GLOW_ALPHA, 0, 255))
    width = max(1, int(round(LINE_WIDTH * scale)))
    glow_width = max(width + 5, int(round((LINE_WIDTH + 4.6) * scale)))

    if CONTINUOUS_START_SPIRAL and shrink_fraction > 0.0:
        center = np.asarray(spiral_center_yx, dtype=np.float32)
        ideal_points = np.concatenate([curve.ideal_yx_mesh for curve in curves if curve.ideal_yx_mesh is not None])
        ideal_radius = np.linalg.norm(ideal_points - center[None, :], axis=1)
        radius_min = float(np.percentile(ideal_radius, 1))
        radius_max = float(np.percentile(ideal_radius, 99))
        turns = max(2, len(curves))

        for segment_idx, curve in enumerate(curves):
            assert curve.ideal_yx_mesh is not None
            n = len(curve.target_yx_mesh)
            if n < 2:
                continue
            theta_unwrapped = np.linspace(
                segment_idx * 2.0 * np.pi,
                (segment_idx + 1) * 2.0 * np.pi,
                n,
                dtype=np.float32,
            )
            alpha = spiral_phase + float(spiral_sense) * theta_unwrapped
            radius = np.linspace(
                radius_min + (radius_max - radius_min) * segment_idx / turns,
                radius_min + (radius_max - radius_min) * (segment_idx + 1) / turns,
                n,
                dtype=np.float32,
            )
            start_yx = np.stack(
                [
                    center[0] + np.sin(alpha) * radius,
                    center[1] + np.cos(alpha) * radius,
                ],
                axis=1,
            )
            target_yx = order_by_spiral_phase(
                curve.target_yx_mesh,
                center,
                spiral_phase,
                spiral_sense,
            )
            scaled_start = center[None, :] + (start_yx - center[None, :]) * spiral_scale
            yx = scaled_start * (1 - deform_t) + target_yx * deform_t
            pts = [
                (
                    (float(x) / mesh_to_zarr_divisor - x0) * scale,
                    (float(y) / mesh_to_zarr_divisor - y0) * scale,
                )
                for y, x in yx
            ]
            glow_draw.line(pts, fill=(255, 255, 255, glow_alpha), width=glow_width, joint="curve")
            line_draw.line(pts, fill=(*color, line_alpha), width=width, joint="curve")
    else:
        for curve in curves:
            assert curve.ideal_yx_mesh is not None
            center = np.asarray(spiral_center_yx, dtype=np.float32)[None, :]
            scaled_ideal = center + (curve.ideal_yx_mesh - center) * spiral_scale
            yx = scaled_ideal * (1 - deform_t) + curve.target_yx_mesh * deform_t
            pts = [
                (
                    (float(x) / mesh_to_zarr_divisor - x0) * scale,
                    (float(y) / mesh_to_zarr_divisor - y0) * scale,
                )
                for y, x in yx
            ]
            if len(pts) < 2:
                continue
            glow_draw.line(pts, fill=(255, 255, 255, glow_alpha), width=glow_width, joint="curve")
            line_draw.line(pts, fill=(*color, line_alpha), width=width, joint="curve")

    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(1.0, 1.2 * scale)))
    frame = Image.alpha_composite(frame.convert("RGBA"), glow)
    frame = Image.alpha_composite(frame, line_layer)
    return frame.convert("RGB")


def main() -> None:
    args = parse_args()
    mesh_to_zarr_divisor = 2.0 ** (ZARR_LEVEL - MESH_ZARR_LEVEL)
    if mesh_to_zarr_divisor <= 0:
        raise ValueError("invalid zarr level mapping")

    all_curves = load_curves(args.mesh_dir, Z_MESH_LEVEL)
    curves = filter_winding_step(all_curves, WINDING_STEP)
    cy, cx, dr, phase, sense = fit_ideal_spiral(curves)

    all_target = np.concatenate([curve.target_yx_mesh for curve in all_curves], axis=0)
    all_level_y = all_target[:, 0] / mesh_to_zarr_divisor
    all_level_x = all_target[:, 1] / mesh_to_zarr_divisor
    y0 = max(0, int(np.floor(all_level_y.min())) - MARGIN)
    y1 = int(np.ceil(all_level_y.max())) + MARGIN
    x0 = max(0, int(np.floor(all_level_x.min())) - MARGIN)
    x1 = int(np.ceil(all_level_x.max())) + MARGIN

    cache_exists = args.background_cache is not None and args.background_cache.exists()
    array = None if cache_exists else open_s3_zarr(ZARR_URL, ZARR_LEVEL)
    if array is not None:
        y1 = min(y1, array.shape[1])
        x1 = min(x1, array.shape[2])
    crop_yx = (y0, y1, x0, x1)
    bg = load_background(
        array,
        Z_MESH_LEVEL,
        crop_yx,
        mesh_to_zarr_divisor,
        args.background_cache,
        args.background_cache_preview,
    )

    native_h, native_w = bg.shape
    scale = min(1.0, MAX_SIZE / max(native_h, native_w))
    out_w = max(1, int(round(native_w * scale)))
    out_h = max(1, int(round(native_h * scale)))
    bg_small = Image.fromarray(bg, mode="L").resize((out_w, out_h), Image.Resampling.BILINEAR)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        draw_frame(
            bg_small,
            curves,
            i / (FRAME_COUNT - 1),
            crop_yx,
            scale,
            mesh_to_zarr_divisor,
            (cy, cx),
            phase,
            sense,
        )
        for i in range(FRAME_COUNT)
    ]
    imageio.mimsave(args.output, frames, duration=0.07, loop=0)
    frames[-1].convert("RGB").save(args.preview)

    print(f"curves: {len(curves)} windings {curves[0].winding}-{curves[-1].winding}")
    print(f"winding step: {WINDING_STEP}")
    print(f"fit: center_yx=({cy:.2f}, {cx:.2f}) dr={dr:.3f} phase={phase:.3f} sense={sense}")
    print(
        f"mesh level {MESH_ZARR_LEVEL} -> zarr level {ZARR_LEVEL}: "
        f"divide mesh z/y/x by {mesh_to_zarr_divisor:g}"
    )
    print(f"crop level {ZARR_LEVEL}: y={y0}:{y1} x={x0}:{x1}; output {out_w}x{out_h}")
    print(f"wrote {args.output}")
    print(f"wrote {args.preview}")
    if args.background_cache is not None:
        print(f"cached background {args.background_cache}")


if __name__ == "__main__":
    main()
