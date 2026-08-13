#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image


@dataclass
class SegmentStats:
    src_dir: Path
    rel_dir: Path
    width: int
    height: int
    scale_x: float
    scale_y: float
    expected_x: float
    expected_y: float
    median_right: float
    median_down: float
    repair_factor_x: float
    repair_factor_y: float
    repair_factor: float
    needs_repair: bool


def tifxyz_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        seg_dir = meta_path.parent
        if all((seg_dir / name).exists() for name in ("x.tif", "y.tif", "z.tif")):
            dirs.append(seg_dir)
    return sorted(dirs)


def load_scale(meta_path: Path) -> tuple[float, float]:
    with meta_path.open() as f:
        meta = json.load(f)
    scale = meta.get("scale")
    if isinstance(scale, (int, float)):
        sx = sy = float(scale)
    elif isinstance(scale, list) and len(scale) >= 2:
        sx = float(scale[0])
        sy = float(scale[1])
    else:
        raise ValueError(f"Unsupported scale in {meta_path}")
    if sx <= 0 or sy <= 0:
        raise ValueError(f"Non-positive scale in {meta_path}")
    return sx, sy


def measure_spacing(seg_dir: Path, input_root: Path, target_spacing: float, threshold: float) -> SegmentStats:
    sx, sy = load_scale(seg_dir / "meta.json")
    x = tifffile.imread(seg_dir / "x.tif").astype(np.float32)
    y = tifffile.imread(seg_dir / "y.tif").astype(np.float32)
    z = tifffile.imread(seg_dir / "z.tif").astype(np.float32)

    pts = np.stack([x, y, z], axis=-1).astype(np.float64)
    valid = (
        np.isfinite(pts[..., 0])
        & np.isfinite(pts[..., 1])
        & np.isfinite(pts[..., 2])
        & (pts[..., 0] != -1.0)
        & (pts[..., 2] > 0)
    )

    right_valid = valid[:, :-1] & valid[:, 1:]
    right_delta = np.linalg.norm(pts[:, 1:, :] - pts[:, :-1, :], axis=-1)
    right_vals = right_delta[right_valid]

    down_valid = valid[:-1, :] & valid[1:, :]
    down_delta = np.linalg.norm(pts[1:, :, :] - pts[:-1, :, :], axis=-1)
    down_vals = down_delta[down_valid]

    if right_vals.size == 0 or down_vals.size == 0:
        raise ValueError(f"No valid neighbor pairs in {seg_dir}")

    expected_x = float(target_spacing)
    expected_y = float(target_spacing)
    median_right = float(np.median(right_vals))
    median_down = float(np.median(down_vals))
    factor_x = median_right / expected_x
    factor_y = median_down / expected_y
    factor = math.sqrt(factor_x * factor_y)
    needs_repair = abs(factor - 1.0) > threshold

    return SegmentStats(
        src_dir=seg_dir,
        rel_dir=seg_dir.relative_to(input_root),
        width=int(x.shape[1]),
        height=int(x.shape[0]),
        scale_x=sx,
        scale_y=sy,
        expected_x=expected_x,
        expected_y=expected_y,
        median_right=median_right,
        median_down=median_down,
        repair_factor_x=factor_x,
        repair_factor_y=factor_y,
        repair_factor=factor,
        needs_repair=needs_repair,
    )


def resample_points(seg_dir: Path, factor: float) -> tuple[int, int]:
    x = tifffile.imread(seg_dir / "x.tif").astype(np.float32)
    y = tifffile.imread(seg_dir / "y.tif").astype(np.float32)
    z = tifffile.imread(seg_dir / "z.tif").astype(np.float32)
    pts = np.stack([x, y, z], axis=-1)

    old_h, old_w = x.shape
    new_w = max(1, int(round(old_w * factor)))
    new_h = max(1, int(round(old_h * factor)))

    resampled = cv2.resize(pts, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    invalid = (
        ~np.isfinite(pts[..., 0])
        | ~np.isfinite(pts[..., 1])
        | ~np.isfinite(pts[..., 2])
        | (pts[..., 0] == -1.0)
        | (pts[..., 2] <= 0)
    ).astype(np.uint8) * 255
    scaled_invalid = cv2.resize(invalid, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), dtype=np.uint8)
    scaled_invalid = cv2.dilate(scaled_invalid, kernel, iterations=1)
    resampled[scaled_invalid > 0] = (-1.0, -1.0, -1.0)

    tifffile.imwrite(seg_dir / "x.tif", resampled[..., 0].astype(np.float32))
    tifffile.imwrite(seg_dir / "y.tif", resampled[..., 1].astype(np.float32))
    tifffile.imwrite(seg_dir / "z.tif", resampled[..., 2].astype(np.float32))
    return old_w, old_h


def maybe_resample_companion_image(path: Path, old_size: tuple[int, int], new_size: tuple[int, int]) -> bool:
    name = path.name.lower()
    if name in {"x.tif", "y.tif", "z.tif", "meta.json"}:
        return False
    if path.suffix.lower() not in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        return False

    try:
        with Image.open(path) as img:
            if img.size != old_size:
                return False
            arr = np.array(img)
            interp = cv2.INTER_NEAREST if any(token in name for token in ("mask", "label", "ink")) else cv2.INTER_LINEAR
            resized = cv2.resize(arr, new_size, interpolation=interp)
            Image.fromarray(resized).save(path)
            return True
    except Exception:
        return False


def restore_scale(meta_path: Path, sx: float, sy: float) -> None:
    with meta_path.open() as f:
        meta = json.load(f)
    meta["scale"] = [sx, sy]
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror a tifxyz tree and repair post-affine undersampled surfaces.")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--target-spacing", type=float, default=20.0, help="Expected voxel spacing between adjacent tifxyz samples")
    parser.add_argument("--threshold", type=float, default=0.15, help="Repair when inferred factor differs from 1.0 by more than this amount")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if output_root.exists():
        raise SystemExit(f"Output root already exists: {output_root}")

    stats = [measure_spacing(seg_dir, input_root, args.target_spacing, args.threshold) for seg_dir in tifxyz_dirs(input_root)]
    flagged = [s for s in stats if s.needs_repair]

    print(f"Found {len(stats)} tifxyz folders under {input_root}")
    print(f"Flagged {len(flagged)} folders for repair")
    for s in flagged:
        print(
            f"[repair] {s.rel_dir}  factor={s.repair_factor:.4f}  "
            f"spacing=({s.median_right:.2f}, {s.median_down:.2f})  "
            f"expected=({s.expected_x:.2f}, {s.expected_y:.2f})"
        )

    if args.dry_run:
        return 0

    shutil.copytree(input_root, output_root)

    for idx, s in enumerate(stats, start=1):
        out_dir = output_root / s.rel_dir
        if not s.needs_repair:
            print(f"[{idx}/{len(stats)}] copy-only {s.rel_dir}")
            continue

        print(f"[{idx}/{len(stats)}] repairing {s.rel_dir} with factor {s.repair_factor:.4f}")
        old_size = resample_points(out_dir, s.repair_factor)
        new_size = (max(1, int(round(old_size[0] * s.repair_factor))), max(1, int(round(old_size[1] * s.repair_factor))))
        restore_scale(out_dir / "meta.json", s.scale_x, s.scale_y)

        updated = 0
        for child in out_dir.iterdir():
            if child.is_file() and maybe_resample_companion_image(child, old_size, new_size):
                updated += 1
        if updated:
            print(f"  updated {updated} companion image(s)")

    print(f"Done. Repaired mirror written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
