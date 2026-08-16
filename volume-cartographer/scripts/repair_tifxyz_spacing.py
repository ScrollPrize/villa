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
    # Declared-vs-measured self-consistency. Independent of --target-spacing:
    # this asks only whether the file agrees with ITSELF, not with an
    # externally supplied expectation. See metadata_factor below for why that
    # distinction matters.
    metadata_factor: float
    metadata_mismatch: bool


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


def measure_spacing(seg_dir: Path, input_root: Path, target_spacing: float,
                    threshold: float, metadata_threshold: float = 0.15) -> SegmentStats:
    """Measure a tifxyz's spacing and check it two independent ways.

    ``needs_repair`` (existing) compares the MEASURED spacing against an
    externally supplied ``target_spacing`` -- a guess about what the spacing
    should be, which this tool cannot derive from the file alone.

    ``metadata_mismatch`` (new) compares the MEASURED spacing against
    ``1 / scale``, the value the file's OWN ``meta.json`` implies under the
    documented convention (every consumer in the repo divides by ``scale`` to
    recover voxels; see the linked issue for the audit of call sites). This
    needs no external guess, so it catches a bad ``scale`` field even when
    ``needs_repair`` cannot -- if the geometry happens to measure close to
    whatever ``--target-spacing`` was passed (its default, in particular),
    ``needs_repair`` is False regardless of what ``scale`` claims.

    That gap is not hypothetical: it is exactly how a real published file
    (PHercParis4's ``outer_shell``, declared scale ~19.997 where every sibling
    in the same pack declares ~0.05, a 400x error) passed this script's
    existing check silently. Its geometry measures as an ordinary ~20-voxel
    grid, which matches ``--target-spacing 20.0`` (the default) to within a
    few percent, so ``needs_repair`` was False. ``metadata_factor`` for that
    file is ~400, immediately visible against any reasonable threshold.

    The two checks answer different questions and are kept separate rather
    than merged: ``needs_repair`` says "this geometry looks resampled and a
    fix would resample it back"; ``metadata_mismatch`` says "this file
    disagrees with itself, and only the METADATA should change" -- resampling
    the points would be wrong here, since the geometry is not the part that's
    broken.
    """
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

    # Self-consistency: declared step (1/scale) vs this file's own measured
    # step, on each axis, then combined the same way as the repair factor
    # above so the two numbers read comparably.
    declared_step_x = 1.0 / sx
    declared_step_y = 1.0 / sy
    meta_factor_x = median_right / declared_step_x
    meta_factor_y = median_down / declared_step_y
    metadata_factor = math.sqrt(meta_factor_x * meta_factor_y)
    metadata_mismatch = abs(metadata_factor - 1.0) > metadata_threshold

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
        metadata_factor=metadata_factor,
        metadata_mismatch=metadata_mismatch,
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


def fix_metadata_scale(meta_path: Path, median_right: float, median_down: float) -> tuple[float, float]:
    """Rewrite ``scale`` to match this file's own measured spacing.

    For a ``metadata_mismatch`` case the geometry is not the problem, so this
    changes only ``scale`` and nothing else -- no resampling, no companion
    image touched. That is what the underlying issue's suggested fix asks
    for: "republish outer_shell/meta.json with scale: [0.05, 0.05]... one
    field, 420-byte file." Uses the same write path as ``restore_scale`` so
    both leave meta.json in an identical shape (indent, trailing newline).
    """
    new_sx = 1.0 / median_right
    new_sy = 1.0 / median_down
    restore_scale(meta_path, new_sx, new_sy)
    return new_sx, new_sy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror a tifxyz tree and repair post-affine undersampled surfaces.")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--target-spacing", type=float, default=20.0, help="Expected voxel spacing between adjacent tifxyz samples")
    parser.add_argument("--threshold", type=float, default=0.15, help="Repair when inferred factor differs from 1.0 by more than this amount")
    parser.add_argument("--metadata-threshold", type=float, default=0.15,
                        help="Flag a file whose declared scale disagrees with its OWN measured "
                             "spacing by more than this factor -- independent of --target-spacing. "
                             "Catches a bad scale field even when the geometry itself measures "
                             "close to --target-spacing (see measure_spacing docstring).")
    parser.add_argument("--fix-metadata", action="store_true",
                        help="For metadata_mismatch files, rewrite ONLY the scale field to match "
                             "measured spacing (no resampling, no companion images touched). "
                             "Independent of --dry-run's geometry repair.")
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

    stats = [
        measure_spacing(seg_dir, input_root, args.target_spacing, args.threshold, args.metadata_threshold)
        for seg_dir in tifxyz_dirs(input_root)
    ]
    flagged = [s for s in stats if s.needs_repair]
    mismatched = [s for s in stats if s.metadata_mismatch]

    print(f"Found {len(stats)} tifxyz folders under {input_root}")
    print(f"Flagged {len(flagged)} folders for geometry repair")
    for s in flagged:
        print(
            f"[repair] {s.rel_dir}  factor={s.repair_factor:.4f}  "
            f"spacing=({s.median_right:.2f}, {s.median_down:.2f})  "
            f"expected=({s.expected_x:.2f}, {s.expected_y:.2f})"
        )

    print(f"Flagged {len(mismatched)} folders where declared scale disagrees with "
          f"the file's OWN measured spacing")
    for s in mismatched:
        declared_step = (1.0 / s.scale_x, 1.0 / s.scale_y)
        print(
            f"[metadata] {s.rel_dir}  factor={s.metadata_factor:.4f}  "
            f"declared scale=({s.scale_x:.6g}, {s.scale_y:.6g}) "
            f"-> step=({declared_step[0]:.3f}, {declared_step[1]:.3f})  "
            f"measured=({s.median_right:.2f}, {s.median_down:.2f})"
        )

    if args.dry_run:
        return 0

    shutil.copytree(input_root, output_root)

    # A file needing BOTH kinds of repair is genuinely ambiguous -- which
    # value (scale or geometry) is the correct one to trust? -- and is left
    # for a human, same as the print above already flags it as "also needs
    # geometry repair" without attempting an automatic fix.
    metadata_only_fix = {s.rel_dir for s in mismatched if args.fix_metadata and not s.needs_repair}

    for idx, s in enumerate(stats, start=1):
        out_dir = output_root / s.rel_dir
        if not s.needs_repair:
            if s.rel_dir in metadata_only_fix:
                new_sx, new_sy = fix_metadata_scale(out_dir / "meta.json", s.median_right, s.median_down)
                print(f"[{idx}/{len(stats)}] fix-metadata {s.rel_dir}  "
                      f"scale {s.scale_x:.6g},{s.scale_y:.6g} -> {new_sx:.6g},{new_sy:.6g}  "
                      f"(geometry untouched)")
            else:
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
