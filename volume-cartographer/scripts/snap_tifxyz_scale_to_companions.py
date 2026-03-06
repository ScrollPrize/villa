#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
import tifffile


CANONICAL_HINTS = ("inklabels", "mask")
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def tifxyz_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        seg_dir = meta_path.parent
        if all((seg_dir / name).exists() for name in ("x.tif", "y.tif", "z.tif")):
            dirs.append(seg_dir)
    return sorted(dirs)


def choose_companion_size(seg_dir: Path) -> tuple[int, int] | None:
    candidates: list[tuple[str, tuple[int, int]]] = []
    for child in sorted(seg_dir.iterdir()):
        if child.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if child.name in {"x.tif", "y.tif", "z.tif"}:
            continue
        lname = child.name.lower()
        if not any(hint in lname for hint in CANONICAL_HINTS):
            continue
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(child) as img:
            candidates.append((child.name, img.size))

    if not candidates:
        return None

    counts = Counter(size for _, size in candidates)
    size, _ = counts.most_common(1)[0]
    return size


def predicted_render_size(cols: int, rows: int, scale_x: float, scale_y: float) -> tuple[int, int]:
    return round(cols / scale_x), round(rows / scale_y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror a tifxyz tree and snap tifxyz scale metadata to companion image sizes.")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
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

    segments = tifxyz_dirs(input_root)
    print(f"Found {len(segments)} tifxyz folders under {input_root}")

    changes: list[tuple[Path, tuple[int, int], tuple[int, int], tuple[float, float], tuple[float, float]]] = []
    for seg_dir in segments:
        size = choose_companion_size(seg_dir)
        if size is None:
            print(f"[skip] {seg_dir.relative_to(input_root)}: no inklabels/mask companion found")
            continue
        cols = tifffile.imread(seg_dir / "x.tif").shape[1]
        rows = tifffile.imread(seg_dir / "x.tif").shape[0]
        with (seg_dir / "meta.json").open() as f:
            meta = json.load(f)
        old_scale = tuple(float(v) for v in meta["scale"][:2])
        new_scale = (cols / size[0], rows / size[1])
        old_pred = predicted_render_size(cols, rows, old_scale[0], old_scale[1])
        changes.append((seg_dir.relative_to(input_root), size, old_pred, old_scale, new_scale))

    if args.dry_run:
        for rel_dir, target, old_pred, old_scale, new_scale in changes:
            print(
                f"[plan] {rel_dir} target={target[0]}x{target[1]} "
                f"old_render={old_pred[0]}x{old_pred[1]} "
                f"old_scale=({old_scale[0]:.8f},{old_scale[1]:.8f}) "
                f"new_scale=({new_scale[0]:.8f},{new_scale[1]:.8f})"
            )
        return 0

    shutil.copytree(input_root, output_root)

    for rel_dir, target, old_pred, old_scale, new_scale in changes:
        meta_path = output_root / rel_dir / "meta.json"
        with meta_path.open() as f:
            meta = json.load(f)
        meta["scale"] = [new_scale[0], new_scale[1]]
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=4)
            f.write("\n")
        print(
            f"[write] {rel_dir} target={target[0]}x{target[1]} "
            f"old_render={old_pred[0]}x{old_pred[1]} "
            f"new_scale=({new_scale[0]:.8f},{new_scale[1]:.8f})"
        )

    print(f"Done. Wrote snapped mirror to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
