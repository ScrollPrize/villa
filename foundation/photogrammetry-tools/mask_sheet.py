#!/usr/bin/env python
"""
Review scroll masks: per-directory checks and an overlay contact sheet.

For every acquisition directory under ROOT that contains ScrollMasks/ (as written by
sam2-photogrammetry/scroll-mask.py), checks that each JPGEnhanced/ image has a same-sized,
non-empty mask, and reports the mask area range relative to its median and the largest
frame-to-frame area change. On a turntable sequence the area varies smoothly with the
viewing angle, so a sudden jump points at a frame where the mask lost or gained a piece.

The sheet shows evenly spaced frames from every directory, cropped around the scroll, with
the mask tinted red: it should cover the scroll and nothing else.

Usage:
    python mask_sheet.py ROOT sheet.jpg [--frames 8]
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

TILE = 220
LABEL_W = 150


def acquisition_dirs(root):
    return sorted(d for d, subdirs, _ in os.walk(root) if "ScrollMasks" in subdirs and "JPGEnhanced" in subdirs)


def check_directory(d):
    """Return (stems, relative areas, problems) for one acquisition directory."""
    stems = sorted(f[:-4] for f in os.listdir(os.path.join(d, "JPGEnhanced")) if f.lower().endswith(".jpg"))
    areas, problems = [], []
    for s in stems:
        mask_path = os.path.join(d, "ScrollMasks", f"{s}_mask.png")
        if not os.path.exists(mask_path):
            problems.append(f"{s}: no mask")
            areas.append(0.0)
            continue
        mask = Image.open(mask_path)
        if mask.size != Image.open(os.path.join(d, "JPGEnhanced", f"{s}.jpg")).size:
            problems.append(f"{s}: mask size {mask.size} differs from image")
        area = float(np.count_nonzero(np.asarray(mask.reduce(8))))
        if area == 0:
            problems.append(f"{s}: empty mask")
        areas.append(area)
    areas = np.array(areas)
    return stems, areas / max(np.median(areas), 1.0), problems


def overlay_tile(d, stem):
    image = Image.open(os.path.join(d, "JPGEnhanced", f"{stem}.jpg")).reduce(4).convert("RGB")
    mask = np.asarray(Image.open(os.path.join(d, "ScrollMasks", f"{stem}_mask.png")).reduce(4)) > 127
    a = np.asarray(image).copy()
    a[mask] = (0.5 * a[mask] + [127, 0, 0]).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(ys):
        cy, cx = int(ys.mean()), int(xs.mean())
        h = int(max(np.ptp(ys), np.ptp(xs)) * 0.75) + 20
        a = a[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h]
    tile = Image.fromarray(a)
    tile.thumbnail((TILE - 4, TILE - 4))
    return tile


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("root", help="directory containing acquisition directories with JPGEnhanced/ and ScrollMasks/")
    parser.add_argument("sheet", help="output contact sheet (.jpg)")
    parser.add_argument("--frames", type=int, default=8, help="frames shown per directory")
    args = parser.parse_args()

    dirs = acquisition_dirs(args.root)
    if not dirs:
        raise SystemExit(f"no directories with JPGEnhanced/ and ScrollMasks/ under {args.root}")
    sheet = Image.new("RGB", (args.frames * TILE + LABEL_W, len(dirs) * TILE), "white")
    draw = ImageDraw.Draw(sheet)
    for row, d in enumerate(dirs):
        name = os.path.relpath(d, args.root)
        stems, rel, problems = check_directory(d)
        jump = np.abs(np.diff(np.log(np.maximum(rel, 1e-3))))
        print(f"{name}: {len(stems)} images; mask area {rel.min():.2f}-{rel.max():.2f} x median; "
              f"largest frame-to-frame change x{np.exp(jump.max()) if len(jump) else 1:.2f}"
              f"{' at ' + stems[jump.argmax() + 1] if len(jump) else ''}; problems {len(problems)}")
        for p in problems:
            print("   ", p)
        draw.text((4, row * TILE + 90), name.replace("_camera_", "\n") + f"\nn={len(stems)}", fill="black")
        for col in range(args.frames):
            stem = stems[(col * len(stems)) // args.frames]
            if os.path.exists(os.path.join(d, "ScrollMasks", f"{stem}_mask.png")):
                sheet.paste(overlay_tile(d, stem), (LABEL_W + col * TILE, row * TILE))
            draw.text((LABEL_W + col * TILE + 2, row * TILE + 2), stem, fill="blue")
    sheet.save(args.sheet, quality=85)
    print("wrote", args.sheet)


if __name__ == "__main__":
    main()
