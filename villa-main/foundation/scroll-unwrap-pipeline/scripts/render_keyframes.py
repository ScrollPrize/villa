#!/usr/bin/env python
"""Render keyframes of a computed unwrap trajectory (frames.npz) for visual QA.

Usage: uv run python scripts/render_keyframes.py outputs/anim/<stem> <group> [--n 8]
Writes outputs/anim/<stem>/keyframes/kf_<idx>_<t>.png plus a contact sheet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from scrollkit.render.scene import SceneRenderer, auto_camera  # noqa: E402


def find_texture(stem: str, group: str) -> Path:
    base = ROOT / ("textured_plys/textured_ply_march" if group == "A" else "textured_plys/textured_ply_may")
    cands = list(base.glob(f"{stem}*.png"))
    if not cands:
        raise FileNotFoundError(f"texture for {stem} in {base}")
    return sorted(cands)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("anim_dir")
    ap.add_argument("group", choices=["A", "B"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=1100)
    ap.add_argument("--texture", default=None)
    args = ap.parse_args()

    adir = Path(args.anim_dir)
    stem = adir.name
    data = np.load(adir / "frames.npz")
    frames, faces, uv, ts = data["frames"], data["faces"], data["uv"], data["t"]
    tex = Path(args.texture) if args.texture else find_texture(stem, args.group)
    orientation = json.loads((ROOT / "reports/audit.json").read_text())["tex_orientation"]
    tex_or = orientation[args.group] if isinstance(orientation[args.group], str) else orientation[args.group]

    # camera fixed over the whole trajectory union
    union = frames.reshape(-1, 3)
    cam = auto_camera(union, aspect=16 / 9, margin_frac=0.07)

    out = adir / "keyframes"
    out.mkdir(exist_ok=True)
    idxs = np.unique(np.linspace(0, len(frames) - 1, args.n).astype(int))
    sr = SceneRenderer(frames[0], faces, uv, tex, tex_or,
                       size=(args.size, int(args.size * 9 / 16)), lighting="studio")
    sr.set_camera(cam)
    tiles = []
    for i in idxs:
        sr.update_points(frames[i])
        img = sr.screenshot()
        p = out / f"kf_{i:04d}_t{ts[i]:.3f}.png"
        Image.fromarray(img).save(p)
        tiles.append(img)
        print("wrote", p)
    rows = []
    per_row = 4
    for r in range(0, len(tiles), per_row):
        row = tiles[r:r + per_row]
        while len(row) < per_row:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    sheet_img = Image.fromarray(sheet)
    sheet_img.thumbnail((2200, 2200))
    sheet_img.save(out / "contact_sheet.jpg", quality=88)
    print("wrote", out / "contact_sheet.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
