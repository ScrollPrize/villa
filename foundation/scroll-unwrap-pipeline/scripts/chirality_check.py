#!/usr/bin/env python
"""Chirality oracle: the flat end-state render must match the texture itself — identity
must beat every mirrored variant by a clear SSIM margin (mirrored ancient Greek is fatal).

Usage: uv run python scripts/chirality_check.py <anim_dir> <group> ...
Appends results to reports/m3_chirality.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

from scrollkit.render.scene import SceneRenderer, auto_camera  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def find_texture(stem: str, group: str) -> Path:
    base = ROOT / ("textured_plys/textured_ply_march" if group == "A" else "textured_plys/textured_ply_may")
    return sorted(base.glob(f"{stem}*.png"))[0]


def check(anim_dir: Path, group: str) -> dict:
    stem = anim_dir.name
    data = np.load(anim_dir / "frames.npz")
    frames, faces, uv = data["frames"], data["faces"], data["uv"]
    flat = frames[-1]
    tex = find_texture(stem, group)
    orientation = json.loads((ROOT / "reports/audit.json").read_text())["tex_orientation"][group]

    # render the flat sheet face-on from the RECTO side: the surface orientation (face
    # winding) defines recto — a PCA normal has arbitrary sign and can show the verso,
    # which is a mirror and falsifies the oracle.
    c = flat.mean(0)
    flat_c = flat - c
    e1 = flat[faces[:, 1]] - flat[faces[:, 0]]
    e2 = flat[faces[:, 2]] - flat[faces[:, 0]]
    normal = np.cross(e1, e2).sum(0)
    normal = normal / np.linalg.norm(normal)
    v_dir = (flat_c * (uv[:, 1:2] - uv[:, 1].mean())).sum(0)
    v_dir -= (v_dir @ normal) * normal
    v_dir /= np.linalg.norm(v_dir)
    extent = float(np.linalg.norm(flat_c, axis=1).max())
    cam = {
        "position": (c + normal * extent * 3).tolist(),
        "focal_point": c.tolist(),
        "up": v_dir.tolist(),
        "parallel": True,
        "parallel_scale": extent * 0.75,
    }
    sr = SceneRenderer(flat, faces, uv, tex, orientation, size=(1024, 1024), lighting="flat")
    sr.set_camera(cam)
    shot = sr.screenshot()
    gray = np.asarray(Image.fromarray(shot).convert("L"), dtype=np.float64)

    # reference: the texture itself (oriented per audit convention so content matches UV space),
    # composited on the render background, downscaled to the same size
    img = Image.open(tex).convert("RGBA")
    arr = np.asarray(img, dtype=np.float64)
    rgb, a = arr[..., :3], arr[..., 3:] / 255.0
    bg = np.array([16.0, 17.0, 20.0])
    comp = rgb * a + bg * (1 - a)
    ref_full = np.asarray(Image.fromarray(comp.astype(np.uint8)).convert("L").resize((768, 768)), dtype=np.float64)

    scores = {}
    for name, op in {
        "identity": lambda x: x,
        "flip_h": lambda x: x[:, ::-1],
        "flip_v": lambda x: x[::-1, :],
        "rot180": lambda x: x[::-1, ::-1],
    }.items():
        best = -1.0
        ref = np.ascontiguousarray(op(ref_full))
        # the sheet occupies an unknown sub-rect of the shot; coarse search over scale-fit:
        # downscale shot to ref size and compare directly (both show the whole sheet)
        shot_s = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize((768, 768)), dtype=np.float64)
        best = structural_similarity(shot_s, ref, data_range=255.0)
        scores[name] = float(best)
    margin = scores["identity"] - max(v for k, v in scores.items() if k != "identity")
    return {"stem": stem, "group": group, "scores": scores, "identity_margin": margin,
            "pass": bool(margin > 0.02 and scores["identity"] == max(scores.values()))}


def main() -> int:
    out = ROOT / "reports/m3_chirality.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    args = sys.argv[1:]
    pairs = [(Path(args[i]), args[i + 1]) for i in range(0, len(args), 2)]
    ok = True
    for adir, group in pairs:
        r = check(adir, group)
        results[r["stem"]] = r
        ok &= r["pass"]
        print(f"{r['stem']}: {r['scores']} margin={r['identity_margin']:.4f} -> {'PASS' if r['pass'] else 'FAIL'}")
    out.write_text(json.dumps(results, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
