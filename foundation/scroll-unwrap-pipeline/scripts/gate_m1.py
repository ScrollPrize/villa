#!/usr/bin/env python
"""Milestone M1 gate: bit-exact reload + texture SHA + render parity (when renderer lands).

Numeric part (always on): every converted OBJ reloads bit-equal to its source PLY.
Render-parity part: enabled once scrollkit.render.parity exists; until then reports SKIP
and the gate stays RED unless --numeric-only is passed explicitly.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scrollkit.io import read_obj, read_ply  # noqa: E402

FAIL: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAIL.append(msg)
        print("  FAIL  " + msg)


def bits(a):
    return a.view(np.uint32) if a is not None else None


def main() -> int:
    numeric_only = "--numeric-only" in sys.argv
    man_p = ROOT / "outputs/obj/manifest.json"
    if not man_p.exists():
        print("manifest missing — run scripts/convert.py first")
        return 1
    man = json.loads(man_p.read_text())
    meshes = man["meshes"]
    check(len(meshes) == 34, f"34 meshes converted (got {len(meshes)})")

    for rec in meshes:
        tag = f"[{rec['group']}] {Path(rec['src']).stem}"
        src = read_ply(ROOT / rec["src"])
        out = read_obj(ROOT / rec["obj"])
        check(np.array_equal(bits(out.vertices), bits(src.vertices)), f"{tag}: vertices bit-equal")
        check(np.array_equal(out.faces, src.faces), f"{tag}: faces identical")
        if src.normals is not None:
            check(np.array_equal(bits(out.normals), bits(src.normals)), f"{tag}: normals bit-equal")
        # UV equivalence: per-corner UVs must match source wedge UVs bit-exactly
        if src.wedge_uv is not None:
            if out.vertex_uv is not None:
                out_wedge = out.vertex_uv[out.faces]
            elif out.wedge_uv is not None:
                out_wedge = out.wedge_uv
            else:
                out_wedge = None
            check(out_wedge is not None and np.array_equal(bits(out_wedge), bits(src.wedge_uv)),
                  f"{tag}: per-corner UV bit-equal")
        if rec["texture"]:
            tex_out = ROOT / Path(rec["obj"]).parent / rec["texture"]
            sha = hashlib.sha256(tex_out.read_bytes()).hexdigest()
            check(sha == rec["texture_sha256"], f"{tag}: texture SHA matches manifest")
            src_sha = hashlib.sha256((ROOT / Path(rec["src"])).parent.joinpath(rec["texture"]).read_bytes()).hexdigest()
            check(sha == src_sha, f"{tag}: texture SHA matches source")
            check(out.texture_files == [rec["texture"]], f"{tag}: MTL references texture")
        print(f"  ok    {tag}")

    if not numeric_only:
        try:
            from scrollkit.render.parity import render_parity_ssim  # noqa: F401

            res_p = ROOT / "reports/m1_render_parity.json"
            check(res_p.exists(), "render-parity report exists (run scripts/render_parity_m1.py)")
            if res_p.exists():
                res = json.loads(res_p.read_text())
                bad = {k: v for k, v in res.get("ssim", {}).items() if v < 0.99}
                check(len(res.get("ssim", {})) == 34 and not bad, f"render parity SSIM ≥0.99 for 34/34 (bad: {bad})")
        except ImportError:
            check(False, "render-parity module not available yet (scrollkit.render.parity)")

    print("== M1:", "GREEN ==" if not FAIL else f"RED ({len(FAIL)} failures) ==")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
