#!/usr/bin/env python
"""Per-frame textured OBJ export from a computed trajectory (frames.npz).

Variant 1 (plain): <out>/plain/frame_0000.obj … + mesh.mtl + texture copy.
Variant 2 (ink):   <out>/ink/frame_0000.obj are HARDLINKS of plain's OBJs (geometry bytes
identical); only ink/mesh.mtl + the overlay texture differ. Zero geometry duplication.

Usage: uv run python scripts/export_frames.py <anim_dir> --texture <plain_tex> [--overlay <ink_tex>]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scrollkit.io import MeshData, ObjFrameWriter, copy_texture, write_mtl  # noqa: E402
from scrollkit.io.obj import ShellObjFrameWriter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("anim_dir")
    ap.add_argument("--texture", required=True)
    ap.add_argument("--overlay")
    ap.add_argument("--out", default=None, help="default: <anim_dir>/frames_obj")
    args = ap.parse_args()

    adir = Path(args.anim_dir)
    data = np.load(adir / "frames.npz")
    frames, faces, uv = data["frames"], data["faces"], data["uv"]
    out = Path(args.out) if args.out else adir / "frames_obj"
    tex = Path(args.texture)

    t0 = time.time()
    plain = out / "plain"
    plain.mkdir(parents=True, exist_ok=True)
    copy_texture(tex, plain / tex.name)
    write_mtl(plain / "mesh.mtl", [("material_0", tex.name)])

    mesh0 = MeshData(vertices=frames[0].astype(np.float32), faces=faces.astype(np.int32),
                     vertex_uv=uv.astype(np.float32))
    writer = ObjFrameWriter(mesh0, texture_file=tex.name, mtl_name="mesh.mtl",
                            header_comment=f"scrollkit unwrap frame ({adir.name})")
    for i in range(len(frames)):
        writer.write_frame(plain / f"frame_{i:04d}.obj", frames[i])

    n_ink = 0
    if args.overlay:
        # Ink variant: OBJ/MTL cannot put different textures on the two sides of one
        # surface (any viewer would paint ink on the verso too — physically wrong).
        # Ship a thin two-sided shell instead: recto surface = ink, verso = plain.
        from scrollkit.render import cinematics

        ink = out / "ink"
        ink.mkdir(parents=True, exist_ok=True)
        ov = Path(args.overlay)
        copy_texture(ov, ink / ov.name)
        copy_texture(tex, ink / tex.name)
        write_mtl(ink / "mesh.mtl", [("ink_recto", ov.name), ("plain_verso", tex.name)])
        inner = cinematics.inner_side_sign(frames[0], faces, uv)["inner_sign"]
        shell = ShellObjFrameWriter(mesh0, inner_sign=float(inner), mtl_name="mesh.mtl",
                                    header_comment=f"scrollkit unwrap frame shell ({adir.name})")
        for i in range(len(frames)):
            shell.write_frame(ink / f"frame_{i:04d}.obj", frames[i])
            n_ink += 1

    sz = sum(f.stat().st_size for f in plain.glob("frame_*.obj"))
    print(f"[{adir.name}] {len(frames)} frame OBJs ({sz / 1e9:.2f} GB geometry, "
          f"{n_ink} two-sided shell ink frames) in {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
