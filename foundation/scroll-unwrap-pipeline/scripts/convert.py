#!/usr/bin/env python
"""M1: exact PLY→OBJ conversion for all 34 meshes.

Per mesh -> outputs/obj/<group>/<stem>/{<stem>.obj, <stem>.mtl, <texture>} with the
texture byte-copied (SHA256-verified). Writes outputs/obj/manifest.json consumed by gate_m1.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scrollkit.io import copy_texture, read_ply, write_obj  # noqa: E402


def discover() -> list[tuple[str, Path]]:
    meshes: list[tuple[str, Path]] = []
    for p in sorted((ROOT / "textured_plys/textured_ply_march").glob("*.ply")):
        meshes.append(("A", p))
    for p in sorted((ROOT / "textured_plys/textured_ply_may").glob("*.ply")):
        meshes.append(("B", p))
    for p in sorted((ROOT / "scroll_meshes-20260610T062158Z-3-001/scroll_meshes").glob("*/*.ply")):
        meshes.append(("C", p))
    return meshes


def convert_one(group: str, src: Path) -> dict:
    t0 = time.time()
    mesh = read_ply(src)
    stem = src.stem
    out_dir = ROOT / "outputs/obj" / group / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    # texture: first declared TextureFile that exists next to the PLY
    tex_name = None
    tex_sha = None
    for cand in mesh.texture_files:
        if (src.parent / cand).exists():
            tex_name = cand
            break
        notes.append(f"declared texture missing on disk: {cand}")
    if tex_name is None and mesh.texture_files:
        raise FileNotFoundError(f"{src}: no declared texture exists on disk")
    if tex_name:
        tex_sha = copy_texture(src.parent / tex_name, out_dir / tex_name)

    if group in ("A", "B"):
        eq = mesh.wedge_equals_vertex_uv()
        uv_mode = "shared" if eq else "wedge"
        if not eq:
            notes.append("wedge UV != vertex UV — wedge-dedup path used")
    else:
        uv_mode = "wedge"
    if mesh.face_texnumber is not None:
        import numpy as np

        full = set(np.unique(mesh.face_texnumber).tolist())
        if full != {0}:
            raise NotImplementedError(f"{src}: multi-material texnumbers {full} — extend writer")
        if len(mesh.texture_files) > 1:
            notes.append(f"texnumber==0 for all faces; extra declared textures ignored: {mesh.texture_files[1:]}")

    obj_path = out_dir / f"{stem}.obj"
    write_obj(
        obj_path,
        mesh,
        texture_file=tex_name,
        uv_mode=uv_mode,
        header_comment=f"source: {src.relative_to(ROOT)} | group {group} | uv_mode {uv_mode}",
    )
    return {
        "group": group,
        "src": str(src.relative_to(ROOT)),
        "obj": str(obj_path.relative_to(ROOT)),
        "mtl": str(obj_path.with_suffix(".mtl").relative_to(ROOT)) if tex_name else None,
        "texture": tex_name,
        "texture_sha256": tex_sha,
        "uv_mode": uv_mode,
        "n_vertices": mesh.n_vertices,
        "n_faces": mesh.n_faces,
        "has_normals": mesh.normals is not None,
        "notes": notes,
        "seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    meshes = discover()
    print(f"converting {len(meshes)} meshes")
    records = []
    for group, src in meshes:
        rec = convert_one(group, src)
        records.append(rec)
        print(f"  [{group}] {src.stem}: {rec['n_faces']} faces, uv={rec['uv_mode']}, {rec['seconds']}s"
              + (f"  NOTES: {rec['notes']}" if rec["notes"] else ""))
    man = ROOT / "outputs/obj/manifest.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps({"meshes": records}, indent=2))
    print(f"wrote {man} ({len(records)} meshes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
