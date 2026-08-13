"""Face visibility from texture alpha, using the audit's normative pixel conventions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_GROUP_DIRS = {
    "textured_ply_march": "A",
    "textured_ply_may": "B",
}


def group_of_source(src: Path) -> str:
    return _GROUP_DIRS.get(src.parent.name, "C")


def uv_to_pixel(uv: np.ndarray, w: int, h: int, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    """Map UV to (row, col) under the audit's convention strings."""
    s, t = uv[:, 0], uv[:, 1]
    if orientation == "topleft":
        rows, cols = t * (h - 1), s * (w - 1)
    elif orientation == "opengl_bottomleft":
        rows, cols = (1.0 - t) * (h - 1), s * (w - 1)
    elif orientation == "rot180":
        rows, cols = (1.0 - t) * (h - 1), (1.0 - s) * (w - 1)
    else:
        raise ValueError(f"unknown tex orientation {orientation!r}")
    return np.clip(rows, 0, h - 1).astype(np.int64), np.clip(cols, 0, w - 1).astype(np.int64)


def visible_face_mask(mesh, src: Path, repo_root: Path, alpha_min: int = 16) -> np.ndarray:
    """True per face iff the texture alpha at the face's UV centroid exceeds alpha_min.
    Meshes without alpha (RGB textures / no texture) are fully visible."""
    src = Path(src)
    group = group_of_source(src)
    tex = None
    for cand in mesh.texture_files:
        p = src.parent / cand
        if p.exists():
            tex = p
            break
    if tex is None or mesh.vertex_uv is None:
        return np.ones(len(mesh.faces), dtype=bool)
    img = Image.open(tex)
    if "A" not in img.getbands():
        return np.ones(len(mesh.faces), dtype=bool)
    alpha = np.asarray(img.getchannel("A"))
    h, w = alpha.shape
    audit = json.loads((Path(repo_root) / "reports/audit.json").read_text())
    orientation = audit["tex_orientation"][group]
    if isinstance(orientation, dict):
        orientation = orientation.get("resolved_orientation", "topleft")
    uvc = mesh.vertex_uv[mesh.faces].mean(axis=1)
    rows, cols = uv_to_pixel(uvc, w, h, orientation)
    return alpha[rows, cols] > alpha_min
