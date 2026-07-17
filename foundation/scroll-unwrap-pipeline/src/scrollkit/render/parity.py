"""Render-parity QA: SSIM between two meshes rendered with identical camera/lighting.

Used by the M1 gate to prove that a converted OBJ renders indistinguishably from its
source PLY (threshold SSIM >= 0.99 per qa-gates). Both meshes go through the same
arrays-only pipeline (scrollkit.render.scene); wedge-UV meshes (Group C) are corner-split
via split_wedge_to_vertex. Two opposed views (front +/-) are rendered and the MIN SSIM
across views is returned, guarding against single-view z-fighting flukes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .scene import SceneRenderer, auto_camera, split_wedge_to_vertex


def _resolve_texture(mesh, tex_dir: str | Path) -> Path:
    """First declared texture that exists in tex_dir (PHerc172 declares a second,
    missing file — audit-known quirk; we render the single existing material)."""
    tex_dir = Path(tex_dir)
    names = list(mesh.texture_files or [])
    for name in names:
        p = tex_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"none of the declared textures {names} found in {tex_dir}")


def _sheet_view_direction(points: np.ndarray) -> np.ndarray:
    """Least-variance principal axis = face-on direction for sheet-like wraps, so the
    parity views actually exercise the texture (deterministic sign convention)."""
    P = np.asarray(points, dtype=np.float64)
    step = max(1, P.shape[0] // 200_000)
    Q = P[::step]
    Q = Q - Q.mean(axis=0)
    cov = (Q.T @ Q) / max(Q.shape[0] - 1, 1)
    _, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
    n = vecs[:, 0]
    k = int(np.argmax(np.abs(n)))
    if n[k] < 0:
        n = -n
    return n


def render_parity_ssim(
    mesh_a,
    mesh_b,
    texture_dir_a,
    texture_dir_b,
    tex_orientation,
    size: int = 1024,
    *,
    return_images: bool = False,
) -> dict:
    """Render mesh_a and mesh_b with identical auto cameras (2 opposed face-on views)
    and flat (unlit, texture-faithful) lighting; compare per view.

    Returns {'ssim': min SSIM across views (grayscale), 'max_px': max abs uint8 pixel
    diff across views} plus per-view diagnostics. With return_images=True, also the
    front-view pair under 'images' (img_a, img_b) for sample sheets.
    """
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity

    Va, Fa, uva = split_wedge_to_vertex(mesh_a)
    Vb, Fb, uvb = split_wedge_to_vertex(mesh_b)
    tex_a = _resolve_texture(mesh_a, texture_dir_a)
    tex_b = _resolve_texture(mesh_b, texture_dir_b)

    union = np.vstack([Va.astype(np.float64, copy=False), Vb.astype(np.float64, copy=False)])
    d = _sheet_view_direction(union)
    views = [
        auto_camera(union, 1.0, direction=d),   # front
        auto_camera(union, 1.0, direction=-d),  # back
    ]

    def shots(V, F, uv, tex_path):
        # GPU is serial: one renderer alive at a time; actors/textures freed on close.
        r = SceneRenderer(V, F, uv, tex_path, tex_orientation,
                          camera=views[0], size=(size, size), lighting="flat")
        try:
            out = []
            for cam in views:
                r.set_camera(cam)
                out.append(r.screenshot())
            return out
        finally:
            r.close()

    imgs_a = shots(Va, Fa, uva, tex_a)
    imgs_b = shots(Vb, Fb, uvb, tex_b)

    ssims: list[float] = []
    max_px = 0
    for ia, ib in zip(imgs_a, imgs_b):
        ga, gb = rgb2gray(ia), rgb2gray(ib)
        ssims.append(float(structural_similarity(ga, gb, data_range=1.0)))
        max_px = max(max_px, int(np.abs(ia.astype(np.int16) - ib.astype(np.int16)).max()))

    out = {
        "ssim": float(min(ssims)),
        "max_px": int(max_px),
        "ssim_views": ssims,
        "views": ["front", "back"],
    }
    if return_images:
        out["images"] = (imgs_a[0], imgs_b[0])
    return out
