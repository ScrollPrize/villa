"""Offscreen (EGL) regression tests pinning scrollkit.render.scene's texture-orientation
transforms and the SceneRenderer animation path.

The expected corner colors are derived ANALYTICALLY from the audit's normative
orientation definitions (reports/audit.json: tex_orientation.definitions):

    topleft:            pixel_col = s*(W-1);     pixel_row = t*(H-1)      (row 0 = file top)
    opengl_bottomleft:  pixel_col = s*(W-1);     pixel_row = (1-t)*(H-1)
    rot180:             pixel_col = (1-s)*(W-1); pixel_row = (1-t)*(H-1)  (== bottomleft
                                                                           with u also flipped)

NOT from the implementation — so any regression in the numpy flips chosen for VTK's
native convention (uv_probe: t=0 samples the BOTTOM array row) fails here.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fixtures import asymmetric_quad

PALETTE = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}
TEX_W = TEX_H = 32  # quadrant texture dims used by every test


def _egl_available() -> bool:
    try:
        import scrollkit.render.scene  # noqa: F401  (pins headless EGL env first)
        import pyvista as pv

        pl = pv.Plotter(off_screen=True, window_size=(32, 32))
        pl.add_mesh(pv.Sphere())
        img = pl.screenshot(return_img=True)
        pl.close()
        return img is not None and img.size > 0
    except Exception:
        return False


EGL_OK = _egl_available()
needs_egl = pytest.mark.skipif(not EGL_OK, reason="EGL offscreen rendering unavailable")


# --------------------------------------------------------------------------------------
# analytic machinery (independent of scrollkit.render internals)
# --------------------------------------------------------------------------------------
def make_quadrant_texture(tmp_path):
    """4-color file image: row 0 = file TOP. TL=red TR=green BL=blue BR=yellow."""
    img = np.zeros((TEX_H, TEX_W, 3), np.uint8)
    img[: TEX_H // 2, : TEX_W // 2] = PALETTE["red"]
    img[: TEX_H // 2, TEX_W // 2 :] = PALETTE["green"]
    img[TEX_H // 2 :, : TEX_W // 2] = PALETTE["blue"]
    img[TEX_H // 2 :, TEX_W // 2 :] = PALETTE["yellow"]
    path = tmp_path / "quadrants.png"
    Image.fromarray(img).save(path)
    return path


def expected_color(s: float, t: float, orientation: str) -> str:
    """Which file-image quadrant does (s,t) address under the audit's definition?"""
    if orientation == "topleft":
        row, col = t * (TEX_H - 1), s * (TEX_W - 1)
    elif orientation == "opengl_bottomleft":
        row, col = (1.0 - t) * (TEX_H - 1), s * (TEX_W - 1)
    elif orientation == "rot180":
        row, col = (1.0 - t) * (TEX_H - 1), (1.0 - s) * (TEX_W - 1)
    else:
        raise ValueError(orientation)
    top, left = row < TEX_H / 2, col < TEX_W / 2
    return {(True, True): "red", (True, False): "green",
            (False, True): "blue", (False, False): "yellow"}[(top, left)]


def classify(px) -> tuple[str, int]:
    px = np.asarray(px, dtype=np.int64)
    name, ref = min(PALETTE.items(), key=lambda kv: int(np.abs(np.array(kv[1]) - px).sum()))
    return name, int(np.abs(np.array(ref) - px).sum())


def world_to_pixel(p, cam, size) -> tuple[int, int]:
    """Project a world point through a parallel camera dict to (row, col) of the
    screenshot (row 0 = screen top). Pixel k spans NDC [(2k/W)-1, (2(k+1)/W)-1]."""
    W, H = size
    pos = np.asarray(cam["position"], float)
    foc = np.asarray(cam["focal_point"], float)
    up = np.asarray(cam["up"], float)
    dop = foc - pos
    dop /= np.linalg.norm(dop)
    right = np.cross(dop, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, dop)
    ps = float(cam["parallel_scale"])
    aspect = W / H
    rel = np.asarray(p, float) - foc
    x = float(rel @ right) / (ps * aspect)  # NDC in [-1, 1]
    y = float(rel @ true_up) / ps
    col = int(round((x + 1.0) * W / 2.0 - 0.5))
    row = int(round((1.0 - y) * H / 2.0 - 0.5))
    return row, col


# --------------------------------------------------------------------------------------
# orientation pinning: asymmetric-quad fixture x 4-color texture x 3 conventions
# --------------------------------------------------------------------------------------
@needs_egl
@pytest.mark.parametrize("orientation", ["opengl_bottomleft", "topleft", "rot180"])
def test_orientation_corner_landing(tmp_path, orientation):
    from scrollkit.render.scene import render_mesh_arrays

    q = asymmetric_quad()
    tex = make_quadrant_texture(tmp_path)
    size = (512, 512)

    # The fixture's two triangles share an identical screen footprint (v3 duplicates
    # v0's position) and would z-fight — render each triangle separately. Together
    # their corners cover all four UV quadrants.
    for tri in range(2):
        F = q["faces"][tri : tri + 1]
        ids = F[0]
        Pw = q["vertices"][ids].astype(np.float64)
        UV = q["vertex_uv"][ids].astype(np.float64)
        cen_p, cen_uv = Pw.mean(axis=0), UV.mean(axis=0)

        # face-on parallel camera looking down -z (quad lives near the z=0 plane)
        lo, hi = Pw[:, :2].min(axis=0), Pw[:, :2].max(axis=0)
        c = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        cam = dict(
            position=(c[0], c[1], 10.0),
            focal_point=(c[0], c[1], 0.0),
            up=(0.0, 1.0, 0.0),
            parallel_scale=float(max(half)) * 1.2,
            parallel=True,
        )
        img = render_mesh_arrays(
            q["vertices"], F, q["vertex_uv"], tex, orientation,
            camera=cam, size=size, background="#000000", lighting="flat",
        )
        assert img.shape == (size[1], size[0], 3) and img.dtype == np.uint8

        for k in range(3):
            # sample pulled 30% toward the centroid: stays inside the triangle (no edge
            # antialiasing) and inside the corner's UV quadrant (margin >= 0.18 from 0.5)
            pw = 0.7 * Pw[k] + 0.3 * cen_p
            s, t = 0.7 * UV[k] + 0.3 * cen_uv
            exp = expected_color(s, t, orientation)
            row, col = world_to_pixel(pw, cam, size)
            patch = img[row - 1 : row + 2, col - 1 : col + 2].reshape(-1, 3).astype(float).mean(axis=0)
            got, dist = classify(patch)
            assert dist < 60, f"{orientation} tri{tri} corner{k}: ambiguous pixel {patch.tolist()}"
            assert got == exp, (
                f"{orientation} tri{tri} corner{k} (s={s:.3f}, t={t:.3f}): rendered {got} "
                f"{patch.tolist()}, audit definition demands {exp}"
            )


# --------------------------------------------------------------------------------------
# wedge corner-split (pure numpy — no GPU needed)
# --------------------------------------------------------------------------------------
def test_split_wedge_to_vertex_bit_exact():
    from scrollkit.io import MeshData
    from scrollkit.render.scene import split_wedge_to_vertex

    q = asymmetric_quad()

    # wedge == vertex (A/B case): pass-through, no resplit
    mesh = MeshData(vertices=q["vertices"], faces=q["faces"],
                    vertex_uv=q["vertex_uv"], wedge_uv=q["wedge_uv"])
    V, F, uv = split_wedge_to_vertex(mesh)
    assert V is mesh.vertices and F is mesh.faces and uv is mesh.vertex_uv

    # wedge-only (C case), wedge derived from vertex table: shared corners merge back
    mesh_c = MeshData(vertices=q["vertices"], faces=q["faces"], wedge_uv=q["wedge_uv"])
    V, F, uv = split_wedge_to_vertex(mesh_c)
    assert V.dtype == np.float32 and uv.dtype == np.float32 and F.dtype == np.int32
    assert V.shape[0] == 4  # 4 unique (vertex, uv) pairs (v0/v3 same position, distinct uv)
    assert np.array_equal(uv[F].view(np.uint32), q["wedge_uv"].view(np.uint32))
    assert np.array_equal(V[F].view(np.uint32), q["vertices"][q["faces"]].view(np.uint32))

    # true per-corner divergence: face 1 re-textures its corners -> vertices must split
    w = q["wedge_uv"].copy()
    w[1] = np.array([[0.5, 0.5], [0.25, 0.75], [0.125, 0.875]], np.float32)
    mesh_s = MeshData(vertices=q["vertices"], faces=q["faces"], wedge_uv=w)
    V, F, uv = split_wedge_to_vertex(mesh_s)
    assert V.shape[0] == 6  # f0:(v0,v1,v2) + f1:(v2',v1',v3) all unique pairs
    assert np.array_equal(uv[F].view(np.uint32), w.view(np.uint32))
    assert np.array_equal(V[F].view(np.uint32), q["vertices"][q["faces"]].view(np.uint32))


# --------------------------------------------------------------------------------------
# auto_camera fit guarantee (pure numpy)
# --------------------------------------------------------------------------------------
def test_auto_camera_fits_union_bbox():
    from scrollkit.render.scene import auto_camera

    rng = np.random.default_rng(7)
    P = rng.uniform(-3.0, 9.0, size=(5000, 3)) * np.array([4.0, 1.0, 0.25])
    aspect, margin = 16 / 9, 0.06
    cam = auto_camera(P, aspect, margin)
    assert cam["parallel"] is True
    pos = np.asarray(cam["position"])
    foc = np.asarray(cam["focal_point"])
    assert np.allclose(foc, 0.5 * (P.min(0) + P.max(0)))
    dop = foc - pos
    dop /= np.linalg.norm(dop)
    up = np.asarray(cam["up"])
    right = np.cross(dop, up)
    right /= np.linalg.norm(right)
    rel = P - foc
    hw = np.abs(rel @ right).max()
    hh = np.abs(rel @ np.cross(right, dop)).max()
    ps = cam["parallel_scale"]
    # every point inside the frame with the 6% margin honored
    assert hh <= ps and hw <= ps * aspect
    assert ps >= max(hh, hw / aspect) * (1 + margin) - 1e-9


# --------------------------------------------------------------------------------------
# SceneRenderer animation path: in-place point update == fresh render
# --------------------------------------------------------------------------------------
@needs_egl
def test_scene_renderer_update_points(tmp_path):
    from scrollkit.render.scene import SceneRenderer, render_mesh_arrays

    tex = make_quadrant_texture(tmp_path)
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    F = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    cam = dict(position=(1.5, 0.5, 5.0), focal_point=(1.5, 0.5, 0.0),
               up=(0.0, 1.0, 0.0), parallel_scale=1.6, parallel=True)

    r = SceneRenderer(V, F, uv, tex, "opengl_bottomleft", camera=cam, size=(256, 256))
    img0 = r.screenshot()
    V2 = V + np.array([2.0, 0.0, 0.0], np.float32)
    r.update_points(V2)
    img1 = r.screenshot()
    r.close()

    assert not np.array_equal(img0, img1), "update_points had no visible effect"
    fresh = render_mesh_arrays(V2, F, uv, tex, "opengl_bottomleft", camera=cam, size=(256, 256))
    assert np.array_equal(img1, fresh), "in-place point update diverges from fresh render"
