"""Headless EGL scene rendering for scroll meshes.

Meshes enter as numpy arrays only (loaded upstream via scrollkit.io — never pv.read).
Textures enter as image files decoded with PIL, transformed to the per-group sampling
orientation from reports/audit.json:tex_orientation, then wrapped in pv.Texture.

Texture-orientation contract (normative derivation)
----------------------------------------------------
Empirical VTK fact, pinned by reports/smoke/uv_probe.json (t0_maps_to = "bottom_row",
pyvista 0.48.4 / VTK 9.6.2): for a numpy array ``A`` of shape (H, W, c) handed to
``pv.Texture`` (array row 0 = first array row), VTK samples

    texel(s, t) = A[(1 - t) * (H - 1),  s * (W - 1)]

i.e. t=0 reads the LAST array row ("bottom"), t=1 reads array row 0, s=0 reads column 0.

The audit's orientation oracle (reports/audit.json: tex_orientation.definitions, binding)
defines how (s, t) addresses the FILE image ``I`` (row 0 = top of the PNG as decoded):

    topleft:            I[ t      * (H-1),   s      * (W-1) ]
    opengl_bottomleft:  I[ (1-t)  * (H-1),   s      * (W-1) ]
    rot180:             I[ (1-t)  * (H-1),   (1-s)  * (W-1) ]   # == bottomleft + u-flip

Solve for the array ``A = T(I)`` such that VTK's native sampling reproduces the file's
true mapping, i.e. ``A[(1-t)(H-1), s(W-1)] == I[row(s,t), col(s,t)]`` for all (s, t):

    opengl_bottomleft:  rows match, cols match            ->  A = I                (no-op)
    topleft:            substitute r = (1-t)(H-1); need
                        A[r, c] = I[(H-1) - r, c]         ->  A = np.flip(I, axis=0)
    rot180:             rows already match (VTK's native t-inversion supplies the
                        vertical flip of rot180); columns need c -> (W-1) - c
                                                          ->  A = np.flip(I, axis=1)

tests/test_render_scene.py pins these transforms against the analytic definitions.
"""

from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------------------------
# EGL/headless environment — must be pinned BEFORE pyvista/vtk import. This is the
# proven recipe from scripts/smoke_render.py (NVIDIA hardware context verified in
# reports/smoke/egl_report.json).
# ------------------------------------------------------------------------------------
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ.pop("DISPLAY", None)  # headless: never touch an X server
os.environ.setdefault("VTK_DEFAULT_EGL_DEVICE_INDEX", "0")

import numpy as np
import pyvista as pv
from PIL import Image


def _vtk_shader_fragment():
    from vtkmodules.vtkRenderingOpenGL2 import vtkShader  # noqa: PLC0415

    return vtkShader.Fragment

pv.OFF_SCREEN = True

ORIENTATIONS = ("opengl_bottomleft", "topleft", "rot180")

__all__ = [
    "ORIENTATIONS",
    "SceneRenderer",
    "auto_camera",
    "load_texture",
    "orient_texture_array",
    "render_mesh_arrays",
    "split_wedge_to_vertex",
]


def orient_texture_array(arr: np.ndarray, tex_orientation: str) -> np.ndarray:
    """Transform a file-decoded image array (row 0 = file top) so that pv.Texture's
    native sampling (t=0 -> bottom array row; see module docstring) reproduces the
    audit's mapping for `tex_orientation`.

        opengl_bottomleft -> identity        (VTK native == the convention)
        topleft           -> np.flip(arr, 0) (vertical flip)
        rot180            -> np.flip(arr, 1) (horizontal flip only: VTK's built-in
                             t-inversion provides rot180's vertical component)
    """
    if tex_orientation == "opengl_bottomleft":
        out = arr
    elif tex_orientation == "topleft":
        out = np.flip(arr, axis=0)
    elif tex_orientation == "rot180":
        out = np.flip(arr, axis=1)
    else:
        raise ValueError(
            f"unknown tex_orientation {tex_orientation!r} (expected one of {ORIENTATIONS})"
        )
    return np.ascontiguousarray(out)


def load_texture(texture_path: str | Path, tex_orientation: str) -> pv.Texture:
    """Decode an image file with PIL, apply the orientation transform, wrap in pv.Texture.

    Textures are never re-encoded or resized here — bytes on disk stay authoritative;
    this is a read-only decode for the GPU.
    """
    texture_path = Path(texture_path)
    if not texture_path.is_file():
        raise FileNotFoundError(f"texture not found: {texture_path}")
    with Image.open(texture_path) as im:
        arr = np.asarray(im.convert("RGBA"))
    arr = orient_texture_array(arr, tex_orientation)
    tex = pv.Texture(arr)
    # bilinear sampling: VTK's default NEAREST makes texels crawl/shimmer under
    # slow subpixel motion on curved flanks (A-B-A flicker in the temporal probe).
    # Mipmaps stay OFF: mip-averaged alpha erodes the 0.5 cutout at grazing angles
    # (the classic disappearing-foliage failure).
    tex.GetInterpolate() or tex.InterpolateOn()
    return tex


def split_wedge_to_vertex(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert any MeshData to per-vertex-UV arrays (V, F, uv) for rendering.

    * vertex_uv present and bit-equal to the wedge table (or no wedge table at all):
      pass-through — arrays returned unchanged (Groups A/B and their OBJ reloads).
    * otherwise (Group C wedge UVs): duplicate vertices per unique (vertex, uv) pair.
      Dedup is bit-exact (uint32 view), mirroring scrollkit.io.dedup_wedge_uv — no
      epsilon merging, so two meshes with bit-equal wedge tables and identical face
      order produce identical splits.
    """
    if mesh.vertex_uv is not None and mesh.wedge_equals_vertex_uv() in (True, None):
        return mesh.vertices, mesh.faces, mesh.vertex_uv
    if mesh.wedge_uv is None:
        raise ValueError(f"{mesh.source_path or 'mesh'}: no UVs (neither vertex_uv nor wedge_uv)")

    m = mesh.faces.shape[0]
    corner_vid = mesh.faces.reshape(-1).astype(np.int64)  # (3m,)
    flat_uv = np.ascontiguousarray(mesh.wedge_uv.reshape(-1, 2).astype(np.float32, copy=False))
    bits = flat_uv.view(np.uint32)  # (3m, 2)

    rec = np.zeros(corner_vid.size, dtype=[("v", "<i8"), ("ub", "<u4"), ("vb", "<u4")])
    rec["v"] = corner_vid
    rec["ub"] = bits[:, 0]
    rec["vb"] = bits[:, 1]
    _, first_idx, inverse = np.unique(rec, return_index=True, return_inverse=True)

    V = np.ascontiguousarray(mesh.vertices[corner_vid[first_idx]])
    uv = np.ascontiguousarray(flat_uv[first_idx])
    F = inverse.reshape(m, 3).astype(np.int32)
    return V, F, uv


def auto_camera(
    all_points,
    aspect: float,
    margin_frac: float = 0.06,
    *,
    direction=None,
    up_hint=None,
    parallel: bool = True,
    distance_factor: float = 3.0,
) -> dict:
    """Fit a camera on the axis-aligned union bbox of ANY point set (e.g. the union of
    every frame of a trajectory) so nothing ever clips, with `margin_frac` margin.

    direction: unit-ish vector from the focal point toward the camera; None gives an
    isometric-ish (1,1,1) view. Returns a camera dict consumable by SceneRenderer /
    render_mesh_arrays: {position, focal_point, up, parallel_scale, parallel}.
    """
    P = np.asarray(all_points, dtype=np.float64).reshape(-1, 3)
    if P.size == 0:
        raise ValueError("auto_camera: empty point set")
    lo, hi = P.min(axis=0), P.max(axis=0)
    center = 0.5 * (lo + hi)
    ext = hi - lo
    diag = float(np.linalg.norm(ext))
    if diag <= 0.0:
        diag = 1.0

    d = np.array([1.0, 1.0, 1.0]) if direction is None else np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm == 0.0:
        raise ValueError("auto_camera: zero view direction")
    d = d / norm

    if up_hint is None:
        up_hint = (0.0, 1.0, 0.0) if abs(d @ np.array([0.0, 1.0, 0.0])) < 0.95 else (0.0, 0.0, 1.0)
    up = np.asarray(up_hint, dtype=np.float64)
    dop = -d  # direction of projection: camera looks along -d toward the focal point
    right = np.cross(dop, up)
    rn = np.linalg.norm(right)
    if rn < 1e-12:  # up parallel to view axis — pick any orthogonal
        up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
        right = np.cross(dop, up)
        rn = np.linalg.norm(right)
    right /= rn
    true_up = np.cross(right, dop)

    # Project the 8 bbox corners onto the screen axes; fit half-extents + margin.
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    rel = corners - center
    hw = max(float(np.abs(rel @ right).max()), 1e-9)
    hh = max(float(np.abs(rel @ true_up).max()), 1e-9)
    parallel_scale = max(hh, hw / max(float(aspect), 1e-9)) * (1.0 + float(margin_frac))

    position = center + d * (diag * float(distance_factor))
    return {
        "position": tuple(float(x) for x in position),
        "focal_point": tuple(float(x) for x in center),
        "up": tuple(float(x) for x in true_up),
        "parallel_scale": float(parallel_scale),
        "parallel": bool(parallel),
    }


class SceneRenderer:
    """Persistent plotter/actor/texture for cheap per-frame re-rendering.

    The PolyData points buffer is updated IN PLACE (`update_points`) + VTK Modified()
    so animation frames cost one render, not one scene rebuild.
    """

    def __init__(
        self,
        vertices,
        faces,
        uv,
        texture_path,
        tex_orientation,
        *,
        camera=None,
        size=(1024, 1024),
        background="#101114",
        lighting="flat",
        double_sided=True,
        back_texture_path=None,
    ) -> None:
        """back_texture_path: when given, the two sides of the sheet get DISTINCT
        textures — `texture_path` on the +winding-normal side (VTK front faces),
        `back_texture_path` on the −side. Used to put ink on the inner face only
        (physically the carbon ink exists on one side of the papyrus)."""
        if lighting not in ("flat", "studio"):
            raise ValueError(f"lighting must be 'flat' or 'studio', got {lighting!r}")
        V = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32))
        F = np.asarray(faces)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"vertices must be (n,3), got {V.shape}")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"faces must be (m,3), got {F.shape}")
        UV = np.ascontiguousarray(np.asarray(uv, dtype=np.float32))
        if UV.shape != (V.shape[0], 2):
            raise ValueError(f"uv must be ({V.shape[0]},2) per-vertex, got {UV.shape}")

        cells = np.empty((F.shape[0], 4), dtype=np.int64)
        cells[:, 0] = 3
        cells[:, 1:] = F
        self.poly = pv.PolyData(V, cells)
        self.poly.active_texture_coordinates = UV
        self.texture = load_texture(texture_path, tex_orientation)

        self.size = (int(size[0]), int(size[1]))
        self.lighting = lighting
        self.plotter = pv.Plotter(off_screen=True, window_size=list(self.size))
        self.plotter.set_background(background)

        if lighting == "flat":
            shading = dict(lighting=False)  # texture-faithful unlit (parity mode)
        else:  # 'studio' — papyrus is matte: minimal specular (docs/RENDER-STYLE.md)
            shading = dict(lighting=True, ambient=0.18, diffuse=0.92,
                           specular=0.03, specular_power=8.0, smooth_shading=False)
        # MASK cutout, properly: any texel with alpha<255 makes VTK classify a normal
        # actor as TRANSLUCENT and render it in the blend pass without per-fragment
        # self-depth correctness — overlapping geometry (roll over the grown flat sheet)
        # then shows far surfaces (ink) bleeding through near papyrus during the
        # late unwrap. Fix: raw vtkOpenGLPolyDataMapper actors with ForceOpaque + a
        # fragment shader alpha test — true cutout with a working z-buffer.
        def _make_actor(texture, culling: str | None):
            import vtkmodules.vtkRenderingOpenGL2 as glmod
            from vtkmodules.vtkRenderingCore import vtkActor

            mapper = glmod.vtkOpenGLPolyDataMapper()
            mapper.SetInputData(self.poly)
            act = vtkActor()
            act.SetMapper(mapper)
            act.SetTexture(texture)
            act.ForceOpaqueOn()
            act.GetShaderProperty().AddFragmentShaderReplacement(
                "//VTK::TCoord::Impl", True,
                "//VTK::TCoord::Impl\n"
                "  if (gl_FragData[0].a < 0.5) { discard; }\n",
                False,
            )
            prop = act.GetProperty()
            if shading.get("lighting") is False:
                prop.LightingOff()
            else:
                prop.SetAmbient(shading["ambient"])
                prop.SetDiffuse(shading["diffuse"])
                prop.SetSpecular(shading["specular"])
                prop.SetSpecularPower(shading["specular_power"])
            if culling == "back":
                prop.BackfaceCullingOn()
            elif culling == "front":
                prop.FrontfaceCullingOn()
            self.plotter.renderer.AddActor(act)
            return act

        if back_texture_path is None:
            self.actor = _make_actor(self.texture, None if double_sided else "back")
            self.actor_back = None
        else:
            # two single-sided actors over the SAME dataset: +n side / −n side textures
            self.back_texture = load_texture(back_texture_path, tex_orientation)
            self.actor = _make_actor(self.texture, "back")
            self.actor_back = _make_actor(self.back_texture, "front")
        self._points = self.poly.points  # live view into the VTK-owned buffer
        self._camera: dict | None = None
        self.set_camera(camera)
        if lighting == "studio":
            self._add_studio_lights()

    # -- camera ---------------------------------------------------------------------
    def set_camera(self, camera: dict | None = None) -> dict:
        """Apply a camera dict ({position, focal_point, up, parallel_scale, parallel});
        None auto-fits an isometric-ish view on the current points."""
        if camera is None:
            camera = auto_camera(self._points, self.size[0] / self.size[1])
        cam = self.plotter.camera
        cam.position = tuple(camera["position"])
        cam.focal_point = tuple(camera["focal_point"])
        cam.up = tuple(camera["up"])
        parallel = bool(camera.get("parallel", True))
        cam.SetParallelProjection(parallel)
        if parallel and "parallel_scale" in camera:
            cam.parallel_scale = float(camera["parallel_scale"])
        if not parallel and "view_angle" in camera:
            cam.SetViewAngle(float(camera["view_angle"]))
        self.plotter.renderer.ResetCameraClippingRange()
        self._camera = dict(camera)
        return self._camera

    def auto_camera(self, all_points, aspect: float | None = None, margin_frac: float = 0.06) -> dict:
        """Fit + apply a camera on the union bbox of ANY point set (e.g. a whole
        unwrap trajectory) so no frame ever clips. Returns the applied camera dict."""
        if aspect is None:
            aspect = self.size[0] / self.size[1]
        cam = auto_camera(all_points, aspect, margin_frac)  # module function (not shadowed here)
        return self.set_camera(cam)

    # -- per-frame animation path ----------------------------------------------------
    def update_points(self, V) -> None:
        """In-place vtk point update + Modified() — no scene rebuild."""
        V = np.asarray(V)
        if V.shape != self._points.shape:
            raise ValueError(f"update_points: shape {V.shape} != {self._points.shape}")
        self._points[:] = V  # writes through into the VTK float32 buffer
        pts = self.poly.GetPoints()
        pts.GetData().Modified()
        pts.Modified()
        self.poly.Modified()

    def screenshot(self) -> np.ndarray:
        """Render and return the frame as (H, W, 3) uint8."""
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()  # explicit: screenshot() alone may return a stale buffer
        img = self.plotter.screenshot(return_img=True)
        return np.asarray(img)

    # -- lifecycle --------------------------------------------------------------------
    def _add_studio_lights(self) -> None:
        """3-point rig per docs/RENDER-STYLE.md: warm key upper camera-left, cool fill
        ~35% camera-right, faint rim behind-above. Positions are computed once from the
        mesh bbox + current camera basis and stay FIXED in world space."""
        self.plotter.remove_all_lights()
        b = np.array(self.poly.bounds, dtype=np.float64).reshape(3, 2)
        center = b.mean(axis=1)
        L = float(np.linalg.norm(b[:, 1] - b[:, 0])) or 1.0
        cam = self._camera or auto_camera(self._points, self.size[0] / self.size[1])
        pos = np.asarray(cam["position"], dtype=np.float64)
        foc = np.asarray(cam["focal_point"], dtype=np.float64)
        d = pos - foc
        d /= np.linalg.norm(d) or 1.0
        up = np.asarray(cam["up"], dtype=np.float64)
        right = np.cross(-d, up)
        right /= np.linalg.norm(right) or 1.0
        true_up = np.cross(right, -d)

        def add(offset, color, intensity):
            p = center + L * (offset[0] * right + offset[1] * true_up + offset[2] * d)
            self.plotter.add_light(pv.Light(position=tuple(p), focal_point=tuple(center),
                                            color=color, intensity=intensity, positional=False))

        add((-1.1, 1.0, 1.3), (1.0, 0.97, 0.92), 1.0)    # key: warm-neutral, upper camera-left
        add((1.2, 0.25, 1.0), (0.84, 0.90, 1.0), 0.35)   # fill: cool, camera-right, 35% of key
        add((0.15, 1.1, -1.3), (1.0, 1.0, 1.0), 0.45)    # rim: behind-above, silhouette separation

    def close(self) -> None:
        if getattr(self, "plotter", None) is not None:
            self.plotter.close()
            self.plotter = None

    def __enter__(self) -> "SceneRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort GPU resource release
        try:
            self.close()
        except Exception:
            pass


def render_mesh_arrays(
    vertices,
    faces,
    uv,
    texture_path,
    tex_orientation,
    *,
    camera=None,
    size=(1024, 1024),
    background="#101114",
    lighting="flat",
    double_sided=True,
) -> np.ndarray:
    """One-shot textured render of numpy mesh arrays; returns (H, W, 3) uint8.

    vertices (n,3) float / faces (m,3) int / uv (n,2) per-vertex in [0,1] (for wedge-UV
    meshes, split first via split_wedge_to_vertex). texture_path is decoded with PIL and
    oriented per `tex_orientation` ('opengl_bottomleft'|'topleft'|'rot180', from
    reports/audit.json) — see module docstring for the derivation. camera=None auto-fits
    an isometric-ish parallel view; lighting='flat' is the texture-faithful parity mode.
    """
    r = SceneRenderer(
        vertices, faces, uv, texture_path, tex_orientation,
        camera=camera, size=size, background=background,
        lighting=lighting, double_sided=double_sided,
    )
    try:
        return r.screenshot()
    finally:
        r.close()
