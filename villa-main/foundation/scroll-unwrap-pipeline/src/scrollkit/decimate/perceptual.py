"""M2 perceptual (SSIM) gate: render source vs decimated through scrollkit.render.scene.

Per mesh, at its chosen rung, four views are compared at 1024 px (grayscale SSIM,
skimage.structural_similarity):

  full_iso   production-ish auto-fit isometric view            SSIM >= 0.97
  full_face  production-ish face-on view (thinnest bbox axis)  SSIM >= 0.97
  close_curv 25%-of-bbox crop at the max mean-curvature region
             (approximated by vertex-normal variation)         SSIM >= 0.95
  close_tex  25%-of-bbox crop at the densest-texture-content
             region (texture alpha centroid mapped through UV) SSIM >= 0.95

Cameras are computed ONCE from the SOURCE mesh and reused for the decimated render.
On failure the rung is retried with extratcoordw=3.0 (UV-smearing remedy), then the
ladder moves up; every attempt is recorded in the mesh's ssim journey.

Group C policy: the ladder for C extends with
GROUP_C_EXTENDED_LADDER ([0.5, 0.65, 0.8]); if NO rung passes, the mesh ships
UNDECIMATED (ship_undecimated_copy — byte-identical M1 copy, SHA-verified) with
decision='no_safe_decimation', and the identity deliverable is still rendered through
this harness as a final wiring check (expected SSIM = 1.0).

GPU rendering is SERIAL — callers must not parallelize this module. A concurrent
concurrent renderers share the GPU: transient EGL/context failures are retried
(30 s sleep, 20 min budget) in _render_views_retrying.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from scrollkit.io import MeshData, read_obj, read_ply

VIEW_DEFS = (
    ("full_iso", False),
    ("full_face", False),
    ("close_curv", True),
    ("close_tex", True),
)
CLOSEUP_BBOX_FRAC = 0.25  # close-up frames 25% of the bbox diagonal
RENDER_SIZE = (1024, 1024)  # comparison resolution (gate spec: 1024 px)
# Measurement protocol (anti-aliasing, pinned by convergence experiments):
# The 8K papyrus textures minify ~10-15x at 1024 px. With VTK's default non-mipmapped
# point sampling the minification aliasing decorrelates between source and decimated
# geometry and dominates SSIM (identical-looking close-ups measured 0.89/0.95/0.96/0.97
# at 1x/2x/3x/4x supersampling — never converging). Enabling trilinear MIPMAP texture
# sampling (the anti-aliased ideal) converges the estimator: ss=4+mipmap and
# ss=8+mipmap agree to 0.0002. So: mipmap+interpolate ON (set on this module's own
# SceneRenderer instances — both sides of every comparison identically) and 4x
# supersampling for geometry-edge AA, LANCZOS-downsampled to the gate's 1024 px.
# SSIM then measures decimation error, not GPU sampling noise.
SUPERSAMPLE = 4
TEXTURE_MIPMAP = True

# GPU contention with the concurrent cinematics renders: EGL context creation can fail
# transiently. Serial retry: sleep 30 s between attempts, give up after a 20 min budget.
RENDER_RETRY_SLEEP_S = 30.0
RENDER_RETRY_BUDGET_S = 20 * 60.0
_RENDER_TRANSIENT_MARKERS = ("egl", "context", "glx", "opengl", "framebuffer", "device")


# ----------------------------------------------------------------- mesh-derived regions
def _vertex_normals(mesh: MeshData) -> np.ndarray:
    """File normals when present (A/B); else area-weighted accumulation (C)."""
    if mesh.normals is not None:
        n = mesh.normals.astype(np.float64)
    else:
        v = mesh.vertices.astype(np.float64)
        f = mesh.faces.astype(np.int64)
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # area-weighted
        n = np.zeros_like(v)
        for k in range(3):
            np.add.at(n, f[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def _unique_edges(faces: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    f = faces.astype(np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    uniq = np.unique(e[:, 0] * n_vertices + e[:, 1])
    return uniq // n_vertices, uniq % n_vertices


def normal_variation_scores(mesh: MeshData) -> np.ndarray:
    """Per-vertex mean angular deviation of neighbor normals (curvature proxy),
    smoothed once over the 1-ring to suppress single-vertex noise."""
    N = _vertex_normals(mesh)
    a, b = _unique_edges(mesh.faces, mesh.n_vertices)
    d = 1.0 - np.einsum("ij,ij->i", N[a], N[b])
    nv = mesh.n_vertices
    acc = np.zeros(nv)
    cnt = np.zeros(nv)
    for idx, val in ((a, d), (b, d)):
        np.add.at(acc, idx, val)
        np.add.at(cnt, idx, 1.0)
    cnt[cnt == 0] = 1.0
    score = acc / cnt
    sm = np.zeros(nv)
    cm = np.zeros(nv)
    for src, dst in ((a, b), (b, a)):
        np.add.at(sm, dst, score[src])
        np.add.at(cm, dst, 1.0)
    cm[cm == 0] = 1.0
    return 0.5 * score + 0.5 * (sm / cm)


def texture_content_uv_centroid(texture_path: str | Path, tex_orientation: str) -> tuple[float, float]:
    """(s, t) of the texture content centroid: alpha-mask moments when the image has a
    meaningful alpha channel, else a luminance mask. Pixel->UV mapping inverts the
    audit's tex_orientation definitions."""
    with Image.open(texture_path) as im:
        alpha = np.asarray(im.getchannel("A")) if im.mode in ("RGBA", "LA", "PA") else None
        if alpha is not None and alpha.min() != alpha.max():
            mask = alpha > 0
        else:
            mask = np.asarray(im.convert("L")) > 16
    h, w = mask.shape
    m = mask.astype(np.float64)
    total = m.sum()
    if total == 0:  # blank texture — fall back to image center
        row_c, col_c = (h - 1) / 2.0, (w - 1) / 2.0
    else:
        row_c = float(m.sum(axis=1) @ np.arange(h)) / total
        col_c = float(m.sum(axis=0) @ np.arange(w)) / total
    rs, cs = row_c / max(h - 1, 1), col_c / max(w - 1, 1)
    if tex_orientation == "topleft":
        s, t = cs, rs
    elif tex_orientation == "opengl_bottomleft":
        s, t = cs, 1.0 - rs
    elif tex_orientation == "rot180":
        s, t = 1.0 - cs, 1.0 - rs
    elif tex_orientation == "topleft_u_flipped":
        s, t = 1.0 - cs, rs
    else:
        raise ValueError(f"unknown tex_orientation {tex_orientation!r}")
    return float(s), float(t)


def _nearest_vertex_by_uv(mesh: MeshData, st: tuple[float, float]) -> int:
    """Vertex whose UV is nearest (s,t). Uses vertex UVs when present, else wedge corners."""
    target = np.array(st)
    if mesh.vertex_uv is not None:
        d = np.linalg.norm(mesh.vertex_uv.astype(np.float64) - target, axis=1)
        return int(np.argmin(d))
    corners = mesh.wedge_uv.reshape(-1, 2).astype(np.float64)
    ci = int(np.argmin(np.linalg.norm(corners - target, axis=1)))
    return int(mesh.faces.reshape(-1)[ci])


def _region_direction(mesh: MeshData, normals: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Mean normal over vertices within `radius` of center (face-on close-up direction);
    falls back to the global thinnest-bbox axis when the region normal cancels out."""
    v = mesh.vertices.astype(np.float64)
    near = np.linalg.norm(v - center, axis=1) <= radius
    d = normals[near].sum(axis=0) if near.any() else np.zeros(3)
    if np.linalg.norm(d) < 1e-9:
        ext = v.max(axis=0) - v.min(axis=0)
        d = np.eye(3)[int(np.argmin(ext))]
    return d / np.linalg.norm(d)


# ----------------------------------------------------------------- cameras + rendering
def build_views(mesh: MeshData, texture_path: Path, tex_orientation: str) -> list[dict]:
    """Four named cameras computed from the SOURCE mesh only."""
    from scrollkit.render.scene import auto_camera

    v = mesh.vertices.astype(np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    ext = hi - lo
    diag = float(np.linalg.norm(ext))
    aspect = RENDER_SIZE[0] / RENDER_SIZE[1]
    normals = _vertex_normals(mesh)

    # production-ish full views
    cam_iso = auto_camera(v, aspect)
    thin = int(np.argmin(ext))
    big = int(np.argmax(ext))
    face_dir = np.eye(3)[thin] + 0.3 * np.eye(3)[big]
    cam_face = auto_camera(v, aspect, direction=face_dir / np.linalg.norm(face_dir))

    # close-up crops: 25% of the bbox
    half_h = 0.5 * CLOSEUP_BBOX_FRAC * diag
    region_r = half_h

    scores = normal_variation_scores(mesh)
    c_curv = v[int(np.argmax(scores))]
    d_curv = _region_direction(mesh, normals, c_curv, region_r)

    st = texture_content_uv_centroid(texture_path, tex_orientation)
    c_tex = v[_nearest_vertex_by_uv(mesh, st)]
    d_tex = _region_direction(mesh, normals, c_tex, region_r)

    def closeup(center: np.ndarray, d: np.ndarray) -> dict:
        up = (0.0, 0.0, 1.0) if abs(d[2]) < 0.95 else (0.0, 1.0, 0.0)
        dop = -d
        right = np.cross(dop, np.asarray(up, dtype=np.float64))
        right /= np.linalg.norm(right)
        true_up = np.cross(right, dop)
        return {
            "position": tuple(float(x) for x in (center + d * diag)),
            "focal_point": tuple(float(x) for x in center),
            "up": tuple(float(x) for x in true_up),
            "parallel_scale": float(half_h),
            "parallel": True,
        }

    cams = {
        "full_iso": cam_iso,
        "full_face": cam_face,
        "close_curv": closeup(c_curv, d_curv),
        "close_tex": closeup(c_tex, d_tex),
    }
    return [{"name": n, "closeup": cu, "camera": cams[n]} for n, cu in VIEW_DEFS]


def _render_views(mesh: MeshData, texture_path: Path, tex_orientation: str, views: list[dict]) -> dict[str, np.ndarray]:
    """One SceneRenderer per mesh; camera swapped per view (GPU-serial).
    Renders SUPERSAMPLE x oversized, then LANCZOS-downsamples to RENDER_SIZE."""
    from scrollkit.render.scene import SceneRenderer, split_wedge_to_vertex

    V, F, UV = split_wedge_to_vertex(mesh)
    big = (RENDER_SIZE[0] * SUPERSAMPLE, RENDER_SIZE[1] * SUPERSAMPLE)
    out: dict[str, np.ndarray] = {}
    with SceneRenderer(V, F, UV, texture_path, tex_orientation, size=big) as r:
        if TEXTURE_MIPMAP:  # converged sampling (see protocol note above); both sides
            r.texture.mipmap = True
            r.texture.interpolate = True
        for view in views:
            r.set_camera(view["camera"])
            img = r.screenshot()
            if SUPERSAMPLE > 1:
                img = np.asarray(Image.fromarray(img).resize(RENDER_SIZE, Image.LANCZOS))
            out[view["name"]] = img
    return out


def _render_views_retrying(
    mesh: MeshData, texture_path: Path, tex_orientation: str, views: list[dict]
) -> dict[str, np.ndarray]:
    """_render_views with retry on transient GPU/EGL context failures (shared GPU)."""
    deadline = time.monotonic() + RENDER_RETRY_BUDGET_S
    attempt = 1
    while True:
        try:
            return _render_views(mesh, texture_path, tex_orientation, views)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            transient = any(k in msg.lower() for k in _RENDER_TRANSIENT_MARKERS)
            if not transient or time.monotonic() + RENDER_RETRY_SLEEP_S > deadline:
                raise
            print(
                f"    render attempt {attempt} hit transient GPU/EGL failure "
                f"({msg[:140]}); retrying in {RENDER_RETRY_SLEEP_S:.0f}s"
            )
            attempt += 1
            time.sleep(RENDER_RETRY_SLEEP_S)


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    ga = a.astype(np.float64).mean(axis=2)
    gb = b.astype(np.float64).mean(axis=2)
    return float(structural_similarity(ga, gb, data_range=255.0))


def evaluate_ssim(
    src_mesh: MeshData,
    dec_obj_path: Path,
    texture_path: Path,
    tex_orientation: str,
    cfg: dict,
    views: list[dict],
    save_dir: Path | None = None,
) -> dict:
    """Render source + decimated with identical cameras, return per-view SSIMs."""
    dec_mesh = read_obj(dec_obj_path)
    imgs_src = _render_views_retrying(src_mesh, texture_path, tex_orientation, views)
    imgs_dec = _render_views_retrying(dec_mesh, texture_path, tex_orientation, views)

    full_min = float(cfg["ssim_production_min"])
    close_min = float(cfg["ssim_closeup_min"])
    results = []
    all_pass = True
    for view in views:
        s = _ssim_gray(imgs_src[view["name"]], imgs_dec[view["name"]])
        thr = close_min if view["closeup"] else full_min
        ok = s >= thr
        all_pass &= ok
        results.append({"name": view["name"], "closeup": view["closeup"], "ssim": round(s, 5), "threshold": thr, "pass": ok})
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for view in views:
            Image.fromarray(imgs_src[view["name"]]).save(save_dir / f"{view['name']}_src.png")
            Image.fromarray(imgs_dec[view["name"]]).save(save_dir / f"{view['name']}_dec.png")
    return {"views": results, "pass": bool(all_pass)}


# ----------------------------------------------------------------- ladder integration
def _apply_redecimation(rec: dict, frag: dict) -> None:
    """Fold a redecimate_at() fragment (attempt + outputs) into the mesh record."""
    att, out = frag["attempt"], frag["out"]
    rec["journey"] = rec.get("journey", []) + [att]
    rec.update(
        rung_chosen=att["keep"],
        extratcoordw=att["extratcoordw"],
        planarquadric=att.get("planarquadric", False),
        keep_achieved=att["keep_achieved"],
        faces_out=att["faces_out"],
        verts_out=att["verts_out"],
        hausdorff=att["hausdorff"],
        hausdorff_ratio=att["hausdorff_ratio"],
        dec_stats=att["dec_stats"],
        boundary_rel_diff=att["boundary_rel_diff"],
        geometry_pass=att["pass"],
        fail_reasons=att["fail_reasons"],
    )
    rec.update(out)
    for n in frag.get("notes", []):
        if n not in rec.setdefault("notes", []):
            rec["notes"].append(n)


_NO_SAFE_NOTE = (
    "no_safe_decimation: extended ladder exhausted — undecimated M1 conversion shipped "
    "as the optimized deliverable (byte-identical copy, SHA-verified)"
)


def _apply_no_safe_decimation(rec: dict, frag: dict) -> None:
    """Fold a ship_undecimated_copy() fragment into the mesh record. The undecimated
    M1 copy becomes the deliverable; the failed ladder attempts stay in the journeys
    as the decision evidence."""
    rec.update(frag)
    if _NO_SAFE_NOTE not in rec.setdefault("notes", []):
        rec["notes"].append(_NO_SAFE_NOTE)


def evaluate_with_ladder(rec: dict, task: dict, cfg: dict, tex_orientation: str, root: Path) -> None:
    """SSIM-gate `rec` at its chosen rung; on failure retry the rung with
    extratcoordw=3.0 (UV-smearing remedy), then climb the group's ladder (Group C gets
    the GROUP_C_EXTENDED_LADDER rungs) — geometry gates are re-run for every
    re-decimation and outputs are only rewritten when they pass, so rec and the on-disk
    OBJ never diverge. When a Group-C mesh exhausts even the extended ladder, the
    undecimated M1 conversion is shipped (decision='no_safe_decimation') and verified
    through the same SSIM harness. Mutates rec in place."""
    from scrollkit.decimate.core import ladder_for_group, redecimate_at, ship_undecimated_copy

    ladder = ladder_for_group(cfg, task["group"])
    src_mesh = read_ply(task["src"])
    tex_name = rec.get("texture")
    texture_path = Path(task["src"]).parent / tex_name if tex_name else None
    if texture_path is None:
        rec["ssim"] = {"views": [], "pass": False, "error": "no texture — SSIM undefined"}
        return
    views = build_views(src_mesh, texture_path, tex_orientation)
    save_root = root / "outputs/decimated/_ssim" / rec["stem"]

    if rec.get("decision") == "no_safe_decimation":
        # Already decided (idempotent re-run): re-check the identity deliverable only,
        # PRESERVING the recorded failed-ladder journey (the gate's decision evidence).
        prior = rec["ssim"]["journey"] if isinstance(rec.get("ssim"), dict) else []
        prior = [j for j in prior if not (j.get("rung") == 1.0 and "no_safe_decimation" in j.get("note", ""))]
        res = evaluate_ssim(src_mesh, Path(rec["obj"]), texture_path, tex_orientation, cfg, views, save_dir=save_root)
        rec["ssim"] = {
            "views": res["views"],
            "pass": bool(res["pass"]),
            "journey": prior + [{
                "rung": 1.0, "extratcoordw": None, "planarquadric": None,
                "pass": bool(res["pass"]),
                "note": "no_safe_decimation: identity deliverable re-check",
                "ssims": {v["name"]: v["ssim"] for v in res["views"]},
            }],
        }
        rec["ssim_renders_dir"] = str(save_root)
        return

    # Candidate sequence per rung: current outputs as-is; extratcoordw=3.0 (the
    # config's sanctioned UV-smearing remedy); planarquadric=True (VCG planar-fitting
    # quadric — measured to strongly reduce texture sliding on sheet-like wraps);
    # then the next rung up, same sub-sequence.
    rung0, w0 = float(rec["rung_chosen"]), float(rec["extratcoordw"])
    pq0 = bool(rec.get("planarquadric", False))
    base_w = float(cfg.get("extratcoordw", 1.0))
    # (rung, extratcoordw, planarquadric, needs_redecimate)
    candidates: list[tuple[float, float, bool, bool]] = [(rung0, w0, pq0, False)]
    if w0 != 3.0:
        candidates.append((rung0, 3.0, False, True))
    if not pq0:
        candidates.append((rung0, base_w, True, True))
    start = ladder.index(rung0) if rung0 in ladder else -1
    for r in ladder[start + 1 :]:
        candidates.append((r, base_w, False, True))
        if base_w != 3.0:
            candidates.append((r, 3.0, False, True))
        candidates.append((r, base_w, True, True))

    journey: list[dict] = []
    final: dict | None = None
    passed = False
    for rung, w, pq, needs_redecimate in candidates:
        if needs_redecimate:
            frag = redecimate_at(task, cfg, rung, w, planarquadric=pq)
            if frag["out"] is None:  # geometry gates failed — disk untouched, skip
                journey.append({"rung": rung, "extratcoordw": w, "planarquadric": pq,
                                "pass": False, "views": [],
                                "note": "geometry gates failed: " + "; ".join(frag["attempt"]["fail_reasons"])})
                continue
            _apply_redecimation(rec, frag)
        res = evaluate_ssim(
            src_mesh, Path(rec["obj"]), texture_path, tex_orientation, cfg, views, save_dir=save_root
        )
        entry = {"rung": rung, "extratcoordw": w, "planarquadric": pq, **res}
        journey.append(entry)
        final = entry
        if res["pass"] and rec.get("geometry_pass"):
            passed = True
            break

    if not passed and task["group"] == "C":
        # Extended C ladder exhausted (policy:
        # decimate aggressively ONLY when visuals are unaffected): no keep fraction is
        # perceptually safe, so the deliverable is the UNDECIMATED M1 conversion
        # (byte-identical copy, SHA-verified). Rendering the identity copy through the
        # same SSIM harness is a final deliverable wiring check (expected SSIM = 1.0;
        # a broken MTL/texture reference would tank it and keep the gate red).
        frag = ship_undecimated_copy(task, root)
        _apply_no_safe_decimation(rec, frag)
        res = evaluate_ssim(
            src_mesh, Path(rec["obj"]), texture_path, tex_orientation, cfg, views, save_dir=save_root
        )
        entry = {"rung": 1.0, "extratcoordw": None, "planarquadric": None,
                 "note": "no_safe_decimation: undecimated M1 copy shipped (identity deliverable check)",
                 **res}
        journey.append(entry)
        final = entry
        passed = bool(res["pass"])
    elif not passed:
        # Ladder exhausted: ship the BEST-margin geometry-green variant (not the last
        # tried), so the on-disk OBJ + record represent the closest-to-green result
        # for escalation review.
        evaluated = [j for j in journey if j.get("views")]
        if evaluated:
            def _margin(j: dict) -> float:
                return min(v["ssim"] - v["threshold"] for v in j["views"])

            best = max(evaluated, key=_margin)
            if best is not final:
                frag = redecimate_at(task, cfg, best["rung"], best["extratcoordw"],
                                     planarquadric=best.get("planarquadric", False))
                if frag["out"] is not None:
                    _apply_redecimation(rec, frag)
                    res = evaluate_ssim(src_mesh, Path(rec["obj"]), texture_path,
                                        tex_orientation, cfg, views, save_dir=save_root)
                    entry = {"rung": best["rung"], "extratcoordw": best["extratcoordw"],
                             "planarquadric": best.get("planarquadric", False),
                             "note": "best-margin variant re-shipped after exhausted ladder", **res}
                    journey.append(entry)
                    final = entry

    rec["ssim"] = {
        "views": final["views"] if final else [],
        "pass": bool(passed),
        "journey": [
            {"rung": j["rung"], "extratcoordw": j["extratcoordw"],
             "planarquadric": j.get("planarquadric", False), "pass": j["pass"],
             **({"note": j["note"]} if "note" in j else {}),
             "ssims": {v["name"]: v["ssim"] for v in j.get("views", [])}}
            for j in journey
        ],
    }
    rec["ssim_renders_dir"] = str(save_root)


__all__ = [
    "build_views",
    "evaluate_ssim",
    "evaluate_with_ladder",
    "normal_variation_scores",
    "texture_content_uv_centroid",
]
