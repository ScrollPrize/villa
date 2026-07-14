"""M4 cinematics: camera planning, chirality-correct screen orientation, studio
lighting, and vignette for the unwrap videos.

Art direction is bound by docs/RENDER-STYLE.md:
  - static per-mesh camera fit on the UNION of the whole trajectory + margin
    (nothing ever clips, verified by projecting every frame's points),
  - 16:9 composition with the unroll axis horizontal (sheet unrolls across the
    width), 9:16 with the unroll axis vertical,
  - gentle linear push-in (default 2.5%) expressed as a per-frame parallel_scale
    schedule (camera frozen during the reveal tail — exact image-space blends),
  - background #101114 + subtle radial vignette (post-multiply in numpy),
  - 3-point studio rig: warm key upper camera-left, cool fill ~35%, faint rim.
    M4 fix over the M3 contact sheets: the key is angled strongly enough along
    the view axis that the FLAT end state (face-on to the camera) reads clearly,
    not just the rolled start. Lights are world-static for the whole clip and are
    positioned relative to the union of start/end geometry.

Camera-'up' rule (chirality)
----------------------------
The flat end state is rendered face-on and must match the team's composite PNG
(the texture file composited on the render background) — mirrored or upside-down
Greek is fatal. The unwrap preserves UV<->3D orientation (M3 `mirror_check` > 0),
so the only remaining freedoms are which SIDE the camera looks from and the sign
of 'up'. We resolve both EMPIRICALLY, exactly like scripts/chirality_check.py but
decision-grade: render the flat frame face-on from the recto (face-winding
normal) and from the verso, each with up = +sheet-v; the up = -sheet-v candidates
are the exact rot180 of those images (parallel face-on projection), so four
hypotheses come from two renders. Each candidate is compared (downscaled SSIM)
against the PNG-composited-on-background reference cropped to the mesh's UV bbox
through the audit's tex_orientation pixel mapping. The argmax wins; all scores
are recorded in the returned plan.

Geometric cross-check (and why the verso can legitimately win): with positive UV
winding the recto view with up=+v shows the FILE transformed by the per-group
tex_orientation (group A 'rot180' -> flip_h(file), group B 'topleft' ->
flip_v(file), 'opengl_bottomleft' -> identity). reports/m3_chirality.json
measured exactly those flips, so for groups A/B the readable side is the verso.
The SSIM probe encodes no convention — it simply finds the screen orientation
that reproduces the composite.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .scene import SceneRenderer

Image.MAX_IMAGE_PIXELS = None

#: render background, bound by docs/RENDER-STYLE.md (#101114)
BACKGROUND_RGB = (16, 17, 20)
BACKGROUND_HEX = "#101114"

#: studio rig v2 (M4). v1 = scene.SceneRenderer._add_studio_lights (M3 contact
#: sheets: flat end state too dark — key cos vs the face-on sheet only ~0.66 and
#: the M3 camera was isometric). v2 keeps the key warm + upper camera-left but
#: angles it strongly along the view axis so the face-on flat sheet holds
#: midtones, fill stays cool at ~35% of key, rim faint from behind-above.
STUDIO_RIG = {
    "version": "m4-v2",
    "key": {"offset": (-0.55, 0.50, 1.45), "color": (1.00, 0.97, 0.92), "intensity": 0.92},
    "fill": {"offset": (1.15, 0.20, 0.95), "color": (0.84, 0.90, 1.00), "intensity": 0.33},
    "rim": {"offset": (0.12, 0.95, -1.40), "color": (1.00, 1.00, 1.00), "intensity": 0.45},
}

#: vignette: corners ~25% darker per the style spec; quadratic-ish falloff keeps the
#: center 80% of frame nearly untouched.
VIGNETTE = {"strength": 0.25, "power": 2.2}

__all__ = [
    "BACKGROUND_HEX",
    "BACKGROUND_RGB",
    "STUDIO_RIG",
    "VIGNETTE",
    "apply_vignette",
    "choose_camera_up",
    "plan_camera",
    "sheet_frame",
    "studio_lights",
    "vignette_mask",
]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length vector")
    return v / n


# --------------------------------------------------------------------------------------
# sheet frame + chirality probe
# --------------------------------------------------------------------------------------
def sheet_frame(flat: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> dict:
    """Orthonormal frame of the FLAT end state.

    n      : recto normal from face winding (sum of cross products — sign is the
             surface orientation, not an arbitrary PCA sign).
    v_hat  : in-plane world direction of increasing uv-v (UV-covariance fit,
             projected into the sheet plane). Camera-up candidates are ±v_hat.
    u_hat  : in-plane direction of increasing uv-u, orthonormalized against v_hat
             (the unroll axis on screen; metrics.json: unroll_axis_uv == 'u').
    """
    flat = np.asarray(flat, dtype=np.float64)
    c = flat.mean(axis=0)
    fc = flat - c
    e1 = flat[faces[:, 1]] - flat[faces[:, 0]]
    e2 = flat[faces[:, 2]] - flat[faces[:, 0]]
    n = _unit(np.cross(e1, e2).sum(axis=0))
    uvc = np.asarray(uv, dtype=np.float64) - np.asarray(uv, dtype=np.float64).mean(axis=0)
    du = (fc * uvc[:, :1]).sum(axis=0)
    dv = (fc * uvc[:, 1:]).sum(axis=0)
    v_hat = _unit(dv - (dv @ n) * n)
    u_raw = du - (du @ n) * n
    u_hat = _unit(u_raw - (u_raw @ v_hat) * v_hat)
    handed = float(np.cross(u_hat, v_hat) @ n)  # +1 for positive UV winding
    return {"center": c, "n": n, "v_hat": v_hat, "u_hat": u_hat, "handedness": handed}


def _reference_image(texture_path, tex_orientation: str, uv: np.ndarray, size_wh) -> np.ndarray:
    """The team's composite as the on-screen ground truth: PNG alpha-composited on
    the render background, cropped to the mesh's UV bbox through the audit's
    tex_orientation pixel mapping, grayscale, resized to the probe render size."""
    with Image.open(texture_path) as im:
        arr = np.asarray(im.convert("RGBA"), dtype=np.float64)
    rgb, a = arr[..., :3], arr[..., 3:] / 255.0
    comp = rgb * a + np.array(BACKGROUND_RGB, dtype=np.float64) * (1.0 - a)
    H, W = comp.shape[:2]

    s0, t0 = (float(x) for x in np.asarray(uv).min(axis=0))
    s1, t1 = (float(x) for x in np.asarray(uv).max(axis=0))
    if tex_orientation == "topleft":            # col = s*(W-1);     row = t*(H-1)
        cols, rows = (s0, s1), (t0, t1)
    elif tex_orientation == "opengl_bottomleft":  # col = s*(W-1);   row = (1-t)*(H-1)
        cols, rows = (s0, s1), (1.0 - t1, 1.0 - t0)
    elif tex_orientation == "rot180":           # col = (1-s)*(W-1); row = (1-t)*(H-1)
        cols, rows = (1.0 - s1, 1.0 - s0), (1.0 - t1, 1.0 - t0)
    else:
        raise ValueError(f"unsupported tex_orientation {tex_orientation!r}")
    c0, c1 = int(np.floor(cols[0] * (W - 1))), int(np.ceil(cols[1] * (W - 1))) + 1
    r0, r1 = int(np.floor(rows[0] * (H - 1))), int(np.ceil(rows[1] * (H - 1))) + 1
    crop = comp[max(r0, 0):min(r1, H), max(c0, 0):min(c1, W)]

    gray = Image.fromarray(crop.astype(np.uint8)).convert("L")
    return np.asarray(gray.resize(size_wh, Image.LANCZOS), dtype=np.float64)


def inner_side_sign(rolled: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> dict:
    """Which side of the sheet faced the umbilicus in the ROLLED state.

    Design rule: the ink lives on the INNER face of the winding (the
    papyrological recto of a rolled scroll); the unwrap must expose it. Geometric rule:
    in the rolled state, the winding axis is the (area-weighted) mean v-tangent direction;
    a face's radial-out vector is its centroid minus its projection on the axis line.
    inner_sign = −sign(median(n̂·radial_out)): +1 means the face-winding normal itself
    points inward (camera goes on the +winding-normal side of the FLAT sheet), −1 the
    opposite. The sign transports to the flat sheet because the morph never flips
    orientation (mirror_check > 0 is gated).
    """
    V = np.asarray(rolled, dtype=np.float64)
    F = np.asarray(faces)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    n = np.cross(e1, e2)
    area = np.linalg.norm(n, axis=1)
    ok = area > 1e-300
    n_hat = np.where(ok[:, None], n / np.maximum(area, 1e-300)[:, None], 0.0)
    # winding axis from uv v-tangents (same construction as the embedding's axis fit)
    duv1 = np.asarray(uv)[F[:, 1]] - np.asarray(uv)[F[:, 0]]
    duv2 = np.asarray(uv)[F[:, 2]] - np.asarray(uv)[F[:, 0]]
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    safe = np.where(np.abs(det) < 1e-18, 1.0, det)
    t_v = (-duv2[:, 0, None] * e1 + duv1[:, 0, None] * e2) / safe[:, None]
    tvn = np.linalg.norm(t_v, axis=1)
    good = (tvn > 1e-12) & ok & (np.abs(det) > 1e-18)
    axis = _unit(((t_v[good] / tvn[good][:, None]) * area[good][:, None]).sum(axis=0))
    cent = V[F].mean(axis=1)
    c0 = V.mean(axis=0)
    rel = cent - c0
    radial_out = rel - (rel @ axis)[:, None] * axis[None, :]
    dot = np.einsum("ij,ij->i", n_hat, radial_out)
    w = area * good
    med = float(np.sign((np.sign(dot) * w).sum()))  # area-weighted majority of sign
    inner = -med if med != 0 else 1.0
    frac_inward = float((np.sign(dot)[w > 0] < 0).mean())
    return {"inner_sign": inner, "frac_normals_inward": frac_inward, "axis": axis.tolist()}


def choose_camera_up(
    flat: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    texture_path,
    tex_orientation: str,
    *,
    probe_px: int = 768,
    rolled: np.ndarray | None = None,
) -> dict:
    """Side + up selection.

    Side: GEOMETRIC when `rolled` is given (design rule: expose the inner face of the
    winding — see inner_side_sign); the PNG-similarity scores remain as a checksum and
    any disagreement is recorded loudly. Up: argmax SSIM against the composite reference
    among ±v_hat on the chosen side (upright, un-mirrored text)."""
    from skimage.metrics import structural_similarity

    fr = sheet_frame(flat, faces, uv)
    c, n, v_hat, u_hat = fr["center"], fr["n"], fr["v_hat"], fr["u_hat"]
    fc = np.asarray(flat, dtype=np.float64) - c
    hh = float(np.abs(fc @ v_hat).max())
    hw = float(np.abs(fc @ u_hat).max())
    Hpx = int(probe_px)
    Wpx = int(np.clip(round(probe_px * hw / hh), 64, 2 * probe_px))
    dist = 3.0 * float(np.linalg.norm(fc, axis=1).max())

    ref = _reference_image(texture_path, tex_orientation, uv, (Wpx, Hpx))

    shots = {}
    sr = SceneRenderer(flat, faces, uv, texture_path, tex_orientation,
                       size=(Wpx, Hpx), background=BACKGROUND_HEX, lighting="flat")
    try:
        for side_name, side in (("recto", 1.0), ("verso", -1.0)):
            sr.set_camera({
                "position": tuple(c + side * n * dist),
                "focal_point": tuple(c),
                "up": tuple(v_hat),
                "parallel": True,
                "parallel_scale": hh * 1.02,
            })
            img = sr.screenshot()
            shots[side_name] = np.asarray(
                Image.fromarray(img).convert("L"), dtype=np.float64
            )
    finally:
        sr.close()

    candidates = {
        ("recto", +1): shots["recto"],
        ("recto", -1): shots["recto"][::-1, ::-1],   # up=-v_hat == screen rot180
        ("verso", +1): shots["verso"],
        ("verso", -1): shots["verso"][::-1, ::-1],
    }
    scores = {
        f"{side}_up{'+' if sgn > 0 else '-'}":
            float(structural_similarity(np.ascontiguousarray(img), ref, data_range=255.0))
        for (side, sgn), img in candidates.items()
    }
    png_best_key = max(scores, key=scores.get)
    runner_up = max(v for k, v in scores.items() if k != png_best_key)

    inner = None
    if rolled is not None:
        inner = inner_side_sign(rolled, faces, uv)
        # 'recto' here = +face-winding-normal side; the inner face is on that side iff
        # inner_sign > 0. The geometric rule decides the SIDE; SSIM decides the UP.
        side_name = "recto" if inner["inner_sign"] > 0 else "verso"
        side_scores = {k: v for k, v in scores.items() if k.startswith(side_name)}
        best_key = max(side_scores, key=side_scores.get)
    else:
        best_key = png_best_key

    side_name, sgn = best_key.rsplit("_up", 1)
    out = {
        "side": side_name,                      # 'recto' (+face-winding normal) | 'verso'
        "side_sign": 1.0 if side_name == "recto" else -1.0,
        "up_sign": 1.0 if sgn == "+" else -1.0,
        "scores": scores,
        "best": best_key,
        "png_best": png_best_key,
        "ssim_best": scores[best_key],
        "ssim_margin": scores[best_key] - runner_up,
        "tex_orientation": tex_orientation,
        "probe_size": [Wpx, Hpx],
        "handedness": fr["handedness"],
        "side_rule": "geometric_inner" if rolled is not None else "png_similarity",
    }
    if inner is not None:
        out["inner_side"] = inner
        out["png_side_agrees"] = png_best_key.startswith(side_name)
    return out


def tight_flat_camera(flatV, faces, uv, plan: dict, margin_frac: float = 0.10) -> dict:
    """Camera framing the FLAT sheet only (reveal push-in target): same view direction
    and up as the production plan, parallel scale fit to the sheet's extent + margin."""
    V = np.asarray(flatV, dtype=np.float64)
    pos = np.asarray(plan["position"], dtype=np.float64)
    foc = np.asarray(plan["focal_point"], dtype=np.float64)
    up = _unit(np.asarray(plan["up"], dtype=np.float64))
    dop = _unit(foc - pos)
    right = _unit(np.cross(dop, up))
    true_up = _unit(np.cross(right, dop))
    c = V.mean(axis=0)
    rel = V - c
    half_w = float(np.abs(rel @ right).max())
    half_h = float(np.abs(rel @ true_up).max())
    aspect = plan.get("aspect")
    if not aspect:
        aspect = half_w / max(half_h, 1e-9)
    scale = max(half_h, half_w / float(aspect)) * (1.0 + margin_frac)
    dist = float(np.linalg.norm(np.asarray(plan["position"]) - np.asarray(plan["focal_point"])))
    return {
        "position": tuple(c - dop * dist),
        "focal_point": tuple(c),
        "up": tuple(up),
        "parallel": True,
        "parallel_scale": scale,
    }


def lerp_camera(a: dict, b: dict, e: float) -> dict:
    """Linear camera interpolation (position/focal/parallel_scale; up held from a)."""
    pa, pb = np.asarray(a["position"], float), np.asarray(b["position"], float)
    fa, fb = np.asarray(a["focal_point"], float), np.asarray(b["focal_point"], float)
    sa, sb = float(a["parallel_scale"]), float(b["parallel_scale"])
    return {
        "position": tuple(pa * (1 - e) + pb * e),
        "focal_point": tuple(fa * (1 - e) + fb * e),
        "up": tuple(a["up"]),
        "parallel": True,
        "parallel_scale": sa * (1 - e) + sb * e,
    }


# --------------------------------------------------------------------------------------
# camera plan
# --------------------------------------------------------------------------------------
def plan_camera(
    frames: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    aspect: float,
    group: str,
    *,
    texture_path=None,
    tex_orientation: str | None = None,
    up_choice: dict | None = None,
    margin_frac: float = 0.06,
    push_in_frac: float = 0.025,
) -> dict:
    """Static per-mesh camera plan for one framing (aspect = width/height).

    View direction: face-on to the flat sheet from the chirality-chosen side.
    Up: aspect > 1 -> the reading orientation itself (unroll axis horizontal,
    text exactly as in the team's composite); aspect < 1 -> the reading frame
    rotated 90° clockwise on screen (unroll axis vertical per the style spec; the
    deterministic CW sign mounts the text top-to-bottom).

    Fit: every frame's points are projected on the screen basis about the
    union-bbox center; the scale schedule (linear push-in of `push_in_frac`
    over the clip) is lifted so that EVERY frame keeps >= margin_frac margin —
    nothing ever clips, including at maximum push-in. Per-frame margins are
    verified and recorded.
    """
    frames = np.asarray(frames)
    if up_choice is None:
        if texture_path is None or tex_orientation is None:
            raise ValueError("plan_camera: need texture_path+tex_orientation or an up_choice dict")
        up_choice = choose_camera_up(frames[-1], faces, uv, texture_path, tex_orientation,
                                     rolled=frames[0])

    fr = sheet_frame(frames[-1], faces, uv)
    d = up_choice["side_sign"] * fr["n"]            # focal -> camera
    reading_up = up_choice["up_sign"] * fr["v_hat"]
    dop = -d                                        # direction of projection
    reading_right = _unit(np.cross(dop, reading_up))
    # Design rule: text ROWS stay horizontal in EVERY framing — the
    # earlier 90°-rotated vertical composition was rejected ("text rotated, not
    # horizontal"). 9:16 fits the wide sheet by pulling back instead.
    up = reading_up
    text_rotation_deg = 0
    right = _unit(np.cross(dop, up))
    true_up = _unit(np.cross(right, dop))

    # union bbox center = focal point (whole trajectory)
    lo = frames.min(axis=(0, 1)).astype(np.float64)
    hi = frames.max(axis=(0, 1)).astype(np.float64)
    focal = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo))

    # per-frame required parallel_scale (half-height units), about the focal point
    F = frames.shape[0]
    req = np.empty(F, dtype=np.float64)
    for f in range(F):
        rel = frames[f].astype(np.float64) - focal
        hw = float(np.abs(rel @ right).max())
        hh = float(np.abs(rel @ true_up).max())
        req[f] = max(hh, hw / float(aspect))

    push = 1.0 - float(push_in_frac) * np.linspace(0.0, 1.0, F)
    s0 = float((req * (1.0 + float(margin_frac)) / push).max())
    per_frame_scale = s0 * push
    margins = per_frame_scale / req - 1.0
    min_margin = float(margins.min())
    if min_margin < float(margin_frac) - 1e-9:      # by construction this cannot happen
        raise AssertionError(f"camera fit violates margin: {min_margin:.4f} < {margin_frac}")

    position = focal + d * (3.0 * diag)
    return {
        "position": [float(x) for x in position],
        "focal_point": [float(x) for x in focal],
        "up": [float(x) for x in up],
        "parallel": True,
        "parallel_scale": float(s0),
        "per_frame_scale": [float(x) for x in per_frame_scale],
        "aspect": float(aspect),
        "group": group,
        "margin_frac": float(margin_frac),
        "push_in_frac": float(push_in_frac),
        "min_margin": min_margin,
        "max_required_scale": float(req.max()),
        "binding_frame": int(np.argmax(req * (1.0 + margin_frac) / push)),
        "text_rotation_deg": text_rotation_deg,
        "screen_right": [float(x) for x in right],
        "screen_up": [float(x) for x in true_up],
        "view_dir_world": [float(x) for x in d],
        "up_choice": up_choice,
        "bounds_lo": [float(x) for x in lo],
        "bounds_hi": [float(x) for x in hi],
    }


# --------------------------------------------------------------------------------------
# studio lighting (M4 rig)
# --------------------------------------------------------------------------------------
def studio_lights(scene_or_plotter, embedding_like_info: dict, rig: dict | None = None) -> dict:
    """Install the M4 3-point rig (replaces any existing lights).

    scene_or_plotter   : SceneRenderer or pv.Plotter.
    embedding_like_info: {'camera': camera/plan dict, 'bounds_lo', 'bounds_hi',
                         optional 'texture_stats': {'p50': .., 'p95': ..} in 0-1}
                         — bounds are the UNION of the trajectory (start+end),
                         so the rig is positioned for both the initial roll and
                         the final tangent-plane sheet.
    Lights are directional (world-static, no per-frame animation). Offsets are
    expressed in the camera basis (right, up, view) at install time and then
    FROZEN in world space. Returns a record of the installed rig including the
    predicted lambert response of the face-on flat sheet (the washed-out-sheet failure mode).

    Exposure adaptation: the style spec pins ON-SCREEN papyrus midtones at 0.65-0.75,
    but composite brightness varies per group (march median ~0.62, may ~0.51).
    When texture_stats are provided (ALWAYS from the plain composite, also for
    the ink variant, so the reveal crossfade blends identically-lit renders),
    key+fill are scaled so the face-on flat sheet's midtone lands at ~0.70,
    capped so its p95 stays below white (no blown highlights); the rim is left
    untouched (silhouette only).
    """
    import pyvista as pv

    plotter = getattr(scene_or_plotter, "plotter", scene_or_plotter)
    rig = rig or STUDIO_RIG
    cam = embedding_like_info["camera"]
    lo = np.asarray(embedding_like_info["bounds_lo"], dtype=np.float64)
    hi = np.asarray(embedding_like_info["bounds_hi"], dtype=np.float64)
    center = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo)) or 1.0

    d = _unit(np.asarray(cam["position"], dtype=np.float64)
              - np.asarray(cam["focal_point"], dtype=np.float64))   # focal -> camera
    up = np.asarray(cam["up"], dtype=np.float64)
    right = _unit(np.cross(-d, up))
    true_up = _unit(np.cross(right, -d))

    plotter.remove_all_lights()
    renderer = plotter.renderer
    renderer.SetTwoSidedLighting(True)  # flat sheet must read whichever side faces the camera

    # exposure adaptation from the plain composite's statistics (see docstring);
    # ambient/diffuse constants mirror scene.py's studio material (0.18 / 0.92)
    AMBIENT, DIFFUSE, MID_TARGET, P95_CAP = 0.18, 0.92, 0.70, 0.97
    dirs, coss = {}, {}
    for name in ("key", "fill", "rim"):
        off = np.asarray(rig[name]["offset"], dtype=np.float64)
        dirs[name] = _unit(off[0] * right + off[1] * true_up + off[2] * d)
        coss[name] = float(dirs[name] @ d)
    exposure_scale = 1.0
    stats = embedding_like_info.get("texture_stats")
    if stats:
        base_diffuse = sum(rig[n]["intensity"] * max(0.0, coss[n]) for n in ("key", "fill"))
        mult_target = MID_TARGET / max(float(stats["p50"]), 1e-3)
        mult_cap = P95_CAP / max(float(stats["p95"]), 1e-3)
        mult = float(np.clip(min(mult_target, mult_cap), 1.0, 1.65))
        exposure_scale = float(np.clip((mult - AMBIENT) / (DIFFUSE * base_diffuse), 0.8, 1.8))

    record = {"rig_version": rig.get("version", "?"), "exposure_scale": exposure_scale,
              "texture_stats": stats, "lights": {}}
    for name in ("key", "fill", "rim"):
        spec = rig[name]
        intensity = spec["intensity"] * (exposure_scale if name in ("key", "fill") else 1.0)
        pos = center + dirs[name] * (2.5 * diag)
        plotter.add_light(pv.Light(position=tuple(pos), focal_point=tuple(center),
                                   color=spec["color"], intensity=intensity,
                                   positional=False))
        record["lights"][name] = {
            "direction_world": [float(x) for x in dirs[name]],
            "color": list(spec["color"]),
            "intensity": round(float(intensity), 4),
            "cos_vs_view_axis": coss[name],  # response of the face-on flat sheet
        }
    # lambert sanity for the flat end state: must hold papyrus midtones ~0.65-0.75
    flat_mult = AMBIENT + DIFFUSE * sum(
        r["intensity"] * max(0.0, r["cos_vs_view_axis"]) for r in record["lights"].values()
    )
    record["flat_sheet_texture_multiplier"] = float(flat_mult)
    if stats:
        record["predicted_flat_midtone"] = round(float(stats["p50"]) * flat_mult, 3)
        record["predicted_flat_p95"] = round(float(stats["p95"]) * flat_mult, 3)
    return record


# --------------------------------------------------------------------------------------
# vignette (post, numpy, before encode)
# --------------------------------------------------------------------------------------
def vignette_mask(width: int, height: int, strength: float | None = None,
                  power: float | None = None) -> np.ndarray:
    """Radial multiply mask as Q8.8 uint16 (256 == 1.0): 1.0 at center, corners
    darkened by `strength` (style spec: ~25%). Apply with apply_vignette()."""
    strength = VIGNETTE["strength"] if strength is None else float(strength)
    power = VIGNETTE["power"] if power is None else float(power)
    ny, nx = np.meshgrid(np.linspace(-1.0, 1.0, height), np.linspace(-1.0, 1.0, width),
                         indexing="ij")
    r = np.sqrt((nx * nx + ny * ny) / 2.0)          # 1.0 exactly at the corners
    scale = 1.0 - strength * np.power(r, power)
    return np.clip(np.round(scale * 256.0), 0, 256).astype(np.uint16)


def apply_vignette(frame_u8: np.ndarray, mask_q8: np.ndarray) -> np.ndarray:
    """Multiply an (H, W, 3) uint8 frame by the Q8.8 mask (exact integer math)."""
    if frame_u8.shape[:2] != mask_q8.shape:
        raise ValueError(f"vignette mask {mask_q8.shape} != frame {frame_u8.shape[:2]}")
    out = (frame_u8.astype(np.uint16) * mask_q8[..., None]) >> 8
    return out.astype(np.uint8)


WATERMARK_TEXT = "© Vesuvius Challenge 2026"


def watermark_overlay(width: int, height: int, text: str = WATERMARK_TEXT,
                      opacity: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    """Pre-rendered subtle watermark for the bottom-right corner (empty background
    in every framing — the subject is center-weighted with generous margins).

    Returns (rgb float32 premultiplied contribution, alpha float32) at frame size;
    composite with apply_watermark(). Soft warm-gray text, ~H/42 tall, antialiased
    at 4x supersampling; margins ~2.2% of the frame height."""
    from PIL import ImageDraw, ImageFont

    ss = 4
    fs = max(12, int(round(height / 42))) * ss
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fs)
    except OSError:
        font = ImageFont.load_default()
    pad_y = int(round(0.022 * height)) * ss
    # right inset scales with WIDTH (a height-based inset is
    # only 1.3% of a 16:9 width — cramped against rounded-corner/UI safe zones)
    pad_x = int(round(0.022 * width)) * ss
    canvas = Image.new("L", (width * ss, height * ss), 0)
    d = ImageDraw.Draw(canvas)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = width * ss - pad_x - tw - bbox[0]
    y = height * ss - pad_y - th - bbox[1]
    d.text((x, y), text, fill=255, font=font)
    a = np.asarray(canvas.resize((width, height), Image.LANCZOS),
                   dtype=np.float32) / 255.0
    a *= opacity
    color = np.array([214.0, 212.0, 206.0], np.float32)   # warm paper-gray
    rgb = a[..., None] * color[None, None, :]
    return rgb, a


def apply_watermark(frame_u8: np.ndarray, wm: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Alpha-composite the pre-rendered watermark over an (H, W, 3) uint8 frame."""
    rgb, a = wm
    out = frame_u8.astype(np.float32) * (1.0 - a[..., None]) + rgb
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
