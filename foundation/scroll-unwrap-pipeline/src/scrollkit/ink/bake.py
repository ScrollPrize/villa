"""Ink overlay bake — M2.5.

Binding spec: docs/RENDER-STYLE.md ('Ink overlay bake') + configs/global.yaml
[ink].  Gate: docs/QUALITY-GATES.md (M2.5).

Per matched wrap (19 March TIF + 11 May JPG, from reports/audit.json: ink):

  base   composite RGBA texture next to the PLY. The alpha channel is carried
         through BYTE-IDENTICAL (numpy slice copy; the gate re-verifies with
         np.array_equal against the saved PNG).
  ink    matched prediction resampled to EXACT base texture dims.
         March (huge deflate TIFs, 4-13x texture scale): integer-factor block
         mean (numpy reshape trick) down to <~2x target, then LANCZOS to exact
         dims. May (1.3-2.4x): single LANCZOS on the grayscale image.
         The same-pixel-grid assumption (no flip vs the texture;
         tex_orientation applies to texture<->UV sampling at RENDER time, not
         here) is sanity-checked per mesh (registration_check). Empirical
         outcome on this dataset: March = identity (violations of the 3%
         outside-mask budget are sheet-edge halos, quantified per mesh); May =
         flipud, decisively (the predictions are pixel-aligned to the
         'max comp 5' composites, which are stored vertically flipped relative
         to the PLY textures — content correlation ~0.9 at zero shift for
         flipud vs ~0.3-0.4 for identity). Both are reported loudly as
         anomalies with evidence.
  pol    normalized to [0,1] with ink=high. The audit polarity label is
         verified per image with the sparse-tail heuristic (smoothstep-active
         ink fraction within the papyrus mask must land in [0.5%, 35%]); the
         label is flipped (and recorded as an anomaly) when the data say so.
         Empirical outcome: all 19 March TIFs are labelled 'positive' in the
         audit but are ink-DARK rasters; they are inverted like the May JPGs.
  grade  opacity = smoothstep(ink, lo, hi) * scale; amber tint #E8B84B;
         out = base*(1-op) + tint*op  (float32, round-half-even to uint8);
         glow G = gaussian(op, sigma) * strength, out = screen(out, tint*G).
         Papyrus fibers stay visible through ink midtones by construction.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # ink predictions are up to ~270 Mpx — trusted local data

# Mirrors configs/global.yaml [ink]; load_params() merges the YAML over this.
DEFAULT_PARAMS: dict = {
    "tint_rgb": [232, 184, 75],  # amber gold #E8B84B
    "opacity_lo": 0.35,
    "opacity_hi": 0.75,
    "opacity_scale": 0.85,
    "glow_sigma_px": 3.0,
    "glow_strength": 0.30,
}

# sparse-tail heuristic band for the in-mask INK FRACTION.  Ink fraction =
# visibly-inked pixels: normalized ink > the smoothstep midpoint (lo+hi)/2,
# i.e. opacity > half its maximum.  (The smoothstep-ONSET fraction (> lo) is
# recorded as a secondary stat; it counts barely-tinted midtone onset and runs
# ~2.5-4x higher on the diffuse March predictions.)
COVERAGE_BAND = (0.005, 0.35)
# registration: fraction of ink-positive px (> 0.3*max) allowed outside alpha>127
REG_OUTSIDE_MAX = 0.03
REG_THR_FRAC = 0.30
# decisive flip: best-variant score must beat the runner-up by >5% (relative)
REG_DECISIVE_REL_MARGIN = 1.05
# may cross-check: the prediction is pixel-aligned to this composite of the SAME wrap
MC5_DIR = "ink det + max comp (may 5th)/max comp 5"


# --------------------------------------------------------------------------- config


def load_params(root: Path) -> dict:
    """ink block of configs/global.yaml merged over DEFAULT_PARAMS."""
    import yaml

    cfg = yaml.safe_load((Path(root) / "configs/global.yaml").read_text())
    p = dict(DEFAULT_PARAMS)
    p.update(cfg.get("ink") or {})
    return p


def load_audit(root: Path) -> dict:
    return json.loads((Path(root) / "reports/audit.json").read_text())


def build_jobs(audit: dict, root: Path) -> list[dict]:
    """One job per matched wrap (19 march + 11 may), from the audit ink tables."""
    root = Path(root)
    mesh_by_path = {m["path"]: m for m in audit["meshes"]}
    jobs: list[dict] = []
    for kind, key, ink_key in (("march", "march_matches", "tif"), ("may", "may_matches", "jpg")):
        for match in audit["ink"][key]:
            mesh = mesh_by_path[match["matched_mesh"]]
            textures = [t for t in mesh["textures"] if t.get("exists")]
            if len(textures) != 1:
                raise ValueError(f"{mesh['path']}: expected exactly 1 texture, got {len(textures)}")
            tex = textures[0]
            jobs.append(
                {
                    "kind": kind,
                    "group": mesh["group"],
                    "stem": mesh["stem"],
                    "mesh_path": mesh["path"],
                    "tex_path": str(Path(mesh["path"]).parent / tex["name"]),
                    "tex_wh": [tex["width"], tex["height"]],
                    "alpha_frac_255": tex["alpha_frac_255"],
                    "ink_path": match[ink_key],
                    "ink_size": match["file_size"],
                    "polarity_label": match["polarity"],
                    "out_path": str(Path("outputs/overlays") / kind / f"{mesh['stem']}_inkoverlay.png"),
                }
            )
    return jobs


# --------------------------------------------------------------------------- resample


def block_mean(arr: np.ndarray, f: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Integer-factor block mean via the numpy reshape trick.

    Trailing rows/cols not divisible by f are cropped (recorded by the caller;
    sub-pixel at these scales: <= f-1 px of >30k px).  Returns float32.
    """
    if f <= 1:
        return arr.astype(np.float32), (0, 0)
    h, w = arr.shape
    h2, w2 = (h // f) * f, (w // f) * f
    out = arr[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3), dtype=np.float64)
    return out.astype(np.float32), (h - h2, w - w2)


def resample_ink(ink: np.ndarray, tex_wh: tuple[int, int], two_step: bool) -> tuple[np.ndarray, list[dict]]:
    """Resample ink raster to exact texture dims. Returns float32 [0,255] + step log.

    two_step (March full-res TIFs): block mean down to <~2x target, then LANCZOS.
    single step (May JPGs, <=2.4x): LANCZOS directly.
    """
    tw, th = int(tex_wh[0]), int(tex_wh[1])
    h, w = ink.shape
    steps: list[dict] = []
    if two_step:
        f = max(1, int(np.ceil(max(h / th, w / tw) / 2.0)))
        if f > 1:
            ink_f, crop = block_mean(ink, f)
            steps.append(
                {
                    "op": "block_mean",
                    "factor": f,
                    "in_hw": [h, w],
                    "out_hw": list(ink_f.shape),
                    "cropped_px_hw": list(crop),
                }
            )
        else:
            ink_f = ink.astype(np.float32)
    else:
        ink_f = ink.astype(np.float32)
    img = Image.fromarray(ink_f, mode="F")
    del ink_f
    img = img.resize((tw, th), Image.Resampling.LANCZOS)
    steps.append({"op": "lanczos", "out_wh": [tw, th]})
    out = np.clip(np.asarray(img, dtype=np.float32), 0.0, 255.0)  # clip LANCZOS over/undershoot
    del img
    return out, steps


# --------------------------------------------------------------------------- polarity


def normalize_ink(ink255: np.ndarray, inverted: bool) -> np.ndarray:
    """uint8-range raster -> [0,1] float32 with ink=high."""
    n = ink255.astype(np.float32) / 255.0
    return (1.0 - n) if inverted else n


def _band_distance(c: float, band: tuple[float, float] = COVERAGE_BAND) -> float:
    """0 inside the band, else log-distance to the nearest edge."""
    lo, hi = band
    eps = 1e-9
    if c < lo:
        return float(np.log((lo + eps) / (c + eps)))
    if c > hi:
        return float(np.log((c + eps) / (hi + eps)))
    return 0.0


def choose_polarity(
    ink255: np.ndarray, mask: np.ndarray, polarity_label: str, thr: float
) -> tuple[np.ndarray, dict]:
    """Verify the audit polarity label with the sparse-tail heuristic.

    coverage(candidate) = in-mask ink fraction: pixels with normalized ink >
    thr (the smoothstep midpoint -> visibly inked).  Must land in
    COVERAGE_BAND.  The audit label wins if in band; otherwise the flipped
    candidate wins if in band; otherwise the candidate closest to the band
    (log distance) wins and the result is flagged out-of-band.
    """
    label_inverted = polarity_label.strip().lower().startswith("inv")
    cov = {}
    for inv in (False, True):
        n = normalize_ink(ink255, inv)
        cov[inv] = float((n[mask] > thr).mean()) if mask.any() else 0.0
        del n
    in_band = {inv: COVERAGE_BAND[0] <= cov[inv] <= COVERAGE_BAND[1] for inv in (False, True)}
    if in_band[label_inverted]:
        use_inv = label_inverted
    elif in_band[not label_inverted]:
        use_inv = not label_inverted
    else:  # neither in band: closest to band (handles ultra-sparse damaged wraps)
        use_inv = min((False, True), key=lambda inv: _band_distance(cov[inv]))
    info = {
        "audit_label": polarity_label,
        "used": "inverted (ink dark)" if use_inv else "positive (ink bright)",
        "flipped_vs_audit": bool(use_inv != label_inverted),
        "coverage_definition": f"in-mask fraction with normalized ink > {thr} (smoothstep midpoint)",
        "coverage_as_audit_label": cov[label_inverted],
        "coverage_flipped": cov[not label_inverted],
        "coverage_used": cov[use_inv],
        "in_band": bool(in_band[use_inv]),
    }
    return normalize_ink(ink255, use_inv), info


# --------------------------------------------------------------------------- registration


def _reg_metrics(S: np.ndarray, mask: np.ndarray) -> dict:
    n_ink = int(S.sum())
    inter = int(np.logical_and(S, mask).sum())
    union = int(np.logical_or(S, mask).sum())
    return {
        "n_ink_px": n_ink,
        "outside_frac": (n_ink - inter) / n_ink if n_ink else 0.0,
        "iou": inter / union if union else 0.0,
    }


def _apply_variant(a: np.ndarray, name: str) -> np.ndarray:
    if name == "fliplr":
        return a[:, ::-1]
    if name == "flipud":
        return a[::-1, :]
    if name == "rot180":
        return a[::-1, ::-1]
    return a


def _content_correlation(tex_gray_ds: np.ndarray, aux_gray_ds: np.ndarray) -> dict[str, float]:
    """Zero-shift normalized correlation of texture content vs an aux render of the
    SAME wrap, per flip variant.  Far stronger registration signal than sparse-ink
    IoU when the aux canvas covers more sheet than the texture alpha."""
    A = tex_gray_ds - tex_gray_ds.mean()
    na = float(np.sqrt((A * A).sum())) or 1.0
    out = {}
    for name in ("identity", "fliplr", "flipud", "rot180"):
        V = _apply_variant(aux_gray_ds, name)
        V = V - V.mean()
        nv = float(np.sqrt((V * V).sum())) or 1.0
        out[name] = float((A * V).sum() / (na * nv))
    return out


def _halo_analysis(S: np.ndarray, mask: np.ndarray, ds: int = 4) -> dict:
    """How close outside-mask ink pixels sit to the mask boundary (edge-halo test)."""
    from scipy.ndimage import distance_transform_edt

    md = mask[::ds, ::ds]
    dist = distance_transform_edt(~md) * ds
    dvals = dist[(S & ~mask)[::ds, ::ds]]
    if dvals.size == 0:
        return {"n_outside_sampled": 0, "frac_within_16px": 1.0, "frac_within_48px": 1.0}
    return {
        "n_outside_sampled": int(dvals.size),
        "frac_within_16px": float((dvals <= 16).mean()),
        "frac_within_48px": float((dvals <= 48).mean()),
        "median_dist_px": float(np.median(dvals)),
    }


def registration_check(
    ink_norm: np.ndarray, mask: np.ndarray, content_pair: tuple[np.ndarray, np.ndarray] | None = None
) -> tuple[np.ndarray, dict]:
    """Sanity-check the same-pixel-grid assumption between ink and base texture.

    S = ink > 0.3*max.  Identity must keep coverage_outside_mask < 3%.  On
    violation the 3 flip variants are tested for THIS mesh; the best-IoU
    variant is used only when decisive (>5% relative margin over the runner-up,
    corroborated by the content cross-check when available); either way the
    event is reported loudly as an anomaly with full evidence.  When identity
    is kept despite the violation, a boundary-halo analysis quantifies whether
    the outside ink is just sheet-edge bleed (prediction silhouette slightly
    fatter than the texture's alpha cutout).
    """
    mx = float(ink_norm.max())
    thr = REG_THR_FRAC * mx
    S = ink_norm > thr
    ident = _reg_metrics(S, mask)
    out = {
        "threshold": thr,
        "threshold_frac_of_max": REG_THR_FRAC,
        "ink_norm_max": mx,
        "identity": ident,
        "outside_frac": ident["outside_frac"],
        "iou": ident["iou"],
        "variant_used": "identity",
        "anomaly": None,
    }
    if ident["outside_frac"] < REG_OUTSIDE_MAX or ident["n_ink_px"] == 0:
        return ink_norm, out

    # violation: test flip variants for this mesh only
    scores = {"identity": ident}
    for name in ("fliplr", "flipud", "rot180"):
        scores[name] = _reg_metrics(_apply_variant(ink_norm, name) > thr, mask)
    ranked = sorted(scores.items(), key=lambda kv: kv[1]["iou"], reverse=True)
    best, second = ranked[0], ranked[1]
    iou_rel = best[1]["iou"] / max(second[1]["iou"], 1e-9)

    content = None
    if content_pair is not None:
        corr = _content_correlation(*content_pair)
        cr = sorted(corr.items(), key=lambda kv: -kv[1])
        content = {
            "method": "zero-shift normalized correlation, texture gray (alpha-weighted) vs "
                      "'max comp 5' composite of the same wrap (prediction canvas), 8x downsampled",
            "scores": corr,
            "best": cr[0][0],
            "rel_margin": cr[0][1] / max(cr[1][1], 1e-9),
        }

    decisive = False
    if best[0] != "identity":
        iou_decisive = iou_rel >= REG_DECISIVE_REL_MARGIN and (content is None or content["best"] == best[0])
        content_decisive = (
            content is not None
            and content["best"] == best[0]
            and content["rel_margin"] >= REG_DECISIVE_REL_MARGIN
        )
        decisive = bool(iou_decisive or content_decisive)
    use = best[0] if (decisive and best[0] != "identity") else "identity"
    out["variant_used"] = use
    out["anomaly"] = {
        "kind": "registration_outside_mask",
        "detail": (
            f"identity coverage_outside_mask {ident['outside_frac']:.4f} >= {REG_OUTSIDE_MAX}; "
            f"flip variants tested: best={best[0]} (IoU {best[1]['iou']:.4f}) vs runner-up "
            f"{second[0]} (IoU {second[1]['iou']:.4f}), rel margin {iou_rel:.3f}; "
            f"decisive={decisive}; using '{use}'"
        ),
        "variant_scores": scores,
        "iou_rel_margin": iou_rel,
        "content_check": content,
        "decisive": decisive,
    }
    if use != "identity":
        out["outside_frac"] = scores[use]["outside_frac"]
        out["iou"] = scores[use]["iou"]
        ink_used = np.ascontiguousarray(_apply_variant(ink_norm, use))
        out["anomaly"]["halo_after_fix"] = _halo_analysis(ink_used > thr, mask)
        return ink_used, out
    out["anomaly"]["halo_identity"] = _halo_analysis(S, mask)
    return ink_norm, out


# --------------------------------------------------------------------------- grading


def smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def grade_overlay(
    base_rgb: np.ndarray, ink_norm: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Ink grade per docs/RENDER-STYLE.md. Returns (rgb uint8, opacity float32).

    1) op  = smoothstep(ink, lo, hi) * scale
    2) out = base*(1-op) + tint*op           float32, round-half-even -> uint8
    3) G   = gaussian(op, sigma) * strength; out = screen(out, tint*G)
    """
    from scipy.ndimage import gaussian_filter

    tint = np.asarray(params["tint_rgb"], dtype=np.float32)
    op = (smoothstep(ink_norm.astype(np.float32), params["opacity_lo"], params["opacity_hi"])
          * np.float32(params["opacity_scale"]))

    out = base_rgb.astype(np.float32)
    out *= (1.0 - op)[..., None]
    out += tint[None, None, :] * op[..., None]
    out_u8 = np.clip(np.rint(out), 0, 255).astype(np.uint8)  # np.rint = round-half-even
    del out
    gc.collect()

    G = gaussian_filter(op, sigma=float(params["glow_sigma_px"]), mode="reflect")
    G *= np.float32(params["glow_strength"])
    # screen blend: 1 - (1-a)(1-b), a = graded base, b = tint*G
    a = out_u8.astype(np.float32) / 255.0
    a *= -1.0
    a += 1.0  # a = 1 - out/255
    final = np.empty_like(a)
    for c in range(3):
        final[..., c] = 1.0 - a[..., c] * (1.0 - (tint[c] / 255.0) * G)
    del a, G
    final *= 255.0
    final_u8 = np.clip(np.rint(final), 0, 255).astype(np.uint8)
    del final
    gc.collect()
    return final_u8, op


def edge_feather_u(ink_norm: np.ndarray, frac: float) -> np.ndarray:
    """Fade the ink signal to 0 toward the chart's u-extremes (canvas x edges).

    The ink models produce edge artifacts at segment borders — a trace's innermost
    edge can carry a column of oversized blobs that the composite shows no support
    for. Texture x == UV u for the A/B canvases, so a
    smoothstep ramp over `frac` of the width on each side suppresses border
    inference without touching interior detections."""
    if frac <= 0:
        return ink_norm
    w = ink_norm.shape[1]
    n = max(2, int(round(w * frac)))
    x = np.arange(w, dtype=np.float32)
    ramp_in = np.clip(x / n, 0.0, 1.0)
    ramp_out = np.clip((w - 1 - x) / n, 0.0, 1.0)
    f = ramp_in * ramp_in * (3 - 2 * ramp_in) * ramp_out * ramp_out * (3 - 2 * ramp_out)
    return (ink_norm.astype(np.float32) * f[None, :]).astype(ink_norm.dtype, copy=False)


def compose_overlay(base_rgba: np.ndarray, ink_norm: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Full overlay: graded RGB + alpha carried through byte-identical (slice copy)."""
    ink_norm = edge_feather_u(ink_norm, float(params.get("edge_feather_u_frac", 0.0)))
    rgb, op = grade_overlay(base_rgba[..., :3], ink_norm, params)
    rgba = np.empty_like(base_rgba)
    rgba[..., :3] = rgb
    rgba[..., 3] = base_rgba[..., 3]  # numpy slice copy — byte-identical
    return rgba, op


# --------------------------------------------------------------------------- stats / previews helpers


def _dense_window(op: np.ndarray, win: int = 1024, cell: int = 64) -> list[int]:
    """[x, y, w, h] of the ~win-px window with the highest summed opacity."""
    h, w = op.shape
    ds, _ = block_mean(op, cell)
    k = max(1, min(win // cell, ds.shape[0], ds.shape[1]))
    ii = np.zeros((ds.shape[0] + 1, ds.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = ds.cumsum(0).cumsum(1)
    sums = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    iy, ix = np.unravel_index(int(np.argmax(sums)), sums.shape)
    x, y = ix * cell, iy * cell
    return [int(min(x, max(w - win, 0))), int(min(y, max(h - win, 0))), int(min(win, w)), int(min(win, h))]


def _load_ink_raster(path: str, kind: str) -> np.ndarray:
    if kind == "march":
        import tifffile

        return tifffile.imread(path)
    img = Image.open(path).convert("L")  # grayscale BEFORE inversion (per spec)
    arr = np.asarray(img, dtype=np.uint8)
    img.close()
    return arr


def _content_pair(base_rgba: np.ndarray, mesh_stem: str, root: Path, ds: int = 8) -> tuple | None:
    """(texture gray * alpha, mc5 gray) both at texture dims downsampled ds x — the
    registration cross-check input for May wraps (prediction is pixel-aligned to the
    'max comp 5' composite; footprint IoU 0.97 at identity on the native canvas)."""
    mc5_path = root / MC5_DIR / f"{mesh_stem}_max_composite_2.jpg"
    if not mc5_path.exists():
        return None
    h, w = base_rgba.shape[:2]
    tg = base_rgba[..., :3].astype(np.float32).mean(axis=2)
    tg *= base_rgba[..., 3].astype(np.float32) / 255.0
    with Image.open(mc5_path) as im:
        mg = np.asarray(im.convert("L").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32)
    tg_ds, _ = block_mean(tg, ds)
    mg_ds, _ = block_mean(mg, ds)
    del tg, mg
    return tg_ds, mg_ds


# --------------------------------------------------------------------------- per-mesh bake


def bake_one(job: dict, params: dict, root: str | Path = ".") -> dict:
    """Bake one overlay. Returns the per-mesh record for reports/m25_ink.json."""
    t0 = time.time()
    root = Path(root)
    rec: dict = {k: job[k] for k in ("kind", "group", "stem", "mesh_path", "tex_path", "ink_path", "tex_wh")}
    rec["polarity_audit_label"] = job["polarity_label"]

    # base texture (RGBA, verbatim alpha)
    base_img = Image.open(root / job["tex_path"])
    if base_img.mode != "RGBA":
        raise ValueError(f"{job['tex_path']}: expected RGBA, got {base_img.mode}")
    base = np.asarray(base_img)
    base_img.close()
    alpha_src = base[..., 3].copy()
    mask = alpha_src > 127

    # ink: load -> resample to exact dims -> free source immediately
    ink_raw = _load_ink_raster(str(root / job["ink_path"]), job["kind"])
    rec["ink_source_hw"] = list(ink_raw.shape)
    rec["ink_source_dtype"] = str(ink_raw.dtype)
    ink255, steps = resample_ink(ink_raw, job["tex_wh"], two_step=(job["kind"] == "march"))
    del ink_raw
    gc.collect()
    rec["resample_steps"] = steps

    # polarity (audit label verified per-image) + normalization to [0,1], ink=high
    lo, hi = params["opacity_lo"], params["opacity_hi"]
    thr_mid = (lo + hi) / 2.0
    ink_norm, pol = choose_polarity(ink255, mask, job["polarity_label"], thr_mid)
    del ink255
    gc.collect()
    rec["normalization"] = "linear v/255 (inverted: 1 - v/255); no per-image min-max stretch"

    # registration sanity check of the same-pixel-grid assumption
    # (May wraps get the mc5 content cross-check: prediction canvas == mc5 canvas)
    content_pair = _content_pair(base, job["stem"], root) if job["kind"] == "may" else None
    ink_norm, reg = registration_check(ink_norm, mask, content_pair)
    del content_pair
    rec["registration"] = reg

    # final coverage is measured AFTER any registration fix
    ink_mask = ink_norm[mask]
    pol["coverage_used"] = float((ink_mask > thr_mid).mean()) if ink_mask.size else 0.0
    cov_mid = pol["coverage_used"]
    cov_active = float((ink_mask > lo).mean()) if ink_mask.size else 0.0
    del ink_mask
    rec["polarity"] = pol

    # grade + compose
    rgba, op = compose_overlay(base, ink_norm, params)
    del ink_norm
    gc.collect()

    # coverage stats (in papyrus mask)
    op_mask = op[mask]
    rec["coverage"] = {
        "ink_frac_in_mask": cov_mid,  # PRIMARY: frac(ink > (lo+hi)/2), visibly inked
        "onset_frac_in_mask": cov_active,  # secondary: frac(ink > lo), any nonzero opacity
        "mean_opacity_in_mask": float(op_mask.mean()) if op_mask.size else 0.0,
        "p99_opacity_in_mask": float(np.percentile(op_mask, 99)) if op_mask.size else 0.0,
        "saturated_frac_in_mask": float((op_mask >= 0.999 * params["opacity_scale"]).mean()) if op_mask.size else 0.0,
        "mask_frac_alpha255": job["alpha_frac_255"],
    }
    band_lo, band_hi = COVERAGE_BAND
    cov = pol["coverage_used"]
    if cov < band_lo:
        rec["coverage"]["explanation"] = (
            f"in-mask ink fraction {cov:.4%} is below the [{band_lo:.1%}, {band_hi:.0%}] band: "
            f"per audit this wrap's texture has alpha_frac_255={job['alpha_frac_255']:.3f} "
            f"(only {job['alpha_frac_255']:.0%} of the canvas is surviving papyrus) — a damaged, "
            f"sparse outer wrap where little ink survives. Not a polarity error: the opposite "
            f"polarity gives {max(pol['coverage_as_audit_label'], pol['coverage_flipped']):.2%} "
            f"(papyrus-as-ink, absurd); this choice is the sparse-tail optimum."
        )
    elif cov > band_hi:
        rec["coverage"]["explanation"] = (
            f"in-mask ink fraction {cov:.4%} is above the [{band_lo:.1%}, {band_hi:.0%}] band "
            f"(texture alpha_frac_255={job['alpha_frac_255']:.3f} per audit): the prediction has "
            f"unusually broad strong-ink support across the surviving papyrus. Mean in-mask "
            f"opacity is {rec['coverage']['mean_opacity_in_mask']:.3f} and the saturated fraction "
            f"{rec['coverage']['saturated_frac_in_mask']:.2%} — fibers stay visible; not a "
            f"polarity error (opposite polarity gives "
            f"{max(pol['coverage_as_audit_label'], pol['coverage_flipped']):.2%})."
        )
    del op_mask
    rec["dense_window_xywh"] = _dense_window(op)
    del op
    gc.collect()

    # save PNG (compress level 6), verify alpha byte-identical on the round trip
    out_path = root / job["out_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out_path, format="PNG", compress_level=6)
    with Image.open(out_path) as chk_img:
        chk = np.asarray(chk_img)
    alpha_ok = bool(np.array_equal(chk[..., 3], alpha_src))
    dims_ok = bool(chk.shape[0] == base.shape[0] and chk.shape[1] == base.shape[1])
    del chk, rgba, base
    gc.collect()

    rec["output"] = {
        "path": job["out_path"],
        "wh": [job["tex_wh"][0], job["tex_wh"][1]],
        "file_size": out_path.stat().st_size,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "alpha_byte_identical_roundtrip": alpha_ok,
        "dims_match_base": dims_ok,
    }
    if not alpha_ok or not dims_ok:
        raise RuntimeError(f"{job['stem']}: saved overlay failed alpha/dims roundtrip check")
    rec["runtime_s"] = round(time.time() - t0, 2)
    return rec
