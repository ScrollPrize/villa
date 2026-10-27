"""The end-to-end check: resample -> band-pass -> register -> metrics -> nulls -> verdict."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import metrics as M
from .register import best_shift, pearson, resample_to_grid, shift_image, downsample


@dataclass
class CheckConfig:
    pred_um: float                      # prediction pixel size (um)
    ref_um: float                       # reference pixel size (um)
    ref_origin: tuple = (0.0, 0.0)      # (y, x) in REFERENCE pixels of prediction pixel (0,0)
    analysis_um: float | None = None    # common grid; default = max(pred_um, ref_um)
    pred_scale: str = "raw"             # 'raw' or 'ink9um' -> clip((v/255-0.25)/0.5, 0, 1)
    pred_zero_invalid: bool = True      # pixels == 0 in prediction = not inferred
    ref_threshold: float | str = "otsu" # reference ink threshold (raw ref units) or 'otsu'
    bp_fine_um: float = 0.0             # DoG fine sigma (0 = keep full resolution)
    bp_coarse_um: float = 500.0         # DoG coarse sigma: removes trends wider than ~1.5 mm
    search_mm: float = 0.5              # translation search radius (each side)
    max_stat_px: int = 2_500_000        # correlation search/nulls run on a grid <= this size
    n_shift_nulls: int = 192            # random large circular shifts of the reference
    min_shift_mm: float = 3.0           # minimum toroidal distance of a shift null
    n_metric_nulls: int = 32            # shift nulls also used for AUC/F1/stroke null (+ dihedral)
    min_stroke_mm2: float = 0.02        # smallest reference ink component counted as a stroke
    stroke_cover: float = 0.50          # fraction of a stroke that must be hit to count as recalled
    alpha_r: float = 0.01               # significance for band-pass correlation (search-matched)
    alpha_stroke: float = 0.05          # significance for stroke recall
    min_r_bp: float = 0.20              # effect size needed for PASS
    min_stroke_recall: float = 0.50     # effect size needed for PASS
    seed: int = 0
    workers: int = 2


@dataclass
class CheckResult:
    verdict: str
    summary: str
    numbers: dict = field(default_factory=dict)
    nulls: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    arrays: dict = field(default_factory=dict, repr=False)  # aligned maps for plotting


def _dihedral(a, kind):
    return {"rot180": lambda x: x[::-1, ::-1], "flipud": lambda x: x[::-1, :],
            "fliplr": lambda x: x[:, ::-1], "rot90": lambda x: np.rot90(x, 1),
            "rot270": lambda x: np.rot90(x, 3), "transpose": lambda x: x.T}[kind](a)


def _apply_null(a, spec, px_um):
    kind = spec[0]
    if kind == "roll":
        dy = int(round(spec[1] * 1000 / px_um))
        dx = int(round(spec[2] * 1000 / px_um))
        return np.roll(a, (dy, dx), axis=(0, 1))
    return np.ascontiguousarray(_dihedral(a, kind))


def _null_specs(shape_mm, cfg, rng):
    Hmm, Wmm = shape_mm
    specs = [("rot180",), ("flipud",), ("fliplr",)]
    if abs(Hmm - Wmm) < 1e-6:
        specs += [("rot90",), ("rot270",), ("transpose",)]
    shifts = []
    tries = 0
    while len(shifts) < cfg.n_shift_nulls and tries < 100 * cfg.n_shift_nulls:
        tries += 1
        dy, dx = rng.uniform(0, Hmm), rng.uniform(0, Wmm)
        ty, tx = min(dy, Hmm - dy), min(dx, Wmm - dx)
        if math.hypot(ty, tx) >= cfg.min_shift_mm:
            shifts.append(("roll", dy, dx))
    return specs, shifts


def _pvalue(obs, null):
    null = np.asarray([v for v in null if np.isfinite(v)])
    if null.size == 0 or not np.isfinite(obs):
        return float("nan"), float("nan")
    p = (1 + np.count_nonzero(null >= obs)) / (1 + null.size)
    sd = null.std()
    z = (obs - null.mean()) / sd if sd > 0 else float("inf")
    return float(p), float(z)


def run_check(pred: np.ndarray, ref: np.ndarray, cfg: CheckConfig, pred_mask: np.ndarray | None = None,
              ref_mask: np.ndarray | None = None, log=print) -> CheckResult:
    t0 = time.time()
    warnings = []
    rng = np.random.default_rng(cfg.seed)
    a_um = cfg.analysis_um or max(cfg.pred_um, cfg.ref_um)

    p = pred.astype(np.float32)
    pm = np.ones(p.shape, bool) if pred_mask is None else pred_mask.astype(bool)
    if cfg.pred_zero_invalid:
        pm &= p > 0
    if cfg.pred_scale == "ink9um":
        p = np.clip((p / 255.0 - 0.25) / 0.5, 0, 1).astype(np.float32)

    Ha = int(round(p.shape[0] * cfg.pred_um / a_um))
    Wa = int(round(p.shape[1] * cfg.pred_um / a_um))
    pa, pva = resample_to_grid(p, cfg.pred_um, a_um, (Ha, Wa), (0, 0), valid=pm)
    oy, ox = cfg.ref_origin
    ra, rva = resample_to_grid(ref.astype(np.float32), cfg.ref_um, a_um, (Ha, Wa), (oy, ox), valid=ref_mask)
    m = pva & rva
    cover = float(m.mean())
    log(f"[inkcheck] analysis grid {Ha}x{Wa} @ {a_um:.3f} um, valid overlap {cover:.1%} "
        f"({m.sum() * a_um ** 2 / 1e8:.2f} cm2)")
    if m.sum() < 1000:
        raise ValueError("almost no overlap between prediction and reference - check --ref-origin / pixel sizes")

    # reference binarisation (on the common grid)
    thr = M.otsu_threshold(ra[m]) if cfg.ref_threshold == "otsu" else float(cfg.ref_threshold)
    rbin = (ra >= thr) & m
    ink_frac = float(rbin.sum() / m.sum())
    if ink_frac < 0.005 or ink_frac > 0.5:
        warnings.append(f"reference ink fraction {ink_frac:.3f} looks odd - check --ref-threshold")

    # band-pass both maps (masked DoG) on the analysis grid
    pbp = M.bandpass(pa, m, a_um, cfg.bp_fine_um, cfg.bp_coarse_um)
    rbp = M.bandpass(ra, m, a_um, cfg.bp_fine_um, cfg.bp_coarse_um)

    # statistics grid (coarser if needed) for search + many nulls
    k = max(1.0, math.sqrt(Ha * Wa / cfg.max_stat_px))
    k = float(math.ceil(k)) if k > 1 else 1.0
    s_um = a_um * k
    ps, ms = downsample(pbp, k, m)
    rs, _ = downsample(rbp, k, m)
    R = max(0, int(round(cfg.search_mm * 1000 / s_um)))
    log(f"[inkcheck] stats grid {ps.shape[0]}x{ps.shape[1]} @ {s_um:.2f} um, search +-{R} px ({cfg.search_mm} mm)")

    (dy, dx), r_obs = best_shift(ps, ms, rs, ms, R, cfg.workers)
    r_zero = pearson(ps, rs, ms)
    raw_r = pearson(pa, ra, m)
    log(f"[inkcheck] band-pass r={r_obs:+.3f} at shift ({dy},{dx}) stat-px; zero-shift {r_zero:+.3f}; raw r {raw_r:+.3f}")

    # aligned reference on the analysis grid
    DY, DX = int(round(dy * k)), int(round(dx * k))
    ra_al, mra = shift_image(ra, m, DY, DX)
    rbp_al, _ = shift_image(rbp, m, DY, DX)
    rbin_al, _ = shift_image(rbin, m, DY, DX)
    mm = m & mra

    ranks_raw = M.rank_scores(pa[mm])
    ranks_bp = M.rank_scores(pbp[mm])
    min_area = max(4, int(round(cfg.min_stroke_mm2 * 1e6 / a_um ** 2)))

    def pix_metrics(rb, mask_sel):
        pos = rb[mm]
        sel = mask_sel[mm]
        out = {}
        if sel.all():
            out["auc_raw"] = M.auc_from_ranks(ranks_raw, pos)
            out["auc_bp"] = M.auc_from_ranks(ranks_bp, pos)
        else:  # recompute ranks within the sub-mask
            out["auc_raw"] = M.auc_from_ranks(M.rank_scores(pa[mm][sel]), pos[sel])
            out["auc_bp"] = M.auc_from_ranks(M.rank_scores(pbp[mm][sel]), pos[sel])
        msk = mm & mask_sel
        kk = int((rb & msk).sum())
        pb = M.topk_mask(pbp, msk, kk)
        tp = int((pb & rb & msk).sum())
        out["breakeven_f1"] = tp / kk if kk else float("nan")
        out.update(M.stroke_overlap(pb, rb, msk, min_area, cfg.stroke_cover, tol_px=0))
        return out

    obs = pix_metrics(rbin_al, np.ones_like(mm))
    f1, f1thr, f1p, f1r = M.best_f1(pa[mm], rbin_al[mm])
    obs.update({"best_f1_raw": f1, "best_f1_precision": f1p, "best_f1_recall": f1r})
    log(f"[inkcheck] AUC raw {obs['auc_raw']:.3f} bp {obs['auc_bp']:.3f}; breakeven F1 {obs['breakeven_f1']:.3f}; "
        f"stroke recall {obs['stroke_recall']:.3f} ({obs['n_ref_strokes']} strokes), precision {obs['stroke_precision']:.3f}")

    # nulls
    Hmm_, Wmm_ = ps.shape[0] * s_um / 1000, ps.shape[1] * s_um / 1000
    dih, shifts = _null_specs((Hmm_, Wmm_), cfg, rng)
    null_r, null_r_dih, null_metrics = [], {}, []
    for i, spec in enumerate(dih + shifts):
        rn = _apply_null(rs, spec, s_um)
        mn = _apply_null(ms, spec, s_um)
        if rn.shape != ps.shape:
            continue
        (ndy, ndx), rv = best_shift(ps, ms, rn, mn, R, cfg.workers)
        if spec[0] == "roll":
            null_r.append(rv)
        else:
            null_r_dih[spec[0]] = rv
        if spec[0] != "roll" or i < len(dih) + cfg.n_metric_nulls:
            rb_n = _apply_null(rbin, spec, a_um) if spec[0] != "roll" else np.roll(
                rbin, (int(round(spec[1] * 1000 / a_um)), int(round(spec[2] * 1000 / a_um))), axis=(0, 1))
            mk_n = _apply_null(m, spec, a_um) if spec[0] != "roll" else np.roll(
                m, (int(round(spec[1] * 1000 / a_um)), int(round(spec[2] * 1000 / a_um))), axis=(0, 1))
            rb_n, mk_s = shift_image(rb_n & mk_n, mk_n, int(round(ndy * k)), int(round(ndx * k)))
            null_metrics.append(pix_metrics(rb_n, mk_s))
        if (i + 1) % 25 == 0:
            log(f"[inkcheck]   nulls {i + 1}/{len(dih) + len(shifts)} ({time.time() - t0:.0f}s)")

    p_r, z_r = _pvalue(r_obs, null_r)
    if 1.0 / (1 + len(null_r)) > cfg.alpha_r:
        warnings.append(f"only {len(null_r)} shift nulls: the smallest attainable p ({1 / (1 + len(null_r)):.3f}) "
                        f"is above alpha={cfg.alpha_r}; increase --nulls")
    for kname, v in null_r_dih.items():
        if v >= r_obs:
            warnings.append(f"reference transformed by {kname} correlates as well or better (r={v:+.3f}) "
                            f"- orientation may be wrong or signal is not spatially specific")
    nm = {key: [d[key] for d in null_metrics] for key in obs if key in null_metrics[0]} if null_metrics else {}
    stats = {}
    for key in ("auc_raw", "auc_bp", "breakeven_f1", "stroke_recall", "stroke_precision"):
        pv, zv = _pvalue(obs[key], nm.get(key, []))
        arr = np.asarray([x for x in nm.get(key, []) if np.isfinite(x)])
        stats[key] = {"obs": obs[key], "null_mean": float(arr.mean()) if arr.size else float("nan"),
                      "null_max": float(arr.max()) if arr.size else float("nan"), "p": pv, "z": zv}

    nr = np.asarray(null_r)
    numbers = {
        "analysis_um": a_um, "stats_um": s_um, "overlap_cm2": float(mm.sum() * a_um ** 2 / 1e8),
        "ref_threshold": thr, "ref_ink_fraction": ink_frac,
        "shift_um": [DY * a_um, DX * a_um],
        "r_bandpass": r_obs, "r_bandpass_zero_shift": r_zero, "r_raw": raw_r,
        "r_bandpass_null_mean": float(nr.mean()), "r_bandpass_null_sd": float(nr.std()),
        "r_bandpass_null_max": float(nr.max()), "r_bandpass_p": p_r, "r_bandpass_z": z_r,
        "r_bandpass_dihedral": null_r_dih, "n_shift_nulls": int(nr.size),
        "n_metric_nulls": len(null_metrics),
        **{f"{k_}": v for k_, v in obs.items()},
        "metric_nulls": stats,
        "seconds": round(time.time() - t0, 1),
    }

    sig_r = np.isfinite(p_r) and p_r <= cfg.alpha_r
    sig_s = np.isfinite(stats["stroke_recall"]["p"]) and stats["stroke_recall"]["p"] <= cfg.alpha_stroke
    big = (r_obs >= cfg.min_r_bp) and (obs["stroke_recall"] >= cfg.min_stroke_recall)
    if sig_r and sig_s and big:
        verdict = "PASS"
        summary = (f"model sees known ink at {cfg.pred_um:g} um: YES (band-pass r={r_obs:.2f}, p={p_r:.3g}; "
                   f"{obs['stroke_recall']:.0%} of reference strokes recovered vs {stats['stroke_recall']['null_mean']:.0%} by chance)")
    elif sig_r:
        verdict = "WEAK"
        summary = (f"model sees known ink at {cfg.pred_um:g} um: NO at letter level - statistically detectable but weak "
                   f"agreement (band-pass r={r_obs:.2f}, p={p_r:.3g}; stroke recall {obs['stroke_recall']:.0%} vs "
                   f"{stats['stroke_recall']['null_mean']:.0%} chance)")
    else:
        verdict = "FAIL"
        summary = (f"model sees known ink at {cfg.pred_um:g} um: NO (band-pass r={r_obs:.2f}, p={p_r:.3g} vs "
                   f"search-matched null)")
    log(f"[inkcheck] VERDICT {verdict}: {summary}")
    cfgd = asdict(cfg)
    return CheckResult(verdict, summary, numbers, {"r_bandpass_shift_nulls": [float(x) for x in nr]}, cfgd,
                       warnings, {"pred": pa, "ref": ra_al, "mask": mm, "ref_bin": rbin_al, "pred_bp": pbp,
                                  "ref_bp": rbp_al})
