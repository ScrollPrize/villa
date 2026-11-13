"""The geometry calibration step: the detector map and the poses of a two-scan helical acquisition, measured from
the acquisition itself with nothing typed in.

    cal = calibrate_geometry([main_h5, companion_h5])        # main: the scan with the rotation axis inside the field
    save(cal, "geometry.json")

What it needs of the acquisition: a helix whose travel per turn is less than the detector's height, a turn at
rest before and after it, the axis inside the field in the first scan, and a second scan moved sideways. What it
finds, in this order:

  layout   frames per turn, the stretches, which way the rows run against the axis stage, a coarse axis column
           and lateral move (profiles of single frames correlated), and the rows that hold something across the
           whole width (the mount), which are kept out of every comparison;
  rows     the time of a detector row within the frame, from the beam dips of a refill if a stretch has one
           (otherwise it must be given);
  map      pixel (c, v) looks along the ray at (c + Eu(c, v), v + Ev(c, v)), Eu and Ev polynomials. Views that
           hold the same lines on different parts of the detector (opposite views, a rest turn against the helix,
           the helix a turn apart, one scan against the other) are registered on the current map's grid, what
           they still disagree by is fitted as a correction of the map, and that is repeated until nothing is
           left. Registering on the corrected grid matters: blocks that an uncorrected term leaves unequal give
           false displacements (development_notes/calibration/05_experiments.md, experiments 10, 23, 25).
  poses    what the same fit gives beside the map: the axis column per rest turn, the second scan's place
           against the first per stretch, the heights of the rest turns against the helix, and for the second
           scan a line in the column per stretch (what differs between the scans and is not the detector's).
  motion   on the finished map's grid, what happens within a turn: the rotating body's place against the axis as
           a function of time, the sample's height once per turn and its drift at rest, and how far the helix is
           further on in angle than the rows' timing gives.

The step writes what it measured; the reconstruction (recon.py) takes its geometry from that alone: frame_poses,
reconstruction_terms and merged_layout turn the result into the pose of every frame, the reconstruction's terms
per projection and the frame the two scans are merged in. The step uses helirec::calibration's session (raw
frames, the scan's own dark and mean flat, the registration) and nothing of the reconstruction's intensity
calibration (calibration.py).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from .solve import fit_line

KINDS = ("opposite", "resthelix", "turn", "pair")


# ---------------------------------------------------------------- the map

def _text(cal):
    """A calibration as the JSON text the C++ side reads (helirec/geometry.hpp)."""
    return cal if isinstance(cal, str) else json.dumps(cal, default=float)


def _map_args(m):
    return [int(x) for x in m["i"]], [int(x) for x in m["j"]], [float(x) for x in m["eu"]], [float(x) for x in m["ev"]], float(m["c0"]), float(m["cs"]), float(m["v0"]), float(m["vs"])


def terms(order):
    """Powers (i, j) of the map's polynomial in the column and the row: all of total order 2 to `order`. The linear
    ones are left out (a scale, a shear and a rotation of the grid cannot be told from the poses), but for the
    rows' slope across the columns, which the opposite views of one scan and the overlap of two do measure.
    (helirec::geometry::map_terms)"""
    from . import _core
    return [tuple(t) for t in _core.geometry_map_terms(int(order))]


def empty_map(rows, cols, order=3):
    from . import _core
    return _core.geometry_empty_map(int(rows), int(cols), int(order))


def _phi(m, c, v, i, j):
    return ((np.asarray(c, float) - m["c0"]) / m["cs"]) ** i * ((np.asarray(v, float) - m["v0"]) / m["vs"]) ** j


def map_displacement(m, c, v):
    """(Eu, Ev) of the map at detector position (c, v). (helirec::geometry::map_displacement)"""
    from . import _core
    c, v = np.broadcast_arrays(np.asarray(c, np.float64), np.asarray(v, np.float64))
    eu, ev = _core.geometry_map_displacement(*_map_args(m), np.ascontiguousarray(c.ravel()), np.ascontiguousarray(v.ravel()))
    return eu.reshape(c.shape)[()], ev.reshape(c.shape)[()]


def with_line(m, a, b, about):
    """The map with a line in the column added for one scan: its blocks are read a (c - about) columns and
    b (c - about) rows further on. (helirec::geometry::with_line)"""
    from . import _core
    return _core.geometry_with_line(*_map_args(m), float(a), float(b), float(about))


def _set_map(session, m, scan=-1):
    session.set_detector_map(m["i"], m["j"], m["eu"], m["ev"], m["c0"], m["cs"], m["v0"], m["vs"], scan)


# ---------------------------------------------------------------- the layout

@dataclass
class Layout:
    rows: int
    cols: int
    per_turn: int
    pixel_um: float
    stretches: list          # per scan: {"rest1": index, "helix": index, "rest2": index}
    lengths: list            # per scan: {name: frames}
    row_sign: float = -1.0
    axis: float = 0.0        # coarse mirror column of the first scan
    move: int = 0            # coarse: the second scan's content lies this many columns higher
    width: int = 1024        # columns of a window
    blocked: dict = field(default_factory=dict)   # (scan, name) -> (frames sampled, mask (samples, rows))

    def is_blocked(self, scan, name, frame, row_lo, row_hi):
        f, mask = self.blocked.get((scan, name), (None, None))
        if f is None:
            return False
        k = int(np.argmin(np.abs(f - frame)))
        return bool(mask[k, max(0, int(row_lo)):max(0, int(row_hi))].any())


def overlap_centre(lay):
    """The middle column, in the first scan, of what both scans record."""
    return (lay.cols - lay.move) / 2.0 if lay.move >= 0 else (lay.cols - lay.move) / 2.0


def stretch_names(session, scan):
    """{'rest1', 'helix', 'rest2'} -> stretch index, from the record's own order of its projection stretches."""
    out, rest = {}, 0
    for st in session.stretches(scan):
        if st["role"] == "projection" and st["latched"]:
            if st["helical"]:
                out["helix"] = st["index"]
            else:
                rest += 1
                out[f"rest{rest}"] = st["index"]
    if set(out) != {"rest1", "helix", "rest2"}:
        raise ValueError(f"the geometry step needs a rest turn, a helix and a rest turn; the record of scan {scan} holds {sorted(out)}")
    return out


def _lag(a, b):
    """cc[k] = sum_c a[c] b[c + k] for k in (-n, n), as an array indexed by k + n."""
    n = len(a)
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * n)) * np.fft.rfft(b, 2 * n), 2 * n)
    return np.concatenate([cc[n:], cc[:n]])


def _profile(session, scan, st, frame, row0, nrows, cols):
    p = session.line_integrals(scan, st, np.array([frame], np.int64), np.array([row0], np.int32), nrows, 0, cols)[0].mean(0)
    k = 31
    return p - np.convolve(np.pad(p, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def read_layout(session, device="auto", log=print):
    info = session.info(0)
    rows, cols = int(info["rows"]), int(info["cols"])
    stretches = [stretch_names(session, k) for k in (0, 1)]
    lengths = [{n: int(next(x["n_frames"] for x in session.stretches(k) if x["index"] == i)) for n, i in stretches[k].items()} for k in (0, 1)]
    step = float(np.median(np.diff(session.series(0, stretches[0]["helix"], "angle_deg"))))
    lay = Layout(rows=rows, cols=cols, per_turn=int(round(360.0 / abs(step))), pixel_um=float(info["pixel_um"]), stretches=stretches, lengths=lengths)
    lay.width = int(2 ** np.floor(np.log2(0.45 * cols)))
    # rows that hold something across the whole width: the mean of the lowest columns of a row is not air there
    for k in (0, 1):
        for name, st in stretches[k].items():
            n = lengths[k][name]
            frames = np.unique(np.linspace(0, n - 1, max(8, n // max(1, lay.per_turn // 16))).astype(np.int64))
            mask = np.zeros((len(frames), rows), bool)
            few = max(8, cols // 50)
            for i, f in enumerate(frames):
                p = session.line_integrals(k, st, np.array([f], np.int64), np.array([0], np.int32), rows, 0, cols)[0]
                q = np.partition(p, few, axis=1)[:, :few].mean(1)
                bad = q > np.median(q) + 0.1
                mask[i] = np.convolve(bad.astype(float), np.ones(49), mode="same") > 0      # with 24 rows around them
            if mask.any():
                lay.blocked[(k, name)] = (frames, mask)
                spans = [(int(np.nonzero(x)[0].min()), int(np.nonzero(x)[0].max())) for x in mask if x.any()]
                where = f"rows {spans[0][0]}..{spans[0][1]}" if len(set(spans)) == 1 else f"rows between {min(a for a, _ in spans)} and {max(b for _, b in spans)}, changing with the frame,"
                log(f"  scan {k} {name}: {where} hold something across the whole width in {len(spans)} of {len(frames)} sampled frames; kept out")
    # the axis column and the lateral move, coarsely: profiles of single frames in rows free of that
    st0, st1 = stretches[0]["rest1"], stretches[1]["rest1"]
    free = ~np.any([lay.blocked[key][1].any(0) for key in ((0, "rest1"), (1, "rest1")) if key in lay.blocked] or [np.zeros(rows, bool)], axis=0)
    r0 = int(np.clip(rows // 2 - 128, 0, rows - 256))
    if not free[r0:r0 + 256].all():
        runs = np.nonzero(np.convolve(free.astype(float), np.ones(256), mode="valid") == 256)[0]
        r0 = int(runs[len(runs) // 2])
    cc_axis, cc_move = 0.0, 0.0
    for p in np.linspace(lay.per_turn // 20, lay.per_turn // 2 - lay.per_turn // 20, 4).astype(int):
        a = _profile(session, 0, st0, int(p), r0, 256, cols)
        b = _profile(session, 0, st0, int(p) + lay.per_turn // 2, r0, 256, cols)
        t = _profile(session, 1, st1, int(p), r0, 256, cols)
        cc_axis = cc_axis + _lag(a, b[::-1])
        cc_move = cc_move + _lag(a, t)
    lay.axis = (cols - 1 - (int(np.argmax(cc_axis)) - cols)) / 2.0
    lay.move = int(np.argmax(cc_move)) - cols
    # which way the rows run against the axis stage: the sense in which the helix a turn apart registers
    best = None
    for sign in (-1.0, 1.0):
        reqs = [dict(scan_a=0, scan_b=0, stretch_a=stretches[0]["helix"], stretch_b=stretches[0]["helix"], frame_a=int(p), frame_hint=lay.per_turn, frames=32, rows=128,
                     cols=lay.width, row_a=-1, col_a=(cols - lay.width) // 2, col_b=(cols - lay.width) // 2) for p in np.linspace(lay.per_turn, lengths[0]["helix"] - 2 * lay.per_turn - 64, 3)]
        got = [r["correlation"] for r in session.measure(reqs, sign, device) if r["planned"] and r["ok"]]
        score = float(np.mean(got)) if got else 0.0
        if best is None or score > best[0]:
            best = (score, sign)
    if best[0] < 0.3:
        raise ValueError("the helix a turn apart does not register in either sense of the rows: the travel per turn must be less than the detector's height")
    lay.row_sign = best[1]
    log(f"  {rows} x {cols} px, {lay.per_turn} frames per turn; mirror column about {lay.axis:.1f}; second scan {lay.move:+d} columns; row sense {lay.row_sign:+.0f}; windows of {lay.width} columns")
    return lay


# ---------------------------------------------------------------- the time of a detector row

def fit_refill_shots(x, n_rows, T, E, pre):
    """Fit exposure-integrated beam dips in C++; return parameters and Ceres diagnostics.
    Insufficient shots, invalid noise and unusable fits return None. Unavailable
    row-time uncertainty is reported as NaN with uncertainty_available=False."""
    from . import _core
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != n_rows:
        raise ValueError("refill levels must have shape (frames, n_rows)")
    fit = _core.calibration_fit_refill_shots(x, T, E, pre)
    return fit if fit["usable"] else None


def row_time_from_refill(session, lay, log=print):
    """Seconds per stored row (signed: negative if a larger row index is exposed earlier), from the beam dips of a
    refill; None if no stretch holds one. The method of development_notes/calibration/04_experiments (exp. 2): each
    injection shot dips the beam for a few milliseconds, the shots sit on a raster, and which rows of which frame
    record a shot depends on the row time through the frame period and the gate only (fit_refill_shots)."""
    for scan in (0, 1):
        for name in ("helix", "rest1", "rest2"):
            st = lay.stretches[scan][name]
            cur, when = session.series(scan, st, "monitor_current"), session.series(scan, st, "monitor_epoch")
            ok = np.isfinite(cur) & np.isfinite(when)
            cur, when = cur[ok], when[ok]
            if len(cur) < 8 or cur.max() - cur[0] < 0.02 * cur[0]:
                continue
            epoch = session.series(scan, st, "epoch_trig")
            lo_i = int(np.argmin(cur))
            t_lo, t_hi = when[lo_i], when[lo_i + int(np.argmax(cur[lo_i:]))]
            f_lo, f_hi = max(20, int(np.searchsorted(epoch, t_lo - 8.0))), min(len(epoch) - 20, int(np.searchsorted(epoch, t_hi + 8.0)))
            if f_hi - f_lo < 100:
                continue
            T = float(np.median(session.series(scan, st, "timer_period")))
            E = float(np.median(session.series(scan, st, "timer_delta")))
            # columns that are air in every frame around the refill
            blk = lay.blocked.get((scan, name))
            free = np.ones(lay.rows, bool) if blk is None else ~blk[1].any(0)
            r_lo, r_hi = lay.rows // 4, 3 * lay.rows // 4
            peak = np.zeros(lay.cols)
            for f in np.linspace(f_lo, f_hi, 24).astype(np.int64):
                p = session.line_integrals(scan, st, np.array([f], np.int64), np.array([r_lo], np.int32), r_hi - r_lo, 0, lay.cols)[0]
                peak = np.maximum(peak, p[free[r_lo:r_hi]].mean(0))
            air = peak < np.percentile(peak, 2) + 0.03
            edges = np.flatnonzero(np.diff(np.concatenate([[0], air.astype(int), [0]])))
            if len(edges) < 2:
                continue
            c0, c1 = max(((b - a, a, b) for a, b in zip(edges[::2], edges[1::2])))[1:]
            c0, c1 = c0 + 8, c1 - 8
            if c1 - c0 < 16:
                continue
            dark = session.dark(scan)[:, c0:c1].mean(1)
            pre = min(400, f_lo - 20)                 # frames before the refill: what a row band scatters by without a shot
            frames = np.arange(f_lo - pre, f_hi + 16, dtype=np.int64)
            x = np.empty((len(frames), lay.rows))
            for i0 in range(0, len(frames), 64):
                raw = session.read_rows(scan, st, frames[i0:i0 + 64], 0, lay.rows)
                x[i0:i0 + 64] = np.log(np.clip(raw[:, :, c0:c1].mean(2, dtype=np.float64) - dark, 1, None))
            got = fit_refill_shots(x, lay.rows, T, E, pre)
            if got is None:
                continue
            params, se, shots = got["x"], got["row_time_se"], got["shots"]
            log(f"  refill in scan {scan} {name}, frames {f_lo}..{f_hi}: {len(shots)} shots on a raster of {params[3] * 1e3:.3f} ms, air columns {c0}..{c1}; "
                f"row time {params[0] * 1e6:+.3f} +- {se * 1e6:.3f} us per stored row")
            return dict(seconds_per_row=float(params[0]), se=se if got["uncertainty_available"] else None,
                        solver={k: got[k] for k in ("solver_version", "termination", "message", "cost", "iterations")}, scan=scan, stretch=name, frames=[f_lo, f_hi], shots=int(len(shots)), raster_s=float(params[3]), air_columns=[int(c0), int(c1)],
                        period_s=T, gate_s=E)
    return None


# ---------------------------------------------------------------- the comparisons

def plan(lay, positions=(8, 16, 12, 10)):
    """The comparisons of one pass, placed from the layout alone: (request for the session, what it compares)."""
    R, C, N, W = lay.rows, lay.cols, lay.per_turn, lay.width
    n_opp, n_rest, n_turn, n_pair = positions
    nrows = min(256, R // 4)
    bands = np.linspace(24, R - nrows - 24, 5).astype(int)
    axis2 = int(round(2 * lay.axis))
    reqs, meta = [], []

    def add(kind, sa, a, sb, b, frame, hint, col_a, col_b, rows_, row_a, mirror=False):
        reqs.append(dict(label=kind, scan_a=sa, scan_b=sb, stretch_a=lay.stretches[sa][a], stretch_b=lay.stretches[sb][b], frame_a=int(frame), frame_hint=int(hint), angle_offset_deg=180.0 if mirror else 0.0,
                         mirror=mirror, frames=32, rows=int(rows_), cols=W, row_a=int(row_a), col_a=int(col_a), col_b=int(col_b)))
        meta.append(dict(kind=kind, scan_a=sa, scan_b=sb, stretch_a=a, stretch_b=b))

    lo, hi = max(0, axis2 - (W - 1) - (C - W)), min(C - W, axis2 - (W - 1))      # windows whose mirror image lies on the detector
    if hi - lo < 32:
        raise ValueError("the rotation axis is too close to the detector's edge for opposite views in the first scan")
    for a in ("rest1", "rest2"):
        for p in np.linspace(16, N // 2 - 80, n_opp).astype(int):
            for row_a in bands:
                for ca in np.linspace(lo + 16, hi - 16, 3).astype(int):
                    add("opposite", 0, a, 0, a, p, N // 2, ca, axis2 - (W - 1) - ca, nrows, row_a, mirror=True)
    across = np.linspace(13, C - W - 13, 3).astype(int)
    for scan in (0, 1):
        hrows = _rows(lay, scan, "helix")
        for a in ("rest1", "rest2"):
            hint = 0 if a == "rest1" else lay.lengths[scan]["helix"] - lay.lengths[scan][a]
            rrows = _rows(lay, scan, a)
            for p in np.linspace(16, lay.lengths[scan][a] - 50, n_rest).astype(int):
                Nr = int(round(hrows[p + 16 + hint] - rrows[p + 16]))
                if abs(Nr) < 48 or abs(Nr) > R - 128 - 60:
                    continue                      # too near (a pattern fixed to the detector would register) or off the detector
                nr = nrows if abs(Nr) <= R - nrows - 84 else 128
                m = 40
                a0, a1 = max(0, m - Nr), min(R - nr, R - nr - m - Nr)
                if a1 < a0:
                    continue
                for row_a in ([a0, (a0 + a1) // 2, a1] if a1 - a0 > 300 else ([a0, a1] if a1 - a0 > 120 else [(a0 + a1) // 2])):
                    for ca in across:
                        add("resthelix", scan, a, scan, "helix", p, hint, ca, ca, nr, row_a)
        for p in np.linspace(16, lay.lengths[scan]["helix"] - N - 100, n_turn).astype(int):
            for ca in across:
                add("turn", scan, "helix", scan, "helix", p, N, ca, ca, 128, -1)
    lo, hi = max(0, -lay.move), min(C - W, C - W - lay.move)                       # windows both scans hold
    if hi - lo < 32:
        raise ValueError("the two scans do not overlap by a window's width")
    for a in ("rest1", "helix", "rest2"):
        for p in np.linspace(16, min(lay.lengths[0][a], lay.lengths[1][a]) - 64, n_pair).astype(int):
            for row_a in bands:
                for ca in np.linspace(lo + 13, hi - 13, 4).astype(int):
                    add("pair", 0, a, 1, a, p, 0, ca, ca + lay.move, nrows, row_a)
    return reqs, meta


def _rows(lay, scan, name):
    return lay._heights[(scan, name)]


def measure(session, lay, m, lines, device="auto", patch=0):
    """One pass: every comparison of the plan registered on the grid of map `m` (the second scan with its line of
    the stretch added). Returns the accepted ones with the quantities the fit is written in.

    patch: 0 registers each block of the plan as one (a rigid displacement of 1024 columns: right only on a grid
    that already holds the map, hence the passes). N > 0 cuts every block into patches of N x N and registers each
    (CalibrationSession.measure_patches: the block's frames read once): the map's stretch inside a patch is a
    hundredth of a pixel, so a pass on the frames as stored already gives the map."""
    reqs, meta = plan(lay)
    out = []
    by = {}
    for q, mm in zip(reqs, meta):
        by.setdefault(mm["stretch_a"] if mm["kind"] == "pair" else "", []).append((q, mm))
    for key, items in by.items():
        _set_map(session, m)
        if key:
            _set_map(session, with_line(m, *lines[key], about=overlap_centre(lay) + lay.move), 1)
        if patch:    # the blocks in the order of their frames: neighbours read the same part of the files
            items.sort(key=lambda qm: (qm[1]["scan_a"], qm[1]["stretch_a"], qm[0]["frame_a"]))
        for i0 in range(0, len(items), 400):
            chunk = items[i0:i0 + 400]
            if patch:
                got = session.measure_patches([c[0] for c in chunk], lay.row_sign, int(patch), int(patch), device)
                results = [(chunk[r["request"]], r) for r in got if r["patch_i"] >= 0]
            else:
                results = zip(chunk, session.measure([c[0] for c in chunk], lay.row_sign, device))
            for (q, mm), r in results:
                if not (r["planned"] and r["ok"] and r["correlation"] > 0.3):
                    continue
                if lay.is_blocked(mm["scan_a"], mm["stretch_a"], r["frame_a"], r["row_a"] - 8, r["row_a"] + r["rows"] + 8):
                    continue
                if lay.is_blocked(mm["scan_b"], mm["stretch_b"], r["frame_b"], r["row_a"] + r["row_offset"] - 8, r["row_a"] + r["row_offset"] + r["rows"] + 8):
                    continue
                d = dict(mm)
                d.update(r)
                out.append(d)
    return out


# ---------------------------------------------------------------- the fit

def fit(rows, lay, m, tau, period, huber=0.03, max_iterations=100):
    """What the comparisons leave, as a correction of the map `m` (same terms), the poses beside it, and the residuals.

    A feature seen at A and at B obeys, with E the part of the detector's map the grid does not hold yet:
        columns, same view direction   du       = u0 - (Eu(B) - Eu(A))
        columns, opposite views        2 mirror = u0 - (Eu(B) + Eu(A)) + 2 psi (v - v0)
        rows                           y        = v0 - (Ev(B) - Ev(A)) [+ 2 psi (c - axis) for opposite views]
    y is the image's row displacement minus the encoder's at the instants the rows were exposed. u0, v0: one pair
    per frame position (whatever the sample and the stages did between positions is theirs, not the map's). psi:
    the roll of the first scan's axis. Between the scans, per stretch, rho (a rotation of the second scan's content
    about the beam) and a line in the column (a, b), whose mean over the stretches is left to the detector."""
    from .solve import Problem

    tm = list(zip(m["i"], m["j"]))
    nt = len(tm)
    fit_u = [k for k, t in enumerate(tm) if t != (1, 0)]            # Eu's line in the column is a scale: not observable
    pair_st = sorted({r["stretch_a"] for r in rows if r["kind"] == "pair"})
    K = len(pair_st)
    cref = overlap_centre(lay)
    groups = sorted({(r["kind"], r["scan_a"], r["stretch_a"], int(r["frame_a"])) for r in rows})
    gi = {g: k for k, g in enumerate(groups)}
    n = len(rows)
    of = lambda key: np.array([r[key] for r in rows], float)
    kinds = np.array([r["kind"] for r in rows])
    mirror, pair = kinds == "opposite", kinds == "pair"
    cA, cB = of("col_a") + (of("cols") - 1) / 2, of("col_b") + (of("cols") - 1) / 2
    vA = of("row_a") + (of("rows") - 1) / 2
    vB = vA + of("row_offset")
    q = of("dq") - of("stated_dq")
    vmid = m["v0"]
    g = np.array([gi[(r["kind"], r["scan_a"], r["stretch_a"], int(r["frame_a"]))] for r in rows], np.int64)
    st = np.array([pair_st.index(r["stretch_a"]) if r["kind"] == "pair" else 0 for r in rows], np.int64)
    p = Problem()
    for name, size in (("eu", nt), ("ev", nt), ("psi", 1), ("rho", K), ("a", max(K - 1, 0)), ("b", max(K - 1, 0)), ("u0", len(groups)), ("v0", len(groups))):
        p.unknowns(name, size)
    columns = p.equations(np.where(mirror, of("col_a") + of("col_b") + of("cols") - 1 - of("du"), of("du")))
    lines_ = p.equations((of("D") - of("stated_D")) - of("slope_b") * q - tau / period * (of("slope_b") * vB - of("slope_a") * vA))
    for k, (ti, tj) in enumerate(tm):
        fa, fb = _phi(m, cA, vA, ti, tj), _phi(m, cB, vB, ti, tj)
        if k in fit_u:
            columns.add("eu", k, np.where(mirror, -(fb + fa), -(fb - fa)))
        lines_.add("ev", k, -(fb - fa))
    columns.add("u0", g, 1.0)
    lines_.add("v0", g, 1.0)
    columns.add("psi", 0, 2.0 * (vA - vmid), mirror)
    lines_.add("psi", 0, 2.0 * (cA - lay.axis), mirror)
    if K:
        columns.add("rho", st, -(vA - vmid), pair)
        lines_.add("rho", st, cA - cref, pair)
    for key, e in (("a", columns), ("b", lines_)):
        for s_ in range(K - 1):                         # the last stretch's line is minus the sum of the others
            e.add(key, s_, np.where(st == s_, cA - cref, -(cA - cref)), pair & ((st == s_) | (st == K - 1)))
    for k, t in enumerate(tm):
        if t == (1, 0):
            p.fix("eu", k)
    sol = p.solve(huber=huber, max_iterations=max_iterations)
    res = np.concatenate([sol.left(columns), sol.left(lines_)])
    out = dict(eu=[float(x) for x in sol["eu"]], ev=[float(x) for x in sol["ev"]], psi=float(sol["psi"][0]), rho={s_: float(x) for s_, x in zip(pair_st, sol["rho"])}, lines={}, n=n, solver=sol.diagnostics)
    for i_, key in enumerate(("a", "b")):
        vals = [float(x) for x in sol[key]]
        vals.append(-sum(vals))
        for s_, v_ in zip(pair_st, vals):
            out["lines"].setdefault(s_, [0.0, 0.0])[i_] = v_
    out["left"] = {k: dict(n=int((kinds == k).sum()), px=float(np.sqrt(np.mean(res[:n][kinds == k] ** 2))), rows=float(np.sqrt(np.mean(res[n:][kinds == k] ** 2)))) for k in KINDS if (kinds == k).any()}
    out["groups"] = {g_: (float(sol["u0"][k]), float(sol["v0"][k])) for g_, k in gi.items()}
    return out


def _least_squares(X, y, columns=None, rows=None):
    """b of X b = y by least squares, through the steps' solver (helirec.solve, as every fit of the calibration).
    columns: the unknowns fitted (the others stay zero); rows: the equations used (the others are left to be
    predicted). Unused columns are fixed; dependent active columns are rejected."""
    from .solve import Problem

    X = np.asarray(X, float)
    cols = np.arange(X.shape[1]) if columns is None else np.asarray(columns, np.int64)
    b = np.zeros(X.shape[1])
    if not len(cols):
        return b
    if not len(X):
        raise ValueError("INSUFFICIENT_SUPPORT: no observations for regression")
    p = Problem()
    p.unknowns("b", len(cols))
    e = p.equations(y, fit=True if rows is None else rows)
    for k, c in enumerate(cols):
        e.add("b", k, X[:, c])
    b[cols] = p.solve()["b"]
    return b


def _update_range(m, d):
    """The largest displacement the correction d makes anywhere on the detector [px, rows]."""
    c, v = np.meshgrid(np.linspace(0, 2 * m["c0"] - 1, 40), np.linspace(0, 2 * m["v0"], 24))
    eu, ev = map_displacement(dict(m, eu=d["eu"], ev=d["ev"]), c, v)
    return float(np.abs(eu - eu.mean()).max()), float(np.abs(ev - ev.mean()).max())


# ---------------------------------------------------------------- the motion within a turn

def _good(n):
    if n < 2 or n % 2:
        return False
    for q in (2, 3, 5, 7):
        while n % q == 0:
            n //= q
    return n == 1


def _block_size(limit):
    """The largest block size up to `limit` the registration takes as it is (quick transform sizes, with a margin)."""
    for n in range(int(limit), 15, -1):
        if _good(n) and any(_good(n + 2 * m_) for m_ in range(8, 33)):
            return n
    raise ValueError("no block size")


def _nodes(t1, t2, th, n_extra, h, total):
    """Design of [d(t2) - d(t1)] . (cos, sin)(th), d at the nodes h, 2h, .. (d(0) = 0), a line between; then n_extra empty columns."""
    K = int(np.ceil(total / h - 1e-9))
    X = np.zeros((len(th), 2 * K + n_extra))
    for sign, t in ((1.0, t2), (-1.0, t1)):
        x = np.clip(t / h, 0, K - 1e-9)
        k = np.floor(x).astype(int)
        f = x - k
        for node, w in ((k, 1 - f), (k + 1, f)):
            sel = node > 0
            np.add.at(X, (np.nonzero(sel)[0], 2 * (node[sel] - 1)), sign * w[sel] * np.cos(th[sel]))
            np.add.at(X, (np.nonzero(sel)[0], 2 * (node[sel] - 1) + 1), sign * w[sel] * np.sin(th[sel]))
    return X, K





def second_scan_fit(table, rows, total):
    """The second scan against the first at equal frames, from the table of their comparisons (stretch, turns from
    the start, angle, middle row, columns, rows, angle in frames). Columns: per stretch an offset and a line in the
    row, in the helix a line in the travel, and once per turn a course with a point every turn (what a body that
    lies elsewhere against the axis shows, and how that changes). Rows and angle: the same with one term once per
    turn. Twice per turn: fitted beside, recorded, not explained. The two bodies' own courses within their scans
    (measure_motion, sideways) do not account for the change of the term once per turn (343P: fitted beside it they
    take coefficients of -2.1 and 0.4 for 1 and -1), so it is taken from these comparisons alone."""
    st = np.array([r[0] for r in table])
    t_, th, v, du, y, q = np.array([r[1:] for r in table], float).T
    names = [a for a in ("rest1", "helix", "rest2") if (st == a).any()]
    vm = v - (rows - 1) / 2
    hel = st == "helix"
    trav = np.where(hel, t_ - (t_[hel].mean() if hel.any() else 0.0), 0.0)             # turns of travel from the helix's middle
    base = [(st == a).astype(float) for a in names] + [(st == a) * vm for a in names] + [trav]
    n1 = len(base)
    K = int(np.ceil(total - 1e-9)) + 1
    hat = [np.clip(1 - np.abs(t_ - k), 0, 1) for k in range(K)]
    course = [w * f_ for w in hat for f_ in (np.cos(th), np.sin(th))]
    once, twice = [np.cos(th), np.sin(th)], [np.cos(2 * th), np.sin(2 * th)]
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))
    P = dict(blocks={a: int((st == a).sum()) for a in names})
    for key, val, harm in (("columns", du, course), ("rows", y, once), ("angle_frames", q, once)):
        Xf = np.stack(base + harm + twice, 1)
        nh = len(harm)
        b, b1, b0 = _least_squares(Xf, val), _least_squares(Xf[:, :n1 + nh], val), _least_squares(Xf[:, :n1], val)
        P[key] = dict(offset={a: float(b1[i]) for i, a in enumerate(names)}, per_row={a: float(b1[len(names) + i]) for i, a in enumerate(names)}, per_turn_of_travel=float(b1[n1 - 1]),
                      twice_per_turn=[float(b[n1 + nh]), float(b[n1 + nh + 1])],
                      left=dict(offsets_only=rms(val - Xf[:, :n1] @ b0), with_once_per_turn=rms(val - Xf[:, :n1 + nh] @ b1), with_twice_per_turn=rms(val - Xf @ b)),
                      left_by_stretch={a: rms((val - Xf[:, :n1 + nh] @ b1)[st == a]) for a in names})
        if key == "columns":
            P[key]["once_per_turn_nodes"] = [[float(b1[n1 + 2 * k]), float(b1[n1 + 2 * k + 1])] for k in range(K)]      # at 0, 1, 2, .. turns from the start
        else:
            P[key]["once_per_turn"] = [float(b1[n1]), float(b1[n1 + 1])]
    return P


def _measure_blocks(session, requests, row_sign, device):
    """session.measure(requests), each block read once: where the registration moves a block by whole frames, rows
    or columns it is cut again from that reading instead of from the files (CalibrationSession.measure_patches with
    the block as its one patch). The same numbers; the files are read about half as often."""
    out = [None] * len(requests)
    for r in session.measure_patches(requests, row_sign, 1 << 20, 1 << 20, device):
        out[r["request"]] = r
    return out


def measure_motion(session, lay, m, lines, tau, period, device="auto", log=print):
    """What the sample and the rotating body do within a turn, on the grid of the map (experiment 21 of the notes):

    sideways    the body's place against the rotation axis in its own frame, d(t): two views at equal angle differ by
                [d(t2) - d(t1)] . (cos, sin)(angle); opposite views see [d(t) - d(t + half a turn)] . n / 2 in their
                mirror column. The first scan has both and gets a point every half turn, the second every turn.
                Beside it the travel's direction against the columns and a constant per rest turn.
    along axis  z(angle): the sample's height once per turn, from opposite views inside the helix (they show -2 z),
                and what the rest turns' opposite views hold beyond it: a drift at rest.
    angle       frames per turn of travel by which the helix is further on than the rows' timing gives.
    heights     the rest turns against the helix (offset, scale) and the helix's own scale, on full-width blocks.
    second scan the second scan against the first at equal frames, on the whole overlap, densely along every
                stretch: per stretch a place (columns, rows, angle); once per turn, common to the stretches, the
                rotating body's place against the axis in the second scan minus the first's and its counterpart
                along the axis; in the helix a line in the travel; twice per turn, recorded and not explained."""
    R, C, N = lay.rows, lay.cols, lay.per_turn
    wide = _block_size(C - 28)
    step = float(np.median(np.diff(session.series(0, lay.stretches[0]["helix"], "angle_deg"))))
    fan = lay.pixel_um * 1e-3 / session.number(0, "scan/source_sample_distance") / np.radians(abs(step))
    start = [{"rest1": 0.0, "helix": lay.lengths[k]["rest1"] / N, "rest2": (lay.lengths[k]["rest1"] + lay.lengths[k]["helix"]) / N} for k in (0, 1)]
    total = max(start[k]["rest2"] + lay.lengths[k]["rest2"] / N for k in (0, 1))
    reqs, meta = [], []

    def add(kind, scan, a, b, frame, hint, cols, col_a, col_b, rows_, row_a, mirror=False):
        reqs.append(dict(label=kind, scan_a=scan, scan_b=scan, stretch_a=lay.stretches[scan][a], stretch_b=lay.stretches[scan][b], frame_a=int(frame), frame_hint=int(hint), angle_offset_deg=180.0 if mirror else 0.0,
                         mirror=mirror, frames=32, rows=int(rows_), cols=int(cols), row_a=int(row_a), col_a=int(col_a), col_b=int(col_b), frame_shear=2 * fan if mirror else 0.0))
        meta.append(dict(kind=kind, scan=scan, a=a, b=b))

    for scan in (0, 1):
        hrows = lay._heights[(scan, "helix")]
        for a in ("rest1", "rest2"):
            hint = 0 if a == "rest1" else lay.lengths[scan]["helix"] - lay.lengths[scan][a]
            for p in range(16, lay.lengths[scan][a] - 48, max(1, N // 125)):
                Nr = abs(hrows[p + 16 + hint] - lay._heights[(scan, a)][p + 16])
                add("resthelix", scan, a, "helix", p, hint, wide, 14, 14, 256 if Nr <= R - 340 else 128, -1)
        for p in range(16, lay.lengths[scan]["helix"] - N - 48, max(1, N // 62)):
            add("turn", scan, "helix", "helix", p, N, wide, 14, 14, 128, -1)
    axis2 = int(round(2 * lay.axis))
    w_opp = _block_size(min(C - 1, axis2) - max(0, axis2 - (C - 1)) - 64)          # the columns whose mirror image lies on the detector
    ca = (axis2 - (w_opp - 1)) // 2
    nrows = min(256, R // 4)
    for a in ("rest1", "rest2"):
        for p in range(16, N // 2 - 80, max(1, N // 62)):
            for row_a in np.linspace(24, R - nrows - 24, 5).astype(int):
                add("opposite", 0, a, a, p, N // 2, w_opp, ca, axis2 - (w_opp - 1) - ca, nrows, row_a, mirror=True)
    hrows = lay._heights[(0, "helix")]
    Dh = int(round(hrows[N // 2 + 2000] - hrows[2000]))
    lo, hi = max(0, 40 - Dh), min(R - nrows, R - nrows - 40 - Dh)
    if hi >= lo:
        for p in range(16, lay.lengths[0]["helix"] - N // 2 - 64, max(1, N // 31)):
            for row_a in sorted({lo, hi}):
                add("helix opposite", 0, "helix", "helix", p, N // 2, w_opp, ca, axis2 - (w_opp - 1) - ca, nrows, row_a, mirror=True)
    # the second scan against the first at equal frames: the whole overlap, three bands of rows
    w_pair = _block_size(C - abs(lay.move) - 28)
    pc = max(0, -lay.move) + 14
    pair_reqs = {}
    for a in ("rest1", "helix", "rest2"):
        for p in range(16, min(lay.lengths[0][a], lay.lengths[1][a]) - 64, max(1, N // 62 if a != "helix" else N // 31)):
            for row_a in np.linspace(24, R - nrows - 24, 3).astype(int):
                pair_reqs.setdefault(a, []).append(dict(label="pair", scan_a=0, scan_b=1, stretch_a=lay.stretches[0][a], stretch_b=lay.stretches[1][a], frame_a=int(p), frame_hint=0, angle_offset_deg=0.0, mirror=False,
                                                        frames=32, rows=int(nrows), cols=int(w_pair), row_a=int(row_a), col_a=int(pc), col_b=int(pc + lay.move), frame_shear=0.0))
    rows = []

    def run(reqs_, meta_):
        for i0 in range(0, len(reqs_), 250):
            for mm, r in zip(meta_[i0:i0 + 250], _measure_blocks(session, reqs_[i0:i0 + 250], lay.row_sign, device)):
                if not (r["planned"] and r["ok"] and r["correlation"] > 0.3):
                    continue
                if lay.is_blocked(mm["scan"], mm["a"], r["frame_a"], r["row_a"] - 8, r["row_a"] + r["rows"] + 8) or \
                   lay.is_blocked(mm["scan_b"], mm["b"], r["frame_b"], r["row_a"] + r["row_offset"] - 8, r["row_a"] + r["row_offset"] + r["rows"] + 8):
                    continue
                d = dict(mm)
                d.update(r)
                vA = r["row_a"] + (r["rows"] - 1) / 2
                d["q"] = r["dq"] - r["stated_dq"]
                d["y"] = (r["D"] - r["stated_D"]) - r["slope_b"] * d["q"] - tau / period * (r["slope_b"] * (vA + r["row_offset"]) - r["slope_a"] * vA)
                d["t1"] = start[mm["scan"]][mm["a"]] + (r["frame_a"] + 16) / N
                d["t2"] = start[mm["scan_b"]][mm["b"]] + (r["frame_b"] + 16) / N
                d["th"] = np.radians(r["angle_a_deg"])
                d["vA"] = vA
                d["X"] = 0.5 * (r["col_a"] + r["col_b"] + r["cols"] - 1 - r["du"])
                for key, sc_, st_, fr in (("Pa", mm["scan"], mm["a"], r["frame_a"]), ("Pb", mm["scan_b"], mm["b"], r["frame_b"])):   # the encoder's rows from the helix's first frame
                    d[key] = float(lay._heights[(sc_, st_)][int(fr) + 16] - lay._heights[(sc_, "helix")][0])
                rows.append(d)

    for mm in meta:
        mm["scan_b"] = mm["scan"]
    _set_map(session, m)
    run(reqs, meta)
    n_planned = len(reqs)
    for a, rq in pair_reqs.items():
        _set_map(session, with_line(m, *lines[a], about=overlap_centre(lay) + lay.move), 1)
        run(rq, [dict(kind="pair", scan=0, scan_b=1, a=a, b=a) for _ in rq])
        n_planned += len(rq)
    _set_map(session, m)
    log(f"  {len(rows)} of {n_planned} comparisons")
    get = lambda kind, scan, key, a=None: np.array([r[key] for r in rows if r["kind"] == kind and r["scan"] == scan and (a is None or r["a"] == a)])
    rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2))) if len(v) else None
    out = dict(turns_at_start=start, comparisons=len(rows))
    for scan in (0, 1):
        S = out[f"scan {scan}"] = {}
        for a in ("rest1", "rest2"):
            D, y = get("resthelix", scan, "stated_D", a), get("resthelix", scan, "y", a)
            if len(D) >= 8:
                sc, off = fit_line(D, y)
                qa = get("resthelix", scan, "q", a) + tau / period * get("resthelix", scan, "row_offset", a)     # the helix's angle beyond the rows' timing [frames]
                qs, q0 = fit_line(D, qa)
                S[a] = dict(offset_rows=float(off), scale_error=float(sc), left_rows=float(np.std(y - np.polyval([sc, off], D))), blocks=len(D),
                            angle_frames=float(q0), angle_frames_per_row=float(qs), angle_left_frames=float(np.std(qa - np.polyval([qs, q0], D))),
                            table=[[float(f_) + 16.0, float(d_), float(y_), float(q_)] for f_, d_, y_, q_ in zip(get("resthelix", scan, "frame_b", a), D, y, qa)])   # helix frame, encoder's rows from the rest turn, image minus encoder, angle
        D, y, q, ro = get("turn", scan, "stated_D"), get("turn", scan, "y"), get("turn", scan, "q"), get("turn", scan, "row_offset")
        S["helix"] = dict(scale_error=float(np.mean(y / D)), rows_per_turn=float(np.mean(D + y)), angle_frames_per_turn=float(np.mean(q + tau / period * ro)), blocks=len(D),
                          table=[[float(f_) + 16.0, float(d_), float(y_), float(a_)] for f_, d_, y_, a_ in zip(get("turn", scan, "frame_a"), D, y, q + tau / period * ro)])   # first frame, encoder's rows to a turn on, image minus encoder, angle [frames]
        same = [r for r in rows if r["scan"] == scan and r["kind"] in ("resthelix", "turn")]
        h = 0.5 if scan == 0 else 1.0
        X, K = _nodes(*(np.array([r[k] for r in same]) for k in ("t1", "t2", "th")), 7, h, total)
        X[:, 2 * K] = [r["stated_D"] for r in same]
        X[:, 2 * K + 1] = [float(r["kind"] == "resthelix" and r["a"] == "rest1") for r in same]
        X[:, 2 * K + 2] = [float(r["kind"] == "resthelix" and r["a"] == "rest2") for r in same]
        y = np.array([r["du"] for r in same])
        kind = ["same angle"] * len(same)
        if scan == 0:
            opp = [r for r in rows if r["kind"] in ("opposite", "helix opposite")]
            Xo, _ = _nodes(*(np.array([r[k] for r in opp]) for k in ("t1", "t2", "th")), 7, h, total)
            Xo[:, :2 * K] *= -0.5
            for j, a in enumerate(("rest1", "helix", "rest2")):
                Xo[:, 2 * K + 3 + j] = [float(r["a"] == a) for r in opp]
            # The axis' column at row v of a frame at travel P: c0 + roll (v - middle) + (travel's direction - roll) P.
            # Opposite views inside the helix lie at two travels: their mirror column holds the direction at their
            # mean and the roll at the first one's (the row is the first one's).
            Xo[:, 2 * K] = [0.5 * (r["Pa"] + r["Pb"]) if r["a"] == "helix" else 0.0 for r in opp]
            Xo[:, 2 * K + 6] = [r["vA"] - (R - 1) / 2 - (r["Pa"] if r["a"] == "helix" else 0.0) for r in opp]
            X, y, kind = np.concatenate([X, Xo]), np.concatenate([y, [r["X"] - lay.axis for r in opp]]), kind + ["opposite"] * len(opp)
        kind = np.array(kind)
        use = np.nonzero(np.abs(X).sum(0) > 0)[0]

        solve = lambda sel, cols_: _least_squares(X, y, cols_, sel)
        every = np.ones(len(y), bool)
        b, b0 = solve(every, use), solve(every, use[use >= 2 * K])
        half = np.arange(len(y)) % 2 == 0
        held, held0 = y[~half] - X[~half] @ solve(half, use), y[~half] - X[~half] @ solve(half, use[use >= 2 * K])
        S["sideways"] = dict(node_spacing_turns=h, nodes_px=[[float(b[2 * k]), float(b[2 * k + 1])] for k in range(K)], travel_px_per_row=float(b[2 * K]), rest1_px=float(b[2 * K + 1]), rest2_px=float(b[2 * K + 2]),
                             left_px={k_: [rms((y - X @ b0)[kind == k_]), rms((y - X @ b)[kind == k_])] for k_ in sorted(set(kind))},
                             left_px_at_positions_left_out={k_: [rms(held0[kind[~half] == k_]), rms(held[kind[~half] == k_])] for k_ in sorted(set(kind))})
        if scan == 0:
            # rest1, rest2: the axis' column at the middle row in those turns; helix: the same at the helix's first frame
            P_end = float(lay._heights[(0, "rest2")][0] - lay._heights[(0, "helix")][0])
            mc = dict(rest1=lay.axis + float(b[2 * K + 3]), helix=lay.axis + float(b[2 * K + 4]), rest2=lay.axis + float(b[2 * K + 5]), per_row=float(b[2 * K + 6]))
            mc["closure_px"] = dict(rest1=float(b[2 * K + 1]) - (mc["helix"] - mc["rest1"]), rest2=float(b[2 * K + 2]) - (mc["helix"] - mc["rest2"] + (float(b[2 * K]) - mc["per_row"]) * P_end))   # same-angle views against opposite views
            S["sideways"]["mirror_column"] = mc
    ho = [r for r in rows if r["kind"] == "helix opposite"]
    if len(ho) >= 16:
        th = np.array([r["th"] for r in ho])
        t = np.array([r["t1"] for r in ho])
        H = np.stack([np.ones_like(th), t - t.mean(), np.array([r["vA"] for r in ho]) - (R - 1) / 2, np.cos(th), np.sin(th)], 1)
        bz = _least_squares(H, np.array([r["y"] for r in ho]))
        A = dict(z_cos=float(-bz[3] / 2), z_sin=float(-bz[4] / 2), amplitude_rows=float(np.hypot(bz[3], bz[4]) / 2), blocks=len(ho), drift_rows_per_turn={})
        for a in ("rest1", "rest2"):
            g = [r for r in rows if r["kind"] == "opposite" and r["a"] == a]
            if len(g) >= 8:
                th = np.array([r["th"] for r in g])
                dv = np.array([r["dv"] for r in g])
                left = dv - (bz[3] * np.cos(th) + bz[4] * np.sin(th))
                A["drift_rows_per_turn"][a] = float(2 * left.mean())
                A[f"{a}_opposite_height_sd"] = [float(dv.std()), float(left.std())]
        out["along_axis"] = A
    pr = [r for r in rows if r["kind"] == "pair"]
    if len(pr) >= 60:
        # stretch, turns from the start, angle [rad], middle row, columns, rows, angle [frames]
        table = [[r["a"], float(r["t1"]), float(r["th"]), float(r["vA"]), float(r["du"]), float(r["y"]), float(r["q"])] for r in pr]
        out["second_scan"] = dict(second_scan_fit(table, R, total), window_columns=[int(pc), int(pc + w_pair - 1)], table=table)
    return out


# ---------------------------------------------------------------- the pose of every frame

def frame_poses(cal, angles_deg, heights_rows):
    """Where every frame of the two scans holds the sample, from the step's result `cal` (with its motion stage).

    angles_deg[s], heights_rows[s]: the record's angle (unwrapped) and height (rows: the layout's row_sign x height
    / pixel) of every frame of scan s, the stretches one after the other (rest1, helix, rest2).

    Returns per scan a dict of arrays with one value per frame. The ray at u columns from the rotation axis (towards
    higher columns) in the slice at height h (rows) is recorded by frame k at the position (X, Y) of the scan's grid,
        Y = h + row[k] + row_per_col[k] u,        X = axis[k] + u + col_per_row[k] (Y - middle row),
    which is the detector pixel (c, v) with
        c + Eu(c, v) + col_scale[k] (c - scale_col) = X,        v + Ev(c, v) + row_scale[k] v = Y
    (Eu, Ev: the step's map), recorded at the angle angle_deg[k] + angle_deg_per_row[k] (v - middle row).

    The first scan is placed by what its own redundant views measured: the helix against the encoder (a line in the
    turn, from views a turn apart), the rest turns against the helix at equal angle, the axis' column of each
    stretch with the rotating body's place against it, the height once per turn. The second scan is the first one's
    model at the same frame plus what the comparison of the two scans at equal frames measured. Terms the step
    measured and this does not use are listed under "not_applied"."""
    from . import _core
    C = _core.GeometryCalibration(_text(cal))          # helirec::geometry::Calibration::frame_poses
    ang = [np.ascontiguousarray(a_, np.float64) for a_ in angles_deg]
    H = [np.ascontiguousarray(h_, np.float64) for h_ in heights_rows]
    try:
        poses, not_applied = C.frame_poses(ang[0], ang[1], H[0], H[1])
    except RuntimeError as e:
        raise ValueError(str(e)) from None
    return poses, json.loads(not_applied)


def reconstruction_terms(pose, heights_rows, layout, rot_center, column_offset, z_readjustment=0.0, row_origin=0.0):
    """One scan's poses (frame_poses) as the reconstruction's per-projection terms (helical.hpp, ScanTrajectory).

    The reconstruction puts the merged column m at m - rot_center columns from the axis and at the scan's column
    m - column_offset, and the slice h at the row h - (z_pix[k] z_translation_scale + z_readjustment +
    row_offset[k]), z_pix the record's height in pixels from the scan's first projection. row_origin: rows added to
    every pose (the slice numbering's origin). Everything the poses hold beyond the record goes into the terms, so
    z_translation_scale is 1."""
    from . import _core
    f = lambda key: np.ascontiguousarray(pose[key], np.float64)
    return _core.geometry_reconstruction_terms(f("row"), f("row_per_col"), f("axis"), f("col_per_row"), f("col_scale"), f("row_scale"), f("angle_deg"), float(pose["scale_col"]),
                                               np.ascontiguousarray(heights_rows, np.float64), float(layout["row_sign"]), int(layout["rows"]), float(rot_center),
                                               float(column_offset), float(z_readjustment), float(row_origin))


def merged_layout(cal, poses, heights_rows, quantum=0.25):
    """The frame the two scans are merged in, from the poses (frame_poses): one rotation axis in the first scan's
    columns and the second scan's place against the first, as the reconstruction lays its merged frame out
    (calibration.placement_from_vector) -> dict(cor_main_detector, vector=(rows, columns), measured=...).

    They are the medians over the helix of the axis' column in each scan and of the second scan's rows against the
    first's at equal frames, rounded to `quantum` pixels. The frame is a nominal one: every projection's own axis
    and row are in its terms (reconstruction_terms), which are counted from this frame, so the rounding moves
    nothing; it fixes the slices' grid, and it is what mode legacy_reference, which has no terms per projection,
    places the scans by (one axis and one vector, on the grid the reference's searches had)."""
    from . import _core
    f = lambda x: np.ascontiguousarray(x, np.float64)
    return _core.GeometryCalibration(_text(cal)).merged_layout(f(poses[0]["axis"]), f(poses[0]["row"]), f(poses[1]["axis"]), f(poses[1]["row"]),
                                                               f(heights_rows[0]), f(heights_rows[1]), float(quantum))


def rolling_shutter_of(cal):
    """The rows' timing as the reconstruction takes it: the time from the first stored row to the last, in frame periods."""
    from . import _core
    return _core.GeometryCalibration(_text(cal)).rolling_shutter


def overlap_centre_of(layout):
    """overlap_centre for the layout as the step saves it: the middle of the columns both scans hold, in the first scan's columns."""
    from . import _core
    return _core.geometry_overlap_centre(int(layout["cols"]), float(layout["coarse_move"]))


# ---------------------------------------------------------------- the step

def calibrate_geometry(scans, row_time=None, device="auto", order=3, passes=6, tolerance=0.02, threads=8, motion=True, log=print, patch=0):
    """The geometry calibration of a two-scan helical acquisition. scans: the two master files, the scan with the
    rotation axis inside the detector's field first. row_time: seconds per stored row (signed), if it is known for
    this camera configuration; otherwise it is measured from a refill, and it is an error if no stretch holds one.
    patch: 0 measures the plan's blocks whole; N cuts them into patches of N x N (see `measure`), with which the
    first pass, on the frames as stored, gives the map and the second confirms it. Returns a dict (see `save`)."""
    from . import _core

    t_start = time.time()
    session = _core.CalibrationSession(list(scans), threads)
    log("layout")
    lay = read_layout(session, device, log)
    lay._heights = {(k, name): lay.row_sign * session.series(k, st, "height_mm") / (lay.pixel_um * 1e-3) for k in (0, 1) for name, st in lay.stretches[k].items()}
    period = float(np.median(session.series(0, lay.stretches[0]["helix"], "timer_period")))
    timing = None
    if row_time is None:
        log("rows' timing")
        timing = row_time_from_refill(session, lay, log)
        if timing is None:
            raise ValueError("no stretch holds a refill to measure the rows' timing from: give row_time (seconds per stored row) for this camera configuration")
        row_time = timing["seconds_per_row"]
    m = empty_map(lay.rows, lay.cols, order)
    lines = {name: [0.0, 0.0] for name in ("rest1", "helix", "rest2")}
    history = []
    log("map")
    for it in range(passes):
        rows = measure(session, lay, m, lines, device, patch)
        f = fit(rows, lay, m, row_time, period)
        du, dv = _update_range(m, f)
        dl = max(abs(v_) for st in f["lines"].values() for v_ in st) * lay.cols / 2
        history.append(dict(accepted=f["n"], correction_px=du, correction_rows=dv, line_change_px=dl, left=f["left"], solver=f["solver"]))
        log(f"  pass {it + 1}: {f['n']} comparisons; the map is corrected by up to {du:.3f} px and {dv:.3f} rows, the second scan's lines by {dl:.3f}; left " +
            ", ".join(f"{k} {v_['px']:.3f} px / {v_['rows']:.3f} rows" for k, v_ in f["left"].items()))
        if max(du, dv, dl) < tolerance and it:
            break
        m["eu"] = [a + b for a, b in zip(m["eu"], f["eu"])]
        m["ev"] = [a + b for a, b in zip(m["ev"], f["ev"])]
        for st in lines:
            lines[st] = [lines[st][0] + f["lines"][st][0], lines[st][1] + f["lines"][st][1]]
    else:
        log(f"  not converged to {tolerance} in {passes} passes")
    # the poses of the last pass
    g = f["groups"]
    mean = lambda vals: float(np.mean(vals)) if len(vals) else None
    poses = dict(axis_roll_rad=f["psi"], rotation_about_beam_rad=f["rho"])
    poses["mirror_column"] = {a: mean([u / 2 for (k, s_, st, _), (u, v_) in g.items() if k == "opposite" and st == a]) for a in ("rest1", "rest2")}
    poses["second_scan"] = {a: dict(columns=lay.move + mean([u for (k, s_, st, _), (u, v_) in g.items() if k == "pair" and st == a]),
                                    rows_beyond_record=mean([v_ for (k, s_, st, _), (u, v_) in g.items() if k == "pair" and st == a]), line=lines[a]) for a in lines}
    heights = {}
    for scan in (0, 1):
        for a in ("rest1", "rest2"):
            sel = [(next(r["stated_D"] for r in rows if r["kind"] == "resthelix" and r["scan_a"] == scan and r["stretch_a"] == a and int(r["frame_a"]) == fr), v_)
                   for (k, s_, st, fr), (u, v_) in g.items() if k == "resthelix" and s_ == scan and st == a]
            if len(sel) >= 4:
                sc, off = fit_line([x for x, _ in sel], [y_ for _, y_ in sel])
                heights[f"scan {scan} {a}"] = dict(offset_rows=float(off), scale_error=float(sc))
        tn = [(v_, next(r["stated_D"] for r in rows if r["kind"] == "turn" and r["scan_a"] == scan and int(r["frame_a"]) == fr)) for (k, s_, st, fr), (u, v_) in g.items() if k == "turn" and s_ == scan]
        if tn:
            heights[f"scan {scan} helix"] = dict(scale_error=float(np.mean([v_ / d for v_, d in tn])), rows_per_turn=float(np.mean([v_ + d for v_, d in tn])))
    poses["heights"] = heights
    if motion:
        log("motion within a turn")
        poses["motion"] = measure_motion(session, lay, m, lines, row_time, period, device, log)
    return dict(step="geometry", scans=[str(x) for x in scans], seconds=round(time.time() - t_start, 1),
                layout=dict(rows=lay.rows, cols=lay.cols, frames=[dict(lay.lengths[k]) for k in (0, 1)], frames_per_turn=lay.per_turn, pixel_um=lay.pixel_um, row_sign=lay.row_sign, coarse_mirror_column=lay.axis, coarse_move=lay.move, window_columns=lay.width,
                            blocked_rows={f"scan {k} {name}": dict(frames=[int(x) for x, hit in zip(fr, mask) if hit.any()], first_row=[int(np.nonzero(hit)[0].min()) for hit in mask if hit.any()],
                                                                    last_row=[int(np.nonzero(hit)[0].max()) for hit in mask if hit.any()]) for (k, name), (fr, mask) in lay.blocked.items()}),
                row_time=dict(seconds_per_row=row_time, frame_period_s=period, measured=timing), map=m, poses=poses, passes=history, converged=bool(max(du, dv, dl) < tolerance),
                patch=int(patch), solver=f["solver"])


def save(cal, path):
    tmp = f"{path}.tmp{os.getpid()}"       # (written whole or not at all: a reconstruction may be reading the cache)
    with open(tmp, "w") as fh:
        json.dump(cal, fh, indent=2, default=float)
        fh.write("\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="helirec-geometry", description="The geometry calibration step: detector map and poses of a two-scan helical acquisition, measured from its own frames.")
    ap.add_argument("main", help="master file of the scan with the rotation axis inside the detector's field")
    ap.add_argument("companion", help="master file of the scan moved sideways")
    ap.add_argument("--out", required=True, help="where the calibration is written (JSON)")
    ap.add_argument("--row-time-us", type=float, default=None, help="time of a stored row within the frame, signed, if known for this camera configuration; measured from a refill otherwise")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--order", type=int, default=3, help="order of the map's polynomial")
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--patch", type=int, default=0, help="0: the plan's blocks registered whole (several passes); N: cut into patches of N x N, the frames of a block read once (try 128)")
    a = ap.parse_args(argv)
    cal = calibrate_geometry([a.main, a.companion], None if a.row_time_us is None else a.row_time_us * 1e-6, a.device, a.order, a.passes, patch=a.patch)
    save(cal, a.out)
    print(f"written to {a.out} ({cal['seconds']} s; {'converged' if cal['converged'] else 'NOT converged'})")


if __name__ == "__main__":
    main()
