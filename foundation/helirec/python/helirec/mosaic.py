"""The geometry calibration step of a mosaic: scans of one sample at several places on the rotation stage (tiles),
each a helix with the rotation axis inside the detector, measured from the acquisition itself.

    cal = calibrate_mosaic([master of the reference tile, masters of the other tiles ...])
    geometry.save(cal, "mosaic_geometry.json")

A sample displaced by d on the rotation stage is seen, in tile t at angle th and detector column c, along the
sample's line s = c - u_t, where u_t is where the sample's origin falls on the detector:

    u_t(f) = a_t + c1_t cos th + s1_t sin th + C_t(f)          H_t(f) = h_t + R_t(f)

a_t the axis column, (c1, s1) the tile's place on the stage (zero for the reference tile: the sample's frame is
the one in which that tile's axis is the origin), and C_t a course along the frames f: what the axis and the
stages do beside, smooth, piecewise linear between knots. H_t, the height of the sample against the record, is
a constant and a course likewise. The angle of a frame is the record's plus a course T(f) that all tiles share
(they run one trajectory on one stage) plus the tile's own yaw_t (its sample turned about the rotation axis
against the reference tile's: the stages' yaw as they travel, or a slip), and rows are exposed one after another,
row v at rho (v - v0) from the frame's angle:

    th = th_record(f) + T(f) + yaw_t + rho (v - v0)

Every comparison of two blocks that hold the same lines is then one equation in these, whichever tiles and views
it joins (E: the part of the detector's map the grid does not hold yet):

    columns, same view direction   cB - cA = u_B - u_A - (Eu(B) - Eu(A)) + psi (vB - vA)
    columns, opposite views        cA + cB = u_A + u_B - (Eu(A) + Eu(B)) + psi (vA + vB - 2 v0)
    rows                           y = H_B - H_A + scale D - (Ev(B) - Ev(A)) [+ chi (cA - cB), opposite views]
    frames                         step dq = T(fA) - T(fB) - rho (vB - vA) [+ fan (cB - cA), same view direction]

Between two tiles the places make u swing by the tiles' distance once per turn, so an error of the angle of
1e-6 rad moves the columns by 0.007 px: the columns of one tile against another are what measures rho and T.
A tile's yaw the columns cannot tell from the direction of its place: the frame shift between two tiles at
equal angle is what measures it (on Paris4 up to 0.12 frames, 0.5 px at the tiles' distance).

All of it is one model, written as equations over named unknowns and solved together by helirec's solver
(helirec.solve, helirec/calibration/solve.hpp): `fit` is where the model is stated, and it is the step's only fit.
What no comparison sees is left at zero by the fit: a course common to all tiles that changes sign over half a
turn (odd harmonics of the turn in the columns; the first of them is the choice of the sample's origin).

What the step finds, in this order (the stages of helirec.geometry, for any number of tiles):

  layout   the helix of each tile and its frames on disk, which way the rows run against the axis stage, a coarse
           axis column per tile, which tiles are neighbours, and their places from single frames;
  passes   the four kinds of comparison (a turn apart and opposite views within a tile, tile against tile at equal
           angle and against the other's opposite views) registered on the current map's grid with the current
           places and rows' timing, and fitted; repeated until a pass corrects the map by less than the tolerance.

The frames may be a partial copy (only what is on disk is read) and stored as JPEG 2000.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import geometry as G

KINDS = ("turn", "opposite", "pair", "pairopp")
NQ = 32                 # frames per block
SWAY_MARGIN = 130       # columns kept free beside a window of a swaying tile: the sway changes by a few px per frame


# ---------------------------------------------------------------- the layout

@dataclass
class Layout:
    rows: int
    cols: int
    per_turn: int
    pixel_um: float
    step_deg: float
    period_s: float
    source_mm: float
    labels: list
    helix: list                    # per tile: stretch index
    on_disk: list                  # per tile: (first, end) of the helix frames on disk
    angle: list = field(default_factory=list, repr=False)       # per tile: stated angle of every helix frame [deg]
    height: list = field(default_factory=list, repr=False)      # per tile: stated height in rows (row_sign z / pixel)
    row_sign: float = -1.0
    axis: list = field(default_factory=list)                    # coarse axis column per tile
    place: list = field(default_factory=list)                   # per tile: (u, v) on the stage [px], from the stage readings
    pairs: list = field(default_factory=list)                   # neighbours (k, j), k < j
    sway: dict = field(default_factory=dict)                    # (k, j) -> (amplitude px, phase deg) from single frames

    @property
    def fan(self):
        """frames per column: the direction of a column's ray against the column"""
        return self.pixel_um * 1e-3 / self.source_mm / np.radians(abs(self.step_deg))

    def angle_at(self, tile, frame):
        """stated angle [deg] at a frame number that need not be whole"""
        f0 = int(np.clip(np.floor(frame), 0, len(self.angle[tile]) - 2))
        return self.angle[tile][f0] + (frame - f0) * (self.angle[tile][f0 + 1] - self.angle[tile][f0])


def _profile(session, scan, st, frame, row0, nrows, cols, k=31):
    p = session.line_integrals(scan, st, np.array([frame], np.int64), np.array([row0], np.int32), nrows, 0, cols)[0].mean(0)
    return p - np.convolve(np.pad(p, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def _lag(a, b, least=600):
    """k for which the correlation of a[c] with b[c + k] over what overlaps is largest, that correlation, and the
    best elsewhere"""
    n = len(a)
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * n)) * np.fft.rfft(b, 2 * n), 2 * n)
    cc = np.concatenate([cc[n:], cc[:n]])
    ks = np.arange(-n, n)
    overlap = np.maximum(n - np.abs(ks), 1)
    score = cc / overlap / max(a.std() * b.std(), 1e-30)
    score[overlap < least] = 0
    best = int(np.argmax(score))
    rest = score.copy()
    rest[max(0, best - 40):best + 41] = 0
    return int(ks[best]), float(score[best]), float(rest.max())


def read_layout(session, n_tiles, device="auto", log=print):
    info = session.info(0)
    rows, cols = int(info["rows"]), int(info["cols"])
    helix, on_disk, angle, zmm = [], [], [], []
    for k in range(n_tiles):
        st = [x for x in session.stretches(k) if x["role"] == "projection" and x["latched"] and x["helical"]]
        if len(st) != 1:
            raise ValueError(f"the mosaic step needs one helix per tile; the record of tile {k} holds {len(st)}")
        held = [tuple(x) for x in st[0]["on_disk"]]
        if not held:
            raise ValueError(f"no frame of the helix of tile {k} is on disk")
        helix.append(st[0]["index"])
        on_disk.append(max(held, key=lambda ab: ab[1] - ab[0]))
        angle.append(session.series(k, helix[k], "angle_deg"))
        zmm.append(session.series(k, helix[k], "height_mm"))
    step = float(np.median(np.diff(angle[0])))
    lay = Layout(rows=rows, cols=cols, per_turn=int(round(360.0 / abs(step))), pixel_um=float(info["pixel_um"]), step_deg=step,
                 period_s=float(np.median(session.series(0, helix[0], "timer_period"))), source_mm=float(session.number(0, "scan/source_sample_distance")),
                 labels=[session.info(k)["label"] for k in range(n_tiles)], helix=helix, on_disk=[(int(a), int(b)) for a, b in on_disk], angle=angle)
    N = lay.per_turn
    for k, (a, b) in enumerate(lay.on_disk):
        if b - a < 2 * N + 4 * NQ:
            raise ValueError(f"tile {lay.labels[k]}: {b - a} helix frames on disk; the step needs more than two turns")
    first, end = max(a for a, _ in lay.on_disk), min(b for _, b in lay.on_disk)
    px_mm = lay.pixel_um * 1e-3
    # which way the rows run against the axis stage: the sense in which the helix a turn apart registers
    best = None
    for sign in (-1.0, 1.0):
        lay.row_sign = sign
        lay.height = [sign * z / px_mm for z in zmm]
        reqs = turn(lay, 0, np.linspace(first + 64, end - N - 64 - NQ, 4).astype(int))
        got = [r["correlation"] for r in session.measure(reqs, sign, device) if r["planned"] and r["ok"]]
        score = float(np.mean(got)) if got else 0.0
        if best is None or score > best[0]:
            best = (score, sign)
    if best[0] < 0.3:
        raise ValueError("the helix a turn apart does not register in either sense of the rows: the travel per turn must be less than the detector's height")
    lay.row_sign = best[1]
    lay.height = [lay.row_sign * z / px_mm for z in zmm]
    # a coarse axis column per tile: single frames half a turn apart, one reversed, on rows half a pitch apart
    for k in range(n_tiles):
        cc = 0
        for p in np.linspace(lay.on_disk[k][0] + 200, lay.on_disk[k][1] - N // 2 - 200, 6).astype(int):
            q = p + N // 2
            off = int(round(lay.height[k][q] - lay.height[k][p]))
            r0 = int(np.clip((rows - 256 - off) // 2, max(0, -off), min(rows - 256, rows - 256 - off)))
            a = _profile(session, k, helix[k], int(p), r0, 256, cols)
            b = _profile(session, k, helix[k], int(q), r0 + off, 256, cols)[::-1]
            c = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * cols)) * np.fft.rfft(b, 2 * cols), 2 * cols)
            c = np.concatenate([c[cols:], c[:cols]])
            ks = np.arange(-cols, cols)
            c = c / np.maximum(cols - np.abs(ks), 1)
            c[cols - np.abs(ks) < 300] = 0
            cc = cc + c
        lay.axis.append((cols - 1 - (int(np.argmax(cc)) - cols)) / 2.0)
    # the tiles' places by the stage, and which are neighbours: closer than twice what a tile sees from its axis
    mu, mv = info.get("sample_u", ""), info.get("sample_v", "")
    if not mu or not mv:
        raise ValueError("the record's tomo configuration names no sample stages on the rotation (sample_u, sample_v)")
    mm = [(session.positioner(k, helix[k], mu), session.positioner(k, helix[k], mv)) for k in range(n_tiles)]
    lay.place = [((u - mm[0][0]) / px_mm, (v - mm[0][1]) / px_mm) for u, v in mm]
    reach = max(max(a, cols - 1 - a) for a in lay.axis)
    lay.pairs = [(k, j) for k, j in itertools.combinations(range(n_tiles), 2) if np.hypot(lay.place[j][0] - lay.place[k][0], lay.place[j][1] - lay.place[k][1]) < 2 * reach - 600]
    if n_tiles > 1 and len({t for p in lay.pairs for t in p}) < n_tiles:
        raise ValueError("a tile has no neighbour that records the same lines")
    # the sway of each pair from single frames at equal frame number: lag(th) = A cos(th - phi)
    frames = (first + 400 + np.arange(144) * (N / 144.0)).astype(int)
    prof = {(k, int(f)): _profile(session, k, helix[k], int(f), rows // 2 - 128, 256, cols) for k in sorted({t for p in lay.pairs for t in p}) for f in frames}
    for k, j in lay.pairs:
        D = float(np.hypot(lay.place[j][0] - lay.place[k][0], lay.place[j][1] - lay.place[k][1]))
        th, d, good = [], [], []
        for f in frames:
            lag, score, other = _lag(prof[(k, int(f))], prof[(j, int(f))])
            th.append(np.radians(angle[k][f])), d.append(lag), good.append(score > 0.3 and score > 1.5 * other)
        th, d, good = np.array(th), np.array(d, float), np.array(good)
        if good.sum() < 12:
            raise ValueError(f"tiles {lay.labels[k]} and {lay.labels[j]} do not register against each other in single frames")
        best = None
        for phi in np.radians(np.arange(0, 360, 0.25)):          # the phase, at the stage's distance
            res = d[good] - D * np.cos(th[good] - phi)
            inl = np.abs(res - np.median(res)) < 60
            if best is None or inl.sum() > best[0]:
                best = (int(inl.sum()), inl)
        idx = np.flatnonzero(good)[best[1]]
        for _ in range(3):
            X = np.stack([np.cos(th[idx]), np.sin(th[idx]), np.ones(len(idx))], 1)
            c, *_ = np.linalg.lstsq(X, d[idx], rcond=None)
            res = d[idx] - X @ c
            idx = idx[np.abs(res) < max(3.0, 3 * res.std())]
        lay.sway[(k, j)] = (float(np.hypot(c[0], c[1])), float(np.degrees(np.arctan2(c[1], c[0])) % 360))
    log(f"  {n_tiles} tiles of {rows} x {cols} px, {N} frames per turn; frames on disk {first}..{end - 1}; row sense {lay.row_sign:+.0f}; axis columns about "
        + ", ".join(f"{a:.1f}" for a in lay.axis) + f"; {len(lay.pairs)} pairs of neighbours")
    return lay


# ---------------------------------------------------------------- the time of a row, from a refill

def row_time_from_refill(session, lay, log=print):
    """Seconds per stored row (signed) from the beam dips of a refill that lies in a tile's frames on disk; None if
    none does. As helirec.geometry measures it (geometry.fit_refill_shots), but with no air to read the beam in:
    a row's level is the mean of all its columns, compared with the same row a few frames before and after."""
    for k, label in enumerate(lay.labels):
        st = lay.helix[k]
        cur, when = session.series(k, st, "monitor_current"), session.series(k, st, "monitor_epoch")
        ok = np.isfinite(cur) & np.isfinite(when)
        cur, when = cur[ok], when[ok]
        epoch = session.series(k, st, "epoch_trig")
        a, b = lay.on_disk[k]
        rises = when[1:][np.diff(cur) > 0.05]
        rises = rises[(rises > epoch[a] + 0.5) & (rises < epoch[b - 1] - 0.5)]
        if len(rises) < 3:
            continue
        f_lo, f_hi = max(a, int(np.searchsorted(epoch, rises.min() - 8.0))), min(b - 20, int(np.searchsorted(epoch, rises.max() + 8.0)))
        if f_hi - f_lo < 300:
            continue
        pre = min(400, f_lo - a)                       # frames on disk before the refill, if the copy has any
        frames = np.arange(f_lo - pre, min(b, f_hi + 600), dtype=np.int64)
        dark = session.dark(k).mean(1)
        x = np.empty((len(frames), lay.rows))
        for i0 in range(0, len(frames), 96):
            raw = session.read_rows(k, st, frames[i0:i0 + 96], 0, lay.rows)
            x[i0:i0 + 96] = np.log(np.clip(raw.mean(2, dtype=np.float64) - dark, 1, None))
        E = float(np.median(session.series(k, st, "timer_delta")))
        got = G.fit_refill_shots(x, lay.rows, lay.period_s, E, pre)
        if got is None:
            continue
        best, se, shots, noise = got
        log(f"  refill in {label}, frames {int(frames[0])}..{int(frames[-1])}: {len(shots)} shots on a raster of {best.x[3] * 1e3:.3f} ms; "
            f"row time {best.x[0] * 1e6:+.3f} +- {se * 1e6:.3f} us per stored row (a row is exposed for {abs(best.x[1]) * 1e3:.2f} ms decay; gate {E * 1e3:.1f} ms)")
        return dict(seconds_per_row=float(best.x[0]), se=se, tile=label, frames=[int(frames[0]), int(frames[-1])], shots=int(len(shots)), raster_s=float(best.x[3]),
                    band_noise=noise, period_s=lay.period_s, gate_s=E)
    return None


# ---------------------------------------------------------------- the model

KNOT_FRAMES = 500       # frames between the knots of a course
TERMS = ("yaw", "depth", "fan", "shared", "tilt", "common", "field", "magnify")      # terms of the model beyond the first form (fit); the step uses "yaw"
COMMON_OWN = 5.0        # with "common": how much harder a tile's own course is held to zero than the shared one
FIELD_NODES = (9, 13)   # with "field": nodes of the detector field along the rows and along the columns


def knots_of(lay):
    """frame numbers of the courses' knots: every KNOT_FRAMES from before the first frame on disk to after the last"""
    lo, hi = min(a for a, _ in lay.on_disk), max(b for _, b in lay.on_disk)
    return np.arange(lo - KNOT_FRAMES, hi + 2 * KNOT_FRAMES, KNOT_FRAMES, dtype=float)


def hat(knots, frame):
    """(index of the knot at or before `frame`, weights on it and the next): piecewise linear between knots"""
    x = (np.asarray(frame, float) - knots[0]) / (knots[1] - knots[0])
    i = np.clip(np.floor(x).astype(int), 0, len(knots) - 2)
    w = x - i
    return i, 1.0 - w, w


def course(knots, values, frame):
    i, w0, w1 = hat(knots, frame)
    return values[..., i] * w0 + values[..., i + 1] * w1


def start_model(lay, order=3, row_time=0.0):
    """Axis columns and places as the layout found them, no courses, an empty map, and the rows' timing if given
    [s per row]."""
    n = len(lay.labels)
    K = knots_of(lay)
    d = np.zeros((n, 2))
    if lay.sway:           # places from the pairs' sways, the reference tile at the origin
        X, y = [], []
        for (k, j), (A, phi) in lay.sway.items():
            for comp, value in ((0, A * np.cos(np.radians(phi))), (1, A * np.sin(np.radians(phi)))):
                row = np.zeros(2 * n)
                row[2 * j + comp], row[2 * k + comp] = 1, -1
                X.append(row), y.append(value)
        sol, *_ = np.linalg.lstsq(np.array(X)[:, 2:], np.array(y), rcond=None)
        d[1:] = sol.reshape(n - 1, 2)
    return dict(knots=K, a=np.array(lay.axis, float), d=d, C=np.zeros((n, len(K))), h=np.zeros(n), R=np.zeros((n, len(K))), T=np.zeros(len(K)), yaw=np.zeros(n),
                map=G.empty_map(lay.rows, lay.cols, order), rho=row_time / lay.period_s * np.radians(abs(lay.step_deg)), scale=0.0, psi=0.0, chi=0.0, fan=1.0)


def sway_of(model, k, j):
    """(amplitude px, phase deg) of where tile j's content lies against tile k's at equal angle"""
    c, s = model["d"][j] - model["d"][k]
    return float(np.hypot(c, s)), float(np.degrees(np.arctan2(s, c)) % 360)


def u_of(model, t, th, frame):
    """where the sample's origin falls on tile t's detector at angle th [rad] and frame number `frame`"""
    return model["a"][t] + model["d"][t, 0] * np.cos(th) + model["d"][t, 1] * np.sin(th) + course(model["knots"], model["C"][t], frame)


def h_of(model, t, frame):
    return model["h"][t] + course(model["knots"], model["R"][t], frame)


# ---------------------------------------------------------------- the comparisons

def bands_for(lay, offset_rows, n, rows=256, margin=40):
    """n first rows of A's block for which B's block, offset_rows higher, also lies on the detector with its margins"""
    lo, hi = max(0, -offset_rows) + margin, min(lay.rows - rows, lay.rows - rows - offset_rows) - margin
    if hi < lo:
        return []
    return [int((lo + hi) / 2)] if n == 1 or hi - lo < 100 else [int(x) for x in np.linspace(lo, hi, n)]


def turn(lay, k, frames, width=1024, n_cols=3, rows=32):
    out = []
    for p in frames:
        for ca in np.linspace(13, lay.cols - width - 13, n_cols).astype(int):
            out.append(dict(label="turn", scan_a=k, scan_b=k, stretch_a=lay.helix[k], stretch_b=lay.helix[k], frame_a=int(p), frame_hint=lay.per_turn, frames=NQ, rows=rows, cols=width,
                            row_a=-1, col_a=int(ca), col_b=int(ca)))
    return out


def opposite(lay, k, frames, axis, hint, n_bands=3, width=512, n_cols=2):
    axis2 = int(round(2 * axis))
    lo, hi = max(0, axis2 - (width - 1) - (lay.cols - width)), min(lay.cols - width, axis2 - (width - 1))
    out = []
    if hi - lo < 64:
        return out
    for p in frames:
        q = p + hint + NQ // 2
        if q < 0 or q >= len(lay.height[k]):
            continue
        for row_a in bands_for(lay, int(round(lay.height[k][q] - lay.height[k][p + NQ // 2])), n_bands):
            for ca in np.linspace(lo + 24, hi - 24, n_cols).astype(int):
                out.append(dict(label="opposite", scan_a=k, scan_b=k, stretch_a=lay.helix[k], stretch_b=lay.helix[k], frame_a=int(p), frame_hint=int(hint), angle_offset_deg=180.0, mirror=True,
                                frames=NQ, rows=256, cols=width, row_a=int(row_a), col_a=int(ca), col_b=int(axis2 - (width - 1) - ca), frame_shear=2 * lay.fan))
    return out


def _sway_fields(model, k, j):
    A, phi = sway_of(model, k, j)
    return dict(sway_px=A, sway_phase_deg=phi, sway_row_deg=float(np.degrees(model["rho"])))


def pair(lay, model, k, j, frames, bands=(150, 1056, 1960), width=1024, n_cols=3):
    """tile k against tile j at equal angle, where both hold a window"""
    A, phi = sway_of(model, k, j)
    out = []
    for p in frames:
        L = A * np.cos(np.radians(lay.angle[k][p + NQ // 2] - phi))
        lo, hi = max(0.0, -L) + SWAY_MARGIN, min(lay.cols - width, lay.cols - width - L) - SWAY_MARGIN
        if hi - lo < 0:
            continue
        cols_a = np.unique(np.linspace(lo, hi, n_cols).astype(int)) if hi - lo > 200 else [int((lo + hi) / 2)]
        for row_a in bands:
            if row_a + 256 + 40 > lay.rows:
                continue
            for ca in cols_a:
                out.append(dict(label="pair", scan_a=k, scan_b=j, stretch_a=lay.helix[k], stretch_b=lay.helix[j], frame_a=int(p), frame_hint=0, frames=NQ, rows=256, cols=width,
                                row_a=int(row_a), col_a=int(ca), col_b=int(ca), **_sway_fields(model, k, j)))
    return out


def pairopp(lay, model, k, j, frames, hint, n_bands=2, width=1024, n_cols=3):
    """tile k at an angle against tile j half a turn on (hint frames further), j's columns reversed"""
    A, phi = sway_of(model, k, j)
    axis2 = int(round(model["a"][k] + model["a"][j]))
    out = []
    for p in frames:
        q = p + hint + NQ // 2
        if q < 0 or q >= len(lay.angle[j]):
            continue
        S = A * np.cos(np.radians(lay.angle[j][q] - phi))               # j's columns are read S further on, at j's angle
        lo = max(0.0, axis2 - (width - 1) + S - (lay.cols - width)) + SWAY_MARGIN
        hi = min(lay.cols - width, axis2 - (width - 1) + S) - SWAY_MARGIN
        if hi - lo < 0:
            continue
        cols_a = np.unique(np.linspace(lo, hi, n_cols).astype(int)) if hi - lo > 200 else [int((lo + hi) / 2)]
        for row_a in bands_for(lay, int(round(lay.height[j][q] - lay.height[k][p + NQ // 2])), n_bands):
            for ca in cols_a:
                out.append(dict(label="pairopp", scan_a=k, scan_b=j, stretch_a=lay.helix[k], stretch_b=lay.helix[j], frame_a=int(p), frame_hint=int(hint), angle_offset_deg=180.0, mirror=True,
                                frames=NQ, rows=256, cols=width, row_a=int(row_a), col_a=int(ca), col_b=int(axis2 - (width - 1) - ca), frame_shear=2 * lay.fan, **_sway_fields(model, k, j)))
    return out


def plan(lay, model, angles=24, in_tile=KNOT_FRAMES):
    """The comparisons of one pass, placed from the layout and the current model. in_tile: frames between the
    positions of the comparisons within a tile (they are what follows a tile's courses along the frames: one window
    per position, the row band and the column going round)."""
    N = lay.per_turn
    reqs = []
    for k in range(len(lay.labels)):
        a, b = lay.on_disk[k]
        for i, p in enumerate(np.arange(a + 64, b - N - 64 - NQ, in_tile)):
            reqs.append(turn(lay, k, [int(p)], n_cols=3)[i % 3])
        for hint, (lo, hi) in ((N // 2, (a + 64, b - N // 2 - 64 - NQ)), (-N // 2, (a + N // 2 + 64, b - 64 - NQ))):
            for i, p in enumerate(np.arange(lo, hi, in_tile)):
                got = opposite(lay, k, [int(p)], model["a"][k], hint)
                if got:
                    reqs.append(got[(i * 5) % len(got)])
    for k, j in lay.pairs:
        a, b = max(lay.on_disk[k][0], lay.on_disk[j][0]), min(lay.on_disk[k][1], lay.on_disk[j][1])
        mid = (a + b) // 2
        frames = np.linspace(mid - N // 2, mid + N // 2, angles, endpoint=False).astype(int)      # one turn, half a turn on disk on either side
        frames = frames[(frames - N // 2 - 16 >= a) & (frames + N // 2 + NQ + 16 < b)]
        reqs += pair(lay, model, k, j, frames)
        for hint in (N // 2, -N // 2):
            reqs += pairopp(lay, model, k, j, frames, hint)
            reqs += pairopp(lay, model, j, k, frames, hint)
    return reqs


def measure(session, lay, model, reqs, device="auto", log=None, chunk=400, patch=0):
    """The requests registered on the grid of the model's map, in the order of their frames (blocks at one frame
    position read the same chunks, which the reader keeps decoded). Returns the accepted ones with the request's
    placement beside the measurement. patch: 0 registers each request's block as one; N cuts it into patches of
    N x N, each registered and returned as a comparison of its own (CalibrationSession.measure_patches: the block's
    frames are read once)."""
    G._set_map(session, model["map"])
    order = sorted(range(len(reqs)), key=lambda i: (reqs[i]["frame_a"], reqs[i]["row_a"], reqs[i]["scan_a"], reqs[i]["scan_b"]))
    out = []
    t0 = time.time()
    for i0 in range(0, len(order), chunk):
        part = [reqs[i] for i in order[i0:i0 + chunk]]
        if patch:
            got = [(part[r["request"]], r) for r in session.measure_patches(part, lay.row_sign, int(patch), int(patch), device) if r["patch_i"] >= 0]
        else:
            got = zip(part, session.measure(part, lay.row_sign, device))
        for q, r in got:
            if not (r["planned"] and r["ok"] and r["correlation"] > 0.3):
                continue
            d = dict(kind=q["label"], ka=q["scan_a"], kb=q["scan_b"], hint=q.get("frame_hint", 0), sway_px=q.get("sway_px", 0.0), sway_phase_deg=q.get("sway_phase_deg", 0.0),
                     sway_row_deg=q.get("sway_row_deg", 0.0))
            d.update({k: v for k, v in r.items() if k != "label"})
            out.append(d)
        if log:
            log(f"    {min(i0 + chunk, len(reqs))} of {len(reqs)} comparisons ({time.time() - t0:.0f} s)")
    return out


# ---------------------------------------------------------------- the fit

def describe(lay, model, rows, grid=None):
    """Per comparison, the quantities the fit is written in: tiles, the middle of each block on the (ideal)
    detector, the frame and the angle of each block, and the measured columns, rows and frames."""
    m = grid or model["map"]
    v0 = m["v0"]
    K = model["knots"]
    step = np.radians(abs(lay.step_deg))
    out = []
    for r in rows:
        ka, kb, mirror = r["ka"], r["kb"], r["kind"] in ("opposite", "pairopp")
        fa = r["frame_a"] + (r["frames"] - 1) / 2
        fb = r["frame_b"] + (r["frames"] - 1) / 2
        va = r["row_a"] + (r["rows"] - 1) / 2
        vb = va + r["row_offset"]
        # the angle of each block's rows by the model: A's, and B's at the frame the registration matched A's to
        yaw = model.get("yaw")
        tha = np.radians(lay.angle_at(ka, fa)) + course(K, model["T"], fa) + model["rho"] * (va - v0) + (yaw[ka] if yaw is not None else 0.0)
        thb = np.radians(lay.angle_at(kb, fb + r["dq"])) + course(K, model["T"], fb) + model["rho"] * (vb - v0) + (yaw[kb] if yaw is not None else 0.0)
        # the sway the session read B at, at B's matched frame and the block's middle row
        S0 = r["sway_px"] * np.cos(np.radians(lay.angle_at(kb, fb + r["dq"]) - r["sway_phase_deg"]) + np.radians(r["sway_row_deg"]) * (vb - v0)) if r["sway_px"] else 0.0
        ca = r["col_a"] + (r["cols"] - 1) / 2
        cb = r["col_b"] + (r["cols"] - 1) / 2 + S0
        q = r["dq"] - r["stated_dq"]
        frac = model["rho"] / step                                   # frames per row
        out.append(dict(kind=r["kind"], ka=ka, kb=kb, mirror=mirror, hint=r["hint"], frame_a=r["frame_a"], fa=fa, fb=fb, tha=tha, thb=thb, va=va, vb=vb, ca=ca, cb=cb, q=q,
                        Y=(r["col_a"] + r["col_b"] + r["cols"] - 1 - r["du"] + S0) if mirror else (r["col_b"] - r["col_a"] + r["du"] + S0),
                        y=(r["D"] - r["stated_D"]) - r["slope_b"] * q - frac * (r["slope_b"] * vb - r["slope_a"] * va), D=r["stated_D"]))
    return out


def fit(lay, model, rows, held_out=None, huber=0.1, grid=None, smooth=0.1, ridge=0.02, frame_weight=3.0, terms=("yaw",)):
    """One solve of the comparisons (helirec.solve: the model as equations over named unknowns, a Huber loss): the
    tiles' axis columns, places and courses, the heights, the angle's course, the rows' timing, the roll, the
    axial scale, and what the map lacks. Returns the new model, the residuals and a summary. held_out: a predicate
    on a comparison (as describe() returns it); those are predicted, not fitted. grid: the map on whose grid the
    comparisons were registered (the model's, unless the fit is repeated on the same comparisons: the angles enter
    through the places, so the equations are lines about the model in hand, and a fit that changes the rows'
    timing or the angle's course is worth another round); the new map is that grid's plus what the fit finds
    missing. smooth, ridge: what holds a course's knots where the comparisons do not (second differences, and the
    values themselves), in the units of the residuals. frame_weight: frames per pixel (a frame shift is known to
    a few hundredths of a frame, a column to a tenth of a pixel).

    The equations, per comparison of blocks A and B (E: what the map lacks; hat: a course between its knots):
        columns   Y = +-a_A + a_B +- d_A.n(th_A) + d_B.n(th_B) +- C_A(f_A) + C_B(f_B) + [the places' swing with the
                      angle] (T, rho) + psi (rows) - (Eu(B) +- Eu(A))            (+ where B is reversed, - where not)
        frames    step q = T(f_A) - T(f_B) - rho (v_B - v_A) [+ fan (c_B - c_A), same direction]
        rows      y = h_B - h_A + R_B(f_B) - R_A(f_A) + scale D [+ chi (c_A - c_B), B reversed] - (Ev(B) - Ev(A))

    terms: further terms of the model, by name (TERMS). "yaw" is part of the step; "depth" and "fan" are there to be
    tried and are not (on Paris4 they do not help where the fit is not fitted: 08_one_solve_plan.md, test 19):
        "yaw"    each tile's sample turned about the rotation axis against the reference tile's (the stages' yaw as
                 they travel, or a slip): the tile's angle is the record's plus yaw_t. Between two tiles it is a
                 frame shift at equal angle, and it swings the tile's place; the columns alone cannot tell it from
                 the direction of the place.
        "depth"  the axis leans towards the beam, so that content at depth z lies tilt x z rows higher. Views in the
                 same direction hold it whatever the content's depth (the tiles' axes lie apart along the beam);
                 reversed views hold twice the content's depth, which is not that of a point fixed in the sample: one
                 number per reversed kind for the part of the lean they show less (08_one_solve_plan.md, test 5).
        "fan"    the beam diverges across the columns: a tile whose axis lies nearer the source is magnified more, a
                 scale of the columns about the axis between two tiles, once per turn. The source's distance is the
                 record's; the factor fitted on it should be 1.
        "shared" the courses along the frames as the stages make them: one course of the columns and one of the
                 heights shared by all tiles at equal frame (knots), and of each tile's own only a drift and terms
                 once and twice per turn (in the columns twice: once per turn is the tile's place). In place of a
                 course of knots per tile.
        "common" a course of the columns and one of the heights shared by all tiles at equal frame, beside each tile's
                 own course of knots, which is then held to zero COMMON_OWN times harder: what the stages repeat is
                 measured by every tile, and a tile's own course is what it adds.
        "magnify" the beam diverges: a view whose tile's place lies Y nearer the source or further is magnified by
                 Y over the source's distance about the axis' column, in every view (the tile's place goes round
                 with the stage, so Y is the place's depth at that view's angle). It enters as the map's terms do,
                 B's less A's with A's sign by the mirror, with one factor on the record's source distance: 1 or
                 -1 (how the depth's sense lies is not known beforehand) if the model is right.
        "field"  what the detector's map lacks as a field on the detector instead of more powers: a displacement
                 along the columns and one along the rows at FIELD_NODES nodes, linear between them, one for all
                 tiles (they share the detector). Beside the map's own terms; held to zero where nothing asks.
        "tilt"   each tile's sample tilted against the reference tile's (the stages' pitch and roll as they travel):
                 its place changes with the row by kappa_t (px per row), and with the same numbers its height
                 changes across the columns. Fitted apart for the columns and the rows, which a rigid tilt ties.
                 The part of the columns' kappa that is every tile's place turned a quarter turn is the rows' timing
                 and is held at zero."""
    from .solve import Problem

    m = grid or model["map"]
    D = describe(lay, model, rows, m)
    n, nt = len(D), len(lay.labels)
    K = model["knots"]
    nk = len(K)
    v0 = m["v0"]
    step = np.radians(abs(lay.step_deg))
    tm = list(zip(m["i"], m["j"]))
    # a column displacement that depends on the row alone is seen only by opposite views half a pitch apart, where
    # an odd term cannot be told from the roll of the axis; its line in the column is a scale
    fit_u = [k for k, (i, j) in enumerate(tm) if not (i == 0 and j % 2 == 1) and (i, j) != (1, 0)]
    fit_v = list(range(len(tm)))
    sel = np.array([not (held_out and held_out(d)) for d in D], bool)
    of = lambda key, kind=float: np.array([d[key] for d in D], kind)
    ka, kb, mirror = of("ka", int), of("kb", int), of("mirror", bool)
    fa, fb, tha, thb, va, vb, ca, cb = (of(key) for key in ("fa", "fb", "tha", "thb", "va", "vb", "ca", "cb"))
    (ia, a0, a1), (ib, b0, b1) = hat(K, fa), hat(K, fb)
    sa = np.where(mirror, 1.0, -1.0)                                 # A's sign in the columns
    ca_, sa_, cb_, sb_ = np.cos(tha), np.sin(tha), np.cos(thb), np.sin(thb)
    # how the columns move with the angle of each block: the places' swing, at the model in hand
    dua = -model["d"][ka, 0] * sa_ + model["d"][ka, 1] * ca_
    dub = -model["d"][kb, 0] * sb_ + model["d"][kb, 1] * cb_

    p = Problem()
    for name, size in (("a", nt), ("d", 2 * (nt - 1)), ("C", nt * nk), ("T", nk), ("rho", 1), ("psi", 1), ("fan", 1), ("eu", len(fit_u)),
                       ("h", nt - 1), ("R", nt * nk), ("scale", 1), ("chi", 1), ("ev", len(fit_v))):
        p.unknowns(name, size)
    unknown = [name for name in terms if name not in TERMS]
    if unknown:
        raise ValueError(f"the model has no terms {unknown}: {TERMS}")
    yaw0 = np.asarray(model.get("yaw", np.zeros(nt)), float)
    depth = lambda d_, th_: -d_[..., 0] * np.sin(th_) + d_[..., 1] * np.cos(th_)        # of a point of the stage: how far along the beam
    if "yaw" in terms:
        p.unknowns("yaw", nt - 1)
    if "depth" in terms:
        p.unknowns("lean", 1)                 # rows per pixel along the beam: the tilt towards it
        p.unknowns("lean_reversed", 2)        # what opposite views within a tile, and of two tiles, show beside it
    if "fan" in terms:
        p.unknowns("fan_scale", 1)
    if "magnify" in terms:
        p.unknowns("magnify", 1)
    if "field" in terms:
        fy, fx = (np.linspace(0, lay.rows - 1, FIELD_NODES[0]), np.linspace(0, lay.cols - 1, FIELD_NODES[1]))
        p.unknowns("field_u", len(fy) * len(fx))
        p.unknowns("field_v", len(fy) * len(fx))

        def nodes(c, v):
            """the four nodes about (column c, row v) and their weights"""
            x = np.clip((c - fx[0]) / (fx[1] - fx[0]), 0, len(fx) - 1 - 1e-9)
            y = np.clip((v - fy[0]) / (fy[1] - fy[0]), 0, len(fy) - 1 - 1e-9)
            i, j = np.floor(y).astype(int), np.floor(x).astype(int)
            wy, wx = y - i, x - j
            return [((i + di) * len(fx) + j + dj, (wy if di else 1 - wy) * (wx if dj else 1 - wx)) for di in (0, 1) for dj in (0, 1)]
    shared = "shared" in terms
    turn_frames = float(lay.per_turn)
    f_mid = 0.5 * (K[0] + K[-1])
    own_c = lambda th_, f_: (np.cos(2 * th_), np.sin(2 * th_), (f_ - f_mid) / turn_frames)                          # a tile's own columns
    own_r = lambda th_, f_: (np.cos(th_), np.sin(th_), np.cos(2 * th_), np.sin(2 * th_), (f_ - f_mid) / turn_frames)  # ... and heights
    if shared:
        p.unknowns("Cs", nk)
        p.unknowns("Cp", 3 * nt)
        p.unknowns("Rs", nk)
        p.unknowns("Rp", 5 * nt)
    common = "common" in terms and not shared
    if common:
        p.unknowns("Cs", nk)
        p.unknowns("Rs", nk)
    if "tilt" in terms:
        p.unknowns("tilt_c", 2 * (nt - 1))    # the place's change with the row, per tile (px per row)
        p.unknowns("tilt_r", 2 * (nt - 1))    # the height's change across the columns, per tile (rows per px)

    def knots(e, block, tile, i, w0, w1, sign=1.0):
        """a course between its two knots"""
        e.add(block, tile * nk + i, sign * w0)
        e.add(block, tile * nk + i + 1, sign * w1)

    def hold(block, offset, unit=1.0, about=None):
        """what holds a course's knots where nothing else does: its values, and its second differences (about: the
        course in hand, where the unknowns are a change of it)"""
        k_, mid = np.arange(nk), np.arange(1, nk - 1)
        e = p.equations(np.zeros(nk) if about is None else -ridge * unit * about)
        e.add(block, offset + k_, ridge * unit)
        e = p.equations(np.zeros(nk - 2) if about is None else -smooth * unit * (about[:-2] - 2 * about[1:-1] + about[2:]))
        for shift, weight in ((-1, 1.0), (0, -2.0), (1, 1.0)):
            e.add(block, offset + mid + shift, weight * smooth * unit)

    # ---- columns
    eu = p.equations(of("Y"), sel)
    eu.add("a", ka, sa)
    eu.add("a", kb, 1.0)
    for tile, sign, cos, sin in ((ka, sa, ca_, sa_), (kb, 1.0, cb_, sb_)):          # the places; the reference tile's is the origin
        eu.add("d", 2 * (tile - 1), sign * cos, tile > 0)
        eu.add("d", 2 * (tile - 1) + 1, sign * sin, tile > 0)
    if shared:
        knots(eu, "Cs", 0, ia, a0, a1, sa)
        knots(eu, "Cs", 0, ib, b0, b1)
        for j, (ba, bb) in enumerate(zip(own_c(tha, fa), own_c(thb, fb))):
            eu.add("Cp", 3 * ka + j, sa * ba)
            eu.add("Cp", 3 * kb + j, bb)
    else:
        knots(eu, "C", ka, ia, a0, a1, sa)
        knots(eu, "C", kb, ib, b0, b1)
    if common:
        knots(eu, "Cs", 0, ia, a0, a1, sa)
        knots(eu, "Cs", 0, ib, b0, b1)
    if "tilt" in terms:
        for tile, sign, cos, sin, v in ((ka, sa, ca_, sa_, va), (kb, 1.0, cb_, sb_, vb)):
            eu.add("tilt_c", 2 * (tile - 1), sign * cos * (v - v0), tile > 0)
            eu.add("tilt_c", 2 * (tile - 1) + 1, sign * sin * (v - v0), tile > 0)
    knots(eu, "T", 0, ia, a0, a1, sa * dua)
    knots(eu, "T", 0, ib, b0, b1, dub)
    eu.add("rho", 0, dub * (vb - v0) + sa * dua * (va - v0))
    eu.add("psi", 0, np.where(mirror, va + vb - 2 * v0, vb - va))
    for c, k in enumerate(fit_u):
        a_, b_ = G._phi(m, ca, va, *tm[k]), G._phi(m, cb, vb, *tm[k])
        eu.add("eu", c, np.where(mirror, -(b_ + a_), -(b_ - a_)))
    if "field" in terms:                      # as the map's terms: B's less A's, A's sign with the mirror
        for (ia_, wa_), (ib_, wb_) in zip(nodes(ca, va), nodes(cb, vb)):
            eu.add("field_u", ia_, np.where(mirror, -wa_, wa_))
            eu.add("field_u", ib_, -wb_)
    if "yaw" in terms:                        # the tile's own angle swings its place, as the angle's course does
        eu.add("yaw", ka - 1, sa * dua, ka > 0)
        eu.add("yaw", kb - 1, dub, kb > 0)
    if "magnify" in terms:
        a_ = (ca - model["a"][ka]) * depth(model["d"][ka], tha) * lay.fan * step
        b_ = (cb - model["a"][kb]) * depth(model["d"][kb], thb) * lay.fan * step
        eu.add("magnify", 0, np.where(mirror, -(b_ + a_), -(b_ - a_)), ka != kb)
    if "fan" in terms:                        # B's columns about its axis, magnified by how much nearer the source its axis lies
        nearer = depth(model["d"][kb], thb) - depth(model["d"][ka], tha)
        eu.add("fan_scale", 0, (cb - model["a"][kb]) * nearer * lay.fan * step, ~mirror & (ka != kb))
    # ---- frames, in radians and weighted; T and rho are fitted as a change of the model's
    w = frame_weight / step
    eq = p.equations(w * (of("q") * step - (course(K, model["T"], fa) - course(K, model["T"], fb)) - (yaw0[ka] - yaw0[kb]) + model["rho"] * (vb - va)), sel)
    if "yaw" in terms:
        eq.add("yaw", ka - 1, w, ka > 0)
        eq.add("yaw", kb - 1, -w, kb > 0)
    knots(eq, "T", 0, ia, a0, a1, w)
    knots(eq, "T", 0, ib, b0, b1, -w)
    eq.add("rho", 0, -w * (vb - va))
    eq.add("fan", 0, w * (cb - ca) * lay.fan * step, ~mirror)
    # ---- rows
    er = p.equations(of("y"), sel)
    er.add("h", ka - 1, -1.0, ka > 0)
    er.add("h", kb - 1, 1.0, kb > 0)
    if shared:
        knots(er, "Rs", 0, ia, a0, a1, -1.0)
        knots(er, "Rs", 0, ib, b0, b1)
        for j, (ba, bb) in enumerate(zip(own_r(tha, fa), own_r(thb, fb))):
            er.add("Rp", 5 * ka + j, -ba)
            er.add("Rp", 5 * kb + j, bb)
    else:
        knots(er, "R", ka, ia, a0, a1, -1.0)
        knots(er, "R", kb, ib, b0, b1)
    if common:
        knots(er, "Rs", 0, ia, a0, a1, -1.0)
        knots(er, "Rs", 0, ib, b0, b1)
    if "tilt" in terms:                       # the same content in both views, x columns from the sample's origin in A's
        x = ca - np.array([u_of(model, int(k_), th_, f_) for k_, th_, f_ in zip(ka, tha, fa)])
        for tile, sign in ((ka, -1.0), (kb, 1.0)):
            er.add("tilt_r", 2 * (tile - 1), sign * x * ca_, (tile > 0) & (ka != kb))
            er.add("tilt_r", 2 * (tile - 1) + 1, sign * x * sa_, (tile > 0) & (ka != kb))
    er.add("scale", 0, of("D"))
    er.add("chi", 0, ca - cb, mirror)
    if "depth" in terms:
        er.add("lean", 0, depth(model["d"][kb], thb) - depth(model["d"][ka], tha), ~mirror & (ka != kb))
        both = model["d"][ka] + model["d"][kb]
        er.add("lean_reversed", np.where(ka == kb, 0, 1), -depth(both, tha), mirror)
    for c, k in enumerate(fit_v):
        er.add("ev", c, -(G._phi(m, cb, vb, *tm[k]) - G._phi(m, ca, va, *tm[k])))
    if "field" in terms:
        for (ia_, wa_), (ib_, wb_) in zip(nodes(ca, va), nodes(cb, vb)):
            er.add("field_v", ia_, wa_)
            er.add("field_v", ib_, -wb_)
        for block in ("field_u", "field_v"):
            n_ = p.blocks[block][1]
            p.equations(np.zeros(n_)).add(block, np.arange(n_), ridge)
    # ---- the courses' knots
    if shared:
        hold("Cs", 0)
        hold("Rs", 0)
        for block in ("Cp", "Rp"):            # what is common to the tiles belongs to the shared course
            o_, n_ = p.blocks[block]
            p.equations(np.zeros(n_)).add(block, np.arange(n_), ridge)
    else:
        for t in range(nt):
            hold("C", t * nk, COMMON_OWN if common else 1.0)
            hold("R", t * nk, COMMON_OWN if common else 1.0)
    if common:
        hold("Cs", 0)
        hold("Rs", 0)
    if "tilt" in terms and nt > 1:            # the rows' timing is the part of kappa that is each place turned a quarter turn
        e = p.equations([0.0])
        scale_ = 1.0 / max(np.abs(model["d"]).max(), 1.0)
        for t_ in range(1, nt):
            e.add("tilt_c", 2 * (t_ - 1), -model["d"][t_, 1] * scale_ * 1e6)
            e.add("tilt_c", 2 * (t_ - 1) + 1, model["d"][t_, 0] * scale_ * 1e6)
    hold("T", 0, 1.0 / step, model["T"])                            # the angle's course in frames, held as a whole

    s = p.solve(huber=huber, iterations=12)
    ru, rv = s.left(eu), s.left(er)
    rq = s.left(eq) / w / step                                      # frames

    new = dict(model, map=dict(m, eu=list(m["eu"]), ev=list(m["ev"])))
    new["a"] = s["a"].copy()
    new["d"] = np.vstack([np.zeros(2), s["d"].reshape(nt - 1, 2)]) if nt > 1 else np.zeros((1, 2))
    if shared:        # onto each tile's knots (a term twice per turn between knots 18 degrees apart: to 5 % of it)
        fk = K
        new["C"] = np.array([s["Cs"] + sum(c_ * b_ for c_, b_ in zip(s["Cp"][3 * t_:3 * t_ + 3], own_c(np.radians([lay.angle_at(t_, f_) for f_ in fk]), fk))) for t_ in range(nt)])
        new["own"] = dict(columns=s["Cp"].reshape(nt, 3).tolist(), heights=s["Rp"].reshape(nt, 5).tolist())
    else:
        new["C"] = s["C"].reshape(nt, nk) + (s["Cs"][None, :] if common else 0.0)
    new["T"] = model["T"] + s["T"]
    new["rho"], new["psi"], new["fan"] = float(model["rho"] + s["rho"][0]), float(s["psi"][0]), float(s["fan"][0])
    new["h"] = np.concatenate([[0.0], s["h"]])
    if shared:
        new["R"] = np.array([s["Rs"] + sum(c_ * b_ for c_, b_ in zip(s["Rp"][5 * t_:5 * t_ + 5], own_r(np.radians([lay.angle_at(t_, f_) for f_ in K]), K))) for t_ in range(nt)])
    else:
        new["R"] = s["R"].reshape(nt, nk) + (s["Rs"][None, :] if common else 0.0)
    if "magnify" in terms:
        new["magnify"] = float(s["magnify"][0])
    if "field" in terms:
        new["field"] = dict(rows=fy.tolist(), cols=fx.tolist(), u=s["field_u"].reshape(len(fy), len(fx)).tolist(), v=s["field_v"].reshape(len(fy), len(fx)).tolist())
    if "tilt" in terms:
        new["tilt_columns"] = np.vstack([np.zeros(2), s["tilt_c"].reshape(nt - 1, 2)]).tolist()
        new["tilt_rows"] = np.vstack([np.zeros(2), s["tilt_r"].reshape(nt - 1, 2)]).tolist()
    new["scale"], new["chi"] = float(s["scale"][0]), float(s["chi"][0])
    if "yaw" in terms:
        new["yaw"] = yaw0 + np.concatenate([[0.0], s["yaw"]])
    if "depth" in terms:
        new["lean"], new["lean_reversed"] = float(s["lean"][0]), [float(x) for x in s["lean_reversed"]]
    if "fan" in terms:
        new["fan_scale"] = float(s["fan_scale"][0])
    corr = dict(eu=[0.0] * len(tm), ev=[0.0] * len(tm))
    for c, k in enumerate(fit_u):
        corr["eu"][k] = float(s["eu"][c])
    for c, k in enumerate(fit_v):
        corr["ev"][k] = float(s["ev"][c])
    new["map"]["eu"] = [a + b for a, b in zip(m["eu"], corr["eu"])]
    new["map"]["ev"] = [a + b for a, b in zip(m["ev"], corr["ev"])]
    du, dv = G._update_range(m, corr)
    summary = dict(n=n, fitted=int(sel.sum()), unknowns=p.n, determined=s.rank, correction_px=du, correction_rows=dv, rho_change=float(s["rho"][0]),
                   angle_change_frames=float(np.abs(s["T"]).max() / step), left={}, structure={})
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else None
    kinds = np.array([d["kind"] for d in D])
    for kind in KINDS:
        k_ = kinds == kind
        if not k_.any():
            continue
        summary["left"][kind] = dict(n=int(k_.sum()), px=rms(ru[k_ & sel]), rows=rms(rv[k_ & sel]), frames=rms(rq[k_ & sel]), held_px=rms(ru[k_ & ~sel]), held_rows=rms(rv[k_ & ~sel]),
                                     held_frames=rms(rq[k_ & ~sel]))
        groups = {}
        for i, d in enumerate(D):
            if d["kind"] == kind:
                groups.setdefault((d["ka"], d["kb"], d["hint"], d["frame_a"]), []).append((ru[i], rv[i]))
        g = [np.array(v) for v in groups.values() if len(v) >= 3]
        if len(g) >= 4:
            within, between = np.sqrt(np.mean([v.var(0, ddof=1) for v in g], 0)), np.std([v.mean(0) for v in g], 0)
            summary["structure"][kind] = dict(positions=len(g), px_within=float(within[0]), px_between=float(between[0]), rows_within=float(within[1]), rows_between=float(between[1]))
    return new, dict(columns=ru, rows=rv, frames=rq, described=D, fitted=sel), summary


# ---------------------------------------------------------------- the step

def summary_of(lay, model):
    """The model in the units a reader wants: per tile the axis column, the place on the stage, the height, and the
    size of its courses."""
    step = np.radians(abs(lay.step_deg))
    on = lambda t: (model["knots"] >= lay.on_disk[t][0]) & (model["knots"] <= lay.on_disk[t][1])
    out = {}
    for t, label in enumerate(lay.labels):
        C, R = model["C"][t][on(t)], model["R"][t][on(t)]
        out[label] = dict(axis_column=float(model["a"][t]), place_px=[float(model["d"][t, 0]), float(model["d"][t, 1])], height_rows=float(model["h"][t]),
                          yaw_rad=float(model["yaw"][t]) if "yaw" in model else 0.0,
                          course_px=dict(rms=float(C.std()), lowest=float(C.min()), highest=float(C.max())), course_rows=dict(rms=float(R.std()), lowest=float(R.min()), highest=float(R.max())),
                          stage_place_px=[float(x) for x in lay.place[t]], column_course=[float(x) for x in model["C"][t]], height_course=[float(x) for x in model["R"][t]])
    return out


def calibrate_mosaic(scans, row_time=None, device="auto", order=3, passes=6, tolerance=0.02, threads=24, angles=24, in_tile=KNOT_FRAMES, start=None, smooth=0.1, log=print,
                     comparisons=None, patch=0):
    """The geometry calibration of a mosaic. scans: the tiles' master files, the reference tile first. row_time:
    seconds per stored row (signed) to start from, if known; it is measured by the passes either way. start: a
    calibration made before for these scans, to go on from. comparisons: a list that receives the last pass's
    comparisons (for a look at what the fit leaves). patch: 0 registers the plan's blocks whole; N cuts each into
    patches of N x N (see measure). Returns a dict (geometry.save writes it)."""
    from . import _core

    t_start = time.time()
    session = _core.CalibrationSession(list(scans), threads)
    if start is None:
        log("layout")
        lay = read_layout(session, len(scans), device, log)
        model = start_model(lay, order, row_time or 0.0)
    else:
        lay, model = restore(session, start)
    refill = None
    if start is None or not start.get("row_time", {}).get("refill"):
        refill = row_time_from_refill(session, lay, log)
    else:
        refill = start["row_time"]["refill"]
    history = []
    held = lambda d: (d["frame_a"] // 8) % 2 == 1          # every other frame position is predicted, not fitted
    log("passes")
    converged = False
    for it in range(passes):
        reqs = plan(lay, model, angles, in_tile)
        rows = measure(session, lay, model, reqs, device, log if it == 0 else None, patch=patch)
        grid = model["map"]
        new = model
        for _ in range(3):
            new, res, s = fit(lay, new, rows, held, grid=grid, smooth=smooth)
        s["rho_rad_per_row"], s["row_time_us"] = new["rho"], new["rho"] / np.radians(abs(lay.step_deg)) * lay.period_s * 1e6
        history.append(s)
        log(f"  pass {it + 1}: {s['n']} comparisons; the map is corrected by up to {s['correction_px']:.3f} px and {s['correction_rows']:.3f} rows; row time {s['row_time_us']:+.3f} us; held out " +
            ", ".join(f"{k} {v['held_px']:.3f} px / {v['held_rows']:.3f} rows" for k, v in s["left"].items() if v["held_px"] is not None))
        moved = float(np.abs(new["d"] - model["d"]).max())
        model = new
        if max(s["correction_px"], s["correction_rows"]) < tolerance and moved < tolerance and it:
            converged = True
            break
    else:
        log(f"  not converged to {tolerance} in {passes} passes")
    if comparisons is not None:
        comparisons[:] = [dict(r, res_px=float(res["columns"][i]), res_rows=float(res["rows"][i]), res_frames=float(res["frames"][i]), fitted=bool(res["fitted"][i])) for i, r in enumerate(rows)]
    return calibration(lay, model, scans, refill, history, converged, time.time() - t_start)


def calibration(lay, model, scans, refill=None, history=(), converged=False, seconds=0.0):
    """A layout and a model as the dict calibrate_mosaic returns (geometry.save writes it, restore reads it, frame_pose
    and tile_reconstructor go by it): for a model fitted otherwise than by calibrate_mosaic's passes, e.g. on
    comparisons measured before."""
    step = np.radians(abs(lay.step_deg))
    return dict(step="mosaic geometry", scans=[str(x) for x in scans], seconds=round(seconds, 1),
                layout=dict(rows=lay.rows, cols=lay.cols, tiles=lay.labels, frames_per_turn=lay.per_turn, pixel_um=lay.pixel_um, step_deg=lay.step_deg, frame_period_s=lay.period_s,
                            row_sign=lay.row_sign, frames_on_disk=[list(x) for x in lay.on_disk], coarse_axis=lay.axis, neighbours=[[lay.labels[k], lay.labels[j]] for k, j in lay.pairs],
                            knot_frames=[float(x) for x in model["knots"]]),
                row_time=dict(seconds_per_row=model["rho"] / step * lay.period_s, rad_per_row=model["rho"], measured="the shear between tiles and the frame shifts (passes)", refill=refill),
                map=model["map"], tiles=summary_of(lay, model), angle_course_rad=[float(x) for x in model["T"]],
                angle_course_frames=dict(rms=float(model["T"].std() / step), lowest=float(model["T"].min() / step), highest=float(model["T"].max() / step)),
                axis_roll_rad=model["psi"], rows_slope_opposite=model["chi"], axial_scale=model["scale"], fan_factor=model["fan"], passes=list(history), converged=converged)


def restore(session, cal):
    """The layout and the model of a calibration made before (calibrate_mosaic's result), for the same scans in the
    same order: to go on from it, or to place further comparisons on its grid, without finding the layout again."""
    L = cal["layout"]
    n = len(L["tiles"])
    helix = []
    for k in range(n):
        st = [x for x in session.stretches(k) if x["role"] == "projection" and x["latched"] and x["helical"]]
        helix.append(st[0]["index"])
    px_mm = L["pixel_um"] * 1e-3
    lay = Layout(rows=L["rows"], cols=L["cols"], per_turn=L["frames_per_turn"], pixel_um=L["pixel_um"], step_deg=L["step_deg"], period_s=L["frame_period_s"],
                 source_mm=float(session.number(0, "scan/source_sample_distance")), labels=list(L["tiles"]), helix=helix, on_disk=[tuple(x) for x in L["frames_on_disk"]],
                 angle=[session.series(k, helix[k], "angle_deg") for k in range(n)], height=[L["row_sign"] * session.series(k, helix[k], "height_mm") / px_mm for k in range(n)],
                 row_sign=L["row_sign"], axis=list(L["coarse_axis"]), place=[tuple(cal["tiles"][t]["stage_place_px"]) for t in L["tiles"]],
                 pairs=[(L["tiles"].index(a), L["tiles"].index(b)) for a, b in L["neighbours"]])
    T = cal["tiles"]
    model = dict(knots=np.array(L["knot_frames"]), a=np.array([T[t]["axis_column"] for t in L["tiles"]]), d=np.array([T[t]["place_px"] for t in L["tiles"]]),
                 C=np.array([T[t]["column_course"] for t in L["tiles"]]), h=np.array([T[t]["height_rows"] for t in L["tiles"]]), R=np.array([T[t]["height_course"] for t in L["tiles"]]),
                 yaw=np.array([T[t].get("yaw_rad", 0.0) for t in L["tiles"]]),
                 T=np.array(cal["angle_course_rad"]), map=dict(cal["map"]), rho=cal["row_time"]["rad_per_row"], scale=cal["axial_scale"], psi=cal["axis_roll_rad"],
                 chi=cal["rows_slope_opposite"], fan=cal.get("fan_factor", 1.0))
    return lay, model


def frame_pose(cal, label, angle_deg, frame, row=None):
    """For frames of a tile: the angle of `row` (the middle row if None) [deg], where the sample's origin falls on
    the detector (column), and how far the sample lies from the record along the axis (rows): th, u and H of the
    module's description. angle_deg, frame: the record's angle and frame number of each frame."""
    L = cal["layout"]
    t = cal["tiles"][label]
    K = np.array(L["knot_frames"])
    v0 = cal["map"]["v0"]
    frame = np.asarray(frame, float)
    th = (np.radians(np.asarray(angle_deg, float)) + course(K, np.array(cal["angle_course_rad"]), frame) + cal["row_time"]["rad_per_row"] * ((v0 if row is None else row) - v0)
          + t.get("yaw_rad", 0.0))
    u = t["axis_column"] + t["place_px"][0] * np.cos(th) + t["place_px"][1] * np.sin(th) + course(K, np.array(t["column_course"]), frame)
    return np.degrees(th), u, t["height_rows"] + course(K, np.array(t["height_course"]), frame)


# ---------------------------------------------------------------- one tile in the reconstruction

def reconstruction_terms(cal, label, angle_deg, height_rows, frame, rot_center, row_origin=0.0):
    """One tile's frames as the reconstruction's per-projection terms (helical.hpp, ScanTrajectory; what
    geometry.reconstruction_terms is for a pair), in the tile's own frame: the slice's centre is the tile's
    rotation axis, and the slice h is the same height of the sample in every tile (row_origin, rows, moves that
    numbering; it must be the same for the tiles that are to be compared). The sample's origin lies at the tile's
    place (frame_pose's swing) from the centre.

    angle_deg, height_rows, frame: the record's angle (unwrapped), its height in rows (the layout's row_sign x
    height / pixel) and the frame number of each frame used. The reconstruction puts the column m of its frame at
    m - rot_center columns from the axis."""
    L = cal["layout"]
    t = cal["tiles"][label]
    frame, Hrec = np.asarray(frame, float), np.asarray(height_rows, float)
    th_deg, u, H = frame_pose(cal, label, angle_deg, frame)                 # at the middle row
    th = np.radians(th_deg)
    axis = u - (t["place_px"][0] * np.cos(th) + t["place_px"][1] * np.sin(th))
    row = (1.0 + cal["axial_scale"]) * Hrec + H                             # content at height w is on the row w + row
    z_pix = L["row_sign"] * (Hrec - Hrec[0])
    # the travel per frame as a line through the 33 frames around (the encoder's own frame-to-frame differences are noise)
    k_ = np.arange(-16, 17)
    pad = np.concatenate([row[0] + (row[1] - row[0]) * np.arange(-16, 0), row, row[-1] + (row[-1] - row[-2]) * np.arange(1, 17)])
    travel = np.convolve(pad, (k_ / (k_ ** 2).sum())[::-1], "valid")
    per_row = cal["row_time"]["rad_per_row"] / np.radians(abs(L["step_deg"]))     # frames per row
    n = len(frame)
    return dict(angles_deg=np.asarray(th_deg, np.float64), z_translation_scale=1.0,
                row_offset=(-(row + row_origin) - z_pix).astype(np.float32), row_slope=np.full(n, -cal["rows_slope_opposite"], np.float32), slope_col=float(rot_center),
                col_offset=(axis - rot_center).astype(np.float32), col_slope=np.full(n, cal["axis_roll_rad"], np.float32), slope_row=(L["rows"] - 1) / 2,
                col_scale=np.zeros(n, np.float32), scale_col=0.0, row_scale=(-travel * per_row).astype(np.float32), scale_row=0.0)


def tile_reconstructor(master, cal, label, device="cuda", params=None, row_origin=None, weight=None, cache_gb=0.0, reduced=None, log=print, cache_dir=None, reduce=0):
    """A reconstructor (helirec SliceReconstructor) of one tile alone, on the mosaic calibration's geometry: the
    detector map, the pose and the angle of every frame, the rows' timing. The tile's own frame (see
    reconstruction_terms); only the helix frames on disk are used. For the check of the calibration on slices, and
    the first step towards reconstructing a mosaic; the intensity side is the pair pipeline's (the scan's own
    darks and flats by the ring current, a weight map from the flats with a taper at the detector's far edge and
    across the axis), and nothing knows yet that every ray passes through sample the tile does not see.

    params: reconstruction parameters over the defaults of a calibration check (fbp, the ramp filter alone).
    row_origin: see reconstruction_terms; None numbers the slices from the record's height of this tile's middle
    frame on disk (the slice (rows - 1) / 2 is then about what the middle row saw there). Tiles that are to be
    compared slice by slice must be given one value (the result's "row_origin" of the first).
    weight: a function (weight map (rows, cols)) -> weight map, to reconstruct from part of the detector.
    reduced: the "reduced" of an earlier call for this tile (its darks and flats as reduced then).
    cache_dir: the reduced darks and flats are kept there (calibration.reduced_darks_flats): reducing them is
    46 s of a tile's start, reading them back 0.3 s.
    reduce: 1, 2 or 3 reconstructs at a half, a quarter, an eighth: the frames are read at the codestream's own
    lower resolution (RawReader(reduce=): decoding 3.4, 12 and 44 times faster), the darks and flats taken
    through the same low-pass (calibration.at_resolution), and the whole geometry divided by 2^reduce, nothing
    added: slice h of a reduced reconstruction is slice h 2^reduce of the full one, and its pixel at (x, y) from
    the axis is the full one's at (x, y) 2^reduce from the axis. The result's rot_center, where the axis lies in
    the slice, is in reduced pixels. row_origin stays in stored rows.
    Returns (reconstructor, dict(rot_center, frames, scan, terms, row_origin, reduced, reduce))."""
    from . import _core, bliss, calibration
    from .recon import CALIBRATION_KEYS, DEFAULTS, angle_subset_config, backprojector_config
    from .sld import sld_config

    L = cal["layout"]
    k = L["tiles"].index(label)
    a, b = L["frames_on_disk"][k]
    scan = bliss.open_scan(master)
    rows, cols = scan.frame_shape
    # the helix among the scan's projections (they are kept one stretch after the other)
    session = _core.CalibrationSession([str(master)], 0)
    st = session.stretches(0)
    helix = next(x for x in st if x["role"] == "projection" and x["latched"] and x["helical"])
    before = sum(x["n_frames"] for x in st if x["role"] == "projection" and x["index"] < helix["index"])
    del session
    frame = np.arange(a, b)
    idx = before + frame
    angle = np.asarray(scan.unwrapped_angles_deg, np.float64)[idx]
    z = np.ascontiguousarray(np.asarray(scan.z_translation_m, np.float64)[idx])
    height_rows = L["row_sign"] * z / scan.pixel_size_m
    t = cal["tiles"][label]
    # The slice's grid has its centre on a pixel or between two, and the reconstruction puts the axis there: the
    # axis' column is rounded to half a pixel of the reconstruction (the rest goes into every frame's column
    # offset), so that rot_center is where the axis lies in the slice. (Rounded to a quarter, as it was, the axis
    # lay a quarter of a pixel from rot_center for some tiles and not for others.)
    rot_center = (1 << int(reduce)) * round(2 * t["axis_column"] / (1 << int(reduce))) / 2.0
    if reduced is None:
        log(f"[{label}] darks and flats ...")
        reduced = calibration.reduced_darks_flats(scan, cache_dir, log)
    dark, flats = reduced
    wm = calibration.weight_map([flats])
    far, near = (50.0, min(2000.0, cols - t["axis_column"])) if t["axis_column"] > (cols - 1) / 2 else (min(2000.0, t["axis_column"]), 50.0)
    w = (wm * _core.apodisation_mask(cols, far, near, 1e-8)[None, :]).astype(np.float32)
    if weight is not None:
        w = np.ascontiguousarray(weight(w), np.float32)
    if row_origin is None:
        row_origin = -(1.0 + cal["axial_scale"]) * float(height_rows[len(height_rows) // 2])
    terms = reconstruction_terms(cal, label, angle, height_rows, frame, rot_center, row_origin)
    m = cal["map"]
    f = 1 << int(reduce)
    pixel_size_m = scan.pixel_size_m * f
    eu, ev = [float(x) for x in m["eu"]], [float(x) for x in m["ev"]]
    c0, cs, v0, vs = (float(m[key]) for key in ("c0", "cs", "v0", "vs"))
    rad_per_row = cal["row_time"]["rad_per_row"]
    if reduce:
        # every position of the stored detector and of the full reconstruction over f; slopes and scales are ratios
        import h5py
        with h5py.File(scan.projections.files[int(scan.projections.file_index[idx[0]])], "r") as h5:
            chunk_rows = h5[scan.projections.dataset].chunks[1]
        dark, flats = calibration.at_resolution(dark, reduce, chunk_rows), calibration.at_resolution(flats, reduce, chunk_rows)
        w = np.ascontiguousarray(w[::f, ::f])                                # the weight of the stored pixel a reduced value sits at
        rows_r, cols_r = rows // f, cols // f
        # the reconstruction counts the rows' timing from its middle row, which is the stored row f (rows_r - 1) / 2
        # and not the stored middle (rows - 1) / 2 the calibration's angles are for
        terms = dict(terms, angles_deg=terms["angles_deg"] + np.degrees(rad_per_row * (f * (rows_r - 1) / 2 - (rows - 1) / 2)),
                     row_offset=terms["row_offset"] / f, col_offset=terms["col_offset"] / f, slope_col=terms["slope_col"] / f, slope_row=terms["slope_row"] / f,
                     scale_col=terms["scale_col"] / f, scale_row=terms["scale_row"] / f)
        eu, ev = [x / f for x in eu], [x / f for x in ev]
        c0, cs, v0, vs = c0 / f, cs / f, v0 / f, vs / f
        rot_center, rad_per_row = rot_center / f, rad_per_row * f
        rows, cols = rows_r, cols_r
    reader = _core.RawReader(list(scan.projections.files), scan.projections.dataset, *scan.frame_shape, reduce=int(reduce))
    if cache_gb:
        reader.set_cache_bytes(int(cache_gb * (1 << 30)))
    sd = _core.make_scan_data(
        terms["angles_deg"], z, pixel_size_m, rows, cols, 0, 0.0, 0.0, np.ascontiguousarray(np.asarray(scan.control_A, np.float64)[idx]),
        np.ascontiguousarray(scan.flats_unwrapped_angles_deg, np.float32), np.ascontiguousarray(dark), np.ascontiguousarray((flats - dark[None]).astype(np.float32)), w, reader,
        np.ascontiguousarray(np.asarray(scan.projections.file_index)[idx]), np.ascontiguousarray(np.asarray(scan.projections.frame_index)[idx]), 1.0,
        row_offset=np.asarray(terms["row_offset"], np.float32).tolist(), row_slope=terms["row_slope"].tolist(), slope_col=terms["slope_col"],
        col_offset=np.asarray(terms["col_offset"], np.float32).tolist(),
        col_slope=terms["col_slope"].tolist(), slope_row=terms["slope_row"], col_scale=terms["col_scale"].tolist(), scale_col=terms["scale_col"],
        row_scale=terms["row_scale"].tolist(), scale_row=terms["scale_row"],
        map_i=[int(x) for x in m["i"]], map_j=[int(x) for x in m["j"]], map_eu=eu, map_ev=ev, map_c0=c0, map_cs=cs, map_v0=v0, map_vs=vs)
    p = dict(DEFAULTS)
    p.update(backprojector="fbp", filters=[], sld="", sharpness_weighting="off", angle_subset="")
    p.update(params or {})
    p.pop("mode")
    cfg = dict(p, device=device, energy_kev=scan.energy_kev, distance_m=scan.distance_m, pixel_size_m=pixel_size_m, rot_center=rot_center, legacy_reference=False)
    cfg.update(backprojector_config(p["backprojector"]))
    if not cfg.get("grid_size"):
        cfg["grid_size"] = int(round(2.0 * rot_center)) + 1
    cfg.update(sld_config(cfg.pop("sld")))
    cfg["exposure_arc"] = 0.0
    cfg["sharpness_weighting"] = ""
    cfg["angle_subset"], cfg["angle_subsets"] = angle_subset_config(cfg.pop("angle_subset"))
    for key in CALIBRATION_KEYS:
        cfg.pop(key)
    cfg["rolling_shutter"] = rad_per_row / np.radians(abs(L["step_deg"])) * (rows - 1)
    return _core.SliceReconstructor([sd], cfg), dict(rot_center=rot_center, frames=(int(a), int(b)), scan=scan, terms=terms, row_origin=row_origin, reduced=reduced,
                                                     reduce=int(reduce))


# ---------------------------------------------------------------- the tiles against each other in the volume

def _local(x, k=25):
    """each slice against its own local mean (a tile alone reconstructs with a slow shading: every ray passes
    through sample the tile does not see)"""
    c = np.cumsum(np.cumsum(np.pad(x, ((0, 0), (k // 2 + 1, k // 2), (k // 2 + 1, k // 2)), mode="edge"), 1, dtype=np.float64), 2)
    box = (c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]) / (k * k)
    return (x - box).astype(np.float32)


def overlaps(cal, masters, heights=2, device="cuda", size=1024, block=(24, 256, 256), margin=(4, 16, 16), stride=180, least=0.1, pairs=None, log=print, cache_dir=None,
             reduce=0, coarse=0):
    """A mosaic calibration measured where a reconstruction sees it: the overlap of every two neighbouring tiles is
    reconstructed from each tile alone (tile_reconstructor: FBP, the ramp filter alone, in the tile's own frame),
    as a slab of slices about the middle between the two axes, and the two slabs are registered block by block
    (one slice at this voxel size is noise; a block of slices registers).

    The sample's origin lies at a tile's place from its axis, in the slice at p_t = (c1, -s1) of the calibration's
    place (the slice's y runs against the swing's sine). A point at P from tile k's axis is at P + p_j - p_k from
    tile j's. Each row of the table is one block: what tile j's image of it is displaced by against tile k's
    beyond that, (dx, dy) in the slice [px] and dz along the axis [slices]; zero for a calibration that puts both
    tiles' rays in one frame.

    masters: the tiles' master files in the calibration's order. heights: how many slabs (spread over the slices
    both tiles can make) or the slabs' first slices. pairs: (k, j) index pairs; None: the layout's neighbours.
    cache_dir: where each tile's reduced darks and flats are kept (tile_reconstructor).
    reduce: 1, 2 or 3 makes the check on reconstructions at a half, a quarter, an eighth (tile_reconstructor):
    the same region of the sample and the same detail (size, the blocks' extent in the slice, stride and the
    registration's band are those of the full resolution over 2^reduce; a block keeps its number of slices and so
    is 2^reduce times as thick). heights and everything returned stay in full-resolution pixels and slices.
    coarse: how far the tiles may lie from where the calibration puts them [full-resolution pixels, in the slice
    and along the axis], for a calibration that has not been fitted yet: the two slabs of an overlap are first
    registered whole, to the pixel, within that distance, and the blocks are cut from there. 0: the blocks alone,
    which find up to two pixels of the reconstruction.
    Returns a list of dicts: k, j, height, x, y (the block's middle from tile k's axis), xj, yj (from tile j's),
    dx, dy, dz, correlation."""
    import gc

    from . import _core

    L = cal["layout"]
    labels = L["tiles"]
    if pairs is None:
        pairs = [(labels.index(a), labels.index(b)) for a, b in L["neighbours"]]
    f = 1 << int(reduce)
    place = np.array([[cal["tiles"][t]["place_px"][0], -cal["tiles"][t]["place_px"][1]] for t in labels]) / f      # in the pixels reconstructed
    size, stride = size // f, max(1, stride // f)
    block, margin = (block[0], block[1] // f, block[2] // f), (margin[0], max(2, margin[1] // f), max(2, margin[2] // f))
    far = int(np.ceil(coarse / f))
    pad, padz = 2 + far, far               # pixels around each box and slices beyond each slab: a block is cut again there
    size += 2 * pad
    session = _core.CalibrationSession([str(m) for m in masters], 0)
    lay, model = restore(session, cal)
    del session
    a, b = lay.on_disk[0]
    row_origin = -(1.0 + model["scale"]) * float(lay.height[0][(a + b) // 2])        # one numbering of the slices for all tiles
    nz, mz = block[0], margin[0]
    ns = nz + 2 * mz
    tiles = sorted({t for p in pairs for t in p})
    # which slices every tile can make: asked of a reconstructor of each (kept: its darks and flats are reduced once)
    slabs, centre, doable = {}, {}, {}
    wanted = {t: [] for t in tiles}
    for k, j in pairs:
        p = place[j] - place[k]
        wanted[k].append(((k, j), -p / 2))
        wanted[j].append(((k, j), p / 2))
    hs = None if isinstance(heights, int) else [int(h) // f for h in heights]
    made = {}
    for stage in ("ask", "make"):
        for t in tiles:
            if stage == "ask" and hs is not None:
                continue
            t0 = time.time()
            rec, info = tile_reconstructor(masters[t], cal, labels[t], device=device, row_origin=row_origin, cache_gb=24.0, reduced=made.get(t), log=lambda *a_: None,
                                           cache_dir=cache_dir, reduce=reduce)
            made[t] = info["reduced"]
            centre[t] = info["rot_center"]
            doable[t] = rec.info["doable"]
            if stage == "make":
                # slab by slab, every overlap of it: the overlaps of one slab are cut from the same frames, which
                # the reader keeps decoded (a slab's chunks are 12 to 24 GB: two slabs in turn would not both stay)
                for h in hs:
                    for pair, pos in wanted[t]:
                        x0, y0 = int(round(centre[t] + pos[0] - size / 2)), int(round(centre[t] + pos[1] - size / 2))
                        slabs[(pair, t, h)] = (_local(rec.reconstruct(h - padz, h + ns + padz, roi=(x0, x0 + size, y0, y0 + size)), (25 // f) | 1), (x0, y0))
                log(f"  {labels[t]}: {len(wanted[t])} overlaps x {len(hs)} slabs reconstructed ({time.time() - t0:.0f} s)")
            del rec
            gc.collect()
        if stage == "ask" and hs is None:
            lo, hi = max(d[0] for d in doable.values()) + padz, min(d[1] for d in doable.values()) - ns - padz
            hs = [int(x) for x in np.linspace(lo + (hi - lo) * 0.3, lo + (hi - lo) * 0.7, heights)]   # well inside: slices with a whole turn of views
            log(f"  slices every tile can make: {lo * f}..{(hi + ns) * f}; slabs at {[h * f for h in hs]}")
    reg = _core.CalibrationRegistrar([nz, block[1], block[2]], list(margin), f_lo=0.01 * f, f_hi=min(0.15 * f, 0.45), search=[2, max(2, 12 // f), max(2, 12 // f)], device=device)
    out = []
    P1, P2, m1, m2 = block[1], block[2], margin[1], margin[2]
    for k, j in pairs:
        p = place[j] - place[k]
        for h in hs:
            A, (ax, ay) = slabs[((k, j), k, h)]
            B, (bx, by) = slabs[((k, j), j, h)]
            # a point at (x, y) of A's box is in B's box at the same place plus what whole pixels did not take
            expect = (np.array([ax, ay]) - centre[k]) + p - (np.array([bx, by]) - centre[j])
            start = (0, 0, 0)
            if far:
                # the two slabs whole, to the pixel: the middle of A's against all of B's, the largest correlation within `far`
                core = np.zeros_like(B)
                core[padz:padz + ns, pad:size - pad, pad:size - pad] = A[padz:padz + ns, pad:size - pad, pad:size - pad]
                cc = np.fft.irfftn(np.conj(np.fft.rfftn(core)) * np.fft.rfftn(B), B.shape)
                win = cc[np.ix_(np.arange(-far, far + 1) % B.shape[0], np.arange(-far, far + 1) % B.shape[1], np.arange(-far, far + 1) % B.shape[2])]
                at = np.unravel_index(int(np.argmax(win)), win.shape)
                start = tuple(int(v) - far for v in at)
                log(f"  {labels[k]}, {labels[j]}, slab {h * f}: whole slabs at (slices, y, x) = {tuple(v * f for v in start)}; peak {win.max() / max(win.std(), 1e-30):.1f} of the window's spread")
            n = 0
            for y in range(m1 + pad, size - P1 - m1 - pad + 1, stride):
                for x in range(m2 + pad, size - P2 - m2 - pad + 1, stride):
                    pa = np.ascontiguousarray(A[padz + mz:padz + mz + nz, y:y + P1, x:x + P2])
                    pa /= max(pa.std(), 1e-12)
                    # The two boxes are cut on whole pixels, up to one pixel apart, and the tiles may differ by a part
                    # of another: a block the registration puts a whole pixel off is cut again there (its `shift` is
                    # then only a first estimate). Before this, every block of such an overlap was dropped.
                    oz, oy, ox = start
                    for _ in range(4):
                        pb = np.ascontiguousarray(B[padz + oz:padz + oz + ns, y - m1 + oy:y + P1 + m1 + oy, x - m2 + ox:x + P2 + m2 + ox])
                        r = reg.run(pa, pb / max(pb.std(), 1e-12))
                        w = tuple(r["whole"])
                        if w == (0, 0, 0) or abs(oz + w[0]) > padz or abs(oy + w[1]) > pad or abs(ox + w[2]) > pad:
                            break
                        oz, oy, ox = oz + w[0], oy + w[1], ox + w[2]
                    if not (r["converged"] and not r["at_edge"] and r["correlation"] > least and w == (0, 0, 0)):
                        continue
                    sh = (r["shift"][0] + oz, r["shift"][1] + oy, r["shift"][2] + ox)
                    xk, yk = ax + x + (P2 - 1) / 2 - centre[k], ay + y + (P1 - 1) / 2 - centre[k]
                    out.append(dict(k=k, j=j, height=h * f, x=float(xk) * f, y=float(yk) * f, xj=float(xk + p[0]) * f, yj=float(yk + p[1]) * f,
                                    dx=float(sh[2] - expect[0]) * f, dy=float(sh[1] - expect[1]) * f, dz=float(sh[0]) * f, correlation=float(r["correlation"])))
                    n += 1
            log(f"  {labels[k]}, {labels[j]}, slab {h * f}: {n} blocks")
    return out


def overlap_means(rows):
    """The table of overlaps() by overlap and slab: {(k, j, height): (median dx, dy, dz, blocks, spread)}; the median
    of an overlap's blocks is what a tile's place answers to, and it does not move with a block gone astray.
    spread: the blocks' robust scatter about it in the slice [px]."""
    out = {}
    for key in sorted({(r["k"], r["j"], r["height"]) for r in rows}):
        d = np.array([[r["dx"], r["dy"], r["dz"]] for r in rows if (r["k"], r["j"], r["height"]) == key])
        med = np.median(d, 0)
        out[key] = (float(med[0]), float(med[1]), float(med[2]), len(d), float(1.4826 * np.median(np.hypot(*(d[:, :2] - med[:2]).T))))
    return out


def place_tiles(cal, rows, least_blocks=4, log=print):
    """The simplest fit of a mosaic in the volume: one shift per tile (x, y in the slice, z along the axis; the
    first tile stays), such that every overlap's two images agree: d(k, j) = s_j - s_k for the median displacement
    of each overlap and slab of overlaps()' table, by least squares over all of them (overlaps with fewer than
    least_blocks blocks left out). Nothing else is fitted. Returns (the calibration with the tiles' places and
    heights moved, summary): summary["shift"] per tile [px, px, slices], "before" and "left" the overlaps' medians
    as measured and as the shifts leave them [px rms in the slice, slices rms along the axis], "worst" the largest
    left in the slice, and "left_by_overlap". What is left is what one shift per tile cannot hold: it closes round
    every loop of tiles or it does not."""
    from .solve import Problem

    labels = cal["layout"]["tiles"]
    n = len(labels)
    means = {key: v for key, v in overlap_means(rows).items() if v[3] >= least_blocks}
    keys = sorted(means)
    k = np.array([key[0] for key in keys])
    j = np.array([key[1] for key in keys])
    d = np.array([means[key][:3] for key in keys])
    prob = Problem()
    prob.unknowns("s", 3 * (n - 1))
    eqs = []
    for c in range(3):
        e = prob.equations(d[:, c])
        e.add("s", 3 * (j - 1) + c, 1.0, j > 0)
        e.add("s", 3 * (k - 1) + c, -1.0, k > 0)
        eqs.append(e)
    sol = prob.solve()
    left = np.stack([sol.left(e) for e in eqs], 1)
    s = np.concatenate([np.zeros(3), np.asarray(sol["s"], float)]).reshape(n, 3)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")
    out = json.loads(json.dumps(cal))
    for t in range(1, n):
        T = out["tiles"][labels[t]]
        T["place_px"] = [T["place_px"][0] + float(s[t, 0]), T["place_px"][1] - float(s[t, 1])]        # the slice's (x, y) is the place's (c, -s)
        T["height_rows"] = T["height_rows"] + PLACE_Z_SIGN * float(s[t, 2])
    summary = dict(shift={labels[t]: [float(v) for v in s[t]] for t in range(n)}, overlaps=len(keys), unknowns=int(sol.rank),
                   before=[rms(np.hypot(d[:, 0], d[:, 1])), rms(d[:, 2])], left=[rms(np.hypot(left[:, 0], left[:, 1])), rms(left[:, 2])],
                   worst=float(np.hypot(left[:, 0], left[:, 1]).max()) if len(keys) else float("nan"),
                   left_by_overlap={f"{labels[a]}, {labels[b]}, {h}": [float(v) for v in left[i]] + [means[(a, b, h)][3], means[(a, b, h)][4]] for i, (a, b, h) in enumerate(keys)})
    log(f"  one shift per tile from {len(keys)} overlap slabs: in the slice {summary['before'][0]:.3f} -> {summary['left'][0]:.3f} px rms (worst {summary['worst']:.3f}), "
        f"along the axis {summary['before'][1]:.3f} -> {summary['left'][1]:.3f} slices")
    return out, summary


# how a tile's height_rows answers to a displacement along the axis measured by overlaps() (dz of tile j against tile k):
# found by moving one tile's height by a known amount (development_notes/calibration/mosaic/exp21_known.py)
PLACE_Z_SIGN = 1.0


# ---------------------------------------------------------------- the axis column along the scan, measured

def mirror_along(session, lay, model, step=600, device="auto", tiles=None):
    """Each tile's mirror column along the frames on disk: the opposite views (opposite(): blocks half a turn apart,
    one reversed) at positions every `step` frames, registered on the grid of the model's map. Nothing is fitted.
    Returns one dict per accepted comparison: tile, frame_a (the first block's first frame), frame (its middle),
    row, col (the block's middle on the detector), mirror (the mirror column), correlation. A comparison of the
    blocks at p and p + N/2 gives m(p) = [C(p) + C(p + N/2)] / 2 of the axis column C: axis_course() makes a
    course of the table."""
    N = lay.per_turn
    out = []
    for k in (range(len(lay.labels)) if tiles is None else tiles):
        a, b = lay.on_disk[k]
        frames = [int(p) for p in np.arange(a + 200, b - N // 2 - 200 - NQ, step)]
        rows = measure(session, lay, model, opposite(lay, k, frames, model["a"][k], N // 2), device=device)
        for x, r in zip(describe(lay, model, rows), rows):
            out.append(dict(tile=k, frame_a=int(x["frame_a"]), frame=float(x["fa"]), row=float(x["va"]), col=float(x["ca"]), mirror=float(x["Y"] / 2), correlation=float(r["correlation"])))
    return out


def course_of_mirror(frames, mirror, at, per_turn):
    """The axis column's course C at the frames `at` from mirror columns m at `frames` (increasing), where
    m(p) = [C(p) + C(p + N/2)] / 2: C(f) = [m(f) + m(f - N/2)] / 2, the two comparisons the frame f is part of, m
    interpolated between its positions (where only one of the two was measured, that one; NaN where neither).
    Exact for a drift and for what repeats every half turn; what changes sign after half a turn (the odd harmonics
    of the turn) is not in opposite views and is not in the course."""
    frames, mirror, at = np.asarray(frames, float), np.asarray(mirror, float), np.asarray(at, float)
    parts = []
    for lag in (0.0, per_turn / 2):
        x = at - lag
        parts.append(np.where((x >= frames[0]) & (x <= frames[-1]), np.interp(x, frames, mirror), np.nan))
    parts = np.stack(parts)
    n = np.isfinite(parts).sum(0)
    return np.where(n > 0, np.nansum(parts, 0) / np.maximum(n, 1), np.nan)


def axis_course(cal, table, common=True, least=3):
    """mirror_along()'s table put into a calibration as the axis column along the scan: per tile and position the
    median of the comparisons about the tile's own median (positions with fewer than `least` comparisons left
    out), and from it each tile's column_course at the layout's knots (course_of_mirror; the end values held
    beyond the measured frames; its mean over the measured knots taken out, so that axis_column stays the tile's
    axis). common: every tile gets the median of the tiles' courses (nothing moves the axis between tiles: what
    the tiles show in common is the stage's, and seven tiles measure it better than one); False: each its own,
    the common one where a tile has no position of its own.
    Returns (the calibration, summary): summary["frames"], "tiles" (tile x position, px about each tile's median),
    "common", and per tile "against_others": [the tile's course rms, rms less the other tiles' median, their
    correlation], which says whether there is a common course at all."""
    L = cal["layout"]
    labels, K, N = L["tiles"], np.array(L["knot_frames"], float), L["frames_per_turn"]
    pos = sorted({r["frame_a"] for r in table})
    frame = np.array([np.mean([r["frame"] for r in table if r["frame_a"] == p]) for p in pos])
    M = np.full((len(labels), len(pos)), np.nan)
    for k in range(len(labels)):
        mine = [r for r in table if r["tile"] == k]
        if not mine:
            continue
        med = np.median([r["mirror"] for r in mine])
        for i, p in enumerate(pos):
            v = [r["mirror"] for r in mine if r["frame_a"] == p]
            if len(v) >= least:
                M[k, i] = np.median(v) - med
    seen = np.isfinite(M).any(0)
    both = np.where(seen, np.nanmedian(np.where(seen, M, 0.0), 0), np.nan)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")
    against = {}
    for k, t in enumerate(labels):
        rest = np.delete(M, k, 0)
        ok = np.isfinite(M[k]) & np.isfinite(rest).any(0)
        if ok.sum() < 3:
            continue
        others = np.nanmedian(rest[:, ok], 0)
        against[t] = [rms(M[k, ok]), rms(M[k, ok] - others), float(np.corrcoef(M[k, ok], others)[0, 1])]
    out = json.loads(json.dumps(cal))
    for k, t in enumerate(labels):
        y = both if common else np.where(np.isfinite(M[k]), M[k], both)
        ok = np.isfinite(y)
        if ok.sum() < 2:
            continue
        c = course_of_mirror(frame[ok], y[ok], K, N)
        inside = np.isfinite(c)
        c = np.interp(K, K[inside], c[inside])
        out["tiles"][t]["column_course"] = [float(x) for x in c - c[inside].mean()]
    return out, dict(frames=[float(x) for x in frame], tiles=M.tolist(), common=[float(x) for x in both], against_others=against)
