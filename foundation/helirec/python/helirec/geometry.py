"""The geometry calibration step: the detector map and the poses of a two-scan helical acquisition, measured from
the acquisition itself with nothing typed in. Python's side of it: everything here calls the C++
(helirec/calibration/geometry_calibration.hpp makes the calibration, helirec/geometry.hpp reads it for the
reconstruction); from a shell it is `helirec-calibration geometry MAIN COMPANION --out geometry.json`.

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

import json
from dataclasses import dataclass

import numpy as np

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


# ---------------------------------------------------------------- the step's stages
# The step and its stages are helirec/calibration/geometry_calibration.hpp (C++; helirec-calibration geometry runs
# it). What is here hands a layout, comparisons and results across as the dicts the callers and the tests know.

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
    # _heights (set by the caller): (scan, name) -> the record's height of every frame in rows


def _layout_text(lay):
    heights = {f"{k} {name}": [float(x) for x in v] for (k, name), v in getattr(lay, "_heights", {}).items()}
    return json.dumps(dict(rows=int(lay.rows), cols=int(lay.cols), per_turn=int(lay.per_turn), pixel_um=float(lay.pixel_um), stretches=lay.stretches, lengths=lay.lengths,
                           row_sign=float(lay.row_sign), axis=float(lay.axis), move=int(lay.move), width=int(lay.width), heights=heights))


def overlap_centre(lay):
    """The middle column, in the first scan, of what both scans record."""
    return (lay.cols - lay.move) / 2.0


def plan(lay):
    """The comparisons of one pass, placed from the layout alone: (requests for the session, what each compares)."""
    from . import _core
    try:
        got = json.loads(_core.geometry_calibration_plan(_layout_text(lay)))
    except RuntimeError as e:
        raise ValueError(str(e)) from None
    return [g[0] for g in got], [g[1] for g in got]


_FIT_KEYS = ("kind", "scan_a", "stretch_a", "frame_a", "row_a", "rows", "row_offset", "col_a", "col_b", "cols", "stated_dq", "dq", "stated_D", "D", "du", "slope_a", "slope_b")


def fit(rows, lay, m, tau, period, huber=0.03, max_iterations=100):
    """What the comparisons leave, as a correction of the map `m` (same terms), the poses beside it, and the residuals.

    A feature seen at A and at B obeys, with E the part of the detector's map the grid does not hold yet:
        columns, same view direction   du       = u0 - (Eu(B) - Eu(A))
        columns, opposite views        2 mirror = u0 - (Eu(B) + Eu(A)) + 2 psi (v - v0)
        rows                           y        = v0 - (Ev(B) - Ev(A)) [+ 2 psi (c - axis) for opposite views]
    y is the image's row displacement minus the encoder's at the instants the rows were exposed. u0, v0: one pair
    per frame position (whatever the sample and the stages did between positions is theirs, not the map's). psi:
    the roll of the first scan's axis. Between the scans, per stretch, rho (a rotation of the second scan's content
    about the beam) and a line in the column (a, b), whose mean over the stretches is left to the detector.
    (helirec::calibration::geometry_calibration::fit, on the steps' solver; an unusable solve is an error)"""
    from . import _core
    text = json.dumps([{k: (r[k] if isinstance(r[k], str) else float(r[k])) for k in _FIT_KEYS} for r in rows])
    try:
        out = json.loads(_core.geometry_calibration_fit(text, _layout_text(lay), json.dumps(m, default=float), float(tau), float(period), float(huber), int(max_iterations)))
    except RuntimeError as e:
        raise ValueError(str(e)) from None
    out["groups"] = {(k, s_, a, f): (u, v) for k, s_, a, f, u, v in out["groups"]}
    return out


def _update_range(m, d):
    """The largest displacement the correction d makes anywhere on the detector [px, rows]."""
    from . import _core
    return tuple(_core.geometry_calibration_update_range(json.dumps(m, default=float), [float(x) for x in d["eu"]], [float(x) for x in d["ev"]]))


def _block_size(limit):
    """The largest block size up to `limit` the registration takes as it is (quick transform sizes, with a margin)."""
    from . import _core
    return _core.geometry_calibration_block_size(int(limit))


def _nodes(t1, t2, th, n_extra, h, total):
    """Design of [d(t2) - d(t1)] . (cos, sin)(th), d at the nodes h, 2h, .. (d(0) = 0), a line between; then n_extra empty columns."""
    from . import _core
    f = lambda x: np.ascontiguousarray(x, np.float64)
    return _core.geometry_calibration_nodes(f(t1), f(t2), f(th), int(n_extra), float(h), float(total))


def second_scan_fit(table, rows, total):
    """The second scan against the first at equal frames, from the table of their comparisons (stretch, turns from
    the start, angle, middle row, columns, rows, angle in frames). Columns: per stretch an offset and a line in the
    row, in the helix a line in the travel, and once per turn a course with a point every turn (what a body that
    lies elsewhere against the axis shows, and how that changes). Rows and angle: the same with one term once per
    turn. Twice per turn: fitted beside, recorded, not explained.
    (helirec::calibration::geometry_calibration::second_scan_fit)"""
    from . import _core
    col = lambda k: np.ascontiguousarray([r[k] for r in table], np.float64)
    return json.loads(_core.geometry_calibration_second_scan_fit([str(r[0]) for r in table], col(1), col(2), col(3), col(4), col(5), col(6), int(rows), float(total)))


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
    patch: 0 measures the plan's blocks whole; N cuts them into patches of N x N, with which the first pass, on
    the frames as stored, gives the map and the second confirms it. Returns a dict (see `save`).
    (helirec::calibration::geometry_calibration::calibrate_geometry; the same as `helirec-calibration geometry`)"""
    from . import _core
    try:
        text = _core.calibrate_geometry([str(x) for x in scans], None if row_time is None else float(row_time), device, int(order), int(passes), float(tolerance), int(threads),
                                        bool(motion), int(patch), log)
    except RuntimeError as e:
        raise ValueError(str(e)) from None
    return json.loads(text)


def save(cal, path):
    """The calibration as JSON, written whole or not at all (a reconstruction may be reading the cache)."""
    from . import _core
    _core.geometry_calibration_save(json.dumps(cal, indent=2, default=float), str(path))
