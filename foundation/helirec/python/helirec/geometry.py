"""The geometry calibration step: the detector's map and the places of the scans of one sample (a scan alone, a
pair moved sideways, the tiles of a mosaic), measured from the acquisition itself with nothing typed in. Python's
side of it: everything here calls the C++ (helirec/calibration/geometry_calibration.hpp makes the calibration and
states its model, helirec/geometry.hpp reads it for the reconstruction); from a shell it is
`helirec-calibration geometry MAIN [COMPANION ..] --out geometry.json`.

    cal = calibrate_geometry([main_h5, companion_h5])        # main: a scan with the rotation axis inside its field
    save(cal, "geometry.json")

What it finds, in this order:

  layout   frames per turn, the stretches on disk, which way the rows run against the axis stage, a coarse axis
           column per scan, the rows that hold something across the whole width (the mount), which are kept out of
           every comparison, and from the record's motors each scan's place and the anchor;
  rows     the time of a detector row within the frame, from the beam dips of a refill if a stretch has one
           (otherwise it must be given; 0 where it is not known);
  views    blocks of frames that hold the same lines on different parts of the detector (opposite views, a rest turn
           against the helix, the helix a turn apart, one scan against another at equal angle), cut into patches
           and registered once on the frames as stored;
  solve    one model for all of it (solve_geometry): the detector's map (pixel (c, v) looks along the ray at
           (c + Eu(c, v), v + Ev(c, v)), Eu and Ev polynomials), each scan's axis column, its slope along the
           travel and its axial scale, the rest turns' heights, every scan's place and height against the
           anchor's, the roll, what the rotation stage does within a turn, and for a mosaic's tiles what their own
           opposite views show of the axis' pitch towards the beam. No comparison has an offset of its own: what
           the model does not explain stays in the residuals, which the result reports per kind;
  report   what the solution leaves of each scan's opposite views and what that says (report), and which scans
           overlap (neighbours).

The step writes what it found; the reconstruction (recon.py) takes its geometry from that alone: frame_poses,
reconstruction_terms and merged_layout turn the result into the pose of every frame, the reconstruction's terms
per projection and the frame the two scans are merged in. The step uses helirec::calibration's session (raw
frames, the scan's own dark and mean flat, the registration) and nothing of the reconstruction's intensity
calibration (calibration.py).
"""
from __future__ import annotations

import json

import numpy as np

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


# ---------------------------------------------------------------- the step's stages
# (helirec/calibration/geometry_calibration.hpp; helirec-calibration geometry runs them in turn)

def acquisition(acq):
    """The scans' layout for the joint model as the C++ side holds it (_core.GeometryAcquisition), from a dict:
    rows, cols, row_time [frames per row], and optionally per_turn, pixel_um, row_sign, fan, per_depth (what the
    beam's divergence magnifies a sample by per pixel it sits nearer the source; 0: left out), width, anchor (the
    scan whose place is the origin); scans = [dict(name, mid_height, axis (the coarse axis column), moved_on_stage
    (the sample sits elsewhere on the rotation stage than in the anchor), on_stage (where, by the motors [px]),
    stretches=[dict(name, helix, angle_deg, rows, and optionally index, first, end (the frames on disk))])]. A
    GeometryAcquisition is returned as it is."""
    from . import _core
    if isinstance(acq, _core.GeometryAcquisition):
        return acq
    f = lambda x: [float(v) for v in x]
    scans = []
    for k, s in enumerate(acq["scans"]):
        stretches = []
        for a, t in enumerate(s["stretches"]):
            d = dict(name=str(t["name"]), index=int(t.get("index", a)), helix=bool(t.get("helix", False)), angle_deg=f(t["angle_deg"]), rows=f(t["rows"]))
            d.update({key: int(t[key]) for key in ("first", "end") if key in t})
            stretches.append(d)
        d = dict(name=str(s.get("name", k)), mid_height=float(s.get("mid_height", 0.0)), stretches=stretches, moved_on_stage=bool(s.get("moved_on_stage", False)))
        if s.get("axis") is not None:
            d["axis"] = float(s["axis"])
        if s.get("on_stage") is not None:
            d["on_stage"] = [float(v) for v in s["on_stage"]]
        scans.append(d)
    top = {key: acq[key] for key in ("per_turn", "pixel_um", "row_sign", "fan", "per_depth", "width", "anchor", "period_s") if key in acq}
    return _core.GeometryAcquisition(json.dumps(dict(top, rows=int(acq["rows"]), cols=int(acq["cols"]), row_time=float(acq.get("row_time", 0.0)), scans=scans)))


def plan_comparisons(acquisition_, angles_per_turn=12, max_positions=48, n_rest=16, n_turn=12):
    """The comparisons of each scan with itself, placed from the layout alone: a list of dicts (kind, scan_a, a, b
    (the stretches in the layout), from_block, and the request's fields).
    (helirec::calibration::geometry_calibration::plan for an Acquisition)"""
    from . import _core
    try:
        return json.loads(_core.geometry_plan(acquisition(acquisition_), int(angles_per_turn), int(max_positions), int(n_rest), int(n_turn)))
    except RuntimeError as e:
        raise ValueError(str(e)) from None


def measure_comparisons(session, acquisition_, device="auto", patch=128, **counts):
    """The plan of each scan's comparisons with itself (plan_comparisons) measured once on a CalibrationSession, in
    patches of patch x patch: the accepted ones as a list of dicts (kind, a, b, opposite, correlation, curvature,
    block: the plan's block it came from), and the number of blocks planned.
    (helirec::calibration::geometry_calibration::measure)"""
    from . import _core
    return _measured(_core.geometry_measure(session, acquisition(acquisition_), device, int(patch), **{k: int(v) for k, v in counts.items()}))


def _measured(got):
    out = [dict(kind=k, a=a, b=b, opposite=o, correlation=float(c), curvature=cu, block=int(bl))
           for k, a, b, o, c, cu, bl in zip(got["kinds"], got["a"], got["b"], got["opposite"], got["correlation"], got["curvature"], got["block"])]
    return out, int(got["blocks"])


def plan_tiles(acquisition_, places, pairs, positions=48, angles=24, bands=3, windows=4):
    """The comparisons of scans that sit at different places on the rotation stage (a mosaic's tiles) with each
    other, placed from `places` (the joint model's parameters: each scan's axis column, d and height). pairs: (j, k)
    of the scans to compare (neighbours(acq)[0]: those whose discs overlap). At `positions` positions along the
    helix, at `angles` angles a turn: "tiles" at equal angle where the two detectors hold the same rays, and
    "tiles_opposite" with the second scan half a turn on, mirrored; up to `bands` bands of rows and `windows` windows
    of columns each. A list of dicts as plan_comparisons gives, with the request's sway.
    (helirec::calibration::geometry_calibration::plan_tiles)"""
    from . import _core
    try:
        return json.loads(_core.geometry_plan_tiles(acquisition(acquisition_), _parameters_text(places), [[int(j), int(k)] for j, k in pairs], int(positions), int(angles), int(bands), int(windows)))
    except RuntimeError as e:
        raise ValueError(str(e)) from None


def measure_tiles(session, acquisition_, places, pairs, device="auto", patch=128, positions=48, angles=24, bands=3, windows=4):
    """plan_tiles measured once on a CalibrationSession, in patches of patch x patch: the accepted comparisons and
    the number of blocks planned, as measure_comparisons returns them.
    (helirec::calibration::geometry_calibration::plan_tiles, measure)"""
    from . import _core
    return _measured(_core.geometry_measure_tiles(session, acquisition(acquisition_), _parameters_text(places), [[int(j), int(k)] for j, k in pairs], device, int(patch), int(positions),
                                                  int(angles), int(bands), int(windows)))


def _parameters_text(p):
    """The joint model's parameters (a start, or a solution with whatever else it holds) as the C++ side reads them."""
    if p is None:
        return ""
    out = dict(roll=p.get("roll", 0.0), pitch=p.get("pitch", 0.0), stage=p.get("stage", {}), about_height=p.get("about_height", 0.0), scans=p["scans"])  # (a scan's height_once and rotation with it)
    if p.get("map") and len(p["map"].get("i", ())):
        out["map"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in p["map"].items()}
    if isinstance(p.get("row_time"), (int, float)):   # frames per row (a solution's); without it, the acquisition's
        out["row_time"] = float(p["row_time"])
    return json.dumps(out, default=float)


def _sightings(rows):
    return np.ascontiguousarray(np.asarray(rows, np.float64).reshape(-1, 5))


def rays(acquisition_, parameters, sightings):
    """The ray each sighting (scan, stretch, frame, row, column) records under the joint model's parameters:
    (n, 3) angle [deg], distance s from the sample's origin across the beam [px], height z [rows].
    (helirec::calibration::geometry_calibration::ray_of)"""
    from . import _core
    return _core.geometry_rays(acquisition(acquisition_), _parameters_text(parameters), _sightings(sightings))


def residuals(acquisition_, parameters, comparisons):
    """What the parameters leave of each comparison: (n, 3) frames (the two rays' angles, less the half turn of
    opposite views), rows and columns. (helirec::calibration::geometry_calibration::residuals)"""
    from . import _core
    return _core.geometry_residuals(acquisition(acquisition_), _parameters_text(parameters), _sightings([c["a"] for c in comparisons]),
                                    _sightings([c["b"] for c in comparisons]), [bool(c.get("opposite", False)) for c in comparisons])


def report(acquisition_, parameters, comparisons):
    """What the parameters leave of each scan's opposite views, and what that says beyond a parameter: per scan
    "opposite_views" (what is left as a mean, once and twice per turn of the angle; the rows' timing their frames
    would give; whether the recorded height's departure from a straight line is the sample's motion: height_slope 1)
    and, for a scan elsewhere on the rotation stage than the anchor, the pitch its own views' height_once says the
    rotation axis has at least; "over_scans" (the spread of the axis column, its slope and the axial scale);
    "edges" and "triangles" (neighbours); "pairs": per pair of scans compared with each other and per kind, what is
    left of frames, rows and columns as an rms, a median, and a mean with a once per turn of the angle (for a
    mosaic's tiles the rows' once per turn at equal angle is what the pitch left, in opposite views what the pitch
    does over the depth of what was registered). (helirec::calibration::geometry_calibration::report)"""
    from . import _core
    return json.loads(_core.geometry_report(acquisition(acquisition_), _parameters_text(parameters), _sightings([c["a"] for c in comparisons]),
                                            _sightings([c["b"] for c in comparisons]), [bool(c.get("opposite", False)) for c in comparisons],
                                            [str(c.get("kind", "")) for c in comparisons]))


def neighbours(acquisition_):
    """The scans whose reconstructions share a part of the sample (a mosaic's neighbours, the two scans of a pair),
    by the motors' places and each scan's disc about its axis, and the triangles of them: (edges, triangles) as
    tuples of scan indices. (helirec::calibration::geometry_calibration::neighbours)"""
    from . import _core
    edges, triangles = _core.geometry_neighbours(acquisition(acquisition_))
    return list(edges), list(triangles)


def _lens(lens):
    """Lens patches as the table the C++ side takes: scan a, scan b, the place, the displacement, its sigma."""
    rows = [[float(l["scan_a"]), float(l["scan_b"]), *map(float, l["p"]), *map(float, l["moved"]), *map(float, l.get("sigma", (1.0, 1.0, 1.0)))] for l in (lens or ())]
    return np.ascontiguousarray(np.asarray(rows, np.float64).reshape(-1, 11))


def lens_residuals(acquisition_, parameters, lens):
    """What the parameters leave of each lens patch: (n, 3) x, y, z [px], the difference of the two scans' rigid
    motions at the patch's place less what was measured. (helirec::calibration::geometry_calibration::lens_residuals)"""
    from . import _core
    return _core.geometry_lens_residuals(acquisition(acquisition_), _parameters_text(parameters), _lens(lens))


def solve_geometry(acquisition_, comparisons, start=None, huber=0.03, max_iterations=100, held=(), freed=(), lens=None, lens_huber=1.5):
    """The joint geometry model solved on comparisons (helirec/calibration/geometry_calibration.hpp states the model).

    acquisition_: the scans' layout (a dict as `acquisition` takes it, or a GeometryAcquisition): what the records
    state per frame. comparisons: dicts with kind, a and b (each a sighting: scan, stretch, frame, row, column) and
    opposite. start: the parameters to start from, as the result
    holds them (None: zeros); its "map" (empty_map) gives the detector map's terms, none without it. held, freed:
    names of unknowns, or their beginnings up to a dot ("map", "stage", "scan1"), to hold or free against the
    default: free are the scans' own axis, axis_slope and axial_scale, the roll, the rotation stage's height once
    per turn and axis column twice per turn, the rest turns' heights, of every scan but the anchor its d and
    height, and of every scan elsewhere on the rotation stage than the anchor (a mosaic's tile) its height_once (the
    height once per turn its own views show beyond the stage's: what an axis pitched towards the beam does to
    opposite views); held are the anchor's d, height and height_once, a helix' height and the map. "pitch" (the
    rotation axis' lean towards the beam; in comparisons of two scans at one angle) is free where such comparisons
    join scans at different places on the rotation stage. A scan's tilt_x, tilt_y are in comparisons of two scans
    too (the part about the beam's direction) and free there only by name; the spin is not in them. The rows of
    two scans' opposite views ("tiles_opposite") are not fitted, their columns are. "row_time" (the rows' timing in
    frames per row: `start`'s, else the acquisition's) is held unless freed by name; scans at different places on
    the rotation stage give it (their column offset at one angle is sheared along the rows by it, and a scan off
    the origin sees its own opposite views that much off half a turn); one scan about its own axis hardly tells it
    from its axial scale ("correlated" lists the two). "scan<k>.column_scale" (the scale of a scan's image across the
    beam against the anchor's: what it records u columns from its axis lies u (1 + column_scale) from it) is held
    unless freed by name; a scan's own views do not see it, its comparisons with another scan do, and the anchor's
    stays at nothing.

    lens: lens patches, the other kind of measurement: two scans' reconstructions of one place of the sample
    registered in 3D. Dicts with scan_a, scan_b, p (the place of the sample: x, y from its origin, z its height),
    moved (where scan b's volume has it against where scan a's has it: x, y, z) and sigma (what the measurement is
    good to per component; 1 if left out). Scan k holds the point p moved by (d_k, height_k) + rotation_k x (p -
    (0, 0, about_height)) against the anchor, about_height taken from `start`. A scan's rotation (tilt_x, tilt_y,
    spin) is free where lens patches name the scan and it is not the anchor. lens_huber: the Huber scale of a lens
    patch's residual, in sigmas.

    Returns the solution (roll, pitch, row_time, stage, about_height, scans, map, free, unused, correlated, solver, left) with
    "residual": (n, 3) frames, rows, columns, and "lens_residual": (n, 3) x, y, z. "correlated" lists pairs of free
    unknowns the measurements hardly tell apart (correlation above 0.9 in size). It is an error if the free
    unknowns depend on each other.
    (helirec::calibration::geometry_calibration::solve_geometry)"""
    from . import _core
    try:
        text, residual, lens_residual = _core.geometry_solve(acquisition(acquisition_), [str(c["kind"]) for c in comparisons], _sightings([c["a"] for c in comparisons]),
                                                             _sightings([c["b"] for c in comparisons]), [bool(c.get("opposite", False)) for c in comparisons], _lens(lens),
                                                             _parameters_text(start), float(huber), int(max_iterations), list(held), list(freed), float(lens_huber))
    except RuntimeError as e:
        raise ValueError(str(e)) from None
    out = json.loads(text)
    out["residual"] = residual
    out["lens_residual"] = lens_residual
    return out


def to_stored_pixels(acquisition_, parameters, rows, cols, f):
    """The joint model's parameters, found on frames read at 1 / f (a session with reduce), in the pixels of the
    frames as stored (rows x cols): positions times f, ratios as they are, each axis column moved along the roll to
    the stored frames' middle row. Returns (the parameters, the layout without its per-frame series) as dicts.
    (helirec::calibration::geometry_calibration::to_stored_pixels)"""
    from . import _core
    p, layout = _core.geometry_to_stored_pixels(acquisition(acquisition_), _parameters_text(parameters), int(rows), int(cols), int(f))
    return json.loads(p), json.loads(layout)


# ---------------------------------------------------------------- the pose of every frame

def frame_poses(cal, angles_deg, heights_rows):
    """Where every frame of the two scans of a pair holds the sample, from the step's result `cal` (or from a file
    of the earlier two-scan step, with its motion stage, which the last paragraph below describes).

    angles_deg[s], heights_rows[s]: the record's angle (unwrapped) and height (rows: the layout's row_sign x height
    / pixel) of every frame of scan s, the stretches one after the other (rest1, helix, rest2).

    Returns per scan a dict of arrays with one value per frame. The ray at u columns from the rotation axis (towards
    higher columns) in the slice at height h (rows) is recorded by frame k at the position (X, Y) of the scan's grid,
        Y = h + row[k] + row_per_col[k] u,        X = axis[k] + u + col_per_row[k] (Y - middle row),
    which is the detector pixel (c, v) with
        c + Eu(c, v) + col_scale[k] (c - scale_col) = X,        v + Ev(c, v) + row_scale[k] v = Y
    (Eu, Ev: the step's map), recorded at the angle angle_deg[k] + angle_deg_per_row[k] (v - middle row).

    The step's poses are its model's, frame by frame (helirec/geometry.hpp lists them): the row from the scan's
    height, its stretch's, the axial scale and the stage's lift once per turn; the axis from the scan's axis
    column, its slope along the travel, the stage's move twice per turn and the scan's place; the slopes from the
    roll; the rows' timing in angle_deg_per_row and row_scale. What the solve left is under "not_applied".

    A file of the earlier two-scan step: the first scan is placed by what its own redundant views measured (the
    helix against the encoder, the rest turns against the helix at equal angle, the axis' column of each stretch
    with the rotating body's place against it, the height once per turn), the second scan is the first one's model
    at the same frame plus what the comparison of the two scans at equal frames measured, and the terms that step
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


def overlap_centre_of(cal):
    """The middle of the columns both scans of a calibration hold, in the first scan's columns."""
    from . import _core
    return _core.GeometryCalibration(_text(cal)).overlap_centre


# ---------------------------------------------------------------- the step

def calibrate_geometry(scans, row_time=None, device="auto", order=3, threads=8, log=print, patch=128, reduce=0, projections=""):
    """The geometry calibration of the scans of one sample. scans: their master files, a scan with the rotation
    axis inside the detector's field first (a pair: the main scan, then the one moved sideways). row_time: seconds
    per stored row (signed), if it is known for this camera configuration (0: not known, done without); otherwise
    it is measured from a refill, and it is an error if no stretch on disk holds one. order: of the detector map's
    polynomial (0: no map). patch: the plan's blocks are cut into patches of patch x patch, each registered once.
    reduce: 1, 2, 3 to measure on the frames read at a half, a quarter, an eighth (the result is in stored pixels
    all the same; the C++ Options say by how much it then differs from the stored frames'). projections: a projection
    store (STORE.zarr of helirec-projections-zarr) to read the frames, darks and flats from, at the level `reduce`.
    Returns a dict (see `save`): layout, row_time, map, parameters, solution (what is left per kind), report, comparisons.
    (helirec::calibration::geometry_calibration::calibrate_geometry; the same as `helirec-calibration geometry`)"""
    from . import _core
    try:
        text = _core.calibrate_geometry([str(x) for x in scans], None if row_time is None else float(row_time), device, int(order), int(threads), int(patch), log, int(reduce), str(projections))
    except RuntimeError as e:
        raise ValueError(str(e)) from None
    return json.loads(text)


def save(cal, path):
    """The calibration as JSON, written whole or not at all (a reconstruction may be reading the cache)."""
    from . import _core
    _core.geometry_calibration_save(json.dumps(cal, indent=2, default=float), str(path))
