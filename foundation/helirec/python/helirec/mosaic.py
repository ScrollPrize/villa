"""One tile of a mosaic, reconstructed alone.

A mosaic is scans of one sample at several places on the rotation stage (tiles), each a helix with the rotation
axis inside the detector. `tile_reconstructor` makes a helirec reconstructor of one tile from a description of
the tiles' geometry.

Nothing in helirec makes such a description at present. The mosaic's geometry calibration that was here (its
fits, and its check of two tiles' overlap in the volume) was removed in November 2026 and is to be done anew;
until then a mosaic cannot be calibrated, and it cannot be reconstructed as one volume. The form of the
description below is what this reconstructor reads. It is not a statement of how a mosaic's geometry should be
modelled or measured.

The description is a dict (JSON as `geometry.save` writes it):

    layout               rows, tiles (the labels), frames_on_disk (per tile, [first, end) of the helix' frames
                         used), row_sign, step_deg, knot_frames (the frame numbers a course is given at)
    map                  the detector's map (helirec.geometry: i, j, eu, ev, c0, cs, v0, vs)
    row_time             rad_per_row: the angle between the exposure of one stored row and the next
    axial_scale, axis_roll_rad, rows_slope_opposite
    angle_course_rad     T, at the knots
    tiles[label]         axis_column a, place_px (c1, s1), height_rows h, yaw_rad, and at the knots
                         column_course C and height_course R

and gives, for the frame f of a tile at the record's angle (courses are lines between their knots):

    th = th_record(f) + T(f) + yaw + rad_per_row (v - v0)      the angle at which the row v was exposed
    u  = a + c1 cos th + s1 sin th + C(f)                       where the sample's origin falls on the detector
    H  = h + R(f)                                               how far the sample lies from the record along the axis

The frames may be a partial copy (only the frames named are read) and stored as JPEG 2000; they can be read at a
half, a quarter or an eighth (`reduce`).
"""
from __future__ import annotations

import numpy as np


def _tiles(cal):
    """The description as the C++ side reads it (helirec/geometry.hpp, TileGeometry)."""
    import json

    from . import _core
    return _core.TileGeometry(cal if isinstance(cal, str) else json.dumps(cal, default=float))


def frame_pose(cal, label, angle_deg, frame, row=None):
    """For frames of a tile: the angle of `row` (the middle row if None) [deg], where the sample's origin falls on
    the detector (column), and how far the sample lies from the record along the axis (rows): th, u and H of the
    module's description, from a description `cal` of the tiles' geometry. angle_deg, frame: the record's angle
    and frame number of each frame. (helirec::geometry::TileGeometry::frame_pose)"""
    f = lambda x: np.ascontiguousarray(np.atleast_1d(x), np.float64)
    return _tiles(cal).frame_pose(label, f(angle_deg), f(frame), -1.0 if row is None else float(row))


def reconstruction_terms(cal, label, angle_deg, height_rows, frame, rot_center, row_origin=0.0):
    """One tile's frames as the reconstruction's per-projection terms (helical.hpp, ScanTrajectory; what
    geometry.reconstruction_terms is for a pair), in the tile's own frame: the slice's centre is the tile's
    rotation axis, and the slice h is the same height of the sample in every tile (row_origin, rows, moves that
    numbering; it must be the same for the tiles that are to be compared). The sample's origin lies at the tile's
    place (frame_pose's swing) from the centre.

    angle_deg, height_rows, frame: the record's angle (unwrapped), its height in rows (the layout's row_sign x
    height / pixel) and the frame number of each frame used. The reconstruction puts the column m of its frame at
    m - rot_center columns from the axis. (helirec::geometry::TileGeometry::reconstruction_terms)"""
    f = lambda x: np.ascontiguousarray(x, np.float64)
    return _tiles(cal).reconstruction_terms(label, f(angle_deg), f(height_rows), f(frame), float(rot_center), float(row_origin))


def tile_reconstructor(master, cal, label, device="cuda", params=None, row_origin=None, weight=None, cache_gb=0.0, reduced=None, log=print, cache_dir=None, reduce=0):
    """A reconstructor (helirec SliceReconstructor) of one tile alone, on a description `cal` of the tiles' geometry
    (the module's description): the detector map, the pose and the angle of every frame, the rows' timing. The tile's own frame (see
    reconstruction_terms); only the helix frames on disk are used. The first step towards reconstructing a mosaic; the intensity side is the pair pipeline's (the scan's own
    darks and flats by the ring current, a weight map from the flats with a taper at the detector's far edge and
    across the axis), and nothing knows yet that every ray passes through sample the tile does not see.

    params: reconstruction parameters over the defaults here (fbp, the ramp filter alone).
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
        # and not the stored middle (rows - 1) / 2 the description's angles are for
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
