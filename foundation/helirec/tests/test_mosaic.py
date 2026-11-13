"""helirec.mosaic: what a description of the tiles' geometry says of a frame (frame_pose), and the per-projection
terms the reconstruction of one tile is given (reconstruction_terms). No frames are read."""
import numpy as np

from helirec import mosaic as M

ROWS, COLS, N = 2368, 4416, 10000


def _description():
    K = np.arange(109000.0, 141500.0, 500.0)
    rng = np.random.default_rng(5)
    tile = lambda axis, place, height, yaw: dict(axis_column=axis, place_px=list(place), height_rows=height, yaw_rad=yaw,
                                                 column_course=(0.4 * rng.standard_normal(len(K))).tolist(), height_course=(0.3 * rng.standard_normal(len(K))).tolist())
    return dict(layout=dict(rows=ROWS, cols=COLS, tiles=["T0", "T1"], frames_on_disk=[[109700, 140300]] * 2, row_sign=-1.0, step_deg=0.036, knot_frames=K.tolist()),
                map=dict(i=[0, 1], j=[1, 0], eu=[0.0, 0.0], ev=[0.0, 0.0], c0=(COLS - 1) / 2, cs=COLS / 2, v0=(ROWS - 1) / 2, vs=ROWS / 2),
                row_time=dict(rad_per_row=-1.5e-7), axial_scale=-1.6e-3, axis_roll_rad=5e-5, rows_slope_opposite=-8e-6,
                angle_course_rad=(1e-5 * rng.standard_normal(len(K))).tolist(),
                tiles=dict(T0=tile(4106.25, (0.0, 0.0), 0.0, 0.0), T1=tile(4106.8, (10.8, -6927.8), 1.55, 3e-5)))


def test_course_is_a_line_between_its_knots():
    K = np.array([100.0, 200.0, 300.0])
    v = np.array([1.0, 3.0, -1.0])
    assert np.allclose(M.course(K, v, [100.0, 150.0, 200.0, 275.0, 300.0]), [1.0, 2.0, 3.0, 0.0, -1.0])
    assert np.allclose(M.course(K, np.stack([v, 2 * v]), 250.0), [1.0, 2.0])           # several courses at once


def test_frame_pose_follows_the_description():
    cal = _description()
    K, v0 = np.array(cal["layout"]["knot_frames"]), cal["map"]["v0"]
    f = np.array([110000.0, 125250.0, 139990.0])
    angle = 0.036 * f
    t = cal["tiles"]["T1"]
    th, u, h = M.frame_pose(cal, "T1", angle, f, row=100.0)
    want = np.radians(angle) + np.interp(f, K, cal["angle_course_rad"]) + t["yaw_rad"] + cal["row_time"]["rad_per_row"] * (100.0 - v0)
    assert np.allclose(np.radians(th), want, rtol=0, atol=1e-12)
    assert np.allclose(u, t["axis_column"] + t["place_px"][0] * np.cos(want) + t["place_px"][1] * np.sin(want) + np.interp(f, K, t["column_course"]))
    assert np.allclose(h, t["height_rows"] + np.interp(f, K, t["height_course"]))
    th0, _, _ = M.frame_pose(cal, "T1", angle, f)                                        # the middle row: no rows' timing
    assert np.allclose(np.radians(th0), want - cal["row_time"]["rad_per_row"] * (100.0 - v0), rtol=0, atol=1e-12)
    _, u0, h0 = M.frame_pose(cal, "T0", angle, f)                                         # the tile at the origin does not swing
    assert np.allclose(u0, 4106.25 + np.interp(f, K, cal["tiles"]["T0"]["column_course"])) and np.allclose(h0, np.interp(f, K, cal["tiles"]["T0"]["height_course"]))


def test_reconstruction_terms_of_a_tile():
    cal = _description()
    K = np.array(cal["layout"]["knot_frames"])
    f = np.arange(120000.0, 120400.0)
    angle = 0.036 * f
    Hrec = 0.2303 * (f - 125000.0)                                                         # the record's height in rows
    t = cal["tiles"]["T1"]
    rot_center, origin = 4107.0, 12.5
    T = M.reconstruction_terms(cal, "T1", angle, Hrec, f, rot_center, origin)
    th, u, H = M.frame_pose(cal, "T1", angle, f)
    assert np.allclose(T["angles_deg"], th) and T["z_translation_scale"] == 1.0
    # the tile's own frame: the axis' column without the place's swing, against the column the reconstruction centres on
    assert np.allclose(T["col_offset"], t["axis_column"] + np.interp(f, K, t["column_course"]) - rot_center, atol=1e-4) and T["slope_col"] == rot_center
    # content at the record's height w lies on the row (1 + scale) w + H; the slices are numbered from row_origin
    row = (1.0 + cal["axial_scale"]) * Hrec + H
    assert np.allclose(T["row_offset"], -(row + origin) - cal["layout"]["row_sign"] * (Hrec - Hrec[0]), atol=1e-3)
    moved = M.reconstruction_terms(cal, "T1", angle, Hrec, f, rot_center, origin + 7.0)
    assert np.allclose(moved["row_offset"] - T["row_offset"], -7.0, atol=1e-4)
    assert np.allclose(T["col_slope"], cal["axis_roll_rad"]) and np.allclose(T["row_slope"], -cal["rows_slope_opposite"]) and T["slope_row"] == (ROWS - 1) / 2
    # the rows' timing as a scale of the rows: the travel per frame times the frames per row
    per_row = cal["row_time"]["rad_per_row"] / np.radians(0.036)
    inner = slice(20, -20)
    assert np.allclose(T["row_scale"][inner], -np.gradient(row)[inner] * per_row, rtol=0.02)
    assert all(len(T[k]) == len(f) for k in ("angles_deg", "row_offset", "col_offset", "row_slope", "col_slope", "row_scale", "col_scale"))
