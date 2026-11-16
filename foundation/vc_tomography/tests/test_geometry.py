"""helirec.geometry: the pose of every frame, the reconstruction's terms and the merged frame from a calibration of
the earlier two-scan step (a file with a motion stage, which helirec/geometry.hpp still reads; nothing writes one
any more). tests/test_geometry_solve.py covers the step that makes calibrations now, and its file."""
import numpy as np
import pytest

from helirec import geometry as G


def _synthetic_geometry(rng):
    """A step result with known terms, for a toy acquisition of 100 frames per turn: one rest turn, a helix of four, one rest turn."""
    N, R = 100, 200
    turn_rows = 150.0
    frames = dict(rest1=N, helix=4 * N, rest2=N)
    ang = np.concatenate([np.arange(N), N + np.arange(4 * N), 5 * N + np.arange(N)]) * 3.6 + 0.3
    H0 = np.concatenate([np.zeros(N), 0.5 + turn_rows * np.arange(4 * N) / N, np.full(N, 0.5 + 4 * turn_rows)])
    H1 = H0 + np.concatenate([np.zeros(N), np.full(4 * N, 0.25), np.zeros(N)])
    tf = (np.arange(0, 3 * N, 5) + 16.0)
    helix_table = lambda c0, c1, g0, g1: [[f, turn_rows, c0 + c1 * f / N, g0 + g1 * f / N] for f in tf]
    rest = lambda o, sc, q0: dict(offset_rows=o, scale_error=sc, angle_frames=q0, angle_frames_per_row=0.0)
    nodes0 = np.cumsum(rng.normal(0, 0.05, (12, 2)), 0).tolist()
    pair = lambda off, once: dict(offset=dict(rest1=off[0], helix=off[1], rest2=off[2]), per_row=dict(rest1=1e-5, helix=2e-5, rest2=3e-5), per_turn_of_travel=0.01, once_per_turn=once, twice_per_turn=[0.0, 0.0])
    second = dict(columns=dict(pair([0.3, 0.5, 0.6], None), once_per_turn_nodes=rng.normal(0, 0.1, (7, 2)).tolist()), rows=pair([-1.0, -0.9, -0.4], [0.02, -0.1]), angle_frames=pair([0.0, 0.0, 0.0], [0.0, 0.0]))
    motion = {"turns_at_start": [dict(rest1=0.0, helix=1.0, rest2=5.0)] * 2,
              "scan 0": dict(rest1=rest(-0.15, 3.4e-3, 0.03), rest2=rest(1.2, 3.3e-3, 0.0), helix=dict(table=helix_table(0.5, 0.03, 0.02, 0.01)),
                             sideways=dict(node_spacing_turns=0.5, nodes_px=nodes0, travel_px_per_row=6e-6, mirror_column=dict(rest1=140.2, helix=140.25, rest2=140.3, per_row=7e-6))),
              "scan 1": dict(helix=dict(table=helix_table(0.55, 0.02, 0.02, 0.01)), sideways=dict(node_spacing_turns=1.0, nodes_px=[[0.0, 0.0]] * 6)),
              "along_axis": dict(z_cos=-0.03, z_sin=-0.01), "second_scan": second}
    cal = dict(layout=dict(rows=R, cols=240, frames=[frames, frames], frames_per_turn=N, row_sign=-1.0, coarse_mirror_column=140.0, coarse_move=70),
               row_time=dict(seconds_per_row=-2e-5, frame_period_s=0.033),
               poses=dict(axis_roll_rad=1.5e-5, rotation_about_beam_rad=dict(rest1=3e-5, helix=2e-5, rest2=1e-5),
                          second_scan={a: dict(line=[1e-4 * k, -2e-5 * k]) for k, a in enumerate(("rest1", "helix", "rest2"), 1)}, motion=motion))
    return cal, [ang, ang.copy()], [H0, H1]


def test_frame_poses_hold_what_was_measured():
    """The poses give back the relations they are built from: the helix a turn apart, the rest turns against the
    helix at equal angle, the second scan against the first at equal frames."""
    cal, ang, H = _synthetic_geometry(np.random.default_rng(3))
    (p0, p1), not_applied = G.frame_poses(cal, ang, H)
    N, mo = 100, cal["poses"]["motion"]
    helix = slice(N, 5 * N)
    f = np.arange(20, 280)                                             # helix frames with a frame a turn on
    row, enc = p0["row"][helix], H[0][helix]
    turn = (row[f + N] - row[f]) - (enc[f + N] - enc[f])               # image minus encoder, a turn apart (z is the same at equal angle)
    assert np.allclose(turn, 0.5 + 0.03 * (f + 0.5) / N, atol=1e-9)
    g = -(p0["angle_deg"][helix] - ang[0][helix]) / 3.6                # frames beyond the encoder's angle
    assert np.allclose(g[f + N] - g[f], 0.02 + 0.01 * (f + 0.5) / N, atol=1e-9)
    k = np.arange(N)
    D = enc[k] - H[0][k]                                               # first rest turn against the helix at equal angle
    assert np.allclose(row[k] - p0["row"][k], D * (1 + 3.4e-3) - 0.15, atol=1e-9)
    assert np.allclose(g[k] - (-(p0["angle_deg"][k] - ang[0][k]) / 3.6), 0.03, atol=1e-9)
    D2 = enc[k + 3 * N] - H[0][5 * N + k]
    assert np.allclose(row[k + 3 * N] - p0["row"][5 * N + k], D2 * (1 + 3.3e-3) + 1.2, atol=1e-9)
    # the second scan at equal frames, at the middle of the overlap: columns and rows of the comparison
    P = mo["second_scan"]
    th = np.radians(ang[1])
    t_abs = (np.arange(6 * N) + 0.5) / N
    nodes = np.array(P["columns"]["once_per_turn_nodes"])
    E = np.stack([np.interp(t_abs, np.arange(7), nodes[:, j]) for j in (0, 1)], 1)
    trav = np.where((t_abs > 1) & (t_abs < 5), t_abs - 3.0, 0.0)
    off = lambda key: np.concatenate([np.full(N, P[key]["offset"]["rest1"]), np.full(4 * N, P[key]["offset"]["helix"]), np.full(N, P[key]["offset"]["rest2"])])
    assert np.allclose(p1["axis"] - p0["axis"] - 70, off("columns") + 0.01 * trav + E[:, 0] * np.cos(th) + E[:, 1] * np.sin(th), atol=1e-9)
    u_ref = G.overlap_centre_of(cal) - p0["axis"]            # the overlap's middle, from the axis (the body's course is in the axis: within 1e-5 rows)
    rows_there = (p1["row"] + p1["row_per_col"] * u_ref) - (p0["row"] + p0["row_per_col"] * u_ref) - (H[1] - H[0])
    bv = np.concatenate([np.full(N, 1e-5), np.full(4 * N, 2e-5), np.full(N, 3e-5)])      # the comparison's line in the row: a scale about the middle row
    assert np.allclose(rows_there + bv * 99.5, off("rows") + 0.01 * trav + 0.02 * np.cos(th) - 0.1 * np.sin(th), atol=2e-5)
    assert np.allclose(p1["row_scale"] - p0["row_scale"], -bv, atol=1e-12)
    assert np.all(p0["row_scale"][:N] == 0) and np.allclose(p0["row_scale"][helix][20:-20], 1.5 * 2e-5 / 0.033)
    assert np.allclose(p0["angle_deg_per_row"], 3.6 * -2e-5 / 0.033)
    assert any("twice per turn" in k_ for k_ in not_applied)
    # as the reconstruction's terms: the slice h lies at the row h - (z_pix + z_readjustment + row_offset) = h + row
    for s, p in enumerate((p0, p1)):
        t = G.reconstruction_terms(p, H[s], cal["layout"], rot_center=210.25, column_offset=70.5 * (1 - s), z_readjustment=0.5 * s, row_origin=0.25)
        z_pix = -(H[s] - H[s][0])
        assert np.allclose(-(z_pix * t["z_translation_scale"] + 0.5 * s + t["row_offset"]), p["row"] + 0.25, atol=2e-4)
        assert np.allclose(210.25 - 70.5 * (1 - s) + t["col_offset"], p["axis"], atol=2e-4)   # the scan's column that the merged axis reads
    assert G.rolling_shutter_of(cal) == pytest.approx(-2e-5 * 199 / 0.033)
