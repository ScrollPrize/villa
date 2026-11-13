"""helirec.geometry: the fit of the detector map recovers a known map, roll, rotation and lines from comparisons made
up from them (no frames), and the plan keeps every block on the detector."""
import numpy as np
import pytest

from helirec import geometry as G


def _layout():
    lay = G.Layout(rows=1200, cols=2528, per_turn=8000, pixel_um=8.64, stretches=[dict(rest1=2, helix=3, rest2=4)] * 2,
                   lengths=[dict(rest1=8000, helix=32000, rest2=8000)] * 2, row_sign=-1.0, axis=1684.8, move=843, width=1024)
    h = np.arange(32000) * (1053.3 / 8000)
    lay._heights = {(k, "rest1"): np.zeros(8000) for k in (0, 1)} | {(k, "rest2"): np.full(8000, h[-1]) for k in (0, 1)} | {(k, "helix"): h for k in (0, 1)}
    return lay


def test_plan_stays_on_the_detector():
    lay = _layout()
    reqs, meta = G.plan(lay)
    assert {m["kind"] for m in meta} == set(G.KINDS) and len(reqs) > 800
    for q, m in zip(reqs, meta):
        assert 0 <= q["col_a"] and q["col_a"] + q["cols"] <= lay.cols and 0 <= q["col_b"] and q["col_b"] + q["cols"] <= lay.cols
        assert q["row_a"] == -1 or (0 <= q["row_a"] and q["row_a"] + q["rows"] <= lay.rows)
        if m["kind"] == "opposite":
            assert q["col_a"] + q["col_b"] + q["cols"] - 1 == round(2 * lay.axis)
        if m["kind"] == "pair":
            assert q["col_b"] - q["col_a"] == lay.move
    with pytest.raises(ValueError):
        G.plan(G.Layout(rows=1200, cols=2528, per_turn=8000, pixel_um=8.64, stretches=lay.stretches, lengths=lay.lengths, axis=2500.0, move=843, width=1024))


def test_fit_recovers_map_and_poses():
    rng = np.random.default_rng(5)
    lay = _layout()
    true = G.empty_map(lay.rows, lay.cols, 3)
    true["eu"] = [0.02, -0.03, -0.58, 0.04, -0.27, 0.07, 0.81, 0.0]          # (0,2) (0,3) (1,1) (1,2) (2,0) (2,1) (3,0) (1,0)
    true["ev"] = [0.03, 0.10, -0.06, 0.05, 0.73, 0.17, -0.20, -0.49]
    psi, rho = -2e-5, dict(rest1=-6e-5, helix=-5e-5, rest2=-4e-5)
    line = dict(rest1=(1.0e-4, 0.8e-4), helix=(-0.2e-4, -0.1e-4), rest2=(-0.8e-4, -0.7e-4))    # mean zero over the stretches
    cref, vmid = G.overlap_centre(lay), true["v0"]
    reqs, meta = G.plan(lay)
    rows = []
    for q, m in zip(reqs, meta):
        N = 0
        if m["kind"] == "turn":
            N = 1053
        elif m["kind"] == "resthelix":
            hint = q["frame_hint"]
            N = int(round(lay._heights[(m["scan_a"], "helix")][q["frame_a"] + 16 + hint] - lay._heights[(m["scan_a"], m["stretch_a"])][q["frame_a"] + 16]))
        row_a = q["row_a"] if q["row_a"] >= 0 else (lay.rows - q["rows"] - N) // 2
        cA, cB = q["col_a"] + (q["cols"] - 1) / 2, q["col_b"] + (q["cols"] - 1) / 2
        vA = row_a + (q["rows"] - 1) / 2
        (eua, eva), (eub, evb) = G.map_displacement(true, cA, vA), G.map_displacement(true, cB, vA + N)
        u0, v0 = 0.3 * np.sin(q["frame_a"] / 900.0), 0.2 * np.cos(q["frame_a"] / 700.0) + 3e-3 * N     # whatever the stages did
        if m["kind"] == "opposite":
            two_mirror = 2 * lay.axis + u0 - (eub + eua) + 2 * psi * (vA - vmid)
            du, yv = q["col_a"] + q["col_b"] + q["cols"] - 1 - two_mirror, v0 - (evb - eva) + 2 * psi * (cA - lay.axis)
        else:
            du, yv = u0 - (eub - eua), v0 - (evb - eva)
            if m["kind"] == "pair":
                st = m["stretch_a"]
                du += -rho[st] * (vA - vmid) + line[st][0] * (cA - cref)
                yv += rho[st] * (cA - cref) + line[st][1] * (cA - cref)
        rows.append(dict(m, frame_a=q["frame_a"], row_a=row_a, rows=q["rows"], row_offset=N, col_a=q["col_a"], col_b=q["col_b"], cols=q["cols"], stated_dq=0.0, dq=0.0, stated_D=float(N),
                         D=float(N) + yv + rng.normal(0, 0.004), du=du + rng.normal(0, 0.004), slope_a=0.0, slope_b=0.0))
    f = G.fit(rows, lay, G.empty_map(lay.rows, lay.cols, 3), tau=-2e-5, period=0.033)
    terms = list(zip(true["i"], true["j"]))
    for k, t in enumerate(terms):
        if t != (1, 0):
            assert f["eu"][k] == pytest.approx(true["eu"][k], abs=0.02), ("eu", t)
        assert f["ev"][k] == pytest.approx(true["ev"][k], abs=0.02), ("ev", t)
    assert f["eu"][terms.index((1, 0))] == 0.0                         # a scale of the columns is not the map's to find
    assert f["psi"] == pytest.approx(psi, abs=3e-6)
    for st in rho:
        assert f["rho"][st] == pytest.approx(rho[st], abs=5e-6)
        assert f["lines"][st] == pytest.approx(list(line[st]), abs=1e-5)
    assert all(v["px"] < 0.01 and v["rows"] < 0.01 for v in f["left"].values())
    # applied, the correction leaves nothing: a second fit on the same comparisons with the map taken out finds no more
    assert max(G._update_range(true, f)) > 0.5


def test_body_motion_design():
    """The design of the body's motion: a course d(t) given at its nodes is found again from what views at equal
    angle (a turn apart) and opposite views (half a turn apart, at half the size and the other sign) show of it."""
    rng = np.random.default_rng(2)
    h, total = 0.5, 6.0
    truth = np.cumsum(rng.normal(0, 0.05, (12, 2)), axis=0)                       # d at 0.5, 1.0, .. 6.0 turns; d(0) = 0
    nodes = np.vstack([[0.0, 0.0], truth])
    d = lambda t: np.stack([np.interp(t, np.arange(13) * h, nodes[:, k]) for k in (0, 1)], 1)
    n = lambda th: np.stack([np.cos(th), np.sin(th)], 1)
    t1 = rng.uniform(0, 5, 400)
    th = 2 * np.pi * t1
    same = ((d(t1 + 1) - d(t1)) * n(th)).sum(1)
    t1o = rng.uniform(0, 5.5, 400)
    tho = 2 * np.pi * t1o
    opp = -0.5 * ((d(t1o + 0.5) - d(t1o)) * n(tho)).sum(1)
    Xs, K = G._nodes(t1, t1 + 1, th, 0, h, total)
    Xo, _ = G._nodes(t1o, t1o + 0.5, tho, 0, h, total)
    assert K == 12
    b = np.linalg.lstsq(np.concatenate([Xs, -0.5 * Xo]), np.concatenate([same, opp]) + rng.normal(0, 1e-4, 800), rcond=None)[0]
    assert np.allclose(b.reshape(12, 2), truth, atol=2e-3)
    # views at equal angle alone cannot tell what the body does within a turn: the half-turn nodes stay open
    assert np.linalg.matrix_rank(Xs) < 24 and np.linalg.matrix_rank(np.concatenate([Xs, Xo])) == 24
    assert G._block_size(2500) == 2500 and G._block_size(1622) == 1620        # 1620 = 2^2 3^4 5; 1622 = 2 x 811 is no quick transform size


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
    u_ref = G.overlap_centre_of(cal["layout"]) - p0["axis"]            # the overlap's middle, from the axis (the body's course is in the axis: within 1e-5 rows)
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


def test_second_scan_fit():
    rng = np.random.default_rng(4)
    n = 900
    t = rng.uniform(0, 6, n)
    st = np.where(t < 1, "rest1", np.where(t < 5, "helix", "rest2"))
    th = 2 * np.pi * t
    v = rng.choice([150.0, 600.0, 1050.0], n)
    nodes = rng.normal(0, 0.1, (7, 2))
    E = np.stack([np.interp(t, np.arange(7), nodes[:, j]) for j in (0, 1)], 1)
    off = np.where(st == "rest1", 0.3, np.where(st == "helix", 0.5, 0.6))
    du = off + 2e-5 * (v - 599.5) + E[:, 0] * np.cos(th) + E[:, 1] * np.sin(th) + rng.normal(0, 1e-3, n)
    y = -off + 0.03 * np.cos(th) - 0.1 * np.sin(th) + rng.normal(0, 1e-3, n)
    P = G.second_scan_fit([[a, b, c, d, e, f, 0.0] for a, b, c, d, e, f in zip(st, t, th, v, du, y)], 1200, 6.0)
    assert np.allclose(P["columns"]["once_per_turn_nodes"], nodes, atol=3e-3)
    assert P["columns"]["offset"]["helix"] == pytest.approx(0.5, abs=1e-3) and P["columns"]["per_row"]["rest1"] == pytest.approx(2e-5, abs=2e-6)
    assert P["rows"]["once_per_turn"] == pytest.approx([0.03, -0.1], abs=5e-4) and P["columns"]["left"]["with_once_per_turn"] < 2e-3

