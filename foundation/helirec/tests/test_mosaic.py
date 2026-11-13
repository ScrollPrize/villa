"""helirec.mosaic: the joint fit of a mosaic's comparisons recovers known places of the tiles, the rows' timing, the
roll, the axial scale and a detector map, and everything of the tiles' courses that a comparison can see, from
comparisons made up from them (no frames); and the plan keeps every block on the detector and on disk."""
import numpy as np
import pytest

from helirec import geometry as G
from helirec import mosaic as M

N, ROWS, COLS, PITCH = 10000, 2368, 4416, 2303.2


def _layout(places=((0.0, 0.0), (6944.2, 0.0), (3472.1, 6014.2))):
    n = len(places)
    f = np.arange(250000)
    lay = M.Layout(rows=ROWS, cols=COLS, per_turn=N, pixel_um=1.129, step_deg=0.036, period_s=0.01471, source_mm=178000.0, labels=[f"T{k}" for k in range(n)],
                   helix=[3] * n, on_disk=[(109700, 140300)] * n, angle=[360.0 + 0.036 * (f + 0.4)] * n, height=[-(f * PITCH / N)] * n, row_sign=-1.0,
                   axis=[4110.0] * n, place=[tuple(p) for p in places])
    lay.pairs = [(k, j) for k in range(n) for j in range(k + 1, n)]
    # the sways as single frames would give them: to a column or two, phase = direction + a zero
    for k, j in lay.pairs:
        du, dv = places[j][0] - places[k][0], places[j][1] - places[k][1]
        lay.sway[(k, j)] = (float(np.hypot(du, dv)) + 1.5, float((np.degrees(np.arctan2(dv, du)) + 270.06) % 360))
    return lay


def _smooth(rng, knots, size, n_turns=3.06):
    """a smooth random course on the knots: a drift and terms up to four times per turn"""
    x = (knots - knots.mean()) / N
    out = rng.normal(0, size) * x / n_turns
    for h in (0.4, 1, 2, 3, 4):
        out = out + rng.normal(0, size / 2) * np.cos(2 * np.pi * h * x) + rng.normal(0, size / 2) * np.sin(2 * np.pi * h * x)
    return out


def _truth(lay, rng):
    n = len(lay.labels)
    true = M.start_model(lay, 3)
    K = true["knots"]
    true["a"] = true["a"] + rng.normal(0, 0.4, n)
    true["d"][1:] += rng.normal(0, 4.0, (n - 1, 2))                 # places: some columns from where single frames put them
    common = _smooth(rng, K, 0.3)                                     # the stages' own course, the same in every tile
    true["C"] = np.array([common + _smooth(rng, K, 0.08) for _ in range(n)])
    true["h"][1:] = rng.normal(0, 4.0, n - 1)
    true["R"] = np.array([_smooth(rng, K, 0.15) for _ in range(n)])
    true["T"] = _smooth(rng, K, 0.05) * np.radians(0.036)            # a twentieth of a frame
    m = true["map"]
    tm = list(zip(m["i"], m["j"]))
    m["eu"] = [0.0 if (i == 0 and j % 2 == 1) or (i, j) == (1, 0) else float(v) for (i, j), v in zip(tm, rng.normal(0, 1.5, len(tm)))]
    m["ev"] = [float(v) for v in rng.normal(0, 1.0, len(tm))]
    true.update(rho=-1.5e-7, psi=8e-5, chi=-3e-5, scale=-4e-4)
    return true


def _measured(lay, true, reqs, rng, noise=0.0):
    """what a registration of each request returns if the acquisition is `true`"""
    m = true["map"]
    K = true["knots"]
    v0, step = m["v0"], np.radians(lay.step_deg)
    slope = -PITCH / N
    out = []
    for q in reqs:
        ka, kb, mirror = q["scan_a"], q["scan_b"], bool(q.get("mirror"))
        nq, nc = q["frames"], q["cols"]
        fa, fb0 = q["frame_a"], q["frame_a"] + q.get("frame_hint", 0)
        off = int(round(lay.height[kb][fb0 + nq // 2] - lay.height[ka][fa + nq // 2]))
        if q["row_a"] < 0:
            row_a = int(np.clip((ROWS - q["rows"] - off) // 2, max(0, -off), min(ROWS - q["rows"], ROWS - q["rows"] - off)))
        else:
            row_a = q["row_a"]
        va = row_a + (q["rows"] - 1) / 2
        vb = va + off
        fam, fbm = fa + (nq - 1) / 2, fb0 + (nq - 1) / 2
        # the same true angle in both blocks: the registration finds B that many frames further on
        dq = (M.course(K, true["T"], fam) - M.course(K, true["T"], fbm) - true["rho"] * (vb - va)) / step
        tha = np.radians(lay.angle_at(ka, fam)) + M.course(K, true["T"], fam) + true["rho"] * (va - v0)
        thb = np.radians(lay.angle_at(kb, fbm + dq)) + M.course(K, true["T"], fbm) + true["rho"] * (vb - v0)
        uA, uB = M.u_of(true, ka, tha, fam), M.u_of(true, kb, thb, fbm)
        hA, hB = M.h_of(true, ka, fam), M.h_of(true, kb, fbm)
        S0 = 0.0
        if q.get("sway_px"):
            S0 = q["sway_px"] * np.cos(np.radians(lay.angle_at(kb, fbm + dq) - q["sway_phase_deg"]) + np.radians(q.get("sway_row_deg", 0.0)) * (vb - v0))
        ca = q["col_a"] + (nc - 1) / 2
        cb = q["col_b"] + (nc - 1) / 2 + S0               # where the session cut B; the truth differs from it by du
        eA, eB = G.map_displacement(m, ca, va), G.map_displacement(m, cb, vb)
        if mirror:
            Y = uA + uB - (eA[0] + eB[0]) + true["psi"] * (va + vb - 2 * v0)
            du = q["col_a"] + q["col_b"] + nc - 1 + S0 - Y
        else:
            Y = uB - uA - (eB[0] - eA[0]) + true["psi"] * (vb - va)
            du = Y - (q["col_b"] - q["col_a"]) - S0
        stated_D = float(lay.height[kb][fb0 + nq // 2] - lay.height[ka][fa + nq // 2])
        y = hB - hA + true["scale"] * stated_D - (eB[1] - eA[1]) + (true["chi"] * (ca - cb) if mirror else 0.0)
        D = stated_D + y + slope * dq + true["rho"] / step * slope * (vb - va)
        out.append(dict(kind=q["label"], ka=ka, kb=kb, hint=q.get("frame_hint", 0), sway_px=q.get("sway_px", 0.0), sway_phase_deg=q.get("sway_phase_deg", 0.0),
                        sway_row_deg=q.get("sway_row_deg", 0.0), frame_a=fa, frame_b=fb0, frames=nq, row_a=row_a, rows=q["rows"], row_offset=off, col_a=q["col_a"], col_b=q["col_b"],
                        cols=nc, stated_dq=0.0, dq=float(dq + rng.normal(0, noise)), du=float(du + rng.normal(0, noise)), stated_D=stated_D, D=float(D + rng.normal(0, noise)),
                        slope_a=slope, slope_b=slope, correlation=0.9))
    return out


def test_plan_stays_on_the_detector_and_on_disk():
    lay = _layout()
    model = M.start_model(lay, 3, -3.5e-6)
    reqs = M.plan(lay, model, angles=24)
    assert {q["label"] for q in reqs} == set(M.KINDS) and len(reqs) > 1000
    first, end = lay.on_disk[0]
    for q in reqs:
        fb = q["frame_a"] + q.get("frame_hint", 0)
        assert first + 8 <= q["frame_a"] and q["frame_a"] + M.NQ + 8 <= end and first + 8 <= fb and fb + M.NQ + 8 <= end
        assert 0 <= q["col_a"] and q["col_a"] + q["cols"] <= COLS and q["rows"] in (32, 256)
        if "sway_px" in q:           # B is cut where its sway puts it: the cut and its margin stay on the detector
            S = q["sway_px"] * np.cos(np.radians(lay.angle[q["scan_b"]][fb + M.NQ // 2] - q["sway_phase_deg"]))
            assert 100 <= q["col_b"] + S and q["col_b"] + S + q["cols"] <= COLS - 100, q
            assert q["sway_row_deg"] == pytest.approx(np.degrees(model["rho"]))
        if q["label"] == "pairopp":
            assert q["mirror"] and abs(q["frame_hint"]) == N // 2
    # within a tile the positions follow each other closely enough for the courses' knots
    for kind in ("turn", "opposite"):
        f = sorted({q["frame_a"] for q in reqs if q["label"] == kind and q["scan_a"] == 1 and q.get("frame_hint", N) > 0})
        assert max(np.diff(f)) <= M.KNOT_FRAMES
    # the model's sways are the layout's, the reference tile at the origin
    assert M.sway_of(model, 0, 1)[0] == pytest.approx(lay.sway[(0, 1)][0], abs=3.0)
    assert np.all(model["d"][0] == 0)


def test_fit_recovers_what_the_comparisons_see():
    rng = np.random.default_rng(7)
    lay = _layout()
    true = _truth(lay, rng)
    model = M.start_model(lay, 3, -3.0e-6)                 # a row time 15 % from the true one
    reqs = M.plan(lay, model, angles=24)
    rows = _measured(lay, true, reqs, rng, noise=0.003)
    held = lambda d: (d["frame_a"] // 8) % 2 == 1
    grid = model["map"]                                    # the comparisons were made on this (empty) map's grid
    for _ in range(4):                                     # the angles enter through the places: a few rounds
        model, res, s = M.fit(lay, model, rows, held, grid=grid)
    assert abs(model["rho"] - true["rho"]) < 1e-9, (model["rho"], true["rho"])
    assert model["psi"] == pytest.approx(true["psi"], abs=3e-6) and model["chi"] == pytest.approx(true["chi"], abs=3e-6)
    # the map over the detector, not coefficient by coefficient
    c, v = np.meshgrid(np.linspace(0, COLS - 1, 23), np.linspace(0, ROWS - 1, 13))
    eu, ev = G.map_displacement(model["map"], c, v)
    tu, tv = G.map_displacement(true["map"], c, v)
    du = (eu - tu) - (eu - tu).mean()                       # a constant of the columns is the axis columns', not the map's
    assert np.abs(du).max() < 0.03 and np.abs(ev - tv).max() < 0.03, (np.abs(du).max(), np.abs(ev - tv).max())
    # every kind is predicted where it was not fitted: far below the courses (0.3 px), near the noise put in
    for kind, left in s["left"].items():
        assert left["held_px"] < 0.03 and left["held_rows"] < 0.03 and left["held_frames"] < 0.01, (kind, left)
    # what a comparison sees of the courses: a tile against another at equal frame, and a tile's two views half a turn apart
    f = np.arange(112000.0, 133000.0, 250.0)
    th = np.radians(360.0 + 0.036 * (f + 0.4))
    u = lambda mdl, t, th_, f_: M.u_of(mdl, t, th_ + M.course(mdl["knots"], mdl["T"], f_), f_)
    for t in (1, 2):
        got = u(model, t, th, f) - u(model, 0, th, f)
        want = u(true, t, th, f) - u(true, 0, th, f)
        assert np.abs(got - want).max() < 0.03, (t, np.abs(got - want).max())
        got = M.h_of(model, t, f) - M.h_of(model, 0, f)
        want = M.h_of(true, t, f) - M.h_of(true, 0, f)
        assert np.abs(got - want).max() < 0.03, (t, np.abs(got - want).max())
    for t in range(3):
        got = u(model, t, th, f) + u(model, t, th + np.pi, f + N / 2)
        want = u(true, t, th, f) + u(true, t, th + np.pi, f + N / 2)
        assert np.abs(got - want).max() < 0.05, (t, np.abs(got - want).max())
    # the places: the swing of one tile against another once per turn
    for t in (1, 2):
        assert np.hypot(*(model["d"][t] - true["d"][t])) < 0.1, (model["d"][t], true["d"][t])


def test_frame_pose_and_restore_keep_the_model():
    lay = _layout()
    true = _truth(lay, np.random.default_rng(3))
    step = np.radians(0.036)
    cal = dict(layout=dict(knot_frames=[float(x) for x in true["knots"]]), map=true["map"], row_time=dict(rad_per_row=true["rho"]), tiles=M.summary_of(lay, true),
               angle_course_rad=[float(x) for x in true["T"]])
    f = np.array([110000.0, 125250.0])
    angle = lay.angle[1][f.astype(int)]
    th, u, h = M.frame_pose(cal, "T1", angle, f, row=100.0)
    want = np.radians(angle) + M.course(true["knots"], true["T"], f) + true["rho"] * (100.0 - true["map"]["v0"])
    assert np.allclose(np.radians(th), want) and np.allclose(u, M.u_of(true, 1, want, f)) and np.allclose(h, M.h_of(true, 1, f))
    assert cal["tiles"]["T1"]["place_px"] == [true["d"][1, 0], true["d"][1, 1]] and cal["tiles"]["T0"]["place_px"] == [0.0, 0.0]


def test_place_tiles_from_overlaps():
    """mosaic.place_tiles: one shift per tile from the overlaps' median displacements. Made-up tiles on a ring with
    one in the middle, displaced by known shifts: the shifts come back, a block gone astray does not move them, the
    calibration's places and heights are moved the way overlaps() measures them (a place by (x, -y)), and what one
    shift per tile cannot hold is reported as left."""
    rng = np.random.default_rng(5)
    labels = ["c", "r0", "r1", "r2", "r3"]
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (1, 4)]
    true = np.array([[0, 0, 0], [1.5, -0.4, 2.0], [-0.7, 0.9, -1.0], [0.3, 0.2, 5.5], [-1.1, -1.3, 0.25]])
    cal = dict(layout=dict(tiles=labels), tiles={t: dict(place_px=[100.0 * i, -50.0 * i], height_rows=float(i)) for i, t in enumerate(labels)})

    def table(extra=None):
        rows = []
        for k, j in pairs:
            for h in (1000, 2000):
                d = true[j] - true[k] + (extra or {}).get((k, j, h), 0.0)
                for b in range(12):
                    x = d + rng.normal(0, 0.02, 3) + (40.0 if b == 0 else 0.0)        # one block of each overlap far off
                    rows.append(dict(k=k, j=j, height=h, x=0.0, y=0.0, xj=0.0, yj=0.0, dx=float(x[0]), dy=float(x[1]), dz=float(x[2]), correlation=0.5))
        return rows

    rows = table()
    means = M.overlap_means(rows)
    assert len(means) == 16 and means[(0, 1, 1000)][3] == 12 and abs(means[(0, 1, 1000)][0] - 1.5) < 0.03
    new, summary = M.place_tiles(cal, rows, log=lambda *a: None)
    got = np.array([summary["shift"][t] for t in labels])
    assert np.allclose(got, true, atol=0.03)
    assert summary["overlaps"] == 16 and summary["unknowns"] == 12 and summary["left"][0] < 0.03 and summary["left"][1] < 0.03 and summary["before"][0] > 1.0
    for i, t in enumerate(labels):
        assert np.allclose(new["tiles"][t]["place_px"], [100.0 * i + got[i, 0], -50.0 * i - got[i, 1]])
        assert np.isclose(new["tiles"][t]["height_rows"], i + M.PLACE_Z_SIGN * got[i, 2])
    assert cal["tiles"]["r0"]["place_px"] == [100.0, -50.0]                              # the calibration given is not changed
    # a displacement of one overlap that no placement of tiles gives stays, and is said
    new2, s2 = M.place_tiles(cal, table({(1, 2, 1000): np.array([0.9, 0.0, 0.0]), (1, 2, 2000): np.array([0.9, 0.0, 0.0])}), log=lambda *a: None)
    assert s2["worst"] > 0.3 and s2["left"][0] > 0.2


def test_axis_course_from_mirror_columns():
    """The axis column along the scan from opposite views: a comparison at p gives [C(p) + C(p + N/2)] / 2; the
    course made of them holds a drift and what repeats every half turn, and leaves out what changes sign after half
    a turn; the tiles' common course is found from tiles that each carry their own constant and noise."""
    N = 1000
    th = lambda f: 2 * np.pi * np.asarray(f, float) / N
    even = lambda f: 0.4 * np.cos(2 * th(f) + 0.3) + 0.1 * np.sin(4 * th(f))
    odd = lambda f: 0.3 * np.cos(3 * th(f)) + 0.2 * np.sin(th(f))
    drift = lambda f: 2e-4 * (np.asarray(f, float) - 5000)
    C = lambda f: even(f) + odd(f) + drift(f)
    p = np.arange(4000.0, 6500.0, 20.0)                       # the comparisons' first blocks; the second half a turn on
    m = (C(p) + C(p + N / 2)) / 2
    at = np.arange(3900.0, 7200.0, 25.0)
    got = M.course_of_mirror(p, m, at, N)
    assert np.isnan(got[at < p[0]]).all() and np.isnan(got[at > p[-1] + N / 2]).all()
    full = (at >= p[0] + N / 2) & (at <= p[-1])              # both comparisons of the frame measured
    assert np.abs(got[full] - (even(at) + drift(at))[full]).max() < 0.01          # the interpolation between positions
    one = (at >= p[0]) & (at < p[0] + N / 2)                  # the first half turn: one comparison, the drift a quarter turn late
    assert np.abs(got[one] - (even(at) + drift(at + N / 4))[one]).max() < 0.01

    labels = ["A", "B", "C", "D"]
    K = np.arange(3500.0, 7600.0, 50.0)
    cal = dict(layout=dict(tiles=labels, knot_frames=[float(x) for x in K], frames_per_turn=N),
               tiles={t: dict(axis_column=4100.0 + i, column_course=[0.0] * len(K)) for i, t in enumerate(labels)})
    rng = np.random.default_rng(3)
    table = []
    for k in range(len(labels)):
        for q in p[::2]:
            for _ in range(4):
                table.append(dict(tile=k, frame_a=int(q), frame=float(q) + 15.5, row=1000.0, col=4000.0, correlation=0.9,
                                  mirror=4100.0 + k + float((C(q + 15.5) + C(q + 15.5 + N / 2)) / 2) + 0.02 * rng.standard_normal()))
    new, summary = M.axis_course(cal, table)
    inside = (K >= p[0] + 15.5 + N / 2) & (K <= p[::2][-1] + 15.5)
    want = even(K) + drift(K)
    for t in labels:
        c = np.array(new["tiles"][t]["column_course"])
        assert new["tiles"][t]["axis_column"] == cal["tiles"][t]["axis_column"]
        assert np.abs((c - want)[inside] - (c - want)[inside].mean()).max() < 0.03
        assert c[0] == c[np.argmax(K >= p[0] + 15.5)] and c[-1] == c[np.nonzero(K <= p[::2][-1] + 15.5 + N / 2)[0][-1]]   # held beyond
        rms_own, rms_less, corr = summary["against_others"][t]
        assert rms_own > 0.2 and rms_less < 0.03 and corr > 0.99
    assert cal["tiles"]["A"]["column_course"] == [0.0] * len(K)          # the calibration given is not changed
    own, _ = M.axis_course(cal, [r for r in table if r["tile"] != 3], common=False)
    assert np.abs(np.array(own["tiles"]["D"]["column_course"]) - np.array(new["tiles"]["D"]["column_course"])).max() < 0.03   # a tile without comparisons: the others' common course
    assert np.abs(np.array(own["tiles"]["A"]["column_course"]) - np.array(new["tiles"]["A"]["column_course"])).max() < 0.05
