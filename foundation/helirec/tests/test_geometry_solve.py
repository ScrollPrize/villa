"""The joint geometry model (helirec.geometry.solve_geometry, rays) against a projection written apart from it:
points in 3D carried by explicit rotation matrices and projected along parallel rays. The projection here fixes
the model's signs and handedness; nothing in it uses the model's formulas."""
import numpy as np
import pytest

from helirec import geometry as G

ROWS, COLS, PER_TURN = 400, 600, 360


def _stage(th):
    """The rotation stage at the angle th: the sample-fixed direction (cos th, sin th) lies along the detector's
    columns, (-sin th, cos th) along the beam."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _turned(w):
    """The rotation by the vector w [rad] (its direction the axis, its length the angle), as a matrix."""
    w = np.asarray(w, float)
    a = np.linalg.norm(w)
    if a == 0:
        return np.eye(3)
    k = w / a
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def _about_beam(a):
    """The detector's (column, row) of a position (x, z) in the frame of the rotation axis: the axis is rolled by a
    against the detector's columns."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s], [-s, c]])


class Scan:
    """A synthetic scan: what its record states per frame, and where it really holds the sample."""

    def __init__(self, stretches, travel=300.0, index=0, **truth):
        # stretches: (name, frames); the helix climbs `travel` rows per turn, a rest turn stands where the helix
        # starts or ends. index: the scan's place among the acquisition's scans. map: the detector's (as
        # G.empty_map gives it, with its coefficients): the pixel (c, v) looks along (c + Eu, v + Ev). pitch: of the
        # rotation axis towards the beam [rad]. moved: the sample sits elsewhere on the stage than in the anchor.
        # rotation: the sample is turned by this vector [rad] about its origin at the height `about`.
        self.t = dict(axis=301.3, axis_slope=0.0, axial_scale=0.0, d=(0.0, 0.0), height=0.0, roll=0.0, row_time=0.0, stretch_height={},
                      height_once=(0.0, 0.0), axis_twice=(0.0, 0.0), map=None, pitch=0.0, moved=False, rotation=(0.0, 0.0, 0.0), about=0.0)
        self.t.update(truth)
        self.index = index
        self.stretches, H, angle0 = [], 0.0, 12.0
        for name, n in stretches:
            f = np.arange(n)
            angle = angle0 + f * (360.0 / PER_TURN)
            rows = H + (f * (travel / PER_TURN) if name == "helix" else np.zeros(n))
            self.stretches.append(dict(name=name, helix=name == "helix", angle_deg=angle, rows=rows))
            H, angle0 = rows[-1], angle[-1] + 360.0 / PER_TURN
        helix = [s for s in self.stretches if s["helix"]]
        self.mid_height = float(helix[0]["rows"][len(helix[0]["rows"]) // 2]) if helix else float(self.stretches[0]["rows"][0])

    def layout(self):
        # (the motors' places are the true ones here: on_stage is d)
        return dict(name="scan", mid_height=self.mid_height, stretches=self.stretches, moved_on_stage=self.t["moved"], on_stage=list(self.t["d"]), axis=self.t["axis"])

    def seen(self, stretch, p, t):
        """Where the point p of the sample is recorded at the time of frame t (fractions allowed) of a stretch:
        (column, row), by explicit rigid motions."""
        st, tr, mid = self.stretches[stretch], self.t, (ROWS - 1) / 2.0
        th = np.radians(np.interp(t, np.arange(len(st["angle_deg"])), st["angle_deg"]))
        H = np.interp(t, np.arange(len(st["rows"])), st["rows"])
        on_stage = self.held(p) - [0.0, 0.0, tr["height"]]                         # the sample's place on the stage (its height below, with the record's)
        x, depth, z = _stage(th) @ on_stage                                        # the stage turned; parallel rays: the depth is dropped,
        z += tr["pitch"] * depth                                                   # but for an axis leaning towards the beam: a row higher per 1 / pitch of depth
        # what the stage does within a turn: it lifts the sample once per turn and moves its axis across the detector twice
        z += tr["height_once"][0] * np.cos(th) + tr["height_once"][1] * np.sin(th)
        x += tr["axis_twice"][0] * np.cos(2 * th) + tr["axis_twice"][1] * np.sin(2 * th)
        x += tr["axis_slope"] * (H - self.mid_height)                              # the axis stage does not travel along the axis as the detector sees it
        z += tr["height"] + tr["stretch_height"].get(st["name"], 0.0) + (1.0 + tr["axial_scale"]) * H   # and it travels further than the record says
        c, v = _about_beam(tr["roll"]) @ [x, z - mid]
        return tr["axis"] + c, mid + v

    def held(self, p):
        """Where this scan holds the point p of the sample: turned about the sample's origin at the height
        `about`, then put at the scan's place and height. What the scan alone reconstructs it at."""
        tr = self.t
        about = np.array([0.0, 0.0, tr["about"]])
        return _turned(tr["rotation"]) @ (np.asarray(p, float) - about) + about + [tr["d"][0], tr["d"][1], tr["height"]]

    def stored(self, X, Y):
        """The stored pixel that looks along the ideal position (X, Y): the map taken off, by going round until it
        stands (the model has the other direction)."""
        m = self.t["map"]
        if m is None:
            return X, Y
        c, v = X, Y
        for _ in range(30):
            phi = ((c - m["c0"]) / m["cs"]) ** np.asarray(m["i"]) * ((v - m["v0"]) / m["vs"]) ** np.asarray(m["j"])
            c, v = X - float(np.dot(m["eu"], phi)), Y - float(np.dot(m["ev"], phi))
        return c, v

    def sighting(self, stretch, p, t):
        """The sighting (frame, row, column) of p whose row was exposed at the time of frame t: the frame's own
        time is t - row_time (row - middle row)."""
        c, v = self.stored(*self.seen(stretch, p, t))
        return [self.index, stretch, t - self.t["row_time"] * (v - (ROWS - 1) / 2.0), v, c]

    def on_detector(self, g, margin=20):
        return margin < g[3] < ROWS - margin and margin < g[4] < COLS - margin

    def parameters(self):
        t = self.t
        return dict(roll=t["roll"], stage=dict(height_once=list(t["height_once"]), axis_twice=list(t["axis_twice"])), scans=[dict(axis=t["axis"], axis_slope=t["axis_slope"], axial_scale=t["axial_scale"], d=list(t["d"]), height=t["height"],
                                                 stretch_height=dict(t["stretch_height"]))])


def _acquisition(*scans):
    return G.acquisition(dict(rows=ROWS, cols=COLS, row_time=scans[0].t["row_time"], scans=[s.layout() for s in scans]))


def _points(rng, scan, stretch, n, radius=250.0):
    """Points of the sample that the stretch records somewhere: (point, time)."""
    st = scan.stretches[stretch]
    out = []
    while len(out) < n:
        t = rng.uniform(2, len(st["rows"]) - 3)
        r, a = radius * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        # a height that this frame holds: the row is about z + H
        z = rng.uniform(30, ROWS - 30) - np.interp(t, np.arange(len(st["rows"])), st["rows"])
        out.append((np.array([r * np.cos(a), r * np.sin(a), z]), t))
    return out


def _same_point(scan, kind, sa, sb, p, ta, tb, rng=None, noise=0.0, other=None):
    """The point p as `scan` holds it at the time ta of its stretch sa and as `other` (the same scan if none is
    given) holds it at the time tb of stretch sb."""
    a, b = scan.sighting(sa, p, ta), (other or scan).sighting(sb, p, tb)
    if not (scan.on_detector(a) and scan.on_detector(b)):
        return None
    if rng is not None:
        b[3] += rng.normal(0, noise)
        b[4] += rng.normal(0, noise)
    return dict(kind=kind, a=a, b=b, opposite=kind.endswith("opposite"))


TRUTH = dict(axis=310.4, axis_slope=4e-4, axial_scale=3.3e-3, d=(7.0, -4.0), height=2.5, roll=3e-4, row_time=2e-3, height_once=(0.21, -0.13), axis_twice=(0.11, 0.17))
STAGE = ["stage.height_once_cos", "stage.height_once_sin", "stage.axis_twice_cos", "stage.axis_twice_sin"]


def _stage_found(got, abs_=2e-3):
    return got["stage"]["height_once"] == pytest.approx(list(TRUTH["height_once"]), abs=abs_) and got["stage"]["axis_twice"] == pytest.approx(list(TRUTH["axis_twice"]), abs=abs_)


def test_the_projection_itself():
    """The projection's own cases by hand: a point on the axis does not move across the detector, and a quarter
    turn carries the direction along the beam onto the columns."""
    scan = Scan([("helix", 3 * PER_TURN)], axis=300.0)
    mid = (ROWS - 1) / 2.0
    for t in (0.0, 77.0, 400.5):
        c, v = scan.seen(0, [0, 0, mid - 100.0], t)
        assert c == pytest.approx(300.0) and v == pytest.approx(mid - 100.0 + t * 300.0 / PER_TURN)
    quarter = (90.0 - 12.0) * PER_TURN / 360.0
    assert scan.seen(0, [0, 50.0, 0.0], quarter)[0] == pytest.approx(350.0)        # at 90 degrees the sample's y lies along the columns
    assert scan.seen(0, [50.0, 0, 0.0], quarter)[0] == pytest.approx(300.0)
    assert scan.seen(0, [50.0, 0, 0.0], quarter + 3 * PER_TURN / 4)[0] == pytest.approx(350.0)   # a whole turn: the sample's x again


def test_rays_are_the_points_lines():
    """Every sighting of a point, under the true parameters, is the line through that point at the angle it was
    seen at: s = p . (cos th, sin th), z = p_z. Signs and handedness of every term at once."""
    rng = np.random.default_rng(1)
    scan = Scan([("rest1", PER_TURN), ("helix", 4 * PER_TURN), ("rest2", PER_TURN)], stretch_height=dict(rest1=-0.4, rest2=1.1), **TRUTH)
    sightings, expect = [], []
    for stretch in range(3):
        for p, t in _points(rng, scan, stretch, 200):
            g = scan.sighting(stretch, p, t)
            if not scan.on_detector(g, 0):
                continue
            th = np.radians(np.interp(t, np.arange(len(scan.stretches[stretch]["angle_deg"])), scan.stretches[stretch]["angle_deg"]))
            sightings.append(g)
            expect.append([np.degrees(th), p[0] * np.cos(th) + p[1] * np.sin(th), p[2]])
    got = G.rays(_acquisition(scan), scan.parameters(), sightings)
    expect = np.array(expect)
    assert len(expect) > 300
    assert np.allclose(got[:, 0], expect[:, 0], atol=1e-9)
    # the model is linear in the small angles; the projection turns by them exactly (roll^2 x 300 px = 3e-5)
    assert np.abs(got[:, 1:] - expect[:, 1:]).max() < 2e-4
    # the beam's divergence is in the angle alone: fan frames of angle per column from the detector's middle
    acq = dict(rows=ROWS, cols=COLS, row_time=scan.t["row_time"], fan=2e-4, scans=[scan.layout()])
    fanned = G.rays(acq, scan.parameters(), sightings)
    cols = np.array(sightings)[:, 4]
    assert np.allclose(fanned[:, 0] - got[:, 0], 2e-4 * (360.0 / PER_TURN) * (cols - (COLS - 1) / 2), atol=1e-12) and np.array_equal(fanned[:, 1:], got[:, 1:])


def _helix_comparisons(rng, scan, stretch, n, noise):
    out = []
    for p, t in _points(rng, scan, stretch, 4 * n):
        last = len(scan.stretches[stretch]["rows"]) - 3
        if t + PER_TURN / 2 < last:
            out.append(_same_point(scan, "opposite", stretch, stretch, p, t, t + PER_TURN / 2, rng, noise))
        if t + PER_TURN < last:
            out.append(_same_point(scan, "turn", stretch, stretch, p, t, t + PER_TURN, rng, noise))
    return [c for c in out if c is not None][: 2 * n]


def test_a_helix_alone_gives_its_own_parameters():
    """A helix alone (what a mosaic's tile has on disk): opposite views and views a turn apart give the axis column,
    its slope, the axial scale and the roll; the scan's place cancels and is not asked for."""
    rng = np.random.default_rng(2)
    scan = Scan([("helix", 5 * PER_TURN)], **TRUTH)
    acq = _acquisition(scan)
    comparisons = _helix_comparisons(rng, scan, 0, 1500, 0.004)
    kinds = {c["kind"] for c in comparisons}
    assert kinds == {"opposite", "turn"} and len(comparisons) > 1000
    # under the truth the comparisons leave the noise
    exact = G.solve_geometry(acq, comparisons, start=scan.parameters(), huber=0)
    assert all(l["rms"][1] < 0.01 and l["rms"][2] < 0.01 for l in exact["left"])
    start = dict(roll=0.0, scans=[dict(axis=TRUTH["axis"] + 2.7)])
    got = G.solve_geometry(acq, comparisons, start=start)
    s = got["scans"][0]
    assert got["free"] == ["roll"] + STAGE + ["scan0.axis", "scan0.axis_slope", "scan0.axial_scale"] and got["unused"] == []
    assert got["solver"]["rank"] == 8 and got["solver"]["usable"] and _stage_found(got)
    # the stage's motion held at nothing, it is what the opposite views are left with: twice the height's once per
    # turn in their rows, twice the axis column's twice per turn in their columns; views a turn apart see none of it
    without = G.solve_geometry(acq, comparisons, start=start, held=STAGE)
    opposite = np.array([c["kind"] == "opposite" for c in comparisons])
    th = np.radians(G.rays(acq, without, [c["a"] for c in comparisons])[:, 0])
    rows, cols = without["residual"][:, 1], without["residual"][:, 2]
    X1 = np.stack([np.cos(th), np.sin(th), np.ones_like(th)], 1)[opposite]
    X2 = np.stack([np.cos(2 * th), np.sin(2 * th), np.ones_like(th)], 1)[opposite]
    assert np.linalg.lstsq(X1, rows[opposite], rcond=None)[0][:2] == pytest.approx([-2 * 0.21, 2 * 0.13], abs=0.02)
    assert np.linalg.lstsq(X2, cols[opposite], rcond=None)[0][:2] == pytest.approx([2 * 0.11, 2 * 0.17], abs=0.02)     # (the columns' residual is a sum)
    assert np.abs(rows[~opposite]).max() < 0.05 and np.abs(cols[~opposite]).max() < 0.05
    assert s["axis"] == pytest.approx(TRUTH["axis"], abs=2e-3)
    assert s["axis_slope"] == pytest.approx(TRUTH["axis_slope"], abs=5e-6)        # 0.006 px over the helix' 1500 rows
    assert s["axial_scale"] == pytest.approx(TRUTH["axial_scale"], abs=2e-5)
    assert got["roll"] == pytest.approx(TRUTH["roll"], abs=1e-5)
    assert s["d"] == [0.0, 0.0] and s["height"] == 0.0                              # held where they started
    for l in got["left"]:
        assert l["rms"][1] < 0.006 and l["rms"][2] < 0.006 and abs(l["median"][0]) < 1e-6, l
    assert got["residual"].shape == (len(comparisons), 3)
    assert np.allclose(got["residual"], G.residuals(acq, got, comparisons))
    # the frames' displacement is reported, not fitted: half a frame given to B's frames is half a frame left
    shifted = [dict(c, b=[c["b"][0], c["b"][1], c["b"][2] + 0.5, c["b"][3], c["b"][4]]) for c in comparisons]
    assert np.allclose(G.residuals(acq, scan.parameters(), shifted)[:, 0], 0.5, atol=1e-4)      # (the rows' noise times the rows' timing)


def test_rest_turns_and_the_refusal_of_dependent_unknowns():
    rng = np.random.default_rng(3)
    heights = dict(rest1=-0.4, rest2=1.1)
    scan = Scan([("rest1", PER_TURN), ("helix", 4 * PER_TURN), ("rest2", PER_TURN)], stretch_height=heights, **TRUTH)
    acq = _acquisition(scan)
    comparisons = _helix_comparisons(rng, scan, 1, 600, 0.004)
    for rest, lo in ((0, 0), (2, 3 * PER_TURN)):
        for p, t in _points(rng, scan, rest, 600):
            if t + PER_TURN / 2 < PER_TURN - 3:
                comparisons.append(_same_point(scan, "opposite", rest, rest, p, t, t + PER_TURN / 2, rng, 0.004))
            # the helix at the same angle: the turn of it nearest the rest turn's height
            comparisons.append(_same_point(scan, "resthelix", rest, 1, p, t, t + lo, rng, 0.004))
    comparisons = [c for c in comparisons if c is not None]
    assert sum(c["kind"] == "resthelix" for c in comparisons) > 200
    start = dict(roll=0.0, scans=[dict(axis=TRUTH["axis"] - 1.9)])
    got = G.solve_geometry(acq, comparisons, start=start)
    s = got["scans"][0]
    assert "scan0.rest1.height" in got["free"] and "scan0.rest2.height" in got["free"] and "scan0.helix.height" not in got["free"]
    assert s["axis"] == pytest.approx(TRUTH["axis"], abs=2e-3) and s["axial_scale"] == pytest.approx(TRUTH["axial_scale"], abs=2e-5)
    assert s["axis_slope"] == pytest.approx(TRUTH["axis_slope"], abs=5e-6) and got["roll"] == pytest.approx(TRUTH["roll"], abs=1e-5)
    assert s["stretch_height"]["rest1"] == pytest.approx(heights["rest1"], abs=3e-3)
    assert s["stretch_height"]["rest2"] == pytest.approx(heights["rest2"], abs=3e-3) and _stage_found(got)
    # every stretch's height free: only their differences are seen, and the solve says so
    with pytest.raises(ValueError, match="RANK_DEFICIENT.*height"):
        G.solve_geometry(acq, comparisons, start=start, freed=["scan0.helix.height"])
    with pytest.raises(ValueError, match="no unknown is named"):
        G.solve_geometry(acq, comparisons, start=start, held=["scan0.pitch"])
    # a rest turn alone stands at one height: its opposite views cannot tell the axis column from its slope (refused),
    # and they hold nothing of the travel or of the stretches' heights (said, and left where they were)
    rest_only = [c for c in comparisons if c["kind"] == "opposite" and c["a"][1] == 0]
    with pytest.raises(ValueError, match="RANK_DEFICIENT.*scan0.axis"):
        G.solve_geometry(acq, rest_only, start=start)
    alone = G.solve_geometry(acq, rest_only, start=start, held=["scan0.axis_slope"])
    assert alone["unused"] == ["scan0.axial_scale", "scan0.rest1.height", "scan0.rest2.height"]
    assert alone["free"] == ["roll"] + STAGE + ["scan0.axis"] and _stage_found(alone, 5e-3)
    # the first rest turn stands at the helix' start, mid_height rows from where the axis column is counted
    there = TRUTH["axis"] + TRUTH["axis_slope"] * (0.0 - scan.mid_height)
    assert alone["scans"][0]["axis"] == pytest.approx(there, abs=2e-3)


def _pair(rng, with_map=True):
    """Two scans of one sample, the second with the rotation stage moved 230 columns across the beam: its axis lies
    60 columns from the detector's edge, and it holds the first's content in the columns both record."""
    m = None
    if with_map:
        m = dict(G.empty_map(ROWS, COLS, 3))
        m["eu"] = rng.uniform(-0.4, 0.4, len(m["i"])) * [0.0 if (i, j) == (1, 0) else 1.0 for i, j in zip(m["i"], m["j"])]
        m["ev"] = rng.uniform(-0.4, 0.4, len(m["i"]))
    stretches = [("rest1", PER_TURN), ("helix", 4 * PER_TURN), ("rest2", PER_TURN)]
    shared = dict(roll=TRUTH["roll"], row_time=TRUTH["row_time"], height_once=TRUTH["height_once"], axis_twice=TRUTH["axis_twice"], d=TRUTH["d"], map=m)
    first = Scan(stretches, index=0, axis=TRUTH["axis"], axis_slope=TRUTH["axis_slope"], axial_scale=TRUTH["axial_scale"], height=TRUTH["height"],
                 stretch_height=dict(rest1=-0.4, rest2=1.1), **shared)
    second = Scan(stretches, index=1, axis=TRUTH["axis"] + 230.0, axis_slope=-2e-4, axial_scale=3.1e-3, height=TRUTH["height"] - 1.3,
                  stretch_height=dict(rest1=0.3, rest2=-0.7), **shared)
    comparisons = []
    for scan in (first, second):
        for c in _helix_comparisons(rng, scan, 1, 700, 0.004):
            if scan is first or c["kind"] == "turn":      # the second scan's axis is at the edge: no opposite views of it
                comparisons.append(c)
        for rest, lo in ((0, 0), (2, 3 * PER_TURN)):
            for p, t in _points(rng, scan, rest, 500):
                if scan is first and t + PER_TURN / 2 < PER_TURN - 3:
                    comparisons.append(_same_point(scan, "opposite", rest, rest, p, t, t + PER_TURN / 2, rng, 0.004))
                comparisons.append(_same_point(scan, "resthelix", rest, 1, p, t, t + lo, rng, 0.004))
    own = [c for c in comparisons if c is not None]
    pairs = []
    for stretch in range(3):                                # the two scans at equal frames
        for p, t in _points(rng, first, stretch, 1500):
            pairs.append(_same_point(first, "pair", stretch, stretch, p, t, t, rng, 0.004, other=second))
    return first, second, own, [c for c in pairs if c is not None]


def test_a_pair_moved_sideways_gives_the_second_scans_place_and_the_map():
    """What a second scan with its axis outside its field is known by: its comparisons with the first. With the
    first's own views they give the detector's map, the second scan's axis column, its slope and its height."""
    rng = np.random.default_rng(5)
    first, second, own, pairs = _pair(rng)
    acq = _acquisition(first, second)
    assert len(pairs) > 800 and sum(c["kind"] == "turn" and c["a"][0] == 1 for c in own) > 100
    truth = first.t["map"]
    start = dict(roll=0.0, scans=[dict(axis=TRUTH["axis"] + 1.6), dict(axis=TRUTH["axis"] + 228.0)], map=G.empty_map(ROWS, COLS, 3))
    got = G.solve_geometry(acq, own + pairs, start=start, freed=["map"])
    assert got["unused"] == [] and "map.eu.i1j0" not in got["free"] and "map.ev.i1j0" in got["free"]
    assert "scan1.height" in got["free"] and "scan1.d0" in got["free"] and "scan0.height" not in got["free"] and "scan0.d0" not in got["free"]
    assert got["solver"]["rank"] == len(got["free"]) == 1 + 4 + 15 + 5 + 8
    a, b = got["scans"]
    assert a["axis"] == pytest.approx(TRUTH["axis"], abs=3e-3) and b["axis"] == pytest.approx(TRUTH["axis"] + 230.0, abs=5e-3)
    assert b["axis_slope"] == pytest.approx(-2e-4, abs=1e-5) and b["axial_scale"] == pytest.approx(3.1e-3, abs=3e-5)
    assert b["height"] == pytest.approx(-1.3, abs=5e-3) and a["height"] == 0.0     # against the anchor's
    assert b["d"] == pytest.approx([0.0, 0.0], abs=5e-3) and a["d"] == [0.0, 0.0]  # and its place: the same as the anchor's here
    assert b["stretch_height"]["rest1"] == pytest.approx(0.3, abs=5e-3) and b["stretch_height"]["rest2"] == pytest.approx(-0.7, abs=5e-3)
    assert got["roll"] == pytest.approx(TRUTH["roll"], abs=2e-5) and _stage_found(got, 3e-3)
    # the map, as the displacement it makes over the detector
    c, v = np.meshgrid(np.linspace(20, COLS - 20, 15), np.linspace(20, ROWS - 20, 11))
    eu, ev = G.map_displacement(got["map"], c, v)
    tu, tv = G.map_displacement(truth, c, v)
    assert np.abs(eu - tu).max() < 0.01 and np.abs(ev - tv).max() < 0.01
    for l in got["left"]:
        assert l["rms"][1] < 0.006 and l["rms"][2] < 0.006, l
    assert {(l["kind"], l["scan_a"], l["scan_b"]) for l in got["left"]} >= {("pair", 0, 1), ("pair", -1, -1), ("turn", 1, 1)}
    # the rays are the points' lines through the map too
    p, t = np.array([40.0, -90.0, 70.0]), 500.25
    g = second.sighting(1, p, t)
    th = np.radians(np.interp(t, np.arange(4 * PER_TURN), second.stretches[1]["angle_deg"]))
    truth_p = dict(roll=TRUTH["roll"], stage=dict(height_once=list(TRUTH["height_once"]), axis_twice=list(TRUTH["axis_twice"])), map=truth,
                   scans=[first.parameters()["scans"][0], second.parameters()["scans"][0]])
    ray = G.rays(acq, truth_p, [g])[0]
    assert ray[1] == pytest.approx(p[0] * np.cos(th) + p[1] * np.sin(th), abs=3e-4) and ray[2] == pytest.approx(p[2], abs=3e-4)
    # held, the map is what the comparisons are left with
    without = G.solve_geometry(acq, own + pairs, start=start)
    assert not any(n.startswith("map") for n in without["free"]) and max(l["rms"][2] for l in without["left"]) > 0.03
    # a sample that sits elsewhere against the axis in the second scan (it slipped in its mount) is found there
    moved = [dict(c, b=list(c["b"])) for c in pairs]
    for c in moved:
        th = np.radians(G.rays(acq, truth_p, [c["b"]])[0][0])
        c["b"][4] += 0.3 * np.cos(th) - 0.2 * np.sin(th)
    slipped = G.solve_geometry(acq, own + moved, start=start, freed=["map"])
    assert slipped["scans"][1]["d"] == pytest.approx([0.3, -0.2], abs=5e-3)
    held = G.solve_geometry(acq, own + moved, start=start, freed=["map"], held=["scan1.d0", "scan1.d1"])
    assert held["scans"][1]["d"] == [0.0, 0.0] and max(l["rms"][2] for l in held["left"] if l["kind"] == "pair") > 0.2
    # the anchor's height beside the second scan's: only their difference is seen
    with pytest.raises(ValueError, match="RANK_DEFICIENT.*height"):
        G.solve_geometry(acq, own + pairs, start=start, freed=["scan0.height"])
    with pytest.raises(ValueError, match="cannot be told from the pixel size"):
        G.solve_geometry(acq, own + pairs, start=start, freed=["map.eu.i1j0"])
    # without its comparisons with the first, nothing gives the second scan's place: said, and left where it was
    alone = G.solve_geometry(acq, own, start=start)
    assert alone["unused"][:3] == ["scan1.d0", "scan1.d1", "scan1.height"] and "scan1.axis" in alone["unused"] and alone["scans"][1]["axis"] == TRUTH["axis"] + 228.0


def test_the_reconstructions_poses_are_the_models():
    """A joint calibration as the reconstruction reads it (helirec/geometry.hpp, Calibration::frame_poses): a pixel
    put through the poses' formula is the ray the model gives it, the slices counted from the first scan's helix."""
    rng = np.random.default_rng(6)
    first, second, _, _ = _pair(rng)
    scans = (first, second)
    acq = _acquisition(first, second)
    truth = dict(roll=TRUTH["roll"], stage=dict(height_once=list(TRUTH["height_once"]), axis_twice=list(TRUTH["axis_twice"])), map=first.t["map"],
                 scans=[dict(sc.parameters()["scans"][0], mid_height=sc.mid_height) for sc in scans])
    offsets = np.cumsum([0] + [len(st["rows"]) for st in first.stretches])
    layout = dict(rows=ROWS, cols=COLS, row_sign=-1.0, scans=[dict(name=str(k), coarse_axis=sc.t["axis"], stretches=[
        dict(name=st["name"], helix=st["helix"], frames=len(st["rows"]), offset=int(o)) for st, o in zip(sc.stretches, offsets)]) for k, sc in enumerate(scans)])
    cal = dict(step="geometry", model="joint", layout=layout, row_time=dict(seconds_per_row=TRUTH["row_time"] * 0.03, frame_period_s=0.03, frames_per_row=TRUTH["row_time"]),
               map={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in truth["map"].items()}, parameters={k: truth[k] for k in ("roll", "stage", "scans")})
    angles = [np.concatenate([st["angle_deg"] for st in sc.stretches]) for sc in scans]
    heights = [np.concatenate([st["rows"] for st in sc.stretches]) for sc in scans]
    poses, _ = G.frame_poses(cal, angles, heights)
    mid = (ROWS - 1) / 2.0
    helix_first = int(offsets[1])
    origin = TRUTH["height"] + (1.0 + TRUTH["axial_scale"]) * heights[0][helix_first]            # the first scan's row at its helix' first frame
    worst = 0.0
    for s, sc in enumerate(scans):
        P = poses[s]
        for _ in range(300):
            stretch = int(rng.integers(0, 3))
            f = int(rng.integers(3, len(sc.stretches[stretch]["rows"]) - 3))
            c, v = rng.uniform(20, COLS - 20), rng.uniform(20, ROWS - 20)
            ray = G.rays(acq, truth, [[s, stretch, f, v, c]])[0]
            k = int(offsets[stretch]) + f
            eu, ev = G.map_displacement(truth["map"], c, v)
            X, Y = c + eu + P["col_scale"][k] * (c - P["scale_col"]), v + ev + P["row_scale"][k] * v
            u = X - P["axis"][k] - P["col_per_row"][k] * (Y - mid)
            h = Y - P["row"][k] - P["row_per_col"][k] * u
            th = np.radians(P["angle_deg"][k] + P["angle_deg_per_row"][k] * (v - mid))
            assert th == pytest.approx(np.radians(ray[0]), abs=1e-9)
            # the poses' axis holds the sample's place d . n(th) at the frame's own angle; the model at the row's
            dn = TRUTH["d"][0] * np.cos(np.radians(P["angle_deg"][k])) + TRUTH["d"][1] * np.sin(np.radians(P["angle_deg"][k]))
            dn_row = TRUTH["d"][0] * np.cos(th) + TRUTH["d"][1] * np.sin(th)
            worst = max(worst, abs(u - (ray[1] + dn_row - dn)), abs(h - (ray[2] + origin)))       # the slice h is the row it has in the helix' first frame
    # (the poses hold the stage's lift and its axis' move at the frame's own angle, the model at the row's: 0.2 px x
    # twice the 0.4 degrees a row here lies from its frame; a real scan's rows lie within 0.04 degrees)
    assert worst < 5e-3
    # a calibration with the axis' pitch towards the beam: every frame's row holds pitch x the depth of the sample's origin
    pitch = 3e-4
    leaning, _ = G.frame_poses(dict(cal, parameters=dict(cal["parameters"], pitch=pitch)), angles, heights)
    for k in (0, 1):
        th, d = np.radians(poses[k]["angle_deg"]), truth["scans"][k]["d"]
        assert np.allclose(leaning[k]["row"] - poses[k]["row"], pitch * (d[1] * np.cos(th) - d[0] * np.sin(th)), atol=1e-9)
        assert np.array_equal(leaning[k]["axis"], poses[k]["axis"])
    # the helix of the first scan, and the merged frame laid out from the poses
    assert G.merged_layout(cal, poses, heights)["measured"]["axis_first_scan"] == pytest.approx(np.median(poses[0]["axis"][offsets[1]:offsets[2]]))
    with pytest.raises(ValueError, match="not of two scans"):
        G.frame_poses(dict(cal, parameters=dict(cal["parameters"], scans=cal["parameters"]["scans"][:1])), angles, heights)


def test_a_tiles_own_views_under_a_pitched_axis():
    """Three tiles of a mosaic, each known by its own opposite views alone, about an axis pitched towards the beam:
    the views are not the same rays, their rows differ by twice the pitch x the depth, and a tile off the middle of
    the sample's detail has that as a sinusoid of the angle. Each tile's height_once takes it (the anchor's is the
    stage's), |height_once| / the tile's distance is the pitch, and the axial scale no longer holds what 2.55
    turns do not average out."""
    rng = np.random.default_rng(8)
    pitch, middle = 2e-4, np.array([400.0, -300.0])              # the sample's detail lies about `middle`, not about the anchor's axis
    places = [(0.0, 0.0), (1500.0, 0.0), (-700.0, 1300.0)]
    shared = {k: TRUTH[k] for k in ("roll", "row_time", "height_once", "axis_twice")}
    scans = [Scan([("helix", int(2.55 * PER_TURN))], index=k, axis=TRUTH["axis"] + 0.3 * k, axis_slope=TRUTH["axis_slope"], axial_scale=TRUTH["axial_scale"] + 1e-4 * k,
                  d=d, pitch=pitch, moved=k > 0, **shared) for k, d in enumerate(places)]
    comparisons = []
    for scan in scans:
        st, d, n = scan.stretches[0], np.array(scan.t["d"]), 0
        while n < 700:
            t = rng.uniform(2, len(st["rows"]) - 3 - PER_TURN / 2)
            th = np.radians(np.interp(t, np.arange(len(st["angle_deg"])), st["angle_deg"]))
            along, beam = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
            # a point the tile's field holds, at a depth about that of the sample's middle
            on_stage = rng.uniform(-200, 200) * along + ((middle + d) @ beam + rng.uniform(-100, 100)) * beam
            z = rng.uniform(30, ROWS - 30) - np.interp(t, np.arange(len(st["rows"])), st["rows"])
            c = _same_point(scan, "opposite", 0, 0, np.array([*(on_stage - d), z]), t, t + PER_TURN / 2, rng, 0.004)
            if c is not None:
                comparisons.append(c)
                n += 1
    acq = _acquisition(*scans)
    start = dict(roll=0.0, scans=[dict(axis=TRUTH["axis"] + 1.0) for _ in scans])
    got = G.solve_geometry(acq, comparisons, start=start)
    once = [f"scan{k}.height_once_{f}" for k in (1, 2) for f in ("cos", "sin")]
    assert set(once) <= set(got["free"]) and "scan0.height_once_cos" not in got["free"]
    assert got["solver"]["rank"] == len(got["free"]) == 1 + 4 + 3 * 3 + 4
    assert got["unused"] == [f"scan{k}.{n}" for k in (1, 2) for n in ("d0", "d1", "height")]      # no tile is compared with another here
    # B + B_k = the stage's lift + pitch x (the middle + d_k) . (-sin th, cos th)
    lean = lambda v: [pitch * v[1], -pitch * v[0]]
    assert got["stage"]["height_once"] == pytest.approx(np.add(TRUTH["height_once"], lean(middle)), abs=0.01)
    for k in (1, 2):
        s = got["scans"][k]
        assert s["height_once"] == pytest.approx(lean(places[k]), abs=0.012)
        assert np.hypot(*s["height_once"]) / np.hypot(*places[k]) == pytest.approx(pitch, rel=0.05)
    assert got["scans"][0]["height_once"] == [0.0, 0.0]
    for k, s in enumerate(got["scans"]):
        assert s["axial_scale"] == pytest.approx(TRUTH["axial_scale"] + 1e-4 * k, abs=4e-5), k
        assert s["axis"] == pytest.approx(TRUTH["axis"] + 0.3 * k, abs=3e-3)
    assert np.allclose(got["residual"], G.residuals(acq, got, comparisons))
    # held, a tile's sinusoid is what its rows are left with, and part of it is in its axial scale
    held = G.solve_geometry(acq, comparisons, start=start, held=once)
    rows = lambda sol, k: np.sqrt(np.mean(sol["residual"][[c["a"][0] == k for c in comparisons], 1] ** 2))
    assert rows(held, 1) > 3 * rows(got, 1) and rows(held, 2) > 3 * rows(got, 2)
    assert max(abs(s["axial_scale"] - TRUTH["axial_scale"] - 1e-4 * k) for k, s in enumerate(held["scans"])) > 1e-4
    # the anchor's own beside the stage's: one of the two, and the solve says so
    with pytest.raises(ValueError, match="RANK_DEFICIENT.*height_once"):
        G.solve_geometry(acq, comparisons, start=start, freed=["scan0.height_once_cos", "scan0.height_once_sin"])
    # the report: the pitch each tile's own views say, their direction a quarter turn from the tile's; the spreads
    told = G.report(acq, got, comparisons)
    assert "pitch_at_least_rad" not in told["scans"][0] and [s["name"] for s in told["scans"]] == ["scan"] * 3
    for k in (1, 2):
        s = told["scans"][k]
        assert s["from_anchor_px"] == pytest.approx(np.hypot(*places[k])) and s["pitch_at_least_rad"] == pytest.approx(pitch, rel=0.05)
        assert s["pitch_phase_less_direction_deg"] == pytest.approx(-90.0, abs=3.0)
    over = told["over_scans"]
    assert over["axis"]["n"] == 3 and over["axis"]["range"] == pytest.approx(0.6, abs=5e-3) and over["axial_scale"]["range"] == pytest.approx(2e-4, abs=5e-5)
    assert over["axial_scale"]["mean"] == pytest.approx(TRUTH["axial_scale"] + 1e-4, abs=3e-5)
    for k, s in enumerate(told["scans"]):
        v = s["opposite_views"]
        assert v["stretch"] == "helix" and v["n"] == 700 and abs(v["rows_apart"]) == pytest.approx(150.0, abs=2.0)
        # the solve took the rows' once per turn; under the held solve it is what the views leave (twice the lift)
        assert np.hypot(*v["left"]["rows"]["once"]) < 0.01 and np.hypot(*v["left"]["columns"]["twice"]) < 0.01
        assert v["row_time_frames_per_row"] == pytest.approx(TRUTH["row_time"], abs=1e-6)
    left = G.report(acq, held, comparisons)["scans"][1]["opposite_views"]["left"]["rows"]
    assert np.hypot(*left["once"]) == pytest.approx(2 * pitch * 1500.0, rel=0.1)
    # an acquisition whose rows' timing is not known: the frames of the opposite views give it (they are exact here)
    unknown = G.acquisition(dict(rows=ROWS, cols=COLS, row_time=0.0, scans=[sc.layout() for sc in scans]))
    for s in G.report(unknown, got, comparisons)["scans"]:
        assert s["opposite_views"]["row_time_frames_per_row"] == pytest.approx(TRUTH["row_time"], rel=0.01)


def test_report_whose_the_recorded_heights_departure_is():
    """A recorded height that leaves a straight line (Paris4: 0.25 rows rms): if the sample moved so, the record
    used frame by frame is right and the report's height_slope is 1; if the encoder alone did, it is 0."""
    def helix(wobble):
        scan = Scan([("helix", 5 * PER_TURN)], **TRUTH)
        f = np.arange(5 * PER_TURN)
        scan.stretches[0]["rows"] = scan.stretches[0]["rows"] + wobble * (0.3 * np.sin(2 * np.pi * f / 70.0) + 0.15 * np.cos(2 * np.pi * f / PER_TURN))
        return scan
    start = dict(roll=0.0, scans=[dict(axis=TRUTH["axis"] + 0.5)])
    moved = helix(1.0)                                             # the sample moves as the record says
    comparisons = _helix_comparisons(np.random.default_rng(9), moved, 0, 1500, 0.004)
    acq = _acquisition(moved)
    told = G.report(acq, G.solve_geometry(acq, comparisons, start=start), comparisons)["scans"][0]["opposite_views"]
    assert told["height_departure_rows"] == pytest.approx(np.hypot(0.3, 0.15) / np.sqrt(2), rel=0.1) and told["height_departure_between_views_rows"] > 0.2
    assert told["height_slope"] == pytest.approx(1.0, abs=0.02) and 0 < told["height_slope_error"] < 0.02
    straight = helix(0.0)                                          # the sample travels along the line; the record of it wobbles
    comparisons = _helix_comparisons(np.random.default_rng(9), straight, 0, 1500, 0.004)
    told = G.report(acq, G.solve_geometry(acq, comparisons, start=start), comparisons)["scans"][0]["opposite_views"]
    assert told["height_slope"] == pytest.approx(0.0, abs=0.05)
    # a rest turn has no travel to depart from: nothing is said
    rest = Scan([("rest1", PER_TURN), ("helix", 4 * PER_TURN)], **TRUTH)
    rng = np.random.default_rng(10)
    views = [c for c in (_same_point(rest, "opposite", 0, 0, p, t, t + PER_TURN / 2, rng, 0.004) for p, t in _points(rng, rest, 0, 900) if t + PER_TURN / 2 < PER_TURN - 3) if c is not None]
    acq = _acquisition(rest)
    told = G.report(acq, G.solve_geometry(acq, views, start=start, held=["scan0.axis_slope"]), views)["scans"][0]["opposite_views"]
    assert told["stretch"] == "rest1" and told["height_slope"] is None and told["row_time_frames_per_row"] is None and abs(told["rows_apart"]) < 1


def test_neighbours_of_a_mosaic_and_of_a_pair():
    """Which scans' reconstructions overlap: Paris4's tiles lie 6944 px apart with the axis 4110 columns from the
    detector's far edge, so neighbours share a lens and tiles two apart do not."""
    def tiles(rings, axis=4110.3, cols=4416):
        cells = [(q, r) for q in range(-rings, rings + 1) for r in range(-rings, rings + 1) if abs(q + r) <= rings]
        stretch = dict(name="helix", helix=True, angle_deg=[0.0, 0.036], rows=[0.0, -0.23])
        return G.acquisition(dict(rows=2368, cols=cols, scans=[dict(name=f"T{q}{r}", axis=axis, moved_on_stage=(q, r) != (0, 0), stretches=[stretch],
                                                                    on_stage=[6944.0 * (q + r / 2), 6944.0 * r * np.sqrt(3) / 2]) for q, r in cells]))
    edges, triangles = G.neighbours(tiles(1))
    assert len(edges) == 12 and len(triangles) == 6 and all(j < k for j, k in edges) and len(set(edges)) == 12
    assert all((i, j) in edges and (i, k) in edges and (j, k) in edges for i, j, k in triangles)
    edges, triangles = G.neighbours(tiles(2))
    assert len(edges) == 42 and len(triangles) == 24
    assert G.neighbours(tiles(1, axis=3000.0)) == ([], [])          # discs of 3000 px about axes 6944 px apart do not meet
    # a pair moved sideways sits at one place: the second scan reconstructs about an axis at its detector's edge,
    # and the two share the first one's disc. A scan whose axis column is not known has no disc.
    stretch = dict(name="helix", helix=True, angle_deg=[0.0, 0.045], rows=[0.0, -0.13])
    pair = lambda second: G.acquisition(dict(rows=1200, cols=2528, scans=[dict(name="HA", axis=1684.8, stretches=[stretch]), dict(name="TA", axis=second, stretches=[stretch])]))
    assert G.neighbours(pair(2528.1)) == ([(0, 1)], []) and G.neighbours(pair(None)) == ([], [])


def _layout(rows, cols, per_turn, travel, axis, stretches, sense=-1.0, second_axis=None, second_height=0.0):
    """A layout as read_acquisition gives it: stretches = (name, frames, first, end); the helix travels `travel`
    rows per turn in the sense of the rows. second_axis: a second scan of the same stretches with its axis there,
    its sample second_height rows higher by the record."""
    out, H, angle0 = [], 0.0, 0.0
    for name, n, first, end in stretches:
        f = np.arange(n)
        h = H + (sense * f * (travel / per_turn) if name == "helix" else np.zeros(n))
        out.append(dict(name=name, helix=name == "helix", angle_deg=angle0 + f * (360.0 / per_turn), rows=h, first=first, end=end))
        H, angle0 = h[-1], angle0 + n * (360.0 / per_turn)
    scans = [dict(name="scan", axis=axis, stretches=out)]
    if second_axis is not None:
        scans.append(dict(name="second", axis=second_axis, stretches=[dict(st, rows=st["rows"] + second_height) for st in out]))
    return G.acquisition(dict(rows=rows, cols=cols, per_turn=per_turn, row_sign=sense, fan=1e-5, width=512, scans=scans))


def test_plan_of_a_helix_alone_stays_in_the_strip():
    """Paris4's layout: a helix of which three turns are on disk, the axis 306 columns from the detector's edge, a
    turn's travel 65 rows short of the detector's height."""
    rows, cols, N, axis = 2368, 4416, 10000, 4110.3
    acq = _layout(rows, cols, N, 2303.0, axis, [("helix", 250000, 109700, 140300)])
    plan = G.plan_comparisons(acq)
    assert {p["kind"] for p in plan} == {"opposite"} and len(plan) > 100          # a turn on shares 65 rows: no turn pairs
    strip = (2 * axis - (cols - 1), cols - 1)
    for p in plan:
        assert p["mirror"] and p["from_block"] and p["cols"] == 512 and p["angle_offset_deg"] == 180.0 and p["frame_hint"] == N // 2
        for c in (p["col_a"], p["col_b"]):
            assert strip[0] + 30 <= c and c + p["cols"] + 30 <= cols                 # inside the strip, with room for the cut's margins
        assert p["col_a"] + p["col_b"] + p["cols"] - 1 == round(2 * axis)
        assert 109700 <= p["frame_a"] and p["frame_a"] + N // 2 + 64 <= 140300      # on disk, half a turn on too
        # A's rows and B's, half a turn's travel lower, on the detector
        assert 0 <= p["row_a"] and p["row_a"] + p["rows"] <= rows and p["row_a"] - 1152 >= 0
    frames = sorted({p["frame_a"] for p in plan})
    assert len(frames) >= 24 and np.allclose(np.diff(frames), N / 12, atol=1)        # twelve angles a turn, evenly
    angles = np.array(frames) % N
    assert len(np.unique(np.round(angles / (N / 12)).astype(int) % 12)) == 12


def test_plan_with_rest_turns():
    """343P's layout: a rest turn, a helix, a rest turn; the axis well inside the field."""
    rows, cols, N = 1200, 2528, 8000
    acq = _layout(rows, cols, N, 1053.0, 1684.8, [("rest1", N, 0, N), ("helix", 7 * N, 0, 7 * N), ("rest2", N, 0, N)])
    plan = G.plan_comparisons(acq)
    kinds = [p["kind"] for p in plan]
    assert set(kinds) == {"opposite", "turn", "resthelix"}
    for p in plan:
        assert 0 <= p["col_a"] and p["col_a"] + p["cols"] <= cols and 0 <= p["col_b"] and p["col_b"] + p["cols"] <= cols
        assert p["row_a"] == -1 or (0 <= p["row_a"] and p["row_a"] + p["rows"] <= rows)
        if p["kind"] == "opposite":
            assert p["col_a"] + p["col_b"] + p["cols"] - 1 == round(2 * 1684.8) and p["from_block"] == (p["a"] == 1)
        else:
            assert p["col_a"] == p["col_b"] and not p["mirror"]
    helix = sorted({p["frame_a"] for p in plan if p["kind"] == "opposite" and p["a"] == 1})
    assert 24 <= len(helix) <= 48                                                    # a long helix: whole turns are passed over
    assert len(np.unique(np.round((np.array(helix) - helix[0]) % N / (N / 12)).astype(int) % 12)) == 12
    assert {p["a"] for p in plan if p["kind"] == "resthelix"} == {0, 2} and {p["b"] for p in plan if p["kind"] == "resthelix"} == {1}
    # an axis at the detector's edge leaves no window for opposite views: the scan has none
    assert {p["kind"] for p in G.plan_comparisons(_layout(rows, cols, N, 1053.0, 2500.0, [("helix", 3 * N, 0, 3 * N)]))} == {"turn"}


def test_plan_of_a_pair_moved_sideways():
    """343P's two scans: the second holds the first's content 843 columns higher, its axis at the detector's edge."""
    rows, cols, N, move = 1200, 2528, 8000, 843
    stretches = [("rest1", N, 0, N), ("helix", 4 * N, 0, 4 * N), ("rest2", N, 0, N)]
    acq = _layout(rows, cols, N, 1053.0, 1684.8, stretches, second_axis=1684.8 + move + 0.3, second_height=-1.0)
    plan = G.plan_comparisons(acq)
    assert {(p["kind"], p["scan_a"], p["scan_b"]) for p in plan} == {("opposite", 0, 0), ("turn", 0, 0), ("resthelix", 0, 0), ("turn", 1, 1), ("resthelix", 1, 1), ("pair", 0, 1)}
    pairs = [p for p in plan if p["kind"] == "pair"]
    covered = np.zeros(cols, bool)
    for p in pairs:
        assert p["a"] == p["b"] and p["request_scan_a"] == 0 and p["request_scan_b"] == 1 and p["from_block"] and not p["mirror"] and p["frame_hint"] == 0
        assert p["col_b"] - p["col_a"] == move and 0 <= p["col_a"] and p["col_b"] + p["cols"] <= cols
        assert 0 <= p["row_a"] - 1 and p["row_a"] + p["rows"] <= rows               # the second scan's block lies a row lower
        covered[p["col_a"]:p["col_a"] + p["cols"]] = True
    assert covered[13:cols - move - 13].all() and not covered[cols - move:].any()   # the columns both hold, all of them
    assert {p["a"] for p in pairs} == {0, 1, 2}
    helix = sorted({p["frame_a"] for p in pairs if p["a"] == 1})
    assert len(helix) == 24 and len(np.unique(np.round((np.array(helix) - helix[0]) % N / (N / 12)).astype(int) % 12)) == 12
    assert len({p["frame_a"] for p in pairs if p["a"] == 0}) == 12
    # scans at other places on the rotation stage are not compared this way
    moved = G.acquisition(dict(rows=rows, cols=cols, per_turn=N, width=512, scans=[
        dict(name=name, axis=1684.8, moved_on_stage=m, stretches=[dict(name="helix", helix=True, angle_deg=np.arange(3 * N) * 360.0 / N, rows=-np.arange(3 * N) * 1053.0 / N)])
        for name, m in (("a", False), ("b", True))]))
    assert "pair" not in {p["kind"] for p in G.plan_comparisons(moved)}


def test_plan_of_tiles_follows_their_places():
    """Two tiles of a mosaic as Paris4's at a quarter: 1736 px apart, the axis 77 columns from the detector's edge.
    At equal angle the second holds the first's rays (d_k - d_j) . (cos th, sin th) columns further on, which leaves
    a window on both detectors only about the two angles at which the tiles line up along the beam; elsewhere the
    second is taken half a turn on, mirrored."""
    rows, cols, N, W, axis = 592, 1104, 10000, 64, 1027.6
    f = np.arange(3 * N + 600)
    stretch = lambda: dict(name="helix", helix=True, angle_deg=f * 360.0 / N, rows=-f * 575.8 / N, first=300, end=3 * N + 300)
    d = (1500.0, -874.0)
    acq = G.acquisition(dict(rows=rows, cols=cols, per_turn=N, row_sign=-1.0, fan=1e-5, width=W, scans=[
        dict(name="T0", axis=axis, on_stage=[0.0, 0.0], stretches=[stretch()]), dict(name="T1", axis=axis + 0.4, on_stage=list(d), moved_on_stage=True, stretches=[stretch()])]))
    places = dict(scans=[dict(axis=axis), dict(axis=axis + 0.4, d=list(d), height=5.0)])
    plan = G.plan_tiles(acq, places, [(0, 1)])
    kinds = {p["kind"] for p in plan}
    assert kinds == {"tiles", "tiles_opposite"} and len(plan) > 200
    apart, phase = np.hypot(*d), np.degrees(np.arctan2(d[1], d[0]))
    offsets = {"tiles": [], "tiles_opposite": []}
    for p in plan:
        assert (p["scan_a"], p["scan_b"], p["request_scan_a"], p["request_scan_b"]) == (0, 1, 0, 1) and p["from_block"] and p["cols"] == W
        assert p["sway_px"] == pytest.approx(apart) and p["sway_phase_deg"] == pytest.approx(phase) and p["sway_row_deg"] == 0.0
        opposite = p["kind"] == "tiles_opposite"
        assert p["mirror"] == opposite and p["angle_offset_deg"] == (180.0 if opposite else 0.0)
        assert p["frame_hint"] == 0 if not opposite else p["frame_hint"] == (N // 2 if p["frame_a"] + N // 2 + 80 <= 3 * N + 300 else -N // 2)
        # where B's block is read: its first column and the sway at the block's middle frame, of B's own angle
        th_b = (p["frame_a"] + 16 + p["frame_hint"]) * 360.0 / N
        first_b = p["col_b"] + apart * np.cos(np.radians(th_b - phase))
        for c in (p["col_a"], first_b):
            assert 13 <= c and c + W + 13 <= cols, p                                 # both windows on the detector, with the cut's margins
        assert 300 <= p["frame_a"] and 300 <= p["frame_a"] + p["frame_hint"] and p["frame_a"] + p["frame_hint"] + 64 <= 3 * N + 300
        assert 0 <= p["row_a"] and p["row_a"] + p["rows"] <= rows
        th_a = np.radians((p["frame_a"] + 16) * 360.0 / N)
        sway = d[0] * np.cos(th_a) + d[1] * np.sin(th_a)
        if opposite:                                                                 # the two windows are mirror images about half the axes' sum less the sway
            assert p["col_a"] + first_b + W - 1 == pytest.approx(2 * axis + 0.4 - sway, abs=1.5)
        else:
            assert first_b - p["col_a"] == pytest.approx(0.4 + sway, abs=1.5)
        offsets[p["kind"]].append(sway)
    # at equal angle only where the offset leaves both windows on the detector; that is both ways the tiles line up
    same = np.array(offsets["tiles"])
    assert np.abs(same).max() < cols - W and same.min() < -500 and same.max() > 500
    # the axis lies 77 columns from the detector's edge: half a turn apart the tiles share rays over the half of the
    # turn in which the second lies on the first's far side, and at about a third of the angles they share none
    mirrored = np.array(offsets["tiles_opposite"])
    assert mirrored.min() > -100 and mirrored.max() > 1500
    frames = sorted({p["frame_a"] for p in plan})
    assert 28 <= len(frames) <= 40 and len(np.unique(np.round(np.array(frames) % N / (N / 24)).astype(int) % 24)) >= 15
    assert {p["frame_hint"] for p in plan} == {0, N // 2, -N // 2}                    # the last half turn's partners lie before it
    # nothing where neither the columns nor the rows of the two blocks lie apart: the detector's own pattern would register
    near = G.plan_tiles(acq, dict(scans=[dict(axis=axis), dict(axis=axis, d=[20.0, 10.0])]), [(0, 1)])
    assert {p["kind"] for p in near} == {"tiles_opposite"}
    with pytest.raises(ValueError, match="two of the acquisition's scans"):
        G.plan_tiles(acq, places, [(0, 2)])


def _tile_comparisons(rng, scans, pairs, n, noise=0.004, radius=260.0):
    """Points of the sample seen by two tiles: at one angle ("tiles") and half a turn apart ("tiles_opposite")."""
    out = []
    for j, k in pairs:
        a, b = scans[j], scans[k]
        middle = -(np.array(a.t["d"]) + np.array(b.t["d"])) / 2                    # between the two axes, in the sample
        st, got = a.stretches[0], {"tiles": 0, "tiles_opposite": 0}
        for _ in range(40 * n):
            if min(got.values()) >= n:
                break
            t = rng.uniform(2, len(st["rows"]) - 3 - PER_TURN / 2)
            r, phi = radius * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            z = rng.uniform(30, ROWS - 30) - np.interp(t, np.arange(len(st["rows"])), st["rows"])
            p = np.array([middle[0] + r * np.cos(phi), middle[1] + r * np.sin(phi), z])
            for kind, later in (("tiles", 0.0), ("tiles_opposite", PER_TURN / 2)):
                c = _same_point(a, kind, 0, 0, p, t, t + later, rng, noise, other=b) if got[kind] < n else None
                if c is not None and b.on_detector(c["b"]):
                    out.append(c)
                    got[kind] += 1
    return out


def test_tiles_at_equal_angle_give_the_pitch():
    """Three tiles about an axis pitched towards the beam, compared with each other in projections. At one angle
    two tiles record the same rays, pitch x (d_k - d_j) . (-sin th, cos th) rows apart whatever the depth: that
    gives the pitch, with the tiles' places and heights. Half a turn apart they are not the same rays (their rows
    differ by twice the pitch x the depth), so those rows are left out of the solve and their columns kept."""
    rng = np.random.default_rng(21)
    pitch = 2e-4
    places = [(0.0, 0.0), (330.0, 60.0), (-150.0, 290.0)]
    shared = {k: TRUTH[k] for k in ("roll", "height_once", "axis_twice")}
    scans = [Scan([("helix", 3 * PER_TURN)], index=k, axis=TRUTH["axis"] - 0.2 * k, axial_scale=TRUTH["axial_scale"] + 1e-4 * k, d=d, height=1.5 * k, pitch=pitch, moved=k > 0, **shared)
             for k, d in enumerate(places)]
    acq = _acquisition(*scans)
    own = [c for sc in scans for c in _helix_comparisons(rng, sc, 0, 500, 0.004) if c["kind"] == "opposite"]
    between = _tile_comparisons(rng, scans, [(0, 1), (0, 2), (1, 2)], 400)
    assert {c["kind"] for c in between} == {"tiles", "tiles_opposite"} and len(between) == 6 * 400
    comparisons = own + between
    start = dict(roll=0.0, scans=[dict(axis=TRUTH["axis"] + 0.5, d=[d[0] + 2.0 * (k > 0), d[1] - 1.5 * (k > 0)]) for k, d in enumerate(places)])   # (the anchor's place is the origin)
    got = G.solve_geometry(acq, comparisons, start=start)
    assert "pitch" in got["free"] and got["solver"]["rank"] == len(got["free"]) == 2 + 4 + 3 * 3 + 2 * 3 + 2 * 2
    assert got["pitch"] == pytest.approx(pitch, rel=0.03)
    for k, (s, d) in enumerate(zip(got["scans"], places)):
        assert s["d"] == pytest.approx(list(d), abs=0.01) and s["height"] == pytest.approx(1.5 * k, abs=0.05), k
        assert s["axis"] == pytest.approx(TRUTH["axis"] - 0.2 * k, abs=5e-3) and s["axial_scale"] == pytest.approx(TRUTH["axial_scale"] + 1e-4 * k, abs=4e-5)
    assert got["roll"] == pytest.approx(TRUTH["roll"], abs=3e-5)
    tiles = np.array([c["kind"] == "tiles" for c in comparisons])
    opposite = np.array([c["kind"] == "tiles_opposite" for c in comparisons])
    assert np.sqrt(np.mean(got["residual"][tiles, 1:] ** 2)) < 0.01                # the same rays: rows and columns to the noise
    assert np.sqrt(np.mean(got["residual"][opposite, 2] ** 2)) < 0.01              # half a turn apart: the columns too,
    assert np.sqrt(np.mean(got["residual"][opposite, 1] ** 2)) > 0.03              # the rows not: twice the pitch x the depth
    # those rows are not in the solve: spoiled by 4 rows, they move nothing (but through the roll, which takes a
    # column from its row: 4 rows x 3e-4)
    spoiled = [dict(c, b=[*c["b"][:3], c["b"][3] + 4.0, c["b"][4]]) if c["kind"] == "tiles_opposite" else c for c in comparisons]
    again = G.solve_geometry(acq, spoiled, start=start)
    assert again["pitch"] == pytest.approx(got["pitch"], rel=1e-5) and again["scans"][1]["height"] == pytest.approx(got["scans"][1]["height"], abs=1e-4)
    # the pitch held at nothing: the tiles' rows at equal angle are left with pitch x their distance once per turn
    told = G.report(acq, G.solve_geometry(acq, comparisons, start=start, held=["pitch"]), comparisons)
    for pr in told["pairs"]:
        if pr["kind"] != "tiles":
            continue
        apart = np.subtract(places[pr["scan_b"]], places[pr["scan_a"]])
        assert np.hypot(*pr["left"]["rows"]["once"]) == pytest.approx(pitch * np.hypot(*apart), rel=0.25), pr
    # solved, nothing of it is left; the report has each pair of tiles under both kinds
    told = G.report(acq, got, comparisons)
    assert sorted((pr["scan_a"], pr["scan_b"], pr["kind"]) for pr in told["pairs"]) == sorted((j, k, kind) for j, k in ((0, 1), (0, 2), (1, 2)) for kind in ("tiles", "tiles_opposite"))
    for pr in told["pairs"]:
        assert pr["n"] == 400 and pr["left"]["columns"]["rms"] < 0.01
        if pr["kind"] == "tiles":
            assert np.hypot(*pr["left"]["rows"]["once"]) < 0.005 and pr["left"]["rows"]["rms"] < 0.01
    # no two scans apart compared at one angle: nothing gives the pitch, and it is not asked for; asked for, it is said
    alone = G.solve_geometry(acq, own, start=start)
    assert "pitch" not in alone["free"] and alone["pitch"] == 0.0
    assert "pitch" in G.solve_geometry(acq, own, start=start, freed=["pitch"])["unused"]
    mirrored = G.solve_geometry(acq, own + [c for c in between if c["kind"] == "tiles_opposite"], start=start)
    assert "pitch" not in mirrored["free"] and mirrored["scans"][1]["d"] == pytest.approx(list(places[1]), abs=0.02)
    assert "scan1.height" in mirrored["unused"]                                    # (their rows say nothing: no height either)


def test_tiles_in_projections_give_their_tilts():
    """Three tiles that hold the sample turned against each other, compared in projections. A projection holds of a
    rotation what is the same at every depth: the tilt's part about the beam's direction moves the columns by it x
    the height and the rows by it x the distance across the beam. So the tile pairs give the tilts; the spin is the
    same view further on in angle (the frames', not fitted) and is not asked of them."""
    rng = np.random.default_rng(31)
    places = [(0.0, 0.0), (330.0, 60.0), (-150.0, 290.0)]
    turns = [(0.0, 0.0, 0.0), (2.5e-4, -1.8e-4, 3e-4), (-2.2e-4, 3.1e-4, -2e-4)]
    about = 60.0
    scans = [Scan([("helix", 3 * PER_TURN)], index=k, d=d, height=1.5 * k, moved=k > 0, rotation=w, about=about) for k, (d, w) in enumerate(zip(places, turns))]
    acq = _acquisition(*scans)
    step = np.radians(360.0 / PER_TURN)
    own = [c for sc in scans for c in _helix_comparisons(rng, sc, 0, 400, 0.004) if c["kind"] == "opposite"]
    between = []
    for j, k in ((0, 1), (0, 2), (1, 2)):
        a, b = scans[j], scans[k]
        later = (turns[k][2] - turns[j][2]) / step                 # tile k holds the same view of the sample this many frames later
        st, n = a.stretches[0], 0
        while n < 500:
            t = rng.uniform(4, len(st["rows"]) - 6 - PER_TURN / 2)
            r, phi = 250.0 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)   # about the sample's origin: no mean depth
            z = rng.uniform(30, ROWS - 30) - np.interp(t, np.arange(len(st["rows"])), st["rows"])
            p = np.array([r * np.cos(phi), r * np.sin(phi), z])
            kind = "tiles" if n % 2 else "tiles_opposite"
            c = _same_point(a, kind, 0, 0, p, t, t + later + (PER_TURN / 2 if n % 2 == 0 else 0.0), rng, 0.004, other=b)
            if c is not None and b.on_detector(c["b"]):
                between.append(c)
                n += 1
    comparisons = own + between
    start = dict(roll=0.0, about_height=about, scans=[dict(axis=301.3, d=list(d)) for d in places])
    tilts = [f"scan{k}.tilt_{c}" for k in (1, 2) for c in "xy"]
    flat = G.solve_geometry(acq, comparisons, start=start, held=["pitch"])
    got = G.solve_geometry(acq, comparisons, start=start, held=["pitch"], freed=tilts)
    assert set(tilts) <= set(got["free"]) and got["solver"]["rank"] == len(got["free"])
    for k in (1, 2):
        assert got["scans"][k]["rotation"][:2] == pytest.approx(list(turns[k][:2]), abs=3e-5), k
        assert got["scans"][k]["rotation"][2] == 0.0 and got["scans"][k]["d"] == pytest.approx(list(places[k]), abs=0.02)
    assert got["scans"][0]["rotation"] == [0.0, 0.0, 0.0]
    kinds = np.array([c["kind"] for c in comparisons])
    cols = lambda sol, kind: np.sqrt(np.mean(sol["residual"][kinds == kind, 2] ** 2))
    assert cols(got, "tiles") < 0.6 * cols(flat, "tiles") and cols(got, "tiles_opposite") < 0.6 * cols(flat, "tiles_opposite")
    # the spin is not in a projection's rows or columns: asked of tile pairs alone, nothing gives it
    assert "scan1.spin" in G.solve_geometry(acq, comparisons, start=start, held=["pitch"], freed=tilts + ["scan1.spin"])["unused"]
    # a scan's own views do not see its rotation at all
    assert set(tilts) <= set(G.solve_geometry(acq, own, start=start, freed=tilts)["unused"])


def test_divergence_between_tiles_by_hand():
    """The beam diverges: a tile that holds the sample nearer the source records it magnified about the detector's
    middle. Paris4 by hand: a neighbour 6944 px along the beam, the pixel over the source's distance 6.34e-9, a
    column 2208 from the middle: 0.097 px, and nothing at the angle at which the tiles lie side by side."""
    per_depth, d = 1.129e-6 / 178.0, (0.0, 6944.0)                  # at the angle 0 the model's y lies along the beam
    f = np.arange(200)
    stretch = dict(name="helix", helix=True, angle_deg=f * 0.9, rows=np.zeros(200))
    layout = lambda k: dict(rows=2368, cols=4416, per_depth=k, scans=[dict(name="a", axis=2207.5, stretches=[stretch]), dict(name="b", axis=2207.5, moved_on_stage=True, on_stage=list(d), stretches=[stretch])])
    p = dict(scans=[dict(axis=2207.5), dict(axis=2207.5, d=list(d))])
    sightings = [[1, 0, 0.0, 1183.5 + 500.0, 2207.5 + 2208.0], [1, 0, 100.0, 1183.5 + 500.0, 2207.5 + 2208.0], [0, 0, 0.0, 1183.5 + 500.0, 2207.5 + 2208.0]]
    plain, magnified = G.rays(layout(0.0), p, sightings), G.rays(layout(per_depth), p, sightings)
    nearer = 6944.0 * per_depth
    assert nearer * 2208.0 == pytest.approx(0.0972, abs=1e-4)
    assert magnified[0, 1] - plain[0, 1] == pytest.approx(-nearer * 2208.0, rel=1e-9) and magnified[0, 2] - plain[0, 2] == pytest.approx(-nearer * 500.0, rel=1e-9)
    assert abs(magnified[1, 1] - plain[1, 1]) < 1e-12                 # a quarter turn on the tile lies beside the axis: no nearer
    assert np.array_equal(magnified[2], plain[2])                     # the scan on the axis


# ---------------------------------------------------------------- lens patches: two scans' volumes of one place

HEX = [(0.0, 0.0)] + [(400.0 * np.cos(a), 400.0 * np.sin(a)) for a in np.radians(np.arange(0, 360, 60))]


def _tiles(rng, about=120.0, rotations=True, own=None):
    """Seven tiles of a mosaic, 400 px apart: each holds the sample at its own place and height, off the motors' by
    a few pixels, and turned by up to 3e-4 rad; the first is the anchor. on_stage (the motors') is the place rounded."""
    scans = []
    for k, c in enumerate(HEX):
        truth = dict(own or {})
        if k:
            truth.update(d=(c[0] + rng.uniform(-6, 6), c[1] + rng.uniform(-6, 6)), height=rng.uniform(-8, 8), moved=True,
                         rotation=tuple(rng.uniform(-3e-4, 3e-4, 3)) if rotations else (0.0, 0.0, 0.0))
        scans.append(Scan([("helix", 3 * PER_TURN)], index=k, about=about, **truth))
    return scans


def _lens_patches(rng, scans, edges, n=60, noise=0.02, about=120.0, sigma=None):
    """Lens patches of the edges: places in the lens of the two tiles' discs (about the middle between their axes,
    150 px along it and 60 across, 150 rows up and down), and where the second tile holds each against the first."""
    out = []
    for a, b in edges:
        ca, cb = -np.array(scans[a].t["d"]), -np.array(scans[b].t["d"])          # the tiles' axes in the sample
        mid, across = (ca + cb) / 2, (cb - ca) / np.linalg.norm(cb - ca)
        along = np.array([-across[1], across[0]])
        for _ in range(n):
            p = np.array([*(mid + rng.uniform(-150, 150) * along + rng.uniform(-60, 60) * across), about + rng.uniform(-150, 150)])
            moved = scans[b].held(p) - scans[a].held(p) + rng.normal(0, noise, 3)
            out.append(dict(scan_a=a, scan_b=b, p=p.tolist(), moved=moved.tolist(), **({} if sigma is None else dict(sigma=sigma))))
    return out


def _edges(scans):
    return G.neighbours(_acquisition(*scans))[0]


def test_lens_patches_give_the_tiles_places_and_rotations():
    """A mosaic's tiles, each reconstructed alone, hold the lens two neighbours share at their own places: the
    patches' displacements give every tile's place, height and rotation against the anchor's."""
    rng = np.random.default_rng(11)
    scans = _tiles(rng)
    acq = _acquisition(*scans)
    edges = _edges(scans)
    assert len(edges) == 12
    lens = _lens_patches(rng, scans, edges, sigma=(0.02, 0.02, 0.02))
    start = dict(about_height=120.0, scans=[dict(d=list(HEX[k])) for k in range(7)])
    got = G.solve_geometry(acq, [], start=start, lens=lens)
    names = [f"scan{k}.{n}" for k in range(1, 7) for n in ("d0", "d1", "height", "tilt_x", "tilt_y", "spin")]
    assert got["free"] == names and got["unused"] == [] and got["solver"]["rank"] == 36 and got["about_height"] == 120.0
    assert not any(n.startswith("scan0.") and n.split(".")[1] in ("d0", "d1", "height", "tilt_x", "tilt_y", "spin") for n in got["free"] + got["unused"])
    for k, sc in enumerate(scans):
        s = got["scans"][k]
        assert s["d"] == pytest.approx(list(sc.t["d"]), abs=0.02) and s["height"] == pytest.approx(sc.t["height"], abs=0.02), k
        assert s["rotation"] == pytest.approx(list(sc.t["rotation"]), abs=3e-5), k
    assert got["lens_residual"].shape == (len(lens), 3) and np.allclose(got["lens_residual"], G.lens_residuals(acq, got, lens))
    left = {(l["scan_a"], l["scan_b"]): l for l in got["left"] if l["kind"] == "lens"}
    assert set(left) == {(-1, -1)} | set(edges) and left[(-1, -1)]["n"] == len(lens)
    assert all(0.012 < r < 0.03 for l in left.values() for r in l["rms"])           # the noise, and no more
    # under the truth the patches leave their noise (the model's cross product against the exact rotation: 1e-5 px)
    truth = dict(about_height=120.0, scans=[dict(d=list(sc.t["d"]), height=sc.t["height"], rotation=list(sc.t["rotation"])) for sc in scans])
    exact = _lens_patches(np.random.default_rng(0), scans, edges, n=5, noise=0.0)
    assert np.abs(G.lens_residuals(acq, truth, exact)).max() < 1e-4
    # the rotations held: what they would have explained is left on the edges
    rigid = G.solve_geometry(acq, [], start=start, lens=lens, held=[f"scan{k}.{n}" for k in range(1, 7) for n in ("tilt_x", "tilt_y", "spin")])
    assert max(l["rms"][j] for l in rigid["left"] for j in range(3)) > 0.03
    # the anchor's rotation beside the others': only differences are seen, and the solve says so
    with pytest.raises(ValueError, match="RANK_DEFICIENT"):
        G.solve_geometry(acq, [], start=start, lens=lens, freed=["scan0.spin"])
    # a tile no patch names keeps its place and has no rotation to ask for; nothing at all is an error
    some = [l for l in lens if 6 not in (l["scan_a"], l["scan_b"])]
    part = G.solve_geometry(acq, [], start=start, lens=some)
    assert "scan6.d0" not in part["free"] and part["unused"] == ["scan6.d0", "scan6.d1", "scan6.height"] and part["scans"][6]["d"] == pytest.approx(list(HEX[6]))
    with pytest.raises(ValueError, match="INSUFFICIENT_SUPPORT"):
        G.solve_geometry(acq, [], start=start)
    with pytest.raises(ValueError, match="with itself"):
        G.solve_geometry(acq, [], start=start, lens=[dict(scan_a=1, scan_b=1, p=[0, 0, 0], moved=[0, 0, 0])])


def test_a_mosaics_first_solve_is_two_that_share_no_unknown():
    """Each tile's own views and the lens patches in one problem: a tile's own views do not see a rigid motion of
    the sample, and a lens patch depends on places and rotations alone, so the places come out the same with the
    tiles' own blocks free, held, or not in the problem at all."""
    rng = np.random.default_rng(12)
    own = dict(axis=TRUTH["axis"], axis_slope=TRUTH["axis_slope"], axial_scale=TRUTH["axial_scale"], roll=TRUTH["roll"], row_time=TRUTH["row_time"],
               height_once=TRUTH["height_once"], axis_twice=TRUTH["axis_twice"])
    scans = _tiles(rng, own=own)[:3]                                              # the anchor and two neighbours: a triangle
    acq = _acquisition(*scans)
    comparisons = []
    for scan in scans:                                                             # points the tile's field holds: about its axis
        st, n = scan.stretches[0], 0
        while n < 300:
            t = rng.uniform(2, len(st["rows"]) - 3 - PER_TURN / 2)
            r, a = 250.0 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            z = rng.uniform(30, ROWS - 30) - np.interp(t, np.arange(len(st["rows"])), st["rows"])
            p = np.array([r * np.cos(a) - scan.t["d"][0], r * np.sin(a) - scan.t["d"][1], z])
            c = _same_point(scan, "opposite", 0, 0, p, t, t + PER_TURN / 2, rng, 0.004)
            if c is not None:
                comparisons.append(c)
                n += 1
    lens = _lens_patches(rng, scans, [(0, 1), (0, 2), (1, 2)], n=200, sigma=(0.02, 0.02, 0.02))
    start = dict(about_height=120.0, roll=0.0, scans=[dict(axis=TRUTH["axis"] + 1.0, d=list(HEX[k])) for k in range(3)])
    both = G.solve_geometry(acq, comparisons, start=start, lens=lens)
    places = [f"scan{k}.{n}" for k in (1, 2) for n in ("d0", "d1", "height", "tilt_x", "tilt_y", "spin")]
    assert set(places) <= set(both["free"]) and both["unused"] == [] and both["solver"]["rank"] == len(both["free"]) == 1 + 4 + 3 * 3 + 4 + 12
    # the tiles' own parameters are the ones their views give although the sample is turned in each (a rigid motion)
    for k, s in enumerate(both["scans"]):
        assert s["axis"] == pytest.approx(TRUTH["axis"], abs=3e-3) and s["axial_scale"] == pytest.approx(TRUTH["axial_scale"], abs=4e-5), k
    assert both["roll"] == pytest.approx(TRUTH["roll"], abs=2e-5) and _stage_found(both, 4e-3)
    alone = G.solve_geometry(acq, [], start=start, lens=lens)
    held = G.solve_geometry(acq, comparisons, start=start, lens=lens, held=["roll", "stage"] + [f"scan{k}.{n}" for k in range(3) for n in ("axis", "axis_slope", "axial_scale", "height_once_cos", "height_once_sin")])
    assert held["free"] == places
    for k in range(3):
        for other in (alone, held):
            assert other["scans"][k]["d"] == pytest.approx(both["scans"][k]["d"], abs=1e-6) and other["scans"][k]["height"] == pytest.approx(both["scans"][k]["height"], abs=1e-6)
            assert other["scans"][k]["rotation"] == pytest.approx(both["scans"][k]["rotation"], abs=1e-9)
        assert both["scans"][k]["d"] == pytest.approx(list(scans[k].t["d"]), abs=0.02) and both["scans"][k]["rotation"] == pytest.approx(list(scans[k].t["rotation"]), abs=5e-5)
    # no pair of a tile's own unknowns with a place is correlated: the two parts share no measurement
    own_names = {n for n in both["free"] if n == "roll" or n.startswith("stage") or n.split(".")[1] in ("axis", "axis_slope", "axial_scale", "height_once_cos", "height_once_sin")}
    assert not any((a in own_names) != (b in own_names) for a, b, _ in both["correlated"])


def test_a_quadratic_row_map_reads_as_tilts_in_the_lens():
    """What the lens can and cannot tell from a rigid motion: a height error of each tile that grows as the square
    of the distance from the tile's axis (the quadratic part of a detector's row map) is, between two tiles at one
    height, a tilt and a height: the solve returns tilts in proportion to the tiles' positions and heights as their
    square, and nothing is left on any edge. Over a slab it is not: a real tilt also moves the lens in the plane by
    the tilt x the height, which a row map does not, and that part is left."""
    rng = np.random.default_rng(13)
    scans = _tiles(rng, rotations=False)
    acq = _acquisition(*scans)
    a = 2e-6                                                                       # rows per px^2
    start = dict(about_height=120.0, scans=[dict(d=list(HEX[k])) for k in range(7)])

    def with_row_map(lens):
        for l in lens:
            ca, cb = -np.array(scans[l["scan_a"]].t["d"]), -np.array(scans[l["scan_b"]].t["d"])
            p = np.array(l["p"][:2])
            l["moved"][2] += a * (np.sum((p - cb) ** 2) - np.sum((p - ca) ** 2))
        return lens

    thin = _lens_patches(rng, scans, _edges(scans), noise=0.0)
    for l in thin:
        l["p"][2] = 120.0                                                          # every patch at the height the rotations are about
    got = G.solve_geometry(acq, [], start=start, lens=with_row_map(thin))
    assert max(l["rms"][j] for l in got["left"] for j in range(3)) < 1e-6
    for k, sc in enumerate(scans):
        d, s = np.array(sc.t["d"]), got["scans"][k]
        # a |p + d|^2 - a |p|^2 = 2 a p . d + a |d|^2: as rotation x p, the tilt (2 a d_y, -2 a d_x)
        assert s["rotation"] == pytest.approx([2 * a * d[1], -2 * a * d[0], 0.0], abs=1e-9), k
        assert s["height"] == pytest.approx(sc.t["height"] + a * d @ d, abs=1e-6) and s["d"] == pytest.approx(list(d), abs=1e-6)
    # a slab of 300 rows: the tilts that take the heights would move the lens' ends by 2 a x 400 px x 150 rows =
    # 0.24 px in the plane, which is not there; the solve settles between the two and leaves it on the edges
    tall = with_row_map(_lens_patches(rng, scans, _edges(scans), noise=0.0))
    slab = G.solve_geometry(acq, [], start=start, lens=tall)
    assert max(l["rms"][j] for l in slab["left"] for j in range(3)) > 0.03
    z = np.array([l["p"][2] - 120.0 for l in tall])
    on_edge = np.array([(l["scan_a"], l["scan_b"]) == (0, 1) for l in tall])
    # tile 1 lies along x from the anchor: its false tilt is about y, and what is left in x on that edge grows with the height
    assert abs(np.corrcoef(z[on_edge], slab["lens_residual"][on_edge, 0])[0, 1]) > 0.9


def test_unknowns_the_lens_hardly_tells_apart_are_listed():
    """One short lens far from the sample's origin: a spin of the tile and its place across the lens' direction
    move the patches alike. The rank check passes them; the solution lists the pair."""
    rng = np.random.default_rng(14)
    scans = [Scan([("helix", 3 * PER_TURN)], index=0), Scan([("helix", 3 * PER_TURN)], index=1, d=(3000.0, 0.0), height=1.0, moved=True, rotation=(0.0, 0.0, 1e-4))]
    acq = _acquisition(*scans)
    lens = []
    for _ in range(80):                                                             # 40 px of lens about (-1500, 0)
        p = np.array([-1500.0 + rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-20, 20)])
        lens.append(dict(scan_a=0, scan_b=1, p=p.tolist(), moved=(scans[1].held(p) - scans[0].held(p) + rng.normal(0, 0.02, 3)).tolist(), sigma=[0.02] * 3))
    got = G.solve_geometry(acq, [], start=dict(scans=[dict(), dict(d=[3000.0, 0.0])]), lens=lens)
    pairs = {(a, b): r for a, b, r in got["correlated"]}
    assert abs(pairs[("scan1.d1", "scan1.spin")]) > 0.99 and abs(pairs[("scan1.height", "scan1.tilt_y")]) > 0.99
    # the well-spread lenses of a mosaic leave no such pair
    scans = _tiles(rng)
    wide = G.solve_geometry(_acquisition(*scans), [], start=dict(about_height=120.0, scans=[dict(d=list(HEX[k])) for k in range(7)]), lens=_lens_patches(rng, scans, _edges(scans)))
    assert wide["correlated"] == []
