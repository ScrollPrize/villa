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


def _about_beam(a):
    """The detector's (column, row) of a position (x, z) in the frame of the rotation axis: the axis is rolled by a
    against the detector's columns."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s], [-s, c]])


class Scan:
    """A synthetic scan: what its record states per frame, and where it really holds the sample."""

    def __init__(self, stretches, travel=300.0, **truth):
        # stretches: (name, frames); the helix climbs `travel` rows per turn, a rest turn stands where the helix
        # starts or ends
        self.t = dict(axis=301.3, axis_slope=0.0, axial_scale=0.0, d=(0.0, 0.0), height=0.0, roll=0.0, row_time=0.0, stretch_height={})
        self.t.update(truth)
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
        return dict(name="scan", mid_height=self.mid_height, stretches=self.stretches)

    def seen(self, stretch, p, t):
        """Where the point p of the sample is recorded at the time of frame t (fractions allowed) of a stretch:
        (column, row), by explicit rigid motions."""
        st, tr, mid = self.stretches[stretch], self.t, (ROWS - 1) / 2.0
        th = np.radians(np.interp(t, np.arange(len(st["angle_deg"])), st["angle_deg"]))
        H = np.interp(t, np.arange(len(st["rows"])), st["rows"])
        on_stage = np.asarray(p, float) + [tr["d"][0], tr["d"][1], 0.0]            # the sample's place on the stage
        x, _, z = _stage(th) @ on_stage                                            # the stage turned; parallel rays: the depth is dropped
        x += tr["axis_slope"] * (H - self.mid_height)                              # the axis stage does not travel along the axis as the detector sees it
        z += tr["height"] + tr["stretch_height"].get(st["name"], 0.0) + (1.0 + tr["axial_scale"]) * H   # and it travels further than the record says
        c, v = _about_beam(tr["roll"]) @ [x, z - mid]
        return tr["axis"] + c, mid + v

    def sighting(self, stretch, p, t):
        """The sighting (frame, row, column) of p whose row was exposed at the time of frame t: the frame's own
        time is t - row_time (row - middle row)."""
        c, v = self.seen(stretch, p, t)
        return [0, stretch, t - self.t["row_time"] * (v - (ROWS - 1) / 2.0), v, c]

    def on_detector(self, g, margin=20):
        return margin < g[3] < ROWS - margin and margin < g[4] < COLS - margin

    def parameters(self):
        t = self.t
        return dict(roll=t["roll"], scans=[dict(axis=t["axis"], axis_slope=t["axis_slope"], axial_scale=t["axial_scale"], d=list(t["d"]), height=t["height"],
                                                 stretch_height=dict(t["stretch_height"]))])


def _acquisition(scan):
    return dict(rows=ROWS, cols=COLS, row_time=scan.t["row_time"], scans=[scan.layout()])


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


def _same_point(scan, kind, sa, sb, p, ta, tb, rng=None, noise=0.0):
    a, b = scan.sighting(sa, p, ta), scan.sighting(sb, p, tb)
    if not (scan.on_detector(a) and scan.on_detector(b)):
        return None
    if rng is not None:
        b[3] += rng.normal(0, noise)
        b[4] += rng.normal(0, noise)
    return dict(kind=kind, a=a, b=b, opposite=kind == "opposite")


TRUTH = dict(axis=310.4, axis_slope=4e-4, axial_scale=3.3e-3, d=(7.0, -4.0), height=2.5, roll=3e-4, row_time=2e-3)


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
    assert got["free"] == ["roll", "scan0.axis", "scan0.axis_slope", "scan0.axial_scale"] and got["unused"] == []
    assert got["solver"]["rank"] == 4 and got["solver"]["usable"]
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
    assert s["stretch_height"]["rest2"] == pytest.approx(heights["rest2"], abs=3e-3)
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
    assert alone["free"] == ["roll", "scan0.axis"]
    # the first rest turn stands at the helix' start, mid_height rows from where the axis column is counted
    there = TRUTH["axis"] + TRUTH["axis_slope"] * (0.0 - scan.mid_height)
    assert alone["scans"][0]["axis"] == pytest.approx(there, abs=2e-3)
