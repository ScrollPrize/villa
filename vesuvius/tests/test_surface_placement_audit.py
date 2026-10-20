"""Surface placement audit: a surface at a known offset from a synthetic sheet must
report that offset, a flat volume must report no peak, and the obj range parser
must survive the partial lines a byte-range read starts and ends with."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import tifffile

pytest.importorskip("scipy")

from vesuvius.image_proc.run import surface_placement_audit as spa  # noqa: E402


def sheet_volume(shape=(64, 64, 64), z0=30.0, width=2.0):
    z = np.arange(shape[0])[:, None, None]
    v = 40 + 150 * np.exp(-0.5 * ((z - z0) / width) ** 2)
    return np.broadcast_to(v, shape).astype(np.uint8)


def write_obj(path, pts_xyz, normals_xyz, faces=True):
    with open(path, "w") as f:
        f.write("# VolCart OBJ File\n# VC OBJ Exporter v1.0\n")
        f.write(f"# Vertices: {len(pts_xyz)}\n")
        for p, n in zip(pts_xyz, normals_xyz):
            f.write(f"v {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}\nvn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        if faces:
            for i in range(len(pts_xyz)):
                f.write(f"vt {i / len(pts_xyz):.4f} 0.5\n")
            for i in range(1, len(pts_xyz) - 1):
                f.write(f"f {i}/{i}/{i} {i + 1}/{i + 1}/{i + 1} {i + 2}/{i + 2}/{i + 2}\n")


def grid_points(z, n=20, lo=16, hi=48):
    yy, xx = np.meshgrid(np.linspace(lo, hi, n), np.linspace(lo, hi, n))
    return np.stack([xx.ravel(), yy.ravel(), np.full(n * n, float(z))], 1)


@pytest.fixture
def ct(tmp_path):
    p = tmp_path / "ct.tif"
    tifffile.imwrite(p, sheet_volume())
    return str(p)


def run(tmp_path, ct, objs, *extra):
    out = tmp_path / "out"
    rc = spa.main(["--surface", *objs, "--volume", ct, "--out", str(out), "--samples", "300",
                   "--seed", "1", "--workers", "1", *extra])
    rows = list(csv.DictReader(open(out / "aggregate.csv")))
    return rc, out, rows


@pytest.mark.parametrize("shift", [0.0, 2.0, -3.0])
def test_known_offset_is_recovered(tmp_path, ct, shift):
    # surface placed `shift` voxels above the sheet peak along +z: the peak is at -shift
    pts = grid_points(30.0 + shift)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    rc, out, rows = run(tmp_path, ct, [str(obj)])
    assert rc == 0
    m = json.load(open(out / "s.json"))
    assert m["n_valid"] == m["n_samples"] > 0
    assert m["offset_median"] == pytest.approx(-shift, abs=0.5)
    assert m["offset_abs_gt_2_frac"] == (1.0 if abs(shift) > 2 else 0.0)
    assert m["off_sheet_frac"] == (1.0 if abs(shift) > 2 else 0.0)
    assert rows[0]["surface"] == "s"


def test_controls_on_a_flat_sheet(tmp_path, ct):
    # on a flat synthetic sheet the snapped points remeasure at 0 and tangent neighbours agree
    pts = grid_points(32.0)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    _, out, _ = run(tmp_path, ct, [str(obj)])
    m = json.load(open(out / "s.json"))
    assert m["snap_remeasure_sd"] == pytest.approx(0.0, abs=0.3)
    assert m["snap_remeasure_abs_gt_2_frac"] == 0.0
    for r in (2, 4, 8):
        assert m[f"tangent_absdiff_median_{r}"] == pytest.approx(0.0, abs=0.5)
    assert m["n_patches"] >= 1


def test_flipped_normal_flips_sign(tmp_path, ct):
    pts = grid_points(32.0)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, -1.0], (len(pts), 1)))
    _, out, _ = run(tmp_path, ct, [str(obj)])
    assert json.load(open(out / "s.json"))["offset_median"] == pytest.approx(2.0, abs=0.5)


def test_flat_volume_reports_no_peak(tmp_path):
    p = tmp_path / "flat.tif"
    tifffile.imwrite(p, np.full((64, 64, 64), 90, np.uint8))
    pts = grid_points(30.0)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    _, out, _ = run(tmp_path, str(p), [str(obj)])
    m = json.load(open(out / "s.json"))
    assert m["n_valid"] == 0 and m["no_peak_frac"] == 1.0
    assert np.isnan(m["offset_median"])


def test_points_outside_volume_are_not_counted(tmp_path, ct):
    pts = np.concatenate([grid_points(30.0, n=10), grid_points(30.0, n=10) + [200, 0, 0]])
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    _, out, _ = run(tmp_path, ct, [str(obj)])
    m = json.load(open(out / "s.json"))
    assert m["n_inside_volume"] == 100 and m["n_valid"] == 100


def test_fail_on(tmp_path, ct):
    pts = grid_points(33.0)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    rc, _, _ = run(tmp_path, ct, [str(obj)], "--fail-on", "offset_abs_gt_2_frac=0.5")
    assert rc == 2
    with pytest.raises(SystemExit):
        spa.parse_fail_on(["nonsense=1"])
    with pytest.raises(SystemExit):
        spa.parse_fail_on(["offset_median=inf"])


def test_samples_csv_has_coordinates(tmp_path, ct):
    pts = grid_points(30.0)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    _, out, _ = run(tmp_path, ct, [str(obj)])
    rows = list(csv.DictReader(open(out / "s.samples.csv")))
    assert len(rows) == 300
    assert float(rows[0]["z"]) == pytest.approx(30.0)
    assert set(rows[0]) >= {"z", "y", "x", "nz", "ny", "nx", "patch", "offset", "contrast", "valid"}


def test_nan_metric_fails_the_gate(tmp_path):
    p = tmp_path / "flat.tif"
    tifffile.imwrite(p, np.full((64, 64, 64), 90, np.uint8))
    pts = grid_points(30.0)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    rc, _, _ = run(tmp_path, str(p), [str(obj)], "--fail-on", "offset_abs_gt_2_frac=0.5")
    assert rc == 2


def test_parse_v_vn_drops_partial_and_unpaired_lines():
    blob = b"1.5 2.5\nv 1 2 3\nvn 0 0 1\nv 4 5 6\nvt 0.1 0.2\nv 7 8 9\nvn 1 0 0\nv 1 2\nvn 0 0 1\nv 3 4 nan\nvn 0 0 1\nv 10 11"
    P, N = spa._parse_v_vn(blob)
    assert P.tolist() == [[1, 2, 3], [7, 8, 9]]
    assert N.tolist() == [[0, 0, 1], [1, 0, 0]]


def test_vertex_region_end_finds_face_block(tmp_path):
    pts = grid_points(30.0, n=40)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    data = open(obj, "rb").read()
    end = spa._vertex_region_end(lambda s, n: data[s:s + n], len(data), probe=256)
    first_vt = data.index(b"\nvt ")
    assert end == first_vt + 1
    assert data[end - 256:end].count(b"\nv ") > 0


def test_obj_sampling_is_seeded(tmp_path):
    pts = grid_points(30.0, n=40)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    a = spa.sample_obj(str(obj), 50, 4, np.random.default_rng(3))
    b = spa.sample_obj(str(obj), 50, 4, np.random.default_rng(3))
    assert np.array_equal(a[0], b[0]) and a[3] == 1600
    assert a[0].shape[1] == 3 and np.allclose(np.linalg.norm(a[1], axis=1), 1)


def test_tifxyz_input(tmp_path, ct):
    from vesuvius.tifxyz.types import Tifxyz
    from vesuvius.tifxyz.writer import write_tifxyz

    yy, xx = np.meshgrid(np.linspace(16, 48, 24), np.linspace(16, 48, 24))
    surf = Tifxyz(_x=xx.astype(np.float32), _y=yy.astype(np.float32),
                  _z=np.full(xx.shape, 32.0, np.float32), _scale=(1.0, 1.0), path=None)
    d = write_tifxyz(tmp_path / "seg", surf)
    assert spa.is_tifxyz(str(d))
    rc, out, rows = run(tmp_path, ct, [str(d)])
    assert rc == 0
    m = json.load(open(out / "seg.json"))
    assert m["n_valid"] > 0
    # the planar surface's normal is +/-z, so the sheet 2 voxels below reads as +/-2
    assert abs(m["offset_median"]) == pytest.approx(2.0, abs=0.5)


def test_tifxyz_points_without_a_normal_are_counted_not_silently_dropped(tmp_path, ct):
    """A hole in the grid leaves its edge points without a normal. They cannot be audited;
    the JSON must say how many so a defect at a hole edge is not hidden (villa PR #2067)."""
    from vesuvius.tifxyz.types import Tifxyz
    from vesuvius.tifxyz.writer import write_tifxyz

    yy, xx = np.meshgrid(np.linspace(16, 48, 24), np.linspace(16, 48, 24))
    x, y, z = xx.astype(np.float32), yy.astype(np.float32), np.full(xx.shape, 32.0, np.float32)
    x[10:13, 10:13] = y[10:13, 10:13] = z[10:13, 10:13] = -1.0  # a 3 x 3 hole
    d = write_tifxyz(tmp_path / "seg", Tifxyz(_x=x, _y=y, _z=z, _scale=(1.0, 1.0), path=None))
    P, N, _, n_ok, n_no_normal, _grid = spa.sample_tifxyz(str(d), 10_000, np.random.default_rng(0))
    assert n_no_normal > 0 and n_ok + n_no_normal == 24 * 24 - 9
    assert np.isfinite(N).all() and len(P) == n_ok
    rc, out, rows = run(tmp_path, ct, [str(d)])
    assert rc == 0
    assert json.load(open(out / "seg.json"))["n_no_normal"] == n_no_normal


def test_grid_spikes_at_hole_edges_are_counted_and_located(tmp_path, ct):
    """A vertex thrown far off the surface next to a hole (villa PR #2067's resampling
    defect) has no normal and is never CT-sampled, so the grid check must catch it from
    geometry alone: count it, locate it, and leave a clean grid at zero."""
    from vesuvius.tifxyz.types import Tifxyz
    from vesuvius.tifxyz.writer import write_tifxyz

    yy, xx = np.meshgrid(np.linspace(16, 48, 24), np.linspace(16, 48, 24))
    x, y, z = xx.astype(np.float32), yy.astype(np.float32), np.full(xx.shape, 32.0, np.float32)
    x[10:13, 10:13] = y[10:13, 10:13] = z[10:13, 10:13] = -1.0
    has = x > 0
    clean = spa.grid_step_check(x, y, z, has)
    assert clean["n_grid_spikes"] == 0 and abs(clean["grid_step_median"] - 32 / 23) < 1e-3
    z[9, 10] = 32.0 + 400.0  # one hole-edge vertex 400 voxels off the plane
    bad = spa.grid_step_check(x, y, z, has)
    assert bad["n_grid_spikes"] == 1 and bad["grid_spike_rc"].tolist() == [[9, 10]]
    assert bad["grid_step_max"] > 399
    d = write_tifxyz(tmp_path / "seg", Tifxyz(_x=x, _y=y, _z=z, _scale=(1.0, 1.0), path=None))
    rc, out, rows = run(tmp_path, ct, [str(d), "--fail-on", "n_grid_spikes=0"])
    assert rc == 2
    j = json.load(open(out / "seg.json"))
    assert j["n_grid_spikes"] == 1 and "grid_spike_rc" not in j
    assert [r for r in csv.DictReader(open(out / "seg.grid_spikes.csv"))][0] == {"row": "9", "col": "10"}


def test_grid_spike_rule_edge_cases():
    """The rule must not fire on things that are not spikes (an untied island, a fold
    joined by long steps, duplicate vertices) and must count a whole spike run."""
    yy, xx = np.meshgrid(np.linspace(16, 48, 24), np.linspace(16, 48, 24))
    x, y, z = xx.astype(np.float32), yy.astype(np.float32), np.full(xx.shape, 32.0, np.float32)
    has = np.ones(x.shape, bool)
    # a 2x2 run of spikes counts 4 and locates all of them
    z2 = z.copy()
    z2[5:7, 5:7] += 400.0
    r = spa.grid_step_check(x, y, z2, has)
    assert r["n_grid_spikes"] == 4 and sorted(r["grid_spike_rc"].tolist()) == [[5, 5], [5, 6], [6, 5], [6, 6]]
    # an island separated from the surface by invalid points is not tied, so not a spike
    has_i = has.copy()
    has_i[8, :] = False
    z_i = z.copy()
    z_i[:8] += 400.0
    assert spa.grid_step_check(x, y, z_i, has_i)["n_grid_spikes"] == 0
    # a fold: the whole top half displaced is a large group, not a spike
    z_f = z.copy()
    z_f[:12] += 400.0
    assert spa.grid_step_check(x, y, z_f, has)["n_grid_spikes"] == 0
    # duplicate vertices: most steps zero, the rule stays quiet instead of flagging everything
    x_d, y_d = x.copy(), y.copy()
    x_d[1:, :] = x_d[:1, :]  # xx varies along axis 0 here, yy along axis 1
    y_d[:, 1:] = y_d[:, :1]
    r = spa.grid_step_check(x_d, y_d, z, has)
    assert r["n_grid_spikes"] == 0 and r["grid_step_median"] == 0.0
    # degenerate grids do not crash
    for h in (np.zeros((3, 3), bool), np.ones((1, 1), bool)):
        sub = (slice(0, h.shape[0]), slice(0, h.shape[1]))
        assert spa.grid_step_check(x[sub], y[sub], z[sub], h)["n_grid_spikes"] == 0
        assert spa.grid_hole_check(h)["n_grid_holes"] == 0


def test_grid_hole_check_is_four_connected():
    """Two interior invalid points touching only at a corner are two holes, not one."""
    has = np.ones((12, 12), bool)
    has[4, 4] = False
    has[5, 5] = False
    assert spa.grid_hole_check(has) == {"n_grid_holes": 2, "grid_hole_points": 2}


def test_obj_input_carries_nan_grid_metrics(tmp_path, ct):
    """An obj surface has no grid; the grid metrics must still exist (NaN) so that
    --fail-on on them does not crash and aggregate.csv keeps the columns whichever
    input comes first."""
    pts = grid_points(30.0)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    rc, out, rows = run(tmp_path, ct, [str(obj), "--fail-on", "n_grid_spikes=0"])
    assert rc == 2  # NaN fails the gate like any unmeasured metric
    j = json.load(open(out / "s.json"))
    assert all(j[k] != j[k] for k in spa.GRID_KEYS)  # NaN
    assert "n_grid_spikes" in rows[0]


def test_interior_grid_holes_are_counted_but_border_notches_are_not():
    """Only holes enclosed by valid points count (villa PR #2128's punched flattenings);
    a notch in the grid edge is the trace ending, not a defect."""
    has = np.ones((20, 20), bool)
    assert spa.grid_hole_check(has) == {"n_grid_holes": 0, "grid_hole_points": 0}
    has[5:8, 5:8] = False  # 9-point interior hole
    has[0:3, 10] = False  # notch touching the border
    has[15, 15] = False  # 1-point interior hole
    assert spa.grid_hole_check(has) == {"n_grid_holes": 2, "grid_hole_points": 10}


def test_surface_outside_the_volume_warns_and_is_reported(tmp_path, ct, capsys):
    """A surface in the wrong coordinate frame (villa PR #2112: missing downscale) lands
    mostly outside the volume; the audit must say so instead of reporting offsets."""
    pts = grid_points(28.0)
    pts[:, :2] += 400.0  # shift x, y far outside the test volume
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    obj = tmp_path / "far.obj"
    write_obj(obj, pts, nrm)
    rc, out, rows = run(tmp_path, ct, [str(obj)])
    assert rc == 0
    j = json.load(open(out / "far.json"))
    assert j["outside_volume_frac"] > 0.9
    assert "outside the volume" in capsys.readouterr().err


def test_surface_in_empty_space_warns(tmp_path, ct, capsys):
    """A missing downscale factor (villa PR #2112) can leave a surface inside the volume
    bounds but away from any sheet; every profile is then empty and the audit must say so."""
    pts = grid_points(14.0)  # 16 voxels below the sheet, further than --half reaches
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    obj = tmp_path / "empty.obj"
    write_obj(obj, pts, nrm)
    rc, out, rows = run(tmp_path, ct, [str(obj)])
    assert rc == 0
    j = json.load(open(out / "empty.json"))
    assert j["outside_volume_frac"] == 0.0 and j["no_peak_frac"] == 1.0
    assert "wrong coordinate frame" in capsys.readouterr().err


def test_bad_arguments(tmp_path, ct):
    with pytest.raises(SystemExit):
        spa.main(["--surface", "x.obj", "--volume", ct, "--out", str(tmp_path), "--fail-on", "nope=1"])
    assert spa.main(["--surface", "x.obj", "--volume", ct, "--out", str(tmp_path), "--half", "0"]) == 1


def test_chunk_cache_matches_direct_reads_and_reuses_chunks(tmp_path):
    rng = np.random.default_rng(3)
    vol = rng.integers(0, 255, (37, 50, 41), dtype=np.uint8)  # not a multiple of the chunk
    cache = spa.ChunkCache(vol, str(tmp_path / "cache"), default_chunk=16)
    windows = [(slice(0, 37), slice(0, 50), slice(0, 41)), (slice(15, 17), slice(31, 33), slice(16, 32)),
               (slice(36, 37), slice(49, 50), slice(40, 41)), (slice(3, 20), slice(0, 7), slice(30, 41))]
    for w in windows:
        np.testing.assert_array_equal(cache[w], vol[w])
    fetched = cache.fetched
    assert fetched > 0 and len(list((tmp_path / "cache").rglob("*.npy"))) == fetched
    for w in windows:
        np.testing.assert_array_equal(cache[w], vol[w])
    assert cache.fetched == fetched  # second pass came from disk


def test_cache_flag_gives_the_same_result(tmp_path, ct):
    pts = grid_points(28.0)
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    write_obj(tmp_path / "s.obj", pts, nrm)
    _, _, direct = run(tmp_path / "a", ct, [str(tmp_path / "s.obj")])
    _, _, cached = run(tmp_path / "b", ct, [str(tmp_path / "s.obj")], "--cache", str(tmp_path / "cache"))
    _, _, again = run(tmp_path / "c", ct, [str(tmp_path / "s.obj")], "--cache", str(tmp_path / "cache"))
    assert direct[0]["offset_median"] == cached[0]["offset_median"] == again[0]["offset_median"]
    assert list((tmp_path / "cache").rglob("*.npy"))


def two_sheet_volume(zs, shape=(64, 64, 64), width=1.5):
    z = np.arange(shape[0])[:, None, None]
    v = 40 + sum(150 * np.exp(-0.5 * ((z - z0) / width) ** 2) for z0 in zs)
    return np.broadcast_to(v, shape).astype(np.uint8)


def one_surface(tmp_path, ct, z, *extra):
    pts = grid_points(z)
    tmp_path.mkdir(parents=True, exist_ok=True)
    obj = tmp_path / "s.obj"
    write_obj(obj, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
    _, out, _ = run(tmp_path, ct, [str(obj)], *extra)
    return json.load(open(out / "s.json"))


def test_cache_does_not_mix_two_volumes_with_the_same_geometry(tmp_path):
    # Same shape, chunks and dtype but different scrolls: a shared --cache dir used to hand the
    # second run the first volume's intensities, so its sheet appeared where the first one's was.
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    tifffile.imwrite(a, sheet_volume(z0=30.0))
    tifffile.imwrite(b, sheet_volume(z0=34.0))
    cache = str(tmp_path / "cache")
    ma = one_surface(tmp_path / "ra", str(a), 30.0, "--cache", cache)
    mb = one_surface(tmp_path / "rb", str(b), 30.0, "--cache", cache)
    assert ma["offset_median"] == pytest.approx(0.0, abs=0.5)
    assert mb["offset_median"] == pytest.approx(4.0, abs=0.5)


def test_cache_keeps_levels_of_one_source_apart(tmp_path):
    vol = np.zeros((20, 20, 20), np.uint8)
    c0 = spa.ChunkCache(vol, str(tmp_path), source="s", level="0")
    c1 = spa.ChunkCache(vol, str(tmp_path), source="s", level="1")
    assert c0.root != c1.root


def test_non_default_spacing_picks_nearest_peak_by_distance_from_zero(tmp_path):
    # step 1.5: t = ..., -3, -1.5, 0, 1.5, 3, .... Sheets at -2 and +2.5 sample at -1.5 and +3;
    # the nearer is the negative one, but picking by index distance from the middle sample
    # (before the grid was built through zero, t=+1) chose the positive sheet.
    p = tmp_path / "ct.tif"
    tifffile.imwrite(p, two_sheet_volume([28.0, 32.5]))
    m = one_surface(tmp_path, str(p), 30.0, "--half", "8", "--step", "1.5", "--sigma", "0")
    assert m["offset_median"] == pytest.approx(-1.5, abs=0.3)


def test_non_default_spacing_reads_the_surface_intensity_at_zero(tmp_path):
    # A narrow sheet exactly under the surface is on-sheet. With no sample at 0 the code read t=+1.
    p = tmp_path / "ct.tif"
    tifffile.imwrite(p, sheet_volume(z0=30.0, width=0.5))
    m = one_surface(tmp_path, str(p), 30.0, "--half", "8", "--step", "1.5", "--sigma", "0")
    assert m["n_valid"] > 0
    assert m["off_sheet_frac"] == 0.0


def test_a_peak_at_the_search_limit_is_censored_not_valid(tmp_path):
    # The sheet sits at +8 = the search limit. It cannot be told from a profile still rising, so it
    # stays out of the offsets (interior rule unchanged) but is counted as censored.
    p = tmp_path / "ct.tif"
    tifffile.imwrite(p, sheet_volume(z0=38.0))
    m = one_surface(tmp_path, str(p), 30.0)
    assert m["n_valid"] == 0
    assert m["censored_frac"] == 1.0


def test_interior_peaks_are_not_censored(tmp_path):
    p = tmp_path / "ct.tif"
    tifffile.imwrite(p, sheet_volume(z0=32.0))
    m = one_surface(tmp_path, str(p), 30.0)
    assert m["n_valid"] > 0 and m["censored_frac"] == 0.0


def test_small_obj_vertex_block_is_found_without_bisection():
    # a sub-2 KB obj with a comment header used to be skipped past its own vertices
    data = b"# made by hand\n# second line\n" + b"".join(
        f"v {i} 0 0\nvn 0 0 1\n".encode() for i in range(3)) + b"f 1//1 2//2 3//3\n"
    end = spa._vertex_region_end(lambda s, n: data[s:s + n], len(data), probe=4096)
    assert data[:end].endswith(b"vn 0 0 1\n") and data[end:].startswith(b"f ")


def test_profile_grid_is_symmetric_and_bounded():
    vol = np.zeros((64, 64, 64), np.float32)
    P = np.array([[32.0, 32.0, 32.0]]); N = np.array([[0.0, 0.0, 1.0]])
    _, t, _ = spa.read_profiles(vol, P, N, half=8, step=1.5, sigma=1.0, workers=1)
    assert t.min() >= -8 and t.max() <= 8 and 0.0 in t
    assert np.allclose(t, -t[::-1])
