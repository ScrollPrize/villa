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


def test_bad_arguments(tmp_path, ct):
    with pytest.raises(SystemExit):
        spa.main(["--surface", "x.obj", "--volume", ct, "--out", str(tmp_path), "--fail-on", "nope=1"])
    assert spa.main(["--surface", "x.obj", "--volume", ct, "--out", str(tmp_path), "--half", "0"]) == 1
