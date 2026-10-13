"""Phantom and parsing tests for interface_distance.py (no network, no zarr)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interface_distance as idist  # noqa: E402


def write_transform(path, matrix3x4, n_landmarks=4):
    """Schema-shaped transform.json with landmarks consistent with the matrix."""
    M = np.vstack([np.asarray(matrix3x4, float), [0, 0, 0, 1]])
    rng = np.random.default_rng(0)
    moving = rng.uniform(10, 100, size=(n_landmarks, 3))
    fixed = (np.c_[moving, np.ones(n_landmarks)] @ M.T)[:, :3]
    path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "fixed_volume": "phantom_fixed",
        "transformation_matrix": np.asarray(matrix3x4, float).tolist(),
        "fixed_landmarks": fixed.tolist(),
        "moving_landmarks": moving.tolist(),
    }))
    return M, fixed, moving


def default_args(**over):
    base = dict(window=20, profile_half=12, smooth_sigma=1.0, sigma_grad=1.5,
                sigma_tensor=3.0, aniso_ratio=2.0, prominence=0.25,
                max_zero_fraction=0.5, normal="volume", moving_offset_zyx=None)
    base.update(over)
    return SimpleNamespace(**base)


def slab_volume(shape, normal_zyx, plane_offset, thickness=12.0, lo=40.0, hi=200.0):
    """Reference volume: value hi inside a slab whose near face passes through
    the point at signed distance plane_offset along normal_zyx."""
    n = np.asarray(normal_zyx, float)
    n /= np.linalg.norm(n)
    zz, yy, xx = np.meshgrid(*(np.arange(s, dtype=float) for s in shape), indexing="ij")
    centre = (np.asarray(shape) - 1) / 2.0
    d = (zz - centre[0]) * n[0] + (yy - centre[1]) * n[1] + (xx - centre[2]) * n[2]
    vol = np.full(shape, lo, np.float32)
    vol[(d >= plane_offset) & (d <= plane_offset + thickness)] = hi
    return vol, n, centre


@pytest.mark.parametrize("mode", ["volume", "points"])
@pytest.mark.parametrize("offset", [0.0, 2.0, 4.0])
def test_phantom_identity_chain_recovers_distance(tmp_path, mode, offset):
    M3 = np.c_[np.eye(3), np.zeros(3)]
    write_transform(tmp_path / "id.json", M3)
    chain, _ = idist.load_chain([str(tmp_path / "id.json")])
    vol, n, centre = slab_volume((96, 96, 96), (0.3, 0.8, 0.52), offset)
    pt = {"id": "p", "zyx": centre, "normal_zyx": n}
    rec = idist.analyze_point(pt, vol, chain, default_args(normal=mode))
    assert rec["evaluable"], rec["non_evaluable_reasons"]
    assert abs(rec["D_voxels"] - offset) <= 0.5
    if mode == "volume":
        got = np.asarray(rec["normal_zyx"])
        assert abs(abs(got @ n) - 1.0) < 0.05


def test_distance_is_in_point_voxels_under_scaled_chain(tmp_path):
    # reference voxels are half the size of point voxels: p_ref = 2 * p_pt + t
    t = np.array([7.0, 3.0, 5.0])
    M3 = np.c_[2 * np.eye(3), t]
    write_transform(tmp_path / "scale.json", M3)
    chain, _ = idist.load_chain([str(tmp_path / "scale.json")])
    # interface at 3 point voxels (= 6 reference voxels) from the mapped point
    vol, n, ref_centre = slab_volume((160, 160, 160), (0.0, 0.6, 0.8), 6.0)
    pt_zyx = (ref_centre[::-1] - t) / 2.0  # xyz algebra, then back to zyx
    pt_zyx = pt_zyx[::-1]
    pt = {"id": "p", "zyx": pt_zyx, "normal_zyx": n}
    rec = idist.analyze_point(pt, vol, chain, default_args(normal="points"))
    assert rec["evaluable"], rec["non_evaluable_reasons"]
    assert abs(rec["D_voxels"] - 3.0) <= 0.5


def test_inverse_step_and_round_trip(tmp_path):
    M3 = np.array([[0.47, 0.006, 0.0, 5805.3],
                   [-0.006, 0.47, -0.001, 5493.9],
                   [0.0, 0.001, 0.47, 28084.0]])
    M, fixed, moving = write_transform(tmp_path / "t.json", M3)
    fwd, desc = idist.load_chain([str(tmp_path / "t.json")])
    inv, _ = idist.load_chain(["inv:" + str(tmp_path / "t.json")])
    assert desc == [{"file": str(tmp_path / "t.json"), "inverse": False}]
    assert np.allclose(inv @ fwd, np.eye(4), atol=1e-9)
    # inv: maps fixed landmarks onto moving landmarks (z,y,x in, z,y,x out)
    got = idist.map_points_zyx(fixed[:, ::-1], inv)
    assert np.allclose(got[:, ::-1], moving, atol=1e-6)
    both, _ = idist.load_chain([str(tmp_path / "t.json"), "inv:" + str(tmp_path / "t.json")])
    assert np.allclose(both, np.eye(4), atol=1e-9)


def test_point_outside_reference_is_rejected_cheaply(tmp_path):
    M3 = np.c_[np.eye(3), np.zeros(3)]
    write_transform(tmp_path / "id.json", M3)
    chain, _ = idist.load_chain([str(tmp_path / "id.json")])
    vol = np.zeros((40, 40, 40), np.float32)
    pt = {"id": "far", "zyx": np.array([500.0, 500.0, 500.0]), "normal_zyx": None}
    rec = idist.analyze_point(pt, vol, chain, default_args())
    assert rec["non_evaluable_reasons"] == ["outside_reference"]
    assert rec["D_voxels"] is None and "profile" not in rec
    # inside the array but in a masked (all-zero) region: also no resampling
    rec = idist.analyze_point({"id": "masked", "zyx": np.array([20.0, 20.0, 20.0]),
                               "normal_zyx": None}, vol, chain, default_args())
    assert rec["non_evaluable_reasons"] == ["masked_zero_fraction_exceeds_cap"]
    assert rec["masked_zero_fraction"] == 1.0 and "profile" not in rec


def test_moving_offset_is_added_before_chain(tmp_path):
    M3 = np.c_[2 * np.eye(3), np.zeros(3)]
    write_transform(tmp_path / "s.json", M3)
    chain, _ = idist.load_chain([str(tmp_path / "s.json")])
    p = np.array([[10.0, 20.0, 30.0]])
    assert np.allclose(idist.map_points_zyx(p, chain, (0, -1, 0)),
                       idist.map_points_zyx(p + [0, -1, 0], chain))


def test_read_points_depth_anchors(tmp_path):
    csv_text = (
        "cell_row,cell_col,y_px,x_px,region,sampled_from,center_layer,"
        "offset_from_plane,half_width_layers,base_x,base_y,base_z,"
        "normal_x,normal_y,normal_z\n"
        "3,4,224,288,15,center,33.5,1.5,2.0,100,200,300,0,0,2\n"
        "3,5,224,352,15,block_median,33.5,1.5,2.0,100,200,300,0,0,2\n"
        "4,4,288,288,15,center,32,0,2.0,10,20,30,0,0,0\n"
    )
    f = tmp_path / "anchors.csv"
    f.write_text(csv_text)
    pts = idist.read_points(str(f))
    assert [p["id"] for p in pts] == ["3_4", "4_4"]  # block_median dropped, zero normal kept
    assert np.allclose(pts[0]["zyx"], [301.5, 200.0, 100.0])  # base + 1.5 * unit normal
    assert np.allclose(pts[0]["normal_zyx"], [1.0, 0.0, 0.0])
    assert pts[0]["region"] == "15"
    assert pts[1]["normal_zyx"] is None and np.allclose(pts[1]["zyx"], [30.0, 20.0, 10.0])
    write_transform(tmp_path / "id.json", np.c_[np.eye(3), np.zeros(3)])
    chain, _ = idist.load_chain([str(tmp_path / "id.json")])
    vol = np.full((48, 48, 48), 100.0, np.float32)
    rec = idist.analyze_point(pts[1], vol, chain, default_args(normal="points"))
    assert not rec["evaluable"] and rec["non_evaluable_reasons"] == ["no_normal_supplied"]
    pts_all = idist.read_points(str(f), include_block_median=True)
    assert [p["id"] for p in pts_all] == ["3_4", "3_5", "4_4"]


def test_read_points_xyz(tmp_path):
    f = tmp_path / "pts.csv"
    f.write_text("id,x,y,z,nx,ny,nz\na,1,2,3,0,1,0\nb,4,5,6,,,\n")
    pts = idist.read_points(str(f))
    assert pts[0]["id"] == "a" and np.allclose(pts[0]["zyx"], [3, 2, 1])
    assert np.allclose(pts[0]["normal_zyx"], [0, 1, 0])
    assert pts[1]["normal_zyx"] is None


def test_summary_and_wilson():
    recs = [{"evaluable": True, "D_voxels": d, "non_evaluable_reasons": []}
            for d in (0, 1, 2, 5, 9)]
    recs.append({"evaluable": False, "D_voxels": None,
                 "non_evaluable_reasons": ["no_qualifying_interface"]})
    s = idist.summarize(recs, 3.0)
    assert (s["n_points"], s["n_evaluable"], s["n_within_threshold"]) == (6, 5, 3)
    assert s["median_D_voxels"] == 2.0
    lo, hi = s["wilson95_fraction_within"]
    assert lo < 0.6 < hi
    assert s["non_evaluable_reasons"] == {"no_qualifying_interface": 1}


def test_wilson_known_values():
    assert idist.wilson(123, 164) == pytest.approx([0.6785, 0.8100], abs=5e-4)
    assert idist.wilson(97, 163) == pytest.approx([0.518, 0.667], abs=5e-4)
    assert idist.wilson(0, 0) is None


def test_anisotropic_chain_keeps_axes_apart(tmp_path):
    # p_ref = (3x, 2y, 1z): a z<->x mix-up in the window or the chain reads the
    # profile along (3,2,1) instead of (1,2,3) and moves D off the analytic 4
    write_transform(tmp_path / "a.json", np.c_[np.diag([3.0, 2.0, 1.0]), np.zeros(3)])
    chain, _ = idist.load_chain([str(tmp_path / "a.json")])
    A = np.diag([1.0, 2.0, 3.0])  # the same map in z,y,x
    n_ref = np.ones(3) / np.sqrt(3)
    stretch = np.linalg.norm(A @ n_ref)  # reference voxels per point voxel along the normal
    vol, n_ref, centre = slab_volume((60, 100, 140), n_ref, 4.0 * stretch)
    pt = {"id": "p", "zyx": centre / np.diag(A), "normal_zyx": A @ n_ref / stretch}
    rec = idist.analyze_point(pt, vol, chain, default_args(normal="points"))
    assert rec["evaluable"], rec["non_evaluable_reasons"]
    assert abs(rec["D_voxels"] - 4.0) <= 0.5


def test_partial_masking_uses_raw_profile_in_points_mode(tmp_path):
    write_transform(tmp_path / "id.json", np.c_[np.eye(3), np.zeros(3)])
    chain, _ = idist.load_chain([str(tmp_path / "id.json")])
    vol, n, c = slab_volume((64, 64, 64), (0.0, 0.0, 1.0), 2.0)
    vol[:, 32:, 40:] = 0  # masked corner: x beyond s = 8 for points with y >= 32
    args = default_args(normal="points", max_zero_fraction=0.1)
    hit = idist.analyze_point({"id": "hit", "zyx": c + [0, 5, 0], "normal_zyx": n},
                              vol, chain, args)
    assert "masked_zero_fraction_exceeds_cap" in hit["non_evaluable_reasons"]
    assert 0.1 < hit["masked_zero_fraction"] < 1.0
    # this point's window also overlaps the masked corner; its profile does not
    miss = idist.analyze_point({"id": "miss", "zyx": c - [0, 5, 0], "normal_zyx": n},
                               vol, chain, args)
    assert miss["evaluable"], miss["non_evaluable_reasons"]
    assert abs(miss["D_voxels"] - 2.0) <= 0.5


def test_main_end_to_end(tmp_path, monkeypatch):
    write_transform(tmp_path / "id.json", np.c_[np.eye(3), np.zeros(3)])
    vol, _, _ = slab_volume((64, 64, 64), (0.0, 0.0, 1.0), 2.0)
    monkeypatch.setattr(idist, "open_reference", lambda path, level: vol)
    (tmp_path / "pts.csv").write_text(
        "id,x,y,z,nx,ny,nz\na,31.5,31.5,31.5,1,0,0\nb,31.5,31.5,31.5,,,\n")
    out, out_csv = tmp_path / "out.json", tmp_path / "out.csv"
    idist.main(["--points", str(tmp_path / "pts.csv"), "--step", str(tmp_path / "id.json"),
                "--reference", "phantom", "--normal", "points", "--window", "20",
                "--profile-half", "12", "--smooth-sigma", "1",
                "--output", str(out), "--csv", str(out_csv)])
    s = json.loads(out.read_text())["summary"]
    assert set(s) == {"n_points", "n_evaluable", "threshold_voxels", "n_within_threshold",
                      "fraction_within_threshold", "wilson95_fraction_within",
                      "median_D_voxels", "non_evaluable_reasons"}
    assert (s["n_points"], s["n_evaluable"], s["n_within_threshold"]) == (2, 1, 1)
    assert s["non_evaluable_reasons"] == {"no_normal_supplied": 1}
    assert len(out_csv.read_text().splitlines()) == 1 + 2  # header + one row per point
