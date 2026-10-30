"""vesuvius.snap_labels on synthetic volumes: single bands, and the gap ladder (three blurred 6 voxel sheets, the
recto-side neighbor of the middle one at gaps of 4, 2, 1, 0.5 and 0 voxels)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from vesuvius.snap_labels import cli
from vesuvius.snap_labels import face as F
from vesuvius.snap_labels import labels as L
from vesuvius.snap_labels.meshio import read_grid_tif, write_grid_tif, load_mesh, write_mesh
from vesuvius.snap_labels.volume import Sampler, Volume

GAPS = (4.0, 2.0, 1.0, 0.5, 0.0)     # gap between the target sheet and its recto-side neighbor, voxels
LADDER_W = 16                        # grid columns per gap
AXIS = ("line", (20.0, 12.0, -400.0), (1.0, 0.0, 0.0))     # far below: outward is +z, so the recto face is below
BAND_AXIS = ("line", (20.0, 20.0, -400.0), (1.0, 0.0, 0.0))


def write_zarr(root, vol, chunk=(16, 16, 16), levels=1):
    """An OME-Zarr v2 group of uint8 levels (level k samples level 0 every 2**k), uncompressed, '/' keys."""
    root = Path(root)
    datasets = []
    for k in range(levels):
        a = vol[::2 ** k, ::2 ** k, ::2 ** k]
        d = root / str(k)
        d.mkdir(parents=True, exist_ok=True)
        (d / ".zarray").write_text(json.dumps({
            "zarr_format": 2, "shape": list(a.shape), "chunks": list(chunk), "dtype": "|u1", "compressor": None,
            "fill_value": 0, "filters": None, "order": "C", "dimension_separator": "/"}))
        for i in range(-(-a.shape[0] // chunk[0])):
            for j in range(-(-a.shape[1] // chunk[1])):
                for m in range(-(-a.shape[2] // chunk[2])):
                    blk = np.zeros(chunk, np.uint8)
                    sub = a[i * chunk[0]:(i + 1) * chunk[0], j * chunk[1]:(j + 1) * chunk[1],
                            m * chunk[2]:(m + 1) * chunk[2]]
                    blk[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                    if not blk.any():
                        continue                      # a missing chunk reads as the fill value
                    p = d / str(i) / str(j)
                    p.mkdir(parents=True, exist_ok=True)
                    (p / str(m)).write_bytes(blk.tobytes())
        datasets.append({"path": str(k), "coordinateTransformations": [
            {"type": "scale", "scale": [float(2 ** k)] * 3}]})
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(json.dumps({"multiscales": [{
        "version": "0.4", "axes": [{"name": a, "type": "space"} for a in "zyx"], "datasets": datasets}]}))
    return root


def plane_mesh(H, W, z, step=2.0, x0=6.0, y0=6.0):
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xs = (x0 + step * c).astype(np.float32)
    ys = (y0 + step * r).astype(np.float32)
    zs = np.full((H, W), z, np.float32)
    return xs, ys, zs


def band(shape, ranges, value=200, background=20):
    v = np.full(shape, background, np.uint8)
    for lo, hi in ranges:
        v[lo:hi + 1] = value
    return v


def ladder_volume(psf=1.0, noise=0.0, seed=0):
    """Three 6-voxel sheets along z, blurred: the target [28, 34), a neighbor on its recto side (below, toward the
    axis) whose gap to the target steps through GAPS across x (LADDER_W voxels each, from x = 4), and one above at a
    gap of 4. Rendered at 8 sub-samples per voxel in z, so a fractional gap is a partial-volume dip."""
    nx = LADDER_W * len(GAPS) + 8
    Z, Y, ss = 64, 24, 8
    v = np.zeros((Z, Y, nx))
    for k in range(ss):
        zc = np.arange(Z) + (k + 0.5) / ss
        val = np.full((Z, Y, nx), 20.0)
        for j in range(nx):
            g = GAPS[min(len(GAPS) - 1, max(0, (j - 4) // LADDER_W))]
            for lo, hi in ((28 - g - 6, 28 - g), (28, 34), (38, 44)):
                val[(zc >= lo) & (zc < hi), :, j] = 200.0
        v += val / ss
    if psf > 0:
        v = F.gaussian_filter(v, psf)
    if noise > 0:
        v = v + np.random.default_rng(seed).normal(0, noise, v.shape)
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def ladder_mesh():
    xs, ys, zs = plane_mesh(9, LADDER_W * len(GAPS), 30.4, step=1.0, x0=4.0, y0=8.0)
    return xs, ys, zs, np.ones(xs.shape, bool)


def cols(g):
    b = GAPS.index(g)
    return slice(b * LADDER_W + 3, b * LADDER_W + LADDER_W - 3)


def sampler_of(root, vol, name="v.zarr"):
    return Sampler(Volume(write_zarr(Path(root) / name, vol)))


def small_mesh(z, H=9, W=9):
    xs, ys, zs = plane_mesh(H, W, z, step=1.0, x0=16.0, y0=16.0)
    return xs, ys, zs, np.ones(xs.shape, bool)


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    root = tmp_path_factory.mktemp("ladder")
    vol = Volume(write_zarr(root / "ladder.zarr", ladder_volume(1.0, 0.0)))
    return root, vol


# ---------------------------------------------------------------- the face snap


def test_faces_and_midline_of_one_band(tmp_path):
    sm = sampler_of(tmp_path, band((48, 40, 40), [(20, 25)]))      # faces at 19.5 and 25.5 (trilinear)
    for z0 in (21.0, 18.0, 24.7):
        for target, want in (("recto", 19.5), ("mid", 22.5), ("verso", 25.5)):
            r = F.snap_face(*small_mesh(z0), sm, axis=BAND_AXIS, opts={"target": target})
            assert r["apply"].all(), (z0, target)
            assert float(np.abs(r["zs"] - want).max()) < 0.05, (z0, target)


def test_recto_sign_and_no_side(tmp_path):
    sm = sampler_of(tmp_path, band((48, 40, 40), [(20, 25)]))
    m = small_mesh(22.0)
    assert abs(float(F.snap_face(*m, sm, recto_sign=1)["zs"].mean()) - 19.5) < 0.05
    assert abs(float(F.snap_face(*m, sm, recto_sign=-1)["zs"].mean()) - 25.5) < 0.05
    none = F.snap_face(*m, sm)                                   # no axis and no sign: the recto side is unknown
    assert not none["apply"].any()
    assert none["report"]["abstained"]["side not named"] == 81


def test_two_bands_about_as_near_abstain(tmp_path):
    sm = sampler_of(tmp_path, band((48, 40, 40), [(10, 14), (20, 24)]))   # faces 14.5 and 19.5
    xs, ys, zs, valid = small_mesh(17.2)                         # 2.7 and 2.3 voxels away: ambiguous
    r = F.snap_face(xs, ys, zs, valid, sm, axis=BAND_AXIS)
    assert not r["apply"].any()
    np.testing.assert_array_equal(r["zs"], zs)
    assert r["report"]["abstained"]["ambiguous or off the profile"] == 81


def test_far_band_tangent_normal_and_bad_target(tmp_path):
    sm = sampler_of(tmp_path, band((48, 40, 40), [(30, 35)]), "far.zarr")
    m = small_mesh(22.0)                                         # 7.5 voxels from the band: past max_move 4
    r = F.snap_face(*m, sm, axis=BAND_AXIS, opts={"half": 16.0})
    assert r["report"]["abstained"]["too far"] == 81
    r = F.snap_face(*m, sm, axis=BAND_AXIS, opts={"half": 16.0, "max_move": 8.0})
    assert float(np.abs(r["zs"] - 29.5).max()) < 0.05
    r = F.snap_face(*m, sm, axis=("line", (20.0, 20.0, 0.0), (0.0, 0.0, 1.0)))
    assert r["report"]["abstained"]["side not named"] == 81      # the normal is tangent to the radial direction
    with pytest.raises(F.SnapError):
        F.snap_face(*m, sm, axis=BAND_AXIS, opts={"target": "edge"})


def test_window16_reads_a_thick_band_the_default_window_cuts_off(tmp_path):
    sm = sampler_of(tmp_path, band((64, 40, 40), [(26, 41)]))      # faces at 25.5 and 41.5
    m = small_mesh(28.0)                                         # upper face 13.5 voxels up: off a +-12 profile
    r = F.snap_face(*m, sm, recto_sign=1, opts=dict(F.FACE_SNAP_PER_VERTEX))
    assert r["report"]["abstained"]["ambiguous or off the profile"] == 81
    assert not np.isfinite(r["detail"]["measured"]).any()
    r = F.snap_face(*m, sm, recto_sign=1, opts=dict(F.FACE_SNAP_WINDOW16))
    assert r["apply"].all() and float(np.abs(r["zs"] - 25.5).max()) < 0.05
    assert r["record"]["max_move"] == 20.0


def test_fallback_window_measures_where_the_wide_one_runs_out_of_volume(tmp_path):
    sm = sampler_of(tmp_path, band((64, 40, 40), [(46, 51)]))      # faces at 45.5 and 51.5
    m = small_mesh(50.0)                                         # +16 reaches z 66, past the volume's end
    wide = F.snap_face(*m, sm, recto_sign=1, opts=dict(F.FACE_SNAP_PER_VERTEX, half=16.0, min_contrast=0.0))
    assert wide["report"]["abstained"]["missing data"] == 81
    narrow = F.snap_face(*m, sm, recto_sign=1, opts=dict(F.FACE_SNAP_PER_VERTEX, half=12.0, min_contrast=0.0))
    both = F.snap_face(*m, sm, recto_sign=1, opts=dict(F.FACE_SNAP_WINDOW16))
    assert both["apply"].all() and float(np.abs(both["zs"] - 45.5).max()) < 0.05
    np.testing.assert_array_equal(both["detail"]["measured"], narrow["detail"]["measured"])
    for fb in (16.0, 20.0, 0.0):
        with pytest.raises(F.SnapError):
            F.snap_face(*m, sm, recto_sign=1, opts=dict(F.FACE_SNAP_WINDOW16, fallback_half=fb))


class ProfileSampler:
    """Returns profile k of vertex (r, c) at (2c, 2r, t) on a flat grid at z 0 (normal +z): the samples exactly."""

    def __init__(self, prof, W, half=16.0, step=0.25):
        self.prof, self.W, self.half, self.step = prof, W, half, step

    def trilinear(self, q):
        q = np.asarray(q, dtype=np.float64)
        c, r = np.rint(q[:, 0] / 2.0).astype(int), np.rint(q[:, 1] / 2.0).astype(int)
        return self.prof[r * self.W + c, np.rint((q[:, 2] + self.half) / self.step).astype(int)]


def random_profiles(rng, t, n):
    """Blurred bands of random thickness, place and contrast, with noise and some missing samples."""
    out = np.empty((n, len(t)))
    for i in range(n):
        p = np.full(len(t), rng.uniform(10, 80))
        for _ in range(rng.integers(1, 4)):
            lo, thick, s = rng.uniform(-20, 14), rng.uniform(0.5, 26), rng.uniform(0.3, 2.0)
            p += rng.uniform(20, 200) * (0.5 * np.tanh((t - lo) / s) - 0.5 * np.tanh((t - lo - thick) / s))
        p += rng.normal(0, rng.choice([0.0, 2.0, 8.0]), len(t))
        if rng.random() < 0.1:
            p[: rng.integers(1, 20)] = np.nan
        out[i] = p
    return out


def test_window16_is_the_wide_window_else_the_fallback_on_random_profiles():
    rng = np.random.default_rng(7)
    t = np.arange(-16.0, 16.125, 0.25)
    H, W = 12, 50
    prof = random_profiles(rng, t, H * W)
    rs = np.where(rng.random((H, W)) < 0.5, 1.0, -1.0)
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mesh = ((2.0 * c).astype(np.float32), (2.0 * r).astype(np.float32), np.zeros((H, W), np.float32),
            np.ones((H, W), bool))

    def measured(opts):
        res = F.snap_face(*mesh, ProfileSampler(prof, W), recto_sign=rs, opts=opts, face_opts={"sigma": 1.0})
        return res["detail"]["measured"]

    both = measured(dict(F.FACE_SNAP_WINDOW16))
    wide = measured(dict(F.FACE_SNAP_PER_VERTEX, half=16.0, min_contrast=0.0))
    narrow = measured(dict(F.FACE_SNAP_PER_VERTEX, half=12.0, min_contrast=0.0))
    want = np.where(np.isfinite(wide), wide, narrow)
    np.testing.assert_array_equal(both, want)
    assert np.isfinite(wide).sum() > 100 and (np.isfinite(narrow) & ~np.isfinite(wide)).sum() > 10
    assert (np.isfinite(wide) & np.isfinite(narrow) & (np.abs(wide - narrow) > 0.25)).any()   # the windows differ


def test_invalid_vertices_stay(tmp_path):
    sm = sampler_of(tmp_path, band((48, 40, 40), [(20, 25)]))
    xs, ys, zs, valid = small_mesh(21.0)
    valid[4, 4] = False
    xs[4, 4] = ys[4, 4] = zs[4, 4] = -1.0
    r = F.snap_face(xs, ys, zs, valid, sm, axis=BAND_AXIS)
    assert not r["apply"][4, 4] and float(r["zs"][4, 4]) == -1.0
    assert float(np.abs(r["zs"][valid] - 19.5).max()) < 0.05


@pytest.mark.parametrize("noise", [0.0, 10.0])
def test_ladder_merged_rule_snaps_from_two_voxels_and_abstains_under_one(tmp_path, noise):
    sm = sampler_of(tmp_path, ladder_volume(1.0, noise))
    r = F.snap_face(*ladder_mesh(), sm, axis=AXIS, opts=dict(F.FACE_SNAP_PER_VERTEX_MERGED), face_opts={"sigma": 1.0})
    for g in (4.0, 2.0):
        assert r["apply"][:, cols(g)].all(), g
        assert float(np.abs(r["zs"][:, cols(g)] - 27.5).max()) < 0.75, g
    for g in (0.5, 0.0):                           # under a 1 voxel gap the band is two sheets: stay put
        assert not r["apply"][:, cols(g)].any(), g
        np.testing.assert_allclose(r["zs"][:, cols(g)], 30.4, atol=1e-5)
    assert r["report"]["abstained"]["too thick"] > 0
    assert abs(r["report"]["ref_band_thick"] - 5.5) < 0.6


def test_ladder_plain_per_vertex_form_lands_on_the_wrong_sheet_under_one_voxel(tmp_path):
    sm = sampler_of(tmp_path, ladder_volume(1.0, 0.0))
    r = F.snap_face(*ladder_mesh(), sm, axis=AXIS, opts=dict(F.FACE_SNAP_PER_VERTEX), face_opts={"sigma": 1.0})
    assert r["report"]["abstained"]["too thick"] == 0
    for g in (0.5, 0.0):
        assert r["apply"][:, cols(g)].all()
        assert float(np.median(np.abs(r["zs"][:, cols(g)] - 27.5))) > 4.0, g   # the neighbor's recto face


def test_self_check_passes_on_a_clean_band_and_can_fail(tmp_path, ladder):
    sm = sampler_of(tmp_path, band((48, 40, 40), [(20, 25)]))
    opts = L.face_opts(merged=True)

    def snapper(a, b, c, v, smp):
        return F.snap_face(a, b, c, v, smp, axis=BAND_AXIS, opts=opts, face_opts={"sigma": 1.0})

    out = F.self_check(*small_mesh(22.0), sm, snapper, F.DEFAULT_CHECK_SHIFT["face"], 1.0, 1.0, 0.8, 3.0, "snapped")
    assert out["passed"], out["why"]
    assert out["control_shifted"]["caught"] and out["control_empty"]["caught"]

    def frozen(a, b, c, v, smp):                   # a "snap" that returns its input: recovers nothing
        return {"xs": a, "ys": b, "zs": c}

    assert not F.self_check(*small_mesh(22.0), sm, frozen, 2.0, 1.0, 1.0, 0.8, 3.0, "snapped")["passed"]
    # on the ladder 60% of the vertices abstain (merged bands and their edges), so the check fails there: it
    # measures recovery over every vertex with a normal, and abstaining vertices do not move back
    _, vol = ladder

    def ladder_snap(a, b, c, v, smp):
        return F.snap_face(a, b, c, v, smp, axis=AXIS, opts=opts, face_opts={"sigma": 1.0})

    assert not F.self_check(*ladder_mesh(), Sampler(vol), ladder_snap, 2.0, 1.0, 1.0, 0.8, 3.0, "snapped")["passed"]


# ---------------------------------------------------------------- label products


def test_label_snap_abstain_reason_and_confidence(ladder):
    _, vol = ladder
    r = L.label_snap(*ladder_mesh(), Sampler(vol), axis=AXIS, merged=True)
    for g in (4.0, 2.0):
        c = cols(g)
        assert (r["abstain"][:, c] == L.ABSTAIN_OK).all() and (r["reason"][:, c] == 0).all()
        assert (r["confidence"][:, c] > 0).all()
        assert float(np.abs(r["zs"][:, c] - 27.5).max()) < 0.75
    for g in (0.5, 0.0):
        c = cols(g)
        assert (r["abstain"][:, c] == L.ABSTAIN_YES).all()
        assert (r["reason"][:, c] == 7).all()                    # too thick: two sheets merged
        assert (r["confidence"][:, c] == 0).all()
    plain = L.label_snap(*ladder_mesh(), Sampler(vol), axis=AXIS)
    assert not L.DEFAULTS["merged"] and not (plain["reason"] == 7).any()


def test_confidence_falls_with_thickness_toward_the_cap():
    det = {"floor": 10.0, "ambiguous": 2.0, "cap": 10.0, "ref_thick": 5.0,
           "contrast": np.array([[40.0, 40.0, 15.0]]), "sep": np.array([[np.inf, np.inf, np.inf]]),
           "thick": np.array([[5.0, 9.0, 5.0]])}
    c = L.confidence(det, np.array([[True, True, True]]))
    assert c[0, 0] == pytest.approx(1.0) and c[0, 1] == pytest.approx(0.2, abs=1e-5)
    assert c[0, 2] == pytest.approx(0.5, abs=1e-5)


def test_components_and_review_regions():
    m = np.array([[1, 1, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]], bool)
    lab = L.components(m)
    assert int(lab.max()) == 3 and int((lab > 0).sum()) == int(m.sum())
    u = np.zeros((5, 5), bool)
    u[:, 0] = u[:, 4] = u[4, :] = True
    assert int(L.components(u).max()) == 1
    ab = np.zeros((10, 10), np.uint8)
    ab[1:4, 1:4] = L.ABSTAIN_YES
    ab[6:8, 6:9] = L.ABSTAIN_YES
    ab[0, 9] = L.ABSTAIN_YES                                     # one vertex: under review_min
    reason = np.where(ab == L.ABSTAIN_YES, 7, 0).astype(np.uint8)
    xs, ys, zs = plane_mesh(10, 10, 5.0, step=1.0, x0=0.0, y0=0.0)
    regs = L.review_regions(xs, ys, zs, ab, reason, min_size=4)
    assert [r["size"] for r in regs] == [9, 6] and regs[0]["reason"] == "too thick"
    doc = L.review_collection(regs)
    assert doc["vc_pointcollections_json_version"] == "1"
    assert doc["collections"]["1"]["points"]["1"]["p"] == [2.0, 2.0, 5.0]


def test_surface_label_values_halo_and_level_scale(tmp_path):
    def write(name, ab, scale=1.0, shape=(40, 40, 40)):
        xs, ys, zs = plane_mesh(6, 6, 20.0)
        valid = np.ones(xs.shape, bool)
        n, _ = F.normals(xs, ys, zs, valid)
        summ = L.write_surface_label(tmp_path / name, xs, ys, zs, valid, ab, n, shape, scale=scale, halo=2.0,
                                     chunk=16)
        shp, blocks = L.read_label_chunks(tmp_path / name)
        full = np.zeros(shp, np.uint8)
        for (cz, cy, cx), b in blocks.items():
            sl = tuple(slice(k * 16, k * 16 + 16) for k in (cz, cy, cx))
            full[sl] = b[:full[sl].shape[0], :full[sl].shape[1], :full[sl].shape[2]]
        return summ, full

    summ, full = write("a.zarr", np.zeros((6, 6), np.uint8))
    assert set(np.unique(full).tolist()) == {0, 1} and (np.nonzero(full == 1)[0] == 20).all()
    ab = np.zeros((6, 6), np.uint8)
    ab[0, 0] = L.ABSTAIN_YES
    summ, full = write("b.zarr", ab)
    assert summ["cells_ignore"] == 1 and set(np.unique(full).tolist()) == {0, 1, 2}
    assert (full[18:23, 7, 7] == 2).all() and int(full[24, 7, 7]) == 0
    assert int(full[20, 8, 10]) == 1                             # an edge shared with a good cell stays surface
    summ, full = write("c.zarr", np.zeros((6, 6), np.uint8), scale=0.5, shape=(20, 20, 20))
    assert (np.nonzero(full == 1)[0] == 10).all()
    with pytest.raises(L.LabelSnapError):
        write("c.zarr", np.zeros((6, 6), np.uint8))
    import zarr
    z = zarr.open(str(tmp_path / "a.zarr" / "0"), mode="r")      # the label reads back through zarr itself
    assert int(z[20, 10, 10]) == 1 and int(z[25, 10, 10]) == 0


# ---------------------------------------------------------------- command line


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    write_zarr(root / "ladder.zarr", ladder_volume(1.0, 0.0))
    write_zarr(root / "ladder2.zarr", ladder_volume(1.0, 0.0), levels=2)
    xs, ys, zs, valid = ladder_mesh()
    write_mesh(None, root / "label", xs, ys, zs, valid, {}, "synthetic")
    out = root / "out"
    code = cli.main(["--merged", str(root / "label"), "--volume", str(root / "ladder.zarr"), "--out", str(out),
                     "--axis-point", "20", "12", "-400", "--axis-dir", "1", "0", "0", "--self-check",
                     "--ignore-halo", "2"])
    return root, out, code


def test_cli_writes_every_product_and_its_check_passes(cli_run):
    root, out, code = cli_run
    assert code == 0
    rec = json.loads((out / "snap_labels.json").read_text())
    assert rec["opts"]["max_thick_ratio"] == 1.8 and "self_check" in rec
    assert rec["vertices"]["abstained"] > 0 and rec["review"]["regions"] > 0
    assert rec["label"]["voxels"]["surface"] > 0 and rec["label"]["voxels"]["ignore"] > 0
    assert L.check_outputs(out) == []
    xs, ys, zs, valid = load_mesh(out / "snapped.tifxyz")
    assert xs.shape == (9, LADDER_W * len(GAPS))
    assert json.loads((out / "snapped.tifxyz" / "meta.json").read_text())["snap"]["mode"] == "face"


def test_check_catches_a_tampered_mask_and_a_missing_file(cli_run):
    root, out, _ = cli_run
    bad = root / "bad"
    shutil.copytree(out, bad)
    ab = read_grid_tif(bad / "abstain.tif").astype(np.uint8)
    ab[0, 0] = L.ABSTAIN_YES if ab[0, 0] == L.ABSTAIN_OK else L.ABSTAIN_OK
    write_grid_tif(bad / "abstain.tif", ab)
    assert any("disagree" in p for p in L.check_outputs(bad))
    (bad / "review.json").unlink()
    assert any("missing" in p for p in L.check_outputs(bad))


def test_label_level_one_halves_the_coordinates(cli_run):
    root, _, _ = cli_run
    out = root / "out_l1"
    code = cli.main(["--merged", str(root / "label"), "--volume", str(root / "ladder2.zarr"), "--out", str(out),
                     "--axis-point", "20", "12", "-400", "--axis-dir", "1", "0", "0", "--label-level", "1",
                     "--ignore-halo", "2", "--json"])
    assert code == 0
    lb = json.loads((out / "snap_labels.json").read_text())["label"]
    assert lb["shape_zyx"] == Volume(root / "ladder2.zarr").shape(1) and lb["samples_outside_volume"] == 0
    attrs = json.loads((out / "surface_label.zarr" / ".zattrs").read_text())
    assert attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [2.0, 2.0, 2.0]
    _, blocks = L.read_label_chunks(out / "surface_label.zarr")
    zz = np.concatenate([np.nonzero(b == L.LABEL_SURFACE)[0] + cz * L.LABEL_CHUNK for (cz, _, _), b in blocks.items()])
    assert len(zz) and ((zz >= 13) & (zz <= 14)).all()          # recto face at level-0 z 27.5 is level-1 z 13.75


def test_cli_window16_options_reach_the_record(cli_run):
    root, _, _ = cli_run
    out = root / "out_w16"
    code = cli.main([str(root / "label"), "--volume", str(root / "ladder.zarr"), "--out", str(out), "--recto-sign",
                     "1", "--half", "16", "--fallback-half", "12", "--min-contrast", "0", "--no-label"])
    assert code == 0
    opts = json.loads((out / "snap_labels.json").read_text())["opts"]
    assert {k: opts[k] for k in F.FACE_SNAP_WINDOW16} == F.FACE_SNAP_WINDOW16
    assert cli.main([str(root / "label"), "--volume", str(root / "ladder.zarr"), "--out", str(root / "out_bad"),
                     "--recto-sign", "1", "--fallback-half", "12"]) == 2        # the fallback must be shorter


def test_refusals(cli_run, capsys):
    root, out, _ = cli_run
    assert cli.main([str(root / "label"), "--volume", str(root / "ladder.zarr"), "--out", str(out),
                     "--recto-sign", "1"]) == 2
    assert "not empty" in capsys.readouterr().err
    assert cli.main([str(root / "label"), "--volume", str(root / "ladder.zarr"), "--out", str(root / "fresh")]) == 2
    assert "recto face is named" in capsys.readouterr().err
