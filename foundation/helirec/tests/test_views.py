"""angle_interp="views": a sinogram row per view, so that the views of a slice that share a grid angle (a turn apart,
where the detector is taller than the travel per turn) are each filtered and backprojected at their own angle instead
of being averaged at their mean. The sinogram kernels for rows over more than one turn (rebalance_turns, the arc
filter and the ramp's padding without the wrap to one turn), and the pipeline."""
import h5py
import numpy as np
import pytest

import helirec
from helirec import _core

cuda_only = pytest.mark.skipif(not helirec.cuda_available(), reason="no CUDA")


# ------------------------------------------------------------------ kernels
def mirror(w, rot):
    """a row's weights at the opposite ray's positions (rebalance: reflection about rot, here a whole number)"""
    n = w.shape[-1]
    shift = int(round(2 * (rot - (n - 1) / 2)))
    out = np.zeros_like(w)
    c = np.arange(n)
    src = n - 1 - (c - shift)
    ok = (c - shift >= 0) & (c - shift < n)
    out[..., c[ok]] = w[..., src[ok]]
    return out


@pytest.mark.parametrize("device", helirec.devices())
def test_rebalance_turns(device):
    rng = np.random.default_rng(1)
    turn, n_proj, rows, cols, rot = 8, 21, 3, 11, 6.0
    R = rng.standard_normal((n_proj, rows, cols)).astype(np.float32)
    W = rng.uniform(0.2, 1, (n_proj, rows, cols)).astype(np.float32)
    W[:, :, :2] = 0                              # columns nothing records: left as they are
    W[3, :, 5:8] = 0                             # samples without weight: the rows a turn on record them
    W[12] = 0                                    # a row without anything, whose angle rows 4 and 20 record
    W[[5, 13]] = 0                               # an angle nothing records (row 21 would be the third)
    r, w = _core.rebalance_turns(R, W, turn, rot, device=device)
    # the empty angle: its first row takes the mean of the rows of the angles either side, of each angle the row with
    # the largest weight in its mid-row
    peak = W[:, rows // 2].max(1)
    best = lambda angle: max((i for i in range(angle, n_proj, turn) if peak[i] > 0), key=lambda i: (peak[i], -i))
    assert best(4) in (4, 20) and best(6) in (6, 14)
    Rf, Wf = R.copy(), W.copy()
    Rf[5], Wf[5] = 0.5 * (R[best(4)] + R[best(6)]), 0.5 * (W[best(4)] + W[best(6)])
    np.testing.assert_allclose(r[5], Rf[5], rtol=1e-6)
    # samples without weight take the value of the row whole turns away whose weight there is largest
    ref = Rf.copy()
    for i in range(n_proj):
        others = sorted((j for j in range(i % turn, n_proj, turn) if j != i), key=lambda j: (abs(j - i), j < i))
        wbest = np.zeros((rows, cols), np.float32)
        for j in others:
            take = (Wf[i] == 0) & (Wf[j] > wbest)
            ref[i][take] = Rf[j][take]
            wbest = np.where(take, Wf[j], wbest)
    np.testing.assert_allclose(r, ref, rtol=1e-6)
    assert not np.array_equal(r[3, :, 5:8], R[3, :, 5:8]) and not np.array_equal(r[12, :, 2:], R[12, :, 2:])
    np.testing.assert_array_equal(r[13, :, 2:], r[5, :, 2:])            # (its only other turn is the filled row)
    np.testing.assert_array_equal(r[:, :, :2], Rf[:, :, :2])
    # the weights: every recording of a line (rows i + k turn / 2, mirrored for odd k) shares it
    ref = np.zeros_like(Wf)
    for i in range(n_proj):
        B = Wf[i].copy()
        for k in range(-6, 7):
            j = i + k * turn // 2
            if k != 0 and 0 <= j < n_proj:
                B += mirror(Wf[j], rot) if k % 2 else Wf[j]
        ref[i] = np.where(Wf[i] == 0, 0, Wf[i] / np.where(B == 0, 1, B))
    np.testing.assert_allclose(w, ref, rtol=2e-6, atol=1e-7)
    # ... a partition of unity over them, wherever the line and its opposite are both on the detector
    for i in range(turn // 2):
        total = sum((mirror(w[j], rot) if (j - i) // (turn // 2) % 2 else w[j]) for j in range(i, n_proj, turn // 2))
        both = (Wf[i] > 0) & (mirror(np.ones(cols, np.float32), rot) > 0)
        np.testing.assert_allclose(total[both], 1, rtol=1e-5)
    # one turn of rows is rebalance itself
    one = (rng.standard_normal((turn, rows, cols)).astype(np.float32), rng.uniform(0.2, 1, (turn, rows, cols)).astype(np.float32))
    one[1][2] = 0
    a, b = _core.rebalance_turns(*one, turn, rot, device=device), _core.rebalance(*one, rot, device=device)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-6)
    np.testing.assert_allclose(a[1], b[1], rtol=2e-6)
    for bad in (dict(turn=7), dict(turn=32)):    # an odd turn; fewer rows than a turn
        with pytest.raises(Exception, match="rebalance_turns"):
            _core.rebalance_turns(R, W, bad["turn"], rot, device=device)
    if device == "cuda":
        r0, w0 = _core.rebalance_turns(R, W, turn, rot, device="cpu")
        np.testing.assert_array_equal(r, r0)
        np.testing.assert_allclose(w, w0, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("device", helirec.devices())
def test_ramp_filter_with_named_opposite_rows(device):
    rng = np.random.default_rng(9)
    n, nd, rot = 16, 40, 27.3                    # (an axis off the middle: the rows are padded from their opposite rows)
    s = rng.standard_normal((n, nd)).astype(np.float32)
    ref = _core.ramp_filter(s, rot, device=device)
    same = _core.ramp_filter(s, rot, device=device, opposite=[(i + n // 2) % n for i in range(n)], turn=n)
    np.testing.assert_array_equal(same, ref)
    # one turn and a half, every row with its opposite row named: the rows of the first turn again
    rows = np.concatenate([s, s[:n // 2]])
    opposite = [i + n // 2 if i + n // 2 < len(rows) else i - n // 2 for i in range(len(rows))]
    out = _core.ramp_filter(rows, rot, device=device, opposite=opposite, turn=n)
    np.testing.assert_allclose(out, np.concatenate([ref, ref[:n // 2]]), rtol=1e-5, atol=1e-6)
    with pytest.raises(Exception, match="opposite"):
        _core.ramp_filter(rows, rot, device=device, opposite=opposite)


# ------------------------------------------------------------------ pipeline
NG, ND, NR = 128, 64, 24                         # grid angles, detector columns and rows
BLOBS = [(21.0, 7.0, 1.3, 0.10), (-11.0, 22.0, 1.3, 0.08), (3.0, -2.0, 2.5, 0.05), (-23.0, -9.0, 1.3, 0.09), (9.0, -24.0, 1.3, 0.07)]


def smooth_border(nr=NR, nd=ND, rows=5.0):
    edge = np.minimum(np.arange(nr), nr - 1 - np.arange(nr))
    t = np.clip((edge + 0.5) / rows, 0, 1)
    return np.repeat((0.5 - 0.5 * np.cos(np.pi * t))[:, None], nd, 1).astype(np.float32)


def blob_scan(tmp_path, per_turn, shutter=0.0, travel=15.0, turns=4.0, skip=(), gain=1.0, name="blobs", weight=None, nr=NR):
    """Gaussian blobs, the same at every height, in a helix of `per_turn` projections per turn that travels `travel`
    rows per turn: a slice is seen during nr / travel turns. Row v of a frame was recorded shutter x (v - middle) /
    (nr - 1) steps from the frame's angle. Returns the scan and the slice."""
    p = np.delete(np.arange(int(round(turns * per_turn))), list(skip))
    angles = p * (360.0 / per_turn) + 0.21 * 360 / NG
    z = p * (travel / per_turn)
    coord = np.arange(ND) - (ND - 1) / 2
    v = (np.arange(nr) - (nr - 1) / 2) / max(nr - 1, 1)
    theta = -np.deg2rad(angles[:, None] + shutter * (360.0 / per_turn) * v[None])[:, :, None]
    line = np.zeros((len(p), nr, ND))
    for x0, y0, s, a in BLOBS:
        line += a * np.sqrt(2 * np.pi) * s * np.exp(-.5 * ((coord[None, None] - (x0 * np.cos(theta) + y0 * np.sin(theta))) / s) ** 2)
    raw = np.rint(50000 * gain * np.exp(-line)).astype(np.uint16)
    path = tmp_path / f"{name}_{per_turn}_{shutter}_{travel}_{len(skip)}_{gain}_{nr}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=raw, chunks=(1, nr, ND))
    reader = _core.RawReader([str(path)], "data", nr, ND)
    scan = _core.make_scan_data(angles.astype(np.float64), z.astype(np.float64) * .01, .01, nr, ND, 0, 0., 0., np.full(len(z), .2),
                                np.array([0], np.float32), np.zeros((nr, ND), np.float32), np.full((1, nr, ND), 50000, np.float32),
                                smooth_border(nr) if weight is None else weight, reader, np.zeros(len(z), np.int32),
                                np.arange(len(z), dtype=np.int32))
    yy, xx = np.mgrid[:ND, :ND] - (ND - 1) / 2
    truth = sum(a * np.exp(-.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / s ** 2) for x0, y0, s, a in BLOBS)
    return scan, truth


def recon(scan, device, **kw):
    cfg = dict(device=device, pixel_size_m=.01, rot_center=(ND - 1) / 2, grid_size=ND, n_grid=NG, backprojector="fbp",
               luminosity_repartition=0., filters=[], block_angles=48, energy_kev=100., distance_m=.1, angle_interp="views")
    cfg.update(kw)
    return _core.SliceReconstructor(scan if isinstance(scan, list) else [scan], cfg)


def middle(r):
    lo, hi = r.info["doable"]
    return (lo + hi) // 2


def error(sl, truth):
    return float(np.abs(sl - truth).max())


@pytest.mark.parametrize("device", helirec.devices())
def test_views_are_the_bins_when_the_turns_agree(tmp_path, device):
    """a whole number of projections per turn and no shutter: the views a turn apart are at one angle, and a row per
    view gives what a row per grid angle gives"""
    scan, truth = blob_scan(tmp_path, NG)
    v, n = recon(scan, device), recon(scan, device, angle_interp="nearest")
    assert v.info["angle_interp"] == "views" and n.info["angle_interp"] == "nearest"
    h = middle(v)
    a, b = v.reconstruct(h, h + 3, debug=True), n.reconstruct(h, h + 3, debug=True)
    np.testing.assert_allclose(a["slices"], b["slices"], atol=3e-7)
    assert error(a["slices"][0], truth) < 4e-4
    # the rows: the bins the chunk's projections span (a slice is seen during 24 / 15 turns), each at its bin's angle
    rows = a["sino"].shape[0]
    assert b["sino"].shape[0] == NG and NG * 1.5 < rows < NG * 1.8 and a["filtered"].shape == (3, rows, ND)
    assert a["angles"].shape == (3, rows) and b["first_bin"] == 0
    bins = a["first_bin"] + np.arange(rows)
    np.testing.assert_allclose(np.exp(1j * a["angles"][1]), np.exp(1j * (bins + 0.21) * 2 * np.pi / NG), atol=2e-5)
    # ... holding what the bin of one turn holds, wherever one view alone is in it, and sharing its weight elsewhere
    w = a["weights"][:, 1, ND // 2]
    one = np.array([np.count_nonzero(w[bins % NG == q] > 1e-6) == 1 for q in bins % NG]) & (w > 1e-6)
    assert one.sum() > NG // 4 and (~one & (w > 1e-6)).sum() > NG // 2
    np.testing.assert_allclose(a["sino"][one, 1], b["sino"][bins[one] % NG, 1], atol=1e-6)
    np.testing.assert_allclose(np.bincount(bins % NG, a["wbal"][:, 1, ND // 2 + 3], NG) + np.bincount((bins + NG // 2) % NG, a["wbal"][:, 1, ND // 2 - 3], NG), 1, rtol=1e-5)
    # no dependence on the chunk
    single = np.concatenate([v.reconstruct(k, k + 1) for k in range(h, h + 3)])
    np.testing.assert_allclose(a["slices"], single, atol=2e-6, rtol=2e-5)
    if device == "cuda":
        np.testing.assert_allclose(a["slices"], recon(scan, "cpu").reconstruct(h, h + 3), atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("per_turn, shutter", [(NG + 0.4, 0.0), (NG, -0.6), (NG + 0.27, 0.72)])
def test_views_a_turn_apart_keep_their_angles(tmp_path, device, per_turn, shutter):
    """a turn that is not a whole number of steps, or rows read one after another: the two views of a slice in one
    bin were recorded at different angles (here 0.4 to 0.8 of a step), which one row at their mean blurs"""
    aligned, truth = blob_scan(tmp_path, NG)
    best = error(recon(aligned, device).reconstruct(middle(recon(aligned, device)), middle(recon(aligned, device)) + 1)[0], truth)
    scan, truth = blob_scan(tmp_path, per_turn, shutter)
    err = {}
    for kind in ("nearest", "views"):
        r = recon(scan, device, angle_interp=kind, rolling_shutter=shutter)
        h = middle(r)
        chunk = r.reconstruct(h, h + 2)
        err[kind] = error(chunk[0], truth)
        np.testing.assert_allclose(chunk, np.concatenate([r.reconstruct(k, k + 1) for k in (h, h + 1)]), atol=2e-6, rtol=2e-5)
    assert err["nearest"] > 2 * best, (err, best)                 # the blur of the mean angle
    assert err["views"] < 1.3 * best and err["views"] < 0.5 * err["nearest"], (err, best)
    if device == "cuda":
        r = recon(scan, "cpu", rolling_shutter=shutter)
        np.testing.assert_allclose(chunk, r.reconstruct(h, h + 2), atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("device", helirec.devices())
def test_views_with_the_other_corrections(tmp_path, device):
    scan, truth = blob_scan(tmp_path, NG + 0.4)
    base = recon(scan, device)
    h = middle(base)
    plain = base.reconstruct(h, h + 1)[0]
    # projections that are missing: where another turn records their angle it takes over, elsewhere the angle is
    # filled from its neighbours, as an empty bin is (with the turns at one angle, to the same slice)
    for per_turn in (NG, NG + 0.4):
        for skip in ((250,), tuple(range(200, 215)), tuple(range(60, 400, 37))):
            holes, _ = blob_scan(tmp_path, per_turn, skip=skip)
            r, n = recon(holes, device), recon(holes, device, angle_interp="nearest")
            sl, ref = r.reconstruct(middle(r), middle(r) + 1)[0], n.reconstruct(middle(r), middle(r) + 1)[0]
            assert np.isfinite(sl).all()
            if per_turn == NG:
                np.testing.assert_allclose(sl, ref, atol=5e-7)
                # (in a chunk of several slices the sinogram also has rows of projections that do not see the
                # slice: the angle is filled from the views that do)
                mid = middle(r) - 2
                np.testing.assert_allclose(r.reconstruct(mid, mid + 5)[2], n.reconstruct(mid, mid + 5)[2], atol=5e-7)
            else:
                assert error(sl, truth) < 0.8 * error(ref, truth), (skip, error(sl, truth), error(ref, truth))
    # the exposure arc is not deconvolved with a row per view (disabled for now)
    arc = recon(scan, device, exposure_arc=0.9)
    assert arc.info["exposure_arc_bins"] == 0 and abs(recon(scan, device, exposure_arc=0.9, angle_interp="nearest").info["exposure_arc_bins"] - 0.9) < 0.01
    np.testing.assert_array_equal(arc.reconstruct(h, h + 1)[0], plain)
    # a phase filter with context rows, the band-dependent weights
    for kw in (dict(filters=[{"type": "paganin", "delta_beta": 50.}], pixel_size_m=1e-5),
               dict(sharpness_weighting="0:0:0:0:1.5:0", sharp_band="0.15:0.30")):
        r = recon(scan, device, **kw)
        chunk = r.reconstruct(h, h + 3)
        assert np.isfinite(chunk).all()
        scale = 1.0 / np.abs(chunk).max()
        np.testing.assert_allclose(chunk * scale, np.concatenate([r.reconstruct(k, k + 1) for k in range(h, h + 3)]) * scale, atol=2e-5)
        assert np.abs(chunk[0] * scale - plain / np.abs(plain).max()).max() > 1e-4 or "sharpness_weighting" in kw
        if device == "cuda":
            np.testing.assert_allclose(chunk * scale, recon(scan, "cpu", **kw).reconstruct(h, h + 3) * scale, atol=5e-5)
    # two scans (the luminosity window's context rows)
    companion, _ = blob_scan(tmp_path, NG + 0.4, gain=0.9)
    r = recon([scan, companion], device, luminosity_repartition=100.)
    chunk = r.reconstruct(h, h + 2)
    np.testing.assert_allclose(chunk, np.concatenate([r.reconstruct(k, k + 1) for k in (h, h + 1)]), atol=2e-6, rtol=2e-5)
    main, _ = blob_scan(tmp_path, NG)             # (the turns at one angle: what a row per grid angle gives)
    other, _ = blob_scan(tmp_path, NG, gain=0.9)
    pair = [recon([main, other], device, luminosity_repartition=100., angle_interp=kind) for kind in ("views", "nearest")]
    np.testing.assert_allclose(pair[0].reconstruct(h, h + 1), pair[1].reconstruct(h, h + 1), atol=5e-7)
    # the backprojectors that take one row per grid angle, and the modes that splat, keep theirs
    assert recon(scan, device, backprojector="sirt", tomo_params={"num_iter": "1"}).info["angle_interp"] == "nearest"
    assert recon(scan, device, backprojector="ghbp").info["angle_interp"] == "linear"
    assert recon(scan, device, legacy_reference=True).info["angle_interp"] == "linear"
    assert recon(scan, device, angle_interp="nearest").info["angle_interp"] == "nearest"
    # a detector many times taller than the travel per turn: more turns of rows than the sinogram takes
    many, _ = blob_scan(tmp_path, NG, travel=1.0, turns=22.0)
    r = recon(many, device)
    with pytest.raises(Exception, match="views"):
        r.reconstruct(middle(r), middle(r) + 1)
    assert np.isfinite(recon(many, device, angle_interp="nearest").reconstruct(middle(r), middle(r) + 1)).all()


@pytest.mark.parametrize("device", helirec.devices())
def test_views_in_a_band(tmp_path, device):
    """the band reconstruction samples the slices of a row per view"""
    scan, _ = blob_scan(tmp_path, NG + 0.4)
    r = recon(scan, device)
    h = middle(r)
    vol = r.reconstruct(h, h + 2)
    yy, xx = np.mgrid[10:54:3, 10:54:3]
    y, x = yy.ravel().astype(np.float32), xx.ravel().astype(np.float32)
    zlo = np.full(len(y), h, np.int32)
    band = _core.BandSampler(zlo, np.zeros(len(y), np.float32), y, x, ND, device=device)
    r.reconstruct_band(h, h + 2, band)
    np.testing.assert_allclose(band.values(), vol[0][yy.ravel(), xx.ravel()], atol=2e-6, rtol=2e-5)
    if device != "cuda":
        return
    # a sweep of filter chains shares the gridding and the weights
    chains = [[{"type": "paganin", "delta_beta": 50.}], [{"type": "paganin", "delta_beta": 200.}]]
    kw = dict(pixel_size_m=1e-5)
    sweep = _core.BandSampler(zlo, np.zeros(len(y), np.float32), y, x, ND, device=device, n_sets=2)
    recon(scan, device, filters=chains[0], **kw).reconstruct_band(h, h + 2, sweep, sweep=chains)
    for k, chain in enumerate(chains):
        ref = recon(scan, device, filters=chain, **kw).reconstruct(h, h + 2)[0][yy.ravel(), xx.ravel()]
        np.testing.assert_allclose(sweep.values(k), ref, atol=2e-5 * np.abs(ref).max(), rtol=2e-4)
