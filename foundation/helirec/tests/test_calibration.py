"""helirec::calibration through its bindings: the registration estimator, and the record and inventory checks on
a small synthetic acquisition record written the way BLISS / Lima write one (virtual frame dataset, one chunk
per frame, per-frame statistics, latched encoders)."""
import json

import numpy as np
import pytest

from helirec import _core

h5py = pytest.importorskip("h5py")

ROWS, COLS = 24, 40
PER_TURN = 16
DEG_PER_COUNT = -9e-5  # the encoder counts down while the angle goes up


def test_selftest():
    assert _core.calibration_selftest()


def test_selftest_cuda():
    if not _core.cuda_available():
        pytest.skip("no CUDA device")
    assert _core.calibration_selftest(device="cuda")


def test_register_blocks_fractional_shift():
    rng = np.random.default_rng(3)
    shape = (40, 160, 320)
    # a smooth periodic field, displaced exactly by a phase ramp
    f = [np.fft.fftfreq(n) for n in shape]
    k = np.meshgrid(*f, indexing="ij")
    spectrum = np.fft.fftn(rng.standard_normal(shape)) * np.exp(-((k[0] / 0.12) ** 2 + (k[1] / 0.08) ** 2 + (k[2] / 0.08) ** 2))
    d = (0.31, -0.42, 0.27)
    field = np.fft.ifftn(spectrum).real
    moved = np.fft.ifftn(spectrum * np.exp(-2j * np.pi * (k[0] * d[0] + k[1] * d[1] + k[2] * d[2]))).real  # moved(x) = field(x - d)
    scale = 1.0 / field.std()
    m = (8, 12, 8)
    a = np.ascontiguousarray(field[12:28, 32:128, 40:280] * scale, dtype=np.float32)
    b = np.ascontiguousarray(moved[12 - m[0]:28 + m[0], 32 - m[1]:128 + m[1], 40 - m[2]:280 + m[2]] * scale, dtype=np.float32)
    a += rng.normal(0, 0.02, a.shape).astype(np.float32)
    b += rng.normal(0, 0.02, b.shape).astype(np.float32)
    r = _core.calibration_register_blocks(a, b, list(m))
    assert r["converged"] and r["whole"] == (0, 0, 0)
    assert np.allclose(r["shift"], d, atol=0.01), r
    # the CUDA registration is the same estimator on the same arithmetic: it differs by float rounding only
    if _core.cuda_available():
        g = _core.calibration_register_blocks(a, b, list(m), device="cuda")
        assert g["converged"] and g["evaluations"] == r["evaluations"]
        assert np.allclose(g["shift"], r["shift"], atol=1e-4), (g, r)
        assert abs(g["correlation"] - r["correlation"]) < 1e-5
        # with a displacement of the rows per frame removed first (the helix against a turn at rest)
        ramp = [0.07 * (q - b.shape[0] / 2) for q in range(b.shape[0])]
        rc = _core.calibration_register_blocks(a, b, list(m), row_shift=ramp)
        rg = _core.calibration_register_blocks(a, b, list(m), row_shift=ramp, device="cuda")
        assert np.allclose(rg["shift"], rc["shift"], atol=1e-4), (rg, rc)


def _frames(rng, n, level):
    return (level + rng.integers(0, 50, size=(n, ROWS, COLS))).astype(np.uint16)


def _write_record(tmp_path, rng):
    """one scan: darks, flats, a turn at rest, a helix of two turns, a turn at rest, flats"""
    master = tmp_path / "sample_XX_0001.h5"
    stretches = [  # entry, title, image_key (None: absent), type, frames, level
        ("2.1", "dark images", 2, "tomo:dark", 4, 290),
        ("3.1", "flat images 1", 1, "tomo:flat", 4, 20000),
        ("4.1", "projections 0 - 16", 0, "tomo:continuous", PER_TURN, 15000),
        ("5.1", "projections 0 - 32", None, "tomo:continuous", 2 * PER_TURN, 15000),
        ("6.1", "projections 0 - 16", 0, "tomo:continuous", PER_TURN, 15000),
        ("7.1", "flat images 16", 1, "tomo:flat", 4, 20000),
    ]
    step = 360.0 / PER_TURN
    with h5py.File(master, "w") as f:
        t = f.create_group("1.1/technique")
        f["1.1/title"] = "tomo:helical"
        f["1.1/sample/name"] = "sample_XX"
        f["1.1/start_time"] = "2025-01-01T00:00:00"
        f["1.1/end_time"] = "2025-01-01T00:10:00"
        f["1.1/end_reason"] = "SUCCESS"
        for key, value in {"dark_n": 4, "flat_n": 4, "tomo_n": PER_TURN, "scan_range": 360.0, "nb_turns": 4, "exposure_time": 30.0, "latency_time": 3.0,
                           "sample_pixel_size": 8.0, "source_sample_distance": 100000.0, "sample_detector_distance": 1000.0,
                           "effective_propagation_distance": 990.1, "energy": 100.0, "z_step": -1.6, "sequence": "tomo:helical"}.items():
            t["scan/" + key] = value
        t["optic/optics_pixel_size"] = 8.0 * 101000.0 / 100000.0
        t["detector/det3/binning"] = [2, 2]
        t["detector/det3/roi"] = [0, 0, COLS, ROWS]
        t["detector/det3/flipping"] = np.array([False, True])
        t["detector/det3/size"] = [COLS, ROWS]
        t["detector/det3/pixel_size"] = [4e-6, 4e-6]
        t["detector/det3/depth"] = [2]
        t["detector/det3/type"] = "TEST"
        t["tomoconfig/detector"] = ["det3"]
        t["tomoconfig/rotation"] = ["mrsrot", "srot"]
        t["tomoconfig/translation_z"] = ["mrsz", "sz"]
        t["tomoconfig/translation_y"] = ["mryrot", "yrot"]
        t["tomoconfig/rotation_is_clockwise"] = np.True_
        f["1.1/instrument/det3_optic/data_axes"] = ["z", "-y"]
        angle0, z0, clock = 0.0, -10.0, 1000.0
        for k, (entry, title, key, kind, n, level) in enumerate(stretches):
            t[f"subscans/scan{k}/type"] = kind
            data = _frames(rng, n, level)
            scan_dir = tmp_path / f"scan{k + 2:04d}"
            scan_dir.mkdir()
            with h5py.File(scan_dir / "det3_0000.h5", "w") as g:
                g.create_dataset("/entry_0000/instrument/det3/data", data=data, chunks=(1, ROWS, COLS))
                g["/entry_0000/instrument/det3/acquisition/nb_frames"] = np.int32(n)
            e = f.create_group(entry)
            e["title"] = title
            e["start_time"] = "2025-01-01T00:00:00"
            e["end_time"] = "2025-01-01T00:01:00"
            e["end_reason"] = "SUCCESS"
            if key is not None:
                e["technique/image_key"] = key
            layout = h5py.VirtualLayout(shape=(n, ROWS, COLS), dtype=np.uint16)
            layout[:] = h5py.VirtualSource(f"scan{k + 2:04d}/det3_0000.h5", "/entry_0000/instrument/det3/data", shape=(n, ROWS, COLS))
            i = e.create_group("instrument")
            i.create_virtual_dataset("det3/image", layout)
            # statistics over the middle rows, as the detector software records them
            y, h = 6, 12
            roi = data[:, y:y + h, :].astype(np.float64)
            i["det3_full/sum"] = roi.sum(axis=(1, 2))
            i["det3_full/avg"] = roi.mean(axis=(1, 2))
            i["det3_full/min"] = roi.min(axis=(1, 2))
            i["det3_full/max"] = roi.max(axis=(1, 2))
            i["det3_full/std"] = roi.std(axis=(1, 2))
            for name, value in {"x": 0, "y": y, "width": COLS, "height": h}.items():
                i["det3_full/selection/" + name] = value
            helix = entry == "5.1"
            z_end = z0 - 2 * 1.6 if helix else z0
            for where, z in (("positioners_start", z0), ("positioners_end", z_end)):
                i[where + "/sz"] = z
                i[where + "/yrot"] = 1.0
            i["machine/current"] = 80.0
            if kind == "tomo:continuous":
                index = np.arange(n)
                edges = np.empty(2 * n + 1)
                edges[0::2] = np.arange(n + 1) * 33000  # start of each gate [us]
                edges[1::2] = index * 33000 + 30000
                i["timer_raw/data"] = edges.astype(np.int32)
                i["timer_trig/data"] = index * 0.033
                i["timer_delta/data"] = np.full(n, 0.030)
                i["timer_period/data"] = np.full(n, 0.033)
                i["epoch_trig/data"] = clock + index * 0.033
                deg = angle0 + np.arange(2 * n + 1) * 0.0  # angle at every edge
                deg[0::2] = angle0 + np.arange(n + 1) * step
                deg[1::2] = angle0 + index * step + step * 30 / 33
                raw = np.round(deg / DEG_PER_COUNT).astype(np.int32)
                trig = raw[0:-1:2] * DEG_PER_COUNT
                delta = (raw[1::2] - raw[0:-1:2]) * DEG_PER_COUNT
                i["mrsrot_raw/data"] = raw
                i["mrsrot_trig/data"] = trig
                i["mrsrot_delta/data"] = delta
                i["mrsrot_period/data"] = np.diff(raw[0::2]) * DEG_PER_COUNT
                i["mrsrot_cen360/data" if helix else "mrsrot_center/data"] = (trig + delta / 2) % 360 if helix else trig + delta / 2
                i["fscan_parameters/npoints"] = n
                i["fscan_parameters/step_size"] = step
                if helix:
                    zr = np.round((z0 - 1.6 * (deg - angle0) / 360.0) / 0.0002).astype(np.int32)
                    i["mrsz_raw/data"] = zr
                    i["mrsz_trig/data"] = zr[0:-1:2] * 0.0002
                    i["mrsz_delta/data"] = (zr[1::2] - zr[0:-1:2]) * 0.0002
                    i["mrsz_center/data"] = (zr[0:-1:2] + zr[1::2]) * 0.0001
                    i["mrsz_period/data"] = np.diff(zr[0::2]) * 0.0002
                    i["fscan_parameters/slave_step_size"] = -1.6 / PER_TURN
                # the monitors of N.1 are in N.2
                major = entry.split(".")[0]
                mon = f.create_group(major + ".2/instrument")
                mon["current/signal"] = np.linspace(80.0, 79.9, 5)
                mon["epoch/value"] = clock + np.linspace(0, n * 0.033, 5)
                angle0 += n * step
                clock += n * 0.033 + 1.0
            else:
                i["epoch/value"] = clock + np.arange(n) * 0.1
                i["elapsed_time/value"] = np.arange(n) * 0.1
                i["current/signal"] = np.full(n, 80.0)
                clock += n * 0.1 + 1.0
            z0 = z_end
    return master


def _by_quantity(report, text):
    found = [f for f in report["findings"] if text in f["quantity"]]
    assert found, text
    return found


def test_record_and_inventory(tmp_path):
    rng = np.random.default_rng(1)
    master = _write_record(tmp_path, rng)
    report = json.loads(_core.calibration_verify([str(master)], checks="record,inventory"))
    for text in ("frame counts of the stretches", "map from frame to file", "hardware timer", "role of each stretch", "every frame on disk"):
        for f in _by_quantity(report, text):
            assert f["verdict"] == "verified" and not f["requires_follow_up"], f
    # the derived angles and heights are the raw counts; the travel per turn is the commanded one
    for f in _by_quantity(report, "rotation angle of each projection"):
        assert f["verdict"] == "verified", f
    travel = _by_quantity(report, "travel along the axis per turn")[0]["numbers"]
    assert abs(travel["mm_per_turn"] + 1.6) < 1e-3
    # the helix has no image_key in the master file: the report says so
    assert "image_key absent" in _by_quantity(report, "role of each stretch")[0]["found"]

    # one altered value in one frame: the inventory finds the frame
    with h5py.File(tmp_path / "scan0005" / "det3_0000.h5", "r+") as g:
        g["/entry_0000/instrument/det3/data"][7, 10, 10] += 1
    report = json.loads(_core.calibration_verify([str(master)], checks="inventory"))
    f = _by_quantity(report, "every frame on disk")[0]
    assert f["verdict"] == "fails" and f["numbers"]["mismatched"] == 1 and "5.1 frame 7" in f["found"]


@pytest.mark.parametrize("values,target,exceed", [
    ([-0.2, 0.2], 0.1, 2),  # cancellation in the mean must not hide drift
    ([0.1001, 0.0, 0.0], 0.1, 1),  # failure despite large SEM / a 3-SEM interval including zero
    ([0.011], 0.01, 1),  # a single observed discrepancy must not be silently dropped
    ([float('nan'), 0.02], 0.01, 1),
])
def test_residual_failures_are_not_discounted(values, target, exceed):
    report = json.loads(_core.calibration_assess_residuals(values, target))
    assert report['position_target_px'] == target
    f = report['findings'][0]
    assert f['verdict'] == 'fails' and f['requires_follow_up']
    assert f['residuals']['above_target'] == exceed
    assert f['residuals']['max_abs'] > target


@pytest.mark.parametrize("values", [[0.001, -0.002], [0.0, 0.0], [], [float('nan')]])
def test_small_or_missing_residuals_do_not_certify_geometry(values):
    f = json.loads(_core.calibration_assess_residuals(values))['findings'][0]
    r = f['residuals']
    assert f['verdict'] == 'unverified' and f['requires_follow_up']
    assert r['target'] == 0.01 and r['above_target'] == 0
    assert r['samples'] == sum(np.isfinite(x) for x in values)
    assert r['invalid'] == sum(not np.isfinite(x) for x in values)
    assert r['nonzero'] == sum(np.isfinite(x) and x != 0 for x in values)
    if values == [0.001, -0.002]:
        assert r['min'] == -0.002 and r['max'] == 0.001
        assert r['rms'] == pytest.approx(np.sqrt(2.5e-6))


@pytest.mark.parametrize("target", [0, -0.01, float('nan'), float('inf')])
def test_invalid_position_target(target):
    with pytest.raises(RuntimeError, match='finite and positive'):
        _core.calibration_assess_residuals([0.1], target)


def test_position_target_controls_record_checks(tmp_path):
    master = _write_record(tmp_path, np.random.default_rng(2))
    with h5py.File(master, 'r+') as f:
        f['1.1/technique/scan/z_step'][...] = -1.6 + 0.008 * 0.2
    loose = json.loads(_core.calibration_verify([str(master)], checks='record', position_target=0.1))
    tight = json.loads(_core.calibration_verify([str(master)], checks='record'))
    assert tight['position_target_px'] == 0.01
    assert loose['position_target_px'] == 0.1
    a = _by_quantity(loose, 'travel along the axis per turn')[0]
    b = _by_quantity(tight, 'travel along the axis per turn')[0]
    assert a['verdict'] == 'unverified' and a['requires_follow_up']
    assert b['verdict'] == 'fails' and b['requires_follow_up']
    assert a['residuals']['max_abs'] == b['residuals']['max_abs']


def test_registrar_matches_register_blocks():
    rng = np.random.default_rng(5)
    shape = (24, 64, 96)
    k = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    spectrum = np.fft.fftn(rng.standard_normal(shape)) * np.exp(-((k[0] / 0.12) ** 2 + (k[1] / 0.08) ** 2 + (k[2] / 0.08) ** 2))
    d = (0.2, -0.3, 0.4)
    field = np.fft.ifftn(spectrum).real
    moved = np.fft.ifftn(spectrum * np.exp(-2j * np.pi * (k[0] * d[0] + k[1] * d[1] + k[2] * d[2]))).real
    scale = 1.0 / field.std()
    m = (4, 8, 8)
    a = np.ascontiguousarray(field[4:20, 8:56, 8:88] * scale, dtype=np.float32)
    b = np.ascontiguousarray(moved * scale, dtype=np.float32)
    once = _core.calibration_register_blocks(a, b, list(m))
    kept = _core.CalibrationRegistrar(list(a.shape), list(m))
    for _ in range(2):  # the same registrar serves many pairs
        r = kept.run(a, b)
        # the same estimator: equal up to the rounding of sums taken by several threads
        assert np.allclose(r["shift"], once["shift"], atol=1e-9) and r["evaluations"] == once["evaluations"]
    assert np.allclose(r["shift"], d, atol=0.02), r
    # one frame against the same frame displaced along rows and columns: the frame axis is not registered
    in_frame = np.fft.ifftn(spectrum * np.exp(-2j * np.pi * (k[1] * d[1] + k[2] * d[2]))).real
    flat = _core.CalibrationRegistrar([1, 48, 80], [0, 8, 8])
    r2 = flat.run(np.ascontiguousarray(a[8:9]), np.ascontiguousarray(in_frame[12:13] * scale, dtype=np.float32))
    assert r2["shift"][0] == 0 and np.allclose(r2["shift"][1:], d[1:], atol=0.02), r2
    with pytest.raises(Exception):
        kept.run(a, a)


def test_session_reads_and_measures(tmp_path, monkeypatch):
    """CalibrationSession: the record's series, the raw frames, and a comparison placed by the caller, cut and
    registered as the view checks do it (here a block against itself: no displacement, correlation 1)."""
    import sys
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROWS", 48)
    monkeypatch.setattr(module, "COLS", 80)
    monkeypatch.setattr(module, "PER_TURN", 40)
    master = _write_record(tmp_path, np.random.default_rng(11))
    s = _core.CalibrationSession([str(master)])
    info = s.info(0)
    assert (info["rows"], info["cols"], info["translation_z"], info["pixel_um"]) == (48, 80, "sz", 8.0)
    st = s.stretches(0)
    assert [(x["entry"], x["role"], x["n_frames"]) for x in st] == [("2.1", "dark", 4), ("3.1", "flat", 4), ("4.1", "projection", 40), ("5.1", "projection", 80),
                                                                    ("6.1", "projection", 40), ("7.1", "flat", 4)]
    assert [x["helical"] for x in st] == [False, False, False, True, False, False]
    angle = s.series(0, 2, "angle_deg")
    assert angle.shape == (40,) and np.allclose(np.diff(angle), 9.0, atol=1e-3)
    assert s.series(0, 3, "height_mm").shape == (80,) and np.all(s.series(0, 2, "height_mm") == -10.0)
    assert s.positioner(0, 3, "sz", True) == pytest.approx(-13.2)
    with pytest.raises(Exception):
        s.series(0, 2, "no such series")
    # raw frames as stored
    with h5py.File(tmp_path / "scan0004" / "det3_0000.h5") as f:
        stored = f["/entry_0000/instrument/det3/data"][:]
    frames = np.array([3, 17, 39], dtype=np.int64)
    assert np.array_equal(s.read_rows(0, 2, frames, 5, 20), stored[frames, 5:25])
    dark, gain = s.dark(0), s.gain(0)
    assert dark.shape == gain.shape == (48, 80) and 290 <= dark.mean() <= 340 and gain.mean() > 19000
    p = s.line_integrals(0, 2, frames, np.array([5, 5, 6], dtype=np.int32), 20, 8, 32)
    want = -np.log(np.maximum(1e-6, (stored[frames[2], 6:26, 8:40].astype(np.float32) - dark[6:26, 8:40]) / gain[6:26, 8:40]))
    assert p.shape == (3, 20, 32) and np.allclose(p[2], want, atol=1e-5)
    assert np.array_equal(s.line_integrals(0, 2, frames, np.array([5, 5, 6], dtype=np.int32), 20, 8, 32, True), p[:, :, ::-1])
    # a block of the rest turn against itself, and a request the record cannot hold
    res = s.measure([dict(label="self", stretch_a=2, frame_a=12, frames=16, rows=16, cols=32, row_a=14, col_a=20, col_b=20),
                     dict(label="outside", stretch_a=2, frame_a=30, frames=16, rows=16, cols=32, row_a=14, col_a=20, col_b=20)], -1.0, "cpu")
    assert [r["label"] for r in res] == ["self", "outside"]
    me = res[0]
    assert me["planned"] and me["ok"] and (me["frame_a"], me["frame_b"], me["frames"], me["rows"], me["cols"], me["row_offset"]) == (12, 12, 16, 16, 32, 0)
    assert abs(me["dq"]) < 1e-3 and abs(me["D"]) < 1e-3 and abs(me["du"]) < 1e-3 and me["correlation"] > 0.999 and me["stated_dq"] == 0
    assert not res[1]["planned"] and not res[1]["ok"]
    with pytest.raises(Exception):
        s.measure([dict(stretch_a=2, frame_a=12)], 0.5, "cpu")


def test_session_detector_map(tmp_path, monkeypatch):
    """A detector map handed to the session: blocks are cut on its ideal grid by cubic convolution, B's frames can be
    sheared along the columns, and without a map nothing changes."""
    import sys
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROWS", 48)
    monkeypatch.setattr(module, "COLS", 80)
    monkeypatch.setattr(module, "PER_TURN", 40)
    master = _write_record(tmp_path, np.random.default_rng(12))
    s = _core.CalibrationSession([str(master)])
    frames = np.array([3, 17, 39], dtype=np.int64)
    plain = s.line_integrals(0, 2, frames, np.full(3, 0, np.int32), 48, 0, 80)
    # no map: the same samples, also flipped
    assert np.array_equal(s.mapped_line_integrals(0, 2, frames, 6, 20, 8, 32), plain[:, 6:26, 8:40])
    assert np.array_equal(s.mapped_line_integrals(0, 2, frames, 6, 20, 8, 32, True), plain[:, 6:26, 8:40][:, :, ::-1])
    # a map that is zero everywhere reads every sample where it is
    s.set_detector_map([1], [1], [0.0], [0.0], 40.0, 40.0, 24.0, 24.0)
    assert np.allclose(s.mapped_line_integrals(0, 2, frames, 6, 20, 8, 32), plain[:, 6:26, 8:40], atol=1e-6)
    # whole pixels: the pixel one column lower and two rows higher looks at this ideal position
    s.set_detector_map([0], [0], [1.0], [-2.0])
    assert np.allclose(s.mapped_line_integrals(0, 2, frames, 6, 20, 8, 32), plain[:, 8:28, 7:39], atol=1e-6)
    assert np.allclose(s.mapped_line_integrals(0, 2, frames, 6, 20, 8, 32, True), plain[:, 8:28, 7:39][:, :, ::-1], atol=1e-6)
    # half a column: the cubic convolution's weights (-1, 9, 9, -1) / 16
    s.set_detector_map([0], [0], [0.5], [0.0])
    want = (-plain[:, 6:26, 6:38] + 9 * plain[:, 6:26, 7:39] + 9 * plain[:, 6:26, 8:40] - plain[:, 6:26, 9:41]) / 16
    assert np.allclose(s.mapped_line_integrals(0, 2, frames, 6, 20, 8, 32), want, atol=1e-5)
    # a map that varies: every sample is read from where the map's inverse puts it (checked against the forward map)
    s.set_detector_map([1, 2], [1, 0], [0.8, -0.3], [0.2, 0.5], 40.0, 40.0, 24.0, 24.0)
    got = s.mapped_line_integrals(0, 2, frames[:1], 10, 24, 12, 48)[0]
    rr, cc = np.mgrid[0:48, 0:80].astype(float)
    x, y = (cc - 40) / 40, (rr - 24) / 24
    ideal_c, ideal_r = cc + 0.8 * x * y - 0.3 * x * x, rr + 0.2 * x * y + 0.5 * x * x
    # a field linear in the ideal coordinates is linear on the detector to second order: compare a smooth field
    smooth = np.sin(ideal_c / 9.0) + np.cos(ideal_r / 7.0)

    def keys(f):
        return np.array([((-0.5 * f + 1) * f - 0.5) * f, (1.5 * f - 2.5) * f * f + 1, ((-1.5 * f + 2) * f + 0.5) * f, (0.5 * f - 0.5) * f * f])

    for r, c in ((0, 0), (5, 9), (23, 47), (11, 30)):
        yi, xi = 10 + r, 12 + c
        cs_, rs_ = float(xi), float(yi)
        for _ in range(3):
            xn, yn = (cs_ - 40) / 40, (rs_ - 24) / 24
            cs_, rs_ = xi - (0.8 * xn * yn - 0.3 * xn * xn), yi - (0.2 * xn * yn + 0.5 * xn * xn)
        r0, c0 = int(np.floor(rs_)), int(np.floor(cs_))
        wr, wc = keys(rs_ - r0), keys(cs_ - c0)
        assert abs(wr @ plain[0, r0 - 1:r0 + 3, c0 - 1:c0 + 3] @ wc - got[r, c]) < 1e-5
        assert abs(wr @ smooth[r0 - 1:r0 + 3, c0 - 1:c0 + 3] @ wc - (np.sin(xi / 9.0) + np.cos(yi / 7.0))) < 2e-3
    # frames sheared along the columns: the middle column stays, the others are read between frames
    s.set_detector_map([], [], [], [])
    run = np.arange(8, 24, dtype=np.int64)
    block = s.line_integrals(0, 2, run, np.full(16, 6, np.int32), 20, 8, 33)
    sheared = s.mapped_line_integrals(0, 2, run, 6, 20, 8, 33, False, 0.25)
    assert np.allclose(sheared[:, :, 16], block[:, :, 16], atol=1e-6)
    assert np.allclose(sheared[4:12, :, 20], block[5:13, :, 20], atol=1e-6)       # four columns on: one frame later
    assert np.allclose(sheared[4:12, :, 18], (-block[3:11, :, 18] + 9 * block[4:12, :, 18] + 9 * block[5:13, :, 18] - block[6:14, :, 18]) / 16, atol=1e-5)
    # a block against itself is still itself with a map, and without one the result is what it was
    req = [dict(label="self", stretch_a=2, frame_a=12, frames=16, rows=16, cols=32, row_a=14, col_a=20, col_b=20)]
    before = s.measure(req, -1.0, "cpu")[0]
    s.set_detector_map([1, 2], [1, 0], [0.8, -0.3], [0.2, 0.5], 40.0, 40.0, 24.0, 24.0)
    with_map = s.measure(req, -1.0, "cpu")[0]
    assert with_map["ok"] and abs(with_map["dq"]) < 1e-3 and abs(with_map["D"]) < 1e-3 and abs(with_map["du"]) < 1e-3 and with_map["correlation"] > 0.999
    s.set_detector_map([], [], [], [])
    after = s.measure(req, -1.0, "cpu")[0]
    assert all(after[k] == pytest.approx(before[k], abs=1e-9) for k in before if k != "label")
    with pytest.raises(Exception):
        s.set_detector_map([1], [1, 2], [0.0], [0.0])


def test_session_detector_map_per_scan(tmp_path, monkeypatch):
    """A map given to one scan of a session alone: the other scan's blocks stay as stored, and a comparison of the
    two finds the displacement the map puts between them."""
    import sys
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROWS", 48)
    monkeypatch.setattr(module, "COLS", 80)
    monkeypatch.setattr(module, "PER_TURN", 40)
    master = _write_record(tmp_path, np.random.default_rng(13))
    s = _core.CalibrationSession([str(master), str(master)])   # the same scan twice: its two copies hold the same lines
    frames = np.array([3, 17, 39], dtype=np.int64)
    plain = s.line_integrals(0, 2, frames, np.full(3, 0, np.int32), 48, 0, 80)
    s.set_detector_map([0], [0], [2.0], [-1.0], scan=1)       # scan 1 alone: read two columns lower and one row higher
    assert np.array_equal(s.mapped_line_integrals(0, 2, frames, 6, 20, 8, 32), plain[:, 6:26, 8:40])
    assert np.allclose(s.mapped_line_integrals(1, 2, frames, 6, 20, 8, 32), plain[:, 7:27, 6:38], atol=1e-6)
    req = [dict(label="pair", scan_a=0, scan_b=1, stretch_a=2, frame_a=12, frames=16, rows=16, cols=32, row_a=14, col_a=20, col_b=20)]
    r = s.measure(req, -1.0, "cpu")[0]
    assert r["ok"] and r["correlation"] > 0.999 and abs(r["dq"]) < 1e-3
    assert abs(r["du"] - 2.0) < 1e-3 and abs(r["D"] + 1.0) < 1e-3, r      # B's content lies where the map moved it from
    s.set_detector_map([0], [0], [2.0], [-1.0])               # every scan: nothing between them again
    r = s.measure(req, -1.0, "cpu")[0]
    assert abs(r["du"]) < 1e-3 and abs(r["D"]) < 1e-3
    s.set_detector_map([], [], [], [])
    r = s.measure(req, -1.0, "cpu")[0]
    assert abs(r["du"]) < 1e-3 and abs(r["D"]) < 1e-3
