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


def _write_frames(path, data, chunk_rows=None):
    """One Lima frame file. chunk_rows: the frames stored as JPEG 2000, one codestream per chunk of that many rows
    through the blosc2 filter (id 32026), as ESRF BM18 writes projections; otherwise one plain chunk per frame."""
    n, rows, cols = data.shape
    with h5py.File(path, "w") as g:
        g["/entry_0000/instrument/det3/acquisition/nb_frames"] = np.int32(n)
        if chunk_rows is None:
            g.create_dataset("/entry_0000/instrument/det3/data", data=data, chunks=(1, rows, cols))
            return
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((1, chunk_rows, cols))
        dcpl.set_filter(32026, h5py.h5z.FLAG_OPTIONAL, (1, 0, 2, chunk_rows * cols * 2, 5, 1, 0, 3, 1, chunk_rows, cols))
        d = h5py.h5d.create(g["/entry_0000/instrument/det3"].id, b"data", h5py.h5t.STD_U16LE, h5py.h5s.create_simple(data.shape), dcpl)
        for k in range(n):
            for r0 in range(0, rows, chunk_rows):
                tile = np.zeros((chunk_rows, cols), np.uint16)      # the last chunk reaches beyond the frame
                piece = data[k, r0:r0 + chunk_rows]
                tile[:piece.shape[0]] = piece
                d.write_direct_chunk((k, r0, 0), _core.jpeg2000_chunk(tile))


def _write_record(tmp_path, rng, name="sample_XX", place=None, compressed=(), frames=None):
    """one scan: darks, flats, a turn at rest, a helix of two turns, a turn at rest, flats.
    place: (sx, sy) [mm], the sample's place on the rotation stage (a tile of a mosaic); compressed: the entries
    whose frames are stored as JPEG 2000; frames: {entry: (n, ROWS, COLS) uint16} to use instead of noise."""
    master = tmp_path / (name + "_0001.h5")
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
        f["1.1/sample/name"] = name
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
        if place is not None:
            t["tomoconfig/sample_u"] = ["mrsx", "sx"]
            t["tomoconfig/sample_v"] = ["mrsy", "sy"]
            t["tomoconfig/sample_x"] = ["mrpsx", "psx"]
            t["tomoconfig/sample_y"] = ["mrpsy", "psy"]
        f["1.1/instrument/det3_optic/data_axes"] = ["z", "-y"]
        angle0, z0, clock = 0.0, -10.0, 1000.0
        for k, (entry, title, key, kind, n, level) in enumerate(stretches):
            t[f"subscans/scan{k}/type"] = kind
            data = _frames(rng, n, level) if frames is None or entry not in frames else frames[entry]
            scan_dir = tmp_path / f"scan{k + 2:04d}"
            scan_dir.mkdir()
            _write_frames(scan_dir / "det3_0000.h5", data, 10 if entry in compressed else None)
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
                if place is not None:
                    turned = np.radians(angle0 + (n * step if where.endswith("end") and kind == "tomo:continuous" else 0.0))
                    i[where + "/sx"], i[where + "/sy"], i[where + "/srot"] = place[0], place[1], np.degrees(turned)
                    i[where + "/psx"] = place[0] * np.cos(turned) + place[1] * np.sin(turned)
                    i[where + "/psy"] = -place[0] * np.sin(turned) + place[1] * np.cos(turned)
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
    report = json.loads(_core.calibration_check_record([str(master)], checks="record,inventory"))
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
    report = json.loads(_core.calibration_check_record([str(master)], checks="inventory"))
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
    loose = json.loads(_core.calibration_check_record([str(master)], checks='record', position_target=0.1))
    tight = json.loads(_core.calibration_check_record([str(master)], checks='record'))
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


def _low_band_53(x):
    """the low band of JPEG 2000's reversible 5/3 wavelet along the last axis (whole-sample symmetric extension):
    sample n sits at the input's position 2 n"""
    x = x.astype(np.int64)
    n = x.shape[-1]
    even, odd = x[..., 0::2], x[..., 1::2]
    right = np.concatenate([even[..., 1:], even[..., -1:]], -1)[..., :odd.shape[-1]] if n % 2 == 0 else even[..., 1:]
    d = odd - (even[..., :odd.shape[-1]] + right) // 2
    left = np.concatenate([d[..., :1], d], -1)[..., :even.shape[-1]]
    here = d if d.shape[-1] == even.shape[-1] else np.concatenate([d, d[..., -1:]], -1)
    return even + (left + here + 2) // 4


def test_frames_at_reduced_resolution(tmp_path):
    """RawReader(reduce=1, 2, 3): JPEG 2000 frames at a half, a quarter, an eighth, from the codestream's own
    lower resolutions. Exactly the wavelet's low band of each chunk (a reversible codestream), whose sample k
    sits at the stored pixel k 2^reduce; any rows, across chunks and beyond the frame; refused for plain frames
    and for a codestream without that many levels."""
    if not _core.jpeg2000_frames_supported():
        pytest.skip("built without OpenJPEG")
    rng = np.random.default_rng(31)
    rows, cols, chunk = 512, 320, 256
    jj, kk = np.mgrid[0:rows, 0:cols]
    data = np.stack([20000 + 37 * jj + 11 * kk,                                    # a ramp: the low band is the ramp at every 2^r-th pixel
                     rng.integers(20000, 30000, (rows, cols))]).astype(np.uint16)       # noise that stays within 16 bits in the low band
    _write_frames(tmp_path / "j.h5", data, chunk_rows=chunk)
    _write_frames(tmp_path / "plain.h5", data)
    dataset = "/entry_0000/instrument/det3/data"
    zero = np.zeros(2, np.int32)
    full = _core.RawReader([str(tmp_path / "j.h5")], dataset, rows, cols)
    assert (full.rows, full.cols, full.reduce) == (rows, cols, 0)
    assert np.array_equal(full.read_frames(zero, np.arange(2, dtype=np.int32)), data)
    for r in (1, 2, 3):
        f = 1 << r
        rd = _core.RawReader([str(tmp_path / "j.h5")], dataset, rows, cols, reduce=r)
        assert (rd.rows, rd.cols, rd.reduce) == (rows // f, cols // f, r)
        got = rd.read_frames(zero, np.arange(2, dtype=np.int32))
        assert got.shape == (2, rows // f, cols // f) and got.dtype == np.uint16
        want = data.astype(np.int64)
        for _ in range(r):                                                      # each chunk is an image of its own: rows first, then columns
            want = np.concatenate([np.swapaxes(_low_band_53(np.swapaxes(want[:, a:a + (chunk >> _)], 1, 2)), 1, 2) for a in range(0, want.shape[1], chunk >> _)], 1)
            want = _low_band_53(want)
        assert np.array_equal(got, want), r
        inner = (slice(4, chunk // f - 4), slice(4, -4))                        # where no edge of the chunk is felt
        assert np.array_equal(got[0][inner], data[0, ::f, ::f][inner])          # not the mean of f x f pixels (that would be 24 (f - 1) more)
        rd.set_cache_bytes(1 << 24)
        for row0, nrows in ((0, rows // f), (3, 5), (chunk // f - 2, 7), (-2, 6), (rows // f - 3, 6)):
            part = rd.read_rows(zero, np.arange(2, dtype=np.int32), np.full(2, row0, np.int32), nrows)
            ref = np.zeros((2, nrows, cols // f), np.uint16)
            a, b = max(0, row0), min(rows // f, row0 + nrows)
            ref[:, a - row0:b - row0] = got[:, a:b]
            assert np.array_equal(part, ref), (r, row0, nrows)
        with pytest.raises(Exception, match="JPEG 2000 only"):
            _core.RawReader([str(tmp_path / "plain.h5")], dataset, rows, cols, reduce=r).read_frames(zero[:1], zero[:1])
    with pytest.raises(Exception, match="reduce must be"):
        _core.RawReader([str(tmp_path / "j.h5")], dataset, rows, cols, reduce=4)
    with pytest.raises(Exception, match="not multiples"):
        _core.RawReader([str(tmp_path / "j.h5")], dataset, rows, cols + 2, reduce=2)
    # a codestream of small chunks has fewer levels than asked for
    small = rng.integers(1000, 60000, (1, 64, 96)).astype(np.uint16)
    _write_frames(tmp_path / "small.h5", small, chunk_rows=32)
    with pytest.raises(Exception, match="levels"):
        _core.RawReader([str(tmp_path / "small.h5")], dataset, 64, 96, reduce=1).read_frames(zero[:1], zero[:1])


def test_images_at_reduced_resolution():
    """calibration.at_resolution: a dark or flat on the grid of RawReader(reduce=) for irreversibly coded frames.
    Unit gain, a ramp kept at every 2^reduce-th pixel (the low band's sample sits on the stored pixel, not between
    pixels), each chunk of rows filtered alone, zeros beyond the frame in a last chunk that is not whole."""
    from helirec import calibration
    jj, kk = np.mgrid[0:512, 0:160].astype(np.float64)
    ramp = 500.0 + 7.0 * jj + 3.0 * kk
    for r in (1, 2, 3):
        f = 1 << r
        got = calibration.at_resolution(ramp, r, 256)
        assert got.shape == (512 // f, 160 // f) and got.dtype == np.float32
        # a symmetric low-pass of unit gain keeps a ramp, 8 reduced samples and more from an edge of a chunk (rows) or of the image (columns)
        for a in range(0, 512 // f, 256 // f):
            inner = (slice(a + 8, a + 256 // f - 8), slice(8, 160 // f - 8))
            assert np.allclose(got[inner], ramp[::f, ::f][inner], atol=2e-3), r
        assert np.allclose(calibration.at_resolution(np.full((2, 512, 160), 41.0), r, 256), 41.0, atol=1e-4)      # a stack, and unit gain
    one = calibration.at_resolution(ramp, 1, 256)
    assert np.allclose(one[:128], calibration.at_resolution(ramp[:256], 1, 256), atol=1e-4)   # a chunk does not see the next
    part = calibration.at_resolution(np.full((48, 80), 100.0), 1, 32)                          # 48 rows in chunks of 32: 16 rows of zeros below
    assert part.shape == (24, 40) and np.allclose(part[:21], 100.0, atol=1e-3) and abs(part[23, 5] - 100.0) > 1.0


def test_frames_stored_as_jpeg2000(tmp_path):
    """Projections stored as one JPEG 2000 codestream per chunk of rows (blosc2 filter 32026): read back exactly
    from a reversible codestream, whatever rows are asked for, and reported by the inventory."""
    if not _core.jpeg2000_frames_supported():
        pytest.skip("built without OpenJPEG")
    master = _write_record(tmp_path, np.random.default_rng(21), compressed=("4.1", "5.1"))
    with h5py.File(tmp_path / "scan0004" / "det3_0000.h5") as f:
        d = f["/entry_0000/instrument/det3/data"]
        assert d.chunks == (1, 10, COLS) and d.id.get_create_plist().get_filter(0)[0] == 32026
        assert len(d.id.read_direct_chunk((0, 20, 0))[1]) < 10 * COLS * 2
    plain = tmp_path / "plain"
    plain.mkdir()
    _write_record(plain, np.random.default_rng(21))
    with h5py.File(plain / "scan0004" / "det3_0000.h5") as f:
        stored = f["/entry_0000/instrument/det3/data"][:]
    s = _core.CalibrationSession([str(master)])
    frames = np.array([0, 7, 15], dtype=np.int64)
    for row0, nrows in ((0, ROWS), (3, 5), (8, 14), (19, 5), (-2, 6), (ROWS - 3, 6)):   # within a chunk, across chunks, beyond the frame
        got = s.read_rows(0, 2, frames, row0, nrows)
        want = np.zeros((3, nrows, COLS), np.uint16)
        a, b = max(0, row0), min(ROWS, row0 + nrows)
        want[:, a - row0:b - row0] = stored[frames, a:b]
        assert np.array_equal(got, want), (row0, nrows)
    # decoded chunks are kept between reads (blocks of a calibration come back to the same frames): the same values
    before = s.cache_counts()
    again = s.read_rows(0, 2, frames, 3, 14)
    after = s.cache_counts()
    assert np.array_equal(again, stored[frames, 3:17]) and after["decoded"] == before["decoded"] and after["hits"] == before["hits"] + 6
    s.set_cache(0.0)
    assert np.array_equal(s.read_rows(0, 2, frames, 3, 14), again) and s.cache_counts()["decoded"] == after["decoded"] + 6
    report = json.loads(_core.calibration_check_record([str(master)], checks="record,inventory"))
    f = _by_quantity(report, "every frame on disk")[0]
    assert f["verdict"] == "verified" and f["numbers"]["mismatched"] == 0, f       # the recorded sums are of these values
    f_all = f["numbers"]["frames"]
    how = _by_quantity(report, "how the frames are stored")[0]
    assert how["verdict"] == "note" and "4.1: JPEG 2000, reversible" in how["found"] and "chunks of 10 rows" in how["found"]
    assert "2.1" not in how["found"]
    # with no statistics recorded for a compressed stretch there is nothing to compare its frames with: one frame
    # in 16 is decoded, every chunk is located, and the report says so
    with h5py.File(master, "r+") as f:
        del f["5.1/instrument/det3_full"]
    sparse = _by_quantity(json.loads(_core.calibration_check_record([str(master)], checks="inventory")), "every frame on disk")[0]
    assert sparse["verdict"] == "unverified" and "5.1: one frame in 16 (2 of 32 on disk)" in sparse["found"], sparse
    assert sparse["numbers"]["located"] == 2 * PER_TURN and sparse["numbers"]["frames"] == f_all - 2 * PER_TURN + 2 and sparse["numbers"]["mismatched"] == 0
    # a container that is not what is read is refused, not guessed at
    with h5py.File(tmp_path / "scan0004" / "det3_0000.h5", "r+") as f:
        d = f["/entry_0000/instrument/det3/data"]
        raw = bytearray(d.id.read_direct_chunk((2, 0, 0))[1])
        raw[2:9] = b"nothing"
        d.id.write_direct_chunk((2, 0, 0), bytes(raw))
    with pytest.raises(Exception, match="not a blosc2 frame"):
        _core.CalibrationSession([str(master)]).read_rows(0, 2, np.array([2], dtype=np.int64), 0, 4)


def test_comparisons_decode_ahead(tmp_path, monkeypatch):
    """Comparisons on compressed frames: the frames of the ones that follow are decoded ahead by the reader's own
    workers and kept; the measurements are the same with and without."""
    if not _core.jpeg2000_frames_supported():
        pytest.skip("built without OpenJPEG")
    import sys
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROWS", 48)
    monkeypatch.setattr(module, "COLS", 80)
    monkeypatch.setattr(module, "PER_TURN", 40)
    master = _write_record(tmp_path, np.random.default_rng(23), compressed=("4.1",))
    reqs = [dict(label=str(p), stretch_a=2, frame_a=p, frames=16, rows=16, cols=32, row_a=14, col_a=20, col_b=20) for p in (10, 11, 12, 13, 14)]
    results = []
    for ahead in (0, 3):
        s = _core.CalibrationSession([str(master)])
        s.set_read_ahead(ahead)
        results.append(s.measure(reqs, -1.0, "cpu"))
        counts = s.cache_counts()
        assert counts["hits"] > counts["decoded"]            # the blocks overlap: most chunks are decoded once
    assert all(r["ok"] and r["correlation"] > 0.999 for r in results[0])
    for r0, r1 in zip(*results):     # the same values were registered: equal up to the rounding of sums taken by several threads
        assert all(r0[key] == pytest.approx(r1[key], abs=1e-9) for key in r0 if key != "label")


def test_frames_not_on_disk(tmp_path):
    """A partial copy of an acquisition: frame files that are absent are reported, and their frames are never
    read (the master's virtual dataset would give zeros for them)."""
    master = _write_record(tmp_path, np.random.default_rng(22))
    (tmp_path / "scan0006" / "det3_0000.h5").unlink()
    with h5py.File(master) as f:
        assert not f["6.1/instrument/det3/image"][0].any()                           # what must not be used
    report = json.loads(_core.calibration_check_record([str(master)], checks="record,inventory,series"))
    f = _by_quantity(report, "map from frame to file")[0]
    assert f["verdict"] == "unverified" and f["requires_follow_up"] and "6.1: none" in f["found"] and "5.1: 0-31" in f["found"]
    assert f["numbers"]["frames_on_disk"] == f["numbers"]["frames"] - PER_TURN and f["numbers"]["files_missing"] == 1
    f = _by_quantity(report, "every frame on disk")[0]
    assert f["verdict"] == "verified" and f["numbers"]["absent"] == PER_TURN and f["numbers"]["compared"] == f["numbers"]["frames"]
    assert "6.1 projection: no frame on disk" in _by_quantity(report, "range of the raw values")[0]["found"]
    s = _core.CalibrationSession([str(master)])
    st = s.stretches(0)
    assert st[4]["on_disk"] == [] and st[4]["n_on_disk"] == 0 and st[3]["on_disk"] == [(0, 2 * PER_TURN)]
    assert s.read_rows(0, 3, np.array([5], dtype=np.int64), 0, 4).shape == (1, 4, COLS)
    with pytest.raises(Exception, match="not on disk"):
        s.read_rows(0, 4, np.array([5], dtype=np.int64), 0, 4)
    with pytest.raises(Exception, match="not on disk"):
        s.line_integrals(0, 4, np.array([5], dtype=np.int64), np.array([0], dtype=np.int32), 4, 0, 8)


def test_tiles_of_a_mosaic(tmp_path):
    """Scans that differ in where the sample sits on the rotation stage: any number of them, checked as tiles."""
    places = {"T0": (1.0, 2.0), "T1": (1.4, 2.0), "T2": (1.2, 2.3)}
    masters = []
    for k, (label, place) in enumerate(places.items()):
        d = tmp_path / label
        d.mkdir()
        masters.append(str(_write_record(d, np.random.default_rng(30 + k), name="sample_" + label, place=place)))
    report = json.loads(_core.calibration_check_record(masters, checks="record"))
    which = _by_quantity(report, "the scans of the acquisition and what differs between them")[0]
    assert which["scan"] == "T0+2" and which["numbers"] == {"scans": 3, "mosaic": 1} and "tiles of a mosaic" in which["found"]
    where = _by_quantity(report, "place of the sample on the rotation stage")[0]
    px = 8.0e-3
    assert where["numbers"]["u_px_T1"] == pytest.approx(0.4 / px) and where["numbers"]["v_px_T2"] == pytest.approx(0.3 / px)
    assert where["numbers"]["lab_sense"] == 1 and where["numbers"]["lab_worst_mm"] < 1e-9 and "the same at the start and end" in where["found"]
    lines = _by_quantity(report, "which lines the scans record")[0]
    assert lines["numbers"]["pairs"] == 3 and lines["numbers"]["uncovered_percent"] == 0
    assert lines["numbers"]["lever_px"] == pytest.approx(0.4 / px + lines["numbers"]["reach_px"])
    for text in ("angle of the scans at equal projection index", "height of the scans at equal projection index", "time between corresponding projections"):
        assert len(_by_quantity(report, text)) == 3                      # one per projection stretch
    assert all(f["residuals"]["max_abs"] == 0 for f in _by_quantity(report, "angle of the scans at equal projection index"))
    # two scans that differ in the lateral stage are still the pair
    with h5py.File(masters[1], "r+") as f:
        for e in ("4.1", "5.1", "6.1"):
            for where_ in ("positioners_start", "positioners_end"):
                f[f"{e}/instrument/{where_}/yrot"][...] = 3.0
    pair = json.loads(_core.calibration_check_record(masters[:2], checks="record"))
    assert _by_quantity(pair, "lateral move between the scans")[0]["numbers"]["move_mm"] == 2.0


def test_sway_between_tiles(tmp_path, monkeypatch):
    """One tile of a mosaic against another: the same lines at equal angle, on columns that differ by a sinusoid of
    the angle. With the sway stated, the registration returns what is left of it."""
    import sys
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROWS", 48)
    monkeypatch.setattr(module, "COLS", 128)
    monkeypatch.setattr(module, "PER_TURN", 40)
    rng = np.random.default_rng(41)
    n, amplitude, phase, left = 40, 3.3, 25.0, 0.3
    k = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(ROWS), np.fft.fftfreq(COLS), indexing="ij")
    spectrum = np.fft.fftn(rng.standard_normal((n, ROWS, COLS))) * np.exp(-((k[0] / 0.12) ** 2 + (k[1] / 0.08) ** 2 + (k[2] / 0.08) ** 2))
    field = np.fft.ifftn(spectrum).real
    field *= 0.2 / field.std()

    def counts(f):
        return np.round(300 + 15000 * np.exp(-f)).astype(np.uint16)

    a_dir, b_dir = tmp_path / "A", tmp_path / "B"
    a_dir.mkdir(), b_dir.mkdir()
    ma = _write_record(a_dir, np.random.default_rng(1), name="sample_A", place=(0.0, 0.0), frames={"4.1": counts(field)})
    # the stated angle of each frame, as the record of B will hold it (the same trajectory)
    with h5py.File(ma) as f:
        angle = f["4.1/instrument/mrsrot_trig/data"][:] + f["4.1/instrument/mrsrot_delta/data"][:] / 2
    sway = amplitude * np.cos(np.radians(angle - phase)) + left
    # B(q, x) = A(q, x - sway(q)): B read sway columns further on is A
    moved = np.fft.ifft(np.fft.fft(field, axis=2) * np.exp(-2j * np.pi * k[2][:1] * sway[:, None, None]), axis=2).real
    mb = _write_record(b_dir, np.random.default_rng(1), name="sample_B", place=(0.1, 0.0), frames={"4.1": counts(moved)})
    s = _core.CalibrationSession([str(ma), str(mb)])
    base = dict(scan_a=0, scan_b=1, stretch_a=2, stretch_b=2, frame_a=12, frames=16, rows=16, cols=32, row_a=14, col_a=44, col_b=44)
    with_sway, without = s.measure([dict(base, sway_px=amplitude, sway_phase_deg=phase), base], -1.0, "cpu")
    assert with_sway["planned"] and with_sway["ok"] and with_sway["correlation"] > 0.98, with_sway
    assert with_sway["du"] == pytest.approx(left, abs=0.01) and abs(with_sway["dq"]) < 0.02 and abs(with_sway["D"]) < 0.01, with_sway
    assert without["correlation"] < with_sway["correlation"] - 0.03, without       # unstated, the sway blurs the match
    # rows exposed one after another: each row of the other tile at its own angle, rho degrees per row from the frame's
    rho = 0.25                                         # small enough for the sway to be a line in the row over the block
    row = np.arange(ROWS)[None, :, None] - (ROWS - 1) / 2
    sway_rows = amplitude * np.cos(np.radians(angle[:, None, None] + rho * row - phase)) + left
    moved = np.fft.ifft(np.fft.fft(field, axis=2) * np.exp(-2j * np.pi * k[2][:1] * sway_rows), axis=2).real
    r_dir = tmp_path / "R"
    r_dir.mkdir()
    mr = _write_record(r_dir, np.random.default_rng(1), name="sample_R", place=(0.1, 0.0), frames={"4.1": counts(moved)})
    s = _core.CalibrationSession([str(ma), str(mr)])
    per_row, per_frame = s.measure([dict(base, sway_px=amplitude, sway_phase_deg=phase, sway_row_deg=rho), dict(base, sway_px=amplitude, sway_phase_deg=phase)], -1.0, "cpu")
    assert per_row["ok"] and per_row["correlation"] > 0.98 and per_row["du"] == pytest.approx(left, abs=0.01), per_row
    assert per_frame["correlation"] < per_row["correlation"], (per_frame, per_row)
    # B's columns reversed (as for opposite views): the sway is stated in B's own columns, before the reversal.
    # C(q, y) = A(q, K - y) read sway columns further on, K the sum of the two blocks' end columns
    K = base["col_a"] + base["col_b"] + base["cols"] - 1
    flipped = field[:, :, (K - np.arange(COLS)) % COLS]
    moved = np.fft.ifft(np.fft.fft(flipped, axis=2) * np.exp(-2j * np.pi * k[2][:1] * sway[:, None, None]), axis=2).real
    c_dir = tmp_path / "C"
    c_dir.mkdir()
    mc = _write_record(c_dir, np.random.default_rng(1), name="sample_C", place=(0.1, 0.0), frames={"4.1": counts(moved)})
    s = _core.CalibrationSession([str(ma), str(mc)])
    mirrored = s.measure([dict(base, mirror=True, sway_px=amplitude, sway_phase_deg=phase)], -1.0, "cpu")[0]
    assert mirrored["ok"] and mirrored["cols"] == base["cols"] and mirrored["correlation"] > 0.98, mirrored
    assert abs(mirrored["du"]) == pytest.approx(left, abs=0.01), mirrored



def test_measure_patches_equals_measure(tmp_path, monkeypatch):
    """CalibrationSession.measure_patches: a block cut into patches, every patch measured from one reading of the
    block's frames. Each patch is what measure() returns for a request of the patch's size at the patch's place: with
    and without a sway, with B's columns reversed, where the registration moves a patch by whole columns inside what
    was read and beyond it (a second reading), and on CUDA as on the CPU."""
    import sys
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROWS", 64)
    monkeypatch.setattr(module, "COLS", 160)
    monkeypatch.setattr(module, "PER_TURN", 40)
    rng = np.random.default_rng(43)
    n, amplitude, phase, left, rho = 40, 3.3, 25.0, 0.3, 0.2
    k = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(ROWS), np.fft.fftfreq(COLS), indexing="ij")
    spectrum = np.fft.fftn(rng.standard_normal((n, ROWS, COLS))) * np.exp(-((k[0] / 0.12) ** 2 + (k[1] / 0.08) ** 2 + (k[2] / 0.08) ** 2))
    field = np.fft.ifftn(spectrum).real
    field *= 0.2 / field.std()

    def counts(f):
        return np.round(300 + 15000 * np.exp(-f)).astype(np.uint16)

    def record(name, data, place):
        d = tmp_path / name
        d.mkdir()
        return _write_record(d, np.random.default_rng(1), name=f"sample_{name}", place=place, frames={"4.1": counts(data)})

    ma = record("A", field, (0.0, 0.0))
    with h5py.File(ma) as f:
        angle = f["4.1/instrument/mrsrot_trig/data"][:] + f["4.1/instrument/mrsrot_delta/data"][:] / 2
    row = np.arange(ROWS)[None, :, None] - (ROWS - 1) / 2
    sway = amplitude * np.cos(np.radians(angle[:, None, None] + rho * row - phase)) + left
    shifted = lambda f, by: np.fft.ifft(np.fft.fft(f, axis=2) * np.exp(-2j * np.pi * k[2][:1] * by), axis=2).real
    mb = record("B", shifted(field, sway), (0.1, 0.0))
    base = dict(scan_a=0, scan_b=1, stretch_a=2, stretch_b=2, frame_a=12, frames=16, rows=32, cols=64, row_a=14, col_a=44, col_b=44)
    K = base["col_a"] + base["col_b"] + base["cols"] - 1
    mc = record("C", shifted(field[:, :, (K - np.arange(COLS)) % COLS], sway), (0.1, 0.0))
    stated = dict(sway_px=amplitude, sway_phase_deg=phase, sway_row_deg=rho)
    integers = ("planned", "ok", "frame_a", "frame_b", "frames", "row_a", "rows", "row_offset", "col_a", "col_b", "cols", "evaluations")
    reals = ("stated_dq", "dq", "stated_dv", "dv", "stated_D", "D", "du", "correlation")

    def check(session, request, device="cpu", expect_ok=True):
        got = session.measure_patches([request], -1.0, 16, 32, device)
        assert [(g["request"], g["patch_i"], g["patch_j"]) for g in got] == [(0, i, j) for j in (0, 1) for i in (0, 1)]
        for g in got:
            assert (g["row_a"], g["col_a"]) == (request["row_a"] + 16 * g["patch_j"], request["col_a"] + 32 * g["patch_i"])
            assert g["col_b"] == request["col_b"] + (32 - 32 * g["patch_i"] if request.get("mirror") else 32 * g["patch_i"])
        alone = session.measure([dict(request, rows=16, cols=32, row_a=g["row_a"], col_a=g["col_a"], col_b=g["col_b"]) for g in got], -1.0, device)
        for g, r in zip(got, alone):
            assert all(g[key] == r[key] for key in integers), (g, r)
            assert all(g[key] == pytest.approx(r[key], abs=2e-5) for key in reals), (g, r)
            if expect_ok:
                assert g["ok"] and g["correlation"] > 0.95, g
                c = g["curvature"]       # a maximum: the curvature is negative along each axis
                assert c[0] < 0 and c[3] < 0 and c[5] < 0, c
        return got

    s = _core.CalibrationSession([str(ma), str(mb)])
    got = check(s, dict(base, **stated))
    assert all(g["du"] == pytest.approx(left, abs=0.02) for g in got), got
    check(s, dict(base, scan_b=0))                                      # a block against itself: no sway, nothing resampled
    moved = check(s, dict(base, **stated, col_b=46))                    # two columns off: cut again from the same reading
    assert all(g["du"] == pytest.approx(left - 2, abs=0.02) for g in moved), moved
    # five columns off: beyond what was read, so B is read again. (A patch this small this far off may find a false
    # maximum, by either route alike; the ones that register are where they should be.)
    far = [g for g in check(s, dict(base, **stated, col_b=49), expect_ok=False) if g["ok"]]
    assert len(far) >= 3 and all(g["du"] == pytest.approx(left - 5, abs=0.02) for g in far), far
    # ... unless the block is registered whole first and its patches start where it ends: then every patch is found,
    # and where both routes find one they end on the same cut and the same numbers
    seeded = s.measure_patches([dict(base, **stated, col_b=49)], -1.0, 16, 32, "cpu", from_block=True)
    assert len(seeded) == 4 and all(g["ok"] and g["du"] == pytest.approx(left - 5, abs=0.02) for g in seeded), seeded
    for g in far:
        r = seeded[2 * g["patch_j"] + g["patch_i"]]
        assert all(g[key] == r[key] for key in integers if key != "evaluations") and all(g[key] == pytest.approx(r[key], abs=2e-5) for key in reals), (g, r)
    # a request the record cannot hold: one entry, no patch
    out = s.measure_patches([dict(base, frame_a=30)], -1.0, 16, 32, "cpu")
    assert len(out) == 1 and (out[0]["patch_i"], out[0]["patch_j"], out[0]["planned"], out[0]["ok"]) == (-1, -1, False, False)
    with pytest.raises(Exception):
        s.measure_patches([base], -1.0, 1, 32, "cpu")
    s = _core.CalibrationSession([str(ma), str(mc)])
    mirrored = check(s, dict(base, mirror=True, **stated))
    assert all(abs(g["du"]) == pytest.approx(left, abs=0.02) for g in mirrored), mirrored
    # B sheared along the frames across its columns (what opposite views in a fan ask for): the patches are sheared
    # side by side, each thread on work memory of its own (they once shared it, and half of them came out wrong)
    for _ in range(4):
        sheared = check(s, dict(base, mirror=True, frame_shear=0.004, **stated), expect_ok=False)
        assert all(g["ok"] and g["correlation"] > 0.9 for g in sheared), sheared
    if _core.cuda_available():
        on_device = s.measure_patches([dict(base, mirror=True, **stated)], -1.0, 16, 32, "cuda")
        for g, r in zip(on_device, mirrored):
            assert all(g[key] == r[key] for key in integers if key != "evaluations") and all(g[key] == pytest.approx(r[key], abs=1e-4) for key in reals), (g, r)
        check(s, dict(base, mirror=True, **stated), "cuda")
