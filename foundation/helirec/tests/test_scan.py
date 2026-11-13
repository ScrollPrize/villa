"""helirec::calibration::Scan (helirec/calibration/scan.hpp) through its bindings, on a small synthetic BLISS master
written the way BLISS / Lima write one: a virtual frame dataset per entry over the detector's files, the projections'
monitors in a frame-less companion entry, wrapped angles, two flat series. The expectations are the reference
conversion's rules (nxtomomill h52nx, nx2etf) written out with numpy."""
from datetime import datetime, timezone

import numpy as np
import pytest

from helirec import _core, bliss

h5py = pytest.importorskip("h5py")

ROWS, COLS = 4, 6


def _ms(iso):
    d = datetime.fromisoformat(iso).astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(d).astype("<M8[ms]")


def _spread(st, en, n):
    return np.linspace(_ms(st).astype(np.longdouble), _ms(en).astype(np.longdouble), num=n, dtype="<M8[ms]")


# entry, title, image_key, frames, start, end, current [mA] (measurement), machine current [mA] (scalar)
ENTRIES = [
    ("2.1", "dark images", 2, 3, "2025-05-10T10:00:00.100000+02:00", "2025-05-10T10:00:04.250300+02:00", [81.7, 81.71, 81.72, 81.7, 81.69], 81.7),
    ("3.1", "flat images 1", 1, 4, "2025-05-10T10:00:04.100000+02:00", "2025-05-10T10:00:09.700000+02:00", [81.5, 81.52, 81.49], 81.5),
    ("4.1", "projections 0 - 10", 0, 10, "2025-05-10T10:00:09.555555+02:00", "2025-05-10T10:00:30.123456+02:00", None, 81.3),
    ("4.2", "projections 0 - 10", 0, 0, "2025-05-10T10:00:09.555555+02:00", "2025-05-10T10:00:30.223456+02:00", [81.32, 81.3, 90.1, 89.9, 89.8, 89.7], 81.3),
    ("5.1", "flat images 10", 1, 4, "2025-05-10T10:00:30.100000+02:00", "2025-05-10T10:00:35.000000+02:00", [88.6, 88.65, 88.7, 88.6], 88.7),
]
ANGLES = np.array([340.0, 350.5, 1.25, 12.0, 23.5, 34.0, 45.0, 56.0, 67.5, 78.0])  # wraps after the second frame


def _write_scan(root, name, yrot_mm=1.5):
    """root/name/name_0001/name_0001.h5 with its detector files in scan000N/; returns the master's path"""
    d = root / name / f"{name}_0001"
    d.mkdir(parents=True)
    master = d / f"{name}_0001.h5"
    with h5py.File(master, "w") as f:
        g = f.create_group("1.1")
        g["title"] = "tomo:helical"
        g["start_time"] = ENTRIES[0][4]
        g["end_time"] = ENTRIES[-1][5]
        g["sample/name"] = name
        tc = g.create_group("technique/tomoconfig")
        tc["detector"] = np.array([b"det3"], dtype=object)
        tc["rotation"] = np.array([b"mrsrot", b"srot"], dtype=object)
        tc["translation_z"] = np.array([b"mrsz", b"sz"], dtype=object)
        tc["translation_y"] = np.array([b"mryrot", b"yrot"], dtype=object)
        sc = g.create_group("technique/scan")
        for key, value, unit in (("energy", 116.0, "keV"), ("effective_propagation_distance", 1191.9883738942317, "mm"), ("exposure_time", 30.0, "ms"),
                                 ("latency_time", 3.0, "ms"), ("sample_pixel_size", 8.64, "um")):
            sc[key] = value
            sc[key].attrs["units"] = unit
        sc["scan_type"] = "CONTINUOUS"
        g["technique/optic/sample_pixel_size"] = 8.64
        g["technique/optic/sample_pixel_size"].attrs["units"] = "um"
        for k, (entry, title, key, n, st, en, cur, machine) in enumerate(ENTRIES):
            e = f.create_group(entry)
            e["title"], e["start_time"], e["end_time"] = title, st, en
            e["technique/image_key"] = key
            e["instrument/machine/current"] = machine
            e["instrument/machine/current"].attrs["units"] = "mA"
            if cur is not None:
                e["measurement/current"] = np.array(cur)
                e["measurement/current"].attrs["units"] = "mA"
            e["instrument/positioners/yrot"] = yrot_mm
            e["instrument/positioners/yrot"].attrs["units"] = "mm"
            e["instrument/positioners/sz"] = 12.5
            e["instrument/positioners/sz"].attrs["units"] = "mm"
            if not n:
                continue
            sub = d / f"scan{k + 2:04d}"
            sub.mkdir()
            # the frames over two detector files, the second holding the entry's last frames
            split = n // 2
            layout = h5py.VirtualLayout(shape=(n, ROWS, COLS), dtype=np.uint16)
            for part, (a, b) in enumerate(((0, split), (split, n))):
                fn = sub / f"det3_{part:04d}.h5"
                with h5py.File(fn, "w") as lf:
                    lf.create_dataset("/entry_0000/instrument/det3/data", data=np.full((b - a + 1, ROWS, COLS), k, np.uint16))
                layout[a:b] = h5py.VirtualSource(f"scan{k + 2:04d}/det3_{part:04d}.h5", "/entry_0000/instrument/det3/data",
                                                 shape=(b - a + 1, ROWS, COLS))[1:]
            e.create_virtual_dataset(f"instrument/det3/image", layout)
            if key == 0:
                e["measurement/mrsrot"] = ANGLES
                e["measurement/mrsrot"].attrs["units"] = "deg"
                e["measurement/mrsrot_delta"] = np.full(n, -0.9)
                e["measurement/mrsrot_period"] = np.full(n, 1.0)
    return master


def _expected_current():
    """nxtomomill's ring current at every frame of the sequence, and the role of each frame"""
    known, frame_t, kinds = {}, [], []
    for entry, title, key, n, st, en, cur, machine in ENTRIES:
        if n:
            frame_t.append(_spread(st, en, n))
            kinds += [key] * n
        values = np.atleast_1d(np.asarray(cur if cur is not None else machine, np.float64)) * 1e-3
        times = [_ms(st)] if len(values) == 1 else _spread(st, en, len(values))
        known.update(zip(times, values))
    items = sorted(known.items())
    kt = np.array([k for k, _ in items], "datetime64[ns]").astype(np.float64)
    ft = np.concatenate(frame_t).astype("datetime64[ns]").astype(np.float64)
    return np.interp(ft, kt, [v for _, v in items]), np.array(kinds)


def test_open_scan(tmp_path):
    master = _write_scan(tmp_path, "sample_HA")
    for path in (master, master.parent, master.parent.parent):   # the master, its scan directory, the directory above
        assert bliss.open_scan(str(path)).master == str(master)
    s = bliss.open_scan(str(master))
    assert s.name == "sample_HA_0001" and s.frame_shape == (ROWS, COLS) and s.dtype == "uint16" and s.detector == "det3"
    assert s.n_proj == 10 and len(s.darks) == 3 and [len(f) for f in s.flats] == [4, 4]
    assert list(s.entries) == ["2.1", "3.1", "4.1", "4.2", "5.1"] and s.entries["4.2"] == dict(role="proj", title="projections 0 - 10", n_frames=0)

    # the frames: two detector files per entry, the first frame of each file skipped by the mapping
    p = s.projections
    assert p.dataset == "/entry_0000/instrument/det3/data"
    assert [f.rsplit("/", 2)[-2:] for f in p.files] == [["scan0004", "det3_0000.h5"], ["scan0004", "det3_0001.h5"]]
    np.testing.assert_array_equal(p.file_index, [0] * 5 + [1] * 5)
    np.testing.assert_array_equal(p.frame_index, list(range(1, 6)) * 2)
    assert len(s.flats[1].files) == 4    # every flat series keeps the files of all flat frames
    frames = _core.RawReader(list(p.files), p.dataset, ROWS, COLS).read_frames(p.file_index, p.frame_index)
    assert (frames == 2).all()
    s.check_files()
    assert len(s.referenced_files()) == 8

    # angles, heights, ring current, the flats' angles
    np.testing.assert_array_equal(s.angles_deg, ANGLES)
    np.testing.assert_array_equal(s.unwrapped_angles_deg, np.unwrap(ANGLES, period=360))
    assert s.unwrapped_angles_deg[2] == 361.25
    np.testing.assert_array_equal(s.z_translation_m, np.full(10, 12.5 * 1e-3))
    current, kinds = _expected_current()
    assert np.array_equal(s.control_A, current[kinds == 0])     # bit for bit
    assert np.array_equal(s.darks_control_A, current[kinds == 2])
    flat = np.flatnonzero(kinds == 1)
    assert np.array_equal(s.flats_control_A[0], current[flat[:4]]) and np.array_equal(s.flats_control_A[1], current[flat[4:]])
    proj = np.flatnonzero(kinds == 0)
    np.testing.assert_array_equal(s.flats_unwrapped_angles_deg,
                                  [np.interp(0.5 * (r[0] + r[-1]), proj, s.unwrapped_angles_deg) for r in (flat[:4], flat[4:])])

    # scalars in SI
    assert s.pixel_size_m == 8.64 * 1e-6 and s.energy_kev == 116.0 and s.distance_m == 1191.9883738942317 * 1e-3
    assert s.y_translation_m == 1.5 * 1e-3 and s.estimated_cor_from_motor_pix == s.y_translation_m / s.pixel_size_m
    assert s.count_time_s == 30.0 * 1e-3 and s.exposure_arc == 0.9
    assert s.detector_pixel_size_m is None


def test_missing_detector_file(tmp_path):
    master = _write_scan(tmp_path, "sample_HA")
    (master.parent / "scan0004" / "det3_0001.h5").unlink()
    s = bliss.open_scan(str(master))           # a partial copy opens ...
    with pytest.raises(RuntimeError, match="1 detector files missing"):
        s.check_files()                        # ... and says what is missing
    assert bliss.discover_scans(str(tmp_path), require_files=True) == []


def test_discover_and_pair(tmp_path):
    _write_scan(tmp_path, "sample_TA", yrot_mm=10.9)
    _write_scan(tmp_path, "sample_HA")
    (tmp_path / "not_a_scan").mkdir()
    scans = bliss.discover_scans(str(tmp_path))
    assert [s.name for s in scans] == ["sample_HA_0001", "sample_TA_0001"]
    pair = bliss.pair_companions(scans)
    assert pair["main"].name == "sample_HA_0001" and pair["companion"].name == "sample_TA_0001"
    assert [s.name for s in pair["companions"]] == ["sample_TA_0001"]
    with pytest.raises(RuntimeError, match="expected one HA/TA family"):
        bliss.pair_companions(scans[:1])


def test_parse_tetra_name():
    assert bliss.parse_tetra_name("SCROLLS_PHerc0343P_HA_0001") == dict(base="SCROLLS_PHerc0343P", kind="HA", rank=0, suffix="_0001", crash=False)
    assert bliss.parse_tetra_name("x_TA2_fc__0003_0002") == dict(base="x", kind="TA2", rank=2, suffix="_0003_0002", crash=True)
    assert bliss.parse_tetra_name("plain_0001") is None
