"""Slab selection, uncompressed exports, restart safety and frame metadata."""
import hashlib
import logging
import queue
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from helirec import _core
from helirec.calibration import at_resolution
from helirec import download_projections as download
from test_calibration import _write_record, _write_frames, ROWS, COLS
from test_htj2k_reader import DATA, _chunk, _write


@pytest.mark.parametrize("dtype", ["uint16", "float32"])
def test_native_writer(tmp_path, dtype):
    path = tmp_path / "out.h5"
    data = np.arange(3 * 7 * 11, dtype=dtype).reshape(3, 7, 11)
    if dtype == "float32":
        data /= 7
    writer = _core.UncompressedFrameWriter(str(path), "entry/data", 3, 7, 11, dtype)
    writer.write(2, data[2:])
    writer.write(0, data[:2])
    with pytest.raises(Exception, match="range"):
        writer.write(3, data[:1])
    with pytest.raises(Exception, match="shape"):
        writer.write(0, data[:, :2])
    with pytest.raises(Exception, match="dtype"):
        writer.write(0, data.astype("float32" if dtype == "uint16" else "uint16"))
    writer.close()
    writer.close()
    with pytest.raises(Exception, match="closed"):
        writer.write(0, data[:1])
    with h5py.File(path, "r") as f:
        d = f["entry/data"]
        np.testing.assert_array_equal(d[:], data)
        assert d.dtype == np.dtype(dtype) and d.id.get_create_plist().get_nfilters() == 0
    with pytest.raises(Exception, match="cannot create"):
        _core.UncompressedFrameWriter(str(path), "entry/data", 3, 7, 11, dtype)


def test_subset_preserves_whole_files_and_all_angles(tmp_path):
    master = _write_record(tmp_path, np.random.default_rng(0))
    with h5py.File(tmp_path / "scan0005/det3_0000.h5", "r") as f:
        data = f["entry_0000/instrument/det3/data"][:]
    layout = h5py.VirtualLayout(data.shape, dtype=np.uint16)
    for k in range(8):
        relative = f"scan0005/det3_{k:04d}.h5"
        _write_frames(tmp_path / relative, data[4*k:4*k+4])
        layout[4*k:4*k+4] = h5py.VirtualSource(relative, "/entry_0000/instrument/det3/data", (4, ROWS, COLS))
    with h5py.File(master, "r+") as f:
        del f["5.1/instrument/det3/image"]
        f.create_virtual_dataset("5.1/instrument/det3/image", layout)
        heights = f["5.1/instrument/mrsz_center/data"][:]
    p = _core.plan_projection_subset(str(master), -11.6, 4, 0)
    assert p["thickness_mm"] == pytest.approx(0.032)
    assert p["pitch_mm"] == pytest.approx(1.6, abs=1e-3)
    assert p["context_mm"] == pytest.approx(24 * 0.008)
    series = {s["entry"]: s for s in p["series"]}
    assert set(series) == {"2.1", "3.1", "5.1", "7.1"}
    s = series["5.1"]
    assert (s["first"], s["count"]) == (12, 8)
    assert [q["first"] for q in s["pieces"]] == [12, 16]
    np.testing.assert_array_equal(s["height_mm"], heights[12:20])
    assert len(s["angle_deg"]) == len(s["epoch"]) == 8
    assert all(series[e]["count"] == 4 for e in ("2.1", "3.1", "7.1"))
    with pytest.raises(Exception, match="beyond the helix"):
        _core.plan_projection_subset(str(master), -11.6, 4, 1.5)
    with pytest.raises(Exception, match="invalid slab"):
        _core.plan_projection_subset(str(master), -11.6, 0, 0)
    # A missing VDS mapping must not silently become all-zero projection pixels.
    layout = h5py.VirtualLayout(data.shape, dtype=np.uint16)
    for k in range(8):
        if k != 3:
            layout[4*k:4*k+4] = h5py.VirtualSource(f"scan0005/det3_{k:04d}.h5", "/entry_0000/instrument/det3/data", (4, ROWS, COLS))
    with h5py.File(master, "r+") as f:
        del f["5.1/instrument/det3/image"]
        f.create_virtual_dataset("5.1/instrument/det3/image", layout)
    with pytest.raises(Exception, match="coverage|gaps"):
        _core.plan_projection_subset(str(master), -11.6, 4, 0)


def _job(tmp_path, role):
    source = tmp_path / "source.h5"
    job = dict(tile="T0-01", relative_file="scan0005/file.h5", output="T0-01/scan0005/file.h5",
               sources=[str(tmp_path / "missing.h5"), str(source)], dataset="data", entry="5.1", role=role,
               first=1200, source_first=1, count=2, rows=256 if role == "projection" else 384, cols=256, reduce=3, chunk_rows=256,
               dtype="uint16" if role == "projection" else "float32", source_master_sha256="source master hash",
               metadata=dict(angle_deg=[1.0, 2.0], height_mm=[-11.6, -11.7], epoch=[100.0, 101.0]))
    job["bytes"] = 2 * (job["rows"] >> 3) * 32 * np.dtype(job["dtype"]).itemsize
    job["id"] = hashlib.sha256(download._json_bytes(job)).hexdigest()
    return source, job


@pytest.mark.skipif(not _core.htj2k_float_frames_supported(), reason="HELIREC_OPENJPH=OFF")
@pytest.mark.parametrize("role", ["projection", "flat"])
def test_export_resume_checksum_and_index(tmp_path, role):
    source, job = _job(tmp_path, role)
    if role == "projection":
        _write(source, _chunk((DATA / "unsigned16_97.j2k").read_bytes(), 256, 256, sample_bytes=2),
               rows=256, cols=256, frames=3, dtype="uint16")
        rd = _core.RawReader([str(source)], "data", 256, 256, reduce=3, decoder="openjph", selective=False)
        want = rd.read_frames(np.zeros(2, np.int32), np.array([1, 2], np.int32), n_threads=1)
    else:
        full = np.random.default_rng(10).integers(20000, 50000, (3, job["rows"], 256), dtype=np.uint16)
        with h5py.File(source, "w") as f:
            f.create_dataset("data", data=full, chunks=(1, job["rows"], 256))
        want = at_resolution(full[1:], 3, 256)
    out = tmp_path / "output"
    events = queue.Queue()
    result = download._download_once(job, out, 1, 1, events)
    assert result["bytes_requested"] > 0 and not result["resumed"]
    assert events.get_nowait()[0] == "progress"
    path = out / job["output"]
    with h5py.File(path, "r") as f:
        got = f["data"][:]
        np.testing.assert_array_equal(got, want)
        assert got.dtype == np.dtype(job["dtype"])
        assert f["data"].id.get_create_plist().get_nfilters() == 0
        assert f.attrs["data_sha256"] == hashlib.sha256(got.tobytes()).hexdigest()
        np.testing.assert_array_equal(f["helirec_metadata/original_frame_index"][:], [1200, 1201])
        np.testing.assert_array_equal(f["helirec_metadata/angle_deg"][:], [1, 2])
    assert not path.with_suffix(".h5.partial").exists()
    assert download._complete(path, job)
    before = path.stat().st_mtime_ns
    source.unlink()  # resuming complete files requires no source reads
    assert download._download_once(job, out, 1, 1, events) == dict(bytes_requested=0, resumed=True)
    assert path.stat().st_mtime_ns == before
    download._index_tile(out, "T0-01", [job])
    with h5py.File(out / "T0-01/index.h5", "r") as f:
        np.testing.assert_array_equal(f["5.1/data"][:], want)
        np.testing.assert_array_equal(f["5.1/original_frame_index"][:], [1200, 1201])
    with h5py.File(path, "r+") as f:
        f.attrs["job_sha256"] = "different job"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        download._download_once(job, out, 1, 1, events)


def test_plan_settings_cannot_change_on_resume(tmp_path):
    args = SimpleNamespace(output=tmp_path, tiles=["T0-01"], slices=256, reduce=3, center_mm=-256.0385,
                           context_turns=1.5, source_root=tmp_path / "remote", local_root=tmp_path / "local")
    download._atomic_json(tmp_path / "plan.json", dict(settings={"slices": 128}, jobs=[]))
    with pytest.raises(ValueError, match="different plan"):
        download.build_plan(args, logging.getLogger(__name__))
