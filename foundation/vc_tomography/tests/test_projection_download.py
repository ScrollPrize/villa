"""Slab selection, uncompressed exports, restart safety and frame metadata."""
import hashlib
import json
import logging
import os
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


@pytest.mark.skipif(not _core.htj2k_float_frames_supported(), reason="HELIREC_OPENJPH=OFF")
@pytest.mark.parametrize("role", ["projection", "flat"])
def test_export_into_a_projection_store(tmp_path, role):
    """--zarr: the same frames, as a level of the store instead of an HDF5 file, and no file is left behind"""
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
    out, store = tmp_path / "output", tmp_path / "projections.zarr"
    plan = dict(jobs=[job], tiles=[dict(tile="T0-01", label="T0-01", rows=job["rows"], cols=256, chunk_rows=256, source_master="/somewhere")])
    args = SimpleNamespace(zarr=store, reduce=3, remote=None, source_root=tmp_path)
    log = logging.getLogger("test")
    download._declare_store(plan, args, log)
    assert not download._complete(out / job["output"], job, str(store))
    events = queue.Queue()
    result = download._download_once(job, out, 1, 1, events, store=str(store))
    assert result["bytes_requested"] > 0 and not result["resumed"] and not out.exists()
    group = str(store / "T0-01" / "scan0005")
    reader = _core.RawReader([group], "", job["rows"], 256, 3, dtype=job["dtype"])
    np.testing.assert_array_equal(reader.read_frames(np.zeros(2, np.int32), np.arange(2, dtype=np.int32)), want)
    attributes = json.loads((store / "T0-01" / "scan0005" / "zarr.json").read_text())["attributes"]["helirec"]
    assert attributes["frame_first"] == 1200 and attributes["files"] == [dict(file="file.h5", source_first=1, first=1200, count=2)]
    assert [level["stored"] for level in attributes["levels"]] == [False, False, False, True]
    source.unlink()  # resuming needs no source reads, and declaring again keeps what is stored
    download._declare_store(plan, args, log)
    assert download._download_once(job, out, 1, 1, events, store=str(store)) == dict(bytes_requested=0, resumed=True)
    # another level of the same frames joins; other frames of the scan are refused
    half = dict(job, reduce=2)
    download._declare_store(dict(plan, jobs=[half]), SimpleNamespace(zarr=store, reduce=2, remote=None, source_root=tmp_path), log)
    attributes = json.loads((store / "T0-01" / "scan0005" / "zarr.json").read_text())["attributes"]["helirec"]
    assert [level["stored"] for level in attributes["levels"]] == [False, False, True, True]
    assert download._complete(out / job["output"], job, str(store)) and not download._complete(out / job["output"], half, str(store))
    with pytest.raises(Exception, match="the store holds frames"):
        download._declare_store(dict(plan, jobs=[dict(job, first=1300)]), args, log)


def test_plan_settings_cannot_change_on_resume(tmp_path):
    args = SimpleNamespace(output=tmp_path, tiles=["T0-01"], slices=256, reduce=3, center_mm=-256.0385,
                           context_turns=1.5, source_root=tmp_path / "remote", local_root=tmp_path / "local")
    download._atomic_json(tmp_path / "plan.json", dict(settings={"slices": 128}, jobs=[]))
    with pytest.raises(ValueError, match="different plan"):
        download.build_plan(args, logging.getLogger(__name__))


class _FakeSFTP:
    """paramiko's SFTPClient as far as --remote uses it, over local files; counts bytes served."""
    def __init__(self):
        self.served = 0
        self.opens = 0

    def open(self, path, mode):
        self.opens += 1
        fake = self
        class File:
            def __init__(self):
                self.f = open(path, "rb")
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                self.f.close()
            def stat(self):
                return SimpleNamespace(st_size=os.fstat(self.f.fileno()).st_size)
            def readv(self, ranges):
                for o, n in ranges:
                    self.f.seek(o)
                    fake.served += n
                    yield self.f.read(n)
        return File()


def _projection_source(source):
    _write(source, _chunk((DATA / "unsigned16_97.j2k").read_bytes(), 256, 256, sample_bytes=2),
           rows=256, cols=256, frames=3, dtype="uint16")


@pytest.mark.skipif(not _core.htj2k_float_frames_supported(), reason="HELIREC_OPENJPH=OFF")
def test_reader_refuses_bytes_that_were_not_fetched(tmp_path):
    source = tmp_path / "source.h5"
    _projection_source(source)
    with h5py.File(source, "r") as f:
        c = f["data"].id.get_chunk_info(1)
    full = _core.RawReader([str(source)], "data", 256, 256, reduce=3, decoder="openjph")
    want = full.read_frames(np.zeros(1, np.int32), np.array([1], np.int32), n_threads=1)
    covered = _core.RawReader([str(source)], "data", 256, 256, reduce=3, decoder="openjph")
    covered.set_fetched(0, [c.byte_offset, c.byte_offset + 10], [10, c.size - 10])  # merged: the whole chunk
    np.testing.assert_array_equal(covered.read_frames(np.zeros(1, np.int32), np.array([1], np.int32), n_threads=1), want)
    partial = _core.RawReader([str(source)], "data", 256, 256, reduce=3, decoder="openjph")
    partial.set_fetched(0, [c.byte_offset], [40])
    with pytest.raises(Exception, match="not fetched"):
        partial.read_frames(np.zeros(1, np.int32), np.array([1], np.int32), n_threads=1)


@pytest.mark.skipif(not _core.htj2k_float_frames_supported(), reason="HELIREC_OPENJPH=OFF")
@pytest.mark.parametrize("role,heads", [("projection", download._HEADS[3]), ("projection", (64, None)), ("flat", None)])
def test_remote_export_equals_local_export(tmp_path, monkeypatch, role, heads):
    source, job = _job(tmp_path, role)
    if role == "projection":
        _projection_source(source)
    else:
        with h5py.File(source, "w") as f:
            f.create_dataset("data", data=np.random.default_rng(10).integers(20000, 50000, (3, job["rows"], 256),
                             dtype=np.uint16), chunks=(1, job["rows"], 256))
    baseline = tmp_path / "baseline"
    download._download(job, baseline, 1, 1, 0, queue.Queue())
    sftp = _FakeSFTP()
    monkeypatch.setattr(download, "_connect", lambda spec: sftp)
    if heads is not None:
        monkeypatch.setattr(download, "_HEADS", {job["reduce"]: heads})
    remote = dict(spec="esrf:/raw", path=str(tmp_path), source_root=str(tmp_path))
    out, events = tmp_path / "remote", queue.Queue()
    result = download._download(job, out, 1, 1, 0, events, remote)
    with h5py.File(baseline / job["output"], "r") as a, h5py.File(out / job["output"], "r") as b:
        np.testing.assert_array_equal(a["data"][:], b["data"][:])
        assert b.attrs["source"] == "esrf:/raw/source.h5"
        assert {k: v for k, v in a.attrs.items() if k != "source"} == {k: v for k, v in b.attrs.items() if k != "source"}
    retries = [e for e in iter(lambda: None if events.empty() else events.get_nowait(), None) if e[0] == "retry"]
    assert bool(retries) == (heads == (64, None))  # 64 bytes per chunk cannot hold a codestream's low resolution
    assert result["bytes_transferred"] > 0 and sftp.served >= result["bytes_transferred"]
    assert not list((out / ".remote").iterdir())
    assert download._download(job, out, 1, 1, 0, events, remote)["resumed"]


@pytest.mark.parametrize("scan", [None, "2.4um_PHerc-Paris4_B_TA"])
def test_plan_reads_master_and_sample_over_sftp(tmp_path, monkeypatch, scan):
    """--remote: the plan needs nothing under --source-root (an SFTP mount that is down must not stop it)."""
    label = scan or "1.129um_mosaic_PHerc_Paris4_ROI-1_T0-01"
    tile = scan or "T0-01"
    served = tmp_path / "served" / label / (label + "_0001")
    served.mkdir(parents=True)
    master = _write_record(served, np.random.default_rng(0), name=label)
    sftp = _FakeSFTP()
    monkeypatch.setattr(download, "_connect", lambda spec: sftp)
    monkeypatch.setattr(download, "_rclone_remote", lambda spec: ({}, str(tmp_path / "served")))
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(output=out, tiles=[tile], scans=[scan] if scan else None, slices=4, reduce=1, center_mm=-11.6, context_turns=0,
                           source_root=tmp_path / "mount_that_is_down", local_root=tmp_path / "local",
                           remote="esrf:/raw")
    # the synthetic frames are plain: the plan gets as far as refusing what the fetched sample file stores
    with pytest.raises(ValueError, match="expects the irreversible HTJ2K"):
        download.build_plan(args, logging.getLogger(__name__))
    assert (out / "metadata" / tile / master.name).read_bytes() == master.read_bytes()
    assert sftp.served > master.stat().st_size  # the master and part of one sample file
    assert not list(out.glob("plan-*")) and not (tmp_path / "mount_that_is_down").exists()


@pytest.mark.parametrize("options", [
    ["--scans", "../scan"], ["--scans", "/scan"], ["--scans", ".."],
    ["--scans", "scan", "scan"], ["--scans", "scan", "--tiles", "T0-01"],
])
def test_explicit_scan_selection_rejects_unsafe_or_ambiguous_names(options, tmp_path):
    with pytest.raises(SystemExit) as error:
        download.main(["--output", str(tmp_path / "output"), *options])
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("role", ["projection", "flat"])
@pytest.mark.parametrize("existing", [None, 1, 2])
def test_multiple_levels_share_remote_transfer_and_resume(tmp_path, monkeypatch, role, existing):
    source, original = _job(tmp_path, role)
    if role == "projection":
        _projection_source(source)
    else:
        full = np.random.default_rng(10).integers(20000, 50000, (3, original["rows"], 256), dtype=np.uint16)
        with h5py.File(source, "w") as f:
            f.create_dataset("data", data=full, chunks=(1, original["rows"], 256))
    job = dict(original, reduce=1, reductions=[1, 2])
    job["bytes"] = sum(download._level_job(job, r)["bytes"] for r in job["reductions"])
    out, store = tmp_path / "output", tmp_path / "projections.zarr"
    plan = dict(jobs=[job], tiles=[dict(tile=job["tile"], chunk_rows=job["chunk_rows"])])
    args = SimpleNamespace(zarr=store, reduce=1, reductions=[1, 2], remote=None, source_root=tmp_path)
    download._declare_store(plan, args, logging.getLogger(__name__))
    if existing:
        download._download_once(download._level_job(job, existing), out, 1, 1, queue.Queue(), store=str(store))
    sftp = _FakeSFTP()
    monkeypatch.setattr(download, "_connect", lambda spec: sftp)
    fetched_heads = []
    fetch = download._fetch
    def track_fetch(*args):
        fetched_heads.append(args[-1])
        return fetch(*args)
    monkeypatch.setattr(download, "_fetch", track_fetch)
    events = queue.Queue()
    remote = dict(spec="esrf:/raw", path=str(tmp_path), source_root=str(tmp_path))
    result = download._download(job, out, 1, 1, 0, events, remote, str(store))
    assert not result["resumed"] and sftp.opens == 1
    assert result["bytes_transferred"] == sftp.served
    finest_missing = 2 if existing == 1 else 1
    assert fetched_heads == [download._HEADS[finest_missing][0] if role == "projection" else None]
    for reduce in [1, 2]:
        if role == "projection":
            reader = _core.RawReader([str(source)], "data", job["rows"], job["cols"], reduce, decoder="openjph")
            want = reader.read_frames(np.zeros(2, np.int32), np.array([1, 2], np.int32), n_threads=1)
        else:
            want = at_resolution(full[1:], reduce, job["chunk_rows"])
        reader = _core.RawReader([download._scan_group(store, job)], "", job["rows"], job["cols"], reduce, dtype=job["dtype"])
        got = reader.read_frames(np.zeros(2, np.int32), np.arange(2, dtype=np.int32), n_threads=1)
        np.testing.assert_array_equal(got, want)
    progress = [event[2] for event in list(events.queue) if event[0] == "progress"]
    assert progress == sorted(progress) and progress[-1] == job["count"]
    source.unlink()
    assert download._download(job, out, 1, 1, 0, events, remote, str(store))["resumed"]
    assert sftp.opens == 1


@pytest.mark.parametrize("options", [["--reduce", "1", "2"], ["--reduce", "1", "1", "--zarr", "unused"]])
def test_multiple_level_cli_validation(options, tmp_path):
    with pytest.raises(SystemExit) as error:
        download.main(["--output", str(tmp_path / "output"), *options])
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_multilevel_plan_uses_native_pixel_size_and_counts_both_levels(tmp_path, monkeypatch):
    label = "2.4um_PHerc-Paris4_B_TA"
    source = tmp_path / "local" / label / (label + "_0001")
    source.mkdir(parents=True)
    master = _write_record(source, np.random.default_rng(0), name=label)
    with h5py.File(master, "r+") as f:
        f["1.1/technique/scan/sample_pixel_size"][()] = 2.4
    # The synthetic record supplies real angles/heights and frame mappings; its plain pixels are irrelevant
    # to planning. Codec/decoded-value coverage is in the multilevel transfer test above.
    class Probe:
        def __init__(self, *args, **kwargs): pass
        def storage(self):
            return dict(compressed=True, codec="JPEG 2000 (HTJ2K)", reversible=False, chunk_rows=16)
    monkeypatch.setattr(download._core, "RawReader", Probe)
    out = tmp_path / "output"
    out.mkdir()
    args = SimpleNamespace(output=out, tiles=[label], scans=[label], slices=4, reduce=1, reductions=[1, 2],
                           center_mm=-11.6, context_turns=0, source_root=tmp_path / "missing",
                           local_root=tmp_path / "local", remote=None, zarr=tmp_path / "projections.zarr")
    plan = download.build_plan(args, logging.getLogger(__name__))
    assert plan["tiles"][0]["thickness_mm"] == pytest.approx(4 * 0.0024)
    assert plan["settings"]["reductions"] == [1, 2]
    assert all(j["reductions"] == [1, 2] for j in plan["jobs"])
    assert plan["total_bytes"] == sum(j["count"] * ((ROWS//2)*(COLS//2) + (ROWS//4)*(COLS//4))
                                    * (2 if j["role"] == "projection" else 4) for j in plan["jobs"])
    assert download.build_plan(args, logging.getLogger(__name__)) == plan
    args.reductions = [1, 3]
    with pytest.raises(ValueError, match="different plan"):
        download.build_plan(args, logging.getLogger(__name__))
