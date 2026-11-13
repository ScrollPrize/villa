"""Frames read from a projection store (helirec/projection_store.hpp; made by helirec-projections-zarr): RawReader
takes a scan group where it takes a file, at the levels the store holds, and frames_in_projection_store finds a
scan's frames in it. The detector's files and the store are made up here; the store's values are the ones a
reduced read would give only by this test's construction (every level keeps the stored pixels (j, k) 2^l)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from helirec import _core

h5py = pytest.importorskip("h5py")
from test_reconstruct_tool import COLS, ROWS, _write  # noqa: E402

_HERE = Path(_core.__file__).resolve().parent
TOOL = next((str(d / "helirec-projections-zarr") for d in (_HERE, *_HERE.parents[:2]) if (d / "helirec-projections-zarr").exists()),
            None) or shutil.which("helirec-projections-zarr")
pytestmark = pytest.mark.skipif(TOOL is None, reason="helirec-projections-zarr is not built")
N, FIRST = 240, 40           # projections of the scan; the store holds [FIRST, N)
DATASET = "/entry_0000/instrument/det3/data"


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """(scan, tile group, frames of the scan's projections file) with levels 2 and 3 stored"""
    root = tmp_path_factory.mktemp("store")
    master = _write(root, "tile", np.linspace(0, 720, N, endpoint=False), np.linspace(-20, 20, N), 3.0)
    scan = _core.open_scan(master)
    files = list(scan.projections.files)
    assert len(files) == 1 and Path(files[0]).parent.name == "scan0004"
    with h5py.File(files[0], "r") as f:
        frames = f[DATASET][:]
    with h5py.File(scan.darks.files[0], "r") as f:
        darks = f[DATASET][:]
    scales = []
    for reduce in (2, 3):
        scale = root / f"scale_{reduce}"
        jobs = []
        # the projections file in two pieces (the second starts inside it), and the darks as float32
        for scan_dir, role, first, count, data in (("scan0004", "projection", FIRST, 100, frames[FIRST:FIRST + 100]),
                                                   ("scan0004", "projection", FIRST + 100, N - FIRST - 100, frames[FIRST + 100:]),
                                                   ("scan0002", "dark", 0, len(darks), darks.astype(np.float32) + 0.25)):
            output = f"T/{scan_dir}/part_{first:04d}.h5"
            (scale / output).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(scale / output, "w") as f:
                f.create_dataset(DATASET, data=data[:, ::1 << reduce, ::1 << reduce], chunks=(1, ROWS >> reduce, COLS >> reduce))
                f.attrs["complete"] = True
            jobs.append(dict(tile="T", output=output, relative_file=f"{scan_dir}/det3_0000.h5", source_first=first, role=role, rows=ROWS, cols=COLS,
                             first=first, count=count, dtype="uint16", dataset=DATASET, reduce=reduce, metadata={}))
        tile = dict(label="T", rows=ROWS, cols=COLS, chunk_rows=64)
        (scale / "plan.json").write_text(json.dumps(dict(settings=dict(reduce=reduce), tiles=[tile], jobs=jobs)))
        scales.append(str(scale))
    out = root / "projections.zarr"
    run = subprocess.run([TOOL, str(out), *scales], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return scan, str(out / "T"), frames, darks


@pytest.mark.parametrize("reduce", [2, 3])
def test_reader_takes_a_scan_group_as_a_file(stored, reduce):
    scan, tile, frames, darks = stored
    group = f"{tile}/scan0004"
    assert _core.is_projection_store_scan(group) and not _core.is_projection_store_scan(tile) and not _core.is_projection_store_scan(scan.master)
    reader = _core.RawReader([group], "", ROWS, COLS, reduce)
    assert (reader.rows, reader.cols) == (ROWS >> reduce, COLS >> reduce)
    which = np.array([0, 57, 100, N - FIRST - 1], np.int32)
    expected = frames[FIRST + which][:, ::1 << reduce, ::1 << reduce]
    assert np.array_equal(np.asarray(reader.read_frames(np.zeros(4, np.int32), which)), expected)
    # rows that start above the frame, cross its chunks and end below it (0 outside), with the chunk cache on
    reader.set_cache_bytes(1 << 24)
    rows = reader.rows
    row0 = np.array([-3, 5, rows - 4, rows // 2], np.int32)
    for _ in range(2):
        got = np.asarray(reader.read_rows(np.zeros(4, np.int32), which, row0, 9))
        for i, r in enumerate(row0):
            want = np.zeros((9, reader.cols), np.uint16)
            a, b = max(0, r), min(rows, r + 9)
            want[a - r:b - r] = expected[i, a:b]
            assert np.array_equal(got[i], want)
    storage = reader.storage(0)
    assert storage["codec"] == "projection store (Zarr)" and storage["frames"] == N - FIRST and storage["chunk_rows"] == 64 and not storage["compressed"]
    with pytest.raises(Exception, match="frame index out of range"):
        reader.read_frames(np.zeros(1, np.int32), np.array([N - FIRST], np.int32))
    floats = _core.RawReader([f"{tile}/scan0002"], "", ROWS, COLS, reduce, dtype="float32")
    assert np.array_equal(np.asarray(floats.read_frames(np.zeros(2, np.int32), np.array([0, 19], np.int32))), (darks[[0, 19]].astype(np.float32) + 0.25)[:, ::1 << reduce, ::1 << reduce])
    with pytest.raises(Exception, match="values are not uint16"):
        _core.RawReader([f"{tile}/scan0002"], "", ROWS, COLS, reduce).read_frames(np.zeros(1, np.int32), np.zeros(1, np.int32))


def test_levels_the_store_does_not_hold_are_refused(stored):
    _, tile, _, _ = stored
    for reduce in (0, 1):
        reader = _core.RawReader([f"{tile}/scan0004"], "", ROWS, COLS, reduce)
        assert reader.storage(0)["chunk_rows"] == 64          # asked of the group, without a level
        with pytest.raises(Exception, match="does not hold it"):
            reader.read_frames(np.zeros(1, np.int32), np.zeros(1, np.int32))
    with pytest.raises(Exception, match="unexpected frame shape"):
        _core.RawReader([f"{tile}/scan0004"], "", ROWS + 8, COLS, 2).read_frames(np.zeros(1, np.int32), np.zeros(1, np.int32))


def test_a_scans_frames_are_found_in_the_store(stored):
    scan, tile, frames, _ = stored
    picked = [FIRST, FIRST + 99, FIRST + 100, N - 1, FIRST + 3]
    there = _core.frames_in_projection_store(scan.projections.subset(picked), tile)
    assert list(there.files) == [f"{tile}/scan0004"] and list(there.file_index) == [0] * 5
    assert list(there.frame_index) == [p - FIRST for p in picked]
    reader = _core.RawReader(list(there.files), there.dataset, ROWS, COLS, 3)
    assert np.array_equal(np.asarray(reader.read_frames(there.file_index, there.frame_index)), frames[picked][:, ::8, ::8])
    with pytest.raises(Exception, match=f"frame {FIRST - 1} of .* is not in"):
        _core.frames_in_projection_store(scan.projections.subset([FIRST - 1]), tile)
    with pytest.raises(Exception, match="has no scan scan0003"):
        _core.frames_in_projection_store(scan.flats[0], tile)
