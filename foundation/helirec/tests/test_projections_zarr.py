"""helirec-projections-zarr (tools/helirec_projections_zarr.cpp): two reduced copies of a small made-up tile, written
the way helirec.download_projections writes them, become one multiscale Zarr store whose stored levels hold the
files' values, whose other levels are declared and empty, and which a second run leaves as it is."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from helirec import _core

h5py = pytest.importorskip("h5py")
open_zarr = pytest.importorskip("vc_reader.reader").open_zarr

_HERE = Path(_core.__file__).resolve().parent
TOOL = next((str(d / "helirec-projections-zarr") for d in (_HERE, *_HERE.parents[:2]) if (d / "helirec-projections-zarr").exists()),
            None) or shutil.which("helirec-projections-zarr")
pytestmark = pytest.mark.skipif(TOOL is None, reason="helirec-projections-zarr is not built")
ROWS, COLS, FIRST = 1040, 64, 500
DATASET = "/entry_0000/instrument/det1/data"


def _copy(root, reduce, rng, complete=True):
    """A scale folder: 130 projections in two files, 20 darks in one, and the plan that lists them."""
    shape = (ROWS >> reduce, COLS >> reduce)
    jobs, values = [], {}
    for scan, role, first, count, dtype in (("scan0005", "projection", FIRST, 100, np.uint16), ("scan0005", "projection", FIRST + 100, 30, np.uint16),
                                            ("scan0002", "dark", 0, 20, np.float32)):
        output = f"T1-01/{scan}/det1_{first // 100:04d}.h5"
        data = (rng.normal(1000, 30, (count,) + shape)).astype(dtype)
        values[scan, first] = data
        (root / output).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(root / output, "w") as f:
            f.create_dataset(DATASET, data=data, chunks=(1,) + shape)
            f.attrs["complete"] = complete or first != FIRST + 100
        metadata = {"epoch": [float(first + i) for i in range(count)], "height_mm": [0.5 * i for i in range(count)]}
        if role == "projection":
            metadata["angle_deg"] = [0.036 * (first + i) for i in range(count)]
        jobs.append(dict(tile="T1-01", output=output, role=role, rows=ROWS, cols=COLS, first=first, count=count, dtype="uint16", dataset=DATASET,
                         reduce=reduce, metadata=metadata))
    tile = dict(label="T1-01", rows=ROWS, cols=COLS, pixel_um=1.129, projection_first=FIRST, projection_count=130, source_master="/somewhere/master.h5")
    (root / "plan.json").write_text(json.dumps(dict(settings=dict(reduce=reduce), tiles=[tile], jobs=jobs)))
    return values


def test_two_scales_become_one_store(tmp_path):
    rng = np.random.default_rng(3)
    quarter, eighth = tmp_path / "scale_0.25", tmp_path / "scale_0.125"
    values = {2: _copy(quarter, 2, rng), 3: _copy(eighth, 3, rng)}
    out = tmp_path / "projections.zarr"
    run = subprocess.run([TOOL, str(out), str(eighth), str(quarter), "--threads", "3"], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "6 files written and verified, 0 already stored" in run.stdout

    group = json.loads((out / "T1-01" / "scan0005" / "zarr.json").read_text())["attributes"]
    assert [(level["path"], level["stored"]) for level in group["helirec"]["levels"]] == [("0", False), ("1", False), ("2", True), ("3", True)]
    assert group["helirec"]["frame_first"] == FIRST and group["helirec"]["frames"] == 130 and group["helirec"]["role"] == "projection"
    datasets = group["multiscales"][0]["datasets"]
    assert [d["coordinateTransformations"][0]["scale"] for d in datasets] == [[1, 1, 1], [1, 2, 2], [1, 4, 4], [1, 8, 8]]
    assert all(d["coordinateTransformations"][1]["translation"] == [0, 0, 0] for d in datasets)
    tile = json.loads((out / "T1-01" / "zarr.json").read_text())["attributes"]["helirec_tile"]
    assert tile["pixel_um"] == 1.129 and "source_master" not in tile

    for reduce in (2, 3):
        array = open_zarr(str(out / "T1-01" / "scan0005" / str(reduce)))
        assert array.shape == (130, ROWS >> reduce, COLS >> reduce) and array.dtype == np.uint16
        meta = json.loads((out / "T1-01" / "scan0005" / str(reduce) / "zarr.json").read_text())
        inner = meta["codecs"][0]["configuration"]
        assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [100, -(-(ROWS >> reduce) // 128) * 128, COLS >> reduce]
        assert inner["chunk_shape"] == [1, 128, COLS >> reduce]
        blosc = next(c for c in inner["codecs"] if c["name"] == "blosc")["configuration"]
        assert blosc["cname"] == "lz4" and blosc["shuffle"] == "bitshuffle"
        stored = np.asarray(array[:, :, :])
        assert np.array_equal(stored[:100], values[reduce]["scan0005", FIRST]) and np.array_equal(stored[100:], values[reduce]["scan0005", FIRST + 100])
        darks = open_zarr(str(out / "T1-01" / "scan0002" / str(reduce)))
        assert darks.dtype == np.float32 and np.array_equal(np.asarray(darks[:, :, :]), values[reduce]["scan0002", 0])
        # one shard per file
        assert len([p for p in (out / "T1-01" / "scan0005" / str(reduce) / "c").rglob("*") if p.is_file()]) == 2
    empty = open_zarr(str(out / "T1-01" / "scan0005" / "0"))
    assert empty.shape == (130, ROWS, COLS) and not (out / "T1-01" / "scan0005" / "0" / "c").exists()

    frames = out / "T1-01" / "scan0005"
    assert np.array_equal(np.asarray(open_zarr(str(frames / "original_frame_index"))[:]), np.arange(FIRST, FIRST + 130))
    assert np.allclose(np.asarray(open_zarr(str(frames / "angle_deg"))[:]), 0.036 * np.arange(FIRST, FIRST + 130))
    assert not (out / "T1-01" / "scan0002" / "angle_deg").exists()

    again = subprocess.run([TOOL, str(out), str(eighth), str(quarter)], capture_output=True, text=True)
    assert again.returncode == 0 and "0 files written and verified, 6 already stored" in again.stdout


def test_an_incomplete_file_is_left_out_and_reported(tmp_path):
    scale = tmp_path / "scale_0.125"
    _copy(scale, 3, np.random.default_rng(4), complete=False)
    out = tmp_path / "projections.zarr"
    run = subprocess.run([TOOL, str(out), str(scale)], capture_output=True, text=True)
    assert run.returncode == 1 and "2 files written and verified, 0 already stored, 1 not complete" in run.stdout
    assert "det1_0006.h5" in run.stderr
    stored = np.asarray(open_zarr(str(out / "T1-01" / "scan0005" / "3"))[:, :, :])
    assert stored[:100].any() and not stored[100:].any()
