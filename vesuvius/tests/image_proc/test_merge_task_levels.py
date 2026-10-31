"""The merge task creates its output levels on every zarr the package accepts (pyproject: zarr>=2.18.7,<4).

MergeTask.prepare and MergeTask._generate_pyramid_levels called Group.create_dataset, which zarr 3
removed, so `vesuvius.zarr_tasks --task merge` crashed under zarr 3 before writing level 0 (#2240).
They now go through create_level_dataset like the other pyramid tasks. CI runs this file on zarr
2.18.7 and 3.2.1.
"""
import json

import numpy as np
import zarr
from numcodecs import Blosc

from vesuvius.image_proc.run.zarr_tasks.tasks.merge import MergeConfig, MergeTask
from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset


def _write_input(path, data):
    # A zarr v2 group with a level "0", written the same way on zarr 2 and zarr 3.
    arr = create_level_dataset(str(path), "0", data.shape, (4, 4, 4), data.dtype, Blosc(cname="zstd", clevel=1))
    arr[...] = data


def _run_merge(tmp_path, num_levels):
    a = np.zeros((8, 8, 8), np.uint8)
    b = np.zeros((8, 8, 8), np.uint8)
    a[:, :, :4] = 10  # left half from input 1
    b[:, :4, :] = 20  # front half from input 2; overlaps the left half, max wins there
    _write_input(tmp_path / "a.zarr", a)
    _write_input(tmp_path / "b.zarr", b)
    out = tmp_path / "out.zarr"
    task = MergeTask(MergeConfig(
        input_zarr=str(tmp_path / "a.zarr"),
        input2_zarr=str(tmp_path / "b.zarr"),
        output_zarr=str(out),
        num_workers=1,
        num_levels=num_levels,
    ))
    task.prepare()
    task._results = [task.process_item(item) for item in task.generate_work_items()]
    task.finalize()
    return out, np.maximum(a, b)


def test_level0_is_the_elementwise_maximum(tmp_path):
    out, expected = _run_merge(tmp_path, num_levels=1)
    np.testing.assert_array_equal(zarr.open(str(out / "0"), mode="r")[...], expected)


def test_pyramid_levels_are_created_and_halved(tmp_path):
    out, expected = _run_merge(tmp_path, num_levels=3)
    shapes = {level: zarr.open(str(out / level), mode="r").shape for level in ("0", "1", "2")}
    assert shapes == {"0": (8, 8, 8), "1": (4, 4, 4), "2": (2, 2, 2)}
    # one 2x2x2 block of level 0 that is all 20 stays 20 after averaging
    assert zarr.open(str(out / "1"), mode="r")[0, 0, 3] == 20
    assert zarr.open(str(out / "1"), mode="r")[0, 3, 0] == 10
    assert zarr.open(str(out / "1"), mode="r")[0, 3, 3] == 0


def test_output_is_zarr_v2_with_multiscales_metadata(tmp_path):
    out, _ = _run_merge(tmp_path, num_levels=2)
    assert json.loads((out / ".zgroup").read_text()) == {"zarr_format": 2}
    assert not (out / "zarr.json").exists()
    for level in ("0", "1"):
        assert (out / level / ".zarray").exists()
    datasets = json.loads((out / ".zattrs").read_text())["multiscales"][0]["datasets"]
    assert [d["path"] for d in datasets] == ["0", "1"]


def test_prepare_replaces_a_stale_output(tmp_path):
    out, expected = _run_merge(tmp_path, num_levels=2)
    stale = out / "7"
    stale.mkdir()
    (stale / "leftover").write_text("x")
    out, expected = _run_merge(tmp_path, num_levels=2)
    assert not stale.exists()
    np.testing.assert_array_equal(zarr.open(str(out / "0"), mode="r")[...], expected)
