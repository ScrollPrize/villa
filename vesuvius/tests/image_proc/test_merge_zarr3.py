"""merge must run end to end under zarr 3 (issue #2240).

MergeTask.prepare and _generate_pyramid_levels called Group.create_dataset,
which zarr 3 removed, so the task crashed with AttributeError before writing
anything.
"""
import numpy as np
import zarr

from vesuvius.data.utils import open_zarr_group
from vesuvius.image_proc.run.zarr_tasks.tasks.merge import MergeConfig, MergeTask


def _write_input(path, level0):
    root = open_zarr_group(str(path), mode="w")
    arr = root.zeros(name="0", shape=level0.shape, chunks=(4, 4, 4), dtype=level0.dtype)
    arr[...] = level0


def test_merge_runs_and_takes_max(tmp_path):
    a = np.zeros((8, 8, 8), dtype=np.uint8)
    b = np.zeros((8, 8, 8), dtype=np.uint8)
    a[:4] = 10
    b[2:6] = 20
    _write_input(tmp_path / "a.zarr", a)
    _write_input(tmp_path / "b.zarr", b)
    out = tmp_path / "out.zarr"

    MergeTask(MergeConfig(
        input_zarr=str(tmp_path / "a.zarr"),
        input2_zarr=str(tmp_path / "b.zarr"),
        output_zarr=str(out),
        num_workers=1,
        num_levels=2,
    )).run()

    root = zarr.open_group(str(out), mode="r")
    np.testing.assert_array_equal(root["0"][...], np.maximum(a, b))
    assert root["1"].shape == (4, 4, 4)
    assert root["0"].dtype == np.uint8
