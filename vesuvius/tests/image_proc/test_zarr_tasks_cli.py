"""zarr_tasks on OME-Zarr groups, and the CLI's task selection, overwrite and listing.

threshold, scale and transpose read the input's shape from level 0 but handed the
group root to their workers, so every run on an OME-Zarr group crashed (KeyError
or TypeError on slicing a Group, AttributeError on Group.ndim). Their outputs also
carried no coordinateTransformations, which OME-Zarr 0.4 requires on every level.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.data.utils import open_zarr, open_zarr_group
from vesuvius.image_proc.run.zarr_tasks import get_task
from vesuvius.image_proc.run.zarr_tasks.tasks.scale import ScaleTask, ScaleConfig
from vesuvius.image_proc.run.zarr_tasks.tasks.threshold import ThresholdConfig, ThresholdTask
from vesuvius.image_proc.run.zarr_tasks.tasks.transpose import TransposeConfig, TransposeTask
from vesuvius.image_proc.run.zarr_tasks.utils import read_level_scale


def _ome_group(tmp_path, data, scale=(7.91, 7.91, 7.91)):
    path = str(tmp_path / "in.zarr")
    root = open_zarr_group(path, mode="w")  # Zarr v2, like the published OME-Zarr volumes
    level0 = open_zarr(f"{path}/0", mode="w", shape=data.shape, chunks=(4, 8, 8),
                       dtype=data.dtype)
    level0[...] = data
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": list(scale)}]}],
    }]
    return path


def _volume():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(8, 16, 12), dtype=np.uint8)


def _datasets(path):
    root = zarr.open_group(path, mode="r")
    return root.attrs["multiscales"][0]


def test_threshold_runs_on_ome_group_and_keeps_voxel_size(tmp_path):
    data = _volume()
    src = _ome_group(tmp_path, data)
    out = str(tmp_path / "out.zarr")
    ThresholdTask(ThresholdConfig(input_zarr=src, output_zarr=out, num_workers=1,
                                  threshold=127, num_levels=2)).run()
    got = zarr.open_group(out, mode="r")["0"][...]
    np.testing.assert_array_equal(got, np.where(data > 127, 255, 0).astype(np.uint8))
    ms = _datasets(out)
    assert [a.get("unit") for a in ms["axes"]] == ["micrometer"] * 3
    scales = [d["coordinateTransformations"][0]["scale"] for d in ms["datasets"]]
    assert scales == [[7.91] * 3, [15.82] * 3]


def test_scale_runs_on_ome_group_and_divides_voxel_size(tmp_path):
    src = _ome_group(tmp_path, _volume(), scale=(8.0, 8.0, 8.0))
    out = str(tmp_path / "out.zarr")
    ScaleTask(ScaleConfig(input_zarr=src, output_zarr=out, num_workers=1,
                          scale_factor=0.5, num_levels=1)).run()
    assert zarr.open_group(out, mode="r")["0"].shape == (4, 8, 6)
    assert _datasets(out)["datasets"][0]["coordinateTransformations"][0]["scale"] == [16.0] * 3


def test_transpose_runs_on_ome_group_and_permutes_voxel_size(tmp_path):
    data = _volume()
    src = _ome_group(tmp_path, data, scale=(1.0, 2.0, 3.0))
    out = str(tmp_path / "out.zarr")
    TransposeTask(TransposeConfig(input_zarr=src, output_zarr=out, num_workers=1,
                                  transpose_order="xzy", num_levels=1)).run()
    got = zarr.open_group(out, mode="r")["0"][...]
    assert got.shape == (data.shape[2], data.shape[0], data.shape[1])
    # Stored z, y, x read as x, z, y: the new axis 0 is the old x (scale 3.0).
    assert _datasets(out)["datasets"][0]["coordinateTransformations"][0]["scale"] == [3.0, 1.0, 2.0]


def test_read_level_scale_on_bare_array_is_none(tmp_path):
    path = str(tmp_path / "bare.zarr")
    open_zarr(path, mode="w", shape=(2, 2, 2), dtype=np.uint8)
    assert read_level_scale(path) is None


def _cli(*args, stdin=subprocess.DEVNULL, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "vesuvius.image_proc.run.zarr_tasks", *args],
        capture_output=True, text=True, stdin=stdin, cwd=cwd, timeout=300,
    )


def test_task_equals_form_selects_the_task(tmp_path):
    src = _ome_group(tmp_path, _volume())
    out = str(tmp_path / "half.zarr")
    result = _cli(src, out, "--task=scale", "--scale-factor", "0.5", "-n", "1",
                  "--num-levels", "1")
    assert result.returncode == 0, result.stderr
    assert zarr.open_group(out, mode="r")["0"].shape == (4, 8, 6)


def test_existing_output_without_a_terminal_fails_clearly(tmp_path):
    src = _ome_group(tmp_path, _volume())
    out = tmp_path / "out.zarr"
    out.mkdir()
    result = _cli(src, str(out), "--task", "threshold", "-n", "1", "--num-levels", "1")
    assert result.returncode != 0
    assert "--yes" in result.stderr
    assert "EOFError" not in result.stderr


def test_yes_overwrites_existing_output(tmp_path):
    src = _ome_group(tmp_path, _volume())
    out = tmp_path / "out.zarr"
    out.mkdir()
    result = _cli(src, str(out), "--task", "threshold", "-n", "1", "--num-levels", "1", "--yes")
    assert result.returncode == 0, result.stderr
    assert zarr.open_group(str(out), mode="r")["0"].shape == (8, 16, 12)


def test_list_tasks_shows_a_summary_per_task():
    result = _cli("--list-tasks")
    assert result.returncode == 0
    line = next(l for l in result.stdout.splitlines() if l.strip().startswith("threshold"))
    assert get_task("threshold").__doc__.split(".")[0] in line
