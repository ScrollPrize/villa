"""Workers must receive the input *array* path, not the pyramid root.

`prepare()` resolved the OME-Zarr level correctly, but `generate_work_items()`
handed workers `config.input_zarr`. On any pyramid input -- including the output
these tasks themselves write -- the worker's `zarr.open(path)` then returned a
Group, and the `input_z[slices]` read raised
`TypeError: sequence item 0: expected str instance, tuple found`, so every chunk
failed and nothing was written. `--level N` was ignored for the same reason.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")


def _write_array(path, shape, chunks, value):
    """Create a filled zarr v2 array, on either zarr generation.

    zarr 2 writes v2 and has no ``zarr_format`` argument; zarr 3 defaults to v3,
    which these tasks cannot consume (they read ``Array.compressor``).
    """
    kwargs = {} if zarr.__version__.startswith("2.") else {"zarr_format": 2}
    array = zarr.open_array(str(path), mode="w", shape=shape, chunks=chunks,
                            dtype="u1", **kwargs)
    array[:] = value
    return array

from vesuvius.image_proc.run.zarr_tasks.tasks.scale import ScaleConfig, ScaleTask
from vesuvius.image_proc.run.zarr_tasks.tasks.threshold import (
    ThresholdConfig,
    ThresholdTask,
)
from vesuvius.image_proc.run.zarr_tasks.tasks.transpose import (
    TransposeConfig,
    TransposeTask,
)


@pytest.fixture
def pyramid(tmp_path):
    """A two-level OME-Zarr group, the v2 layout these tasks read and write."""
    root = tmp_path / "input.zarr"
    root.mkdir()
    # the layout create_level_dataset writes: a .zgroup beside the level arrays
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    _write_array(root / "0", (8, 8, 8), (4, 4, 4), 200)
    _write_array(root / "1", (4, 4, 4), (4, 4, 4), 100)
    return str(root)


@pytest.fixture
def bare_array(tmp_path):
    """A plain array input, which must keep working unchanged."""
    path = tmp_path / "plain.zarr"
    _write_array(path, (8, 8, 8), (4, 4, 4), 200)
    return str(path)


def _threshold_task(input_path, tmp_path, level=None):
    config = ThresholdConfig(input_zarr=input_path,
                             output_zarr=str(tmp_path / "out.zarr"),
                             num_workers=1, level=level)
    task = ThresholdTask(config)
    task.prepare()
    return task


def test_work_items_carry_an_array_path(pyramid, tmp_path):
    task = _threshold_task(pyramid, tmp_path)
    input_path = next(iter(task.generate_work_items()))[0]

    opened = zarr.open(input_path, mode="r")
    assert not isinstance(opened, zarr.Group)
    # the read the worker performs must succeed on that path
    assert opened[(slice(0, 4), slice(0, 4), slice(0, 4))].shape == (4, 4, 4)


def test_processing_a_chunk_writes_the_expected_values(pyramid, tmp_path):
    task = _threshold_task(pyramid, tmp_path)
    item = next(iter(task.generate_work_items()))

    task.process_item(item)

    written = zarr.open(str(tmp_path / "out.zarr" / "0"), mode="r")
    assert np.all(written[(slice(0, 4), slice(0, 4), slice(0, 4))] == 255)


def test_requested_level_reaches_the_worker(pyramid, tmp_path):
    task = _threshold_task(pyramid, tmp_path, level=1)
    input_path = next(iter(task.generate_work_items()))[0]

    assert input_path.endswith("/1")
    assert zarr.open(input_path, mode="r").shape == (4, 4, 4)


def test_bare_array_input_is_passed_through(bare_array, tmp_path):
    task = _threshold_task(bare_array, tmp_path)
    input_path = next(iter(task.generate_work_items()))[0]

    assert input_path == bare_array
    assert zarr.open(input_path, mode="r").shape == (8, 8, 8)


@pytest.mark.parametrize("task_cls, config_cls, extra", [
    (ScaleTask, ScaleConfig, {}),
    (TransposeTask, TransposeConfig, {}),
])
def test_sibling_tasks_resolve_the_level_too(task_cls, config_cls, extra,
                                             pyramid, tmp_path):
    config = config_cls(input_zarr=pyramid, output_zarr=str(tmp_path / "out.zarr"),
                        num_workers=1, **extra)
    task = task_cls(config)
    task.prepare()

    input_path = next(iter(task.generate_work_items()))[0]
    assert not isinstance(zarr.open(input_path, mode="r"), zarr.Group)
