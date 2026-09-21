"""create_level_dataset has to work on every zarr the package accepts (pyproject: zarr>=2.18.7,<4).

Under zarr 3 it raised AttributeError on zarr.NestedDirectoryStore before touching any data, which
took the threshold, transpose, scale and recompress tasks down with it (#1670). CI runs this file on
both ends of that range, zarr 2.18.7 and 3.2.1, so each case is checked on each.
"""
import json

import numpy as np
from numcodecs import Blosc

from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset


def _level(tmp_path, **kwargs):
    args = dict(
        root_group_path=str(tmp_path / "vol.zarr"),
        level_name="0",
        shape=(8, 8, 8),
        chunks=(4, 4, 4),
        dtype=np.uint8,
        compressor=Blosc(cname="zstd", clevel=1),
    )
    args.update(kwargs)
    return create_level_dataset(**args)


def _chunk_files(level):
    return sorted(p.relative_to(level).as_posix() for p in level.rglob("*") if p.is_file() and not p.name.startswith("."))


def test_writes_and_reads_back(tmp_path):
    arr = _level(tmp_path)
    data = np.arange(512, dtype=np.uint8).reshape(8, 8, 8)
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)


def test_keeps_the_nested_zarr_v2_layout(tmp_path):
    """NestedDirectoryStore was only ever there for nested chunk directories. Keep that layout."""
    arr = _level(tmp_path)
    arr[:] = 1
    level = tmp_path / "vol.zarr" / "0"
    meta = json.loads((level / ".zarray").read_text())
    assert meta["zarr_format"] == 2
    assert meta["dimension_separator"] == "/"
    assert "0/0/0" in _chunk_files(level)
    assert not (level / "0.0.0").exists()
    assert json.loads((tmp_path / "vol.zarr" / ".zgroup").read_text())["zarr_format"] == 2


def test_chunks_that_hold_only_the_fill_value_are_not_written(tmp_path):
    arr = _level(tmp_path)
    arr[:] = 0
    level = tmp_path / "vol.zarr" / "0"
    assert _chunk_files(level) == []
    arr[0:4, 0:4, 0:4] = 7
    assert _chunk_files(level) == ["0/0/0"]


def test_overwrite_replaces_an_existing_level(tmp_path):
    _level(tmp_path)[:] = 5
    arr = _level(tmp_path, overwrite=True)
    assert int(np.asarray(arr[:]).max()) == 0
