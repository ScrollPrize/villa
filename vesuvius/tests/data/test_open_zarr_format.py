"""open_zarr must create stores the rest of the pipeline can read.

blend_logits validates that logits stores are zarr_format 2, and every writer in
the package builds numcodecs compressors, which zarr 3 rejects on a v3 array.
"""

from __future__ import annotations

import json
from pathlib import Path

import numcodecs
import pytest

from vesuvius.data.utils import open_zarr


def _stored_format(path: Path) -> int:
    if (path / ".zarray").exists():
        return json.loads((path / ".zarray").read_text())["zarr_format"]
    return json.loads((path / "zarr.json").read_text())["zarr_format"]


def test_created_array_defaults_to_zarr_format_2(tmp_path: Path) -> None:
    path = tmp_path / "out.zarr"
    open_zarr(path=str(path), mode="w", shape=(4, 8, 8), chunks=(1, 8, 8), dtype="f2")
    assert _stored_format(path) == 2


def test_numcodecs_compressor_accepted(tmp_path: Path) -> None:
    path = tmp_path / "compressed.zarr"
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.blosc.SHUFFLE)
    arr = open_zarr(path=str(path), mode="w", shape=(4, 8, 8), chunks=(1, 8, 8),
                    dtype="f2", compressor=compressor)
    arr[0] = 1.0
    assert _stored_format(path) == 2
    assert open_zarr(path=str(path), mode="r")[0].max() == 1.0


def test_redundant_overwrite_kwarg_is_accepted(tmp_path: Path) -> None:
    # finalize_outputs passes overwrite=True alongside mode='w'; zarr 3 derives
    # overwrite from the mode and rejects the duplicate.
    path = tmp_path / "overwritten.zarr"
    open_zarr(path=str(path), mode="w", shape=(4, 8, 8), chunks=(1, 8, 8),
              dtype="u1", overwrite=True)
    assert _stored_format(path) == 2


def test_contradictory_overwrite_false_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        open_zarr(path=str(tmp_path / "no.zarr"), mode="w", shape=(4, 8, 8),
                  chunks=(1, 8, 8), dtype="u1", overwrite=False)


def test_explicit_zarr_format_3_still_possible(tmp_path: Path) -> None:
    path = tmp_path / "v3.zarr"
    open_zarr(path=str(path), mode="w", shape=(4, 8, 8), chunks=(1, 8, 8),
              dtype="f2", zarr_format=3)
    assert _stored_format(path) == 3
