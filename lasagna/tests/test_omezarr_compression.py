from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
import zarr

from lasagna.tiled_predict3d import (
    DEFAULT_OME_COMPRESSOR,
    _create_omezarr,
    _open_or_create_omezarr,
    omezarr_compressor_config,
)


EXPECTED_ZSTD = {
    "id": "blosc",
    "cname": "zstd",
    "clevel": 3,
    "shuffle": 1,
    "blocksize": 0,
}


def _metadata(root: Path, level: int) -> dict:
    return json.loads((root / str(level) / ".zarray").read_text(encoding="utf-8"))


def _tensorstore_array(root: Path, level: int):
    return ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(root / str(level))},
        },
        open=True,
    ).result()


def test_new_omezarr_levels_use_exact_shared_zstd_and_open_in_both_readers(tmp_path: Path):
    root = tmp_path / "fiber.ome.zarr"
    group = _create_omezarr(str(root), (9, 10, 11), 0, 3, 4, "presence")

    assert omezarr_compressor_config(DEFAULT_OME_COMPRESSOR) == EXPECTED_ZSTD
    for level in range(3):
        assert _metadata(root, level)["compressor"] == EXPECTED_ZSTD
        zarr_array = group[str(level)]
        assert tuple(zarr_array.shape) == tuple(_tensorstore_array(root, level).shape)

    group["0"][0:1, 0:1, 0:1] = np.asarray([[[17]]], dtype=np.uint8)
    assert int(_tensorstore_array(root, 0)[0, 0, 0].read().result()) == 17


def test_none_override_creates_uncompressed_v2_arrays(tmp_path: Path):
    root = tmp_path / "legacy-compatible.ome.zarr"
    _create_omezarr(str(root), (4, 4, 4), 0, 2, 4, "value", "none")

    assert _metadata(root, 0)["compressor"] is None
    assert _metadata(root, 1)["compressor"] is None


def test_resume_preserves_existing_compressor_and_reports_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    root = tmp_path / "existing.ome.zarr"
    _create_omezarr(str(root), (8, 8, 8), 0, 2, 4, "value", "none")

    _open_or_create_omezarr(
        str(root), (8, 8, 8), 0, 2, 4, "value", DEFAULT_OME_COMPRESSOR,
    )

    assert _metadata(root, 0)["compressor"] is None
    assert _metadata(root, 1)["compressor"] is None
    output = capsys.readouterr().out
    assert "preserving existing compressor(s)" in output
    assert "requested" in output
