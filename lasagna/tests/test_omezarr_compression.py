from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vc_reader.reader import open_zarr

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


def test_new_omezarr_levels_use_exact_shared_zstd_and_open(tmp_path: Path):
    root = tmp_path / "fiber.ome.zarr"
    group = _create_omezarr(str(root), (9, 10, 11), 0, 3, 4, "presence")

    assert omezarr_compressor_config(DEFAULT_OME_COMPRESSOR) == EXPECTED_ZSTD
    for level in range(3):
        assert _metadata(root, level)["compressor"] == EXPECTED_ZSTD
        assert tuple(group[str(level)].shape) == tuple(open_zarr(root / str(level)).shape)

    group["0"][0:1, 0:1, 0:1] = np.asarray([[[17]]], dtype=np.uint8)
    assert int(open_zarr(root / "0")[0, 0, 0]) == 17


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
