"""In-place recompression must work on both supported zarr versions.

The task builds a numcodecs compressor. zarr 3 rejects ``compressor=`` on the
v3 array ``zarr.open`` creates by default, so ``--inplace`` dies before it
copies a chunk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from vesuvius.data.utils import open_zarr
from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import (
    RecompressConfig,
    RecompressTask,
)


def test_inplace_recompression_preserves_values(tmp_path: Path) -> None:
    path = tmp_path / "vol.zarr"
    data = np.arange(64, dtype="u1").reshape(4, 4, 4) + 1
    open_zarr(
        path=str(path),
        mode="w",
        shape=(4, 4, 4),
        chunks=(2, 2, 2),
        dtype="u1",
    )[:] = data

    RecompressTask(
        RecompressConfig(
            input_zarr=str(path),
            output_zarr=None,
            num_workers=1,
            inplace=True,
        )
    ).run()

    meta = json.loads((path / ".zarray").read_text())
    assert meta["zarr_format"] == 2
    assert meta["compressor"]["cname"] == "zstd"
    assert np.array_equal(zarr.open(str(path), mode="r")[:], data)
