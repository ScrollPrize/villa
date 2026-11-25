"""STEP=reduce writes a local OUTPUT_PATH given relative to the working directory, with or without ./"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import entrypoint  # noqa: E402
import processing  # noqa: E402


@pytest.mark.parametrize("output_path", ["pred.tif", "./pred.tif"])
def test_reduce_writes_output_path_relative_to_cwd(output_path, monkeypatch, tmp_path):
    parts = tmp_path / "partitions"
    for kind in ("pred", "count"):
        array = zarr.open(
            str(parts / f"mask_{kind}_part_000.zarr"), mode="w", shape=(8, 8), chunks=(4, 4),
            dtype=np.float32, compressor=None, zarr_format=2,
        )
        array[:] = 1.0
    monkeypatch.setattr(processing, "PARTITION_CACHE_DIR", str(tmp_path / "partition_cache"))
    model = f"reduce_output_path_test_{os.getpid()}"
    for key, value in {"MODEL": model, "START_LAYER": "0", "END_LAYER": "4", "STEP": "reduce",
                       "ZARR_OUTPUT_DIR": str(parts), "OUTPUT_PATH": output_path}.items():
        monkeypatch.setenv(key, value)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    try:
        entrypoint.run_reduce_step(entrypoint.parse_env())
    finally:
        Path(f"/tmp/prediction_{model}_00_04.tif").unlink(missing_ok=True)
    assert (cwd / "pred.tif").stat().st_size > 0
