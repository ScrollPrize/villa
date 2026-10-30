"""reduce_partitions copies partitions to a local cache; a second reduce in the same process (or container) with other
partitions must not reuse the first run's copies."""

import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import processing  # noqa: E402


def _write_partition(out_dir, value, shape=(32, 32)):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = zarr.open(str(out_dir / "mask_pred_part_000.zarr"), mode="w", shape=shape, dtype="f4")
    pred[:] = value
    count = zarr.open(str(out_dir / "mask_count_part_000.zarr"), mode="w", shape=shape, dtype="f4")
    count[:] = 1.0


def _reduce(out_dir):
    tiles, _ = processing.reduce_partitions(str(out_dir), num_parts=1, pred_shape=(32, 32), tile_size=32)
    return np.concatenate([np.asarray(t).ravel() for t in tiles])


def test_second_reduce_reads_its_own_partitions(tmp_path):
    _write_partition(tmp_path / "run_a", 0.1)
    _write_partition(tmp_path / "run_b", 0.9)
    first = _reduce(tmp_path / "run_a")
    second = _reduce(tmp_path / "run_b")
    assert first.mean() < second.mean()
