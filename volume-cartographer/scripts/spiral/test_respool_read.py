"""Tests for respool_read: a synthetic sidecar, read back brick by brick.

No network and no CUDA: the fixture writes a sidecar in the layout that
``pack_resident_pools.py`` documents, so the reader is checked against the
format rather than against a particular published field.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from respool_read import ResidentPool

BRICK = (4, 4, 4)
GRID = (3, 3, 3)
SHAPE = tuple(g * b for g, b in zip(GRID, BRICK))


@pytest.fixture()
def sidecar(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A sidecar where every second brick is absent, plus the dense truth."""
    rng = np.random.default_rng(0)
    dense = np.zeros(SHAPE, np.uint8)
    table = np.zeros(GRID, np.int32)
    coords = [(-1, -1, -1)]
    pool = [np.zeros(int(np.prod(BRICK)), np.uint8)]     # row 0 is all zeros

    row = 1
    for gz in range(GRID[0]):
        for gy in range(GRID[1]):
            for gx in range(GRID[2]):
                if (gz + gy + gx) % 2:                    # leave a hole
                    continue
                brick = rng.integers(1, 255, BRICK, dtype=np.uint8)
                z, y, x = gz * BRICK[0], gy * BRICK[1], gx * BRICK[2]
                dense[z:z + BRICK[0], y:y + BRICK[1], x:x + BRICK[2]] = brick
                table[gz, gy, gx] = row
                coords.append((gz, gy, gx))
                pool.append(brick.ravel())
                row += 1

    root = tmp_path / 'field.respool_g1'
    root.mkdir()
    (root / 'meta.json').write_text(json.dumps({
        'format': 'respool', 'version': 2,
        'array_shape': list(SHAPE), 'brick_shape': list(BRICK),
        'grid_shape': list(GRID), 'rows': row, 'dtype': 'u1',
    }))
    np.save(root / 'table.npy', table)
    np.save(root / 'brick_coords.npy', np.array(coords, np.int32))
    (root / 'channel_0.u8').write_bytes(np.concatenate(pool).tobytes())
    return root, dense


def test_reads_whole_array(sidecar):
    root, dense = sidecar
    pool = ResidentPool(root)
    assert pool.array_shape == SHAPE
    got = pool.read(0, SHAPE[0], 0, SHAPE[1], 0, SHAPE[2])
    np.testing.assert_array_equal(got, dense)


def test_absent_bricks_read_as_fill(sidecar):
    root, _ = sidecar
    pool = ResidentPool(root)
    # brick (0, 0, 1) was skipped by the fixture
    got = pool.read(0, BRICK[0], 0, BRICK[1], BRICK[2], 2 * BRICK[2], fill=7)
    assert np.all(got == 7)


@pytest.mark.parametrize('box', [
    (1, 6, 2, 9, 0, 5),
    (0, 1, 0, 1, 0, 1),
    (5, SHAPE[0], 3, SHAPE[1], 7, SHAPE[2]),
])
def test_subblocks_match_dense(sidecar, box):
    root, dense = sidecar
    pool = ResidentPool(root)
    z0, z1, y0, y1, x0, x1 = box
    np.testing.assert_array_equal(pool.read(*box), dense[z0:z1, y0:y1, x0:x1])


def test_rejects_out_of_range(sidecar):
    root, _ = sidecar
    pool = ResidentPool(root)
    with pytest.raises(ValueError):
        pool.read(0, SHAPE[0] + 1, 0, 1, 0, 1)
    with pytest.raises(ValueError):
        pool.read(2, 2, 0, 1, 0, 1)


def test_rejects_non_respool(tmp_path: Path):
    root = tmp_path / 'not_a_sidecar'
    root.mkdir()
    (root / 'meta.json').write_text(json.dumps({'format': 'zarr'}))
    with pytest.raises(ValueError):
        ResidentPool(root)


def test_cache_dir_avoids_refetch(sidecar, tmp_path: Path):
    root, dense = sidecar
    cache = tmp_path / 'cache'
    first = ResidentPool(root, cache_dir=cache)
    first.read(0, SHAPE[0], 0, SHAPE[1], 0, SHAPE[2])
    assert any(cache.rglob('*.bin'))
    second = ResidentPool(root, cache_dir=cache)
    np.testing.assert_array_equal(
        second.read(0, SHAPE[0], 0, SHAPE[1], 0, SHAPE[2]), dense)


def test_occupancy(sidecar):
    root, _ = sidecar
    pool = ResidentPool(root)
    assert 0.4 < pool.occupancy() < 0.6
