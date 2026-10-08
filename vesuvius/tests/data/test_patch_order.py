"""Chunk-local patch order: the same patches and data, in a sequence a chunk cache can serve.

VCDataset enumerates sliding-window patches row-major (z, then y, then x). Patches
overlap and are not aligned to the input's chunk grid, so each chunk is read by
many patches; a chunk cache only avoids the repeat fetches if those patches come
close together and through the same process. These tests pin down that the
Morton ordering changes nothing but the sequence, and that the sequence really is
cheaper for an LRU cache.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict

import numpy as np
import pytest
import torch
import zarr

from vesuvius.data.patch_order import (
    PATCH_ORDERS,
    WorkerContiguousBatchSampler,
    chunk_local_order,
    interleave_contiguous_blocks,
    morton_keys,
)
from vesuvius.data.vc_dataset import VCDataset
from vesuvius.utils.models.helpers import compute_steps_for_sliding_window

PATCH = (24, 24, 24)
CHUNK = (16, 16, 16)
SHAPE = (96, 80, 112)
STEP = 0.5


@pytest.fixture(scope="module")
def volume_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("patch_order") / "vol.zarr"
    arr = zarr.open(str(path), mode="w", shape=SHAPE, chunks=CHUNK, dtype="u1")
    arr[:] = np.random.default_rng(7).integers(0, 255, size=SHAPE, dtype=np.uint8)
    return str(path)


def _dataset(volume_path: str, **kwargs) -> VCDataset:
    return VCDataset(
        input_path=volume_path,
        patch_size=PATCH,
        mode="infer",
        skip_empty_patches=False,
        normalization_scheme="none",
        return_as_type="np.float32",
        verbose=False,
        **kwargs,
    )


def _row_major_positions() -> list[tuple[int, int, int]]:
    grids = [compute_steps_for_sliding_window(SHAPE[a], PATCH[a], STEP) for a in range(3)]
    return [(z, y, x) for z in grids[0] for y in grids[1] for x in grids[2]]


def _chunks_touched(pos) -> set[tuple[int, int, int]]:
    ranges = [range(pos[a] // CHUNK[a], (pos[a] + PATCH[a] - 1) // CHUNK[a] + 1) for a in range(3)]
    return set(itertools.product(*ranges))


def _lru_misses(positions, capacity: int) -> int:
    """Chunk fetches an LRU cache of `capacity` chunks makes while reading `positions` in order."""
    cache: OrderedDict = OrderedDict()
    misses = 0
    for pos in positions:
        for key in sorted(_chunks_touched(pos)):
            if key in cache:
                cache.move_to_end(key)
                continue
            misses += 1
            cache[key] = True
            if len(cache) > capacity:
                cache.popitem(last=False)
    return misses


# --- the Morton key and the sort ------------------------------------------------------

def test_morton_keys_are_injective_and_keep_2x2x2_blocks_together() -> None:
    idx = np.array(list(itertools.product(range(8), repeat=3)), dtype=np.int64)
    keys = morton_keys(idx)
    assert len(set(keys.tolist())) == len(idx)
    # The eight chunks of any aligned 2x2x2 block have eight consecutive keys.
    for base in itertools.product(range(0, 8, 2), repeat=3):
        block = np.array([tuple(b + d for b, d in zip(base, delta)) for delta in itertools.product((0, 1), repeat=3)])
        block_keys = sorted(morton_keys(block).tolist())
        assert block_keys == list(range(block_keys[0], block_keys[0] + 8))


def test_morton_keys_reject_bad_shapes_and_ranges() -> None:
    with pytest.raises(ValueError):
        morton_keys(np.zeros((3, 2), dtype=np.int64))
    with pytest.raises(ValueError):
        morton_keys(np.array([[0, -1, 0]]))
    with pytest.raises(ValueError):
        morton_keys(np.array([[1 << 21, 0, 0]]))


def test_chunk_local_order_is_a_permutation_with_row_major_ties() -> None:
    positions = _row_major_positions()
    order = chunk_local_order(positions, CHUNK)
    assert sorted(order) == list(range(len(positions)))
    # Patches whose start voxels share a chunk keep their row-major relative order.
    seen: dict = {}
    for rank, i in enumerate(order):
        key = tuple(positions[i][a] // CHUNK[a] for a in range(3))
        if key in seen:
            assert seen[key] < i
        seen[key] = i
    assert chunk_local_order([], CHUNK) == []
    with pytest.raises(ValueError):
        chunk_local_order(positions, (16, 16))


# --- VCDataset --------------------------------------------------------------------

def test_default_order_is_chunk_and_chunk_shape_is_detected(volume_path: str) -> None:
    ds = _dataset(volume_path)
    assert ds.patch_order == "chunk"
    assert ds.chunk_shape == CHUNK


def test_zyx_order_is_the_row_major_enumeration(volume_path: str) -> None:
    assert _dataset(volume_path, patch_order="zyx").all_positions == _row_major_positions()


def test_chunk_order_has_the_same_patches_in_a_different_sequence(volume_path: str) -> None:
    chunk = _dataset(volume_path, patch_order="chunk").all_positions
    zyx = _dataset(volume_path, patch_order="zyx").all_positions
    assert sorted(chunk) == sorted(zyx)
    assert len(chunk) == len(set(chunk))
    assert chunk != zyx


def test_chunk_order_is_deterministic(volume_path: str) -> None:
    assert _dataset(volume_path).all_positions == _dataset(volume_path).all_positions


def test_each_patch_reads_the_same_data_in_either_order(volume_path: str) -> None:
    chunk = _dataset(volume_path, patch_order="chunk")
    zyx = _dataset(volume_path, patch_order="zyx")
    by_position = {zyx.all_positions[i]: zyx[i] for i in range(len(zyx))}
    for i in range(len(chunk)):
        item = chunk[i]
        reference = by_position[item["pos"]]
        assert item["index"] == i
        assert torch.equal(item["data"], reference["data"])


@pytest.mark.parametrize("capacity", [16, 32, 64])
def test_chunk_order_costs_an_lru_cache_fewer_fetches_than_row_major(volume_path: str, capacity: int) -> None:
    # A 24^3 patch over 16^3 chunks touches up to 27 chunks, and one row of the
    # row-major sweep touches far more than these caches hold (the regime a
    # remote whole-scroll run is in), so the sequence decides how often a chunk
    # is fetched again. Both orders touch exactly the same chunks.
    chunk = _dataset(volume_path, patch_order="chunk").all_positions
    zyx = _dataset(volume_path, patch_order="zyx").all_positions
    assert set().union(*map(_chunks_touched, chunk)) == set().union(*map(_chunks_touched, zyx))
    assert _lru_misses(chunk, capacity) < _lru_misses(zyx, capacity)


def test_unknown_chunk_shape_falls_back_to_row_major(volume_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VCDataset, "_spatial_chunk_shape", lambda self: None)
    ds = _dataset(volume_path, patch_order="chunk")
    assert ds.chunk_shape is None
    assert ds.all_positions == _row_major_positions()


def test_invalid_patch_order_is_rejected(volume_path: str) -> None:
    with pytest.raises(ValueError, match="patch_order"):
        _dataset(volume_path, patch_order="hilbert")
    assert "chunk" in PATCH_ORDERS and "zyx" in PATCH_ORDERS


def test_parts_and_bbox_still_partition_the_same_patches(volume_path: str) -> None:
    bbox = (20, 70, 10, 60, 30, 100)
    single = _dataset(volume_path, bbox=bbox)
    union: list = []
    for part_id in range(3):
        union.extend(_dataset(volume_path, bbox=bbox, num_parts=3, part_id=part_id).all_positions)
    assert sorted(union) == sorted(single.all_positions)
    assert sorted(single.all_positions) == sorted(_dataset(volume_path, bbox=bbox, patch_order="zyx").all_positions)


# --- worker dispatch ------------------------------------------------------------------

@pytest.mark.parametrize("total", [0, 1, 5, 8, 9, 10])
@pytest.mark.parametrize("num_workers", [1, 2, 3, 4])
def test_interleave_gives_each_round_robin_worker_a_contiguous_block(total: int, num_workers: int) -> None:
    batches = [[i] for i in range(total)]
    out = interleave_contiguous_blocks(batches, num_workers)
    assert sorted(b[0] for b in out) == list(range(total))
    # DataLoader gives batch i to worker i % num_workers; every worker's share must be
    # one ascending, gap-free run of the original sequence.
    runs = [[out[i][0] for i in range(len(out)) if i % num_workers == w] for w in range(num_workers)]
    flat = [v for run in runs for v in run]
    assert flat == list(range(total))
    assert max(len(r) for r in runs) - min(len(r) for r in runs) <= 1 if total else True


def test_sampler_emits_the_plain_batches_exactly_once() -> None:
    sampler = WorkerContiguousBatchSampler(num_items=10, batch_size=3, num_workers=2)
    batches = list(sampler)
    assert len(sampler) == len(batches) == 4
    assert sorted(batches) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert batches == [[0, 1, 2], [6, 7, 8], [3, 4, 5], [9]]


def test_sampler_with_one_worker_or_no_items_is_plain_order() -> None:
    assert list(WorkerContiguousBatchSampler(5, 2, 1)) == [[0, 1], [2, 3], [4]]
    assert list(WorkerContiguousBatchSampler(0, 2, 4)) == []
    with pytest.raises(ValueError):
        WorkerContiguousBatchSampler(5, 0, 2)
