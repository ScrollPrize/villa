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
from collections import OrderedDict, defaultdict

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader, Dataset, get_worker_info

from vesuvius.data.patch_order import (
    PATCH_ORDERS,
    WorkerContiguousBatchSampler,
    chunk_local_order,
    interleave_contiguous_blocks,
    morton_keys,
    resolve_patch_order,
)
from vesuvius.data.vc_dataset import VCDataset
from vesuvius.data.volume import Volume
from vesuvius.utils.models.helpers import compute_steps_for_sliding_window

_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3
requires_zarr_v3 = pytest.mark.skipif(not _ZARR_V3, reason="the chunk cache needs zarr>=3")

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


@pytest.fixture(scope="module")
def multiscale_group_path(tmp_path_factory) -> str:
    """A multiscale group with levels "0" (CHUNK chunks) and "1" (8^3 chunks), zarr v2 format.

    Group creation is version-branched as in test_volume_multiscale.py.
    """
    path = str(tmp_path_factory.mktemp("patch_order_group") / "group.zarr")
    full = np.random.default_rng(3).integers(0, 255, size=SHAPE, dtype=np.uint8)
    half = tuple(s // 2 for s in SHAPE)
    if _ZARR_V3:
        root = zarr.open_group(store=path, mode="w", zarr_format=2)
        a0 = root.create_array("0", shape=SHAPE, chunks=CHUNK, dtype="uint8")
        a1 = root.create_array("1", shape=half, chunks=(8, 8, 8), dtype="uint8")
    else:
        root = zarr.open_group(path, mode="w")
        a0 = root.create_dataset("0", shape=SHAPE, chunks=CHUNK, dtype="uint8")
        a1 = root.create_dataset("1", shape=half, chunks=(8, 8, 8), dtype="uint8")
    a0[:] = full
    a1[:] = full[::2, ::2, ::2]
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}, {"path": "1"}]}]
    return path


@pytest.fixture(scope="module")
def channels_first_path(tmp_path_factory) -> str:
    """A (C, Z, Y, X) array whose chunks carry a leading channel axis."""
    path = tmp_path_factory.mktemp("patch_order_4d") / "vol4d.zarr"
    shape = (2, 64, 64, 64)
    arr = zarr.open(str(path), mode="w", shape=shape, chunks=(1, *CHUNK), dtype="u1")
    arr[:] = np.random.default_rng(5).integers(0, 255, size=shape, dtype=np.uint8)
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


# --- resolving 'auto' -------------------------------------------------------------

def test_resolve_patch_order() -> None:
    assert PATCH_ORDERS == ("auto", "chunk", "zyx")
    assert resolve_patch_order("auto", cache_enabled=False) == "zyx"
    assert resolve_patch_order("auto", cache_enabled=True) == "chunk"
    assert resolve_patch_order("chunk", cache_enabled=False) == "chunk"
    assert resolve_patch_order("zyx", cache_enabled=True) == "zyx"
    with pytest.raises(ValueError, match="patch_order"):
        resolve_patch_order("hilbert", cache_enabled=True)


# --- VCDataset --------------------------------------------------------------------

def test_default_is_auto_which_is_row_major_without_the_cache(volume_path: str) -> None:
    # Without a chunk cache the order changes nothing about what is fetched, so the
    # default leaves the pipeline exactly as it was: row-major positions, and no
    # chunk-shape lookup (a metadata probe on remote volumes).
    ds = _dataset(volume_path)
    assert ds.patch_order == "auto"
    assert ds.effective_patch_order == "zyx"
    assert ds.chunk_shape is None
    assert ds.all_positions == _row_major_positions()


@requires_zarr_v3
def test_auto_is_the_chunk_order_once_the_cache_is_on(volume_path: str) -> None:
    ds = _dataset(volume_path, cache=True, cache_size_mb=1)
    assert ds.effective_patch_order == "chunk"
    assert ds.chunk_shape == CHUNK
    assert ds.all_positions == _dataset(volume_path, patch_order="chunk").all_positions


def test_chunk_order_does_not_need_the_cache(volume_path: str) -> None:
    ds = _dataset(volume_path, patch_order="chunk")
    assert ds.effective_patch_order == "chunk"
    assert ds.chunk_shape == CHUNK
    assert ds.all_positions != _row_major_positions()


@requires_zarr_v3
def test_zyx_order_stays_row_major_with_the_cache(volume_path: str) -> None:
    ds = _dataset(volume_path, patch_order="zyx", cache=True, cache_size_mb=1)
    assert ds.effective_patch_order == "zyx"
    assert ds.chunk_shape is None
    assert ds.all_positions == _row_major_positions()


def test_chunk_shape_comes_from_level_0_of_a_multiscale_group(
    multiscale_group_path: str, volume_path: str
) -> None:
    ds = _dataset(multiscale_group_path, patch_order="chunk")
    assert isinstance(ds.volume.data, zarr.Group)
    assert ds.chunk_shape == CHUNK  # level "0", not the 8^3 chunks of level "1"
    assert ds.all_positions == _dataset(volume_path, patch_order="chunk").all_positions


def test_chunk_shape_drops_the_channel_axis_of_a_4d_array(channels_first_path: str) -> None:
    ds = _dataset(channels_first_path, patch_order="chunk")
    assert len(ds.input_shape) == 4
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
    first = _dataset(volume_path, patch_order="chunk").all_positions
    assert first == _dataset(volume_path, patch_order="chunk").all_positions


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


def test_failed_level0_lookup_warns_and_yields_no_chunk_shape(volume_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # A transient metadata error while resolving the level-0 array must not abort
    # the run, nor pass silently: the lookup warns and reports no chunk shape, and
    # the order then stays row-major (test_unknown_chunk_shape_falls_back_to_row_major).
    # Volume.shape() resolves the level too, so the failure is injected after the
    # dataset is built.
    ds = _dataset(volume_path, patch_order="zyx")
    assert isinstance(ds.volume, Volume)

    def unavailable(idx=0):
        raise OSError("metadata fetch failed")

    monkeypatch.setattr(ds.volume, "_level", unavailable)
    with pytest.warns(UserWarning, match="level-0 array.*metadata fetch failed"):
        assert ds._level0_array() is None
    with pytest.warns(UserWarning, match="level-0 array"):
        assert ds._spatial_chunk_shape() is None


def test_invalid_patch_order_is_rejected(volume_path: str) -> None:
    with pytest.raises(ValueError, match="patch_order"):
        _dataset(volume_path, patch_order="hilbert")


def test_parts_and_bbox_still_partition_the_same_patches(volume_path: str) -> None:
    bbox = (20, 70, 10, 60, 30, 100)
    single = _dataset(volume_path, bbox=bbox, patch_order="chunk")
    union: list = []
    for part_id in range(3):
        part = _dataset(volume_path, bbox=bbox, num_parts=3, part_id=part_id, patch_order="chunk")
        union.extend(part.all_positions)
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


class _WorkerIdDataset(Dataset):
    """Reports which DataLoader worker fetched each index. Module-level so it pickles under spawn."""

    def __init__(self, num_items: int):
        self.num_items = num_items

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> dict:
        info = get_worker_info()
        return {"index": idx, "worker": -1 if info is None else info.id}


def _collate_keep_index(batch: list) -> dict:
    return {"index": [item["index"] for item in batch], "worker": [item["worker"] for item in batch]}


def test_real_multi_worker_loader_gives_each_worker_one_contiguous_run() -> None:
    # The sampler relies on DataLoader handing batch i to worker i % num_workers
    # (its in_order=True default). Iterate a real two-worker loader, built the way
    # Inferer builds it, and check what each worker actually fetched.
    num_items, batch_size, num_workers = 30, 4, 2
    loader = DataLoader(
        _WorkerIdDataset(num_items),
        batch_sampler=WorkerContiguousBatchSampler(num_items, batch_size, num_workers),
        num_workers=num_workers,
        collate_fn=_collate_keep_index,
    )
    fetched: dict = defaultdict(list)
    seen: list = []
    for batch in loader:
        assert len(set(batch["worker"])) == 1, batch  # one worker fetches a whole batch
        fetched[batch["worker"][0]].extend(batch["index"])
        seen.extend(batch["index"])
    assert sorted(seen) == list(range(num_items))  # every index exactly once
    assert set(fetched) == set(range(num_workers))  # both workers took part
    for run in fetched.values():
        assert run == list(range(run[0], run[0] + len(run)))  # ascending, gap-free
    assert [i for w in range(num_workers) for i in fetched[w]] == list(range(num_items))
