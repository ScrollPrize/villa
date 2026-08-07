"""Patch traversal order.

Row-major traversal walks an entire row of x before moving on, so by the time
it reaches the neighbouring row the chunks it needs have been evicted from any
cache and are fetched again. Ordering positions along a Z-order curve over
chunk indices keeps patches that share chunks close together in time.

The invariant these tests protect: reordering must change only the sequence,
never which patches exist.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

from vesuvius.data.vc_dataset import VCDataset


# --------------------------------------------------------------------------
# The curve itself
# --------------------------------------------------------------------------

def test_morton_interleaves_bits() -> None:
    assert VCDataset._morton(0, 0, 0) == 0
    # one bit per axis lands on its own slot
    assert VCDataset._morton(1, 0, 0) == 0b001
    assert VCDataset._morton(0, 1, 0) == 0b010
    assert VCDataset._morton(0, 0, 1) == 0b100


def test_morton_is_injective_on_a_small_grid() -> None:
    """Distinct chunk triples must not collide, or ordering would be arbitrary."""
    keys = {VCDataset._morton(a, b, c)
            for a in range(8) for b in range(8) for c in range(8)}
    assert len(keys) == 8 ** 3


def test_morton_keeps_neighbours_close() -> None:
    """Points inside a 2x2x2 block occupy a contiguous run of keys."""
    block = [VCDataset._morton(a, b, c)
             for a in range(2) for b in range(2) for c in range(2)]
    assert sorted(block) == list(range(min(block), min(block) + 8))


# --------------------------------------------------------------------------
# Reordering preserves the work
# --------------------------------------------------------------------------

def _fake_dataset(positions, chunks, order):
    """A VCDataset shell carrying just what reordering touches."""
    ds = VCDataset.__new__(VCDataset)
    ds.all_positions = list(positions)
    ds.traversal_order = order
    ds.verbose = False
    ds._input_chunk_shape = lambda: chunks
    return ds


GRID = [(z, y, x)
        for z in range(0, 512, 96)
        for y in range(0, 512, 96)
        for x in range(0, 512, 96)]


def test_reordering_preserves_the_patch_set() -> None:
    ds = _fake_dataset(GRID, (128, 128, 128), 'morton')
    ds._reorder_positions_for_locality()

    assert sorted(ds.all_positions) == sorted(GRID)
    assert len(ds.all_positions) == len(GRID)
    assert ds.all_positions != GRID          # order did change


def test_zyx_leaves_order_untouched() -> None:
    ds = _fake_dataset(GRID, (128, 128, 128), 'zyx')
    ds._reorder_positions_for_locality()
    assert ds.all_positions == GRID


def test_unknown_chunk_shape_falls_back_to_row_major() -> None:
    """Reordering must never fail the run just because chunks are unknown."""
    ds = _fake_dataset(GRID, None, 'morton')
    ds._reorder_positions_for_locality()
    assert ds.all_positions == GRID


def test_empty_position_list_is_safe() -> None:
    ds = _fake_dataset([], (128, 128, 128), 'morton')
    ds._reorder_positions_for_locality()
    assert ds.all_positions == []


# --------------------------------------------------------------------------
# The point of the exercise: fewer refetches
# --------------------------------------------------------------------------

def _fetches_under_lru(positions, patch, chunk, capacity):
    """Chunk fetches a traversal costs with an LRU cache of `capacity` chunks."""
    cache: OrderedDict = OrderedDict()
    fetches = 0
    for z, y, x in positions:
        for cz in range(z // chunk, (z + patch - 1) // chunk + 1):
            for cy in range(y // chunk, (y + patch - 1) // chunk + 1):
                for cx in range(x // chunk, (x + patch - 1) // chunk + 1):
                    key = (cz, cy, cx)
                    if key in cache:
                        cache.move_to_end(key)
                        continue
                    fetches += 1
                    cache[key] = True
                    if len(cache) > capacity:
                        cache.popitem(last=False)
    return fetches


@pytest.mark.parametrize("capacity", [16, 32, 64])
def test_zorder_costs_fewer_fetches_than_row_major(capacity: int) -> None:
    ds = _fake_dataset(GRID, (128, 128, 128), 'morton')
    ds._reorder_positions_for_locality()

    row_major = _fetches_under_lru(GRID, 192, 128, capacity)
    zorder = _fetches_under_lru(ds.all_positions, 192, 128, capacity)

    assert zorder < row_major, (
        f"capacity {capacity}: Z-order {zorder} should beat row-major {row_major}"
    )


def test_both_orders_reach_the_same_chunks() -> None:
    """Ordering changes when chunks are fetched, never which ones."""
    ds = _fake_dataset(GRID, (128, 128, 128), 'morton')
    ds._reorder_positions_for_locality()

    def touched(positions):
        out = set()
        for z, y, x in positions:
            for cz in range(z // 128, (z + 191) // 128 + 1):
                for cy in range(y // 128, (y + 191) // 128 + 1):
                    for cx in range(x // 128, (x + 191) // 128 + 1):
                        out.add((cz, cy, cx))
        return out

    assert touched(GRID) == touched(ds.all_positions)
