"""Patch traversal order and DataLoader dispatch for sliding-window inference.

Sliding-window inference reads overlapping patches that are not aligned to the
input Zarr's chunk grid, so every chunk is needed by several patches. A chunk
cache (``Volume(cache=True)``, ``vesuvius.predict --chunk_cache_mb``) only
helps when the patches that share a chunk are read close together in time and
by the same process. Two things in the plain pipeline work against that:

* Positions are enumerated row-major (``for z: for y: for x``). A whole row of
  x is visited before the next y row, so by the time the traversal comes back
  to the chunks it just used they have been evicted.
* ``torch.utils.data.DataLoader`` hands consecutive batches to its workers
  round-robin. Neighbouring patches therefore land in different processes, and
  each worker's cache has to fetch the chunks its neighbours already hold.

``chunk_local_order`` sorts patch positions along a Morton (Z-order) curve over
the chunk indices of their start voxel, so patches that touch the same chunks
are adjacent in the sequence. ``WorkerContiguousBatchSampler`` emits batches in
an order that gives every DataLoader worker one contiguous run of that
sequence. Neither changes *which* patches are read or what the model sees for
each one: only the sequence, and which worker reads it.

Without a chunk cache the order changes nothing about what is fetched (every
chunk is downloaded once per patch that needs it either way), so the ``'auto'``
order keeps the plain row-major pipeline unless the cache is on; see
``resolve_patch_order``.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np
from torch.utils.data import Sampler

# 'chunk': Morton order over the input chunk grid and one contiguous run of it
#          per DataLoader worker.
# 'zyx':   row-major order and plain batching (the pipeline before this option).
# 'auto':  'chunk' when the chunk cache is enabled, 'zyx' otherwise.
PATCH_ORDERS = ("auto", "chunk", "zyx")

_MORTON_BITS = 21  # three 21-bit coordinates interleave into one 63-bit key


def resolve_patch_order(patch_order: str, cache_enabled: bool) -> str:
    """Resolve ``'auto'`` to ``'chunk'`` or ``'zyx'``; return an explicit order unchanged.

    The chunk order only pays off when a chunk cache can serve the patches that
    share a chunk; with the cache off every chunk is fetched once per patch
    whatever the sequence. ``'auto'`` is therefore ``'chunk'`` exactly when
    ``cache_enabled`` (``vesuvius.predict --chunk_cache_mb > 0``) and ``'zyx'``
    otherwise, so a run without the cache is sequence-for-sequence the plain
    pipeline.
    """
    if patch_order not in PATCH_ORDERS:
        raise ValueError(f"patch_order must be one of {PATCH_ORDERS}, got {patch_order!r}")
    if patch_order == "auto":
        return "chunk" if cache_enabled else "zyx"
    return patch_order


def _spread_bits(values: np.ndarray) -> np.ndarray:
    """Spread the low 21 bits of each value so two zero bits follow every bit."""
    v = values.astype(np.uint64) & np.uint64((1 << _MORTON_BITS) - 1)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


def morton_keys(chunk_indices: np.ndarray) -> np.ndarray:
    """Morton (Z-order) key of each ``(cz, cy, cx)`` chunk index row.

    Keys are injective for indices below ``2**21`` per axis, and the eight chunks
    of any aligned 2x2x2 block map to eight consecutive keys, which is what keeps
    chunk-sharing patches together once positions are sorted by key.
    """
    idx = np.asarray(chunk_indices, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError(f"chunk_indices must have shape (N, 3), got {idx.shape}")
    if idx.size and (idx.min() < 0 or idx.max() >= (1 << _MORTON_BITS)):
        raise ValueError(f"chunk indices must lie in [0, 2**{_MORTON_BITS}), got range {idx.min()}..{idx.max()}")
    return (
        (_spread_bits(idx[:, 0]) << np.uint64(2))
        | (_spread_bits(idx[:, 1]) << np.uint64(1))
        | _spread_bits(idx[:, 2])
    )


def chunk_local_order(positions: Sequence[Tuple[int, int, int]], chunk_shape: Sequence[int]) -> List[int]:
    """Indices that reorder ``positions`` along a Morton curve over chunk indices.

    ``positions`` are ``(z, y, x)`` patch start voxels and ``chunk_shape`` the
    spatial chunk shape of the array they are read from. Patches whose start
    voxels fall in the same chunk keep their row-major relative order, so the
    result is deterministic. The returned list is a permutation of
    ``range(len(positions))``; the set of positions is never changed.
    """
    n = len(positions)
    if n == 0:
        return []
    chunk = tuple(int(c) for c in chunk_shape)
    if len(chunk) != 3 or any(c <= 0 for c in chunk):
        raise ValueError(f"chunk_shape must be three positive ints, got {chunk_shape}")
    pos = np.asarray(positions, dtype=np.int64).reshape(n, 3)
    keys = morton_keys(pos // np.asarray(chunk, dtype=np.int64))
    # lexsort sorts by the last key first: Morton key, then z, y, x as tie-breakers.
    order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0], keys))
    return order.tolist()


def interleave_contiguous_blocks(batches: Sequence[Sequence[int]], num_workers: int) -> List[List[int]]:
    """Reorder ``batches`` so a round-robin consumer gives each worker a contiguous block.

    ``DataLoader`` assigns the ``i``-th batch it fetches to worker ``i % num_workers``.
    Splitting the batch list into ``num_workers`` contiguous blocks (the first
    ``len(batches) % num_workers`` blocks one batch longer) and emitting
    ``block[0][0], block[1][0], ..., block[0][1], block[1][1], ...`` therefore
    sends block ``w`` to worker ``w`` in its original order. Every batch is
    emitted exactly once.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    batches = [list(b) for b in batches]
    total = len(batches)
    if num_workers == 1 or total == 0:
        return batches
    base, extra = divmod(total, num_workers)
    blocks, start = [], 0
    for w in range(num_workers):
        size = base + (1 if w < extra else 0)
        blocks.append(batches[start:start + size])
        start += size
    out: List[List[int]] = []
    for r in range(base + (1 if extra else 0)):
        for block in blocks:
            if r < len(block):
                out.append(block[r])
    return out


class WorkerContiguousBatchSampler(Sampler):
    """Batches of consecutive indices, ordered so each DataLoader worker reads one contiguous run.

    Use as ``DataLoader(dataset, batch_sampler=WorkerContiguousBatchSampler(len(dataset), batch_size, num_workers))``
    in place of ``DataLoader(dataset, batch_size=batch_size, shuffle=False)``. The
    batches are the same ``[0..b), [b..2b), ...`` slices plain batching produces,
    only their order differs, so every index is still visited exactly once.
    """

    def __init__(self, num_items: int, batch_size: int, num_workers: int):
        if num_items < 0:
            raise ValueError(f"num_items must be >= 0, got {num_items}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.num_items = int(num_items)
        self.batch_size = int(batch_size)
        self.num_workers = max(1, int(num_workers))
        plain = [list(range(i, min(i + self.batch_size, self.num_items))) for i in range(0, self.num_items, self.batch_size)]
        self._batches = interleave_contiguous_blocks(plain, self.num_workers)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)
