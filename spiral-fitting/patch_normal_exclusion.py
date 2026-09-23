"""Cached Euclidean dilation of patch coverage, sampled with one bit lookup.

Distances are between export-cell centers. No spatial downsampling is used.
Empty/full bricks share bitmap rows 0/1; only mixed bricks need storage.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def pack_presence_bits(present):
    """Pack (rows, cells) booleans into native int64 words, low bit first."""
    rows, cells = present.shape
    padded = present
    if cells % 64:
        padded = np.zeros((rows, ((cells + 63) // 64) * 64), dtype=bool)
        padded[:, :cells] = present
    return np.packbits(padded, axis=1, bitorder='little').view('<i8').astype(np.int64, copy=False)


def _dilate_tile(tile, table, bits, brick, shape, radius, tile_bricks):
    import edt

    start = np.array(tile) * tile_bricks
    end = np.minimum(start + tile_bricks, table.shape)
    origin = start * brick
    halo = int(np.ceil(radius))
    source_lo = np.maximum(origin - halo, 0)
    source_hi = np.minimum(end * brick + halo, shape)
    brick_lo = source_lo // brick
    brick_hi = (source_hi + brick - 1) // brick
    rows = table[tuple(slice(a, b) for a, b in zip(brick_lo, brick_hi))]
    target_grid = end - start
    target_shape = target_grid * brick
    if not rows.any():
        return start, end, None, None
    # Unpack only this tile and its halo, including coverage in adjacent bricks.
    words = np.ascontiguousarray(bits[rows].astype('<i8', copy=False))
    occupied = np.unpackbits(words.view(np.uint8), axis=-1, bitorder='little',
                             count=int(np.prod(brick)))
    occupied = occupied.reshape(*rows.shape, *brick).transpose(0, 3, 1, 4, 2, 5)
    occupied = occupied.reshape((brick_hi - brick_lo) * brick)
    relative_lo = source_lo - brick_lo * brick
    relative_hi = source_hi - brick_lo * brick
    occupied = occupied[tuple(slice(a, b) for a, b in zip(relative_lo, relative_hi))]
    if not occupied.any():
        return start, end, None, None
    # black_border=False is essential: the outside of a tile is not coverage.
    distance_sq = edt.edtsq(np.ascontiguousarray(occupied == 0),
                           black_border=False, parallel=1)
    valid_shape = np.minimum(end * brick, shape) - origin
    offset = origin - source_lo
    near = np.zeros(target_shape, dtype=bool)
    near[tuple(slice(0, n) for n in valid_shape)] = distance_sq[
        tuple(slice(a, a + n) for a, n in zip(offset, valid_shape))] <= radius**2
    near = near.reshape(target_grid[0], brick[0], target_grid[1], brick[1],
                        target_grid[2], brick[2]).transpose(0, 2, 4, 1, 3, 5)
    near = near.reshape(-1, int(np.prod(brick)))
    full = near.all(axis=1)
    mixed = near.any(axis=1) & ~full
    # Local ids: 0 empty, 1 full, >=2 indexes this tile's mixed rows.
    local_ids = full.astype(np.int32)
    local_ids[mixed] = np.arange(2, 2 + int(mixed.sum()), dtype=np.int32)
    return start, end, local_ids.reshape(target_grid), pack_presence_bits(near[mixed])


def build_exclusion_mask(table, bits, brick, shape, radius, *, z_roi=None,
                         tile_bricks=8, workers=4, progress_callback=None):
    """Bounded-memory CPU dilation; returns an int32 table and int64 bitmaps."""
    from scipy.ndimage import maximum_filter

    if not np.isfinite(radius) or radius < 0:
        raise ValueError('patch exclusion radius must be finite and nonnegative')
    brick, shape = np.array(brick), np.array(shape)
    if tile_bricks <= 0 or workers <= 0:
        raise ValueError('tile_bricks and workers must be positive')
    tile_shape = (np.array(table.shape) + tile_bricks - 1) // tile_bricks
    padded = np.zeros(tile_shape * tile_bricks, dtype=bool)
    padded[tuple(slice(0, n) for n in table.shape)] = table != 0
    active = padded.reshape(tile_shape[0], tile_bricks, tile_shape[1], tile_bricks,
                            tile_shape[2], tile_bricks).any(axis=(1, 3, 5))
    del padded
    tile_halo = np.ceil(radius / (brick * tile_bricks)).astype(int)
    active = maximum_filter(active, size=tuple(2 * tile_halo + 1), mode='constant', cval=0)
    if z_roi is not None:
        z = np.arange(len(active)) * tile_bricks * brick[0]
        active &= ((z < z_roi[1]) & (z + tile_bricks * brick[0] > z_roi[0]))[:, None, None]
    tiles = np.argwhere(active)
    result = np.zeros_like(table, dtype=np.int32)
    words = (int(np.prod(brick)) + 63) // 64
    chunks = [np.zeros((1, words), dtype=np.int64), np.full((1, words), -1, dtype=np.int64)]
    next_row = 2

    def work(tile):
        return _dilate_tile(tile, table, bits, brick, shape, radius, tile_bricks)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # buffersize prevents queued completed tiles from exhausting host RAM.
        for i, (start, end, local_ids, packed) in enumerate(
                executor.map(work, tiles, buffersize=workers * 2)):
            if local_ids is not None:
                local_ids[local_ids >= 2] += next_row - 2
                result[tuple(slice(a, b) for a, b in zip(start, end))] = local_ids
                if len(packed):
                    chunks.append(packed)
                    next_row += len(packed)
            if progress_callback and (i % 32 == 0 or i + 1 == len(tiles)):
                progress_callback(i + 1, len(tiles), 'building patch-normal exclusion mask')
    return result, np.concatenate(chunks)


def exclusion_cache_path(cache_directory, sidecar_dir, *, shape, brick, z_roi, radius):
    if cache_directory is None:
        return None
    root = Path(sidecar_dir).resolve()
    names = ['meta.json', 'table.npy', 'brick_coords.npy'] + [f'channel_{i}.u8' for i in range(3)]
    signature = dict(version=1, source=str(root), shape=list(shape), brick=list(brick),
                     z_roi=z_roi, radius=float(radius),
                     files=[(name, (root / name).stat().st_size, (root / name).stat().st_mtime_ns)
                            for name in names])
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return Path(cache_directory) / 'patch-normal-exclusion' / f'{digest}.npz'


def load_exclusion_mask(path, grid_shape, words):
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        table, bits = data['table'], data['bits']
    if (table.shape != tuple(grid_shape) or table.dtype != np.int32
            or bits.dtype != np.int64 or bits.ndim != 2 or bits.shape[1] != words
            or len(bits) < 2 or bits[0].any() or not (bits[1] == -1).all()
            or table.min() < 0 or table.max() >= len(bits)):
        raise ValueError(f'{path}: invalid patch-normal exclusion cache')
    return table, bits


def save_exclusion_mask(path, table, bits):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent builders may replace the same deterministic artifact safely;
    # readers never see a partially written archive.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
            tmp = Path(f.name)
            np.savez(f, table=table, bits=bits)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


class PatchExclusionMask:
    def __init__(self, table, bits, device):
        self.table = torch.from_numpy(table).to(device)
        self.bits = torch.from_numpy(bits).to(device)
        self.nbytes = table.nbytes + bits.nbytes

    def gather_at(self, brick_indices, word, bit):
        """Reuse the normal gather's indices; no radius search or host work."""
        row = self.table[brick_indices[:, 0], brick_indices[:, 1], brick_indices[:, 2]].long()
        return ((self.bits[row, word] >> bit) & 1).bool()

    def close(self):
        self.table = self.bits = None
