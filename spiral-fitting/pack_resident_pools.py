"""Pack sparse Lasagna or signed patch-normal Zarr stores into flat brick pools.

A sidecar holds every *occupied* brick of a (chunk-sparse) uint8 zarr array as
one flat pool file per channel, plus an int32 brick-grid table mapping brick
coordinates to pool rows (0 = absent, served by a reserved all-zero row).
Loading it is a single sequential read per channel instead of tens of
thousands of demand-fills through tensorstore, and a fully resident pool never
evicts, so the per-step LRU thrash disappears entirely.

CT masking: the giant outside-the-scroll mask region (CT volume reads 0) can
be byte-occupied in a source store. Passing ``--ct`` zeroes every voxel whose
(ratio-mapped) CT voxel is 0 while packing; combined with a sub-chunk brick
size this drops the mask skin from the pool, and the zeros read back as
no-data through the existing sampling contract.

Sidecar layout (``<zarr>.respool_g<group>[_pair]/``):
  meta.json         format/version, shapes, brick grid, sources, ct_mask info
  table.npy         int32 (gz, gy, gx); 0 = absent, i >= 1 = pool row i
  brick_coords.npy  int32 (rows, 3) brick coords per row; row 0 = -1
  channel_<i>.u8    uint8 (rows, brick_voxels) raw pool; row 0 = zeros

Usage (defaults match the fit_spiral input conventions):
  python pack_resident_pools.py /path/to/lasagna_inputs \
      --ct /path/to/s1_ds2.zarr --ct-group 2 --verify 2000

Signed normals encoded as three uint8 channels (no presence/alignment output):
  python pack_resident_pools.py /path/to/patch_normals_zarr \
      --what signed_normals --signed-brick 16 --io-threads 2 --verify 2000
This writes version 4 with explicit channel layouts and normal encoding metadata.
It is a storage format only; the existing fitter expects version 2 uint8 pools.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from zarr_chunk_io import (
    _chunk_path, _list_chunk_keys, _load_delta3d, _make_chunk_reader,
)

RESPOOL_FORMAT_VERSION = 2
TYPED_RESPOOL_FORMAT_VERSION = 3
SIGNED_NORMAL_U8_FORMAT_VERSION = 4
NORMAL_U8_ENCODING = {
    'name': 'signed_unit_vector_u8',
    'components': ['nx', 'ny', 'nz'],
    'encode': 'round_ties_to_even(127 * clip(component, -1, 1)) + 128',
    'decode': '(float32(value) - 128) / 127; normalize the three-vector',
    'offset': 128, 'scale': 127, 'renormalize_after_decode': True,
    'component_zero': 128, 'component_min': 1, 'component_max': 255,
    'missing_vector': [0, 0, 0], 'presence': 'any component != 0',
}


def encode_signed_normals_u8(normals, valid=None):
    """Quantize unit XYZ (or ZYX) vectors; reserve [0,0,0] for missing data."""
    normals = np.asarray(normals, dtype=np.float32)
    if normals.shape[-1:] != (3,):
        raise ValueError('normals must have three components')
    valid = np.any(normals != 0, axis=-1) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != normals.shape[:-1]:
        raise ValueError('normal validity shape mismatch')
    selected = normals[valid]
    if (not np.isfinite(selected).all()
            or np.any(np.abs(np.linalg.norm(selected, axis=-1) - 1) > 1e-4)):
        raise ValueError('occupied normals must be finite unit vectors')
    result = np.zeros(normals.shape, dtype=np.uint8)
    result[valid] = (np.rint(np.clip(selected, -1, 1) * 127) + 128).astype(np.uint8)
    return result


def decode_signed_normals_u8(encoded):
    """Return (normalized vectors, valid); retain the input component order."""
    encoded = np.asarray(encoded)
    if encoded.dtype != np.uint8 or encoded.shape[-1:] != (3,):
        raise ValueError('encoded normals must be uint8 three-vectors')
    valid = np.any(encoded != 0, axis=-1)
    vectors = (encoded.astype(np.float32) - 128) / 127
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(valid & (np.any(encoded == 0, axis=-1) | (length[..., 0] == 0))):
        raise ValueError('invalid encoded normal triplet')
    return np.where(valid[..., None], vectors / np.maximum(length, 1e-12), 0), valid


def sidecar_path(zarr_path: str, group: str, *, pair: bool = False) -> str:
    """Canonical sidecar directory for a store; the pair suffix marks the
    two-channel nx+ny pool that lives next to the nx zarr."""
    suffix = f'.respool_g{group}' + ('_pair' if pair else '')
    return str(zarr_path).rstrip('/') + suffix


def _read_zarray_meta(array_dir: Path, *, typed=False) -> dict:
    with open(array_dir / '.zarray') as f:
        meta = json.load(f)
    if meta['dtype'] not in (('|u1', '<f4') if typed else ('|u1',)):
        expected = '|u1 or <f4' if typed else '|u1'
        raise ValueError(f'{array_dir}: expected {expected} dtype, got {meta["dtype"]!r}')
    if meta.get('fill_value', 0) != 0:
        raise ValueError(f'{array_dir}: only zero-fill arrays are supported')
    if meta.get('order', 'C') != 'C':
        raise ValueError(f'{array_dir}: expected C order')
    if meta.get('filters'):
        raise ValueError(f'{array_dir}: filters are not supported')
    return meta




class CtMasker:
    """Voxel-level ``CT == 0`` mask, ratio-mapped onto a target array grid.

    ``ratio`` is target voxels per CT voxel, per axis (integer, e.g. 1 for the
    group-4 lasagna stores against s1_ds2 group 2). Decoded CT chunks are kept in a small LRU because consecutive
    target chunks in key order revisit the same CT chunks.
    """

    def __init__(self, ct_zarr_path: str, ct_group: str, target_shape,
                 cache_chunks: int = 512):
        self.array_dir = Path(ct_zarr_path) / str(ct_group)
        meta = _read_zarray_meta(self.array_dir)
        self.shape = tuple(meta['shape'])
        self.chunks = tuple(meta['chunks'])
        self.separator = meta.get('dimension_separator', '.')
        self._reader = _make_chunk_reader(
            meta['compressor'], int(np.prod(self.chunks)))
        self.ratio = tuple(
            int(round(ts / cs)) for ts, cs in zip(target_shape, self.shape))
        for axis, (ts, cs, r) in enumerate(
                zip(target_shape, self.shape, self.ratio)):
            if r < 1 or not (cs - 1 <= (ts + r - 1) // r <= cs + 1):
                raise ValueError(
                    f'CT shape {self.shape} does not map onto target shape '
                    f'{tuple(target_shape)} with an integer per-axis ratio '
                    f'(axis {axis}: ratio {r})')
        self._keys = _list_chunk_keys(self.array_dir, self.separator)
        self._cache: OrderedDict[tuple, np.ndarray | None] = OrderedDict()
        self._cache_chunks = int(cache_chunks)
        self._lock = threading.Lock()

    def _ct_chunk(self, key: tuple[int, int, int]) -> np.ndarray | None:
        """Decoded CT chunk, or None when absent (all fill / all masked)."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        value = None
        if key in self._keys:
            value = self._reader(
                _chunk_path(self.array_dir, key, self.separator)
            ).reshape(self.chunks)
        with self._lock:
            self._cache[key] = value
            while len(self._cache) > self._cache_chunks:
                self._cache.popitem(last=False)
        return value

    def ct_region(self, lo, hi) -> np.ndarray:
        """CT voxels for the half-open CT-grid box [lo, hi); OOB reads 0."""
        lo = [max(0, v) for v in lo]
        hi = [min(s, v) for s, v in zip(self.shape, hi)]
        out = np.zeros([max(0, h - l) for l, h in zip(lo, hi)], dtype=np.uint8)
        if out.size == 0:
            return out
        c0 = [l // c for l, c in zip(lo, self.chunks)]
        c1 = [(h - 1) // c for h, c in zip(hi, self.chunks)]
        for cz in range(c0[0], c1[0] + 1):
            for cy in range(c0[1], c1[1] + 1):
                for cx in range(c0[2], c1[2] + 1):
                    chunk = self._ct_chunk((cz, cy, cx))
                    if chunk is None:
                        continue
                    base = (cz * self.chunks[0], cy * self.chunks[1],
                            cx * self.chunks[2])
                    src = tuple(
                        slice(max(l, b) - b, min(h, b + c) - b)
                        for l, h, b, c in zip(lo, hi, base, self.chunks))
                    dst = tuple(
                        slice(max(l, b) - l, min(h, b + c) - l)
                        for l, h, b, c in zip(lo, hi, base, self.chunks))
                    out[dst] = chunk[src]
        return out

    def occupied_block(self, origin_zyx, block_shape) -> np.ndarray:
        """Boolean ``CT != 0`` for a target-grid box, upsampled by ratio.

        ``origin_zyx`` must be aligned to the ratio (chunk origins are).
        """
        for o, r in zip(origin_zyx, self.ratio):
            if o % r:
                raise ValueError(f'block origin {origin_zyx} not aligned to CT ratio {self.ratio}')
        lo = [o // r for o, r in zip(origin_zyx, self.ratio)]
        hi = [-(-(o + b) // r) for o, b, r in zip(origin_zyx, block_shape, self.ratio)]
        region = self.ct_region(lo, hi) != 0
        padded = np.zeros([h - l for l, h in zip(lo, hi)], dtype=bool)
        padded[tuple(slice(0, s) for s in region.shape)] = region
        for axis, r in enumerate(self.ratio):
            if r > 1:
                padded = padded.repeat(r, axis=axis)
        return padded[tuple(slice(0, b) for b in block_shape)]

    def ct_value_at(self, target_zyx) -> int:
        ct = [t // r for t, r in zip(target_zyx, self.ratio)]
        region = self.ct_region(ct, [c + 1 for c in ct])
        return int(region[0, 0, 0]) if region.size else 0

    def describe(self) -> dict:
        return {
            'path': str(self.array_dir.parent),
            'group': self.array_dir.name,
            'ratio': list(self.ratio),
        }


def pack_arrays(
    array_dirs: list[str],
    out_dir: str,
    *,
    label: str,
    brick_shape: tuple[int, int, int] | None = None,
    ct_masker: CtMasker | None = None,
    io_threads: int = 16,
    force: bool = False,
    progress_callback=None,
    channel_labels: list[str] | None = None,
    validity_channel: int | None = None,
    source_metadata: dict | None = None,
    encode_normals_u8: bool = False,
) -> str:
    """Pack same-geometry Zarr arrays, with uint8 version 2 as the default.

    ``channel_labels`` enables version 3 and per-channel uint8/float32 dtypes.
    ``validity_channel`` uses presence to mask values and select occupied bricks.
    ``encode_normals_u8`` writes version 4: three quantized signed components,
    using source presence without storing a separate validity channel.

    ``brick_shape`` defaults to the source chunk shape and must divide it;
    bricks that are entirely zero across all channels (after CT masking) are
    dropped from the pool.
    """
    out_dir = Path(out_dir)
    if encode_normals_u8 and (channel_labels != ['nx', 'ny', 'nz', 'presence'] or validity_channel != 3):
        raise ValueError('uint8 normal encoding requires nx, ny, nz and source presence')
    packed_labels = channel_labels[:3] if encode_normals_u8 else channel_labels
    version = (SIGNED_NORMAL_U8_FORMAT_VERSION if encode_normals_u8 else
               TYPED_RESPOOL_FORMAT_VERSION if channel_labels is not None else RESPOOL_FORMAT_VERSION)
    packed_sources = array_dirs[:3] if encode_normals_u8 else array_dirs
    meta_path = out_dir / 'meta.json'
    if meta_path.exists() and not force:
        if channel_labels is not None:
            previous = json.loads(meta_path.read_text())
            if (previous.get('version') != version
                    or [c['name'] for c in previous.get('channel_layout', [])] != packed_labels
                    or previous['channels'] != [str(Path(d).resolve()) for d in packed_sources]
                    or previous['brick_shape'] != list(brick_shape if brick_shape is not None
                                                       else previous['source_chunk_shape'])
                    or previous.get('source_metadata') != source_metadata):
                raise ValueError('existing typed pool settings differ; choose a new output or --force')
            open_pool(str(out_dir))  # Check file sizes before accepting a completed sidecar.
        print(f'{label}: {out_dir} already exists, skipping (--force to rebuild)')
        return str(out_dir)

    if not array_dirs or io_threads < 1:
        raise ValueError('at least one array and one I/O thread are required')
    typed = channel_labels is not None
    if typed and (len(channel_labels) != len(array_dirs)
                  or len(set(channel_labels)) != len(channel_labels)):
        raise ValueError('channel labels must be unique and match the array count')
    if validity_channel is not None and not 0 <= validity_channel < len(array_dirs):
        raise ValueError('invalid validity channel index')
    array_dirs = [Path(d).resolve() for d in array_dirs]
    metas = [_read_zarray_meta(d, typed=typed) for d in array_dirs]
    dtypes = [np.dtype(m['dtype']) for m in metas]
    if encode_normals_u8 and [d.str for d in dtypes] != ['<f4', '<f4', '<f4', '|u1']:
        raise ValueError('normal encoding requires float32 components and uint8 source presence')
    packed_dtypes = [np.dtype('u1')] * 3 if encode_normals_u8 else dtypes
    if validity_channel is not None and dtypes[validity_channel] != np.dtype('u1'):
        raise ValueError('presence must be uint8')
    shape = tuple(metas[0]['shape'])
    chunks = tuple(metas[0]['chunks'])
    if len(shape) != 3 or len(chunks) != 3 or any(s <= 0 for s in (*shape, *chunks)):
        raise ValueError('source arrays must have three positive shape/chunk dimensions')
    for d, m in zip(array_dirs, metas):
        if tuple(m['shape']) != shape or tuple(m['chunks']) != chunks:
            raise ValueError(
                f'{d}: shape/chunks {m["shape"]}/{m["chunks"]} differ from '
                f'{array_dirs[0]}: {shape}/{chunks}')
    separators = [m.get('dimension_separator', '.') for m in metas]
    chunk_voxels = int(np.prod(chunks))
    brick = tuple(chunks) if brick_shape is None else tuple(brick_shape)
    if len(brick) != 3 or any(b <= 0 or int(b) != b for b in brick):
        raise ValueError('brick shape must contain three positive integers')
    brick = tuple(map(int, brick))
    if any(c % b for c, b in zip(chunks, brick)):
        raise ValueError(f'brick {brick} must divide the source chunks {chunks}')
    sub = tuple(c // b for c, b in zip(chunks, brick))
    subs_per_chunk = int(np.prod(sub))
    brick_voxels = int(np.prod(brick))
    grid = tuple((s + b - 1) // b for s, b in zip(shape, brick))

    per_channel_keys = [
        _list_chunk_keys(d, sep) for d, sep in zip(array_dirs, separators)
    ]
    keys = sorted(set().union(*per_channel_keys))
    if any(any(k < 0 or k*c >= s for k, c, s in zip(key, chunks, shape)) for key in keys):
        raise ValueError('source chunk key outside array shape')
    if not keys:
        raise RuntimeError(f'{label}: no chunk files found under {array_dirs}')
    for d, channel_keys in zip(array_dirs, per_channel_keys):
        missing = len(keys) - len(channel_keys)
        if missing:
            print(f'{label}: WARNING {d.name} is missing {missing} of '
                  f'{len(keys)} union chunks; they read as zero (no-data)')

    bytes_per_voxel = sum(d.itemsize for d in packed_dtypes)
    upper_gib = (len(keys) * subs_per_chunk + 1) * brick_voxels * bytes_per_voxel / 1024 ** 3
    print(f'{label}: packing {len(keys):,} chunks of {chunks} into bricks of '
          f'{brick} x{len(packed_dtypes)} channel(s), <= {upper_gib:.1f} GiB, '
          f'ct_mask={"on" if ct_masker else "off"} -> {out_dir}')
    if progress_callback is not None:
        progress_callback(0, len(keys), 'discovering occupied bricks')

    readers = [
        _make_chunk_reader(m['compressor'], chunk_voxels * dtype.itemsize)
        for m, dtype in zip(metas, dtypes)
    ]

    def read_chunk(key):
        """Returns (key, kept sub-brick index array, per-channel (n, vox))."""
        channels = []
        for d, sep, reader, channel_keys, dtype in zip(
                array_dirs, separators, readers, per_channel_keys, dtypes):
            if key in channel_keys:
                channels.append(reader(_chunk_path(d, key, sep)).view(dtype).reshape(chunks))
            else:
                channels.append(None)
        origin = tuple(k * c for k, c in zip(key, chunks))
        # Padding outside the logical array is never an occupied voxel.
        inside = np.ones(chunks, dtype=bool)
        for axis, (o, c, s) in enumerate(zip(origin, chunks, shape)):
            index_shape = [1, 1, 1]
            index_shape[axis] = c
            inside &= (np.arange(c).reshape(index_shape) + o < s)
        if validity_channel is not None:
            presence = channels[validity_channel]
            if presence is None:
                raise ValueError(f'{label}: missing presence chunk {key}')
            if not np.isin(presence, [0, 255]).all():
                raise ValueError(f'{label}: invalid presence encoding in chunk {key}')
            inside &= presence != 0
            if inside.any():
                if any(a is None for a in channels):
                    raise ValueError(f'{label}: missing channel in occupied chunk {key}')
                if any(not np.isfinite(a[inside]).all() for a in channels):
                    raise ValueError(f'{label}: nonfinite occupied data in chunk {key}')
        if ct_masker is not None:
            inside &= ct_masker.occupied_block(origin, chunks)
        channels = [None if a is None else np.where(inside, a, 0) for a in channels]
        if encode_normals_u8:
            # Presence is consumed during packing, but never written to a pool.
            vectors = np.stack([np.zeros(chunks, np.float32) if a is None else a
                                for a in channels[:3]], axis=-1)
            encoded = encode_signed_normals_u8(vectors, inside)
            channels = [encoded[..., i] for i in range(3)]
        split = [
            None if a is None else np.ascontiguousarray(
                a.reshape(sub[0], brick[0], sub[1], brick[1], sub[2], brick[2])
                .transpose(0, 2, 4, 1, 3, 5)
            ).reshape(subs_per_chunk, brick_voxels)
            for a in channels
        ]
        keep = np.zeros(subs_per_chunk, dtype=bool)
        for a in (split if validity_channel is None or encode_normals_u8 else [split[validity_channel]]):
            if a is not None:
                keep |= a.any(axis=1)
        keep_idx = np.flatnonzero(keep)
        return key, keep_idx, [
            (np.zeros((len(keep_idx), brick_voxels), dtype=dtype)
             if a is None else a[keep_idx])
            for a, dtype in zip(split, packed_dtypes)
        ]

    started = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    if meta_path.exists():
        meta_path.unlink()  # invalidate a stale sidecar while rebuilding

    rows = 1
    coords_path = out_dir / 'brick_coords.bin.tmp'
    sub_grid = np.stack(np.unravel_index(np.arange(subs_per_chunk), sub), axis=1)
    filenames = [f'channel_{i}.{"f32" if dtype.kind == "f" else "u8"}'
                 for i, dtype in enumerate(packed_dtypes)]
    if force:
        for stale in out_dir.glob('channel_*'):
            if (stale.stem.removeprefix('channel_').isdigit()
                    and stale.suffix in ('.u8', '.f32') and stale.name not in filenames):
                stale.unlink()
    channel_files = [open(out_dir / name, 'wb', buffering=1024 * 1024) for name in filenames]
    try:
        for f, dtype in zip(channel_files, packed_dtypes):
            f.write(bytes(brick_voxels * dtype.itemsize))  # reserved zero brick
        with coords_path.open('wb') as coordinates, ThreadPoolExecutor(io_threads) as executor:
            np.asarray([[-1, -1, -1]], dtype='<i4').tofile(coordinates)
            done = 0
            for key, keep_idx, channel_bricks in executor.map(
                    read_chunk, keys, buffersize=io_threads * 2):
                base = np.multiply(key, sub)
                rows += len(keep_idx)
                if rows > np.iinfo(np.int32).max:
                    raise ValueError('pool exceeds int32 row-index capacity')
                (sub_grid[keep_idx] + base).astype('<i4').tofile(coordinates)
                for f, bricks in zip(channel_files, channel_bricks):
                    f.write(bricks.tobytes())
                done += 1
                if done % 2000 == 0 or done == len(keys):
                    if progress_callback is not None:
                        progress_callback(
                            done, len(keys),
                            f'{rows - 1:,} bricks kept')
                    elapsed = time.perf_counter() - started
                    print(f'{label}: {done:,}/{len(keys):,} chunks, '
                          f'{rows - 1:,} bricks kept '
                          f'({done / len(keys) * 100:.0f}%, {elapsed:.0f}s)',
                          flush=True)
    finally:
        for f in channel_files:
            f.close()

    coords = np.lib.format.open_memmap(out_dir / 'brick_coords.npy', mode='w+',
                                      dtype='<i4', shape=(rows, 3))
    table = np.lib.format.open_memmap(out_dir / 'table.npy', mode='w+', dtype='<i4', shape=grid)
    # Explicitly clear a possibly reused file without allocating the full table.
    for z in range(grid[0]):
        table[z] = 0
    with coords_path.open('rb') as stream:
        for start in range(0, rows, 100_000):
            end = min(rows, start + 100_000)
            block = np.fromfile(stream, dtype='<i4', count=(end-start)*3).reshape(-1, 3)
            coords[start:end] = block
            begin = max(start, 1)
            valid = block[begin-start:]
            table[valid[:, 0], valid[:, 1], valid[:, 2]] = np.arange(begin, end, dtype=np.int32)
    coords.flush(); table.flush()
    del coords, table
    coords_path.unlink()
    pool_gib = rows * brick_voxels * bytes_per_voxel / 1024 ** 3
    meta = {
        'format': 'respool',
        'version': version,
        'array_shape': list(shape),
        'source_chunk_shape': list(chunks),
        'brick_shape': list(brick),
        'grid_shape': list(grid),
        'rows': rows,
        'channels': [str(d) for d in (array_dirs[:3] if encode_normals_u8 else array_dirs)],
        'channel_names': [d.parent.name + '/' + d.name
                          for d in (array_dirs[:3] if encode_normals_u8 else array_dirs)],
        'dtype': 'per_channel' if typed else 'u1',
        'ct_mask': ct_masker.describe() if ct_masker is not None else None,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    if typed:
        meta.update(channel_layout=[dict(name=name, file=filename, dtype=dtype.str,
                    shape=[rows, brick_voxels], order='C')
                    for name, filename, dtype in zip(packed_labels, filenames, packed_dtypes)],
                    axis_order='zyx', local_index='(z * brick_y + y) * brick_x + x',
                    absent_row=0, validity_channel=validity_channel,
                    occupancy='presence != 0' if validity_channel is not None else 'any nonzero channel',
                    source_metadata=source_metadata)
    if encode_normals_u8:
        meta.update(normal_encoding=NORMAL_U8_ENCODING, validity_channel=None,
                    source_validity_array=str(array_dirs[validity_channel]),
                    occupancy='any encoded normal component != 0')
    with open(meta_path.with_suffix('.json.tmp'), 'w') as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_path.with_suffix('.json.tmp'), meta_path)
    print(f'{label}: done in {time.perf_counter() - started:.0f}s, '
          f'{rows - 1:,} bricks = {pool_gib:.1f} GiB')
    return str(out_dir)


def open_pool(sidecar_dir: str):
    """Open a sidecar read-only: (meta, table, coords, [channel memmaps])."""
    sidecar_dir = Path(sidecar_dir)
    with open(sidecar_dir / 'meta.json') as f:
        meta = json.load(f)
    if meta.get('format') != 'respool' or meta.get('version') not in (
            RESPOOL_FORMAT_VERSION, TYPED_RESPOOL_FORMAT_VERSION, SIGNED_NORMAL_U8_FORMAT_VERSION):
        raise ValueError(f'{sidecar_dir}: unsupported sidecar format {meta.get("format")!r} '
                         f'v{meta.get("version")!r}')
    table = np.load(sidecar_dir / 'table.npy', mmap_mode='r', allow_pickle=False)
    coords = np.load(sidecar_dir / 'brick_coords.npy', mmap_mode='r', allow_pickle=False)
    rows = int(meta['rows'])
    brick_voxels = int(np.prod(meta['brick_shape']))
    pools = []
    if table.shape != tuple(meta['grid_shape']) or coords.shape != (rows, 3):
        raise ValueError('pool index shapes do not match metadata')
    layout = (meta['channel_layout'] if meta['version'] != RESPOOL_FORMAT_VERSION
              else [dict(file=f'channel_{i}.u8', dtype='u1', shape=[rows, brick_voxels], order='C')
                    for i in range(len(meta['channels']))])
    if len(layout) != len(meta['channels']):
        raise ValueError('channel layout count does not match sources')
    if meta['version'] == SIGNED_NORMAL_U8_FORMAT_VERSION:
        if (meta.get('normal_encoding') != NORMAL_U8_ENCODING
                or [c['name'] for c in layout] != ['nx', 'ny', 'nz']
                or any(c['dtype'] != '|u1' for c in layout)):
            raise ValueError('unsupported signed uint8 normal encoding')
    for channel in layout:
        dtype = np.dtype(channel['dtype'])
        if (dtype.str not in ('|u1', '<f4') or channel['shape'] != [rows, brick_voxels]
                or channel['order'] != 'C' or Path(channel['file']).name != channel['file']):
            raise ValueError('unsupported channel layout')
        path = sidecar_dir / channel['file']
        expected = rows * brick_voxels * dtype.itemsize
        if path.stat().st_size != expected:
            raise ValueError(f'{path}: size {path.stat().st_size} != expected {expected}')
        pools.append(np.memmap(path, dtype=dtype, mode='r',
                               shape=(rows, brick_voxels)))
    return meta, table, coords, pools


def verify_pool(sidecar_dir: str, num_samples: int, seed: int = 0) -> None:
    """Compare random voxels against the source zarrs. Where the sidecar was
    CT-masked, a pool zero is accepted iff the mapped CT voxel is zero."""
    import zarr

    meta, table, _coords, pools = open_pool(sidecar_dir)
    arrays = [zarr.open(path, mode='r') for path in meta['channels']]
    masker = None
    if meta.get('ct_mask'):
        masker = CtMasker(meta['ct_mask']['path'], meta['ct_mask']['group'],
                          meta['array_shape'])
    shape = np.array(meta['array_shape'])
    brick = np.array(meta['brick_shape'])
    rng = np.random.default_rng(seed)
    points = (rng.random((num_samples, 3)) * shape).astype(np.int64)
    if len(_coords) > 1:
        count = num_samples // 2
        chosen = _coords[rng.integers(1, len(_coords), size=count)]
        points[:count] = np.minimum(chosen * brick + (rng.random((count, 3))*brick).astype(int), shape-1)
    brick_idx = points // brick
    local = points - brick_idx * brick
    linear = (local[:, 0] * brick[1] + local[:, 1]) * brick[2] + local[:, 2]
    rows = table[brick_idx[:, 0], brick_idx[:, 1], brick_idx[:, 2]]
    encoded_expected = None
    if meta['version'] == SIGNED_NORMAL_U8_FORMAT_VERSION:
        presence = zarr.open(meta['source_validity_array'], mode='r')
        normal = np.asarray([[array[tuple(p)] for array in arrays] for p in points],
                            dtype=np.float32).reshape(-1, 3)
        valid = np.asarray([presence[tuple(p)] != 0 for p in points], dtype=bool)
        encoded_expected = encode_signed_normals_u8(normal, valid)
    mismatches = masked = 0
    for channel, (pool, array) in enumerate(zip(pools, arrays)):
        got = pool[rows, linear]
        for i in range(num_samples):
            expected = (encoded_expected[i, channel] if encoded_expected is not None
                        else array[points[i, 0], points[i, 1], points[i, 2]])
            if got[i] == expected:
                continue
            validity = meta.get('validity_channel')
            if (validity is not None and got[i] == 0
                    and arrays[validity][tuple(points[i])] == 0):
                masked += 1
                continue
            if (masker is not None and got[i] == 0
                    and masker.ct_value_at(points[i]) == 0):
                masked += 1
                continue
            mismatches += 1
            if mismatches <= 10:
                print(f'MISMATCH ch{channel} zyx={tuple(points[i])}: '
                      f'pool={got[i]} zarr={expected}')
    if mismatches:
        raise SystemExit(f'{sidecar_dir}: {mismatches} mismatching voxels')
    print(f'{sidecar_dir}: verified {num_samples} random voxels '
          f'x{len(pools)} channel(s); all match '
          f'({masked} differ only under the CT/presence mask)')


def pack_signed_normals(folder, out_dir=None, *, brick_edge=16, io_threads=2, force=False):
    """Write just three signed-vector uint8 components from a completed export."""
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text())
    if (manifest.get('artifact_type') != 'signed_patch_normal_volume'
            or manifest.get('complete') is not True):
        raise ValueError('signed-normal packing requires a complete signed_patch_normal_volume export')
    names = ['nx', 'ny', 'nz', 'presence']
    arrays = []
    for name in names:
        relative = Path(manifest['channels'][name])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('channel paths must be relative to the export directory')
        array = folder / relative
        meta = _read_zarray_meta(array, typed=True)
        expected = '|u1' if name == 'presence' else '<f4'
        if meta['dtype'] != expected or meta['shape'] != manifest['shape_zyx']:
            raise ValueError(f'{name}: incompatible channel dtype/shape')
        arrays.append(str(array))
    provenance = dict(export_manifest=str(folder / 'manifest.json'),
        vector_components=['nx', 'ny', 'nz'], signed=True,
        sign_convention=manifest['sign_convention'],
        cell_size_fitter_voxels=manifest['cell_size_fitter_voxels'],
        fitter_voxel_scale_base_voxels=manifest['fitter_voxel_scale_base_voxels'],
        scale_base_voxels=manifest['scale_base_voxels'],
        translation_base_voxels=manifest['translation_base_voxels'],
        indexing=manifest['indexing'], source_metadata=manifest['source_metadata'])
    return pack_arrays(arrays, str(out_dir or folder / 'signed_normals_u8.respool'),
        label='signed patch normals', brick_shape=(brick_edge,)*3,
        io_threads=io_threads, force=force, channel_labels=names,
        validity_channel=names.index('presence'), source_metadata=provenance, encode_normals_u8=True)


def _find_one(folder: str, pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    if not hits:
        return None
    if len(hits) > 1:
        print(f'WARNING: multiple matches for {pattern}, using {hits[0]}')
    return hits[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Pack Lasagna or signed patch-normal Zarr stores into sparse brick pools.')
    parser.add_argument('folder', help='Lasagna input directory, or a completed '
                        'patch-normal export directory for --what signed_normals')
    parser.add_argument('--what', default='normals,grad_mag',
                        help='comma list of normals,grad_mag (default all), or signed_normals')
    parser.add_argument('--signed-brick', type=int, default=16,
                        help='signed-normal brick edge in output cells (default 16)')
    parser.add_argument('--signed-output', type=Path,
                        help='signed uint8 pool directory (default FOLDER/signed_normals_u8.respool)')
    parser.add_argument('--normal-group', default='4',
                        help='group for both normals and grad_mag')
    parser.add_argument('--ct', default=None, metavar='ZARR',
                        help='CT zarr whose zero voxels are masked out of every '
                             'packed store (e.g. .../s1_ds2.zarr)')
    parser.add_argument('--ct-group', default='2')
    parser.add_argument('--io-threads', type=int,
                        help='bounded chunk workers (default 2 for signed normals, 16 otherwise)')
    parser.add_argument('--force', action='store_true',
                        help='rebuild sidecars that already exist')
    parser.add_argument('--verify', type=int, default=0, metavar='N',
                        help='after packing, compare N random voxels per '
                             'store against the source zarr')
    args = parser.parse_args(argv)

    what = {w.strip() for w in args.what.split(',') if w.strip()}
    unknown = what - {'normals', 'grad_mag', 'signed_normals'}
    if unknown:
        parser.error(f'unknown --what entries: {sorted(unknown)}')
    if args.verify < 0 or (args.io_threads is not None and args.io_threads < 1):
        parser.error('verify must be nonnegative and io-threads must be positive')
    if 'signed_normals' in what:
        if what != {'signed_normals'} or args.ct:
            parser.error('signed_normals is a standalone mode; CT masking is not supported in this mode')
        out = pack_signed_normals(args.folder, args.signed_output,
            brick_edge=args.signed_brick,
            io_threads=args.io_threads or 2, force=args.force)
        if args.verify:
            verify_pool(out, args.verify)
        return [out]
    args.io_threads = args.io_threads or 16

    def masker_for(array_dir):
        if args.ct is None:
            return None
        shape = _read_zarray_meta(Path(array_dir))['shape']
        return CtMasker(args.ct, args.ct_group, shape)

    built = []
    if 'normals' in what:
        nx = _find_one(args.folder, '*_nx.ome.zarr')
        ny = _find_one(args.folder, '*_ny.ome.zarr')
        if nx is None or ny is None:
            print('missing *_nx/*_ny ome.zarr, skipping normals')
        else:
            arrays = [os.path.join(nx, args.normal_group),
                      os.path.join(ny, args.normal_group)]
            out = sidecar_path(nx, args.normal_group, pair=True)
            pack_arrays(arrays, out, label='normals',
                        ct_masker=masker_for(arrays[0]),
                        io_threads=args.io_threads, force=args.force)
            built.append(out)
    if 'grad_mag' in what:
        grad = _find_one(args.folder, '*_grad_mag.ome.zarr')
        if grad is None:
            print('no *_grad_mag.ome.zarr found, skipping grad_mag')
        else:
            array = os.path.join(grad, args.normal_group)
            out = sidecar_path(grad, args.normal_group)
            pack_arrays([array], out, label='grad_mag',
                        ct_masker=masker_for(array),
                        io_threads=args.io_threads, force=args.force)
            built.append(out)

    if args.verify:
        for out in built:
            verify_pool(out, args.verify)
    return built


if __name__ == '__main__':
    main(sys.argv[1:])
