"""Reusable, lossless on-disk form of the compact patch-normal pool.

The source respool is left intact. All rows are prepacked once, so a fit can
select its z range without rereading the three dense uint8 channel files.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from pack_resident_pools import open_pool
from patch_normal_exclusion import pack_presence_bits


FORMAT_VERSION = 1


def pack_rows(values):
    """Pack (rows, cells, 3) bytes, including partially-zero valid vectors."""
    present = np.any(values != 0, axis=-1)
    rows, cells = present.shape
    words = (cells + 63) // 64
    padded = np.zeros((rows, words * 64), dtype=bool)
    padded[:, :cells] = present
    counts = padded.reshape(rows, words, 64).sum(axis=-1, dtype=np.int64)
    prefix = (counts.cumsum(axis=1) - counts).astype(np.int16)
    return pack_presence_bits(padded), prefix, counts.sum(axis=1), values[present]


def prepacked_cache_path(cache_directory, sidecar_dir):
    """Key the cache by source location, sizes, and modification times."""
    if cache_directory is None:
        return None
    source = Path(sidecar_dir).resolve()
    files = ['meta.json', 'table.npy', 'brick_coords.npy']
    files += [f'channel_{i}.u8' for i in range(3)]
    signature = {
        'version': FORMAT_VERSION,
        'source': str(source),
        'files': [(name, (source / name).stat().st_size,
                   (source / name).stat().st_mtime_ns) for name in files],
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return Path(cache_directory) / 'patch-normal-prepacked' / digest


@dataclass
class PrepackedPatchNormals:
    bits: np.ndarray
    prefix: np.ndarray
    offsets: np.ndarray
    values: np.ndarray


def open_direct_compact_export(sidecar_dir):
    """Open a completed compact export without touching any dense channel files."""
    sidecar_dir = Path(sidecar_dir)
    meta = json.loads((sidecar_dir / 'meta.json').read_text())
    if meta.get('format') != 'compact_patch_normals' or meta.get('version') != 1:
        raise ValueError(f'{sidecar_dir}: unsupported direct compact pool')
    rows = int(meta['rows'])
    cells = int(np.prod(meta['brick_shape']))
    words = (cells + 63) // 64
    total = int(meta['total_values'])
    if rows < 2 or total < 2 or cells > 32767:
        raise ValueError(f'{sidecar_dir}: invalid compact pool dimensions')
    expected = {'brick_coords.i32': rows * 3 * 4,
                'bits.i64': rows * words * 8,
                'prefix.i16': rows * words * 2,
                'offsets.i64': (rows + 1) * 8,
                'values.u8': total * 3}
    for name, size in expected.items():
        if (sidecar_dir / name).stat().st_size != size:
            raise ValueError(f'{sidecar_dir / name}: unexpected file size')
    table = np.load(sidecar_dir / 'table.npy', mmap_mode='r', allow_pickle=False)
    if table.shape != tuple(meta['grid_shape']) or table.dtype != np.int32:
        raise ValueError(f'{sidecar_dir}: invalid compact table')
    coords = np.memmap(sidecar_dir / 'brick_coords.i32', dtype=np.int32,
                       mode='r', shape=(rows, 3))
    bits = np.memmap(sidecar_dir / 'bits.i64', dtype=np.int64,
                     mode='r', shape=(rows, words))
    prefix = np.memmap(sidecar_dir / 'prefix.i16', dtype=np.int16,
                       mode='r', shape=(rows, words))
    offsets = np.memmap(sidecar_dir / 'offsets.i64', dtype=np.int64,
                        mode='r', shape=(rows + 1,))
    values = np.memmap(sidecar_dir / 'values.u8', dtype=np.uint8,
                       mode='r', shape=(total, 3))
    if (offsets[0] != 1 or offsets[1] != 1 or offsets[-1] != total
            or np.any(np.diff(offsets) < 0) or bits[0].any()
            or np.any(values[0] != 0) or not np.array_equal(coords[0], [-1, -1, -1])):
        raise ValueError(f'{sidecar_dir}: invalid compact reserved row or offsets')
    return meta, table, coords, PrepackedPatchNormals(bits, prefix, offsets, values)


def open_prepacked_cache(path, source_meta):
    """Open a complete cache as read-only memmaps; return None if absent."""
    if path is None or not path.exists():
        return None
    try:
        info = json.loads((path / 'meta.json').read_text())
        rows = int(source_meta['rows'])
        cells = int(np.prod(source_meta['brick_shape']))
        words = (cells + 63) // 64
        total = int(info['total_values'])
        if (info.get('format_version') != FORMAT_VERSION
                or info.get('rows') != rows
                or info.get('brick_shape') != source_meta['brick_shape']
                or total < 1
                or (path / 'values.u8').stat().st_size != total * 3):
            raise ValueError('metadata differs from the source pool')
        bits = np.load(path / 'bits.npy', mmap_mode='r', allow_pickle=False)
        prefix = np.load(path / 'prefix.npy', mmap_mode='r', allow_pickle=False)
        offsets = np.load(path / 'offsets.npy', mmap_mode='r', allow_pickle=False)
        if (bits.shape != (rows, words) or bits.dtype != np.int64
                or prefix.shape != (rows, words) or prefix.dtype != np.int16
                or offsets.shape != (rows + 1,) or offsets.dtype != np.int64
                or offsets[0] != 1 or offsets[1] != 1
                or offsets[-1] != total or np.any(np.diff(offsets) < 0)):
            raise ValueError('array layout is invalid')
        values = np.memmap(path / 'values.u8', dtype=np.uint8, mode='r',
                           shape=(total, 3))
        if np.any(values[0] != 0):
            raise ValueError('reserved zero vector is invalid')
        return PrepackedPatchNormals(bits, prefix, offsets, values)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f'{path}: invalid prepacked patch-normal cache: {exc}') from exc


def preprocess_patch_normal_pool(sidecar_dir, cache_directory, *, batch_rows=4096,
                                 progress_callback=None):
    """Stream the full source pool into an atomic, reusable compact cache."""
    if batch_rows <= 0:
        raise ValueError('batch_rows must be positive')
    meta, _table, _coords, pools = open_pool(sidecar_dir)
    rows = int(meta['rows'])
    cells = int(np.prod(meta['brick_shape']))
    if len(pools) != 3 or cells > 32767:
        raise ValueError('compact patch normals require three channels and <=32767 cells per brick')
    target = prepacked_cache_path(cache_directory, sidecar_dir)
    if target is None:
        raise ValueError('cache_directory is required to preprocess patch normals')
    if open_prepacked_cache(target, meta) is not None:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        words = (cells + 63) // 64
        bits = np.lib.format.open_memmap(
            temporary / 'bits.npy', mode='w+', dtype=np.int64, shape=(rows, words))
        prefix = np.lib.format.open_memmap(
            temporary / 'prefix.npy', mode='w+', dtype=np.int16, shape=(rows, words))
        offsets = np.lib.format.open_memmap(
            temporary / 'offsets.npy', mode='w+', dtype=np.int64, shape=(rows + 1,))
        offsets[0] = 1
        total_values = 1
        with (temporary / 'values.u8').open('wb') as output:
            output.write(b'\0\0\0')
            for lo in range(0, rows, batch_rows):
                hi = min(lo + batch_rows, rows)
                raw = np.stack([channel[lo:hi] for channel in pools], axis=-1)
                batch_bits, batch_prefix, counts, packed = pack_rows(raw)
                if lo == 0 and counts[0] != 0:
                    raise ValueError('patch-normal reserved row 0 must be empty')
                bits[lo:hi] = batch_bits
                prefix[lo:hi] = batch_prefix
                offsets[lo + 1:hi + 1] = total_values + counts.cumsum()
                total_values += int(counts.sum())
                output.write(packed.tobytes())
                if progress_callback:
                    progress_callback(hi, rows, 'prepacking patch normals')
        bits.flush()
        prefix.flush()
        offsets.flush()
        del bits, prefix, offsets
        (temporary / 'meta.json').write_text(json.dumps({
            'format_version': FORMAT_VERSION,
            'rows': rows,
            'brick_shape': meta['brick_shape'],
            'total_values': total_values,
        }))
        if target.exists():
            # Another builder completed the same source while this one ran.
            open_prepacked_cache(target, meta)
        else:
            os.rename(temporary, target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None):
    import argparse
    from patch_normals import patch_normal_export_info

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='patch_normals.zarr export directory')
    parser.add_argument('--cache-directory', type=Path, required=True,
                        help='the same cache root configured for the Spiral service')
    parser.add_argument('--batch-rows', type=int, default=4096)
    args = parser.parse_args(argv)
    sidecar, _cell_size, _shape = patch_normal_export_info(args.source)

    last_reported = -1

    def progress(current, total, _detail):
        nonlocal last_reported
        milestone = min(20, 20 * current // total)
        if milestone > last_reported:
            last_reported = milestone
            print(f'prepacked {current:,}/{total:,} bricks', flush=True)

    result = preprocess_patch_normal_pool(
        sidecar, args.cache_directory, batch_rows=args.batch_rows,
        progress_callback=progress)
    print(f'\nPatch normals prepacked at {result}')


if __name__ == '__main__':
    main()
