"""Pack sparse lasagna zarr stores into flat resident-pool sidecars.

A sidecar holds every *occupied* brick of a (chunk-sparse) uint8 zarr array as
one flat pool file per channel, plus an int32 brick-grid table mapping brick
coordinates to pool rows (0 = absent, served by a reserved all-zero row).
Loading it is a single sequential read per channel instead of tens of
thousands of demand-fills through tensorstore, and a fully resident pool never
evicts, so the per-step LRU thrash disappears entirely.

CT masking: the surf-SDT builder zeroes the *surface prediction* where the CT
volume reads 0, but the EDT then fills those voxels with capped far-field
distances, so the giant outside-the-scroll mask region is byte-occupied in the
store. Passing ``--ct`` zeroes every voxel whose (ratio-mapped) CT voxel is 0
while packing; combined with a sub-chunk brick size this drops the mask skin
from the pool, and the zeros read back as no-data through the existing
sampling contract.

Sidecar layout (``<zarr>.respool_g<group>[_pair]/``):
  meta.json         format/version, shapes, brick grid, sources, ct_mask info
  table.npy         int32 (gz, gy, gx); 0 = absent, i >= 1 = pool row i
  brick_coords.npy  int32 (rows, 3) brick coords per row; row 0 = -1
  channel_<i>.u8    uint8 (rows, brick_voxels) raw pool; row 0 = zeros

Usage (defaults match the fit_spiral input conventions):
  python pack_resident_pools.py /path/to/lasagna_inputs \
      --ct /path/to/s1_ds2.zarr --ct-group 2 --verify 2000
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

RESPOOL_FORMAT_VERSION = 2


def sidecar_path(zarr_path: str, group: str, *, pair: bool = False) -> str:
    """Canonical sidecar directory for a store; the pair suffix marks the
    two-channel nx+ny pool that lives next to the nx zarr."""
    suffix = f'.respool_g{group}' + ('_pair' if pair else '')
    return str(zarr_path).rstrip('/') + suffix


def _read_zarray_meta(array_dir: Path) -> dict:
    with open(array_dir / '.zarray') as f:
        meta = json.load(f)
    if meta['dtype'] != '|u1':
        raise ValueError(f'{array_dir}: expected |u1 dtype, got {meta["dtype"]!r}')
    if meta.get('order', 'C') != 'C':
        raise ValueError(f'{array_dir}: expected C order')
    if meta.get('filters'):
        raise ValueError(f'{array_dir}: filters are not supported')
    return meta


def _load_delta3d(codec_id: str):
    try:
        import vc_delta3d
    except ImportError as exc:
        raise RuntimeError(
            f'the store uses the {codec_id} codec; install the vc-delta3d package'
        ) from exc
    return vc_delta3d


def _list_chunk_keys(array_dir: Path, separator: str) -> set[tuple[int, int, int]]:
    keys = set()
    if separator == '/':
        for path in glob.iglob(str(array_dir / '*' / '*' / '*')):
            parts = path.rsplit(os.sep, 3)[-3:]
            try:
                keys.add((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    else:
        for name in os.listdir(array_dir):
            parts = name.split('.')
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                keys.add((int(parts[0]), int(parts[1]), int(parts[2])))
    return keys


def _chunk_path(array_dir: Path, key: tuple[int, int, int], separator: str) -> Path:
    if separator == '/':
        return array_dir / str(key[0]) / str(key[1]) / str(key[2])
    return array_dir / f'{key[0]}.{key[1]}.{key[2]}'


def _make_chunk_reader(compressor: dict | None, chunk_voxels: int):
    if compressor is None:
        def read(path: Path) -> np.ndarray:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size != chunk_voxels:
                raise ValueError(f'{path}: raw chunk has {data.size} bytes, expected {chunk_voxels}')
            return data
        return read
    codec_id = compressor.get('id')
    if codec_id == 'vcz1':
        delta3d = _load_delta3d(codec_id)

        def read(path: Path) -> np.ndarray:
            with open(path, 'rb') as f:
                decoded = delta3d.decompress_with_magic(
                    f.read(), chunk_voxels, b'VCZ1')
            decoded = np.frombuffer(decoded, dtype=np.uint8)
            if decoded.size != chunk_voxels:
                raise ValueError(
                    f'{path}: decoded chunk has {decoded.size} bytes, '
                    f'expected {chunk_voxels}')
            return decoded
        return read
    if codec_id == 'vc-delta3d':
        _load_delta3d(codec_id)  # Import registers the numcodecs adapter.
    import numcodecs
    codec = numcodecs.get_codec(compressor)

    def read(path: Path) -> np.ndarray:
        try:
            with open(path, 'rb') as f:
                decoded = codec.decode(f.read())
        except Exception as exc:
            # Without the path, a codec failure on one chunk out of millions
            # is a stack trace with nothing to act on.
            raise RuntimeError(f'{path}: codec decode failed: {exc}') from exc
        decoded = np.frombuffer(decoded, dtype=np.uint8)
        if decoded.size != chunk_voxels:
            raise ValueError(f'{path}: decoded chunk has {decoded.size} bytes, expected {chunk_voxels}')
        return decoded
    return read


class CtMasker:
    """Voxel-level ``CT == 0`` mask, ratio-mapped onto a target array grid.

    ``ratio`` is target voxels per CT voxel, per axis (integer, e.g. 2 for the
    surf-SDT group 1 against s1_ds2 group 2, 1 for the group-4 lasagna
    stores). Decoded CT chunks are kept in a small LRU because consecutive
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
) -> str:
    """Pack one or more same-geometry uint8 zarr arrays (one per channel).

    ``brick_shape`` defaults to the source chunk shape and must divide it;
    bricks that are entirely zero across all channels (after CT masking) are
    dropped from the pool.
    """
    out_dir = Path(out_dir)
    meta_path = out_dir / 'meta.json'
    if meta_path.exists() and not force:
        print(f'{label}: {out_dir} already exists, skipping (--force to rebuild)')
        return str(out_dir)

    array_dirs = [Path(d) for d in array_dirs]
    metas = [_read_zarray_meta(d) for d in array_dirs]
    shape = tuple(metas[0]['shape'])
    chunks = tuple(metas[0]['chunks'])
    for d, m in zip(array_dirs, metas):
        if tuple(m['shape']) != shape or tuple(m['chunks']) != chunks:
            raise ValueError(
                f'{d}: shape/chunks {m["shape"]}/{m["chunks"]} differ from '
                f'{array_dirs[0]}: {shape}/{chunks}')
    separators = [m.get('dimension_separator', '.') for m in metas]
    chunk_voxels = int(np.prod(chunks))
    brick = tuple(chunks) if brick_shape is None else tuple(brick_shape)
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
    if not keys:
        raise RuntimeError(f'{label}: no chunk files found under {array_dirs}')
    for d, channel_keys in zip(array_dirs, per_channel_keys):
        missing = len(keys) - len(channel_keys)
        if missing:
            print(f'{label}: WARNING {d.name} is missing {missing} of '
                  f'{len(keys)} union chunks; they read as zero (no-data)')

    upper_gib = (len(keys) * subs_per_chunk + 1) * brick_voxels * len(array_dirs) / 1024 ** 3
    print(f'{label}: packing {len(keys):,} chunks of {chunks} into bricks of '
          f'{brick} x{len(array_dirs)} channel(s), <= {upper_gib:.1f} GiB, '
          f'ct_mask={"on" if ct_masker else "off"} -> {out_dir}')
    if progress_callback is not None:
        progress_callback(0, len(keys), 'discovering occupied bricks')

    readers = [
        _make_chunk_reader(m['compressor'], chunk_voxels) for m in metas
    ]

    def read_chunk(key):
        """Returns (key, kept sub-brick index array, per-channel (n, vox))."""
        channels = []
        for d, sep, reader, channel_keys in zip(
                array_dirs, separators, readers, per_channel_keys):
            if key in channel_keys:
                channels.append(reader(_chunk_path(d, key, sep)).reshape(chunks))
            else:
                channels.append(None)
        if ct_masker is not None:
            origin = tuple(k * c for k, c in zip(key, chunks))
            occupied = ct_masker.occupied_block(origin, chunks)
            channels = [
                None if a is None else np.where(occupied, a, np.uint8(0))
                for a in channels
            ]
        split = [
            None if a is None else np.ascontiguousarray(
                a.reshape(sub[0], brick[0], sub[1], brick[1], sub[2], brick[2])
                .transpose(0, 2, 4, 1, 3, 5)
            ).reshape(subs_per_chunk, brick_voxels)
            for a in channels
        ]
        keep = np.zeros(subs_per_chunk, dtype=bool)
        for a in split:
            if a is not None:
                keep |= a.any(axis=1)
        keep_idx = np.flatnonzero(keep)
        return key, keep_idx, [
            (np.zeros((len(keep_idx), brick_voxels), dtype=np.uint8)
             if a is None else a[keep_idx])
            for a in split
        ]

    started = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    if meta_path.exists():
        meta_path.unlink()  # invalidate a stale sidecar while rebuilding

    coords = [(-1, -1, -1)]
    sub_grid = np.stack(np.unravel_index(np.arange(subs_per_chunk), sub), axis=1)
    channel_files = [
        open(out_dir / f'channel_{i}.u8', 'wb', buffering=1024 * 1024)
        for i in range(len(array_dirs))
    ]
    try:
        for f in channel_files:
            f.write(bytes(brick_voxels))  # row 0: reserved all-zero brick
        with ThreadPoolExecutor(io_threads) as executor:
            done = 0
            for key, keep_idx, channel_bricks in executor.map(
                    read_chunk, keys, chunksize=4):
                base = np.multiply(key, sub)
                for sz, sy, sx in sub_grid[keep_idx]:
                    coords.append((base[0] + sz, base[1] + sy, base[2] + sx))
                for f, bricks in zip(channel_files, channel_bricks):
                    f.write(bricks.tobytes())
                done += 1
                if done % 2000 == 0 or done == len(keys):
                    if progress_callback is not None:
                        progress_callback(
                            done, len(keys),
                            f'{len(coords) - 1:,} bricks kept')
                    elapsed = time.perf_counter() - started
                    print(f'{label}: {done:,}/{len(keys):,} chunks, '
                          f'{len(coords) - 1:,} bricks kept '
                          f'({done / len(keys) * 100:.0f}%, {elapsed:.0f}s)',
                          flush=True)
    finally:
        for f in channel_files:
            f.close()

    coords = np.asarray(coords, dtype=np.int32)
    rows = len(coords)
    table = np.zeros(grid, dtype=np.int32)
    table[coords[1:, 0], coords[1:, 1], coords[1:, 2]] = np.arange(
        1, rows, dtype=np.int32)
    np.save(out_dir / 'table.npy', table)
    np.save(out_dir / 'brick_coords.npy', coords)
    pool_gib = rows * brick_voxels * len(array_dirs) / 1024 ** 3
    meta = {
        'format': 'respool',
        'version': RESPOOL_FORMAT_VERSION,
        'array_shape': list(shape),
        'source_chunk_shape': list(chunks),
        'brick_shape': list(brick),
        'grid_shape': list(grid),
        'rows': rows,
        'channels': [str(d) for d in array_dirs],
        'channel_names': [d.parent.name + '/' + d.name for d in array_dirs],
        'dtype': 'u1',
        'ct_mask': ct_masker.describe() if ct_masker is not None else None,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'{label}: done in {time.perf_counter() - started:.0f}s, '
          f'{rows - 1:,} bricks = {pool_gib:.1f} GiB')
    return str(out_dir)


def open_pool(sidecar_dir: str):
    """Open a sidecar read-only: (meta, table, coords, [channel memmaps])."""
    sidecar_dir = Path(sidecar_dir)
    with open(sidecar_dir / 'meta.json') as f:
        meta = json.load(f)
    if meta.get('format') != 'respool' or meta.get('version') != RESPOOL_FORMAT_VERSION:
        raise ValueError(f'{sidecar_dir}: unsupported sidecar format {meta.get("format")!r} '
                         f'v{meta.get("version")!r}')
    table = np.load(sidecar_dir / 'table.npy')
    coords = np.load(sidecar_dir / 'brick_coords.npy')
    rows = int(meta['rows'])
    brick_voxels = int(np.prod(meta['brick_shape']))
    pools = []
    for i in range(len(meta['channels'])):
        path = sidecar_dir / f'channel_{i}.u8'
        expected = rows * brick_voxels
        if path.stat().st_size != expected:
            raise ValueError(f'{path}: size {path.stat().st_size} != expected {expected}')
        pools.append(np.memmap(path, dtype=np.uint8, mode='r',
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
    brick_idx = points // brick
    local = points - brick_idx * brick
    linear = (local[:, 0] * brick[1] + local[:, 1]) * brick[2] + local[:, 2]
    rows = table[brick_idx[:, 0], brick_idx[:, 1], brick_idx[:, 2]]
    mismatches = masked = 0
    for channel, (pool, array) in enumerate(zip(pools, arrays)):
        got = pool[rows, linear]
        for i in range(num_samples):
            expected = array[points[i, 0], points[i, 1], points[i, 2]]
            if got[i] == expected:
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
          f'({masked} differ only under the CT mask)')


def _find_one(folder: str, pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    if not hits:
        return None
    if len(hits) > 1:
        print(f'WARNING: multiple matches for {pattern}, using {hits[0]}')
    return hits[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Pack lasagna zarr stores into resident-pool sidecars.')
    parser.add_argument('folder', help='lasagna_inputs directory holding the '
                        '*_surf_sdt / *_nx / *_ny / *_grad_mag ome.zarr stores')
    parser.add_argument('--what', default='sdt,normals,grad_mag',
                        help='comma list of sdt,normals,grad_mag (default all)')
    parser.add_argument('--sdt-group', default='1')
    parser.add_argument('--normal-group', default='4',
                        help='group for both normals and grad_mag')
    parser.add_argument('--ct', default=None, metavar='ZARR',
                        help='CT zarr whose zero voxels are masked out of every '
                             'packed store (e.g. .../s1_ds2.zarr)')
    parser.add_argument('--ct-group', default='2')
    parser.add_argument('--sdt-brick', type=int, default=None,
                        help='SDT pool brick edge; defaults to 32 when --ct is '
                             'given (so masked bricks are actually dropped), '
                             'else the source chunk size')
    parser.add_argument('--io-threads', type=int, default=16)
    parser.add_argument('--force', action='store_true',
                        help='rebuild sidecars that already exist')
    parser.add_argument('--verify', type=int, default=0, metavar='N',
                        help='after packing, compare N random voxels per '
                             'store against the source zarr')
    args = parser.parse_args(argv)

    what = {w.strip() for w in args.what.split(',') if w.strip()}
    unknown = what - {'sdt', 'normals', 'grad_mag'}
    if unknown:
        parser.error(f'unknown --what entries: {sorted(unknown)}')

    def masker_for(array_dir):
        if args.ct is None:
            return None
        shape = _read_zarray_meta(Path(array_dir))['shape']
        return CtMasker(args.ct, args.ct_group, shape)

    built = []
    if 'sdt' in what:
        sdt = _find_one(args.folder, '*_surf_sdt.ome.zarr')
        if sdt is None:
            print('no *_surf_sdt.ome.zarr found, skipping sdt')
        else:
            array = os.path.join(sdt, args.sdt_group)
            edge = args.sdt_brick or (32 if args.ct else None)
            pack_arrays([array], sidecar_path(sdt, args.sdt_group),
                        label='surf_sdt',
                        brick_shape=(edge,) * 3 if edge else None,
                        ct_masker=masker_for(array),
                        io_threads=args.io_threads, force=args.force)
            built.append(sidecar_path(sdt, args.sdt_group))
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
