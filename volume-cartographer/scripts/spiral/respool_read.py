"""Read resident-pool sidecars on the CPU, locally or straight off a server.

``pack_resident_pools.py`` writes a sidecar per (chunk-sparse) uint8 zarr
array: ``meta.json``, ``table.npy``, ``brick_coords.npy`` and one flat
``channel_<i>.u8`` pool. The existing consumer, ``lasagna_data.py``, loads a
sidecar as a fully-resident sparse CUDA pool, so reading one currently costs a
CUDA device and a local copy of the whole pool.

That is a steep entry price for anyone who only wants to look at a field. The
published PHercParis4 sidecars run to tens of gigabytes per channel, and the
CT-support field alone is 35 GB; a contributor checking one patch against the
predicted surface distance should not need a GPU and a full download.

This module reads the same sidecars with numpy alone. A brick is 32 KiB, and
``channel_<i>.u8`` stores bricks contiguously by pool row, so a single brick is
one HTTP range request at offset ``row * brick_voxels``. Reading a 900x1250
slice of a published field costs about 38 MB over roughly 1100 requests
instead of the full pool.

Local sidecars are read the same way through plain file seeks.

Example
-------
    from respool_read import ResidentPool

    pool = ResidentPool(
        'https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4'
        '/lasagna_inputs/las_008_surf_sdt.ome.zarr.respool_g1',
        auth=('registeredusers', 'only'),
        cache_dir='~/.cache/respool',
    )
    block = pool.read(z0=5302, z1=5303, y0=1300, y1=2200, x0=1650, x1=2900)

The sidecar's ``array_shape`` is the shape of the *source array*, which for the
published fields is a downsampled pyramid level rather than the full-resolution
volume: check ``pool.array_shape`` before mixing coordinates from another
level.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = ['ResidentPool']

_TIMEOUT = 120


class ResidentPool:
    """A resident-pool sidecar, read brick by brick.

    Parameters
    ----------
    root:
        Sidecar directory: a local path, or an ``http(s)://`` prefix.
    channel:
        Channel index; ``pair`` sidecars carry two.
    auth:
        Optional ``(user, password)`` for basic auth on remote roots.
    cache_dir:
        Optional directory for fetched bricks. Strongly recommended for remote
        roots: without it every repeated read pays the request cost again.
    """

    def __init__(self, root: str, channel: int = 0,
                 auth: tuple[str, str] | None = None,
                 cache_dir: str | os.PathLike[str] | None = None) -> None:
        self.root = str(root).rstrip('/')
        self.remote = self.root.startswith(('http://', 'https://'))
        self.channel = int(channel)
        self._auth = auth
        self.meta = json.loads(self._fetch('meta.json').decode())
        if self.meta.get('format') != 'respool':
            raise ValueError(f'{self.root}: not a resident-pool sidecar')
        self.array_shape = tuple(int(v) for v in self.meta['array_shape'])
        self.brick_shape = tuple(int(v) for v in self.meta['brick_shape'])
        self.brick_voxels = int(np.prod(self.brick_shape))
        if self.meta.get('dtype', 'u1') != 'u1':
            raise ValueError(f'{self.root}: only uint8 pools are supported')
        self.table = np.load(io.BytesIO(self._fetch('table.npy')))
        self._mem: dict[int, np.ndarray] = {}
        self.cache_dir: Path | None = None
        if cache_dir is not None:
            self.cache_dir = Path(os.path.expanduser(cache_dir)) / (
                f'{Path(self.root).name}_c{self.channel}')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bytes_fetched = 0
        self.requests = 0

    # ------------------------------------------------------------------ io
    def _fetch(self, name: str, byte_range: tuple[int, int] | None = None) -> bytes:
        if not self.remote:
            path = Path(self.root) / name
            if byte_range is None:
                return path.read_bytes()
            start, end = byte_range
            with path.open('rb') as fh:
                fh.seek(start)
                return fh.read(end - start + 1)
        request = urllib.request.Request(f'{self.root}/{name}')
        if self._auth is not None:
            token = base64.b64encode(
                f'{self._auth[0]}:{self._auth[1]}'.encode()).decode()
            request.add_header('Authorization', f'Basic {token}')
        if byte_range is not None:
            request.add_header('Range', f'bytes={byte_range[0]}-{byte_range[1]}')
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return response.read()

    # --------------------------------------------------------------- bricks
    def brick(self, gz: int, gy: int, gx: int) -> np.ndarray | None:
        """One brick, or ``None`` where the sidecar holds no data."""
        table = self.table
        if not (0 <= gz < table.shape[0] and 0 <= gy < table.shape[1]
                and 0 <= gx < table.shape[2]):
            return None
        row = int(table[gz, gy, gx])
        if row <= 0:                     # 0 marks an absent brick
            return None
        cached = self._mem.get(row)
        if cached is not None:
            return cached
        raw = self._read_row(row)
        if raw is None:
            return None
        brick = np.frombuffer(raw, np.uint8).reshape(self.brick_shape)
        if len(self._mem) < 20_000:      # bounded, bricks are 32 KiB each
            self._mem[row] = brick
        return brick

    def _read_row(self, row: int) -> bytes | None:
        path = None if self.cache_dir is None else self.cache_dir / f'{row}.bin'
        if path is not None and path.exists():
            return path.read_bytes()
        offset = row * self.brick_voxels
        raw = self._fetch(f'channel_{self.channel}.u8',
                          (offset, offset + self.brick_voxels - 1))
        self.bytes_fetched += len(raw)
        self.requests += 1
        if len(raw) != self.brick_voxels:
            return None
        if path is not None:
            path.write_bytes(raw)
        return raw

    # ---------------------------------------------------------------- reads
    def read(self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
             fill: int = 0) -> np.ndarray:
        """A block in source-array coordinates; absent bricks read as ``fill``."""
        for lo, hi, size, axis in ((z0, z1, self.array_shape[0], 'z'),
                                   (y0, y1, self.array_shape[1], 'y'),
                                   (x0, x1, self.array_shape[2], 'x')):
            if not 0 <= lo < hi <= size:
                raise ValueError(
                    f'{axis} range {lo}..{hi} outside array extent {size}')
        bz, by, bx = self.brick_shape
        out = np.full((z1 - z0, y1 - y0, x1 - x0), fill, np.uint8)
        for gz in range(z0 // bz, (z1 - 1) // bz + 1):
            for gy in range(y0 // by, (y1 - 1) // by + 1):
                for gx in range(x0 // bx, (x1 - 1) // bx + 1):
                    brick = self.brick(gz, gy, gx)
                    if brick is None:
                        continue
                    oz, oy, ox = gz * bz, gy * by, gx * bx
                    sz0, sy0, sx0 = max(z0, oz), max(y0, oy), max(x0, ox)
                    sz1, sy1, sx1 = min(z1, oz + bz), min(y1, oy + by), min(x1, ox + bx)
                    out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                        brick[sz0 - oz:sz1 - oz, sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox]
        return out

    def occupancy(self) -> float:
        """Fraction of the brick grid that holds data."""
        return float((self.table > 0).mean())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        where = 'remote' if self.remote else 'local'
        return (f'<ResidentPool {where} shape={self.array_shape} '
                f'brick={self.brick_shape} occupancy={self.occupancy():.1%}>')


def _cli(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('root', help='sidecar directory or URL')
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--auth', help='user:password for remote basic auth')
    parser.add_argument('--cache-dir')
    parser.add_argument('--z', type=int, nargs=2, metavar=('Z0', 'Z1'))
    parser.add_argument('--y', type=int, nargs=2, metavar=('Y0', 'Y1'))
    parser.add_argument('--x', type=int, nargs=2, metavar=('X0', 'X1'))
    parser.add_argument('--out', help='write the block as .npy')
    args = parser.parse_args(list(argv) if argv is not None else None)

    auth = tuple(args.auth.split(':', 1)) if args.auth else None
    pool = ResidentPool(args.root, args.channel, auth=auth,
                        cache_dir=args.cache_dir)
    print(pool)
    if not (args.z and args.y and args.x):
        return 0
    block = pool.read(*args.z, *args.y, *args.x)
    filled = float((block > 0).mean())
    print(f'block {block.shape}: {filled:.1%} non-zero, '
          f'values {block.min()}..{block.max()}')
    print(f'fetched {pool.bytes_fetched / 1e6:.1f} MB in {pool.requests} requests')
    if args.out:
        np.save(args.out, block)
        print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli())
