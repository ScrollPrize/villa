#!/usr/bin/env python3
"""Re-encode a resident-pool sidecar to min3: one aligned 32-bit word per 2^3 block.

Layout, little-endian, one word per block:

    | byte 0 = block minimum | bits 8..31 = eight 3-bit codes, voxel j at 8+3j |

    value = (w & 0xFF) + ((w >> (8 + 3*j)) & 7)

Lossless exactly when every 2^3 block spans at most 7. On las_008_surf_sdt that
was checked block by block over the whole store -- 4,951,375,872 blocks, worst
span 6, none over 7 -- not sampled. This packer re-checks every block anyway and
refuses to write if one is over, because a sidecar that is silently lossy is
worse than no sidecar.

Why 2^3 and not min4's 4^3, given 4 bits per voxel sounds cheaper than 3 plus a
bigger minimum table: it is not the ratio that decides, it is the number of
memory transactions. min4 stores codes and minima in two arrays, so every gather
is two random reads; measured on the real 2.03 GiB pool that costs 71% over an
uncompressed read. Here the minimum travels inside the same word as its codes,
so a gather is one aligned 32-bit load -- 4.6% over uncompressed, at a better
ratio: 4 bytes per 8 voxels is exactly 2.000x against 1.939x.

    python pack_min3.py <sidecar>              # writes <sidecar>.min3
    python pack_min3.py <sidecar> --analyze    # check and size it, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

BLOCK = 2
MAX_CODE = 7


def encode_bricks(bricks: np.ndarray, side: int = BLOCK) -> np.ndarray:
    """(rows, edge, edge, edge) uint8 -> (rows, blocks) uint32. Raises if inexact."""
    rows, edge = bricks.shape[0], bricks.shape[1]
    c = edge // side
    b = bricks.reshape(rows, c, side, c, side, c, side)
    mins = b.min(axis=(2, 4, 6))
    span = b.max(axis=(2, 4, 6)).astype(np.int16) - mins.astype(np.int16)
    worst = int(span.max()) if span.size else 0
    if worst > MAX_CODE:
        bad = int(np.count_nonzero(span > MAX_CODE))
        raise ValueError(
            f'{bad} block(s) of {span.size} span more than {MAX_CODE}; '
            f'largest span {worst} -- not exact at side {side}')
    codes = (b - mins[:, :, None, :, None, :, None]).astype(np.uint32)
    w = mins.astype(np.uint32)                              # byte 0
    for dz in range(side):
        for dy in range(side):
            for dx in range(side):
                j = (dz * side + dy) * side + dx
                w |= codes[:, :, dz, :, dy, :, dx] << (8 + 3 * j)
    return np.ascontiguousarray(w.reshape(rows, c ** 3)), worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('sidecar')
    ap.add_argument('--out', default=None, help='default <sidecar>.min3')
    ap.add_argument('--side', type=int, default=BLOCK)
    ap.add_argument('--group', type=int, default=64, help='bricks per read')
    ap.add_argument('--analyze', action='store_true',
                    help='check exactness and report the size, write nothing')
    args = ap.parse_args()

    if args.side ** 3 * 3 > 24:
        raise SystemExit(f'side {args.side} needs {args.side**3 * 3} code bits, '
                         f'only 24 fit beside the minimum in a 32-bit word')

    src = Path(args.sidecar)
    meta = json.loads((src / 'meta.json').read_text())
    if meta.get('encoding', 'raw') != 'raw':
        raise SystemExit(f'{src}: expected a raw sidecar, found '
                         f'{meta.get("encoding")!r}')
    brick = tuple(int(v) for v in meta['brick_shape'])
    if len(set(brick)) != 1:
        raise SystemExit(f'non-cubic brick {brick} is not handled')
    edge = brick[0]
    if edge % args.side:
        raise SystemExit(f'brick edge {edge} is not a multiple of {args.side}')
    brick_voxels = edge ** 3
    rows = int(meta['rows'])
    channels = len(meta['channels'])
    blocks = (edge // args.side) ** 3

    dst = Path(args.out) if args.out else Path(str(src) + '.min3')
    if not args.analyze:
        dst.mkdir(parents=True, exist_ok=True)
        for name in ('table.npy', 'brick_coords.npy'):
            shutil.copy2(src / name, dst / name)

    started = time.perf_counter()
    worst_overall = 0
    for ci in range(channels):
        mm = np.memmap(src / f'channel_{ci}.u8', dtype=np.uint8, mode='r',
                       shape=(rows, brick_voxels))
        out = (None if args.analyze else
               (dst / f'channel_{ci}.u32').open('wb', buffering=8 << 20))
        try:
            for lo in range(0, rows, args.group):
                hi = min(lo + args.group, rows)
                buf = np.asarray(mm[lo:hi]).reshape(-1, edge, edge, edge)
                words, worst = encode_bricks(buf, args.side)
                worst_overall = max(worst_overall, worst)
                if out is not None:
                    # A writable memmap of the whole 19 GiB output can leave
                    # almost all pages dirty and make flush() fail with ENOMEM
                    # on a 15 GiB, swap-free WSL host.  Stream bounded groups
                    # in row order; the on-disk layout is byte-for-byte the
                    # same and kernel writeback can throttle each group.
                    encoded = words.astype('<u4', copy=False)
                    out.write(memoryview(encoded).cast('B'))
                if (lo // args.group) % 32 == 0:
                    print(f'  channel {ci}: {hi:,}/{rows:,} bricks '
                          f'({time.perf_counter() - started:.0f}s, '
                          f'worst span {worst_overall})', flush=True)
            if out is not None:
                out.flush()
                os.fsync(out.fileno())
        finally:
            if out is not None:
                out.close()

    raw = rows * brick_voxels * channels
    enc = rows * blocks * 4 * channels
    print(f'\n{rows:,} bricks x {channels} channel(s) in '
          f'{time.perf_counter() - started:.0f}s')
    print(f'raw {raw / 2**30:.2f} GiB -> encoded {enc / 2**30:.2f} GiB '
          f'= {raw / enc:.3f}x   (side {args.side}, worst span '
          f'{worst_overall}/{MAX_CODE})')
    if args.analyze:
        print('analyze only, nothing written.')
        return 0

    out_meta = dict(meta)
    out_meta['encoding'] = 'min3'
    out_meta['encode_block'] = args.side
    # Which sidecar this came from, not where it happened to sit: the absolute
    # path identifies the machine, and a sidecar that has been re-encoded since
    # has the same path and a different table.
    out_meta['encoded_from'] = src.name
    out_meta['encoded_from_table_sha256'] = hashlib.sha256(
        (src / 'table.npy').read_bytes()).hexdigest()
    out_meta['worst_block_span'] = worst_overall
    # meta.json last: it is the completion sentinel every reader keys on, so a
    # conversion killed part way through leaves a directory nothing will load
    # rather than a short one something might.
    (dst / 'meta.json').write_text(json.dumps(out_meta, indent=2))
    print(f'written: {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
