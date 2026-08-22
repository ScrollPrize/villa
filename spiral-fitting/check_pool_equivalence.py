#!/usr/bin/env python3
"""Does the packed pool return exactly what the raw pool returns?

Lossless is a claim about every voxel, so this gathers the same indices from
both pools and requires torch.equal -- not a tolerance, not a sample mean. The
indices are drawn to hit the cases a packing gets wrong when its layout is
wrong: both parities of the linear index, block boundaries, brick boundaries,
and the reserved all-zero row.

The second argument's own meta.json says which packing it is, and the pool
class comes from the same dispatch the fitter uses, so this cannot pass against
a class the fitter would not have loaded.

Run:
  python check_pool_equivalence.py <raw sidecar> <packed sidecar> [--z-roi lo hi]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from pathlib import Path

# Everything this needs is now in the tree beside it, so a run cannot pick up
# a pool class from somewhere the commit does not describe.  absolute(), not
# resolve(): resolve() expands symlinks into every traceback.
_HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(_HERE))
from sparse_cuda_cache import ResidentBrickPool          # noqa: E402


def open_packed(sidecar, **kwargs):
    """Whichever pool the sidecar asks for -- same choice the fitter makes."""
    import json
    encoding = json.loads(
        (Path(sidecar) / 'meta.json').read_text()).get('encoding', 'raw')
    if encoding == 'min3':
        from min3_pool import Min3BrickPool
        return Min3BrickPool(sidecar, **kwargs), encoding
    raise SystemExit(f'{sidecar}: encoding {encoding!r} is not a packed pool')


def draw_indices(z_lo, z_hi, shape, brick, side, n, seed):
    """Random points plus every boundary the packing can be wrong about.

    Boundaries are absolute, not relative to the ROI: brick b spans
    [128b, 128b+128), so offsets from an arbitrary z_lo land nowhere near one.
    An earlier version of this generator shifted relative offsets by z_lo and
    therefore tested no boundary at all.
    """
    rng = np.random.default_rng(seed)
    _, ys, xs = shape
    pts = [np.stack([rng.integers(z_lo, z_hi, n),
                     rng.integers(0, ys, n),
                     rng.integers(0, xs, n)], axis=1)]

    def edges(step, lo, hi, take):
        """Absolute multiples of `step` inside [lo, hi), and their neighbours."""
        first = ((lo + step - 1) // step) * step
        out = set()
        for k in range(take):
            m = first + k * step
            if m >= hi:
                break
            for d in (-1, 0, 1):
                v = m + d
                if lo <= v < hi:
                    out.add(int(v))
        return sorted(out)

    ez = edges(brick[0], z_lo, z_hi, 6) + edges(side, z_lo, z_hi, 8)
    ey = edges(brick[1], 0, ys, 6) + edges(side, 0, ys, 8)
    ex = edges(brick[2], 0, xs, 6) + edges(side, 0, xs, 8)
    if ez and ey and ex:
        g = np.array(np.meshgrid(ez, ey, ex, indexing='ij')).reshape(3, -1).T
        pts.append(g.astype(np.int64))

    # A contiguous run along x so both nibble parities appear consecutively,
    # crossing a brick edge on the way.
    span = min(4096, xs)
    x0 = max(0, min(xs - span, (brick[2] * 3) - span // 2))
    run = np.stack([np.full(span, (z_lo + z_hi) // 2),
                    np.full(span, ys // 2),
                    np.arange(x0, x0 + span)], axis=1)
    pts.append(run.astype(np.int64))
    return np.concatenate(pts, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('packed')
    ap.add_argument('--z-roi', type=int, nargs=2, default=None)
    ap.add_argument('--n', type=int, default=2_000_000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    z_roi = tuple(args.z_roi) if args.z_roi else None
    raw = ResidentBrickPool(args.raw, origin_zyx=(0, 0, 0), z_roi=z_roi,
                            label='raw')
    packed, enc = open_packed(args.packed, origin_zyx=(0, 0, 0),
                              z_roi=z_roi, label='packed')
    print(f'\nVRAM: raw {raw.pool_bytes / 2**30:.2f} GiB vs '
          f'{enc} {packed.pool_bytes / 2**30:.2f} GiB '
          f'= {raw.pool_bytes / packed.pool_bytes:.3f}x smaller')

    shape = raw.shape_zyx
    brick = tuple(int(v) for v in raw.meta['brick_shape'])
    side = int(packed.meta['encode_block'])
    lo = z_roi[0] if z_roi else 0
    hi = min(z_roi[1], shape[0]) if z_roi else shape[0]
    idx = draw_indices(lo, hi, shape, brick, side, args.n, args.seed)
    on_brick = int(np.count_nonzero(idx[:, 2] % brick[2] == 0))
    on_block = int(np.count_nonzero(idx[:, 2] % side == 0))
    odd = int(np.count_nonzero((idx[:, 2] & 1) == 1))
    oob = ((idx < 0) | (idx >= np.array(shape))).any(axis=1)
    if oob.any():
        raise SystemExit(
            f'the index generator produced {int(oob.sum())} out-of-range '
            f'points, e.g. {idx[oob][0].tolist()} against shape {shape} -- '
            f'fix the generator, do not let the pool decide')
    print(f'comparing {len(idx):,} gathers  '
          f'(on a brick edge in x: {on_brick:,}; on a block edge: {on_block:,}; '
          f'odd nibble: {odd:,})')

    bad = 0
    CH = 1 << 20
    for s in range(0, len(idx), CH):
        t = torch.from_numpy(idx[s:s + CH])
        a = raw.gather(t)
        b = packed.gather(t)
        if not torch.equal(a, b):
            d = (a != b).nonzero()
            bad += int(d.shape[0])
            if bad and d.shape[0]:
                k = int(d[0, 0])
                print(f'  MISMATCH at index {s + k}: zyx={idx[s + k].tolist()} '
                      f'raw={a[k].tolist()} {enc}={b[k].tolist()}')
                break
    if bad:
        print(f'\nFAIL: {bad:,} differing gathers')
        return 1
    print(f'\nPASS: every one of {len(idx):,} gathers is bitwise equal')

    # Halving VRAM is worthless if the fit slows down: decode adds a second
    # dependent load and an add per gathered voxel. Time both on the same
    # indices with the device synchronised, so this compares kernels rather
    # than queue depth.
    bench = torch.from_numpy(idx[:min(len(idx), 8 << 20)])
    for pool, name in ((raw, 'raw '), (packed, enc)):
        for _ in range(3):
            pool.gather(bench)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            pool.gather(bench)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 100.0
        # AGENTS.md 1.4 wants iteration counts and a distribution, not one
        # number: time each iteration separately and report min/median/max.
        per = []
        for _ in range(10):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            pool.gather(bench)
            torch.cuda.synchronize()
            per.append((time.perf_counter() - t1) * 1000.0)
        per.sort()
        print(f'  {name} gather over {len(bench):,} indices, 10 iterations: '
              f'min {per[0]:.2f} / median {per[len(per)//2]:.2f} / '
              f'max {per[-1]:.2f} ms   (mean of a separate 10-run batch '
              f'{ms:.2f} ms)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
