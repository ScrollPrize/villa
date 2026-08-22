#!/usr/bin/env python3
"""How much scratch VRAM does one gather cost, on top of the pool it reads?

Halving the resident pool is only worth what reaches the fitter's peak.  A
decode that allocates a few int64 temporaries per gathered voxel can hand back
everything the smaller pool won, and the pool-size number alone will not show
it: the pool is steady state, the temporaries are the peak.

For each pool this reports

    resident   bytes the pool holds for the whole fit
    transient  max_memory_allocated - memory_allocated across one gather,
               i.e. everything the call touched that the pool does not own
    output     bytes of the value tensor the call returns

Transient is the number that competes with the model, the optimiser and the
other pools for the same allocator budget.

The instrument is checked before it is used: --self-test allocates a known
number of bytes inside a fake gather and requires the reading back to match,
and the run fails if the reading is off by more than one allocator block.
Without that, a decode change that does nothing would read as a win.

Run:
  python benchmark_resident_pool_memory.py <raw sidecar> <min3 sidecar> \
      [--z-roi lo hi] [--n 4000000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# absolute(), not resolve(): resolve() expands symlinks, which puts the real
# checkout path into every traceback this script can raise.
sys.path.insert(0, str(Path(__file__).absolute().parent))

from min3_pool import Min3BrickPool             # noqa: E402
from sparse_cuda_cache import ResidentBrickPool  # noqa: E402


def measure(fn, *args):
    """Peak allocated bytes above the resting level, across one call."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    resting = torch.cuda.memory_allocated()
    out = fn(*args)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    kept = out.numel() * out.element_size()
    del out
    return peak - resting, kept


class _Waster:
    """A gather that does nothing but allocate a known number of bytes."""

    def __init__(self, want_bytes):
        self.want_bytes = want_bytes

    def gather(self, indices_zyx):
        scratch = torch.empty(self.want_bytes, dtype=torch.uint8,
                              device='cuda')
        scratch.fill_(1)
        out = torch.zeros((indices_zyx.shape[0], 1), dtype=torch.uint8,
                          device='cuda')
        del scratch
        return out


def self_test():
    """Does the meter move by exactly what a known allocation costs?"""
    idx = torch.zeros((1024, 3), dtype=torch.long, device='cuda')
    small, _ = measure(_Waster(1 << 20).gather, idx)
    large, _ = measure(_Waster(64 << 20).gather, idx)
    grew = large - small
    want = (64 - 1) << 20
    # The allocator rounds; one 2 MiB block of slack is fine, a factor is not.
    if abs(grew - want) > (2 << 20):
        raise SystemExit(
            f'self-test failed: a {want:,}-byte increase read as {grew:,}. '
            f'The meter is not measuring what this script claims.')
    print(f'self-test: a {want / 2**20:.0f} MiB increase reads as '
          f'{grew / 2**20:.0f} MiB -- meter live')


def draw(z_lo, z_hi, shape, n, seed=0):
    rng = np.random.default_rng(seed)
    _, ys, xs = shape
    return np.stack([rng.integers(z_lo, z_hi, n),
                     rng.integers(0, ys, n),
                     rng.integers(0, xs, n)], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('min3')
    ap.add_argument('--z-roi', type=int, nargs=2, default=None)
    ap.add_argument('--n', type=int, default=4_000_000)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('this benchmark needs a CUDA device')
    self_test()

    z_roi = tuple(args.z_roi) if args.z_roi else None
    raw = ResidentBrickPool(args.raw, origin_zyx=(0, 0, 0), z_roi=z_roi,
                            label='raw')
    packed = Min3BrickPool(args.min3, origin_zyx=(0, 0, 0), z_roi=z_roi,
                           label='min3')

    lo = z_roi[0] if z_roi else 0
    hi = min(z_roi[1], raw.shape_zyx[0]) if z_roi else raw.shape_zyx[0]
    idx = torch.from_numpy(draw(lo, hi, raw.shape_zyx, args.n)).cuda()

    report = {'n': args.n, 'pools': {}}
    print(f'\none gather over {args.n:,} indices\n')
    print(f'{"pool":6} {"resident GiB":>13} {"transient MiB":>14} '
          f'{"output MiB":>11} {"transient/index B":>18}')
    for pool, name in ((raw, 'raw'), (packed, 'min3')):
        pool.gather(idx[:1024])                      # warm the kernels
        readings = [measure(pool.gather, idx)[0] for _ in range(3)]
        transient = min(readings)
        _, kept = measure(pool.gather, idx)
        print(f'{name:6} {pool.pool_bytes / 2**30:13.2f} '
              f'{transient / 2**20:14.1f} {kept / 2**20:11.1f} '
              f'{transient / args.n:18.1f}')
        report['pools'][name] = {
            'resident_bytes': int(pool.pool_bytes),
            'transient_bytes': int(transient),
            'transient_readings': [int(v) for v in readings],
            'output_bytes': int(kept),
        }

    r, m = report['pools']['raw'], report['pools']['min3']
    saved = r['resident_bytes'] - m['resident_bytes']
    paid = m['transient_bytes'] - r['transient_bytes']
    report['resident_saved_bytes'] = int(saved)
    report['transient_paid_bytes'] = int(paid)
    report['net_bytes'] = int(saved - paid)
    print(f'\nresident saved   {saved / 2**30:+.3f} GiB')
    print(f'transient paid   {-paid / 2**30:+.3f} GiB')
    print(f'net at the peak  {(saved - paid) / 2**30:+.3f} GiB')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f'\nwritten: {args.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
