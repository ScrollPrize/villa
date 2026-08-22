#!/usr/bin/env python3
"""A ResidentBrickPool that keeps the pool in min3 words on the device.

Same public surface as the upstream class -- ``gather(indices_zyx)`` returning
``(..., channels)`` uint8 -- so the fitter cannot tell the difference except by
how much VRAM is left. Decode is per voxel, out of one aligned 32-bit word:

    value = (w & 0xFF) + ((w >> (8 + 3*j)) & 7)      j = voxel index in its 2^3 block

One load. That is the whole point of the layout, and it is why this replaces
Min4BrickPool rather than sitting beside it: min4 keeps codes and minima in two
arrays, so every gather is two random reads, and on the real 2.03 GiB pool that
cost 71% over an uncompressed read where this costs 4.6% -- at a better ratio,
2.000x against 1.939x.

No overflow is possible in the add: the block minimum plus a code of at most 7
is the block maximum, which was a uint8 in the source.

Words are held as int32 rather than uint32 because the arithmetic below is all
mask-and-shift and int32 is the type every torch version handles the same way.
The sign bit is the top code bit; ``& 7`` after the shift discards the sign
extension, so the value is unaffected.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch


class Min3BrickPool:
    """Resident pool over a `<sidecar>.min3` directory."""

    def __init__(self, sidecar_dir, *, origin_zyx=(0, 0, 0), z_roi=None,
                 device: torch.device | str = 'cuda', label: str,
                 expected_channels: int | None = None,
                 expected_shape_zyx: tuple[int, int, int] | None = None,
                 progress_callback=None,
                 bounds_check: bool = True):
        started = time.perf_counter()
        sidecar_dir = Path(sidecar_dir)
        meta = json.loads((sidecar_dir / 'meta.json').read_text())
        if meta.get('encoding') != 'min3':
            raise ValueError(f'{sidecar_dir}: not a min3 sidecar')
        self.meta = meta
        self.label = str(label)
        self.device = torch.device(device)
        self.side = int(meta['encode_block'])

        self.sidecar_dir = str(sidecar_dir)
        self.shape_zyx = tuple(int(v) for v in meta['array_shape'])
        self.origin_zyx = tuple(int(v) for v in origin_zyx)
        # The upstream pool refuses a sidecar that does not match the array it
        # is standing in for. Keeping the same two checks matters more here,
        # not less: a re-encoded sidecar is one more place the two can drift.
        if (expected_shape_zyx is not None
                and tuple(int(v) for v in expected_shape_zyx) != self.shape_zyx):
            raise ValueError(
                f'{label}: sidecar {sidecar_dir} covers array shape '
                f'{self.shape_zyx}, expected {tuple(expected_shape_zyx)}')
        brick = tuple(int(v) for v in meta['brick_shape'])
        self._edge = brick[0]
        rows = int(meta['rows'])
        self.channels = len(meta['channels'])
        if expected_channels is not None and self.channels != expected_channels:
            raise ValueError(
                f'{label}: sidecar {sidecar_dir} has {self.channels} '
                f'channel(s), expected {expected_channels}')
        blocks_per_brick = (self._edge // self.side) ** 3

        table_np = np.load(sidecar_dir / 'table.npy')
        coords_np = np.load(sidecar_dir / 'brick_coords.npy')

        keep = np.ones(rows, dtype=bool)
        if z_roi is not None:
            z_lo, z_hi = int(z_roi[0]), int(z_roi[1])
            brick_z = coords_np[:, 0].astype(np.int64)
            keep = (brick_z * brick[0] < z_hi) & ((brick_z + 1) * brick[0] > z_lo)
            keep[0] = True                      # the reserved all-zero brick
        kept_ids = np.flatnonzero(keep)
        remap = np.zeros(rows, dtype=np.int32)
        remap[kept_ids] = np.arange(len(kept_ids), dtype=np.int32)

        self.resident_bricks = len(kept_ids)
        self.pool_bytes = self.channels * len(kept_ids) * blocks_per_brick * 4
        try:
            self.words = torch.empty(
                (self.channels, len(kept_ids), blocks_per_brick),
                dtype=torch.int32, device=self.device)
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                f'Could not allocate the {self.label} min3 pool '
                f'({self.pool_bytes / 1024**3:.2f} GiB, {len(kept_ids)} bricks)'
            ) from exc

        slab = max(1, (256 << 20) // max(1, blocks_per_brick * 4))
        for ci in range(self.channels):
            wm = np.memmap(sidecar_dir / f'channel_{ci}.u32', dtype=np.uint32,
                           mode='r', shape=(rows, blocks_per_brick))
            done = 0
            total = self.channels * len(kept_ids)
            for lo in range(0, len(kept_ids), slab):
                ids = kept_ids[lo:lo + slab]
                block = np.ascontiguousarray(wm[ids]).view(np.int32)
                self.words[ci, lo:lo + len(ids)] = torch.from_numpy(block).to(
                    self.device)
                done += len(ids)
                if progress_callback is not None:
                    progress_callback(ci * len(kept_ids) + done, total,
                                      f'{self.label} min3 pool')

        self.table = torch.from_numpy(remap[table_np]).to(self.device)
        self._origin = torch.tensor(
            [int(v) for v in origin_zyx], device=self.device, dtype=torch.long)
        self._brick = torch.tensor(list(brick), device=self.device,
                                   dtype=torch.long)
        self._shape = torch.tensor(list(self.shape_zyx), device=self.device,
                                   dtype=torch.long)
        self._bps = self._edge // self.side          # blocks per brick edge
        # Upstream gates this on an environment variable and leaves it off by
        # default. Defaulting it on here would make the drop-in stricter than
        # what it replaces -- it would raise on indices the pool it stands in
        # for accepts -- so the substitution would change behaviour, which is
        # the one thing it must not do.
        self._bounds_check = (
            bool(bounds_check)
            and os.environ.get('FIT_SPIRAL_RESIDENT_BOUNDS_CHECK') == '1')
        self.total_bricks = rows
        self._gathers = 0
        self._gather_seconds = 0.0
        # fit_spiral reads store.last_timings after every phase step, so the
        # drop-in has to carry it or the substitution fails at run time rather
        # than at load.
        self.last_timings: dict[str, float | int] = {}
        self.load_seconds = time.perf_counter() - started
        print(f'{self.label}: min3 pool {self.resident_bricks:,}/{rows:,} bricks '
              f'({self.pool_bytes / 1024**3:.2f} GiB) loaded in '
              f'{self.load_seconds:.1f}s from {sidecar_dir.name}', flush=True)

    def gather(self, indices_zyx: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        original_shape = tuple(indices_zyx.shape[:-1])
        flat = indices_zyx.detach().reshape(-1, 3)
        if flat.shape[0] == 0:
            return torch.empty((*original_shape, self.channels),
                               dtype=torch.uint8, device=self.device)
        flat = flat.to(device=self.device, dtype=torch.long)
        source = flat + self._origin
        if self._bounds_check and bool(
                ((source < 0) | (source >= self._shape)).any()):
            raise IndexError(f'{self.label} gather received an out-of-bounds index')
        brick_idx = torch.div(source, self._brick, rounding_mode='floor')
        slots = self.table[brick_idx[:, 0], brick_idx[:, 1],
                           brick_idx[:, 2]].to(torch.long)
        # local = source - brick_idx * brick, through the two tensors that
        # already exist rather than into a third.  At the batch sizes the
        # fitter gathers, one more live (n, 3) int64 is 24 bytes per index of
        # peak, which is six times what the words it decodes cost.
        brick_idx.mul_(self._brick)
        source.sub_(brick_idx)
        del brick_idx
        lz, ly, lx = source[:, 0], source[:, 1], source[:, 2]

        side = self.side
        bz = torch.div(lz, side, rounding_mode='floor')
        by = torch.div(ly, side, rounding_mode='floor')
        bx = torch.div(lx, side, rounding_mode='floor')
        # The shift operand, 8 + 3j, built in int32 and in place.  j is the
        # position inside the 2^3 block, in the same (z,y,x) order the packer
        # laid the codes down; it never reaches 8, so carrying it as int64 and
        # then widening it to the shape of the words -- which is what the first
        # version of this did -- cost more scratch than the codes it decodes.
        shift = lz.sub(bz * side).to(torch.int32)
        shift.mul_(side).add_(ly.sub(by * side).to(torch.int32))
        shift.mul_(side).add_(lx.sub(bx * side).to(torch.int32))
        shift.mul_(3).add_(8)
        block = bz.mul_(self._bps).add_(by).mul_(self._bps).add_(bx)
        del by, bx, lz, ly, lx, source

        # Advanced indexing returns a fresh tensor, so the decode can run in
        # place without touching the resident words.
        w = self.words[:, slots, block]                     # (channels, n) int32
        del block, slots
        code = torch.bitwise_right_shift(w, shift)   # (channels, n) >> (n,)
        code.bitwise_and_(7)
        w.bitwise_and_(0xFF)                                # w is now the minima
        w.add_(code)
        del code, shift
        values = w.to(torch.uint8).transpose(0, 1)

        elapsed = time.perf_counter() - started
        self._gather_seconds += elapsed
        self._gathers += 1
        self.last_timings = {
            'gather_seconds': elapsed,
            'resident_bricks': self.resident_bricks,
            'resident_mib': self.pool_bytes / 1024 ** 2,
        }
        return values.reshape(*original_shape, self.channels)

    def stats(self) -> dict[str, float | int]:
        return {
            'resident_bricks': self.resident_bricks,
            'total_bricks': self.total_bricks,
            'gathers': self._gathers,
            'gather_seconds': self._gather_seconds,
            'load_seconds': self.load_seconds,
            'pool_bytes': self.pool_bytes,
        }
