"""Lossless, resident patch normals: bitmap/rank lookup into packed uint8 values.

The on-disk respool remains unchanged. A prepacked cache can skip CPU
compaction; otherwise startup stages compact CPU batches as before.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from pack_resident_pools import open_pool
from prepacked_patch_normals import (open_direct_compact_export, open_standalone_prepacked,
                                    open_prepacked_cache, pack_rows,
                                    prepacked_cache_path)


def _popcount(value):
    # Signed int64 shifts are arithmetic; masks discard the sign extension.
    value = value - ((value >> 1) & 0x5555555555555555)
    value = (value & 0x3333333333333333) + ((value >> 2) & 0x3333333333333333)
    value = (value + (value >> 4)) & 0x0F0F0F0F0F0F0F0F
    value = value + (value >> 8)
    value = value + (value >> 16)
    return (value + (value >> 32)) & 127


class CompactPatchNormalPool:
    """Same gather contract as ResidentBrickPool, without empty-cell payloads."""

    def __init__(self, sidecar_dir, *, z_roi=None, device='cuda',
                 expected_shape_zyx=None, progress_callback=None, batch_rows=4096,
                 exclusion_radius_cells=0., cache_directory=None):
        started = time.perf_counter()
        self.device = torch.device(device)
        import json
        from pathlib import Path
        source_format = json.loads((Path(sidecar_dir) / 'meta.json').read_text()).get('format')
        standalone = source_format == 'prepacked_patch_normals'
        direct = source_format == 'compact_patch_normals'
        if standalone:
            meta, table, coords, direct_data = open_standalone_prepacked(sidecar_dir)
            pools = None
        elif direct:
            meta, table, coords, direct_data = open_direct_compact_export(sidecar_dir)
            pools = None
        else:
            meta, table, coords, pools = open_pool(sidecar_dir)
        self.meta = meta
        self.shape_zyx = tuple(meta['array_shape'])
        brick = tuple(meta['brick_shape'])
        cells = int(np.prod(brick))
        if (pools is not None and len(pools) != 3) or cells > 32767:
            raise ValueError('compact patch normals require three channels and <=32767 cells per brick')
        if expected_shape_zyx is not None and tuple(expected_shape_zyx) != self.shape_zyx:
            raise ValueError('patch-normal pool shape differs from manifest')
        if batch_rows <= 0:
            raise ValueError('batch_rows must be positive')
        rows = int(meta['rows'])
        if not np.isfinite(exclusion_radius_cells) or exclusion_radius_cells < 0:
            raise ValueError('patch exclusion radius must be finite and nonnegative')
        keep = np.ones(rows, dtype=bool)
        if z_roi is not None:
            bz = coords[:, 0].astype(np.int64) * brick[0]
            # Coverage just outside the fit must still exclude nearby fallback.
            halo = int(np.ceil(exclusion_radius_cells))
            keep = (bz < z_roi[1] + halo) & (bz + brick[0] > z_roi[0] - halo)
        keep[0] = True
        ids = np.flatnonzero(keep)
        self.resident_bricks = len(ids)
        self.total_bricks = rows
        remap = np.zeros(rows, dtype=np.int32)
        remap[ids] = np.arange(len(ids), dtype=np.int32)
        prepacked_path = (None if direct or standalone else
                          prepacked_cache_path(cache_directory, sidecar_dir))
        prepacked = (direct_data if direct or standalone else
                     open_prepacked_cache(prepacked_path, meta))
        self.prepacked_cache_used = prepacked is not None
        batches = []
        total_values = 1  # reserved zero vector for absent cells
        selected_offsets = None
        if prepacked is not None:
            counts = prepacked.offsets[ids + 1] - prepacked.offsets[ids]
            selected_offsets = np.empty(len(ids) + 1, dtype=np.int64)
            selected_offsets[0] = 1
            selected_offsets[1:] = 1 + counts.cumsum()
            total_values = int(selected_offsets[-1])
            if progress_callback:
                progress_callback(1, 1, 'reusing prepacked patch normals')
        else:
            # Stage only compact bytes on the host; raw batches are bounded.
            for lo in range(0, len(ids), batch_rows):
                selected = ids[lo:lo + batch_rows]
                raw = np.stack([p[selected] for p in pools], axis=-1)
                bits, prefix, counts, packed = pack_rows(raw)
                if lo == 0 and counts[0] != 0:
                    raise ValueError('patch-normal reserved row 0 must be empty')
                offsets = counts.cumsum() - counts + total_values
                total_values += int(counts.sum())
                batches.append((lo, bits, prefix, offsets, packed))
                del raw
                if progress_callback:
                    progress_callback(min(lo + batch_rows, len(ids)), len(ids),
                                      'compacting patch normals on CPU')
        words = (cells + 63) // 64
        table = remap[table]
        exclusion_arrays = None
        self.exclusion = None
        if exclusion_radius_cells > 0:
            from patch_normal_exclusion import (
                build_exclusion_mask, exclusion_cache_path, load_exclusion_mask,
                save_exclusion_mask, select_exclusion_mask_z_roi)
            cache_path = exclusion_cache_path(
                cache_directory, sidecar_dir, shape=self.shape_zyx, brick=brick,
                z_roi=z_roi, radius=exclusion_radius_cells)
            exclusion_arrays = load_exclusion_mask(cache_path, table.shape, words)
            if exclusion_arrays is None and z_roi is not None:
                # A full-volume cache can serve any fit range of this export.
                full_cache_path = exclusion_cache_path(
                    cache_directory, sidecar_dir, shape=self.shape_zyx,
                    brick=brick, z_roi=None, radius=exclusion_radius_cells)
                exclusion_arrays = load_exclusion_mask(full_cache_path, table.shape, words)
                if exclusion_arrays is not None:
                    exclusion_arrays = select_exclusion_mask_z_roi(
                        *exclusion_arrays, z_roi, brick[0])
            if exclusion_arrays is None:
                source_bits = (np.asarray(prepacked.bits[ids]) if prepacked is not None
                               else np.concatenate([batch[1] for batch in batches]))
                exclusion_arrays = build_exclusion_mask(
                    table, source_bits, brick, self.shape_zyx, exclusion_radius_cells,
                    z_roi=z_roi, progress_callback=progress_callback)
                del source_bits
                save_exclusion_mask(cache_path, *exclusion_arrays)
            elif progress_callback:
                progress_callback(1, 1, 'reusing cached patch-normal exclusion mask')
        self.pool_bytes = (total_values * 3 + len(ids) * (words * 10 + 8)
                           + table.nbytes)
        self.dense_pool_bytes = len(ids) * cells * 3 + table.nbytes
        self.bits = torch.empty((len(ids), words), dtype=torch.int64, device=self.device)
        self.prefix = torch.empty((len(ids), words), dtype=torch.int16, device=self.device)
        self.offsets = torch.empty(len(ids), dtype=torch.int64, device=self.device)
        self.values = torch.empty((total_values, 3), dtype=torch.uint8, device=self.device)
        self.values[0] = 0
        if prepacked is not None:
            for lo in range(0, len(ids), batch_rows):
                hi = min(lo + batch_rows, len(ids))
                selected = ids[lo:hi]
                self.bits[lo:hi].copy_(torch.from_numpy(np.asarray(prepacked.bits[selected])))
                self.prefix[lo:hi].copy_(torch.from_numpy(np.asarray(prepacked.prefix[selected])))
                self.offsets[lo:hi].copy_(torch.from_numpy(selected_offsets[lo:hi].copy()))
                if progress_callback:
                    progress_callback(hi, len(ids), 'loading prepacked patch normals')
            # Consecutive source rows have consecutive packed values. Copy
            # each run in bounded chunks without unpacking the dense channels.
            breaks = np.flatnonzero(np.diff(ids) != 1) + 1
            run_starts = np.r_[0, breaks]
            run_ends = np.r_[breaks, len(ids)]
            for first, end in zip(run_starts, run_ends):
                source = int(prepacked.offsets[ids[first]])
                destination = int(selected_offsets[first])
                remaining = int(selected_offsets[end] - selected_offsets[first])
                while remaining:
                    count = min(remaining, 8_000_000)
                    values = np.asarray(prepacked.values[source:source + count]).copy()
                    self.values[destination:destination + count].copy_(torch.from_numpy(values))
                    source += count
                    destination += count
                    remaining -= count
        else:
            # Pop batches as they are uploaded so CPU staging memory is released.
            while batches:
                lo, bits, prefix, offsets, packed = batches.pop()
                end = lo + len(bits)
                self.bits[lo:end].copy_(torch.from_numpy(bits))
                self.prefix[lo:end].copy_(torch.from_numpy(prefix))
                self.offsets[lo:end].copy_(torch.from_numpy(offsets))
                begin = int(offsets[0])
                self.values[begin:begin + len(packed)].copy_(torch.from_numpy(packed))
        self.table = torch.from_numpy(table).to(self.device)
        self.exclusion_bytes = 0
        if exclusion_arrays is not None:
            from patch_normal_exclusion import PatchExclusionMask
            self.exclusion = PatchExclusionMask(*exclusion_arrays, self.device)
            self.exclusion_bytes = self.exclusion.nbytes
            self.pool_bytes += self.exclusion_bytes
        self._brick = torch.tensor(brick, device=self.device, dtype=torch.long)
        self._stride_z, self._stride_y = brick[1] * brick[2], brick[2]
        self.load_seconds = time.perf_counter() - started
        print(f'patch normals: compact pool {self.pool_bytes / 1024**3:.2f} GiB '
              f'(formerly {self.dense_pool_bytes / 1024**3:.2f} GiB), '
              f'{total_values - 1:,} occupied cells loaded in {self.load_seconds:.1f}s'
              f'{" from standalone export" if standalone else " from direct compact export" if direct else " from prepacked cache" if self.prepacked_cache_used else ""}', flush=True)
        if self.exclusion is not None:
            print(f'patch normals: exclusion mask {self.exclusion_bytes / 1024**2:.1f} MiB '
                  f'(included above), radius {exclusion_radius_cells:g} export cells', flush=True)

    def gather(self, indices_zyx):
        return self._gather(indices_zyx, return_exclusion=False)

    def gather_with_exclusion(self, indices_zyx):
        return self._gather(indices_zyx, return_exclusion=True)

    def _gather(self, indices_zyx, *, return_exclusion):
        shape = indices_zyx.shape[:-1]
        flat = indices_zyx.detach().reshape(-1, 3).to(self.device, dtype=torch.long)
        b = torch.div(flat, self._brick, rounding_mode='floor')
        slots = self.table[b[:, 0], b[:, 1], b[:, 2]].long()
        local = flat - b * self._brick
        linear = local[:, 0] * self._stride_z + local[:, 1] * self._stride_y + local[:, 2]
        word, bit = linear // 64, linear % 64
        bits = self.bits[slots, word]
        present = ((bits >> bit) & 1).bool()
        lower_mask = (torch.ones_like(bit) << bit) - 1
        rank = self.prefix[slots, word].long() + _popcount(bits & lower_mask)
        index = torch.where(present, self.offsets[slots] + rank, 0)
        values = self.values[index].reshape(*shape, 3)
        if return_exclusion:
            near = (self.exclusion.gather_at(b, word, bit)
                    if self.exclusion is not None else present)
            return values, near.reshape(shape)
        return values

    def close(self):
        if self.exclusion is not None:
            self.exclusion.close()
            self.exclusion = None
        self.bits = self.prefix = self.offsets = self.values = self.table = None
