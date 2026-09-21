"""Remove entire winding rays that leave the reviewed CT envelope.

Writes a new crossing store usable by WindingInferenceStore, including density
and relative supervision. Coordinates in output are unchanged. See
docs/ct_region_masking.md for the coordinate mapping and usage.
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ct_region_mask import MaskSettings, make_ct_mask
from mask_zarr_from_ct import LocalArray, bounded_map, read_ct_batch, write_json
from winding_inference_io import (
    ARTIFACT_TYPE, FORMAT_VERSION, _load_array, _sha256, manifest_fingerprint,
)


class PackedMask:
    """Bit-packed full XY slices; only the rays' Z interval needs storage."""

    def __init__(self, data, shape, z_start=0):
        self.data, self.shape, self.z_start = data, np.asarray(shape), z_start

    def contains(self, cells):
        inside = ((cells >= 0) & (cells < self.shape)).all(axis=1)
        inside &= (cells[:, 0] >= self.z_start)
        inside &= (cells[:, 0] < self.z_start + len(self.data))
        z, y, x = cells[inside].T
        inside[inside] = ((self.data[z - self.z_start, y, x // 8]
                           >> (x % 8)) & 1).astype(bool)
        return inside


def segments_inside(mask, starts, ends):
    """Visit every voxel traversed by each straight segment (3D grid DDA).

    CT indices use floor, matching the Zarr masker. At simultaneous grid-plane
    crossings all adjacent cells are checked conservatively. This also checks
    narrow excursions between crossing points which fixed-step sampling misses.
    Zero-length segments check their point; nonfinite/out-of-volume rays fail.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    keep = (np.isfinite(starts).all(1) & np.isfinite(ends).all(1)
            & (starts >= 0).all(1) & (ends >= 0).all(1)
            & (starts < mask.shape).all(1) & (ends < mask.shape).all(1))
    ids = np.flatnonzero(keep)
    p, q = starts[ids], ends[ids]
    cell, final = np.floor(p).astype(np.int64), np.floor(q).astype(np.int64)
    valid = mask.contains(cell) & mask.contains(final)
    keep[ids[~valid]] = False
    ids, p, q, cell, final = (a[valid] for a in (ids, p, q, cell, final))
    delta = q - p
    step = np.sign(delta).astype(np.int64)
    stride = np.full(delta.shape, np.inf)
    np.divide(1., np.abs(delta), out=stride, where=delta != 0)
    boundary = cell + (step > 0)
    next_t = np.full(delta.shape, np.inf)
    np.divide(boundary - p, delta, out=next_t, where=delta != 0)
    # A segment crosses at most the Manhattan distance between endpoint cells.
    budget = int(np.abs(final - cell).sum(1).max(initial=0)) + 1
    for _ in range(budget):
        active = (cell != final).any(1)
        if not active.any():
            return keep
        ids, cell, final, step, stride, next_t = (
            a[active] for a in (ids, cell, final, step, stride, next_t))
        next_t[cell == final] = np.inf
        event = next_t.min(1)
        # A tiny roundoff guard only ADDS conservative neighbor checks; the
        # traversal advances at the exact minimum so it cannot skip short cells.
        tied = np.abs(next_t - event[:, None]) <= 8 * np.finfo(float).eps
        multiple = np.flatnonzero(tied.sum(1) > 1)
        valid = np.ones(len(ids), dtype=bool)
        if len(multiple):
            for bits in range(1, 8):
                axes = np.array([(bits >> axis) & 1 for axis in range(3)], bool)
                rows = multiple[(tied[multiple][:, axes]).all(1)]
                valid[rows] &= mask.contains(cell[rows] + step[rows] * axes)
        advance = next_t == event[:, None]
        cell += advance * step
        next_t += np.where(advance, stride, 0.)
        valid &= mask.contains(cell)
        keep[ids[~valid]] = False
        ids, cell, final, step, stride, next_t = (
            a[valid] for a in (ids, cell, final, step, stride, next_t))
    raise RuntimeError('ray traversal did not converge')


def load_source(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest.get('artifact_type') != ARTIFACT_TYPE
            or manifest.get('format_version') != FORMAT_VERSION
            or manifest.get('coordinate_order') != 'zyx'):
        raise ValueError('requires a version-1 ZYX winding crossing store')
    if manifest.get('fingerprint') != manifest_fingerprint(manifest):
        raise ValueError('winding-inference manifest fingerprint mismatch')
    ray_length = manifest.get('source_attributes', {}).get('ray_length')
    if not isinstance(ray_length, int) or ray_length < 2:
        raise ValueError('source_attributes.ray_length is required for full-ray checking')
    shards, names = [], set()
    for shard in manifest['shards']:
        name = shard['name']
        if not name or Path(name).name != name or name in ('.', '..') or name in names:
            raise ValueError('shard names must be unique directory names')
        names.add(name)
        arrays = shard['arrays']
        for desc in arrays.values():
            if Path(desc['file']).name != desc['file']:
                raise ValueError('array filenames must stay within their shard')
        a = {key: _load_array(path / name, desc, verify=True)
             for key, desc in arrays.items()}
        if any(a[k].dtype != np.dtype('float32') for k in
               ('ray_origin_zyx', 'ray_step_zyx', 'crossing_t')):
            raise ValueError(f'{name}: exported ray coordinates and t must be float32')
        n, m = len(a['ray_origin_zyx']), len(a['crossing_t'])
        offsets = a['crossing_offsets']
        if (a['ray_origin_zyx'].shape != (n, 3)
                or a['ray_step_zyx'].shape != (n, 3)
                or a['seed_winding'].shape != (n,)
                or offsets.shape != (n + 1,) or offsets.dtype.kind not in 'iu'
                or offsets[0] != 0 or offsets[-1] != m
                or (offsets[1:] <= offsets[:-1]).any()
                or a['crossing_t'].shape != (m,) or a['crossing_level'].shape != (m,)
                or n != shard['num_retained_rays'] or m != shard['num_crossings']):
            raise ValueError(f'inconsistent ray/crossing arrays in {name}')
        t = a['crossing_t']
        if not np.isfinite(t).all() or (t < 0).any() or (t > ray_length - 1).any():
            raise ValueError(f'crossings outside the declared full ray in {name}')
        descending = np.diff(t) < 0
        descending[offsets[1:-1] - 1] = False
        if descending.any():
            raise ValueError(f'crossings are not sorted within rays in {name}')
        shards.append((shard, a))
    if (sum(len(a['seed_winding']) for _, a in shards) != manifest['num_rays']
            or sum(len(a['crossing_t']) for _, a in shards) != manifest['num_crossings']):
        raise ValueError('manifest totals disagree with shards')
    return manifest, shards, ray_length


def rays_inside(mask, arrays, begin, end, scale, ray_end):
    origin = arrays['ray_origin_zyx'][begin:end].astype(np.float64)
    step = arrays['ray_step_zyx'][begin:end].astype(np.float64)
    keep = segments_inside(mask, origin * scale, (origin + ray_end * step) * scale)
    # Also check the actual float32 crossing points used by the fitting loader:
    # rounding can put a point across a mask face even when the exact line is in.
    offsets = arrays['crossing_offsets']
    lengths = np.diff(offsets[begin:end + 1])
    owner = np.repeat(np.arange(begin, end), lengths)
    selected = np.repeat(keep, lengths)
    owner = owner[selected]
    t = arrays['crossing_t'][offsets[begin]:offsets[end]][selected]
    points = arrays['ray_origin_zyx'][owner] + t[:, None] * arrays['ray_step_zyx'][owner]
    cells = np.floor(points.astype(np.float64) * scale).astype(np.int64)
    keep[owner[~mask.contains(cells)] - begin] = False
    return keep


def filter_arrays(arrays, keep):
    """Keep entire ragged records, preserving t, winding labels and ray order."""
    lengths = np.diff(arrays['crossing_offsets'])
    crossings = np.repeat(keep, lengths)
    return {
        **{key: arrays[key][keep] for key in
           ('ray_origin_zyx', 'ray_step_zyx', 'seed_winding')},
        **{key: arrays[key][crossings] for key in ('crossing_t', 'crossing_level')},
        'crossing_offsets': np.r_[np.int64(0), np.cumsum(lengths[keep], dtype=np.int64)],
    }


def save_shard(out, original, arrays, keep):
    name = original['name']
    root = out / name
    root.mkdir()
    result = copy.deepcopy(original)
    result['arrays'] = {}
    for key, value in filter_arrays(arrays, keep).items():
        path = root / f'{key}.npy'
        np.save(path, value, allow_pickle=False)
        result['arrays'][key] = dict(file=path.name, shape=list(value.shape),
                                    dtype=value.dtype.str, bytes=path.stat().st_size,
                                    sha256=_sha256(path))
    lengths = np.diff(arrays['crossing_offsets'])[keep]
    result.update(num_retained_rays=int(keep.sum()), num_crossings=int(lengths.sum()),
                  crossings_per_ray=dict(min=int(lengths.min()) if len(lengths) else 0,
                                         max=int(lengths.max()) if len(lengths) else 0,
                                         median=float(np.median(lengths)) if len(lengths) else 0.))
    result['ct_mask_removed_rays'] = int((~keep).sum())
    write_json(root / 'manifest.json', result)
    # Audit indices refer to input shard rows, not the original neural cache.
    np.save(out / 'filter_audit' / f'{name}_source_ray_indices.npy', np.flatnonzero(keep))
    return result


def run(ct_path, source_path, output_path, *, coordinate_scale, workers=2,
        mask_workers=4, ray_workers=4, rays_per_task=16384, settings=MaskSettings(),
        temp_dir=None):
    started = time.perf_counter()
    ct = LocalArray(ct_path)
    source, out = Path(source_path).resolve(), Path(output_path).resolve()
    scale = np.asarray(coordinate_scale, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('coordinate scale requires three finite positive ZYX factors')
    if min(workers, mask_workers, ray_workers, rays_per_task) < 1:
        raise ValueError('worker counts and rays per task must be positive')
    if out.exists() or out.is_relative_to(source) or out.is_relative_to(ct.path):
        raise ValueError('output must be a new directory outside the inputs')
    manifest, shards, ray_length = load_source(source)
    spans = []
    for _, a in shards:
        start = a['ray_origin_zyx'].astype(np.float64) * scale
        end = (a['ray_origin_zyx'].astype(np.float64)
               + (ray_length - 1) * a['ray_step_zyx'].astype(np.float64)) * scale
        eligible = (np.isfinite(start).all(1) & np.isfinite(end).all(1)
                    & (start >= 0).all(1) & (end >= 0).all(1)
                    & (start < ct.shape).all(1) & (end < ct.shape).all(1))
        if eligible.any():
            spans.extend((start[eligible, 0].min(), start[eligible, 0].max(),
                          end[eligible, 0].min(), end[eligible, 0].max()))
    z0 = max(0, int(np.floor(min(spans))) - 1) if spans else 0
    z1 = min(ct.shape[0], int(np.floor(max(spans))) + 2) if spans else 1
    report = dict(complete=False, source=str(source), source_fingerprint=manifest['fingerprint'],
                  ct=str(ct.path), ct_shape=list(ct.shape), settings=asdict(settings),
                  coordinate_scale_zyx=scale.tolist(), include_internal_cavities=True,
                  ray_t_range=[0, ray_length - 1], mask_z_range=[z0, z1],
                  mapping='CT index = floor(reference coordinate * scale), zero origin',
                  rule='remove entire ray if any traversed CT mask voxel is exterior',
                  workers=workers, mask_workers=mask_workers, ray_workers=ray_workers,
                  shards=[])
    out.mkdir(parents=True)
    (out / 'filter_audit').mkdir()
    write_json(out / 'mask_report.json', report)
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix='winding_ct_mask_', dir=temp_dir) as scratch:
        packed = np.memmap(Path(scratch) / 'mask.bits', mode='w+', dtype=np.uint8,
                           shape=(z1 - z0, ct.shape[1], (ct.shape[2] + 7) // 8))
        mask = PackedMask(packed, ct.shape, z0)
        with ThreadPoolExecutor(workers) as io_pool, ThreadPoolExecutor(mask_workers) as mask_pool:
            begin = z0
            while begin < z1:
                end = min(z1, (begin // ct.chunks[0] + 1) * ct.chunks[0])
                slab = read_ct_batch(ct, begin, end, io_pool, workers)

                def make_slice(z):
                    packed[z - z0] = np.packbits(make_ct_mask(slab[z - begin], settings),
                                                axis=1, bitorder='little')

                for _ in bounded_map(mask_pool, make_slice, range(begin, end), mask_workers):
                    pass
                del slab
                print(f'Mask Z {begin}:{end} / {z1}; elapsed {time.perf_counter()-started:.1f}s', flush=True)
                begin = end
        report['mask_seconds'] = time.perf_counter() - started
        filtered = copy.deepcopy(manifest)
        filtered['shards'] = []
        with ThreadPoolExecutor(ray_workers) as ray_pool:
            for original, a in shards:
                keep = np.empty(len(a['seed_winding']), dtype=bool)

                def check(begin):
                    end = min(len(keep), begin + rays_per_task)
                    keep[begin:end] = rays_inside(mask, a, begin, end, scale, ray_length - 1)

                for _ in bounded_map(ray_pool, check, range(0, len(keep), rays_per_task), ray_workers):
                    pass
                result = save_shard(out, original, a, keep)
                filtered['shards'].append(result)
                stats = dict(name=original['name'], input_rays=len(keep),
                             kept_rays=int(keep.sum()), removed_rays=int((~keep).sum()),
                             input_crossings=original['num_crossings'],
                             kept_crossings=result['num_crossings'])
                report['shards'].append(stats)
                write_json(out / 'mask_report.json', report)
                print(f'{original["name"]}: kept {keep.sum():,} / {len(keep):,} rays', flush=True)
        del mask, packed
    filtered['num_rays'] = sum(s['num_retained_rays'] for s in filtered['shards'])
    filtered['num_crossings'] = sum(s['num_crossings'] for s in filtered['shards'])
    filtered['ct_mask_filter'] = {k: v for k, v in report.items()
                                  if k not in ('complete', 'shards', 'mask_seconds')}
    filtered['fingerprint'] = manifest_fingerprint(filtered)
    report.update(complete=True, elapsed_seconds=time.perf_counter() - started,
                  kept_rays=filtered['num_rays'], kept_crossings=filtered['num_crossings'])
    write_json(out / 'mask_report.json', report)
    # A usable store manifest appears only after all arrays have been written.
    write_json(out / 'manifest.json', filtered)
    print(f'Finished in {report["elapsed_seconds"]:.1f}s: {out}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ct', type=Path, help='CT Zarr v2 array at the reviewed mask scale')
    parser.add_argument('source', type=Path, help='winding-inference crossing store')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--coordinate-scale', required=True, nargs=3, type=float,
                        metavar=('Z', 'Y', 'X'), help='CT voxels per reference-coordinate voxel')
    parser.add_argument('--workers', default=2, type=int, help='CT read workers (default 2)')
    parser.add_argument('--mask-workers', default=4, type=int)
    parser.add_argument('--ray-workers', default=4, type=int)
    parser.add_argument('--rays-per-task', default=16384, type=int)
    parser.add_argument('--temp-dir', type=Path, help='temporary packed mask location')
    args = parser.parse_args()
    run(args.ct, args.source, args.output_dir, coordinate_scale=args.coordinate_scale,
        workers=args.workers, mask_workers=args.mask_workers, ray_workers=args.ray_workers,
        rays_per_task=args.rays_per_task, temp_dir=args.temp_dir)


if __name__ == '__main__':
    main()
