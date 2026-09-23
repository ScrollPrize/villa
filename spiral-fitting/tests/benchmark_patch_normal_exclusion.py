"""CPU-only, bounded real-data benchmark; never opens a CUDA device.

Run from spiral-fitting:
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python \
    tests/benchmark_patch_normal_exclusion.py \
    --source /mnt/raid_nvme/spiral_dataset_working/patch_normals.zarr

Copies a small real brick neighborhood into a temporary sidecar, retaining
its exact uint8 bytes. Times patch sampling before/after exclusion, with one
warmup and fresh uniformly scattered points for every paired measurement.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from compact_patch_normals import CompactPatchNormalPool
from pack_resident_pools import open_pool
from patch_normals import PatchNormals, override_patch_normals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=200000)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((args.source / 'manifest.json').read_text())
    cell = float(manifest['cell_size_fitter_voxels'])
    meta, table, coords, pools = open_pool(args.source / 'signed_normals_u8.respool')
    brick = np.array(meta['brick_shape'])
    # Four bricks either side of a real occupied brick; leave a halo around
    # the query box so this is an interior region of the original export.
    center = coords[len(coords) // 2]
    lo = np.maximum(center - 4, 0)
    hi = np.minimum(center + 5, table.shape)
    original_ids = table[tuple(slice(a, b) for a, b in zip(lo, hi))]
    ids = np.unique(np.append(original_ids, 0))
    remap = np.zeros(meta['rows'], dtype=np.int32)
    remap[ids] = np.arange(len(ids), dtype=np.int32)
    local_table = remap[original_ids]
    local_coords = coords[ids].copy() - lo
    local_coords[0] = -1
    shape = tuple(((hi - lo) * brick).tolist())
    with tempfile.TemporaryDirectory(prefix='patch-exclusion-bench-') as directory:
        root = Path(directory)
        local_meta = dict(format='respool', version=2, rows=len(ids),
                          array_shape=shape, brick_shape=brick.tolist(), channels=['nx', 'ny', 'nz'])
        (root / 'meta.json').write_text(json.dumps(local_meta))
        np.save(root / 'table.npy', local_table)
        np.save(root / 'brick_coords.npy', local_coords)
        present = np.zeros((len(ids), int(np.prod(brick))), dtype=bool)
        for ch, pool in enumerate(pools):
            values = pool[ids]
            values.tofile(root / f'channel_{ch}.u8')
            present |= values != 0
        coverage = present[local_table].reshape(*local_table.shape, *brick)
        coverage = coverage.transpose(0, 3, 1, 4, 2, 5).reshape(shape)
        from scipy.ndimage import distance_transform_edt
        reference_near = distance_transform_edt(~coverage) <= 8. / cell
        start = time.perf_counter()
        cache = CompactPatchNormalPool(root, device='cpu', exclusion_radius_cells=8. / cell)
        build_seconds = time.perf_counter() - start
        store = PatchNormals(cache, cell, shape, (0, shape[0]))
        mask = cache.exclusion
        rng = np.random.default_rng(42)
        durations = {'pointwise_ms': [], 'exclusion_ms': []}
        fallback = torch.tensor([0., 1., 0.]).expand(args.samples, 3)
        weights = torch.ones(args.samples)
        for i in range(args.repeats + 1):
            p = (rng.random((args.samples, 3)) * (np.array(shape) - 16) + 8) * cell
            points = torch.from_numpy(p.astype(np.float32))
            for name, exclusion in [('pointwise_ms', None), ('exclusion_ms', mask)]:
                cache.exclusion = exclusion
                start = time.perf_counter()
                _, result_weight = override_patch_normals({'patch_normals': store}, points, fallback, weights)
                elapsed = (time.perf_counter() - start) * 1000
                if i:
                    durations[name].append(elapsed)
        cache.exclusion = mask
        _, _, near = store.sample(points, return_exclusion=True)
        indices = torch.floor(points / cell).long().numpy()
        np.testing.assert_array_equal(near.numpy(), reference_near[tuple(indices.T)])
        output = dict(source=str(args.source), source_grid_origin=(lo * brick).tolist(),
                      shape=shape, samples=args.samples, repeats=args.repeats,
                      cpu_threads=1, build_seconds=build_seconds,
                      verified_mask_lookups=args.samples,
                      compact_bytes=cache.pool_bytes - cache.exclusion_bytes,
                      exclusion_bytes=cache.exclusion_bytes,
                      excluded_empty_fraction=float((result_weight == 0).float().mean()))
        for name, times in durations.items():
            output[name] = dict(mean=float(np.mean(times)), min=min(times),
                                median=float(np.median(times)), p95=float(np.percentile(times, 95)),
                                max=max(times))
        print(json.dumps(output, indent=2))
        store.close()


if __name__ == '__main__':
    main()
