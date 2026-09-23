"""Bounded-memory bilinear sampling, signed box filtering, and packed export."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import numpy as np

from patch_normal_filter import box_average, cube_neighbors
from patch_normal_samples import (NormalPool, _worker_init, bilinear_samples,
                                  orient_normals, patch_quads, thin_samples)
from tifxyz import load_tifxyz


def _read_quads(work):
    path, scale, roi = work
    patch = load_tifxyz(str(path), skip_empty=True)
    if patch is None:
        return np.empty((0, 4, 3))
    q = patch_quads(patch) * scale
    return q[(q[:, :, 0].max(axis=1) >= roi[0]) & (q[:, :, 0].min(axis=1) < roi[1])]


def stage_quads(directory, batches, *, tile_edge, radius, z_roi):
    """Write quad shards in source order, including complete convolution halos.

    Only geometry is staged, never the much larger one-voxel point cloud.
    Source order within each shard preserves nearest-cell tie breaking.
    """
    directory = Path(directory)
    directory.mkdir()
    tiles = set()
    count = empty = 0
    last = time.monotonic()
    for batch in batches:
        count += 1
        empty += not len(batch)
        for begin in range(0, len(batch), 20_000):
            q = batch[begin:begin + 20_000]
            lo = np.floor((q.min(axis=1) - radius) / tile_edge).astype(np.int64)
            hi = np.floor((q.max(axis=1) + radius) / tile_edge).astype(np.int64)
            # A tile core must intersect the half-open output ROI.
            lo[:, 0] = np.maximum(lo[:, 0], int(np.floor(z_roi[0] / tile_edge)))
            hi[:, 0] = np.minimum(hi[:, 0], int(np.ceil(z_roi[1] / tile_edge)) - 1)
            extent = np.maximum(hi - lo + 1, 0).max(axis=0)
            all_keys, all_rows = [], []
            for offset in itertools.product(*(range(int(e)) for e in extent)):
                key = lo + offset
                rows = np.flatnonzero((key <= hi).all(axis=1))
                if len(rows):
                    all_keys.append(key[rows]); all_rows.append(rows)
            if not all_rows:
                continue
            keys, rows = np.concatenate(all_keys), np.concatenate(all_rows)
            order = np.lexsort((rows, keys[:, 2], keys[:, 1], keys[:, 0]))
            keys, rows = keys[order], rows[order]
            starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1]
            for a, b in zip(starts, np.r_[starts[1:], len(rows)]):
                key = tuple(keys[a].tolist())
                tiles.add(key)
                with (directory / ('_'.join(map(str, key)) + '.bin')).open('ab') as stream:
                    q[rows[a:b]].tofile(stream)
        if time.monotonic() - last > 10:
            print(f'Staged {count:,} patches in {len(tiles):,} spatial tiles', flush=True)
            last = time.monotonic()
    return sorted(tiles), {'patch_count': count, 'patches_without_quads_in_roi': empty}


def pool_tile(work):
    path, key, tile_edge, radius, roi, spacing = work
    lo = np.asarray(key) * tile_edge
    hi = lo + tile_edge
    pool = NormalPool(1)
    q = np.memmap(path, dtype=np.float64, mode='r').reshape(-1, 4, 3)
    # Limit both quad bookkeeping and generated sample batches.
    for begin in range(0, len(q), 20_000):
        for p, n in bilinear_samples(q[begin:begin + 20_000], spacing):
            keep = (((p >= lo - radius) & (p < hi + radius)).all(axis=1)
                    & (p[:, 0] >= roi[0]) & (p[:, 0] < roi[1]))
            pool.add(p[keep], n[keep])
    p, n = pool.finish()
    return key, p, n


def filter_tile(p, n, key, *, tile_edge, cell_size, width, reference,
                device, chunk_size):
    """Match the experiment: dedup fine voxels, sign, average, sample, re-sign."""
    import torch
    lo = np.asarray(key) * tile_edge
    core = np.flatnonzero(((p >= lo) & (p < lo + tile_edge)).all(axis=1))
    positions, selected = thin_samples(p[core], core[:, None], cell_size)
    chosen = selected[:, 0]
    if not len(chosen):
        return None
    model = reference['model']
    transform, dr = model.get_slice_to_spiral_transform(), model.get_dr_per_winding()
    with torch.no_grad():
        signed = orient_normals(p, n, transform, dr, device=device, chunk_size=chunk_size)
        normals = signed['normal_zyx'].astype(np.float64)
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
        neighbor = cube_neighbors(np.floor(p).astype(np.int64),
                                  np.floor(p[chosen]).astype(np.int64), width)
        mean, _, _, valid = box_average(normals, neighbor, signed['sign_valid'])
        valid &= signed['sign_valid'][chosen]
        # Keep occupied cells even when the mean cancels or has no signed support.
        result = dict(position_zyx=positions, normal_zyx=np.zeros((len(chosen), 3), np.float32),
                      sign_valid=np.zeros(len(chosen), bool),
                      inward_alignment=np.zeros(len(chosen), np.float32),
                      presence=np.zeros(len(chosen), np.uint8))
        if valid.any():
            oriented = orient_normals(positions[valid], mean[valid], transform, dr,
                                      device=device, chunk_size=chunk_size)
            for name in result:
                if name != 'position_zyx':
                    result[name][valid] = oriented[name]
    # Preserve float64 locations: float32 rounding can move a sample into a
    # neighboring output voxel. Zarr rasterization must recover the same cells.
    return result


class PackedWriter:
    """Spool columns and publish a standard compressed NPZ without full arrays."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.count = 0
        self.valid_count = 0
        self.schema = {}

    def append(self, arrays):
        for name, values in arrays.items():
            self.schema[name] = (values.dtype, values.shape[1:])
            with (self.directory / f'{name}.bin').open('ab') as stream:
                values.tofile(stream)
        self.count += len(arrays['position_zyx'])
        self.valid_count += int(arrays['sign_valid'].sum())

    def finish(self, output, metadata):
        if not self.count:
            raise ValueError('no valid surface samples in the requested ROI')
        temporary = self.directory / 'packed.npz'
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as archive:
            for name, (dtype, trailing) in self.schema.items():
                with archive.open(f'{name}.npy', 'w', force_zip64=True) as dest:
                    np.lib.format.write_array_header_2_0(dest, {
                        'descr': np.lib.format.dtype_to_descr(dtype), 'fortran_order': False,
                        'shape': (self.count, *trailing)})
                    with (self.directory / f'{name}.bin').open('rb') as source:
                        shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
                (self.directory / f'{name}.bin').unlink()
            with archive.open('metadata_json.npy', 'w', force_zip64=True) as dest:
                np.lib.format.write_array(dest, np.asarray(json.dumps(metadata)), allow_pickle=False)
        os.link(temporary, output)


def export_filtered(entries, output, reference, *, coordinate_scale, z_roi, cell_size,
                    surface_spacing, width, tile_edge, workers, device, chunk_size,
                    metadata, output_format='npz', base_scale=4, sign_grid_spacing=0,
                    resume=False, geometry_batch_quads=250_000):
    """Stage geometry, process independent tiles, and stream the resulting store."""
    if str(device).startswith('cuda'):
        import torch
        from patch_normal_cuda_export import export_cuda
        with torch.cuda.device(device):
            return export_cuda(entries,output,reference,coordinate_scale=coordinate_scale,
                z_roi=z_roi,cell_size=cell_size,surface_spacing=surface_spacing,width=width,
                tile_edge=tile_edge,workers=workers,chunk_size=chunk_size,metadata=metadata,
                output_format=output_format,base_scale=base_scale,sign_grid_spacing=sign_grid_spacing,
                resume=resume,geometry_batch_quads=geometry_batch_quads)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = last = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as temporary, ExitStack() as stack:
        directory = Path(temporary)
        executor = (stack.enter_context(ProcessPoolExecutor(
            workers, initializer=_worker_init, mp_context=multiprocessing.get_context('spawn')))
            if workers > 1 else None)
        def ordered_map(function, work):
            return (executor.map(function, work, buffersize=workers * 2)
                    if executor is not None else map(function, work))
        batches = ordered_map(_read_quads, ((p, coordinate_scale, z_roi) for p in entries))
        tiles, stats = stage_quads(directory / 'geometry', batches, tile_edge=tile_edge,
                                  radius=width // 2, z_roi=z_roi)
        metadata.update(stats, spatial_tiles=len(tiles), tile_edge_fitter_voxels=tile_edge)
        if output_format == 'npz':
            writer = PackedWriter(directory)
        elif output_format == 'compact':
            from patch_normal_compact_export import CompactPatchNormalWriter
            if len(tiles) == 0:
                raise ValueError('no valid surface samples in the requested ROI')
            shape = ((np.max(tiles, axis=0) + 1) * (tile_edge // int(cell_size))).tolist()
            writer = CompactPatchNormalWriter(output, shape, cell_size, base_scale,
                                              tile_edge // int(cell_size), metadata)
        else:
            from patch_normal_volume import TiledVolumeWriter
            if len(tiles) == 0:
                raise ValueError('no valid surface samples in the requested ROI')
            shape = ((np.max(tiles, axis=0) + 1) * (tile_edge // int(cell_size))).tolist()
            writer = TiledVolumeWriter(output, shape, cell_size, base_scale,
                                       tile_edge // int(cell_size), metadata)
        work = ((directory / 'geometry' / ('_'.join(map(str, key)) + '.bin'),
                 key, tile_edge, width // 2, z_roi, surface_spacing) for key in tiles)
        for index, (key, p, n) in enumerate(ordered_map(pool_tile, work), 1):
            result = filter_tile(p, n, key, tile_edge=tile_edge, cell_size=cell_size,
                                 width=width, reference=reference, device=device, chunk_size=chunk_size)
            if result is not None:
                writer.append(result)
            (directory / 'geometry' / ('_'.join(map(str, key)) + '.bin')).unlink()
            if time.monotonic() - last > 10 or index == len(tiles):
                print(f'Filtered {index:,}/{len(tiles):,} tiles; {writer.count:,} output normals', flush=True)
                last = time.monotonic()
        metadata.update(sample_count=writer.count, pooled_position_count=writer.count,
                        valid_sign_count=writer.valid_count, processing_seconds=time.monotonic() - start)
        writer.finish(output, metadata)
    print(f'Wrote {writer.count:,} box-filtered normals to {output}', flush=True)
    return metadata
