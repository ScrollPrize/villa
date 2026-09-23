"""CUDA-resident geometry, pooling, signed filtering, and streamed Zarr export."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import multiprocessing
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from patch_normal_cuda import InwardGrid, filter_cuda, pool_quads_cuda
from patch_normal_tiled import PackedWriter, _read_quads
from patch_normal_samples import _worker_init


def check_memory_headroom(minimum_gib=16):
    """Stop between batches/tiles if Linux reports little available system RAM."""
    path = Path('/proc/meminfo')
    if path.exists():
        values = dict(line.split(':', 1) for line in path.read_text().splitlines())
        available = int(values['MemAvailable'].split()[0]) * 1024
        if available < minimum_gib * 2**30:
            raise MemoryError(f'Only {available / 2**30:.1f} GiB system RAM available; '
                              'stopping export to preserve desktop headroom. Resume later.')


def stage_quads_cuda(directory, entries, coordinate_scale, z_roi, workers,
                     tile_edge, radius, batch_quads):
    """Bound geometry residency; group on CUDA and spool tile geometry to disk."""
    directory = Path(directory)
    directory.mkdir()
    tiles = set()
    pending = []
    count = total = empty = 0
    last = time.monotonic()

    def flush():
        q = torch.as_tensor(np.concatenate(pending), device='cuda', dtype=torch.float64)
        keys, starts, ends, sources = group_quads_cuda(q, tile_edge, radius, z_roi)
        # Transfer one tile at a time: never materialize all duplicated quads.
        for key, a, b in zip(keys, starts, ends):
            key = tuple(map(int, key))
            tiles.add(key)
            block = q[sources[int(a):int(b)].long()].cpu().numpy()
            with (directory / ('_'.join(map(str, key)) + '.bin')).open('ab') as stream:
                block.tofile(stream)
        del q, sources

    with ProcessPoolExecutor(workers, initializer=_worker_init,
                             mp_context=multiprocessing.get_context('spawn')) as readers:
        work = ((p, coordinate_scale, z_roi) for p in entries)
        for index, quads in enumerate(readers.map(_read_quads, work, buffersize=workers), 1):
            total += len(quads)
            empty += not len(quads)
            for start in range(0, len(quads), batch_quads):
                part = quads[start:start + batch_quads]
                if count and count + len(part) > batch_quads:
                    check_memory_headroom()
                    flush()
                    pending.clear()
                    count = 0
                    torch.cuda.empty_cache()
                pending.append(part)
                count += len(part)
            if time.monotonic() - last > 10:
                print(f'Staged {index:,}/{len(entries):,} patches; bounded CUDA batches', flush=True)
                last = time.monotonic()
        if count:
            check_memory_headroom()
            flush()
    torch.cuda.empty_cache()
    if not tiles:
        raise ValueError('no valid surface samples in the requested ROI')
    return np.asarray(sorted(tiles)), dict(patch_count=len(entries),
        patches_without_quads_in_roi=empty, quad_count=total,
        geometry_batch_quads=batch_quads)


def group_quads_cuda(q,tile_edge,radius,z_roi):
    """Group overlapping quad bounds on CUDA, preserving source order per tile."""
    lo=torch.floor((q.amin(1)-radius)/tile_edge).to(torch.int32)
    hi=torch.floor((q.amax(1)+radius)/tile_edge).to(torch.int32)
    lo[:,0].clamp_(min=int(np.floor(z_roi[0]/tile_edge)))
    hi[:,0].clamp_(max=int(np.ceil(z_roi[1]/tile_edge))-1)
    extent=(hi-lo+1).amax(0).tolist()
    origin=lo.amin(0);shape=(hi.amax(0)-origin+1).tolist()
    all_keys=[];all_sources=[]
    source=torch.arange(len(q),device='cuda',dtype=torch.int32)
    for offset in itertools.product(*(range(int(e)) for e in extent)):
        tile=lo+torch.tensor(offset,device='cuda',dtype=torch.int32)
        good=(tile<=hi).all(1)
        local=tile[good]-origin
        all_keys.append((local[:,0]*shape[1]+local[:,1])*shape[2]+local[:,2])
        all_sources.append(source[good])
    del lo,hi,source,tile,good,local
    keys=torch.cat(all_keys);sources=torch.cat(all_sources)
    del all_keys,all_sources
    order=torch.argsort(keys.long()*len(q)+sources.long())
    keys=keys[order];sources=sources[order]
    unique,counts=torch.unique_consecutive(keys,return_counts=True)
    unique=unique.cpu().numpy();counts=counts.cpu().numpy();origin=origin.cpu().numpy()
    tiles=np.stack((unique//(shape[1]*shape[2]),unique//shape[2]%shape[1],unique%shape[2]),1)+origin
    ends=np.cumsum(counts);starts=np.r_[0,ends[:-1]]
    return tiles,starts,ends,sources


def export_cuda(entries, output, reference, *, coordinate_scale, z_roi, cell_size,
                surface_spacing, width, tile_edge, workers, chunk_size, metadata,
                output_format, base_scale, sign_grid_spacing=0, resume=False,
                geometry_batch_quads=250_000):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = last = time.monotonic()
    model = reference['model']
    metadata.update(tile_edge_fitter_voxels=tile_edge, geometry_backend='cuda',
        geometry_residency='bounded batches; tile geometry staged on disk',
        sign_grid_spacing_fitter_voxels=float(sign_grid_spacing),
        inward_alignment_evaluation='interpolated checkpoint direction with exact ambiguous fallback'
            if sign_grid_spacing else 'exact checkpoint direction',
        sign_grid_fallback_min_corner_cosine=.75 if sign_grid_spacing else None)
    # Reject incompatible resumes before reading or staging the input geometry.
    if resume:
        from patch_normal_volume import TiledVolumeWriter
        TiledVolumeWriter.check_resume(output, cell_size, base_scale,
                                      tile_edge // int(cell_size), metadata)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}.', dir=output.parent) as tmp:
        geometry = Path(tmp) / 'geometry'
        tiles, stats = stage_quads_cuda(geometry, entries, coordinate_scale, z_roi,
                                        workers, tile_edge, width // 2, geometry_batch_quads)
        metadata.update(stats, spatial_tiles=len(tiles), staging_seconds=time.monotonic()-started)
        print(f'Staged {len(tiles):,} tiles; releasing grouping allocations', flush=True)
        torch.cuda.empty_cache()
        with torch.no_grad(), ThreadPoolExecutor(workers) as writers:
            transform = model.get_slice_to_spiral_transform()
            dr = model.get_dr_per_winding()
            metadata['sign_grid_build_seconds'] = 0.
            metadata['sign_grid_nodes'] = 0
            if output_format in ('zarr', 'compact'):
                if output_format == 'compact':
                    from patch_normal_compact_export import CompactPatchNormalWriter as Writer
                else:
                    from patch_normal_volume import TiledVolumeWriter as Writer
                shape = ((tiles.max(0)+1)*(tile_edge//int(cell_size))).tolist()
                writer = Writer(output, shape, cell_size, base_scale,
                                tile_edge//int(cell_size), metadata,
                                **({'resume': resume} if output_format == 'zarr' else {}))
            else:
                writer = PackedWriter(tmp)
            pending = deque()
            fine_count = fallback_count = reused = 0
            for index, key in enumerate(tiles, 1):
                check_memory_headroom()
                path = geometry / ('_'.join(map(str, key)) + '.bin')
                if resume and writer.reuse_chunk(tuple(map(int, key))):
                    reused += 1
                else:
                    quads = np.fromfile(path, dtype=np.float64).reshape(-1, 4, 3)
                    lo = key*tile_edge-width//2
                    hi = (key+1)*tile_edge+width//2
                    p, n = pool_quads_cuda(quads, lo, hi, z_roi, surface_spacing)
                    grid = None
                    if sign_grid_spacing and len(p):
                        before = time.monotonic()
                        bounds = torch.stack((p.amin(0), p.amax(0))).cpu().numpy()
                        grid = InwardGrid(bounds[0], bounds[1], sign_grid_spacing, transform, dr, chunk_size)
                        metadata['sign_grid_build_seconds'] += time.monotonic()-before
                        metadata['sign_grid_nodes'] += len(grid.field)
                    result, counts = filter_cuda(p, n, key, tile_edge=tile_edge, cell_size=cell_size,
                        width=width, transform=transform, dr=dr, grid=grid, chunk_size=chunk_size)
                    fine_count += counts['fine_samples']
                    fallback_count += counts['exact_fallback']
                    if result is not None:
                        arrays = {k: v.cpu().numpy() for k, v in result.items()}
                        if output_format in ('zarr', 'compact'):
                            pending.append(writers.submit(writer.append, arrays))
                            if len(pending) >= workers:
                                pending.popleft().result()
                        else:
                            writer.append(arrays)
                    del quads, p, n, grid, result
                path.unlink()
                if index % 32 == 0:
                    torch.cuda.empty_cache()
                if time.monotonic()-last > 10 or index == len(tiles):
                    print(f'CUDA tiles {index:,}/{len(tiles):,}; reused {reused:,}; '
                          f'{writer.valid_count:,} valid normals; {time.monotonic()-started:.1f}s; '
                          f'CUDA reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB', flush=True)
                    last = time.monotonic()
            for future in pending:
                future.result()
            unknown = bool(getattr(writer, 'reused_chunks', 0))
            metadata.update(sample_count=None if unknown else writer.count,
                pooled_position_count=None if unknown else writer.count,
                valid_sign_count=writer.valid_count, processing_seconds=time.monotonic()-started,
                fine_samples_including_halos=fine_count, exact_sign_fallback_count=fallback_count,
                reused_chunks=reused, counts_scope='current run only for fine samples/grid/fallback',
                cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            writer.finish(output, metadata)
    torch.cuda.empty_cache()
    print(f'Wrote {writer.valid_count:,} valid normals with CUDA to {output}', flush=True)
    return metadata
