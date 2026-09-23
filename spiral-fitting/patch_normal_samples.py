"""Export spatially pooled, inward-facing patch normals as packed NPZ or Zarr.

Defaults: one-voxel bilinear sampling, signed 3x3x3 box averaging, 4-voxel output.
A reference spiral checkpoint chooses signs using its decreasing-winding gradient.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from tifxyz import load_tifxyz


def patch_quads(patch, *, with_indices=False):
    """Finite, fully valid corners in TL, TR, BL, BR order, in stored coordinates."""
    grid = patch.zyxs.numpy().astype(np.float64)
    valid = patch.valid_quad_mask.numpy().copy()
    finite = np.isfinite(grid).all(axis=-1)
    valid &= (finite[:-1, :-1] & finite[1:, :-1]
              & finite[:-1, 1:] & finite[1:, 1:])
    quads = np.stack((grid[:-1, :-1][valid], grid[:-1, 1:][valid],
                      grid[1:, :-1][valid], grid[1:, 1:][valid]), axis=1)
    return (quads, np.argwhere(valid)) if with_indices else quads


def triangle_normals_from_quads(quads, coordinate_scale=1.0):
    """Return triangle centroids and unit geometric normals in ZYX order."""
    tl, tr, bl, br = np.moveaxis(quads, 1, 0)
    # Match Patch.project's diagonal from bottom-left to top-right.
    a, b, c = np.concatenate((tl, br)), np.concatenate((bl, tr)), np.concatenate((tr, bl))
    edge, other = b - a, c - a
    normal = np.cross(edge, other)
    area2 = np.linalg.norm(normal, axis=-1)
    edge_length = np.linalg.norm(edge, axis=-1)
    other_length = np.linalg.norm(other, axis=-1)
    keep = (edge_length > 0) & (area2 > 1e-8 * edge_length * other_length)
    positions = (a[keep] + b[keep] + c[keep]) / 3 * coordinate_scale
    return positions, normal[keep] / area2[keep, None]


def triangle_normals(patch, coordinate_scale=1.0):
    return triangle_normals_from_quads(patch_quads(patch), coordinate_scale)


def bilinear_geometry(quads, uv, *, return_area=False):
    """Evaluate points and analytic surface derivatives for paired quad/UV rows."""
    tl, tr, bl, br = np.moveaxis(quads, 1, 0)
    u, v = uv[:, :1], uv[:, 1:]
    du = (bl - tl) * (1 - v) + (br - tr) * v
    dv = (tr - tl) * (1 - u) + (br - bl) * u
    position = (tl * (1 - u) + bl * u) * (1 - v) + (tr * (1 - u) + br * u) * v
    normal = np.cross(du, dv)
    norm = np.linalg.norm(normal, axis=1)
    valid = norm > 1e-8 * np.linalg.norm(du, axis=1) * np.linalg.norm(dv, axis=1)
    normal = normal / np.where(valid, norm, 1)[:, None]
    return (position, normal, valid, norm) if return_area else (position, normal, valid)


def bilinear_samples(quads, spacing=1., *, coordinate_scale=1., batch_size=250_000,
                     with_provenance=False):
    """Yield bounded batches on each bilinear quad's parameter-cell centers.

    Subdivision counts use the longest opposing edges after coordinate scaling.
    Adjacent samples along either parameter axis are at most ``spacing`` apart
    in world space. This does not add geometric information to the input mesh.
    With provenance, also yield source-quad indices, UV coordinates, and the
    surface-area quadrature weight for each parameter-cell center.
    """
    if not np.isfinite(spacing) or spacing <= 0 or batch_size < 1:
        raise ValueError('surface spacing and batch size must be positive')
    quads = np.asarray(quads, dtype=np.float64) * coordinate_scale
    if not np.isfinite(quads).all():
        raise ValueError('quad coordinates must be finite')
    if not len(quads):
        return
    tl, tr, bl, br = np.moveaxis(quads, 1, 0)
    nu = np.maximum(1, np.ceil(np.maximum(np.linalg.norm(bl - tl, axis=1),
                                         np.linalg.norm(br - tr, axis=1)) / spacing)).astype(np.int64)
    nv = np.maximum(1, np.ceil(np.maximum(np.linalg.norm(tr - tl, axis=1),
                                         np.linalg.norm(br - bl, axis=1)) / spacing)).astype(np.int64)
    ends = np.cumsum(nu * nv)
    starts = np.r_[0, ends[:-1]]
    for begin in range(0, int(ends[-1]), batch_size):
        sample = np.arange(begin, min(begin + batch_size, ends[-1]))
        quad = np.searchsorted(ends, sample, side='right')
        local = sample - starts[quad]
        uv = np.stack(((local // nv[quad] + .5) / nu[quad],
                       (local % nv[quad] + .5) / nv[quad]), axis=1)
        position, normal, valid, area = bilinear_geometry(quads[quad], uv, return_area=True)
        if with_provenance:
            yield (position[valid], normal[valid], quad[valid], uv[valid],
                   area[valid] / (nu[quad[valid]] * nv[quad[valid]]))
        else:
            yield position[valid], normal[valid]


class NormalPool:
    """Deterministic hierarchical pooling without retaining all dense samples."""

    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.levels = []

    def add(self, positions, normals):
        if not len(positions):
            return
        pooled = thin_samples(positions, normals, self.cell_size)
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            previous = self.levels[level]
            pooled = thin_samples(np.concatenate((previous[0], pooled[0])),
                                  np.concatenate((previous[1], pooled[1])), self.cell_size)
            self.levels[level] = None
            level += 1
        if level == len(self.levels):
            self.levels.append(pooled)
        else:
            self.levels[level] = pooled

    def finish(self):
        pooled = [item for item in reversed(self.levels) if item is not None]
        if not pooled:
            return np.empty((0, 3)), np.empty((0, 3))
        return thin_samples(np.concatenate([item[0] for item in pooled]),
                            np.concatenate([item[1] for item in pooled]), self.cell_size)


def thin_samples(positions, normals, cell_size):
    """Keep the centroid nearest each cell center; stable ties use input order."""
    if not len(positions) or cell_size == 0:
        return positions, normals
    cells = np.floor(positions / cell_size).astype(np.int64)
    distance2 = np.square(positions / cell_size - cells - 0.5).sum(axis=1)
    order = np.lexsort((distance2, cells[:, 2], cells[:, 1], cells[:, 0]))
    ordered = cells[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    indices = order[first]
    return positions[indices], normals[indices]


def load_sign_reference(checkpoint_path, umbilicus=None, device='cpu'):
    """Reconstruct the checkpoint using the existing surface-export loader."""
    import torch
    from checkpoint_io import load_checkpoint_cpu
    from flatten_spiral_checkpoint import (_build_model, _checkpoint_config,
                                          _resolve_umbilicus)

    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = load_checkpoint_cpu(str(checkpoint_path))
    if checkpoint.get('frozen_epochs'):
        raise ValueError('signed export does not yet support baked/frozen epochs')
    if umbilicus is None:
        recorded = (checkpoint.get('input_manifest') or {}).get('umbilicus')
        if recorded and Path(recorded).is_file():
            umbilicus = Path(recorded)
    umbilicus = _resolve_umbilicus(
        checkpoint_path, Path(umbilicus) if umbilicus is not None else None)
    # Historical shell-loss parameter is unrelated to the spatial transform.
    state = dict(checkpoint['spiral_and_transform'])
    state.pop('shell_outer_winding_param', None)
    checkpoint = {**checkpoint, 'spiral_and_transform': state}
    model = _build_model(checkpoint, _checkpoint_config(checkpoint),
                         umbilicus, torch.device(device))
    return {
        'model': model,
        'z_roi': (int(checkpoint['z_begin']), int(checkpoint['z_end'])),
        'metadata': {
            'checkpoint': str(checkpoint_path),
            'model_state_sha256': checkpoint.get('model_state_sha256'),
            'completed_iterations': checkpoint.get('completed_iterations'),
            'umbilicus': str(umbilicus),
            'checkpoint_z_roi': [checkpoint['z_begin'], checkpoint['z_end']],
        },
    }


def orient_normals(positions, normals, transform, dr_per_winding, *, device='cpu',
                   chunk_size=8192, progress=False):
    """Choose each patch normal's sign using the exact winding covector.

    Undefined or nearly orthogonal reference gradients produce sign_valid=False
    and zero normals. Alignment is the absolute cosine with the inward gradient.
    """
    from fit_spiral import inward_winding_direction

    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    normal = np.asarray(normals, dtype=np.float64).copy()
    norm = np.linalg.norm(normal, axis=-1)
    if not np.isfinite(normal).all() or np.any(norm == 0):
        raise ValueError('patch normals must be finite nonzero vectors')
    normal /= norm[:, None]
    dot = np.empty(len(positions), dtype=np.float64)
    report_batch = max(chunk_size, (65536 // chunk_size) * chunk_size)
    next_report = time.monotonic() + 10
    for start in range(0, len(positions), report_batch):
        end = min(start + report_batch, len(positions))
        inward = inward_winding_direction(
            transform, positions[start:end], device=device, chunk_size=chunk_size,
            dr_per_winding=dr_per_winding)
        dot[start:end] = np.sum(normal[start:end] * inward, axis=-1)
        if progress and (time.monotonic() >= next_report or end == len(positions)):
            print(f'oriented {end:,}/{len(positions):,} normals', flush=True)
            next_report = time.monotonic() + 10
    valid = np.isfinite(dot) & (np.abs(dot) > 1e-6)
    normal *= np.where(dot < 0, -1., 1.)[:, None]
    normal[~valid] = 0
    return {
        'position_zyx': np.asarray(positions, dtype=np.float32),
        'normal_zyx': normal.astype(np.float32),
        'sign_valid': valid,
        'inward_alignment': np.where(valid, np.clip(np.abs(dot), 0, 1), 0).astype(np.float32),
        'presence': valid.astype(np.uint8) * 255,
    }


def _worker_init():
    import torch
    torch.set_num_threads(1)


def _sample_patch(work):
    path, coordinate_scale, z_roi, cell_size, surface_spacing = work
    # A triangle may cross the ROI without any vertex inside it, so avoid
    # the loader's vertex-based z filter.
    patch = load_tifxyz(str(path), skip_empty=True)
    if patch is None:
        return np.empty((0, 3)), np.empty((0, 3))
    if surface_spacing is not None:
        quads = patch_quads(patch)
        zs = quads[:, :, 0] * coordinate_scale
        quads = quads[(zs.max(axis=1) >= z_roi[0]) & (zs.min(axis=1) < z_roi[1])]
        pool = NormalPool(cell_size)
        for position, normal in bilinear_samples(quads, surface_spacing, coordinate_scale=coordinate_scale):
            keep = (position[:, 0] >= z_roi[0]) & (position[:, 0] < z_roi[1])
            pool.add(position[keep], normal[keep])
        return pool.finish()
    position, normal = triangle_normals(patch, coordinate_scale)
    keep = (position[:, 0] >= z_roi[0]) & (position[:, 0] < z_roi[1])
    return thin_samples(position[keep], normal[keep], cell_size)


def pool_patch_normals(entries, coordinate_scale, z_roi, cell_size, workers, surface_spacing=None):
    """Hierarchical merging bounds memory when many input patches overlap."""
    work = ((path, coordinate_scale, z_roi, cell_size, surface_spacing) for path in entries)
    executor = (ProcessPoolExecutor(workers, initializer=_worker_init,
                                   mp_context=multiprocessing.get_context('spawn'))
                if workers > 1 else None)
    pool = NormalPool(cell_size)
    positions, normals, pending = [], [], 0
    local_count = empty_count = 0
    try:
        results = (executor.map(_sample_patch, work, buffersize=workers * 2)
                   if executor is not None else map(_sample_patch, work))
        for index, (position, normal) in enumerate(results, 1):
            positions.append(position)
            normals.append(normal)
            pending += len(position)
            local_count += len(position)
            empty_count += not len(position)
            if pending >= 1_000_000 or index == len(entries):
                pool.add(np.concatenate(positions), np.concatenate(normals))
                positions, normals, pending = [], [], 0
            if index % 1000 == 0 or index == len(entries):
                print(f'processed {index:,}/{len(entries):,} patches', flush=True)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    position, normal = pool.finish()
    return position, normal, {
        'patch_count': len(entries),
        'patches_without_samples': empty_count,
        'positions_before_global_pooling': local_count,
        'pooled_position_count': len(position),
    }


def extract_patch_normals(patches_dir, output, *, checkpoint, coordinate_scale=1.0,
                          z_roi=None, cell_size=None, umbilicus=None, device='auto',
                          chunk_size=65536, workers=1, surface_spacing=None,
                          box_width=None, tile_edge=None, output_format='npz', base_scale=4.,
                          sign_grid_spacing=0., resume=False, cuda_memory_gb=8.,
                          geometry_batch_quads=250_000):
    """Write one signed normal store; all spatial arguments use fitter voxels."""
    patches_dir, output = Path(patches_dir), Path(output)
    packed_input = patches_dir.is_file()
    if cell_size is None:
        if packed_input:
            with np.load(patches_dir) as samples:
                cell_size = json.loads(str(samples['metadata_json']))['cell_size_output_voxels']
        else:
            cell_size = 4.
    if box_width is None:
        box_width = 0 if packed_input else 3
    if box_width not in (0, 3, 5):
        raise ValueError('box_width must be 0 (disabled), 3, or 5')
    if packed_input and box_width:
        raise ValueError('box filtering requires original patches, not packed samples')
    if box_width and surface_spacing is None:
        surface_spacing = 1.
    if output_format not in ('npz', 'zarr', 'compact'):
        raise ValueError('output_format must be npz, zarr, or compact')
    if output_format in ('zarr', 'compact') and not box_width:
        raise ValueError('direct volume export requires box filtering; use patch_normal_volume.py for packed input')
    if not np.isfinite(base_scale) or base_scale <= 0:
        raise ValueError('base_scale must be finite and positive')
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError('coordinate_scale must be finite and positive')
    if not np.isfinite(cell_size) or cell_size < 0:
        raise ValueError('cell_size must be finite and nonnegative')
    if surface_spacing is not None and (not np.isfinite(surface_spacing) or surface_spacing <= 0):
        raise ValueError('surface spacing must be finite and positive')
    if tile_edge is None:
        tile_edge=256 if str(device).startswith('cuda') or device=='auto' else 64
    if not np.isfinite(sign_grid_spacing) or sign_grid_spacing<0:
        raise ValueError('sign grid spacing must be finite and nonnegative')
    if sign_grid_spacing and (not box_width or packed_input):
        raise ValueError('sign grid lookup requires original patches and box filtering')
    if box_width and (cell_size < 1 or int(cell_size) != cell_size
                      or tile_edge < 1 or int(tile_edge) != tile_edge or tile_edge % cell_size):
        raise ValueError('box filtering requires integer output cells and a tile edge divisible by cell size')
    if z_roi is not None and (len(z_roi) != 2 or not np.isfinite(z_roi).all()
                              or z_roi[0] >= z_roi[1]):
        raise ValueError('z_roi must be a finite increasing pair')
    if resume and (output_format != 'zarr' or not box_width or packed_input):
        raise ValueError('resume requires filtered CUDA Zarr export from original patches')
    if not np.isfinite(cuda_memory_gb) or cuda_memory_gb <= 0 or geometry_batch_quads < 1:
        raise ValueError('CUDA memory budget and geometry batch size must be positive')
    if output.exists() and not resume:
        raise FileExistsError(output)
    if chunk_size <= 0 or workers <= 0:
        raise ValueError('chunk_size and workers must be positive')
    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if resume and not str(device).startswith('cuda'):
        raise ValueError('resume requires CUDA')
    if sign_grid_spacing and not str(device).startswith('cuda'):
        raise ValueError('accelerated sign grid lookup requires CUDA')
    print(f'using {device} for checkpoint evaluation', flush=True)
    if str(device).startswith('cuda'):
        total = torch.cuda.get_device_properties(device).total_memory
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(min(cuda_memory_gb * 2**30 / total, 1.), index)
        # Establish the CUDA context before CPU checkpoint loading pressures
        # shared system memory on integrated/unified-memory GPUs.
        torch.empty(1,device=device)
    reference = load_sign_reference(checkpoint, umbilicus, device)
    domain = reference['z_roi']
    if z_roi is None:
        z_roi = domain
    elif z_roi[0] < domain[0] or z_roi[1] > domain[1]:
        raise ValueError(f'z_roi extends outside checkpoint domain {domain}')
    base_metadata = {
        **reference['metadata'],
        'artifact_type': 'patch_normal_samples', 'format_version': 1,
        'source_type': 'patch_surface_normals', 'signed': True,
        'patches_dir': str(patches_dir.resolve()), 'coordinate_scale': float(coordinate_scale),
        'z_roi': list(z_roi), 'cell_size_output_voxels': float(cell_size),
        'normal_convention': 'patch normal aligned with negative winding gradient',
        'sign_valid_min_abs_cosine': 1e-6, 'finite_difference_epsilon': 2.0, 'device': str(device),
    }
    source_metadata = {}
    if patches_dir.is_file():
        if surface_spacing is not None:
            raise ValueError('bilinear resampling requires original patches, not packed samples')
        if coordinate_scale != 1:
            raise ValueError('packed sample input is already in fitter coordinates')
        with np.load(patches_dir) as samples:
            source_metadata = json.loads(str(samples['metadata_json']))
            if (source_metadata.get('artifact_type') != 'patch_normal_samples'
                    or source_metadata.get('format_version') != 1):
                raise ValueError('unsupported packed normal store')
            if source_metadata.get('cell_size_output_voxels') != cell_size:
                raise ValueError('packed input must use the requested cell size')
            position, normal = samples['position_zyx'], samples['normal_zyx']
            if (position.ndim != 2 or position.shape[1] != 3
                    or normal.shape != position.shape):
                raise ValueError('packed normal positions/vectors must have shape (N, 3)')
            keep = ((position[:, 0] >= z_roi[0]) & (position[:, 0] < z_roi[1])
                    & np.isfinite(position).all(axis=1)
                    & np.isfinite(normal).all(axis=1) & (np.linalg.norm(normal, axis=1) > 0))
            if not keep.all():
                position, normal = position[keep], normal[keep]
        stats = {'source_samples': str(patches_dir.resolve()),
                 'patch_count': source_metadata.get('patch_count'),
                 'pooled_position_count': len(position)}
    else:
        entries = sorted(p for p in patches_dir.iterdir()
                         if p.is_dir() and (p / 'meta.json').is_file()
                         and all((p / f'{axis}.tif').is_file() for axis in 'xyz'))
        if not entries:
            raise ValueError(f'no TIFXYZ patches found in {patches_dir}')
        if box_width:
            from patch_normal_tiled import export_filtered
            return export_filtered(
                entries, output, reference, coordinate_scale=coordinate_scale, z_roi=z_roi,
                cell_size=cell_size, surface_spacing=surface_spacing, width=box_width,
                tile_edge=int(tile_edge), workers=workers, device=device, chunk_size=chunk_size,
                output_format=output_format, base_scale=base_scale,sign_grid_spacing=sign_grid_spacing,
                resume=resume, geometry_batch_quads=geometry_batch_quads,
                metadata={**base_metadata, 'sampling': 'bilinear_surface_signed_box_filter',
                          'cuda_memory_limit_gib': cuda_memory_gb,
                          'surface_spacing_output_voxels': float(surface_spacing),
                          'fine_cell_size_output_voxels': 1., 'box_kernel_zyx': [box_width] * 3,
                          'box_normalization': 'occupied signed fine voxels only; normalize mean vector',
                          'overlap_policy': 'nearest fine-cell center; stable source order on ties',
                          'output_selection': 'fine representative nearest coarse-cell center',
                          'filter_across_patches_and_windings': True,
                          'reorient_after_filter': True, 'position_dtype': 'float64'})
        position, normal, stats = pool_patch_normals(
            entries, coordinate_scale, z_roi, cell_size, workers, surface_spacing)
    if not len(position):
        raise ValueError('no valid triangle centroids in the requested ROI')
    model = reference['model']
    with torch.no_grad():
        signed = orient_normals(
            position, normal, model.get_slice_to_spiral_transform(),
            model.get_dr_per_winding(), device=device, chunk_size=chunk_size, progress=True)
    if packed_input:
        signed['position_zyx'] = position  # Preserve original precision and cell membership.
    metadata = {
        **source_metadata, **base_metadata, **stats,
        'patches_dir': source_metadata.get('patches_dir', str(patches_dir.resolve())),
        'coordinate_scale': source_metadata.get('coordinate_scale', float(coordinate_scale)),
        'z_roi': list(z_roi),
        'cell_size_output_voxels': float(cell_size),
        'sampling': source_metadata.get('sampling', 'bilinear_surface_nearest_cell_center'
                                       if surface_spacing is not None else 'triangle_centroids_nearest_cell_center'),
        'surface_spacing_output_voxels': source_metadata.get('surface_spacing_output_voxels', surface_spacing),
        'sample_count': len(position),
        'normal_convention': 'patch normal aligned with negative winding gradient',
        'sign_valid_min_abs_cosine': 1e-6,
        'finite_difference_epsilon': 2.0,
        'device': str(device),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f'.{output.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **signed, metadata_json=np.asarray(json.dumps(metadata)))
        # Publish only a complete archive, and never replace a racing writer.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'wrote {len(position):,} normals ({int(signed["sign_valid"].sum()):,} '
          f'valid signs) to {output}', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('patches_dir', type=Path,
                        help='patch directory, or pooled normal NPZ to reorient')
    parser.add_argument('output', type=Path,
                        help='signed normal .npz file, Zarr directory, or dataset/patch-normals directory')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='reference fit used to choose inward signs')
    parser.add_argument('--umbilicus', type=Path, help='umbilicus used by the reference fit')
    parser.add_argument('--coordinate-scale', type=float, default=1.0,
                        help='multiply TIFF coordinates by this (base /0 to /2: 0.25)')
    parser.add_argument('--z-roi', type=float, nargs=2, metavar=('BEGIN', 'END'),
                        help='half-open fitter range; defaults to the checkpoint range')
    parser.add_argument('--cell-size', type=float,
                        help='output cell edge in fitter voxels (default 4; packed input inherits its cell size)')
    parser.add_argument('--surface-spacing', type=float,
                        help='bilinear sample spacing (default 1 with box filtering)')
    parser.add_argument('--box-width', type=int, choices=(0, 3, 5),
                        help='box kernel on the one-voxel lattice (default 3; 0 disables smoothing). '
                             'With 0, omitting --surface-spacing selects legacy triangle centroids')
    parser.add_argument('--tile-edge', type=int,
                        help='processing tile edge in fitter voxels (CUDA/auto default 256, CPU 64); '
                             'must divide into whole output cells')
    parser.add_argument('--sign-grid-spacing', type=float, default=0.,
                        help='optional approximate CUDA checkpoint-direction grid in fitter voxels; '
                             '0 evaluates every point exactly. Ambiguous cells use exact evaluation')
    parser.add_argument('--resume', action='store_true',
                        help='validate and reuse complete chunks in an interrupted CUDA Zarr export')
    parser.add_argument('--cuda-memory-gb', type=float, default=8.,
                        help='PyTorch CUDA allocation limit in GiB (default 8; excludes driver/CPU memory)')
    parser.add_argument('--geometry-batch-quads', type=int, default=250_000,
                        help='maximum quads per CUDA grouping batch; geometry is staged on disk')
    parser.add_argument('--output-format', choices=('npz', 'zarr', 'compact'), default='npz',
                        help='packed NPZ, sparse OME-Zarr, or direct fitter-ready compact normals')
    parser.add_argument('--base-scale', type=float, default=4.,
                        help='base voxels per fitter voxel for direct Zarr metadata (default 4)')
    parser.add_argument('--device', default='auto', help='auto (prefer CUDA), cpu or cuda')
    parser.add_argument('--chunk-size', type=int, default=65536,
                        help='positions per reference-gradient batch')
    parser.add_argument('--workers', type=int, default=1, help='parallel patch readers')
    parser.add_argument('--torch-threads', type=int, default=4,
                        help='CPU threads for reference-gradient evaluation')
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error('--torch-threads must be positive')
    import torch
    torch.set_num_threads(args.torch_threads)
    extract_patch_normals(args.patches_dir, args.output, checkpoint=args.checkpoint,
                          coordinate_scale=args.coordinate_scale, z_roi=args.z_roi,
                          cell_size=args.cell_size, umbilicus=args.umbilicus,
                          device=args.device, chunk_size=args.chunk_size, workers=args.workers,
                          surface_spacing=args.surface_spacing, box_width=args.box_width,
                          tile_edge=args.tile_edge, output_format=args.output_format,
                          base_scale=args.base_scale,sign_grid_spacing=args.sign_grid_spacing,
                          resume=args.resume, cuda_memory_gb=args.cuda_memory_gb,
                          geometry_batch_quads=args.geometry_batch_quads)


if __name__ == '__main__':
    main()
