import json
import os

import numpy as np
import torch
import zarr

from pack_resident_pools import sidecar_path


def _require_sidecar(zarr_path, group, *, pair=False, label):
    """Resolve a store's resident-pool sidecar or fail with the pack command."""
    sidecar = sidecar_path(zarr_path, group, pair=pair)
    if not os.path.exists(os.path.join(sidecar, 'meta.json')):
        raise RuntimeError(
            f'{label}: resident-pool sidecar {sidecar!r} not found; build it '
            f'with pack_resident_pools.py (pointed at the lasagna_inputs '
            f'folder, with --ct for CT masking)')
    return sidecar


def _read_sidecar_meta(sidecar):
    with open(os.path.join(sidecar, 'meta.json')) as f:
        return json.load(f)


def prepare_lasagna_volume(
    scroll_zarr,
    *,
    use_normals,
    use_spacing,
    normal_nx_zarr_path,
    normal_ny_zarr_path,
    grad_mag_zarr_path,
    normal_zarr_group,
    z_begin,
    z_end,
    lasagna_scale,
    storage_backend='sparse_cuda',
    cache_directory=None,
    yx_bounds_working=None,
    interior_fn=None,
    paged_chunk=64,
):
    """Open normals/grad-magnitude as fully-resident sparse brick pools.

    The pools load from ``pack_resident_pools.py`` sidecars next to the
    source zarrs; there is no other loading path.
    """
    if not use_normals and not use_spacing:
        return None
    if storage_backend != 'sparse_cuda':
        raise ValueError(
            f"storage_backend={storage_backend!r} is no longer supported; "
            "use 'sparse_cuda'")
    if not torch.cuda.is_available():
        raise RuntimeError('sparse CUDA volume sampling requires an available CUDA device')

    if use_normals and (not normal_nx_zarr_path or not normal_ny_zarr_path):
        raise RuntimeError('normal sampling is enabled, but one of the nx/ny zarr paths is not set')
    if use_spacing and not grad_mag_zarr_path:
        raise RuntimeError('dense spacing loss is enabled, but grad_mag zarr path is not set')

    group = str(normal_zarr_group)
    print(f'loading lasagna resident pools, group {group}')
    normal_sidecar = grad_sidecar = None
    reference_shape = None
    if use_normals:
        normal_sidecar = _require_sidecar(
            normal_nx_zarr_path, group, pair=True, label='lasagna normals')
        normal_meta = _read_sidecar_meta(normal_sidecar)
        reference_shape = tuple(normal_meta['array_shape'])
        expected_pair = [
            os.path.basename(str(normal_nx_zarr_path).rstrip('/')) + '/' + group,
            os.path.basename(str(normal_ny_zarr_path).rstrip('/')) + '/' + group,
        ]
        if normal_meta.get('channel_names') != expected_pair:
            print(f'WARNING: normal sidecar channels '
                  f'{normal_meta.get("channel_names")} do not match the '
                  f'configured nx/ny stores {expected_pair}')
    if use_spacing:
        grad_sidecar = _require_sidecar(
            grad_mag_zarr_path, group, label='lasagna grad_mag')
        grad_meta = _read_sidecar_meta(grad_sidecar)
        if reference_shape is None:
            reference_shape = tuple(grad_meta['array_shape'])
        elif tuple(grad_meta['array_shape']) != reference_shape:
            raise ValueError(
                f'grad_mag sidecar shape {grad_meta["array_shape"]} differs '
                f'from dense normal shape {reference_shape}')

    if scroll_zarr is not None:
        expected_shape = tuple(np.ceil(np.array(scroll_zarr.shape, dtype=np.float64) / lasagna_scale).astype(np.int64))
        if tuple(reference_shape) != expected_shape:
            print(
                f'WARNING: lasagna store shape {reference_shape} does not match '
                f'ceil(scroll_zarr.shape / lasagna_scale) {expected_shape}'
            )

    z_size = int(reference_shape[0])
    z_lo = max(0, int(np.floor(z_begin / lasagna_scale)))
    z_hi = min(z_size, int(np.ceil(z_end / lasagna_scale)))
    if z_hi <= z_lo:
        raise RuntimeError(f'lasagna z-ROI [{z_lo}, {z_hi}) is empty (store z size {z_size})')

    roi_shape = (z_hi - z_lo, reference_shape[1], reference_shape[2])
    from sparse_cuda_cache import ResidentBrickPool, SparseLasagnaStore
    device = torch.device('cuda')
    normal_cache = None
    if use_normals:
        normal_cache = ResidentBrickPool(
            normal_sidecar,
            origin_zyx=(z_lo, 0, 0),
            z_roi=(z_lo, z_hi),
            device=device,
            label='lasagna normals',
            expected_channels=2,
        )
    grad_cache = None
    if use_spacing:
        grad_cache = ResidentBrickPool(
            grad_sidecar,
            origin_zyx=(z_lo, 0, 0),
            z_roi=(z_lo, z_hi),
            device=device,
            label='lasagna grad_mag',
            expected_channels=1,
            expected_shape_zyx=reference_shape,
        )
    return {
        'backend': 'sparse_cuda',
        'store': SparseLasagnaStore(
            normal_cache=normal_cache, grad_cache=grad_cache),
        'z_origin': z_lo,
        'lasagna_scale': lasagna_scale,
        'shape': roi_shape,
    }


def _resolve_ome_group_scale(root_attrs, group_name):
    """Per-axis scale of one OME multiscales dataset, in working voxels per
    stored grid voxel. Rejects datasets with a nonzero translation: the fitter
    assumes a shared origin with the working volume."""
    multiscales = root_attrs.get('multiscales')
    for dataset in (multiscales[0].get('datasets', []) if multiscales else []):
        if str(dataset.get('path')) != str(group_name):
            continue
        scale = None
        for transformation in dataset.get('coordinateTransformations', []):
            if transformation.get('type') == 'scale':
                scale = tuple(float(s) for s in transformation['scale'])
            elif transformation.get('type') == 'translation':
                if any(abs(float(t)) > 1e-9 for t in transformation.get('translation', ())):
                    raise RuntimeError(
                        f'OME dataset {group_name!r} carries a nonzero translation; '
                        'the fitter only supports stores sharing the working-volume origin')
        return scale
    return None


def _merged_ranges_cover(ranges, lo, hi):
    """Whether the union of [lo, hi) working-z intervals covers [lo, hi)."""
    covered_to = lo
    for range_lo, range_hi in sorted((float(a), float(b)) for a, b in ranges):
        if range_lo > covered_to:
            return False
        covered_to = max(covered_to, range_hi)
        if covered_to >= hi:
            return True
    return covered_to >= hi


def prepare_surf_sdt_volume(
    sdt_zarr_path,
    sdt_zarr_group,
    *,
    z_begin,
    z_end,
    cache_directory,
    storage_backend='sparse_cuda',
    workers=None,
    yx_bounds_working=None,
    interior_fn=None,
    paged_chunk=64,
):
    """Resolve and validate a surf-SDT store as a sparse CUDA input.

    Geometry and encoding are read from the store's own metadata - never from
    ``normal_zarr_group``/``lasagna_scale``. The scale convention is working
    voxels per stored grid voxel (group 1 of the standard build = 2.0), so
    sampling maps ``working_zyx / scale`` into the store grid.
    """
    if storage_backend != 'sparse_cuda':
        raise ValueError(
            f"storage_backend={storage_backend!r} is no longer supported; "
            "use 'sparse_cuda'")
    if not torch.cuda.is_available():
        raise RuntimeError('sparse CUDA SDT sampling requires an available CUDA device')
    root = zarr.open_group(sdt_zarr_path, mode='r')
    attrs = dict(root.attrs)
    group_name = str(sdt_zarr_group)
    if group_name not in root:
        raise RuntimeError(f'group {group_name!r} not found in {sdt_zarr_path}')
    array = root[group_name]

    scale_zyx = _resolve_ome_group_scale(attrs, group_name)
    if scale_zyx is None:
        raise RuntimeError(
            f'no OME multiscales scale for group {group_name!r} in {sdt_zarr_path}; '
            'the fitter refuses to infer the store geometry')

    if attrs.get('kind') != 'surf_sdt':
        raise RuntimeError(
            f"{sdt_zarr_path} has kind={attrs.get('kind')!r}, expected 'surf_sdt'")
    for key in ('unit_working_voxels', 'offset', 'cap_working_voxels'):
        if key not in attrs:
            raise RuntimeError(f'{sdt_zarr_path} is missing encoding attribute {key!r}')
    unit = float(attrs['unit_working_voxels'])
    offset = int(attrs['offset'])
    cap = float(attrs['cap_working_voxels'])
    declared = attrs.get('scale_vs_working')
    if declared is not None:
        declared = [declared] * 3 if np.isscalar(declared) else list(declared)
        base_scale = [s / 2 ** _pyramid_level(attrs, group_name) for s in scale_zyx]
        if any(abs(a - b) > 1e-6 for a, b in zip(base_scale, declared)):
            print(f'WARNING: {sdt_zarr_path} attrs scale_vs_working {declared} does not '
                  f'match the OME scale {scale_zyx} for group {group_name}')
    # Coverage: the store is trusted only when it is stamped complete or its
    # embedded built working-z ranges cover the requested fit range. The
    # done_tiles sidecar is deliberately not consulted - it may not travel
    # with the zarr.
    if not attrs.get('complete', False):
        ranges = attrs.get('built_z_ranges_working')
        if not ranges and 'z_range_working' in attrs:
            ranges = [attrs['z_range_working']]
        if not ranges or not _merged_ranges_cover(ranges, z_begin, z_end):
            raise RuntimeError(
                f'{sdt_zarr_path} is not stamped complete and its built working-z ranges '
                f'{ranges!r} do not cover the fit range [{z_begin}, {z_end}); rebuild or '
                'extend the store (unbuilt tiles read as no-data and would silently '
                'disable the SDT losses there)')
    volume_kind = 'sdt'

    z_size = int(array.shape[0])
    z_lo = max(0, int(np.floor(z_begin / scale_zyx[0])))
    z_hi = min(z_size, int(np.ceil(z_end / scale_zyx[0])))
    if z_hi <= z_lo:
        raise RuntimeError(f'surf_sdt z-ROI [{z_lo}, {z_hi}) is empty (z size {z_size})')

    fingerprint = {
        'path': os.path.abspath(sdt_zarr_path),
        'group': group_name,
        'kind': attrs.get('kind'),
        'source': attrs.get('source'),
        'source_group': attrs.get('source_group'),
        'threshold': attrs.get('threshold'),
        'unit_working_voxels': attrs.get('unit_working_voxels'),
        'offset': attrs.get('offset'),
        'cap_working_voxels': attrs.get('cap_working_voxels'),
        'erode_source_voxels': attrs.get('erode_source_voxels'),
        'ct_mask': (attrs.get('ct_mask') or {}).get('group'),
        'ct_zero': (attrs.get('ct_zero') or {}).get('group'),
        'scale_zyx': list(scale_zyx),
        'complete': bool(attrs.get('complete', False)),
        'z_range_working': attrs.get('z_range_working'),
        'built_z_ranges_working': attrs.get('built_z_ranges_working'),
        'created': attrs.get('created'),
        'git_commit': attrs.get('git_commit'),
    }

    shape = (z_hi - z_lo, int(array.shape[1]), int(array.shape[2]))
    from sparse_cuda_cache import ResidentBrickPool, SparseScalarStore
    sidecar = _require_sidecar(sdt_zarr_path, group_name, label='surf_sdt')
    cache = ResidentBrickPool(
        sidecar,
        origin_zyx=(z_lo, 0, 0),
        z_roi=(z_lo, z_hi),
        device=torch.device('cuda'),
        label='surf_sdt',
        expected_channels=1,
        expected_shape_zyx=tuple(int(v) for v in array.shape),
    )
    fingerprint['respool_ct_mask'] = cache.meta.get('ct_mask')
    common = {
        'kind': volume_kind,
        'backend': 'sparse_cuda',
        'store': SparseScalarStore(cache),
        'z_origin': z_lo,
        'scale_zyx': tuple(scale_zyx),
        'unit': unit,
        'offset': offset,
        'cap': cap,
        'shape': shape,
        'fingerprint': fingerprint,
    }
    return common


def _pyramid_level(root_attrs, group_name):
    multiscales = root_attrs.get('multiscales')
    datasets = multiscales[0].get('datasets', []) if multiscales else []
    for level, dataset in enumerate(datasets):
        if str(dataset.get('path')) == str(group_name):
            return level
    return 0
