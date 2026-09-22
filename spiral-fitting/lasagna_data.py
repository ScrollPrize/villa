import json
import os

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on Windows
    fcntl = None

import numpy as np
import torch

from pack_resident_pools import pack_arrays, sidecar_path


def _ensure_sidecar(array_dirs, sidecar, *, label, stage_name, progress=None):
    """Build one missing resident-pool sidecar, safely across fit processes."""
    meta_path = os.path.join(sidecar, 'meta.json')
    if os.path.exists(meta_path):
        return sidecar

    if progress is not None:
        progress.begin(
            'building', stage_name, step=0, total_steps=0, unit='chunks',
            detail=sidecar)
    print(f'{label}: sparse resident pool not found; building {sidecar}',
          flush=True)

    # Multiple fits can start against the same dataset at once. The metadata
    # is written last by pack_arrays(), so a waiter can distinguish a complete
    # pool from a partial one after it acquires the lock.
    lock_path = sidecar + '.lock'
    with open(lock_path, 'a+b') as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows fallback
            import msvcrt
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b'\0')
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            if os.path.exists(meta_path):
                if progress is not None:
                    progress.finish(detail='built by another process')
                return sidecar

            pack_arrays(
                array_dirs,
                sidecar,
                label=label,
                progress_callback=(
                    (lambda current, total, detail: progress.update(
                        current, total_steps=total, detail=detail))
                    if progress is not None else None
                ),
            )
        finally:
            if fcntl is None:  # pragma: no cover - Windows fallback
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    if progress is not None:
        progress.finish(detail=f'sparse store ready at {sidecar}')
    return sidecar


def ensure_fit_sparse_stores(
    *,
    use_normals,
    use_spacing,
    normal_nx_zarr_path,
    normal_ny_zarr_path,
    grad_mag_zarr_path,
    normal_zarr_group,
    progress=None,
):
    """Build the resident pools required by one fit when they are absent."""
    normal_group = str(normal_zarr_group)
    if use_normals:
        if not normal_nx_zarr_path or not normal_ny_zarr_path:
            raise RuntimeError(
                'normal sampling is enabled, but one of the nx/ny zarr paths '
                'is not set')
        sidecar = sidecar_path(
            normal_nx_zarr_path, normal_group, pair=True)
        _ensure_sidecar(
            [os.path.join(normal_nx_zarr_path, normal_group),
             os.path.join(normal_ny_zarr_path, normal_group)],
            sidecar,
            label='lasagna normals',
            stage_name='Building Lasagna normal sparse store',
            progress=progress,
        )

    if use_spacing:
        if not grad_mag_zarr_path:
            raise RuntimeError(
                'dense spacing loss is enabled, but grad_mag zarr path is not set')
        sidecar = sidecar_path(grad_mag_zarr_path, normal_group)
        _ensure_sidecar(
            [os.path.join(grad_mag_zarr_path, normal_group)],
            sidecar,
            label='lasagna grad_mag',
            stage_name='Building Lasagna gradient sparse store',
            progress=progress,
        )


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
    progress=None,
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
        if progress is not None:
            progress.begin(
                'loading', 'Loading normal volumes onto GPU',
                step=0, total_steps=0, unit='bricks')
        normal_cache = ResidentBrickPool(
            normal_sidecar,
            origin_zyx=(z_lo, 0, 0),
            z_roi=(z_lo, z_hi),
            device=device,
            label='lasagna normals',
            expected_channels=2,
            progress_callback=(
                (lambda current, total, detail: progress.update(
                    current, total_steps=total, detail=detail))
                if progress is not None else None
            ),
        )
    grad_cache = None
    if use_spacing:
        if progress is not None:
            progress.begin(
                'loading', 'Loading gradient volume onto GPU',
                step=0, total_steps=0, unit='bricks')
        grad_cache = ResidentBrickPool(
            grad_sidecar,
            origin_zyx=(z_lo, 0, 0),
            z_roi=(z_lo, z_hi),
            device=device,
            label='lasagna grad_mag',
            expected_channels=1,
            expected_shape_zyx=reference_shape,
            progress_callback=(
                (lambda current, total, detail: progress.update(
                    current, total_steps=total, detail=detail))
                if progress is not None else None
            ),
        )
    return {
        'backend': 'sparse_cuda',
        'store': SparseLasagnaStore(
            normal_cache=normal_cache, grad_cache=grad_cache),
        'z_origin': z_lo,
        'lasagna_scale': lasagna_scale,
        'shape': roi_shape,
    }

