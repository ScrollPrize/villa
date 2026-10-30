import json
import os

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on Windows
    fcntl = None

import numpy as np
import torch

from devices import fit_device
from pack_resident_pools import pack_arrays


def _ensure_sidecar(array_dirs, sidecar, *, identity, label, stage_name, progress=None):
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
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
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
                lasagna_identity=identity,
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


# Pack kind -> (label, the stage that builds it).
_PACKS = {
    'normals': ('lasagna normals', 'Building Lasagna normal sparse store'),
    'grad_mag': ('lasagna grad_mag', 'Building Lasagna gradient sparse store'),
}


def _fit_packs(use_normals, use_spacing):
    return [kind for kind, used in (('normals', use_normals), ('grad_mag', use_spacing))
            if used]


def ensure_fit_sparse_stores(
    *,
    source,
    cache_directory,
    use_normals,
    use_spacing,
    progress=None,
):
    """Build the resident pools one fit needs from the dataset's arrays when
    they are absent (the command-line fit; a service never builds them)."""
    for kind in _fit_packs(use_normals, use_spacing):
        label, stage_name = _PACKS[kind]
        _ensure_sidecar(
            source.arrays(kind), source.pack_directory(cache_directory, kind),
            identity=source.identity, label=label, stage_name=stage_name,
            progress=progress)


def _require_sidecar(source, cache_directory, kind):
    """The pack ``kind`` of ``source`` in the cache, or fail with the command
    that builds it."""
    label = _PACKS[kind][0]
    sidecar = source.pack_directory(cache_directory, kind)
    if not os.path.exists(os.path.join(sidecar, 'meta.json')):
        raise RuntimeError(
            f'{label}: no resident pool of Lasagna dataset {source.identity} '
            f'({source.location}) at {sidecar}; build it where the dataset\'s '
            f'arrays are with pack_resident_pools.py <its .lasagna.json> --cache '
            f'<cache> --what {kind} and copy lasagna/{source.identity}/ into '
            f'{cache_directory}')
    packed = _read_sidecar_meta(sidecar).get('lasagna_identity')
    if packed is not None and packed != source.identity:
        raise RuntimeError(
            f'{label}: {sidecar} holds Lasagna dataset {packed}, not '
            f'{source.identity}')
    return sidecar


def _read_sidecar_meta(sidecar):
    with open(os.path.join(sidecar, 'meta.json')) as f:
        return json.load(f)


def prepare_lasagna_volume(
    scroll_zarr,
    *,
    use_normals,
    use_spacing,
    source,
    cache_directory,
    z_begin,
    z_end,
    lasagna_scale,
    storage_backend='sparse_cuda',
    yx_bounds_working=None,
    interior_fn=None,
    paged_chunk=64,
    device=None,
    progress=None,
):
    """Open normals/grad-magnitude as fully-resident sparse brick pools.

    The pools load from the packs of ``source`` (a
    lasagna_source.LasagnaSource) in ``cache_directory``; there is no other
    loading path.
    """
    if not use_normals and not use_spacing:
        return None
    if storage_backend != 'sparse_cuda':
        raise ValueError(
            f"storage_backend={storage_backend!r} is no longer supported; "
            "use 'sparse_cuda'")
    device = fit_device() if device is None else torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('sparse CUDA volume sampling requires an available CUDA device')
    if source is None:
        raise RuntimeError('Lasagna sampling is enabled, but the fit names no '
                           'Lasagna dataset')
    if not cache_directory:
        raise RuntimeError('Lasagna sampling needs a cache directory holding its packs')

    print(f'loading lasagna resident pools of {source.identity}, group {source.group}')
    normal_sidecar = grad_sidecar = None
    reference_shape = None
    if use_normals:
        normal_sidecar = _require_sidecar(source, cache_directory, 'normals')
        reference_shape = tuple(_read_sidecar_meta(normal_sidecar)['array_shape'])
    if use_spacing:
        grad_sidecar = _require_sidecar(source, cache_directory, 'grad_mag')
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

