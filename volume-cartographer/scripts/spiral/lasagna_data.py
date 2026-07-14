import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import zarr


def _read_zarr_zslab_chunked(zarr_array, z_lo, z_hi, max_workers=32):
    """Fast read of a local v2 zarr z slab, with safe fallback to zarr indexing."""
    store = getattr(zarr_array, 'store', None)
    root = getattr(store, 'root', None)
    try:
        compressors = zarr_array.compressors
        codec = compressors[0] if compressors else None
    except Exception:
        codec = None

    chunks = tuple(zarr_array.chunks)
    shape = tuple(zarr_array.shape)
    chunk_dir = None if root is None else os.path.join(str(root), zarr_array.path)
    fallback = (
        root is None or codec is None or len(chunks) != 3
        or zarr_array.dtype != np.uint8 or chunk_dir is None
        or not os.path.isdir(chunk_dir)
    )
    if fallback:
        return np.ascontiguousarray(zarr_array[z_lo:z_hi], dtype=np.uint8)

    z_size, y_size, x_size = shape
    chunk_z, chunk_y, chunk_x = chunks
    out = np.zeros((z_hi - z_lo, y_size, x_size), dtype=np.uint8)
    z_chunk_lo = z_lo // chunk_z
    z_chunk_hi = (z_hi - 1) // chunk_z

    coords = []
    for z_chunk in range(z_chunk_lo, z_chunk_hi + 1):
        z_dir = os.path.join(chunk_dir, str(z_chunk))
        if not os.path.isdir(z_dir):
            continue
        for y_name in os.listdir(z_dir):
            y_dir = os.path.join(z_dir, y_name)
            if not os.path.isdir(y_dir):
                continue
            for x_name in os.listdir(y_dir):
                coords.append((z_chunk, int(y_name), int(x_name)))

    def load_and_place(coord):
        z_chunk, y_chunk, x_chunk = coord
        chunk_path = os.path.join(chunk_dir, str(z_chunk), str(y_chunk), str(x_chunk))
        try:
            with open(chunk_path, 'rb') as fp:
                raw = fp.read()
        except FileNotFoundError:
            return
        buf = np.frombuffer(codec.decode(raw), dtype=np.uint8).reshape(chunk_z, chunk_y, chunk_x)
        z0 = z_chunk * chunk_z
        out_z0 = max(z0, z_lo) - z_lo
        out_z1 = min(z0 + chunk_z, z_hi) - z_lo
        buf_z0 = out_z0 + z_lo - z0
        y0 = y_chunk * chunk_y
        x0 = x_chunk * chunk_x
        y1 = min(y0 + chunk_y, y_size)
        x1 = min(x0 + chunk_x, x_size)
        out[out_z0:out_z1, y0:y1, x0:x1] = buf[
            buf_z0:buf_z0 + (out_z1 - out_z0), :y1 - y0, :x1 - x0
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(load_and_place, coords))
    return out


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
    storage_backend='dense_cuda',
    cache_directory=None,
):
    # Densely load the precomputed nx/ny normal-component and grad_mag
    # (windings-per-base-voxel) zarrs over the z-ROI into a compact uint8 volume.
    if not use_normals and not use_spacing:
        return None

    if use_normals and (not normal_nx_zarr_path or not normal_ny_zarr_path):
        raise RuntimeError('dense normal loss is enabled, but one of the nx/ny zarr paths is not set')
    if use_spacing and not grad_mag_zarr_path:
        raise RuntimeError('dense spacing loss is enabled, but grad_mag zarr path is not set')

    print(f'loading lasagna zarrs group {normal_zarr_group}')
    nx_array = ny_array = grad_mag_array = None
    reference_shape = None
    if use_normals:
        nx_root = zarr.open(normal_nx_zarr_path, mode='r')
        ny_root = zarr.open(normal_ny_zarr_path, mode='r')
        nx_array = nx_root[normal_zarr_group]
        ny_array = ny_root[normal_zarr_group]
        if nx_array.shape != ny_array.shape:
            raise ValueError(f'nx/ny normal zarr shapes differ: {nx_array.shape} vs {ny_array.shape}')
        reference_shape = nx_array.shape
    if use_spacing:
        grad_mag_root = zarr.open(grad_mag_zarr_path, mode='r')
        grad_mag_array = grad_mag_root[normal_zarr_group]
        if reference_shape is None:
            reference_shape = grad_mag_array.shape
        elif grad_mag_array.shape != reference_shape:
            raise ValueError(f'grad_mag zarr shape {grad_mag_array.shape} differs from dense normal shape {reference_shape}')

    if scroll_zarr is not None:
        expected_shape = tuple(np.ceil(np.array(scroll_zarr.shape, dtype=np.float64) / lasagna_scale).astype(np.int64))
        if tuple(reference_shape) != expected_shape:
            print(
                f'WARNING: lasagna zarr shape {reference_shape} does not match '
                f'ceil(scroll_zarr.shape / lasagna_scale) {expected_shape}'
            )

    z_size = int(reference_shape[0])
    z_lo = max(0, int(np.floor(z_begin / lasagna_scale)))
    z_hi = min(z_size, int(np.ceil(z_end / lasagna_scale)))
    if z_hi <= z_lo:
        raise RuntimeError(f'lasagna z-ROI [{z_lo}, {z_hi}) is empty (zarr z size {z_size})')

    roi_shape = (z_hi - z_lo, reference_shape[1], reference_shape[2])
    if storage_backend in ('auto', 'mmap'):
        from lasagna_mmap import prepare_lasagna_mmap
        store = prepare_lasagna_mmap(
            nx_array=nx_array, ny_array=ny_array, grad_mag_array=grad_mag_array,
            source_paths={'normal_x': normal_nx_zarr_path, 'normal_y': normal_ny_zarr_path,
                          'gradient_magnitude': grad_mag_zarr_path},
            group=normal_zarr_group, z_lo=z_lo, z_hi=z_hi,
            lasagna_scale=lasagna_scale, cache_directory=cache_directory,
        )
        print(f'lasagna: using mmap cache {store.directory} with {store.worker_count} gather workers')
        return {'backend': 'mmap', 'store': store, 'z_origin': z_lo,
                'lasagna_scale': lasagna_scale, 'shape': roi_shape}

    print(f'loading lasagna for z in [{z_lo}, {z_hi}) (shape {roi_shape[0]}, {roi_shape[1]}, {roi_shape[2]})')
    with ThreadPoolExecutor(max_workers=3) as executor:
        nx_future = executor.submit(_read_zarr_zslab_chunked, nx_array, z_lo, z_hi) if use_normals else None
        ny_future = executor.submit(_read_zarr_zslab_chunked, ny_array, z_lo, z_hi) if use_normals else None
        grad_mag_future = executor.submit(_read_zarr_zslab_chunked, grad_mag_array, z_lo, z_hi) if use_spacing else None
        nx_u8 = nx_future.result() if nx_future is not None else np.zeros(roi_shape, dtype=np.uint8)
        ny_u8 = ny_future.result() if ny_future is not None else np.zeros(roi_shape, dtype=np.uint8)
        grad_mag_u8 = grad_mag_future.result() if grad_mag_future is not None else np.zeros(roi_shape, dtype=np.uint8)
    volume = np.stack([nx_u8, ny_u8, grad_mag_u8], axis=0)  # 3 (nx, ny, grad_mag), z, y, x  uint8
    print(f'lasagna: loaded {volume.nbytes / 1e9:.2f} GB volume {volume.shape}')
    volume = torch.from_numpy(volume).to(device='cuda')
    return {
        'backend': 'dense_cuda',
        'volume': volume,
        'z_origin': z_lo,
        'lasagna_scale': lasagna_scale,
        'shape': tuple(volume.shape[1:]),  # z, y, x
    }
