import numpy as np
import torch
import zarr


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
    downsample_factor,
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
        expected_shape = tuple(np.ceil(np.array(scroll_zarr.shape, dtype=np.float64) / downsample_factor).astype(np.int64))
        if tuple(reference_shape) != expected_shape:
            print(
                f'WARNING: lasagna zarr shape {reference_shape} does not match '
                f'ceil(scroll_zarr.shape / downsample_factor) {expected_shape}'
            )

    z_size = int(reference_shape[0])
    z_lo = max(0, z_begin)
    z_hi = min(z_size, z_end)
    if z_hi <= z_lo:
        raise RuntimeError(f'lasagna z-ROI [{z_lo}, {z_hi}) is empty (zarr z size {z_size})')

    roi_shape = (z_hi - z_lo, reference_shape[1], reference_shape[2])
    print(f'loading lasagna for z in [{z_lo}, {z_hi}) (shape {roi_shape[0]}, {roi_shape[1]}, {roi_shape[2]})')
    nx_u8 = np.ascontiguousarray(nx_array[z_lo:z_hi], dtype=np.uint8) if use_normals else np.zeros(roi_shape, dtype=np.uint8)
    ny_u8 = np.ascontiguousarray(ny_array[z_lo:z_hi], dtype=np.uint8) if use_normals else np.zeros(roi_shape, dtype=np.uint8)
    grad_mag_u8 = np.ascontiguousarray(grad_mag_array[z_lo:z_hi], dtype=np.uint8) if use_spacing else np.zeros(roi_shape, dtype=np.uint8)
    volume = np.stack([nx_u8, ny_u8, grad_mag_u8], axis=0)  # 3 (nx, ny, grad_mag), z, y, x  uint8
    print(f'lasagna: loaded {volume.nbytes / 1e9:.2f} GB volume {volume.shape}')
    volume = torch.from_numpy(volume).to(device='cuda')
    return {
        'volume': volume,
        'z_origin': z_lo,
        'shape': tuple(volume.shape[1:]),  # z, y, x
    }
