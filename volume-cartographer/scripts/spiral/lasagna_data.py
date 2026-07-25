import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import zarr
from tqdm import tqdm


class _TqdmSlabProgress:
    """Adapter from the mmap builders' (name, channel, done, total) callback to
    one tqdm bar per streamed source (auto-disabled on non-TTY output)."""

    def __init__(self, label):
        self._label = label
        self._bar = None
        self._key = None
        self._done = 0

    def __call__(self, name, channel, done, total):
        key = (name, channel)
        if key != self._key:
            self.close()
            self._key = key
            self._done = 0
            suffix = f'[{channel}]' if name == 'normals' else ''
            self._bar = tqdm(desc=f'{self._label} {name}{suffix}', total=total,
                             unit='slice', disable=None)
        if self._bar is not None:
            self._bar.update(done - self._done)
        self._done = done

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


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


def _dense_gpu_store_allowed(nbytes, *, forced=False):
    """Whether a prepared uint8 store should be promoted to a dense GPU tensor.

    The mmap gather path costs ~0.12 s per million trilinear samples on the
    GB10 and dominates the phase-bundle step; a resident uint8 tensor makes
    the same lookups (bit-identical values) essentially free. Gate:
    ``FIT_SPIRAL_DENSE_GPU_STORES`` = ``auto`` (default: promote when the
    slab fits within a fraction of currently free GPU memory), ``0`` (never),
    ``1`` (always), or a float fraction overriding auto's 0.30.
    """
    mode = os.environ.get('FIT_SPIRAL_DENSE_GPU_STORES', 'auto').strip().lower()
    if mode in ('0', 'false', 'no'):
        return False
    if not torch.cuda.is_available():
        return False
    if forced or mode in ('1', 'true', 'yes', 'force'):
        return True
    fraction = 0.30
    try:
        fraction = float(mode)
    except ValueError:
        pass
    free, _total = torch.cuda.mem_get_info()
    return nbytes <= fraction * free


def _upload_slabs_to_gpu(destination, reader, slab=64):
    """Fill a dense CUDA uint8 tensor slab-by-slab from a memmap-backed
    reader (z-slab callable), never materialising the full array in RAM."""
    z_size = destination.shape[0]
    for z_lo in range(0, z_size, slab):
        z_hi = min(z_lo + slab, z_size)
        # copy: a contiguous memmap slice arrives read-only, which
        # torch.from_numpy warns about (we never write, but stay clean).
        destination[z_lo:z_hi].copy_(
            torch.from_numpy(np.array(reader(z_lo, z_hi), dtype=np.uint8)))
    return destination


def _paged_chunk_mask(interior_fn, grid_shape, scale_zyx, z_origin, chunk):
    """Bool [nzc, nyc, nxc] of store-grid chunks intersecting the interior.

    ``interior_fn(points_working_zyx) -> bool`` decides membership in FULL-RES
    working coordinates (e.g. inside the outer-shell envelope + margin). Each
    chunk is tested at its center with the chunk's half-diagonal folded into
    the caller's margin, so a partially-inside chunk is always kept.
    """
    nzc = -(-grid_shape[0] // chunk)
    nyc = -(-grid_shape[1] // chunk)
    nxc = -(-grid_shape[2] // chunk)
    zc, yc, xc = np.meshgrid(
        np.arange(nzc), np.arange(nyc), np.arange(nxc), indexing='ij')
    centers_grid = np.stack([zc, yc, xc], axis=-1).reshape(-1, 3) * chunk + chunk / 2.0
    centers_grid[:, 0] += z_origin
    centers_working = centers_grid * np.asarray(scale_zyx, dtype=np.float64)
    keep = interior_fn(centers_working.astype(np.float32))
    return np.asarray(keep, dtype=bool).reshape(nzc, nyc, nxc)


def _build_paged_store(reader, grid_shape, keep_mask, chunk, channels=1):
    """Pack the kept chunks of a z-slab reader into a paged GPU store.

    reader(ch, z_lo, z_hi) -> np.uint8 [z, Y, X] for channel ch. Returns
    (chunk_grid int32 CUDA [nzc, nyc, nxc] slot-or--1, pool uint8 CUDA
    [channels, n_kept, chunk**3]). Boundary chunks are zero-padded; 0 is the
    stores' no-data code, so padding reads as invalid.
    """
    nzc, nyc, nxc = keep_mask.shape
    slots = np.full(keep_mask.shape, -1, dtype=np.int32)
    slots[keep_mask] = np.arange(int(keep_mask.sum()), dtype=np.int32)
    n_kept = int(keep_mask.sum())
    pool = torch.zeros((channels, n_kept, chunk ** 3), dtype=torch.uint8,
                       device='cuda')
    pad_y = nyc * chunk - grid_shape[1]
    pad_x = nxc * chunk - grid_shape[2]
    for zc in range(nzc):
        row_mask = keep_mask[zc]
        if not row_mask.any():
            continue
        z_lo = zc * chunk
        z_hi = min(z_lo + chunk, grid_shape[0])
        row_slots = slots[zc][row_mask]
        keep_flat = np.flatnonzero(row_mask.reshape(-1))
        for ch in range(channels):
            slab = np.asarray(reader(ch, z_lo, z_hi), dtype=np.uint8)
            if slab.shape[0] < chunk or pad_y or pad_x:
                slab = np.pad(slab, ((0, chunk - slab.shape[0]),
                                     (0, pad_y), (0, pad_x)))
            # [chunk, nyc, chunk, nxc, chunk] -> [nyc*nxc, chunk^3]
            tiles = slab.reshape(chunk, nyc, chunk, nxc, chunk)
            tiles = tiles.transpose(1, 3, 0, 2, 4).reshape(nyc * nxc, chunk ** 3)
            pool[ch, torch.from_numpy(row_slots.astype(np.int64)).cuda()] = \
                torch.from_numpy(np.ascontiguousarray(tiles[keep_flat])).cuda()
    chunk_grid = torch.from_numpy(slots).to(device='cuda')
    return chunk_grid, pool


def gather_paged_u8(volume_dict, zi, yi, xi, channel=0):
    """Gather uint8 values from a paged store at store-grid indices (any
    shape). Indices must already be clamped in-bounds; reads landing in a
    dropped (outside-interior) chunk return 0 = the no-data code."""
    chunk = volume_dict['paged_chunk']
    grid = volume_dict['chunk_grid']
    pool = volume_dict['pool']
    cz, cy, cx = zi // chunk, yi // chunk, xi // chunk
    slot = grid[cz, cy, cx].long()
    local = (((zi - cz * chunk) * chunk + (yi - cy * chunk)) * chunk
             + (xi - cx * chunk))
    values = pool[channel, slot.clamp(min=0), local]
    return torch.where(slot >= 0, values, torch.zeros_like(values))


def _grid_yx_window(yx_bounds_working, y_scale, x_scale, y_size, x_size):
    """Convert a full-res working-coordinate (y0, y1, x0, x1) crop window to
    store-grid index bounds, clamped to the plane. None -> the full plane.

    The crop applies only to dense-GPU promotion: the store's disk cache stays
    full-plane, and out-of-window reads under the cropped tensor fall out of
    bounds -> invalid, exactly like reads past the canvas edge today."""
    if yx_bounds_working is None:
        return 0, int(y_size), 0, int(x_size)
    y0, y1, x0, x1 = yx_bounds_working
    y_lo = max(0, int(np.floor(float(y0) / y_scale)))
    y_hi = min(int(y_size), int(np.ceil(float(y1) / y_scale)))
    x_lo = max(0, int(np.floor(float(x0) / x_scale)))
    x_hi = min(int(x_size), int(np.ceil(float(x1) / x_scale)))
    if y_hi <= y_lo or x_hi <= x_lo:
        raise RuntimeError(
            f'yx crop window {yx_bounds_working} is empty on the '
            f'{y_size}x{x_size} store plane')
    return y_lo, y_hi, x_lo, x_hi


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
    yx_bounds_working=None,
    interior_fn=None,
    paged_chunk=64,
):
    # Densely load the precomputed nx/ny normal-component and grad_mag
    # (windings-per-base-voxel) zarrs over the z-ROI into a compact uint8 volume.
    if not use_normals and not use_spacing:
        return None

    if use_normals and (not normal_nx_zarr_path or not normal_ny_zarr_path):
        raise RuntimeError('normal sampling is enabled, but one of the nx/ny zarr paths is not set')
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
        if nx_array.dtype != np.dtype('uint8') or ny_array.dtype != np.dtype('uint8'):
            raise ValueError(
                f'nx/ny normal zarrs must use the production uint8 encoding; '
                f'got {nx_array.dtype} and {ny_array.dtype}')
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
        progress = _TqdmSlabProgress('lasagna mmap cache:')
        try:
            store = prepare_lasagna_mmap(
                nx_array=nx_array, ny_array=ny_array, grad_mag_array=grad_mag_array,
                source_paths={'normal_x': normal_nx_zarr_path, 'normal_y': normal_ny_zarr_path,
                              'gradient_magnitude': grad_mag_zarr_path},
                group=normal_zarr_group, z_lo=z_lo, z_hi=z_hi,
                lasagna_scale=lasagna_scale, cache_directory=cache_directory,
                progress=progress,
            )
        finally:
            progress.close()
        if interior_fn is not None and storage_backend == 'auto':
            keep = _paged_chunk_mask(
                interior_fn, roi_shape, (lasagna_scale,) * 3, z_lo, paged_chunk)
            pool_bytes = 3 * int(keep.sum()) * paged_chunk ** 3
            if _dense_gpu_store_allowed(pool_bytes):
                readers = (
                    lambda a, b: store.normals[a:b, :, :, 0],
                    lambda a, b: store.normals[a:b, :, :, 1],
                    lambda a, b: store.grad_mag[a:b],
                )
                chunk_grid, pool = _build_paged_store(
                    lambda ch, a, b: readers[ch](a, b),
                    roi_shape, keep, paged_chunk, channels=3)
                store._drop_resident_pages()
                store.close()
                print(f'lasagna: paged GPU store from cache {store.directory} '
                      f'({pool_bytes / 1e9:.1f} GB, {int(keep.sum())}/{keep.size} '
                      f'chunks of {paged_chunk}^3 kept)')
                return {'backend': 'dense_cuda_paged', 'chunk_grid': chunk_grid,
                        'pool': pool, 'paged_chunk': paged_chunk,
                        'z_origin': z_lo, 'lasagna_scale': lasagna_scale,
                        'shape': roi_shape}
        y_lo, y_hi, x_lo, x_hi = _grid_yx_window(
            yx_bounds_working, lasagna_scale, lasagna_scale,
            roi_shape[1], roi_shape[2])
        dense_shape = (roi_shape[0], y_hi - y_lo, x_hi - x_lo)
        if storage_backend == 'auto' and _dense_gpu_store_allowed(
                3 * int(np.prod(dense_shape, dtype=np.int64))):
            # Promote the cached slab to a resident GPU tensor: identical
            # uint8 values, no per-step sparse gather. The disk cache stays
            # for the next mmap-served consumer. With a yx crop only the
            # sampled window is uploaded (out-of-window -> invalid).
            volume = torch.empty((3, *dense_shape), dtype=torch.uint8,
                                 device='cuda')
            _upload_slabs_to_gpu(
                volume[0], lambda a, b: store.normals[a:b, y_lo:y_hi, x_lo:x_hi, 0])
            _upload_slabs_to_gpu(
                volume[1], lambda a, b: store.normals[a:b, y_lo:y_hi, x_lo:x_hi, 1])
            _upload_slabs_to_gpu(
                volume[2], lambda a, b: store.grad_mag[a:b, y_lo:y_hi, x_lo:x_hi])
            store._drop_resident_pages()
            store.close()
            print(f'lasagna: dense GPU store from cache {store.directory} '
                  f'({volume.numel() / 1e9:.1f} GB, yx window '
                  f'[{y_lo},{y_hi})x[{x_lo},{x_hi}))')
            return {'backend': 'dense_cuda', 'volume': volume,
                    'z_origin': z_lo, 'y_origin': y_lo, 'x_origin': x_lo,
                    'lasagna_scale': lasagna_scale,
                    'shape': dense_shape}
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
    storage_backend='auto',
    workers=None,
    yx_bounds_working=None,
    interior_fn=None,
    paged_chunk=64,
):
    """Resolve and validate a surf-SDT store as a mmap-served Lasagna input.

    Geometry and encoding are read from the store's own metadata - never from
    ``normal_zarr_group``/``lasagna_scale``. The scale convention is working
    voxels per stored grid voxel (group 1 of the standard build = 2.0), so
    sampling maps ``working_zyx / scale`` into the store grid.
    """
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

    from lasagna_mmap import prepare_scalar_mmap
    progress = _TqdmSlabProgress('surf_sdt mmap cache:')
    try:
        store = prepare_scalar_mmap(
            array=array, source_path=sdt_zarr_path, group=group_name,
            z_lo=z_lo, z_hi=z_hi, coordinate_scale=list(scale_zyx),
            cache_directory=cache_directory, kind='surf_sdt',
            extra_fingerprint={'encoding': [unit, offset, cap]},
            workers=workers, progress=progress,
        )
    finally:
        progress.close()
    shape = (z_hi - z_lo, int(array.shape[1]), int(array.shape[2]))
    common = {
        'kind': volume_kind,
        'z_origin': z_lo,
        'scale_zyx': tuple(scale_zyx),
        'unit': unit,
        'offset': offset,
        'cap': cap,
        'shape': shape,
        'fingerprint': fingerprint,
    }
    if interior_fn is not None and storage_backend == 'auto':
        keep = _paged_chunk_mask(interior_fn, shape, scale_zyx, z_lo, paged_chunk)
        pool_bytes = int(keep.sum()) * paged_chunk ** 3
        if _dense_gpu_store_allowed(pool_bytes):
            chunk_grid, pool = _build_paged_store(
                lambda ch, a, b: store.volume[a:b],
                shape, keep, paged_chunk, channels=1)
            store._drop_resident_pages()
            store.close()
            print(f'surf_sdt: paged GPU store from cache {store.directory} '
                  f'({pool_bytes / 1e9:.1f} GB, {int(keep.sum())}/{keep.size} '
                  f'chunks of {paged_chunk}^3 kept)')
            return {'backend': 'dense_cuda_paged', 'chunk_grid': chunk_grid,
                    'pool': pool, 'paged_chunk': paged_chunk, **common}
    y_lo, y_hi, x_lo, x_hi = _grid_yx_window(
        yx_bounds_working, scale_zyx[1], scale_zyx[2], shape[1], shape[2])
    dense_shape = (shape[0], y_hi - y_lo, x_hi - x_lo)
    nbytes = int(np.prod(dense_shape, dtype=np.int64))
    if storage_backend in ('auto', 'dense_cuda') and _dense_gpu_store_allowed(
            nbytes, forced=storage_backend == 'dense_cuda'):
        volume = torch.empty(dense_shape, dtype=torch.uint8, device='cuda')
        _upload_slabs_to_gpu(
            volume, lambda a, b: store.volume[a:b, y_lo:y_hi, x_lo:x_hi])
        store._drop_resident_pages()
        store.close()
        print(f'surf_sdt: dense GPU store from cache {store.directory} '
              f'({nbytes / 1e9:.1f} GB, yx window [{y_lo},{y_hi})x[{x_lo},{x_hi}))')
        return {'backend': 'dense_cuda', 'volume': volume, **common,
                'y_origin': y_lo, 'x_origin': x_lo, 'shape': dense_shape}
    print(f'surf_sdt: using mmap cache {store.directory} '
          f'({np.prod(store.shape) / 1e9:.1f} GB, {store.worker_count} gather workers)')
    return {'backend': 'mmap', 'store': store, **common}


def _pyramid_level(root_attrs, group_name):
    multiscales = root_attrs.get('multiscales')
    datasets = multiscales[0].get('datasets', []) if multiscales else []
    for level, dataset in enumerate(datasets):
        if str(dataset.get('path')) == str(group_name):
            return level
    return 0
