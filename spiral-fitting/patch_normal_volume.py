"""Export signed packed patch normals to sparse OME-Zarr and compare local cues."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Zstd


def bounded_map(executor, function, items, limit):
    """Keep at most ``limit`` chunk jobs in flight."""
    from collections import deque
    pending = deque()
    for item in items:
        pending.append(executor.submit(function, item))
        if len(pending) >= limit:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


class LocalArray:
    """Minimal chunk reader used by export validation."""
    def __init__(self, path):
        self.array = zarr.open_array(str(path), mode='r')
        self.shape = self.array.shape
        self.chunks = np.asarray(self.array.chunks)
        self.dtype = self.array.dtype
        self.fill = self.array.fill_value

    def read_chunk(self, key):
        slices = tuple(slice(int(k * c), int(min((k + 1) * c, s)))
                       for k, c, s in zip(key, self.chunks, self.shape))
        chunk = np.full(tuple(self.chunks), self.fill, dtype=self.dtype)
        region = self.array[slices]
        chunk[tuple(slice(0, n) for n in region.shape)] = region
        return chunk


def log(message):
    print(message, flush=True)


def cell_index(positions, cell_size, shape):
    cells = np.floor(np.asarray(positions, dtype=np.float64) / cell_size).astype(np.int64)
    inside = ((cells >= 0) & (cells < np.asarray(shape))).all(axis=1)
    keys = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]
    return cells, keys, inside


def representative_rows(positions, valid, cell_size, shape):
    """Resolve any float32 boundary collisions by nearest centroid, stable on ties."""
    cells, keys, inside = cell_index(positions, cell_size, shape)
    if not inside.all():
        raise ValueError('output shape must cover every input position')
    rows = np.flatnonzero(valid)
    rows = rows[np.argsort(keys[rows], kind='stable')]
    ordered = keys[rows]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    ends = np.r_[starts[1:], len(rows)]
    selected = rows[starts].copy()
    for group in np.flatnonzero(ends - starts > 1):
        candidates = rows[starts[group]:ends[group]]
        centers = (cells[candidates] + .5) * cell_size
        dist = np.square(positions[candidates].astype(np.float64) - centers).sum(axis=1)
        selected[group] = candidates[np.argmin(dist)]
    return cells, keys[selected], selected


def create_volume(output, shape, cell_size, base_scale, chunk_edge, info):
    output.mkdir(parents=True)
    codec = Zstd(level=3)
    roots = {}
    arrays = {}
    for name in ('nx', 'ny', 'nz', 'presence', 'inward_alignment'):
        relative = f'patch_{name}.ome.zarr'
        root = zarr.open_group(str(output / relative), mode='w', zarr_format=2)
        root.create_array('0', shape=shape, chunks=(chunk_edge,) * 3,
                          dtype='uint8' if name == 'presence' else 'float32',
                          fill_value=0, compressors=codec,
                          chunk_key_encoding={'name': 'v2', 'separator': '/'})
        root.attrs.update({'complete': False, 'signed': True,
                           'multiscales': [{'version': '0.4', 'name': name,
                            'axes': [{'name': a, 'type': 'space', 'unit': 'pixel'} for a in 'zyx'],
                            'datasets': [{'path': '0', 'coordinateTransformations': [
                                {'type': 'scale', 'scale': [cell_size * base_scale] * 3},
                                {'type': 'translation', 'translation': [.5 * cell_size * base_scale] * 3}]}]}]})
        roots[name] = root
        arrays[name] = output / relative / '0'
        info['channels'][name] = f'{relative}/0'
    (output / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    return roots, arrays, codec


def write_volume_chunk(arrays, codec, key, local, normals, alignment, chunk_edge):
    for name, directory in arrays.items():
        dense = np.zeros((chunk_edge,) * 3, dtype='uint8' if name == 'presence' else 'float32')
        values = (255 if name == 'presence' else alignment
                  if name == 'inward_alignment' else normals[:, {'nz': 0, 'ny': 1, 'nx': 2}[name]])
        dense[local] = values
        path = directory.joinpath(*(str(k) for k in key))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_bytes(codec.encode(dense))
        os.replace(temporary, path)


class TiledVolumeWriter:
    """Write nonoverlapping output tiles directly, without a packed intermediate."""

    def __init__(self, output, shape, cell_size, base_scale, chunk_edge, metadata, resume=False):
        self.output = Path(output)
        if self.output.exists() and not resume:
            raise FileExistsError(self.output)
        if base_scale <= 0 or not np.isfinite(base_scale):
            raise ValueError('base scale must be finite and positive')
        self.shape, self.cell_size, self.chunk_edge = shape, cell_size, chunk_edge
        self.count = self.valid_count = 0
        self.reused_chunks = 0
        self.written = set()
        self.lock = threading.Lock()
        self.checks = []
        if resume:
            self.info = self.check_resume(output, cell_size, base_scale, chunk_edge, metadata)
            if list(shape) != self.info['shape_zyx']:
                raise ValueError('resume input geometry gives a different output shape')
            self.codec = Zstd(level=3)
            self.arrays = {name: self.output / relative
                           for name, relative in self.info['channels'].items()}
            self.roots = {name: zarr.open_group(str(path.parent), mode='r+')
                          for name, path in self.arrays.items()}
            return
        self.info = dict(artifact_type='signed_patch_normal_volume', format_version=1,
                         complete=False, source_metadata=metadata, shape_zyx=shape,
                         cell_size_fitter_voxels=cell_size,
                         fitter_voxel_scale_base_voxels=base_scale,
                         scale_base_voxels=cell_size * base_scale,
                         translation_base_voxels=.5 * cell_size * base_scale,
                         indexing='floor(position_zyx / cell_size); OME coordinates at cell centers',
                         missing='presence=0; other components zero; no interpolation or filling',
                         normal_encoding='float32 signed Cartesian components; vector order ZYX',
                         sign_convention='negative fitted winding gradient (inward)',
                         collision_policy='nearest cell-center sample on the deduplicated fine lattice',
                         chunks_zyx=[chunk_edge] * 3, channels={})
        self.roots, self.arrays, self.codec = create_volume(
            self.output, shape, cell_size, base_scale, chunk_edge, self.info)

    @staticmethod
    def check_resume(output, cell_size, base_scale, chunk_edge, metadata):
        info = json.loads((Path(output) / 'manifest.json').read_text())
        if info.get('complete'):
            raise ValueError('output is already complete; no resume needed')
        if (info.get('artifact_type') != 'signed_patch_normal_volume'
                or info['cell_size_fitter_voxels'] != cell_size
                or info['fitter_voxel_scale_base_voxels'] != base_scale
                or info['chunks_zyx'] != [chunk_edge] * 3):
            raise ValueError('resume output layout does not match requested settings')
        previous = info['source_metadata']
        for field in ('model_state_sha256', 'umbilicus', 'patches_dir', 'coordinate_scale',
                      'z_roi', 'surface_spacing_output_voxels', 'box_kernel_zyx',
                      'sign_grid_spacing_fitter_voxels', 'sign_grid_fallback_min_corner_cosine',
                      'overlap_policy', 'output_selection'):
            if previous.get(field) != metadata.get(field):
                raise ValueError(f'resume setting differs: {field}')
        expected = {name: f'patch_{name}.ome.zarr/0'
                    for name in ('nx', 'ny', 'nz', 'presence', 'inward_alignment')}
        if info['channels'] != expected:
            raise ValueError('unexpected channel layout in resume store')
        for name, relative in expected.items():
            array = json.loads((Path(output) / relative / '.zarray').read_text())
            if (array['shape'] != info['shape_zyx'] or array['chunks'] != [chunk_edge] * 3
                    or np.dtype(array['dtype']) != np.dtype('u1' if name == 'presence' else '<f4')
                    or array.get('compressor', {}).get('id') != 'zstd'
                    or array.get('order') != 'C' or array.get('filters')):
                raise ValueError(f'incompatible resume channel metadata: {name}')
        return info

    def reuse_chunk(self, key):
        """Validate legacy chunks without receipts; never trust file existence alone."""
        paths = {name: directory.joinpath(*(str(k) for k in key))
                 for name, directory in self.arrays.items()}
        if not any(path.exists() for path in paths.values()):
            return False
        try:
            size = self.chunk_edge ** 3
            def decode(name):
                dtype = np.dtype('u1' if name == 'presence' else '<f4')
                raw = self.codec.decode(paths[name].read_bytes())
                if len(raw) != size * dtype.itemsize:
                    raise ValueError('wrong decoded chunk length')
                return np.frombuffer(raw, dtype=dtype)
            presence = decode('presence')
            if not np.isin(presence, [0, 255]).all():
                raise ValueError('invalid presence encoding')
            valid = presence == 255
            count = int(valid.sum())
            if not count:
                raise ValueError('legacy exporter does not write empty chunks')
            rows = np.flatnonzero(valid)
            rows = rows[np.linspace(0, len(rows)-1, min(4, len(rows)), dtype=int)]
            norms = np.zeros(count, dtype=np.float64)
            samples = {}
            for name in ('nz', 'ny', 'nx', 'inward_alignment'):
                values = decode(name)
                if not np.isfinite(values).all() or np.any(values[~valid] != 0):
                    raise ValueError('nonfinite data or nonzero missing cells')
                selected = values[valid]
                if name == 'inward_alignment':
                    if np.any((selected <= 1e-6) | (selected > 1.00001)):
                        raise ValueError('invalid alignment')
                else:
                    norms += selected.astype(np.float64) ** 2
                samples[name] = values[rows].copy()
            if np.any(np.abs(norms-1) > 2e-6):
                raise ValueError('invalid normal length')
        except (OSError, ValueError, RuntimeError):
            # An interrupted channel set must be replaced as a whole, including
            # when recomputation yields no valid normals and writes no chunk.
            for path in paths.values():
                path.unlink(missing_ok=True)
            return False
        cells = np.column_stack(np.unravel_index(rows, [self.chunk_edge] * 3))
        p = (cells + np.asarray(key)*self.chunk_edge + .5) * self.cell_size
        n = np.column_stack([samples[name] for name in ('nz', 'ny', 'nx')])
        with self.lock:
            self.valid_count += count
            self.count += count
            self.reused_chunks += 1
            self.written.add(key)
            if len(self.checks) < 1024:
                self.checks.append((p, n, samples['inward_alignment']))
        return True

    def append(self, result):
        valid = result['sign_valid']
        with self.lock:
            self.count += len(result['position_zyx'])
            self.valid_count += int(valid.sum())
        if not valid.any():
            return
        p, n = result['position_zyx'][valid], result['normal_zyx'][valid]
        alignment = result['inward_alignment'][valid]
        cells, keys, inside = cell_index(p, self.cell_size, self.shape)
        if not inside.all() or len(np.unique(keys)) != len(keys):
            raise ValueError('tile contains duplicate or out-of-bounds output cells')
        chunks = cells // self.chunk_edge
        key = tuple(chunks[0].tolist())
        with self.lock:
            if not (chunks == chunks[0]).all() or key in self.written:
                raise ValueError('each output chunk must be written exactly once')
            self.written.add(key)
        write_volume_chunk(self.arrays, self.codec, key, tuple((cells % self.chunk_edge).T),
                           n, alignment, self.chunk_edge)
        with self.lock:
            if len(self.checks) < 1024:
                rows = np.linspace(0, len(p) - 1, min(4, len(p)), dtype=int)
                self.checks.append((p[rows], n[rows], alignment[rows]))

    def finish(self, output, metadata):
        if not self.valid_count:
            raise ValueError('no valid signed normals in the requested ROI')
        p, n, a = (np.concatenate(parts) for parts in zip(*self.checks))
        self.info.update(source_metadata=metadata,
                         input_rows=None if self.reused_chunks else self.count,
                         invalid_sign_rows=None if self.reused_chunks else self.count - self.valid_count,
                         resumed_chunks=self.reused_chunks,
                         row_count_note='legacy chunks do not record invalid sample counts'
                            if self.reused_chunks else None,
                         occupied_cells=self.valid_count, collision_rows=0,
                         spatial_chunks=len(self.written),
                         validation=validate_volume(self.output, p, n, a, self.cell_size))
        for root in self.roots.values():
            root.attrs['complete'] = True
        self.info['complete'] = True
        (self.output / 'manifest.json').write_text(json.dumps(self.info, indent=2) + '\n')


def export_volume(output, positions, normals, valid, alignment, metadata, *,
                  cell_size=8., base_scale=4., chunk_edge=64, workers=8):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if cell_size <= 0 or base_scale <= 0 or chunk_edge < 1 or workers < 1:
        raise ValueError('scales, chunk edge and worker count must be positive')
    if len(positions) == 0 or not valid.any():
        raise ValueError('no valid signed normals')
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise ValueError('positions and normals must be finite')
    shape = tuple((np.floor(positions.max(axis=0) / cell_size).astype(int) + 1).tolist())
    log('Indexing pooled cells')
    cells, keys, rows = representative_rows(positions, valid, cell_size, shape)
    info = dict(artifact_type='signed_patch_normal_volume', format_version=1,
                complete=False, source_metadata=metadata, shape_zyx=shape,
                cell_size_fitter_voxels=cell_size, fitter_voxel_scale_base_voxels=base_scale,
                scale_base_voxels=cell_size * base_scale,
                translation_base_voxels=.5 * cell_size * base_scale,
                indexing='floor(position_zyx / cell_size); OME coordinates at cell centers',
                missing='presence=0; other components zero; no interpolation or filling',
                normal_encoding='float32 signed Cartesian components; vector order ZYX',
                sign_convention='negative fitted winding gradient (inward)',
                input_rows=len(positions), invalid_sign_rows=int((~valid).sum()),
                occupied_cells=len(rows), collision_rows=int(valid.sum()) - len(rows),
                collision_policy='nearest centroid to cell center; stable input order on ties',
                chunks_zyx=[chunk_edge] * 3, channels={})
    roots, arrays, codec = create_volume(output, shape, cell_size, base_scale, chunk_edge, info)
    chunk_shape = (np.asarray(shape) + chunk_edge - 1) // chunk_edge
    ck = cells[rows] // chunk_edge
    chunk_keys = (ck[:, 0] * chunk_shape[1] + ck[:, 1]) * chunk_shape[2] + ck[:, 2]
    order = np.argsort(chunk_keys, kind='stable')
    chunk_keys = chunk_keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(chunk_keys)) + 1]
    ends = np.r_[starts[1:], len(order)]

    def write_chunk(group):
        start, end = starts[group], ends[group]
        subset = rows[order[start:end]]
        key = tuple((cells[subset[0]] // chunk_edge).tolist())
        local = tuple((cells[subset] % chunk_edge).T)
        write_volume_chunk(arrays, codec, key, local, normals[subset], alignment[subset], chunk_edge)

    start = last = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, _ in enumerate(bounded_map(pool, write_chunk, range(len(starts)), 2 * workers), 1):
            if time.monotonic() - last > 10:
                log(f'Zarr chunks {count:,}/{len(starts):,}')
                last = time.monotonic()
    info.update(spatial_chunks=len(starts), write_seconds=time.monotonic() - start)
    check_rows = rows[np.linspace(0, len(rows) - 1, min(4096, len(rows)), dtype=int)]
    info['validation'] = validate_volume(output, positions[check_rows], normals[check_rows],
                                         alignment[check_rows], cell_size, workers)
    for root in roots.values():
        root.attrs['complete'] = True
    info['complete'] = True
    (output / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    log(f'Exported {len(rows):,} occupied cells; {info["collision_rows"]:,} boundary collisions')
    return info, keys, rows


def iter_npz_rows(path, name, batch_rows=2_000_000):
    """Stream numeric row-major NPY members without materializing their full array."""
    if batch_rows < 1:
        raise ValueError('batch_rows must be positive')
    with zipfile.ZipFile(path) as archive, archive.open(f'{name}.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError(f'unsupported NPY header {version}')
        shape, fortran, dtype = reader(stream)
        if fortran or dtype.hasobject or not shape:
            raise ValueError('requires row-major numeric arrays')
        row_bytes = dtype.itemsize * int(np.prod(shape[1:]))
        last = time.monotonic()
        for start in range(0, shape[0], batch_rows):
            end = min(start + batch_rows, shape[0])
            raw = stream.read((end - start) * row_bytes)
            block = np.frombuffer(raw, dtype=dtype).reshape((end - start, *shape[1:]))
            yield start, block
            if time.monotonic() - last > 15:
                log(f'Reading fiber {name}: {end:,}/{shape[0]:,}')
                last = time.monotonic()


def read_npz_rows(path, names, indices, batch_rows=2_000_000):
    """Read selected rows without allocating the full (potentially billion-row) arrays."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) and (indices[0] < 0 or np.any(np.diff(indices) < 0)):
        raise ValueError('indices must be sorted and nonnegative')
    result = {}
    for name in names:
        pieces = []
        end = 0
        for start, block in iter_npz_rows(path, name, batch_rows):
            end = start + len(block)
            lo, hi = np.searchsorted(indices, [start, end])
            # Copies retain only selected rows, not the backing streamed block.
            pieces.append(block[indices[lo:hi] - start])
        if len(indices) and indices[-1] >= end:
            raise IndexError('sample row outside array')
        if not pieces:
            # Empty arrays are small and retain their exact trailing dimensions.
            with np.load(path) as archive:
                result[name] = archive[name][indices]
        else:
            result[name] = np.concatenate(pieces)
        log(f'Read sampled fiber {name}')
    return result


def sample_local_array(path, indices, workers=8):
    array = LocalArray(path)
    indices = np.asarray(indices, dtype=np.int64)
    result = np.full(len(indices), array.fill, dtype=array.dtype)
    good = np.flatnonzero(((indices >= 0) & (indices < array.shape)).all(axis=1))
    if not len(good):
        return result
    chunks = indices[good] // array.chunks
    shape = (np.asarray(array.shape) + array.chunks - 1) // array.chunks
    key = (chunks[:, 0] * shape[1] + chunks[:, 1]) * shape[2] + chunks[:, 2]
    order = np.argsort(key, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(key[order])) + 1]
    ends = np.r_[starts[1:], len(order)]

    def sample(group):
        rows = good[order[starts[group]:ends[group]]]
        chunk = array.read_chunk(tuple(indices[rows[0]] // array.chunks))
        result[rows] = chunk[tuple((indices[rows] % array.chunks).T)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in bounded_map(pool, sample, range(len(starts)), 2 * workers):
            pass
    return result


def decode_axes(nx, ny):
    # Use the fitter's exact hemisphere decoding, including disk/clamp behavior.
    import torch
    from losses import _decode_uint8_normal
    torch.set_num_threads(4)
    vectors, valid = _decode_uint8_normal(torch.from_numpy(nx), torch.from_numpy(ny))
    return vectors.numpy(), valid.numpy().astype(bool)


def validate_volume(output, positions, normals, alignment, cell_size, workers=8):
    indices = np.floor(positions.astype(np.float64) / cell_size).astype(np.int64)
    channels = {}
    for name in ('nx', 'ny', 'nz', 'presence', 'inward_alignment'):
        channels[name] = sample_local_array(Path(output) / f'patch_{name}.ome.zarr' / '0', indices, workers)
    recovered = np.stack([channels[k] for k in ('nz', 'ny', 'nx')], axis=1)
    np.testing.assert_array_equal(recovered, normals)
    np.testing.assert_array_equal(channels['presence'], 255)
    np.testing.assert_array_equal(channels['inward_alignment'], alignment)
    return {'readback_samples': len(positions), 'signed_components_bit_exact': True,
            'presence_and_alignment_exact': True}


def angular_error(a, b, *, tangent=False):
    cosine = np.clip(np.abs(np.einsum('ij,ij->i', a, b)), 0., 1.)
    return np.degrees(np.arcsin(cosine) if tangent else np.arccos(cosine))


def statistics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0}
    return dict(count=len(values), mean_deg=float(values.mean()),
                median_deg=float(np.median(values)), p90_deg=float(np.percentile(values, 90)),
                p95_deg=float(np.percentile(values, 95)),
                fraction_within_10_deg=float(np.mean(values <= 10)),
                fraction_within_20_deg=float(np.mean(values <= 20)))


def lasagna_axes(directory, positions, base_scale, workers):
    channels = []
    scale = None
    for name in ('nx', 'ny'):
        root = Path(directory) / f'las_008_{name}.ome.zarr'
        attrs = json.loads((root / '.zattrs').read_text())
        ms = attrs['multiscales'][0]
        if [a['name'] for a in ms['axes']] != ['z', 'y', 'x']:
            raise ValueError('Lasagna axes must be ZYX')
        transforms = ms.get('coordinateTransformations', [])
        if transforms:
            raise ValueError('group-level Lasagna transforms are unsupported')
        level = min(ms['datasets'], key=lambda d: np.prod(next(
            t['scale'] for t in d['coordinateTransformations'] if t['type'] == 'scale')))
        ts = level['coordinateTransformations']
        if len(ts) != 1 or ts[0]['type'] != 'scale':
            raise ValueError('Lasagna requires a single zero-origin scale transform')
        current_scale = np.asarray(ts[0]['scale']) / base_scale
        if scale is not None and not np.array_equal(current_scale, scale):
            raise ValueError('Lasagna component scales disagree')
        scale = current_scale
        # Match losses.get_dense_* nearest-neighbor sampling (round to even).
        indices = np.rint(positions / scale).astype(np.int64)
        channels.append(sample_local_array(root / level['path'], indices, workers))
    vectors, valid = decode_axes(*channels)
    return vectors, valid, scale


def compare_cues(output, positions, normals, alignment, cell_keys, rows, shape, *,
                 cell_size, base_scale, fibers, lasagna, sample_count=100_000,
                 fiber_candidates=2_000_000, seed=20260917, workers=8):
    output = Path(output)
    output.mkdir()
    rng = np.random.default_rng(seed)
    sample_rows = rng.choice(rows, min(sample_count, len(rows)), replace=False)
    log(f'Sampling Lasagna at {len(sample_rows):,} uniformly selected patch cells')
    las, las_valid, las_scale = lasagna_axes(lasagna, positions[sample_rows], base_scale, workers)
    patch_las = angular_error(normals[sample_rows], las)
    patch_las[~las_valid] = np.nan
    np.savez_compressed(output / 'patch_lasagna_samples.npz', patch_row=sample_rows,
                        position_zyx=positions[sample_rows], normal_zyx=normals[sample_rows],
                        lasagna_normal_zyx=las, lasagna_valid=las_valid,
                        patch_lasagna_deg=patch_las, inward_alignment=alignment[sample_rows])
    with np.load(fibers) as data:
        fiber_meta = json.loads(data['metadata_json'].item())
    total = fiber_meta['sample_count']
    fiber_rows = np.sort(rng.choice(total, min(fiber_candidates, total), replace=False))
    candidate = read_npz_rows(fibers, ('position_zyx', 'nx', 'ny', 'presence'), fiber_rows)
    p = candidate['position_zyx'] * (float(fiber_meta['output_scale_base_voxels']) / base_scale)
    _, fk, inside = cell_index(p, cell_size, shape)
    where = np.searchsorted(cell_keys, fk)
    where = np.minimum(where, len(cell_keys) - 1)
    matched = inside & (cell_keys[where] == fk)
    pi = rows[where]
    distance = np.linalg.norm(p - positions[pi], axis=1)
    fvec, fvalid = decode_axes(candidate['nx'], candidate['ny'])
    matched &= fvalid & (candidate['presence'] > 0) & (distance <= cell_size)
    matched_indices = np.flatnonzero(matched)
    # One sampled fiber per patch cell; nearest sampled fiber, then input row.
    order = np.lexsort((fiber_rows[matched_indices], distance[matched_indices], pi[matched_indices]))
    chosen = matched_indices[order]
    chosen = chosen[np.r_[True, np.diff(pi[chosen]) != 0]] if len(chosen) else chosen
    eligible = len(chosen)
    if eligible > sample_count:
        chosen = np.sort(rng.choice(chosen, sample_count, replace=False))
    if not len(chosen):
        raise ValueError('no spatially matched patch/fiber samples')
    pr = pi[chosen]
    pp = positions[pr]
    pn = normals[pr]
    fv = fvec[chosen]
    log(f'Comparing {len(chosen):,} distinct patch/fiber sites from {eligible:,} eligible cells')
    ln, lv, _ = lasagna_axes(lasagna, pp, base_scale, workers)
    # At fiber positions too: separate normal-source error from positional separation.
    fln, flv, _ = lasagna_axes(lasagna, p[chosen], base_scale, workers)
    metrics = {'patch_lasagna_deg': angular_error(pn, ln),
               'patch_fiber_plane_deg': angular_error(pn, fv, tangent=True),
               'lasagna_fiber_plane_deg': angular_error(ln, fv, tangent=True),
               'lasagna_at_fiber_plane_deg': angular_error(fln, fv, tangent=True)}
    metrics['patch_lasagna_deg'][~lv] = np.nan
    metrics['lasagna_fiber_plane_deg'][~lv] = np.nan
    metrics['lasagna_at_fiber_plane_deg'][~flv] = np.nan
    fields = dict(patch_row=pr, fiber_row=fiber_rows[chosen], position_zyx=pp,
                  normal_zyx=pn, fiber_position_zyx=p[chosen], fiber_direction_zyx=fv,
                  fiber_presence=candidate['presence'][chosen], lasagna_normal_zyx=ln,
                  lasagna_valid=lv, lasagna_at_fiber_normal_zyx=fln,
                  lasagna_at_fiber_valid=flv, separation_voxels=distance[chosen],
                  inward_alignment=alignment[pr], **metrics)
    np.savez_compressed(output / 'matched_samples.npz', **fields)
    with (output / 'matched_spots.csv').open('w') as stream:
        writer = csv.writer(stream)
        header = ['patch_row', 'fiber_row', 'z', 'y', 'x', 'fiber_z', 'fiber_y', 'fiber_x',
                  'normal_z', 'normal_y', 'normal_x', 'fiber_dz', 'fiber_dy', 'fiber_dx',
                  'lasagna_nz', 'lasagna_ny', 'lasagna_nx', 'separation_voxels',
                  'inward_alignment', 'fiber_presence', 'lasagna_valid', *metrics]
        writer.writerow(header)
        for i in range(len(chosen)):
            writer.writerow([int(pr[i]), int(fiber_rows[chosen[i]]), *pp[i], *p[chosen[i]],
                             *pn[i], *fv[i], *ln[i], distance[chosen[i]], alignment[pr[i]],
                             int(candidate['presence'][chosen[i]]), bool(lv[i]),
                             *(v[i] for v in metrics.values())])
    report = dict(seed=seed, packed_fibers=str(Path(fibers).resolve()),
                  lasagna=str(Path(lasagna).resolve()), fitter_voxel_scale_base_voxels=base_scale,
                  lasagna_scale_fitter_voxels=las_scale.tolist(),
                  uniform_patch_sample_count=len(sample_rows),
                  uniform_patch_lasagna=statistics(patch_las),
                  fiber_candidate_count=len(fiber_rows), fiber_sample_count_total=total,
                  matched_candidate_count=int(matched.sum()), eligible_distinct_patch_cells=eligible,
                  matched_site_count=len(chosen), lasagna_valid_at_patch=int(lv.sum()),
                  sampling='uniform patch cells; separate uniform fiber-row sample matched within same patch cell',
                  matching='nearest sampled fiber in same cell, at most cell_size away; not global nearest fiber',
                  reference_sign='Lasagna and fibers are unsigned axes; comparison cannot validate inward sign',
                  angle_conventions='normal agreement acos(abs(dot)); fiber angle from tangent plane asin(abs(dot)); smaller is better',
                  all_matched={k: statistics(v) for k, v in metrics.items()},
                  same_sites_with_lasagna={k: statistics(v[lv]) for k, v in metrics.items()},
                  by_max_separation={str(d): {k: statistics(v[distance[chosen] <= d])
                                             for k, v in metrics.items()} for d in (2, 4, 8)},
                  by_z=[],
                  separation_percentiles_voxels=np.percentile(distance[chosen], [0, 50, 90, 100]).tolist())
    z_edges = np.linspace(float(positions[rows, 0].min()),
                          np.nextafter(float(positions[rows, 0].max()), np.inf), 15)
    for lo, hi in zip(z_edges[:-1], z_edges[1:]):
        in_bin = (pp[:, 0] >= lo) & (pp[:, 0] < hi)
        report['by_z'].append(dict(z_begin=float(lo), z_end=float(hi),
                                   **{k: statistics(v[in_bin]) for k, v in metrics.items()}))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    make_plots(output, pp, metrics, patch_las, z_edges)
    plot_spot_directions(output, fields)
    return report


def make_plots(output, positions, metrics, patch_las, z_edges):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    labels = {'patch_lasagna_deg': 'Patch vs Lasagna normals',
              'patch_fiber_plane_deg': 'Fiber vs patch tangent plane',
              'lasagna_fiber_plane_deg': 'Fiber vs Lasagna tangent plane'}
    for key, label in labels.items():
        values = metrics[key]
        values = np.sort(values[np.isfinite(values)])
        if len(values):
            axes[0, 0].plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
        medians = [np.nanmedian(metrics[key][(positions[:, 0] >= lo) & (positions[:, 0] < hi)])
                   for lo, hi in zip(z_edges[:-1], z_edges[1:])]
        axes[0, 1].plot((z_edges[:-1] + z_edges[1:]) / 2, medians, '.-', label=label)
    axes[0, 0].set(xlabel='Angular error (degrees; smaller is better)', ylabel='Cumulative fraction', xlim=(0, 90))
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel='Z (fitting voxels)', ylabel='Median error (degrees)')
    axes[1, 0].hist(patch_las[np.isfinite(patch_las)], bins=np.arange(0, 92, 2), color='steelblue')
    axes[1, 0].set(xlabel='Patch–Lasagna angle (degrees)', ylabel='Uniformly sampled patch cells')
    selection = np.linspace(0, len(positions) - 1, min(15000, len(positions)), dtype=int)
    scatter = axes[1, 1].scatter(positions[selection, 2], positions[selection, 0], s=2,
                                c=metrics['patch_fiber_plane_deg'][selection], vmin=0, vmax=45,
                                cmap='viridis', rasterized=True)
    axes[1, 1].set(xlabel='X (fitting voxels)', ylabel='Z (fitting voxels)', title='Matched locations (XZ projection)')
    fig.colorbar(scatter, ax=axes[1, 1], label='Fiber angle from patch plane (degrees)')
    fig.suptitle('Signed patch normals: unsigned reference comparisons')
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)


def plot_spot_directions(output, fields):
    """Show median and upper-tail examples across Z, with fixed world axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    p = fields['position_zyx']
    valid = fields['lasagna_valid']
    error = fields['patch_lasagna_deg']
    z_edges = np.linspace(p[:, 0].min(), p[:, 0].max() + .01, 5)
    fig = plt.figure(figsize=(14, 18))
    fig.subplots_adjust(left=.03, right=.97, top=.91, bottom=.06, hspace=.42, wspace=.15)
    selected = []
    for band, (lo, hi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        rows = np.flatnonzero(valid & (p[:, 0] >= lo) & (p[:, 0] < hi))
        if not len(rows):
            continue
        ordered = rows[np.argsort(error[rows], kind='stable')]
        for column, quantile in enumerate((.1, .5, .9)):
            row = ordered[round(quantile * (len(ordered) - 1))]
            selected.append({'matched_row': int(row), 'z_band': band,
                             'normal_error_quantile': quantile,
                             'position_zyx': p[row].tolist()})
            ax = fig.add_subplot(4, 3, band * 3 + column + 1, projection='3d')
            normal = fields['normal_zyx'][row][::-1]
            lasagna = fields['lasagna_normal_zyx'][row][::-1]
            if np.dot(normal, lasagna) < 0:
                lasagna = -lasagna
            fiber = fields['fiber_direction_zyx'][row][::-1]
            for vector, color in ((normal, 'tab:red'), (lasagna, 'tab:blue')):
                ax.quiver(0, 0, 0, *vector, color=color, linewidth=2, arrow_length_ratio=.18)
            ax.plot([-fiber[0], fiber[0]], [-fiber[1], fiber[1]], [-fiber[2], fiber[2]],
                    color='tab:green', linewidth=2)
            ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                   xlabel='X', ylabel='Y', zlabel='Z', xticks=[-1, 0, 1],
                   yticks=[-1, 0, 1], zticks=[-1, 0, 1])
            ax.set_box_aspect((1, 1, 1))
            ax.tick_params(labelsize=7, pad=0)
            z, y, x = p[row]
            ax.set_title(f'ZYX ({z:.0f}, {y:.0f}, {x:.0f}) · p{quantile*100:.0f}\n'
                         f'Normal {error[row]:.1f}° · fiber/plane '
                         f'{fields["patch_fiber_plane_deg"][row]:.1f}°', fontsize=9)
    fig.legend(handles=[Line2D([0], [0], color='tab:red', label='Signed patch normal'),
                        Line2D([0], [0], color='tab:blue', label='Lasagna normal (sign aligned for display)'),
                        Line2D([0], [0], color='tab:green', label='Fiber axis')],
               loc='lower center', ncol=3, bbox_to_anchor=(.5, .01))
    fig.suptitle('12 matched spots: 10th / 50th / 90th normal-error percentiles in four Z bands\n'
                 'Vectors shown at a common origin; coordinates in fitting voxels')
    fig.savefig(Path(output) / 'spot_directions.png', dpi=150)
    (Path(output) / 'displayed_spots.json').write_text(json.dumps(selected, indent=2) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--base-scale', type=float, required=True, help='base /0 voxels per fitting voxel')
    parser.add_argument('--chunk-edge', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--fibers', type=Path)
    parser.add_argument('--lasagna', type=Path)
    parser.add_argument('--sample-count', type=int, default=100_000)
    parser.add_argument('--fiber-candidates', type=int, default=2_000_000)
    parser.add_argument('--seed', type=int, default=20260917)
    args = parser.parse_args()
    if bool(args.fibers) != bool(args.lasagna):
        parser.error('--fibers and --lasagna must be supplied together')
    if args.sample_count < 1 or args.fiber_candidates < 1:
        parser.error('sample counts must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    log('Loading packed patch normals')
    with np.load(args.samples) as data:
        metadata = json.loads(data['metadata_json'].item())
        if metadata.get('artifact_type') != 'patch_normal_samples' or not metadata.get('signed'):
            raise ValueError('requires signed patch_normal_samples')
        p, n = data['position_zyx'], data['normal_zyx']
        valid, alignment = data['sign_valid'], data['inward_alignment']
    metadata['packed_store'] = str(args.samples.resolve())
    cell_size = float(metadata['cell_size_output_voxels'])
    info, keys, rows = export_volume(args.output, p, n, valid, alignment, metadata,
                                    cell_size=cell_size, base_scale=args.base_scale,
                                    chunk_edge=args.chunk_edge, workers=args.workers)
    if args.fibers:
        report = compare_cues(args.output / 'comparison', p, n, alignment, keys, rows,
                              info['shape_zyx'], cell_size=cell_size, base_scale=args.base_scale,
                              fibers=args.fibers, lasagna=args.lasagna,
                              sample_count=args.sample_count, fiber_candidates=args.fiber_candidates,
                              seed=args.seed, workers=args.workers)
        log(json.dumps(report['all_matched'], indent=2))


if __name__ == '__main__':
    main()
