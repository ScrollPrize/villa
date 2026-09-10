"""Rasterize DBM track polylines into a proximity-colored uint8 OME-Zarr.

Track databases produced by ``extract_surface_tracks.py`` contain pickled
lists of ``(N, 3)`` integer ZYX polylines.  This converter reserves value 0
for background and assigns values 1..255 so that tracks near one another do
not normally reuse a value.  The values are intended for the VC3D Glasbey
colormap, where they become categorical colors.

The input can be very large.  DBM records are streamed, CPU-heavy polyline
rasterization runs in worker processes, and distinct compressed Zarr chunks
are written concurrently by threads.  At no point is a whole input database
or output volume held in memory.
"""

import base64
import dbm
import itertools
import json
import math
import os
import pickle
import shlex
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import zarr
from numcodecs import Zstd
from tqdm import tqdm


LABEL_COUNT = 255
ALL_LABEL_BITS = sum(1 << label for label in range(1, LABEL_COUNT + 1))
LABEL_MASK_BYTES = (LABEL_COUNT + 1) // 8
LABEL_MODES = ('local', 'proximity', 'round-robin')
LABEL_MODE_DESCRIPTIONS = {
    'local': 'greedy reuse avoidance within coarse cells crossed by each track',
    'proximity': 'proximity-aware greedy coloring across neighboring coarse cells',
    'round-robin': 'global round-robin assignment without spatial reuse checks',
}


def _parse_zyx(text, option_name, cast=int):
    parts = [cast(p.strip()) for p in text.split(',')]
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3 or any(v <= 0 for v in parts):
        raise click.UsageError(f'{option_name} must be one positive value or Z,Y,X')
    return tuple(parts)


def _normalize_dbm_path(path):
    """Return the base path accepted by dbm.whichdb/open.

    ndbm commonly creates ``name.db`` while callers naturally pass either
    ``name`` or ``name.db``.  SQLite-backed DBMs are single files and keep
    their supplied suffix.
    """
    path = os.path.abspath(os.fspath(path))
    if dbm.whichdb(path):
        return path
    if path.endswith('.db') and dbm.whichdb(path[:-3]):
        return path[:-3]
    raise click.UsageError(f'{path} is not a recognized DBM database')


class TrackDatabase:
    """Read a DBM without making the SQLite backend create WAL files."""

    def __init__(self, path):
        self.path = _normalize_dbm_path(path)
        self.kind = dbm.whichdb(self.path)
        self._keys = None

    def __len__(self):
        if self.kind == 'dbm.sqlite3':
            with closing(self._sqlite()) as connection:
                return int(connection.execute('SELECT COUNT(*) FROM Dict').fetchone()[0])
        return len(self._dbm_keys())

    def records(self):
        if self.kind == 'dbm.sqlite3':
            with closing(self._sqlite()) as connection:
                cursor = connection.execute('SELECT key, value FROM Dict ORDER BY key')
                for key, value in cursor:
                    yield bytes(key), bytes(value)
            return

        with dbm.open(self.path, 'r') as database:
            for key in self._dbm_keys():
                yield bytes(key), bytes(database[key])

    def _dbm_keys(self):
        # Some ndbm implementations discover keys by scanning a large hash
        # file. Cache that small key list because tqdm sizing, an automatic
        # extent pass, and conversion may all ask for it.
        if self._keys is None:
            with dbm.open(self.path, 'r') as database:
                self._keys = sorted(database.keys())
        return self._keys

    def _sqlite(self):
        # immutable=1 is important for read-only inputs and prevents large
        # source DBMs from acquiring sidecar journal/WAL files.
        uri = f'file:{Path(self.path).as_posix()}?mode=ro&immutable=1'
        return sqlite3.connect(uri, uri=True)


def _decode_tracks(payload, source, key):
    entries = pickle.loads(payload)
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f'{source} key {key!r}: value is not a list of tracks')
    tracks = []
    for index, entry in enumerate(entries):
        track = np.asarray(entry)
        if track.ndim != 2 or track.shape[1:] != (3,):
            raise ValueError(
                f'{source} key {key!r} track {index}: expected shape (N, 3), got {track.shape}'
            )
        if len(track):
            if not np.issubdtype(track.dtype, np.integer):
                if not np.all(np.isfinite(track)) or not np.all(track == np.rint(track)):
                    raise ValueError(
                        f'{source} key {key!r} track {index}: coordinates are not integers'
                    )
            track = track.astype(np.int64, copy=False)
            tracks.append(track)
    return tracks


def _record_id(source_index, key):
    encoded = base64.urlsafe_b64encode(key).decode('ascii')
    return f'{source_index}:{encoded}'


def _coarse_track_cells(track, cell_size):
    """Grid cells crossed by a polyline, including long inter-vertex spans."""
    cells = set(map(tuple, track // cell_size))
    if len(track) <= 1:
        return cells

    deltas = np.diff(track, axis=0)
    # Normal skeleton vertices are much closer than one cell, in which case
    # their endpoint cells are sufficient. Only interpolate genuinely long
    # jumps; avoiding a tiny allocation per ordinary segment matters for the
    # multi-gigabyte databases this script targets.
    long_segments = np.flatnonzero(np.abs(deltas).max(axis=1) > cell_size)
    for index in long_segments:
        start = track[index]
        delta = deltas[index]
        subdivisions = int(math.ceil(np.abs(delta).max() / cell_size))
        fractions = np.arange(subdivisions + 1, dtype=np.float64) / subdivisions
        samples = start + np.rint(fractions[:, None] * delta).astype(np.int64)
        cells.update(map(tuple, samples // cell_size))
    return cells


def _cell_key_geometry(shape, cell_size):
    """Return collision-free integer-key strides for coarse cells plus a halo."""
    coarse_shape = tuple((int(size) + cell_size - 1) // cell_size for size in shape)
    halo_y = coarse_shape[1] + 2
    halo_x = coarse_shape[2] + 2
    return coarse_shape, (halo_y * halo_x, halo_x)


def _coarse_track_keys(track, cell_size, key_strides):
    """Integer keys for coarse cells crossed by a polyline."""
    z_stride, y_stride = key_strides

    def encode(points):
        divided = points // cell_size + 1
        return (divided[:, 0] * z_stride
                + divided[:, 1] * y_stride
                + divided[:, 2]).astype(np.int64, copy=False)

    key_parts = [encode(track)]
    if len(track) > 1:
        deltas = np.diff(track, axis=0)
        long_segments = np.flatnonzero(np.abs(deltas).max(axis=1) > cell_size)
        for index in long_segments:
            start = track[index]
            delta = deltas[index]
            subdivisions = int(math.ceil(np.abs(delta).max() / cell_size))
            fractions = np.arange(subdivisions + 1, dtype=np.float64) / subdivisions
            samples = start + np.rint(fractions[:, None] * delta).astype(np.int64)
            key_parts.append(encode(samples))
    return np.unique(np.concatenate(key_parts))


class SpatialLabeler:
    """Greedy spatial graph coloring backed by a coarse occupancy hash."""

    def __init__(self, reuse_distance, shape=None):
        self.cell_size = int(reuse_distance)
        self.occupancy = {}
        self.uses = np.zeros(LABEL_COUNT + 1, dtype=np.int64)
        self.forced_reuses = 0
        self.next_label = 1
        self.key_strides = None
        self.neighbour_offsets = None
        if shape is not None:
            _, self.key_strides = _cell_key_geometry(shape, self.cell_size)
            z_stride, y_stride = self.key_strides
            self.neighbour_offsets = tuple(
                dz * z_stride + dy * y_stride + dx
                for dz, dy, dx in itertools.product((-1, 0, 1), repeat=3)
            )

    def assign(self, track):
        if self.key_strides is None:
            cells = _coarse_track_cells(track, self.cell_size)
        else:
            cells = [int(key) for key in _coarse_track_keys(
                track, self.cell_size, self.key_strides,
            )]
        neighbour_masks = []
        forbidden = 0
        neighbour_cells = set()
        if self.neighbour_offsets is None:
            for cell in cells:
                for offset in itertools.product((-1, 0, 1), repeat=3):
                    neighbour_cells.add(tuple(a + b for a, b in zip(cell, offset)))
        else:
            for cell in cells:
                for offset in self.neighbour_offsets:
                    neighbour_cells.add(cell + offset)
        for cell in neighbour_cells:
            mask = self.occupancy.get(cell, 0)
            if mask:
                neighbour_masks.append(mask)
                forbidden |= mask

        available = ALL_LABEL_BITS & ~forbidden
        if available:
            # Round-robin keeps global use balanced without examining all 255
            # counters for every one of potentially millions of tracks. In
            # the common case the first candidate is locally available.
            for offset in range(LABEL_COUNT):
                candidate = 1 + (self.next_label - 1 + offset) % LABEL_COUNT
                if available & (1 << candidate):
                    label = candidate
                    self.next_label = 1 + candidate % LABEL_COUNT
                    break
        else:
            # More than 255 mutually local tracks cannot be represented in a
            # byte. Reuse the value occurring in the fewest nearby cells.
            # Expanding all masks together keeps this dense fallback in NumPy;
            # iterating over every mask and label in Python dominates runtime
            # once a region has exhausted all 255 values.
            packed_masks = np.frombuffer(
                b''.join(mask.to_bytes(LABEL_MASK_BYTES, 'little')
                         for mask in neighbour_masks),
                dtype=np.uint8,
            ).reshape(-1, LABEL_MASK_BYTES)
            local_uses = np.unpackbits(
                packed_masks, axis=1, bitorder='little',
            ).sum(axis=0)
            least_local_uses = local_uses[1:].min()
            candidates = np.flatnonzero(local_uses[1:] == least_local_uses) + 1
            label = min((int(value) for value in candidates),
                        key=lambda value: (local_uses[value], self.uses[value], value))
            self.forced_reuses += 1

        bit = 1 << label
        for cell in cells:
            self.occupancy[cell] = self.occupancy.get(cell, 0) | bit
        self.uses[label] += 1
        return label

    def assign_many(self, tracks):
        return [self.assign(track) for track in tracks]


class LocalLabeler:
    """Avoid label reuse only in coarse cells actually crossed by a track."""

    def __init__(self, reuse_distance, shape):
        self.cell_size = int(reuse_distance)
        _, self.key_strides = _cell_key_geometry(shape, self.cell_size)
        self.occupancy = {}
        self.forced_reuses = 0
        self.next_label = 1

    def assign(self, track):
        cells = _coarse_track_keys(track, self.cell_size, self.key_strides)
        forbidden = 0
        for cell in cells:
            forbidden |= self.occupancy.get(int(cell), 0)

        available = ALL_LABEL_BITS & ~forbidden
        label = self.next_label
        if available:
            while not available & (1 << label):
                label = 1 + label % LABEL_COUNT
        else:
            # A byte cannot distinguish more than 255 tracks crossing the same
            # coarse cells. Keep the global cycle balanced when reuse is forced.
            self.forced_reuses += 1
        self.next_label = 1 + label % LABEL_COUNT

        bit = 1 << label
        for cell in cells:
            cell = int(cell)
            self.occupancy[cell] = self.occupancy.get(cell, 0) | bit
        return label

    def assign_many(self, tracks):
        return [self.assign(track) for track in tracks]


class RoundRobinLabeler:
    """Assign a balanced global label cycle without spatial bookkeeping."""

    def __init__(self):
        self.forced_reuses = 0
        self.next_label = 1

    def assign(self, track):
        del track
        label = self.next_label
        self.next_label = 1 + label % LABEL_COUNT
        return label

    def assign_many(self, tracks):
        count = len(tracks)
        labels = 1 + (
            self.next_label - 1 + np.arange(count, dtype=np.int64)
        ) % LABEL_COUNT
        self.next_label = 1 + (self.next_label - 1 + count) % LABEL_COUNT
        return labels.astype(np.uint8, copy=False)


def _make_labeler(label_mode, reuse_distance, shape):
    if label_mode == 'local':
        return LocalLabeler(reuse_distance, shape)
    if label_mode == 'proximity':
        return SpatialLabeler(reuse_distance, shape)
    if label_mode == 'round-robin':
        return RoundRobinLabeler()
    raise ValueError(f'unknown label mode {label_mode!r}')


def _rasterize_polyline(track):
    """Return integer voxels on each 3-D segment, with no gaps."""
    if len(track) <= 1:
        return track.copy()
    deltas = np.diff(track, axis=0)
    steps = np.abs(deltas).max(axis=1).astype(np.int64)
    keep = steps > 0
    if not np.any(keep):
        return track[:1].copy()
    starts = track[:-1][keep]
    deltas = deltas[keep]
    steps = steps[keep]
    total = int(steps.sum())
    segment_ids = np.repeat(np.arange(len(steps), dtype=np.int64), steps)
    segment_starts = np.repeat(np.cumsum(steps) - steps, steps)
    within = np.arange(total, dtype=np.int64) - segment_starts + 1
    points = starts[segment_ids] + np.rint(
        deltas[segment_ids] * (within / steps[segment_ids])[:, None]
    ).astype(np.int64)
    return np.concatenate((track[:1], points), axis=0)


def _rasterize_batch(items, chunk_edge):
    """Worker entry point: rasterize tracks and bucket voxels by chunk."""
    point_parts = []
    value_parts = []
    for track, label in items:
        points = _rasterize_polyline(track)
        point_parts.append(points)
        value_parts.append(np.full(len(points), label, dtype=np.uint8))
    if not point_parts:
        return []

    points = np.concatenate(point_parts, axis=0)
    values = np.concatenate(value_parts, axis=0)
    chunk_indices = points // chunk_edge
    local = points - chunk_indices * chunk_edge
    order = np.lexsort((chunk_indices[:, 2], chunk_indices[:, 1], chunk_indices[:, 0]))
    chunk_indices = chunk_indices[order]
    local = local[order]
    values = values[order]

    boundaries = np.flatnonzero(np.any(chunk_indices[1:] != chunk_indices[:-1], axis=1)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(points)]))
    updates = []
    for start, stop in zip(starts, stops):
        coords = local[start:stop]
        flat = ((coords[:, 0] * chunk_edge + coords[:, 1]) * chunk_edge
                + coords[:, 2]).astype(np.uint32)
        updates.append((tuple(int(v) for v in chunk_indices[start]), flat, values[start:stop]))
    return updates


def _make_batches(tracks, labels, target_points):
    batch = []
    points = 0
    for track, label in zip(tracks, labels):
        if batch and points + len(track) > target_points:
            yield batch
            batch = []
            points = 0
        batch.append((track, label))
        points += len(track)
    if batch:
        yield batch


def _merge_updates(results):
    chunks = {}
    for result in results:  # submission order preserves first-track-wins
        for chunk_index, flat, values in result:
            bucket = chunks.setdefault(chunk_index, ([], []))
            bucket[0].append(flat)
            bucket[1].append(values)
    merged = {}
    for chunk_index, (flat_parts, value_parts) in chunks.items():
        flat = np.concatenate(flat_parts)
        values = np.concatenate(value_parts)
        _, first = np.unique(flat, return_index=True)
        first.sort()
        merged[chunk_index] = (flat[first], values[first])
    return merged


def _write_chunk(array, shape, chunk_edge, chunk_index, flat, values, fresh=False):
    starts = tuple(index * chunk_edge for index in chunk_index)
    stops = tuple(min(start + chunk_edge, size) for start, size in zip(starts, shape))
    slices = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    if fresh:
        data = np.zeros(tuple(stop - start for start, stop in zip(starts, stops)), dtype=np.uint8)
    else:
        data = np.asarray(array[slices]).copy()
    z = flat // (chunk_edge * chunk_edge)
    rem = flat % (chunk_edge * chunk_edge)
    y = rem // chunk_edge
    x = rem % chunk_edge
    empty = data[z, y, x] == 0
    if not np.any(empty):
        return 0, False
    data[z[empty], y[empty], x[empty]] = values[empty]
    array[slices] = data
    return int(empty.sum()), True


def _infer_shape(databases):
    maximum = np.full(3, -1, dtype=np.int64)
    total = sum(len(database) for database in databases)
    with tqdm(total=total, desc='scanning track extents', unit='record') as progress:
        for database in databases:
            for key, payload in database.records():
                for track in _decode_tracks(payload, database.path, key):
                    if np.any(track < 0):
                        raise ValueError(f'{database.path} key {key!r}: negative coordinate')
                    maximum = np.maximum(maximum, track.max(axis=0))
                progress.update()
    if np.any(maximum < 0):
        raise click.UsageError('the input databases contain no non-empty tracks')
    return tuple(int(v + 1) for v in maximum)


def _reference_geometry(path, group_name):
    root = zarr.open_group(path, mode='r')
    if group_name not in root:
        raise click.UsageError(f'group {group_name!r} not found in reference {path}')
    array = root[group_name]
    if array.ndim != 3:
        raise click.UsageError(f'reference group is {array.ndim}-D, expected Z,Y,X')
    scale = None
    unit = None
    multiscales = root.attrs.get('multiscales', [])
    if multiscales:
        metadata = multiscales[0]
        axes = metadata.get('axes', [])
        units = [axis.get('unit') for axis in axes]
        if len(units) == 3 and len(set(units)) == 1:
            unit = units[0]
        for dataset in metadata.get('datasets', []):
            if dataset.get('path') == group_name:
                for transformation in dataset.get('coordinateTransformations', []):
                    if transformation.get('type') == 'scale':
                        scale = tuple(float(v) for v in transformation['scale'])
    return tuple(int(v) for v in array.shape), scale, unit


def _progress_path(output_path):
    return f'{output_path}.tracks-progress.json'


def _write_progress(path, parameters, done):
    payload = {'parameters': parameters, 'done': sorted(done)}
    temporary = f'{path}.tmp'
    with open(temporary, 'w') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def _load_progress(path, parameters):
    with open(path) as stream:
        payload = json.load(stream)
    if payload.get('parameters') != parameters:
        raise click.UsageError(f'{path} was created with different conversion parameters')
    return set(payload.get('done', []))


def _create_output(output_path, shape, chunk_edge, scale, unit, sources, parameters, resume):
    exists = os.path.exists(output_path)
    if exists and not resume:
        raise click.UsageError(f'{output_path} already exists; pass --resume to continue it')
    if resume and not exists:
        raise click.UsageError(f'--resume given but {output_path} does not exist')

    root = zarr.open_group(output_path, mode='a', zarr_format=2)
    if '0' not in root:
        root.create_array(
            '0', shape=shape, chunks=(chunk_edge,) * 3, dtype='uint8', fill_value=0,
            compressors=Zstd(level=3),
            chunk_key_encoding={'name': 'v2', 'separator': '/'},
        )
    array = root['0']
    if tuple(array.shape) != tuple(shape) or np.dtype(array.dtype) != np.dtype('uint8'):
        raise click.UsageError('existing output group 0 has a different shape or dtype')

    if not exists:
        axes = [{'name': axis, 'type': 'space'} for axis in 'zyx']
        if unit:
            for axis in axes:
                axis['unit'] = unit
        root.attrs.update({
            'kind': 'spatially_colored_tracks',
            'created': datetime.now(timezone.utc).isoformat(),
            'command_line': shlex.join(sys.argv),
            'sources': sources,
            'background_value': 0,
            'track_label_values': [1, 255],
            'label_mode': parameters['label_mode'],
            'label_assignment': LABEL_MODE_DESCRIPTIONS[parameters['label_mode']],
            'overlap_policy': 'first track wins',
            'reuse_distance_voxels': parameters['reuse_distance'],
            'compression': {'codec': 'zstd', 'level': 3},
            'complete': False,
            'multiscales': [{
                'version': '0.4',
                'name': 'tracks',
                'axes': axes,
                'datasets': [{
                    'path': '0',
                    'coordinateTransformations': [{'type': 'scale', 'scale': list(scale)}],
                }],
            }],
        })
    return root, array


@click.command(help='Rasterize pickled ZYX track DBMs into a uint8 OME-Zarr.')
@click.argument('track_dbms', nargs=-1, required=True, type=click.Path())
@click.option('--out', 'output_path', required=True, type=click.Path(),
              help='output .ome.zarr directory')
@click.option('--shape', 'shape_text', help='output Z,Y,X shape; avoids an extent prepass')
@click.option('--like', 'reference_path', type=click.Path(exists=True, file_okay=False),
              help='copy shape and geometry from a reference OME-Zarr')
@click.option('--like-group', default='0', show_default=True,
              help='array group within --like')
@click.option('--voxel-size', default=None,
              help='OME scale as one value or Z,Y,X; overrides --like geometry')
@click.option('--unit', default=None,
              help='OME spatial unit, for example micrometer; omit for pixel coordinates')
@click.option('--chunk', 'chunk_edge', default=128, type=click.IntRange(16, 1024),
              show_default=True, help='cubic output chunk edge')
@click.option('--reuse-distance', default=128, type=click.IntRange(min=1), show_default=True,
              help='coarse cell edge used for spatial label reuse, in voxels')
@click.option('--label-mode', type=click.Choice(LABEL_MODES), default='local', show_default=True,
              help='local is fast and avoids reuse in cells crossed by each track')
@click.option('--workers', default=4,
              type=click.IntRange(min=1), show_default=True,
              help='polyline rasterization worker processes')
@click.option('--write-threads', default=4, type=click.IntRange(min=1), show_default=True,
              help='concurrent Zarr chunk compression/write threads')
@click.option('--batch-points', default=100_000, type=click.IntRange(min=1), show_default=True,
              help='approximate input vertices per rasterization task')
@click.option('--records-per-flush', default=128, type=click.IntRange(min=1), show_default=True,
              help='DBM records accumulated before each chunk write pass')
@click.option('--resume', is_flag=True,
              help='resume an interrupted output, rebuilding coloring state while scanning')
def main(track_dbms, output_path, shape_text, reference_path, like_group, voxel_size, unit,
         chunk_edge, reuse_distance, label_mode, workers, write_threads, batch_points,
         records_per_flush, resume):
    if shape_text and reference_path:
        raise click.UsageError('pass only one of --shape and --like')
    databases = [TrackDatabase(path) for path in track_dbms]
    sources = [database.path for database in databases]

    reference_scale = None
    reference_unit = None
    if reference_path:
        shape, reference_scale, reference_unit = _reference_geometry(reference_path, like_group)
    elif shape_text:
        shape = _parse_zyx(shape_text, '--shape', int)
    else:
        shape = _infer_shape(databases)

    if voxel_size:
        scale = _parse_zyx(voxel_size, '--voxel-size', float)
    else:
        scale = reference_scale or (1.0, 1.0, 1.0)
    unit = unit or reference_unit

    parameters = {
        'sources': sources,
        'shape': list(shape),
        'chunk': chunk_edge,
        'reuse_distance': reuse_distance,
        'label_mode': label_mode,
        'scale': list(scale),
        'unit': unit,
        'overlap_policy': 'first_track_wins',
        'codec': {'name': 'zstd', 'level': 3},
    }
    progress_path = _progress_path(output_path)
    if resume:
        if not os.path.exists(progress_path):
            raise click.UsageError(f'--resume requires {progress_path}')
        done = _load_progress(progress_path, parameters)
    else:
        done = set()

    root, output = _create_output(
        output_path, shape, chunk_edge, scale, unit, sources, parameters, resume,
    )
    if not resume:
        # Create the checkpoint before the first expensive record so even an
        # early interruption leaves an output that --resume can recognize.
        _write_progress(progress_path, parameters, done)
    labeler = _make_labeler(label_mode, reuse_distance, shape)
    record_total = sum(len(database) for database in databases)
    tracks_seen = 0
    voxels_written = 0
    chunks_written = 0
    shape_array = np.asarray(shape)
    pending_updates = {}
    pending_record_ids = []
    inflight_flush = None
    written_chunks = set()

    process_pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        with ThreadPoolExecutor(max_workers=write_threads) as write_pool:
            def finish_flush(wait):
                nonlocal voxels_written, chunks_written, inflight_flush
                if inflight_flush is None:
                    return False
                chunk_futures = inflight_flush['chunk_futures']
                if not wait and not all(future.done() for _, future in chunk_futures):
                    return False
                for chunk_index, future in chunk_futures:
                    new_voxels, wrote = future.result()
                    voxels_written += new_voxels
                    chunks_written += int(wrote)
                    written_chunks.add(chunk_index)
                done.update(inflight_flush['record_ids'])
                _write_progress(progress_path, parameters, done)
                inflight_flush = None
                return True

            def launch_flush():
                nonlocal pending_updates, pending_record_ids, inflight_flush
                if not pending_record_ids:
                    return
                # Only one write generation may touch the output at once. The
                # next records can still be labeled and rasterized while it runs.
                finish_flush(wait=True)
                updates_to_flush = pending_updates
                record_ids = pending_record_ids
                pending_updates = {}
                pending_record_ids = []
                writes = {}
                for chunk_index, (flat_parts, value_parts) in updates_to_flush.items():
                    flat = np.concatenate(flat_parts)
                    values = np.concatenate(value_parts)
                    _, first = np.unique(flat, return_index=True)
                    first.sort()
                    writes[chunk_index] = (flat[first], values[first])
                chunk_futures = []
                for chunk_index, (flat, values) in writes.items():
                    fresh = not resume and chunk_index not in written_chunks
                    future = write_pool.submit(
                        _write_chunk, output, shape, chunk_edge,
                        chunk_index, flat, values, fresh,
                    )
                    chunk_futures.append((chunk_index, future))
                inflight_flush = {
                    'record_ids': record_ids,
                    'chunk_futures': chunk_futures,
                }

            def submit_raster_job(identifier, tracks, labels):
                should_write = identifier not in done
                futures = None
                results = None
                if should_write and tracks:
                    batches = list(_make_batches(tracks, labels, batch_points))
                    if process_pool is None:
                        results = [_rasterize_batch(batch, chunk_edge) for batch in batches]
                    else:
                        futures = [
                            process_pool.submit(_rasterize_batch, batch, chunk_edge)
                            for batch in batches
                        ]
                return identifier, should_write, futures, results

            def finish_raster_job(job, progress):
                nonlocal pending_updates, pending_record_ids
                identifier, should_write, futures, results = job
                if futures is not None:
                    results = [future.result() for future in futures]
                if results is not None:
                    updates = _merge_updates(results)
                    for chunk_index, (flat, values) in updates.items():
                        bucket = pending_updates.setdefault(chunk_index, ([], []))
                        bucket[0].append(flat)
                        bucket[1].append(values)

                if should_write:
                    pending_record_ids.append(identifier)
                if len(pending_record_ids) >= records_per_flush:
                    launch_flush()
                finish_flush(wait=False)
                progress.update()
                progress.set_postfix(
                    tracks=tracks_seen,
                    voxels=voxels_written,
                    chunks=chunks_written,
                    forced_reuse=labeler.forced_reuses,
                    refresh=False,
                )

            with tqdm(total=record_total, desc='writing tracks', unit='record') as progress:
                previous_raster_job = None
                for source_index, database in enumerate(databases):
                    for key, payload in database.records():
                        tracks = _decode_tracks(payload, database.path, key)
                        for track in tracks:
                            if np.any(track < 0) or np.any(track >= shape_array):
                                raise ValueError(
                                    f'{database.path} key {key!r}: track coordinate outside '
                                    f'output shape {shape}'
                                )
                        labels = labeler.assign_many(tracks)
                        tracks_seen += len(tracks)

                        identifier = _record_id(source_index, key)
                        current_raster_job = submit_raster_job(identifier, tracks, labels)
                        if previous_raster_job is not None:
                            finish_raster_job(previous_raster_job, progress)
                        previous_raster_job = current_raster_job

                if previous_raster_job is not None:
                    finish_raster_job(previous_raster_job, progress)
                launch_flush()
                finish_flush(wait=True)
    finally:
        if process_pool is not None:
            process_pool.shutdown(wait=True, cancel_futures=True)

    root.attrs['tracks_written'] = tracks_seen
    root.attrs['nonbackground_voxels_written_this_run'] = voxels_written
    root.attrs['forced_local_label_reuses'] = labeler.forced_reuses
    root.attrs['complete'] = True
    if os.path.exists(progress_path):
        os.remove(progress_path)
    click.echo(
        f'wrote {tracks_seen:,} tracks to {output_path} with {chunks_written:,} chunk writes; '
        f'{labeler.forced_reuses:,} forced local label reuses'
    )


if __name__ == '__main__':
    main()
