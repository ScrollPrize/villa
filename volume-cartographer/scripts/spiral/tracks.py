import colorsys
import dbm
import importlib
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
import shutil
import struct
from pathlib import Path
import tempfile

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from tqdm import tqdm

import prefetch
from dt_targets import snap_dt_target, strip_dt_target_in_sample_frame
from loss_maps import diagnostics_enabled, record_loss_samples
import geom_utils
from sample_spiral import (
    get_theta_and_radii,
    get_theta_crossing_step_adjustments,
    radius_from_unwrapped_shifted,
    unwrap_shifted_radii,
)


TRACK_CROSSING_CACHE_VERSION = 1
TRACK_CROSSING_CACHE_SUFFIX = '.crossings.npz'
TRACK_STORE_SUFFIX = '.vctracks'
TRACK_STORE_VERSION = 1
TRACK_STORE_MAGIC = b'VCTRK01\0'


def _load_native_track_crossings():
    """Load the optional VC native crossing kernel without making Spiral depend on it."""
    if os.environ.get('VC_DISABLE_NATIVE_TRACK_CROSSINGS') == '1':
        return None
    try:
        return importlib.import_module('vc.track_crossings')
    except ImportError:
        # Editable VC installs may point ``vc`` at site-packages even while a
        # newer extension exists in this checkout's build tree. Make that
        # developer build discoverable so repo scripts use the matching native
        # kernel immediately after ``ninja -C build vc_track_crossings``.
        try:
            import vc
            build_package = Path(__file__).resolve().parents[2] / 'build/python/vc'
            if build_package.is_dir() and str(build_package) not in vc.__path__:
                vc.__path__.append(str(build_package))
            return importlib.import_module('vc.track_crossings')
        except ImportError:
            return None


def _load_native_track_store():
    """Load the mmap-backed packed-track extension from install or build tree."""
    try:
        return importlib.import_module('vc.track_store')
    except ImportError:
        try:
            import vc
            build_package = Path(__file__).resolve().parents[2] / 'build/python/vc'
            if build_package.is_dir() and str(build_package) not in vc.__path__:
                vc.__path__.append(str(build_package))
            return importlib.import_module('vc.track_store')
        except ImportError:
            return None


class PackedTrackCollection:
    """A ragged track collection backed by a few contiguous native arrays."""

    def __init__(
            self, coordinates, offsets, source_ids, family_codes,
            arclengths, tortuosities, *, store_path=None, rows=None):
        self.coordinates = np.asarray(coordinates, dtype=np.float32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.source_ids = np.asarray(source_ids, dtype=np.uint64)
        self.family_codes = np.asarray(family_codes, dtype=np.int8)
        self.arclengths = np.asarray(arclengths, dtype=np.float64)
        self.tortuosities = np.asarray(tortuosities, dtype=np.float64)
        self.store_path = None if store_path is None else Path(store_path)
        self.rows = (np.arange(len(self.source_ids), dtype=np.int64)
                     if rows is None else np.asarray(rows, dtype=np.int64))
        if self.offsets.shape != (len(self.source_ids) + 1,):
            raise ValueError('packed track offsets are not parallel to metadata')
        if (self.family_codes.shape != self.source_ids.shape
                or self.arclengths.shape != self.source_ids.shape
                or self.tortuosities.shape != self.source_ids.shape):
            raise ValueError('packed track metadata arrays are not parallel')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.subset(np.arange(len(self), dtype=np.int64)[index])
        row = int(self.rows[index])
        return self.coordinates[self.offsets[row]:self.offsets[row + 1]]

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return PackedTrackCollection(
            self.coordinates, self.offsets, self.source_ids,
            self.family_codes, self.arclengths, self.tortuosities,
            store_path=self.store_path, rows=self.rows[indices])

    @property
    def selected_source_ids(self):
        return self.source_ids[self.rows]

    @property
    def selected_family_codes(self):
        return self.family_codes[self.rows]

    @property
    def selected_arclengths(self):
        return self.arclengths[self.rows]

    @property
    def selected_tortuosities(self):
        return self.tortuosities[self.rows]

    @property
    def selected_lengths(self):
        return self.offsets[self.rows + 1] - self.offsets[self.rows]

    def materialize(self, workers=None):
        identity = (len(self.rows) == len(self.source_ids)
                    and np.array_equal(
                        self.rows, np.arange(len(self.rows), dtype=np.int64)))
        if identity:
            return self.coordinates, self.offsets
        native = _load_native_track_store()
        if native is not None and hasattr(native, 'compact'):
            result = native.compact(
                self.coordinates, self.offsets, self.rows,
                workers=min(32, os.cpu_count() or 1)
                if workers is None else int(workers))
            return (np.asarray(result['coordinates']),
                    np.asarray(result['offsets']))
        tracks = [self[index] for index in range(len(self))]
        lengths = np.fromiter(
            (len(track) for track in tracks), dtype=np.int64, count=len(tracks))
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        return np.concatenate(tracks, axis=0), offsets

    def as_packed_polylines(self):
        """Return contiguous coordinates and offsets for bulk geometry output."""
        return self.materialize()


class _NativeCrossingProgress:
    """Present native phase counters as one honest tqdm bar at a time."""
    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self.phase = None
        self.progress = None
        self.completed = 0

    def __call__(self, phase, completed, total):
        if not self.enabled:
            return
        completed = int(completed)
        total = int(total)
        if phase != self.phase:
            self.close()
            self.phase = phase
            self.completed = 0
            units = {
                'radix sorting packed voxel keys': 'passes',
                'finding exact crossings': 'points',
                'sorting crossing events': 'events',
                'computing track arclengths': 'tracks',
                'consolidating track pairs': 'pairs',
                'encoding crossing CSR': 'pairs',
            }
            self.progress = tqdm(
                total=total, desc=phase, unit=units.get(phase, 'items'))
        if self.progress.total != total:
            self.progress.total = total
            self.progress.refresh()
        self.progress.update(max(0, completed - self.completed))
        self.completed = completed
        if completed >= total:
            self.close()

    def close(self):
        if self.progress is not None:
            self.progress.close()
        self.progress = None
        self.phase = None
        self.completed = 0


def normalize_tracks_dbm_path(path):
    """Return the logical DBM path, accepting a common ``.db`` backing path."""
    text = os.fspath(path)
    if dbm.whichdb(text):
        return text
    if text.endswith('.db') and dbm.whichdb(text[:-3]):
        return text[:-3]
    raise FileNotFoundError(f'not a readable DBM logical path: {path}')


def track_crossing_cache_path(path):
    """Return the conventional exact-crossing sidecar path for a tracks DBM."""
    return Path(normalize_tracks_dbm_path(path) + TRACK_CROSSING_CACHE_SUFFIX)


def _tracks_db_signature(path):
    """Fingerprint the portable set of files backing a logical DBM."""
    logical = Path(normalize_tracks_dbm_path(path))
    candidates = [
        logical,
        *[Path(str(logical) + suffix)
          for suffix in ('.db', '.dat', '.dir', '.bak', '.pag')],
    ]
    result = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved in seen or not candidate.is_file():
                continue
            seen.add(resolved)
            stat = candidate.stat()
            result.append((candidate.name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            continue
    if not result:
        raise FileNotFoundError(f'could not stat DBM backing files for {path}')
    return sorted(result)


def track_store_path(path):
    """Return the conventional packed-store directory beside a tracks DBM."""
    return Path(normalize_tracks_dbm_path(path) + TRACK_STORE_SUFFIX)


def write_packed_track_store(
        path, destination=None, *, force=False, show_progress=True):
    """Stream a legacy pickle DBM into the mmap-friendly packed track format."""
    logical = normalize_tracks_dbm_path(path)
    destination = (track_store_path(logical) if destination is None
                   else Path(destination))
    if destination.exists() and not force:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f'.{destination.name}.', dir=destination.parent))
    length_batches = []
    source_id_batches = []
    family_code_batches = []
    z_bound_batches = []
    arclength_batches = []
    tortuosity_batches = []
    point_count = 0
    try:
        with (dbm.open(logical, 'r') as database,
              (temporary / 'coordinates.i32').open('wb') as coordinate_stream):
            keys = sorted(database.keys())
            for key_ordinal, key in enumerate(tqdm(
                    keys, desc='packing track store', unit='keys',
                    disable=not show_progress)):
                entries = pickle.loads(database[key])
                nonempty = np.fromiter(
                    (index for index, entry in enumerate(entries) if len(entry)),
                    dtype=np.int64)
                if not len(nonempty):
                    continue
                lengths = np.fromiter(
                    (len(entries[index]) for index in nonempty),
                    dtype=np.int64, count=len(nonempty))
                coordinates = np.concatenate([
                    np.asarray(entries[index]) for index in nonempty])
                if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                    raise ValueError(
                        f'track DBM key {key!r} contains invalid coordinates')
                if (not np.issubdtype(coordinates.dtype, np.integer)
                        and (not np.isfinite(coordinates).all()
                             or not np.all(coordinates == np.rint(coordinates)))):
                    raise ValueError(
                        f'track DBM key {key!r} contains non-integral coordinates')
                coordinates = np.asarray(coordinates, dtype='<i4')
                coordinates.tofile(coordinate_stream)
                starts = np.empty(len(lengths), dtype=np.int64)
                starts[0] = 0
                np.cumsum(lengths[:-1], out=starts[1:])
                z_values = coordinates[:, 0]
                z_bounds = np.column_stack((
                    np.minimum.reduceat(z_values, starts),
                    np.maximum.reduceat(z_values, starts),
                )).astype('<i4', copy=False)
                steps = np.empty(len(coordinates), dtype=np.float64)
                steps[0] = 0.0
                deltas = np.subtract(
                    coordinates[1:], coordinates[:-1], dtype=np.float64)
                steps[1:] = np.linalg.norm(deltas, axis=1)
                steps[starts] = 0.0
                arclengths = np.add.reduceat(steps, starts)
                chords = np.linalg.norm(
                    coordinates[starts + lengths - 1].astype(np.float64)
                    - coordinates[starts].astype(np.float64), axis=1)
                tortuosities = np.full(len(lengths), np.inf, dtype=np.float64)
                np.divide(
                    arclengths, chords, out=tortuosities, where=chords > 0)
                prefix = key.decode().split(':', 1)[0]
                family_code = 0 if prefix == 'h' else (
                    1 if prefix in ('vx', 'vy') else -1)
                length_batches.append(lengths)
                source_id_batches.append(
                    (np.uint64(key_ordinal) << np.uint64(32))
                    | nonempty.astype(np.uint64, copy=False))
                family_code_batches.append(np.full(
                    len(nonempty), family_code, dtype=np.int8))
                z_bound_batches.append(z_bounds)
                arclength_batches.append(arclengths)
                tortuosity_batches.append(tortuosities)
                point_count += len(coordinates)

        lengths = (np.concatenate(length_batches)
                   if length_batches else np.empty(0, dtype=np.int64))
        track_count = len(lengths)
        offsets = np.empty(track_count + 1, dtype='<i8')
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        if int(offsets[-1]) != point_count:
            raise RuntimeError('packed track point count changed while writing')
        source_ids = (np.concatenate(source_id_batches).astype('<u8', copy=False)
                      if source_id_batches else np.empty(0, dtype='<u8'))
        family_codes = (np.concatenate(family_code_batches)
                        if family_code_batches else np.empty(0, dtype=np.int8))
        z_bounds = (np.concatenate(z_bound_batches).astype('<i4', copy=False)
                    if z_bound_batches else np.empty((0, 2), dtype='<i4'))
        arclengths = (np.concatenate(arclength_batches).astype('<f8', copy=False)
                      if arclength_batches else np.empty(0, dtype='<f8'))
        tortuosities = (
            np.concatenate(tortuosity_batches).astype('<f8', copy=False)
            if tortuosity_batches else np.empty(0, dtype='<f8'))
        offsets.tofile(temporary / 'offsets.i64')
        source_ids.tofile(temporary / 'source_ids.u64')
        family_codes.tofile(temporary / 'family_codes.i8')
        z_bounds.tofile(temporary / 'z_bounds.i32')
        arclengths.tofile(temporary / 'arclengths.f64')
        tortuosities.tofile(temporary / 'tortuosities.f64')
        with (temporary / 'header.bin').open('wb') as stream:
            stream.write(struct.pack(
                '<8sIIQQ4Q', TRACK_STORE_MAGIC, TRACK_STORE_VERSION, 64,
                track_count, point_count, 0, 0, 0, 0))
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {
            'version': TRACK_STORE_VERSION,
            'source_db_signature': [list(item) for item in _tracks_db_signature(logical)],
            'track_count': track_count,
            'point_count': point_count,
        }
        with (temporary / 'metadata.json').open('w', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            if not force:
                raise FileExistsError(destination)
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(temporary, destination)
        temporary = None
        print(
            f'wrote packed track store {destination}: {track_count} tracks, '
            f'{point_count} points')
        return destination
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _load_packed_track_collection(path, z_lo=None, z_hi=None, warn=True):
    native = _load_native_track_store()
    if native is None:
        return None
    try:
        store = track_store_path(path)
        if not store.is_dir():
            return None
        with (store / 'metadata.json').open('r', encoding='utf-8') as stream:
            metadata = json.load(stream)
        if metadata.get('version') != TRACK_STORE_VERSION:
            raise ValueError('unsupported packed track-store version')
        expected = [list(item) for item in _tracks_db_signature(path)]
        if metadata.get('source_db_signature') != expected:
            raise ValueError('source DBM changed after the packed store was written')
        result = native.load(
            os.fspath(store),
            z_minimum=-(1 << 63) if z_lo is None else int(z_lo),
            z_maximum=(1 << 63) - 1 if z_hi is None else int(z_hi),
            workers=min(32, os.cpu_count() or 1),
        )
        collection = PackedTrackCollection(
            result['coordinates'], result['offsets'], result['source_ids'],
            result['family_codes'], result['arclengths'],
            result['tortuosities'], store_path=store)
        print(
            f'loaded packed track store {store}: {len(collection)} tracks, '
            f'{len(collection.coordinates)} points in ROI')
        return collection
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        if warn:
            print(f'WARNING: ignoring packed track store: {error}')
        return None


def _packed_store_if_current(path):
    """Return an adjacent packed store only when it matches the source DBM."""
    store = track_store_path(path)
    if not store.is_dir():
        return None
    try:
        with (store / 'metadata.json').open('r', encoding='utf-8') as stream:
            metadata = json.load(stream)
        if metadata.get('version') != TRACK_STORE_VERSION:
            return None
        expected = [list(item) for item in _tracks_db_signature(path)]
        if metadata.get('source_db_signature') != expected:
            return None
        return store
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _dbm_source_id(key_ordinal, source_index):
    if key_ordinal >= (1 << 32) or source_index >= (1 << 32):
        raise ValueError('track DBM key and entry indices must fit in 32 bits')
    return (int(key_ordinal) << 32) | int(source_index)


def load_tracks_from_dbm(
        path, z_lo=None, z_hi=None, return_families=False,
        return_source_ids=False, show_progress=True, low_memory=False):
    # Load tracks written by extract_surface_tracks.py. Each DBM value is a
    # pickled list of (N, 3) int32 zyx arrays; keep only tracks that lie entirely
    # within the full-resolution [z_lo, z_hi) ROI.
    path = normalize_tracks_dbm_path(path)
    if not low_memory:
        packed = _load_packed_track_collection(path, z_lo, z_hi)
        if packed is not None:
            if return_families and return_source_ids:
                return (packed, packed.selected_family_codes,
                        packed.selected_source_ids)
            if return_families:
                return packed, packed.selected_family_codes
            if return_source_ids:
                return packed, packed.selected_source_ids
            return packed
    tracks = []
    families = []
    source_ids = []
    with dbm.open(path, 'r') as db:
        keys = db.keys()
        key_ordinals = {key: index for index, key in enumerate(sorted(keys))}
        for key in tqdm(keys, desc='loading tracks', disable=not show_progress):
            family = None
            if return_families:
                prefix = key.decode().split(':', 1)[0]
                family = 'horizontal' if prefix == 'h' else (
                    'vertical' if prefix in ('vx', 'vy') else None)
            entries = pickle.loads(db[key])
            if not entries:
                continue
            if low_memory:
                for source_index, entry in enumerate(entries):
                    if not len(entry):
                        continue
                    z_column = entry[:, 0]
                    if z_lo is not None and z_column.min() < z_lo:
                        continue
                    if z_hi is not None and z_column.max() >= z_hi:
                        continue
                    # The crossing builder consumes the DBM's native int32
                    # voxel coordinates. Avoid a second per-key float32 copy.
                    tracks.append(np.asarray(entry))
                    if return_families:
                        families.append(family)
                    if return_source_ids:
                        source_ids.append(_dbm_source_id(
                            key_ordinals[key], source_index))
                continue
            # Vectorize the per-track z min/max across the whole key: concatenate
            # every non-empty track's z column and reduce per segment, rather
            # than calling .min()/.max() once per track.
            idx = [i for i in range(len(entries)) if len(entries[i])]
            if not idx:
                continue
            lengths = np.fromiter((len(entries[i]) for i in idx), dtype=np.intp, count=len(idx))
            zcat = np.concatenate([entries[i][:, 0] for i in idx])
            offsets = np.zeros(len(idx), dtype=np.intp)
            np.cumsum(lengths[:-1], out=offsets[1:])
            zmins = np.minimum.reduceat(zcat, offsets)
            zmaxs = np.maximum.reduceat(zcat, offsets)
            keep = np.ones(len(idx), dtype=bool)
            if z_lo is not None:
                keep &= zmins >= z_lo
            if z_hi is not None:
                keep &= zmaxs < z_hi
            for j in np.nonzero(keep)[0]:
                source_index = idx[j]
                tracks.append(entries[source_index].astype(np.float32))
                if return_families:
                    families.append(family)
                if return_source_ids:
                    source_ids.append(_dbm_source_id(
                        key_ordinals[key], source_index))
    if return_families and return_source_ids:
        return tracks, families, np.asarray(source_ids, dtype=np.uint64)
    if return_families:
        return tracks, families
    if return_source_ids:
        return tracks, np.asarray(source_ids, dtype=np.uint64)
    return tracks


def _iter_tracks_from_dbm(path, z_lo=None, z_hi=None, show_progress=True):
    """Yield native voxel tracks in stable source-id order, one DBM key at a time."""
    path = normalize_tracks_dbm_path(path)
    with dbm.open(path, 'r') as database:
        keys = sorted(database.keys())
        for key_ordinal, key in enumerate(tqdm(
                keys, desc='streaming tracks', disable=not show_progress)):
            prefix = key.decode().split(':', 1)[0]
            family = 'horizontal' if prefix == 'h' else (
                'vertical' if prefix in ('vx', 'vy') else None)
            entries = pickle.loads(database[key])
            for source_index, entry in enumerate(entries):
                if not len(entry):
                    continue
                z_column = entry[:, 0]
                if z_lo is not None and z_column.min() < z_lo:
                    continue
                if z_hi is not None and z_column.max() >= z_hi:
                    continue
                yield (
                    np.asarray(entry), family,
                    _dbm_source_id(key_ordinal, source_index),
                )


def validate_track_sampling_config(config):
    """Validate and normalize the optional session-scoped track policies."""
    weights = config.get('track_length_bin_weights')
    if weights is not None:
        if not isinstance(weights, (list, tuple)) or len(weights) != 3:
            raise ValueError('track_length_bin_weights must be null or [short, medium, long]')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) or float(value) < 0
               for value in weights):
            raise ValueError('track_length_bin_weights values must be finite and non-negative')
        weights = np.asarray(weights, dtype=np.float64)
        if float(weights.sum()) <= 0:
            raise ValueError('track_length_bin_weights must contain at least one positive value')
        weights = weights / weights.sum()

    max_tortuosity = config.get('track_max_tortuosity')
    if max_tortuosity is not None:
        if (isinstance(max_tortuosity, bool)
                or not isinstance(max_tortuosity, (int, float))
                or not math.isfinite(float(max_tortuosity))
                or float(max_tortuosity) < 1.0):
            raise ValueError('track_max_tortuosity must be null or a finite number >= 1')
        max_tortuosity = float(max_tortuosity)

    max_crossings = config.get('max_track_crossing_per_step', 0)
    if (isinstance(max_crossings, bool) or not isinstance(max_crossings, (int, float))
            or not math.isfinite(float(max_crossings))
            or not float(max_crossings).is_integer() or int(max_crossings) < 0):
        raise ValueError('max_track_crossing_per_step must be a non-negative integer')

    crossing_precompute_max = config.get('track_crossing_precompute_max', 8)
    if (isinstance(crossing_precompute_max, bool)
            or not isinstance(crossing_precompute_max, (int, float))
            or not math.isfinite(float(crossing_precompute_max))
            or not float(crossing_precompute_max).is_integer()
            or int(crossing_precompute_max) < 0):
        raise ValueError('track_crossing_precompute_max must be a non-negative integer')
    # Existing profiles may already request more than the default ceiling.
    # Honor that initial value by preparing at least as many slots; the
    # resident session still rejects later Run requests above what it prepared.
    crossing_precompute_max = max(
        int(crossing_precompute_max), int(max_crossings))

    sample_spacings = {}
    for key, default in (
            ('track_min_sample_spacing', 20.0),
            ('track_max_sample_spacing', 60.0)):
        value = config.get(key, default)
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0):
            raise ValueError(f'{key} must be a finite number > 0')
        sample_spacings[key] = float(value)
    if (sample_spacings['track_min_sample_spacing']
            > sample_spacings['track_max_sample_spacing']):
        raise ValueError(
            'track_min_sample_spacing must be <= track_max_sample_spacing')

    return {
        'length_bin_weights': weights,
        'max_tortuosity': max_tortuosity,
        'max_crossings': int(max_crossings),
        'crossing_precompute_max': crossing_precompute_max,
        'min_sample_spacing': sample_spacings['track_min_sample_spacing'],
        'max_sample_spacing': sample_spacings['track_max_sample_spacing'],
    }


def _polyline_arclengths(tracks):
    return np.asarray([
        np.linalg.norm(np.diff(np.asarray(track, dtype=np.float64), axis=0), axis=1).sum()
        if len(track) >= 2 else 0.0
        for track in tracks
    ], dtype=np.float64)


def _track_tortuosities(tracks, arclengths):
    chords = np.asarray([
        np.linalg.norm(np.asarray(track[-1], dtype=np.float64)
                       - np.asarray(track[0], dtype=np.float64))
        if len(track) >= 2 else 0.0
        for track in tracks
    ], dtype=np.float64)
    result = np.full(len(tracks), np.inf, dtype=np.float64)
    np.divide(arclengths, chords, out=result, where=chords > 0)
    return result


def filter_tracks_to_outer_shell(
        tracks, shell_envelope, track_families=None, max_points_per_chunk=500_000,
        return_indices=False):
    """Drop tracks whose two endpoints are outside the shell envelope."""
    if shell_envelope is None or not tracks:
        result = (tracks, track_families)
        if return_indices:
            return (*result, np.arange(len(tracks), dtype=np.int64))
        return result
    if track_families is not None and len(track_families) != len(tracks):
        raise ValueError('track_families must be parallel to tracks')

    if isinstance(tracks, PackedTrackCollection):
        keep = np.zeros(len(tracks), dtype=bool)
        tracks_per_chunk = max(1, int(max_points_per_chunk) // 2)
        for start in range(0, len(tracks), tracks_per_chunk):
            local_rows = np.arange(
                start, min(start + tracks_per_chunk, len(tracks)),
                dtype=np.int64)
            rows = tracks.rows[local_rows]
            begin = tracks.offsets[rows]
            end = tracks.offsets[rows + 1] - 1
            endpoints = np.stack((
                tracks.coordinates[begin], tracks.coordinates[end]), axis=1)
            with torch.no_grad():
                target_radius, radius, _confidence, _valid = shell_envelope.lookup(
                    torch.from_numpy(endpoints.reshape(-1, 3)))
                endpoint_inside = (radius <= target_radius).view(-1, 2).any(dim=1)
            keep[local_rows] = endpoint_inside.cpu().numpy()
        kept_indices = np.flatnonzero(keep)
        filtered_tracks = tracks.subset(kept_indices)
        filtered_families = (
            np.asarray(track_families)[kept_indices]
            if track_families is not None else None)
        print(
            f'outer-shell track filter: kept {len(filtered_tracks)} / {len(tracks)} '
            f'tracks; excluded {len(tracks) - len(filtered_tracks)} entirely '
            'outside tracks (packed)')
        result = (filtered_tracks, filtered_families)
        if return_indices:
            return (*result, kept_indices)
        return result

    keep = np.zeros(len(tracks), dtype=bool)
    nonempty = np.asarray(
        [index for index, track in enumerate(tracks) if len(track)], dtype=np.int64)
    tracks_per_chunk = max(1, int(max_points_per_chunk) // 2)
    for start in range(0, len(nonempty), tracks_per_chunk):
        batch_indices = nonempty[start:start + tracks_per_chunk]
        endpoints = np.stack([
            np.stack([tracks[index][0], tracks[index][-1]])
            for index in batch_indices
        ]).astype(np.float32, copy=False)
        with torch.no_grad():
            target_radius, radius, _confidence, _valid = shell_envelope.lookup(
                torch.from_numpy(endpoints.reshape(-1, 3)))
            endpoint_inside = (radius <= target_radius).view(-1, 2).any(dim=1)
        keep[batch_indices] = endpoint_inside.cpu().numpy()

    kept_indices = np.flatnonzero(keep)
    filtered_tracks = [tracks[index] for index in kept_indices]
    filtered_families = (
        [track_families[index] for index in kept_indices]
        if track_families is not None else None)
    print(
        f'outer-shell track filter: kept {len(filtered_tracks)} / {len(tracks)} tracks; '
        f'excluded {len(tracks) - len(filtered_tracks)} entirely outside tracks'
    )
    result = (filtered_tracks, filtered_families)
    if return_indices:
        return (*result, kept_indices)
    return result


def _length_bin_probabilities(arclengths, weights, device):
    edges = np.quantile(arclengths, [1 / 3, 2 / 3])
    bin_ids = (arclengths > edges[0]).astype(np.int64)
    bin_ids += (arclengths > edges[1]).astype(np.int64)
    counts = np.bincount(bin_ids, minlength=3)
    available_weights = np.where(counts > 0, weights, 0.0)
    if available_weights.sum() <= 0:
        raise ValueError(
            'track_length_bin_weights assign no probability to the non-empty length bins')
    available_weights /= available_weights.sum()
    probabilities = available_weights[bin_ids] / counts[bin_ids]
    print(
        'track length bins: '
        f'short <= {edges[0]:.1f} ({counts[0]}), '
        f'medium <= {edges[1]:.1f} ({counts[1]}), long ({counts[2]}); '
        f'effective weights {available_weights.tolist()}'
    )
    return torch.as_tensor(probabilities, dtype=torch.float32, device=device)


def configure_prepared_track_sampling(prepared_tracks, config):
    """Apply Run-scoped track policies to an already prepared track pool."""
    if prepared_tracks is None:
        return
    current = {
        'track_length_bin_weights': prepared_tracks.get('length_bin_weights'),
        'max_track_crossing_per_step': prepared_tracks.get('active_max_crossings', 0),
        'track_crossing_precompute_max': prepared_tracks.get(
            'crossing_precompute_max', 0),
    }
    current.update({
        key: config[key] for key in current
        if key in config
    })
    policy = validate_track_sampling_config(current)

    if 'track_length_bin_weights' in config:
        weights = policy['length_bin_weights']
        if weights is None:
            prepared_tracks.pop('sampling_probabilities', None)
            prepared_tracks['length_bin_weights'] = None
        else:
            prepared_tracks['sampling_probabilities'] = _length_bin_probabilities(
                prepared_tracks['arclengths_cpu'], weights,
                prepared_tracks['device'])
            prepared_tracks['length_bin_weights'] = weights.tolist()

    maximum = policy['max_crossings']
    prepared_maximum = prepared_tracks.get('crossing_precompute_max', 0)
    if maximum > prepared_maximum:
        raise ValueError(
            f'max_track_crossing_per_step={maximum} exceeds the session\'s '
            f'prepared crossing ceiling ({prepared_maximum}); reload with a larger '
            'track_crossing_precompute_max')
    prepared_tracks['active_max_crossings'] = maximum


def _track_tangent(track, raw_index, radius_voxels=12.0):
    point = np.asarray(track[raw_index], dtype=np.float64)
    left = raw_index
    while left > 0 and np.linalg.norm(np.asarray(track[left], dtype=np.float64) - point) < radius_voxels:
        left -= 1
    right = raw_index
    while (right + 1 < len(track)
           and np.linalg.norm(np.asarray(track[right], dtype=np.float64) - point) < radius_voxels):
        right += 1
    vector = np.asarray(track[right], dtype=np.float64) - np.asarray(track[left], dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def _pack_track_points(
        points, chunk_size=2_000_000, show_progress=False, destination=None):
    """Pack zyx coordinates without materialising an (N, 3) int64 copy.

    Track pools routinely contain tens of millions of float32 points.  Casting
    the complete array to int64 used more memory than the coordinates
    themselves and created another similarly sized temporary while combining
    the columns.  Fill the packed output a chunk at a time instead.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('track points must have shape (N, 3)')
    if destination is None:
        packed = np.empty(len(points), dtype=np.uint64)
    else:
        packed = np.asarray(destination)
        if packed.dtype != np.uint64 or packed.shape != (len(points),):
            raise ValueError(
                'packed track destination must be a uint64 vector parallel to points')
    chunk_size = max(1, int(chunk_size))
    coordinate_limit = 1 << 20
    shift_z = np.uint64(40)
    shift_y = np.uint64(20)
    chunks = range(0, len(points), chunk_size)
    if show_progress:
        chunks = tqdm(
            chunks, total=math.ceil(len(points) / chunk_size),
            desc='packing track coordinates', unit='chunks')
    for start in chunks:
        stop = min(start + chunk_size, len(points))
        source = points[start:stop]
        if (not np.all(np.isfinite(source)) or np.any(source < 0)
                or np.any(source >= coordinate_limit)):
            raise ValueError(
                'track coordinates must lie in [0, 2**20) for crossing pairing')
        destination = packed[start:stop]
        np.copyto(destination, source[:, 0], casting='unsafe')
        destination <<= shift_z
        temporary = source[:, 1].astype(np.uint64)
        temporary <<= shift_y
        destination |= temporary
        np.copyto(temporary, source[:, 2], casting='unsafe')
        destination |= temporary
    return packed


def _pack_track_collection(
        tracks, offsets, show_progress=False, chunk_size=2_000_000):
    """Pack a ragged track collection without concatenating its coordinates."""
    packed = np.empty(int(offsets[-1]), dtype=np.uint64)
    chunk_size = max(1, int(chunk_size))
    progress = tqdm(
        total=len(tracks), desc='packing track coordinates', unit='tracks',
        disable=not show_progress)
    track_begin = 0
    try:
        while track_begin < len(tracks):
            point_begin = int(offsets[track_begin])
            point_limit = point_begin + chunk_size
            track_end = int(np.searchsorted(
                offsets, point_limit, side='right')) - 1
            track_end = max(track_begin + 1, min(track_end, len(tracks)))
            point_end = int(offsets[track_end])
            if track_end == track_begin + 1:
                coordinates = tracks[track_begin]
            else:
                coordinates = np.concatenate(
                    tracks[track_begin:track_end], axis=0)
            _pack_track_points(
                coordinates, destination=packed[point_begin:point_end])
            progress.update(track_end - track_begin)
            track_begin = track_end
    finally:
        progress.close()
    return packed


def _select_spaced_crossing_partners(candidates, maximum):
    """Choose distinct partners, spreading their crossings along the primary."""
    if not candidates or maximum <= 0:
        return []
    remaining = list(candidates)
    if maximum == 1 or len(remaining) == 1:
        return [max(remaining, key=lambda item: (item[2], -item[0]))[0]]

    # Seed the set with the widest-separated pair, then use maximin spacing
    # for any remaining slots. Clearance and partner id make ties stable.
    first = min(remaining, key=lambda item: (item[1], -item[2], item[0]))
    selected = [first]
    remaining.remove(first)
    second = max(
        remaining,
        key=lambda item: (abs(item[1] - first[1]), item[2], -item[0]),
    )
    selected.append(second)
    remaining.remove(second)
    while remaining and len(selected) < maximum:
        positions = [item[1] for item in selected]
        choice = max(
            remaining,
            key=lambda item: (
                min(abs(item[1] - position) for position in positions),
                item[2],
                -item[0],
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return [item[0] for item in selected]


_CROSSING_WORKER_STATE = None
_CROSSING_DISK_WORKER_STATE = None


def _find_crossing_events_chunk(group_range):
    """Scan a bounded range of the sorted packed-coordinate index."""
    if _CROSSING_WORKER_STATE is None:
        raise RuntimeError('crossing worker state is not initialized')
    (tracks, families, offsets, packed, order,
     angle_cutoff) = _CROSSING_WORKER_STATE
    position_begin, position_end = group_range
    flat_indices = order[position_begin:position_end]
    sorted_packed = packed[flat_indices]
    boundaries = np.flatnonzero(sorted_packed[1:] != sorted_packed[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(sorted_packed)]
    shared = stops - starts >= 2
    starts = starts[shared]
    stops = stops[shared]
    tangent_cache = {}
    raw_events = {}
    for start, stop in zip(starts, stops):
        group_flat_indices = flat_indices[int(start):int(stop)]
        track_ids = np.searchsorted(
            offsets[1:], group_flat_indices, side='right')
        local_indices = group_flat_indices - offsets[track_ids]
        unique = {}
        for track_id, local_index in zip(track_ids, local_indices):
            track_id = int(track_id)
            local_index = int(local_index)
            previous = unique.get(track_id)
            if previous is None or local_index < previous:
                unique[track_id] = local_index
        if len(unique) < 2:
            continue
        for first, second in itertools.combinations(unique, 2):
            if (families[first] is None or families[second] is None
                    or families[first] == families[second]):
                continue
            first_index, second_index = unique[first], unique[second]
            first_key, second_key = (first, first_index), (second, second_index)
            if first_key not in tangent_cache:
                tangent_cache[first_key] = _track_tangent(tracks[first], first_index)
            if second_key not in tangent_cache:
                tangent_cache[second_key] = _track_tangent(tracks[second], second_index)
            first_tangent = tangent_cache[first_key]
            second_tangent = tangent_cache[second_key]
            if first_tangent is None or second_tangent is None:
                continue
            if abs(float(np.dot(first_tangent, second_tangent))) > angle_cutoff:
                continue
            raw_events.setdefault((first, second), []).append(
                (first_index, second_index))
    return position_end - position_begin, raw_events


def _crossing_events_to_csr(
        raw_events, tracks, source_ids, show_progress=False):
    """Consolidate exact-voxel events into the cached partner CSR."""
    crossing_track_ids = set(itertools.chain.from_iterable(raw_events))
    needed_local_indices = {track_id: set() for track_id in crossing_track_ids}
    for (first, second), events in raw_events.items():
        needed_local_indices[first].update(event[0] for event in events)
        needed_local_indices[second].update(event[1] for event in events)
    event_positions = {}
    track_arclengths = {}
    for track_id, local_indices in needed_local_indices.items():
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(
            np.diff(np.asarray(tracks[track_id], dtype=np.float64), axis=0),
            axis=1))]
        event_positions[track_id] = {
            local_index: float(cumulative[local_index])
            for local_index in local_indices
        }
        track_arclengths[track_id] = float(cumulative[-1])
    del needed_local_indices
    adjacency = {}
    accepted_events = 0
    pair_iterator = raw_events.items()
    if show_progress:
        pair_iterator = tqdm(
            pair_iterator, total=len(raw_events),
            desc='consolidating track pairs', unit='pairs')
    for (first, second), events in pair_iterator:
        events.sort()
        clusters = []
        cluster = []
        for event in events:
            if (cluster and abs(event[0] - cluster[-1][0]) <= 4
                    and abs(event[1] - cluster[-1][1]) <= 4):
                cluster.append(event)
            else:
                if cluster:
                    clusters.append(cluster)
                cluster = [event]
        if cluster:
            clusters.append(cluster)
        representatives = [cluster[len(cluster) // 2] for cluster in clusters]
        best = None
        for first_index, second_index in representatives:
            first_position = event_positions[first][first_index]
            second_position = event_positions[second][second_index]
            clearance = min(
                first_position, track_arclengths[first] - first_position,
                second_position, track_arclengths[second] - second_position,
            )
            candidate = (
                clearance, first_position, second_position,
                first_index, second_index,
            )
            if best is None or candidate > best:
                best = candidate
        (clearance, first_position, second_position,
         best_first_index, best_second_index) = best
        adjacency.setdefault(first, []).append((
            second, first_position, clearance,
            best_first_index, best_second_index,
        ))
        adjacency.setdefault(second, []).append((
            first, second_position, clearance,
            best_second_index, best_first_index,
        ))
        accepted_events += len(representatives)
    del raw_events, event_positions, track_arclengths

    counts = np.zeros(len(tracks), dtype=np.int64)
    for track_id, candidates in adjacency.items():
        counts[track_id] = len(candidates)
    csr_offsets = np.empty(len(tracks) + 1, dtype=np.int64)
    csr_offsets[0] = 0
    np.cumsum(counts, out=csr_offsets[1:])
    partner_slots = int(csr_offsets[-1])
    partners = np.empty(partner_slots, dtype=np.int32)
    self_local = np.empty(partner_slots, dtype=np.int32)
    partner_local = np.empty(partner_slots, dtype=np.int32)
    positions = np.empty(partner_slots, dtype=np.float64)
    clearances = np.empty(partner_slots, dtype=np.float64)
    track_iterator = adjacency.items()
    if show_progress:
        track_iterator = tqdm(
            track_iterator, total=len(adjacency),
            desc='encoding crossing CSR', unit='paired tracks')
    for track_id, candidates in track_iterator:
        start = int(csr_offsets[track_id])
        candidates.sort(key=lambda candidate: int(source_ids[candidate[0]]))
        for slot, candidate in enumerate(candidates, start=start):
            partners[slot] = candidate[0]
            positions[slot] = candidate[1]
            clearances[slot] = candidate[2]
            self_local[slot] = candidate[3]
            partner_local[slot] = candidate[4]
    paired_tracks = int(np.count_nonzero(counts))
    print(
        f'track crossings: {accepted_events} exact crossing events, '
        f'{paired_tracks}/{len(tracks)} tracks have partners, '
        f'{partner_slots} directed partner records retained'
    )
    return {
        'source_ids': source_ids,
        'offsets': csr_offsets,
        'partners': partners,
        'self_local': self_local,
        'partner_local': partner_local,
        'positions': positions,
        'clearances': clearances,
    }


class _MemmapTrackCollection:
    def __init__(self, coordinates, offsets):
        self.coordinates = coordinates
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, track_id):
        return self.coordinates[
            int(self.offsets[track_id]):int(self.offsets[track_id + 1])]


def _find_crossing_events_disk_chunk(position_range):
    """Scan one bounded range of the in-memory packed-coordinate order."""
    if _CROSSING_DISK_WORKER_STATE is None:
        raise RuntimeError('disk crossing worker state is not initialized')
    (tracks, offsets, family_codes, packed, order,
     angle_cutoff) = _CROSSING_DISK_WORKER_STATE
    position_begin, position_end = position_range
    flat_indices = order[position_begin:position_end]
    sorted_packed = packed[flat_indices]
    boundaries = np.flatnonzero(
        sorted_packed[1:] != sorted_packed[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(sorted_packed)]
    shared = stops - starts >= 2
    starts = starts[shared]
    stops = stops[shared]
    tangent_cache = {}
    raw_events = {}
    for start, stop in zip(starts, stops):
        group_flat_indices = flat_indices[int(start):int(stop)]
        track_ids = np.searchsorted(
            offsets[1:], group_flat_indices, side='right')
        local_indices = group_flat_indices - offsets[track_ids]
        unique = {}
        for track_id, local_index in zip(track_ids, local_indices):
            track_id = int(track_id)
            local_index = int(local_index)
            previous = unique.get(track_id)
            if previous is None or local_index < previous:
                unique[track_id] = local_index
        if len(unique) < 2:
            continue
        for first, second in itertools.combinations(unique, 2):
            if (family_codes[first] < 0 or family_codes[second] < 0
                    or family_codes[first] == family_codes[second]):
                continue
            first_index, second_index = unique[first], unique[second]
            first_key, second_key = (first, first_index), (second, second_index)
            if first_key not in tangent_cache:
                tangent_cache[first_key] = _track_tangent(
                    tracks[first], first_index)
            if second_key not in tangent_cache:
                tangent_cache[second_key] = _track_tangent(
                    tracks[second], second_index)
            first_tangent = tangent_cache[first_key]
            second_tangent = tangent_cache[second_key]
            if first_tangent is None or second_tangent is None:
                continue
            if abs(float(np.dot(first_tangent, second_tangent))) > angle_cutoff:
                continue
            raw_events.setdefault((first, second), []).append(
                (first_index, second_index))
    return position_end - position_begin, raw_events


def _write_packed_store_crossing_index(
        store, coordinates_path, packed_path,
        z_lo=None, z_hi=None, show_progress=True):
    with (store / 'metadata.json').open('r', encoding='utf-8') as stream:
        metadata = json.load(stream)
    track_count = int(metadata['track_count'])
    point_count = int(metadata['point_count'])
    coordinates = np.memmap(
        store / 'coordinates.i32', mode='r', dtype='<i4',
        shape=(point_count, 3))
    offsets = np.memmap(
        store / 'offsets.i64', mode='r', dtype='<i8',
        shape=(track_count + 1,))
    source_ids = np.memmap(
        store / 'source_ids.u64', mode='r', dtype='<u8',
        shape=(track_count,))
    family_codes = np.memmap(
        store / 'family_codes.i8', mode='r', dtype=np.int8,
        shape=(track_count,))
    z_bounds = np.memmap(
        store / 'z_bounds.i32', mode='r', dtype='<i4',
        shape=(track_count, 2))
    keep = offsets[1:] > offsets[:-1]
    if z_lo is not None:
        keep &= z_bounds[:, 0] >= z_lo
    if z_hi is not None:
        keep &= z_bounds[:, 1] < z_hi
    selected = np.flatnonzero(keep)
    lengths = (offsets[selected + 1] - offsets[selected]).astype(
        np.int64, copy=False)
    output_offsets = np.empty(len(selected) + 1, dtype=np.int64)
    output_offsets[0] = 0
    np.cumsum(lengths, out=output_offsets[1:])

    progress = tqdm(
        total=len(selected), desc='staging packed tracks', unit='tracks',
        disable=not show_progress)
    try:
        with (open(coordinates_path, 'wb') as coordinate_stream,
              open(packed_path, 'wb') as packed_stream):
            begin = 0
            while begin < len(selected):
                point_limit = int(output_offsets[begin]) + 2_000_000
                end = int(np.searchsorted(
                    output_offsets, point_limit, side='right')) - 1
                end = max(begin + 1, min(end, len(selected)))
                rows = selected[begin:end]
                batch = np.concatenate([
                    coordinates[offsets[row]:offsets[row + 1]]
                    for row in rows], axis=0)
                batch = np.asarray(batch, dtype='<i4')
                batch.tofile(coordinate_stream)
                _pack_track_points(batch).tofile(packed_stream)
                progress.update(end - begin)
                begin = end
    finally:
        progress.close()
    print(f'track crossings: staged directly from packed store {store}')
    return (
        output_offsets,
        np.asarray(source_ids[selected]).copy(),
        np.asarray(family_codes[selected]).copy(),
    )


def _write_disk_crossing_index(
        path, coordinates_path, packed_path,
        z_lo=None, z_hi=None, show_progress=True):
    length_batches = []
    source_id_batches = []
    family_code_batches = []
    path = normalize_tracks_dbm_path(path)
    packed_store = _packed_store_if_current(path)
    if packed_store is not None:
        return _write_packed_store_crossing_index(
            packed_store, coordinates_path, packed_path,
            z_lo=z_lo, z_hi=z_hi, show_progress=show_progress)
    with (open(coordinates_path, 'wb') as coordinate_stream,
          open(packed_path, 'wb') as packed_stream,
          dbm.open(path, 'r') as database):
        keys = sorted(database.keys())
        for key_ordinal, key in enumerate(tqdm(
                keys, desc='staging track keys', unit='keys',
                disable=not show_progress)):
            entries = pickle.loads(database[key])
            nonempty = np.fromiter(
                (index for index, entry in enumerate(entries) if len(entry)),
                dtype=np.int64)
            if not len(nonempty):
                continue
            lengths = np.fromiter(
                (len(entries[index]) for index in nonempty),
                dtype=np.int64, count=len(nonempty))
            z_values = np.concatenate([
                np.asarray(entries[index])[:, 0] for index in nonempty])
            starts = np.empty(len(lengths), dtype=np.int64)
            starts[0] = 0
            np.cumsum(lengths[:-1], out=starts[1:])
            keep = np.ones(len(nonempty), dtype=bool)
            if z_lo is not None:
                keep &= np.minimum.reduceat(z_values, starts) >= z_lo
            if z_hi is not None:
                keep &= np.maximum.reduceat(z_values, starts) < z_hi
            selected = nonempty[keep]
            if not len(selected):
                continue
            selected_lengths = lengths[keep]
            coordinates = np.concatenate([
                np.asarray(entries[index]) for index in selected])
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise ValueError('track points must have shape (N, 3)')
            coordinates = np.asarray(coordinates, dtype='<i4')
            coordinates.tofile(coordinate_stream)
            packed = _pack_track_points(coordinates)
            packed.tofile(packed_stream)
            prefix = key.decode().split(':', 1)[0]
            family_code = 0 if prefix == 'h' else (1 if prefix in ('vx', 'vy') else -1)
            length_batches.append(selected_lengths)
            source_id_batches.append(
                (np.uint64(key_ordinal) << np.uint64(32))
                | selected.astype(np.uint64, copy=False))
            family_code_batches.append(np.full(
                len(selected), family_code, dtype=np.int8))

    if not length_batches:
        return (
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.int8),
        )
    lengths = np.concatenate(length_batches)
    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    source_ids = np.concatenate(source_id_batches)
    family_codes = np.concatenate(family_code_batches)
    return offsets, source_ids, family_codes


def _scan_hybrid_crossing_index(
        tracks, offsets, family_codes, packed, order, workers, show_progress):
    point_count = len(order)
    # Keep result latency low enough that the progress bar reflects real work.
    # The previous point_count/(workers*16) policy produced ~4-million-point
    # tasks on billion-point datasets and appeared stuck until a whole task
    # returned.
    chunk_size = 500_000
    tasks = []
    begin = 0
    while begin < point_count:
        stop = min(begin + chunk_size, point_count)
        while (stop < point_count
               and packed[order[stop - 1]] == packed[order[stop]]):
            stop += 1
        tasks.append((begin, stop))
        begin = stop

    angle_cutoff = math.cos(math.radians(30.0))
    global _CROSSING_DISK_WORKER_STATE
    _CROSSING_DISK_WORKER_STATE = (
        tracks, offsets, family_codes, packed, order, angle_cutoff)
    raw_events = {}

    def merge_chunk_result(result):
        completed, chunk_events = result
        for pair, events in chunk_events.items():
            raw_events.setdefault(pair, []).extend(events)
        return completed

    progress = tqdm(
        total=point_count, desc='finding exact crossings',
        unit='points', disable=not show_progress)
    try:
        use_fork_workers = workers > 1 and 'fork' in mp.get_all_start_methods()
        if workers > 1 and not use_fork_workers:
            print('WARNING: fork multiprocessing is unavailable; '
                  'finding crossings with one worker')
        if use_fork_workers and tasks:
            context = mp.get_context('fork')
            with context.Pool(processes=workers) as pool:
                for result in pool.imap_unordered(
                        _find_crossing_events_disk_chunk, tasks, chunksize=1):
                    progress.update(merge_chunk_result(result))
        else:
            for task in tasks:
                progress.update(merge_chunk_result(
                    _find_crossing_events_disk_chunk(task)))
    finally:
        progress.close()
        _CROSSING_DISK_WORKER_STATE = None
    return raw_events


def build_crossing_partner_csr_disk_backed(
        path, z_lo=None, z_hi=None, workers=1, show_progress=True,
        temporary_directory=None):
    """Build exact crossings with mmap coordinates and an in-memory key sort."""
    workers = max(1, int(workers))
    with tempfile.TemporaryDirectory(
            prefix='track-crossings-', dir=temporary_directory) as workspace:
        coordinates_path = Path(workspace) / 'coordinates.i32'
        packed_path = Path(workspace) / 'packed-voxels.u64'
        offsets, source_ids, family_codes = _write_disk_crossing_index(
            path, coordinates_path, packed_path,
            z_lo=z_lo, z_hi=z_hi, show_progress=show_progress)
        if len(source_ids) == 0:
            raise RuntimeError(f'no tracks found in {path}')
        point_count = int(offsets[-1])
        temporary_gib = (
            coordinates_path.stat().st_size + packed_path.stat().st_size
        ) / (1 << 30)
        print(
            f'track crossings: staged {len(source_ids)} tracks and '
            f'{point_count} points in {temporary_gib:.1f} GiB of temporary files')
        coordinates = np.memmap(
            coordinates_path, mode='r', dtype='<i4', shape=(point_count, 3))
        packed_gib = packed_path.stat().st_size / (1 << 30)
        print(
            f'track crossings: loading {packed_gib:.1f} GiB of packed voxel '
            'keys into memory')
        packed = np.fromfile(packed_path, dtype='<u8', count=point_count)
        if len(packed) != point_count:
            raise RuntimeError(
                f'packed voxel file contains {len(packed)} of '
                f'{point_count} expected records')
        native = _load_native_track_crossings()
        if native is None:
            sort_gib = point_count * 16 / (1 << 30)
            print(
                'track crossings: sorting packed voxel keys in memory '
                f'(~{sort_gib:.1f} GiB for keys and order)')
            order = np.argsort(packed, kind='quicksort')
        else:
            sort_peak_gib = point_count * 32 / (1 << 30)
            print(
                'track crossings: parallel radix-sorting packed voxel keys '
                f'(~{sort_peak_gib:.1f} GiB peak sort workspace)')
            sort_progress = _NativeCrossingProgress(show_progress)
            try:
                order = native.parallel_argsort(
                    packed, workers=workers, progress=sort_progress)
            finally:
                sort_progress.close()
        tracks = _MemmapTrackCollection(coordinates, offsets)
        if native is None:
            print(
                'WARNING: vc.track_crossings is unavailable; using the much '
                'slower high-memory Python crossing consolidator')
            raw_events = _scan_hybrid_crossing_index(
                tracks, offsets, family_codes, packed, order,
                workers=workers, show_progress=show_progress)
            csr = _crossing_events_to_csr(
                raw_events, tracks, source_ids, show_progress=show_progress)
            del order, packed
        else:
            print(
                f'track crossings: using native shared-memory kernel with '
                f'{workers} workers')
            progress = _NativeCrossingProgress(show_progress)
            try:
                events = native.scan_crossing_events(
                    coordinates, offsets, family_codes, packed, order,
                    workers=workers, progress=progress)
                event_gib = events.memory_bytes / (1 << 30)
                print(
                    f'track crossings: retained {events.event_count} compact raw '
                    f'events ({event_gib:.2f} GiB)')
                # The native event buffer no longer references the 15+ GiB key
                # and order arrays. Release them before allocating arclength and
                # CSR storage so consolidation cannot overlap all three.
                del order, packed
                csr = native.consolidate_crossing_events(
                    events, coordinates, offsets, source_ids,
                    workers=workers, progress=progress)
            finally:
                progress.close()
            accepted_events = int(csr.pop('accepted_events'))
            paired_tracks = int(csr.pop('paired_tracks'))
            print(
                f'track crossings: {accepted_events} exact crossing events, '
                f'{paired_tracks}/{len(tracks)} tracks have partners, '
                f'{len(csr["partners"])} directed partner records retained')
        del tracks, coordinates
        return csr


def _build_crossing_partner_csr(
        tracks, families, flat_points=None, offsets=None, source_ids=None,
        workers=None, show_progress=False, worker_chunk_groups=None):
    """Build all exact opposite-family crossing partners as CPU CSR arrays."""
    if families is None or len(families) != len(tracks):
        raise ValueError('crossing-track sampling requires DBM track-family provenance')
    if source_ids is None:
        source_ids = np.arange(len(tracks), dtype=np.uint64)
    else:
        source_ids = np.asarray(source_ids, dtype=np.uint64)
        if source_ids.shape != (len(tracks),):
            raise ValueError('crossing source ids must contain one id per track')
        if len(np.unique(source_ids)) != len(source_ids):
            raise ValueError('crossing source ids must be unique')
    if not tracks:
        return {
            'source_ids': source_ids,
            'offsets': np.zeros(1, dtype=np.int64),
            'partners': np.empty(0, dtype=np.int32),
            'self_local': np.empty(0, dtype=np.int32),
            'partner_local': np.empty(0, dtype=np.int32),
            'positions': np.empty(0, dtype=np.float64),
            'clearances': np.empty(0, dtype=np.float64),
        }

    if offsets is None:
        lengths = np.fromiter(
            (len(track) for track in tracks), dtype=np.int64, count=len(tracks))
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
    else:
        offsets = np.asarray(offsets, dtype=np.int64)
        if offsets.shape != (len(tracks) + 1,):
            raise ValueError('crossing offsets must have one entry per track plus one')
    if flat_points is not None and len(flat_points) != int(offsets[-1]):
        raise ValueError('flat crossing points do not match track offsets')

    point_count = int(offsets[-1])
    print(
        f'track crossings: indexing {point_count} points from '
        f'{len(tracks)} tracks (all partners)'
    )
    workers = (min(32, os.cpu_count() or 1) if workers is None
               else max(1, int(workers)))
    native = _load_native_track_crossings()
    if native is not None:
        if flat_points is None:
            coordinates = np.concatenate([
                np.asarray(track, dtype=np.int32) for track in tracks], axis=0)
        else:
            coordinates = np.asarray(flat_points, dtype=np.int32)
        coordinates = np.ascontiguousarray(coordinates, dtype=np.int32)
        family_codes = np.fromiter((
            (0 if family in ('horizontal', 0) else
             1 if family in ('vertical', 1) else -1)
            for family in families
        ), dtype=np.int8, count=len(families))
        packed = _pack_track_points(coordinates, show_progress=show_progress)
        progress = _NativeCrossingProgress(show_progress)
        try:
            order = native.parallel_argsort(
                packed, workers=workers, progress=progress)
            events = native.scan_crossing_events(
                coordinates, offsets, family_codes, packed, order,
                workers=workers, progress=progress)
            del order, packed
            csr = native.consolidate_crossing_events(
                events, coordinates, offsets, source_ids,
                workers=workers, progress=progress)
        finally:
            progress.close()
        accepted_events = int(csr.pop('accepted_events'))
        paired_tracks = int(csr.pop('paired_tracks'))
        print(
            f'track crossings: {accepted_events} exact crossing events, '
            f'{paired_tracks}/{len(tracks)} tracks have partners, '
            f'{len(csr["partners"])} directed partner records retained '
            '(native in-memory fallback)')
        return csr
    if flat_points is None:
        packed = _pack_track_collection(
            tracks, offsets, show_progress=show_progress)
    else:
        packed = _pack_track_points(
            flat_points, show_progress=show_progress)
    if show_progress:
        with tqdm(total=1, desc='sorting packed coordinates', unit='sort') as progress:
            order = np.argsort(packed, kind='quicksort')
            progress.update()
    else:
        order = np.argsort(packed, kind='quicksort')

    angle_cutoff = math.cos(math.radians(30.0))
    global _CROSSING_WORKER_STATE
    _CROSSING_WORKER_STATE = (
        tracks, families, offsets, packed, order, angle_cutoff)
    raw_events = {}
    if worker_chunk_groups is None:
        worker_chunk_groups = max(
            100_000, math.ceil(point_count / max(1, workers * 16)))
    worker_chunk_groups = max(1, int(worker_chunk_groups))
    tasks = []
    begin = 0
    while begin < point_count:
        stop = min(begin + worker_chunk_groups, point_count)
        # Exact-coordinate groups must remain in one task so workers can
        # consolidate repeated voxels independently.
        while (stop < point_count
               and packed[order[stop - 1]] == packed[order[stop]]):
            stop += 1
        tasks.append((begin, stop))
        begin = stop

    def merge_chunk_result(result):
        completed, chunk_events = result
        for pair, events in chunk_events.items():
            raw_events.setdefault(pair, []).extend(events)
        return completed

    progress = tqdm(
        total=point_count, desc='finding exact crossings',
        unit='points', disable=not show_progress)
    try:
        use_fork_workers = workers > 1 and 'fork' in mp.get_all_start_methods()
        if workers > 1 and not use_fork_workers:
            print('WARNING: fork multiprocessing is unavailable; '
                  'finding crossings with one worker')
        if use_fork_workers and tasks:
            context = mp.get_context('fork')
            with context.Pool(processes=workers) as pool:
                for result in pool.imap_unordered(
                        _find_crossing_events_chunk, tasks, chunksize=1):
                    progress.update(merge_chunk_result(result))
        else:
            for task in tasks:
                progress.update(merge_chunk_result(
                    _find_crossing_events_chunk(task)))
    finally:
        progress.close()
        _CROSSING_WORKER_STATE = None
    del packed, order

    return _crossing_events_to_csr(
        raw_events, tracks, source_ids, show_progress=show_progress)


def _materialize_crossing_partner_table(csr, maximum, device):
    """Select a spaced fixed-width training table from a CPU CSR graph."""
    maximum = int(maximum)
    if maximum <= 0:
        return None
    offsets = csr['offsets']
    num_tracks = len(offsets) - 1
    table = np.full((num_tracks, maximum), -1, dtype=np.int64)
    self_local = np.full((num_tracks, maximum), -1, dtype=np.int64)
    partner_local = np.full((num_tracks, maximum), -1, dtype=np.int64)
    for track_id in range(num_tracks):
        start, stop = int(offsets[track_id]), int(offsets[track_id + 1])
        candidates = [
            (int(csr['partners'][index]), float(csr['positions'][index]),
             float(csr['clearances'][index]), int(csr['self_local'][index]),
             int(csr['partner_local'][index]))
            for index in range(start, stop)
        ]
        chosen = _select_spaced_crossing_partners(candidates, maximum)
        table[track_id, :len(chosen)] = chosen
        by_partner = {candidate[0]: candidate for candidate in candidates}
        for slot, partner in enumerate(chosen):
            candidate = by_partner[partner]
            self_local[track_id, slot] = candidate[3]
            partner_local[track_id, slot] = candidate[4]
    partner_slots = int(np.count_nonzero(table >= 0))
    print(
        f'track crossings: {partner_slots} partner slots selected '
        f'(max {maximum} per primary)'
    )
    return (
        torch.from_numpy(table).to(device=device),
        torch.from_numpy(self_local).to(device=device),
        torch.from_numpy(partner_local).to(device=device),
    )


def _materialize_cached_crossing_partner_table(
        csr, source_ids, maximum, device, workers=None):
    """Fuse whole-cache restriction and fixed-width selection in native code."""
    native = _load_native_track_crossings()
    if native is None or not hasattr(native, 'materialize_partner_table'):
        restricted = _restrict_crossing_partner_csr(csr, source_ids)
        return _materialize_crossing_partner_table(
            restricted, maximum, device)
    if workers is None:
        workers = min(32, os.cpu_count() or 1)
    try:
        result = native.materialize_partner_table(
            np.asarray(csr['source_ids'], dtype=np.uint64),
            np.asarray(csr['offsets'], dtype=np.int64),
            np.asarray(csr['partners'], dtype=np.int32),
            np.asarray(csr['self_local'], dtype=np.int32),
            np.asarray(csr['partner_local'], dtype=np.int32),
            np.asarray(csr['positions'], dtype=np.float64),
            np.asarray(csr['clearances'], dtype=np.float64),
            np.asarray(source_ids, dtype=np.uint64),
            maximum=int(maximum), workers=int(workers),
        )
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    partner_slots = int(result['selected_slots'])
    print(
        f'track crossings: {partner_slots} partner slots selected '
        f'(max {int(maximum)} per primary; native fused cache path)'
    )
    return (
        torch.from_numpy(np.asarray(result['partners'])).to(device=device),
        torch.from_numpy(np.asarray(result['self_local'])).to(device=device),
        torch.from_numpy(np.asarray(result['partner_local'])).to(device=device),
    )


def _build_crossing_partner_table(
        tracks, families, maximum, device, flat_points=None, offsets=None):
    """Compatibility path which discovers crossings and selects a dense table."""
    if maximum <= 0:
        return None
    csr = _build_crossing_partner_csr(
        tracks, families, flat_points=flat_points, offsets=offsets)
    return _materialize_crossing_partner_table(csr, maximum, device)


def _validate_crossing_partner_csr(csr, require_sorted_source_ids=False):
    required = {
        'source_ids', 'offsets', 'partners', 'self_local', 'partner_local',
        'positions', 'clearances',
    }
    missing = required - set(csr)
    if missing:
        raise ValueError(f'crossing cache is missing arrays: {sorted(missing)}')
    source_ids = np.asarray(csr['source_ids'])
    offsets = np.asarray(csr['offsets'])
    if source_ids.ndim != 1 or offsets.shape != (len(source_ids) + 1,):
        raise ValueError('crossing cache source_ids/offsets have invalid shapes')
    if (require_sorted_source_ids and len(source_ids)
            and np.any(source_ids[1:] <= source_ids[:-1])):
        raise ValueError('crossing cache source_ids must be strictly increasing')
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError('crossing cache offsets must be monotonic from zero')
    record_count = int(offsets[-1])
    for name in required - {'source_ids', 'offsets'}:
        if np.asarray(csr[name]).shape != (record_count,):
            raise ValueError(f'crossing cache {name} has an invalid shape')
    partners = np.asarray(csr['partners'])
    if (record_count and
            (np.any(partners < 0) or np.any(partners >= len(source_ids)))):
        raise ValueError('crossing cache contains an out-of-range partner row')
    if (np.any(np.asarray(csr['self_local']) < 0)
            or np.any(np.asarray(csr['partner_local']) < 0)):
        raise ValueError('crossing cache contains a negative local index')
    if (not np.all(np.isfinite(csr['positions']))
            or not np.all(np.isfinite(csr['clearances']))):
        raise ValueError('crossing cache contains non-finite geometry')


def write_track_crossing_cache(
        path, csr, destination=None, source_signature=None,
        z_lo=None, z_hi=None):
    """Atomically write a versioned exact-crossing CSR sidecar."""
    _validate_crossing_partner_csr(csr, require_sorted_source_ids=True)
    destination = (track_crossing_cache_path(path) if destination is None
                   else Path(destination))
    signature = (_tracks_db_signature(path) if source_signature is None
                 else source_signature)
    metadata = {
        'version': TRACK_CROSSING_CACHE_VERSION,
        'db_signature': [list(item) for item in signature],
        'angle_degrees': 30.0,
        'tangent_radius_voxels': 12.0,
        'cluster_local_index_radius': 4,
        'z_range': [z_lo, z_hi],
    }
    temporary = destination.with_name(
        destination.name + f'.tmp-{os.getpid()}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open('wb') as stream:
            np.savez(
                stream,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                source_ids=np.asarray(csr['source_ids'], dtype=np.uint64),
                offsets=np.asarray(csr['offsets'], dtype=np.int64),
                partners=np.asarray(csr['partners'], dtype=np.int32),
                self_local=np.asarray(csr['self_local'], dtype=np.int32),
                partner_local=np.asarray(csr['partner_local'], dtype=np.int32),
                positions=np.asarray(csr['positions'], dtype=np.float64),
                clearances=np.asarray(csr['clearances'], dtype=np.float64),
            )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def load_track_crossing_cache(path, warn=True, expected_z_range=None):
    """Load a valid adjacent CSR sidecar, or return ``None`` on a safe miss."""
    try:
        cache_path = track_crossing_cache_path(path)
    except (FileNotFoundError, OSError) as error:
        if warn:
            print(f'WARNING: cannot resolve tracks DBM for crossing cache: {error}')
        return None
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored['metadata'].item()))
            if metadata.get('version') != TRACK_CROSSING_CACHE_VERSION:
                raise ValueError(
                    f"unsupported version {metadata.get('version')!r}")
            expected_signature = [list(item) for item in _tracks_db_signature(path)]
            if metadata.get('db_signature') != expected_signature:
                raise ValueError('tracks DBM has changed since the cache was built')
            if (expected_z_range is not None
                    and metadata.get('z_range', [None, None])
                    != list(expected_z_range)):
                raise ValueError(
                    'crossing cache was built for a different z range')
            csr = {
                name: stored[name]
                for name in (
                    'source_ids', 'offsets', 'partners', 'self_local',
                    'partner_local', 'positions', 'clearances')
            }
        _validate_crossing_partner_csr(csr, require_sorted_source_ids=True)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if warn:
            print(f'WARNING: ignoring invalid track crossing cache '
                  f'{cache_path}: {error}')
        return None
    print(
        f'loaded track crossing cache {cache_path}: '
        f'{len(csr["source_ids"])} tracks, {len(csr["partners"])} directed records'
    )
    return csr


def _restrict_crossing_partner_csr(csr, source_ids):
    """Remap a whole-DB crossing graph onto the fit's surviving track rows."""
    _validate_crossing_partner_csr(csr, require_sorted_source_ids=True)
    source_ids = np.asarray(source_ids, dtype=np.uint64)
    cached_source_ids = np.asarray(csr['source_ids'], dtype=np.uint64)
    rows = np.searchsorted(cached_source_ids, source_ids)
    valid_rows = rows < len(cached_source_ids)
    if not np.all(valid_rows):
        raise ValueError('crossing cache does not contain every selected track')
    if not np.array_equal(cached_source_ids[rows], source_ids):
        raise ValueError('crossing cache does not match selected track identities')

    global_to_local = np.full(len(cached_source_ids), -1, dtype=np.int64)
    global_to_local[rows] = np.arange(len(rows), dtype=np.int64)
    old_offsets = np.asarray(csr['offsets'], dtype=np.int64)
    counts = old_offsets[rows + 1] - old_offsets[rows]
    expanded_rows = np.repeat(rows, counts)
    expanded_local_rows = np.repeat(np.arange(len(rows), dtype=np.int64), counts)
    expanded_starts = np.repeat(old_offsets[rows], counts)
    local_starts = np.repeat(np.cumsum(np.r_[0, counts[:-1]]), counts)
    edge_indices = expanded_starts + (
        np.arange(int(counts.sum()), dtype=np.int64) - local_starts)
    partner_rows = np.asarray(csr['partners'], dtype=np.int64)[edge_indices]
    remapped_partners = global_to_local[partner_rows]
    keep = remapped_partners >= 0
    kept_local_rows = expanded_local_rows[keep]
    new_counts = np.bincount(kept_local_rows, minlength=len(rows))
    new_offsets = np.empty(len(rows) + 1, dtype=np.int64)
    new_offsets[0] = 0
    np.cumsum(new_counts, out=new_offsets[1:])
    kept_edges = edge_indices[keep]
    result = {
        'source_ids': source_ids.copy(),
        'offsets': new_offsets,
        'partners': remapped_partners[keep].astype(np.int32, copy=False),
        'self_local': np.asarray(csr['self_local'])[kept_edges].copy(),
        'partner_local': np.asarray(csr['partner_local'])[kept_edges].copy(),
        'positions': np.asarray(csr['positions'])[kept_edges].copy(),
        'clearances': np.asarray(csr['clearances'])[kept_edges].copy(),
    }
    _validate_crossing_partner_csr(result)
    return result


def _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding):
    # Compatibility wrapper for callers/tests that only need the unwrapped values.
    return unwrap_shifted_radii(theta, shifted_radii, dr_per_winding)[0]


def _aggregate_dt_track_losses(track_losses, across_p, active_mask=None):
    if active_mask is not None:
        track_losses = track_losses[active_mask]
    if track_losses.numel() == 0:
        return torch.zeros([], device=track_losses.device)
    return ((track_losses ** across_p).sum() / track_losses.numel()) ** (1 / across_p)


def _progressive_dt_active_mask(snapped_winding, dr_per_winding, dt_max_winding):
    if dt_max_winding is None:
        return None
    winding_idx = (snapped_winding / dr_per_winding).detach()
    return winding_idx <= dt_max_winding


def _build_track_flat_bundle(tracks, device):
    valid_track_indices = [i for i, track in enumerate(tracks) if len(track) >= 2]
    if not valid_track_indices:
        return None, torch.zeros(0, dtype=torch.int64), valid_track_indices

    pairs = [
        (
            np.asarray(tracks[i], dtype=np.float32),
            np.zeros(len(tracks[i]), dtype=np.float32),
        )
        for i in valid_track_indices
    ]
    lengths_np = np.fromiter((len(z) for z, _ in pairs), dtype=np.int64, count=len(pairs))
    starts_np = np.empty(len(pairs) + 1, dtype=np.int64)
    starts_np[0] = 0
    np.cumsum(lengths_np, out=starts_np[1:])
    total = int(starts_np[-1])
    if total == 0:
        return None, torch.from_numpy(lengths_np), valid_track_indices

    zyxs_flat = np.concatenate([z for z, _ in pairs], axis=0).astype(np.float32, copy=False)
    windings_flat = np.concatenate([w for _, w in pairs], axis=0).astype(np.float32, copy=False)
    strip_id_np = np.repeat(np.arange(len(pairs), dtype=np.int64), lengths_np)
    flat = {
        'zyxs': torch.from_numpy(zyxs_flat).to(device=device),
        'windings': torch.from_numpy(windings_flat).to(device=device),
        'strip_id': torch.from_numpy(strip_id_np).to(device=device),
        'starts': torch.from_numpy(starts_np).to(device=device),
        'lengths': torch.from_numpy(lengths_np).to(device=device),
        'lengths_cpu': torch.from_numpy(lengths_np),
        'total': total,
    }
    return flat, flat['lengths_cpu'], valid_track_indices


def _build_track_spiral_context(slice_to_spiral_transform, dr_per_winding, flat, num_tracks, metrics_config):
    spiral_tolerance = dr_per_winding.detach() * metrics_config['satisfaction_radius_tolerance']
    scan_tolerance = metrics_config['satisfaction_distance_tolerance']
    dr = dr_per_winding.detach()
    device = dr_per_winding.device

    if flat is None or flat['total'] == 0:
        lengths_cpu = flat['lengths_cpu'] if flat is not None else torch.zeros(num_tracks, dtype=torch.int64)
        return None, lengths_cpu, num_tracks

    chunk = 65536

    def transform_in_chunks(zyxs, fn):
        if zyxs.shape[0] <= chunk:
            return fn(zyxs)
        pieces = []
        for st in range(0, zyxs.shape[0], chunk):
            pieces.append(fn(zyxs[st:st + chunk]))
        return torch.cat(pieces, dim=0)

    zyxs = flat['zyxs']
    windings = flat['windings']
    track_id = flat['strip_id']
    starts = flat['starts']
    lengths = flat['lengths']
    lengths_cpu = flat['lengths_cpu']
    total = flat['total']

    with torch.no_grad():
        spiral_zyxs = transform_in_chunks(zyxs, slice_to_spiral_transform)
        theta, _, shifted_radii = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)

        if total > 1:
            same_track = track_id[1:] == track_id[:-1]
            step_adj = get_theta_crossing_step_adjustments(theta, dr)
            step_adj = torch.where(same_track, step_adj, torch.zeros_like(step_adj))
            cumsum_inner = torch.cumsum(step_adj, dim=0)
            cumsum_flat = torch.cat([
                torch.zeros(1, device=device, dtype=cumsum_inner.dtype),
                cumsum_inner,
            ], dim=0)
            adjustments = cumsum_flat - cumsum_flat[starts[:-1][track_id]]
        else:
            adjustments = torch.zeros_like(shifted_radii)
        unwrapped_shifted = shifted_radii + adjustments
        normalised_radii = unwrapped_shifted - windings * dr

    return {
        'spiral_tolerance': spiral_tolerance,
        'scan_tolerance': scan_tolerance,
        'dr': dr,
        'device': device,
        'num_tracks': num_tracks,
        'slice_to_spiral_transform': slice_to_spiral_transform,
        'transform_in_chunks': transform_in_chunks,
        'zyxs': zyxs,
        'windings': windings,
        'track_id': track_id,
        'lengths_cpu': lengths_cpu,
        'spiral_zyxs': spiral_zyxs,
        'theta': theta,
        'adjustments': adjustments,
        'unwrapped_shifted': unwrapped_shifted,
        'normalised_radii': normalised_radii,
    }, lengths_cpu, num_tracks


def _mode_winding_per_track(track_id, winding_idx_per_point, num_tracks, device):
    mode_winding_per_track = torch.zeros(num_tracks, dtype=torch.int64, device=device)
    if winding_idx_per_point.numel() == 0:
        return mode_winding_per_track

    w_min = winding_idx_per_point.min()
    w_max = winding_idx_per_point.max()
    w_span = (w_max - w_min + 1).to(torch.int64)
    composite = track_id.to(torch.int64) * w_span + (winding_idx_per_point - w_min).to(torch.int64)
    sorted_comp, _ = torch.sort(composite)
    unique_comp, counts = torch.unique_consecutive(sorted_comp, return_counts=True)
    u_track = unique_comp // w_span
    u_widx = (unique_comp % w_span) + w_min

    counts_max = counts.max().to(torch.int64)
    widx_min = u_widx.min().to(torch.int64)
    widx_max = u_widx.max().to(torch.int64)
    widx_span = (widx_max - widx_min + 1).to(torch.int64)
    key = (
        u_track * ((counts_max + 1) * widx_span)
        + (counts_max - counts.to(torch.int64)) * widx_span
        + (u_widx.to(torch.int64) - widx_min)
    )
    order = torch.argsort(key)
    sorted_track = u_track[order]
    sorted_widx = u_widx[order]
    new_track = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        sorted_track[1:] != sorted_track[:-1],
    ])
    first_idx = torch.nonzero(new_track, as_tuple=False).squeeze(-1)
    mode_winding_per_track[sorted_track[first_idx]] = sorted_widx[first_idx].to(torch.int64)
    return mode_winding_per_track


def _track_satisfaction_from_target(ctx, target_normalised_per_track):
    dr = ctx['dr']
    device = ctx['device']
    num_tracks = ctx['num_tracks']
    track_id = ctx['track_id']
    windings = ctx['windings']
    theta = ctx['theta']
    adjustments = ctx['adjustments']
    unwrapped_shifted = ctx['unwrapped_shifted']
    spiral_zyxs = ctx['spiral_zyxs']
    zyxs = ctx['zyxs']
    lengths_cpu = ctx['lengths_cpu']
    spiral_tolerance = ctx['spiral_tolerance']
    scan_tolerance = ctx['scan_tolerance']
    transform_in_chunks = ctx['transform_in_chunks']
    slice_to_spiral_transform = ctx['slice_to_spiral_transform']

    with torch.no_grad():
        target_normalised = target_normalised_per_track[track_id]
        target_shifted = target_normalised + windings * dr
        spiral_in_band = (unwrapped_shifted - target_shifted).abs() <= spiral_tolerance

        target_radii = radius_from_unwrapped_shifted(
            theta, target_shifted, adjustments, dr,
        )
        target_spiral_zyxs = torch.stack([
            spiral_zyxs[..., 0],
            torch.sin(theta) * target_radii,
            torch.cos(theta) * target_radii,
        ], dim=-1)
        target_scroll_zyxs = transform_in_chunks(target_spiral_zyxs, slice_to_spiral_transform.inv)
        scan_distances = torch.linalg.norm(target_scroll_zyxs - zyxs, dim=-1)
        scan_in_band = scan_distances <= scan_tolerance

        satisfied = spiral_in_band & scan_in_band
        satisfied_counts_dev = torch.zeros(num_tracks, dtype=torch.int64, device=device)
        satisfied_counts_dev.scatter_add_(0, track_id, satisfied.to(torch.int64))
        satisfied_counts = satisfied_counts_dev.cpu()
        per_point_satisfaction = list(torch.split(satisfied.cpu(), lengths_cpu.tolist()))

    return satisfied_counts, per_point_satisfaction


def get_track_satisfied_counts(slice_to_spiral_transform, dr_per_winding, tracks, metrics_config):
    device = dr_per_winding.device
    if not tracks:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty, empty, [], empty

    flat, lengths_cpu, valid_track_indices = _build_track_flat_bundle(tracks, device)
    if not valid_track_indices:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty, empty, [], empty

    num_tracks = len(valid_track_indices)
    ctx, lengths_cpu, num_tracks = _build_track_spiral_context(
        slice_to_spiral_transform, dr_per_winding, flat, num_tracks, metrics_config,
    )
    valid_track_indices_t = torch.tensor(valid_track_indices, dtype=torch.int64)
    if ctx is None:
        per_point = [torch.zeros([int(n.item())], dtype=torch.bool) for n in lengths_cpu]
        return valid_track_indices_t, torch.zeros(num_tracks, dtype=torch.int64), lengths_cpu.clone(), per_point, torch.zeros(num_tracks, dtype=torch.int64)

    dr = ctx['dr']
    track_id = ctx['track_id']
    normalised_radii = ctx['normalised_radii']

    with torch.no_grad():
        winding_idx_per_point = torch.round(normalised_radii / dr).to(torch.int64)
        mode_winding_per_track = _mode_winding_per_track(track_id, winding_idx_per_point, num_tracks, device)
        target_normalised_per_track = mode_winding_per_track.to(dr.dtype) * dr

    satisfied_counts, per_point_satisfaction = _track_satisfaction_from_target(ctx, target_normalised_per_track)
    return (
        valid_track_indices_t,
        satisfied_counts,
        lengths_cpu.clone(),
        per_point_satisfaction,
        mode_winding_per_track.cpu(),
    )


def get_track_satisfied_counts_in_chunks(slice_to_spiral_transform, dr_per_winding, tracks, metrics_config, chunk_size=500_000):
    sat_parts, tot_parts = [], []
    for start in range(0, len(tracks), chunk_size):
        chunk = tracks[start:start + chunk_size]
        _, sat, tot, _, _ = get_track_satisfied_counts(
            slice_to_spiral_transform, dr_per_winding, chunk, metrics_config,
        )
        sat_parts.append(sat)
        tot_parts.append(tot)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not sat_parts:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty
    return torch.cat(sat_parts), torch.cat(tot_parts)


def _build_anchor_kdtree(anchor_zyx):
    if anchor_zyx is None:
        return None
    if isinstance(anchor_zyx, torch.Tensor):
        anchor_np = anchor_zyx.detach().cpu().numpy()
    else:
        anchor_np = np.asarray(anchor_zyx)
    if anchor_np.shape[0] == 0:
        return None
    return cKDTree(np.ascontiguousarray(anchor_np, dtype=np.float32))


def _track_points_far_from_anchors_mask(track_zyx, anchor_tree, threshold):
    if isinstance(track_zyx, torch.Tensor):
        track_np = track_zyx.detach().cpu().numpy()
    else:
        track_np = np.asarray(track_zyx)
    track_np = np.ascontiguousarray(track_np, dtype=np.float32)
    if threshold <= 0 or anchor_tree is None:
        return np.ones(track_np.shape[0], dtype=bool)
    dist, _ = anchor_tree.query(track_np, k=1, distance_upper_bound=float(threshold), workers=-1)
    return np.isinf(dist)


def prepare_main_phase_tracks(
        tracks, anchor_scroll_zyxs, exclusion_radius, device, anchor_tree=None,
        sampling_config=None, track_families=None, track_source_ids=None,
        crossing_cache=None):
    if not tracks:
        return None
    if sampling_config is not None and 'length_bin_weights' in sampling_config:
        policy = sampling_config
    else:
        policy = validate_track_sampling_config(sampling_config or {})
    weights = policy['length_bin_weights']
    max_tortuosity = policy['max_tortuosity']
    max_crossings = policy['max_crossings']
    crossing_precompute_max = policy['crossing_precompute_max']

    input_track_count = len(tracks)
    packed_input = isinstance(tracks, PackedTrackCollection)
    working_tracks = tracks if packed_input else list(tracks)
    working_families = (
        np.asarray(track_families) if packed_input and track_families is not None
        else (list(track_families) if track_families is not None else None))
    working_source_ids = (
        np.asarray(track_source_ids, dtype=np.uint64)
        if track_source_ids is not None else None)
    if (working_source_ids is not None
            and working_source_ids.shape != (len(working_tracks),)):
        raise ValueError('track_source_ids must be parallel to tracks')
    if crossing_cache is not None and working_source_ids is None:
        raise ValueError('a crossing cache requires stable track_source_ids')
    if crossing_cache is not None and exclusion_radius > 0:
        print(
            'WARNING: track crossing cache cannot be used after point-level '
            'track exclusion; rebuilding crossings from the clipped tracks')
        crossing_cache = None
    arclengths = (working_tracks.selected_arclengths
                  if packed_input else _polyline_arclengths(working_tracks))
    if max_tortuosity is not None:
        tortuosities = (working_tracks.selected_tortuosities if packed_input
                        else _track_tortuosities(working_tracks, arclengths))
        keep = np.flatnonzero(tortuosities <= max_tortuosity)
        working_tracks = (working_tracks.subset(keep) if packed_input else
                          [working_tracks[index] for index in keep])
        if working_families is not None:
            working_families = (working_families[keep] if packed_input else
                                [working_families[index] for index in keep])
        if working_source_ids is not None:
            working_source_ids = working_source_ids[keep]
        arclengths = arclengths[keep]
        print(
            f'track tortuosity <= {max_tortuosity:g}: kept '
            f'{len(working_tracks)} / {input_track_count} tracks'
        )
        if not working_tracks:
            return None

    print('removing tracks near patches')
    if anchor_tree is None:
        anchor_tree = _build_anchor_kdtree(anchor_scroll_zyxs)

    def finish_prepared(
            flat_zyx_np, lengths_new, surviving_indices, prepared_track_list):
        offsets_new = np.empty(len(lengths_new) + 1, dtype=np.int64)
        offsets_new[0] = 0
        np.cumsum(lengths_new, out=offsets_new[1:])
        prepared = {
            # The full point cloud can be tens of millions of points, while a
            # step consumes only a sampled batch. Keep coordinates in host RAM.
            'flat_zyx_cpu': torch.from_numpy(flat_zyx_np).contiguous(),
            'offsets_cpu': torch.from_numpy(offsets_new),
            'offsets': torch.from_numpy(offsets_new).to(device=device),
            'lengths': torch.from_numpy(lengths_new).to(device=device),
            'device': torch.device(device),
            'staging': None,
            'resampled_cache': {},
            'arclengths_cpu': arclengths[surviving_indices],
            'length_bin_weights': None,
            'active_max_crossings': 0,
            'crossing_precompute_max': (
                crossing_precompute_max
                if working_families is not None or crossing_cache is not None
                else 0),
        }
        if crossing_precompute_max > 0:
            tables = None
            if crossing_cache is not None:
                eligible_source_ids = working_source_ids[surviving_indices]
                try:
                    tables = _materialize_cached_crossing_partner_table(
                        crossing_cache, eligible_source_ids,
                        crossing_precompute_max, device)
                    print(
                        f'track crossings: used cached CSR for '
                        f'{len(eligible_source_ids)} surviving tracks')
                except ValueError as error:
                    print(
                        'WARNING: track crossing cache could not be remapped; '
                        f'rebuilding exact crossings: {error}')
            if tables is None and working_families is not None:
                eligible_families = [
                    working_families[index] for index in surviving_indices]
                tables = _build_crossing_partner_table(
                    prepared_track_list, eligible_families,
                    crossing_precompute_max, device,
                    flat_points=flat_zyx_np, offsets=offsets_new)
            if tables is not None:
                (prepared['crossing_partners'],
                 prepared['crossing_self_local'],
                 prepared['crossing_partner_local']) = tables
        configure_prepared_track_sampling(prepared, {
            'track_length_bin_weights': (
                None if weights is None else weights.tolist()),
            'max_track_crossing_per_step': max_crossings,
        })
        return prepared

    # The common configuration has no exclusion radius.  The generic path
    # below creates several point-count-sized int64 arrays and stable-sorts the
    # already grouped tracks; none of that changes the result when every point
    # is kept.  Concatenate only the surviving (length >= 2) tracks directly.
    if exclusion_radius <= 0 or anchor_tree is None:
        if packed_input:
            surviving = np.flatnonzero(working_tracks.selected_lengths >= 2)
            surviving_collection = working_tracks.subset(surviving)
            print(
                f'kept {len(surviving_collection)} / {len(working_tracks)} tracks '
                '(packed)')
            if not surviving_collection:
                return None
            flat_zyx_np, offsets_new = surviving_collection.materialize()
            lengths_new = np.diff(offsets_new)
            prepared_track_list = _MemmapTrackCollection(
                flat_zyx_np, offsets_new)
            print(
                f'track radius loss: {len(surviving_collection)}/'
                f'{len(working_tracks)} tracks survive exclusion '
                f'(radius {exclusion_radius:.1f}); '
                f'{len(flat_zyx_np)} points retained'
            )
            return finish_prepared(
                flat_zyx_np, lengths_new, surviving, prepared_track_list)
        surviving = np.asarray([
            index for index, track in enumerate(working_tracks) if len(track) >= 2
        ], dtype=np.int64)
        surviving_tracks = [
            np.asarray(working_tracks[index], dtype=np.float32)
            for index in surviving
        ]
        print(f'kept {len(surviving_tracks)} / {len(working_tracks)} tracks')
        if not surviving_tracks:
            return None
        lengths_new = np.fromiter(
            (len(track) for track in surviving_tracks),
            dtype=np.int64,
            count=len(surviving_tracks),
        )
        flat_zyx_np = np.concatenate(surviving_tracks, axis=0)
        print(
            f'track radius loss: {len(surviving_tracks)}/{len(working_tracks)} tracks survive exclusion '
            f'(radius {exclusion_radius:.1f}); {int(lengths_new.sum())} points retained'
        )
        return finish_prepared(
            flat_zyx_np, lengths_new, surviving, surviving_tracks)

    if packed_input:
        flat_materialized, packed_offsets = working_tracks.materialize()
        working_tracks = _MemmapTrackCollection(
            flat_materialized, packed_offsets)

    flat_zyx_np = np.concatenate([t.astype(np.float32) for t in working_tracks], axis=0)
    track_id_np = np.concatenate([
        np.full(len(t), i, dtype=np.int64) for i, t in enumerate(working_tracks)
    ])
    keep_np = _track_points_far_from_anchors_mask(flat_zyx_np, anchor_tree, exclusion_radius)
    flat_zyx_np = flat_zyx_np[keep_np]
    track_id_np = track_id_np[keep_np]
    num_tracks_orig = len(working_tracks)
    new_lengths = np.bincount(track_id_np, minlength=num_tracks_orig)
    surviving = np.where(new_lengths >= 2)[0]
    print(f'kept {len(surviving)} / {len(working_tracks)} tracks')
    if len(surviving) == 0:
        return None
    old_to_new = -np.ones(num_tracks_orig, dtype=np.int64)
    old_to_new[surviving] = np.arange(len(surviving))
    new_id = old_to_new[track_id_np]
    keep2 = new_id >= 0
    flat_zyx_np = flat_zyx_np[keep2]
    new_id = new_id[keep2]
    sort_idx = np.argsort(new_id, kind='stable')
    flat_zyx_np = flat_zyx_np[sort_idx]
    lengths_new = new_lengths[surviving].astype(np.int64)
    prepared_track_list = list(np.split(flat_zyx_np, np.cumsum(lengths_new)[:-1]))
    print(
        f'track radius loss: {len(surviving)}/{num_tracks_orig} tracks survive exclusion '
        f'(radius {exclusion_radius:.1f}); {int(lengths_new.sum())} points retained'
    )
    return finish_prepared(
        flat_zyx_np, lengths_new, surviving, prepared_track_list)


def _build_resampled_track_bundle(prepared_tracks, min_spacing, max_spacing):
    """Resample complete tracks between mandatory polyline anchors."""
    min_spacing = float(min_spacing)
    max_spacing = float(max_spacing)
    cache_key = (min_spacing, max_spacing)
    cached = prepared_tracks['resampled_cache'].get(cache_key)
    if cached is not None:
        return cached

    flat_source = prepared_tracks['flat_zyx_cpu'].numpy()
    offsets = prepared_tracks['offsets_cpu'].numpy()
    num_tracks = len(offsets) - 1
    crossing_partners = prepared_tracks.get('crossing_partners')
    if crossing_partners is not None:
        partners_np = crossing_partners.cpu().numpy()
        self_local_np = prepared_tracks['crossing_self_local'].cpu().numpy()
        partner_local_np = prepared_tracks['crossing_partner_local'].cpu().numpy()
    else:
        partners_np = self_local_np = partner_local_np = None

    native = _load_native_track_crossings()
    if native is not None and hasattr(native, 'resample_tracks'):
        if partners_np is None:
            empty = np.empty((num_tracks, 0), dtype=np.int32)
            partners_native = self_local_native = partner_local_native = empty
        else:
            partners_native = np.asarray(partners_np, dtype=np.int32)
            self_local_native = np.asarray(self_local_np, dtype=np.int32)
            partner_local_native = np.asarray(partner_local_np, dtype=np.int32)
        result = native.resample_tracks(
            np.asarray(flat_source, dtype=np.float32),
            np.asarray(offsets, dtype=np.int64),
            partners_native, self_local_native, partner_local_native,
            minimum_spacing=min_spacing, maximum_spacing=max_spacing,
            workers=min(32, os.cpu_count() or 1),
        )
        sampled_lengths = np.asarray(result['lengths'])
        device = prepared_tracks['device']
        bundle = {
            'flat_zyx_cpu': torch.from_numpy(
                np.asarray(result['coordinates'])).contiguous(),
            'flat_source_local_cpu': torch.from_numpy(
                np.asarray(result['source_local'])).contiguous(),
            'offsets': torch.from_numpy(
                np.asarray(result['offsets'])).to(device=device),
            'lengths': torch.from_numpy(sampled_lengths).to(device=device),
        }
        if partners_np is not None:
            bundle['crossing_self_sample'] = torch.from_numpy(
                np.asarray(result['crossing_self_sample'])).to(device=device)
            bundle['crossing_partner_sample'] = torch.from_numpy(
                np.asarray(result['crossing_partner_sample'])).to(device=device)
        prepared_tracks['resampled_cache'].clear()
        prepared_tracks['resampled_cache'][cache_key] = bundle
        minimum_observed_spacing = float(result['minimum_observed_spacing'])
        maximum_observed_spacing = float(result['maximum_observed_spacing'])
        undersized_anchor_gaps = int(result['undersized_anchor_gaps'])
        observed_min_text = (
            f'{minimum_observed_spacing:.2f}'
            if math.isfinite(minimum_observed_spacing) else 'n/a')
        print(
            f'track point spacing: {int(sampled_lengths.sum())} cached points, '
            f'observed {observed_min_text}-{maximum_observed_spacing:.2f} voxels '
            f'(limits {min_spacing:g}-{max_spacing:g}; '
            f'{undersized_anchor_gaps} mandatory-anchor exceptions; native)'
        )
        return bundle

    mandatory = [set() for _ in range(num_tracks)]
    if partners_np is not None:
        for track_id, slot in zip(*np.nonzero(partners_np >= 0)):
            partner = int(partners_np[track_id, slot])
            mandatory[track_id].add(int(self_local_np[track_id, slot]))
            mandatory[partner].add(int(partner_local_np[track_id, slot]))

    sampled_parts = []
    source_index_parts = []
    sampled_lengths = np.empty(num_tracks, dtype=np.int64)
    local_maps = []
    minimum_observed_spacing = math.inf
    maximum_observed_spacing = 0.0
    undersized_anchor_gaps = 0
    for track_id in range(num_tracks):
        track = flat_source[offsets[track_id]:offsets[track_id + 1]]
        steps = np.linalg.norm(np.diff(track.astype(np.float64), axis=0), axis=1)
        cumulative = np.r_[0.0, np.cumsum(steps)]
        total = float(cumulative[-1])
        if total > 0:
            anchor_indices = np.fromiter(
                sorted({0, len(track) - 1, *mandatory[track_id]}),
                dtype=np.int64,
            )
            anchor_positions = np.unique(cumulative[anchor_indices])
            position_parts = []
            for anchor_left, anchor_right in zip(
                    anchor_positions[:-1], anchor_positions[1:]):
                span = float(anchor_right - anchor_left)
                min_intervals = max(1, int(math.ceil(span / max_spacing)))
                max_intervals = int(math.floor(span / min_spacing))
                if min_intervals <= max_intervals:
                    segment = np.linspace(
                        anchor_left, anchor_right, min_intervals + 1,
                        dtype=np.float64)
                else:
                    # An endpoint or exact crossing makes both bounds
                    # geometrically impossible. Keep the maximum hard and put
                    # the one short residual directly beside the right anchor.
                    segment = np.r_[
                        anchor_left
                        + np.arange(min_intervals, dtype=np.float64) * max_spacing,
                        anchor_right,
                    ]
                position_parts.append(segment[:-1])
            positions = np.concatenate(position_parts + [anchor_positions[-1:]])
            right = np.searchsorted(cumulative, positions, side='right')
            right = np.clip(right, 1, len(track) - 1)
            left = right - 1
            denom = cumulative[right] - cumulative[left]
            alpha = np.divide(
                positions - cumulative[left], denom,
                out=np.zeros_like(positions), where=denom > 0)
            sampled = (
                track[left] * (1.0 - alpha[:, None])
                + track[right] * alpha[:, None]
            ).astype(np.float32)
            nearest_right = (
                np.abs(cumulative[right] - positions)
                < np.abs(positions - cumulative[left]))
            source_indices = np.where(nearest_right, right, left).astype(np.int64)
            observed = np.diff(positions)
            minimum_observed_spacing = min(
                minimum_observed_spacing,
                float(observed.min(initial=math.inf)),
            )
            maximum_observed_spacing = max(
                maximum_observed_spacing,
                float(observed.max(initial=0.0)),
            )
            undersized_anchor_gaps += int(np.count_nonzero(
                observed < min_spacing - 1.e-9))
        else:
            sampled = track[:1].copy()
            positions = np.zeros(1, dtype=np.float64)
            source_indices = np.zeros(1, dtype=np.int64)

        sampled_parts.append(sampled)
        source_index_parts.append(source_indices)
        sampled_lengths[track_id] = len(sampled)
        local_maps.append({
            original_index: int(np.argmin(np.abs(
                positions - cumulative[original_index])))
            for original_index in mandatory[track_id]
        })

    sampled_offsets = np.empty(num_tracks + 1, dtype=np.int64)
    sampled_offsets[0] = 0
    np.cumsum(sampled_lengths, out=sampled_offsets[1:])
    device = prepared_tracks['device']
    bundle = {
        'flat_zyx_cpu': torch.from_numpy(np.concatenate(sampled_parts)).contiguous(),
        'flat_source_local_cpu': torch.from_numpy(
            np.concatenate(source_index_parts)).contiguous(),
        'offsets': torch.from_numpy(sampled_offsets).to(device=device),
        'lengths': torch.from_numpy(sampled_lengths).to(device=device),
    }
    if partners_np is not None:
        crossing_self_sample = np.full_like(self_local_np, -1)
        crossing_partner_sample = np.full_like(partner_local_np, -1)
        for track_id, slot in zip(*np.nonzero(partners_np >= 0)):
            partner = int(partners_np[track_id, slot])
            crossing_self_sample[track_id, slot] = local_maps[track_id][
                int(self_local_np[track_id, slot])]
            crossing_partner_sample[track_id, slot] = local_maps[partner][
                int(partner_local_np[track_id, slot])]
        bundle['crossing_self_sample'] = torch.from_numpy(
            crossing_self_sample).to(device=device)
        bundle['crossing_partner_sample'] = torch.from_numpy(
            crossing_partner_sample).to(device=device)

    # Run-scoped spacing edits can request a different density. Keep the cache
    # bounded to one whole-dataset resample rather than retaining every setting
    # tried during an interactive session.
    prepared_tracks['resampled_cache'].clear()
    prepared_tracks['resampled_cache'][cache_key] = bundle
    observed_min_text = (
        f'{minimum_observed_spacing:.2f}'
        if math.isfinite(minimum_observed_spacing) else 'n/a')
    print(
        f'track point spacing: {int(sampled_lengths.sum())} cached points, '
        f'observed {observed_min_text}-{maximum_observed_spacing:.2f} voxels '
        f'(limits {min_spacing:g}-{max_spacing:g}; '
        f'{undersized_anchor_gaps} mandatory-anchor exceptions)'
    )
    return bundle


def _draw_track_sample(
        prepared_tracks, resampled, k, target_points, max_crossings,
        generator=None):
    # Track selection and ragged gather stay vectorized.  Every selected row
    # contains the entire spacing-bounded resample of that track.
    device = prepared_tracks['device']
    num_tracks = int(prepared_tracks['lengths'].numel())
    sampling_probabilities = prepared_tracks.get('sampling_probabilities')
    if sampling_probabilities is None:
        primary_track_idx = torch.randint(
            num_tracks, (k,), device=device, generator=generator)
    else:
        primary_track_idx = torch.multinomial(
            sampling_probabilities, k, replacement=True, generator=generator)

    crossing_partners = prepared_tracks.get('crossing_partners')
    partner_group = torch.empty(0, dtype=torch.int64, device=device)
    partner_slot = torch.empty(0, dtype=torch.int64, device=device)
    partner_track_idx = torch.empty(0, dtype=torch.int64, device=device)
    if (crossing_partners is not None and crossing_partners.shape[1] > 0
            and max_crossings > 0):
        selected = crossing_partners[primary_track_idx, :max_crossings]
        valid = selected >= 0
        partner_track_idx = selected[valid].to(torch.int64)
        partner_group = torch.arange(k, device=device)[:, None].expand_as(
            selected)[valid]
        partner_slot = torch.arange(
            selected.shape[1], device=device)[None, :].expand_as(selected)[valid]

    track_idx = torch.cat([primary_track_idx, partner_track_idx])
    group_id = torch.cat([
        torch.arange(k, device=device), partner_group,
    ])
    row_slot = torch.cat([
        torch.zeros(k, dtype=torch.int64, device=device), partner_slot + 1,
    ])
    row_lengths = resampled['lengths'][track_idx]
    row_starts = torch.zeros(
        len(track_idx) + 1, dtype=torch.int64, device=device)
    torch.cumsum(row_lengths, dim=0, out=row_starts[1:])
    row_id = torch.repeat_interleave(
        torch.arange(len(track_idx), device=device), row_lengths)
    local_idx = torch.arange(int(row_starts[-1]), device=device) - row_starts[:-1][row_id]
    flat_idx = resampled['offsets'][track_idx][row_id] + local_idx

    flat_idx_cpu = flat_idx.cpu()
    sampled_cpu = resampled['flat_zyx_cpu'][flat_idx_cpu]
    if device.type == 'cuda':
        staging = prepared_tracks.get('staging')
        if staging is None or staging.shape != sampled_cpu.shape:
            staging = torch.empty(sampled_cpu.shape, dtype=torch.float32, pin_memory=True)
            prepared_tracks['staging'] = staging
        staging.copy_(sampled_cpu)
        sampled_scroll = staging.to(device=device, non_blocking=False)
    else:
        sampled_scroll = sampled_cpu.to(device=device)

    if target_points == 1:
        target_local = torch.zeros(
            [len(track_idx), 1], dtype=torch.int64, device=device)
    else:
        fractions = torch.linspace(
            0.0, 1.0, target_points, device=device)
        target_local = torch.round(
            fractions[None, :] * (row_lengths[:, None] - 1)).to(torch.int64)
    target_flat_idx = row_starts[:-1, None] + target_local
    target_source_idx_cpu = resampled['flat_source_local_cpu'][
        flat_idx_cpu[target_flat_idx.reshape(-1).cpu()]].reshape(
            len(track_idx), target_points)
    target_source_idx = target_source_idx_cpu.to(device=device)

    primary_cross_flat = torch.empty(0, dtype=torch.int64, device=device)
    partner_cross_flat = torch.empty(0, dtype=torch.int64, device=device)
    partner_rows = torch.empty(0, dtype=torch.int64, device=device)
    if partner_track_idx.numel() > 0:
        self_cross_local = resampled['crossing_self_sample'][
            primary_track_idx[partner_group], partner_slot]
        partner_cross_local = resampled['crossing_partner_sample'][
            primary_track_idx[partner_group], partner_slot]
        partner_rows = torch.arange(
            k, len(track_idx), dtype=torch.int64, device=device)
        primary_cross_flat = row_starts[partner_group] + self_cross_local
        partner_cross_flat = row_starts[partner_rows] + partner_cross_local

    return {
        'track_idx': track_idx,
        'sampled_scroll': sampled_scroll,
        'row_id': row_id,
        'row_starts': row_starts,
        'row_lengths': row_lengths,
        'group_id': group_id,
        'row_slot': row_slot,
        'num_groups': k,
        'group_width': max_crossings + 1,
        'target_flat_idx': target_flat_idx,
        'target_source_idx': target_source_idx,
        'primary_cross_flat': primary_cross_flat,
        'partner_cross_flat': partner_cross_flat,
        'partner_rows': partner_rows,
    }


def _sample_prepared_track_points(
        prepared_tracks, num_tracks_per_step, num_points_per_track,
        min_sample_spacing=20.0, max_sample_spacing=60.0,
        max_crossings=None):
    device = prepared_tracks['device']
    num_tracks = int(prepared_tracks['lengths'].numel())
    if num_tracks == 0 or num_tracks_per_step <= 0 or num_points_per_track <= 0:
        return None
    k = min(int(num_tracks_per_step), num_tracks)
    prepared_crossings = int(prepared_tracks.get('crossing_precompute_max', 0))
    if max_crossings is None:
        max_crossings = prepared_tracks.get('active_max_crossings', 0)
    max_crossings = int(max_crossings)
    if max_crossings < 0 or max_crossings > prepared_crossings:
        raise ValueError(
            f'max crossings must lie in [0, {prepared_crossings}] for this session')
    resampled = _build_resampled_track_bundle(
        prepared_tracks, min_sample_spacing, max_sample_spacing)

    if prefetch.prefetch_enabled() and device.type == 'cuda':
        pf = prefetch.get_prefetcher()
        generator = pf.torch_rng('tracks', device)
        return pf.pop_or_run(
            ('tracks', k, num_points_per_track, float(min_sample_spacing),
             float(max_sample_spacing), max_crossings),
            lambda: _draw_track_sample(
                prepared_tracks, resampled, k, num_points_per_track,
                max_crossings, generator),
        )
    return _draw_track_sample(
        prepared_tracks, resampled, k, num_points_per_track, max_crossings)


@geom_utils.maybe_compile
def _same_radius_loss_tensor(
    shifted_radii,
    dr_per_winding,
    use_median_target,
    radius_loss_margin,
    within_p,
):
    radius_hinge_margin = dr_per_winding.detach() * radius_loss_margin
    if use_median_target:
        radius_target_per_track = shifted_radii.median(dim=-1, keepdim=True).values
    else:
        radius_target_per_track = shifted_radii.mean(dim=-1, keepdim=True)
    deviations = (shifted_radii - radius_target_per_track).abs()
    hinged = F.relu(deviations - radius_hinge_margin)
    if within_p == 1.0:
        return hinged.mean()
    per_track = ((hinged + 1.e-5) ** within_p).mean(dim=-1) ** (1.0 / within_p)
    return per_track.mean()


def _same_radius_loss_for_shifted_radii(shifted_radii, dr_per_winding, cfg):
    target = cfg['track_radius_target']
    if target not in ('mean', 'median'):
        raise ValueError(f"track_radius_target must be 'mean' or 'median', got {target!r}")
    return _same_radius_loss_tensor(
        shifted_radii,
        dr_per_winding,
        target == 'median',
        cfg['track_radius_loss_margin'],
        cfg['track_radius_within_norm_p'],
    )


def _unwrap_ragged_track_radii(
        theta, shifted_radii, dr_per_winding, row_id, row_starts):
    """Unwrap a flat, row-contiguous batch without padding its tracks."""
    if shifted_radii.numel() <= 1:
        return shifted_radii, torch.zeros_like(shifted_radii)
    same_row = row_id[1:] == row_id[:-1]
    step_adjustments = get_theta_crossing_step_adjustments(
        theta, dr_per_winding)
    step_adjustments = torch.where(
        same_row, step_adjustments, torch.zeros_like(step_adjustments))
    cumulative_inner = torch.cumsum(step_adjustments, dim=0)
    cumulative = torch.cat([
        torch.zeros(1, device=theta.device, dtype=theta.dtype),
        cumulative_inner,
    ])
    adjustments = cumulative - cumulative[row_starts[:-1][row_id]]
    return shifted_radii + adjustments, adjustments


def _grouped_radius_targets(
        target_values, group_id, row_slot, num_groups, group_width,
        use_median):
    """One robust target for every primary track and all crossing partners."""
    points_per_row = target_values.shape[1]
    values = torch.full(
        [num_groups, group_width, points_per_row], torch.inf,
        device=target_values.device, dtype=target_values.dtype)
    values[group_id, row_slot] = target_values
    valid = torch.isfinite(values)
    if not use_median:
        return torch.where(valid, values, torch.zeros_like(values)).sum(
            dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)

    flattened = values.flatten(1).sort(dim=-1).values
    counts = valid.sum(dim=(1, 2))
    median_index = torch.div(counts - 1, 2, rounding_mode='floor')
    return flattened.gather(1, median_index[:, None]).squeeze(1)


def _grouped_same_radius_loss(
        shifted_radii, target_values, row_id, group_id, row_slot,
        num_groups, group_width, dr_per_winding, cfg):
    target_mode = cfg['track_radius_target']
    if target_mode not in ('mean', 'median'):
        raise ValueError(
            f"track_radius_target must be 'mean' or 'median', got {target_mode!r}")
    targets = _grouped_radius_targets(
        target_values, group_id, row_slot, num_groups, group_width,
        target_mode == 'median')
    flat_group_id = group_id[row_id]
    deviations = (shifted_radii - targets[flat_group_id]).abs()
    hinged = F.relu(
        deviations
        - dr_per_winding.detach() * cfg['track_radius_loss_margin'])
    within_p = cfg['track_radius_within_norm_p']
    if within_p == 1.0:
        per_point = hinged
    else:
        per_point = (hinged + 1.e-5) ** within_p
    sums = torch.zeros(
        num_groups, device=shifted_radii.device, dtype=shifted_radii.dtype)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, flat_group_id, per_point)
    counts.scatter_add_(0, flat_group_id, torch.ones_like(per_point))
    per_group = sums / counts.clamp(min=1)
    if within_p != 1.0:
        per_group = per_group ** (1.0 / within_p)
    return per_group.mean(), targets, hinged


def iter_track_losses(slice_to_spiral_transform, dr_per_winding, prepared_tracks, cfg, compute_dt=True, dt_max_winding=None, dt_target_cache=None):
    """Yield radius then DT losses so the caller can backward them separately.

    The DT target is detached before its inverse transform, so its graph does
    not depend on the radius-loss forward graph.  Yielding at that boundary
    prevents both large transform graphs from being resident together.
    """
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if prepared_tracks is None:
        yield 'track_radius', zero
        yield 'track_dt', zero
        return
    sample = _sample_prepared_track_points(
        prepared_tracks,
        cfg['track_num_per_step'],
        cfg['track_num_points_per_step'],
        cfg.get('track_min_sample_spacing', 20.0),
        cfg.get('track_max_sample_spacing', 60.0),
        cfg.get('max_track_crossing_per_step', 0),
    )
    if sample is None:
        yield 'track_radius', zero
        yield 'track_dt', zero
        return
    track_idx = sample['track_idx']
    sampled_scroll = sample['sampled_scroll']
    row_id = sample['row_id']
    row_starts = sample['row_starts']
    group_id = sample['group_id']
    num_groups = sample['num_groups']
    sampled_spiral = slice_to_spiral_transform(sampled_scroll)
    theta, _, shifted_radii = get_theta_and_radii(sampled_spiral[..., 1:], dr_per_winding)
    shifted_radii, crossing_adjustments = _unwrap_ragged_track_radii(
        theta, shifted_radii, dr_per_winding, row_id, row_starts,
    )

    # Every appended partner is put into its primary track's unwrap frame at
    # their exact shared voxel.  A single group target can then constrain all
    # points on both complete tracks to one winding.
    if sample['partner_rows'].numel() > 0:
        row_alignment = torch.zeros(
            len(track_idx), device=device, dtype=shifted_radii.dtype)
        row_alignment[sample['partner_rows']] = (
            shifted_radii[sample['primary_cross_flat']]
            - shifted_radii[sample['partner_cross_flat']]
        ).detach()
        flat_alignment = row_alignment[row_id]
        shifted_radii = shifted_radii + flat_alignment
        crossing_adjustments = crossing_adjustments + flat_alignment

    dt_hinge_margin = dr_per_winding.detach() * cfg['track_dt_loss_margin']
    target_values = shifted_radii[sample['target_flat_idx']]
    radius_loss, group_radius_targets, diagnostic_radius = \
        _grouped_same_radius_loss(
            shifted_radii, target_values, row_id, group_id,
            sample['row_slot'], num_groups, sample['group_width'],
            dr_per_winding, cfg)
    if diagnostics_enabled():
        diagnostic_radius_target = group_radius_targets[group_id[row_id]]
        diagnostic_target_radii = radius_from_unwrapped_shifted(
            theta, diagnostic_radius_target, crossing_adjustments,
            dr_per_winding,
        )
        diagnostic_target_spiral = torch.stack([
            sampled_spiral[..., 0],
            torch.sin(theta) * diagnostic_target_radii,
            torch.cos(theta) * diagnostic_target_radii,
        ], dim=-1).detach()
        record_loss_samples(
            'track_radius', sampled_spiral, diagnostic_radius,
            display_spiral_zyx=diagnostic_target_spiral,
        )

    if not compute_dt:
        yield 'track_radius', radius_loss
        del radius_loss, sampled_spiral, theta, shifted_radii, crossing_adjustments
        yield 'track_dt', zero
        return

    if dt_target_cache is None:
        target_shifted_per_group = snap_dt_target(
            group_radius_targets[:, None], dr_per_winding)
    else:
        # Primary rows occupy [0, num_groups). Their cached whole-track target
        # becomes the target of every crossing partner in the aligned group.
        target_shifted_per_group = strip_dt_target_in_sample_frame(
            target_values[:num_groups],
            sample['target_source_idx'][:num_groups],
            theta[sample['target_flat_idx'][:num_groups]],
            crossing_adjustments[sample['target_flat_idx'][:num_groups]],
            dr_per_winding, dt_target_cache, track_idx[:num_groups],
        )
    target_shifted_radii = target_shifted_per_group[group_id[row_id], 0]
    target_radii = radius_from_unwrapped_shifted(
        theta, target_shifted_radii, crossing_adjustments, dr_per_winding,
    )
    target_spiral_zyxs = torch.stack([
        sampled_spiral[..., 0],
        torch.sin(theta) * target_radii,
        torch.cos(theta) * target_radii,
    ], dim=-1).detach()
    active_mask = _progressive_dt_active_mask(
        target_shifted_per_group.squeeze(-1), dr_per_winding,
        dt_max_winding)

    yield 'track_radius', radius_loss
    # The caller has now released the radius graph.  Keep only detached DT
    # inputs before constructing the inverse-transform graph.
    del radius_loss, sampled_spiral, theta, shifted_radii, crossing_adjustments
    del target_radii, target_shifted_radii
    target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs)

    within_p = cfg['track_dt_within_track_norm_p']
    across_p = cfg['track_dt_norm_p']
    point_distances = torch.linalg.norm(sampled_scroll - target_scroll_zyxs, dim=-1)
    point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5
    flat_group_id = group_id[row_id]
    sums = torch.zeros(
        num_groups, device=device, dtype=point_distances.dtype)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, flat_group_id, point_distances ** within_p)
    counts.scatter_add_(0, flat_group_id, torch.ones_like(point_distances))
    group_losses = (sums / counts.clamp(min=1)) ** (1 / within_p)
    dt_loss = _aggregate_dt_track_losses(
        group_losses, across_p, active_mask)
    record_loss_samples(
        'track_dt', target_spiral_zyxs, point_distances,
        active_mask[flat_group_id] if active_mask is not None else None,
    )

    yield 'track_dt', dt_loss


def render_spiral_on_tracks_for_slice(
    spiral_zyx, spiral_density, dr_per_winding,
    slice_z, all_tracks, snapped_tracks,
    out_path, name_suffix,
    render_volume_scale=1,
):
    z_window = 20
    point_radius = 1
    target_ids = {id(t) for t in snapped_tracks}

    def track_colour(track, is_target):
        hue = ((id(track) * 2654435761) & 0xFFFFFFFF) / 2 ** 32
        sat, val = (0.9, 1.0) if is_target else (0.35, 0.75)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return (int(r * 255), int(g * 255), int(b * 255))

    _, _, shifted_radius = get_theta_and_radii(spiral_zyx[..., 1:], dr_per_winding)
    winding_idx = (shifted_radius / dr_per_winding).round().to(torch.int64).clamp_min(0)
    num_winding_hues = 6
    hue_min, hue_max = 1.5 / 6, 5.25 / 6
    hue_fraction = hue_min + (winding_idx % num_winding_hues).to(torch.float32) / num_winding_hues * (hue_max - hue_min)
    hue = hue_fraction * 2 * np.pi
    hsv = torch.stack([hue, torch.full_like(hue, 0.5), torch.ones_like(hue)])
    spiral_colours = kornia.color.hsv_to_rgb(hsv).permute(1, 2, 0) * 255
    canvas = (spiral_colours * spiral_density[..., None]).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)

    for is_target in (False, True):
        for track in all_tracks:
            if (id(track) in target_ids) != is_target:
                continue
            zs = track[:, 0]
            in_slab = np.abs(zs.astype(np.float32) - float(slice_z)) <= z_window
            if not in_slab.any():
                continue
            colour = track_colour(track, is_target)
            for idx in np.nonzero(in_slab)[0]:
                y = float(track[idx, 1]) / render_volume_scale
                x = float(track[idx, 2]) / render_volume_scale
                draw.ellipse(
                    [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                    fill=colour,
                )
    image.save(f'{out_path}/spiral_on_tracks_s{int(slice_z):05}_{name_suffix}.png', compress_level=3)
