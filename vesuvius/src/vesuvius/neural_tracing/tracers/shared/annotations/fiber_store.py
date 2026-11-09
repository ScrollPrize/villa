"""Fibers in a VC object store (``vc_object_store``): worker-local training access, raw polylines, a content
digest, and writing a derived store.

A fiber is identified by its object UUID. Its geometry is the ``line_points`` array, stored in the frame its
``meta.coordinates`` declares ({volume_uuid, scale_factor}: stored = volume voxel * scale_factor); everything here
returns volume voxels of one named volume (an open-data volume id, ``open_data_volume``). Fibers not bound to that
volume are not listed; a fiber without line points is skipped.
"""
from collections import OrderedDict
from collections.abc import Sequence
import getpass
import hashlib
import os
from pathlib import Path
import tempfile
import uuid as uuidlib

import numpy as np
from vc_object_store import Fiber, ObjectStore, CollectionReader, import_initial_records, geometry_scale_to_volume
from vesuvius.neural_tracing.tracers.shared.annotations.repair import ANNOTATION_REPAIR, repair_kinks
from vesuvius.neural_tracing.tracers.shared.annotations.traced import TracedFiber
from vesuvius.neural_tracing.tracers.shared.annotations.splits import heldout_ids
from vesuvius.neural_tracing.tracers.shared.geometry import arclength, interp_at, dense_line

PAGE = 8192
WRITE_BATCH = 256


def open_store(path):
    """A read-only handle; a context manager. Not shareable across processes: open one per process."""
    return ObjectStore.open(str(path), read_only=True, reader_pool_size=1)


def fiber_volume(path):
    """The one volume every fiber is bound to, across the complete collection."""
    with CollectionReader(str(path)) as store:
        records = store.objects('fiber')
    if not records:
        raise ValueError(f'No fibers in {path}')
    volumes = {(r['meta'].get('coordinates') or {}).get('volume_uuid') for r in records}
    if None in volumes or len(volumes) != 1:
        raise ValueError(f'Fibers in {path} must be bound to one volume')
    return volumes.pop()


def fiber_catalog(path, volume):
    """UUID-ordered catalog, including shards/references; geometry is not decoded."""
    with CollectionReader(str(path)) as store:
        records = store.objects('fiber')
    rows = []
    for record in records:
        fiber = record.get('fiber') or {}
        scale = geometry_scale_to_volume(record['meta'], volume)
        if scale is None or fiber.get('line_point_count', 0) < 2:
            continue
        bbox = fiber.get('bbox')
        rows.append(dict(id=record['uuid'], name=record['name'], family=fiber.get('family', ''),
                         length=fiber['length']*scale, point_count=fiber['line_point_count'],
                         bbox=None if bbox is None else [v*scale for v in bbox], scale=scale, meta=record['meta']))
    return rows


def iter_polylines(path, fiber_ids=None, *, volume=None):
    """Raw UUID-ordered line points in volume voxels, without repair or resampling."""
    volume = volume or fiber_volume(path)
    selected = None if fiber_ids is None else sorted(set(map(str, fiber_ids)))
    with CollectionReader(str(path)) as store:
        if selected is None:
            records = [r for r in store.objects('fiber')
                       if geometry_scale_to_volume(r['meta'], volume) is not None
                       and (r.get('fiber') or {}).get('line_point_count', 0) >= 2]
        else:
            records = [store.object(fid) for fid in selected]
            if any(r is None or r['kind'] != 'fiber' for r in records):
                raise ValueError('Unknown requested fiber')
        # Bounded native geometry batches, not one Python/native call per sample.
        for begin in range(0, len(records), PAGE):
            batch = records[begin:begin+PAGE]
            # Same-volume scalar conversion preserves the existing fast path;
            # registered frames use the shared native double-precision affine.
            mapped=any(geometry_scale_to_volume(r['meta'],volume) is None for r in batch)
            arrays = store.geometry_arrays([r['uuid'] for r in batch], 'line_points',volume if mapped else None)
            for record, array in zip(batch, arrays):
                scale = 1. if mapped else geometry_scale_to_volume(record['meta'], volume)
                xyz = np.frombuffer(array['data'], dtype=array['dtype']).reshape(array['shape']).astype(np.float64)
                if scale is None or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2:
                    raise ValueError(f'Fiber {record["uuid"]} is unbound or has no line')
                xyz *= scale
                fiber = record.get('fiber') or {}
                yield (dict(id=record['uuid'], name=record['name'], family=fiber.get('family', ''),
                                    length=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                                    point_count=len(xyz), meta=record['meta']), xyz)



def read_polylines(path, fiber_ids=None, *, volume=None):
    """Materialize raw curves; use iter_polylines for bounded streaming."""
    return list(iter_polylines(path, fiber_ids, volume=volume))


class FiberBlockIndex:
    """Ephemeral double-precision 256-segment bounds, including slim stores.

    The same overlapping block bounds used by the former AFV spatial seed
    query; no persistent spatial tables are required. The KD tree is only a
    conservative broad phase; inclusive XYZ-box intersection is checked exactly.
    """
    def __init__(self, path, ids, *, volume=None):
        from scipy.spatial import cKDTree
        owners, bounds = [], []
        for row, xyz in iter_polylines(path, ids, volume=volume):
            for first in range(0, len(xyz)-1, 256):
                block = xyz[first:first+257]
                owners.append(row['id'])
                bounds.append([block.min(0), block.max(0)])
        self.ids = np.asarray(owners, dtype='U36')
        self.bounds = np.asarray(bounds, dtype=np.float64).reshape(-1, 2, 3)
        centers = self.bounds.mean(1)
        self.radius = np.max((self.bounds[:,1]-self.bounds[:,0])/2) if len(bounds) else 0.
        self.tree = cKDTree(centers)

    def query(self, samples, radius):
        selected = set()
        for point, candidates in zip(samples, self.tree.query_ball_point(samples, float(np.max(radius))+self.radius, p=np.inf)):
            indices = np.asarray(candidates, dtype=np.int64)
            bounds = self.bounds[indices]
            hit = np.all(bounds[:,1] >= point-radius, axis=1) & np.all(bounds[:,0] <= point+radius, axis=1)
            selected.update(self.ids[indices[hit]])
        return selected


FIBERS_DIGEST_FORMAT = 'fibers_digest v2'


def fibers_digest(path, fiber_ids=None):
    """Content digest of a store's live fibers (all, or the given ones): SHA-256 over a format tag and each fiber's
    UUID and numbered version (with its history generation), in UUID order, read in one snapshot. A store file's
    bytes are not a stable identity (WAL). Format v2 (numbered versions) never equals a v1 digest (geometry
    revision UUIDs and record generations), so data derived under v1 is reported changed and must be regenerated."""
    selected = [] if fiber_ids is None else sorted(set(map(str, fiber_ids)))
    with CollectionReader(str(path)) as store:
        rows = store.objects('fiber') if fiber_ids is None else [store.object(fid) for fid in selected]
        if any(row is None for row in rows):
            raise ValueError('Unknown requested fiber')
    if fiber_ids is not None and sorted(r['uuid'] for r in rows) != selected:
        raise ValueError('Unknown requested fiber')
    digest = hashlib.sha256(f'{FIBERS_DIGEST_FORMAT}\n'.encode())
    for row in sorted(rows, key=lambda r: r['uuid']):
        digest.update(f"{row['uuid']} {row['version_generation']} {row['version']}\n".encode())
    return digest.hexdigest()


def write_fibers(destination, fibers, *, volume, author=None):
    """A new store holding ``fibers``: (row, xyz) pairs, xyz in voxels of ``volume`` (bound with scale 1). row keys:
    id (a UUID to keep, or absent/None for a new one), name, family, optional meta (its coordinates are replaced).
    Each fiber gets end-point controls 0,1 and every sample as line points. Written to a sibling staging file and
    renamed into place; an existing destination is refused. Returns dict(fibers, points, ids)."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'{destination} exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    coordinates = dict(volume_uuid=str(volume), scale_factor=1)
    ids, points_total = [], 0
    with tempfile.TemporaryDirectory(prefix='.'+destination.name+'-', dir=destination.parent) as staging:
        staged = Path(staging)/'fibers.vcobj'
        with ObjectStore.create(str(staged), author=author or getpass.getuser(), reader_pool_size=1,
                                geometry_cache_bytes=0, spatial_cache_bytes=0) as store:
            pending = []

            def flush():
                if pending:
                    with store.write(expected_versions={r['uuid']: None for r, _ in pending}) as tx:
                        for record, geometry in pending:
                            tx.upsert_object(record)
                            tx.put_geometry(record['uuid'], geometry)
                    pending.clear()

            for row, xyz in fibers:
                xyz = np.ascontiguousarray(xyz, dtype='<f8')
                if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
                    raise ValueError(f'Invalid polyline for fiber {row.get("id")}')
                fid = str(uuidlib.UUID(str(row['id']))) if row.get('id') else str(uuidlib.uuid4())
                ids.append(fid)
                points_total += len(xyz)
                record = dict(uuid=fid, kind='fiber', name=row.get('name', ''), format_version=4,
                              fiber=dict(family=row.get('family', ''), control_point_count=2,
                                         line_point_count=len(xyz),
                                         length=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                                         bbox=[*map(float, xyz.min(0)), *map(float, xyz.max(0))]),
                              meta={**(row.get('meta') or {}), 'coordinates': coordinates})
                pending.append((record, Fiber.from_arrays(np.ascontiguousarray(xyz[[0, -1]]), xyz)))
                if len(pending) >= WRITE_BATCH:
                    flush()
            flush()
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate fiber UUID')
        if any(Path(str(staged)+suffix).exists() for suffix in ('-wal', '-shm')):
            raise RuntimeError('Store was not checkpointed on close')
        os.link(staged, destination)
    return dict(fibers=len(ids), points=points_total, ids=ids)


def write_initial_fibers(destination, fibers, *, volume, workers=32, packed=False):
    """Immutable batch import of traced polylines; no invented control points.

    Uses the same UUID-routed native queues as AFV and prepared imports. The
    native importer owns staging, failure cleanup, sealing and atomic publication.
    Returns the actual root locator (collection.vcobj for unpacked shards).
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records, ids, points_total = [], [], 0
    for row, points in fibers:
        xyz = np.ascontiguousarray(points, dtype='<f8')
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
            raise ValueError(f'Invalid polyline for fiber {row.get("id")}')
        fid = str(uuidlib.UUID(str(row['id']))) if row.get('id') else str(uuidlib.uuid4())
        ids.append(fid)
        points_total += len(xyz)
        record = dict(uuid=fid, kind='fiber', name=row.get('name', ''), format_version=4,
                      fiber=dict(family=row.get('family', ''), control_point_count=0,
                                 line_point_count=len(xyz),
                                 length=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                                 bbox=[*map(float, xyz.min(0)), *map(float, xyz.max(0))]),
                      meta={**(row.get('meta') or {}), 'coordinates': dict(volume_uuid=str(volume), scale_factor=1)})
        records.append(dict(record=record, geometry=dict(kind='fiber', format_version=1,
                            arrays=[dict(name='control_points', dtype='<f8', shape=[0,3], data=b''),
                                    dict(name='control_point_ids', dtype='<i8', shape=[0], data=b''),
                                    dict(name='line_points', dtype='<f8', shape=list(xyz.shape), data=xyz.tobytes())])))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate fiber UUID')
    result = import_initial_records(str(destination), records, workers=workers, packed=packed)
    return dict(result, fibers=len(ids), points=points_total, ids=ids)


class FiberStore(Sequence):
    """Training access to the fibers of one store bound to one volume: a catalog in memory, polylines decoded into
    a bounded LRU, one read-only handle per process (reopened after a fork; dropped when pickled).

    Validation holds out whole fibers by UUID (fiber_hash), including from neighbor queries.
    """
    def __init__(self, path, volume, grid_scale=1., *, validation=None, split='train', cache_size=32):
        self.path = str(Path(path).resolve())
        self.volume = str(volume)
        self.grid_scale = float(grid_scale)
        if not np.isfinite(self.grid_scale) or self.grid_scale <= 0:
            raise ValueError('Fiber grid_scale must be finite and positive')
        if split not in ('train', 'validation', 'all'):
            raise ValueError('Invalid fiber split')
        self.cache_size = cache_size
        self._store = None
        self._pid = None
        self._cache = OrderedDict()
        self.catalog = fiber_catalog(self.path, self.volume)
        if validation is not None:
            heldout = heldout_ids([r['id'] for r in self.catalog], validation)
            if split != 'all':
                self.catalog = [r for r in self.catalog if ((r['id'] in heldout) == (split == 'validation'))]
        if not self.catalog:
            raise ValueError(f'No fibers bound to volume {self.volume} in {split} split')
        self.ids = {r['id'] for r in self.catalog}
        self.id_to_index = {r['id']: i for i, r in enumerate(self.catalog)}
        # Catalog lengths are cheap sampling weights. Geometry bounds must use
        # the recomputed arc array: densification can change its last bits.
        self.lengths = np.array([r['length']/self.grid_scale for r in self.catalog])

    def store(self):
        if self._store is None or self._pid != os.getpid():
            self._store = open_store(self.path)
            self._pid = os.getpid()
        return self._store

    def __getstate__(self):
        return {**self.__dict__, '_store': None, '_pid': None, '_cache': OrderedDict()}

    def __len__(self):
        return len(self.catalog)

    def manifest_entries(self):
        ids = [r['id'] for r in self.catalog]
        # A compact collection manifest avoids hashing/expanding all polylines
        # each time a replay cache is opened by a loader worker.
        return [dict(kind='fiber_store_catalog_v2', digest=fibers_digest(self.path, ids), volume=self.volume,
                     grid_scale=self.grid_scale, fiber_count=len(self),
                     fiber_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                     annotation_repair=ANNOTATION_REPAIR)]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[j] for j in range(*index.indices(len(self)))]
        # Catalog lookup also enforces Sequence's IndexError iteration contract.
        return _Fiber(self, index, self.catalog[index])

    def geometry(self, index):
        if index not in self._cache:
            row = self.catalog[index]
            with self.store().geometry_snapshot() as snapshot:
                xyz = snapshot.fiber(row['id']).line_points_array()
            if len(xyz) < 2 or not np.isfinite(xyz).all():
                raise ValueError(f'Invalid fiber geometry: {row["id"]}')
            points = xyz*(row['scale']/self.grid_scale)
            # Preserve corners while bounding nearest-vertex clearance error.
            points = dense_line(points, 1.)
            points, repairs = repair_kinks(points, arclength(points))
            self._cache[index] = TracedFiber(row['name'], points, arclength(points), row['family'],
                endpoint_stop=(False, False), source_hash=row['id'], kink_repairs=len(repairs))
        self._cache.move_to_end(index)
        result = self._cache[index]
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def _regions(self, lo, hi):
        """The store's spatial blocks (<= 257 line points) of this split whose bounds overlap [lo, hi] (trace grid)."""
        bbox = [*(np.asarray(lo)*self.grid_scale), *(np.asarray(hi)*self.grid_scale)]
        return [r for r in self.store().spatial_regions(bbox, object_kind='fiber', volume=self.volume)
                if r['array_name'] == 'line_points' and r['object_uuid'] in self.ids]

    def _blocks(self, lo, hi):
        """The overlapping blocks' points in trace grid units: (fiber id, first vertex, points)."""
        regions = self._regions(lo, hi)
        blocks = []
        with self.store().geometry_snapshot() as snapshot:
            for r in regions:
                fid = r['object_uuid']
                scale = self.catalog[self.id_to_index[fid]]['scale']/self.grid_scale
                line = snapshot.fiber(fid, r['revision_uuid'])
                blocks.append((fid, int(r['start'][0]), line.line_points_array(int(r['start'][0]), int(r['shape'][0]))*scale))
        yield from blocks

    def nearby_blocks(self, bounds_xyz, exclude_id, *, records=False):
        lo, hi = np.asarray(bounds_xyz)
        for fid, first, points in self._blocks(lo, hi):
            if fid == exclude_id:
                continue
            arc = arclength(points)
            samples = interp_at(points, arc, np.arange(0., arc[-1]+1e-9, .25))
            yield dict(points=points, samples=samples, shard=f'fiber:{fid}', index=first) if records else samples

    def nearby_fiber_ids(self, point, radius, exclude_id):
        point = np.asarray(point)
        return sorted({r['object_uuid'] for r in self._regions(point-radius, point+radius)} - {exclude_id})


class _Fiber:
    def __init__(self, collection, index, row):
        self.collection, self.index = collection, index
        self.name, self.tag = row['name'], row['family']
        self.endpoint_stop = (False, False)

    @property
    def length(self):
        return self.collection.geometry(self.index).length

    def __getattr__(self, key):
        return getattr(self.collection.geometry(self.index), key)
