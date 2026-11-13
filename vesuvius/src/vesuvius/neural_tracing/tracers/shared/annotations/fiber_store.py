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
from vc_object_store import Fiber, ObjectStore, import_initial_records, canonical_uuid, geometry_scale_to_volume, object_reference
from vc_object_store.derivation import derived_metadata
from vesuvius.neural_tracing.tracers.shared.annotations.repair import ANNOTATION_REPAIR, repair_kinks
from vesuvius.neural_tracing.tracers.shared.annotations.traced import TracedFiber
from vesuvius.neural_tracing.tracers.shared.annotations.splits import heldout_ids
from vesuvius.neural_tracing.tracers.shared.geometry import arclength, interp_at, dense_line

WRITE_BATCH = 256


def open_store(path):
    """A read-only handle; a context manager. Not shareable across processes: open one per process."""
    return ObjectStore.open(str(path), read_only=True, reader_pool_size=1)


def fiber_volume(path):
    """The one volume every fiber of the store is bound to; raises when they are unbound or bound to several."""
    with open_store(path) as store:
        records = store.select_objects(object_kind='fiber', fields=['meta'])
        return _fiber_records_volume(records, path)


def _fiber_records_volume(records, path):
    if not records:
        raise ValueError(f'No fibers in {path}')
    volume = (records[0]['meta'].get('coordinates') or {}).get('volume_uuid')
    if volume is None or any(geometry_scale_to_volume(row['meta'], volume) is None for row in records):
        raise ValueError(f'Fibers in {path} must be bound to one volume')
    return volume


def fiber_catalog(path, volume=None, *, source_version=None):
    """Catalog rows of the fibers bound to ``volume``, in UUID order, measured from selected geometry:
    dict(id, name, family, length, point_count, bbox, scale, reference); length and bbox (minx,miny,minz,maxx,maxy,maxz, or
    None) are volume voxels. Frozen evidence can require its recorded ``source_version``."""
    with open_store(path) as store:
        _require_source_version(store, source_version)
        rows = _catalog(store, volume)
        _require_source_version(store, source_version)
        return rows


def fiber_point_counts(path, *, source_version=None):
    """UUID-ordered selected line-point counts without reading XYZ payloads.

    Descriptor interpretation and retained/current integrity checks are native.
    Frozen evidence may require the same store position on both sides.
    """
    with open_store(path) as store:
        _require_source_version(store, source_version)
        columns = store.geometry_array_counts('fiber', 'line_points')
        _require_source_version(store, source_version)
    ids = np.frombuffer(columns['uuid_ascii'], dtype='S36').astype('U36')
    return ids, columns['count']


def fiber_count(path, *, source_version=None):
    """Count live fibers natively without materializing their identities."""
    with open_store(path) as store:
        _require_source_version(store, source_version)
        result = store.object_count('fiber')
        _require_source_version(store, source_version)
    return result


def _catalog(store, volume):
    records = store.select_objects(object_kind='fiber', volume=volume, fields=['name', 'fiber', 'meta'])
    volume = volume or _fiber_records_volume(records, 'fiber store')
    return [row for row, _ in _read_rows(store, records, volume, require_lines=False)]


def _read_rows(store, records, volume, *, require_lines=True):
    # One selected identity per object; bound payload memory rather than loading
    # an entire volume merely to derive current lengths/counts/bounds.
    for start in range(0, len(records), WRITE_BATCH):
        selected = records[start:start+WRITE_BATCH]
        with store.graph_read(versions=[object_reference(r) for r in selected], edge_limit=0) as view:
            yield from _view_rows(view, volume, require_lines=require_lines)


def _view_rows(view, volume, *, require_lines=True):
    """Adapt selected mapped geometry only for callers requiring row records."""
    records = view.objects(['name', 'fiber', 'meta'])
    for start in range(0, len(records), WRITE_BATCH):
        selected = records[start:start+WRITE_BATCH]
        packed = view.read_geometry_arrays('line_points', nodes=list(range(start, start+len(selected))), volume=volume)
        values, offsets = packed.values, packed.offsets
        for node, record in enumerate(selected):
            xyz = np.array(values[offsets[node]:offsets[node+1]], dtype=np.float64)
            if len(xyz) < 2:
                if require_lines:
                    raise ValueError(f'Fiber {record["uuid"]} is not bound to volume {volume} or has no line')
                continue
            scale = geometry_scale_to_volume(record['meta'], volume)
            fiber = record.get('fiber') or {}
            yield dict(id=record['uuid'], name=record['name'], family=fiber.get('family', ''),
                       length=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                       point_count=len(xyz), bbox=[*map(float, xyz.min(0)), *map(float, xyz.max(0))],
                       scale=1. if scale is None else scale,
                       meta=record['meta'], reference=object_reference(record)), xyz


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


def _source_version(store):
    status = store.status()
    return {key: status[key] for key in ('store', 'generation', 'commit_sequence')}


def fiber_source_version(path):
    """Compact source identity for frozen evidence, including coordinate-registration changes."""
    with open_store(path) as store:
        return _source_version(store)


def _require_source_version(store, expected):
    if expected is not None and expected != _source_version(store):
        raise ValueError('Fiber source changed or missing source version; regenerate frozen evidence')


def _position(store):
    return tuple(_source_version(store).values())


def iter_polylines(path, fiber_ids=None, *, volume=None, source_version=None):
    """[(row, xyz)] of the requested fibers (all bound ones when None), in UUID order; xyz are the raw line points in
    volume voxels (no repair or densifying). row = dict(id, name, family, length, point_count, meta, reference). ``volume``
    defaults to the store's one volume (fiber_volume). Raises for an unknown, unbound or line-less requested fiber.
    Frozen evidence supplies ``source_version``; both sides of the read must match that store snapshot."""
    volume = volume or fiber_volume(path)
    selected = None if fiber_ids is None else sorted({canonical_uuid(str(fid)) for fid in fiber_ids})
    with open_store(path) as store:
        _require_source_version(store, source_version)
        if selected == []:
            return
        position = source_version or _source_version(store)
        cursor = None
        while True:
            with store.geometry_read(object_kind='fiber', object_uuids=selected or [], volume=volume,
                                     limit=None if selected is not None else WRITE_BATCH,
                                     after_uuid=cursor, expected_position=position) as view:
                ids = view.object_uuids
                if selected is not None and ids != selected:
                    raise ValueError('Unknown requested fiber')
                yield from _view_rows(view, volume, require_lines=selected is not None)
            if selected is not None or len(ids) < WRITE_BATCH:
                break
            cursor = ids[-1]
        _require_source_version(store, position)


def read_polylines(path, fiber_ids=None, *, volume=None, source_version=None):
    """Materialize raw curves; use iter_polylines for bounded streaming with the same source pin."""
    return list(iter_polylines(path, fiber_ids, volume=volume, source_version=source_version))


def read_polyline_arrays(path, fiber_ids, *, volume=None, source_version=None):
    """Current packed XYZ and UUID columns, without Python per-fiber records.

    Native reading captures metadata/geometry together and performs coordinate
    mapping under the same source-position guard. Returned buffers own their
    data after the store transaction closes.
    """
    if fiber_ids is None:raise ValueError('Packed polyline reads require explicit fiber IDs')
    volume = volume or fiber_volume(path)
    selected = _canonical_fiber_ids(fiber_ids)
    with open_store(path) as store:
        with store.geometry_read(object_kind='fiber', object_uuids=selected, volume=volume,
                                 limit=0 if not selected else None, expected_position=source_version) as view:
            ids = view.object_uuids
            if ids != selected:raise ValueError('Unknown requested fiber')
            packed = view.read_geometry_arrays('line_points', volume=volume)
    return np.asarray(ids, dtype='U36'), packed


def _canonical_fiber_ids(fiber_ids):
    """Shared bulk identity adaptation without a Python per-object loop."""
    from vc_object_store import canonical_uuid_ascii
    raw = np.asarray(fiber_ids if isinstance(fiber_ids,np.ndarray) else list(fiber_ids), dtype=str)
    if raw.ndim != 1:raise ValueError('Fiber IDs must be a one-dimensional column')
    if not len(raw):return []
    encoded = np.char.encode(raw, 'ascii')
    column = encoded.view(np.uint8).reshape(len(raw), encoded.dtype.itemsize)
    canonical = canonical_uuid_ascii(column, workers=1)
    return np.unique(canonical.view('S36').ravel().astype('U36')).tolist()


FIBERS_DIGEST_FORMAT = 'fibers_digest v3'


def fibers_digest(path, fiber_ids=None, *, source_version=None):
    """Content digest of a store's live fibers (all, or the given ones): SHA-256 over a format tag and each fiber's
    complete owning-store/generation/object/version identity, in UUID order, read in one snapshot.
    Format v3 includes source ownership, so divergent branches cannot collide at equal version counters."""
    selected = None if fiber_ids is None else _canonical_fiber_ids(fiber_ids)
    with open_store(path) as store:
        _require_source_version(store, source_version)
        result = store.selected_reference_digest(object_kind='fiber',object_uuids=selected,domain=FIBERS_DIGEST_FORMAT)
        _require_source_version(store, source_version)
    return result


def verify_fiber_source(path, digest, source_version=None):
    """Validate saved evidence and return the snapshot to use for subsequent reads."""
    version = source_version if source_version is not None else fiber_source_version(path)
    if fibers_digest(path, source_version=version) != digest:
        raise ValueError('Saved fiber source differs; regenerate frozen evidence')
    return version


def verify_fiber_signature(path, signature):
    """Validate flat or subset tensor provenance and return digest and pin."""
    digest = (signature.get('inputs') or signature).get('fibers')
    if not digest:
        raise ValueError('Saved tensors have no fiber provenance; regenerate frozen evidence')
    return digest, verify_fiber_source(path, digest, signature.get('fiber_source'))


def _references_digest(references):
    from vc_object_store import object_references_digest
    return object_references_digest(list(references),domain=FIBERS_DIGEST_FORMAT)


def write_fibers(destination, fibers, *, volume, author=None, metadata_policy='portable', replace=False):
    """A new store holding ``fibers``: (row, xyz) pairs, xyz in voxels of ``volume`` (bound with scale 1). row keys:
    id (a UUID to keep, or absent/None for a new one), name, family, optional meta (its coordinates are replaced).
    Each fiber gets end-point controls 0,1 and every sample as line points. Written to a sibling staging file and
    published into place; an existing destination is refused unless replace=True. Managed extension/artifact descriptors are rejected
    unless metadata_policy='geometry_only' explicitly requests new geometry without those payloads.
    This is a derived-data writer, not a store/relationship/history copy. Returns dict(fibers, points, ids)."""
    destination = Path(destination)
    if destination.exists() and not replace:
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
                fid = canonical_uuid(str(row['id'])) if row.get('id') else str(uuidlib.uuid4())
                ids.append(fid)
                points_total += len(xyz)
                record = dict(uuid=fid, kind='fiber', name=row.get('name', ''), format_version=4,
                              fiber=dict(family=row.get('family', ''), control_point_count=2,
                                         line_point_count=len(xyz),
                                         length=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                                         bbox=[*map(float, xyz.min(0)), *map(float, xyz.max(0))]),
                              meta={**derived_metadata(row.get('meta') or {}, policy=metadata_policy,
                                                       reference=row.get('reference')), 'coordinates': coordinates})
                pending.append((record, Fiber.from_arrays(np.ascontiguousarray(xyz[[0, -1]]), xyz)))
                if len(pending) >= WRITE_BATCH:
                    flush()
            flush()
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate fiber UUID')
        if any(Path(str(staged)+suffix).exists() for suffix in ('-wal', '-shm')):
            raise RuntimeError('Store was not checkpointed on close')
        if replace:
            staged.replace(destination)
        else:
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
        with open_store(self.path) as store:
            self._position = _position(store)
            self.catalog = _catalog(store, self.volume)
            self._require_unchanged(store)
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

    def require_digest(self, expected):
        """Reject a cached selection from another source before interpreting its ordinal indices."""
        self._require_unchanged()
        if not expected or expected != fibers_digest(self.path):
            raise ValueError('Fiber source changed or missing source digest; rebuild the index')
        self._require_unchanged()

    def _require_unchanged(self, store=None):
        if _position(store or self.store()) != self._position:
            raise RuntimeError('Fiber source changed after catalog selection; reopen the dataset')

    def manifest_entries(self):
        self._require_unchanged()
        ids = [r['id'] for r in self.catalog]
        # A compact collection manifest avoids hashing/expanding all polylines
        # each time a replay cache is opened by a loader worker.
        return [dict(kind='fiber_store_catalog_v3', digest=_references_digest(r['reference'] for r in self.catalog), volume=self.volume,
                     grid_scale=self.grid_scale, fiber_count=len(self),
                     fiber_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                     annotation_repair=ANNOTATION_REPAIR)]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[j] for j in range(*index.indices(len(self)))]
        # Catalog lookup also enforces Sequence's IndexError iteration contract.
        return _Fiber(self, index, self.catalog[index])

    def geometry(self, index):
        self._require_unchanged()
        if index not in self._cache:
            row = self.catalog[index]
            with self.store().graph_read(versions=[row['reference']], edge_limit=0) as view:
                xyz = view.read_geometry_arrays('line_points').values
            self._require_unchanged()
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
        self._require_unchanged()
        bbox = [*(np.asarray(lo)*self.grid_scale), *(np.asarray(hi)*self.grid_scale)]
        regions = [r for r in self.store().spatial_regions(bbox, object_kind='fiber', volume=self.volume)
                   if r['array_name'] == 'line_points' and r['object_uuid'] in self.ids]
        self._require_unchanged()
        return regions

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
        self._require_unchanged()
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
