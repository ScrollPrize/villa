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
from vc_object_store import Fiber, ObjectStore, canonical_uuid, geometry_scale_to_volume, object_reference
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
        first = store.select_objects(object_kind='fiber', limit=1, fields=['meta'])
        if not first:
            raise ValueError(f'No fibers in {path}')
        volume = (first[0]['meta'].get('coordinates') or {}).get('volume_uuid')
        if volume is None:
            raise ValueError(f'Fibers in {path} are not bound to a volume')
        if store.unbound_object_count('fiber', volume):
            raise ValueError(f'Fibers in {path} are bound to several volumes; name one')
    return volume


def fiber_catalog(path, volume, *, source_version=None):
    """Catalog rows of the fibers bound to ``volume``, in UUID order, measured from selected geometry:
    dict(id, name, family, length, point_count, bbox, scale, reference); length and bbox (minx,miny,minz,maxx,maxy,maxz, or
    None) are volume voxels. Frozen evidence can require its recorded ``source_version``."""
    with open_store(path) as store:
        _require_source_version(store, source_version)
        rows = _catalog(store, volume)
        _require_source_version(store, source_version)
        return rows


def _catalog(store, volume):
    records = store.select_objects(object_kind='fiber', volume=volume, fields=['name', 'fiber', 'meta'])
    return [row for row, _ in _read_rows(store, records, volume, require_lines=False)]


def _read_rows(store, records, volume, *, require_lines=True):
    # One selected identity per object; bound payload memory rather than loading
    # an entire volume merely to derive current lengths/counts/bounds.
    for start in range(0, len(records), WRITE_BATCH):
        selected = records[start:start+WRITE_BATCH]
        with store.graph_read(versions=[object_reference(r) for r in selected]) as view:
            packed = view.read_geometry_arrays('line_points')
            values, offsets = packed.values, packed.offsets
            for node, record in enumerate(view.objects(['name', 'fiber', 'meta'])):
                scale = geometry_scale_to_volume(record['meta'], volume)
                xyz = np.array(values[offsets[node]:offsets[node+1]], dtype=np.float64)
                if scale is None or len(xyz) < 2:
                    if require_lines:
                        raise ValueError(f'Fiber {record["uuid"]} is not bound to volume {volume} or has no line')
                    continue
                xyz *= scale
                fiber = record.get('fiber') or {}
                yield dict(id=record['uuid'], name=record['name'], family=fiber.get('family', ''),
                           length=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                           point_count=len(xyz), bbox=[*map(float, xyz.min(0)), *map(float, xyz.max(0))],
                           scale=scale, meta=record['meta'], reference=object_reference(record)), xyz


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


def read_polylines(path, fiber_ids=None, *, volume=None, source_version=None):
    """[(row, xyz)] of the requested fibers (all bound ones when None), in UUID order; xyz are the raw line points in
    volume voxels (no repair or densifying). row = dict(id, name, family, length, point_count, meta, reference). ``volume``
    defaults to the store's one volume (fiber_volume). Raises for an unknown, unbound or line-less requested fiber.
    Frozen evidence supplies ``source_version``; both sides of the read must match that store snapshot."""
    volume = volume or fiber_volume(path)
    selected = None if fiber_ids is None else sorted({canonical_uuid(str(fid)) for fid in fiber_ids})
    with open_store(path) as store:
        _require_source_version(store, source_version)
        if selected == []:
            return []
        records = store.select_objects(object_kind='fiber', object_uuids=selected or [],
                                       volume=volume, fields=['name', 'fiber', 'meta'])
        if selected is not None and sorted(r['uuid'] for r in records) != selected:
            raise ValueError('Unknown requested fiber')
        rows = sorted(_read_rows(store, records, volume, require_lines=selected is not None), key=lambda item: item[0]['id'])
        _require_source_version(store, source_version)
        return rows


FIBERS_DIGEST_FORMAT = 'fibers_digest v3'


def fibers_digest(path, fiber_ids=None):
    """Content digest of a store's live fibers (all, or the given ones): SHA-256 over a format tag and each fiber's
    complete owning-store/generation/object/version identity, in UUID order, read in one snapshot.
    Format v3 includes source ownership, so divergent branches cannot collide at equal version counters."""
    selected = [] if fiber_ids is None else sorted({canonical_uuid(str(fid)) for fid in fiber_ids})
    with open_store(path) as store:
        rows = (store.select_objects(object_kind='fiber', object_uuids=selected, fields=['name'])
                if selected or fiber_ids is None else [])
    if fiber_ids is not None and sorted(r['uuid'] for r in rows) != selected:
        raise ValueError('Unknown requested fiber')
    return _references_digest(object_reference(row) for row in rows)


def _references_digest(references):
    digest = hashlib.sha256(f'{FIBERS_DIGEST_FORMAT}\n'.encode())
    for ref in sorted(references, key=lambda r: r['object']):
        digest.update(f"{ref['store']} {ref['generation']} {ref['object']} {ref['version']}\n".encode())
    return digest.hexdigest()


def write_fibers(destination, fibers, *, volume, author=None, metadata_policy='portable'):
    """A new store holding ``fibers``: (row, xyz) pairs, xyz in voxels of ``volume`` (bound with scale 1). row keys:
    id (a UUID to keep, or absent/None for a new one), name, family, optional meta (its coordinates are replaced).
    Each fiber gets end-point controls 0,1 and every sample as line points. Written to a sibling staging file and
    renamed into place; an existing destination is refused. Managed extension/artifact descriptors are rejected
    unless metadata_policy='geometry_only' explicitly requests new geometry without those payloads.
    This is a derived-data writer, not a store/relationship/history copy. Returns dict(fibers, points, ids)."""
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
        os.link(staged, destination)
    return dict(fibers=len(ids), points=points_total, ids=ids)


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
            with self.store().graph_read(versions=[row['reference']]) as view:
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
