"""Worker-local access to native-L0 Automated Fiber Volumes (read), and writing a derived AFV in the same format."""
from collections import OrderedDict
from collections.abc import Sequence
import json
import hashlib
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit

import numpy as np

from vesuvius.neural_tracing.tracers.shared.annotations.repair import ANNOTATION_REPAIR, repair_kinks
from vesuvius.neural_tracing.tracers.shared.annotations.traced import TracedFiber
from vesuvius.neural_tracing.tracers.shared.annotations.splits import heldout_ids
from vesuvius.neural_tracing.tracers.shared.geometry import arclength, interp_at, dense_line


AFV_APPLICATION_ID, AFV_USER_VERSION = 0x56434643, 1
AFV_SCHEMA = (  # VC3D's Automated Fiber Volume format, version 1
    'CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)',
    '''CREATE TABLE fibers(id INTEGER PRIMARY KEY, name TEXT NOT NULL, family TEXT NOT NULL,
 point_count INTEGER NOT NULL CHECK(point_count >= 2), length REAL NOT NULL,
 min_x REAL NOT NULL, max_x REAL NOT NULL, min_y REAL NOT NULL, max_y REAL NOT NULL,
 min_z REAL NOT NULL, max_z REAL NOT NULL, annotation TEXT NOT NULL)''',
    '''CREATE TABLE blocks(id INTEGER PRIMARY KEY, fiber_id INTEGER NOT NULL REFERENCES fibers(id),
 first_segment INTEGER NOT NULL, points BLOB NOT NULL, UNIQUE(fiber_id, first_segment))''',
    'CREATE INDEX fibers_length ON fibers(length DESC, id)',
    'CREATE VIRTUAL TABLE block_bounds USING rtree(id, min_x, max_x, min_y, max_y, min_z, max_z)')
OPEN_DATA_BUCKET = 'vesuvius-challenge-open-data'
OPEN_DATA_HTTPS = f'https://{OPEN_DATA_BUCKET}.s3.us-east-1.amazonaws.com'


def open_data_volume(url):
    """(sample, volume id, base voxel um, https path) of a Vesuvius open-data CT store
    (``s3://vesuvius-challenge-open-data/SAMPLE/volumes/ID-2.400um-....zarr`` or its https form), else None."""
    source = urlsplit(str(url).rstrip('/'))
    if (source.scheme, source.netloc) not in (('s3', OPEN_DATA_BUCKET), ('https', urlsplit(OPEN_DATA_HTTPS).netloc)):
        return None
    parts = source.path.strip('/').split('/')
    match = re.match(r'(\d+)-(\d+(?:\.\d+)?)um', parts[-1])
    if len(parts) != 3 or parts[1] != 'volumes' or not match:
        return None
    return parts[0], match[1], float(match[2]), f'{OPEN_DATA_HTTPS}/{"/".join(parts)}'


def open_data_frame(url, base_shape_zyx):
    """AFV 'frame' and 'root' metadata of native L0 XYZ coordinates of an open-data CT store (``open_data_volume``),
    as VC3D's fiber collection dock reads them; None for another store."""
    volume = open_data_volume(url)
    if volume is None:
        return None
    sample, volume_id, um, https = volume
    tags = dict(vc_open_data_coordinate_space=f'{sample}/{volume_id}', vc_open_data_source_coordinate_level=0,
                vc_open_data_source_coordinate_scale_factor=1)
    root = dict(type='vc3d_fiber_collection', version=1, scale=1, **tags, coordinate_order='XYZ',
                coordinate_units='voxel', voxel_size_mm=um/1000, vc_open_data_source_original_resolution=um,
                coordinate_base_shape_zyx=[int(v) for v in base_shape_zyx], vc_open_data_source_path=https)
    return dict(frame=tags, root=root)


def file_sha256(path):
    """SHA-256 of a whole file, read in 8 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class AFVFibers(Sequence):
    """Keep only a catalog in memory; decode polylines into a bounded LRU.

    Validation holds out entire fiber IDs, including from neighbor queries.
    """
    def __init__(self, path, grid_scale=1., *, validation=None, split='train', cache_size=32, sha256=None):
        self.path = str(Path(path).resolve())
        self.grid_scale = float(grid_scale)
        if not np.isfinite(self.grid_scale) or self.grid_scale <= 0:
            raise ValueError('AFV grid_scale must be finite and positive')
        if split not in ('train', 'validation', 'all'):
            raise ValueError('Invalid AFV split')
        self.cache_size = cache_size
        self.sha256 = sha256
        self._connection = None
        self._pid = None
        self._cache = OrderedDict()
        c = self.connection()
        if (c.execute('PRAGMA application_id').fetchone()[0] != 0x56434643
                or c.execute('PRAGMA user_version').fetchone()[0] != 1):
            raise ValueError('Expected AFV format version 1')
        self.metadata = {k: json.loads(v) for k, v in c.execute('SELECT key,value FROM metadata')}
        if self.metadata.get('complete') is not True:
            raise ValueError('Incomplete AFV')
        frame = self.metadata['frame']
        if frame.get('vc_open_data_source_coordinate_level', 0) != 0 or frame.get('vc_open_data_source_coordinate_scale_factor', 1) != 1:
            raise ValueError('AFV geometry must be native L0 XYZ')
        self.catalog = c.execute('SELECT id,name,family,length,min_z,max_z FROM fibers ORDER BY id').fetchall()
        if len(self.catalog) != self.metadata['fiber_count']:
            raise ValueError('AFV fiber count mismatch')
        if validation is not None:
            heldout = heldout_ids([r[0] for r in self.catalog],validation)
            if split != 'all':
                self.catalog = [r for r in self.catalog if ((r[0] in heldout) == (split == 'validation'))]
        if not self.catalog:
            raise ValueError(f'No AFV fibers in {split} split')
        self.ids = {r[0] for r in self.catalog}
        self.id_to_index = {r[0]: i for i,r in enumerate(self.catalog)}
        # Catalog lengths are cheap sampling weights. Geometry bounds must use
        # the recomputed arc array: densification can change its last bits.
        self.lengths = np.array([r[3]/self.grid_scale for r in self.catalog])

    def connection(self):
        if self._connection is None or self._pid != os.getpid():
            if self._connection is not None:
                self._connection.close()
            self._connection = sqlite3.connect(Path(self.path).as_uri()+'?mode=ro', uri=True)
            self._pid = os.getpid()
        return self._connection

    def __getstate__(self):
        return {**self.__dict__, '_connection': None, '_pid': None, '_cache': OrderedDict()}

    def __len__(self):
        return len(self.catalog)

    def manifest_entries(self):
        if self.sha256 is None:
            self.sha256 = file_sha256(self.path)
        ids = np.array([r[0] for r in self.catalog],dtype='<i8')
        # A compact immutable collection manifest avoids hashing/expanding all
        # polylines each time a replay cache is opened by a loader worker.
        return [dict(kind='afv_catalog_v1', sha256=self.sha256, grid_scale=self.grid_scale,
                     fiber_count=len(self), fiber_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                     annotation_repair=ANNOTATION_REPAIR)]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[j] for j in range(*index.indices(len(self))) ]
        # Catalog lookup also enforces Sequence's IndexError iteration contract.
        row = self.catalog[index]
        return _Fiber(self, index, row)

    def geometry(self, index):
        if index not in self._cache:
            fiber_id, name, family, _, _, _ = self.catalog[index]
            pieces, next_segment = [], 0
            for first, blob in self.connection().execute(
                    'SELECT first_segment,points FROM blocks WHERE fiber_id=? ORDER BY first_segment', (fiber_id,)):
                xyz = np.frombuffer(blob, dtype='<f8').reshape(-1, 3)
                if first != next_segment or not 2 <= len(xyz) <= 257 or not np.isfinite(xyz).all():
                    raise ValueError(f'Invalid AFV geometry: {fiber_id}')
                if pieces and not np.array_equal(pieces[-1][-1], xyz[0]):
                    raise ValueError(f'Discontinuous AFV blocks: {fiber_id}')
                pieces.append(xyz if not pieces else xyz[1:])
                next_segment += len(xyz)-1
            points = np.concatenate(pieces)/self.grid_scale
            # Preserve corners while bounding nearest-vertex clearance error.
            points = dense_line(points, 1.)
            points, repairs = repair_kinks(points, arclength(points))
            self._cache[index] = TracedFiber(name, points, arclength(points), family,
                endpoint_stop=(False, False), source_hash=f'{self.metadata["uuid"]}:{fiber_id}',
                kink_repairs=len(repairs))
        self._cache.move_to_end(index)
        result = self._cache[index]
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def nearby_blocks(self, bounds_xyz, exclude_id, *, records=False):
        lo, hi = np.asarray(bounds_xyz)*self.grid_scale
        rows = self.connection().execute('''SELECT b.id,b.fiber_id,b.points FROM block_bounds r
            JOIN blocks b ON b.id=r.id WHERE r.max_x>=? AND r.min_x<=?
            AND r.max_y>=? AND r.min_y<=? AND r.max_z>=? AND r.min_z<=?''',
            (lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]))
        for bid, fid, blob in rows:
            if fid == exclude_id or fid not in self.ids:
                continue
            points = np.frombuffer(blob, dtype='<f8').reshape(-1,3)/self.grid_scale
            arc = arclength(points)
            samples = interp_at(points, arc, np.arange(0., arc[-1]+1e-9, .25))
            yield dict(points=points,samples=samples,shard=f'fiber:{fid}',index=bid) if records else samples

    def nearby_fiber_ids(self, point, radius, exclude_id):
        point = np.asarray(point)*self.grid_scale
        lo, hi = point-radius*self.grid_scale, point+radius*self.grid_scale
        rows = self.connection().execute('''SELECT DISTINCT b.fiber_id FROM block_bounds r
            JOIN blocks b ON b.id=r.id WHERE r.max_x>=? AND r.min_x<=?
            AND r.max_y>=? AND r.min_y<=? AND r.max_z>=? AND r.min_z<=?''',
            (lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]))
        return sorted(r[0] for r in rows if r[0] != exclude_id and r[0] in self.ids)


class _Fiber:
    def __init__(self, collection, index, row):
        self.collection, self.index = collection, index
        self.name, self.tag = row[1:3]
        self.endpoint_stop = (False, False)

    @property
    def length(self):
        return self.collection.geometry(self.index).length

    def __getattr__(self, key):
        return getattr(self.collection.geometry(self.index), key)


def read_afv_polylines(path):
    """Every fiber of an AFV as (fibers row dict, native xyz (N, 3) polyline), in id order."""
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) as c:
        columns = [r[1] for r in c.execute('PRAGMA table_info(fibers)')]
        rows = {r[0]: dict(zip(columns, r)) for r in c.execute(f'SELECT {",".join(columns)} FROM fibers ORDER BY id')}
        pieces = {}
        for fid, first, blob in c.execute('SELECT fiber_id,first_segment,points FROM blocks ORDER BY fiber_id,first_segment'):
            xyz = np.frombuffer(blob, dtype='<f8').reshape(-1, 3)
            seen = fid in pieces  # tested before setdefault adds the key: blocks after the first share their first point
            pieces.setdefault(fid, []).append(xyz[1:] if seen else xyz)
    return [(rows[fid], np.concatenate(pieces[fid])) for fid in rows]


def write_afv(source, destination, fibers, metadata_updates, *, block_points=257, keep_metadata=None):
    """A new AFV with ``source``'s schema, pragmas and metadata (updated by ``metadata_updates``, point and fiber
    counts recomputed) holding ``fibers``: (fibers row dict, native xyz polyline) pairs. Each polyline is stored in
    blocks of at most ``block_points`` points sharing their end points, each block with its R-tree bounds; the row's
    point count, length and bounds are recomputed. ``complete`` is written last. ``keep_metadata`` (keys) limits the
    metadata taken from ``source``, e.g. to its coordinate frame for an unrelated collection; default all. Without a
    ``source`` (None), the format is AFV_SCHEMA and the metadata only ``metadata_updates`` (which then must give the
    coordinate 'frame' and 'root', e.g. open_data_frame)."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'{destination} exists')
    if source is None:
        schema, pragmas, metadata = list(AFV_SCHEMA), [AFV_APPLICATION_ID, AFV_USER_VERSION], {}
    else:
        with sqlite3.connect(f'file:{Path(source).resolve()}?mode=ro', uri=True) as src:
            schema = [sql for (sql,) in src.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND "
                                                      "name NOT LIKE 'block_bounds_%' AND type IN ('table','index')")]
            pragmas = [src.execute(f'PRAGMA {p}').fetchone()[0] for p in ('application_id', 'user_version')]
            metadata = {k: json.loads(v) for k, v in src.execute('SELECT key,value FROM metadata')
                        if keep_metadata is None or k in keep_metadata}
    partial = destination.with_suffix('.partial.afv')
    partial.unlink(missing_ok=True)
    dst = sqlite3.connect(partial)
    dst.execute('PRAGMA journal_mode=OFF')
    dst.execute('PRAGMA synchronous=OFF')
    dst.execute(f'PRAGMA application_id = {int(pragmas[0])}')
    dst.execute(f'PRAGMA user_version = {int(pragmas[1])}')
    for sql in schema:
        dst.execute(sql)
    columns = [r[1] for r in dst.execute('PRAGMA table_info(fibers)')]
    fiber_rows, block_rows, bound_rows, points_total, block_id = [], [], [], 0, 0
    for row, xyz in fibers:
        xyz = np.ascontiguousarray(xyz, dtype='<f8')
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
            raise ValueError(f'Invalid polyline for fiber {row["id"]}')
        row = dict(row, point_count=len(xyz), length=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                   **{f'{bound}_{axis}': float(f(xyz[:, k])) for k, axis in enumerate('xyz')
                      for bound, f in (('min', np.min), ('max', np.max))})
        fiber_rows.append(tuple(row[c] for c in columns))
        points_total += len(xyz)
        for first in range(0, len(xyz)-1, block_points-1):
            block = xyz[first:first+block_points]
            block_id += 1
            block_rows.append((block_id, row['id'], first, block.tobytes()))
            lo, hi = block.min(0), block.max(0)
            bound_rows.append((block_id, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    dst.executemany(f'INSERT INTO fibers ({",".join(columns)}) VALUES ({",".join("?"*len(columns))})', fiber_rows)
    dst.executemany('INSERT INTO blocks (id,fiber_id,first_segment,points) VALUES (?,?,?,?)', block_rows)
    dst.executemany('INSERT INTO block_bounds VALUES (?,?,?,?,?,?,?)', bound_rows)
    metadata.update(metadata_updates, fiber_count=len(fiber_rows), point_count=points_total)
    metadata.pop('complete', None)
    dst.executemany('INSERT INTO metadata VALUES (?, ?)', [(k, json.dumps(v)) for k, v in metadata.items()])
    dst.execute("INSERT INTO metadata VALUES ('complete', 'true')")
    dst.commit()
    dst.close()
    partial.replace(destination)
    return dict(fibers=len(fiber_rows), points=points_total, blocks=block_id)
