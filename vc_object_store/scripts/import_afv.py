#!/usr/bin/env python3
"""Stream an Automated Fiber Volume v1 into a NEW object store.

All AFV handling lives here; writes use the public vc_object_store Python API.
Requires the existing Python extension and NumPy (no setup is performed):

  PYTHONPATH=build/python python3 scripts/import_afv.py INPUT.afv OUTPUT.vcobj \
      --volume-id CATALOG_VOLUME_ID --volume /path/to/volume.zarr

--volume-id is the volume's id from the open-data metadata.json, stored verbatim
in coordinates.volume_uuid. It is never hashed or converted to a generated UUID.
Identity does not depend on the cache path or source URL.
The volume opened in VC3D must have the same catalog ID. For a local cache,
set its root meta.json "uuid" to that ID; this script does not modify CT metadata.
--scale-factor is stored units per active volume voxel (1 for native L0, 4 for
an explicitly rebased level-2 volume). Coordinates are NEVER rescaled.
Use --unbound explicitly when the authoritative volume identity is unknown.
The --volume path is provenance only; it is not used to guess identity.

Paris 4:
  PYTHONPATH=build/python python3 scripts/import_afv.py \
      /mnt/bigpc/tmp/fibers.afv paris4-fibers.vcobj \
      --volume-id 20260411134726 \
      --volume /mnt/bigpc/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr

vc_reader identifies this as PHercParis4/20260411134726. Its catalog volume ID
20260411134726 is stored unchanged as coordinates.volume_uuid.

Every original float64 XYZ sample becomes line_points. Two endpoint controls
with stable IDs 0,1 make the polyline editable without duplicating all samples.
Names, families, catalog summaries and per-fiber annotations are retained;
collection metadata/provenance is stored once in a generic collection object.
Trace annotations such as fiber_follow.links remain provenance: they describe
proximity, not validated branch anchors, and are not invented into branch edges.

Uses one ordered SQLite cursor, binary arrays, and bounded checked transactions.
The source stays read-only. A private sibling staging directory keeps failures
from publishing a partial destination. Existing output is never overwritten.
--limit supports repeatable subset benchmarks (not resume or migration).
Compare --batch-size 1 with the default on the same --limit; Python's cProfile
can profile this script. --self-test runs the embedded regression harness.

Measured on Linux arm64, Python 3.14, existing RelWithDebInfo (-O2) extension,
warm NFS input /mnt/bigpc/tmp/fibers.afv and local /tmp output: first 10,000
fibers (3,790,226 points), three sequential runs per batch size. With
--unbound --limit 10000 --batch-size 1: mean 27.257s, min/median/max
27.225/27.266/27.281s. Default batch size 256: mean 4.534s, min/median/max
4.527/4.537/4.538s (6.0x faster). A 1,000-fiber cProfile run attributes 75%
of import time to store writes/commits, 7% to reading/validating AFV blocks.
A separate --limit 100000 run imported 42,315,891 points in 65.46s with
115.2 MiB peak RSS (/usr/bin/time -v); 102 distributed fibers were independently
compared byte-for-byte with the source. These timings exclude no import work.
No rounding, resampling, SQLite durability changes, or core changes are used.
"""
from __future__ import annotations

import argparse
import contextlib
import getpass
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
import time
import uuid

import numpy as np
from vc_object_store import ObjectStore


@contextlib.contextmanager
def open_afv(path):
    # Hold one read snapshot: metadata and blocks must describe the same source.
    with contextlib.closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA cache_size=-32768')
        db.execute('BEGIN')
        if (db.execute('PRAGMA application_id').fetchone()[0],
                db.execute('PRAGMA user_version').fetchone()[0]) != (0x56434643, 1):
            raise ValueError('Expected AFV application ID and format version 1')
        metadata = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
        if metadata.get('complete') is not True:
            raise ValueError('AFV is incomplete')
        uuid.UUID(metadata['uuid'])
        for key in ('fiber_count', 'point_count'):
            if type(metadata[key]) is not int or metadata[key] < 0:
                raise ValueError(f'Invalid AFV {key}')
        frame, root = metadata['frame'], metadata['root']
        for declaration in (frame, root):
            if (declaration.get('vc_open_data_source_coordinate_level', 0) != 0 or
                    declaration.get('vc_open_data_source_coordinate_scale_factor', 1) != 1):
                raise ValueError('Only native L0 AFV coordinates are supported')
        if root.get('scale', 1) != 1:
            raise ValueError('Only native L0 AFV coordinates are supported')
        if root.get('coordinate_order') != 'XYZ' or root.get('coordinate_units') != 'voxel':
            raise ValueError('AFV must declare XYZ voxel coordinates')
        yield db, metadata


def fibers(db, limit=None):
    """One fiber in memory, no per-fiber SQL and no Python coordinate lists."""
    # The UNIQUE(fiber_id,first_segment) index supplies order, without a blob sort.
    query = '''SELECT f.id,f.name,f.family,f.point_count,f.length,
        f.min_x,f.min_y,f.min_z,f.max_x,f.max_y,f.max_z,f.annotation,
        b.first_segment,b.points FROM fibers f
        LEFT JOIN blocks b ON b.fiber_id=f.id ORDER BY f.id,b.first_segment'''
    for _, group in itertools.islice(itertools.groupby(db.execute(query), key=lambda row: row[0]), limit):
        first = next(group)
        fid, name, family, count, length, *rest = first[:12]
        bounds, annotation = rest[:6], rest[6]
        if count < 2 or not math.isfinite(length) or length < 0:
            raise ValueError(f'Invalid catalog entry for fiber {fid}')
        pieces, next_segment, previous = [], 0, None
        for row in itertools.chain((first,), group):
            start, blob = row[-2:]
            if not isinstance(blob, bytes) or len(blob) % 24 or not 2 <= len(blob) // 24 <= 257:
                raise ValueError(f'Invalid block size for fiber {fid}')
            if start != next_segment or (previous is not None and blob[:24] != previous):
                raise ValueError(f'Discontinuous blocks for fiber {fid}')
            pieces.append(blob if previous is None else blob[24:])
            previous = blob[-24:]
            next_segment += len(blob) // 24 - 1
        if next_segment + 1 != count:
            raise ValueError(f'Point count mismatch for fiber {fid}')
        raw = b''.join(pieces)
        points = np.frombuffer(raw, dtype='<f8').reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError(f'Non-finite coordinates in fiber {fid}')
        actual = np.concatenate((points.min(axis=0), points.max(axis=0)))
        if not np.array_equal(actual, bounds):
            raise ValueError(f'Bounds mismatch for fiber {fid}')
        yield fid, name, family, count, length, bounds, json.loads(annotation), raw


def prepared(row, collection_uuid, coordinates):
    fid, name, family, count, length, bounds, annotation, raw = row
    object_uuid = str(uuid.uuid5(uuid.UUID(collection_uuid), f'fiber:{fid}'))
    record = dict(uuid=object_uuid, kind='fiber', name=name, format_version=4,
        fiber=dict(family=family, control_point_count=2, line_point_count=count,
                   length=length, bbox=bounds),
        meta=dict(coordinates=coordinates, source=dict(afv=dict(
            collection_uuid=collection_uuid, fiber_id=fid, annotation=annotation))))
    geometry = dict(kind='fiber', arrays=[
        dict(name='control_points', dtype='<f8', shape=[2, 3], data=raw[:24] + raw[-24:]),
        dict(name='control_point_ids', dtype='<i8', shape=[2], data=struct.pack('<2q', 0, 1)),
        dict(name='line_points', dtype='<f8', shape=[count, 3], data=raw)])
    return record, geometry


def import_afv(source, destination, *, volume_id=None, volume=None, scale_factor=1,
               unbound=False, author=None, batch_size=256, batch_mib=64, limit=None, progress=True):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise ValueError('Output parent directory must already exist')
    if bool(volume_id) == bool(unbound):
        raise ValueError('Specify either --volume-id or --unbound')
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError('scale-factor must be finite and positive')
    if batch_size < 1 or not math.isfinite(batch_mib) or batch_mib <= 0 or (limit is not None and limit < 1):
        raise ValueError('Batch sizes and limit must be positive')
    if volume is not None and not Path(volume).is_dir():
        raise ValueError('Volume path must be an existing directory')
    if volume_id is not None and (not isinstance(volume_id, str) or not volume_id):
        raise ValueError('volume-id must be a nonempty string from metadata.json')
    coordinates = dict(volume_uuid=volume_id if volume_id else None, scale_factor=scale_factor)
    started = time.perf_counter()
    count = points = payload = 0
    with open_afv(source) as (input_db, metadata), tempfile.TemporaryDirectory(
            prefix='.' + destination.name + '-', dir=destination.parent) as staging:
        collection_uuid = str(uuid.UUID(metadata['uuid']))
        staged = Path(staging) / 'objects.vcobj'
        with ObjectStore.create(str(staged), author=author or getpass.getuser(),
                                reader_pool_size=1, geometry_cache_bytes=0, spatial_cache_bytes=0) as store:
            collection = dict(uuid=collection_uuid, kind='collection', name=source.stem,
                meta=dict(coordinates=coordinates, afv=dict(metadata=metadata, source=str(source),
                    volume=str(Path(volume).absolute()) if volume else None)))
            with store.write(expected_revisions={collection_uuid: None}) as tx:
                tx.upsert_object(collection)
            pending, pending_bytes = [], 0
            last_report = started

            def flush():
                nonlocal pending_bytes, last_report
                if not pending:
                    return
                with store.write(expected_revisions={r['uuid']: None for r, _ in pending}) as tx:
                    for record, geometry in pending:
                        tx.upsert_object(record)
                        tx.put_geometry(record['uuid'], geometry)
                pending.clear()
                pending_bytes = 0
                now = time.perf_counter()
                if progress and now - last_report >= 5:
                    elapsed = now - started
                    print(f'{count:,}/{min(limit or metadata["fiber_count"], metadata["fiber_count"]):,} fibers; '
                          f'{points:,} points; {count/elapsed:,.0f} fibers/s; {elapsed:.1f}s', flush=True)
                    last_report = now

            for row in fibers(input_db, limit):
                size = len(row[-1]) + len(json.dumps(row[-2])) + 64
                if pending and (len(pending) >= batch_size or pending_bytes + size > batch_mib * 1024**2):
                    flush()
                pending.append(prepared(row, collection_uuid, coordinates))
                pending_bytes += size
                count += 1
                points += row[3]
                payload += len(row[-1])
            flush()
            if (limit is None or count == metadata['fiber_count']) and (
                    count != metadata['fiber_count'] or points != metadata['point_count']):
                raise ValueError('AFV catalog totals do not match collection metadata')
            with store.write(expected_revisions={collection_uuid: store.current_object_revision(collection_uuid)['uuid']}) as tx:
                tx.set_meta(collection_uuid, 'import', dict(fibers=count, points=points,
                    complete=count == metadata['fiber_count'], limit=limit))
        # A closed store has checkpointed its WAL. Link publishes atomically
        # without a check/replace race; staging is on the destination filesystem.
        os.link(staged, destination)
    elapsed = time.perf_counter() - started
    result = dict(fibers=count, points=points, seconds=elapsed, fibers_per_second=count/elapsed,
                  payload_mib=payload/1024**2, output_bytes=destination.stat().st_size,
                  coordinates=coordinates, destination=str(destination))
    if progress:
        print(json.dumps(result, indent=2), flush=True)
    return result


def self_test():
    """Small real-store regressions, kept here so AFV has no other code surface."""
    def fixture(path):
        points = np.arange(780, dtype='<f8').reshape(-1, 3) / 8
        points[0, 0] = -0.0
        metadata = dict(uuid='1482bcc8-f93e-4c3b-b380-2ddf369318f4', complete=True,
            fiber_count=2, point_count=262, frame={},
            root=dict(coordinate_order='XYZ', coordinate_units='voxel'), provenance={'test': True})
        with sqlite3.connect(path) as db:
            db.executescript('''PRAGMA application_id=1447249475; PRAGMA user_version=1;
                CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT,family TEXT,point_count INTEGER,length REAL,
                    min_x REAL,max_x REAL,min_y REAL,max_y REAL,min_z REAL,max_z REAL,annotation TEXT);
                CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER,first_segment INTEGER,points BLOB,
                    UNIQUE(fiber_id,first_segment));''')
            db.executemany('INSERT INTO metadata VALUES(?,?)', [(k, json.dumps(v)) for k, v in metadata.items()])
            for fid, xyz in ((0, points), (9, points[-2:])):
                lo, hi = xyz.min(axis=0), xyz.max(axis=0)
                db.execute('INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    (fid, 'same name', 'H' if fid == 0 else 'V', len(xyz), 7.25,
                     lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], json.dumps({'fiber_follow': {'links': [-1, 0]}})))
                for start in range(0, len(xyz) - 1, 256):
                    db.execute('INSERT INTO blocks(fiber_id,first_segment,points) VALUES(?,?,?)',
                               (fid, start, xyz[start:start + 257].tobytes()))
        return metadata, points

    with tempfile.TemporaryDirectory(prefix='afv-import-test-') as directory:
        root = Path(directory)
        source = root / 'input.afv'
        metadata, points = fixture(source)
        original = source.read_bytes()
        output = root / 'objects.vcobj'
        result = import_afv(source, output, volume_id='20260411134726', scale_factor=4,
                            batch_size=1, progress=False)
        assert result['fibers'] == 2 and result['points'] == 262
        assert source.read_bytes() == original
        with ObjectStore.open(str(output), read_only=True) as store:
            objects = store.objects_by_kind('fiber')
            assert len(objects) == 2 and objects[0]['uuid'] != objects[1]['uuid']
            for record in objects:
                fid = record['meta']['source']['afv']['fiber_id']
                expected = points if fid == 0 else points[-2:]
                revision = store.current_geometry(record['uuid'])['uuid']
                assert store.read_geometry_array(revision, 'line_points') == expected.tobytes()
                assert store.read_geometry_array(revision, 'control_points') == expected[[0, -1]].tobytes()
                assert store.read_geometry_array(revision, 'control_point_ids') == struct.pack('<2q', 0, 1)
                assert len(store.object_history(record['uuid'])) == 1
                assert record['meta']['coordinates'] == dict(
                    volume_uuid='20260411134726', scale_factor=4)
                assert record['fiber']['family'] == ('H' if fid == 0 else 'V')
                assert record['meta']['source']['afv']['annotation']['fiber_follow']['links'] == [-1, 0]
                store.export_geometry(record['uuid'], root / (record['uuid'] + '.json'))
            collection = store.object(metadata['uuid'])
            assert collection['meta']['afv']['metadata'] == metadata
            assert collection['meta']['import']['complete']
        try:
            import_afv(source, output, unbound=True, progress=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output must not be overwritten')
        subset = root / 'subset.vcobj'
        import_afv(source, subset, unbound=True, limit=1, batch_mib=.001, progress=False)
        with ObjectStore.open(str(subset), read_only=True) as store:
            assert store.object_count('fiber') == 1
            assert not store.object(metadata['uuid'])['meta']['import']['complete']
        failures = [
            'PRAGMA user_version=2',
            "UPDATE metadata SET value='false' WHERE key='complete'",
            "UPDATE metadata SET value='263' WHERE key='point_count'",
            'UPDATE blocks SET first_segment=255 WHERE first_segment=256',
            'DELETE FROM blocks WHERE fiber_id=9',
            'UPDATE blocks SET points=zeroblob(25) WHERE id=1',
            'UPDATE blocks SET points=zeroblob(length(points)) WHERE first_segment=256',
            'UPDATE fibers SET point_count=261 WHERE id=0',
            'UPDATE fibers SET max_x=max_x+1 WHERE id=0',
            "UPDATE metadata SET value=json_set(value, '$.scale', 2) WHERE key='root'",
        ]
        for i, mutation in enumerate(failures):
            broken = root / f'broken-{i}.afv'
            broken.write_bytes(original)
            with sqlite3.connect(broken) as db:
                db.execute(mutation)
            target = root / f'failed-{i}.vcobj'
            try:
                import_afv(broken, target, unbound=True, batch_size=1, progress=False)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Expected rejection: {mutation}')
            assert not target.exists(), 'Failure must not publish even previously committed batches'
            assert not list(root.glob('.' + target.name + '-*')), 'Failure must remove staging files'
    print('AFV importer self-test passed (exact bytes, metadata, checked history, subset, malformed input, atomic output)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, nargs='?')
    parser.add_argument('destination', type=Path, nargs='?')
    binding = parser.add_mutually_exclusive_group()
    binding.add_argument('--volume-id', help='Volume id from metadata.json, stored verbatim')
    binding.add_argument('--unbound', action='store_true', help='Explicitly leave authoritative volume identity unknown')
    parser.add_argument('--volume', type=Path, help='CT directory, retained as provenance')
    parser.add_argument('--scale-factor', type=float, default=1)
    parser.add_argument('--author', default=getpass.getuser())
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--batch-mib', type=float, default=64)
    parser.add_argument('--limit', type=int, help='Import first N fibers in ID order for benchmarking')
    parser.add_argument('--self-test', action='store_true')
    args = vars(parser.parse_args())
    if args.pop('self_test'):
        self_test()
        return
    if args['source'] is None or args['destination'] is None:
        parser.error('source and destination are required')
    try:
        import_afv(**args)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f'Import failed: {error}\n')


if __name__ == '__main__':
    main()
