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


def fibers(db):
    """One fiber in memory, no per-fiber SQL and no Python coordinate lists."""
    # The UNIQUE(fiber_id,first_segment) index supplies order, without a blob sort.
    query = '''SELECT f.id,f.name,f.family,f.point_count,f.length,
        f.min_x,f.min_y,f.min_z,f.max_x,f.max_y,f.max_z,f.annotation,
        b.first_segment,b.points FROM fibers f
        LEFT JOIN blocks b ON b.fiber_id=f.id ORDER BY f.id,b.first_segment'''
    for _, group in itertools.groupby(db.execute(query), key=lambda row: row[0]):
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
               unbound=False, author=None, batch_size=256, batch_mib=64, progress=True):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise ValueError('Output parent directory must already exist')
    if bool(volume_id) == bool(unbound):
        raise ValueError('Specify either --volume-id or --unbound')
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError('scale-factor must be finite and positive')
    if batch_size < 1 or not math.isfinite(batch_mib) or batch_mib <= 0:
        raise ValueError('Batch sizes must be positive')
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
                    print(f'{count:,}/{metadata["fiber_count"]:,} fibers; '
                          f'{points:,} points; {count/elapsed:,.0f} fibers/s; {elapsed:.1f}s', flush=True)
                    last_report = now

            for row in fibers(input_db):
                size = len(row[-1]) + len(json.dumps(row[-2])) + 64
                if pending and (len(pending) >= batch_size or pending_bytes + size > batch_mib * 1024**2):
                    flush()
                pending.append(prepared(row, collection_uuid, coordinates))
                pending_bytes += size
                count += 1
                points += row[3]
                payload += len(row[-1])
            flush()
            if count != metadata['fiber_count'] or points != metadata['point_count']:
                raise ValueError('AFV catalog totals do not match collection metadata')
            with store.write(expected_revisions={collection_uuid: store.current_object_revision(collection_uuid)['uuid']}) as tx:
                tx.set_meta(collection_uuid, 'import', dict(fibers=count, points=points))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    binding = parser.add_mutually_exclusive_group()
    binding.add_argument('--volume-id', help='Volume id from metadata.json, stored verbatim')
    binding.add_argument('--unbound', action='store_true', help='Explicitly leave authoritative volume identity unknown')
    parser.add_argument('--volume', type=Path, help='CT directory, retained as provenance')
    parser.add_argument('--scale-factor', type=float, default=1)
    parser.add_argument('--author', default=getpass.getuser())
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--batch-mib', type=float, default=64)
    args = vars(parser.parse_args())
    try:
        import_afv(**args)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f'Import failed: {error}\n')


if __name__ == '__main__':
    main()
