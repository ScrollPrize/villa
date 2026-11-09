#!/usr/bin/env python3
"""Stream an Automated Fiber Volume v1 into a NEW object store.

AFV decoding uses the shared native reader; writes use the public Python API.
Requires the existing Python extension (no setup is performed):

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
The --volume path only checks local input availability; it is not used to guess identity.

Paris 4:
  PYTHONPATH=build/python python3 scripts/import_afv.py \
      /mnt/bigpc/tmp/fibers.afv paris4-fibers.vcobj \
      --volume-id 20260411134726 \
      --volume /mnt/bigpc/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr

vc_reader identifies this as PHercParis4/20260411134726. Its catalog volume ID
20260411134726 is stored unchanged as coordinates.volume_uuid.

Every original float64 XYZ sample becomes line_points. Control arrays stay empty:
AFV traces do not contain control points, and none are invented during import.
Names, families, catalog summaries and per-fiber annotations are retained;
the source UUID and declared counts are stored in compact extension provisioning.
Full source metadata, paths, reports and ID lists are not copied into provisioning.
Numeric source IDs and annotations use typed extension rows, not object metadata.
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
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid

from vc_object_store import ObjectStore, AfvSource, fiber_source_extension, fiber_source_row


@contextlib.contextmanager
def open_afv(path):
    # Hold one read snapshot: metadata and blocks must describe the same source.
    source = AfvSource(Path(path).resolve())
    yield source


def fibers(db):
    """Native streaming cursor shared with the C++ bulk import workflow."""
    yield from db


def prepared(row, collection_uuid, coordinates):
    fid, name, family, count, length, bounds, annotation, raw = row
    object_uuid = str(uuid.uuid5(uuid.UUID(collection_uuid), f'fiber:{fid}'))
    record = dict(uuid=object_uuid, kind='fiber', name=name, format_version=4,
        fiber=dict(family=family, control_point_count=0, line_point_count=count,
                   length=length, bbox=bounds),
        meta=dict(coordinates=coordinates))
    geometry = dict(kind='fiber', arrays=[
        dict(name='control_points', dtype='<f8', shape=[0, 3], data=b''),
        dict(name='control_point_ids', dtype='<i8', shape=[0], data=b''),
        dict(name='line_points', dtype='<f8', shape=[count, 3], data=raw)])
    return record, geometry, fiber_source_row(object_uuid, fid, annotation)


def import_afv(source, destination, *, volume_id=None, volume=None, scale_factor=1,
               unbound=False, author=None, batch_size=256, batch_mib=64, progress=True, source_sha256=None):
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
    with open_afv(source) as input_db, tempfile.TemporaryDirectory(
            prefix='.' + destination.name + '-', dir=destination.parent) as staging:
        collection_uuid = input_db.uuid
        source_fibers, source_points = input_db.fiber_count, input_db.point_count
        staged = Path(staging) / 'objects.vcobj'
        with ObjectStore.create(str(staged), author=author or getpass.getuser(),
                                reader_pool_size=1, geometry_cache_bytes=0, spatial_cache_bytes=0) as store:
            source_table = fiber_source_extension(collection_uuid, sha256=source_sha256,
                                                   fiber_count=source_fibers, point_count=source_points)
            store.create_extension_table(source_table)
            pending, pending_bytes = [], 0
            last_report = started

            def flush():
                nonlocal pending_bytes, last_report
                if not pending:
                    return
                with store.write(expected_versions={r['uuid']: None for r, _, _ in pending}) as tx:
                    for record, geometry, _ in pending:
                        tx.upsert_object(record)
                        tx.put_geometry(record['uuid'], geometry)
                    tx.append_extension_rows(source_table['name'], [extension['values'] for _, _, extension in pending])
                pending.clear()
                pending_bytes = 0
                now = time.perf_counter()
                if progress and now - last_report >= 5:
                    elapsed = now - started
                    print(f'{count:,}/{source_fibers:,} fibers; '
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
            if count != source_fibers or points != source_points:
                raise ValueError('AFV catalog totals do not match collection metadata')
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
    parser.add_argument('--volume', type=Path, help='Optional CT directory availability check; never stored in provisioning')
    parser.add_argument('--source-sha256', help='Known source-file SHA-256, if available (not computed during import)')
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
