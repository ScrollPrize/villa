#!/usr/bin/env python3
"""Correct an explicit volume binding, retaining geometry and revision history.

Dry-run by default. --apply uses checked object-store transactions in bounded
batches; rerunning resumes records still carrying --from-id. Existing revisions
keep the old value. This tool refuses stores with relationships, whose coordinate
metadata dependencies require case-specific review. Close VC3D before applying.

  PYTHONPATH=build/python python scripts/rebind_volume.py objects.vcobj \
      --from-id OLD_VALUE --to-id CATALOG_VOLUME_ID [--apply]
"""
import argparse
import contextlib
import getpass
import json
from pathlib import Path
import sqlite3
import uuid


def rebind(path, old, new, *, apply=False, batch_size=256):
    if not isinstance(new, str) or not new or not isinstance(old, str) or not old:
        raise ValueError('Volume IDs must be nonempty strings')
    if old == new or batch_size < 1:
        raise ValueError('Specify different IDs and a positive batch size')
    with contextlib.closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True,
                                          isolation_level=None)) as reader:
        records = reader.execute("SELECT kind,count(*) FROM objects WHERE json_extract(meta,'$.coordinates.volume_uuid')=? GROUP BY kind", (old,)).fetchall()
        report = {'from_id': old, 'to_id': new, 'matching': dict(records), 'updated': 0}
        if not apply or not records:
            return report
        if reader.execute('SELECT count(*) FROM edges').fetchone()[0]:
            raise ValueError('Store contains relationships; coordinate dependencies need explicit review')
        from vc_object_store import ObjectStore
        with ObjectStore.open(str(path), author=getpass.getuser()) as store:
            after = 0
            while True:
                rows = reader.execute("""SELECT o.id,o.uuid,o.meta,h.revision_uuid
                    FROM objects o JOIN object_heads h ON h.object_uuid=
                    lower(substr(hex(o.uuid),1,8)||'-'||substr(hex(o.uuid),9,4)||'-'||
                          substr(hex(o.uuid),13,4)||'-'||substr(hex(o.uuid),17,4)||'-'||substr(hex(o.uuid),21,12))
                    WHERE o.id>? AND json_extract(o.meta,'$.coordinates.volume_uuid')=?
                    ORDER BY o.id LIMIT ?""", (after, old, batch_size)).fetchall()
                if not rows:
                    break
                changes = [(str(uuid.UUID(bytes=blob)), json.loads(meta)['coordinates'], revision)
                           for _, blob, meta, revision in rows]
                with store.write({identity: revision for identity, _, revision in changes}) as txn:
                    for identity, coordinates, _ in changes:
                        coordinates['volume_uuid'] = new
                        txn.set_meta(identity, 'coordinates', coordinates)
                report['updated'] += len(changes)
                after = rows[-1][0]
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--from-id', required=True)
    parser.add_argument('--to-id', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(rebind(args.database, args.from_id, args.to_id, apply=args.apply), indent=2))
