#!/usr/bin/env python3
"""Verify typed and native roundtrips on real sources without modifying them."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if build_python := os.environ.get("VC_OBJECT_STORE_BUILD_PYTHON_PATH"):
    sys.path.insert(0, str(Path(build_python).resolve()))
from vc_object_store import ObjectStore


def ingest(store, kind, source):
    report = store.ingest_geometry_source(kind, source)
    if report["problems"]:
        raise RuntimeError(report["problems"])
    records = store.objects_by_kind(kind)
    if not records:
        raise RuntimeError(f"source contains no {kind} objects: {source}")
    return records


def arrays(store, revision):
    return {a["name"]: store.read_geometry_array(revision["uuid"], a["name"])
            for a in revision["arrays"]}


def validate(kind, source, root):
    with ObjectStore.create(str(root / "input.vcobj"), author="validation") as store:
        records = ingest(store, kind, source)
        point_count = 0
        for record in records:
            uid = record["uuid"]
            with store.geometry_snapshot() as snapshot:
                view = snapshot.geometry(uid)
                owned, revision = view.load(), view.revision
            point_count += len(owned.control_points) + len(owned.line_points) if kind == "fiber" else len(owned.points)
            expected = arrays(store, revision)
            clone = store.upsert_object({"kind": kind})
            copy = store.put_geometry(clone, owned)
            assert arrays(store, store.geometry_revision(copy)) == expected
            output = root / uid
            store.export_geometry(uid, output)
            with ObjectStore.create(str(root / f"{uid}.vcobj"), author="validation") as other:
                imported = ingest(other, kind, output)[0]["uuid"]
                assert arrays(other, other.current_geometry(imported)) == expected
        return dict(kind=kind, source=str(source), objects=len(records), points=point_count, exact_arrays_and_native_export=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = {"point-collections": "point_collection", "fiber": "fiber", "patch": "patch", "tracks": "track"}
    for flag in flags:
        parser.add_argument(f"--{flag}", action="append", type=Path, default=[])
    args = vars(parser.parse_args())
    if not any(args.values()):
        parser.error("supply at least one source")
    for flag, kind in flags.items():
        for source in args[flag.replace('-', '_')]:
            with tempfile.TemporaryDirectory(prefix="vc-typed-validation-") as directory:
                print(json.dumps(validate(kind, source, Path(directory))), flush=True)


if __name__ == "__main__":
    main()
