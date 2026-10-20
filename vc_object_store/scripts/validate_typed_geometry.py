#!/usr/bin/env python3
"""Check typed geometry against real native sources, using temporary databases."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if build_python := os.environ.get("VC_OBJECT_STORE_BUILD_PYTHON_PATH"):
    sys.path.insert(0, str(Path(build_python).resolve()))

from vc_object_store import ObjectStore


def validate(kind, source, root):
    source_collections = {}
    if kind == "point_collection":
        source_collections = {
            value["uuid"]: value
            for value in json.loads(source.read_text())["collections"].values()
        }
    with ObjectStore.create(str(root / "input.vcobj"), author="geometry-validation") as db:
        report = db.ingest_geometry_source(kind, source)
        if report["problems"]:
            raise RuntimeError(report["problems"])
        records = db.objects_by_kind(kind)
        if not records:
            raise RuntimeError(f"source contains no {kind} objects: {source}")
        point_count = 0
        for record in records:
            uid = record["uuid"]
            with db.geometry_snapshot() as snapshot:
                view = snapshot.geometry(uid)
                owned = view.load()
                revision = view.revision
                if kind == "point_collection":
                    original = source_collections[uid]["points"]
                    assert owned.ids == sorted(int(key) for key in original)
                    assert dict(zip(owned.ids, owned.points)) == {
                        int(key): value["p"] for key, value in original.items()
                    }
                    point_count += len(owned.points)
                    for index in sorted({0, len(owned.ids) - 1}):
                        if owned.ids:
                            assert view.point(owned.ids[index]) == owned.points[index]
                elif kind == "fiber":
                    point_count += len(owned.control_points) + len(owned.line_points)
                    if source.is_file():
                        original = json.loads(source.read_text())
                        for field in ("control_points", "line_points"):
                            values = [
                                p["position"] if isinstance(p, dict) else p
                                for p in original.get(field, [])
                            ]
                            assert values == getattr(owned, field)
                else:
                    point_count += len(owned.points)
                if kind == "patch" and view.rows and view.columns:
                    row, column = min(127, view.rows - 1), min(127, view.columns - 1)
                    rows, columns = min(4, view.rows - row), min(4, view.columns - column)
                    tile = view.read_tile(row, column, rows, columns)
                    for r in range(rows):
                        for c in range(columns):
                            assert tile.vertex(r, c) == owned.vertex(row + r, column + c)
            clone = db.upsert_object({"kind": kind})
            copy = db.put_geometry(clone, owned)
            for array in revision["arrays"]:
                assert db.read_geometry_array(copy, array["name"]) == db.read_geometry_array(
                    revision["uuid"], array["name"]
                ), (source, uid, array["name"])

            # Native output is reconstructed from source annotations/topology.
            output = root / uid
            db.export_geometry(uid, output)
            if kind == "point_collection":
                exported = next(iter(json.loads(output.read_text())["collections"].values()))
                assert {int(k): v["p"] for k, v in exported["points"].items()} == dict(
                    zip(owned.ids, owned.points)
                )
            elif kind in ("patch", "track"):
                with ObjectStore.create(str(root / f"{uid}.vcobj"), author="geometry-validation") as other:
                    imported = other.ingest_geometry_source(kind, output)
                    assert not imported["problems"], imported
                    imported_id = other.objects_by_kind(kind)[0]["uuid"]
                    imported_revision = other.current_geometry(imported_id)["uuid"]
                    for array in revision["arrays"]:
                        assert other.read_geometry_array(imported_revision, array["name"]) == db.read_geometry_array(
                            revision["uuid"], array["name"]
                        )
            else:
                exported = json.loads(output.read_text())
                for field in ("control_points", "line_points"):
                    values = [p["position"] if isinstance(p, dict) else p for p in exported[field]]
                    assert values == getattr(owned, field)
        return {"kind": kind, "source": str(source), "objects": len(records),
                "points": point_count, "exact_arrays_and_native_export": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("point-collections", "fiber", "patch", "tracks"):
        parser.add_argument(f"--{flag}", action="append", type=Path, default=[])
    args = parser.parse_args()
    sources = [(kind, source) for kind, paths in (
        ("point_collection", args.point_collections), ("fiber", args.fiber),
        ("patch", args.patch), ("track", args.tracks),
    ) for source in paths]
    if not sources:
        parser.error("supply at least one source")
    for kind, source in sources:
        with tempfile.TemporaryDirectory(prefix="vc-typed-validation-") as directory:
            print(json.dumps(validate(kind, source, Path(directory))), flush=True)


if __name__ == "__main__":
    main()
