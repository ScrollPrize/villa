#!/usr/bin/env python3
"""Scale point coordinates in a VC3D point-collection JSON file in place."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale all point coordinates in a VC3D point-collection JSON file.",
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="VC3D point-collection JSON file to update in place.",
    )
    parser.add_argument(
        "factor",
        type=float,
        help="Uniform scale factor applied to each point coordinate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the number of scaled points without writing.",
    )
    return parser.parse_args()


def scale_point_collection(data: dict[str, Any], factor: float) -> int:
    if data.get("vc_pointcollections_json_version") != "1":
        raise ValueError("input is missing VC3D point-collection JSON version 1")

    collections = data.get("collections")
    if not isinstance(collections, dict):
        raise ValueError("input is missing object field 'collections'")

    scaled = 0
    for collection_id, collection in collections.items():
        if not isinstance(collection, dict):
            raise ValueError(f"collection {collection_id!r} is not an object")

        points = collection.get("points")
        if not isinstance(points, dict):
            raise ValueError(f"collection {collection_id!r} is missing object field 'points'")

        for point_id, point in points.items():
            if not isinstance(point, dict):
                raise ValueError(f"point {point_id!r} in collection {collection_id!r} is not an object")

            coords = point.get("p")
            if not (
                isinstance(coords, list)
                and len(coords) == 3
                and all(isinstance(value, (int, float)) for value in coords)
            ):
                raise ValueError(
                    f"point {point_id!r} in collection {collection_id!r} has invalid 'p' coordinates"
                )

            point["p"] = [float(value) * factor for value in coords]
            scaled += 1

    return scaled


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path = path.resolve()
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=4)
            out.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()

    with args.json_path.open("r", encoding="utf-8") as src:
        data = json.load(src)
    if not isinstance(data, dict):
        raise ValueError("input JSON root must be an object")

    scaled = scale_point_collection(data, args.factor)
    if not args.dry_run:
        write_json_atomic(args.json_path, data)

    action = "would scale" if args.dry_run else "scaled"
    print(f"{action} {scaled} point(s) in {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
