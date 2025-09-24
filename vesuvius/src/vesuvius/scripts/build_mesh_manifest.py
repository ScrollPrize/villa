#!/usr/bin/env python3
"""Generate a mesh manifest from a directory of OBJ/PLY files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None


def _parse_transform(path: Optional[str], values: Optional[Iterable[float]]) -> Optional[list]:
    if path and values:
        raise ValueError("Specify either --transform-file or --transform, not both")
    if path:
        tf_path = Path(path)
        with tf_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "transform" in data:
            data = data["transform"]
        return _normalise_transform(data)
    if values:
        return _normalise_transform(list(values))
    return None


def _normalise_transform(data) -> list:
    import numpy as np

    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        if array.size == 12:
            array = array.reshape(3, 4)
        elif array.size == 16:
            array = array.reshape(4, 4)
        else:
            raise ValueError("Transform list must contain 12 or 16 values")
    if array.ndim != 2:
        raise ValueError("Transform must be a 2D matrix")
    if array.shape == (3, 4):
        array = np.vstack([array, np.array([0.0, 0.0, 0.0, 1.0], dtype=float)])
    if array.shape != (4, 4):
        raise ValueError("Transform must be 3x4 or 4x4")
    return array.tolist()


def _parse_attributes(attrs: Optional[Iterable[str]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    if not attrs:
        return result
    for entry in attrs:
        if "=" not in entry:
            raise ValueError(f"Invalid attribute '{entry}'. Use key=value format.")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Try to parse numbers/booleans via json
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        result[key] = parsed
    return result


def _relative_path(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mesh manifest from a directory of OBJ/PLY files.")
    parser.add_argument("path", help="Directory containing mesh files")
    parser.add_argument("output", help="Output manifest file (YAML or JSON)")
    parser.add_argument("--source-volume", required=True, help="Volume ID associated with all meshes")
    parser.add_argument("--transform", nargs=12, type=float, help="Row-major 3x4/4x4 transform values (supply 12 or 16 numbers)")
    parser.add_argument("--transform-file", help="JSON file containing a 3x4 or 4x4 transform matrix")
    parser.add_argument("--attribute", action="append", help="Extra attribute in key=value format (repeatable)")
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="Manifest output format")
    parser.add_argument("--mesh-dir", help="Optional root to store relative paths (defaults to input path)")

    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.is_dir():
        raise SystemExit(f"Mesh directory not found: {root}")

    mesh_root = Path(args.mesh_dir).resolve() if args.mesh_dir else root
    transform = _parse_transform(args.transform_file, args.transform)
    attributes = _parse_attributes(args.attribute)

    entries = []
    for path in sorted(root.glob("**/*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".obj", ".ply"}:
            continue
        mesh_id = path.stem
        entry = {
            "mesh_id": mesh_id,
            "path": _relative_path(mesh_root, path.resolve()),
            "source_volume": args.source_volume,
        }
        if transform is not None:
            entry["transform"] = transform
        if attributes:
            entry["attributes"] = dict(attributes)
        entries.append(entry)

    if not entries:
        raise SystemExit("No OBJ/PLY files found in the specified directory.")

    manifest = {"meshes": entries}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        if yaml is None:
            raise SystemExit("PyYAML is required to write YAML manifests; install 'pyyaml' or use --format json")
        with output_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(manifest, fh, sort_keys=False)

    print(f"Manifest written to {output_path}")


if __name__ == "__main__":
    main()
