#!/usr/bin/env python3
"""Assign persistent UUIDs to a copy of the initial graph objects.

The input directory must contain verified_patches/, fibers/, and the three
top-level point-collection JSON files. An identity manifest is published before
any object is changed, so an interrupted run can resume with the same UUIDs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
import uuid
from pathlib import Path


PCL_FILES = ("abs_winding.json", "relative_windings.json", "sean_fiber_pcls.json")
MANIFEST_NAME = "graph_object_uuid_manifest.json"
UUID_TEXT = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict, indent: int) -> None:
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=indent)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def is_uuid(value: object) -> bool:
    return isinstance(value, str) and UUID_TEXT.fullmatch(value) is not None


def inventory(root: Path) -> tuple[list[Path], list[Path], dict[str, list[str]]]:
    patch_root = root / "verified_patches"
    fiber_root = root / "fibers"
    if not patch_root.is_dir() or not fiber_root.is_dir():
        raise ValueError("Expected verified_patches/ and fibers/ in the copied dataset")
    patches = sorted(path for path in patch_root.iterdir() if path.is_dir())
    fibers = sorted(fiber_root.glob("*.json"))
    if not patches or not fibers:
        raise ValueError("Patch or fiber folder is empty")
    collections: dict[str, list[str]] = {}
    for file_name in PCL_FILES:
        path = root / file_name
        doc = read_json(path)
        if doc.get("vc_pointcollections_json_version") != "1" or not isinstance(
            doc.get("collections"), dict
        ):
            raise ValueError(f"Invalid point-collection document: {path}")
        collections[file_name] = sorted(doc["collections"])
    for path in patches:
        if not (path / "meta.json").is_file():
            raise ValueError(f"Patch is missing meta.json: {path}")
    return patches, fibers, collections


def create_manifest(
    root: Path, patches: list[Path], fibers: list[Path], collections: dict[str, list[str]]
) -> dict:
    manifest: dict = {
        "version": 1,
        "patches": {},
        "fibers": {},
        "point_collections": {},
    }
    used: set[str] = set()

    def assign(old: object, context: str) -> str:
        if is_uuid(old):
            result = str(old)
        else:
            result = str(uuid.uuid4())
        if result.lower() in used:
            raise ValueError(f"Duplicate object UUID at {context}: {result}")
        used.add(result.lower())
        return result

    for patch in patches:
        path = patch / "meta.json"
        old = read_json(path).get("uuid")
        if not isinstance(old, str) or not old:
            raise ValueError(f"Patch is missing its former identity: {path}")
        key = patch.name
        manifest["patches"][key] = {"old_uuid": old, "uuid": assign(old, str(path))}
    for path in fibers:
        doc = read_json(path)
        if doc.get("type") != "vc3d_fiber":
            raise ValueError(f"Not a vc3d_fiber: {path}")
        old = doc.get("uuid")
        if old is not None and not is_uuid(old):
            raise ValueError(f"Invalid existing fiber UUID: {path}")
        manifest["fibers"][path.name] = {"old_uuid": old, "uuid": assign(old, str(path))}
    for file_name, ids in collections.items():
        doc = read_json(root / file_name)
        records = {}
        for collection_id in ids:
            old = doc["collections"][collection_id].get("uuid")
            if old is not None and not is_uuid(old):
                raise ValueError(f"Invalid existing collection UUID: {file_name}:{collection_id}")
            records[collection_id] = {
                "old_uuid": old,
                "uuid": assign(old, f"{file_name}:{collection_id}"),
            }
        manifest["point_collections"][file_name] = records
    return manifest


def validate_manifest(
    manifest: dict, patches: list[Path], fibers: list[Path], collections: dict[str, list[str]]
) -> None:
    if manifest.get("version") != 1:
        raise ValueError("Unknown UUID manifest version")
    expected = (
        {path.name for path in patches},
        {path.name for path in fibers},
        set(collections),
    )
    actual = (
        set(manifest["patches"]),
        set(manifest["fibers"]),
        set(manifest["point_collections"]),
    )
    if actual != expected:
        raise ValueError("Manifest does not match the copied object inventory")
    used: set[str] = set()
    for file_name, ids in collections.items():
        if set(manifest["point_collections"][file_name]) != set(ids):
            raise ValueError(f"Manifest does not match collections in {file_name}")
    records = list(manifest["patches"].values()) + list(manifest["fibers"].values())
    records += [
        record
        for per_file in manifest["point_collections"].values()
        for record in per_file.values()
    ]
    for record in records:
        value = record["uuid"]
        if not is_uuid(value) or value.lower() in used:
            raise ValueError(f"Invalid or duplicate UUID in manifest: {value}")
        used.add(value.lower())


def apply_record(path: Path, record: dict, indent: int) -> None:
    doc = read_json(path)
    current = doc.get("uuid")
    if current not in (record["old_uuid"], record["uuid"]):
        raise ValueError(f"UUID changed since manifest creation: {path}")
    if current != record["uuid"]:
        doc["uuid"] = record["uuid"]
        write_json_atomic(path, doc, indent)


def apply_manifest(root: Path, manifest: dict) -> None:
    for directory, record in manifest["patches"].items():
        apply_record(root / "verified_patches" / directory / "meta.json", record, 4)
    for file_name, record in manifest["fibers"].items():
        apply_record(root / "fibers" / file_name, record, 2)
    for file_name, records in manifest["point_collections"].items():
        path = root / file_name
        doc = read_json(path)
        changed = False
        for collection_id, record in records.items():
            collection = doc["collections"][collection_id]
            current = collection.get("uuid")
            if current not in (record["old_uuid"], record["uuid"]):
                raise ValueError(f"UUID changed since manifest creation: {file_name}:{collection_id}")
            if current != record["uuid"]:
                collection["uuid"] = record["uuid"]
                changed = True
        if changed:
            write_json_atomic(path, doc, 4)


def verify(root: Path, manifest: dict) -> None:
    for directory, record in manifest["patches"].items():
        if read_json(root / "verified_patches" / directory / "meta.json").get("uuid") != record["uuid"]:
            raise ValueError(f"Patch verification failed: {directory}")
    fiber_names = set(manifest["fibers"])
    for file_name, record in manifest["fibers"].items():
        doc = read_json(root / "fibers" / file_name)
        if doc.get("uuid") != record["uuid"]:
            raise ValueError(f"Fiber verification failed: {file_name}")
        for branch in doc.get("branches", []):
            if branch.get("branch_file") not in fiber_names:
                raise ValueError(f"Unresolved branch in {file_name}: {branch.get('branch_file')}")
    for file_name, records in manifest["point_collections"].items():
        doc = read_json(root / file_name)
        for collection_id, record in records.items():
            if doc["collections"][collection_id].get("uuid") != record["uuid"]:
                raise ValueError(f"Collection verification failed: {file_name}:{collection_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Copied object folder to update in place")
    args = parser.parse_args()
    root = args.root.resolve()
    patches, fibers, collections = inventory(root)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = create_manifest(root, patches, fibers, collections)
    validate_manifest(manifest, patches, fibers, collections)
    if not manifest_path.exists():
        write_json_atomic(manifest_path, manifest, 2)
    apply_manifest(root, manifest)
    verify(root, manifest)
    print(
        f"Verified {len(patches)} patches, {len(fibers)} fibers, "
        f"{sum(map(len, collections.values()))} point collections; "
        f"manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
