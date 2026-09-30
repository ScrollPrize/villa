#!/usr/bin/env python3
"""Build a graph edge import manifest from a UUID-converted dataset copy.

The input directory must already have been converted by
assign_graph_object_uuids.py: it contains fibers/*.json with top-level fiber
UUIDs and graph_object_uuid_manifest.json. No input file is changed. The only
output is graph_edge_import_manifest.json in the same directory, written
atomically.

Edges produced:

* fiber_branch / fiber_adjacent_branch: the reciprocal ``branches`` and
  ``adjacent_branches`` records of vc3d_fiber files, one undirected edge per
  logical link. Peers are resolved only by ``branch_file`` basename;
  ``branch_fiber_id`` is a stale VC3D runtime ID and is kept only in the
  source trace.
* fiber_patch_explicit_target: ``explicit_link:<patch id>`` fiber tags,
  directed fiber -> patch.

Inconsistent or unresolvable input is listed under ``problems`` instead of
being repaired. If the output already exists, edge UUIDs are reused for edges
with the same canonical key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from assign_graph_object_uuids import MANIFEST_NAME, is_uuid, read_json, write_json_atomic


EDGE_MANIFEST_NAME = "graph_edge_import_manifest.json"
EDGE_MANIFEST_VERSION = 1
BRANCH_ARRAYS = {"branches": "fiber_branch", "adjacent_branches": "fiber_adjacent_branch"}
EXPLICIT_KIND = "fiber_patch_explicit_target"
EXPLICIT_LINK_TAG_PREFIX = "explicit_link:"
# Same tolerance VC3D uses when it checks a stored branch position against the
# owning fiber's control point (pointsApproximatelyEqual).
CONTROL_POINT_TOLERANCE = 1.0e-6
REQUIRED_BRANCH_FIELDS = (
    "control_point_index",
    "branch_file",
    "branch_control_point_index",
    "control_point_position",
    "control_point_direction",
    "branch_control_point_position",
    "branch_control_point_direction",
)


class Fiber:
    def __init__(self, name: str, doc: dict) -> None:
        self.name = name
        self.doc = doc
        self.uuid = str(doc["uuid"])
        self.generation = doc.get("generation")
        points = doc.get("control_points")
        self.control_points = points if isinstance(points, list) else []


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_vec3(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def control_point_position(point: object) -> object:
    if isinstance(point, dict):
        return point.get("position")
    return point


def load_inputs(root: Path) -> tuple[dict, dict[str, Fiber]]:
    """Read the node manifest and all fibers; refuse if their UUIDs disagree."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"Missing node UUID manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("version") != 1:
        raise ValueError("Unknown node UUID manifest version")
    for section in ("patches", "fibers", "point_collections"):
        if not isinstance(manifest.get(section), dict):
            raise ValueError(f"Node manifest lacks the {section} map")
    fiber_root = root / "fibers"
    if not fiber_root.is_dir():
        raise ValueError(f"Missing fibers/ folder: {fiber_root}")
    paths = sorted(fiber_root.glob("*.json"))
    names = {path.name for path in paths}
    if names != set(manifest["fibers"]):
        missing = sorted(set(manifest["fibers"]) - names)[:5]
        extra = sorted(names - set(manifest["fibers"]))[:5]
        raise ValueError(
            f"Fiber files do not match the node manifest (missing {missing}, unlisted {extra})"
        )
    fibers: dict[str, Fiber] = {}
    used: dict[str, str] = {}
    for path in paths:
        doc = read_json(path)
        if doc.get("type") != "vc3d_fiber":
            raise ValueError(f"Not a vc3d_fiber: {path}")
        expected = manifest["fibers"][path.name].get("uuid")
        if not is_uuid(doc.get("uuid")) or doc["uuid"] != expected:
            raise ValueError(
                f"Fiber UUID does not match the node manifest: {path} "
                f"(file {doc.get('uuid')!r}, manifest {expected!r})"
            )
        key = doc["uuid"].lower()
        if key in used:
            raise ValueError(f"Duplicate fiber UUID in {used[key]} and {path.name}")
        used[key] = path.name
        fibers[path.name] = Fiber(path.name, doc)
    for name, record in manifest["patches"].items():
        if not is_uuid(record.get("uuid")):
            raise ValueError(f"Invalid patch UUID in node manifest: {name}")
    return manifest, fibers


def source_trace(fiber: Fiber, array: str, index: int, record: object) -> dict:
    trace = {"fiber_file": fiber.name, "array": array, "index": index}
    if isinstance(record, dict):
        trace["branch_file"] = record.get("branch_file")
        trace["branch_fiber_id"] = record.get("branch_fiber_id")
    return trace


def collect_branch_records(
    fibers: dict[str, Fiber], problems: list[dict]
) -> list[dict]:
    """Validate every branch record and return the resolvable ones."""
    records = []
    for fiber in fibers.values():
        for array, kind in BRANCH_ARRAYS.items():
            if array not in fiber.doc:
                continue
            values = fiber.doc[array]
            if not isinstance(values, list):
                problems.append({
                    "code": "invalid_record",
                    "detail": f"{array} is not an array",
                    "sources": [{"fiber_file": fiber.name, "array": array}],
                })
                continue
            for index, record in enumerate(values):
                trace = source_trace(fiber, array, index, record)
                if not isinstance(record, dict):
                    problems.append({"code": "invalid_record", "detail": "not an object",
                                     "sources": [trace]})
                    continue
                missing = [field for field in REQUIRED_BRANCH_FIELDS if field not in record]
                pending = record.get("pending", False)
                if missing or not isinstance(record["branch_file"], str) or not isinstance(pending, bool):
                    problems.append({
                        "code": "invalid_record",
                        "detail": f"missing or malformed fields: {missing or ['branch_file/pending']}",
                        "sources": [trace],
                    })
                    continue
                vectors = [f for f in REQUIRED_BRANCH_FIELDS[3:] if not is_vec3(record[f])]
                local_index = record["control_point_index"]
                remote_index = record["branch_control_point_index"]
                if vectors or not is_index(local_index) or not is_index(remote_index):
                    problems.append({
                        "code": "invalid_record",
                        "detail": f"non-integer index or non-3-vector fields: {vectors}",
                        "sources": [trace],
                    })
                    continue
                target_name = Path(record["branch_file"]).name
                target = fibers.get(target_name)
                if target is None:
                    problems.append({"code": "missing_target_file", "branch_file": target_name,
                                     "sources": [trace]})
                    continue
                bad_ends = []
                if not 0 <= local_index < len(fiber.control_points):
                    bad_ends.append({"side": "local", "fiber_file": fiber.name,
                                     "control_point_index": local_index,
                                     "control_point_count": len(fiber.control_points)})
                if not 0 <= remote_index < len(target.control_points):
                    bad_ends.append({"side": "remote", "fiber_file": target.name,
                                     "control_point_index": remote_index,
                                     "control_point_count": len(target.control_points)})
                if bad_ends:
                    problems.append({"code": "index_out_of_range", "endpoints": bad_ends,
                                     "sources": [trace]})
                    continue
                if target.uuid == fiber.uuid:
                    problems.append({"code": "self_link", "sources": [trace]})
                    continue
                stored = control_point_position(fiber.control_points[local_index])
                if not is_vec3(stored) or math.dist(stored, record["control_point_position"]) > CONTROL_POINT_TOLERANCE:
                    # VC3D rejects such a record on load ("local CP position mismatch").
                    problems.append({
                        "code": "control_point_position_mismatch",
                        "fiber_file": fiber.name,
                        "control_point_index": local_index,
                        "control_point": stored,
                        "stored_in_record": record["control_point_position"],
                        "sources": [trace],
                    })
                records.append({
                    "kind": kind,
                    "owner": fiber,
                    "target": target,
                    "local": (fiber.uuid, local_index),
                    "remote": (target.uuid, remote_index),
                    "record": record,
                    "pending": pending,
                    "trace": trace,
                })
    return records


def endpoint_sort_key(end: tuple[str, int]) -> tuple[str, int]:
    return (end[0].lower(), end[1])


def fiber_endpoint(fiber: Fiber, index: int, position: list, direction: list) -> dict:
    return {
        "fiber_uuid": fiber.uuid,
        "control_point_index": index,
        "fiber_generation": fiber.generation,
        "position": position,
        "direction": direction,
        "fiber_file": fiber.name,
    }


def branch_key(kind: str, a: tuple[str, int], b: tuple[str, int]) -> str:
    return f"{kind}|{a[0].lower()}#{a[1]}|{b[0].lower()}#{b[1]}"


def build_branch_edges(records: list[dict], problems: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for item in records:
        pair = tuple(sorted((item["local"], item["remote"]), key=endpoint_sort_key))
        groups[pair].append(item)
    edges = []
    for pair in sorted(groups, key=lambda p: (endpoint_sort_key(p[0]), endpoint_sort_key(p[1]))):
        items = groups[pair]
        traces = [item["trace"] for item in items]
        kinds = sorted({item["kind"] for item in items})
        if len(kinds) > 1:
            problems.append({"code": "kind_mismatch", "kinds": kinds, "sources": traces})
            continue
        kind = kinds[0]
        a, b = pair
        # Each side is the set of records stored by the fiber that owns that endpoint.
        sides = [[item for item in items if item["local"] == end] for end in (a, b)]
        if any(len(side) > 1 for side in sides):
            identical = all(item["record"] == side[0]["record"] for side in sides for item in side)
            problems.append({"code": "duplicate_record", "kind": kind, "records": len(items),
                             "identical_copies": identical, "sources": traces})
        first = [side[0] if side else None for side in sides]
        endpoints = []
        for end, own, peer in ((a, first[0], first[1]), (b, first[1], first[0])):
            fiber = own["owner"] if own else peer["target"]
            if own:
                pos = own["record"]["control_point_position"]
                direction = own["record"]["control_point_direction"]
            else:
                pos = peer["record"]["branch_control_point_position"]
                direction = peer["record"]["branch_control_point_direction"]
            endpoints.append(fiber_endpoint(fiber, end[1], pos, direction))
        pending_values = [item["pending"] for item in items]
        status = "reciprocated" if all(first) else "unreciprocated"
        if status == "unreciprocated":
            present = first[0] or first[1]
            problems.append({
                "code": "missing_reciprocal",
                "kind": kind,
                "stored_by": present["owner"].name,
                "missing_from": present["target"].name,
                "sources": traces,
            })
        else:
            differences = {}
            if len(set(pending_values)) > 1:
                differences["pending"] = {
                    f"{item['trace']['fiber_file']}[{item['trace']['index']}]": item["pending"]
                    for item in items
                }
            for own, peer, end in ((first[0], first[1], "a"), (first[1], first[0], "b")):
                for field in ("position", "direction"):
                    owner_value = own["record"][f"control_point_{field}"]
                    peer_value = peer["record"][f"branch_control_point_{field}"]
                    if owner_value != peer_value:
                        differences[f"endpoint_{end}_{field}"] = {
                            own["owner"].name: owner_value,
                            peer["owner"].name: peer_value,
                        }
            if differences:
                deltas = {
                    field: max(abs(x - y) for x, y in zip(*values.values()))
                    for field, values in differences.items()
                    if field != "pending"
                }
                problems.append({"code": "reciprocal_disagreement", "kind": kind,
                                 "differences": differences, "max_abs_delta": deltas,
                                 "sources": traces})
        edges.append({
            "key": branch_key(kind, a, b),
            "kind": kind,
            "directed": False,
            "status": status,
            # Conservative when copies disagree: pending unless every copy says otherwise.
            "pending": any(pending_values),
            "endpoints": endpoints,
            "sources": traces,
        })
    return edges


def patch_lookup(manifest: dict) -> dict[str, tuple[str, str]]:
    """Map patch directory names, old UUIDs, and new UUIDs to (directory, UUID)."""
    lookup: dict[str, tuple[str, str]] = {}
    for directory, record in manifest["patches"].items():
        lookup[directory] = (directory, record["uuid"])
    for directory, record in manifest["patches"].items():
        for alias in (record.get("old_uuid"), record["uuid"]):
            if isinstance(alias, str) and alias:
                previous = lookup.setdefault(alias, (directory, record["uuid"]))
                if previous[0] != directory:
                    raise ValueError(f"Patch reference {alias!r} is ambiguous in the node manifest")
    return lookup


def build_explicit_edges(
    fibers: dict[str, Fiber], manifest: dict, problems: list[dict]
) -> tuple[list[dict], int]:
    lookup = patch_lookup(manifest)
    edges: dict[str, dict] = {}
    tag_count = 0
    for fiber in fibers.values():
        tags = fiber.doc.get("tags", [])
        if not isinstance(tags, list):
            problems.append({"code": "invalid_tags", "detail": f"tags is {type(tags).__name__}",
                             "sources": [{"fiber_file": fiber.name, "array": "tags"}]})
            continue
        for index, tag in enumerate(tags):
            if not isinstance(tag, str) or not tag.startswith(EXPLICIT_LINK_TAG_PREFIX):
                continue
            reference = tag[len(EXPLICIT_LINK_TAG_PREFIX):].strip()
            if not reference:
                continue
            tag_count += 1
            trace = {"fiber_file": fiber.name, "array": "tags", "index": index, "tag": tag}
            resolved = lookup.get(reference)
            if resolved is None:
                problems.append({"code": "unresolved_patch_reference", "patch_reference": reference,
                                 "sources": [trace]})
                continue
            directory, patch_uuid = resolved
            key = f"{EXPLICIT_KIND}|{fiber.uuid.lower()}|{patch_uuid.lower()}"
            edge = edges.setdefault(key, {
                "key": key,
                "kind": EXPLICIT_KIND,
                "directed": True,
                "status": "authored",
                "pending": False,
                "endpoints": [
                    {"fiber_uuid": fiber.uuid, "fiber_file": fiber.name},
                    {"patch_uuid": patch_uuid, "patch_directory": directory},
                ],
                "sources": [],
            })
            edge["sources"].append(trace)
    return list(edges.values()), tag_count


def endpoint_identity(endpoint: dict) -> tuple[str, str, str] | None:
    if "fiber_uuid" in endpoint and "fiber_file" in endpoint:
        return ("fibers", endpoint["fiber_file"], endpoint["fiber_uuid"])
    if "patch_uuid" in endpoint and "patch_directory" in endpoint:
        return ("patches", endpoint["patch_directory"], endpoint["patch_uuid"])
    return None


def previous_edge_uuids(path: Path, manifest: dict) -> dict[str, str]:
    """Edge UUIDs from an earlier run, after checking its node references."""
    if not path.exists():
        return {}
    previous = read_json(path)
    if previous.get("version") != EDGE_MANIFEST_VERSION or not isinstance(previous.get("edges"), list):
        raise ValueError(f"Unknown edge manifest format: {path}")
    by_uuid = {
        section: {record["uuid"].lower(): name for name, record in manifest[section].items()}
        for section in ("fibers", "patches")
    }
    result: dict[str, str] = {}
    used: set[str] = set()
    for edge in previous["edges"]:
        key, value = edge.get("key"), edge.get("uuid")
        if not isinstance(key, str) or not is_uuid(value) or key in result or value.lower() in used:
            raise ValueError(f"Invalid or duplicate edge in existing manifest: {key!r} {value!r}")
        for endpoint in edge.get("endpoints", []):
            identity = endpoint_identity(endpoint)
            if identity is None:
                raise ValueError(f"Existing edge {value} has an unrecognized endpoint")
            section, name, node_uuid = identity
            current = manifest[section].get(name, {}).get("uuid")
            owner = by_uuid[section].get(str(node_uuid).lower())
            if (current is not None and current != node_uuid) or (owner is not None and owner != name):
                raise ValueError(
                    f"Node UUIDs changed since {path.name} was written: {section} {name!r} "
                    f"was {node_uuid}, node manifest now has {current} (UUID owner {owner!r})"
                )
        result[key] = value
        used.add(value.lower())
    return result


def build_manifest(root: Path) -> tuple[dict, dict]:
    manifest, fibers = load_inputs(root)
    manifest_path = root / MANIFEST_NAME
    output_path = root / EDGE_MANIFEST_NAME
    reuse = previous_edge_uuids(output_path, manifest)
    problems: list[dict] = []
    records = collect_branch_records(fibers, problems)
    edges = build_branch_edges(records, problems)
    explicit, tag_count = build_explicit_edges(fibers, manifest, problems)
    edges += explicit
    edges.sort(key=lambda edge: (edge["kind"], edge["key"]))
    keys = [edge["key"] for edge in edges]
    if len(keys) != len(set(keys)):
        raise AssertionError("Canonical edge keys are not unique")
    reused = 0
    used = set(value.lower() for value in reuse.values())
    records_out = []
    for edge in edges:
        value = reuse.get(edge["key"])
        if value is None:
            value = str(uuid.uuid4())
            while value.lower() in used:
                value = str(uuid.uuid4())
            used.add(value.lower())
        else:
            reused += 1
        records_out.append({"uuid": value, **edge})
    problems.sort(key=lambda problem: (problem["code"], json.dumps(problem, sort_keys=True)))
    array_counts = Counter()
    for fiber in fibers.values():
        for array in BRANCH_ARRAYS:
            if isinstance(fiber.doc.get(array), list):
                array_counts[array] += len(fiber.doc[array])
    result = {
        "version": EDGE_MANIFEST_VERSION,
        "created_from": {
            "root": str(root),
            "node_manifest": str(manifest_path),
            "node_manifest_sha256": file_sha256(manifest_path),
            "node_counts": {
                "patches": len(manifest["patches"]),
                "fibers": len(manifest["fibers"]),
                "point_collections": sum(len(v) for v in manifest["point_collections"].values()),
            },
            "source_counts": {
                "fiber_files": len(fibers),
                "branches_records": array_counts["branches"],
                "adjacent_branches_records": array_counts["adjacent_branches"],
                "explicit_link_tags": tag_count,
            },
        },
        "counts": {
            "edges": dict(sorted(Counter(edge["kind"] for edge in edges).items())),
            "edge_status": dict(sorted(Counter(edge["status"] for edge in edges).items())),
            "problems": dict(sorted(Counter(problem["code"] for problem in problems).items())),
        },
        "edges": records_out,
        "problems": problems,
    }
    stats = {
        "reused": reused,
        "new": len(edges) - reused,
        "retired": len(set(reuse) - set(keys)),
        "output": output_path,
    }
    return result, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="UUID-converted dataset copy (read-only input)")
    args = parser.parse_args()
    root = args.root.resolve()
    result, stats = build_manifest(root)
    write_json_atomic(stats["output"], result, 2)
    counts = result["counts"]
    sources = result["created_from"]["source_counts"]
    print(f"Read {sources['fiber_files']} fibers: {sources['branches_records']} branches records, "
          f"{sources['adjacent_branches_records']} adjacent_branches records, "
          f"{sources['explicit_link_tags']} explicit_link tags")
    print("Edges:" + ("" if counts["edges"] else " none"))
    for kind, count in counts["edges"].items():
        print(f"  {kind}: {count}")
    print("Edge status: " + ", ".join(f"{k} {v}" for k, v in counts["edge_status"].items()))
    print(f"Edge UUIDs: {stats['reused']} reused, {stats['new']} new, "
          f"{stats['retired']} earlier edges no longer present")
    print("Problems:" + ("" if counts["problems"] else " none"))
    for code, count in counts["problems"].items():
        print(f"  {code}: {count}")
    print(f"Manifest: {stats['output']}")


if __name__ == "__main__":
    main()
