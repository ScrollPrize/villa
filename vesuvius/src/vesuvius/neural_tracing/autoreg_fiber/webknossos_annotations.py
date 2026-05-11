from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_WEBKNOSSOS_URL = ""
DEFAULT_CACHE_DIR = Path("webknossos_annotations")
FIBER_NAME_MARKERS: dict[str, str] = {
    "fibers_s3": "PHerc0332",
    "fibers_s1a": "PHercParis4",
}
_RELAXED_MARKER_TOKENS: dict[str, tuple[str, str]] = {
    "s3": ("fibers_s3", "PHerc0332"),
    "s1a": ("fibers_s1a", "PHercParis4"),
}


@dataclass(frozen=True)
class AnnotationCandidate:
    annotation_id: str
    name: str
    owner: str | None
    marker: str
    target_volume: str
    dataset_id: str | None = None
    dataset_name: str | None = None
    annotation_type: str | None = None
    state: str | None = None
    modified: int | None = None
    source: str = "readable"


@dataclass(frozen=True)
class TreeInventoryRecord:
    annotation_id: str
    annotation_name: str
    owner: str | None
    marker: str
    target_volume: str
    dataset_id: str | None
    dataset_name: str | None
    annotation_type: str | None
    state: str | None
    source: str
    tree_id: int | str | None
    tree_name: str
    status: str
    reject_reason: str | None
    node_count: int
    edge_count: int
    bbox_xyz_min: list[float] | None
    bbox_xyz_max: list[float] | None
    path_length_voxels: float
    downloaded_path: str | None


@dataclass(frozen=True)
class _SkeletonProtoNode:
    id: int
    position: tuple[float, float, float]


@dataclass(frozen=True)
class _SkeletonProtoTree:
    id: int
    name: str
    nodes: list[_SkeletonProtoNode]
    edges: list[tuple[_SkeletonProtoNode, _SkeletonProtoNode]]


@dataclass(frozen=True)
class _SkeletonProtoSkeleton:
    trees: list[_SkeletonProtoTree]

    def flattened_trees(self) -> Iterable[_SkeletonProtoTree]:
        return iter(self.trees)


@dataclass(frozen=True)
class _SkeletonProtoAnnotation:
    skeleton: _SkeletonProtoSkeleton


def sanitize_filename(value: str, *, max_length: int = 96) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not cleaned:
        cleaned = "annotation"
    return cleaned[:max_length]


def infer_fiber_marker(annotation_name: str) -> tuple[str, str] | None:
    lowered = str(annotation_name).lower()
    matches = [
        (marker, target)
        for marker, target in FIBER_NAME_MARKERS.items()
        if marker in lowered
    ]
    if not matches and "fiber" in lowered:
        for token, marker_info in _RELAXED_MARKER_TOKENS.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered):
                matches.append(marker_info)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"annotation name matches multiple fiber markers: {annotation_name!r}")
    return matches[0]


def resolve_token_path(token_file: str | Path | None = None, *, start_dir: str | Path | None = None) -> Path:
    if token_file is not None:
        return Path(token_file).expanduser().resolve()

    start = Path(start_dir or Path.cwd()).resolve()
    candidates: list[Path] = []
    for parent in (start, *start.parents):
        candidates.append(parent / "webknossos-api-token.txt")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find webknossos-api-token.txt walking up from the current directory"
    )


def read_token_file(token_file: str | Path) -> str:
    token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("WebKnossos token file is empty")
    return token


def _load_webknossos_module():
    try:
        import webknossos as wk  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The WebKnossos client is not installed. Run with the optional extra, "
            "for example: uv run --extra webknossos python -m "
            "vesuvius.neural_tracing.autoreg_fiber.webknossos_annotations ..."
        ) from exc
    return wk


def _get_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _owner_to_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    if isinstance(value, Mapping):
        value = _get_attr(value, ("id", "user_id", "userId", "_id"), default)
    return _none_or_str(value)


def _enum_or_str(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = _get_attr(value, ("value",), None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)


def _candidate_from_annotation_info(
    info: Any,
    *,
    owner: str | None,
    source: str,
) -> AnnotationCandidate | None:
    name = str(_get_attr(info, ("name",), ""))
    marker_info = infer_fiber_marker(name)
    if marker_info is None:
        return None
    marker, target_volume = marker_info
    annotation_id = str(_get_attr(info, ("id", "annotation_id")))
    if not annotation_id or annotation_id == "None":
        raise ValueError(f"matching annotation {name!r} has no id")
    return AnnotationCandidate(
        annotation_id=annotation_id,
        name=name,
        owner=_owner_to_str(_get_attr(info, ("owner_id", "owner", "ownerId"), owner), owner),
        marker=marker,
        target_volume=target_volume,
        dataset_id=_none_or_str(_get_attr(info, ("dataset_id", "datasetId"))),
        dataset_name=_none_or_str(_get_attr(info, ("dataset_name", "datasetName", "dataSetName"))),
        annotation_type=_enum_or_str(_get_attr(info, ("type", "typ"))),
        state=_enum_or_str(_get_attr(info, ("state",))),
        modified=_none_or_int(_get_attr(info, ("modified",))),
        source=source,
    )


def annotation_info_to_candidate(info: Any, *, owner: str | None) -> AnnotationCandidate | None:
    return _candidate_from_annotation_info(info, owner=owner, source="readable")


def annotation_payload_to_candidate(
    payload: Mapping[str, Any],
    *,
    owner: str | None,
    source: str,
) -> AnnotationCandidate | None:
    name = str(_get_attr(payload, ("name",), ""))
    marker_info = infer_fiber_marker(name)
    if marker_info is None:
        return None
    marker, target_volume = marker_info
    annotation_id = str(_get_attr(payload, ("id", "_id", "annotation_id", "annotationId")))
    if not annotation_id or annotation_id == "None":
        raise ValueError(f"matching annotation {name!r} has no id")
    owner_value = _get_attr(payload, ("owner_id", "ownerId", "owner", "user_id", "userId"), owner)
    return AnnotationCandidate(
        annotation_id=annotation_id,
        name=name,
        owner=_owner_to_str(owner_value, owner),
        marker=marker,
        target_volume=target_volume,
        dataset_id=_none_or_str(_get_attr(payload, ("dataset_id", "datasetId", "_dataset"))),
        dataset_name=_none_or_str(_get_attr(payload, ("dataset_name", "datasetName", "dataSetName"))),
        annotation_type=_none_or_str(_get_attr(payload, ("typ", "type"))),
        state=_none_or_str(_get_attr(payload, ("state",))),
        modified=_none_or_int(_get_attr(payload, ("modified",))),
        source=source,
    )


def _managed_owner_keys(wk: Any) -> list[tuple[str | None, str]]:
    user_cls = getattr(wk, "User", None)
    if user_cls is None:
        return [(None, "current")]

    users: list[Any] = []
    for method_name in ("get_current_user", "get_all_managed_users"):
        method = getattr(user_cls, method_name, None)
        if method is None:
            continue
        try:
            result = method()
        except Exception:
            continue
        if result is None:
            continue
        if isinstance(result, list):
            users.extend(result)
        else:
            users.append(result)

    owners: list[tuple[str | None, str]] = []
    seen: set[str | None] = set()
    for user in users:
        owner = _get_attr(user, ("user_id", "id", "email", "username", "name"))
        if owner is None:
            continue
        owner_str = str(owner)
        if owner_str in seen:
            continue
        seen.add(owner_str)
        label = str(_get_attr(user, ("email", "username", "name", "user_id", "id"), owner_str))
        owners.append((owner_str, label))
    return owners or [(None, "current")]


def _wk_api_client_or_none(wk: Any) -> Any | None:
    if getattr(wk, "__name__", None) != "webknossos":
        client = _get_attr(wk, ("api_client", "_api_client"), None)
        return client if client is not None and hasattr(client, "_get") else None
    try:
        from webknossos.client.context import _get_api_client
    except Exception:
        return None
    try:
        return _get_api_client(True)
    except Exception:
        return None


def _client_get_json_list(
    client: Any,
    route: str,
    *,
    query: Mapping[str, Any] | None = None,
) -> tuple[list[Mapping[str, Any]], Mapping[str, str]]:
    if not hasattr(client, "_get"):
        raise RuntimeError("WebKnossos API client does not expose _get")
    response = client._get(route, query=dict(query or {}))
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"WebKnossos endpoint {route} did not return a list")
    return data, getattr(response, "headers", {}) or {}


def _read_proto_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 64:
            raise ValueError("protobuf varint is too long")


def _iter_proto_fields(data: bytes) -> Iterable[tuple[int, int, Any]]:
    offset = 0
    while offset < len(data):
        key, offset = _read_proto_varint(data, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if wire_type == 0:
            value, offset = _read_proto_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise ValueError("truncated protobuf fixed64 field")
            value = data[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _read_proto_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf length-delimited field")
            value = data[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise ValueError("truncated protobuf fixed32 field")
            value = data[offset:end]
            offset = end
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def _parse_vec3_int_proto(data: bytes) -> tuple[float, float, float]:
    coords = {1: 0, 2: 0, 3: 0}
    for field_number, wire_type, value in _iter_proto_fields(data):
        if field_number in coords and wire_type == 0:
            coords[field_number] = int(value)
    return (float(coords[1]), float(coords[2]), float(coords[3]))


def _parse_skeleton_node_proto(data: bytes) -> _SkeletonProtoNode | None:
    node_id: int | None = None
    position: tuple[float, float, float] | None = None
    for field_number, wire_type, value in _iter_proto_fields(data):
        if field_number == 1 and wire_type == 0:
            node_id = int(value)
        elif field_number == 2 and wire_type == 2:
            position = _parse_vec3_int_proto(value)
    if node_id is None or position is None:
        return None
    return _SkeletonProtoNode(id=node_id, position=position)


def _parse_skeleton_edge_proto(data: bytes) -> tuple[int, int] | None:
    source: int | None = None
    target: int | None = None
    for field_number, wire_type, value in _iter_proto_fields(data):
        if field_number == 1 and wire_type == 0:
            source = int(value)
        elif field_number == 2 and wire_type == 0:
            target = int(value)
    if source is None or target is None:
        return None
    return source, target


def _parse_skeleton_tree_proto(data: bytes, *, name_prefix: str = "") -> _SkeletonProtoTree | None:
    tree_id: int | None = None
    name = ""
    nodes: list[_SkeletonProtoNode] = []
    edge_ids: list[tuple[int, int]] = []
    for field_number, wire_type, value in _iter_proto_fields(data):
        if field_number == 1 and wire_type == 0:
            tree_id = int(value)
        elif field_number == 2 and wire_type == 2:
            node = _parse_skeleton_node_proto(value)
            if node is not None:
                nodes.append(node)
        elif field_number == 3 and wire_type == 2:
            edge = _parse_skeleton_edge_proto(value)
            if edge is not None:
                edge_ids.append(edge)
        elif field_number == 7 and wire_type == 2:
            name = value.decode("utf-8", errors="replace")
    if tree_id is None:
        return None
    nodes_by_id = {node.id: node for node in nodes}
    edges = [
        (nodes_by_id[source], nodes_by_id[target])
        for source, target in edge_ids
        if source in nodes_by_id and target in nodes_by_id
    ]
    tree_name = f"{name_prefix}{name}" if name_prefix and name else name_prefix or name
    return _SkeletonProtoTree(id=tree_id, name=tree_name, nodes=nodes, edges=edges)


def skeleton_trees_from_tracing_proto(data: bytes, *, name_prefix: str = "") -> list[_SkeletonProtoTree]:
    trees: list[_SkeletonProtoTree] = []
    for field_number, wire_type, value in _iter_proto_fields(data):
        if field_number == 2 and wire_type == 2:
            tree = _parse_skeleton_tree_proto(value, name_prefix=name_prefix)
            if tree is not None:
                trees.append(tree)
    return trees


def _tracingstore_client_or_none() -> Any | None:
    try:
        from webknossos.client.context import _get_context
    except Exception:
        return None
    try:
        return _get_context().get_tracingstore_api_client(require_auth=True)
    except Exception:
        return None


def _download_skeleton_only_annotation(
    client: Any,
    candidate: AnnotationCandidate,
    annotations_dir: Path,
    *,
    version: int = 0,
) -> tuple[_SkeletonProtoAnnotation, Path]:
    tracing_client = _tracingstore_client_or_none()
    if tracing_client is None or not hasattr(tracing_client, "_get"):
        raise RuntimeError("WebKnossos tracing-store client is unavailable")

    info_response = client._get(f"/annotations/{candidate.annotation_id}/info")
    info = info_response.json()
    skeleton_layers = [
        layer
        for layer in info.get("annotationLayers", [])
        if str(layer.get("typ", "")).lower() == "skeleton" and layer.get("tracingId")
    ]
    if not skeleton_layers:
        raise RuntimeError("annotation has no skeleton layers")

    raw_tracings: dict[str, bytes] = {}
    trees: list[_SkeletonProtoTree] = []
    for index, layer in enumerate(skeleton_layers):
        tracing_id = str(layer["tracingId"])
        response = tracing_client._get(
            f"/skeleton/{tracing_id}",
            query={
                "annotationId": candidate.annotation_id,
                "version": version,
            },
            timeout_seconds=120,
        )
        raw = bytes(response.content)
        raw_tracings[tracing_id] = raw
        prefix = "" if len(skeleton_layers) == 1 else f"{layer.get('name', index)}:"
        trees.extend(skeleton_trees_from_tracing_proto(raw, name_prefix=prefix))

    annotations_dir.mkdir(parents=True, exist_ok=True)
    out_path = annotations_dir / f"{sanitize_filename(candidate.name)}_{candidate.annotation_id}.skeleton-v{version}.zip"
    if not out_path.exists():
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "annotation_id": candidate.annotation_id,
                        "annotation_name": candidate.name,
                        "skeleton_version": version,
                        "source": "tracingstore_skeleton_proto",
                        "skeleton_layers": skeleton_layers,
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            for tracing_id, raw in raw_tracings.items():
                zf.writestr(f"{tracing_id}.SkeletonTracing.pb", raw)

    return _SkeletonProtoAnnotation(_SkeletonProtoSkeleton(trees)), out_path


def collect_matching_user_endpoint_annotations(
    client: Any,
    owners: Sequence[tuple[str | None, str]],
    *,
    endpoint: str,
    source: str,
    page_limit: int = 1000,
) -> tuple[list[AnnotationCandidate], dict[str, Any]]:
    candidates: list[AnnotationCandidate] = []
    warnings: list[str] = []
    scanned = 0
    owners_scanned = 0
    endpoints_attempted = 0

    for owner_id, owner_label in owners:
        if owner_id is None:
            continue
        owners_scanned += 1
        page_number = 0
        expected_total: int | None = None
        fetched_for_owner = 0
        while True:
            endpoints_attempted += 1
            try:
                payloads, headers = _client_get_json_list(
                    client,
                    f"/users/{owner_id}/{endpoint}",
                    query={
                        "includeTotalCount": page_number == 0,
                        "limit": page_limit,
                        "pageNumber": page_number,
                    },
                )
            except Exception as exc:
                warnings.append(
                    f"failed to list {endpoint} for {owner_label}: {type(exc).__name__}: {str(exc)[:300]}"
                )
                break

            if page_number == 0:
                header_count = _get_attr(headers, ("X-Total-Count", "x-total-count"), None)
                expected_total = int(header_count) if header_count is not None else None

            scanned += len(payloads)
            fetched_for_owner += len(payloads)
            for payload in payloads:
                candidate = annotation_payload_to_candidate(payload, owner=owner_id, source=source)
                if candidate is not None:
                    candidates.append(candidate)

            if not payloads:
                break
            if expected_total is not None and fetched_for_owner >= expected_total:
                break
            if len(payloads) < page_limit and expected_total is None:
                break
            page_number += 1

    return candidates, {
        "scanned": scanned,
        "owners_scanned": owners_scanned,
        "endpoints_attempted": endpoints_attempted,
        "warnings": warnings,
    }


def collect_matching_annotations(wk: Any) -> tuple[list[AnnotationCandidate], dict[str, Any]]:
    owners = _managed_owner_keys(wk)
    info_cls = getattr(wk, "AnnotationInfo", None)
    if info_cls is None or not hasattr(info_cls, "get_remote_annotations"):
        raise RuntimeError("webknossos.AnnotationInfo.get_remote_annotations is unavailable")

    candidates: list[AnnotationCandidate] = []
    warnings: list[str] = []
    readable_scanned = 0
    user_annotations_scanned = 0
    user_task_annotations_scanned = 0
    seen_ids: set[str] = set()
    state_queries = (False, True)

    for owner_arg, owner_label in owners:
        owner_infos: list[Any] = []
        owner_seen_info_ids: set[str] = set()
        for is_finished in state_queries:
            try:
                infos = list(info_cls.get_remote_annotations(is_finished=is_finished, owner=owner_arg))
            except TypeError:
                try:
                    infos = list(info_cls.get_remote_annotations(owner=owner_arg))
                except Exception as exc:
                    warnings.append(f"failed to list annotations for {owner_label}: {type(exc).__name__}: {exc}")
                    break
            except Exception as exc:
                warnings.append(f"failed to list annotations for {owner_label} finished={is_finished}: {type(exc).__name__}: {exc}")
                continue
            readable_scanned += len(infos)
            for info in infos:
                info_id = str(_get_attr(info, ("id", "annotation_id"), ""))
                if info_id in owner_seen_info_ids:
                    continue
                owner_seen_info_ids.add(info_id)
                owner_infos.append(info)

        for info in owner_infos:
            candidate = annotation_info_to_candidate(info, owner=owner_arg)
            if candidate is None or candidate.annotation_id in seen_ids:
                continue
            seen_ids.add(candidate.annotation_id)
            candidates.append(candidate)

    client = _wk_api_client_or_none(wk)
    endpoint_stats: dict[str, Any] = {}
    if client is None:
        endpoint_stats["user_endpoint_available"] = False
    else:
        endpoint_stats["user_endpoint_available"] = True
        for endpoint, source in (
            ("annotations", "user_annotations"),
            ("tasks", "user_tasks"),
        ):
            direct_candidates, direct_stats = collect_matching_user_endpoint_annotations(
                client,
                owners,
                endpoint=endpoint,
                source=source,
            )
            warnings.extend(direct_stats.get("warnings", []))
            endpoint_stats[f"{source}_endpoints_attempted"] = direct_stats.get("endpoints_attempted", 0)
            endpoint_stats[f"{source}_owners_scanned"] = direct_stats.get("owners_scanned", 0)
            if source == "user_annotations":
                user_annotations_scanned = int(direct_stats.get("scanned", 0))
            else:
                user_task_annotations_scanned = int(direct_stats.get("scanned", 0))
            for candidate in direct_candidates:
                if candidate.annotation_id in seen_ids:
                    continue
                seen_ids.add(candidate.annotation_id)
                candidates.append(candidate)

    return candidates, {
        "annotations_scanned": readable_scanned + user_annotations_scanned + user_task_annotations_scanned,
        "readable_annotations_scanned": readable_scanned,
        "user_annotations_scanned": user_annotations_scanned,
        "user_task_annotations_scanned": user_task_annotations_scanned,
        "owners_scanned": len(owners),
        "warnings": warnings,
        **endpoint_stats,
    }


def _download_annotation(wk: Any, candidate: AnnotationCandidate, annotations_dir: Path):
    annotations_dir.mkdir(parents=True, exist_ok=True)
    try:
        annotation = wk.Annotation.download(candidate.annotation_id, skip_volume_data=True)
        filename = f"{sanitize_filename(candidate.name)}_{candidate.annotation_id}.zip"
        out_path = annotations_dir / filename
        if hasattr(annotation, "save") and not out_path.exists():
            annotation.save(out_path)
        return annotation, out_path
    except Exception as normal_export_exc:
        client = _wk_api_client_or_none(wk)
        if client is None:
            raise
        try:
            return _download_skeleton_only_annotation(
                client,
                candidate,
                annotations_dir,
                version=0,
            )
        except Exception as skeleton_only_exc:
            raise RuntimeError(
                "full annotation export and skeleton-only tracing-store export both failed: "
                f"{type(normal_export_exc).__name__}: {normal_export_exc}; "
                f"{type(skeleton_only_exc).__name__}: {skeleton_only_exc}"
            ) from skeleton_only_exc


def _flattened_trees(annotation: Any) -> list[Any]:
    skeleton = getattr(annotation, "skeleton", None)
    if skeleton is None:
        return []
    flattened = getattr(skeleton, "flattened_trees", None)
    if callable(flattened):
        return list(flattened())
    trees = getattr(skeleton, "trees", None)
    if trees is None:
        return []
    return list(trees() if callable(trees) else trees)


def _tree_nodes(tree: Any) -> list[Any]:
    nodes = getattr(tree, "nodes", [])
    if callable(nodes):
        nodes = nodes()
    return list(nodes)


def _tree_edges(tree: Any) -> list[tuple[Any, Any]]:
    edges = getattr(tree, "edges", [])
    if callable(edges):
        edges = edges()
    normalized: list[tuple[Any, Any]] = []
    for edge in list(edges):
        if len(edge) < 2:
            continue
        normalized.append((edge[0], edge[1]))
    return normalized


def _node_key(node: Any) -> tuple[str, Any]:
    node_id = _get_attr(node, ("id", "_id"), None)
    if node_id is not None:
        return ("id", node_id)
    if isinstance(node, (str, int)):
        return ("value", node)
    return ("object", id(node))


def _position_from_value(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = _get_attr(value, ("position", "pos", "xyz"), None)
    else:
        value = _get_attr(value, ("position", "pos", "xyz"), value)
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr


def _node_positions_by_key(tree: Any, nodes: Sequence[Any]) -> dict[tuple[str, Any], np.ndarray]:
    positions: dict[tuple[str, Any], np.ndarray] = {}
    node_view = getattr(tree, "nodes", None)
    for node in nodes:
        pos = _position_from_value(node)
        if pos is None and node_view is not None:
            try:
                pos = _position_from_value(node_view[node])
            except Exception:
                pos = None
        if pos is None:
            continue
        positions[_node_key(node)] = pos
    return positions


def _path_qc(nodes: Sequence[Any], edges: Sequence[tuple[Any, Any]]) -> str | None:
    if len(nodes) < 2:
        return "too_few_nodes"
    node_keys = [_node_key(node) for node in nodes]
    node_key_set = set(node_keys)
    adjacency = {key: set() for key in node_keys}
    for left, right in edges:
        left_key = _node_key(left)
        right_key = _node_key(right)
        if left_key not in node_key_set or right_key not in node_key_set:
            return "edge_references_missing_node"
        adjacency[left_key].add(right_key)
        adjacency[right_key].add(left_key)
    if len(edges) != len(nodes) - 1:
        return "cyclic_or_disconnected"
    start = node_keys[0]
    visited = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for nxt in adjacency[current]:
            if nxt in visited:
                continue
            visited.add(nxt)
            stack.append(nxt)
    if len(visited) != len(nodes):
        return "disconnected"
    degrees = [len(adjacency[key]) for key in node_keys]
    if max(degrees) > 2:
        return "branching"
    if degrees.count(1) != 2:
        return "not_a_single_path"
    return None


def tree_inventory_record(
    *,
    candidate: AnnotationCandidate,
    tree: Any,
    downloaded_path: Path | None,
) -> TreeInventoryRecord:
    nodes = _tree_nodes(tree)
    edges = _tree_edges(tree)
    positions_by_key = _node_positions_by_key(tree, nodes)
    positions = np.asarray(list(positions_by_key.values()), dtype=np.float64)
    reject_reason = _path_qc(nodes, edges)
    if len(positions_by_key) != len(nodes):
        reject_reason = reject_reason or "missing_node_positions"

    path_length = 0.0
    for left, right in edges:
        left_pos = positions_by_key.get(_node_key(left))
        right_pos = positions_by_key.get(_node_key(right))
        if left_pos is None or right_pos is None:
            continue
        path_length += float(np.linalg.norm(right_pos - left_pos))

    bbox_min = bbox_max = None
    if positions.size > 0:
        bbox_min = positions.min(axis=0).astype(float).tolist()
        bbox_max = positions.max(axis=0).astype(float).tolist()

    return TreeInventoryRecord(
        annotation_id=candidate.annotation_id,
        annotation_name=candidate.name,
        owner=candidate.owner,
        marker=candidate.marker,
        target_volume=candidate.target_volume,
        dataset_id=candidate.dataset_id,
        dataset_name=candidate.dataset_name,
        annotation_type=candidate.annotation_type,
        state=candidate.state,
        source=candidate.source,
        tree_id=_get_attr(tree, ("id", "_id"), None),
        tree_name=str(_get_attr(tree, ("name",), "")),
        status="accepted" if reject_reason is None else "rejected",
        reject_reason=reject_reason,
        node_count=len(nodes),
        edge_count=len(edges),
        bbox_xyz_min=bbox_min,
        bbox_xyz_max=bbox_max,
        path_length_voxels=path_length,
        downloaded_path=None if downloaded_path is None else str(downloaded_path),
    )


def write_annotation_summary_csv(payload: Mapping[str, Any], path: str | Path) -> None:
    annotations = {
        str(candidate["annotation_id"]): dict(candidate)
        for candidate in payload.get("annotations", [])
    }
    grouped_records: dict[str, list[Mapping[str, Any]]] = {key: [] for key in annotations}
    for record in payload.get("records", []):
        annotation_id = str(record.get("annotation_id"))
        grouped_records.setdefault(annotation_id, []).append(record)
        if annotation_id not in annotations:
            annotations[annotation_id] = {
                "annotation_id": annotation_id,
                "name": record.get("annotation_name"),
                "owner": record.get("owner"),
                "marker": record.get("marker"),
                "target_volume": record.get("target_volume"),
                "dataset_id": record.get("dataset_id"),
                "dataset_name": record.get("dataset_name"),
                "annotation_type": record.get("annotation_type"),
                "state": record.get("state"),
                "source": record.get("source"),
            }

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "annotation_id",
        "owner",
        "name",
        "marker",
        "target_volume",
        "dataset_id",
        "dataset_name",
        "annotation_type",
        "state",
        "source",
        "tree_records",
        "trees_accepted",
        "trees_rejected",
        "node_count_total",
        "edge_count_total",
        "path_length_voxels_total",
        "downloaded_paths",
        "reject_reasons",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for annotation_id in sorted(annotations):
            candidate = annotations[annotation_id]
            records = grouped_records.get(annotation_id, [])
            accepted = sum(1 for record in records if record.get("status") == "accepted")
            rejected = len(records) - accepted
            downloaded_paths = sorted(
                {
                    str(record.get("downloaded_path"))
                    for record in records
                    if record.get("downloaded_path")
                }
            )
            reject_reasons = sorted(
                {
                    str(record.get("reject_reason"))
                    for record in records
                    if record.get("reject_reason")
                }
            )
            writer.writerow(
                {
                    "annotation_id": annotation_id,
                    "owner": candidate.get("owner"),
                    "name": candidate.get("name"),
                    "marker": candidate.get("marker"),
                    "target_volume": candidate.get("target_volume"),
                    "dataset_id": candidate.get("dataset_id"),
                    "dataset_name": candidate.get("dataset_name"),
                    "annotation_type": candidate.get("annotation_type"),
                    "state": candidate.get("state"),
                    "source": candidate.get("source"),
                    "tree_records": len(records),
                    "trees_accepted": accepted,
                    "trees_rejected": rejected,
                    "node_count_total": sum(int(record.get("node_count") or 0) for record in records),
                    "edge_count_total": sum(int(record.get("edge_count") or 0) for record in records),
                    "path_length_voxels_total": sum(float(record.get("path_length_voxels") or 0.0) for record in records),
                    "downloaded_paths": ";".join(downloaded_paths),
                    "reject_reasons": ";".join(reject_reasons),
                }
            )


def run_inventory(
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    token_file: str | Path | None = None,
    token: str | None = None,
    webknossos_url: str = DEFAULT_WEBKNOSSOS_URL,
    inventory_json: str | Path | None = None,
    summary_csv: str | Path | None = None,
    limit: int | None = None,
    wk_module: Any | None = None,
) -> dict[str, Any]:
    if token is None:
        token_path = resolve_token_path(token_file)
        token = read_token_file(token_path)

    wk = wk_module if wk_module is not None else _load_webknossos_module()
    root = Path(cache_dir).expanduser().resolve()
    annotations_dir = root / "annotations"
    inventory_path = Path(inventory_json).expanduser().resolve() if inventory_json else root / "inventory.json"
    root.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    records: list[TreeInventoryRecord] = []
    candidates: list[AnnotationCandidate] = []
    scan_stats: dict[str, Any] = {}
    annotations_downloaded = 0
    warnings: list[str] = []

    with wk.webknossos_context(url=webknossos_url, token=token):
        candidates, scan_stats = collect_matching_annotations(wk)
        warnings.extend(scan_stats.get("warnings", []))
        if limit is not None:
            candidates = candidates[: int(limit)]
        for candidate in candidates:
            try:
                annotation, downloaded_path = _download_annotation(wk, candidate, annotations_dir)
            except Exception as exc:
                reason = f"download_failed:{type(exc).__name__}:{str(exc)[:500]}"
                warnings.append(f"failed to download annotation {candidate.annotation_id}: {type(exc).__name__}: {str(exc)[:300]}")
                records.append(
                    TreeInventoryRecord(
                        annotation_id=candidate.annotation_id,
                        annotation_name=candidate.name,
                        owner=candidate.owner,
                        marker=candidate.marker,
                        target_volume=candidate.target_volume,
                        dataset_id=candidate.dataset_id,
                        dataset_name=candidate.dataset_name,
                        annotation_type=candidate.annotation_type,
                        state=candidate.state,
                        source=candidate.source,
                        tree_id=None,
                        tree_name="",
                        status="rejected",
                        reject_reason=reason,
                        node_count=0,
                        edge_count=0,
                        bbox_xyz_min=None,
                        bbox_xyz_max=None,
                        path_length_voxels=0.0,
                        downloaded_path=None,
                    )
                )
                continue
            annotations_downloaded += 1
            trees = _flattened_trees(annotation)
            if not trees:
                records.append(
                    TreeInventoryRecord(
                        annotation_id=candidate.annotation_id,
                        annotation_name=candidate.name,
                        owner=candidate.owner,
                        marker=candidate.marker,
                        target_volume=candidate.target_volume,
                        dataset_id=candidate.dataset_id,
                        dataset_name=candidate.dataset_name,
                        annotation_type=candidate.annotation_type,
                        state=candidate.state,
                        source=candidate.source,
                        tree_id=None,
                        tree_name="",
                        status="rejected",
                        reject_reason="no_trees",
                        node_count=0,
                        edge_count=0,
                        bbox_xyz_min=None,
                        bbox_xyz_max=None,
                        path_length_voxels=0.0,
                        downloaded_path=str(downloaded_path),
                    )
                )
                continue
            for tree in trees:
                records.append(
                    tree_inventory_record(
                        candidate=candidate,
                        tree=tree,
                        downloaded_path=downloaded_path,
                    )
                )

    elapsed = time.perf_counter() - t0
    accepted = sum(1 for record in records if record.status == "accepted")
    rejected = len(records) - accepted
    payload = {
        "webknossos_url": webknossos_url,
        "cache_dir": str(root),
        "counts": {
            "annotations_scanned": int(scan_stats.get("annotations_scanned", 0)),
            "owners_scanned": int(scan_stats.get("owners_scanned", 0)),
            "annotations_matched": len(candidates),
            "annotations_downloaded": annotations_downloaded,
            "trees_total": len(records),
            "trees_accepted": accepted,
            "trees_rejected": rejected,
        },
        "benchmark": {
            "elapsed_seconds": elapsed,
            "trees_per_second": 0.0 if elapsed <= 0.0 else len(records) / elapsed,
        },
        "warnings": warnings,
        "scan": scan_stats,
        "annotations": [asdict(candidate) for candidate in candidates],
        "records": [asdict(record) for record in records],
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if summary_csv is not None:
        write_annotation_summary_csv(payload, summary_csv)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory WebKnossos fiber skeleton annotations.")
    parser.add_argument("--webknossos-url", default=DEFAULT_WEBKNOSSOS_URL)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--inventory-json", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    payload = run_inventory(
        cache_dir=args.cache_dir,
        token_file=args.token_file,
        webknossos_url=args.webknossos_url,
        inventory_json=args.inventory_json,
        summary_csv=args.summary_csv,
        limit=args.limit,
    )
    counts = payload["counts"]
    print(
        "WebKnossos fiber inventory: "
        f"{counts['annotations_matched']} annotations, "
        f"{counts['trees_accepted']} accepted trees, "
        f"{counts['trees_rejected']} rejected trees"
    )


if __name__ == "__main__":
    main()
