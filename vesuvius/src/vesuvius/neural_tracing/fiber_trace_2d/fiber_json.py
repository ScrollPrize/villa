from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vesuvius.neural_tracing.fiber_trace.fiber_json import (
    Vc3dFiber,
    load_vc3d_fiber,
    parse_vc3d_fiber,
)


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _node_sort_key(node_id: str) -> tuple[int, int | str]:
    try:
        return 0, int(node_id)
    except ValueError:
        return 1, node_id


def _finite_float(raw: Any, *, label: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _ordered_simple_path(
    *,
    component: set[str],
    adjacency: dict[str, set[str]],
    label: str,
) -> list[str] | None:
    degrees = {node_id: len(adjacency[node_id] & component) for node_id in component}
    if any(degree > 2 for degree in degrees.values()):
        warnings.warn(
            f"skipping NML fiber component with branch node: {label}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    endpoints = sorted(
        [node_id for node_id, degree in degrees.items() if degree == 1],
        key=_node_sort_key,
    )
    if len(endpoints) != 2:
        warnings.warn(
            f"skipping NML fiber component that is not an open simple path: {label}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    ordered: list[str] = []
    previous: str | None = None
    current = endpoints[0]
    while True:
        ordered.append(current)
        if current == endpoints[1]:
            break
        next_nodes = sorted(
            (adjacency[current] & component) - ({previous} if previous is not None else set()),
            key=_node_sort_key,
        )
        if len(next_nodes) != 1:
            warnings.warn(
                f"skipping NML fiber component with ambiguous traversal: {label}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        previous, current = current, next_nodes[0]
    if len(ordered) != len(component):
        warnings.warn(
            f"skipping NML fiber component with incomplete traversal: {label}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return ordered


def _connected_components(
    node_ids: set[str], adjacency: dict[str, set[str]]
) -> list[set[str]]:
    remaining = set(node_ids)
    components: list[set[str]] = []
    while remaining:
        start = min(remaining, key=_node_sort_key)
        stack = [start]
        component: set[str] = set()
        remaining.remove(start)
        while stack:
            node_id = stack.pop()
            component.add(node_id)
            for neighbor in sorted(adjacency[node_id], key=_node_sort_key):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def parse_nml_fibers(obj: ET.ElementTree | ET.Element, *, path: str | Path | None = None) -> list[Vc3dFiber]:
    """Parse Knossos/WebKnossos NML simple path components as VC3D fibers.

    NML stores nodes in XML and uses edges for ordering. Each open simple path
    component becomes one ``Vc3dFiber`` with the ordered node coordinates used
    as both line points and control points.
    """

    root = obj.getroot() if isinstance(obj, ET.ElementTree) else obj
    fiber_path = Path(path) if path is not None else None
    things = [element for element in root.iter() if _local_name(element.tag) == "thing"]
    if not things:
        things = [root]

    fibers: list[Vc3dFiber] = []
    for thing_index, thing in enumerate(things):
        thing_id = str(thing.attrib.get("id", thing_index))
        thing_name = str(thing.attrib.get("name", ""))
        nodes: dict[str, np.ndarray] = {}
        adjacency: dict[str, set[str]] = {}
        for element in thing.iter():
            if _local_name(element.tag) != "node":
                continue
            node_id = element.attrib.get("id")
            if node_id is None:
                continue
            try:
                point = np.asarray(
                    [
                        _finite_float(element.attrib["x"], label="NML node x"),
                        _finite_float(element.attrib["y"], label="NML node y"),
                        _finite_float(element.attrib["z"], label="NML node z"),
                    ],
                    dtype=np.float32,
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"invalid NML node in {fiber_path}: thing_id={thing_id!r} "
                    f"node_id={node_id!r}: {exc}"
                ) from exc
            nodes[str(node_id)] = point
            adjacency.setdefault(str(node_id), set())

        missing_edge_refs: list[tuple[str, str]] = []
        for element in thing.iter():
            if _local_name(element.tag) != "edge":
                continue
            source = element.attrib.get("source")
            target = element.attrib.get("target")
            if source is None or target is None:
                continue
            source_s = str(source)
            target_s = str(target)
            if source_s not in nodes or target_s not in nodes:
                missing_edge_refs.append((source_s, target_s))
                continue
            adjacency[source_s].add(target_s)
            adjacency[target_s].add(source_s)

        if missing_edge_refs:
            warnings.warn(
                f"skipping {len(missing_edge_refs)} NML edge(s) with missing node refs "
                f"in {fiber_path}: thing_id={thing_id!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        if len(nodes) < 2:
            continue

        for component_index, component in enumerate(_connected_components(set(nodes), adjacency)):
            if len(component) < 2:
                continue
            label = (
                f"path='{fiber_path}' thing_id={thing_id!r} "
                f"thing_name={thing_name!r} component_index={component_index}"
            )
            ordered_ids = _ordered_simple_path(
                component=component,
                adjacency=adjacency,
                label=label,
            )
            if ordered_ids is None:
                continue
            points = np.stack([nodes[node_id] for node_id in ordered_ids], axis=0).astype(
                np.float32,
                copy=False,
            )
            fibers.append(
                Vc3dFiber(
                    path=fiber_path,
                    version=1,
                    line_points_xyz=points,
                    control_points_xyz=points.copy(),
                    generation=1,
                    metadata={
                        "source_format": "nml",
                        "nml_thing_index": thing_index,
                        "nml_thing_id": thing_id,
                        "nml_thing_name": thing_name,
                        "nml_component_index": component_index,
                        "nml_node_ids": tuple(ordered_ids),
                    },
                )
            )
    if not fibers:
        raise ValueError(f"NML file did not contain any usable simple fiber paths: {fiber_path}")
    return fibers


def load_nml_fibers(path: str | Path) -> list[Vc3dFiber]:
    fiber_path = Path(path)
    return parse_nml_fibers(ET.parse(fiber_path), path=fiber_path)


def _homogeneous_matrix_xyz(matrix: np.ndarray) -> np.ndarray:
    raw = np.asarray(matrix, dtype=np.float64)
    if raw.shape == (3, 4):
        raw = np.vstack([raw, [0.0, 0.0, 0.0, 1.0]])
    if raw.shape != (4, 4):
        raise ValueError(f"fiber transform matrix must be 3x4 or 4x4, got {raw.shape}")
    if not bool(np.isfinite(raw).all()):
        raise ValueError("fiber transform matrix contains non-finite values")
    if not np.allclose(raw[3], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)):
        raise ValueError("fiber transform matrix must be homogeneous with last row [0, 0, 0, 1]")
    return raw


def apply_fiber_transform(
    fiber: Vc3dFiber,
    matrix_xyz: np.ndarray,
    *,
    transform_identity: str | None = None,
) -> Vc3dFiber:
    matrix = _homogeneous_matrix_xyz(matrix_xyz)

    def apply(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        ones = np.ones((int(pts.shape[0]), 1), dtype=np.float64)
        transformed = (matrix @ np.concatenate([pts, ones], axis=1).T).T[:, :3]
        return transformed.astype(np.float32)

    metadata = dict(fiber.metadata)
    metadata["fiber_transform_applied"] = True
    if transform_identity:
        metadata["fiber_transform_identity"] = str(transform_identity)
    return replace(
        fiber,
        line_points_xyz=apply(fiber.line_points_xyz),
        control_points_xyz=apply(fiber.control_points_xyz),
        metadata=metadata,
    )


def load_fiber_file(
    path: str | Path,
    *,
    transform_xyz: np.ndarray | None = None,
    transform_identity: str | None = None,
) -> list[Vc3dFiber]:
    fiber_path = Path(path)
    suffix = fiber_path.suffix.lower()
    if suffix == ".json":
        fibers = [load_vc3d_fiber(fiber_path)]
    elif suffix == ".nml":
        fibers = load_nml_fibers(fiber_path)
    else:
        raise ValueError(f"unsupported fiber source extension {suffix!r}: {fiber_path}")
    if transform_xyz is not None:
        fibers = [
            apply_fiber_transform(
                fiber,
                transform_xyz,
                transform_identity=transform_identity,
            )
            for fiber in fibers
        ]
    return fibers


__all__ = [
    "Vc3dFiber",
    "apply_fiber_transform",
    "load_fiber_file",
    "load_nml_fibers",
    "load_vc3d_fiber",
    "parse_nml_fibers",
    "parse_vc3d_fiber",
]
