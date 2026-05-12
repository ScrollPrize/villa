from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vesuvius.data import affine
from vesuvius.neural_tracing.autoreg_fiber.webknossos_annotations import (
    _node_key,
    _node_positions_by_key,
    _path_qc,
    _tree_edges,
    _tree_nodes,
    sanitize_filename,
)


S3_TRANSFORM_URL = (
    "s3://vesuvius-challenge-open-data/PHerc0332/volumes/"
    "20251211183505-2.399um-0.2m-78keV-masked.zarr/transform.json"
)
S1A_TRANSFORM_URL = (
    "s3://vesuvius-challenge-open-data/PHercParis4/volumes/"
    "20260411134726-2.400um-0.2m-78keV-masked.zarr/transform.json"
)
TRANSFORM_URL_BY_MARKER = {
    "fibers_s3": S3_TRANSFORM_URL,
    "fibers_s1a": S1A_TRANSFORM_URL,
}


class FiberGeometryError(ValueError):
    """Raised when a skeleton tree cannot be converted into one ordered fiber."""


@dataclass(frozen=True)
class OrderedTreePath:
    node_keys: list[str]
    points_xyz: np.ndarray


@dataclass(frozen=True)
class FiberPath:
    annotation_id: str
    tree_id: str
    target_volume: str
    marker: str
    source_points_xyz: np.ndarray
    points_zyx: np.ndarray
    transform_checksum: str
    densify_step: float | None
    coordinate_convention: str = "wk_xyz_to_new_zyx"


def xyz_to_zyx(points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {points.shape}")
    return points[:, ::-1].astype(np.float64, copy=True)


def zyx_to_xyz(points_zyx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_zyx, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_zyx must have shape (N, 3), got {points.shape}")
    return points[:, ::-1].astype(np.float64, copy=True)


def apply_affine_xyz(matrix_xyz: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix_xyz, dtype=np.float64)
    points = np.asarray(points_xyz, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix_xyz must have shape (4, 4), got {matrix.shape}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {points.shape}")
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def old_xyz_to_new_zyx(points_old_xyz: np.ndarray, *, new_to_old_matrix_xyz: np.ndarray) -> np.ndarray:
    old_to_new = affine.invert_affine_matrix(np.asarray(new_to_old_matrix_xyz, dtype=np.float64))
    points_new_xyz = apply_affine_xyz(old_to_new, points_old_xyz)
    return xyz_to_zyx(points_new_xyz)


def _key_sort_value(key: tuple[str, Any]) -> str:
    return f"{key[0]}:{key[1]}"


def order_tree_path_xyz(tree: Any) -> OrderedTreePath:
    nodes = _tree_nodes(tree)
    edges = _tree_edges(tree)
    reject_reason = _path_qc(nodes, edges)
    if reject_reason is not None:
        raise FiberGeometryError(f"tree is not a single path: {reject_reason}")

    positions = _node_positions_by_key(tree, nodes)
    if len(positions) != len(nodes):
        raise FiberGeometryError("tree has nodes without XYZ positions")

    node_keys = [_node_key(node) for node in nodes]
    adjacency: dict[tuple[str, Any], list[tuple[str, Any]]] = {key: [] for key in node_keys}
    for left, right in edges:
        left_key = _node_key(left)
        right_key = _node_key(right)
        adjacency[left_key].append(right_key)
        adjacency[right_key].append(left_key)

    endpoints = sorted(
        [key for key, neighbors in adjacency.items() if len(neighbors) == 1],
        key=_key_sort_value,
    )
    if len(endpoints) != 2:
        raise FiberGeometryError("tree path does not have exactly two endpoints")

    ordered: list[tuple[str, Any]] = []
    previous: tuple[str, Any] | None = None
    current = endpoints[0]
    while True:
        ordered.append(current)
        next_nodes = [key for key in adjacency[current] if key != previous]
        if not next_nodes:
            break
        previous, current = current, sorted(next_nodes, key=_key_sort_value)[0]

    if len(ordered) != len(node_keys):
        raise FiberGeometryError("tree traversal did not visit every node")

    points = np.stack([positions[key] for key in ordered], axis=0)
    return OrderedTreePath(
        node_keys=[_key_sort_value(key) for key in ordered],
        points_xyz=points.astype(np.float64, copy=False),
    )


def densify_polyline(points: np.ndarray, *, max_step: float | None) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {arr.shape}")
    if arr.shape[0] <= 1 or max_step is None or float(max_step) <= 0.0:
        return arr.astype(np.float64, copy=True)

    step = float(max_step)
    out = [arr[0]]
    for start, stop in zip(arr[:-1], arr[1:]):
        delta = stop - start
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            continue
        pieces = max(1, int(np.ceil(length / step)))
        for idx in range(1, pieces + 1):
            out.append(start + delta * (idx / pieces))
    return np.stack(out, axis=0).astype(np.float64, copy=False)


def resample_polyline_uniform(points: np.ndarray, *, target_spacing: float) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {arr.shape}")
    if arr.shape[0] <= 1 or float(target_spacing) <= 0.0:
        return arr.astype(np.float64, copy=True)
    seg_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    arc_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(arc_length[-1])
    if total <= 0.0:
        return arr[[0]].astype(np.float64, copy=True)
    n = max(2, int(round(total / float(target_spacing))) + 1)
    targets = np.linspace(0.0, total, n)
    out = np.stack(
        [np.interp(targets, arc_length, arr[:, axis]) for axis in range(3)],
        axis=1,
    )
    return out.astype(np.float64, copy=False)


def tree_to_fiber_path(
    tree: Any,
    *,
    annotation_id: str,
    marker: str,
    target_volume: str,
    new_to_old_matrix_xyz: np.ndarray,
    densify_step: float | None = None,
) -> FiberPath:
    ordered = order_tree_path_xyz(tree)
    source_points_xyz = densify_polyline(ordered.points_xyz, max_step=densify_step)
    points_zyx = old_xyz_to_new_zyx(
        source_points_xyz,
        new_to_old_matrix_xyz=new_to_old_matrix_xyz,
    )
    return FiberPath(
        annotation_id=str(annotation_id),
        tree_id=str(getattr(tree, "id", "")),
        target_volume=str(target_volume),
        marker=str(marker),
        source_points_xyz=source_points_xyz,
        points_zyx=points_zyx,
        transform_checksum=affine.matrix_checksum(np.asarray(new_to_old_matrix_xyz, dtype=np.float64)),
        densify_step=None if densify_step is None else float(densify_step),
    )


def fiber_cache_key(
    *,
    annotation_id: str,
    tree_id: str | int,
    transform_checksum: str,
    densify_step: float | None,
    coordinate_convention: str = "wk_xyz_to_new_zyx",
) -> str:
    payload = json.dumps(
        {
            "annotation_id": str(annotation_id),
            "tree_id": str(tree_id),
            "transform_checksum": str(transform_checksum),
            "densify_step": None if densify_step is None else float(densify_step),
            "coordinate_convention": coordinate_convention,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def write_fiber_cache(fiber: FiberPath, output_dir: str | Path) -> Path:
    key = fiber_cache_key(
        annotation_id=fiber.annotation_id,
        tree_id=fiber.tree_id,
        transform_checksum=fiber.transform_checksum,
        densify_step=fiber.densify_step,
        coordinate_convention=fiber.coordinate_convention,
    )
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_filename(fiber.annotation_id)}_{sanitize_filename(fiber.tree_id)}_{key}.npz"
    path = out_dir / filename
    metadata = {
        "annotation_id": fiber.annotation_id,
        "tree_id": fiber.tree_id,
        "target_volume": fiber.target_volume,
        "marker": fiber.marker,
        "transform_checksum": fiber.transform_checksum,
        "densify_step": fiber.densify_step,
        "coordinate_convention": fiber.coordinate_convention,
        "point_count": int(fiber.points_zyx.shape[0]),
    }
    np.savez_compressed(
        path,
        points_zyx=fiber.points_zyx.astype(np.float32),
        source_points_xyz=fiber.source_points_xyz.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def load_fiber_cache(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    blob = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(blob["metadata"].item()))
    return np.asarray(blob["points_zyx"], dtype=np.float32), metadata


__all__ = [
    "FiberGeometryError",
    "FiberPath",
    "OrderedTreePath",
    "S1A_TRANSFORM_URL",
    "S3_TRANSFORM_URL",
    "TRANSFORM_URL_BY_MARKER",
    "apply_affine_xyz",
    "densify_polyline",
    "fiber_cache_key",
    "load_fiber_cache",
    "old_xyz_to_new_zyx",
    "order_tree_path_xyz",
    "resample_polyline_uniform",
    "tree_to_fiber_path",
    "write_fiber_cache",
    "xyz_to_zyx",
    "zyx_to_xyz",
]
