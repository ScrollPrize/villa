from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def resolve_point_collection_path(path: str | Path) -> Path:
    """Expand a user's home directory without changing relative-path semantics."""
    return Path(path).expanduser()


def load_point_collection(path: str | Path) -> np.ndarray:
    """Load all XYZ points from a version-1 point-collection JSON file."""
    resolved_path = resolve_point_collection_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Point collection does not exist: {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed point collection JSON {resolved_path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise ValueError(f"Point collection {resolved_path} must contain a JSON object.")
    version = document.get("version")
    if version not in (1, "1", "1.0"):
        raise ValueError(
            f"Unsupported point collection version {version!r} in {resolved_path}; expected version 1."
        )
    collections = document.get("collections")
    if not isinstance(collections, Sequence) or isinstance(collections, (str, bytes)):
        raise ValueError(f"Point collection {resolved_path} must contain a collections array.")

    points: list[tuple[float, float, float]] = []
    for collection_index, collection in enumerate(collections):
        if not isinstance(collection, Mapping):
            raise ValueError(
                f"Collection {collection_index} in {resolved_path} must be a JSON object."
            )
        collection_points = collection.get("points")
        if not isinstance(collection_points, Sequence) or isinstance(collection_points, (str, bytes)):
            raise ValueError(
                f"Collection {collection_index} in {resolved_path} must contain a points array."
            )
        for point_index, point in enumerate(collection_points):
            if not isinstance(point, Mapping) or "p" not in point:
                raise ValueError(
                    f"Point {point_index} in collection {collection_index} of {resolved_path} must contain p."
                )
            coordinates = point["p"]
            if (
                not isinstance(coordinates, Sequence)
                or isinstance(coordinates, (str, bytes))
                or len(coordinates) != 3
            ):
                raise ValueError(
                    f"Point {point_index} in collection {collection_index} of {resolved_path} "
                    "must have exactly three XYZ coordinates."
                )
            try:
                xyz = tuple(float(value) for value in coordinates)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Point {point_index} in collection {collection_index} of {resolved_path} "
                    "contains a non-numeric coordinate."
                ) from exc
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError(
                    f"Point {point_index} in collection {collection_index} of {resolved_path} "
                    "contains a non-finite coordinate."
                )
            points.append(xyz)

    if not points:
        raise ValueError(f"Point collection {resolved_path} contains no usable points.")
    return np.asarray(points, dtype=np.float64)


def xyz_to_zyx(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 XYZ point array, got shape {points_xyz.shape}.")
    return points_xyz[:, ::-1].copy()


def map_scale0_voxel_centers(
    points_zyx: np.ndarray,
    scale0_shape: Sequence[int],
    selected_shape: Sequence[int],
) -> np.ndarray:
    """Map scale-0 voxel centers to a co-registered pyramid level."""
    points_zyx = np.asarray(points_zyx, dtype=np.float64)
    scale0 = np.asarray(tuple(int(value) for value in scale0_shape), dtype=np.float64)
    selected = np.asarray(tuple(int(value) for value in selected_shape), dtype=np.float64)
    if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
        raise ValueError(f"Expected an Nx3 ZYX point array, got shape {points_zyx.shape}.")
    if scale0.shape != (3,) or selected.shape != (3,) or np.any(scale0 <= 0) or np.any(selected <= 0):
        raise ValueError(
            f"Scale shapes must contain three positive dimensions, got {tuple(scale0_shape)} and "
            f"{tuple(selected_shape)}."
        )
    return (points_zyx + 0.5) * (selected / scale0) - 0.5
