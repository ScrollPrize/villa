from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FiberTraceSegmentMetadata:
    normal_manifest: str
    fiber_manifest: str
    trace_to_base_scale: float
    max_endpoint_error_base_voxels: float
    config: dict[str, float | int]


@dataclass(frozen=True)
class Vc3dFiber:
    """Parsed VC3D fiber geometry.

    VC3D stores points as x, y, z. Zarr volume access uses z, y, x, so both
    orders are exposed explicitly.
    """

    path: Path | None
    version: int
    line_points_xyz: np.ndarray
    control_points_xyz: np.ndarray
    control_point_segments: tuple[FiberTraceSegmentMetadata | None, ...]
    generation: int
    metadata: dict[str, Any]

    @property
    def line_points_zyx(self) -> np.ndarray:
        return self.line_points_xyz[:, (2, 1, 0)].astype(np.float32, copy=True)

    @property
    def control_points_zyx(self) -> np.ndarray:
        return self.control_points_xyz[:, (2, 1, 0)].astype(np.float32, copy=True)


def _parse_points(
    raw: Any, *, key: str, path: Path | None, min_count: int
) -> np.ndarray:
    label = f"vc3d_fiber {key}"
    if path is not None:
        label = f"{label} in {path}"
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list")
    if len(raw) < min_count:
        raise ValueError(f"{label} must contain at least {min_count} point(s)")

    points = np.asarray(raw, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must have shape [N, 3]")
    if not bool(np.isfinite(points).all()):
        raise ValueError(f"{label} contains non-finite values")
    return points.astype(np.float32, copy=False)


_SEGMENT_KEYS = {
    "optimizer",
    "metadata_version",
    "tracer_version",
    "normal_manifest",
    "fiber_manifest",
    "trace_to_base_scale",
    "max_endpoint_error_base_voxels",
    "config",
}
_CONFIG_KEYS = {
    "step_voxels",
    "cone_angle_degrees",
    "cone_angle_step_degrees",
    "cone_grid_size",
    "beam_width",
    "beam_prune_distance_voxels",
    "beam_lookahead_steps",
    "smoothness_weight",
    "smoothness_normal_weight",
    "smoothness_tangent_weight",
    "smoothness_free_angle_degrees",
    "cumulative_smoothness_steps",
    "cumulative_smoothness_tangent_weight",
    "initial_free_angle_degrees",
    "max_step_factor",
    "fusion_gap_factor",
    "endpoint_accept_threshold_base_voxels",
}


def _parse_segment_metadata(raw: Any) -> FiberTraceSegmentMetadata:
    if not isinstance(raw, dict) or set(raw) != _SEGMENT_KEYS:
        raise ValueError("segment_to_next has missing or unknown fields")
    if raw["optimizer"] != "native_fiber_trace3d":
        raise ValueError(f"unsupported segment_to_next optimizer: {raw['optimizer']!r}")
    if raw["metadata_version"] != 1 or raw["tracer_version"] != 1:
        raise ValueError("unsupported segment_to_next metadata/tracer version")
    config = raw["config"]
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("segment_to_next config has missing or unknown fields")
    normal_manifest = raw["normal_manifest"]
    fiber_manifest = raw["fiber_manifest"]
    if not isinstance(normal_manifest, str) or not normal_manifest:
        raise ValueError("segment_to_next normal_manifest must be a non-empty string")
    if not isinstance(fiber_manifest, str) or not fiber_manifest:
        raise ValueError("segment_to_next fiber_manifest must be a non-empty string")
    trace_scale = float(raw["trace_to_base_scale"])
    endpoint_error = float(raw["max_endpoint_error_base_voxels"])
    if not math.isfinite(trace_scale) or trace_scale <= 0:
        raise ValueError("segment_to_next trace_to_base_scale must be positive")
    if not math.isfinite(endpoint_error) or endpoint_error < 0:
        raise ValueError("segment_to_next endpoint error must be non-negative")
    normalized_config: dict[str, float | int] = {}
    integer_keys = {
        "cone_grid_size",
        "beam_width",
        "beam_lookahead_steps",
        "cumulative_smoothness_steps",
    }
    for key in _CONFIG_KEYS:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"segment_to_next config {key} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"segment_to_next config {key} must be finite")
        normalized_config[key] = int(value) if key in integer_keys else float(value)
    if normalized_config["cone_grid_size"] <= 0 or normalized_config["beam_width"] <= 0:
        raise ValueError("segment_to_next grid and beam sizes must be positive")
    if normalized_config["beam_lookahead_steps"] < 0 or normalized_config["cumulative_smoothness_steps"] < 0:
        raise ValueError("segment_to_next step counts must be non-negative")
    return FiberTraceSegmentMetadata(
        normal_manifest=normal_manifest,
        fiber_manifest=fiber_manifest,
        trace_to_base_scale=trace_scale,
        max_endpoint_error_base_voxels=endpoint_error,
        config=normalized_config,
    )


def _parse_control_points(
    raw: Any, *, version: int, path: Path | None
) -> tuple[np.ndarray, tuple[FiberTraceSegmentMetadata | None, ...]]:
    if version == 1:
        points = _parse_points(raw, key="control_points", path=path, min_count=1)
        return points, tuple(None for _ in range(len(points)))
    if not isinstance(raw, list) or not raw:
        raise ValueError("vc3d_fiber control_points must be a non-empty list")
    positions: list[Any] = []
    segments: list[FiberTraceSegmentMetadata | None] = []
    for index, control in enumerate(raw):
        if not isinstance(control, dict) or not set(control) <= {"position", "segment_to_next"}:
            raise ValueError("version-2 control points must contain only position and segment_to_next")
        if "position" not in control:
            raise ValueError("version-2 control point is missing position")
        positions.append(control["position"])
        segment = control.get("segment_to_next")
        if segment is not None and index + 1 == len(raw):
            raise ValueError("the final control point cannot contain segment_to_next")
        segments.append(None if segment is None else _parse_segment_metadata(segment))
    return (
        _parse_points(positions, key="control_points", path=path, min_count=1),
        tuple(segments),
    )


def parse_vc3d_fiber(
    obj: dict[str, Any], *, path: str | Path | None = None
) -> Vc3dFiber:
    fiber_path = Path(path) if path is not None else None
    if not isinstance(obj, dict):
        raise ValueError(f"vc3d_fiber JSON must be an object, got {type(obj).__name__}")
    if obj.get("type", "vc3d_fiber") != "vc3d_fiber":
        raise ValueError(
            f"vc3d_fiber type must be 'vc3d_fiber', got {obj.get('type')!r}"
        )

    version = int(obj.get("version", 1))
    if version not in {1, 2}:
        raise ValueError(f"only vc3d_fiber versions 1 and 2 are supported, got {version}")

    line_points = _parse_points(
        obj.get("line_points"), key="line_points", path=fiber_path, min_count=2
    )
    control_points, control_point_segments = _parse_control_points(
        obj.get("control_points"), version=version, path=fiber_path
    )
    generation = int(obj.get("generation", 1))

    metadata = {
        key: value
        for key, value in obj.items()
        if key not in {"type", "version", "line_points", "control_points", "generation"}
    }
    return Vc3dFiber(
        path=fiber_path,
        version=version,
        line_points_xyz=line_points,
        control_points_xyz=control_points,
        control_point_segments=control_point_segments,
        generation=generation,
        metadata=metadata,
    )


def load_vc3d_fiber(path: str | Path) -> Vc3dFiber:
    fiber_path = Path(path)
    with fiber_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    return parse_vc3d_fiber(obj, path=fiber_path)
