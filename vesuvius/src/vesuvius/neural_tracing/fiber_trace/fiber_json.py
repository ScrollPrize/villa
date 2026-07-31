from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FiberTraceSegmentMetadata:
    interp_goal: str
    interp_mode: str
    metric: float | None
    msg: str
    outcome: str
    normal_manifest: str
    fiber_manifest: str
    trace_to_base_scale: float
    meeting_error_base_voxels: float | None
    meeting_error_ratio: float | None
    meeting_source: str
    failure_code: str
    failure_detail: str
    lasagna_failure_code: str
    lasagna_failure_detail: str
    config: dict[str, float | int]

    @property
    def max_endpoint_error_base_voxels(self) -> float:
        """Previous-schema compatibility alias for accepted-only callers."""
        return (
            math.nan
            if self.meeting_error_base_voxels is None
            else self.meeting_error_base_voxels
        )


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


_SEGMENT_KEYS_V3 = {
    "optimizer",
    "metadata_version",
    "tracer_version",
    "interp_goal",
    "interp_mode",
    "metric",
    "msg",
    "normal_manifest",
    "fiber_manifest",
    "trace_to_base_scale",
    "meeting_error_base_voxels",
    "meeting_error_ratio",
    "meeting_source",
    "failure_code",
    "failure_detail",
    "lasagna_failure_code",
    "lasagna_failure_detail",
    "config",
}
_CONFIG_KEYS_COMMON = {
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
    "endpoint_accept_threshold_base_voxels",
}
_CONFIG_KEYS_V3 = _CONFIG_KEYS_COMMON | {"meeting_accept_max_error_ratio"}


def _parse_segment_metadata(raw: Any) -> FiberTraceSegmentMetadata:
    if not isinstance(raw, dict):
        raise ValueError("segment_to_next has missing or unknown fields")
    if raw["optimizer"] != "native_fiber_trace3d":
        raise ValueError(f"unsupported segment_to_next optimizer: {raw['optimizer']!r}")
    version = (raw.get("metadata_version"), raw.get("tracer_version"))
    if version != (3, 2):
        raise ValueError("unsupported segment_to_next metadata/tracer version")
    if set(raw) != _SEGMENT_KEYS_V3:
        raise ValueError("segment_to_next has missing or unknown fields")
    config = raw["config"]
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS_V3:
        raise ValueError("segment_to_next config has missing or unknown fields")
    normal_manifest = raw["normal_manifest"]
    fiber_manifest = raw["fiber_manifest"]
    if not isinstance(normal_manifest, str):
        raise ValueError("segment_to_next normal_manifest must be a string")
    if not isinstance(fiber_manifest, str):
        raise ValueError("segment_to_next fiber_manifest must be a string")
    trace_scale = float(raw["trace_to_base_scale"])
    if not math.isfinite(trace_scale) or trace_scale <= 0:
        raise ValueError("segment_to_next trace_to_base_scale must be positive")
    interp_goal = raw["interp_goal"]
    interp_mode = raw["interp_mode"]
    if interp_goal not in {"global", "cspline", "lasagna", "trace"}:
        raise ValueError("segment_to_next interp_goal is invalid")
    if interp_mode not in {"cspline", "lasagna", "trace"}:
        raise ValueError("segment_to_next interp_mode is invalid")
    metric = None if raw["metric"] is None else float(raw["metric"])
    if metric is not None and (not math.isfinite(metric) or metric < 0):
        raise ValueError("segment_to_next metric must be non-negative")
    if interp_mode == "cspline" and metric is not None:
        raise ValueError("cspline segment_to_next cannot contain metric")
    msg = raw["msg"]
    failure_code = raw["failure_code"]
    failure_detail = raw["failure_detail"]
    lasagna_failure_code = raw["lasagna_failure_code"]
    lasagna_failure_detail = raw["lasagna_failure_detail"]
    if not all(isinstance(value, str) for value in (
        msg, failure_code, failure_detail,
        lasagna_failure_code, lasagna_failure_detail,
    )):
        raise ValueError("segment_to_next diagnostics must be strings")
    raw_error = raw["meeting_error_base_voxels"]
    raw_ratio = raw["meeting_error_ratio"]
    meeting_error = None if raw_error is None else float(raw_error)
    meeting_ratio = None if raw_ratio is None else float(raw_ratio)
    meeting_source = raw["meeting_source"]
    if interp_mode == "trace":
        if (meeting_error is None or meeting_ratio is None or not meeting_source or
            metric is None or failure_code or failure_detail or
            not normal_manifest or not fiber_manifest):
            raise ValueError("trace segment_to_next is inconsistent")
        outcome = "accepted_native"
    else:
        if meeting_error is not None or meeting_ratio is not None or meeting_source:
            raise ValueError("non-trace segment contains meeting diagnostics")
        outcome = "lasagna_fallback"
    normalized_config: dict[str, float | int] = {}
    integer_keys = {
        "cone_grid_size",
        "beam_width",
        "beam_lookahead_steps",
        "cumulative_smoothness_steps",
    }
    for key in _CONFIG_KEYS_V3:
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
    if not 0 <= normalized_config["meeting_accept_max_error_ratio"] <= 1:
        raise ValueError("segment_to_next meeting_accept_max_error_ratio must be in [0, 1]")
    return FiberTraceSegmentMetadata(
        interp_goal=interp_goal,
        interp_mode=interp_mode,
        metric=metric,
        msg=msg,
        outcome=outcome,
        normal_manifest=normal_manifest,
        fiber_manifest=fiber_manifest,
        trace_to_base_scale=trace_scale,
        meeting_error_base_voxels=meeting_error,
        meeting_error_ratio=meeting_ratio,
        meeting_source=meeting_source,
        failure_code=failure_code,
        failure_detail=failure_detail,
        lasagna_failure_code=lasagna_failure_code,
        lasagna_failure_detail=lasagna_failure_detail,
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
            raise ValueError("version-3 control points must contain only position and segment_to_next")
        if "position" not in control:
            raise ValueError("version-3 control point is missing position")
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
    if version not in {1, 3}:
        raise ValueError(f"only vc3d_fiber versions 1 and 3 are supported, got {version}")

    line_points = _parse_points(
        obj.get("line_points"), key="line_points", path=fiber_path, min_count=2
    )
    control_points, control_point_segments = _parse_control_points(
        obj.get("control_points"), version=version, path=fiber_path
    )
    if version == 1:
        control_point_segments = tuple(
            None if index + 1 == len(control_point_segments) else FiberTraceSegmentMetadata(
                interp_goal="global",
                interp_mode="lasagna",
                metric=None,
                msg="lasagna",
                outcome="lasagna_fallback",
                normal_manifest="",
                fiber_manifest="",
                trace_to_base_scale=1.0,
                meeting_error_base_voxels=None,
                meeting_error_ratio=None,
                meeting_source="",
                failure_code="",
                failure_detail="",
                lasagna_failure_code="",
                lasagna_failure_detail="",
                config={},
            )
            for index in range(len(control_point_segments))
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
