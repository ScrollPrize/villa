from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SEGMENT_KEYS_V3 = {
    "optimizer", "metadata_version", "tracer_version", "interp_goal",
    "interp_mode", "metric", "msg", "normal_manifest", "fiber_manifest",
    "trace_to_base_scale", "meeting_error_base_voxels",
    "meeting_error_ratio", "meeting_source", "failure_code",
    "failure_detail", "lasagna_failure_code", "lasagna_failure_detail",
    "config",
}
_CONFIG_KEYS_V3 = {
    "step_voxels", "cone_angle_degrees", "cone_angle_step_degrees",
    "cone_grid_size", "beam_width", "beam_prune_distance_voxels",
    "beam_lookahead_steps", "smoothness_weight",
    "smoothness_normal_weight", "smoothness_tangent_weight",
    "smoothness_free_angle_degrees", "cumulative_smoothness_steps",
    "cumulative_smoothness_tangent_weight", "initial_free_angle_degrees",
    "max_step_factor", "meeting_accept_max_error_ratio",
    "endpoint_accept_threshold_base_voxels",
}
_INTEGER_CONFIG_KEYS = {
    "cone_grid_size", "beam_width", "beam_lookahead_steps",
    "cumulative_smoothness_steps",
}
_POSITIVE_CONFIG_KEYS = {
    "step_voxels", "cone_angle_degrees", "cone_angle_step_degrees",
    "cone_grid_size", "beam_width", "beam_prune_distance_voxels",
    "max_step_factor", "endpoint_accept_threshold_base_voxels",
}
_NON_NEGATIVE_CONFIG_KEYS = _CONFIG_KEYS_V3 - _POSITIVE_CONFIG_KEYS


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
        return (
            math.nan
            if self.meeting_error_base_voxels is None
            else self.meeting_error_base_voxels
        )


@dataclass(frozen=True)
class ParsedVc3dFiber:
    path: Path | None
    version: int
    optimization_mode: str
    line_points_xyz: tuple[tuple[float, float, float], ...]
    control_points_xyz: tuple[tuple[float, float, float], ...]
    control_point_segments: tuple[FiberTraceSegmentMetadata | None, ...]
    generation: int
    metadata: dict[str, Any]


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _points(raw: Any, *, label: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list")
    result: list[tuple[float, float, float]] = []
    for index, point in enumerate(raw):
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"{label}[{index}] must be [x, y, z]")
        result.append(tuple(
            _number(value, label=f"{label}[{index}]") for value in point
        ))
    return tuple(result)


def parse_segment_metadata(raw: Any) -> FiberTraceSegmentMetadata:
    if not isinstance(raw, dict) or set(raw) != _SEGMENT_KEYS_V3:
        raise ValueError("segment_to_next has missing or unknown fields")
    if raw["optimizer"] != "native_fiber_trace3d":
        raise ValueError(f"unsupported segment_to_next optimizer: {raw['optimizer']!r}")
    if (raw["metadata_version"], raw["tracer_version"]) != (3, 2):
        raise ValueError("unsupported segment_to_next metadata/tracer version")
    if not isinstance(raw["normal_manifest"], str) or not isinstance(raw["fiber_manifest"], str):
        raise ValueError("segment_to_next manifests must be strings")

    goal = raw["interp_goal"]
    mode = raw["interp_mode"]
    if not isinstance(goal, str) or goal not in {"global", "cspline", "lasagna", "trace"}:
        raise ValueError("segment_to_next interp_goal is invalid")
    if not isinstance(mode, str) or mode not in {"cspline", "lasagna", "trace"}:
        raise ValueError("segment_to_next interp_mode is invalid")

    trace_scale = _number(raw["trace_to_base_scale"], label="trace_to_base_scale")
    if trace_scale <= 0:
        raise ValueError("segment_to_next trace_to_base_scale must be positive")
    metric = None if raw["metric"] is None else _number(raw["metric"], label="metric")
    if metric is not None and metric < 0:
        raise ValueError("segment_to_next metric must be non-negative")
    if mode == "cspline" and metric is not None:
        raise ValueError("cspline segment_to_next cannot contain metric")

    string_keys = {
        "msg", "meeting_source", "failure_code", "failure_detail",
        "lasagna_failure_code", "lasagna_failure_detail",
    }
    if not all(isinstance(raw[key], str) for key in string_keys):
        raise ValueError("segment_to_next diagnostics must be strings")

    error = None if raw["meeting_error_base_voxels"] is None else _number(
        raw["meeting_error_base_voxels"], label="meeting_error_base_voxels"
    )
    ratio = None if raw["meeting_error_ratio"] is None else _number(
        raw["meeting_error_ratio"], label="meeting_error_ratio"
    )
    if ((error is not None and error < 0)
            or (ratio is not None and ratio < 0)):
        raise ValueError("trace meeting diagnostics must be non-negative")
    if mode == "trace":
        if (
            metric is None or error is None or ratio is None
            or not raw["meeting_source"] or raw["failure_code"]
            or raw["failure_detail"] or not raw["normal_manifest"]
            or not raw["fiber_manifest"]
        ):
            raise ValueError("trace segment_to_next is inconsistent")
        outcome = "accepted_native"
    else:
        if error is not None or ratio is not None or raw["meeting_source"]:
            raise ValueError("non-trace segment contains meeting diagnostics")
        outcome = "lasagna_fallback"

    config = raw["config"]
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS_V3:
        raise ValueError("segment_to_next config has missing or unknown fields")
    normalized_config: dict[str, float | int] = {}
    for key in _CONFIG_KEYS_V3:
        value = _number(config[key], label=f"segment_to_next config {key}")
        if key in _INTEGER_CONFIG_KEYS:
            if not isinstance(config[key], int) or isinstance(config[key], bool):
                raise ValueError(f"segment_to_next config {key} must be an integer")
            normalized_config[key] = int(config[key])
        else:
            normalized_config[key] = value
        if key in _POSITIVE_CONFIG_KEYS and value <= 0:
            raise ValueError(f"segment_to_next config {key} must be positive")
        if key in _NON_NEGATIVE_CONFIG_KEYS and value < 0:
            raise ValueError(f"segment_to_next config {key} must be non-negative")
    if normalized_config["meeting_accept_max_error_ratio"] > 1:
        raise ValueError("segment_to_next meeting_accept_max_error_ratio must be in [0, 1]")

    return FiberTraceSegmentMetadata(
        interp_goal=goal,
        interp_mode=mode,
        metric=metric,
        msg=raw["msg"],
        outcome=outcome,
        normal_manifest=raw["normal_manifest"],
        fiber_manifest=raw["fiber_manifest"],
        trace_to_base_scale=trace_scale,
        meeting_error_base_voxels=error,
        meeting_error_ratio=ratio,
        meeting_source=raw["meeting_source"],
        failure_code=raw["failure_code"],
        failure_detail=raw["failure_detail"],
        lasagna_failure_code=raw["lasagna_failure_code"],
        lasagna_failure_detail=raw["lasagna_failure_detail"],
        config=normalized_config,
    )


def legacy_lasagna_segments(count: int) -> tuple[FiberTraceSegmentMetadata | None, ...]:
    return tuple(
        None if index + 1 == count else FiberTraceSegmentMetadata(
            interp_goal="global", interp_mode="lasagna", metric=None,
            msg="lasagna", outcome="lasagna_fallback", normal_manifest="",
            fiber_manifest="", trace_to_base_scale=1.0,
            meeting_error_base_voxels=None, meeting_error_ratio=None,
            meeting_source="", failure_code="", failure_detail="",
            lasagna_failure_code="", lasagna_failure_detail="", config={},
        )
        for index in range(count)
    )


def parse_vc3d_fiber_format(
    obj: Any, *, path: str | Path | None = None
) -> ParsedVc3dFiber:
    fiber_path = Path(path) if path is not None else None
    label = f"vc3d_fiber in {fiber_path}" if fiber_path is not None else "vc3d_fiber"
    if not isinstance(obj, dict):
        raise ValueError(f"{label} JSON must be an object")
    if obj.get("type", "vc3d_fiber") != "vc3d_fiber":
        raise ValueError(f"{label} type must be 'vc3d_fiber'")
    version = obj.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version not in {1, 3}:
        raise ValueError(f"only vc3d_fiber versions 1 and 3 are supported, got {version!r}")
    if version == 3 and "type" not in obj:
        raise ValueError("version-3 vc3d_fiber is missing type")

    if version == 3 and "optimization_mode" not in obj:
        raise ValueError("version-3 vc3d_fiber is missing optimization_mode")
    optimization_mode = obj.get("optimization_mode", "lasagna")
    if not isinstance(optimization_mode, str) or optimization_mode not in {
        "lasagna", "native_fiber_trace3d"
    }:
        raise ValueError("vc3d_fiber optimization_mode is invalid")

    line_points = _points(obj.get("line_points"), label=f"{label} line_points")
    raw_controls = obj.get("control_points")
    if not isinstance(raw_controls, list):
        raise ValueError(f"{label} control_points must be a list")
    if version == 1:
        control_points = _points(raw_controls, label=f"{label} control_points")
        segments = legacy_lasagna_segments(len(control_points))
    else:
        positions: list[Any] = []
        parsed_segments: list[FiberTraceSegmentMetadata | None] = []
        for index, control in enumerate(raw_controls):
            if not isinstance(control, dict) or not set(control) <= {"position", "segment_to_next"}:
                raise ValueError("version-3 control points must contain only position and segment_to_next")
            if "position" not in control:
                raise ValueError("version-3 control point is missing position")
            positions.append(control["position"])
            if index + 1 == len(raw_controls):
                if "segment_to_next" in control:
                    raise ValueError("the final control point cannot contain segment_to_next")
                parsed_segments.append(None)
            else:
                if "segment_to_next" not in control:
                    raise ValueError("a non-final control point is missing segment_to_next")
                parsed_segments.append(parse_segment_metadata(control["segment_to_next"]))
        control_points = _points(positions, label=f"{label} control_points")
        segments = tuple(parsed_segments)

    generation = obj.get("generation", 1)
    if version == 3:
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ValueError("vc3d_fiber generation must be a non-negative integer")
    else:
        try:
            generation = int(generation)
        except (TypeError, ValueError) as exc:
            raise ValueError("vc3d_fiber generation must be an integer") from exc
    metadata = {
        key: value for key, value in obj.items()
        if key not in {
            "type", "version", "line_points", "control_points", "generation",
            "optimization_mode",
        }
    }
    return ParsedVc3dFiber(
        path=fiber_path,
        version=version,
        optimization_mode=optimization_mode,
        line_points_xyz=line_points,
        control_points_xyz=control_points,
        control_point_segments=segments,
        generation=generation,
        metadata=metadata,
    )
