"""View a base-coordinate crop of a fiber-presence OME-Zarr in napari."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OmeZarrLevel:
    root: Path
    path: str
    scale_zyx: tuple[float, float, float]
    translation_zyx: tuple[float, float, float]

    @property
    def array_path(self) -> Path:
        return self.root / self.path


@dataclass(frozen=True)
class CropSelection:
    requested_base_xyzwhd: tuple[int, int, int, int, int, int]
    slices_zyx: tuple[slice, slice, slice]
    origin_base_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class LineObjGeometry:
    paths_zyx: list[np.ndarray]
    total_groups: int
    trace_loss_total: list[float]
    loss_per_prediction_voxel: list[float]
    relative_quality: list[float]


@dataclass(frozen=True)
class AnchorCellGeometry:
    centers_zyx: np.ndarray
    displacements_zyx: list[np.ndarray]


@dataclass(frozen=True)
class AnchorStageGeometry:
    stage: str
    paths_zyx: list[np.ndarray]
    features: dict[str, list]
    record_count: int
    geometric_record_count: int
    reasons: dict[str, int]
    binding: dict
    records: tuple[dict, ...]


@dataclass(frozen=True)
class SurfaceObjGeometry:
    vertices_zyx: np.ndarray
    triangles: np.ndarray
    normalized_ct_intensity: np.ndarray
    component_count: int


@dataclass(frozen=True)
class ReplayStripGeometry:
    reference: SurfaceObjGeometry
    greedy: SurfaceObjGeometry
    fiberlet: SurfaceObjGeometry


@dataclass(frozen=True)
class FiberReplayBundle:
    path: Path
    tracer: str
    tracer_failure_index: int
    status: str
    crop_xyzwhd: tuple[int, int, int, int, int, int]
    prediction_shape_zyx: tuple[int, int, int]
    prediction_to_base_scale: float
    fiber_manifest_content_hash: str
    reference_zyx: np.ndarray
    greedy_segments_zyx: tuple[np.ndarray, ...]
    fiberlet_segments_zyx: tuple[np.ndarray, ...]
    failure_zyx: np.ndarray | None
    tube_radius_base_voxels: float | None
    anchors_obj: Path | None
    anchor_cells_obj: Path | None
    anchor_stages: tuple[AnchorStageGeometry, ...]
    paths_obj: Path | None
    strip_metadata: dict | None = None
    strip_artifacts: tuple[tuple[Path, Path, Path], ...] | None = None


@dataclass(frozen=True)
class ReplayVisualArtifacts:
    anchors: LineObjGeometry
    anchor_cells: AnchorCellGeometry
    anchor_stages: tuple[AnchorStageGeometry, ...]
    fiberlets: LineObjGeometry
    strips: ReplayStripGeometry | None = None


@dataclass
class ReplayGeometryFilter:
    key: str
    layer: object
    source_data: np.ndarray | tuple[np.ndarray, ...]
    source_features: dict[str, tuple] | None
    distances_base_voxels: np.ndarray
    color_attribute: str
    color_value: str
    empty_color_value: str | None
    display_width: float | None
    display_size: float | None


_LINE_OBJ_HEADERS = {
    "anchors": "# vc_fiberlet_anchors version 1",
    "paths": "# vc_fiberlets version 1",
}


_FIBERLET_QUALITY_COLORMAP = "red-yellow-green"
_DEFAULT_PRESENCE_RADIUS_BASE_VOXELS = 32.0
_DEFAULT_ANCHOR_RADIUS_BASE_VOXELS = 32.0
_DEFAULT_FIBERLET_RADIUS_BASE_VOXELS = 16.0
_ANCHOR_STAGE_NAMES = ("initialized", "refined", "support", "selection", "nms")
_ANCHOR_STAGE_COLORS = {
    "initialized": "blue",
    "refined": "magenta",
    "support": "orange",
    "selection": "white",
    "nms": "lime",
}


def replay_display_radius_defaults_base() -> dict[str, float]:
    """Return independent replay display radii in base voxels."""
    return {
        "presence": _DEFAULT_PRESENCE_RADIUS_BASE_VOXELS,
        "anchors": _DEFAULT_ANCHOR_RADIUS_BASE_VOXELS,
        "fiberlets": _DEFAULT_FIBERLET_RADIUS_BASE_VOXELS,
    }


def _fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def _strict_xyz_points(value, name: str, minimum: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < minimum:
        raise ValueError(f"{name} must contain at least {minimum} XYZ points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return points


def _polyline_arcs(points: np.ndarray, name: str) -> np.ndarray:
    arcs = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    if arcs[-1] <= 1.0e-12:
        raise ValueError(f"{name} has no non-degenerate edge")
    return arcs


def _sample_polyline_arc(
    points: np.ndarray, arcs: np.ndarray, arc: float
) -> np.ndarray:
    segment = max(
        0,
        min(int(np.searchsorted(arcs, arc, side="right") - 1), len(points) - 2),
    )
    while segment + 1 < len(points) and arcs[segment + 1] <= arcs[segment] + 1.0e-12:
        segment += 1
    if segment + 1 >= len(points):
        segment = len(points) - 2
        while segment > 0 and arcs[segment + 1] <= arcs[segment] + 1.0e-12:
            segment -= 1
    fraction = (arc - arcs[segment]) / (arcs[segment + 1] - arcs[segment])
    return points[segment] + np.clip(fraction, 0.0, 1.0) * (
        points[segment + 1] - points[segment]
    )


def _slice_polyline_arc(points: np.ndarray, begin: float, end: float) -> np.ndarray:
    arcs = _polyline_arcs(points, "trace_points_base_xyz")
    selected = [_sample_polyline_arc(points, arcs, begin)]
    selected.extend(
        points[index]
        for index in range(1, len(points) - 1)
        if arcs[index] > begin + 1.0e-12 and arcs[index] < end - 1.0e-12
    )
    endpoint = _sample_polyline_arc(points, arcs, end)
    if np.linalg.norm(endpoint - selected[-1]) > 1.0e-12:
        selected.append(endpoint)
    return np.asarray(selected, dtype=np.float64)


def _read_replay_obj(path: Path, header: str, point_record: bool) -> np.ndarray:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read replay OBJ {path}: {exc}") from exc
    if not lines or lines[0].strip() != header:
        raise ValueError(f"{path}: unsupported replay OBJ header")
    vertices: list[list[float]] = []
    records: list[list[int]] = []
    record_name = "p" if point_record else "l"
    for line_number, raw in enumerate(lines[1:], start=2):
        fields = raw.strip().split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) == 4:
            try:
                vertices.append([float(item) for item in fields[1:]])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid vertex") from exc
        elif fields[0] == record_name:
            try:
                records.append([int(item) for item in fields[1:]])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid index") from exc
        else:
            raise ValueError(f"{path}:{line_number}: unsupported OBJ record")
    expected = (
        [1]
        if point_record
        else ([1, 1] if len(vertices) == 1 else list(range(1, len(vertices) + 1)))
    )
    if len(records) != 1 or records[0] != expected:
        raise ValueError(f"{path}: replay OBJ topology does not match its vertices")
    return _strict_xyz_points(vertices, str(path), 1)


def read_anchor_cell_obj(path: str | Path) -> AnchorCellGeometry:
    """Read cell-center points and center-to-anchor displacement lines."""
    obj_path = Path(path)
    try:
        lines = obj_path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read anchor-cell OBJ {obj_path}: {exc}") from exc
    if not lines or lines[0].strip() != "# vc_fiberlet_anchor_cells version 1":
        raise ValueError(f"{obj_path}: unsupported anchor-cell OBJ header")
    vertices: list[list[float]] = []
    points: list[int] = []
    connectors: list[tuple[int, int]] = []
    has_group = False
    for line_number, raw in enumerate(lines[1:], start=2):
        fields = raw.strip().split()
        if not fields:
            continue
        if fields[0] == "g" and len(fields) == 2:
            has_group = True
        elif fields[0] == "v" and len(fields) == 4 and has_group:
            try:
                vertex = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"{obj_path}:{line_number}: invalid vertex") from exc
            if not np.isfinite(vertex).all():
                raise ValueError(f"{obj_path}:{line_number}: non-finite vertex")
            vertices.append(vertex)
        elif fields[0] == "p" and len(fields) == 2 and has_group:
            try:
                points.append(int(fields[1]))
            except ValueError as exc:
                raise ValueError(
                    f"{obj_path}:{line_number}: invalid point index"
                ) from exc
        elif fields[0] == "l" and len(fields) == 3 and has_group:
            try:
                connectors.append((int(fields[1]), int(fields[2])))
            except ValueError as exc:
                raise ValueError(
                    f"{obj_path}:{line_number}: invalid line index"
                ) from exc
        else:
            raise ValueError(
                f"{obj_path}:{line_number}: unsupported anchor-cell OBJ record"
            )
    if not points or len(set(points)) != len(points):
        raise ValueError(f"{obj_path}: anchor-cell points are empty or duplicated")
    if any(index < 1 or index > len(vertices) for index in points):
        raise ValueError(f"{obj_path}: anchor-cell point index is out of range")
    point_set = set(points)
    if any(
        start not in point_set
        or target < 1
        or target > len(vertices)
        or target in point_set
        for start, target in connectors
    ):
        raise ValueError(f"{obj_path}: anchor-cell connector topology is invalid")
    xyz = np.asarray(vertices, dtype=np.float64)
    centers = xyz[np.asarray(points, dtype=np.int64) - 1, ::-1].copy()
    displacements = [
        xyz[np.asarray((start, target), dtype=np.int64) - 1, ::-1].copy()
        for start, target in connectors
    ]
    return AnchorCellGeometry(centers, displacements)


def _finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def read_anchor_stage_json(
    path: str | Path, expected_stage: str
) -> AnchorStageGeometry:
    """Read one compatible anchor-pipeline diagnostic stage."""
    stage_path = Path(path)
    try:
        root = json.loads(stage_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read anchor stage {stage_path}: {exc}") from exc
    required_root = {
        "format",
        "version",
        "stage",
        "source",
        "coordinates",
        "selection",
        "glyph_length_base_voxels",
        "summary",
        "records",
    }
    if (
        not isinstance(root, dict)
        or not required_root.issubset(root)
        or root.get("format") != "vc_fiberlet_anchor_stage"
        or root.get("version") != 1
        or root.get("stage") != expected_stage
        or expected_stage not in _ANCHOR_STAGE_NAMES
    ):
        raise ValueError(f"{stage_path}: unsupported anchor-stage schema")
    source = root["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"manifest", "manifest_content_hash"}
        or not all(isinstance(value, str) and value for value in source.values())
    ):
        raise ValueError(f"{stage_path}: invalid anchor-stage source")
    coordinates = root["coordinates"]
    coordinate_keys = {
        "position_order",
        "cell_index_order",
        "position_space",
        "prediction_to_base_scale",
        "prediction_shape_zyx",
    }
    if not isinstance(coordinates, dict) or set(coordinates) not in (
        coordinate_keys,
        coordinate_keys | {"base_voxel_size_um"},
    ):
        raise ValueError(f"{stage_path}: invalid anchor-stage coordinates")
    scale = coordinates.get("prediction_to_base_scale")
    shape = coordinates.get("prediction_shape_zyx")
    if (
        coordinates.get("position_order") != "XYZ"
        or coordinates.get("cell_index_order") != "ZYX"
        or coordinates.get("position_space") != "base_volume"
        or not _finite_number(scale)
        or scale <= 0
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in shape
        )
    ):
        raise ValueError(f"{stage_path}: invalid anchor-stage grid")
    if "base_voxel_size_um" in coordinates and (
        not _finite_number(coordinates["base_voxel_size_um"])
        or coordinates["base_voxel_size_um"] <= 0
    ):
        raise ValueError(f"{stage_path}: invalid base voxel size")
    glyph = root["glyph_length_base_voxels"]
    if not _finite_number(glyph) or glyph <= 0:
        raise ValueError(f"{stage_path}: invalid anchor-stage dimensions")
    selection = root["selection"]
    cells = selection.get("cells_zyx") if isinstance(selection, dict) else None
    if (
        not isinstance(selection, dict)
        or set(selection) != {"cells_zyx"}
        or not isinstance(cells, list)
        or not cells
    ):
        raise ValueError(f"{stage_path}: invalid anchor-stage selection")
    cell_keys = []
    for cell in cells:
        if (
            not isinstance(cell, list)
            or len(cell) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in cell
            )
            or any(value < 0 for value in cell)
        ):
            raise ValueError(f"{stage_path}: invalid selected anchor cell")
        cell_keys.append(tuple(cell))
    if cell_keys != sorted(set(cell_keys)):
        raise ValueError(f"{stage_path}: anchor-stage cells are not canonical")

    record_keys = {
        "cell_zyx",
        "candidate_id",
        "parent_ids",
        "geometry",
        "metrics",
        "transition",
    }
    metric_keys = {
        "assigned_observations",
        "objective_contribution",
        "aligned_support",
        "directional_coherence",
        "refinement_score",
        "refinement_iterations",
    }
    transition_keys = {
        "outcome",
        "reason",
        "successor_id",
        "tested_value",
        "threshold",
        "suppressor",
    }
    records = root["records"]
    if not isinstance(records, list):
        raise ValueError(f"{stage_path}: records must be a list")  # noqa: TRY004
    normalized = []
    paths = []
    features = {
        name: []
        for name in (
            "cell_zyx",
            "candidate_id",
            "parent_ids",
            "assigned_observations",
            "aligned_support",
            "directional_coherence",
            "refinement_score",
            "refinement_iterations",
            "transition",
            "reason",
            "tested_value",
            "threshold",
            "suppressor",
        )
    }
    selected_set = set(cell_keys)
    previous_key = None
    outcome_counts = {}
    reason_counts = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != record_keys:
            raise ValueError(f"{stage_path}: malformed anchor-stage record")
        cell = record["cell_zyx"]
        candidate_id = record["candidate_id"]
        parents = record["parent_ids"]
        if (
            not isinstance(cell, list)
            or tuple(cell) not in selected_set
            or not isinstance(candidate_id, int)
            or isinstance(candidate_id, bool)
            or candidate_id < 0
            or not isinstance(parents, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in parents
            )
            or len(parents) != len(set(parents))
        ):
            raise ValueError(f"{stage_path}: invalid anchor-stage identity")
        key = (tuple(cell), candidate_id)
        if previous_key is not None and key <= previous_key:
            raise ValueError(f"{stage_path}: records are not canonical")
        previous_key = key
        geometry = record["geometry"]
        path_zyx = None
        if geometry is not None:
            if not isinstance(geometry, dict) or set(geometry) != {
                "position_base_xyz",
                "axis_xyz",
            }:
                raise ValueError(f"{stage_path}: malformed anchor geometry")
            position = _strict_xyz_points(
                [geometry["position_base_xyz"]], "position_base_xyz", 1
            )[0]
            axis = _strict_xyz_points([geometry["axis_xyz"]], "axis_xyz", 1)[0]
            if not math.isclose(
                float(np.linalg.norm(axis)), 1.0, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"{stage_path}: anchor axis is not unit length")
            if axis[int(np.argmax(np.abs(axis)))] < 0:
                raise ValueError(f"{stage_path}: anchor axis is not canonical")
            half = axis * (float(glyph) * 0.5)
            path_zyx = np.asarray([position - half, position + half])[:, ::-1]
        metrics = record["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != metric_keys:
            raise ValueError(f"{stage_path}: malformed anchor metrics")
        for name, value in metrics.items():
            if value is None:
                continue
            if name in {"assigned_observations", "refinement_iterations"}:
                valid = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                )
            else:
                valid = _finite_number(value)
            if not valid:
                raise ValueError(f"{stage_path}: invalid metric {name}")
        transition = record["transition"]
        if not isinstance(transition, dict) or set(transition) != transition_keys:
            raise ValueError(f"{stage_path}: malformed anchor transition")
        outcome = transition["outcome"]
        reason = transition["reason"]
        if outcome not in {"continue", "rejected", "merged", "final"}:
            raise ValueError(f"{stage_path}: invalid transition outcome")
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise ValueError(f"{stage_path}: invalid transition reason")
        successor = transition["successor_id"]
        if successor is not None and (
            not isinstance(successor, int)
            or isinstance(successor, bool)
            or successor < 0
        ):
            raise ValueError(f"{stage_path}: invalid transition successor")
        for name in ("tested_value", "threshold"):
            if transition[name] is not None and not _finite_number(transition[name]):
                raise ValueError(f"{stage_path}: invalid transition {name}")
        suppressor = transition["suppressor"]
        if suppressor is not None:
            suppressor_keys = {
                "cell_zyx",
                "candidate_id",
                "external_context",
                "aligned_support",
                "directional_coherence",
            }
            if (
                not isinstance(suppressor, dict)
                or set(suppressor) != suppressor_keys
                or not isinstance(suppressor["cell_zyx"], list)
                or len(suppressor["cell_zyx"]) != 3
                or not isinstance(suppressor["candidate_id"], int)
                or suppressor["candidate_id"] < 0
                or not isinstance(suppressor["external_context"], bool)
                or not _finite_number(suppressor["aligned_support"])
                or not _finite_number(suppressor["directional_coherence"])
            ):
                raise ValueError(f"{stage_path}: invalid NMS suppressor")
        if outcome == "merged":
            if reason != "merged_same_direction" or successor is None:
                raise ValueError(f"{stage_path}: invalid merge transition")
        elif successor is not None:
            raise ValueError(f"{stage_path}: unexpected merge successor")
        if outcome in {"continue", "final"} and reason is not None:
            raise ValueError(f"{stage_path}: surviving transition has a reason")
        if (reason == "nms_suppressed") != (suppressor is not None):
            raise ValueError(f"{stage_path}: inconsistent NMS suppressor")
        if reason in {"below_support", "outside_selection"} and (
            transition["tested_value"] is None or transition["threshold"] is None
        ):
            raise ValueError(f"{stage_path}: threshold rejection lacks values")
        allowed_outcomes = {
            "initialized": {"continue", "rejected", "merged"},
            "refined": {"continue", "rejected"},
            "support": {"continue", "rejected"},
            "selection": {"continue", "rejected"},
            "nms": {"final"},
        }
        allowed_reasons = {
            "initialized": {None, "empty", "degenerate", "merged_same_direction"},
            "refined": {None, "empty", "below_support"},
            "support": {None, "outside_selection"},
            "selection": {None, "nms_suppressed"},
            "nms": {None},
        }
        if (
            outcome not in allowed_outcomes[expected_stage]
            or reason not in allowed_reasons[expected_stage]
        ):
            raise ValueError(f"{stage_path}: transition does not belong to its stage")
        if expected_stage != "initialized" and geometry is None:
            raise ValueError(
                f"{stage_path}: post-initialization record has no geometry"
            )
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if reason is not None:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        normalized.append(record)
        if path_zyx is not None:
            paths.append(path_zyx)
            features["cell_zyx"].append("_".join(str(value) for value in cell))
            features["candidate_id"].append(candidate_id)
            features["parent_ids"].append(",".join(str(value) for value in parents))
            for name in (
                "assigned_observations",
                "aligned_support",
                "directional_coherence",
                "refinement_score",
                "refinement_iterations",
            ):
                features[name].append(metrics[name])
            features["transition"].append(outcome)
            features["reason"].append(reason or "")
            features["tested_value"].append(transition["tested_value"])
            features["threshold"].append(transition["threshold"])
            features["suppressor"].append(
                ""
                if suppressor is None
                else (
                    f"{suppressor['cell_zyx']}:{suppressor['candidate_id']}"
                    f" external={suppressor['external_context']}"
                )
            )
    if expected_stage == "initialized":
        expected = [(cell, candidate) for cell in cell_keys for candidate in (0, 1)]
        actual = [
            (tuple(record["cell_zyx"]), record["candidate_id"]) for record in normalized
        ]
        if actual != expected:
            raise ValueError(f"{stage_path}: initialized attempts are incomplete")
    expected_summary = {
        "record_count": len(normalized),
        "geometric_record_count": len(paths),
        "outcomes": outcome_counts,
        "reasons": reason_counts,
    }
    if root["summary"] != expected_summary:
        raise ValueError(f"{stage_path}: summary is inconsistent")
    return AnchorStageGeometry(
        stage=expected_stage,
        paths_zyx=paths,
        features=features,
        record_count=len(normalized),
        geometric_record_count=len(paths),
        reasons=reason_counts,
        binding={
            "source": source,
            "coordinates": coordinates,
            "selection": selection,
            "glyph_length_base_voxels": glyph,
        },
        records=tuple(normalized),
    )


def validate_anchor_stage_chain(
    stages: Sequence[AnchorStageGeometry], final_anchors: LineObjGeometry
) -> None:
    """Validate lineage and unchanged geometry through all anchor stages."""
    if tuple(stage.stage for stage in stages) != _ANCHOR_STAGE_NAMES:
        raise ValueError("anchor-stage chain is incomplete or out of order")
    if any(stage.binding != stages[0].binding for stage in stages[1:]):
        raise ValueError("anchor-stage files have mixed bindings")
    indexed = [
        {
            (tuple(record["cell_zyx"]), record["candidate_id"]): record
            for record in stage.records
        }
        for stage in stages
    ]
    expected_refined = {}
    for key, record in indexed[0].items():
        transition = record["transition"]
        if transition["outcome"] == "continue":
            expected_refined[key] = (record["candidate_id"],)
        elif transition["outcome"] == "merged":
            successor = (key[0], transition["successor_id"])
            expected_refined[successor] = tuple(
                sorted((*expected_refined.get(successor, ()), record["candidate_id"]))
            )
    if set(indexed[1]) != set(expected_refined):
        raise ValueError("initialized-to-refined anchor lineage is inconsistent")
    for key, parents in expected_refined.items():
        if tuple(indexed[1][key]["parent_ids"]) != parents:
            raise ValueError("refined anchor parent lineage is inconsistent")
    for stage_index in range(1, 4):
        expected_keys = {
            key
            for key, record in indexed[stage_index].items()
            if record["transition"]["outcome"] == "continue"
        }
        if set(indexed[stage_index + 1]) != expected_keys:
            raise ValueError("anchor-stage survivor set is inconsistent")
        for key in expected_keys:
            before = indexed[stage_index][key]
            after = indexed[stage_index + 1][key]
            if (
                before["geometry"] != after["geometry"]
                or before["metrics"] != after["metrics"]
            ):
                raise ValueError(
                    "anchor geometry or metrics changed between filter stages"
                )
    for record in stages[3].records:
        suppressor = record["transition"]["suppressor"]
        if suppressor is None or suppressor["external_context"]:
            continue
        suppressor_key = (tuple(suppressor["cell_zyx"]), suppressor["candidate_id"])
        if suppressor_key not in indexed[3]:
            raise ValueError("internal NMS suppressor is absent from selection")
        metrics = indexed[3][suppressor_key]["metrics"]
        if (
            metrics["aligned_support"] != suppressor["aligned_support"]
            or metrics["directional_coherence"] != suppressor["directional_coherence"]
        ):
            raise ValueError("internal NMS suppressor metrics are inconsistent")
    if any(record["transition"]["outcome"] != "final" for record in stages[-1].records):
        raise ValueError("NMS stage contains non-final records")
    unmatched = list(final_anchors.paths_zyx)
    if len(unmatched) != len(stages[-1].paths_zyx):
        raise ValueError("NMS stage differs from authoritative final anchors")
    for diagnostic in stages[-1].paths_zyx:
        match = next(
            (
                index
                for index, final in enumerate(unmatched)
                if np.allclose(diagnostic, final, rtol=0.0, atol=1e-5)
            ),
            None,
        )
        if match is None:
            raise ValueError("NMS stage differs from authoritative final anchors")
        unmatched.pop(match)


def load_anchor_stage_directory(
    directory: str | Path,
    final_anchor_obj: str | Path,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
) -> tuple[AnchorStageGeometry, ...]:
    """Load and cross-check the complete stage set from an anchor output."""
    stage_directory = Path(directory)
    stages = tuple(
        read_anchor_stage_json(stage_directory / f"{stage}.json", stage)
        for stage in _ANCHOR_STAGE_NAMES
    )
    validate_anchor_stage_chain(
        stages, read_line_obj(final_anchor_obj, "anchors", crop_xyzwhd)
    )
    return stages


def _resolve_replay_artifacts(
    base: Path, descriptors: object, expected: set[str], label: str
) -> dict[str, Path]:
    if not isinstance(descriptors, dict) or set(descriptors) != expected:
        raise ValueError(f"{label} artifact set is invalid")
    resolved: dict[str, Path] = {}
    for key, descriptor in descriptors.items():
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "content_hash",
        }:
            raise ValueError(f"{label} artifact descriptor {key!r} is invalid")
        relative = Path(descriptor["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} artifact path {relative} escapes the bundle")
        try:
            artifact = (base / relative).resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"cannot resolve {label} artifact {relative}: {exc}"
            ) from exc
        if not artifact.is_relative_to(base):
            raise ValueError(f"{label} artifact path {relative} escapes the bundle")
        try:
            content = artifact.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read {label} artifact {artifact}: {exc}") from exc
        if _fnv1a64(content) != descriptor["content_hash"]:
            raise ValueError(f"{label} artifact hash mismatch: {relative}")
        resolved[key] = artifact
    return resolved


def _strict_segment_list(value: object, name: str) -> tuple[np.ndarray, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")  # noqa: TRY004
    return tuple(
        _strict_xyz_points(segment, f"{name}[{index}]", 1)
        for index, segment in enumerate(value)
    )


def _read_segmented_replay_obj(path: Path, header: str) -> tuple[np.ndarray, ...]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read replay OBJ {path}: {exc}") from exc
    if not lines or lines[0].strip() != header:
        raise ValueError(f"{path}: unsupported replay OBJ header")
    vertices: list[list[float]] = []
    records: list[list[int]] = []
    for line_number, raw in enumerate(lines[1:], start=2):
        fields = raw.strip().split()
        if not fields or fields[0] == "g":
            continue
        if fields[0] == "v" and len(fields) == 4:
            try:
                vertex = [float(item) for item in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid vertex") from exc
            if not np.isfinite(vertex).all():
                raise ValueError(f"{path}:{line_number}: non-finite vertex")
            vertices.append(vertex)
        elif fields[0] == "l" and len(fields) >= 3:
            try:
                records.append([int(item) for item in fields[1:]])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid index") from exc
        else:
            raise ValueError(f"{path}:{line_number}: unsupported OBJ record")
    result = []
    used: list[int] = []
    for record in records:
        if any(index < 1 or index > len(vertices) for index in record):
            raise ValueError(f"{path}: replay OBJ index is out of range")
        canonical = (
            record[:-1] if len(record) == 2 and record[0] == record[1] else record
        )
        used.extend(canonical)
        result.append(
            _strict_xyz_points(
                [vertices[index - 1] for index in canonical], str(path), 1
            )
        )
    if used != list(range(1, len(vertices) + 1)):
        raise ValueError(f"{path}: replay OBJ topology does not match its vertices")
    return tuple(result)


def _read_replay_strip_obj(
    artifacts: tuple[Path, Path, Path],
    header: str,
    source_segments_xyz: Sequence[np.ndarray],
    metadata: dict,
) -> SurfaceObjGeometry:
    """Read the standard VC3D textured-surface OBJ/TIFF strip artifact."""
    path, mtl_path, texture_path = artifacts
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read replay strip OBJ {path}: {exc}") from exc
    if not lines or lines[0].strip() != header:
        raise ValueError(f"{path}: unsupported replay strip OBJ header")

    stem = path.stem
    material = f"{stem}_texture"
    if path.parent / mtl_path.name != mtl_path:
        raise ValueError(f"{path}: replay strip MTL is not local to its OBJ")
    if mtl_path.parent / texture_path.name != texture_path:
        raise ValueError(f"{mtl_path}: replay strip texture is not local to its MTL")
    expected_mtl = [
        f"newmtl {material}",
        "Ka 1 1 1",
        "Kd 1 1 1",
        "Ks 0 0 0",
        "d 1",
        "illum 1",
        f"map_Kd {texture_path.name}",
    ]
    try:
        mtl_lines = [line.strip() for line in mtl_path.read_text().splitlines() if line.strip()]
    except OSError as exc:
        raise ValueError(f"cannot read replay strip MTL {mtl_path}: {exc}") from exc
    if mtl_lines != expected_mtl:
        raise ValueError(f"{mtl_path}: invalid replay strip material")
    try:
        from PIL import Image

        with Image.open(texture_path) as image:
            if image.format != "TIFF" or image.mode != "L":
                raise ValueError(
                    f"{texture_path}: replay strip texture must be grayscale TIFF"
                )
            if image.tag_v2.get(259, 1) != 1:
                raise ValueError(
                    f"{texture_path}: replay strip TIFF must be uncompressed"
                )
            texture = np.asarray(image, dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"cannot read replay strip texture {texture_path}: {exc}") from exc

    expected_cross = int(metadata["cross_samples"])
    render_scale = int(metadata["values"]["render_scale"])
    expected_sources = [
        (index, np.asarray(segment, dtype=np.float64))
        for index, segment in enumerate(source_segments_xyz)
        if len(segment) >= 2
    ]
    expected_vertices = sum(expected_cross * len(points) for _, points in expected_sources)
    expected_faces = sum(
        (expected_cross - 1) * (len(points) - 1)
        for _, points in expected_sources
    )

    if expected_sources:
        expected_prefix = [
            header,
            f"mtllib {mtl_path.name}",
            f"o {stem}",
            f"usemtl {material}",
        ]
    else:
        expected_prefix = [header, f"mtllib {mtl_path.name}", f"usemtl {material}"]
    if [line.strip() for line in lines[: len(expected_prefix)]] != expected_prefix:
        raise ValueError(f"{path}: invalid replay strip material binding")

    vertices: list[list[float]] = []
    texture_coordinates: list[list[float]] = []
    triangles: list[list[int]] = []
    faces: list[tuple[list[int], list[int], int]] = []
    section = "vertices"
    for line_number, raw in enumerate(
        lines[len(expected_prefix) :], start=len(expected_prefix) + 1
    ):
        fields = raw.strip().split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) == 4:
            if section != "vertices":
                raise ValueError(f"{path}:{line_number}: misplaced strip vertex")
            try:
                vertex = [float(item) for item in fields[1:4]]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid strip vertex") from exc
            if not np.isfinite(vertex).all():
                raise ValueError(f"{path}:{line_number}: non-finite strip vertex")
            vertices.append(vertex)
        elif fields[0] == "vt" and len(fields) == 3:
            if section == "faces":
                raise ValueError(f"{path}:{line_number}: misplaced strip texture coordinate")
            section = "texture_coordinates"
            try:
                coordinate = [float(item) for item in fields[1:]]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid strip texture coordinate"
                ) from exc
            if not np.isfinite(coordinate).all() or any(
                value < 0.0 or value > 1.0 for value in coordinate
            ):
                raise ValueError(
                    f"{path}:{line_number}: invalid strip texture coordinate"
                )
            texture_coordinates.append(coordinate)
        elif fields[0] == "f" and len(fields) == 5:
            if section == "vertices":
                raise ValueError(f"{path}:{line_number}: misplaced strip face")
            section = "faces"
            try:
                pairs = [item.split("/") for item in fields[1:]]
                if any(len(pair) != 2 for pair in pairs):
                    raise ValueError
                face = [int(pair[0]) for pair in pairs]
                texture_face = [int(pair[1]) for pair in pairs]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid strip face") from exc
            faces.append((face, texture_face, line_number))
        else:
            raise ValueError(f"{path}:{line_number}: unsupported strip OBJ record")
    if len(vertices) != expected_vertices or len(texture_coordinates) != expected_vertices:
        raise ValueError(f"{path}: strip component vertex/UV count is invalid")
    if len(faces) != expected_faces:
        raise ValueError(f"{path}: strip component face count is invalid")

    vertex_array = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
    texture_array = np.asarray(texture_coordinates, dtype=np.float64).reshape((-1, 2))
    expected_texture_height = expected_cross * render_scale + 2 if expected_sources else 1
    expected_texture_width = (
        sum(len(points) * render_scale + 2 for _, points in expected_sources)
        if expected_sources
        else 1
    )
    if texture.shape != (expected_texture_height, expected_texture_width):
        raise ValueError(f"{texture_path}: replay strip atlas dimensions are invalid")
    intensities: list[float] = []
    atlas_column = 0
    vertex_offset = 0
    face_offset = 0
    for _, source in expected_sources:
        longitudinal = len(source)
        component_vertices = expected_cross * longitudinal
        grid = vertex_array[
            vertex_offset : vertex_offset + component_vertices
        ].reshape((expected_cross, longitudinal, 3))
        center = grid[expected_cross // 2]
        if not np.allclose(center, source, rtol=1e-5, atol=1e-4):
            raise ValueError(f"{path}: strip centerline differs from trace geometry")
        uv_grid = texture_array[
            vertex_offset : vertex_offset + component_vertices
        ].reshape(
            (expected_cross, longitudinal, 2)
        )
        texture_columns = longitudinal * render_scale
        texture_rows = expected_cross * render_scale
        local_u = np.linspace(0.0, 1.0, longitudinal)
        local_v = np.linspace(1.0, 0.0, expected_cross)
        left = (atlas_column + 1.5) / expected_texture_width
        right = (atlas_column + 0.5 + texture_columns) / expected_texture_width
        bottom = 1.0 - (texture_rows + 0.5) / expected_texture_height
        top = 1.0 - 1.5 / expected_texture_height
        expected_u = left + local_u * (right - left)
        expected_v = bottom + local_v * (top - bottom)
        if not np.allclose(
            uv_grid[:, :, 0], expected_u[None, :], rtol=0.0, atol=1e-5
        ) or not np.allclose(
            uv_grid[:, :, 1], expected_v[:, None], rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"{path}: strip UVs do not address atlas texel centers")

        component_faces = (expected_cross - 1) * (longitudinal - 1)
        for local_face_index, (face, texture_face, line_number) in enumerate(
            faces[face_offset : face_offset + component_faces]
        ):
            row, column = divmod(local_face_index, longitudinal - 1)
            expected = [
                vertex_offset + row * longitudinal + column + 1,
                vertex_offset + row * longitudinal + column + 2,
                vertex_offset + (row + 1) * longitudinal + column + 2,
                vertex_offset + (row + 1) * longitudinal + column + 1,
            ]
            if face != expected or texture_face != expected:
                raise ValueError(
                    f"{path}:{line_number}: strip face crosses or scrambles components"
                )
            zero = [item - 1 for item in face]
            triangles.extend(
                ([zero[0], zero[1], zero[2]], [zero[0], zero[2], zero[3]])
            )

        tile = texture[
            1 : texture_rows + 1,
            atlas_column + 1 : atlas_column + texture_columns + 1,
        ]
        padded = texture[
            : texture_rows + 2,
            atlas_column : atlas_column + texture_columns + 2,
        ]
        if (
            not np.array_equal(padded[0, 1:-1], tile[0])
            or not np.array_equal(padded[-1, 1:-1], tile[-1])
            or not np.array_equal(padded[1:-1, 0], tile[:, 0])
            or not np.array_equal(padded[1:-1, -1], tile[:, -1])
            or padded[0, 0] != tile[0, 0]
            or padded[-1, 0] != tile[-1, 0]
            or padded[0, -1] != tile[0, -1]
            or padded[-1, -1] != tile[-1, -1]
        ):
            raise ValueError(f"{texture_path}: replay strip atlas padding is invalid")
        sample_rows = np.rint(
            np.linspace(0, texture_rows - 1, expected_cross)
        ).astype(np.int64)
        sample_columns = np.rint(
            np.linspace(0, texture_columns - 1, longitudinal)
        ).astype(np.int64)
        intensities.extend(
            (tile[np.ix_(sample_rows, sample_columns)].astype(np.float32) / 255.0)
            .reshape(-1)
            .tolist()
        )
        atlas_column += texture_columns + 2
        vertex_offset += component_vertices
        face_offset += component_faces

    return SurfaceObjGeometry(
        vertices_zyx=vertex_array[:, ::-1].copy(),
        triangles=np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        normalized_ct_intensity=np.asarray(intensities, dtype=np.float32),
        component_count=len(expected_sources),
    )


def _load_legacy_fiber_replay(
    bundle_path: Path, root: dict, include_anchor_stages: bool
) -> FiberReplayBundle:
    required = {
        "format",
        "version",
        "coordinates",
        "sources",
        "bindings",
        "trace_config",
        "status",
        "termination_reason",
        "reference_points_base_xyz",
        "trace_points_base_xyz",
        "comparison_trace_points_base_xyz",
        "comparison",
        "trace_cumulative_losses",
        "matching",
        "postroll",
        "failure_trace_point_index",
        "failure_reference_arc_base",
        "fiberlet_replay",
        "tube",
        "volume_crop_base_xyzwhd",
        "artifacts",
    }
    if set(root) != required:
        raise ValueError("legacy replay bundle fields do not match version 1")
    coordinates = {
        "position_order": "XYZ",
        "position_space": "base_volume",
        "distance_unit": "base_voxels",
    }
    if root["coordinates"] != coordinates:
        raise ValueError("legacy replay coordinate contract is unsupported")
    sources = root["sources"]
    source_fields = {
        "fiber_manifest",
        "fiber_manifest_content_hash",
        "normal_manifest",
        "normal_manifest_content_hash",
        "fiber_json",
        "fiber_json_content_hash",
    }
    binding = root.get("bindings", {}).get("prediction")
    shape = binding.get("prediction_shape_zyx") if isinstance(binding, dict) else None
    scale = (
        binding.get("prediction_to_base_scale") if isinstance(binding, dict) else None
    )
    if (
        not isinstance(sources, dict)
        or set(sources) != source_fields
        or any(not isinstance(value, str) or not value for value in sources.values())
        or not isinstance(binding, dict)
        or binding.get("mode") != "canonical_stored_grid"
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(item, int) or item <= 0 for item in shape)
        or isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("legacy replay source or prediction binding is invalid")
    reference_xyz = _strict_xyz_points(
        root["reference_points_base_xyz"], "reference_points_base_xyz", 1
    )
    trace_xyz = _strict_xyz_points(
        root["trace_points_base_xyz"], "trace_points_base_xyz", 1
    )
    comparison_trace_xyz = _strict_xyz_points(
        root["comparison_trace_points_base_xyz"],
        "comparison_trace_points_base_xyz",
        1,
    )
    status = root["status"]
    failed = status in {"failure_with_postroll", "failure_truncated"}
    if status not in {
        "failure_with_postroll",
        "failure_truncated",
        "no_failure",
        "trace_terminated_before_failure",
    }:
        raise ValueError("legacy replay status is invalid")
    crop_value = root["volume_crop_base_xyzwhd"]
    if failed:
        if not isinstance(crop_value, list) or len(crop_value) != 6:
            raise ValueError("legacy failure replay crop is invalid")
        try:
            crop = parse_crop(",".join(str(item) for item in crop_value))
        except argparse.ArgumentTypeError as exc:
            raise ValueError("legacy failure replay crop is invalid") from exc
    else:
        if crop_value is not None:
            raise ValueError("legacy nonfailure replay must not contain a crop")
        low = np.floor(reference_xyz.min(axis=0)).astype(int)
        high = np.ceil(reference_xyz.max(axis=0)).astype(int) + 1
        crop = (*low.tolist(), *(high - low).tolist())
    artifacts = root["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("legacy replay artifacts must be an object")  # noqa: TRY004
    resolved = _resolve_replay_artifacts(
        bundle_path.parent, artifacts, set(artifacts), "legacy replay"
    )
    reference_obj = _read_replay_obj(
        resolved["replay/reference.obj"],
        "# vc_fiber_replay_reference version 1",
        False,
    )
    trace_obj = _read_replay_obj(
        resolved["replay/trace.obj"],
        "# vc_fiber_replay_trace version 1",
        False,
    )
    if not np.array_equal(reference_obj, reference_xyz) or not np.array_equal(
        trace_obj, comparison_trace_xyz
    ):
        raise ValueError("legacy replay OBJ geometry differs from its JSON")
    fiberlet_xyz: tuple[np.ndarray, ...] = ()
    fiberlet = root["fiberlet_replay"]
    if isinstance(fiberlet, dict) and fiberlet.get("status") not in {None, "not_run"}:
        route = _strict_xyz_points(
            fiberlet.get("route_points_base_xyz"),
            "fiberlet_replay.route_points_base_xyz",
            1,
        )
        fiberlet_xyz = (route,)
        if "replay/fiberlet_trace.obj" in resolved:
            route_obj = _read_replay_obj(
                resolved["replay/fiberlet_trace.obj"],
                "# vc_fiberlet_graph_replay version 1",
                False,
            )
            if not np.array_equal(route_obj, route):
                raise ValueError("legacy fiberlet OBJ differs from its JSON")
    failure_zyx = None
    radius = None
    anchors_obj = None
    anchor_cells_obj = None
    paths_obj = None
    anchor_stages: tuple[AnchorStageGeometry, ...] = ()
    if failed:
        failure_index = root["failure_trace_point_index"]
        if not isinstance(failure_index, int) or not 0 <= failure_index < len(
            trace_xyz
        ):
            raise ValueError("legacy replay failure index is invalid")
        failure_obj = _read_replay_obj(
            resolved["replay/failure.obj"],
            "# vc_fiber_replay_failure version 1",
            True,
        )
        if not np.array_equal(failure_obj[0], trace_xyz[failure_index]):
            raise ValueError("legacy replay failure OBJ differs from its JSON")
        failure_zyx = failure_obj[:, ::-1].copy()
        tube = root["tube"]
        radius = tube.get("radius_base_voxels") if isinstance(tube, dict) else None
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("legacy replay tube radius is invalid")
        anchors_obj = resolved["anchors/anchors.obj"]
        anchor_cells_obj = resolved["anchors/anchor_cells.obj"]
        paths_obj = resolved["paths/fiberlets.obj"]
        if include_anchor_stages:
            anchor_stages = load_anchor_stage_directory(
                resolved["anchors/stages/initialized.json"].parent,
                anchors_obj,
                crop,
            )
    return FiberReplayBundle(
        path=bundle_path,
        tracer="greedy",
        tracer_failure_index=0,
        status=status,
        crop_xyzwhd=crop,
        prediction_shape_zyx=tuple(shape),
        prediction_to_base_scale=float(scale),
        fiber_manifest_content_hash=sources["fiber_manifest_content_hash"],
        reference_zyx=reference_xyz[:, ::-1].copy(),
        greedy_segments_zyx=(comparison_trace_xyz[:, ::-1].copy(),),
        fiberlet_segments_zyx=tuple(route[:, ::-1].copy() for route in fiberlet_xyz),
        failure_zyx=failure_zyx,
        tube_radius_base_voxels=float(radius) if radius is not None else None,
        anchors_obj=anchors_obj,
        anchor_cells_obj=anchor_cells_obj,
        anchor_stages=anchor_stages,
        paths_obj=paths_obj,
        strip_metadata=None,
        strip_artifacts=None,
    )


def load_fiber_replay_bundle(
    path: str | Path, *, include_anchor_stages: bool = True
) -> FiberReplayBundle:
    """Load one directly selected replay visualization manifest."""
    bundle_path = Path(path).expanduser().resolve()
    try:
        root = json.loads(bundle_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read replay bundle {bundle_path}: {exc}") from exc
    if not isinstance(root, dict):
        raise ValueError("replay manifest root must be an object")  # noqa: TRY004
    if root.get("format") == "vc_fiber_replay" and root.get("version") == 1:
        return _load_legacy_fiber_replay(bundle_path, root, include_anchor_stages)
    if root.get("format") == "vc_fiber_replay" and root.get("version") == 2:
        visualizations = root.get("visualizations")
        if not isinstance(visualizations, list) or not visualizations:
            raise ValueError(
                "aggregate replay contains no visualization manifests; "
                "rerun fiberlet-replay with --vis"
            )
        paths = [
            item.get("manifest", {}).get("path")
            for item in visualizations
            if isinstance(item, dict)
        ]
        usable = next((item for item in paths if isinstance(item, str)), None)
        suffix = f"; open {usable!r} directly" if usable is not None else ""
        raise ValueError(
            "aggregate replay is an index, not a visualization manifest" + suffix
        )
    local = root
    manifest_path = bundle_path
    coordinates = {
        "position_order": "XYZ",
        "position_space": "base_volume",
        "distance_unit": "base_voxels",
    }
    local_fields = {
        "format",
        "version",
        "identity",
        "coordinates",
        "sources",
        "prediction_binding",
        "failure",
        "tube",
        "volume_crop_base_xyzwhd",
        "reference_points_base_xyz",
        "greedy_trace_segments_base_xyz",
        "fiberlet_trace_segments_base_xyz",
        "artifacts",
    }
    has_strips = "trace_strips" in local
    expected_local_fields = local_fields | ({"trace_strips"} if has_strips else set())
    if (
        not isinstance(local, dict)
        or set(local) != expected_local_fields
        or local.get("format") != "vc_fiber_replay_visualization"
        or local.get("version") != 1
        or local.get("coordinates") != coordinates
    ):
        raise ValueError("replay visualization manifest is invalid")
    identity = local["identity"]
    if (
        not isinstance(identity, dict)
        or set(identity) != {"global_index", "tracer", "tracer_failure_index"}
        or not isinstance(identity["global_index"], int)
        or identity["global_index"] < 0
        or identity["tracer"] not in {"greedy", "fiberlet"}
        or not isinstance(identity["tracer_failure_index"], int)
        or identity["tracer_failure_index"] < 0
    ):
        raise ValueError("replay visualization identity is invalid")
    sources = local["sources"]
    source_fields = {
        "fiber_manifest",
        "fiber_manifest_content_hash",
        "normal_manifest",
        "normal_manifest_content_hash",
        "fiber_json",
        "fiber_json_content_hash",
    }
    if (
        not isinstance(sources, dict)
        or set(sources) != source_fields
        or any(not isinstance(value, str) or not value for value in sources.values())
    ):
        raise ValueError("replay visualization sources are invalid")
    binding = local["prediction_binding"]
    if not isinstance(binding, dict):
        raise ValueError(  # noqa: TRY004
            "replay visualization prediction binding is invalid"
        )
    shape = binding.get("prediction_shape_zyx")
    scale = binding.get("prediction_to_base_scale")
    if (
        binding.get("mode") != "canonical_stored_grid"
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(item, int) or item <= 0 for item in shape)
        or not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("replay visualization prediction grid is invalid")
    crop_value = local["volume_crop_base_xyzwhd"]
    if (
        not isinstance(crop_value, list)
        or len(crop_value) != 6
        or any(not isinstance(item, int) for item in crop_value)
    ):
        raise ValueError("replay visualization crop is invalid")
    try:
        crop = parse_crop(",".join(str(item) for item in crop_value))
    except argparse.ArgumentTypeError as exc:
        raise ValueError("replay visualization crop is invalid") from exc
    reference_xyz = _strict_xyz_points(
        local["reference_points_base_xyz"], "reference_points_base_xyz", 2
    )
    greedy_xyz = _strict_segment_list(
        local["greedy_trace_segments_base_xyz"], "greedy_trace_segments_base_xyz"
    )
    fiberlet_xyz = _strict_segment_list(
        local["fiberlet_trace_segments_base_xyz"], "fiberlet_trace_segments_base_xyz"
    )
    failure = local["failure"]
    failure_fields = {
        "index",
        "segment_index",
        "reason",
        "reference_arc_base",
        "reference_arc_fraction",
        "reference_point_base_xyz",
        "evaluator_point_base_xyz",
        "segment_point_index",
        "candidate_index",
        "arc_index",
        "candidate_path_point_index",
        "error_base_voxels",
        "error_ratio",
    }
    if (
        not isinstance(failure, dict)
        or set(failure) != failure_fields
        or failure.get("index") != identity["tracer_failure_index"]
        or not isinstance(failure.get("reason"), str)
        or not failure["reason"]
        or not _finite_number(failure.get("reference_arc_base"))
        or not _finite_number(failure.get("reference_arc_fraction"))
        or not 0.0 <= failure["reference_arc_fraction"] <= 1.0
    ):
        raise ValueError("replay visualization failure metadata is invalid")
    marker_xyz = _strict_xyz_points(
        [failure["evaluator_point_base_xyz"] or failure["reference_point_base_xyz"]],
        "failure marker",
        1,
    )
    tube = local["tube"]
    tube_fields = {
        "begin_arc_base",
        "end_arc_base",
        "radius_base_voxels",
        "reference_points_base_xyz",
        "cells_zyx",
    }
    if (
        not isinstance(tube, dict)
        or set(tube) != tube_fields
        or not isinstance(tube.get("radius_base_voxels"), (int, float))
        or isinstance(tube.get("radius_base_voxels"), bool)
        or not math.isfinite(tube["radius_base_voxels"])
        or tube["radius_base_voxels"] <= 0
        or not _finite_number(tube.get("begin_arc_base"))
        or not _finite_number(tube.get("end_arc_base"))
        or tube["begin_arc_base"] > failure["reference_arc_base"]
        or tube["end_arc_base"] < failure["reference_arc_base"]
        or tube["begin_arc_base"] >= tube["end_arc_base"]
        or not isinstance(tube.get("cells_zyx"), list)
        or any(
            not isinstance(cell, list)
            or len(cell) != 3
            or any(not isinstance(value, int) or value < 0 for value in cell)
            for cell in tube["cells_zyx"]
        )
        or not np.array_equal(
            _strict_xyz_points(tube["reference_points_base_xyz"], "tube reference", 2),
            reference_xyz,
        )
    ):
        raise ValueError("replay visualization tube is invalid")
    strip_metadata = None
    if has_strips:
        strip_metadata = local["trace_strips"]
        strip_fields = {"orientation", "geometry_builder", "cross_samples", "values"}
        values = (
            strip_metadata.get("values") if isinstance(strip_metadata, dict) else None
        )
        value_fields = {
            "semantic",
            "encoding",
            "renderer",
            "render_scale",
            "atlas_padding_pixels",
            "texture_format",
            "source_locator",
            "source_dtype",
            "source_shape_zyx",
            "source_group_scale_from_base_xyz",
            "source_group_offset_from_base_xyz",
            "source_storage_order",
            "vertex_position_order",
            "position_space",
            "scale_xyz",
            "translation_xyz",
        }
        if (
            not isinstance(strip_metadata, dict)
            or set(strip_metadata) != strip_fields
            or strip_metadata.get("orientation") != "sheet_aligned_normal_cross_tangent"
            or strip_metadata.get("geometry_builder")
            != "buildLineViewSurfaces_default"
            or strip_metadata.get("cross_samples") != 21
            or not isinstance(values, dict)
            or set(values) != value_fields
            or values.get("semantic") != "ct_intensity"
            or values.get("encoding") != "obj_uv_grayscale_tiff_u8"
            or values.get("renderer") != "vc_line_probe_fine_to_coarse"
            or not isinstance(values.get("render_scale"), int)
            or isinstance(values.get("render_scale"), bool)
            or values["render_scale"] < 1
            or values.get("atlas_padding_pixels") != 1
            or values.get("texture_format") != "tiff_gray_u8_uncompressed"
            or not isinstance(values.get("source_locator"), str)
            or not values["source_locator"]
            or values.get("source_dtype") != "uint8"
            or not isinstance(values.get("source_shape_zyx"), list)
            or len(values["source_shape_zyx"]) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in values["source_shape_zyx"]
            )
            or any(
                not isinstance(values.get(field), list)
                or len(values[field]) != 3
                or any(not _finite_number(value) for value in values[field])
                for field in (
                    "source_group_scale_from_base_xyz",
                    "source_group_offset_from_base_xyz",
                )
            )
            or any(
                value <= 0
                for value in values["source_group_scale_from_base_xyz"]
            )
            or values.get("source_storage_order") != "ZYX"
            or values.get("vertex_position_order") != "XYZ"
            or values.get("position_space") != "base_volume"
            or values.get("scale_xyz") != [1.0, 1.0, 1.0]
            or values.get("translation_xyz") != [0.0, 0.0, 0.0]
        ):
            raise ValueError("replay visualization strip metadata is invalid")
    expected_artifacts = {
        "replay/reference.obj",
        "replay/greedy.obj",
        "replay/fiberlet.obj",
        "replay/failure.obj",
        "anchors/anchors.json",
        "anchors/anchors.obj",
        "anchors/anchors_0.obj",
        "anchors/anchors_1.obj",
        "anchors/anchor_cells.obj",
        "anchors/stages/initialized.json",
        "anchors/stages/refined.json",
        "anchors/stages/support.json",
        "anchors/stages/selection.json",
        "anchors/stages/nms.json",
        "paths/fiberlets.json",
        "paths/fiberlets.obj",
        "paths/fiberlet_graph.json",
    }
    strip_artifacts = {
        "replay/reference_strip.obj",
        "replay/reference_strip.mtl",
        "replay/reference_strip.tif",
        "replay/greedy_strip.obj",
        "replay/greedy_strip.mtl",
        "replay/greedy_strip.tif",
        "replay/fiberlet_strip.obj",
        "replay/fiberlet_strip.mtl",
        "replay/fiberlet_strip.tif",
    }
    actual_artifacts = local["artifacts"]
    if not isinstance(actual_artifacts, dict):
        raise ValueError("replay visualization artifact set is invalid")  # noqa: TRY004
    present_strip_artifacts = set(actual_artifacts) & strip_artifacts
    if bool(strip_metadata) != bool(present_strip_artifacts) or (
        present_strip_artifacts and present_strip_artifacts != strip_artifacts
    ):
        raise ValueError("replay visualization strip artifact set is incomplete")
    if strip_metadata is not None:
        expected_artifacts |= strip_artifacts
    resolved = _resolve_replay_artifacts(
        manifest_path.parent,
        actual_artifacts,
        expected_artifacts,
        "replay visualization",
    )
    if not np.array_equal(
        _read_replay_obj(
            resolved["replay/reference.obj"],
            "# vc_fiber_replay_reference version 2",
            False,
        ),
        reference_xyz,
    ):
        raise ValueError("replay reference OBJ differs from visualization metadata")
    if len(greedy_xyz) != len(
        greedy_obj := _read_segmented_replay_obj(
            resolved["replay/greedy.obj"], "# vc_greedy_fiber_replay version 2"
        )
    ) or any(
        not np.array_equal(left, right)
        for left, right in zip(greedy_xyz, greedy_obj, strict=True)
    ):
        raise ValueError("greedy replay OBJ differs from visualization metadata")
    if len(fiberlet_xyz) != len(
        fiberlet_obj := _read_segmented_replay_obj(
            resolved["replay/fiberlet.obj"], "# vc_fiberlet_graph_replay version 2"
        )
    ) or any(
        not np.array_equal(left, right)
        for left, right in zip(fiberlet_xyz, fiberlet_obj, strict=True)
    ):
        raise ValueError("fiberlet replay OBJ differs from visualization metadata")
    failure_obj = _read_replay_obj(
        resolved["replay/failure.obj"],
        "# vc_fiber_replay_failure version 2",
        True,
    )
    if not np.array_equal(failure_obj, marker_xyz):
        raise ValueError("replay failure OBJ differs from visualization metadata")
    resolved_strip_artifacts = None
    if strip_metadata is not None:
        resolved_strip_artifacts = (
            (
                resolved["replay/reference_strip.obj"],
                resolved["replay/reference_strip.mtl"],
                resolved["replay/reference_strip.tif"],
            ),
            (
                resolved["replay/greedy_strip.obj"],
                resolved["replay/greedy_strip.mtl"],
                resolved["replay/greedy_strip.tif"],
            ),
            (
                resolved["replay/fiberlet_strip.obj"],
                resolved["replay/fiberlet_strip.mtl"],
                resolved["replay/fiberlet_strip.tif"],
            ),
        )
    anchor_stages: tuple[AnchorStageGeometry, ...] = ()
    if include_anchor_stages:
        anchor_stages = load_anchor_stage_directory(
            resolved["anchors/stages/initialized.json"].parent,
            resolved["anchors/anchors.obj"],
            crop,
        )
    return FiberReplayBundle(
        path=bundle_path,
        tracer=identity["tracer"],
        tracer_failure_index=identity["tracer_failure_index"],
        status=failure["reason"],
        crop_xyzwhd=crop,
        prediction_shape_zyx=tuple(shape),
        prediction_to_base_scale=float(scale),
        fiber_manifest_content_hash=sources["fiber_manifest_content_hash"],
        reference_zyx=reference_xyz[:, ::-1].copy(),
        greedy_segments_zyx=tuple(value[:, ::-1].copy() for value in greedy_xyz),
        fiberlet_segments_zyx=tuple(value[:, ::-1].copy() for value in fiberlet_xyz),
        failure_zyx=marker_xyz[:, ::-1].copy(),
        tube_radius_base_voxels=float(tube["radius_base_voxels"]),
        anchors_obj=resolved["anchors/anchors.obj"],
        anchor_cells_obj=resolved["anchors/anchor_cells.obj"],
        anchor_stages=anchor_stages,
        paths_obj=resolved["paths/fiberlets.obj"],
        strip_metadata=strip_metadata,
        strip_artifacts=resolved_strip_artifacts,
    )


def load_replay_visual_artifacts(replay: FiberReplayBundle) -> ReplayVisualArtifacts:
    """Load every failed-replay visualization artifact through strict readers."""
    if (
        replay.failure_zyx is None
        or replay.tube_radius_base_voxels is None
        or replay.anchors_obj is None
        or replay.anchor_cells_obj is None
        or replay.paths_obj is None
        or len(replay.anchor_stages) not in {0, len(_ANCHOR_STAGE_NAMES)}
    ):
        raise ValueError("artifact reload requires a failed replay bundle")
    strips = None
    if replay.strip_artifacts is not None:
        if replay.strip_metadata is None:
            raise ValueError("replay strip paths have no metadata")
        strips = ReplayStripGeometry(
            reference=_read_replay_strip_obj(
                replay.strip_artifacts[0],
                "# vc_fiber_replay_reference_strip version 4",
                (replay.reference_zyx[:, ::-1],),
                replay.strip_metadata,
            ),
            greedy=_read_replay_strip_obj(
                replay.strip_artifacts[1],
                "# vc_greedy_fiber_replay_strip version 4",
                tuple(value[:, ::-1] for value in replay.greedy_segments_zyx),
                replay.strip_metadata,
            ),
            fiberlet=_read_replay_strip_obj(
                replay.strip_artifacts[2],
                "# vc_fiberlet_graph_replay_strip version 4",
                tuple(value[:, ::-1] for value in replay.fiberlet_segments_zyx),
                replay.strip_metadata,
            ),
        )
    return ReplayVisualArtifacts(
        anchors=read_line_obj(replay.anchors_obj, "anchors", replay.crop_xyzwhd),
        anchor_cells=read_anchor_cell_obj(replay.anchor_cells_obj),
        anchor_stages=replay.anchor_stages,
        fiberlets=read_line_obj(replay.paths_obj, "paths", replay.crop_xyzwhd),
        strips=strips,
    )


def replay_strip_contrast_limits(
    strips: ReplayStripGeometry,
) -> tuple[float, float]:
    """Return one robust display range shared by all stored CT strip values."""
    values = (
        np.concatenate(
            [
                geometry.normalized_ct_intensity
                for geometry in (strips.reference, strips.greedy, strips.fiberlet)
                if len(geometry.normalized_ct_intensity)
            ]
        )
        if any(
            len(geometry.normalized_ct_intensity)
            for geometry in (strips.reference, strips.greedy, strips.fiberlet)
        )
        else np.empty(0, dtype=np.float32)
    )
    if not len(values):
        return (0.0, 1.0)
    lower, upper = (float(value) for value in np.percentile(values, (1.0, 99.0)))
    if not upper > lower:
        lower = float(np.min(values))
        upper = float(np.max(values))
    if not upper > lower:
        return (0.0, 1.0)
    return (lower, upper)


def replay_visual_topology(
    replay: FiberReplayBundle,
    artifacts: ReplayVisualArtifacts,
) -> tuple:
    """Describe the layer-presence topology that an in-place reload must retain."""
    return (
        replay.failure_zyx is not None,
        bool(replay.greedy_segments_zyx),
        bool(replay.fiberlet_segments_zyx),
        tuple(stage.stage for stage in artifacts.anchor_stages),
        artifacts.strips is not None,
    )


def validate_replay_reload_compatibility(
    current_replay: FiberReplayBundle,
    current_artifacts: ReplayVisualArtifacts,
    replacement_replay: FiberReplayBundle,
    replacement_artifacts: ReplayVisualArtifacts,
) -> None:
    """Reject a replay replacement that cannot reuse the current viewer/Zarr."""
    if current_replay.fiber_manifest_content_hash != (
        replacement_replay.fiber_manifest_content_hash
    ):
        raise ValueError("reloaded replay uses a different fiber prediction source")
    if current_replay.prediction_shape_zyx != replacement_replay.prediction_shape_zyx:
        raise ValueError("reloaded replay prediction shape differs")
    if current_replay.prediction_to_base_scale != (
        replacement_replay.prediction_to_base_scale
    ):
        raise ValueError("reloaded replay prediction scale differs")
    if current_replay.crop_xyzwhd != replacement_replay.crop_xyzwhd:
        raise ValueError("reloaded replay crop differs")
    if current_replay.tube_radius_base_voxels != (
        replacement_replay.tube_radius_base_voxels
    ):
        raise ValueError("reloaded replay extraction radius differs")
    if replay_visual_topology(current_replay, current_artifacts) != (
        replay_visual_topology(replacement_replay, replacement_artifacts)
    ):
        raise ValueError("reloaded replay visual layer topology differs")


def commit_with_rollback(
    commit: Callable[[], None],
    rollback: Callable[[], None],
) -> None:
    """Run one reload commit and guarantee an attempted rollback on failure."""
    try:
        commit()
    except Exception as update_error:
        try:
            rollback()
        except Exception as rollback_error:
            raise RuntimeError(
                "artifact reload failed and rollback also failed: "
                f"reload={update_error}; rollback={rollback_error}"
            ) from rollback_error
        raise RuntimeError(
            f"artifact reload commit failed and was rolled back: {update_error}"
        ) from update_error


def fiberlet_layer_features(fiberlets: LineObjGeometry) -> dict[str, np.ndarray]:
    """Return the aligned napari feature columns for fiberlet paths."""
    features = {
        "trace_loss_total": np.asarray(fiberlets.trace_loss_total, dtype=np.float64),
        "loss_per_prediction_voxel": np.asarray(
            fiberlets.loss_per_prediction_voxel, dtype=np.float64
        ),
        "relative_quality": np.asarray(fiberlets.relative_quality, dtype=np.float64),
    }
    if any(len(values) != len(fiberlets.paths_zyx) for values in features.values()):
        raise ValueError("fiberlet feature rows do not match path geometry")
    return features


def anchor_stage_layer_name(stage: AnchorStageGeometry) -> str:
    """Return a full-population diagnostic name independent of display filters."""
    rejected = sum(stage.reasons.values())
    return (
        f"anchor {stage.stage} "
        f"[{stage.geometric_record_count}/{stage.record_count}; rejected={rejected}]"
    )


def fiberlet_quality_colormap_spec() -> tuple[str, np.ndarray, np.ndarray]:
    """Return the default quality colormap without importing napari."""
    return (
        _FIBERLET_QUALITY_COLORMAP,
        np.asarray(
            [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
    )


def fiberlet_colormap_names(available: Sequence[str]) -> tuple[str, ...]:
    """Put the custom quality ramp first and sort unique napari names."""
    return (
        _FIBERLET_QUALITY_COLORMAP,
        *sorted(set(available) - {_FIBERLET_QUALITY_COLORMAP}),
    )


def parse_crop(value: str) -> tuple[int, int, int, int, int, int]:
    try:
        crop = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "crop must contain six comma-separated integers: X,Y,Z,W,H,D"
        ) from exc
    if len(crop) != 6:
        raise argparse.ArgumentTypeError(
            "crop must contain six comma-separated integers: X,Y,Z,W,H,D"
        )
    if any(item < 0 for item in crop[:3]):
        raise argparse.ArgumentTypeError("crop origin must be non-negative")
    if any(item <= 0 for item in crop[3:]):
        raise argparse.ArgumentTypeError("crop dimensions must be positive")
    return crop


def _path_intersects_crop(
    path_zyx: np.ndarray,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
) -> bool:
    x, y, z, width, height, depth = crop_xyzwhd
    low_zyx = np.asarray([z, y, x], dtype=np.float32)
    high_zyx = low_zyx + np.asarray([depth, height, width], dtype=np.float32)
    return bool(
        np.all(np.max(path_zyx, axis=0) >= low_zyx)
        and np.all(np.min(path_zyx, axis=0) < high_zyx)
    )


def read_line_obj(
    path: str | Path,
    kind: str,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
) -> LineObjGeometry:
    """Read one ordered base-XYZ line per group from a fiberlet OBJ."""
    if kind not in _LINE_OBJ_HEADERS:
        raise ValueError(f"unknown line OBJ kind: {kind!r}")

    obj_path = Path(path).expanduser()
    expected_header = _LINE_OBJ_HEADERS[kind]
    paths_have_quality = kind == "paths"
    paths_zyx: list[np.ndarray] = []
    trace_loss_total: list[float] = []
    loss_per_prediction_voxel: list[float] = []
    relative_quality: list[float] = []
    total_groups = 0
    vertex_count = 0
    group_name: str | None = None
    group_vertices: dict[int, tuple[float, float, float]] = {}
    group_lines: list[list[int]] = []
    group_metrics: dict[str, float] = {}
    header_seen = False
    report_metadata: dict[str, str] = {}
    group_records: list[tuple[np.ndarray, bool, str, float, float, float]] = []
    group_names: set[str] = set()

    def fail(line_number: int, message: str) -> ValueError:
        return ValueError(f"{obj_path}:{line_number}: {message}")

    def finish_group(line_number: int) -> None:
        nonlocal total_groups
        if group_name is None:
            return
        total_groups += 1
        if not group_lines:
            raise fail(line_number, f"group {group_name!r} has no line record")

        ordered_indices: list[int] = []
        for indices in group_lines:
            if not ordered_indices:
                ordered_indices.extend(indices)
            elif ordered_indices[-1] == indices[0]:
                ordered_indices.extend(indices[1:])
            else:
                raise fail(
                    line_number,
                    f"group {group_name!r} line records do not form one ordered path",
                )
        try:
            xyz = np.asarray(
                [group_vertices[index] for index in ordered_indices],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise fail(
                line_number,
                f"group {group_name!r} references vertex {exc.args[0]} outside the group",
            ) from exc
        if xyz.shape[0] < 2:
            raise fail(line_number, f"group {group_name!r} has fewer than two points")
        path_zyx = xyz[:, ::-1].copy()
        intersects = _path_intersects_crop(path_zyx, crop_xyzwhd)
        if paths_have_quality:
            required_metrics = {
                "trace_loss_total",
                "trace_loss_per_prediction_voxel",
                "trace_quality_relative",
            }
            if set(group_metrics) != required_metrics:
                raise fail(
                    line_number,
                    f"group {group_name!r} has incomplete trace-quality metadata",
                )
            total = group_metrics["trace_loss_total"]
            density = group_metrics["trace_loss_per_prediction_voxel"]
            quality = group_metrics["trace_quality_relative"]
            if (
                not math.isfinite(total)
                or total < 0.0
                or not math.isfinite(density)
                or density < 0.0
                or not math.isfinite(quality)
                or not 0.0 <= quality <= 1.0
            ):
                raise fail(
                    line_number,
                    f"group {group_name!r} trace-quality values are invalid",
                )
            group_records.append(
                (
                    path_zyx,
                    intersects,
                    group_name,
                    total,
                    density,
                    quality,
                )
            )
        elif intersects:
            paths_zyx.append(path_zyx)

    try:
        with obj_path.open() as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if line == expected_header:
                        if header_seen:
                            raise fail(line_number, "duplicate OBJ header")
                        header_seen = True
                        continue
                    if not paths_have_quality:
                        continue
                    fields = line[1:].strip().split()
                    if not fields:
                        continue
                    key = fields[0]
                    report_keys = {
                        "trace_quality_population",
                        "trace_loss_density_unit",
                        "trace_quality_formula",
                        "trace_quality_count",
                        "trace_loss_density_min",
                        "trace_loss_density_max",
                    }
                    metric_keys = {
                        "trace_loss_total",
                        "trace_loss_per_prediction_voxel",
                        "trace_quality_relative",
                    }
                    if key in report_keys:
                        if group_name is not None or len(fields) != 2:
                            raise fail(line_number, f"invalid report metadata {key!r}")
                        if key in report_metadata:
                            raise fail(
                                line_number, f"duplicate report metadata {key!r}"
                            )
                        report_metadata[key] = fields[1]
                    elif key in metric_keys:
                        if group_name is None or len(fields) != 2:
                            raise fail(line_number, f"invalid group metadata {key!r}")
                        if key in group_metrics:
                            raise fail(line_number, f"duplicate group metadata {key!r}")
                        try:
                            group_metrics[key] = float(fields[1])
                        except ValueError as exc:
                            raise fail(
                                line_number, f"group metadata {key!r} must be numeric"
                            ) from exc
                    else:
                        raise fail(line_number, f"unsupported path OBJ comment {key!r}")
                    continue

                fields = line.split()
                record = fields[0]
                if record == "g":
                    if len(fields) != 2:
                        raise fail(
                            line_number, "group record must contain exactly one name"
                        )
                    finish_group(line_number)
                    group_name = fields[1]
                    if group_name in group_names:
                        raise fail(line_number, f"duplicate group {group_name!r}")
                    group_names.add(group_name)
                    group_vertices = {}
                    group_lines = []
                    group_metrics = {}
                elif record == "v":
                    if group_name is None:
                        raise fail(line_number, "vertex appears before the first group")
                    if len(fields) != 4:
                        raise fail(line_number, "vertex record must contain X Y Z")
                    try:
                        xyz = tuple(float(value) for value in fields[1:])
                    except ValueError as exc:
                        raise fail(
                            line_number, "vertex coordinates must be numeric"
                        ) from exc
                    if not np.isfinite(xyz).all():
                        raise fail(line_number, "vertex coordinates must be finite")
                    vertex_count += 1
                    group_vertices[vertex_count] = xyz
                elif record == "l":
                    if group_name is None:
                        raise fail(line_number, "line appears before the first group")
                    if len(fields) < 3 or (paths_have_quality and len(fields) != 3):
                        raise fail(
                            line_number,
                            "path line record must reference exactly two vertices"
                            if paths_have_quality
                            else "line record must reference at least two vertices",
                        )
                    try:
                        indices = [int(value) for value in fields[1:]]
                    except ValueError as exc:
                        raise fail(
                            line_number, "line indices must be integers"
                        ) from exc
                    if any(index <= 0 for index in indices):
                        raise fail(line_number, "line indices must be positive")
                    group_lines.append(indices)
                else:
                    raise fail(line_number, f"unsupported OBJ record {record!r}")
            finish_group(line_number + 1 if "line_number" in locals() else 1)
    except OSError as exc:
        raise ValueError(f"cannot read line OBJ {obj_path}: {exc}") from exc

    if not header_seen:
        raise ValueError(f"{obj_path} is not a supported {kind} OBJ")
    if paths_have_quality:
        required_report = {
            "trace_quality_population",
            "trace_loss_density_unit",
            "trace_quality_formula",
            "trace_quality_count",
            "trace_loss_density_min",
            "trace_loss_density_max",
        }
        if set(report_metadata) != required_report:
            raise ValueError(f"{obj_path}: incomplete trace-quality report metadata")
        expected_values = {
            "trace_quality_population": "successful_scored_fiberlets",
            "trace_loss_density_unit": "prediction_voxel",
            "trace_quality_formula": "inverse_min_max_low_loss_is_one",
        }
        for key, expected in expected_values.items():
            if report_metadata[key] != expected:
                raise ValueError(f"{obj_path}: unsupported {key} value")
        try:
            expected_count = int(report_metadata["trace_quality_count"])
        except ValueError as exc:
            raise ValueError(
                f"{obj_path}: trace_quality_count must be an integer"
            ) from exc
        if expected_count < 0 or expected_count != total_groups:
            raise ValueError(f"{obj_path}: trace-quality count does not match groups")
        densities = [record[4] for record in group_records]
        minimum_text = report_metadata["trace_loss_density_min"]
        maximum_text = report_metadata["trace_loss_density_max"]
        if expected_count == 0:
            if minimum_text != "none" or maximum_text != "none":
                raise ValueError(f"{obj_path}: empty trace-quality bounds must be none")
            minimum = maximum = None
        else:
            try:
                minimum = float(minimum_text)
                maximum = float(maximum_text)
            except ValueError as exc:
                raise ValueError(
                    f"{obj_path}: trace-quality bounds must be numeric"
                ) from exc
            if (
                not math.isfinite(minimum)
                or not math.isfinite(maximum)
                or minimum < 0.0
                or minimum > maximum
                or min(densities) != minimum
                or max(densities) != maximum
            ):
                raise ValueError(
                    f"{obj_path}: trace-quality bounds do not match groups"
                )
        for path_zyx, intersects, name, total, density, quality in group_records:
            expected_quality = (
                1.0 if minimum == maximum else (maximum - density) / (maximum - minimum)
            )
            if not math.isclose(
                quality, expected_quality, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{obj_path}: group {name!r} quality is inconsistent")
            if intersects:
                paths_zyx.append(path_zyx)
                trace_loss_total.append(total)
                loss_per_prediction_voxel.append(density)
                relative_quality.append(quality)
    lengths = {
        len(paths_zyx),
        len(trace_loss_total) if paths_have_quality else len(paths_zyx),
        len(loss_per_prediction_voxel) if paths_have_quality else len(paths_zyx),
        len(relative_quality) if paths_have_quality else len(paths_zyx),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"{obj_path}: cropped fiberlet geometry and metrics are misaligned"
        )
    return LineObjGeometry(
        paths_zyx=paths_zyx,
        total_groups=total_groups,
        trace_loss_total=trace_loss_total,
        loss_per_prediction_voxel=loss_per_prediction_voxel,
        relative_quality=relative_quality,
    )


def clipping_plane_in_layer_data(
    layer,
    position_base_zyx: Sequence[float],
    normal_base_zyx: Sequence[float],
) -> dict:
    """Transform a base-coordinate clipping plane into layer data coordinates."""
    position_world = np.asarray(position_base_zyx, dtype=np.float64)
    normal_world = np.asarray(normal_base_zyx, dtype=np.float64)
    if position_world.shape != (3,) or normal_world.shape != (3,):
        raise ValueError("clipping-plane position and normal must be 3D")
    if not np.isfinite(position_world).all() or not np.isfinite(normal_world).all():
        raise ValueError("clipping-plane position and normal must be finite")
    if np.linalg.norm(normal_world) == 0:
        raise ValueError("clipping-plane normal must be nonzero")

    position_data = np.asarray(layer.world_to_data(position_world), dtype=np.float64)
    normal_tip_data = np.asarray(
        layer.world_to_data(position_world + normal_world), dtype=np.float64
    )
    normal_data = normal_tip_data - position_data
    normal_length = np.linalg.norm(normal_data)
    if not np.isfinite(normal_length) or normal_length == 0:
        raise ValueError("layer transform makes the clipping-plane normal invalid")
    normal_data /= normal_length
    return {
        "position": position_data,
        "normal": normal_data,
        "enabled": True,
    }


def crop_clipping_planes_in_layer_data(
    layer,
    lower_base_zyx: Sequence[float],
    upper_base_zyx: Sequence[float],
) -> list[dict]:
    """Build the six inward-facing planes of a base-coordinate crop box."""
    return [
        clipping_plane_in_layer_data(layer, plane["position"], plane["normal"])
        for plane in crop_clipping_planes_in_base(lower_base_zyx, upper_base_zyx)
    ]


def crop_clipping_planes_in_base(
    lower_base_zyx: Sequence[float],
    upper_base_zyx: Sequence[float],
) -> list[dict]:
    """Build six inward-facing crop planes in base/world coordinates."""
    lower = np.asarray(lower_base_zyx, dtype=np.float64)
    upper = np.asarray(upper_base_zyx, dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("crop bounds must be 3D")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("crop bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("crop lower bounds must not exceed upper bounds")

    planes: list[dict] = []
    for axis in range(3):
        lower_position = lower.copy()
        upper_position = upper.copy()
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        planes.append({"position": lower_position, "normal": normal, "enabled": True})
        planes.append({"position": upper_position, "normal": -normal, "enabled": True})
    return planes


def common_shape_edge_width(layer, default: float = 2.0) -> float:
    """Read the common width from napari's per-shape edge-width collection."""
    if layer is None:
        return default
    stored_width = getattr(layer, "_vc_display_edge_width", None)
    if stored_width is not None:
        width = float(stored_width)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("shape edge width must be positive and finite")
        return width
    widths = np.asarray(layer.edge_width, dtype=np.float64).reshape(-1)
    if widths.size == 0:
        return default
    width = float(widths[0])
    if not math.isfinite(width) or width <= 0:
        raise ValueError("shape edge width must be positive and finite")
    return width


def set_common_shape_edge_width(layer, width: float) -> None:
    """Set every shape width and notify napari's VisPy layer explicitly."""
    width = float(width)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("shape edge width must be positive and finite")
    layer._vc_display_edge_width = width
    layer.edge_width = width
    layer.events.edge_width()


def add_clipping_controls(
    viewer,
    volume_layer,
    anchors_layer,
    paths_layer,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
    path_quality_colormaps: dict[str, object] | None = None,
    reference_layer=None,
    trace_layer=None,
    failure_layer=None,
    anchor_cell_centers_layer=None,
    anchor_displacements_layer=None,
    anchor_stage_layers: Sequence | None = None,
    additional_layers: Sequence | None = None,
    presence_radius_base_voxels: float | None = None,
    maximum_presence_radius_base_voxels: float | None = None,
    set_presence_radius: Callable[[float], None] | None = None,
    anchor_radius_base_voxels: float | None = None,
    maximum_anchor_radius_base_voxels: float | None = None,
    set_anchor_radius: Callable[[float], None] | None = None,
    fiberlet_radius_base_voxels: float | None = None,
    maximum_fiberlet_radius_base_voxels: float | None = None,
    set_fiberlet_radius: Callable[[float], None] | None = None,
    reload_artifacts: Callable[[], str] | None = None,
) -> None:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QPushButton,
        QSlider,
        QSpinBox,
        QWidget,
    )

    x, y, z, width, height, depth = crop_xyzwhd
    axes = {
        "X": (2, x, x + width),
        "Y": (1, y, y + height),
        "Z": (0, z, z + depth),
    }

    widget = QWidget()
    form = QFormLayout(widget)
    bound_controls: dict[tuple[str, str], tuple[QSlider, QSpinBox]] = {}

    for axis_name, (_, minimum, maximum) in axes.items():
        for side, initial in (("min", minimum), ("max", maximum)):
            control = QWidget()
            layout = QHBoxLayout(control)
            layout.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            spin = QSpinBox()
            slider.setRange(minimum, maximum)
            spin.setRange(minimum, maximum)
            slider.setValue(initial)
            spin.setValue(initial)
            layout.addWidget(slider, stretch=1)
            layout.addWidget(spin)
            form.addRow(f"{axis_name} {side}", control)
            bound_controls[(axis_name, side)] = (slider, spin)

    def add_width_control(label: str, layer) -> tuple[QSlider, QDoubleSpinBox]:
        control = QWidget()
        layout = QHBoxLayout(control)
        layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1, 1000)
        slider.setTracking(False)
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 10.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        initial = common_shape_edge_width(layer)
        slider.setValue(round(initial * 100))
        spin.setValue(initial)
        slider.setEnabled(layer is not None)
        spin.setEnabled(layer is not None)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(spin)
        form.addRow(label, control)
        return slider, spin

    anchor_stage_layers = tuple(anchor_stage_layers or ())
    additional_layers = tuple(additional_layers or ())
    anchor_width_source = (
        anchors_layer
        if anchors_layer is not None
        else (anchor_stage_layers[0] if anchor_stage_layers else None)
    )
    anchors_width = add_width_control("Anchor width", anchor_width_source)
    paths_width = add_width_control("Path width", paths_layer)
    reference_width = add_width_control("Reference width", reference_layer)
    trace_width = add_width_control("Trace width", trace_layer)
    anchor_displacements_width = add_width_control(
        "Anchor offset width", anchor_displacements_layer
    )
    failure_size = QDoubleSpinBox()
    failure_size.setRange(0.1, 100.0)
    failure_size.setValue(
        float(np.asarray(failure_layer.size).reshape(-1)[0])
        if failure_layer is not None
        else 4.0
    )
    failure_size.setEnabled(failure_layer is not None)
    form.addRow("Failure size", failure_size)
    anchor_cell_size = QDoubleSpinBox()
    anchor_cell_size.setRange(0.1, 100.0)
    anchor_cell_size.setValue(
        float(np.asarray(anchor_cell_centers_layer.size).reshape(-1)[0])
        if anchor_cell_centers_layer is not None
        and np.asarray(anchor_cell_centers_layer.size).size
        else 2.0
    )
    anchor_cell_size.setEnabled(anchor_cell_centers_layer is not None)
    form.addRow("Cell center size", anchor_cell_size)
    quality_colormap_combo: QComboBox | None = None
    if paths_layer is not None and path_quality_colormaps:
        quality_colormap_combo = QComboBox()
        quality_colormap_combo.addItems(path_quality_colormaps)
        form.addRow("Path colormap", quality_colormap_combo)

    def add_radius_control(
        label: str,
        initial_radius: float,
        maximum_radius: float,
    ) -> tuple[QSlider, QDoubleSpinBox]:
        if (
            not math.isfinite(initial_radius)
            or initial_radius < 0
            or not math.isfinite(maximum_radius)
            or maximum_radius < 0
        ):
            raise ValueError(f"{label.lower()} values must be finite and non-negative")
        maximum_radius = max(
            float(initial_radius),
            float(maximum_radius),
        )
        control = QWidget()
        layout = QHBoxLayout(control)
        layout.setContentsMargins(0, 0, 0, 0)
        maximum_tenths = max(1, math.ceil(maximum_radius * 10.0))
        displayed_maximum = maximum_tenths / 10.0
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, maximum_tenths)
        slider.setTracking(False)
        spin = QDoubleSpinBox()
        spin.setRange(0.0, displayed_maximum)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        slider.setValue(round(initial_radius * 10.0))
        spin.setValue(initial_radius)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(spin)
        form.addRow(label, control)
        return slider, spin

    radius_controls: list[
        tuple[
            tuple[QSlider, QDoubleSpinBox],
            float,
            Callable[[float], None],
        ]
    ] = []
    if set_presence_radius is not None:
        if (
            presence_radius_base_voxels is None
            or maximum_presence_radius_base_voxels is None
        ):
            raise ValueError(
                "presence radius controls require initial and maximum values"
            )
        radius_controls.append(
            (
                add_radius_control(
                    "Presence radius",
                    presence_radius_base_voxels,
                    maximum_presence_radius_base_voxels,
                ),
                presence_radius_base_voxels,
                set_presence_radius,
            )
        )
    if set_anchor_radius is not None:
        if (
            anchor_radius_base_voxels is None
            or maximum_anchor_radius_base_voxels is None
        ):
            raise ValueError(
                "anchor radius controls require initial and maximum values"
            )
        radius_controls.append(
            (
                add_radius_control(
                    "Anchor radius",
                    anchor_radius_base_voxels,
                    maximum_anchor_radius_base_voxels,
                ),
                anchor_radius_base_voxels,
                set_anchor_radius,
            )
        )
    if set_fiberlet_radius is not None:
        if (
            fiberlet_radius_base_voxels is None
            or maximum_fiberlet_radius_base_voxels is None
        ):
            raise ValueError(
                "fiberlet radius controls require initial and maximum values"
            )
        radius_controls.append(
            (
                add_radius_control(
                    "Fiberlet radius",
                    fiberlet_radius_base_voxels,
                    maximum_fiberlet_radius_base_voxels,
                ),
                fiberlet_radius_base_voxels,
                set_fiberlet_radius,
            )
        )
    reload_button = None
    if reload_artifacts is not None:
        reload_button = QPushButton("Reload artifacts")
        form.addRow(reload_button)
    reset_button = QPushButton("Reset")
    form.addRow(reset_button)

    shape_layers = tuple(
        layer
        for layer in (
            anchors_layer,
            paths_layer,
            reference_layer,
            trace_layer,
            failure_layer,
            anchor_cell_centers_layer,
            anchor_displacements_layer,
            *anchor_stage_layers,
            *additional_layers,
        )
        if layer is not None
    )
    updating_bounds = False

    def current_bounds() -> tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(3, dtype=np.float64)
        upper = np.zeros(3, dtype=np.float64)
        for axis_name, (axis_index, _, _) in axes.items():
            lower[axis_index] = bound_controls[(axis_name, "min")][0].value()
            upper[axis_index] = bound_controls[(axis_name, "max")][0].value()
        return lower, upper

    def update_clipping(*_args) -> None:
        if updating_bounds:
            return
        lower, upper = current_bounds()
        # VisPy's volume clipper consumes scene coordinates after napari reverses
        # ZYX to XYZ; passing crop-local image coordinates clips translated crops.
        volume_layer.experimental_clipping_planes = crop_clipping_planes_in_base(
            lower, upper
        )
        for layer in shape_layers:
            layer.experimental_clipping_planes = crop_clipping_planes_in_layer_data(
                layer, lower, upper
            )

    def bound_changed(axis_name: str, side: str, value: int) -> None:
        nonlocal updating_bounds
        if updating_bounds:
            return
        updating_bounds = True
        slider, spin = bound_controls[(axis_name, side)]
        slider.setValue(value)
        spin.setValue(value)
        other_side = "max" if side == "min" else "min"
        other_slider, other_spin = bound_controls[(axis_name, other_side)]
        if (side == "min" and value > other_slider.value()) or (
            side == "max" and value < other_slider.value()
        ):
            other_slider.setValue(value)
            other_spin.setValue(value)
        updating_bounds = False
        update_clipping()

    for (axis_name, side), (slider, spin) in bound_controls.items():
        slider.valueChanged.connect(
            lambda value, axis_name=axis_name, side=side: bound_changed(
                axis_name, side, value
            )
        )
        spin.valueChanged.connect(
            lambda value, axis_name=axis_name, side=side: bound_changed(
                axis_name, side, value
            )
        )

    def connect_width_control(control, layer) -> None:
        slider, spin = control
        if layer is None:
            return
        updating_width = False

        def slider_changed(value: int) -> None:
            nonlocal updating_width
            if updating_width:
                return
            updating_width = True
            width = value / 100.0
            spin.setValue(width)
            updating_width = False
            set_common_shape_edge_width(layer, width)

        def spin_changed(value: float) -> None:
            nonlocal updating_width
            if updating_width:
                return
            updating_width = True
            slider.setValue(round(value * 100))
            updating_width = False
            set_common_shape_edge_width(layer, value)

        slider.valueChanged.connect(slider_changed)
        spin.valueChanged.connect(spin_changed)

    connect_width_control(anchors_width, anchors_layer)
    for anchor_stage_layer in anchor_stage_layers:
        connect_width_control(anchors_width, anchor_stage_layer)
    connect_width_control(paths_width, paths_layer)
    connect_width_control(reference_width, reference_layer)
    connect_width_control(trace_width, trace_layer)
    connect_width_control(anchor_displacements_width, anchor_displacements_layer)
    if failure_layer is not None:
        failure_size.valueChanged.connect(
            lambda value: setattr(failure_layer, "size", float(value))
        )
    if anchor_cell_centers_layer is not None:
        anchor_cell_size.valueChanged.connect(
            lambda value: setattr(anchor_cell_centers_layer, "size", float(value))
        )

    if quality_colormap_combo is not None and path_quality_colormaps is not None:

        def quality_colormap_changed(name: str) -> None:
            paths_layer.edge_colormap = path_quality_colormaps[name]
            paths_layer.refresh_colors()

        quality_colormap_combo.currentTextChanged.connect(quality_colormap_changed)

    radius_resets: list[Callable[[], None]] = []

    def connect_radius_control(
        control: tuple[QSlider, QDoubleSpinBox],
        initial_radius: float,
        setter: Callable[[float], None],
    ) -> None:
        radius_slider, radius_spin = control
        updating_radius = False

        def set_control_radius(radius: float) -> None:
            nonlocal updating_radius
            updating_radius = True
            radius_slider.setValue(round(radius * 10.0))
            radius_spin.setValue(radius)
            updating_radius = False
            setter(radius)

        def radius_slider_changed(value: int) -> None:
            nonlocal updating_radius
            if updating_radius:
                return
            updating_radius = True
            radius = value / 10.0
            radius_spin.setValue(radius)
            updating_radius = False
            setter(radius)

        def radius_spin_changed(value: float) -> None:
            nonlocal updating_radius
            if updating_radius:
                return
            updating_radius = True
            radius_slider.setValue(round(value * 10.0))
            updating_radius = False
            setter(float(value))

        radius_slider.valueChanged.connect(radius_slider_changed)
        radius_spin.valueChanged.connect(radius_spin_changed)
        radius_resets.append(lambda: set_control_radius(initial_radius))

    for radius_control, initial_radius, setter in radius_controls:
        connect_radius_control(radius_control, initial_radius, setter)

    def reset_bounds(*_args) -> None:
        nonlocal updating_bounds
        updating_bounds = True
        for axis_name, (_, minimum, maximum) in axes.items():
            for side, value in (("min", minimum), ("max", maximum)):
                slider, spin = bound_controls[(axis_name, side)]
                slider.setValue(value)
                spin.setValue(value)
        updating_bounds = False
        update_clipping()
        if quality_colormap_combo is not None:
            quality_colormap_combo.setCurrentIndex(0)
        for reset_radius in radius_resets:
            reset_radius()

    reset_button.clicked.connect(reset_bounds)
    if reload_button is not None and reload_artifacts is not None:

        def reload_clicked(*_args) -> None:
            reload_button.setEnabled(False)
            try:
                message = reload_artifacts()
                viewer.status = message
                print(message)
            except Exception as exc:  # noqa: BLE001 - Qt callback must stay alive
                message = f"Replay artifact reload failed: {exc}"
                viewer.status = message
                print(message, file=sys.stderr)
            finally:
                reload_button.setEnabled(True)

        reload_button.clicked.connect(reload_clicked)
    update_clipping()

    viewer.window.add_dock_widget(widget, area="right", name="Clip")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing OME-Zarr metadata: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read OME-Zarr metadata {path}: {exc}") from exc


def _find_multiscale_root(path: Path) -> tuple[Path, dict]:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        attrs_path = candidate / ".zattrs"
        if not attrs_path.is_file():
            continue
        attrs = _read_json(attrs_path)
        if isinstance(attrs.get("multiscales"), list):
            return candidate, attrs
    raise ValueError(f"{path} is not an OME-Zarr pyramid root or an array inside one")


def _compose_transforms(
    transforms: Sequence[dict], ndim: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    scale = np.ones(ndim, dtype=np.float64)
    translation = np.zeros(ndim, dtype=np.float64)
    for transform in transforms:
        transform_type = transform.get("type")
        if transform_type == "scale":
            values = np.asarray(transform.get("scale"), dtype=np.float64)
            if (
                values.shape != (ndim,)
                or not np.isfinite(values).all()
                or np.any(values <= 0)
            ):
                raise ValueError(
                    "OME-Zarr scale must contain one positive value per axis"
                )
            scale *= values
            translation *= values
        elif transform_type == "translation":
            values = np.asarray(transform.get("translation"), dtype=np.float64)
            if values.shape != (ndim,) or not np.isfinite(values).all():
                raise ValueError(
                    "OME-Zarr translation must contain one finite value per axis"
                )
            translation += values
        else:
            raise ValueError(
                f"unsupported OME-Zarr coordinate transform: {transform_type!r}"
            )
    return tuple(float(item) for item in scale), tuple(
        float(item) for item in translation
    )


def resolve_ome_zarr_level(
    zarr_path: str | Path, level: str | None = None
) -> OmeZarrLevel:
    requested = Path(zarr_path).expanduser().resolve()
    root, attrs = _find_multiscale_root(requested)
    multiscales = attrs["multiscales"]
    if len(multiscales) != 1 or not isinstance(multiscales[0], dict):
        raise ValueError(
            "fiber presence OME-Zarr must contain exactly one multiscale image"
        )

    multiscale = multiscales[0]
    axes = multiscale.get("axes")
    axis_names = tuple(
        axis.get("name") if isinstance(axis, dict) else axis for axis in axes or ()
    )
    if axis_names != ("z", "y", "x"):
        raise ValueError(f"fiber presence axes must be Z,Y,X, got {axis_names}")

    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("OME-Zarr multiscale metadata has no datasets")

    if level is not None:
        selected_path = level.strip("/")
    elif requested == root:
        selected_path = str(datasets[0].get("path", "")).strip("/")
    else:
        try:
            selected_path = requested.relative_to(root).as_posix().strip("/")
        except ValueError as exc:
            raise ValueError(f"{requested} is not inside OME-Zarr root {root}") from exc

    matches = [
        item
        for item in datasets
        if str(item.get("path", "")).strip("/") == selected_path
    ]
    if len(matches) != 1:
        available = ", ".join(str(item.get("path")) for item in datasets)
        raise ValueError(
            f"OME-Zarr level {selected_path!r} not found; available levels: {available}"
        )
    if not (root / selected_path / ".zarray").is_file():
        raise ValueError(
            f"OME-Zarr level is not a local Zarr v2 array: {root / selected_path}"
        )

    root_transforms = multiscale.get("coordinateTransformations", [])
    dataset_transforms = matches[0].get("coordinateTransformations", [])
    scale, translation = _compose_transforms(
        [*dataset_transforms, *root_transforms], ndim=3
    )
    return OmeZarrLevel(
        root=root,
        path=selected_path,
        scale_zyx=scale,
        translation_zyx=translation,
    )


def _ceil_lattice_coordinate(value: float) -> int:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest)
    return math.ceil(value)


def select_base_crop(
    shape_zyx: Sequence[int],
    level: OmeZarrLevel,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
) -> CropSelection:
    if len(shape_zyx) != 3:
        raise ValueError(
            f"fiber presence array must be 3D, got shape {tuple(shape_zyx)}"
        )

    x, y, z, width, height, depth = crop_xyzwhd
    low_base = (z, y, x)
    high_base = (z + depth, y + height, x + width)
    lows: list[int] = []
    highs: list[int] = []
    for axis in range(3):
        scale = level.scale_zyx[axis]
        translation = level.translation_zyx[axis]
        low = _ceil_lattice_coordinate((low_base[axis] - translation) / scale)
        high = _ceil_lattice_coordinate((high_base[axis] - translation) / scale)
        lows.append(max(0, min(int(shape_zyx[axis]), low)))
        highs.append(max(0, min(int(shape_zyx[axis]), high)))

    if any(low >= high for low, high in zip(lows, highs, strict=True)):
        raise ValueError(
            "crop does not contain any samples from the selected OME-Zarr level"
        )

    slices = tuple(slice(low, high) for low, high in zip(lows, highs, strict=True))
    origin = tuple(
        level.translation_zyx[axis] + lows[axis] * level.scale_zyx[axis]
        for axis in range(3)
    )
    return CropSelection(
        requested_base_xyzwhd=crop_xyzwhd,
        slices_zyx=slices,
        origin_base_zyx=origin,
        shape_zyx=tuple(high - low for low, high in zip(lows, highs, strict=True)),
    )


def open_lazy_crop(
    level: OmeZarrLevel, crop_xyzwhd: tuple[int, int, int, int, int, int]
):
    try:
        import dask.array as da
        import zarr
    except ImportError as exc:
        raise RuntimeError("fiber presence viewing requires dask and zarr") from exc

    array = zarr.open_array(str(level.array_path), mode="r")
    selection = select_base_crop(array.shape, level, crop_xyzwhd)
    lazy_array = da.from_zarr(array)[selection.slices_zyx]
    return lazy_array, selection


def replay_distance_transform_base(
    reference_base_zyx: np.ndarray,
    trace_segments_base_zyx: np.ndarray | Sequence[np.ndarray],
    selection: CropSelection,
    scale_zyx: Sequence[float],
) -> np.ndarray:
    """Return base-voxel distance to the reference-or-replay line union."""
    from scipy.ndimage import distance_transform_edt

    scale = np.asarray(scale_zyx, dtype=np.float64)
    origin = np.asarray(selection.origin_base_zyx, dtype=np.float64)
    shape = np.asarray(selection.shape_zyx, dtype=np.int64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("replay distance-transform scale is invalid")
    if np.any(shape <= 0):
        raise ValueError("replay distance-transform crop is empty")

    centerline = np.zeros(selection.shape_zyx, dtype=bool)
    trace_segments = (
        (trace_segments_base_zyx,)
        if isinstance(trace_segments_base_zyx, np.ndarray)
        else tuple(trace_segments_base_zyx)
    )
    polylines = (("reference", reference_base_zyx),) + tuple(
        (f"trace segment {index}", value) for index, value in enumerate(trace_segments)
    )
    for name, polyline_value in polylines:
        polyline = np.asarray(polyline_value, dtype=np.float64)
        if polyline.ndim != 2 or polyline.shape[1:] != (3,) or len(polyline) < 1:
            raise ValueError(f"{name} polyline must contain ZYX points")
        if not np.isfinite(polyline).all():
            raise ValueError(f"{name} polyline contains non-finite coordinates")
        polyline_data = (polyline - origin) / scale
        rasterized: list[np.ndarray] = []
        if len(polyline_data) == 1:
            rasterized.append(polyline_data)
        else:
            for start, end in pairwise(polyline_data):
                steps = max(1, math.ceil(np.max(np.abs(end - start))))
                fractions = np.linspace(0.0, 1.0, steps + 1)[:, np.newaxis]
                rasterized.append(start + fractions * (end - start))
        for samples in rasterized:
            indices = np.rint(samples).astype(np.int64)
            valid = np.all((indices >= 0) & (indices < shape), axis=1)
            indices = indices[valid]
            if len(indices):
                centerline[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    if not centerline.any():
        raise ValueError("replay lines do not intersect the selected crop")
    return distance_transform_edt(~centerline, sampling=scale).astype(
        np.float32, copy=False
    )


def mask_presence_by_distance(data, distance_data, radius_base_voxels: float):
    """Apply a hard base-distance tube mask without materializing presence."""
    if not math.isfinite(radius_base_voxels) or radius_base_voxels < 0:
        raise ValueError("presence radius must be finite and non-negative")
    if data.shape != distance_data.shape:
        raise ValueError("presence and reference-distance shapes differ")
    return data.map_blocks(
        lambda presence, distance: np.where(
            distance <= radius_base_voxels,
            presence,
            np.zeros((), dtype=presence.dtype),
        ),
        distance_data,
        dtype=data.dtype,
    )


def polyline_union_distances_base(
    points_base_zyx: np.ndarray,
    polylines_base_zyx: Sequence[np.ndarray],
) -> np.ndarray:
    """Return exact point-to-polyline-union distances in base voxels."""
    points = np.asarray(points_base_zyx, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("anchor representatives must be an Nx3 ZYX array")
    if not np.isfinite(points).all():
        raise ValueError("anchor representatives contain non-finite coordinates")
    if not polylines_base_zyx:
        raise ValueError("anchor distance requires at least one polyline")

    segment_starts = []
    segment_ends = []
    for index, polyline_value in enumerate(polylines_base_zyx):
        polyline = np.asarray(polyline_value, dtype=np.float64)
        if polyline.ndim != 2 or polyline.shape[1:] != (3,) or len(polyline) < 1:
            raise ValueError(f"polyline {index} must contain ZYX points")
        if not np.isfinite(polyline).all():
            raise ValueError(f"polyline {index} contains non-finite coordinates")
        if len(polyline) == 1:
            segment_starts.append(polyline)
            segment_ends.append(polyline)
        else:
            segment_starts.append(polyline[:-1])
            segment_ends.append(polyline[1:])

    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    starts = np.concatenate(segment_starts, axis=0)
    ends = np.concatenate(segment_ends, axis=0)
    directions = ends - starts
    squared_lengths = np.einsum("ij,ij->i", directions, directions)
    distances_squared = np.full(len(points), np.inf, dtype=np.float64)

    # Bound pairwise scratch while retaining vectorized point-to-segment math.
    point_batch = 512
    segment_batch = 512
    for point_start in range(0, len(points), point_batch):
        point_stop = min(len(points), point_start + point_batch)
        point_block = points[point_start:point_stop]
        block_minimum = np.full(len(point_block), np.inf, dtype=np.float64)
        for segment_start in range(0, len(starts), segment_batch):
            segment_stop = min(len(starts), segment_start + segment_batch)
            starts_block = starts[segment_start:segment_stop]
            directions_block = directions[segment_start:segment_stop]
            lengths_block = squared_lengths[segment_start:segment_stop]
            relative = point_block[:, np.newaxis, :] - starts_block[np.newaxis, :, :]
            numerators = np.einsum("ijk,jk->ij", relative, directions_block)
            fractions = np.divide(
                numerators,
                lengths_block,
                out=np.zeros_like(numerators),
                where=lengths_block > 0,
            )
            np.clip(fractions, 0.0, 1.0, out=fractions)
            offsets = relative - fractions[:, :, np.newaxis] * directions_block
            candidate_squared = np.einsum("ijk,ijk->ij", offsets, offsets)
            block_minimum = np.minimum(
                block_minimum,
                np.min(candidate_squared, axis=1),
            )
        distances_squared[point_start:point_stop] = block_minimum
    return np.sqrt(distances_squared)


def anchor_path_representatives(
    paths_zyx: Sequence[np.ndarray], *, target_endpoint: bool = False
) -> np.ndarray:
    """Return the semantic base-coordinate representative of each anchor glyph."""
    representatives = []
    for index, path_value in enumerate(paths_zyx):
        path = np.asarray(path_value, dtype=np.float64)
        if path.ndim != 2 or path.shape[1:] != (3,) or len(path) < 1:
            raise ValueError(f"anchor geometry {index} must contain ZYX points")
        if not np.isfinite(path).all():
            raise ValueError(f"anchor geometry {index} contains non-finite coordinates")
        representatives.append(
            path[-1] if target_endpoint else (path[0] + path[-1]) * 0.5
        )
    if not representatives:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(representatives, dtype=np.float64)


def replay_anchor_representatives(
    artifacts: ReplayVisualArtifacts,
) -> dict[str, np.ndarray]:
    """Return stable layer-keyed anchor representatives for replay filtering."""
    representatives = {}
    if artifacts.anchors.paths_zyx:
        representatives["anchors"] = anchor_path_representatives(
            artifacts.anchors.paths_zyx
        )
    for stage in artifacts.anchor_stages:
        if stage.paths_zyx:
            representatives[f"stage:{stage.stage}"] = anchor_path_representatives(
                stage.paths_zyx
            )
    if len(artifacts.anchor_cells.centers_zyx):
        representatives["cell_centers"] = np.asarray(
            artifacts.anchor_cells.centers_zyx, dtype=np.float64
        )
    if artifacts.anchor_cells.displacements_zyx:
        representatives["displacements"] = anchor_path_representatives(
            artifacts.anchor_cells.displacements_zyx,
            target_endpoint=True,
        )
    return representatives


def replay_anchor_distances_base(
    replay: FiberReplayBundle,
    artifacts: ReplayVisualArtifacts,
) -> dict[str, np.ndarray]:
    """Compute exact union distances while deduplicating shared representatives."""
    representatives = replay_anchor_representatives(artifacts)
    if not representatives:
        return {}
    keys = tuple(representatives)
    counts = [len(representatives[key]) for key in keys]
    all_representatives = np.concatenate([representatives[key] for key in keys], axis=0)
    unique_representatives, inverse = np.unique(
        all_representatives,
        axis=0,
        return_inverse=True,
    )
    unique_distances = polyline_union_distances_base(
        unique_representatives,
        (
            replay.reference_zyx,
            *replay.greedy_segments_zyx,
            *replay.fiberlet_segments_zyx,
        ),
    )
    all_distances = unique_distances[inverse]
    result = {}
    offset = 0
    for key, count in zip(keys, counts, strict=True):
        result[key] = all_distances[offset : offset + count]
        offset += count
    return result


def replay_fiberlet_distances_base(
    replay: FiberReplayBundle,
    fiberlets: LineObjGeometry,
) -> np.ndarray:
    """Return each rendered fiberlet polyline's exact replay-line distance."""
    if not fiberlets.paths_zyx:
        return np.empty(0, dtype=np.float64)
    target_starts = []
    target_ends = []
    replay_lines = (
        replay.reference_zyx,
        *replay.greedy_segments_zyx,
        *replay.fiberlet_segments_zyx,
    )
    for index, value in enumerate(replay_lines):
        starts, ends = _polyline_segments_base(value, f"replay polyline {index}")
        target_starts.append(starts)
        target_ends.append(ends)
    targets_start = np.concatenate(target_starts, axis=0)
    targets_end = np.concatenate(target_ends, axis=0)

    result = np.full(len(fiberlets.paths_zyx), np.inf, dtype=np.float64)
    segment_batch = 256
    for path_index, path in enumerate(fiberlets.paths_zyx):
        starts, ends = _polyline_segments_base(path, f"fiberlet geometry {path_index}")
        minimum_squared = np.inf
        for source_offset in range(0, len(starts), segment_batch):
            source_slice = slice(source_offset, source_offset + segment_batch)
            for target_offset in range(0, len(targets_start), segment_batch):
                target_slice = slice(target_offset, target_offset + segment_batch)
                pairwise = _segment_pair_distances_squared(
                    starts[source_slice],
                    ends[source_slice],
                    targets_start[target_slice],
                    targets_end[target_slice],
                )
                minimum_squared = min(minimum_squared, float(np.min(pairwise)))
        result[path_index] = math.sqrt(max(0.0, minimum_squared))
    return result


def _polyline_segments_base(
    value: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    polyline = np.asarray(value, dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[1:] != (3,) or len(polyline) < 1:
        raise ValueError(f"{name} must contain ZYX points")
    if not np.isfinite(polyline).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    if len(polyline) == 1:
        return polyline, polyline
    return polyline[:-1], polyline[1:]


def _segment_pair_distances_squared(
    first_starts: np.ndarray,
    first_ends: np.ndarray,
    second_starts: np.ndarray,
    second_ends: np.ndarray,
) -> np.ndarray:
    """Return exact pairwise 3D segment distances with bounded caller batches."""
    first_directions = first_ends - first_starts
    second_directions = second_ends - second_starts
    relative = first_starts[:, np.newaxis, :] - second_starts[np.newaxis, :, :]
    first_length = np.einsum("ij,ij->i", first_directions, first_directions)[
        :, np.newaxis
    ]
    second_length = np.einsum("ij,ij->i", second_directions, second_directions)[
        np.newaxis, :
    ]
    cross = first_directions @ second_directions.T
    first_relative = np.einsum("ijk,ik->ij", relative, first_directions)
    second_relative = np.einsum("ijk,jk->ij", relative, second_directions)
    epsilon = np.finfo(np.float64).eps

    def squared_at(first_fraction, second_fraction):
        offset = (
            relative
            + first_fraction[:, :, np.newaxis] * first_directions[:, np.newaxis, :]
            - second_fraction[:, :, np.newaxis] * second_directions[np.newaxis, :, :]
        )
        return np.einsum("ijk,ijk->ij", offset, offset)

    zeros = np.zeros_like(cross)
    minimum = np.full_like(cross, np.inf)

    denominator = first_length * second_length - cross * cross
    nonparallel = denominator > epsilon * first_length * second_length
    first_fraction = np.divide(
        cross * second_relative - second_length * first_relative,
        denominator,
        out=zeros.copy(),
        where=nonparallel,
    )
    second_fraction = np.divide(
        first_length * second_relative - cross * first_relative,
        denominator,
        out=zeros.copy(),
        where=nonparallel,
    )
    interior = (
        nonparallel
        & (first_fraction >= 0.0)
        & (first_fraction <= 1.0)
        & (second_fraction >= 0.0)
        & (second_fraction <= 1.0)
    )
    minimum = np.minimum(
        minimum,
        np.where(interior, squared_at(first_fraction, second_fraction), np.inf),
    )

    second_on_first_start = np.clip(
        np.divide(
            second_relative,
            second_length,
            out=zeros.copy(),
            where=second_length > 0.0,
        ),
        0.0,
        1.0,
    )
    minimum = np.minimum(minimum, squared_at(zeros, second_on_first_start))
    second_on_first_end = np.clip(
        np.divide(
            second_relative + cross,
            second_length,
            out=zeros.copy(),
            where=second_length > 0.0,
        ),
        0.0,
        1.0,
    )
    minimum = np.minimum(minimum, squared_at(np.ones_like(cross), second_on_first_end))
    first_on_second_start = np.clip(
        np.divide(
            -first_relative,
            first_length,
            out=zeros.copy(),
            where=first_length > 0.0,
        ),
        0.0,
        1.0,
    )
    minimum = np.minimum(minimum, squared_at(first_on_second_start, zeros))
    first_on_second_end = np.clip(
        np.divide(
            cross - first_relative,
            first_length,
            out=zeros.copy(),
            where=first_length > 0.0,
        ),
        0.0,
        1.0,
    )
    return np.minimum(minimum, squared_at(first_on_second_end, np.ones_like(cross)))


def distance_visibility_mask(
    distances_base_voxels: np.ndarray,
    radius_base_voxels: float,
) -> np.ndarray:
    """Return the inclusive visibility mask for a base-voxel radius."""
    distances = np.asarray(distances_base_voxels, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("replay geometry distances must be one-dimensional")
    if np.isnan(distances).any() or np.any(distances < 0):
        raise ValueError("replay geometry distances must be non-negative and not NaN")
    if not math.isfinite(radius_base_voxels) or radius_base_voxels < 0:
        raise ValueError("replay geometry radius must be finite and non-negative")
    return distances <= radius_base_voxels


def make_replay_geometry_filter(
    *,
    key: str,
    layer,
    source_data: np.ndarray | Sequence[np.ndarray],
    distances_base_voxels: np.ndarray,
    color_attribute: str,
    color_value: str,
    empty_color_value: str | None = None,
    source_features: dict[str, Sequence] | None = None,
) -> ReplayGeometryFilter:
    """Retain a defensive full-population copy for reversible layer filtering."""
    if isinstance(source_data, np.ndarray):
        copied_data: np.ndarray | tuple[np.ndarray, ...] = np.array(
            source_data, copy=True
        )
    else:
        copied_data = tuple(np.array(value, copy=True) for value in source_data)
    count = len(copied_data)
    distances = np.array(distances_base_voxels, dtype=np.float64, copy=True)
    if distances.shape != (count,):
        raise ValueError("replay geometry distances do not match geometry")
    distance_visibility_mask(distances, 0.0)

    copied_features = None
    if source_features is not None:
        copied_features = {}
        for name, values in source_features.items():
            if len(values) != count:
                raise ValueError(
                    f"replay geometry feature {name!r} does not match geometry"
                )
            copied_features[name] = tuple(copy.deepcopy(value) for value in values)
    return ReplayGeometryFilter(
        key=key,
        layer=layer,
        source_data=copied_data,
        source_features=copied_features,
        distances_base_voxels=distances,
        color_attribute=color_attribute,
        color_value=color_value,
        empty_color_value=empty_color_value,
        display_width=(
            common_shape_edge_width(layer) if hasattr(layer, "edge_width") else None
        ),
        display_size=(
            float(np.asarray(layer.size).reshape(-1)[0])
            if hasattr(layer, "size") and np.asarray(layer.size).size
            else None
        ),
    )


def filtered_replay_geometry_data(
    visual_filter: ReplayGeometryFilter,
    radius_base_voxels: float,
) -> tuple[np.ndarray | list[np.ndarray], dict[str, list] | None]:
    """Return the source-ordered geometry and features inside a replay radius."""
    visible = distance_visibility_mask(
        visual_filter.distances_base_voxels, radius_base_voxels
    )
    indices = np.flatnonzero(visible)
    if isinstance(visual_filter.source_data, np.ndarray):
        data: np.ndarray | list[np.ndarray] = np.array(
            visual_filter.source_data[indices], copy=True
        )
    else:
        data = [
            np.array(visual_filter.source_data[index], copy=True) for index in indices
        ]
    features = None
    if visual_filter.source_features is not None:
        features = {
            name: [copy.deepcopy(values[index]) for index in indices]
            for name, values in visual_filter.source_features.items()
        }
    return data, features


def apply_replay_geometry_filter(
    visual_filter: ReplayGeometryFilter,
    radius_base_voxels: float,
) -> None:
    """Physically remove out-of-radius geometry from one Napari layer."""
    layer = visual_filter.layer
    data, features = filtered_replay_geometry_data(visual_filter, radius_base_voxels)
    if hasattr(layer, "edge_width") and (
        getattr(layer, "_vc_display_edge_width", None) is not None
        or np.asarray(layer.edge_width).size
    ):
        visual_filter.display_width = common_shape_edge_width(layer)
    if hasattr(layer, "size") and np.asarray(layer.size).size:
        visual_filter.display_size = float(np.asarray(layer.size).reshape(-1)[0])
    if hasattr(layer, "selected_data"):
        layer.selected_data = set()
    layer.data = data
    if features is not None:
        layer.features = features
    color_value = (
        visual_filter.empty_color_value
        if not len(visual_filter.source_data)
        and visual_filter.empty_color_value is not None
        else visual_filter.color_value
    )
    setattr(layer, visual_filter.color_attribute, color_value)
    if visual_filter.display_width is not None:
        layer.edge_width = visual_filter.display_width
    if visual_filter.display_size is not None:
        layer.size = visual_filter.display_size


def replace_replay_geometry_filter_sources(
    templates: Sequence[ReplayGeometryFilter],
    sources: dict[
        str,
        tuple[np.ndarray | Sequence[np.ndarray], dict[str, Sequence] | None],
    ],
    distances_by_key: dict[str, np.ndarray],
) -> list[ReplayGeometryFilter]:
    """Prepare reload filter state without mutating any displayed layer."""
    template_by_key = {value.key: value for value in templates}
    if len(template_by_key) != len(templates):
        raise ValueError("replay geometry filter keys are duplicated")
    if set(sources) != set(template_by_key) or not set(distances_by_key).issubset(
        template_by_key
    ):
        raise ValueError("reloaded replay geometry layer topology differs")
    replacements = []
    for key, template in template_by_key.items():
        source_data, source_features = sources[key]
        distances = distances_by_key.get(key)
        if distances is None:
            if len(source_data):
                raise ValueError("reloaded replay geometry distances are missing")
            distances = np.empty(0, dtype=np.float64)
        replacement = make_replay_geometry_filter(
            key=key,
            layer=template.layer,
            source_data=source_data,
            source_features=source_features,
            distances_base_voxels=distances,
            color_attribute=template.color_attribute,
            color_value=template.color_value,
            empty_color_value=template.empty_color_value,
        )
        replacement.display_width = template.display_width
        replacement.display_size = template.display_size
        replacements.append(replacement)
    return replacements


def launch_viewer(
    level: OmeZarrLevel,
    crop_xyzwhd: tuple[int, int, int, int, int, int],
    anchors_obj: str | Path | None = None,
    paths_obj: str | Path | None = None,
    replay: FiberReplayBundle | None = None,
    anchor_stages: Sequence[AnchorStageGeometry] = (),
) -> None:
    try:
        import napari
        from napari.utils import Colormap
        from napari.utils.colormaps import AVAILABLE_COLORMAPS
    except ImportError as exc:
        raise RuntimeError(
            "napari is not installed; install the vesuvius GUI extra"
        ) from exc

    display_radius_defaults = replay_display_radius_defaults_base()
    data, selection = open_lazy_crop(level, crop_xyzwhd)
    if replay is not None:
        import zarr

        array = zarr.open_array(str(level.array_path), mode="r")
        if tuple(array.shape) != replay.prediction_shape_zyx:
            raise ValueError(
                "external fiber-presence Zarr shape does not match replay metadata"
            )
        expected_scale = (replay.prediction_to_base_scale,) * 3
        if not np.allclose(level.scale_zyx, expected_scale, rtol=0.0, atol=1e-12):
            raise ValueError(
                "external fiber-presence Zarr scale does not match replay metadata"
            )
    dense_gib = int(np.prod(selection.shape_zyx)) * data.dtype.itemsize / 1024**3
    stored_bounds = ",".join(
        f"{axis_slice.start}:{axis_slice.stop}" for axis_slice in selection.slices_zyx
    )
    print(f"OME-Zarr: {level.root}")
    print(f"Level: {level.path} scale_zyx={level.scale_zyx}")
    print(f"Stored crop ZYX: {stored_bounds} shape={selection.shape_zyx}")
    print(f"Dense crop size: {dense_gib:.3f} GiB")

    presence_source_data = data
    presence_distance_base = None
    presence_distance_data = None
    presence_radius_base_voxels = None
    if replay is not None and replay.tube_radius_base_voxels is not None:
        import dask.array as da

        print("Rasterizing replay reference/trace and computing presence-tube EDT...")
        presence_distance_base = replay_distance_transform_base(
            replay.reference_zyx,
            (*replay.greedy_segments_zyx, *replay.fiberlet_segments_zyx),
            selection,
            level.scale_zyx,
        )
        presence_distance_data = da.from_array(
            presence_distance_base,
            chunks=data.chunks,
        )
        presence_radius_base_voxels = display_radius_defaults["presence"]
        data = mask_presence_by_distance(
            presence_source_data,
            presence_distance_data,
            presence_radius_base_voxels,
        )

    replay_artifacts = (
        load_replay_visual_artifacts(replay)
        if replay is not None and replay.failure_zyx is not None
        else None
    )
    anchors = (
        replay_artifacts.anchors
        if replay_artifacts is not None
        else (
            read_line_obj(anchors_obj, "anchors", crop_xyzwhd)
            if anchors_obj is not None
            else None
        )
    )
    anchor_cells = (
        replay_artifacts.anchor_cells if replay_artifacts is not None else None
    )
    fiberlets = (
        replay_artifacts.fiberlets
        if replay_artifacts is not None
        else (
            read_line_obj(paths_obj, "paths", crop_xyzwhd)
            if paths_obj is not None
            else None
        )
    )
    if anchors is not None:
        print(
            f"Anchors: {len(anchors.paths_zyx)}/{anchors.total_groups} groups intersect crop"
        )
    if anchor_cells is not None:
        print(
            f"Anchor cells: {len(anchor_cells.centers_zyx)} centers, "
            f"{len(anchor_cells.displacements_zyx)} accepted offsets"
        )
    anchor_stage_geometry = (
        replay_artifacts.anchor_stages
        if replay_artifacts is not None
        else tuple(anchor_stages)
    )
    for stage in anchor_stage_geometry:
        reasons = (
            ",".join(f"{name}={count}" for name, count in sorted(stage.reasons.items()))
            or "none"
        )
        print(
            f"Anchor stage {stage.stage}: records={stage.record_count} "
            f"geometry={stage.geometric_record_count} reasons={reasons}"
        )
    if fiberlets is not None:
        print(
            f"Fiberlets: {len(fiberlets.paths_zyx)}/{fiberlets.total_groups} groups intersect crop"
        )

    viewer = napari.Viewer(ndisplay=3, title="Fiber presence")
    if np.issubdtype(data.dtype, np.integer):
        contrast_limits = (0, np.iinfo(data.dtype).max)
    else:
        contrast_limits = (0.0, 1.0)
    volume_layer = viewer.add_image(
        data,
        name=f"fiber presence [{level.path}]",
        scale=level.scale_zyx,
        translate=selection.origin_base_zyx,
        colormap="HiLo",
        contrast_limits=contrast_limits,
        rendering="attenuated_mip",
    )
    derived_state = {
        "presence_distance_base": presence_distance_base,
        "presence_distance_data": presence_distance_data,
        "presence_radius": presence_radius_base_voxels,
        "anchor_visual_filters": [],
        "anchor_radius": (
            display_radius_defaults["anchors"]
            if replay is not None and replay.tube_radius_base_voxels is not None
            else None
        ),
        "fiberlet_visual_filter": None,
        "fiberlet_radius": (
            display_radius_defaults["fiberlets"]
            if replay is not None and replay.tube_radius_base_voxels is not None
            else None
        ),
    }

    def set_presence_radius(radius_base_voxels: float) -> None:
        distance_data = derived_state["presence_distance_data"]
        if distance_data is None:
            return
        derived_state["presence_radius"] = radius_base_voxels
        volume_layer.data = mask_presence_by_distance(
            presence_source_data,
            distance_data,
            radius_base_voxels,
        )

    def set_anchor_radius(radius_base_voxels: float) -> None:
        distance_visibility_mask(np.empty(0, dtype=np.float64), radius_base_voxels)
        derived_state["anchor_radius"] = radius_base_voxels
        for visual_filter in derived_state["anchor_visual_filters"]:
            apply_replay_geometry_filter(visual_filter, radius_base_voxels)

    def set_fiberlet_radius(radius_base_voxels: float) -> None:
        distance_visibility_mask(np.empty(0, dtype=np.float64), radius_base_voxels)
        derived_state["fiberlet_radius"] = radius_base_voxels
        visual_filter = derived_state["fiberlet_visual_filter"]
        if visual_filter is not None:
            apply_replay_geometry_filter(visual_filter, radius_base_voxels)

    anchor_filter_specs: list[tuple] = []
    anchors_layer = None
    if anchors is not None and (anchors.paths_zyx or replay_artifacts is not None):
        anchors_layer = viewer.add_shapes(
            anchors.paths_zyx or None,
            ndim=3,
            shape_type="line",
            name="fiber anchors",
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
            visible=not anchor_stage_geometry,
        )
        anchor_filter_specs.append(
            (
                "anchors",
                anchors_layer,
                "edge_color",
                "cyan",
                anchors.paths_zyx,
                None,
                anchor_path_representatives(anchors.paths_zyx),
            )
        )
    anchor_stage_layers = []
    anchor_stage_layers_by_name = {}
    for stage in anchor_stage_geometry:
        if not stage.paths_zyx and replay_artifacts is None:
            continue
        layer = viewer.add_shapes(
            stage.paths_zyx or None,
            ndim=3,
            shape_type="line",
            name=anchor_stage_layer_name(stage),
            features=stage.features,
            edge_color=_ANCHOR_STAGE_COLORS[stage.stage],
            edge_width=2,
            face_color="transparent",
            visible=stage.stage == "nms",
        )
        anchor_stage_layers.append(layer)
        anchor_stage_layers_by_name[stage.stage] = layer
        anchor_filter_specs.append(
            (
                f"stage:{stage.stage}",
                layer,
                "edge_color",
                _ANCHOR_STAGE_COLORS[stage.stage],
                stage.paths_zyx,
                stage.features,
                anchor_path_representatives(stage.paths_zyx),
            )
        )
    anchor_cell_centers_layer = None
    anchor_displacements_layer = None
    if anchor_cells is not None:
        anchor_cell_centers_layer = viewer.add_points(
            anchor_cells.centers_zyx,
            name="anchor cell centers",
            face_color="yellow",
            size=2,
        )
        anchor_filter_specs.append(
            (
                "cell_centers",
                anchor_cell_centers_layer,
                "face_color",
                "yellow",
                anchor_cells.centers_zyx,
                None,
                np.asarray(anchor_cells.centers_zyx, dtype=np.float64),
            )
        )
        if anchor_cells.displacements_zyx or replay_artifacts is not None:
            anchor_displacements_layer = viewer.add_shapes(
                anchor_cells.displacements_zyx or None,
                ndim=3,
                shape_type="line",
                name="anchor refinement offsets",
                edge_color="orange",
                edge_width=1,
                face_color="transparent",
            )
            anchor_filter_specs.append(
                (
                    "displacements",
                    anchor_displacements_layer,
                    "edge_color",
                    "orange",
                    anchor_cells.displacements_zyx,
                    None,
                    anchor_path_representatives(
                        anchor_cells.displacements_zyx,
                        target_endpoint=True,
                    ),
                )
            )
    paths_layer = None
    path_quality_colormaps: dict[str, object] | None = None
    if fiberlets is not None and (fiberlets.paths_zyx or replay_artifacts is not None):
        custom_name, custom_colors, custom_controls = fiberlet_quality_colormap_spec()
        custom_colormap = Colormap(
            custom_colors,
            controls=custom_controls,
            name=custom_name,
        )
        path_quality_colormaps = {
            name: custom_colormap if name == custom_name else name
            for name in fiberlet_colormap_names(tuple(AVAILABLE_COLORMAPS))
        }
        paths_layer = viewer.add_shapes(
            fiberlets.paths_zyx or None,
            ndim=3,
            shape_type="path",
            name="fiberlet paths",
            features=fiberlet_layer_features(fiberlets),
            edge_color=("relative_quality" if fiberlets.paths_zyx else "gray"),
            edge_colormap=custom_colormap,
            edge_contrast_limits=(0.0, 1.0),
            edge_width=2,
            face_color="transparent",
        )
    reference_layer = None
    trace_layer = None
    fiberlet_route_layer = None
    failure_layer = None
    strip_layers: tuple = ()
    if replay is not None:
        reference_layer = viewer.add_shapes(
            [replay.reference_zyx],
            shape_type="path",
            name="reference fiber",
            edge_color="white",
            edge_width=2,
            face_color="transparent",
        )
        trace_layer = viewer.add_shapes(
            list(replay.greedy_segments_zyx) or None,
            ndim=3,
            shape_type="path",
            name="greedy replay",
            edge_color="magenta",
            edge_width=2,
            face_color="transparent",
        )
        if replay.fiberlet_segments_zyx:
            fiberlet_route_layer = viewer.add_shapes(
                list(replay.fiberlet_segments_zyx),
                ndim=3,
                shape_type="path",
                name="fiberlet graph replay",
                edge_color="lime",
                edge_width=3,
                face_color="transparent",
            )
        if replay.failure_zyx is not None:
            failure_layer = viewer.add_points(
                replay.failure_zyx,
                name="replay failure",
                face_color="red",
                size=4,
            )
        if replay_artifacts is not None and replay_artifacts.strips is not None:
            strip_contrast_limits = replay_strip_contrast_limits(
                replay_artifacts.strips
            )
            strip_specs = (
                (
                    "reference CT strip",
                    replay_artifacts.strips.reference,
                ),
                (
                    "greedy CT strip",
                    replay_artifacts.strips.greedy,
                ),
                (
                    "fiberlet CT strip",
                    replay_artifacts.strips.fiberlet,
                ),
            )
            strip_layers = tuple(
                viewer.add_surface(
                    (
                        geometry.vertices_zyx,
                        geometry.triangles,
                        geometry.normalized_ct_intensity,
                    ),
                    name=name,
                    colormap="gray",
                    contrast_limits=strip_contrast_limits,
                    shading="none",
                    visible=False,
                )
                for name, geometry in strip_specs
            )

    anchor_radius_base_voxels = None
    maximum_anchor_radius_base_voxels = None
    set_anchor_radius_callback = None
    if (
        replay is not None
        and replay_artifacts is not None
        and replay.tube_radius_base_voxels is not None
        and anchor_filter_specs
    ):
        print("Computing exact replay distance for anchor diagnostics...")
        anchor_distances = replay_anchor_distances_base(replay, replay_artifacts)
        anchor_visual_filters = []
        for (
            key,
            layer,
            color_attribute,
            color_value,
            source_data,
            source_features,
            representatives,
        ) in anchor_filter_specs:
            distances = anchor_distances.get(key, np.empty(0, dtype=np.float64))
            count = len(representatives)
            if len(distances) != count:
                raise ValueError("anchor diagnostic distances do not match geometry")
            anchor_visual_filters.append(
                make_replay_geometry_filter(
                    key=key,
                    layer=layer,
                    source_data=source_data,
                    source_features=source_features,
                    distances_base_voxels=distances,
                    color_attribute=color_attribute,
                    color_value=color_value,
                )
            )

        anchor_radius_base_voxels = display_radius_defaults["anchors"]
        maximum_anchor_radius_base_voxels = max(
            anchor_radius_base_voxels,
            max(
                (
                    float(np.max(values))
                    for values in anchor_distances.values()
                    if len(values)
                ),
                default=anchor_radius_base_voxels,
            ),
        )
        derived_state["anchor_visual_filters"] = anchor_visual_filters
        set_anchor_radius(anchor_radius_base_voxels)
        set_anchor_radius_callback = set_anchor_radius

    fiberlet_radius_base_voxels = None
    maximum_fiberlet_radius_base_voxels = None
    set_fiberlet_radius_callback = None
    if (
        replay is not None
        and replay_artifacts is not None
        and replay.tube_radius_base_voxels is not None
        and paths_layer is not None
    ):
        print("Computing exact replay distance for fiberlet paths...")
        fiberlet_distances = replay_fiberlet_distances_base(replay, fiberlets)
        fiberlet_visual_filter = make_replay_geometry_filter(
            key="fiberlets",
            layer=paths_layer,
            source_data=fiberlets.paths_zyx,
            source_features=fiberlet_layer_features(fiberlets),
            distances_base_voxels=fiberlet_distances,
            color_attribute="edge_color",
            color_value="relative_quality",
            empty_color_value="gray",
        )
        fiberlet_radius_base_voxels = display_radius_defaults["fiberlets"]
        maximum_fiberlet_radius_base_voxels = max(
            fiberlet_radius_base_voxels,
            float(np.max(fiberlet_distances))
            if len(fiberlet_distances)
            else fiberlet_radius_base_voxels,
        )
        derived_state["fiberlet_visual_filter"] = fiberlet_visual_filter
        set_fiberlet_radius(fiberlet_radius_base_voxels)
        set_fiberlet_radius_callback = set_fiberlet_radius

    reload_artifacts_callback = None
    if replay is not None and replay_artifacts is not None:
        import dask.array as da

        replay_root_path = replay.path
        replay_state = {
            "replay": replay,
            "artifacts": replay_artifacts,
        }

        def build_reloaded_anchor_filters(
            artifacts: ReplayVisualArtifacts,
            distances_by_key: dict[str, np.ndarray],
        ) -> list[ReplayGeometryFilter]:
            sources: dict[str, tuple[object, dict[str, Sequence] | None]] = {
                "anchors": (artifacts.anchors.paths_zyx, None),
                "cell_centers": (artifacts.anchor_cells.centers_zyx, None),
                "displacements": (
                    artifacts.anchor_cells.displacements_zyx,
                    None,
                ),
            }
            sources.update(
                {
                    f"stage:{stage.stage}": (stage.paths_zyx, stage.features)
                    for stage in artifacts.anchor_stages
                }
            )
            return replace_replay_geometry_filter_sources(
                derived_state["anchor_visual_filters"],
                sources,
                distances_by_key,
            )

        def build_reloaded_fiberlet_filter(
            artifacts: ReplayVisualArtifacts,
            distances: np.ndarray,
        ) -> ReplayGeometryFilter:
            template = derived_state["fiberlet_visual_filter"]
            if template is None:
                raise ValueError("fiberlet replay filter is unavailable")
            return replace_replay_geometry_filter_sources(
                [template],
                {
                    "fiberlets": (
                        artifacts.fiberlets.paths_zyx,
                        fiberlet_layer_features(artifacts.fiberlets),
                    )
                },
                {"fiberlets": distances},
            )[0]

        def mutable_replay_layers() -> tuple:
            return tuple(
                layer
                for layer in (
                    anchors_layer,
                    *anchor_stage_layers,
                    anchor_cell_centers_layer,
                    anchor_displacements_layer,
                    paths_layer,
                    reference_layer,
                    trace_layer,
                    fiberlet_route_layer,
                    failure_layer,
                    *strip_layers,
                    volume_layer,
                )
                if layer is not None
            )

        def apply_replay_artifacts(
            candidate_replay: FiberReplayBundle,
            candidate_artifacts: ReplayVisualArtifacts,
            candidate_presence_data,
            candidate_presence_distance_base: np.ndarray,
            candidate_presence_distance_data,
            candidate_anchor_filters: list[ReplayGeometryFilter],
            candidate_fiberlet_filter: ReplayGeometryFilter,
            selected_data: dict[int, set] | None = None,
        ) -> None:
            layers = mutable_replay_layers()
            widths = {
                id(layer): common_shape_edge_width(layer)
                for layer in layers
                if hasattr(layer, "edge_width")
            }
            sizes = {
                id(layer): float(np.asarray(layer.size).reshape(-1)[0])
                for layer in layers
                if hasattr(layer, "size") and np.asarray(layer.size).size
            }
            with ExitStack() as stack:
                for layer in layers:
                    blocker = getattr(layer.events, "blocker_all", None)
                    if blocker is not None:
                        stack.enter_context(blocker())
                    if hasattr(layer, "selected_data"):
                        layer.selected_data = set()

                for layer in layers:
                    if id(layer) in widths:
                        layer.edge_width = widths[id(layer)]
                    if id(layer) in sizes:
                        layer.size = sizes[id(layer)]
                reference_layer.data = [candidate_replay.reference_zyx]
                trace_layer.data = list(candidate_replay.greedy_segments_zyx)
                if fiberlet_route_layer is not None:
                    fiberlet_route_layer.data = list(
                        candidate_replay.fiberlet_segments_zyx
                    )
                failure_layer.data = candidate_replay.failure_zyx
                if strip_layers:
                    if candidate_artifacts.strips is None or len(strip_layers) != 3:
                        raise ValueError("replacement replay strip data is incomplete")
                    strip_data = (
                        candidate_artifacts.strips.reference,
                        candidate_artifacts.strips.greedy,
                        candidate_artifacts.strips.fiberlet,
                    )
                    for layer, geometry in zip(strip_layers, strip_data, strict=True):
                        layer.data = (
                            geometry.vertices_zyx,
                            geometry.triangles,
                            geometry.normalized_ct_intensity,
                        )
                for stage in candidate_artifacts.anchor_stages:
                    layer = anchor_stage_layers_by_name.get(stage.stage)
                    if layer is None:
                        continue
                    layer.name = anchor_stage_layer_name(stage)
                derived_state["presence_distance_base"] = (
                    candidate_presence_distance_base
                )
                derived_state["presence_distance_data"] = (
                    candidate_presence_distance_data
                )
                derived_state["anchor_visual_filters"] = candidate_anchor_filters
                derived_state["fiberlet_visual_filter"] = candidate_fiberlet_filter
                volume_layer.data = candidate_presence_data
                for visual_filter in candidate_anchor_filters:
                    apply_replay_geometry_filter(
                        visual_filter, derived_state["anchor_radius"]
                    )
                apply_replay_geometry_filter(
                    candidate_fiberlet_filter,
                    derived_state["fiberlet_radius"],
                )

                for layer in layers:
                    if id(layer) in widths:
                        layer.edge_width = widths[id(layer)]
                    if id(layer) in sizes:
                        layer.size = sizes[id(layer)]
                    if selected_data is not None and id(layer) in selected_data:
                        layer.selected_data = selected_data[id(layer)]

            if paths_layer is not None:
                paths_layer.refresh_colors()
            for layer in layers:
                layer.refresh()

        def reload_artifacts() -> str:
            current_replay = replay_state["replay"]
            current_artifacts = replay_state["artifacts"]
            replacement_replay = load_fiber_replay_bundle(
                replay_root_path,
                include_anchor_stages=bool(current_replay.anchor_stages),
            )
            replacement_artifacts = load_replay_visual_artifacts(replacement_replay)
            validate_replay_reload_compatibility(
                current_replay,
                current_artifacts,
                replacement_replay,
                replacement_artifacts,
            )
            replacement_presence_distance_base = replay_distance_transform_base(
                replacement_replay.reference_zyx,
                (
                    *replacement_replay.greedy_segments_zyx,
                    *replacement_replay.fiberlet_segments_zyx,
                ),
                selection,
                level.scale_zyx,
            )
            replacement_presence_distance_data = da.from_array(
                replacement_presence_distance_base,
                chunks=presence_source_data.chunks,
            )
            replacement_presence_data = mask_presence_by_distance(
                presence_source_data,
                replacement_presence_distance_data,
                derived_state["presence_radius"],
            )
            replacement_anchor_filters = build_reloaded_anchor_filters(
                replacement_artifacts,
                replay_anchor_distances_base(
                    replacement_replay,
                    replacement_artifacts,
                ),
            )
            replacement_fiberlet_filter = build_reloaded_fiberlet_filter(
                replacement_artifacts,
                replay_fiberlet_distances_base(
                    replacement_replay,
                    replacement_artifacts.fiberlets,
                ),
            )

            old_presence_data = volume_layer.data
            old_presence_distance_base = derived_state["presence_distance_base"]
            old_presence_distance_data = derived_state["presence_distance_data"]
            old_anchor_filters = derived_state["anchor_visual_filters"]
            old_fiberlet_filter = derived_state["fiberlet_visual_filter"]
            old_selected_data = {
                id(layer): set(layer.selected_data)
                for layer in mutable_replay_layers()
                if hasattr(layer, "selected_data")
            }

            def commit() -> None:
                apply_replay_artifacts(
                    replacement_replay,
                    replacement_artifacts,
                    replacement_presence_data,
                    replacement_presence_distance_base,
                    replacement_presence_distance_data,
                    replacement_anchor_filters,
                    replacement_fiberlet_filter,
                )

            def rollback() -> None:
                derived_state["presence_distance_base"] = old_presence_distance_base
                derived_state["presence_distance_data"] = old_presence_distance_data
                derived_state["anchor_visual_filters"] = old_anchor_filters
                derived_state["fiberlet_visual_filter"] = old_fiberlet_filter
                apply_replay_artifacts(
                    current_replay,
                    current_artifacts,
                    old_presence_data,
                    old_presence_distance_base,
                    old_presence_distance_data,
                    old_anchor_filters,
                    old_fiberlet_filter,
                    old_selected_data,
                )

            commit_with_rollback(commit, rollback)

            replay_state["replay"] = replacement_replay
            replay_state["artifacts"] = replacement_artifacts
            return (
                "Reloaded replay artifacts: "
                f"anchors={len(replacement_artifacts.anchors.paths_zyx)} "
                f"fiberlets={len(replacement_artifacts.fiberlets.paths_zyx)}"
            )

        reload_artifacts_callback = reload_artifacts

    maximum_display_radius = math.sqrt(
        crop_xyzwhd[3] ** 2 + crop_xyzwhd[4] ** 2 + crop_xyzwhd[5] ** 2
    )
    add_clipping_controls(
        viewer=viewer,
        volume_layer=volume_layer,
        anchors_layer=anchors_layer,
        paths_layer=paths_layer,
        crop_xyzwhd=crop_xyzwhd,
        path_quality_colormaps=path_quality_colormaps,
        reference_layer=reference_layer,
        trace_layer=trace_layer,
        failure_layer=failure_layer,
        anchor_cell_centers_layer=anchor_cell_centers_layer,
        anchor_displacements_layer=anchor_displacements_layer,
        anchor_stage_layers=anchor_stage_layers,
        additional_layers=(fiberlet_route_layer, *strip_layers),
        presence_radius_base_voxels=presence_radius_base_voxels,
        maximum_presence_radius_base_voxels=(
            maximum_display_radius if presence_distance_base is not None else None
        ),
        set_presence_radius=(
            set_presence_radius if presence_distance_data is not None else None
        ),
        anchor_radius_base_voxels=anchor_radius_base_voxels,
        maximum_anchor_radius_base_voxels=(
            maximum_display_radius
            if maximum_anchor_radius_base_voxels is not None
            else None
        ),
        set_anchor_radius=set_anchor_radius_callback,
        fiberlet_radius_base_voxels=fiberlet_radius_base_voxels,
        maximum_fiberlet_radius_base_voxels=(
            maximum_display_radius
            if maximum_fiberlet_radius_base_voxels is not None
            else None
        ),
        set_fiberlet_radius=set_fiberlet_radius_callback,
        reload_artifacts=reload_artifacts_callback,
    )
    viewer.reset_view()
    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zarr",
        help="Local fiber-presence OME-Zarr pyramid root or array level",
    )
    parser.add_argument(
        "--crop",
        type=parse_crop,
        metavar="X,Y,Z,W,H,D",
        help="Half-open crop in base voxels",
    )
    parser.add_argument(
        "--level",
        help="OME-Zarr dataset path; defaults to the finest (first) level",
    )
    parser.add_argument(
        "--anchors",
        help="Fiberlet anchors OBJ to show as a separate line layer",
    )
    parser.add_argument(
        "--paths",
        help="Fiberlet paths OBJ to show as a separate path layer",
    )
    parser.add_argument(
        "--replay",
        help="Direct replay visualization manifest",
    )
    parser.add_argument(
        "--anchor-stages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the five detailed anchor diagnostic layers (default: enabled)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.replay is not None and any(
            value is not None for value in (args.crop, args.anchors, args.paths)
        ):
            raise ValueError(
                "--replay cannot be combined with --crop, --anchors, or --paths"
            )
        if args.replay is None and args.crop is None:
            raise ValueError("manual mode requires --crop")
        replay = (
            load_fiber_replay_bundle(
                args.replay,
                include_anchor_stages=args.anchor_stages,
            )
            if args.replay
            else None
        )
        crop = replay.crop_xyzwhd if replay is not None else args.crop
        anchors = replay.anchors_obj if replay is not None else args.anchors
        paths = replay.paths_obj if replay is not None else args.paths
        anchor_stages = ()
        if replay is None and anchors is not None and args.anchor_stages:
            stage_directory = Path(anchors).expanduser().resolve().parent / "stages"
            if stage_directory.exists():
                anchor_stages = load_anchor_stage_directory(
                    stage_directory, anchors, crop
                )
        resolved = resolve_ome_zarr_level(args.zarr, args.level)
        launch_viewer(
            resolved,
            crop,
            anchors,
            paths,
            replay,
            anchor_stages,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
