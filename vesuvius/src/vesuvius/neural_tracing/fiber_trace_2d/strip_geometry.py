from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace.fiber_json import Vc3dFiber


_EPS = 1.0e-12
_ROLL_SMOOTHNESS = 4.0
_ROLL_SMOOTH_ITERATIONS = 80


@dataclass(frozen=True)
class FiberStripFrame:
    tangent_xyz: np.ndarray
    side_xyz: np.ndarray
    mesh_normal_xyz: np.ndarray


@dataclass(frozen=True)
class FiberStripGrid:
    coords_xyz: np.ndarray
    coords_zyx: np.ndarray
    valid_mask: np.ndarray
    frame: FiberStripFrame
    offset_axis_xyz: np.ndarray | None = None
    offset_axis_zyx: np.ndarray | None = None


@dataclass(frozen=True)
class FiberStripGridTorch:
    coords_xyz: torch.Tensor
    coords_zyx: torch.Tensor
    valid_mask: torch.Tensor
    frame: FiberStripFrame
    offset_axis_xyz: torch.Tensor | None = None
    offset_axis_zyx: torch.Tensor | None = None

    def to_numpy(self) -> FiberStripGrid:
        coords_xyz = self.coords_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        coords_zyx = self.coords_zyx.detach().cpu().numpy().astype(np.float32, copy=False)
        valid_mask = self.valid_mask.detach().cpu().numpy().astype(bool, copy=False)
        offset_axis_xyz = (
            None
            if self.offset_axis_xyz is None
            else self.offset_axis_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        offset_axis_zyx = (
            None
            if self.offset_axis_zyx is None
            else self.offset_axis_zyx.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        return FiberStripGrid(
            coords_xyz=coords_xyz,
            coords_zyx=coords_zyx,
            valid_mask=valid_mask,
            frame=self.frame,
            offset_axis_xyz=offset_axis_xyz,
            offset_axis_zyx=offset_axis_zyx,
        )


@dataclass(frozen=True)
class FiberStripLineWindow:
    line_points_xyz: np.ndarray
    original_line_indices: np.ndarray
    local_control_point_index: int


def _finite(v: np.ndarray) -> bool:
    return bool(np.isfinite(v).all())


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalized_or_zero(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if not _finite(v):
        return np.zeros(3, dtype=np.float64)
    n = _norm(v)
    if n <= _EPS:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _valid_direction(v: np.ndarray) -> bool:
    return _finite(v) and _norm(v) > _EPS


def _axis_fallback_least_aligned_with(reference: np.ndarray) -> np.ndarray:
    ref = _normalized_or_zero(reference)
    axes = np.eye(3, dtype=np.float64)
    dots = np.abs(axes @ ref)
    return axes[int(np.argmin(dots))]


def _project_to_tangent_plane(vector: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    tangent = _normalized_or_zero(tangent)
    projected = np.asarray(vector, dtype=np.float64) - tangent * float(np.dot(vector, tangent))
    return _normalized_or_zero(projected)


def _side_direction(normal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    side = _normalized_or_zero(np.cross(normal, tangent))
    if _valid_direction(side):
        return side
    side = _normalized_or_zero(np.cross(_axis_fallback_least_aligned_with(tangent), tangent))
    if _valid_direction(side):
        return side
    return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def _rotate_around_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    unit_axis = _normalized_or_zero(axis)
    if not _valid_direction(unit_axis):
        return np.asarray(vector, dtype=np.float64)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    vector = np.asarray(vector, dtype=np.float64)
    return (
        vector * c
        + np.cross(unit_axis, vector) * s
        + unit_axis * float(np.dot(unit_axis, vector)) * (1.0 - c)
    )


def _transport_normal(
    previous_normal: np.ndarray, previous_tangent: np.ndarray, tangent: np.ndarray
) -> np.ndarray:
    axis = np.cross(previous_tangent, tangent)
    sin_angle = _norm(axis)
    cos_angle = float(np.clip(np.dot(previous_tangent, tangent), -1.0, 1.0))
    transported = np.asarray(previous_normal, dtype=np.float64)
    if sin_angle > _EPS:
        transported = _rotate_around_axis(previous_normal, axis, float(np.arctan2(sin_angle, cos_angle)))
    transported = _project_to_tangent_plane(transported, tangent)
    if _valid_direction(transported):
        return transported
    side = _side_direction(_axis_fallback_least_aligned_with(tangent), tangent)
    transported = _normalized_or_zero(np.cross(tangent, side))
    if _valid_direction(transported):
        return transported
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


def _unwrap_axis_near(angle: float, reference: float) -> float:
    half_pi = 0.5 * np.pi
    while angle - reference > half_pi:
        angle -= np.pi
    while angle - reference < -half_pi:
        angle += np.pi
    return float(angle)


def _smooth_roll_angles(targets: np.ndarray) -> np.ndarray:
    angles = np.asarray(targets, dtype=np.float64).copy()
    if angles.size < 2:
        return angles
    for _ in range(_ROLL_SMOOTH_ITERATIONS):
        old = angles.copy()
        for i in range(angles.size):
            neighbor_sum = 0.0
            neighbor_count = 0.0
            if i > 0:
                neighbor_sum += old[i - 1]
                neighbor_count += 1.0
            if i + 1 < angles.size:
                neighbor_sum += old[i + 1]
                neighbor_count += 1.0
            angles[i] = (targets[i] + _ROLL_SMOOTHNESS * neighbor_sum) / (
                1.0 + _ROLL_SMOOTHNESS * neighbor_count
            )
    return angles


def _tangent_at(points_xyz: np.ndarray, index: int) -> np.ndarray:
    if points_xyz.shape[0] < 2:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if index == 0:
        tangent = points_xyz[1] - points_xyz[0]
    elif index + 1 == points_xyz.shape[0]:
        tangent = points_xyz[index] - points_xyz[index - 1]
    else:
        tangent = points_xyz[index + 1] - points_xyz[index - 1]
    tangent = _normalized_or_zero(tangent)
    if _valid_direction(tangent):
        return tangent
    return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


def _arc_lengths(points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.shape[0] <= 1:
        return np.zeros(points_xyz.shape[0], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points_xyz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _line_index_for_control_point(line_points_xyz: np.ndarray, control_point_xyz: np.ndarray) -> int:
    matches = np.flatnonzero(np.all(line_points_xyz == control_point_xyz[None, :], axis=1))
    if matches.size == 0:
        raise ValueError(
            "control point is not an exact member of line_points; fiber JSON is inconsistent"
        )
    return int(matches[0])


def side_strip_line_window(
    fiber: Vc3dFiber,
    *,
    control_point_index: int,
    patch_shape_hw: tuple[int, int],
    pixel_spacing_base: float = 1.0,
    interpolation_point_margin: int = 2,
) -> FiberStripLineWindow:
    control_points = np.asarray(fiber.control_points_xyz, dtype=np.float64)
    if control_point_index < 0 or control_point_index >= control_points.shape[0]:
        raise IndexError(
            f"control_point_index {control_point_index} out of range for {control_points.shape[0]} control points"
        )
    line_points = np.asarray(fiber.line_points_xyz, dtype=np.float64)
    if line_points.ndim != 2 or line_points.shape[1] != 3 or line_points.shape[0] == 0:
        raise ValueError("fiber points must have shape [N, 3] with N > 0")
    _, width = (int(v) for v in patch_shape_hw)
    if width <= 0:
        raise ValueError(f"patch_shape_hw must contain positive values, got {patch_shape_hw}")
    pixel_spacing_base = float(pixel_spacing_base)
    if not np.isfinite(pixel_spacing_base) or pixel_spacing_base <= 0.0:
        raise ValueError(f"pixel_spacing_base must be positive and finite, got {pixel_spacing_base}")

    line_index = _line_index_for_control_point(line_points, control_points[control_point_index])
    cumulative = _arc_lengths(line_points)
    anchor_arc = float(cumulative[line_index])
    half_width = (float(width) - 1.0) * 0.5 * pixel_spacing_base
    start_arc = anchor_arc - half_width
    end_arc = anchor_arc + half_width
    start_index = int(np.searchsorted(cumulative, start_arc, side="right") - 1)
    end_index = int(np.searchsorted(cumulative, end_arc, side="left"))
    margin = max(0, int(interpolation_point_margin))
    start_index = max(0, min(start_index, line_index) - margin)
    end_index = min(line_points.shape[0] - 1, max(end_index, line_index) + margin)
    local_points = line_points[start_index : end_index + 1]
    original_indices = np.arange(start_index, end_index + 1, dtype=np.int64)
    return FiberStripLineWindow(
        line_points_xyz=local_points.astype(np.float64, copy=False),
        original_line_indices=original_indices,
        local_control_point_index=line_index - start_index,
    )


def control_point_line_index(fiber: Vc3dFiber, control_point_index: int) -> int:
    control_points = np.asarray(fiber.control_points_xyz, dtype=np.float64)
    if control_point_index < 0 or control_point_index >= control_points.shape[0]:
        raise IndexError(
            f"control_point_index {control_point_index} out of range for {control_points.shape[0]} control points"
        )
    line_points = np.asarray(fiber.line_points_xyz, dtype=np.float64)
    return _line_index_for_control_point(line_points, control_points[control_point_index])


def side_strip_segment_line_window(
    fiber: Vc3dFiber,
    *,
    start_control_point_index: int,
    target_control_point_index: int,
    margin_px: float,
    pixel_spacing_base: float = 1.0,
    interpolation_point_margin: int = 2,
) -> FiberStripLineWindow:
    if start_control_point_index == target_control_point_index:
        raise ValueError("start and target control points must be different")
    line_points = np.asarray(fiber.line_points_xyz, dtype=np.float64)
    if line_points.ndim != 2 or line_points.shape[1] != 3 or line_points.shape[0] == 0:
        raise ValueError("fiber points must have shape [N, 3] with N > 0")
    pixel_spacing_base = float(pixel_spacing_base)
    if not np.isfinite(pixel_spacing_base) or pixel_spacing_base <= 0.0:
        raise ValueError(f"pixel_spacing_base must be positive and finite, got {pixel_spacing_base}")

    start_line_index = control_point_line_index(fiber, start_control_point_index)
    target_line_index = control_point_line_index(fiber, target_control_point_index)
    cumulative = _arc_lengths(line_points)
    start_arc = float(cumulative[start_line_index])
    target_arc = float(cumulative[target_line_index])
    margin_base = max(0.0, float(margin_px)) * pixel_spacing_base
    start_arc_window = min(start_arc, target_arc) - margin_base
    end_arc_window = max(start_arc, target_arc) + margin_base
    start_index = int(np.searchsorted(cumulative, start_arc_window, side="right") - 1)
    end_index = int(np.searchsorted(cumulative, end_arc_window, side="left"))
    margin = max(0, int(interpolation_point_margin))
    start_index = max(0, min(start_index, start_line_index, target_line_index) - margin)
    end_index = min(line_points.shape[0] - 1, max(end_index, start_line_index, target_line_index) + margin)
    local_points = line_points[start_index : end_index + 1]
    original_indices = np.arange(start_index, end_index + 1, dtype=np.int64)
    return FiberStripLineWindow(
        line_points_xyz=local_points.astype(np.float64, copy=False),
        original_line_indices=original_indices,
        local_control_point_index=start_line_index - start_index,
    )


def source_line_xy_from_line_window(
    line_window: FiberStripLineWindow,
    *,
    patch_shape_hw: tuple[int, int],
    anchor_column_px: float | None = None,
    pixel_spacing_base: float = 1.0,
) -> np.ndarray:
    line_points = np.asarray(line_window.line_points_xyz, dtype=np.float64)
    height, width = (int(v) for v in patch_shape_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"patch_shape_hw must contain positive values, got {patch_shape_hw}")
    pixel_spacing_base = float(pixel_spacing_base)
    if not np.isfinite(pixel_spacing_base) or pixel_spacing_base <= 0.0:
        raise ValueError(f"pixel_spacing_base must be positive and finite, got {pixel_spacing_base}")
    anchor_col = (float(width) - 1.0) * 0.5 if anchor_column_px is None else float(anchor_column_px)
    anchor_index = int(line_window.local_control_point_index)
    if anchor_index < 0 or anchor_index >= line_points.shape[0]:
        raise IndexError(
            f"local_control_point_index {anchor_index} out of range for {line_points.shape[0]} line points"
        )
    cumulative = _arc_lengths(line_points)
    anchor_arc = float(cumulative[anchor_index])
    x = anchor_col + (cumulative - anchor_arc) / pixel_spacing_base
    y = np.full_like(x, (float(height) - 1.0) * 0.5, dtype=np.float64)
    return np.stack([x, y], axis=1).astype(np.float32)


def source_point_xy_for_line_index(
    line_window: FiberStripLineWindow,
    *,
    original_line_index: int,
    patch_shape_hw: tuple[int, int],
    anchor_column_px: float | None = None,
    pixel_spacing_base: float = 1.0,
) -> np.ndarray:
    matches = np.flatnonzero(np.asarray(line_window.original_line_indices, dtype=np.int64) == int(original_line_index))
    if matches.size == 0:
        raise ValueError(f"line index {original_line_index} is not inside the line window")
    return source_line_xy_from_line_window(
        line_window,
        patch_shape_hw=patch_shape_hw,
        anchor_column_px=anchor_column_px,
        pixel_spacing_base=pixel_spacing_base,
    )[int(matches[0])]


def _arc_derivatives(points_xyz: np.ndarray, cumulative: np.ndarray) -> np.ndarray:
    derivatives = np.zeros_like(points_xyz, dtype=np.float64)
    if points_xyz.shape[0] < 2:
        derivatives[:] = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        return derivatives
    for i in range(points_xyz.shape[0]):
        if i == 0:
            denom = cumulative[1] - cumulative[0]
            delta = points_xyz[1] - points_xyz[0]
        elif i + 1 == points_xyz.shape[0]:
            denom = cumulative[i] - cumulative[i - 1]
            delta = points_xyz[i] - points_xyz[i - 1]
        else:
            denom = cumulative[i + 1] - cumulative[i - 1]
            delta = points_xyz[i + 1] - points_xyz[i - 1]
        if denom > _EPS:
            derivatives[i] = delta / denom
        else:
            derivatives[i] = _tangent_at(points_xyz, i)
    return derivatives


def _arc_segment_indices(cumulative: np.ndarray, arc_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = float(cumulative[-1])
    valid = (arc_coord >= 0.0) & (arc_coord <= total)
    arc = np.clip(arc_coord, 0.0, total)
    c1 = np.searchsorted(cumulative, arc, side="right")
    c1 = np.clip(c1, 1, cumulative.shape[0] - 1)
    c0 = c1 - 1
    span = cumulative[c1] - cumulative[c0]
    t = np.divide(
        arc - cumulative[c0],
        span,
        out=np.zeros_like(arc, dtype=np.float64),
        where=span > _EPS,
    )
    return c0, c1, t, valid


def _cubic_hermite_line(points_xyz: np.ndarray, arc_coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_xyz.shape[0] == 1:
        center = np.broadcast_to(points_xyz[0], arc_coord.shape + (3,)).astype(np.float64)
        return center, np.ones(arc_coord.shape, dtype=bool)
    cumulative = _arc_lengths(points_xyz)
    derivatives = _arc_derivatives(points_xyz, cumulative)
    c0, c1, t, valid = _arc_segment_indices(cumulative, arc_coord)
    span = cumulative[c1] - cumulative[c0]
    p0 = points_xyz[c0]
    p1 = points_xyz[c1]
    m0 = derivatives[c0]
    m1 = derivatives[c1]
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    center = (
        h00[..., None] * p0
        + h10[..., None] * span[..., None] * m0
        + h01[..., None] * p1
        + h11[..., None] * span[..., None] * m1
    )
    return center, valid


def _cubic_hermite_tangent(points_xyz: np.ndarray, arc: float) -> np.ndarray:
    if points_xyz.shape[0] < 2:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    cumulative = _arc_lengths(points_xyz)
    derivatives = _arc_derivatives(points_xyz, cumulative)
    c0, c1, t, _ = _arc_segment_indices(cumulative, np.asarray(arc, dtype=np.float64))
    i0 = int(c0)
    i1 = int(c1)
    span = float(cumulative[i1] - cumulative[i0])
    tv = float(t)
    if span <= _EPS:
        return _tangent_at(points_xyz, i0)
    p0 = points_xyz[i0]
    p1 = points_xyz[i1]
    m0 = derivatives[i0]
    m1 = derivatives[i1]
    dh00 = 6.0 * tv * tv - 6.0 * tv
    dh10 = 3.0 * tv * tv - 4.0 * tv + 1.0
    dh01 = -6.0 * tv * tv + 6.0 * tv
    dh11 = 3.0 * tv * tv - 2.0 * tv
    tangent = (dh00 * p0 + dh10 * span * m0 + dh01 * p1 + dh11 * span * m1) / span
    tangent = _normalized_or_zero(tangent)
    if _valid_direction(tangent):
        return tangent
    return _tangent_at(points_xyz, i0)


def _fallback_mesh_normal_for_tangent(tangent: np.ndarray) -> np.ndarray:
    side = _side_direction(_axis_fallback_least_aligned_with(tangent), tangent)
    normal = _normalized_or_zero(np.cross(tangent, side))
    if _valid_direction(normal):
        return normal
    normal = _project_to_tangent_plane(np.asarray([0.0, 0.0, 1.0]), tangent)
    if _valid_direction(normal):
        return normal
    return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def _frame_from_mesh_normal(mesh_normal: np.ndarray, tangent: np.ndarray) -> FiberStripFrame:
    mesh_normal = _project_to_tangent_plane(mesh_normal, tangent)
    if not _valid_direction(mesh_normal):
        mesh_normal = _fallback_mesh_normal_for_tangent(tangent)
    side = _normalized_or_zero(np.cross(mesh_normal, tangent))
    if not _valid_direction(side):
        side = _side_direction(_axis_fallback_least_aligned_with(tangent), tangent)
        mesh_normal = _normalized_or_zero(np.cross(tangent, side))
    return FiberStripFrame(
        tangent_xyz=tangent.astype(np.float32),
        side_xyz=side.astype(np.float32),
        mesh_normal_xyz=mesh_normal.astype(np.float32),
    )


def _resolved_normals(points_xyz: np.ndarray, sampled_normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(sampled_normals, dtype=np.float64)
    if normals.shape == (3,):
        normals = np.repeat(normals[None, :], points_xyz.shape[0], axis=0)
    if normals.shape != points_xyz.shape:
        raise ValueError(
            f"sampled Lasagna normals must have shape (3,) or {points_xyz.shape}, got {normals.shape}"
        )
    resolved = np.zeros_like(points_xyz, dtype=np.float64)
    previous = np.zeros(3, dtype=np.float64)
    for i, normal in enumerate(normals):
        unit = _normalized_or_zero(normal)
        if not _valid_direction(unit):
            unit = previous if _valid_direction(previous) else _fallback_mesh_normal_for_tangent(_tangent_at(points_xyz, i))
        if _valid_direction(previous) and float(np.dot(unit, previous)) < 0.0:
            unit *= -1.0
        resolved[i] = unit
        previous = unit
    return resolved


def _build_side_strip_frames_for_points(points: np.ndarray, *, sampled_normals: np.ndarray) -> list[FiberStripFrame]:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("fiber points must have shape [N, 3] with N > 0")

    tangents = np.stack([_tangent_at(points, i) for i in range(points.shape[0])], axis=0)
    normals = _resolved_normals(points, sampled_normals)

    anchor = points.shape[0] // 2
    base_normals = np.zeros_like(points, dtype=np.float64)
    base_normals[anchor] = _project_to_tangent_plane(normals[anchor], tangents[anchor])
    if not _valid_direction(base_normals[anchor]):
        base_normals[anchor] = _fallback_mesh_normal_for_tangent(tangents[anchor])
    for row in range(anchor + 1, points.shape[0]):
        base_normals[row] = _transport_normal(base_normals[row - 1], tangents[row - 1], tangents[row])
    for row in range(anchor, 0, -1):
        base_normals[row - 1] = _transport_normal(base_normals[row], tangents[row], tangents[row - 1])

    roll_targets = np.zeros(points.shape[0], dtype=np.float64)

    def target_axis_angle(row: int) -> float | None:
        axis = _project_to_tangent_plane(normals[row], tangents[row])
        if not _valid_direction(axis):
            return None
        binormal = _normalized_or_zero(np.cross(tangents[row], base_normals[row]))
        if not _valid_direction(binormal):
            return None
        return float(np.arctan2(float(np.dot(axis, binormal)), float(np.dot(axis, base_normals[row]))))

    angle = target_axis_angle(anchor)
    if angle is not None:
        roll_targets[anchor] = _unwrap_axis_near(angle, 0.0)
    for row in range(anchor + 1, points.shape[0]):
        angle = target_axis_angle(row)
        roll_targets[row] = (
            _unwrap_axis_near(angle, roll_targets[row - 1]) if angle is not None else roll_targets[row - 1]
        )
    for row in range(anchor, 0, -1):
        angle = target_axis_angle(row - 1)
        roll_targets[row - 1] = (
            _unwrap_axis_near(angle, roll_targets[row]) if angle is not None else roll_targets[row]
        )

    roll_angles = _smooth_roll_angles(roll_targets)
    frames: list[FiberStripFrame | None] = [None] * points.shape[0]
    frames[anchor] = _frame_from_mesh_normal(
        _rotate_around_axis(base_normals[anchor], tangents[anchor], float(roll_angles[anchor])),
        tangents[anchor],
    )
    for row in range(anchor + 1, points.shape[0]):
        mesh_normal = _rotate_around_axis(base_normals[row], tangents[row], float(roll_angles[row]))
        prev_frame = frames[row - 1]
        assert prev_frame is not None
        transported = _transport_normal(prev_frame.mesh_normal_xyz, tangents[row - 1], tangents[row])
        if _valid_direction(transported) and float(np.dot(mesh_normal, transported)) < 0.0:
            mesh_normal *= -1.0
        frames[row] = _frame_from_mesh_normal(mesh_normal, tangents[row])
    for row in range(anchor, 0, -1):
        mesh_normal = _rotate_around_axis(base_normals[row - 1], tangents[row - 1], float(roll_angles[row - 1]))
        next_frame = frames[row]
        assert next_frame is not None
        transported = _transport_normal(next_frame.mesh_normal_xyz, tangents[row], tangents[row - 1])
        if _valid_direction(transported) and float(np.dot(mesh_normal, transported)) < 0.0:
            mesh_normal *= -1.0
        frames[row - 1] = _frame_from_mesh_normal(mesh_normal, tangents[row - 1])

    return [frame for frame in frames if frame is not None]


def build_vc3d_side_strip_frames(
    fiber: Vc3dFiber, *, sampled_normal: np.ndarray | None = None, sampled_normals: np.ndarray | None = None
) -> list[FiberStripFrame]:
    """Port of VC3D/Lasagna LineViewBuilder frame construction.

    VC3D builds side strips from fiber-line samples, transported tangents,
    sampled normals projected into the tangent plane, and smoothed roll angles.
    This Python port keeps those semantics so downstream code samples explicit
    coordinates instead of using the neural-tracing crop reader.
    """

    points = np.asarray(fiber.line_points_xyz, dtype=np.float64)
    normals = sampled_normals if sampled_normals is not None else sampled_normal
    if normals is None:
        raise ValueError("sampled_normal or sampled_normals is required")
    return _build_side_strip_frames_for_points(points, sampled_normals=normals)


def _frame_at_arc(points_xyz: np.ndarray, frames: list[FiberStripFrame], arc: float) -> FiberStripFrame:
    tangent = _cubic_hermite_tangent(points_xyz, arc)
    if points_xyz.shape[0] == 1:
        return _frame_from_mesh_normal(frames[0].mesh_normal_xyz, tangent)
    cumulative = _arc_lengths(points_xyz)
    c0, c1, t, _ = _arc_segment_indices(cumulative, np.asarray(arc, dtype=np.float64))
    i0 = int(c0)
    i1 = int(c1)
    tv = float(t)
    n0 = np.asarray(frames[i0].mesh_normal_xyz, dtype=np.float64)
    n1 = np.asarray(frames[i1].mesh_normal_xyz, dtype=np.float64)
    normal = n0 * (1.0 - tv) + n1 * tv
    return _frame_from_mesh_normal(normal, tangent)


def _interpolate_line_side_slice(
    points_xyz: np.ndarray,
    frames: list[FiberStripFrame],
    arc_coord: np.ndarray,
    normal_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_xyz.shape[0] == 1:
        center = np.broadcast_to(points_xyz[0], arc_coord.shape + (3,)).astype(np.float64)
        normal = np.broadcast_to(frames[0].mesh_normal_xyz, arc_coord.shape + (3,)).astype(np.float64)
        return center + normal * normal_offsets[..., None], normal, np.ones(arc_coord.shape, dtype=bool)

    cumulative = _arc_lengths(points_xyz)
    center, valid = _cubic_hermite_line(points_xyz, arc_coord)
    c0, c1, t, _ = _arc_segment_indices(cumulative, arc_coord)

    n0 = np.stack([frames[int(i)].mesh_normal_xyz for i in c0.reshape(-1)], axis=0).reshape(
        arc_coord.shape + (3,)
    )
    n1 = np.stack([frames[int(i)].mesh_normal_xyz for i in c1.reshape(-1)], axis=0).reshape(
        arc_coord.shape + (3,)
    )
    normal = n0 * (1.0 - t[..., None]) + n1 * t[..., None]
    normal_len = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.divide(normal, normal_len, out=np.zeros_like(normal), where=normal_len > _EPS)
    return center + normal * normal_offsets[..., None], normal, valid


def _frame_normal_array(frames: list[FiberStripFrame]) -> np.ndarray:
    return np.stack([np.asarray(frame.mesh_normal_xyz, dtype=np.float64) for frame in frames], axis=0)


def _cubic_hermite_line_torch(
    points_xyz: torch.Tensor,
    cumulative: torch.Tensor,
    derivatives: torch.Tensor,
    arc_coord: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = cumulative[-1]
    valid = (arc_coord >= 0.0) & (arc_coord <= total)
    arc = torch.clamp(arc_coord, 0.0, float(total.item()))
    c1 = torch.searchsorted(cumulative, arc.contiguous(), right=True)
    c1 = torch.clamp(c1, 1, int(cumulative.numel()) - 1)
    c0 = c1 - 1
    span = cumulative[c1] - cumulative[c0]
    t = torch.where(span > _EPS, (arc - cumulative[c0]) / span, torch.zeros_like(arc))

    p0 = points_xyz[c0]
    p1 = points_xyz[c1]
    m0 = derivatives[c0]
    m1 = derivatives[c1]
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    center = (
        h00[..., None] * p0
        + h10[..., None] * span[..., None] * m0
        + h01[..., None] * p1
        + h11[..., None] * span[..., None] * m1
    )
    return center, c0, c1, t, valid


def _interpolate_line_side_slice_torch(
    points_xyz: np.ndarray,
    frames: list[FiberStripFrame],
    arc_coord: torch.Tensor,
    normal_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = arc_coord.device
    dtype = torch.float64
    points = torch.as_tensor(points_xyz, dtype=dtype, device=device)
    if points.shape[0] == 1:
        center = points[0].view(1, 1, 3).expand(arc_coord.shape + (3,))
        normal = torch.as_tensor(frames[0].mesh_normal_xyz, dtype=dtype, device=device).view(1, 1, 3)
        normal = normal.expand(arc_coord.shape + (3,))
        return center + normal * normal_offsets[..., None], normal, torch.ones_like(arc_coord, dtype=torch.bool)

    cumulative_np = _arc_lengths(points_xyz)
    derivatives_np = _arc_derivatives(points_xyz, cumulative_np)
    cumulative = torch.as_tensor(cumulative_np, dtype=dtype, device=device)
    derivatives = torch.as_tensor(derivatives_np, dtype=dtype, device=device)
    frame_normals = torch.as_tensor(_frame_normal_array(frames), dtype=dtype, device=device)

    center, c0, c1, t, valid = _cubic_hermite_line_torch(points, cumulative, derivatives, arc_coord)
    n0 = frame_normals[c0]
    n1 = frame_normals[c1]
    normal = n0 * (1.0 - t[..., None]) + n1 * t[..., None]
    normal_len = torch.linalg.norm(normal, dim=-1, keepdim=True)
    normal = torch.where(normal_len > _EPS, normal / normal_len.clamp_min(_EPS), torch.zeros_like(normal))
    return center + normal * normal_offsets[..., None], normal, valid


def build_side_strip_patch_grid_tensor_from_line_window(
    line_window: FiberStripLineWindow,
    *,
    patch_shape_hw: tuple[int, int],
    strip_z_offset: float,
    sampled_normal: np.ndarray | None = None,
    sampled_normals: np.ndarray | None = None,
    pixel_spacing_base: float = 1.0,
    anchor_column_px: float | None = None,
    device: torch.device | str | None = None,
) -> FiberStripGridTorch:
    line_points = np.asarray(line_window.line_points_xyz, dtype=np.float64)
    height, width = (int(v) for v in patch_shape_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"patch_shape_hw must contain positive values, got {patch_shape_hw}")
    pixel_spacing_base = float(pixel_spacing_base)
    if not np.isfinite(pixel_spacing_base) or pixel_spacing_base <= 0.0:
        raise ValueError(f"pixel_spacing_base must be positive and finite, got {pixel_spacing_base}")

    normals = sampled_normals if sampled_normals is not None else sampled_normal
    if normals is None:
        raise ValueError("sampled_normal or sampled_normals is required")
    frames = _build_side_strip_frames_for_points(line_points, sampled_normals=normals)
    line_index = int(line_window.local_control_point_index)
    if line_index < 0 or line_index >= line_points.shape[0]:
        raise IndexError(
            f"local_control_point_index {line_index} out of range for {line_points.shape[0]} line points"
        )

    torch_device = torch.device("cpu") if device is None else torch.device(device)
    dtype = torch.float64
    row_offsets = (
        (torch.arange(height, dtype=dtype, device=torch_device) - (float(height) - 1.0) * 0.5)
        + float(strip_z_offset)
    ) * pixel_spacing_base
    anchor_col = (float(width) - 1.0) * 0.5 if anchor_column_px is None else float(anchor_column_px)
    col_offsets = (
        torch.arange(width, dtype=dtype, device=torch_device) - anchor_col
    ) * pixel_spacing_base
    row_grid, col_grid = torch.meshgrid(row_offsets, col_offsets, indexing="ij")
    cumulative = _arc_lengths(line_points)
    anchor_arc = float(cumulative[line_index])
    arc_coord = anchor_arc + col_grid
    coords_xyz_t, offset_axis_xyz_t, valid_t = _interpolate_line_side_slice_torch(
        line_points, frames, arc_coord, row_grid
    )
    shape_valid_t = torch.isfinite(coords_xyz_t).all(dim=-1)
    frame = _frame_at_arc(line_points, frames, anchor_arc)
    return FiberStripGridTorch(
        coords_xyz=coords_xyz_t.to(dtype=torch.float32),
        coords_zyx=coords_xyz_t[..., (2, 1, 0)].to(dtype=torch.float32),
        valid_mask=valid_t & shape_valid_t,
        frame=frame,
        offset_axis_xyz=offset_axis_xyz_t.to(dtype=torch.float32),
        offset_axis_zyx=offset_axis_xyz_t[..., (2, 1, 0)].to(dtype=torch.float32),
    )


def build_side_strip_patch_grid_from_line_window_torch(
    line_window: FiberStripLineWindow,
    *,
    patch_shape_hw: tuple[int, int],
    strip_z_offset: float,
    sampled_normal: np.ndarray | None = None,
    sampled_normals: np.ndarray | None = None,
    pixel_spacing_base: float = 1.0,
    anchor_column_px: float | None = None,
    device: torch.device | str | None = None,
) -> FiberStripGrid:
    return build_side_strip_patch_grid_tensor_from_line_window(
        line_window,
        patch_shape_hw=patch_shape_hw,
        strip_z_offset=strip_z_offset,
        sampled_normal=sampled_normal,
        sampled_normals=sampled_normals,
        pixel_spacing_base=pixel_spacing_base,
        anchor_column_px=anchor_column_px,
        device=device,
    ).to_numpy()


def build_side_strip_patch_grid_from_line_window(
    line_window: FiberStripLineWindow,
    *,
    patch_shape_hw: tuple[int, int],
    strip_z_offset: float,
    sampled_normal: np.ndarray | None = None,
    sampled_normals: np.ndarray | None = None,
    pixel_spacing_base: float = 1.0,
    anchor_column_px: float | None = None,
) -> FiberStripGrid:
    line_points = np.asarray(line_window.line_points_xyz, dtype=np.float64)
    height, width = (int(v) for v in patch_shape_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"patch_shape_hw must contain positive values, got {patch_shape_hw}")
    pixel_spacing_base = float(pixel_spacing_base)
    if not np.isfinite(pixel_spacing_base) or pixel_spacing_base <= 0.0:
        raise ValueError(f"pixel_spacing_base must be positive and finite, got {pixel_spacing_base}")

    normals = sampled_normals if sampled_normals is not None else sampled_normal
    if normals is None:
        raise ValueError("sampled_normal or sampled_normals is required")
    frames = _build_side_strip_frames_for_points(line_points, sampled_normals=normals)
    row_offsets = ((np.arange(height, dtype=np.float64) - (height - 1) * 0.5) + float(strip_z_offset)) * pixel_spacing_base
    anchor_col = (float(width) - 1.0) * 0.5 if anchor_column_px is None else float(anchor_column_px)
    col_offsets = (np.arange(width, dtype=np.float64) - anchor_col) * pixel_spacing_base
    row_grid, col_grid = np.meshgrid(row_offsets, col_offsets, indexing="ij")
    line_index = int(line_window.local_control_point_index)
    if line_index < 0 or line_index >= line_points.shape[0]:
        raise IndexError(
            f"local_control_point_index {line_index} out of range for {line_points.shape[0]} line points"
        )
    anchor_arc = float(_arc_lengths(line_points)[line_index])
    arc_coord = anchor_arc + col_grid
    coords_xyz, offset_axis_xyz, valid = _interpolate_line_side_slice(line_points, frames, arc_coord, row_grid)
    shape_valid = np.isfinite(coords_xyz).all(axis=-1)
    coords_zyx = coords_xyz[..., (2, 1, 0)].astype(np.float32)
    offset_axis_zyx = offset_axis_xyz[..., (2, 1, 0)].astype(np.float32)
    frame = _frame_at_arc(line_points, frames, anchor_arc)
    return FiberStripGrid(
        coords_xyz=coords_xyz.astype(np.float32),
        coords_zyx=coords_zyx,
        valid_mask=(valid & shape_valid),
        frame=frame,
        offset_axis_xyz=offset_axis_xyz.astype(np.float32),
        offset_axis_zyx=offset_axis_zyx,
    )


def build_side_strip_patch_grid(
    fiber: Vc3dFiber,
    *,
    control_point_index: int,
    patch_shape_hw: tuple[int, int],
    strip_z_offset: float,
    sampled_normal: np.ndarray | None = None,
    sampled_normals: np.ndarray | None = None,
    pixel_spacing_base: float = 1.0,
) -> FiberStripGrid:
    line_window = side_strip_line_window(
        fiber,
        control_point_index=control_point_index,
        patch_shape_hw=patch_shape_hw,
        pixel_spacing_base=pixel_spacing_base,
        interpolation_point_margin=0,
    )
    normals = sampled_normals
    if normals is not None:
        normals_array = np.asarray(normals)
        full_count = np.asarray(fiber.line_points_xyz).shape[0]
        if normals_array.ndim == 2 and normals_array.shape[0] == full_count:
            normals = normals_array[line_window.original_line_indices]
    return build_side_strip_patch_grid_from_line_window(
        line_window,
        patch_shape_hw=patch_shape_hw,
        strip_z_offset=strip_z_offset,
        sampled_normal=sampled_normal,
        sampled_normals=normals,
        pixel_spacing_base=pixel_spacing_base,
    )
