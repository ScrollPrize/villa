from __future__ import annotations

from dataclasses import dataclass
import numpy as np

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


def _resolved_normals(points_xyz: np.ndarray, sampled_normal: np.ndarray) -> np.ndarray:
    normal = _normalized_or_zero(sampled_normal)
    if not _valid_direction(normal):
        raise ValueError("sampled Lasagna normal is invalid")
    return np.repeat(normal[None, :], points_xyz.shape[0], axis=0)


def build_vc3d_side_strip_frames(
    fiber: Vc3dFiber, *, sampled_normal: np.ndarray
) -> list[FiberStripFrame]:
    """Port of VC3D/Lasagna LineViewBuilder frame construction.

    VC3D builds side strips from control-point samples, transported tangents,
    sampled normals projected into the tangent plane, and smoothed roll angles.
    This Python port keeps those semantics so downstream code samples explicit
    coordinates instead of using the neural-tracing crop reader.
    """

    points = np.asarray(fiber.control_points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("fiber control_points_xyz must have shape [N, 3] with N > 0")

    tangents = np.stack([_tangent_at(points, i) for i in range(points.shape[0])], axis=0)
    normals = _resolved_normals(points, sampled_normal)

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


def _interpolate_line_side_slice(
    points_xyz: np.ndarray,
    frames: list[FiberStripFrame],
    arc_coord: np.ndarray,
    normal_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if points_xyz.shape[0] == 1:
        center = np.broadcast_to(points_xyz[0], arc_coord.shape + (3,)).astype(np.float64)
        normal = np.broadcast_to(frames[0].mesh_normal_xyz, arc_coord.shape + (3,)).astype(np.float64)
        return center + normal * normal_offsets[..., None], np.ones(arc_coord.shape, dtype=bool)

    segment_lengths = np.linalg.norm(np.diff(points_xyz, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    valid = (arc_coord >= 0.0) & (arc_coord <= total)
    arc = np.clip(arc_coord, 0.0, total)
    c1 = np.searchsorted(cumulative, arc, side="right")
    c1 = np.clip(c1, 1, points_xyz.shape[0] - 1)
    c0 = c1 - 1
    span = cumulative[c1] - cumulative[c0]
    t = np.divide(
        arc - cumulative[c0],
        span,
        out=np.zeros_like(arc, dtype=np.float64),
        where=span > _EPS,
    )

    p0 = points_xyz[c0]
    p1 = points_xyz[c1]
    n0 = np.stack([frames[int(i)].mesh_normal_xyz for i in c0.reshape(-1)], axis=0).reshape(
        arc.shape + (3,)
    )
    n1 = np.stack([frames[int(i)].mesh_normal_xyz for i in c1.reshape(-1)], axis=0).reshape(
        arc.shape + (3,)
    )
    center = p0 * (1.0 - t[..., None]) + p1 * t[..., None]
    normal = n0 * (1.0 - t[..., None]) + n1 * t[..., None]
    normal_len = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.divide(normal, normal_len, out=np.zeros_like(normal), where=normal_len > _EPS)
    return center + normal * normal_offsets[..., None], valid


def build_side_strip_patch_grid(
    fiber: Vc3dFiber,
    *,
    control_point_index: int,
    patch_shape_hw: tuple[int, int],
    strip_z_offset: float,
    sampled_normal: np.ndarray,
    pixel_spacing_base: float = 1.0,
) -> FiberStripGrid:
    points = np.asarray(fiber.control_points_xyz, dtype=np.float64)
    if control_point_index < 0 or control_point_index >= points.shape[0]:
        raise IndexError(
            f"control_point_index {control_point_index} out of range for {points.shape[0]} control points"
        )
    height, width = (int(v) for v in patch_shape_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"patch_shape_hw must contain positive values, got {patch_shape_hw}")
    pixel_spacing_base = float(pixel_spacing_base)
    if not np.isfinite(pixel_spacing_base) or pixel_spacing_base <= 0.0:
        raise ValueError(f"pixel_spacing_base must be positive and finite, got {pixel_spacing_base}")

    frames = build_vc3d_side_strip_frames(fiber, sampled_normal=sampled_normal)
    row_offsets = ((np.arange(height, dtype=np.float64) - (height - 1) * 0.5) + float(strip_z_offset)) * pixel_spacing_base
    col_offsets = (np.arange(width, dtype=np.float64) - (width - 1) * 0.5) * pixel_spacing_base
    row_grid, col_grid = np.meshgrid(row_offsets, col_offsets, indexing="ij")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    arc_coord = float(cumulative[control_point_index]) + col_grid
    coords_xyz, valid = _interpolate_line_side_slice(points, frames, arc_coord, row_grid)
    shape_valid = np.isfinite(coords_xyz).all(axis=-1)
    coords_zyx = coords_xyz[..., (2, 1, 0)].astype(np.float32)
    frame = frames[control_point_index]
    return FiberStripGrid(
        coords_xyz=coords_xyz.astype(np.float32),
        coords_zyx=coords_zyx,
        valid_mask=(valid & shape_valid),
        frame=frame,
    )


def build_planar_side_strip_patch_grid(
    fiber: Vc3dFiber,
    *,
    control_point_index: int,
    patch_shape_hw: tuple[int, int],
    strip_z_offset: float,
    sampled_normal: np.ndarray,
    pixel_spacing_base: float = 1.0,
) -> FiberStripGrid:
    """Build a local planar debug slice from the same CP frame as the side strip."""

    points = np.asarray(fiber.control_points_xyz, dtype=np.float64)
    if control_point_index < 0 or control_point_index >= points.shape[0]:
        raise IndexError(
            f"control_point_index {control_point_index} out of range for {points.shape[0]} control points"
        )
    height, width = (int(v) for v in patch_shape_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"patch_shape_hw must contain positive values, got {patch_shape_hw}")
    pixel_spacing_base = float(pixel_spacing_base)
    if not np.isfinite(pixel_spacing_base) or pixel_spacing_base <= 0.0:
        raise ValueError(f"pixel_spacing_base must be positive and finite, got {pixel_spacing_base}")

    frames = build_vc3d_side_strip_frames(fiber, sampled_normal=sampled_normal)
    frame = frames[control_point_index]
    origin = points[control_point_index]
    row_offsets = ((np.arange(height, dtype=np.float64) - (height - 1) * 0.5) + float(strip_z_offset)) * pixel_spacing_base
    col_offsets = (np.arange(width, dtype=np.float64) - (width - 1) * 0.5) * pixel_spacing_base
    row_grid, col_grid = np.meshgrid(row_offsets, col_offsets, indexing="ij")
    coords_xyz = (
        origin[None, None, :]
        + row_grid[..., None] * np.asarray(frame.mesh_normal_xyz, dtype=np.float64)
        + col_grid[..., None] * np.asarray(frame.tangent_xyz, dtype=np.float64)
    )
    shape_valid = np.isfinite(coords_xyz).all(axis=-1)
    return FiberStripGrid(
        coords_xyz=coords_xyz.astype(np.float32),
        coords_zyx=coords_xyz[..., (2, 1, 0)].astype(np.float32),
        valid_mask=shape_valid,
        frame=frame,
    )
