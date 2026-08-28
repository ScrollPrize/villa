"""Export the largest approved-fiber component in a metric UV layout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from . import LayoutOptions, SpiralThetaProvider, WindingGraph
from . import fit_rigid_registration
from . import layout_largest_fiber_component
from . import refine_patch_pose_graph


@dataclass(frozen=True)
class _Pose:
    matrix: np.ndarray
    translation: np.ndarray

    @property
    def reflected(self) -> bool:
        return bool(np.linalg.det(self.matrix) < 0.0)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return points @ self.matrix.T + self.translation

    def inverse(self, points: np.ndarray) -> np.ndarray:
        return (points - self.translation) @ self.matrix


@dataclass
class _Fit:
    patch_id: str
    pose: _Pose
    rms: float
    inliers: int
    area: float
    winding_offset: int
    round: int = 0
    anchors: list[tuple[np.ndarray, float]] = field(default_factory=list)
    region_winding_offsets: dict[int, int] = field(default_factory=dict)


def _patch_color(patch_id: str) -> np.ndarray:
    digest = hashlib.blake2b(patch_id.encode(), digest_size=3).digest()
    raw = np.frombuffer(digest, np.uint8).astype(np.uint16)
    return (64 + raw * 160 // 255).astype(np.uint8)


def _manifest(cache: Path) -> dict:
    document = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if document.get("schema") != "spiral-winding-graph":
        raise ValueError("unsupported graph cache")
    return document


def _patch_specs(cache: Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for entry in _manifest(cache)["patches"]:
        path = Path(entry["path"])
        metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        scale = metadata.get("scale", [1.0, 1.0])
        if len(scale) != 2 or not all(float(value) > 0 for value in scale):
            raise ValueError(f"invalid TIFXYZ scale for {entry['id']}")
        output[entry["id"]] = {
            "path": path,
            "scale_row": float(scale[0]),
            "scale_col": float(scale[1]),
            "valid": bool(entry.get("valid", True)),
        }
    return output


def _native_metadata(
    native, provider
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    fibers = []
    zyx: list[list[float]] = []
    uv: list[list[float]] = []
    turn: list[float] = []
    valid_point_records: list[dict] = []
    for fiber in native.fibers:
        points = []
        for point in fiber.points:
            if point.theta_valid:
                zyx.append([point.z, point.y, point.x])
                uv.append([point.u, point.v])
                turn.append(point.winding + point.fractional_winding)
            points.append(
                {
                    "zyx": [point.z, point.y, point.x],
                    "uv": [point.u, point.v],
                    "theta_valid": point.theta_valid,
                    "winding": point.winding if point.theta_valid else None,
                    "checkpoint_seam_phase": (
                        point.fractional_winding if point.theta_valid else None
                    ),
                    "fractional_winding": None,
                }
            )
            if point.theta_valid:
                valid_point_records.append(points[-1])
        fibers.append(
            {
                "id": fiber.id,
                "axis": fiber.axis,
                "logical_track": fiber.logical_track,
                "reversed": fiber.reversed,
                "arclength": fiber.arclength,
                "winding_offset": fiber.winding_offset,
                "points": points,
            }
        )
    zyx_array = np.asarray(zyx, dtype=np.float32)
    if len(zyx_array):
        if not hasattr(provider, "geometric_theta"):
            raise TypeError("theta provider must define geometric_theta(zyx)")
        actual_theta = np.asarray(
            provider.geometric_theta(np.ascontiguousarray(zyx_array)),
            dtype=np.float64,
        )
        if actual_theta.shape != (len(zyx_array),):
            raise ValueError("geometric theta provider returned the wrong shape")
        actual_fraction = np.mod(actual_theta / (2.0 * np.pi), 1.0)
        for record, fraction in zip(
            valid_point_records, actual_fraction, strict=True
        ):
            record["fractional_winding"] = float(fraction)
    metadata = {
        "root_fiber": native.root_fiber,
        "fiber_count": len(fibers),
        "excluded_fibers": list(native.excluded_fibers),
        "theta_covered_points": native.theta_covered_points,
        "theta_uncovered_points": native.theta_uncovered_points,
        "total_arclength": native.total_arclength,
        "solver": {
            "initial_cost": native.initial_cost,
            "final_cost": native.final_cost,
            "iterations": native.solver_iterations,
        },
        "crossings": [
            {
                "first_fiber": knot.first_fiber,
                "first_point": knot.first_point,
                "second_fiber": knot.second_fiber,
                "second_point": knot.second_point,
                "u_residual": knot.u_residual,
                "v_residual": knot.v_residual,
            }
            for knot in native.crossings
        ],
        "fibers": fibers,
    }
    return (
        metadata,
        zyx_array,
        np.asarray(uv, dtype=np.float64),
        np.asarray(turn, dtype=np.float64),
    )


def _metric_contact(hit, spec: dict) -> np.ndarray:
    return np.array(
        [hit.column / spec["scale_col"], hit.row / spec["scale_row"]],
        dtype=np.float64,
    )


def _spatially_distinct(points: np.ndarray, tolerance: float) -> int:
    if not len(points):
        return 0
    cells = np.floor(points / max(tolerance, 1e-6)).astype(np.int64)
    return len({(int(a), int(b)) for a, b in cells})


def _tiff_shape(path: Path) -> tuple[int, int]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        shape = tif.pages[0].shape
    return int(shape[0]), int(shape[1])


def _fit_patch(
    patch_id: str,
    correspondences: list[tuple[np.ndarray, np.ndarray, float]],
    spec: dict,
    options: LayoutOptions,
) -> tuple[_Fit | None, str]:
    if len(correspondences) < options.min_inliers:
        return None, "too_few_contacts"
    correspondences.sort(
        key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1])
    )
    source = np.stack([item[0] for item in correspondences])
    target = np.stack([item[1] for item in correspondences])
    target_turn = np.asarray([item[2] for item in correspondences])
    native = fit_rigid_registration(
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(target, dtype=np.float64),
        options,
    )
    if not native.accepted:
        return None, native.rejection
    pose = _Pose(
        np.array([[native.r00, native.r01], [native.r10, native.r11]]),
        np.array([native.translation_u, native.translation_v]),
    )
    error = np.linalg.norm(pose.apply(source) - target, axis=1)
    included = error <= options.uv_ransac_tolerance
    rms = native.rms
    rows, columns = _tiff_shape(spec["path"] / "x.tif")
    area = (rows - 1) * (columns - 1) / (
        spec["scale_row"] * spec["scale_col"]
    )
    return (
        _Fit(
            patch_id,
            pose,
            rms,
            native.inliers,
            area,
            0,
            anchors=[
                (source[index].copy(), float(target_turn[index]))
                for index in np.nonzero(included)[0]
            ],
        ),
        "accepted",
    )


def _initial_correspondences(graph, zyx, uv, turn, specs, tolerance):
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = defaultdict(list)
    for point_uv, point_turn, hits in zip(
        uv, turn, graph.inspect_contacts(zyx, tolerance), strict=True
    ):
        for hit in hits:
            spec = specs.get(hit.patch_id)
            if spec is not None and spec["valid"]:
                grouped[hit.patch_id].append(
                    (_metric_contact(hit, spec), point_uv.copy(), float(point_turn))
                )
    return grouped


def _patch_vertices(graph, patch_id: str, spec: dict):
    layout = graph.patch_layout(patch_id)
    ij = np.asarray(layout["vertex_ij"], dtype=np.float64)
    local = np.column_stack(
        (ij[:, 1] / spec["scale_col"], ij[:, 0] / spec["scale_row"])
    )
    return local, np.asarray(layout["vertex_zyx"], dtype=np.float32)


def _checkpoint_valid(provider, xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = valid.copy()
    if hasattr(provider, "z_begin") and hasattr(provider, "z_end"):
        output &= xyz[..., 0] >= float(provider.z_begin)
        output &= xyz[..., 0] < float(provider.z_end)
    return output


def _patch_turn_field(provider, spec: dict):
    xyz, valid = _read_patch(spec)
    valid = _checkpoint_valid(provider, xyz, valid)
    seam_theta = np.zeros(valid.shape, dtype=np.float32)
    points = np.ascontiguousarray(xyz[valid], dtype=np.float32)
    seam_theta[valid] = provider(points)
    local_turn, regions = _unwrap_turn(seam_theta, valid)
    return xyz, valid, local_turn, regions


def _resolve_patch_winding(
    fit: _Fit,
    provider,
    spec: dict,
    min_inliers: int,
):
    xyz, valid, local_turn, regions = _patch_turn_field(provider, spec)
    local = np.stack([anchor[0] for anchor in fit.anchors])
    row = local[:, 1] * spec["scale_row"]
    column = local[:, 0] * spec["scale_col"]
    _, sampled_turn, supported = _sample_patch(
        xyz, local_turn, valid, row, column
    )
    nearest_row = np.clip(np.rint(row).astype(np.int64), 0, valid.shape[0] - 1)
    nearest_column = np.clip(
        np.rint(column).astype(np.int64), 0, valid.shape[1] - 1
    )
    anchor_regions = regions[nearest_row, nearest_column]
    supported &= anchor_regions >= 0
    anchor_turn = np.asarray([anchor[1] for anchor in fit.anchors])
    resolved: dict[int, int] = {}
    support_counts: dict[int, int] = {}
    saw_phase_failure = False
    saw_ambiguity = False
    saw_disagreement = False
    for region in sorted(set(map(int, anchor_regions[supported]))):
        selected = supported & (anchor_regions == region)
        delta = anchor_turn[selected] - sampled_turn[selected]
        votes = np.rint(delta).astype(np.int64)
        if len(votes) < min_inliers:
            continue
        phase_inlier = np.abs(delta - votes) <= 0.25
        if np.count_nonzero(phase_inlier) < min_inliers:
            saw_phase_failure = True
            continue
        votes = votes[phase_inlier]
        counts = Counter(map(int, votes))
        ordered = counts.most_common()
        if ordered[0][1] < min_inliers or ordered[0][1] * 2 <= len(votes):
            saw_ambiguity = True
            continue
        if len(ordered) > 1:
            saw_disagreement = True
            continue
        resolved[region] = ordered[0][0]
        support_counts[region] = len(votes)
    # Each disconnected valid grid region has an independent unwrap gauge.
    # Only regions with a full inlier consensus may enter the established
    # layout; otherwise unrelated mask islands inherit an arbitrary integer.
    if not resolved:
        if saw_disagreement:
            return None, "overlap_winding_disagreement", None
        if saw_ambiguity:
            return None, "ambiguous_winding", None
        if saw_phase_failure:
            return None, "winding_phase_gate", None
        return None, "winding_inlier_gate", None
    fit.region_winding_offsets = resolved
    primary_region = min(
        resolved, key=lambda region: (-support_counts[region], region)
    )
    fit.winding_offset = resolved[primary_region]
    resolved_valid = np.zeros_like(valid)
    resolved_turn = local_turn.copy()
    for region, offset in resolved.items():
        selected = regions == region
        resolved_valid |= selected
        resolved_turn[selected] += offset
    resolved_rows, resolved_columns = np.nonzero(resolved_valid)
    physical = np.column_stack(
        (
            resolved_columns / spec["scale_col"],
            resolved_rows / spec["scale_row"],
        )
    )
    global_uv = fit.pose.apply(physical)
    turns = resolved_turn[resolved_valid]
    if len(turns) > 1 and np.cov(global_uv[:, 0], turns)[0, 1] <= 0:
        return None, "checkpoint_turn_decreases_with_u", None
    if (
        len(turns) > 1
        and np.cov(global_uv[:, 1], xyz[..., 0][resolved_valid])[0, 1] < 0
    ):
        return None, "physical_z_decreases_with_v", None
    return fit, "accepted", (xyz, resolved_valid, resolved_turn)


def _grow_patches(graph, specs, initial, options: LayoutOptions, provider):
    placed: dict[str, _Fit] = {}
    rejected: dict[str, str] = {}
    relative_contacts: dict[
        tuple[str, str], list[tuple[np.ndarray, np.ndarray, float]]
    ] = defaultdict(list)
    # Registration evidence is component-global and monotonic.  A candidate
    # can touch several established patches in different growth rounds; if we
    # retain only the most recent frontier, a later local fit can forget an
    # earlier constraint and close a globally inconsistent cycle.
    reservoir: dict[
        str, list[tuple[np.ndarray, np.ndarray, float]]
    ] = defaultdict(list)
    for patch_id, values in initial.items():
        reservoir[patch_id].extend(values)
    dirty = set(reservoir)
    acceptance_round = 0
    while dirty:
        candidates = []
        work = sorted(
            (patch, reservoir[patch])
            for patch in dirty
            if patch not in placed and patch in specs
        )
        dirty.clear()
        workers = options.workers if options.workers > 0 else None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_fit_patch, patch, list(values), specs[patch], options)
                for patch, values in work
            ]
            for (patch, _), future in zip(work, futures, strict=True):
                fit, reason = future.result()
                if fit is None:
                    rejected[patch] = reason
                else:
                    candidates.append(fit)
        resolved = []
        turn_fields = {}
        for candidate in candidates:
            patch_id = candidate.patch_id
            fit, reason, turn_field = _resolve_patch_winding(
                candidate, provider, specs[patch_id], options.min_inliers
            )
            if fit is None:
                rejected[patch_id] = reason
                continue
            resolved.append(fit)
            turn_fields[fit.patch_id] = turn_field
        resolved.sort(key=lambda fit: (fit.rms, -fit.area, fit.patch_id))
        candidates = []
        deferred: set[str] = set()
        committed_bounds = []
        for fit in resolved:
            local, physical = _patch_vertices(
                graph, fit.patch_id, specs[fit.patch_id]
            )
            footprint = fit.pose.apply(local)
            uv_bounds = (footprint.min(axis=0), footprint.max(axis=0))
            xyz_bounds = (physical.min(axis=0), physical.max(axis=0))
            collision = False
            for other_uv, other_xyz in committed_bounds:
                uv_intersects = np.all(uv_bounds[0] <= other_uv[1]) and np.all(
                    other_uv[0] <= uv_bounds[1]
                )
                xyz_intersects = np.all(
                    xyz_bounds[0] <= other_xyz[1] + options.contact_tolerance
                ) and np.all(
                    other_xyz[0] <= xyz_bounds[1] + options.contact_tolerance
                )
                if uv_intersects or xyz_intersects:
                    collision = True
                    break
            if collision:
                deferred.add(fit.patch_id)
            else:
                candidates.append(fit)
                committed_bounds.append((uv_bounds, xyz_bounds))
        if not candidates:
            break
        acceptance_round += 1
        for fit in candidates:
            fit.round = acceptance_round
            placed[fit.patch_id] = fit
            rejected.pop(fit.patch_id, None)

        dirty.update(deferred)
        for fit in candidates:
            local, zyx = _patch_vertices(graph, fit.patch_id, specs[fit.patch_id])
            global_uv = fit.pose.apply(local)
            ij = np.asarray(graph.patch_layout(fit.patch_id)["vertex_ij"], dtype=np.int64)
            global_turn = turn_fields[fit.patch_id][2][ij[:, 0], ij[:, 1]]
            supported = turn_fields[fit.patch_id][1][ij[:, 0], ij[:, 1]]
            for point_local, point_uv, point_turn, hits in zip(
                local[supported],
                global_uv[supported],
                global_turn[supported],
                graph.inspect_contacts(zyx[supported], options.contact_tolerance),
                strict=True,
            ):
                for hit in hits:
                    if hit.patch_id in placed or hit.patch_id == fit.patch_id:
                        continue
                    target_spec = specs.get(hit.patch_id)
                    if target_spec is None or not target_spec["valid"]:
                        continue
                    target_local = _metric_contact(hit, target_spec)
                    relative_contacts[(fit.patch_id, hit.patch_id)].append(
                        (point_local.copy(), target_local.copy(), float(point_turn))
                    )
                    reservoir[hit.patch_id].append(
                        (
                            target_local,
                            point_uv.copy(),
                            float(point_turn),
                        )
                    )
                    dirty.add(hit.patch_id)
    for patch_id, spec in specs.items():
        if spec["valid"] and patch_id not in placed:
            rejected.setdefault(patch_id, "unreachable")
    return placed, rejected, relative_contacts


def _distinct_reservoir(
    source: np.ndarray,
    target: np.ndarray,
    tolerance: float,
    maximum: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((target[:, 1], target[:, 0], source[:, 1], source[:, 0]))
    cells: set[tuple[int, int]] = set()
    selected = []
    for index in order:
        cell = tuple(np.floor(source[index] / tolerance).astype(np.int64))
        if cell in cells:
            continue
        cells.add(cell)
        selected.append(int(index))
    if len(selected) > maximum:
        positions = np.linspace(0, len(selected) - 1, maximum).astype(np.int64)
        selected = [selected[index] for index in positions]
    indices = np.asarray(selected, dtype=np.int64)
    return source[indices], target[indices]


def _refine_pose_graph(
    placed: dict[str, _Fit],
    initial,
    relative_contacts,
    options: LayoutOptions,
    provider,
    specs,
):
    patch_ids = sorted(placed)
    patch_index = {patch_id: index for index, patch_id in enumerate(patch_ids)}
    poses = np.zeros((len(patch_ids), 2, 3), dtype=np.float64)
    for patch_id, index in patch_index.items():
        poses[index, :, :2] = placed[patch_id].pose.matrix
        poses[index, :, 2] = placed[patch_id].pose.translation

    absolute_patch = []
    absolute_local = []
    absolute_target = []
    for patch_id in patch_ids:
        values = initial.get(patch_id, ())
        if not values:
            continue
        source = np.stack([item[0] for item in values])
        target = np.stack([item[1] for item in values])
        error = np.linalg.norm(placed[patch_id].pose.apply(source) - target, axis=1)
        included = error <= options.uv_ransac_tolerance
        if np.count_nonzero(included) < options.min_inliers:
            continue
        source, target = _distinct_reservoir(
            source[included], target[included], options.uv_ransac_tolerance
        )
        absolute_patch.extend([patch_index[patch_id]] * len(source))
        absolute_local.extend(source)
        absolute_target.extend(target)

    relative_patch = []
    relative_first = []
    relative_second = []
    accepted_edges = 0
    rejected_edges = 0
    winding_rejected_edges = 0
    turn_fields = {}
    edge_records = []
    for (first_id, second_id), values in sorted(relative_contacts.items()):
        if first_id not in placed or second_id not in placed:
            continue
        first = np.stack([item[0] for item in values])
        second = np.stack([item[1] for item in values])
        first_turn = np.asarray([item[2] for item in values])
        if second_id not in turn_fields:
            xyz, valid, local_turn, regions = _patch_turn_field(
                provider, specs[second_id]
            )
            resolved_valid = np.zeros_like(valid)
            for region, offset in placed[second_id].region_winding_offsets.items():
                selected = regions == region
                resolved_valid |= selected
                local_turn[selected] += offset
            turn_fields[second_id] = (xyz, resolved_valid, local_turn)
        xyz, valid, second_turn_field = turn_fields[second_id]
        row = second[:, 1] * specs[second_id]["scale_row"]
        column = second[:, 0] * specs[second_id]["scale_col"]
        _, second_turn, supported = _sample_patch(
            xyz, second_turn_field, valid, row, column
        )
        turn_agrees = supported & (np.abs(first_turn - second_turn) <= 0.25)
        if np.count_nonzero(turn_agrees) < options.min_inliers:
            winding_rejected_edges += 1
            continue
        first = first[turn_agrees]
        second = second[turn_agrees]
        native = fit_rigid_registration(
            np.ascontiguousarray(first, dtype=np.float64),
            np.ascontiguousarray(second, dtype=np.float64),
            options,
        )
        if not native.accepted:
            rejected_edges += 1
            continue
        matrix = np.array(
            [[native.r00, native.r01], [native.r10, native.r11]]
        )
        expected_reflection = (
            placed[first_id].pose.reflected != placed[second_id].pose.reflected
        )
        if bool(np.linalg.det(matrix) < 0) != expected_reflection:
            rejected_edges += 1
            continue
        translation = np.array(
            [native.translation_u, native.translation_v], dtype=np.float64
        )
        error = np.linalg.norm(first @ matrix.T + translation - second, axis=1)
        included = error <= options.uv_ransac_tolerance
        if np.count_nonzero(included) < options.min_inliers:
            rejected_edges += 1
            continue
        first, second = _distinct_reservoir(
            first[included], second[included], options.uv_ransac_tolerance
        )
        relative_patch.extend(
            [(patch_index[first_id], patch_index[second_id])] * len(first)
        )
        relative_first.extend(first)
        relative_second.extend(second)
        edge_records.append((first_id, second_id, first, second))
        accepted_edges += 1

    if not relative_patch and len(patch_ids) == 1:
        return {
            "absolute_constraints": len(absolute_patch),
            "relative_constraints": 0,
            "accepted_edges": 0,
            "rejected_edges": rejected_edges,
            "winding_rejected_edges": winding_rejected_edges,
            "initial_cost": 0.0,
            "final_cost": 0.0,
            "iterations": 0,
        }
    if len(absolute_patch) < options.min_inliers or not relative_patch:
        raise RuntimeError("patch pose graph lacks fiber anchors or overlap edges")
    absolute_patch_array = np.asarray(absolute_patch, dtype=np.int64)
    absolute_local_array = np.asarray(absolute_local, dtype=np.float64).reshape(-1, 2)
    absolute_target_array = np.asarray(absolute_target, dtype=np.float64).reshape(-1, 2)
    relative_patch_array = np.asarray(relative_patch, dtype=np.int64).reshape(-1, 2)
    relative_first_array = np.asarray(relative_first, dtype=np.float64).reshape(-1, 2)
    relative_second_array = np.asarray(relative_second, dtype=np.float64).reshape(-1, 2)
    result = refine_patch_pose_graph(
        np.ascontiguousarray(poses),
        np.ascontiguousarray(absolute_patch_array),
        np.ascontiguousarray(absolute_local_array),
        np.ascontiguousarray(absolute_target_array),
        np.ascontiguousarray(relative_patch_array),
        np.ascontiguousarray(relative_first_array),
        np.ascontiguousarray(relative_second_array),
        options,
    )
    if not result.usable or len(result.poses) != len(patch_ids):
        raise RuntimeError("global patch pose graph refinement failed")

    solved = {
        patch_id: _Pose(
            np.array([[pose.r00, pose.r01], [pose.r10, pose.r11]]),
            np.array([pose.translation_u, pose.translation_v]),
        )
        for patch_id, pose in zip(patch_ids, result.poses, strict=True)
    }
    good_edges = []
    postsolve_rejected_edges = 0
    for first_id, second_id, first, second in edge_records:
        error = np.linalg.norm(
            solved[first_id].apply(first) - solved[second_id].apply(second), axis=1
        )
        included = error <= options.uv_ransac_tolerance
        distinct = _spatially_distinct(
            first[included], options.uv_ransac_tolerance
        )
        rms = (
            float(np.sqrt(np.mean(np.square(error[included]))))
            if np.any(included)
            else math.inf
        )
        if distinct < options.min_inliers or rms > options.max_refit_rms:
            postsolve_rejected_edges += 1
            continue
        good_edges.append(
            (first_id, second_id, first[included], second[included])
        )

    anchored = {patch_ids[int(index)] for index in absolute_patch_array}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for first_id, second_id, _, _ in good_edges:
        adjacency[first_id].add(second_id)
        adjacency[second_id].add(first_id)
    retained = set(anchored)
    queue = deque(sorted(anchored))
    while queue:
        patch_id = queue.popleft()
        for other in sorted(adjacency[patch_id]):
            if other not in retained:
                retained.add(other)
                queue.append(other)
    quarantined = sorted(set(patch_ids) - retained)

    if retained:
        retained_ids = sorted(retained)
        retained_index = {
            patch_id: index for index, patch_id in enumerate(retained_ids)
        }
        retained_poses = np.zeros((len(retained_ids), 2, 3), dtype=np.float64)
        for patch_id, index in retained_index.items():
            retained_poses[index, :, :2] = solved[patch_id].matrix
            retained_poses[index, :, 2] = solved[patch_id].translation
        keep_absolute = np.array(
            [patch_ids[int(index)] in retained for index in absolute_patch_array]
        )
        final_absolute_patch = np.asarray(
            [
                retained_index[patch_ids[int(index)]]
                for index in absolute_patch_array[keep_absolute]
            ],
            dtype=np.int64,
        )
        final_relative_patch = []
        final_relative_first = []
        final_relative_second = []
        for first_id, second_id, first, second in good_edges:
            if first_id not in retained or second_id not in retained:
                continue
            first, second = _distinct_reservoir(
                first, second, options.uv_ransac_tolerance
            )
            final_relative_patch.extend(
                [(retained_index[first_id], retained_index[second_id])] * len(first)
            )
            final_relative_first.extend(first)
            final_relative_second.extend(second)
        final_result = refine_patch_pose_graph(
            np.ascontiguousarray(retained_poses),
            np.ascontiguousarray(final_absolute_patch),
            np.ascontiguousarray(absolute_local_array[keep_absolute]),
            np.ascontiguousarray(absolute_target_array[keep_absolute]),
            np.ascontiguousarray(final_relative_patch, dtype=np.int64).reshape(-1, 2),
            np.ascontiguousarray(final_relative_first, dtype=np.float64).reshape(-1, 2),
            np.ascontiguousarray(final_relative_second, dtype=np.float64).reshape(-1, 2),
            options,
        )
        if not final_result.usable or len(final_result.poses) != len(retained_ids):
            raise RuntimeError("quarantined patch pose graph refinement failed")
        for patch_id, pose in zip(retained_ids, final_result.poses, strict=True):
            placed[patch_id].pose = _Pose(
                np.array([[pose.r00, pose.r01], [pose.r10, pose.r11]]),
                np.array([pose.translation_u, pose.translation_v]),
            )
    else:
        raise RuntimeError("patch pose graph quarantine removed every patch")

    return {
        "absolute_constraints": len(absolute_patch),
        "relative_constraints": len(relative_patch),
        "accepted_edges": accepted_edges,
        "rejected_edges": rejected_edges,
        "winding_rejected_edges": winding_rejected_edges,
        "postsolve_rejected_edges": postsolve_rejected_edges,
        "quarantined_patches": quarantined,
        "initial_cost": result.initial_cost,
        "final_cost": final_result.final_cost,
        "iterations": result.iterations + final_result.iterations,
    }


def _read_patch(spec: dict):
    import tifffile

    def read_tiff(path: Path) -> np.ndarray:
        try:
            return tifffile.imread(path)
        except ValueError as error:
            if "imagecodecs" not in str(error):
                raise
            import cv2

            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"OpenCV could not decode {path}") from error
            return image

    path = spec["path"]
    xyz = np.stack(
        [read_tiff(path / f"{axis}.tif") for axis in "zyx"], axis=-1
    ).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz >= 0).any(axis=-1)
    if (path / "mask.tif").is_file():
        valid &= read_tiff(path / "mask.tif") != 0
    return xyz, valid


def _unwrap_turn(
    theta: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output = np.full(theta.shape, np.nan, dtype=np.float64)
    regions = np.full(theta.shape, -1, dtype=np.int32)
    region = 0
    for start in zip(*np.nonzero(valid), strict=True):
        if np.isfinite(output[start]):
            continue
        output[start] = (float(theta[start]) / (2.0 * np.pi)) % 1.0
        regions[start] = region
        queue = deque([start])
        while queue:
            row, column = queue.popleft()
            for r, c in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if r < 0 or c < 0 or r >= theta.shape[0] or c >= theta.shape[1]:
                    continue
                if not valid[r, c] or np.isfinite(output[r, c]):
                    continue
                delta = (
                    float(theta[r, c])
                    - float(theta[row, column])
                    + np.pi
                ) % (2 * np.pi) - np.pi
                output[r, c] = output[row, column] + delta / (2.0 * np.pi)
                regions[r, c] = region
                queue.append((r, c))
        region += 1
    return output, regions


def _sample_patch(xyz, turn, valid, row, column):
    rows, columns = valid.shape
    inside = (
        (row >= 0)
        & (column >= 0)
        & (row <= rows - 1)
        & (column <= columns - 1)
    )
    row = np.clip(row, 0, rows - 1)
    column = np.clip(column, 0, columns - 1)
    row = np.where(np.abs(row - np.rint(row)) < 1e-8, np.rint(row), row)
    column = np.where(
        np.abs(column - np.rint(column)) < 1e-8, np.rint(column), column
    )
    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(column).astype(np.int64)
    r1 = np.ceil(row).astype(np.int64)
    c1 = np.ceil(column).astype(np.int64)
    fr = row - r0
    fc = column - c0
    supported = (
        inside
        & valid[r0, c0]
        & valid[r0, c1]
        & valid[r1, c0]
        & valid[r1, c1]
    )
    weights = np.stack(
        ((1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc),
        axis=-1,
    )
    corners_xyz = np.stack(
        (xyz[r0, c0], xyz[r0, c1], xyz[r1, c0], xyz[r1, c1]), axis=-2
    )
    corners_turn = np.stack(
        (turn[r0, c0], turn[r0, c1], turn[r1, c0], turn[r1, c1]), axis=-1
    )
    sampled_xyz = np.sum(corners_xyz * weights[..., None], axis=-2)
    sampled_turn = np.sum(corners_turn * weights, axis=-1)
    sampled_xyz[~supported] = np.nan
    sampled_turn[~supported] = np.nan
    return sampled_xyz, sampled_turn, supported


def _supported_vertex_mask(occupied: np.ndarray, supported: np.ndarray) -> np.ndarray:
    output = occupied.copy()
    while output.shape[0] > 1 and output.shape[1] > 1:
        quads = (
            output[:-1, :-1]
            & output[1:, :-1]
            & output[:-1, 1:]
            & output[1:, 1:]
        )
        unsupported = quads & ~supported
        if not unsupported.any():
            break
        supported_now = quads & supported
        cost = np.zeros_like(output, dtype=np.uint8)
        cost[:-1, :-1] += supported_now
        cost[1:, :-1] += supported_now
        cost[:-1, 1:] += supported_now
        cost[1:, 1:] += supported_now
        rows, columns = np.nonzero(unsupported)
        corner_cost = np.stack(
            (
                cost[rows, columns],
                cost[rows + 1, columns],
                cost[rows, columns + 1],
                cost[rows + 1, columns + 1],
            ),
            axis=1,
        )
        choice = np.argmin(corner_cost, axis=1)
        output[
            rows + ((choice == 1) | (choice == 3)),
            columns + ((choice == 2) | (choice == 3)),
        ] = False
    if output.shape[0] > 1 and output.shape[1] > 1:
        quads = (
            output[:-1, :-1]
            & output[1:, :-1]
            & output[:-1, 1:]
            & output[1:, 1:]
            & supported
        )
        incident = np.zeros_like(output)
        incident[:-1, :-1] |= quads
        incident[1:, :-1] |= quads
        incident[:-1, 1:] |= quads
        incident[1:, 1:] |= quads
        output &= incident
    else:
        output[:] = False
    return output


def _rasterize(
    provider, placed, specs, spacing, maximum, contact_tolerance=2.0
):
    prepared = []
    u_min = v_min = math.inf
    u_max = v_max = -math.inf
    ordered_placed = sorted(
        placed.items(),
        key=lambda item: (
            item[1].rms,
            -item[1].inliers,
            -item[1].area,
            item[0],
        ),
    )
    for patch_id, fit in ordered_placed:
        xyz, valid, local_turn, regions = _patch_turn_field(
            provider, specs[patch_id]
        )
        if fit.region_winding_offsets:
            resolved_valid = np.zeros_like(valid)
            for region, offset in fit.region_winding_offsets.items():
                selected = regions == region
                resolved_valid |= selected
                local_turn[selected] += offset
            valid &= resolved_valid
        else:
            local_turn += fit.winding_offset
        row, column = np.nonzero(valid)
        local = np.column_stack(
            (
                column / specs[patch_id]["scale_col"],
                row / specs[patch_id]["scale_row"],
            )
        )
        footprint = fit.pose.apply(local)
        u_min = min(u_min, float(footprint[:, 0].min()))
        u_max = max(u_max, float(footprint[:, 0].max()))
        v_min = min(v_min, float(footprint[:, 1].min()))
        v_max = max(v_max, float(footprint[:, 1].max()))
        prepared.append((patch_id, fit, xyz, valid, local_turn))
    width = max(1, int(math.ceil((u_max - u_min) / spacing)) + 1)
    height = max(1, int(math.ceil((v_max - v_min) / spacing)) + 1)
    if width * height > maximum:
        raise RuntimeError(
            f"layout would contain {width * height:,} samples, exceeding "
            f"the {maximum:,} maximum; increase --spacing"
        )
    u = u_min + np.arange(width) * spacing
    v = v_max - np.arange(height) * spacing
    grid_u, grid_v = np.meshgrid(u, v)
    sample_uv = np.stack((grid_u, grid_v), axis=-1)
    xyz_sum = np.zeros((height, width, 3), dtype=np.float64)
    turn_sum = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.uint16)
    labels = np.full((height, width), -1, dtype=np.int32)
    conflict = np.zeros((height, width), dtype=bool)
    source_supported_quads = np.zeros(
        (max(0, height - 1), max(0, width - 1)), bool
    )
    patch_records = []
    overlap_conflicts = []
    for patch_index, (patch_id, fit, xyz, valid, local_turn) in enumerate(
        prepared
    ):
        local = fit.pose.inverse(sample_uv)
        row = local[..., 1] * specs[patch_id]["scale_row"]
        column = local[..., 0] * specs[patch_id]["scale_col"]
        contribution_xyz, contribution_turn, supported = _sample_patch(
            xyz, local_turn, valid, row, column
        )
        if height > 1 and width > 1:
            source_supported_quads |= (
                supported[:-1, :-1]
                & supported[1:, :-1]
                & supported[:-1, 1:]
                & supported[1:, 1:]
            )
        existing = count > 0
        mean_xyz = np.zeros_like(xyz_sum)
        mean_xyz[existing] = xyz_sum[existing] / count[existing, None]
        existing_winding = np.zeros_like(labels)
        existing_winding[existing] = np.floor(
            turn_sum[existing] / count[existing]
        ).astype(np.int32)
        contribution_winding = np.full(labels.shape, np.iinfo(np.int32).min)
        contribution_winding[supported] = np.floor(
            contribution_turn[supported]
        ).astype(np.int32)
        xyz_disagreement = supported & existing & (
            np.linalg.norm(contribution_xyz - mean_xyz, axis=-1)
            > contact_tolerance
        )
        winding_disagreement = supported & existing & (
            contribution_winding != existing_winding
        )
        disagreement = xyz_disagreement | winding_disagreement
        overlap = supported & existing
        agreeing_overlap = overlap & ~disagreement
        if disagreement.any():
            rows, columns = np.nonzero(disagreement)
            overlap_conflicts.append(
                {
                    "patch_id": patch_id,
                    "samples": int(len(rows)),
                    "xyz_samples": int(np.count_nonzero(xyz_disagreement)),
                    "winding_samples": int(
                        np.count_nonzero(winding_disagreement)
                    ),
                    "first_sample": [int(rows[0]), int(columns[0])],
                }
            )
        conflict |= disagreement
        accept = supported & ~disagreement & ~conflict
        xyz_sum[accept] += contribution_xyz[accept]
        turn_sum[accept] += contribution_turn[accept]
        count[accept] += 1
        labels[accept & (labels < 0)] = patch_index
        patch_records.append(
            {
                "patch_id": patch_id,
                "source_path": str(specs[patch_id]["path"]),
                "matrix": fit.pose.matrix.tolist(),
                "translation": fit.pose.translation.tolist(),
                "reflected": fit.pose.reflected,
                "rms": fit.rms,
                "inliers": fit.inliers,
                "physical_area": fit.area,
                "winding_offset": fit.winding_offset,
                "region_winding_offsets": [
                    {"region": region, "offset": offset}
                    for region, offset in sorted(
                        fit.region_winding_offsets.items()
                    )
                ],
                "acceptance_round": fit.round,
                "supported_samples": int(np.count_nonzero(supported)),
                "overlap_samples": int(np.count_nonzero(overlap)),
                "agreeing_overlap_samples": int(
                    np.count_nonzero(agreeing_overlap)
                ),
                "conflict_samples": int(np.count_nonzero(disagreement)),
            }
        )
    count[conflict] = 0
    labels[conflict] = -1
    occupied = _supported_vertex_mask(count > 0, source_supported_quads)
    count[~occupied] = 0
    labels[~occupied] = -1
    output_xyz = np.full((height, width, 3), -1.0, dtype=np.float32)
    output_turn = np.full((height, width), np.nan, dtype=np.float64)
    output_xyz[occupied] = (xyz_sum[occupied] / count[occupied, None]).astype(
        np.float32
    )
    output_turn[occupied] = turn_sum[occupied] / count[occupied]
    valid_quads = (
        occupied[:-1, :-1]
        & occupied[1:, :-1]
        & occupied[:-1, 1:]
        & occupied[1:, 1:]
        & source_supported_quads
        if height > 1 and width > 1
        else np.zeros((0, 0), bool)
    )
    winding = np.full(
        (height, width), np.iinfo(np.int32).min, dtype=np.int32
    )
    fractional = np.full((height, width), np.nan, dtype=np.float32)
    winding[occupied] = np.floor(output_turn[occupied]).astype(np.int32)
    if occupied.any():
        if not hasattr(provider, "geometric_theta"):
            raise TypeError("theta provider must define geometric_theta(zyx)")
        actual_theta = np.asarray(
            provider.geometric_theta(
                np.ascontiguousarray(output_xyz[occupied], dtype=np.float32)
            ),
            dtype=np.float64,
        )
        if actual_theta.shape != (int(occupied.sum()),):
            raise ValueError("geometric theta provider returned the wrong shape")
        fractional[occupied] = np.mod(
            actual_theta / (2.0 * np.pi), 1.0
        ).astype(np.float32)
    distances = []
    for a, b, both in (
        (
            output_xyz[:, :-1],
            output_xyz[:, 1:],
            occupied[:, :-1] & occupied[:, 1:],
        ),
        (
            output_xyz[:-1],
            output_xyz[1:],
            occupied[:-1] & occupied[1:],
        ),
    ):
        if both.any():
            distances.append(np.linalg.norm(a[both] - b[both], axis=1))
    edge = np.concatenate(distances) if distances else np.empty(0)
    percentiles = {
        str(value): float(np.percentile(edge, value)) if len(edge) else None
        for value in (5, 25, 50, 75, 95)
    }
    return {
        "xyz": output_xyz,
        "labels": labels,
        "winding": winding,
        "fractional": fractional,
        "occupied": occupied,
        "valid_quads": valid_quads,
        "patches": patch_records,
        "conflicts": overlap_conflicts,
        "bounds": {
            "u_min": u_min,
            "u_max": u_max,
            "v_min": v_min,
            "v_max": v_max,
        },
        "edge_percentiles": percentiles,
    }


def _write_preview(path: Path, labels: np.ndarray, patches: list[dict]) -> None:
    from PIL import Image

    rgb = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    for index, patch in enumerate(patches):
        rgb[labels == index] = _patch_color(patch["patch_id"])
    image = Image.fromarray(rgb, mode="RGB")
    if max(image.size) > 4096:
        image.thumbnail((4096, 4096), Image.Resampling.NEAREST)
    image.save(path)


def _inconsistent_raster_patches(patches: list[dict], min_inliers: int) -> list[str]:
    return sorted(
        patch["patch_id"]
        for patch in patches
        if patch["overlap_samples"] >= min_inliers
        and patch["conflict_samples"] > patch["agreeing_overlap_samples"]
    )


def export_component(
    cache: Path,
    checkpoint: Path,
    output: Path,
    *,
    spacing: float = 20.0,
    umbilicus: Path | None = None,
    device: str = "cuda",
    workers: int = 0,
    contact_tolerance: float = 2.0,
    min_inliers: int = 16,
    uv_ransac_tolerance: float = 3.0,
    max_refit_rms: float = 2.0,
    ransac_hypotheses: int = 512,
    max_raster_samples: int = 100_000_000,
    theta_provider=None,
) -> dict:
    cache = Path(cache)
    checkpoint = Path(checkpoint)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    provider = theta_provider or SpiralThetaProvider(
        checkpoint, umbilicus=umbilicus, device=device
    )
    options = LayoutOptions()
    options.contact_tolerance = contact_tolerance
    options.min_inliers = min_inliers
    options.uv_ransac_tolerance = uv_ransac_tolerance
    options.max_refit_rms = max_refit_rms
    options.ransac_hypotheses = ransac_hypotheses
    options.max_raster_samples = max_raster_samples
    options.workers = workers
    native = layout_largest_fiber_component(cache, provider, options)
    fiber_metadata, zyx, uv, turn = _native_metadata(native, provider)
    # The v1 layout uses the saved cache only as an immutable patch surface
    # and contact index.  Its legacy theta-provider identity and winding graph
    # are intentionally outside the layout authority.
    graph = WindingGraph.open(cache)
    specs = _patch_specs(cache)
    initial = _initial_correspondences(
        graph, zyx, uv, turn, specs, contact_tolerance
    )
    placed, rejected, relative_contacts = _grow_patches(
        graph, specs, initial, options, provider
    )
    if not placed:
        raise RuntimeError("no patch passed the fiber registration inlier gate")
    pose_graph = _refine_pose_graph(
        placed, initial, relative_contacts, options, provider, specs
    )
    for patch_id in pose_graph.get("quarantined_patches", ()):
        placed.pop(patch_id, None)
        rejected[patch_id] = "pose_graph_quarantine"
    raster_quarantined = []
    while True:
        raster = _rasterize(
            provider,
            placed,
            specs,
            spacing,
            max_raster_samples,
            contact_tolerance,
        )
        inconsistent = _inconsistent_raster_patches(
            raster["patches"], min_inliers
        )
        if not inconsistent:
            break
        if len(inconsistent) == len(placed):
            raise RuntimeError("raster consistency quarantine removed every patch")
        for patch_id in inconsistent:
            placed.pop(patch_id, None)
            rejected[patch_id] = "raster_overlap_quarantine"
        raster_quarantined.extend(inconsistent)
    pose_graph["raster_quarantined_patches"] = raster_quarantined
    metadata = {
        "schema": "spiral-fiber-component-layout",
        "version": 1,
        "cache": str(cache.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "coordinate_system": {
            "u": "checkpoint-seam-positive horizontal fiber arclength",
            "v": "z-positive vertical fiber arclength",
            "integer_winding": (
                "component-relative lift propagated from the root H fiber"
            ),
            "fractional_winding": "raw polar theta around the umbilicus",
            "spacing": spacing,
            "raster_row_zero": "maximum-v/highest-z",
            **raster["bounds"],
        },
        "registration_options": {
            "contact_tolerance": contact_tolerance,
            "min_inliers": min_inliers,
            "uv_ransac_tolerance": uv_ransac_tolerance,
            "max_refit_rms": max_refit_rms,
            "ransac_hypotheses": ransac_hypotheses,
            "max_raster_samples": max_raster_samples,
        },
        "fiber_component": fiber_metadata,
        "patches": raster["patches"],
        "patch_pose_graph": pose_graph,
        "rejected_or_unplaced": [
            {"patch_id": patch_id, "reason": reason}
            for patch_id, reason in sorted(rejected.items())
        ],
        "overlap_conflicts": raster["conflicts"],
        "raster": {
            "height": int(raster["labels"].shape[0]),
            "width": int(raster["labels"].shape[1]),
            "samples": int(raster["labels"].size),
            "valid_samples": int(raster["occupied"].sum()),
            "valid_quads": int(raster["valid_quads"].sum()),
            "uv_20_to_xyz_edge_length_percentiles": raster["edge_percentiles"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        import tifffile

        _write_preview(
            temporary / "overview.png", raster["labels"], raster["patches"]
        )
        (temporary / "layout.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tifffile.imwrite(
            temporary / "patch_index.tif", raster["labels"], compression=None
        )
        tifffile.imwrite(
            temporary / "winding.tif", raster["winding"], compression=None
        )
        tifffile.imwrite(
            temporary / "fractional_winding.tif",
            raster["fractional"],
            compression=None,
        )
        surface = temporary / "surface.tifxyz"
        surface.mkdir()
        tifffile.imwrite(
            surface / "x.tif", raster["xyz"][..., 2], compression=None
        )
        tifffile.imwrite(
            surface / "y.tif", raster["xyz"][..., 1], compression=None
        )
        tifffile.imwrite(
            surface / "z.tif", raster["xyz"][..., 0], compression=None
        )
        tifffile.imwrite(
            surface / "mask.tif",
            raster["occupied"].astype(np.uint8) * 255,
            compression=None,
        )
        (surface / "meta.json").write_text(
            json.dumps(
                {
                    "format": "tifxyz",
                    "type": "seg",
                    "uuid": f"fiber-component-{native.root_fiber}",
                    "scale": [1.0 / spacing, 1.0 / spacing],
                    "step_size": [spacing, spacing],
                    "layout": "fiber-first-u-v",
                    "row_zero": "maximum-v",
                    "source": str(cache.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=20.0)
    parser.add_argument("--umbilicus", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--contact-tolerance", type=float, default=2.0)
    parser.add_argument("--min-inliers", type=int, default=16)
    parser.add_argument("--uv-ransac-tolerance", type=float, default=3.0)
    parser.add_argument("--max-refit-rms", type=float, default=2.0)
    parser.add_argument("--ransac-hypotheses", type=int, default=512)
    parser.add_argument("--max-raster-samples", type=int, default=100_000_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    metadata = export_component(
        args.cache,
        args.checkpoint,
        args.output,
        spacing=args.spacing,
        umbilicus=args.umbilicus,
        device=args.device,
        workers=args.workers,
        contact_tolerance=args.contact_tolerance,
        min_inliers=args.min_inliers,
        uv_ransac_tolerance=args.uv_ransac_tolerance,
        max_refit_rms=args.max_refit_rms,
        ransac_hypotheses=args.ransac_hypotheses,
        max_raster_samples=args.max_raster_samples,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fibers": metadata["fiber_component"]["fiber_count"],
                "patches": len(metadata["patches"]),
                **metadata["raster"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
