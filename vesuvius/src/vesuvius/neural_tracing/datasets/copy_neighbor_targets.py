from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid_masked


@dataclass(frozen=True)
class CopyNeighborTargetPayload:
    cond_gt: np.ndarray
    target_seg: np.ndarray
    domain: np.ndarray
    velocity_dir: np.ndarray
    velocity_loss_weight: np.ndarray
    progress_phi: np.ndarray
    progress_phi_weight: np.ndarray
    surface_attract: np.ndarray
    surface_attract_weight: np.ndarray
    stop: np.ndarray
    stop_weight: np.ndarray
    target_edt: np.ndarray
    endpoint_seed_points: np.ndarray
    endpoint_seed_mask: np.ndarray
    debug: dict


def _config_float(config: Mapping[str, object], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite, got {value!r}")
    return value


def _config_int(config: Mapping[str, object], key: str, default: int) -> int:
    return int(config.get(key, default))


def _ball_structure(radius: float) -> np.ndarray:
    radius = float(radius)
    if radius <= 0.0:
        return np.ones((1, 1, 1), dtype=bool)
    ceil_radius = int(math.ceil(radius))
    zz, yy, xx = np.mgrid[
        -ceil_radius : ceil_radius + 1,
        -ceil_radius : ceil_radius + 1,
        -ceil_radius : ceil_radius + 1,
    ]
    return (zz * zz + yy * yy + xx * xx) <= radius * radius


def _finite_in_crop(surface: np.ndarray, crop_size: tuple[int, int, int]) -> np.ndarray:
    surface = np.asarray(surface, dtype=np.float32)
    finite = np.isfinite(surface).all(axis=-1)
    lower = (surface >= 0.0).all(axis=-1)
    upper = (
        (surface[..., 0] <= crop_size[0] - 1)
        & (surface[..., 1] <= crop_size[1] - 1)
        & (surface[..., 2] <= crop_size[2] - 1)
    )
    return finite & lower & upper


def _voxelize(surface: np.ndarray, crop_size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    valid = _finite_in_crop(surface, crop_size)
    voxels = voxelize_surface_grid_masked(
        np.asarray(surface, dtype=np.float32),
        crop_size,
        valid.astype(np.bool_),
    )
    return voxels.astype(bool, copy=False), valid


def _draw_segment(mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
    delta = b - a
    steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) * 2.0)))
    for t in np.linspace(0.0, 1.0, steps + 1, dtype=np.float32):
        p = np.rint(a + delta * t).astype(np.int64)
        z, y, x = int(p[0]), int(p[1]), int(p[2])
        if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
            mask[z, y, x] = True


def _keep_components_touching_source_and_target(
    domain: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    debug: dict,
) -> np.ndarray:
    labels, num_labels = ndimage.label(domain)
    debug["connected_component_count"] = int(num_labels)
    if num_labels == 0:
        debug["source_to_target_connected_component_count"] = 0
        return np.zeros_like(domain, dtype=bool)

    source_labels = set(int(v) for v in np.unique(labels[source_mask]) if int(v) != 0)
    target_labels = set(int(v) for v in np.unique(labels[target_mask]) if int(v) != 0)
    keep_labels = np.array(sorted(source_labels & target_labels), dtype=np.int32)
    debug["source_to_target_connected_component_count"] = int(keep_labels.size)
    if keep_labels.size == 0:
        return np.zeros_like(domain, dtype=bool)
    return np.isin(labels, keep_labels)


def _connector_domain(
    *,
    crop_size: tuple[int, int, int],
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    side_hint_vector: np.ndarray,
    config: Mapping[str, object],
    debug: dict,
) -> np.ndarray:
    stride = max(1, _config_int(config, "copy_neighbor_connector_sample_stride", 4))
    dilation_radius = _config_float(config, "copy_neighbor_bridge_dilation_radius", 2.0)
    closing_radius = _config_float(config, "copy_neighbor_bridge_closing_radius", 2.0)

    bridge = np.zeros(crop_size, dtype=bool)
    sampled_source = source_points[::stride]
    if sampled_source.size == 0 or target_points.size == 0:
        debug["failure_reason"] = "empty connector source or target points"
        return np.zeros(crop_size, dtype=bool)

    tree = cKDTree(target_points.astype(np.float32, copy=False))
    _, nearest = tree.query(sampled_source.astype(np.float32, copy=False), k=1)
    nearest_target = target_points[np.asarray(nearest, dtype=np.int64)]
    for source_point, target_point in zip(sampled_source, nearest_target):
        _draw_segment(bridge, source_point, target_point)

    debug["connector_segment_count"] = int(sampled_source.shape[0])
    debug["bridge_voxels_raw"] = int(bridge.sum())

    if dilation_radius > 0.0:
        bridge = ndimage.binary_dilation(bridge, structure=_ball_structure(dilation_radius))
    if closing_radius > 0.0:
        bridge = ndimage.binary_closing(bridge, structure=_ball_structure(closing_radius))
    debug["bridge_voxels_after_morphology"] = int(bridge.sum())

    domain = bridge | source_mask | target_mask
    if bool(config.get("copy_neighbor_fill_domain_holes", True)):
        domain = ndimage.binary_fill_holes(domain)

    if bool(config.get("copy_neighbor_keep_touching_components", True)):
        domain = _keep_components_touching_source_and_target(domain, source_mask, target_mask, debug)

    return _apply_side_hint(domain, source_mask, target_mask, side_hint_vector, config, debug)


def _distance_score_domain(
    *,
    crop_size: tuple[int, int, int],
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    side_hint_vector: np.ndarray,
    d_source: np.ndarray,
    d_target: np.ndarray,
    config: Mapping[str, object],
    debug: dict,
) -> np.ndarray:
    max_source = _config_float(config, "copy_neighbor_max_source_distance", 16.0)
    max_target = _config_float(config, "copy_neighbor_max_target_distance", 16.0)
    default_threshold = max_source + max_target
    score_threshold = _config_float(config, "copy_neighbor_bridge_score_threshold", default_threshold)
    domain = (
        (d_source <= max_source)
        & (d_target <= max_target)
        & ((d_source + d_target) <= score_threshold)
    )
    domain = domain | source_mask | target_mask
    debug["bridge_voxels_raw"] = int(domain.sum())
    debug["bridge_voxels_after_morphology"] = int(domain.sum())
    if bool(config.get("copy_neighbor_keep_touching_components", True)):
        domain = _keep_components_touching_source_and_target(domain, source_mask, target_mask, debug)
    return _apply_side_hint(domain, source_mask, target_mask, side_hint_vector, config, debug)


def _apply_side_hint(
    domain: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    side_hint_vector: np.ndarray,
    config: Mapping[str, object],
    debug: dict,
) -> np.ndarray:
    tolerance = _config_float(config, "copy_neighbor_side_tolerance", 1.0)
    if not bool(domain.any()):
        debug["rejected_by_side_count"] = 0
        return domain

    _, nearest_source_idx = ndimage.distance_transform_edt(
        ~source_mask,
        return_distances=True,
        return_indices=True,
    )
    coords = np.indices(domain.shape, dtype=np.float32)
    nearest = nearest_source_idx.astype(np.float32, copy=False)
    offset = coords - nearest
    score = (
        offset[0] * float(side_hint_vector[0])
        + offset[1] * float(side_hint_vector[1])
        + offset[2] * float(side_hint_vector[2])
    )
    selected_side = score >= -tolerance
    rejected = domain & ~selected_side
    debug["rejected_by_side_count"] = int(rejected.sum())
    return (domain & selected_side) | source_mask | target_mask


def _edt_progress(d_source: np.ndarray, d_target: np.ndarray, domain: np.ndarray) -> np.ndarray:
    denom = d_source + d_target + np.float32(1e-6)
    phi = np.zeros_like(d_source, dtype=np.float32)
    phi[domain] = np.clip(d_source[domain] / denom[domain], 0.0, 1.0)
    return phi


def _harmonic_progress(
    *,
    edt_phi: np.ndarray,
    domain: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    config: Mapping[str, object],
    debug: dict,
) -> np.ndarray | None:
    unknown = domain & ~source_mask & ~target_mask
    min_unknown = _config_int(config, "copy_neighbor_harmonic_min_unknown_voxels", 1)
    if int(unknown.sum()) < min_unknown:
        debug["harmonic_failure_reason"] = "too few unknown voxels"
        return None

    max_iters = max(1, _config_int(config, "copy_neighbor_harmonic_max_iters", 250))
    tolerance = _config_float(config, "copy_neighbor_harmonic_tolerance", 1e-3)
    required_converged = bool(config.get("copy_neighbor_harmonic_required_converged", False))

    phi = edt_phi.astype(np.float32, copy=True)
    phi[source_mask] = 0.0
    phi[target_mask] = 1.0
    phi[~domain] = 0.0

    start = time.perf_counter()
    residual = np.inf
    slices = (
        ((slice(1, None), slice(None), slice(None)), (slice(None, -1), slice(None), slice(None))),
        ((slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
        ((slice(None), slice(1, None), slice(None)), (slice(None), slice(None, -1), slice(None))),
        ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
        ((slice(None), slice(None), slice(1, None)), (slice(None), slice(None), slice(None, -1))),
        ((slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None))),
    )
    for iteration in range(max_iters):
        neighbor_sum = np.zeros_like(phi, dtype=np.float32)
        neighbor_count = np.zeros_like(phi, dtype=np.float32)
        for src_slice, dst_slice in slices:
            valid_neighbor = domain[src_slice]
            neighbor_sum[dst_slice] += phi[src_slice] * valid_neighbor
            neighbor_count[dst_slice] += valid_neighbor.astype(np.float32, copy=False)

        updatable = unknown & (neighbor_count > 0.0)
        next_phi = phi.copy()
        next_phi[updatable] = neighbor_sum[updatable] / neighbor_count[updatable]
        residual = float(np.max(np.abs(next_phi[updatable] - phi[updatable]))) if bool(updatable.any()) else 0.0
        phi = next_phi
        phi[source_mask] = 0.0
        phi[target_mask] = 1.0
        phi[~domain] = 0.0
        if residual <= tolerance:
            break

    converged = residual <= tolerance
    debug["harmonic_iterations"] = int(iteration + 1)
    debug["harmonic_residual"] = float(residual)
    debug["harmonic_converged"] = bool(converged)
    debug["harmonic_seconds"] = float(time.perf_counter() - start)
    if required_converged and not converged:
        debug["harmonic_failure_reason"] = "not converged"
        return None
    return np.clip(phi, 0.0, 1.0).astype(np.float32, copy=False)


def _normalized_gradient(phi: np.ndarray, domain: np.ndarray) -> np.ndarray:
    grads = np.gradient(phi.astype(np.float32, copy=False))
    velocity = np.stack(grads, axis=0).astype(np.float32, copy=False)
    norm = np.linalg.norm(velocity, axis=0)
    valid = domain & np.isfinite(norm) & (norm > 1e-6)
    out = np.zeros_like(velocity, dtype=np.float32)
    out[:, valid] = velocity[:, valid] / norm[valid][None]
    return out


def _fixed_endpoint_seeds(
    source_points: np.ndarray,
    target_edt: np.ndarray,
    config: Mapping[str, object],
    debug: dict,
) -> tuple[np.ndarray, np.ndarray]:
    num_seeds = max(1, _config_int(config, "copy_neighbor_endpoint_num_seeds", 256))
    seeds = np.zeros((num_seeds, 3), dtype=np.float32)
    mask = np.zeros((num_seeds,), dtype=np.float32)
    if source_points.size == 0:
        debug["endpoint_step_count"] = 0
        return seeds, mask

    order = np.linspace(0, source_points.shape[0] - 1, min(num_seeds, source_points.shape[0]), dtype=np.int64)
    selected = source_points[order].astype(np.float32, copy=False)
    seeds[: selected.shape[0]] = selected
    mask[: selected.shape[0]] = 1.0

    step_size = max(_config_float(config, "copy_neighbor_endpoint_step_size", 1.0), 1e-6)
    if str(config.get("copy_neighbor_endpoint_steps_mode", "adaptive")) == "fixed":
        step_count = max(1, _config_int(config, "copy_neighbor_endpoint_steps", 8))
    else:
        percentile = _config_float(config, "copy_neighbor_endpoint_distance_percentile", 75.0)
        margin = _config_int(config, "copy_neighbor_endpoint_step_margin", 2)
        max_steps = max(1, _config_int(config, "copy_neighbor_endpoint_max_steps", 64))
        rounded = np.rint(selected).astype(np.int64)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, target_edt.shape[0] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, target_edt.shape[1] - 1)
        rounded[:, 2] = np.clip(rounded[:, 2], 0, target_edt.shape[2] - 1)
        distances = target_edt[rounded[:, 0], rounded[:, 1], rounded[:, 2]]
        representative_distance = float(np.percentile(distances, percentile)) if distances.size else 0.0
        step_count = int(math.ceil(representative_distance / step_size))
        step_count = min(max(step_count + margin, 1), max_steps)
        debug["endpoint_representative_distance"] = representative_distance
    debug["endpoint_step_count"] = int(step_count)
    return seeds, mask


def build_copy_neighbor_targets(
    crop_size: tuple[int, int, int],
    source_surface_local: np.ndarray,
    target_surface_local: np.ndarray,
    side_hint_vector: np.ndarray,
    config: Mapping[str, object],
) -> CopyNeighborTargetPayload | None:
    """Build dense copy-neighbor trace/ODE targets in crop-local ZYX coordinates."""
    crop_size = tuple(int(v) for v in crop_size)
    debug: dict[str, object] = {}
    source_surface = np.asarray(source_surface_local, dtype=np.float32)
    target_surface = np.asarray(target_surface_local, dtype=np.float32)
    side_hint = np.asarray(side_hint_vector, dtype=np.float32)
    side_norm = float(np.linalg.norm(side_hint))
    if source_surface.ndim != 3 or source_surface.shape[-1] != 3:
        debug["failure_reason"] = "source surface must have shape [H, W, 3]"
        return None
    if target_surface.ndim != 3 or target_surface.shape[-1] != 3:
        debug["failure_reason"] = "target surface must have shape [H, W, 3]"
        return None
    if not np.isfinite(side_norm) or side_norm <= 1e-6:
        debug["failure_reason"] = "invalid side hint vector"
        return None
    side_hint = side_hint / np.float32(side_norm)

    source_mask, source_valid = _voxelize(source_surface, crop_size)
    target_mask, target_valid = _voxelize(target_surface, crop_size)
    debug["source_voxel_count"] = int(source_mask.sum())
    debug["target_voxel_count"] = int(target_mask.sum())
    debug["source_valid_fraction"] = float(source_valid.mean()) if source_valid.size else 0.0
    debug["target_valid_fraction"] = float(target_valid.mean()) if target_valid.size else 0.0

    min_source_voxels = _config_int(config, "copy_neighbor_min_source_voxels", 8)
    min_target_voxels = _config_int(config, "copy_neighbor_min_target_voxels", 8)
    if int(source_mask.sum()) < min_source_voxels:
        debug["failure_reason"] = "too few source voxels"
        return None
    if int(target_mask.sum()) < min_target_voxels:
        debug["failure_reason"] = "too few target voxels"
        return None

    source_points = source_surface[source_valid]
    target_points = target_surface[target_valid]
    d_source = ndimage.distance_transform_edt(~source_mask).astype(np.float32, copy=False)
    d_target = ndimage.distance_transform_edt(~target_mask).astype(np.float32, copy=False)
    target_edt = d_target.astype(np.float32, copy=False)

    domain_builder = str(config.get("copy_neighbor_domain_builder", "connectors"))
    if domain_builder == "connectors":
        domain = _connector_domain(
            crop_size=crop_size,
            source_mask=source_mask,
            target_mask=target_mask,
            source_points=source_points,
            target_points=target_points,
            side_hint_vector=side_hint,
            config=config,
            debug=debug,
        )
    elif domain_builder == "distance_score":
        domain = _distance_score_domain(
            crop_size=crop_size,
            source_mask=source_mask,
            target_mask=target_mask,
            side_hint_vector=side_hint,
            d_source=d_source,
            d_target=d_target,
            config=config,
            debug=debug,
        )
    else:
        raise ValueError(f"Unsupported copy_neighbor_domain_builder={domain_builder!r}")

    debug["domain_voxel_count"] = int(domain.sum())
    if int(domain.sum()) < _config_int(config, "copy_neighbor_min_domain_voxels", 16):
        debug["failure_reason"] = "too few domain voxels"
        return None

    debug["d_source_domain_min"] = float(np.min(d_source[domain]))
    debug["d_source_domain_mean"] = float(np.mean(d_source[domain]))
    debug["d_source_domain_max"] = float(np.max(d_source[domain]))
    debug["d_target_domain_min"] = float(np.min(d_target[domain]))
    debug["d_target_domain_mean"] = float(np.mean(d_target[domain]))
    debug["d_target_domain_max"] = float(np.max(d_target[domain]))

    edt_phi = _edt_progress(d_source, d_target, domain)
    progress_builder = str(config.get("copy_neighbor_progress_builder", "harmonic"))
    if progress_builder == "edt":
        progress_phi = edt_phi
    elif progress_builder == "harmonic":
        progress_phi = _harmonic_progress(
            edt_phi=edt_phi,
            domain=domain,
            source_mask=source_mask,
            target_mask=target_mask,
            config=config,
            debug=debug,
        )
        if progress_phi is None:
            debug.setdefault("failure_reason", debug.get("harmonic_failure_reason", "harmonic progress failed"))
            return None
    else:
        raise ValueError(f"Unsupported copy_neighbor_progress_builder={progress_builder!r}")

    velocity_dir = _normalized_gradient(progress_phi, domain)
    velocity_weight = (domain & (np.linalg.norm(velocity_dir, axis=0) > 0.0))[None].astype(np.float32, copy=False)
    progress_weight = domain[None].astype(np.float32, copy=False)

    _, nearest_target_idx = ndimage.distance_transform_edt(
        ~target_mask,
        return_distances=True,
        return_indices=True,
    )
    coords = np.indices(crop_size, dtype=np.float32)
    nearest_target = nearest_target_idx.astype(np.float32, copy=False)
    surface_attract = (nearest_target - coords).astype(np.float32, copy=False)
    surface_radius = _config_float(config, "copy_neighbor_surface_attract_radius", 4.0)
    surface_weight = (target_edt <= surface_radius)[None].astype(np.float32, copy=False)
    surface_attract[:, surface_weight[0] <= 0.0] = 0.0

    stop_radius = _config_float(config, "copy_neighbor_stop_radius", 1.5)
    stop = (target_edt <= stop_radius)[None].astype(np.float32, copy=False)
    stop_weight = (domain | (target_edt <= stop_radius))[None].astype(np.float32, copy=False)

    endpoint_seed_points, endpoint_seed_mask = _fixed_endpoint_seeds(
        source_points,
        target_edt,
        config,
        debug,
    )

    return CopyNeighborTargetPayload(
        cond_gt=source_mask.astype(np.float32, copy=False),
        target_seg=target_mask.astype(np.float32, copy=False),
        domain=domain.astype(np.float32, copy=False),
        velocity_dir=velocity_dir.astype(np.float32, copy=False),
        velocity_loss_weight=velocity_weight,
        progress_phi=progress_phi[None].astype(np.float32, copy=False),
        progress_phi_weight=progress_weight,
        surface_attract=surface_attract.astype(np.float32, copy=False),
        surface_attract_weight=surface_weight,
        stop=stop,
        stop_weight=stop_weight,
        target_edt=target_edt[None].astype(np.float32, copy=False),
        endpoint_seed_points=endpoint_seed_points,
        endpoint_seed_mask=endpoint_seed_mask,
        debug=debug,
    )
