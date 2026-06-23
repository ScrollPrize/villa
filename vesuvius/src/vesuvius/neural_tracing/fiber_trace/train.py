from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace.dataset import (
    FiberTraceBatch,
    FiberTraceBatchBuilder,
    FiberTraceDebugTableState,
    ZarrChunkRequest,
)
from vesuvius.neural_tracing.fiber_trace.labels import (
    IGNORE_ID,
    IGNORE_INDEX,
    NEGATIVE_LABEL,
    NEGATIVE_ONLY_ID,
    POSITIVE_LABEL,
)
from vesuvius.neural_tracing.fiber_trace.losses import (
    compute_fiber_trace_loss,
    sample_contrastive_pair_indices,
)
from vesuvius.neural_tracing.fiber_trace.model import build_fiber_trace_model


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config JSON must be an object, got {type(config).__name__}")
    config.setdefault("_config_dir", str(config_path.parent))
    return config


def _sanitize_run_name(value: Any) -> str:
    name = str(value or "fiber_trace").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._-") or "fiber_trace"


def _reject_legacy_checkpoint_path(config: dict[str, Any]) -> None:
    if "checkpoint_path" in config:
        raise ValueError(
            "checkpoint_path was replaced by run_path/run_name; snapshots are "
            "written to <run_path>/<run_name>_<datestr>/snapshots/current.pt "
            "and best.pt"
        )


def _resolve_run_layout(config: dict[str, Any]) -> tuple[Path, Path]:
    _reject_legacy_checkpoint_path(config)
    run_path = Path(str(config.get("run_path", "runs/fiber_trace")))
    run_name = _sanitize_run_name(config.get("run_name", "fiber_trace"))
    date_str = str(
        config.get("run_datestr") or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = run_path / f"{run_name}_{date_str}"
    snapshot_dir = run_dir / "snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, snapshot_dir


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return f"<array shape={tuple(shape)} dtype={dtype}>"
    return repr(value)


def _config_json_text(config: dict[str, Any]) -> str:
    return json.dumps(_json_safe(config), indent=2, sort_keys=True)


def _make_summary_writer(log_dir: Path, *, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires the tensorboard package. "
            "Install tensorboard or set tensorboard_enabled=false."
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))


def _make_test_config(config: dict[str, Any]) -> dict[str, Any] | None:
    if config.get("_test_array_records"):
        test_config = dict(config)
        test_config["_array_records"] = config["_test_array_records"]
        return test_config

    if config.get("test_datasets"):
        test_config = dict(config)
        test_config["datasets"] = config["test_datasets"]
        test_config.pop("_array_records", None)
        return test_config

    datasets = config.get("datasets")
    if not isinstance(datasets, list):
        return None

    test_datasets: list[dict[str, Any]] = []
    for dataset_raw in datasets:
        dataset = dict(dataset_raw)
        test_paths = dataset.get("test_fiber_paths")
        test_glob = dataset.get("test_fiber_glob")
        if not test_paths and not test_glob:
            continue

        dataset.pop("fiber_paths", None)
        dataset.pop("fiber_glob", None)
        if test_paths:
            dataset["fiber_paths"] = test_paths
        if test_glob:
            dataset["fiber_glob"] = test_glob
        test_datasets.append(dataset)

    if not test_datasets:
        return None

    test_config = dict(config)
    test_config["datasets"] = test_datasets
    test_config.pop("_array_records", None)
    return test_config


def _loss_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = dict(config.get("loss", {}))
    return {
        "temperature": float(loss_cfg.get("temperature", 0.1)),
        "contrastive_weight": float(loss_cfg.get("contrastive_weight", 1.0)),
        "max_contrastive_samples": int(loss_cfg.get("max_contrastive_samples", 4096)),
    }


def _normalize_vectors_torch(vec: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    return vec.to(dtype=torch.float32) / torch.linalg.vector_norm(
        vec.to(dtype=torch.float32), dim=-1, keepdim=True
    ).clamp_min(float(eps))


def _nearest_polyline_projection_torch(
    coords_xyz: torch.Tensor, line_points_xyz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = coords_xyz.reshape(-1, 3).to(dtype=torch.float32)
    points = line_points_xyz.to(device=flat.device, dtype=torch.float32)
    best_dist_sq = torch.full(
        (flat.shape[0],), float("inf"), device=flat.device, dtype=torch.float32
    )
    best_tangent = torch.zeros((flat.shape[0], 3), device=flat.device)
    best_closest = torch.zeros((flat.shape[0], 3), device=flat.device)

    for start, end in zip(points[:-1].unbind(0), points[1:].unbind(0), strict=True):
        seg = end - start
        seg_len_sq = torch.sum(seg * seg)
        valid_seg = torch.isfinite(seg_len_sq) & (seg_len_sq > 1e-12)
        safe_len_sq = seg_len_sq.clamp_min(1e-12)
        tangent = seg / torch.sqrt(safe_len_sq)
        rel = flat - start
        t = ((rel @ seg) / safe_len_sq).clamp(0.0, 1.0)
        closest = start + t[:, None] * seg
        dist_sq = torch.sum((flat - closest) ** 2, dim=1)
        update = (dist_sq < best_dist_sq) & valid_seg
        best_dist_sq = torch.where(update, dist_sq, best_dist_sq)
        best_tangent = torch.where(update[:, None], tangent[None, :], best_tangent)
        best_closest = torch.where(update[:, None], closest, best_closest)

    out_shape = coords_xyz.shape[:-1]
    return (
        torch.sqrt(best_dist_sq).reshape(out_shape),
        best_tangent.reshape(*out_shape, 3),
        best_closest.reshape(*out_shape, 3),
    )


def classify_batch_on_device(
    batch: FiberTraceBatch, config: dict[str, Any]
) -> FiberTraceBatch:
    device = batch.volume.device
    crop_shape = tuple(int(v) for v in batch.volume.shape[-3:])
    bsz = int(batch.volume.shape[0])
    zz, yy, xx = torch.meshgrid(
        torch.arange(crop_shape[0], device=device, dtype=torch.float32),
        torch.arange(crop_shape[1], device=device, dtype=torch.float32),
        torch.arange(crop_shape[2], device=device, dtype=torch.float32),
        indexing="ij",
    )
    base_coords_zyx = torch.stack([zz, yy, xx], dim=-1)

    valid_mask = batch.mask_values > 0.0

    labels = torch.full_like(batch.labels, int(IGNORE_INDEX))
    target_id = torch.full_like(batch.target_id, int(IGNORE_ID))
    target_fw_xyz = torch.zeros_like(batch.target_fw_xyz)

    normal_plane_jitter = float(config["normal_plane_jitter_voxels"])
    normal_perp_jitter = float(config["normal_perpendicular_jitter_voxels"])
    positive_along_fiber_limit = float(config["positive_along_fiber_limit_voxels"])
    negative_cone_distance = float(config["negative_cone_distance_voxels"])
    for idx in range(bsz):
        origin_zyx = batch.crop_origin_zyx[idx].to(device=device, dtype=torch.float32)
        coords_zyx = base_coords_zyx + origin_zyx
        coords_xyz = torch.stack(
            [coords_zyx[..., 2], coords_zyx[..., 1], coords_zyx[..., 0]], dim=-1
        )
        _, tangent_xyz, closest_xyz = _nearest_polyline_projection_torch(
            coords_xyz, batch.line_points_xyz[idx]
        )
        tangent_xyz = _normalize_vectors_torch(tangent_xyz)

        normal_i = batch.center_normal_xyz[idx].to(
            device=device, dtype=torch.float32
        )
        normal_geometry_valid = torch.isfinite(normal_i).all() & (
            torch.linalg.vector_norm(normal_i) > 1e-6
        )
        normal_for_geometry = normal_i / torch.linalg.vector_norm(normal_i).clamp_min(
            1e-6
        )

        valid = valid_mask[idx] & normal_geometry_valid
        offset_xyz = coords_xyz - closest_xyz
        plane_offset = torch.sum(offset_xyz * normal_for_geometry, dim=-1)
        in_plane_offset = offset_xyz - plane_offset[..., None] * normal_for_geometry
        in_plane_distance = torch.linalg.vector_norm(in_plane_offset, dim=-1)
        perpendicular_distance = plane_offset.abs()
        positive_center_zyx = origin_zyx + batch.sample_local_zyx[idx].to(
            device=device, dtype=torch.float32
        )
        positive_center_xyz = torch.stack(
            [positive_center_zyx[2], positive_center_zyx[1], positive_center_zyx[0]]
        )
        along_fiber_distance = torch.sum(
            (closest_xyz - positive_center_xyz.view(1, 1, 1, 3)) * tangent_xyz,
            dim=-1,
        ).abs()

        positive_zone = (
            valid
            & (in_plane_distance <= normal_plane_jitter)
            & (perpendicular_distance <= normal_perp_jitter)
            & (along_fiber_distance <= positive_along_fiber_limit)
        )

        cone_radius = (perpendicular_distance - negative_cone_distance).clamp_min(0.0)
        inside_normal_cone = (
            (perpendicular_distance >= negative_cone_distance)
            & (in_plane_distance <= cone_radius)
        )
        cone_negative = valid & inside_normal_cone
        if str(batch.crop_kinds[idx]) == "random_negative":
            positive = torch.zeros_like(valid, dtype=torch.bool)
            negative = valid
        else:
            positive = positive_zone
            negative = cone_negative & ~positive

        labels[idx] = torch.where(
            negative, torch.full_like(labels[idx], int(NEGATIVE_LABEL)), labels[idx]
        )
        target_id[idx] = torch.where(
            negative,
            torch.full_like(target_id[idx], int(NEGATIVE_ONLY_ID)),
            target_id[idx],
        )
        labels[idx] = torch.where(
            positive, torch.full_like(labels[idx], int(POSITIVE_LABEL)), labels[idx]
        )
        positive_id = batch.positive_target_id[idx].to(
            device=device, dtype=target_id.dtype
        )
        target_id[idx] = torch.where(
            positive,
            positive_id.expand_as(target_id[idx]),
            target_id[idx],
        )

        target_fw_xyz[idx] = tangent_xyz.movedim(-1, 0)

    return batch.with_classification(
        valid_mask=valid_mask,
        labels=labels,
        target_id=target_id,
        target_fw_xyz=target_fw_xyz,
    )


def _compute_losses(
    model: torch.nn.Module,
    batch_builder: FiberTraceBatchBuilder,
    device: torch.device,
    config: dict[str, Any],
    loss_kwargs: dict[str, Any],
    *,
    iteration: int | None = None,
    debug_batch: bool = True,
) -> tuple[Any, FiberTraceBatch, dict[str, torch.Tensor], dict[str, float]]:
    data_t0 = time.perf_counter()
    batch = batch_builder.sample_batch(
        iteration=iteration, debug=debug_batch, emit_debug_row=False
    )
    data_ms = (time.perf_counter() - data_t0) * 1000.0

    batch = batch.to(device)
    batch = classify_batch_on_device(batch, config)

    contrastive_samples = sample_contrastive_pair_indices(
        batch.labels,
        batch.target_id,
        max_samples=loss_kwargs["max_contrastive_samples"],
        seed=0 if iteration is None else int(iteration),
    )
    outputs = model(
        batch.volume,
        batch.cond_fw_xyz,
        sample_indices=contrastive_samples.flat_indices,
    )
    losses = compute_fiber_trace_loss(
        outputs, batch, contrastive_samples=contrastive_samples, **loss_kwargs
    )
    timings = {"data_ms": data_ms}
    cache_stats = getattr(batch_builder, "last_cache_stats", None)
    if cache_stats is not None:
        cache_mib = float(getattr(cache_stats, "cache_hit_bytes", 0)) / (
            1024.0 * 1024.0
        )
        download_mib = float(getattr(cache_stats, "download_bytes", 0)) / (
            1024.0 * 1024.0
        )
        cache_ms = float(getattr(cache_stats, "cache_hit_ms", 0.0))
        download_ms = float(getattr(cache_stats, "download_ms", 0.0))
        timings.update(
            {
                "cache_hits": float(getattr(cache_stats, "cache_hits", 0)),
                "cache_downloads": float(getattr(cache_stats, "downloads", 0)),
                "cache_missing": float(
                    int(getattr(cache_stats, "missing", 0))
                    + int(getattr(cache_stats, "negative_hits", 0))
                ),
                "cache_hit_ms": cache_ms,
                "cache_download_ms": download_ms,
                "cache_mib_s": (
                    cache_mib / max(cache_ms / 1000.0, 1e-9)
                    if cache_mib > 0.0
                    else 0.0
                ),
                "download_mib_s": (
                    download_mib / max(download_ms / 1000.0, 1e-9)
                    if download_mib > 0.0
                    else 0.0
                ),
            }
        )
    return losses, batch, outputs, timings


def _loss_scalars(losses) -> dict[str, float]:
    return {
        "total": float(losses.total.detach().cpu()),
        "contrastive": float(losses.contrastive.detach().cpu()),
    }


def _average_scalar_dicts(scalars: list[dict[str, float]]) -> dict[str, float]:
    if not scalars:
        return {}
    keys = sorted(set().union(*(item.keys() for item in scalars)))
    return {
        key: sum(float(item[key]) for item in scalars if key in item)
        / float(sum(1 for item in scalars if key in item))
        for key in keys
    }


def _compute_test_scalars(
    model: torch.nn.Module,
    batch_builder: FiberTraceBatchBuilder,
    device: torch.device,
    config: dict[str, Any],
    loss_kwargs: dict[str, Any],
    *,
    start_iteration: int,
    sample_count: int,
) -> dict[str, float]:
    was_training = bool(model.training)
    model.eval()
    scalars: list[dict[str, float]] = []
    try:
        with torch.no_grad():
            for offset in range(int(sample_count)):
                losses, _, _, _ = _compute_losses(
                    model,
                    batch_builder,
                    device,
                    config,
                    loss_kwargs,
                    iteration=int(start_iteration) + int(offset),
                    debug_batch=False,
                )
                scalars.append(_loss_scalars(losses))
    finally:
        if was_training:
            model.train()
    return _average_scalar_dicts(scalars)


def _sample_classified_batch(
    batch_builder: FiberTraceBatchBuilder,
    device: torch.device,
    config: dict[str, Any],
    *,
    iteration: int,
) -> FiberTraceBatch:
    batch = batch_builder.sample_batch(
        iteration=int(iteration), debug=False, emit_debug_row=False
    ).to(device)
    return classify_batch_on_device(batch, config)


def _log_scalars(writer: Any, prefix: str, scalars: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for name, value in scalars.items():
        writer.add_scalar(f"{prefix}/{name}", value, step)


def _print_step_timing(
    *,
    step: int,
    timings: dict[str, float],
    train_scalars: dict[str, float] | None,
    test_scalars: dict[str, float] | None,
    row_index: int,
    header_every: int,
    print_legend: bool,
) -> None:
    def _loss_value(scalars: dict[str, float] | None, name: str) -> str:
        if scalars is None:
            return f"{'-':>9}"
        return f"{float(scalars[name]):9.4f}"

    if print_legend:
        print(
            "fiber_trace step columns: it=iteration data=batch sample/load ms "
            "train=device transfer + labels + forward/loss + backward + optimizer ms "
            "hit/dl/mis=cache events hms/dms=cache-hit/download ms "
            "hMiB/s/dMiB/s=cache-hit/download throughput "
            "trn/tst losses=total/contrastive",
            flush=True,
        )
    if row_index % max(1, int(header_every)) == 0:
        print(
            "   it      data     train "
            " hit  dl mis      hms      dms   hMiB/s   dMiB/s   trn_tot   trn_con   tst_tot   tst_con",
            flush=True,
        )
    print(
        f"{int(step):5d} "
        f"{float(timings.get('data_ms', 0.0)):9.1f} "
        f"{float(timings.get('train_ms', 0.0)):9.1f} "
        f"{int(timings.get('cache_hits', 0.0)):4d} "
        f"{int(timings.get('cache_downloads', 0.0)):3d} "
        f"{int(timings.get('cache_missing', 0.0)):3d} "
        f"{float(timings.get('cache_hit_ms', 0.0)):8.1f} "
        f"{float(timings.get('cache_download_ms', 0.0)):8.1f} "
        f"{float(timings.get('cache_mib_s', 0.0)):8.1f} "
        f"{float(timings.get('download_mib_s', 0.0)):8.1f} "
        f"{_loss_value(train_scalars, 'total')} "
        f"{_loss_value(train_scalars, 'contrastive')} "
        f"{_loss_value(test_scalars, 'total')} "
        f"{_loss_value(test_scalars, 'contrastive')}",
        flush=True,
    )


def _normalize_vector(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    vec = vec.to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(vec)
    if (
        bool(torch.isfinite(norm).detach().cpu())
        and float(norm.detach().cpu()) > 1e-6
    ):
        return vec / norm.clamp_min(1e-6)
    return fallback.to(device=vec.device, dtype=torch.float32)


def _sample_local_zyx_tuple(batch: FiberTraceBatch, sample_index: int) -> tuple[int, int, int]:
    shape_zyx = tuple(int(v) for v in batch.labels.shape[-3:])
    local = batch.sample_local_zyx[int(sample_index)].detach().cpu().numpy()
    return tuple(
        max(0, min(int(shape_zyx[axis]) - 1, int(local[axis]))) for axis in range(3)
    )


def _orthonormal_sample_frame(
    batch: FiberTraceBatch, sample_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_index = int(sample_index)
    device = batch.volume.device
    center_zyx = _sample_local_zyx_tuple(batch, sample_index)
    fallback_fw = _normalize_vector(
        batch.cond_fw_xyz[sample_index],
        torch.tensor([1.0, 0.0, 0.0], device=device),
    )
    fw = _normalize_vector(
        batch.target_fw_xyz[(sample_index, slice(None)) + center_zyx],
        fallback_fw,
    )
    fallback_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    if abs(float(torch.sum(fw * fallback_axis).detach().cpu())) > 0.9:
        fallback_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
    normal_axis = fallback_axis
    normal_candidate = batch.center_normal_xyz[sample_index].to(
        device=device, dtype=torch.float32
    )
    normal_norm = torch.linalg.vector_norm(normal_candidate)
    if (
        bool(torch.isfinite(normal_norm).detach().cpu())
        and float(normal_norm.detach().cpu()) > 1e-6
    ):
        normal_axis = normal_candidate / normal_norm.clamp_min(1e-6)
    normal_axis = normal_axis - torch.sum(normal_axis * fw) * fw
    normal_axis = _normalize_vector(normal_axis, fallback_axis)
    right = torch.cross(fw, normal_axis, dim=0)
    if float(torch.linalg.vector_norm(right).detach().cpu()) <= 1e-6:
        normal_axis = _normalize_vector(
            fallback_axis - torch.sum(fallback_axis * fw) * fw,
            fallback_axis,
        )
        right = torch.cross(fw, normal_axis, dim=0)
    right = _normalize_vector(right, torch.tensor([0.0, 0.0, 1.0], device=device))
    normal_axis = _normalize_vector(torch.cross(right, fw, dim=0), normal_axis)
    return fw, normal_axis, right


def _normalize_image_for_tb(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().to(dtype=torch.float32)
    image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    finite = torch.isfinite(image)
    if not bool(finite.any().detach().cpu()):
        return torch.zeros_like(image)
    values = image[finite]
    min_value = values.min()
    max_value = values.max()
    denom = max_value - min_value
    if float(denom.detach().cpu()) <= 1e-6:
        return torch.zeros_like(image)
    return ((image - min_value) / denom.clamp_min(1e-6)).clamp(0.0, 1.0)


def _slice_grid(
    *,
    shape_zyx: tuple[int, int, int],
    center_xyz: torch.Tensor,
    axis_u_xyz: torch.Tensor,
    axis_v_xyz: torch.Tensor,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32)
    coords = coords - float(size // 2)
    grid_v, grid_u = torch.meshgrid(coords, coords, indexing="ij")
    pos_xyz = (
        center_xyz.view(1, 1, 3)
        + grid_u.unsqueeze(-1) * axis_u_xyz.view(1, 1, 3)
        + grid_v.unsqueeze(-1) * axis_v_xyz.view(1, 1, 3)
    )
    depth, height, width = (int(v) for v in shape_zyx)
    x = pos_xyz[..., 0]
    y = pos_xyz[..., 1]
    z = pos_xyz[..., 2]
    x_norm = (
        torch.zeros_like(x) if width <= 1 else (2.0 * x / float(width - 1)) - 1.0
    )
    y_norm = (
        torch.zeros_like(y) if height <= 1 else (2.0 * y / float(height - 1)) - 1.0
    )
    z_norm = (
        torch.zeros_like(z) if depth <= 1 else (2.0 * z / float(depth - 1)) - 1.0
    )
    return torch.stack((x_norm, y_norm, z_norm), dim=-1).view(1, 1, size, size, 3)


def _draw_center_cross(
    image: torch.Tensor,
    *,
    center_yx: tuple[int, int],
    low_value: float,
    high_value: float,
) -> torch.Tensor:
    marked = image.clone()
    height, width = (int(v) for v in marked.shape[-2:])
    if height <= 0 or width <= 0:
        return marked
    center_y = max(0, min(height - 1, int(center_yx[0])))
    center_x = max(0, min(width - 1, int(center_yx[1])))
    radius = max(2, min(6, min(height, width) // 16))
    gap = max(1, min(3, radius // 2))
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    left_end = max(x0, center_x - gap)
    right_start = min(x1, center_x + gap + 1)
    top_end = max(y0, center_y - gap)
    bottom_start = min(y1, center_y + gap + 1)
    if x0 < left_end:
        marked[center_y, x0:left_end] = high_value
    if right_start < x1:
        marked[center_y, right_start:x1] = high_value
    if y0 < top_end:
        marked[y0:top_end, center_x] = low_value
    if bottom_start < y1:
        marked[bottom_start:y1, center_x] = low_value
    return marked


def _sample_plane_image(
    volume: torch.Tensor,
    labels: torch.Tensor,
    grid: torch.Tensor,
    embedding_similarity: torch.Tensor | None = None,
    center_marker_yx: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    image = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0, 0]
    encoded_labels = torch.full(labels.shape, 127.0, device=labels.device)
    encoded_labels = torch.where(
        labels == int(NEGATIVE_LABEL),
        torch.zeros_like(encoded_labels),
        encoded_labels,
    )
    encoded_labels = torch.where(
        labels == int(POSITIVE_LABEL),
        torch.full_like(encoded_labels, 255.0),
        encoded_labels,
    )
    sampled_labels = F.grid_sample(
        encoded_labels.unsqueeze(0).unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0, 0]
    in_bounds = (grid[0, 0].abs() <= 1.0).all(dim=-1)
    yy, xx = torch.meshgrid(
        torch.arange(sampled_labels.shape[0], device=sampled_labels.device),
        torch.arange(sampled_labels.shape[1], device=sampled_labels.device),
        indexing="ij",
    )
    outside_checker = torch.where(
        ((yy // 8 + xx // 8) % 2) == 0,
        torch.full_like(sampled_labels, 63.0),
        torch.full_like(sampled_labels, 191.0),
    )
    sampled_labels = torch.where(
        in_bounds,
        sampled_labels,
        outside_checker,
    )
    image_out = _normalize_image_for_tb(image)
    label_out = sampled_labels.round().clamp(0.0, 255.0).to(torch.uint8)
    if center_marker_yx is not None:
        image_out = _draw_center_cross(
            image_out,
            center_yx=center_marker_yx,
            low_value=0.0,
            high_value=1.0,
        )
        label_out = _draw_center_cross(
            label_out,
            center_yx=center_marker_yx,
            low_value=0.0,
            high_value=255.0,
        )
    images = {
        "image": image_out,
        "labels": label_out,
    }
    if embedding_similarity is not None:
        sampled_cos = F.grid_sample(
            embedding_similarity.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, 0]
        sampled_cos = torch.where(
            in_bounds,
            sampled_cos,
            (outside_checker / 127.5) - 1.0,
        )
        cos_out = ((sampled_cos.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(
            0.0, 1.0
        )
        if center_marker_yx is not None:
            cos_out = _draw_center_cross(
                cos_out,
                center_yx=center_marker_yx,
                low_value=0.0,
                high_value=1.0,
            )
        images["cos_emb_cp"] = cos_out
    return images


def _visualization_positive_sample_indices(
    batch: FiberTraceBatch, *, fallback_index: int
) -> list[int]:
    batch_size = int(batch.volume.shape[0])
    indices = [
        idx
        for idx, kind in enumerate(batch.crop_kinds)
        if idx < batch_size and str(kind) == "gt_control"
    ][:2]
    if indices:
        return indices
    if batch_size <= 0:
        return []
    return [max(0, min(int(fallback_index), batch_size - 1))]


def _embedding_similarity_to_reference(
    embedding: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    embedding = embedding.to(dtype=torch.float32)
    reference = reference.to(device=embedding.device, dtype=torch.float32)
    reference = reference / torch.linalg.vector_norm(reference).clamp_min(1e-6)
    embedding_n = embedding / torch.linalg.vector_norm(
        embedding, dim=0, keepdim=True
    ).clamp_min(1e-6)
    return torch.sum(embedding_n * reference.view(-1, 1, 1, 1), dim=0)


def _cos_similarity_to_tb_image(
    similarity: torch.Tensor,
    *,
    center_marker_yx: tuple[int, int] | None = None,
) -> torch.Tensor:
    image = ((similarity.to(dtype=torch.float32).clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(
        0.0, 1.0
    )
    if center_marker_yx is not None:
        image = _draw_center_cross(
            image,
            center_yx=center_marker_yx,
            low_value=0.0,
            high_value=1.0,
        )
    return image


def _principal_cosine_slices(
    similarity_zyx: torch.Tensor, cp_local_zyx: tuple[int, int, int]
) -> dict[str, torch.Tensor]:
    depth, height, width = (int(v) for v in similarity_zyx.shape[-3:])
    z = max(0, min(depth - 1, int(cp_local_zyx[0])))
    y = max(0, min(height - 1, int(cp_local_zyx[1])))
    x = max(0, min(width - 1, int(cp_local_zyx[2])))
    return {
        "principal_yx": _cos_similarity_to_tb_image(
            similarity_zyx[z, :, :], center_marker_yx=(y, x)
        ),
        "principal_zx": _cos_similarity_to_tb_image(
            similarity_zyx[:, y, :], center_marker_yx=(z, x)
        ),
        "principal_zy": _cos_similarity_to_tb_image(
            similarity_zyx[:, :, x], center_marker_yx=(z, y)
        ),
    }


def _log_stitched_training_sample_visualization(
    writer: Any,
    model: torch.nn.Module | None,
    batch: FiberTraceBatch,
    *,
    step: int,
    sample_indices: list[int],
    tag_prefix: str = "train_sample",
) -> None:
    if writer is None or not hasattr(writer, "add_image") or not sample_indices:
        return
    shape_zyx = tuple(int(v) for v in batch.labels.shape[-3:])
    depth, height, width = shape_zyx
    size = max(depth, height, width)
    device = batch.volume.device

    sample_payloads: list[dict[str, Any]] = []
    ideal_volumes: list[torch.Tensor] = []
    ideal_fw: list[torch.Tensor] = []
    for sample_index in sample_indices:
        cp_local_zyx = _sample_local_zyx_tuple(batch, sample_index)
        cp_local_xyz = torch.tensor(
            [float(cp_local_zyx[2]), float(cp_local_zyx[1]), float(cp_local_zyx[0])],
            device=device,
            dtype=torch.float32,
        )
        fw, normal_axis, right = _orthonormal_sample_frame(batch, sample_index)
        sample_payloads.append(
            {
                "sample_index": int(sample_index),
                "cp_local_zyx": cp_local_zyx,
                "cp_local_xyz": cp_local_xyz,
                "fw": fw,
                "planes": {
                    "side": (fw, normal_axis),
                    "top": (fw, right),
                    "cross": (right, normal_axis),
                },
            }
        )
        ideal_volumes.append(batch.volume[sample_index : sample_index + 1])
        ideal_fw.append(fw.to(device=device, dtype=batch.volume.dtype))

    embeddings: list[torch.Tensor | None] = [None] * len(sample_payloads)
    refs: list[torch.Tensor | None] = [None] * len(sample_payloads)
    if model is not None and ideal_volumes:
        volume = torch.cat(ideal_volumes, dim=0)
        cond_fw = torch.stack(ideal_fw, dim=0)
        was_training = bool(model.training)
        model.eval()
        try:
            ideal_outputs = model(volume, cond_fw)
        finally:
            if was_training:
                model.train()
        dense_embedding = ideal_outputs["embedding"].to(
            device=device, dtype=torch.float32
        )
        for pos, payload in enumerate(sample_payloads):
            embedding = dense_embedding[pos]
            cp_local_zyx = payload["cp_local_zyx"]
            embeddings[pos] = embedding
            refs[pos] = embedding[(slice(None),) + cp_local_zyx]

    stitched: dict[tuple[str, str], list[torch.Tensor]] = {}
    for pos, payload in enumerate(sample_payloads):
        sample_index = int(payload["sample_index"])
        volume = batch.volume[sample_index : sample_index + 1]
        labels = batch.labels[sample_index].to(device=device)
        embedding = embeddings[pos]
        own_ref = refs[pos]
        other_ref = None
        if len(refs) > 1:
            other_ref = refs[(pos + 1) % len(refs)]
        own_similarity = (
            None
            if embedding is None or own_ref is None
            else _embedding_similarity_to_reference(embedding, own_ref)
        )
        other_similarity = (
            None
            if embedding is None or other_ref is None
            else _embedding_similarity_to_reference(embedding, other_ref)
        )
        for plane_name, (axis_u, axis_v) in payload["planes"].items():
            grid = _slice_grid(
                shape_zyx=shape_zyx,
                center_xyz=payload["cp_local_xyz"],
                axis_u_xyz=axis_u,
                axis_v_xyz=axis_v,
                size=size,
                device=device,
            )
            images = _sample_plane_image(
                volume,
                labels,
                grid,
                embedding_similarity=own_similarity,
                center_marker_yx=(size // 2, size // 2),
            )
            if other_similarity is not None:
                images["cos_emb_other_cp"] = _sample_plane_image(
                    volume,
                    labels,
                    grid,
                    embedding_similarity=other_similarity,
                    center_marker_yx=(size // 2, size // 2),
                )["cos_emb_cp"]
            for image_name, image in images.items():
                stitched.setdefault((plane_name, image_name), []).append(image)
        if own_similarity is not None:
            for plane_name, image in _principal_cosine_slices(
                own_similarity, payload["cp_local_zyx"]
            ).items():
                stitched.setdefault((plane_name, "cos_emb_cp"), []).append(image)

    for (plane_name, image_name), images in stitched.items():
        combined = torch.cat(images, dim=-1)
        writer.add_image(
            f"{tag_prefix}/{plane_name}/{image_name}",
            combined.detach().cpu().unsqueeze(0),
            int(step),
        )


def _log_training_sample_visualization(
    writer: Any,
    model: torch.nn.Module | None,
    batch: FiberTraceBatch,
    *,
    step: int,
    sample_index: int = 0,
    tag_prefix: str = "train_sample",
) -> None:
    _log_stitched_training_sample_visualization(
        writer,
        model,
        batch,
        step=step,
        sample_indices=_visualization_positive_sample_indices(
            batch, fallback_index=sample_index
        ),
        tag_prefix=tag_prefix,
    )


def _save_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    steps: int,
    metric_name: str,
    metric_value: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": _json_safe(config),
            "step": int(step),
            "steps": int(steps),
            "metric_name": str(metric_name),
            "metric_value": float(metric_value),
        },
        path,
    )


def _dedupe_chunk_requests(
    requests: list[ZarrChunkRequest],
) -> list[ZarrChunkRequest]:
    seen: set[tuple[str, str]] = set()
    unique: list[ZarrChunkRequest] = []
    for request in requests:
        key = (request.store_identity, request.key)
        if key in seen:
            continue
        seen.add(key)
        unique.append(request)
    return unique


def _fetch_prefetch_chunk(request: ZarrChunkRequest) -> tuple[str, int, float]:
    start = time.perf_counter()
    try:
        data = request.store[request.key]
    except KeyError:
        return "missing", 0, (time.perf_counter() - start) * 1000.0
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    return "ok", len(data), (time.perf_counter() - start) * 1000.0


def _prefetch_chunk_is_cached(request: ZarrChunkRequest) -> bool:
    cache_dir = getattr(request.store, "_cache_dir", None)
    if cache_dir is None:
        return False
    cached = os.path.join(str(cache_dir), request.key)
    if os.path.isfile(cached):
        return True
    marker_suffix = getattr(request.store, "_NEGATIVE_MARKER_SUFFIX", ".__notfound__")
    return os.path.isfile(cached + str(marker_suffix))


def _format_eta(seconds: float | None) -> str:
    if seconds is None or not bool(seconds >= 0.0) or seconds == float("inf"):
        return "--:--"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _prefetch_progress_line(
    *,
    done: int,
    total: int,
    bytes_read: int,
    started_at: float,
    missing: int,
    errors: int,
) -> str:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    mib = float(bytes_read) / (1024.0 * 1024.0)
    mib_s = mib / elapsed
    eta_seconds = None
    if done > 0:
        eta_seconds = elapsed * float(max(total - done, 0)) / float(done)
    width = 24
    filled = int(width * done / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"prefetch [{bar}] {done}/{total} chunks "
        f"{mib:.1f} MiB {mib_s:.1f} MiB/s "
        f"eta={_format_eta(eta_seconds)} missing={missing} errors={errors}"
    )


def run_prefetch(
    config: dict[str, Any],
    *,
    max_workers: int | None = None,
) -> dict[str, Any]:
    _reject_legacy_checkpoint_path(config)
    steps = int(config.get("num_steps", config.get("steps", 1)))
    if steps <= 0:
        raise ValueError(f"num_steps/steps must be positive, got {steps}")
    configured_workers = (
        max_workers
        if max_workers is not None
        else config.get("prefetch_workers", 16)
    )
    if configured_workers is None:
        configured_workers = 16
    worker_count = int(configured_workers)
    worker_count = max(1, min(16, worker_count))

    debug_table_state = FiberTraceDebugTableState()
    batch_builder = FiberTraceBatchBuilder(
        config, debug_table_state=debug_table_state
    )
    test_config = _make_test_config(config)
    test_batch_builder = (
        FiberTraceBatchBuilder(test_config, debug_table_state=debug_table_state)
        if test_config is not None
        else None
    )

    requests: list[ZarrChunkRequest] = []
    sample_limit = getattr(batch_builder, "sample_limit", None)
    train_prefetch_steps = (
        min(steps, int(sample_limit)) if sample_limit is not None else steps
    )
    for step in range(1, train_prefetch_steps + 1):
        requests.extend(
            batch_builder.prefetch_chunk_requests_for_iteration(iteration=step)
        )
    if test_batch_builder is not None:
        test_sample_count = max(1, int(config.get("test_sample_count", 1)))
        test_start_iteration = int(config.get("test_start_iteration", 1))
        for offset in range(test_sample_count):
            requests.extend(
                test_batch_builder.prefetch_chunk_requests_for_iteration(
                    iteration=test_start_iteration + offset
                )
            )

    unique_requests = _dedupe_chunk_requests(requests)
    cached_requests = [
        request for request in unique_requests if _prefetch_chunk_is_cached(request)
    ]
    pending_requests = [
        request for request in unique_requests if not _prefetch_chunk_is_cached(request)
    ]
    total = len(pending_requests)
    print(
        f"prefetch chunks: generated={len(requests)} unique={len(unique_requests)} "
        f"cached={len(cached_requests)} pending={total} "
        f"steps={steps} sample_steps={train_prefetch_steps} workers={worker_count}",
        flush=True,
    )
    if total == 0:
        return {
            "generated_chunks": len(requests),
            "unique_chunks": len(unique_requests),
            "cached_chunks": len(cached_requests),
            "pending_chunks": 0,
            "bytes": 0,
            "missing": 0,
            "errors": 0,
            "elapsed_seconds": 0.0,
            "mib_s": 0.0,
        }

    started_at = time.perf_counter()
    done = 0
    bytes_read = 0
    missing = 0
    error_count = 0
    errors: list[str] = []
    next_print_at = 0.0
    pool_workers = min(worker_count, total)
    with ThreadPoolExecutor(max_workers=pool_workers) as executor:
        futures = [
            executor.submit(_fetch_prefetch_chunk, request)
            for request in pending_requests
        ]
        for future in as_completed(futures):
            done += 1
            try:
                status, byte_count, _elapsed_ms = future.result()
            except Exception as exc:
                status = "error"
                byte_count = 0
                error_count += 1
                if len(errors) < 5:
                    errors.append(f"{type(exc).__name__}: {exc}")
            if status == "missing":
                missing += 1
            elif status == "error":
                pass
            else:
                bytes_read += int(byte_count)

            now = time.perf_counter()
            if now >= next_print_at or done == total:
                print(
                    "\r"
                    + _prefetch_progress_line(
                        done=done,
                        total=total,
                        bytes_read=bytes_read,
                        started_at=started_at,
                        missing=missing,
                        errors=error_count,
                    ),
                    end="",
                    flush=True,
                )
                next_print_at = now + 0.25
    print("", flush=True)

    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    mib = float(bytes_read) / (1024.0 * 1024.0)
    if error_count:
        preview = "; ".join(errors)
        raise RuntimeError(f"prefetch failed for {error_count} chunks: {preview}")

    return {
        "generated_chunks": len(requests),
        "unique_chunks": len(unique_requests),
        "cached_chunks": len(cached_requests),
        "pending_chunks": total,
        "bytes": bytes_read,
        "missing": missing,
        "errors": error_count,
        "elapsed_seconds": elapsed_seconds,
        "mib_s": mib / elapsed_seconds,
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if "seed" in config:
        torch.manual_seed(int(config["seed"]))
    _reject_legacy_checkpoint_path(config)

    debug_table_state = FiberTraceDebugTableState()
    batch_builder = FiberTraceBatchBuilder(
        config, debug_table_state=debug_table_state
    )
    test_config = _make_test_config(config)
    test_batch_builder = (
        FiberTraceBatchBuilder(test_config, debug_table_state=debug_table_state)
        if test_config is not None
        else None
    )
    run_dir, snapshot_dir = _resolve_run_layout(config)
    current_snapshot = snapshot_dir / "current.pt"
    best_snapshot = snapshot_dir / "best.pt"
    writer = _make_summary_writer(
        run_dir, enabled=bool(config.get("tensorboard_enabled", True))
    )

    model = build_fiber_trace_model(config).to(device)
    opt_cfg = dict(config.get("optimizer", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("learning_rate", opt_cfg.get("lr", 1e-3))),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )

    loss_kwargs = _loss_kwargs(config)
    steps = int(config.get("num_steps", config.get("steps", 1)))
    log_every = max(1, int(config.get("log_every", 100)))
    test_every = int(config.get("test_every", log_every))
    if test_every < 0:
        raise ValueError("test_every must be >= 0")
    test_sample_count = int(config.get("test_sample_count", 1))
    if test_sample_count <= 0:
        raise ValueError("test_sample_count must be positive")
    test_start_iteration = int(config.get("test_start_iteration", 1))
    sample_visualization_every = int(config.get("sample_visualization_every", 10000))
    if sample_visualization_every < 0:
        raise ValueError("sample_visualization_every must be >= 0")
    sample_visualization_index = int(config.get("sample_visualization_index", 0))
    test_visualization_every = int(
        config.get("test_visualization_every", sample_visualization_every)
    )
    if test_visualization_every < 0:
        raise ValueError("test_visualization_every must be >= 0")
    best_metric = float("inf")
    test_enabled = test_batch_builder is not None and test_every > 0
    best_metric_name = "test/total" if test_enabled else "train/total"
    debug_timing = bool(config.get("debug_sampling", False) or config.get("debug_cache", False))
    debug_step_rows = 0
    debug_step_header_every = int(config.get("debug_step_header_every", 20))
    debug_step_legend_printed = False

    try:
        if writer is not None:
            writer.add_text(
                "config/json", f"```json\n{_config_json_text(config)}\n```", 0
            )
            writer.flush()

        model.train()
        for step in range(1, steps + 1):
            step_t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses, train_batch, train_outputs, step_timings = _compute_losses(
                model, batch_builder, device, config, loss_kwargs, iteration=step
            )
            losses.total.backward()
            optimizer.step()

            should_log = step % log_every == 0 or step == steps
            should_test = test_enabled and (step % test_every == 0 or step == steps)
            should_visualize = (
                writer is not None
                and sample_visualization_every > 0
                and step % sample_visualization_every == 0
            )
            should_visualize_test = (
                writer is not None
                and test_batch_builder is not None
                and test_visualization_every > 0
                and step % test_visualization_every == 0
            )

            train_row_scalars = (
                _loss_scalars(losses)
                if debug_timing or should_log or should_test
                else None
            )
            train_scalars = train_row_scalars if should_log else None
            step_ms = (time.perf_counter() - step_t0) * 1000.0
            step_timings["train_ms"] = max(
                0.0, step_ms - float(step_timings.get("data_ms", 0.0))
            )
            test_scalars = None
            if should_test and test_batch_builder is not None:
                test_scalars = _compute_test_scalars(
                    model,
                    test_batch_builder,
                    device,
                    test_config,
                    loss_kwargs,
                    start_iteration=test_start_iteration,
                    sample_count=test_sample_count,
                )

            if should_log and train_scalars is not None:
                _log_scalars(writer, "train", train_scalars, step)
            if test_scalars is not None:
                _log_scalars(writer, "test", test_scalars, step)
            if should_visualize:
                with torch.no_grad():
                    _log_training_sample_visualization(
                        writer,
                        model,
                        train_batch,
                        step=step,
                        sample_index=sample_visualization_index,
                    )
            if should_visualize_test and test_batch_builder is not None:
                with torch.no_grad():
                    test_batch = _sample_classified_batch(
                        test_batch_builder,
                        device,
                        test_config,
                        iteration=test_start_iteration,
                    )
                    _log_training_sample_visualization(
                        writer,
                        model,
                        test_batch,
                        step=step,
                        sample_index=sample_visualization_index,
                        tag_prefix="test_sample",
                    )
            if writer is not None and (
                should_log or should_test or should_visualize or should_visualize_test
            ):
                writer.flush()

            if debug_timing:
                _print_step_timing(
                    step=step,
                    timings=step_timings,
                    train_scalars=train_row_scalars,
                    test_scalars=test_scalars,
                    row_index=debug_step_rows,
                    header_every=debug_step_header_every,
                    print_legend=not debug_step_legend_printed,
                )
                debug_step_rows += 1
                debug_step_legend_printed = True

            should_snapshot = should_test or (not test_enabled and should_log)
            if not should_snapshot or train_row_scalars is None:
                continue

            metric_value = (
                test_scalars["total"]
                if test_scalars is not None
                else train_row_scalars["total"]
            )
            _save_snapshot(
                current_snapshot,
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                steps=steps,
                metric_name=best_metric_name,
                metric_value=metric_value,
            )
            if metric_value < best_metric:
                best_metric = metric_value
                _save_snapshot(
                    best_snapshot,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    steps=steps,
                    metric_name=best_metric_name,
                    metric_value=metric_value,
                )

            message = (
                f"step={step} train_total={train_row_scalars['total']:.6f} "
                f"train_contrastive={train_row_scalars['contrastive']:.6f}"
            )
            if test_scalars is not None:
                message += (
                    f" test_total={test_scalars['total']:.6f} "
                    f"test_contrastive={test_scalars['contrastive']:.6f}"
                )
            if not debug_timing:
                print(message, flush=True)
    finally:
        if writer is not None:
            writer.close()

    return {
        "run_dir": str(run_dir),
        "snapshot_dir": str(snapshot_dir),
        "current_snapshot": str(current_snapshot),
        "best_snapshot": str(best_snapshot),
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the fiber tracing MVP model.")
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help=(
            "Generate the zarr chunk list for this training config and download "
            "those chunks into the configured cache instead of training."
        ),
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=None,
        help="Maximum parallel chunk downloads for --prefetch, capped at 16.",
    )
    parser.add_argument("config", help="Path to a fiber trace training config JSON.")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if args.prefetch:
        run_prefetch(config, max_workers=args.prefetch_workers)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
