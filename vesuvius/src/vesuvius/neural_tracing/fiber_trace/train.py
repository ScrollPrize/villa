from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace.dataset import (
    FiberTraceBatch,
    FiberTraceBatchBuilder,
)
from vesuvius.neural_tracing.fiber_trace.labels import (
    IGNORE_ID,
    IGNORE_INDEX,
    NEGATIVE_LABEL,
    NEGATIVE_ONLY_ID,
    POSITIVE_LABEL,
)
from vesuvius.neural_tracing.fiber_trace.losses import compute_fiber_trace_loss
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
        "fw_weight": float(loss_cfg.get("fw_weight", 1.0)),
        "up_weight": float(loss_cfg.get("up_weight", 1.0)),
        "max_contrastive_samples": int(loss_cfg.get("max_contrastive_samples", 4096)),
    }


def _normalize_vectors_torch(vec: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    return vec.to(dtype=torch.float32) / torch.linalg.vector_norm(
        vec.to(dtype=torch.float32), dim=-1, keepdim=True
    ).clamp_min(float(eps))


def _fallback_up_vectors_torch(fw_xyz: torch.Tensor) -> torch.Tensor:
    fw = _normalize_vectors_torch(fw_xyz)
    helper_x = torch.zeros_like(fw)
    helper_x[..., 0] = 1.0
    helper_y = torch.zeros_like(fw)
    helper_y[..., 1] = 1.0
    helper = torch.where((fw[..., 0].abs() < 0.9)[..., None], helper_x, helper_y)
    return _normalize_vectors_torch(torch.cross(fw, helper, dim=-1))


def _construct_up_vectors_torch(
    fw_xyz: torch.Tensor,
    normal_xyz: torch.Tensor | None,
    *,
    allow_arbitrary_up_fallback: bool,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    fw = _normalize_vectors_torch(fw_xyz, eps=eps)
    if normal_xyz is None:
        if not allow_arbitrary_up_fallback:
            raise ValueError("normal_xyz is required for fiber trace label geometry")
        return _fallback_up_vectors_torch(fw), torch.ones(
            fw.shape[:-1], device=fw.device, dtype=torch.bool
        )

    normal = _normalize_vectors_torch(normal_xyz, eps=eps)
    up = normal - torch.sum(normal * fw, dim=-1, keepdim=True) * fw
    up_norm = torch.linalg.vector_norm(up, dim=-1, keepdim=True)
    valid = torch.isfinite(normal_xyz).all(dim=-1) & (up_norm[..., 0] > float(eps))
    up = up / up_norm.clamp_min(float(eps))
    if allow_arbitrary_up_fallback:
        fallback = _fallback_up_vectors_torch(fw)
        up = torch.where(valid[..., None], up, fallback)
        valid = torch.ones_like(valid)
    return up, valid


def _folded_frame_agreement_torch(
    cond_fw_xyz: torch.Tensor,
    cond_up_xyz: torch.Tensor | None,
    target_fw_xyz: torch.Tensor,
    target_up_xyz: torch.Tensor | None,
    *,
    target_up_valid: torch.Tensor | None,
) -> torch.Tensor:
    cond_fw = _normalize_vectors_torch(cond_fw_xyz)
    target_fw = _normalize_vectors_torch(target_fw_xyz)
    fw_agreement = torch.sum(cond_fw * target_fw, dim=-1).abs()
    if cond_up_xyz is None or target_up_xyz is None:
        return fw_agreement.clamp(0.0, 1.0)

    cond_up_raw = cond_up_xyz.to(dtype=torch.float32)
    target_up_raw = target_up_xyz.to(dtype=torch.float32)
    cond_up = _normalize_vectors_torch(cond_up_raw)
    target_up = _normalize_vectors_torch(target_up_raw)
    up_agreement = torch.sum(cond_up * target_up, dim=-1).abs()
    frame_agreement = torch.minimum(fw_agreement, up_agreement)
    up_ok = (torch.linalg.vector_norm(cond_up_raw, dim=-1) > 1e-6) & (
        torch.linalg.vector_norm(target_up_raw, dim=-1) > 1e-6
    )
    if target_up_valid is not None:
        up_ok = up_ok & target_up_valid
    return torch.where(up_ok, frame_agreement, fw_agreement).clamp(0.0, 1.0)


def _decode_lasagna_normals_torch(
    nx_values: torch.Tensor, ny_values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    nx = (nx_values.to(dtype=torch.float32) - 128.0) / 127.0
    ny = (ny_values.to(dtype=torch.float32) - 128.0) / 127.0
    nz = torch.sqrt((1.0 - nx * nx - ny * ny).clamp_min(0.0))
    normal = torch.stack([nx, ny, nz], dim=-1)
    norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    valid = torch.isfinite(normal).all(dim=-1) & (norm[..., 0] > 1e-6)
    normal = normal / norm.clamp_min(1e-6)
    normal = torch.where(valid[..., None], normal, torch.zeros_like(normal))
    return normal, valid


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
    allow_fallback = bool(config.get("allow_arbitrary_up_fallback", False))
    if batch.nx_values is None or batch.ny_values is None:
        if not allow_fallback:
            raise ValueError("Lasagna nx/ny normal channels are required")
        normal_xyz = None
        normal_valid = torch.ones_like(valid_mask, dtype=torch.bool)
    else:
        normal_xyz, normal_valid = _decode_lasagna_normals_torch(
            batch.nx_values, batch.ny_values
        )
        valid_mask = valid_mask & normal_valid

    labels = torch.full_like(batch.labels, int(IGNORE_INDEX))
    target_id = torch.full_like(batch.target_id, int(IGNORE_ID))
    target_fw_xyz = torch.zeros_like(batch.target_fw_xyz)
    target_up_xyz = torch.zeros_like(batch.target_up_xyz)
    target_up_valid = torch.zeros_like(batch.target_up_valid)

    positive_radius = float(config.get("positive_radius", 1.5))
    ignore_radius = float(config.get("ignore_radius", max(3.0, positive_radius)))
    normal_plane_jitter = float(
        config.get("normal_plane_jitter_voxels", positive_radius)
    )
    normal_perp_jitter = float(
        config.get("normal_perpendicular_jitter_voxels", positive_radius)
    )
    negative_cone_distance = float(
        config.get("negative_cone_distance_voxels", ignore_radius)
    )
    positive_cosine = float(config.get("positive_cosine", 0.8660254037844386))
    negative_cosine = float(config.get("negative_cosine", 0.5))
    degenerate_up_policy = str(config.get("degenerate_up_policy", "invalid"))

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

        normal_i = None if normal_xyz is None else normal_xyz[idx]
        if normal_i is None:
            normal_for_geometry, normal_geometry_valid = _construct_up_vectors_torch(
                tangent_xyz,
                None,
                allow_arbitrary_up_fallback=allow_fallback,
            )
        else:
            normal_for_geometry = _normalize_vectors_torch(normal_i)
            normal_geometry_valid = torch.isfinite(normal_i).all(dim=-1) & (
                torch.linalg.vector_norm(normal_i, dim=-1) > 1e-6
            )

        valid = valid_mask[idx] & normal_geometry_valid
        target_up_i, target_up_valid_i = _construct_up_vectors_torch(
            tangent_xyz,
            normal_i,
            allow_arbitrary_up_fallback=allow_fallback,
        )
        target_up_valid_i = target_up_valid_i & valid
        agreement = _folded_frame_agreement_torch(
            batch.cond_fw_xyz[idx],
            batch.cond_up_xyz[idx],
            tangent_xyz,
            target_up_i,
            target_up_valid=target_up_valid_i,
        )

        offset_xyz = coords_xyz - closest_xyz
        plane_offset = torch.sum(offset_xyz * normal_for_geometry, dim=-1)
        in_plane_offset = offset_xyz - plane_offset[..., None] * normal_for_geometry
        in_plane_distance = torch.linalg.vector_norm(in_plane_offset, dim=-1)
        perpendicular_distance = plane_offset.abs()

        aligned = agreement >= (positive_cosine - 1e-6)
        disagreed = agreement <= (negative_cosine + 1e-6)
        positive_zone = (
            valid
            & (in_plane_distance <= normal_plane_jitter)
            & (perpendicular_distance <= normal_perp_jitter)
        )

        lateral = torch.cross(normal_for_geometry, tangent_xyz, dim=-1)
        lateral_norm = torch.linalg.vector_norm(lateral, dim=-1, keepdim=True)
        lateral_valid = lateral_norm[..., 0] > 1e-6
        lateral = lateral / lateral_norm.clamp_min(1e-6)
        in_plane_norm = torch.linalg.vector_norm(in_plane_offset, dim=-1, keepdim=True)
        in_plane_dir = in_plane_offset / in_plane_norm.clamp_min(1e-6)
        lateral_alignment = torch.sum(in_plane_dir * lateral, dim=-1).abs()
        inside_lateral_cone = (
            lateral_valid
            & (in_plane_norm[..., 0] > 1e-6)
            & (lateral_alignment >= 0.7071067811865476)
        )

        cone_negative = (
            valid
            & inside_lateral_cone
            & (perpendicular_distance >= negative_cone_distance)
        )
        direction_negative = positive_zone & disagreed
        positive = positive_zone & aligned
        negative = (cone_negative | direction_negative) & ~positive

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

        if degenerate_up_policy == "raise" and bool((positive & ~target_up_valid_i).any()):
            raise ValueError("positive voxels contain degenerate Lasagna normal up vectors")

        target_fw_xyz[idx] = tangent_xyz.movedim(-1, 0)
        target_up_xyz[idx] = target_up_i.movedim(-1, 0)
        target_up_valid[idx] = target_up_valid_i

    return batch.with_classification(
        valid_mask=valid_mask,
        labels=labels,
        target_id=target_id,
        target_fw_xyz=target_fw_xyz,
        target_up_xyz=target_up_xyz,
        target_up_valid=target_up_valid,
    )


def _compute_losses(
    model: torch.nn.Module,
    batch_builder: FiberTraceBatchBuilder,
    device: torch.device,
    config: dict[str, Any],
    loss_kwargs: dict[str, Any],
) -> tuple[Any, FiberTraceBatch]:
    batch = batch_builder.sample_batch().to(device)
    batch = classify_batch_on_device(batch, config)
    outputs = model(batch.volume, batch.cond_fw_xyz, batch.cond_up_xyz)
    return compute_fiber_trace_loss(outputs, batch, **loss_kwargs), batch


def _loss_scalars(losses) -> dict[str, float]:
    return {
        "total": float(losses.total.detach().cpu()),
        "contrastive": float(losses.contrastive.detach().cpu()),
        "fw": float(losses.fw.detach().cpu()),
        "up": float(losses.up.detach().cpu()),
    }


def _log_scalars(writer: Any, prefix: str, scalars: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for name, value in scalars.items():
        writer.add_scalar(f"{prefix}/{name}", value, step)


def _normalize_vector(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    vec = vec.to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(vec)
    if (
        bool(torch.isfinite(norm).detach().cpu())
        and float(norm.detach().cpu()) > 1e-6
    ):
        return vec / norm.clamp_min(1e-6)
    return fallback.to(device=vec.device, dtype=torch.float32)


def _orthonormal_sample_frame(
    batch: FiberTraceBatch, sample_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_index = int(sample_index)
    device = batch.volume.device
    shape_zyx = batch.labels.shape[-3:]
    center_zyx = tuple(min(int(size) - 1, int(size) // 2) for size in shape_zyx)
    fallback_fw = _normalize_vector(
        batch.cond_fw_xyz[sample_index],
        torch.tensor([1.0, 0.0, 0.0], device=device),
    )
    fallback_up = _normalize_vector(
        batch.cond_up_xyz[sample_index],
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )
    fw = _normalize_vector(
        batch.target_fw_xyz[(sample_index, slice(None)) + center_zyx],
        fallback_fw,
    )
    up_valid = bool(batch.target_up_valid[(sample_index,) + center_zyx].detach().cpu())
    up_candidate = batch.target_up_xyz[(sample_index, slice(None)) + center_zyx]
    up_source = up_candidate if up_valid else fallback_up
    up = _normalize_vector(up_source, fallback_up)
    up = up - torch.sum(up * fw) * fw
    up = _normalize_vector(up, fallback_up)
    right = torch.cross(fw, up, dim=0)
    if float(torch.linalg.vector_norm(right).detach().cpu()) <= 1e-6:
        fallback_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
        if abs(float(torch.sum(fw * fallback_axis).detach().cpu())) > 0.9:
            fallback_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
        up = _normalize_vector(
            fallback_axis - torch.sum(fallback_axis * fw) * fw,
            fallback_up,
        )
        right = torch.cross(fw, up, dim=0)
    right = _normalize_vector(right, torch.tensor([0.0, 0.0, 1.0], device=device))
    up = _normalize_vector(torch.cross(right, fw, dim=0), up)
    return fw, up, right


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
    coords = coords - (float(size) - 1.0) * 0.5
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


def _sample_plane_image(
    volume: torch.Tensor,
    labels: torch.Tensor,
    grid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    image = F.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0, 0]
    masks = {
        "positive": labels == int(POSITIVE_LABEL),
        "undef": (labels != int(POSITIVE_LABEL)) & (labels != int(NEGATIVE_LABEL)),
        "negative": labels == int(NEGATIVE_LABEL),
    }
    sampled_masks = {}
    for name, mask in masks.items():
        sampled_masks[name] = F.grid_sample(
            mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, 0]
    return {"image": _normalize_image_for_tb(image), **sampled_masks}


def _log_training_sample_visualization(
    writer: Any,
    batch: FiberTraceBatch,
    *,
    step: int,
    sample_index: int = 0,
) -> None:
    if writer is None or not hasattr(writer, "add_image"):
        return
    batch_size = int(batch.volume.shape[0])
    if batch_size <= 0:
        return
    sample_index = max(0, min(int(sample_index), batch_size - 1))
    shape_zyx = tuple(int(v) for v in batch.labels.shape[-3:])
    depth, height, width = shape_zyx
    size = max(depth, height, width)
    device = batch.volume.device
    center_xyz = torch.tensor(
        [float(width // 2), float(height // 2), float(depth // 2)],
        device=device,
        dtype=torch.float32,
    )
    fw, up, right = _orthonormal_sample_frame(batch, sample_index)
    planes = {
        "side": (fw, up),
        "top": (fw, right),
        "cross": (right, up),
    }
    volume = batch.volume[sample_index : sample_index + 1]
    labels = batch.labels[sample_index].to(device=device)
    for plane_name, (axis_u, axis_v) in planes.items():
        grid = _slice_grid(
            shape_zyx=shape_zyx,
            center_xyz=center_xyz,
            axis_u_xyz=axis_u,
            axis_v_xyz=axis_v,
            size=size,
            device=device,
        )
        images = _sample_plane_image(volume, labels, grid)
        for image_name, image in images.items():
            writer.add_image(
                f"train_sample/{plane_name}/{image_name}",
                image.detach().cpu().unsqueeze(0),
                int(step),
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


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if "seed" in config:
        torch.manual_seed(int(config["seed"]))
    _reject_legacy_checkpoint_path(config)

    batch_builder = FiberTraceBatchBuilder(config)
    test_config = _make_test_config(config)
    test_batch_builder = (
        FiberTraceBatchBuilder(test_config) if test_config is not None else None
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
    sample_visualization_every = int(config.get("sample_visualization_every", 10000))
    if sample_visualization_every < 0:
        raise ValueError("sample_visualization_every must be >= 0")
    sample_visualization_index = int(config.get("sample_visualization_index", 0))
    best_metric = float("inf")
    best_metric_name = (
        "test/total" if test_batch_builder is not None else "train/total"
    )

    try:
        if writer is not None:
            writer.add_text(
                "config/json", f"```json\n{_config_json_text(config)}\n```", 0
            )
            writer.flush()

        model.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses, train_batch = _compute_losses(
                model, batch_builder, device, config, loss_kwargs
            )
            losses.total.backward()
            optimizer.step()

            should_log = step % log_every == 0 or step == steps
            should_visualize = (
                writer is not None
                and sample_visualization_every > 0
                and step % sample_visualization_every == 0
            )
            if not should_log and not should_visualize:
                continue

            train_scalars = _loss_scalars(losses) if should_log else None
            test_scalars = None
            if should_log and test_batch_builder is not None:
                model.eval()
                with torch.no_grad():
                    test_losses, _ = _compute_losses(
                        model, test_batch_builder, device, test_config, loss_kwargs
                    )
                test_scalars = _loss_scalars(test_losses)
                model.train()

            if should_log and train_scalars is not None:
                _log_scalars(writer, "train", train_scalars, step)
                if test_scalars is not None:
                    _log_scalars(writer, "test", test_scalars, step)
            if should_visualize:
                with torch.no_grad():
                    _log_training_sample_visualization(
                        writer,
                        train_batch,
                        step=step,
                        sample_index=sample_visualization_index,
                    )
            if writer is not None:
                writer.flush()

            if not should_log or train_scalars is None:
                continue

            metric_value = (
                test_scalars["total"]
                if test_scalars is not None
                else train_scalars["total"]
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
                f"step={step} train_total={train_scalars['total']:.6f} "
                f"train_contrastive={train_scalars['contrastive']:.6f} "
                f"train_fw={train_scalars['fw']:.6f} "
                f"train_up={train_scalars['up']:.6f}"
            )
            if test_scalars is not None:
                message += (
                    f" test_total={test_scalars['total']:.6f} "
                    f"test_contrastive={test_scalars['contrastive']:.6f} "
                    f"test_fw={test_scalars['fw']:.6f} "
                    f"test_up={test_scalars['up']:.6f}"
                )
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
    parser.add_argument("config", help="Path to a fiber trace training config JSON.")
    args = parser.parse_args(argv)
    run_training(_load_config(args.config))


if __name__ == "__main__":
    main()
