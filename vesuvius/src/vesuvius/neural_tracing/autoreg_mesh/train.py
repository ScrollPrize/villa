from __future__ import annotations

import functools
import importlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.neural_tracing.autoreg_mesh.config import load_autoreg_mesh_config, validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.dataset import AutoregMeshDataset, autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.infer import infer_autoreg_mesh
from vesuvius.neural_tracing.autoreg_mesh.losses import (
    _boundary_touch_fraction_from_sequence,
    _coarse_accuracy_metrics,
    _first_strip_wrong_side_rate_from_sequence,
    _invalid_vertex_fraction_from_sequence,
    _l1_xyz_metric,
    _pred_oob_fraction_from_sequence,
    _seam_edge_error_from_sequence,
    _triangle_flip_rate_from_sequence,
    compute_autoreg_mesh_losses,
)
from vesuvius.neural_tracing.autoreg_mesh.model import AutoregMeshModel
from vesuvius.neural_tracing.autoreg_mesh.serialization import deserialize_continuation_grid


@dataclass
class _DistributedRuntime:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None
    initialized_process_group: bool

    @property
    def is_main_process(self) -> bool:
        return int(self.rank) == 0


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _initialize_distributed_runtime(device: str | torch.device | None = None) -> _DistributedRuntime:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    requested_device = torch.device(device) if device is not None else None
    if env_world_size <= 1:
        runtime_device = requested_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _DistributedRuntime(
            is_distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=runtime_device,
            backend=None,
            initialized_process_group=False,
        )

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if requested_device is not None and requested_device.type != "cuda":
        runtime_device = requested_device
        backend = "gloo"
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        runtime_device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        runtime_device = torch.device("cpu")
        backend = "gloo"

    initialized = False
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required when WORLD_SIZE > 1")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        initialized = True
    return _DistributedRuntime(
        is_distributed=True,
        rank=int(dist.get_rank()),
        local_rank=int(local_rank),
        world_size=int(dist.get_world_size()),
        device=runtime_device,
        backend=str(backend),
        initialized_process_group=initialized,
    )


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _wrap_model_for_ddp(model: AutoregMeshModel, runtime: _DistributedRuntime):
    if not runtime.is_distributed:
        return model
    ddp_kwargs = {
        "device_ids": [runtime.local_rank] if runtime.device.type == "cuda" else None,
        "output_device": runtime.local_rank if runtime.device.type == "cuda" else None,
        "find_unused_parameters": False,
        "broadcast_buffers": False,
    }
    return DDP(model, **ddp_kwargs)


def _seed_worker(worker_id: int, *, base_seed: int, rank: int) -> None:
    worker_seed = int(base_seed) + (int(rank) * 1000) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def _metric_dict_mean_across_ranks(
    metrics: dict[str, float],
    *,
    device: torch.device,
    runtime: _DistributedRuntime,
) -> dict[str, float]:
    if not runtime.is_distributed or not metrics:
        return dict(metrics)
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(runtime.world_size)
    return {key: float(value.item()) for key, value in zip(keys, values, strict=True)}


def _distributed_all_finite(value: float, *, device: torch.device, runtime: _DistributedRuntime) -> bool:
    finite = torch.tensor(1 if np.isfinite(float(value)) else 0, device=device, dtype=torch.int32)
    if runtime.is_distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(int(finite.item()) == 1)


def _maybe_barrier(runtime: _DistributedRuntime) -> None:
    if runtime.is_distributed:
        dist.barrier()


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif key == "prompt_tokens":
            moved[key] = {
                inner_key: inner_value.to(device) if torch.is_tensor(inner_value) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def _next_batch(iterator, dataloader, *, on_reset=None):
    try:
        batch = next(iterator)
    except StopIteration:
        if on_reset is not None:
            on_reset()
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


@dataclass
class _SpatialSplitRecord:
    sample_index: int
    source_key: tuple[Any, ...]
    world_bbox: tuple[float, float, float, float, float, float]
    chunk_id: tuple[int, int, int] | None


def _split_indices(num_items: int, *, seed: int, val_fraction: float) -> tuple[list[int], list[int]]:
    if num_items <= 0:
        raise ValueError("autoreg_mesh training requires a non-empty dataset")
    if num_items < 2 or float(val_fraction) <= 0.0:
        return list(range(num_items)), []

    num_val = int(round(num_items * float(val_fraction)))
    num_val = max(1, min(num_val, num_items - 1))
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(num_items).tolist()
    return indices[num_val:], indices[:num_val]


def _patch_source_key(patch) -> tuple[Any, ...]:
    volume = getattr(patch, "volume", None)
    volume_store = getattr(volume, "store", None)
    volume_path = getattr(volume, "path", None)
    store_path = getattr(volume_store, "path", None)
    if volume_path is not None:
        source_id = str(volume_path)
    elif store_path is not None:
        source_id = str(store_path)
    else:
        source_id = f"volume_object:{id(volume)}"
    return (source_id, int(getattr(patch, "scale", 0)))


def _expanded_bboxes_overlap(
    bbox_a: tuple[float, float, float, float, float, float],
    bbox_b: tuple[float, float, float, float, float, float],
    *,
    margin_voxels: float,
) -> bool:
    margin = float(margin_voxels)
    return (
        (float(bbox_a[0]) - margin) < (float(bbox_b[1]) + margin) and
        (float(bbox_b[0]) - margin) < (float(bbox_a[1]) + margin) and
        (float(bbox_a[2]) - margin) < (float(bbox_b[3]) + margin) and
        (float(bbox_b[2]) - margin) < (float(bbox_a[3]) + margin) and
        (float(bbox_a[4]) - margin) < (float(bbox_b[5]) + margin) and
        (float(bbox_b[4]) - margin) < (float(bbox_a[5]) + margin)
    )


def _build_spatial_split_records(dataset: AutoregMeshDataset) -> list[_SpatialSplitRecord]:
    records: list[_SpatialSplitRecord] = []
    for sample_idx, sample_key in enumerate(dataset.sample_index):
        patch_idx, wrap_idx = sample_key
        patch = dataset.patches[int(patch_idx)]
        records.append(
            _SpatialSplitRecord(
                sample_index=int(sample_idx),
                source_key=_patch_source_key(patch),
                world_bbox=tuple(float(v) for v in patch.world_bbox),
                chunk_id=tuple(int(v) for v in patch.chunk_id) if getattr(patch, "chunk_id", None) is not None else None,
            )
        )
    return records


def _build_spatial_split_groups(
    dataset: AutoregMeshDataset,
    *,
    margin_voxels: float,
) -> list[list[int]]:
    records = _build_spatial_split_records(dataset)
    if not records:
        return []
    parents = list(range(len(records)))

    def _find(idx: int) -> int:
        while parents[idx] != idx:
            parents[idx] = parents[parents[idx]]
            idx = parents[idx]
        return idx

    def _union(left: int, right: int) -> None:
        root_left = _find(left)
        root_right = _find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    grouped_by_source: dict[tuple[Any, ...], list[_SpatialSplitRecord]] = {}
    for record in records:
        grouped_by_source.setdefault(record.source_key, []).append(record)

    for source_records in grouped_by_source.values():
        ordered = sorted(source_records, key=lambda record: record.world_bbox[0])
        active: list[_SpatialSplitRecord] = []
        for record in ordered:
            current_z_min = float(record.world_bbox[0])
            active = [
                other for other in active
                if (float(other.world_bbox[1]) + float(margin_voxels)) > (current_z_min - float(margin_voxels))
            ]
            for other in active:
                if _expanded_bboxes_overlap(record.world_bbox, other.world_bbox, margin_voxels=margin_voxels):
                    _union(record.sample_index, other.sample_index)
            active.append(record)

    groups: dict[int, list[int]] = {}
    for record in records:
        groups.setdefault(_find(record.sample_index), []).append(int(record.sample_index))
    return [sorted(indices) for _, indices in sorted(groups.items(), key=lambda item: min(item[1]))]


def _split_spatial_groups(
    groups: list[list[int]],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[int], list[int]]:
    if not groups:
        raise ValueError("autoreg_mesh training requires a non-empty dataset")
    if len(groups) < 2 or float(val_fraction) <= 0.0:
        train_indices = [idx for group in groups for idx in group]
        return sorted(train_indices), []
    num_val_groups = int(round(len(groups) * float(val_fraction)))
    num_val_groups = max(1, min(num_val_groups, len(groups) - 1))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(groups)).tolist()
    val_group_ids = set(order[:num_val_groups])
    train_indices, val_indices = [], []
    for group_idx, group in enumerate(groups):
        if group_idx in val_group_ids:
            val_indices.extend(group)
        else:
            train_indices.extend(group)
    return sorted(train_indices), sorted(val_indices)


def _spatial_split_diagnostics(groups: list[list[int]], train_indices: list[int], val_indices: list[int], *, total_items: int) -> dict[str, float]:
    train_groups = [group for group in groups if any(idx in train_indices for idx in group)]
    val_groups = [group for group in groups if any(idx in val_indices for idx in group)]
    train_sizes = [len(group) for group in train_groups]
    val_sizes = [len(group) for group in val_groups]
    all_sizes = [len(group) for group in groups]
    return {
        "num_train_groups": float(len(train_groups)),
        "num_val_groups": float(len(val_groups)),
        "mean_train_group_size": float(np.mean(train_sizes)) if train_sizes else 0.0,
        "mean_val_group_size": float(np.mean(val_sizes)) if val_sizes else 0.0,
        "max_group_size": float(max(all_sizes)) if all_sizes else 0.0,
        "val_fraction_actual": (float(len(val_indices)) / float(max(total_items, 1))) if total_items > 0 else 0.0,
    }


def _find_cross_split_bbox_overlap(
    dataset: AutoregMeshDataset,
    train_indices: list[int],
    val_indices: list[int],
    *,
    margin_voxels: float,
) -> tuple[int, int] | None:
    records = _build_spatial_split_records(dataset)
    train_records = [records[int(idx)] for idx in train_indices]
    val_records = [records[int(idx)] for idx in val_indices]
    val_by_source: dict[tuple[Any, ...], list[_SpatialSplitRecord]] = {}
    for record in val_records:
        val_by_source.setdefault(record.source_key, []).append(record)
    for train_record in train_records:
        for val_record in val_by_source.get(train_record.source_key, ()):
            if _expanded_bboxes_overlap(train_record.world_bbox, val_record.world_bbox, margin_voxels=margin_voxels):
                return train_record.sample_index, val_record.sample_index
    return None


def _restrict_dataset_samples(dataset: Dataset, selected_indices: list[int]) -> Dataset:
    if not hasattr(dataset, "sample_index"):
        raise TypeError("dataset does not expose sample_index for in-place split restriction")
    dataset.sample_index = [dataset.sample_index[int(i)] for i in selected_indices]
    return dataset


def _clone_autoreg_mesh_dataset(
    cfg: dict,
    patch_metadata,
    *,
    apply_spatial_augmentation: bool,
    apply_volume_only_augmentation: bool,
) -> AutoregMeshDataset:
    return AutoregMeshDataset(
        cfg,
        patch_metadata=patch_metadata,
        apply_augmentation=False,
        apply_perturbation=False,
        apply_spatial_augmentation=bool(apply_spatial_augmentation),
        apply_volume_only_augmentation=bool(apply_volume_only_augmentation),
    )


def _split_dataset(dataset: Dataset, *, cfg: dict, seed: int, val_fraction: float) -> tuple[Dataset, Dataset | None, dict[str, float]]:
    total = len(dataset)
    split_diagnostics: dict[str, float] = {}
    train_indices, val_indices = _split_indices(total, seed=seed, val_fraction=val_fraction)

    if isinstance(dataset, AutoregMeshDataset):
        if str(cfg.get("val_split_mode", "spatial_groups")) == "spatial_groups":
            groups = _build_spatial_split_groups(
                dataset,
                margin_voxels=float(cfg.get("validation_leakage_margin_voxels", 0.0)),
            )
            if float(val_fraction) > 0.0 and len(groups) < 2:
                raise ValueError(
                    "autoreg_mesh leakage-safe spatial split collapsed into fewer than two groups; "
                    "unable to form a held-out validation set"
                )
            train_indices, val_indices = _split_spatial_groups(groups, seed=seed, val_fraction=val_fraction)
            split_diagnostics = _spatial_split_diagnostics(groups, train_indices, val_indices, total_items=total)
            overlap_pair = _find_cross_split_bbox_overlap(
                dataset,
                train_indices,
                val_indices,
                margin_voxels=float(cfg.get("validation_leakage_margin_voxels", 0.0)),
            )
            if overlap_pair is not None:
                raise RuntimeError(
                    "autoreg_mesh spatial validation split still has leakage between train and val; "
                    f"overlapping sample indices={overlap_pair}"
                )
        patch_metadata = dataset.export_patch_metadata()
        train_dataset = _restrict_dataset_samples(
            _clone_autoreg_mesh_dataset(
                cfg,
                patch_metadata,
                apply_spatial_augmentation=bool(cfg.get("spatial_augmentation", {}).get("enabled", False)),
                apply_volume_only_augmentation=bool(cfg.get("volume_only_augmentation", {}).get("enabled", False)),
            ),
            train_indices,
        )
        if not val_indices:
            return train_dataset, None, split_diagnostics
        val_dataset = _restrict_dataset_samples(
            _clone_autoreg_mesh_dataset(
                cfg,
                patch_metadata,
                apply_spatial_augmentation=False,
                apply_volume_only_augmentation=False,
            ),
            val_indices,
        )
        return train_dataset, val_dataset, split_diagnostics

    if not val_indices:
        return dataset, None, split_diagnostics
    return Subset(dataset, train_indices), Subset(dataset, val_indices), split_diagnostics


def _make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    sampler=None,
    worker_init_fn=None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=int(num_workers),
        collate_fn=autoreg_mesh_collate,
        persistent_workers=bool(num_workers > 0),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def _maybe_import_wandb(cfg: dict):
    if not cfg.get("wandb_project"):
        return None
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise ImportError(
            "wandb_project is configured for autoreg_mesh training, but the 'wandb' package is not installed."
        ) from exc


def _load_checkpoint_payload(path: str | Path | None):
    if path is None:
        return None
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _resolve_wandb_run_id(cfg: dict, ckpt_payload: dict | None) -> str | None:
    run_id = cfg.get("wandb_run_id")
    if run_id is not None:
        return str(run_id)
    if not bool(cfg.get("wandb_resume", False)) or ckpt_payload is None:
        return None
    run_id = ckpt_payload.get("wandb_run_id")
    if run_id is None:
        ckpt_config = ckpt_payload.get("config", {})
        if isinstance(ckpt_config, dict):
            run_id = ckpt_config.get("wandb_run_id")
    return None if run_id is None else str(run_id)


def _checkpoint_coarse_prediction_mode(ckpt_payload: dict | None) -> str:
    if ckpt_payload is None:
        return "joint_pointer"
    ckpt_config = ckpt_payload.get("config", {})
    if isinstance(ckpt_config, dict):
        return str(ckpt_config.get("coarse_prediction_mode", "joint_pointer"))
    return "joint_pointer"


def _validate_checkpoint_compatibility(cfg: dict, ckpt_payload: dict | None) -> None:
    if ckpt_payload is None:
        return
    expected_mode = str(cfg.get("coarse_prediction_mode", "joint_pointer"))
    checkpoint_mode = _checkpoint_coarse_prediction_mode(ckpt_payload)
    if checkpoint_mode != expected_mode:
        raise ValueError(
            "load_ckpt uses incompatible coarse_prediction_mode: "
            f"checkpoint={checkpoint_mode!r} current_config={expected_mode!r}"
        )


def _make_checkpoint_payload(
    *,
    model: AutoregMeshModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    step: int,
) -> dict:
    raw_model = _unwrap_model(model)
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": int(step),
        "wandb_run_id": config.get("wandb_run_id"),
        "distributed_world_size": int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1,
        "distributed_backend": dist.get_backend() if dist.is_available() and dist.is_initialized() else None,
        "mixed_precision": str(config.get("mixed_precision", "no")),
    }
    if scheduler is not None:
        payload["lr_scheduler"] = scheduler.state_dict()
    return payload


def _save_checkpoint(
    *,
    out_dir: Path,
    filename: str,
    model: AutoregMeshModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    step: int,
) -> Path:
    path = out_dir / filename
    torch.save(
        _make_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
        ),
        path,
    )
    return path


def _loss_dict_to_metrics(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    metrics = {}
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            metrics[key] = float(value.detach().cpu().item())
        else:
            metrics[key] = float(value)
    return metrics


def _mean_metric_dict(metric_dicts: list[dict[str, float]], *, prefix: str) -> dict[str, float]:
    if not metric_dicts:
        return {}
    sums: dict[str, float] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    return {f"{prefix}{key}": value / float(len(metric_dicts)) for key, value in sums.items()}


def _mean_batch_sample_metrics(batch: dict, *, prefix: str = "") -> dict[str, float]:
    metrics = {}
    if "target_invalid_fraction" in batch:
        metrics[f"{prefix}target_invalid_fraction"] = float(batch["target_invalid_fraction"].to(torch.float32).mean().item())
    if "frontier_invalid_fraction" in batch:
        metrics[f"{prefix}frontier_invalid_fraction"] = float(batch["frontier_invalid_fraction"].to(torch.float32).mean().item())
    if "touches_crop_boundary" in batch:
        metrics[f"{prefix}touches_crop_boundary"] = float(batch["touches_crop_boundary"].to(torch.float32).mean().item())
    return metrics


def _make_rollout_metric_batch(raw_sample: dict, inference_result: dict) -> tuple[dict, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    batch = autoreg_mesh_collate([raw_sample])
    target_len = int(batch["target_lengths"][0].item())
    pred_xyz = np.asarray(inference_result["predicted_continuation_vertices_local"], dtype=np.float32)
    pred_len = int(pred_xyz.shape[0])
    pred_xyz_padded = torch.full((1, target_len, 3), float("nan"), dtype=torch.float32)
    overlap = min(target_len, pred_len)
    if overlap > 0:
        pred_xyz_padded[0, :overlap] = torch.from_numpy(pred_xyz[:overlap])

    pred_coarse_ids = torch.full((1, target_len), -100, dtype=torch.long)
    pred_coarse_np = np.asarray(inference_result["predicted_coarse_ids"], dtype=np.int64)
    if overlap > 0:
        pred_coarse_ids[0, :overlap] = torch.from_numpy(pred_coarse_np[:overlap])

    pred_axis_ids = None
    if "predicted_coarse_axis_ids" in inference_result:
        pred_axis_ids = {}
        for axis_name in ("z", "y", "x"):
            axis_padded = torch.full((1, target_len), -100, dtype=torch.long)
            axis_np = np.asarray(inference_result["predicted_coarse_axis_ids"][axis_name], dtype=np.int64)
            if overlap > 0:
                axis_padded[0, :overlap] = torch.from_numpy(axis_np[:overlap])
            pred_axis_ids[axis_name] = axis_padded
    return batch, pred_xyz_padded, pred_coarse_ids, pred_axis_ids


@torch.no_grad()
def _evaluate_rollout_validation(
    *,
    model: AutoregMeshModel,
    dataset: Dataset,
    cfg: dict,
) -> dict[str, float]:
    num_examples = min(int(cfg.get("rollout_val_examples_per_log", 1)), len(dataset))
    if num_examples <= 0:
        return {}

    seam_band_width = int(cfg.get("seam_band_width", 1))
    max_steps_cfg = cfg.get("rollout_val_max_steps")
    metric_dicts: list[dict[str, float]] = []
    model.eval()
    for sample_idx in range(num_examples):
        raw_sample = dataset[sample_idx]
        inference_result = infer_autoreg_mesh(
            model,
            raw_sample,
            greedy=True,
            max_steps=None if max_steps_cfg is None else int(max_steps_cfg),
        )
        batch, pred_xyz_padded, pred_coarse_ids, pred_axis_ids = _make_rollout_metric_batch(raw_sample, inference_result)
        coarse_metrics = _coarse_accuracy_metrics(
            {
                "pred_coarse_ids": pred_coarse_ids,
                "pred_coarse_axis_ids": pred_axis_ids if pred_axis_ids is not None else {
                    "z": torch.zeros_like(pred_coarse_ids),
                    "y": torch.zeros_like(pred_coarse_ids),
                    "x": torch.zeros_like(pred_coarse_ids),
                },
                "coarse_grid_shape": tuple(int(v) for v in model.coarse_grid_shape),
            },
            batch,
        )
        target_len = int(batch["target_lengths"][0].item())
        pred_len = int(np.asarray(inference_result["predicted_continuation_vertices_local"]).shape[0])
        metric_dicts.append(
            {
                "xyz_l1_refined": float(_l1_xyz_metric(pred_xyz_padded, batch).item()),
                "seam_edge_error": float(_seam_edge_error_from_sequence(pred_xyz_padded, batch, band_width=seam_band_width).item()),
                "coarse_exact_acc": float(coarse_metrics["coarse_exact_acc"].item()),
                "triangle_flip_rate": float(_triangle_flip_rate_from_sequence(pred_xyz_padded, batch).item()),
                "first_strip_wrong_side_rate": float(_first_strip_wrong_side_rate_from_sequence(pred_xyz_padded, batch).item()),
                "pred_oob_fraction": float(_pred_oob_fraction_from_sequence(pred_xyz_padded, batch).item()),
                "invalid_vertex_fraction": float(_invalid_vertex_fraction_from_sequence(pred_xyz_padded, batch).item()),
                "boundary_touch_fraction": float(_boundary_touch_fraction_from_sequence(pred_xyz_padded, batch).item()),
                "stop_count_error": float(abs(pred_len - target_len)),
            }
        )
    model.train()
    return _mean_metric_dict(metric_dicts, prefix="rollout_val_")


def _scheduled_sampling_prob(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("scheduled_sampling_enabled", False)):
        return 0.0
    start_step = int(cfg.get("scheduled_sampling_start_step", 0))
    if global_step < start_step:
        return 0.0
    max_prob = float(cfg.get("scheduled_sampling_max_prob", 0.0))
    ramp_steps = int(cfg.get("scheduled_sampling_ramp_steps", 0))
    if ramp_steps <= 0:
        return max_prob
    progress = min(1.0, max(0.0, float(global_step - start_step) / float(ramp_steps)))
    return max_prob * progress


def _position_refine_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("position_refine_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("position_refine_start_step", 5000)):
        return 0.0
    return float(cfg.get("position_refine_weight", 0.0))


def _xyz_soft_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("xyz_soft_loss_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("xyz_soft_loss_start_step", 0)):
        return 0.0
    return float(cfg.get("xyz_soft_loss_weight", 0.0))


def _seam_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("seam_loss_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("seam_loss_start_step", 0)):
        return 0.0
    return float(cfg.get("seam_loss_weight", 0.0))


def _triangle_barrier_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("triangle_barrier_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("triangle_barrier_start_step", 0)):
        return 0.0
    return float(cfg.get("triangle_barrier_weight", 0.0))


def _boundary_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    del cfg, global_step
    return 0.0


def _geometry_metric_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("geometry_metric_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("geometry_metric_start_step", 2000)):
        return 0.0
    return float(cfg.get("geometry_metric_weight", 0.0))


def _geometry_sd_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("geometry_sd_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("geometry_sd_start_step", 2000)):
        return 0.0
    return float(cfg.get("geometry_sd_weight", 0.0))


def _joint_valid_aux_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("joint_valid_aux_loss_enabled", False)):
        return 0.0
    if int(global_step) < int(cfg.get("joint_valid_aux_loss_start_step", 0)):
        return 0.0
    return float(cfg.get("joint_valid_aux_loss_weight", 0.0))


def _offset_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    start = int(cfg.get("offset_loss_start_step", 0))
    if int(global_step) < start:
        return 0.0
    ramp = int(cfg.get("offset_loss_ramp_steps", 0))
    final = float(cfg.get("offset_loss_weight", 1.0))
    if ramp <= 0:
        return final
    t = min(1.0, float(int(global_step) - start) / float(ramp))
    return final * t


def _scheduled_sampling_feedback_state(cfg: dict, *, global_step: int) -> tuple[bool, bool]:
    offset_feedback_start = cfg.get("scheduled_sampling_offset_feedback_start_step")
    if offset_feedback_start is None:
        offset_feedback_enabled = _offset_loss_weight_active(cfg, global_step=global_step) > 0.0
    else:
        offset_feedback_enabled = int(global_step) >= int(offset_feedback_start)
    refine_feedback_enabled = _position_refine_weight_active(cfg, global_step=global_step) > 0.0
    return bool(offset_feedback_enabled), bool(offset_feedback_enabled and refine_feedback_enabled)


def _as_numpy_grid(grid) -> np.ndarray:
    if torch.is_tensor(grid):
        return grid.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(grid, dtype=np.float32)


def _draw_line_2d(canvas: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    while True:
        if 0 <= r0 < canvas.shape[0] and 0 <= c0 < canvas.shape[1]:
            canvas[r0, c0] = 1.0
        if r0 == r1 and c0 == c1:
            break
        err2 = 2 * err
        if err2 > -dr:
            err -= dr
            c0 += sc
        if err2 < dc:
            err += dc
            r0 += sr


def _draw_line_2d_thick(canvas: np.ndarray, r0: int, c0: int, r1: int, c1: int, *, thickness: int) -> None:
    radius = max(0, int(thickness) - 1)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            _draw_line_2d(canvas, r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _normalize_slice_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    slice_arr = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(slice_arr)
    if not bool(finite.any()):
        gray = np.zeros_like(slice_arr, dtype=np.uint8)
    else:
        low = float(slice_arr[finite].min())
        high = float(slice_arr[finite].max())
        if high <= low + 1e-6:
            gray = np.zeros_like(slice_arr, dtype=np.uint8)
        else:
            normalized = np.clip((slice_arr - low) / (high - low), 0.0, 1.0)
            gray = (255.0 * normalized).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _iter_grid_edges_xy(grid_local: np.ndarray):
    grid = np.asarray(grid_local, dtype=np.float32)
    valid = np.isfinite(grid).all(axis=-1)
    rows, cols = grid.shape[:2]
    for row_idx in range(rows):
        for col_idx in range(cols - 1):
            if valid[row_idx, col_idx] and valid[row_idx, col_idx + 1]:
                yield grid[row_idx, col_idx], grid[row_idx, col_idx + 1]
    for row_idx in range(rows - 1):
        for col_idx in range(cols):
            if valid[row_idx, col_idx] and valid[row_idx + 1, col_idx]:
                yield grid[row_idx, col_idx], grid[row_idx + 1, col_idx]


def _edge_segment_on_z_slice(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    z_slice: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    z_min = float(z_slice) - float(depth_tolerance)
    z_max = float(z_slice) + float(depth_tolerance)
    z0 = float(p0[0])
    z1 = float(p1[0])
    dz = z1 - z0

    if abs(dz) < 1e-6:
        if z_min <= z0 <= z_max:
            return p0[1:].copy(), p1[1:].copy()
        return None

    t0 = (z_min - z0) / dz
    t1 = (z_max - z0) / dz
    t_lo = max(0.0, min(t0, t1))
    t_hi = min(1.0, max(t0, t1))
    if t_hi < 0.0 or t_lo > 1.0 or t_lo > t_hi:
        return None

    pa = p0 + t_lo * (p1 - p0)
    pb = p0 + t_hi * (p1 - p0)
    return pa[1:].copy(), pb[1:].copy()


def _count_slice_support(grid_local: np.ndarray, *, z_slice: int, depth_tolerance: float) -> int:
    count = 0
    for p0, p1 in _iter_grid_edges_xy(grid_local):
        if _edge_segment_on_z_slice(p0, p1, z_slice=z_slice, depth_tolerance=depth_tolerance) is not None:
            count += 1
    return count


def _choose_best_xy_slice(
    grids_local: list[np.ndarray],
    *,
    depth: int,
    depth_tolerance: float,
) -> int:
    finite_z = []
    for grid in grids_local:
        grid_arr = np.asarray(grid, dtype=np.float32)
        valid = np.isfinite(grid_arr).all(axis=-1)
        if bool(valid.any()):
            finite_z.append(grid_arr[..., 0][valid])
    if not finite_z:
        return max(0, min(int(depth) - 1, int(depth) // 2))

    z_values = np.concatenate(finite_z, axis=0)
    z_lo = max(0, int(np.floor(float(z_values.min()) - float(depth_tolerance))))
    z_hi = min(int(depth) - 1, int(np.ceil(float(z_values.max()) + float(depth_tolerance))))
    if z_hi < z_lo:
        return max(0, min(int(depth) - 1, int(np.rint(float(np.median(z_values))))))

    best_slice = z_lo
    best_score = -1
    center = 0.5 * float(max(0, int(depth) - 1))
    for z_slice in range(z_lo, z_hi + 1):
        score = sum(_count_slice_support(grid, z_slice=z_slice, depth_tolerance=depth_tolerance) for grid in grids_local)
        if score > best_score:
            best_score = score
            best_slice = z_slice
        elif score == best_score and abs(float(z_slice) - center) < abs(float(best_slice) - center):
            best_slice = z_slice
    return int(best_slice)


def _blend_line_mask(canvas: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float = 0.8) -> np.ndarray:
    blended = np.asarray(canvas, dtype=np.float32).copy()
    mask_bool = mask > 0.0
    if not bool(mask_bool.any()):
        return canvas
    color_arr = np.asarray(color, dtype=np.float32)
    blended[mask_bool] = (1.0 - float(alpha)) * blended[mask_bool] + float(alpha) * color_arr[None, :]
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _rasterize_grid_on_xy_slice(
    grid_local: np.ndarray,
    *,
    z_slice: int,
    panel_shape: tuple[int, int],
    line_thickness: int,
    depth_tolerance: float,
) -> np.ndarray:
    mask = np.zeros(panel_shape, dtype=np.float32)
    for p0, p1 in _iter_grid_edges_xy(grid_local):
        clipped = _edge_segment_on_z_slice(p0, p1, z_slice=z_slice, depth_tolerance=depth_tolerance)
        if clipped is None:
            continue
        a_xy, b_xy = clipped
        r0 = int(np.clip(np.rint(a_xy[0]), 0, panel_shape[0] - 1))
        c0 = int(np.clip(np.rint(a_xy[1]), 0, panel_shape[1] - 1))
        r1 = int(np.clip(np.rint(b_xy[0]), 0, panel_shape[0] - 1))
        c1 = int(np.clip(np.rint(b_xy[1]), 0, panel_shape[1] - 1))
        _draw_line_2d_thick(mask, r0, c0, r1, c1, thickness=int(line_thickness))
    return mask


def _make_xy_slice_overlay_canvas(
    *,
    volume: np.ndarray,
    prompt_grid_local: np.ndarray,
    target_grid_local: np.ndarray,
    pred_grid_local: np.ndarray,
    line_thickness: int,
    depth_tolerance: float,
) -> np.ndarray:
    volume_np = np.asarray(volume, dtype=np.float32)
    if volume_np.ndim == 4:
        volume_np = volume_np[0]
    if volume_np.ndim != 3:
        raise ValueError(f"volume must have shape [D,H,W] or [1,D,H,W], got {volume_np.shape!r}")
    z_slice = _choose_best_xy_slice(
        [prompt_grid_local, target_grid_local, pred_grid_local],
        depth=volume_np.shape[0],
        depth_tolerance=float(depth_tolerance),
    )
    slice_rgb = _normalize_slice_to_rgb(volume_np[z_slice])
    panel_shape = tuple(int(v) for v in volume_np.shape[1:])
    prompt_mask = _rasterize_grid_on_xy_slice(
        prompt_grid_local,
        z_slice=z_slice,
        panel_shape=panel_shape,
        line_thickness=int(line_thickness),
        depth_tolerance=float(depth_tolerance),
    )
    target_mask = _rasterize_grid_on_xy_slice(
        target_grid_local,
        z_slice=z_slice,
        panel_shape=panel_shape,
        line_thickness=int(line_thickness),
        depth_tolerance=float(depth_tolerance),
    )
    pred_mask = _rasterize_grid_on_xy_slice(
        pred_grid_local,
        z_slice=z_slice,
        panel_shape=panel_shape,
        line_thickness=int(line_thickness),
        depth_tolerance=float(depth_tolerance),
    )
    overlay = slice_rgb
    overlay = _blend_line_mask(overlay, prompt_mask, color=(90, 180, 255))
    overlay = _blend_line_mask(overlay, target_mask, color=(110, 235, 110))
    overlay = _blend_line_mask(overlay, pred_mask, color=(255, 190, 90))
    return _add_header(
        overlay,
        title=f"XY Slice z={z_slice}  prompt/gt/pred",
        labels=["XY"],
        background=(24, 24, 24),
    )


def _render_surface_projection(
    grid_local: np.ndarray,
    *,
    axes: tuple[int, int],
    panel_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.asarray(grid_local, dtype=np.float32)
    panel = np.zeros(panel_shape, dtype=np.float32)
    valid = np.isfinite(grid).all(axis=-1)

    def _project(point: np.ndarray) -> tuple[int, int]:
        row = int(np.clip(np.rint(point[axes[0]]), 0, panel_shape[0] - 1))
        col = int(np.clip(np.rint(point[axes[1]]), 0, panel_shape[1] - 1))
        return row, col

    rows, cols = grid.shape[:2]
    for row_idx in range(rows):
        for col_idx in range(cols):
            if not valid[row_idx, col_idx]:
                continue
            r0, c0 = _project(grid[row_idx, col_idx])
            panel[r0, c0] = 1.0
            if col_idx + 1 < cols and valid[row_idx, col_idx + 1]:
                r1, c1 = _project(grid[row_idx, col_idx + 1])
                _draw_line_2d(panel, r0, c0, r1, c1)
            if row_idx + 1 < rows and valid[row_idx + 1, col_idx]:
                r1, c1 = _project(grid[row_idx + 1, col_idx])
                _draw_line_2d(panel, r0, c0, r1, c1)
    return panel


def _voxelize_grid_projection_panels(grid_local: np.ndarray, crop_shape: tuple[int, int, int]) -> list[tuple[str, np.ndarray]]:
    return [
        ("ZY", _render_surface_projection(grid_local, axes=(0, 1), panel_shape=(crop_shape[0], crop_shape[1]))),
        ("ZX", _render_surface_projection(grid_local, axes=(0, 2), panel_shape=(crop_shape[0], crop_shape[2]))),
        ("YX", _render_surface_projection(grid_local, axes=(1, 2), panel_shape=(crop_shape[1], crop_shape[2]))),
    ]


def _panel_to_rgb(panel: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    clipped = np.clip(panel, 0.0, 1.0)
    image = np.zeros((*clipped.shape, 3), dtype=np.uint8)
    for channel, value in enumerate(color):
        image[..., channel] = (clipped * float(value)).astype(np.uint8)
    return image


def _pad_panel_height(panel: np.ndarray, *, height: int) -> np.ndarray:
    if int(panel.shape[0]) >= int(height):
        return panel
    pad_rows = int(height) - int(panel.shape[0])
    return np.pad(panel, ((0, pad_rows), (0, 0)), mode="constant")


def _add_header(canvas: np.ndarray, *, title: str, labels: list[str], background: tuple[int, int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    header_height = 22
    header = np.zeros((header_height, canvas.shape[1], 3), dtype=np.uint8)
    header[..., 0] = background[0]
    header[..., 1] = background[1]
    header[..., 2] = background[2]
    image = Image.fromarray(header)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((6, 5), title, fill=(255, 255, 255), font=font)

    panel_width = canvas.shape[1] // len(labels)
    for idx, label in enumerate(labels):
        x = idx * panel_width + max(6, panel_width // 2 - 12)
        draw.text((x, 5), label, fill=(230, 230, 230), font=font)
    return np.concatenate([np.asarray(image, dtype=np.uint8), canvas], axis=0)


def _make_labeled_triptych(
    *,
    grid_local: np.ndarray,
    crop_shape: tuple[int, int, int],
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    projection_panels = _voxelize_grid_projection_panels(grid_local, crop_shape)
    target_height = max(int(panel.shape[0]) for _, panel in projection_panels)
    rgb_panels = [
        _panel_to_rgb(_pad_panel_height(panel, height=target_height), color=color)
        for _, panel in projection_panels
    ]
    separator = np.full((target_height, 3, 3), 28, dtype=np.uint8)
    body = np.concatenate(
        [rgb_panels[0], separator, rgb_panels[1], separator.copy(), rgb_panels[2]],
        axis=1,
    )
    return _add_header(
        body,
        title=title,
        labels=[name for name, _ in projection_panels],
        background=tuple(max(12, int(value * 0.18)) for value in color),
    )


def _make_projection_canvas(
    *,
    prompt_grid_local: np.ndarray,
    target_grid_local: np.ndarray,
    pred_grid_local: np.ndarray,
    crop_shape: tuple[int, int, int],
) -> np.ndarray:
    prompt_panel = _make_labeled_triptych(
        grid_local=prompt_grid_local,
        crop_shape=crop_shape,
        title="Prompt",
        color=(90, 180, 255),
    )
    target_panel = _make_labeled_triptych(
        grid_local=target_grid_local,
        crop_shape=crop_shape,
        title="Target",
        color=(110, 235, 110),
    )
    pred_panel = _make_labeled_triptych(
        grid_local=pred_grid_local,
        crop_shape=crop_shape,
        title="Prediction",
        color=(255, 190, 90),
    )
    target_height = max(int(prompt_panel.shape[0]), int(target_panel.shape[0]), int(pred_panel.shape[0]))
    prompt_panel = _pad_panel_height(prompt_panel, height=target_height)
    target_panel = _pad_panel_height(target_panel, height=target_height)
    pred_panel = _pad_panel_height(pred_panel, height=target_height)
    separator = np.full((target_height, 8, 3), 24, dtype=np.uint8)
    return np.concatenate([prompt_panel, separator, target_panel, separator.copy(), pred_panel], axis=1)


def _make_teacher_forced_prediction_canvas(batch: dict, outputs: dict, *, sample_idx: int = 0) -> np.ndarray:
    count = int(batch["target_lengths"][sample_idx].item())
    grid_shape = tuple(int(v) for v in batch["target_grid_shape"][sample_idx].tolist())
    direction = str(batch["direction"][sample_idx])
    pred_xyz = outputs.get("pred_xyz_refined", outputs["pred_xyz"])[sample_idx, :count].detach().cpu().numpy()
    pred_grid_local = deserialize_continuation_grid(pred_xyz, direction=direction, grid_shape=grid_shape)
    prompt_grid_local = _as_numpy_grid(batch["prompt_grid_local"][sample_idx])
    target_grid_local = _as_numpy_grid(batch["target_grid_local"][sample_idx])
    crop_shape = tuple(int(v) for v in batch["volume"][sample_idx].shape[-3:])
    return _make_projection_canvas(
        prompt_grid_local=prompt_grid_local,
        target_grid_local=target_grid_local,
        pred_grid_local=pred_grid_local,
        crop_shape=crop_shape,
    )


def _make_teacher_forced_xy_slice_canvas(
    batch: dict,
    outputs: dict,
    *,
    sample_idx: int = 0,
    line_thickness: int,
    depth_tolerance: float,
) -> np.ndarray:
    count = int(batch["target_lengths"][sample_idx].item())
    grid_shape = tuple(int(v) for v in batch["target_grid_shape"][sample_idx].tolist())
    direction = str(batch["direction"][sample_idx])
    pred_xyz = outputs.get("pred_xyz_refined", outputs["pred_xyz"])[sample_idx, :count].detach().cpu().numpy()
    pred_grid_local = deserialize_continuation_grid(pred_xyz, direction=direction, grid_shape=grid_shape)
    prompt_grid_local = _as_numpy_grid(batch["prompt_grid_local"][sample_idx])
    target_grid_local = _as_numpy_grid(batch["target_grid_local"][sample_idx])
    volume = batch["volume"][sample_idx].detach().cpu().numpy()
    return _make_xy_slice_overlay_canvas(
        volume=volume,
        prompt_grid_local=prompt_grid_local,
        target_grid_local=target_grid_local,
        pred_grid_local=pred_grid_local,
        line_thickness=int(line_thickness),
        depth_tolerance=float(depth_tolerance),
    )


def _make_inference_prediction_canvas(raw_sample: dict, inference_result: dict) -> np.ndarray:
    prompt_grid_local = _as_numpy_grid(raw_sample["prompt_grid_local"])
    target_grid_local = _as_numpy_grid(raw_sample["target_grid_local"])
    pred_grid_local = np.asarray(inference_result["continuation_grid_local"], dtype=np.float32)
    crop_shape = tuple(int(v) for v in raw_sample["volume"].shape[-3:])
    return _make_projection_canvas(
        prompt_grid_local=prompt_grid_local,
        target_grid_local=target_grid_local,
        pred_grid_local=pred_grid_local,
        crop_shape=crop_shape,
    )


def _make_inference_xy_slice_canvas(
    raw_sample: dict,
    inference_result: dict,
    *,
    line_thickness: int,
    depth_tolerance: float,
) -> np.ndarray:
    prompt_grid_local = _as_numpy_grid(raw_sample["prompt_grid_local"])
    target_grid_local = _as_numpy_grid(raw_sample["target_grid_local"])
    pred_grid_local = np.asarray(inference_result["continuation_grid_local"], dtype=np.float32)
    volume = _as_numpy_grid(raw_sample["volume"])
    return _make_xy_slice_overlay_canvas(
        volume=volume,
        prompt_grid_local=prompt_grid_local,
        target_grid_local=target_grid_local,
        pred_grid_local=pred_grid_local,
        line_thickness=int(line_thickness),
        depth_tolerance=float(depth_tolerance),
    )


@torch.no_grad()
def _evaluate_validation(
    *,
    model: AutoregMeshModel,
    dataloader: DataLoader,
    iterator,
    cfg: dict,
    device: torch.device,
    global_step: int,
) -> tuple[dict[str, float], Any]:
    model.eval()
    amp_enabled = str(cfg.get("mixed_precision", "no")).lower() != "no" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    metric_dicts: list[dict[str, float]] = []
    for _ in range(int(cfg["val_batches_per_log"])):
        raw_batch, iterator = _next_batch(iterator, dataloader)
        batch = _move_batch_to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(batch, scheduled_sampling_prob=0.0)
            loss_dict = compute_autoreg_mesh_losses(
                outputs,
                batch,
                offset_num_bins=tuple(int(v) for v in cfg["offset_num_bins"]),
                occupancy_loss_weight=float(cfg.get("occupancy_loss_weight", 0.0)),
                offset_loss_weight_active=_offset_loss_weight_active(cfg, global_step=global_step),
                position_refine_weight_active=_position_refine_weight_active(cfg, global_step=global_step),
                position_refine_loss_type=str(cfg.get("position_refine_loss", "huber")),
                xyz_soft_loss_weight_active=_xyz_soft_loss_weight_active(cfg, global_step=global_step),
                xyz_soft_loss_type=str(cfg.get("xyz_soft_loss", "huber")),
                seam_loss_weight_active=_seam_loss_weight_active(cfg, global_step=global_step),
                seam_loss_type=str(cfg.get("seam_loss", "edge_huber")),
                seam_band_width=int(cfg.get("seam_band_width", 1)),
                triangle_barrier_weight_active=_triangle_barrier_weight_active(cfg, global_step=global_step),
                triangle_barrier_margin=float(cfg.get("triangle_barrier_margin", 0.05)),
                boundary_loss_weight_active=_boundary_loss_weight_active(cfg, global_step=global_step),
                geometry_metric_weight_active=_geometry_metric_weight_active(cfg, global_step=global_step),
                geometry_metric_loss_type=str(cfg.get("geometry_metric_loss", "huber")),
                geometry_sd_weight_active=_geometry_sd_weight_active(cfg, global_step=global_step),
                distance_aware_coarse_targets_enabled=bool(cfg.get("distance_aware_coarse_targets_enabled", True)),
                distance_aware_coarse_target_radius=int(cfg.get("distance_aware_coarse_target_radius", 1)),
                distance_aware_coarse_target_sigma=float(cfg.get("distance_aware_coarse_target_sigma", 1.0)),
                distance_aware_coarse_target_loss=str(cfg.get("distance_aware_coarse_target_loss", "soft_ce")),
                joint_valid_aux_loss_weight_active=_joint_valid_aux_loss_weight_active(cfg, global_step=global_step),
                distance_aware_offset_targets_enabled=bool(cfg.get("distance_aware_offset_targets_enabled", False)),
                distance_aware_offset_target_radius=int(cfg.get("distance_aware_offset_target_radius", 1)),
                distance_aware_offset_target_sigma=float(cfg.get("distance_aware_offset_target_sigma", 0.75)),
            )
        metrics = _loss_dict_to_metrics(loss_dict)
        metrics.update(_mean_batch_sample_metrics(batch))
        metric_dicts.append(metrics)
    model.train()
    return _mean_metric_dict(metric_dicts, prefix="val_"), iterator


def run_autoreg_mesh_training(
    config: dict,
    *,
    dataset=None,
    model: AutoregMeshModel | None = None,
    device: str | torch.device | None = None,
    max_steps: int | None = None,
) -> dict:
    cfg = validate_autoreg_mesh_config(config)
    runtime = _initialize_distributed_runtime(device)
    _seed_everything(int(cfg["seed"]) + int(runtime.rank))

    out_dir = Path(cfg["out_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = AutoregMeshDataset(cfg)
    train_dataset, val_dataset, split_diagnostics = _split_dataset(
        dataset,
        cfg=cfg,
        seed=int(cfg["seed"]),
        val_fraction=float(cfg.get("val_fraction", 0.0)),
    )
    train_worker_init = functools.partial(_seed_worker, base_seed=int(cfg["seed"]), rank=int(runtime.rank))
    val_worker_init = functools.partial(_seed_worker, base_seed=int(cfg["seed"]) + 1, rank=0)
    train_sampler = None
    if runtime.is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=int(runtime.world_size),
            rank=int(runtime.rank),
            shuffle=True,
            drop_last=False,
            seed=int(cfg["seed"]),
        )
    train_dataloader = _make_dataloader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        shuffle=not runtime.is_distributed,
        seed=int(cfg["seed"]),
        sampler=train_sampler,
        worker_init_fn=train_worker_init,
    )
    val_dataloader = None
    if val_dataset is not None and runtime.is_main_process:
        val_dataloader = _make_dataloader(
            val_dataset,
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg["val_num_workers"]),
            shuffle=False,
            seed=int(cfg["seed"]) + 1,
            worker_init_fn=val_worker_init,
        )

    if model is None:
        model = AutoregMeshModel(cfg)
    raw_model = model.to(runtime.device)
    raw_model.train()

    optimizer = create_optimizer(dict(cfg["optimizer"]), raw_model)
    scheduler = None
    scheduler_name = str(cfg.get("scheduler", "constant")).lower()
    total_steps = int(max_steps or cfg["num_steps"])
    if scheduler_name != "constant":
        from vesuvius.models.training.lr_schedulers import get_scheduler

        scheduler = get_scheduler(
            scheduler_type=scheduler_name,
            optimizer=optimizer,
            initial_lr=float(cfg["optimizer"]["learning_rate"]),
            max_steps=total_steps,
            **dict(cfg.get("scheduler_kwargs") or {}),
        )

    preloaded_ckpt = _load_checkpoint_payload(cfg.get("load_ckpt"))
    _validate_checkpoint_compatibility(cfg, preloaded_ckpt)
    resolved_wandb_run_id = _resolve_wandb_run_id(cfg, preloaded_ckpt)
    if resolved_wandb_run_id is not None:
        cfg["wandb_run_id"] = resolved_wandb_run_id

    start_step = 0
    if preloaded_ckpt is not None:
        raw_model.load_state_dict(preloaded_ckpt["model"])
        if not bool(cfg.get("load_weights_only", False)):
            start_step = int(preloaded_ckpt.get("step", 0))
            if "optimizer" in preloaded_ckpt:
                optimizer.load_state_dict(preloaded_ckpt["optimizer"])
            if scheduler is not None and "lr_scheduler" in preloaded_ckpt:
                scheduler.load_state_dict(preloaded_ckpt["lr_scheduler"])

    train_model = _wrap_model_for_ddp(raw_model, runtime)
    wandb = _maybe_import_wandb(cfg) if runtime.is_main_process else None
    wandb_run = None
    saved_checkpoints: list[str] = []
    final_checkpoint_path = None
    history: list[dict[str, float]] = []
    startup_started = time.perf_counter()
    progress_bar = None
    startup_ms = 0.0

    try:
        if wandb is not None and runtime.is_main_process:
            wandb_kwargs = {
                "project": cfg["wandb_project"],
                "config": cfg,
            }
            if cfg.get("wandb_entity") is not None:
                wandb_kwargs["entity"] = cfg["wandb_entity"]
            if cfg.get("wandb_run_name") is not None:
                wandb_kwargs["name"] = cfg["wandb_run_name"]
            if bool(cfg.get("wandb_resume", False)):
                wandb_kwargs["resume"] = cfg.get("wandb_resume_mode", "allow")
                if cfg.get("wandb_run_id") is not None:
                    wandb_kwargs["id"] = cfg["wandb_run_id"]
            wandb_run = wandb.init(**wandb_kwargs)
            active_run = getattr(wandb, "run", None) or wandb_run
            active_run_id = getattr(active_run, "id", None)
            if active_run_id is not None:
                cfg["wandb_run_id"] = str(active_run_id)

        if bool(cfg.get("ckpt_at_step_zero", False)) and start_step == 0 and runtime.is_main_process:
            ckpt_path = _save_checkpoint(
                out_dir=out_dir,
                filename="ckpt_000000.pth",
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=cfg,
                step=0,
            )
            saved_checkpoints.append(str(ckpt_path))
        if runtime.is_distributed:
            _maybe_barrier(runtime)

        train_sampler_epoch = 0
        if train_sampler is not None:
            train_sampler.set_epoch(train_sampler_epoch)
        def _on_train_iterator_reset():
            nonlocal train_sampler_epoch
            if train_sampler is not None:
                train_sampler_epoch += 1
                train_sampler.set_epoch(train_sampler_epoch)
        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader) if val_dataloader is not None else None
        global_step = int(start_step)
        progress_bar = tqdm(total=max(0, total_steps - global_step), desc="autoreg_mesh", leave=False) if runtime.is_main_process else None
        startup_ms = 1000.0 * (time.perf_counter() - startup_started)
        amp_enabled = str(cfg.get("mixed_precision", "no")).lower() != "no" and runtime.device.type == "cuda"
        amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

        while global_step < total_steps:
            raw_batch, train_iterator = _next_batch(
                train_iterator,
                train_dataloader,
                on_reset=_on_train_iterator_reset if train_sampler is not None else None,
            )
            batch = _move_batch_to_device(raw_batch, runtime.device)

            optimizer.zero_grad(set_to_none=True)
            scheduled_sampling_prob = _scheduled_sampling_prob(cfg, global_step=global_step)
            offset_loss_weight_active = _offset_loss_weight_active(cfg, global_step=global_step)
            position_refine_weight_active = _position_refine_weight_active(cfg, global_step=global_step)
            xyz_soft_loss_weight_active = _xyz_soft_loss_weight_active(cfg, global_step=global_step)
            seam_loss_weight_active = _seam_loss_weight_active(cfg, global_step=global_step)
            triangle_barrier_weight_active = _triangle_barrier_weight_active(cfg, global_step=global_step)
            boundary_loss_weight_active = _boundary_loss_weight_active(cfg, global_step=global_step)
            geometry_metric_weight_active = _geometry_metric_weight_active(cfg, global_step=global_step)
            geometry_sd_weight_active = _geometry_sd_weight_active(cfg, global_step=global_step)
            joint_valid_aux_loss_weight_active = _joint_valid_aux_loss_weight_active(cfg, global_step=global_step)
            offset_feedback_enabled, refine_feedback_enabled = _scheduled_sampling_feedback_state(cfg, global_step=global_step)
            with torch.autocast(device_type=runtime.device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = train_model(
                    batch,
                    scheduled_sampling_prob=scheduled_sampling_prob,
                    scheduled_sampling_pattern=str(cfg.get("scheduled_sampling_pattern", "stripwise_full_strip_greedy")),
                    scheduled_sampling_offset_feedback_enabled=offset_feedback_enabled,
                    scheduled_sampling_refine_feedback_enabled=refine_feedback_enabled,
                )
                loss_dict = compute_autoreg_mesh_losses(
                    outputs,
                    batch,
                    offset_num_bins=tuple(int(v) for v in cfg["offset_num_bins"]),
                    occupancy_loss_weight=float(cfg.get("occupancy_loss_weight", 0.0)),
                    offset_loss_weight_active=offset_loss_weight_active,
                    position_refine_weight_active=position_refine_weight_active,
                    position_refine_loss_type=str(cfg.get("position_refine_loss", "huber")),
                    xyz_soft_loss_weight_active=xyz_soft_loss_weight_active,
                    xyz_soft_loss_type=str(cfg.get("xyz_soft_loss", "huber")),
                    seam_loss_weight_active=seam_loss_weight_active,
                    seam_loss_type=str(cfg.get("seam_loss", "edge_huber")),
                    seam_band_width=int(cfg.get("seam_band_width", 1)),
                    triangle_barrier_weight_active=triangle_barrier_weight_active,
                    triangle_barrier_margin=float(cfg.get("triangle_barrier_margin", 0.05)),
                    boundary_loss_weight_active=boundary_loss_weight_active,
                    geometry_metric_weight_active=geometry_metric_weight_active,
                    geometry_metric_loss_type=str(cfg.get("geometry_metric_loss", "huber")),
                    geometry_sd_weight_active=geometry_sd_weight_active,
                    distance_aware_coarse_targets_enabled=bool(cfg.get("distance_aware_coarse_targets_enabled", True)),
                    distance_aware_coarse_target_radius=int(cfg.get("distance_aware_coarse_target_radius", 1)),
                    distance_aware_coarse_target_sigma=float(cfg.get("distance_aware_coarse_target_sigma", 1.0)),
                    distance_aware_coarse_target_loss=str(cfg.get("distance_aware_coarse_target_loss", "soft_ce")),
                    joint_valid_aux_loss_weight_active=joint_valid_aux_loss_weight_active,
                    distance_aware_offset_targets_enabled=bool(cfg.get("distance_aware_offset_targets_enabled", False)),
                    distance_aware_offset_target_radius=int(cfg.get("distance_aware_offset_target_radius", 1)),
                    distance_aware_offset_target_sigma=float(cfg.get("distance_aware_offset_target_sigma", 0.75)),
                )
            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Encountered non-finite training loss at step {global_step}: {loss.item()}")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=float(cfg["grad_clip"]))
            grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            skipped_step = 0.0
            if _distributed_all_finite(grad_norm_value, device=runtime.device, runtime=runtime):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                skipped_step = 1.0
            global_step += 1

            metrics = _loss_dict_to_metrics(loss_dict)
            metrics["current_lr"] = float(optimizer.param_groups[0]["lr"])
            metrics["grad_norm"] = grad_norm_value
            metrics["scheduled_sampling_prob"] = float(scheduled_sampling_prob)
            metrics["offset_loss_weight_active"] = float(offset_loss_weight_active)
            metrics["position_refine_weight_active"] = float(position_refine_weight_active)
            metrics["xyz_soft_loss_weight_active"] = float(xyz_soft_loss_weight_active)
            metrics["seam_loss_weight_active"] = float(seam_loss_weight_active)
            metrics["triangle_barrier_weight_active"] = float(triangle_barrier_weight_active)
            metrics["boundary_loss_weight_active"] = float(boundary_loss_weight_active)
            metrics["geometry_metric_weight_active"] = float(geometry_metric_weight_active)
            metrics["geometry_sd_weight_active"] = float(geometry_sd_weight_active)
            metrics["joint_valid_aux_loss_weight_active"] = float(joint_valid_aux_loss_weight_active)
            metrics["step"] = float(global_step)
            metrics.update(_mean_batch_sample_metrics(batch, prefix="train_"))
            metrics.update(split_diagnostics)
            if skipped_step > 0.0:
                metrics["skipped_step_nonfinite_grad"] = skipped_step
            metrics = _metric_dict_mean_across_ranks(metrics, device=runtime.device, runtime=runtime)

            should_run_validation_step = (
                val_dataset is not None and
                global_step % int(cfg["log_frequency"]) == 0
            )
            should_run_validation = runtime.is_main_process and val_dataloader is not None and should_run_validation_step
            if should_run_validation:
                val_metrics, val_iterator = _evaluate_validation(
                    model=raw_model,
                    dataloader=val_dataloader,
                    iterator=val_iterator,
                    cfg=cfg,
                    device=runtime.device,
                    global_step=global_step,
                )
                metrics.update(val_metrics)
                metrics.update(
                    _evaluate_rollout_validation(
                        model=raw_model,
                        dataset=val_dataset,
                        cfg=cfg,
                    )
                )

            wandb_payload = dict(metrics)
            should_log_projection_images_step = (
                bool(cfg.get("wandb_log_images", True)) and
                global_step % int(cfg["wandb_image_frequency"]) == 0
            )
            should_log_xy_images_step = (
                bool(cfg.get("wandb_log_images", True)) and
                bool(cfg.get("wandb_log_xy_slice_images", True)) and
                global_step % int(cfg["wandb_xy_slice_image_frequency"]) == 0
            )
            should_log_projection_images = runtime.is_main_process and wandb is not None and should_log_projection_images_step
            should_log_xy_images = runtime.is_main_process and wandb is not None and should_log_xy_images_step
            if should_log_projection_images or should_log_xy_images:
                raw_val_sample = None
                val_infer = None
                need_val_visual = val_dataset is not None and len(val_dataset) > 0
                if need_val_visual:
                    raw_val_sample = val_dataset[0]
                    raw_model.eval()
                    val_infer = infer_autoreg_mesh(raw_model, raw_val_sample, greedy=True)
                    raw_model.train()

            if should_log_projection_images:
                train_projection_image = wandb.Image(
                    _make_teacher_forced_prediction_canvas(batch, outputs, sample_idx=0),
                    caption=f"step={global_step} train teacher-forced",
                )
                wandb_payload["train_example"] = train_projection_image
                wandb_payload["train_example_projection"] = train_projection_image
                if raw_val_sample is not None and val_infer is not None:
                    val_projection_image = wandb.Image(
                        _make_inference_prediction_canvas(raw_val_sample, val_infer),
                        caption=f"step={global_step} val autoregressive",
                    )
                    wandb_payload["val_example"] = val_projection_image
                    wandb_payload["val_example_projection"] = val_projection_image
            if should_log_xy_images:
                wandb_payload["train_example_xy"] = wandb.Image(
                    _make_teacher_forced_xy_slice_canvas(
                        batch,
                        outputs,
                        sample_idx=0,
                        line_thickness=int(cfg.get("wandb_xy_slice_line_thickness", 1)),
                        depth_tolerance=float(cfg.get("wandb_xy_slice_depth_tolerance", 0.75)),
                    ),
                    caption=f"step={global_step} train xy slice",
                )
                if raw_val_sample is not None and val_infer is not None:
                    wandb_payload["val_example_xy"] = wandb.Image(
                        _make_inference_xy_slice_canvas(
                            raw_val_sample,
                            val_infer,
                            line_thickness=int(cfg.get("wandb_xy_slice_line_thickness", 1)),
                            depth_tolerance=float(cfg.get("wandb_xy_slice_depth_tolerance", 0.75)),
                        ),
                        caption=f"step={global_step} val xy slice",
                    )

            if runtime.is_main_process:
                history.append(dict(metrics))
            if wandb is not None and runtime.is_main_process:
                wandb.log(wandb_payload, step=global_step)

            if progress_bar is not None:
                progress_bar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
                progress_bar.update(1)

            should_write_checkpoint_step = (global_step % int(cfg["ckpt_frequency"]) == 0)
            should_write_checkpoint = runtime.is_main_process and should_write_checkpoint_step
            if should_write_checkpoint:
                ckpt_path = _save_checkpoint(
                    out_dir=out_dir,
                    filename=f"ckpt_{global_step:06}.pth",
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=cfg,
                    step=global_step,
                )
                saved_checkpoints.append(str(ckpt_path))
            if runtime.is_distributed and (
                should_run_validation_step or
                should_log_projection_images_step or
                should_log_xy_images_step or
                should_write_checkpoint_step
            ):
                _maybe_barrier(runtime)

        if progress_bar is not None:
            progress_bar.close()

        if bool(cfg.get("save_final_checkpoint", True)) and runtime.is_main_process:
            final_ckpt = _save_checkpoint(
                out_dir=out_dir,
                filename="final.pth",
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=cfg,
                step=global_step,
            )
            final_checkpoint_path = str(final_ckpt)
            saved_checkpoints.append(final_checkpoint_path)

        return {
            "model": raw_model,
            "optimizer": optimizer,
            "history": history if runtime.is_main_process else [],
            "final_metrics": history[-1] if history and runtime.is_main_process else {},
            "start_step": start_step,
            "wandb_run_id": cfg.get("wandb_run_id"),
            "checkpoint_paths": saved_checkpoints if runtime.is_main_process else [],
            "final_checkpoint_path": final_checkpoint_path if runtime.is_main_process else None,
            "out_dir": str(out_dir),
            "is_main_process": runtime.is_main_process,
            "rank": int(runtime.rank),
            "world_size": int(runtime.world_size),
            "device": str(runtime.device),
            "startup_ms": float(startup_ms),
            "split_diagnostics": dict(split_diagnostics),
        }
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if wandb is not None and runtime.is_main_process:
            wandb.finish()
        if runtime.is_distributed and dist.is_available() and dist.is_initialized():
            if runtime.initialized_process_group:
                dist.destroy_process_group()


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    cfg = load_autoreg_mesh_config(Path(config_path))
    result = run_autoreg_mesh_training(cfg)
    if bool(result.get("is_main_process", True)):
        print(json.dumps(result["final_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    train()
