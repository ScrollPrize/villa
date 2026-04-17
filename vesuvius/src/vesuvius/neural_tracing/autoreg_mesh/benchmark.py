from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from vesuvius.neural_tracing.autoreg_mesh.config import load_autoreg_mesh_config, validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.dataset import AutoregMeshDataset, autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.infer import infer_autoreg_mesh
from vesuvius.neural_tracing.autoreg_mesh.model import AutoregMeshModel
from vesuvius.neural_tracing.autoreg_mesh.train import run_autoreg_mesh_training


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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_ms(fn, *, device: torch.device, warmup: int = 1, repeats: int = 1) -> float:
    for _ in range(int(warmup)):
        fn()
        _sync(device)
    start = time.perf_counter()
    for _ in range(int(repeats)):
        fn()
        _sync(device)
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / float(max(1, repeats))


def run_autoreg_mesh_benchmark(
    config: dict,
    *,
    dataset=None,
    model: AutoregMeshModel | None = None,
    device: str | torch.device | None = None,
    sample_count: int = 32,
    training_steps: int = 0,
) -> dict:
    cfg = validate_autoreg_mesh_config(config)
    dataset = dataset or AutoregMeshDataset(cfg)
    if len(dataset) <= 0:
        raise ValueError("autoreg_mesh benchmark requires a non-empty dataset")

    sample_count = max(1, min(int(sample_count), len(dataset)))
    prompt_lengths = []
    target_lengths = []
    prompt_valid_counts = []
    target_valid_counts = []
    for idx in range(sample_count):
        sample = dataset[idx]
        prompt_lengths.append(int(sample["prompt_tokens"]["coarse_ids"].shape[0]))
        target_lengths.append(int(sample["target_coarse_ids"].shape[0]))
        prompt_valid_counts.append(int(sample["prompt_tokens"]["valid_mask"].sum().item()))
        target_valid_counts.append(int(sample["target_valid_mask"].sum().item()))

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model or AutoregMeshModel(cfg)
    model = model.to(device).eval()

    sample = dataset[0]
    batch = autoreg_mesh_collate([sample])
    batch = _move_batch_to_device(batch, device)

    @torch.no_grad()
    def _forward():
        return model(batch)

    @torch.no_grad()
    def _infer():
        infer_autoreg_mesh(
            model,
            sample,
            max_steps=min(64, int(sample["target_coarse_ids"].shape[0])),
            greedy=True,
            stop_probability_threshold=1.1,
        )

    forward_outputs = _forward()
    forward_ms = _measure_ms(_forward, device=device)
    infer_ms = _measure_ms(_infer, device=device)

    result = {
        "device": str(device),
        "dataset_length": int(len(dataset)),
        "sample_count": int(sample_count),
        "pointer_temperature": float(cfg.get("pointer_temperature", 0.25)),
        "coarse_prediction_mode": str(cfg.get("coarse_prediction_mode", "joint_pointer")),
        "conditioning_feature_debias_mode": str(cfg.get("conditioning_feature_debias_mode", "none")),
        "conditioning_feature_debias_basis_source": str(cfg.get("conditioning_feature_debias_basis_source", "zero_volume_svd")),
        "conditioning_feature_debias_components": int(cfg.get("conditioning_feature_debias_components", 16)),
        "conditioning_feature_debias_norm_ratio": float(getattr(model, "last_conditioning_feature_debias_norm_ratio", 1.0)),
        "coarse_grid_shape": list(int(v) for v in model.coarse_grid_shape),
        "coarse_axis_sizes": {
            "z": int(model.coarse_grid_shape[0]),
            "y": int(model.coarse_grid_shape[1]),
            "x": int(model.coarse_grid_shape[2]),
        },
        "distance_aware_coarse_targets_enabled": bool(cfg.get("distance_aware_coarse_targets_enabled", True)),
        "distance_aware_coarse_target_radius": int(cfg.get("distance_aware_coarse_target_radius", 1)),
        "distance_aware_coarse_target_sigma": float(cfg.get("distance_aware_coarse_target_sigma", 1.0)),
        "geometry_metric_enabled": bool(cfg.get("geometry_metric_enabled", True)),
        "geometry_metric_weight": float(cfg.get("geometry_metric_weight", 0.01)),
        "geometry_metric_start_step": int(cfg.get("geometry_metric_start_step", 2000)),
        "geometry_sd_enabled": bool(cfg.get("geometry_sd_enabled", True)),
        "geometry_sd_weight": float(cfg.get("geometry_sd_weight", 0.005)),
        "geometry_sd_start_step": int(cfg.get("geometry_sd_start_step", 2000)),
        "cross_attention_use_rope": bool(cfg.get("cross_attention_use_rope", True)),
        "rope_normalize_coords": cfg.get("rope_normalize_coords"),
        "rope_shift_coords": cfg.get("rope_shift_coords"),
        "rope_jitter_coords": cfg.get("rope_jitter_coords"),
        "rope_rescale_coords": cfg.get("rope_rescale_coords"),
        "median_prompt_length": float(np.median(prompt_lengths)),
        "median_target_length": float(np.median(target_lengths)),
        "median_valid_prompt_tokens": float(np.median(prompt_valid_counts)),
        "median_valid_target_tokens": float(np.median(target_valid_counts)),
        "refine_head_present": bool("pred_refine_residual" in forward_outputs and "pred_xyz_refined" in forward_outputs),
        "forward_ms": float(forward_ms),
        "infer_ms": float(infer_ms),
    }
    if int(training_steps) > 0:
        bench_cfg = dict(cfg)
        bench_cfg["wandb_project"] = None
        bench_cfg["val_fraction"] = 0.0
        bench_cfg["save_final_checkpoint"] = False
        bench_cfg["ckpt_at_step_zero"] = False
        bench_cfg["log_frequency"] = max(int(training_steps) + 1, int(cfg.get("log_frequency", 100)))
        bench_cfg["ckpt_frequency"] = max(int(training_steps) + 1, int(cfg.get("ckpt_frequency", 5000)))
        with tempfile.TemporaryDirectory(prefix="autoreg_mesh_bench_") as tmpdir:
            bench_cfg["out_dir"] = tmpdir
            start = time.perf_counter()
            train_result = run_autoreg_mesh_training(
                bench_cfg,
                dataset=dataset,
                model=None,
                device=device,
                max_steps=int(training_steps),
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
        startup_ms = float(train_result.get("startup_ms", 0.0))
        measured_train_ms = max(0.0, elapsed_ms - startup_ms)
        mean_train_step_ms = measured_train_ms / float(max(1, int(training_steps)))
        world_size = int(train_result.get("world_size", 1))
        per_rank_batch_size = int(cfg["batch_size"])
        effective_global_batch_size = int(per_rank_batch_size * world_size)
        result.update(
            {
                "training_steps": int(training_steps),
                "rank": int(train_result.get("rank", 0)),
                "world_size": world_size,
                "per_rank_batch_size": per_rank_batch_size,
                "effective_global_batch_size": effective_global_batch_size,
                "startup_ms": startup_ms,
                "mean_train_step_ms": mean_train_step_ms,
                "samples_per_sec_effective": (
                    (effective_global_batch_size * 1000.0 / mean_train_step_ms)
                    if mean_train_step_ms > 0.0 else 0.0
                ),
            }
        )
    return result


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--sample-count", default=32, type=int, show_default=True)
@click.option("--device", default=None, type=str)
@click.option("--training-steps", default=0, type=int, show_default=True)
def benchmark(config_path: str, sample_count: int, device: str | None, training_steps: int) -> None:
    cfg = load_autoreg_mesh_config(Path(config_path))
    result = run_autoreg_mesh_benchmark(cfg, sample_count=sample_count, device=device, training_steps=training_steps)
    if int(result.get("world_size", 1)) <= 1 or int(result.get("rank", 0)) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    benchmark()
