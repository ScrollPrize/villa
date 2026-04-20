from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import tifffile
from torch.utils.data import Dataset

import vesuvius.neural_tracing.autoreg_mesh.infer as autoreg_mesh_infer
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.models.build.pretrained_backbones.rope import MixedRopePositionEmbedding
from vesuvius.neural_tracing.autoreg_mesh.benchmark import run_autoreg_mesh_benchmark
from vesuvius.neural_tracing.autoreg_mesh.config import validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.dataset import (
    _apply_spatial_augmentation,
    _apply_volume_only_augmentation,
    _compute_target_boundary_stats,
    _fast_prefilter_plan,
    _should_reject_boundary_stats,
    autoreg_mesh_collate,
)
from vesuvius.neural_tracing.autoreg_mesh.infer import infer_autoreg_mesh
from vesuvius.neural_tracing.autoreg_mesh.losses import (
    _build_distance_aware_coarse_targets,
    _geometry_metric_loss,
    _geometry_sd_loss,
    _masked_mean,
    _seam_edge_loss_from_sequence,
    _sequence_to_grid_torch,
    _triangle_barrier_loss_from_sequence,
    compute_autoreg_mesh_losses,
)
from vesuvius.neural_tracing.autoreg_mesh.model import AutoregMeshModel
from vesuvius.neural_tracing.autoreg_mesh.model import _batched_rope_from_coords
from vesuvius.neural_tracing.autoreg_mesh.serialization import deserialize_continuation_grid, serialize_split_conditioning_example
from vesuvius.neural_tracing.autoreg_mesh.train import (
    _build_spatial_split_groups,
    _choose_best_xy_slice,
    _edge_segment_on_z_slice,
    _expanded_bboxes_overlap,
    _find_cross_split_bbox_overlap,
    _geometry_metric_weight_active,
    _geometry_sd_weight_active,
    _initialize_distributed_runtime,
    _make_xy_slice_overlay_canvas,
    _metric_dict_mean_across_ranks,
    _next_batch,
    _rasterize_grid_on_xy_slice,
    _restrict_dataset_samples,
    _scheduled_sampling_feedback_state,
    _split_dataset,
    run_autoreg_mesh_training,
)
from vesuvius.neural_tracing.datasets.triplet_resampling import choose_replacement_index


def _tiny_dinovol_model_config() -> dict:
    return {
        "model_type": "v2",
        "input_channels": 1,
        "global_crops_size": [16, 16, 16],
        "local_crops_size": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "embed_dim": 48,
        "depth": 2,
        "num_heads": 4,
        "num_reg_tokens": 2,
        "mlp_ratio": 2.0,
        "drop_path_rate": 0.0,
        "qkv_fused": True,
    }


def _write_local_guide_checkpoint(path: Path) -> None:
    backbone = build_dinovol_2_backbone(_tiny_dinovol_model_config())
    teacher_state = {f"backbone.{key}": value.cpu() for key, value in backbone.state_dict().items()}
    checkpoint = {
        "config": {
            "model": _tiny_dinovol_model_config(),
            "dataset": {
                "global_crop_size": [16, 16, 16],
                "local_crop_size": [16, 16, 16],
            },
        },
        "teacher": teacher_state,
    }
    torch.save(checkpoint, path)


def _make_surface(rows: int, cols: int, *, z_offset: float, y_offset: float, x_offset: float) -> np.ndarray:
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    col_axis = np.arange(cols, dtype=np.float32)[None, :]
    row_grid = np.broadcast_to(row_axis, (rows, cols))
    col_grid = np.broadcast_to(col_axis, (rows, cols))
    z = z_offset + (row_grid * 0.5) + (col_grid * 0.25)
    y = y_offset + (row_grid * 2.0)
    x = x_offset + (col_grid * 2.0)
    return np.stack([z, y, x], axis=-1).astype(np.float32)


def _make_large_surface(rows: int, cols: int, *, horizontal_split: bool) -> np.ndarray:
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    col_axis = np.arange(cols, dtype=np.float32)[None, :]
    row_grid = np.broadcast_to(row_axis, (rows, cols))
    col_grid = np.broadcast_to(col_axis, (rows, cols))
    if horizontal_split:
        z = 8.0 + (row_grid * 0.25) + (col_grid * 0.05)
        y = 10.0 + (row_grid * 4.0)
        x = 15.0 + (col_grid * 2.0)
    else:
        z = 8.0 + (row_grid * 0.05) + (col_grid * 0.25)
        y = 15.0 + (row_grid * 2.0)
        x = 10.0 + (col_grid * 4.0)
    return np.stack([z, y, x], axis=-1).astype(np.float32)


def _make_sample(direction: str) -> dict:
    if direction in {"left", "right"}:
        full = _make_surface(3, 4, z_offset=4.0, y_offset=2.0, x_offset=3.0)
        cond = full[:, :2] if direction == "left" else full[:, 2:]
        masked = full[:, 2:] if direction == "left" else full[:, :2]
    else:
        full = _make_surface(4, 3, z_offset=5.0, y_offset=1.0, x_offset=4.0)
        cond = full[:2, :] if direction == "up" else full[2:, :]
        masked = full[2:, :] if direction == "up" else full[:2, :]

    serialized = serialize_split_conditioning_example(
        cond_zyxs_local=cond,
        masked_zyxs_local=masked,
        direction=direction,
        volume_shape=(16, 16, 16),
        patch_size=(8, 8, 8),
        offset_num_bins=(4, 4, 4),
        frontier_band_width=4,
    )
    volume = torch.randn(1, 16, 16, 16, dtype=torch.float32)
    return {
        "volume": volume,
        "vol_tokens": None,
        "prompt_tokens": {
            "coarse_ids": torch.from_numpy(serialized["prompt_tokens"]["coarse_ids"]).to(torch.long),
            "offset_bins": torch.from_numpy(serialized["prompt_tokens"]["offset_bins"]).to(torch.long),
            "xyz": torch.from_numpy(serialized["prompt_tokens"]["xyz"]).to(torch.float32),
            "strip_positions": torch.from_numpy(serialized["prompt_tokens"]["strip_positions"]).to(torch.long),
            "strip_coords": torch.from_numpy(serialized["prompt_tokens"]["strip_coords"]).to(torch.float32),
            "valid_mask": torch.from_numpy(serialized["prompt_tokens"]["valid_mask"]).to(torch.bool),
        },
        "prompt_meta": serialized["prompt_meta"],
        "conditioning_grid_local": torch.from_numpy(serialized["conditioning_grid_local"]).to(torch.float32),
        "prompt_anchor_xyz": torch.from_numpy(serialized["prompt_anchor_xyz"]).to(torch.float32),
        "prompt_anchor_valid": torch.tensor(bool(serialized["prompt_anchor_valid"]), dtype=torch.bool),
        "prompt_grid_local": torch.from_numpy(serialized["prompt_grid_local"]).to(torch.float32),
        "target_coarse_ids": torch.from_numpy(serialized["target_coarse_ids"]).to(torch.long),
        "target_offset_bins": torch.from_numpy(serialized["target_offset_bins"]).to(torch.long),
        "target_valid_mask": torch.from_numpy(serialized["target_valid_mask"]).to(torch.bool),
        "target_stop": torch.from_numpy(serialized["target_stop"]).to(torch.float32),
        "target_xyz": torch.from_numpy(serialized["target_xyz"]).to(torch.float32),
        "target_bin_center_xyz": torch.from_numpy(serialized["target_bin_center_xyz"]).to(torch.float32),
        "target_strip_positions": torch.from_numpy(serialized["target_strip_positions"]).to(torch.long),
        "target_strip_coords": torch.from_numpy(serialized["target_strip_coords"]).to(torch.float32),
        "target_grid_local": torch.from_numpy(serialized["target_grid_local"]).to(torch.float32),
        "direction": direction,
        "direction_id": torch.tensor(serialized["direction_id"], dtype=torch.long),
        "strip_length": torch.tensor(serialized["strip_length"], dtype=torch.long),
        "num_strips": torch.tensor(serialized["num_strips"], dtype=torch.long),
        "min_corner": torch.zeros(3, dtype=torch.float32),
        "world_bbox": torch.tensor((0.0, 16.0, 0.0, 16.0, 0.0, 16.0), dtype=torch.float32),
        "target_grid_shape": torch.tensor(serialized["target_grid_shape"], dtype=torch.long),
        "wrap_metadata": {"segment_uuid": f"synthetic_{direction}"},
    }


def _make_large_sample(direction: str) -> dict:
    if direction in {"left", "right"}:
        full = _make_large_surface(3, 4, horizontal_split=True)
        cond = full[:, :2] if direction == "left" else full[:, 2:]
        masked = full[:, 2:] if direction == "left" else full[:, :2]
    else:
        full = _make_large_surface(4, 3, horizontal_split=False)
        cond = full[:2, :] if direction == "up" else full[2:, :]
        masked = full[2:, :] if direction == "up" else full[:2, :]

    serialized = serialize_split_conditioning_example(
        cond_zyxs_local=cond,
        masked_zyxs_local=masked,
        direction=direction,
        volume_shape=(32, 32, 32),
        patch_size=(8, 8, 8),
        offset_num_bins=(4, 4, 4),
        frontier_band_width=4,
    )
    volume = torch.randn(1, 32, 32, 32, dtype=torch.float32)
    return {
        "volume": volume,
        "vol_tokens": None,
        "prompt_tokens": {
            "coarse_ids": torch.from_numpy(serialized["prompt_tokens"]["coarse_ids"]).to(torch.long),
            "offset_bins": torch.from_numpy(serialized["prompt_tokens"]["offset_bins"]).to(torch.long),
            "xyz": torch.from_numpy(serialized["prompt_tokens"]["xyz"]).to(torch.float32),
            "strip_positions": torch.from_numpy(serialized["prompt_tokens"]["strip_positions"]).to(torch.long),
            "strip_coords": torch.from_numpy(serialized["prompt_tokens"]["strip_coords"]).to(torch.float32),
            "valid_mask": torch.from_numpy(serialized["prompt_tokens"]["valid_mask"]).to(torch.bool),
        },
        "prompt_meta": serialized["prompt_meta"],
        "conditioning_grid_local": torch.from_numpy(serialized["conditioning_grid_local"]).to(torch.float32),
        "prompt_anchor_xyz": torch.from_numpy(serialized["prompt_anchor_xyz"]).to(torch.float32),
        "prompt_anchor_valid": torch.tensor(bool(serialized["prompt_anchor_valid"]), dtype=torch.bool),
        "prompt_grid_local": torch.from_numpy(serialized["prompt_grid_local"]).to(torch.float32),
        "target_coarse_ids": torch.from_numpy(serialized["target_coarse_ids"]).to(torch.long),
        "target_offset_bins": torch.from_numpy(serialized["target_offset_bins"]).to(torch.long),
        "target_valid_mask": torch.from_numpy(serialized["target_valid_mask"]).to(torch.bool),
        "target_stop": torch.from_numpy(serialized["target_stop"]).to(torch.float32),
        "target_xyz": torch.from_numpy(serialized["target_xyz"]).to(torch.float32),
        "target_bin_center_xyz": torch.from_numpy(serialized["target_bin_center_xyz"]).to(torch.float32),
        "target_strip_positions": torch.from_numpy(serialized["target_strip_positions"]).to(torch.long),
        "target_strip_coords": torch.from_numpy(serialized["target_strip_coords"]).to(torch.float32),
        "target_grid_local": torch.from_numpy(serialized["target_grid_local"]).to(torch.float32),
        "direction": direction,
        "direction_id": torch.tensor(serialized["direction_id"], dtype=torch.long),
        "strip_length": torch.tensor(serialized["strip_length"], dtype=torch.long),
        "num_strips": torch.tensor(serialized["num_strips"], dtype=torch.long),
        "min_corner": torch.zeros(3, dtype=torch.float32),
        "world_bbox": torch.tensor((0.0, 32.0, 0.0, 32.0, 0.0, 32.0), dtype=torch.float32),
        "target_grid_shape": torch.tensor(serialized["target_grid_shape"], dtype=torch.long),
        "wrap_metadata": {"segment_uuid": f"large_{direction}"},
    }


def _make_cached_vol_tokens(decoder_dim: int = 96) -> torch.Tensor:
    values = torch.linspace(-1.0, 1.0, steps=8 * decoder_dim, dtype=torch.float32)
    return values.reshape(8, decoder_dim)


def _make_sample_with_cached_tokens(direction: str, *, decoder_dim: int = 96) -> dict:
    sample = _make_sample(direction)
    sample["vol_tokens"] = _make_cached_vol_tokens(decoder_dim=decoder_dim)
    return sample


def _make_config(checkpoint_path: Path) -> dict:
    return {
        "dinov2_backbone": str(checkpoint_path),
        "input_shape": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "offset_num_bins": [4, 4, 4],
        "decoder_dim": 96,
        "decoder_depth": 2,
        "decoder_num_heads": 4,
        "cross_attention_every_n_blocks": 1,
        "frontier_band_width": 4,
        "batch_size": 1,
        "num_workers": 0,
        "val_num_workers": 0,
        "num_steps": 2,
        "val_fraction": 0.0,
        "val_batches_per_log": 1,
        "optimizer": {"name": "adamw", "learning_rate": 1e-3, "weight_decay": 0.0},
        "log_frequency": 1,
        "ckpt_frequency": 1,
        "save_final_checkpoint": False,
    }


def _make_cached_token_config() -> dict:
    return {
        "dinov2_backbone": None,
        "input_shape": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "offset_num_bins": [4, 4, 4],
        "decoder_dim": 96,
        "decoder_depth": 2,
        "decoder_num_heads": 4,
        "cross_attention_every_n_blocks": 1,
        "frontier_band_width": 4,
        "batch_size": 1,
        "num_workers": 0,
        "val_num_workers": 0,
        "num_steps": 2,
        "val_fraction": 0.0,
        "val_batches_per_log": 1,
        "optimizer": {"name": "adamw", "learning_rate": 1e-3, "weight_decay": 0.0},
        "log_frequency": 1,
        "ckpt_frequency": 1,
        "save_final_checkpoint": False,
    }


def _make_debias_config(checkpoint_path: Path) -> dict:
    config = _make_config(checkpoint_path)
    config["conditioning_feature_debias_mode"] = "orthogonal_project"
    config["conditioning_feature_debias_components"] = 4
    return config


def _make_factorized_cached_token_config() -> dict:
    config = _make_cached_token_config()
    config["coarse_prediction_mode"] = "axis_factorized"
    return config


def _make_muon_cached_token_config() -> dict:
    config = _make_cached_token_config()
    config["optimizer"] = {
        "name": "muon",
        "learning_rate": 0.02,
        "momentum": 0.95,
        "weight_decay": 0.01,
        "weight_decouple": True,
        "nesterov": True,
        "ns_steps": 5,
        "use_adjusted_lr": False,
        "adamw_lr": 3e-4,
        "adamw_betas": [0.9, 0.95],
        "adamw_wd": 0.01,
        "adamw_eps": 1e-10,
        "maximize": False,
    }
    return config


def _make_large_cached_token_config() -> dict:
    config = _make_cached_token_config()
    config["input_shape"] = [32, 32, 32]
    config["crop_size"] = [32, 32, 32]
    return config


class _ListDataset(Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[int(idx)]


class _ResamplingSampleIndexDataset(Dataset):
    def __init__(self, items) -> None:
        self.sample_index = list(items)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        idx = int(idx)
        value = self.sample_index[idx]
        if value == (0, 0):
            replacement = choose_replacement_index(self.sample_index, attempted_indices={idx})
            if replacement is None:
                raise RuntimeError("failed to resample")
            return self[int(replacement)]
        return {"value": value}


class _SplitCloneAutoregDataset:
    def __init__(self, apply_volume_only_augmentation: bool, apply_spatial_augmentation: bool = False, patch_metadata=None) -> None:
        self.sample_index = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self._apply_volume_only_augmentation = bool(apply_volume_only_augmentation)
        self._apply_spatial_augmentation = bool(apply_spatial_augmentation)
        self._patch_metadata = patch_metadata or {"sample_index": tuple(self.sample_index)}

    def __len__(self) -> int:
        return len(self.sample_index)

    def export_patch_metadata(self):
        return dict(self._patch_metadata)


def _make_fake_patch(source: str, chunk_id: tuple[int, int, int], world_bbox: tuple[float, float, float, float, float, float]):
    return SimpleNamespace(
        volume=SimpleNamespace(path=str(source)),
        scale=0,
        chunk_id=tuple(int(v) for v in chunk_id),
        world_bbox=tuple(float(v) for v in world_bbox),
    )


class _LeakageSplitAutoregDataset:
    def __init__(self, patch_metadata=None) -> None:
        if patch_metadata is None:
            self.patches = [
                _make_fake_patch("volA", (0, 0, 0), (0.0, 64.0, 0.0, 64.0, 0.0, 64.0)),
                _make_fake_patch("volA", (0, 1, 0), (0.0, 64.0, 32.0, 96.0, 0.0, 64.0)),
                _make_fake_patch("volA", (0, 2, 0), (0.0, 64.0, 128.0, 192.0, 0.0, 64.0)),
                _make_fake_patch("volA", (0, 3, 0), (0.0, 64.0, 224.0, 288.0, 0.0, 64.0)),
            ]
            self.sample_index = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)]
            self._patch_metadata = {
                "patches": self.patches,
                "sample_index": tuple(self.sample_index),
            }
        else:
            self._patch_metadata = dict(patch_metadata)
            self.patches = list(self._patch_metadata["patches"])
            self.sample_index = list(self._patch_metadata["sample_index"])

    def __len__(self) -> int:
        return len(self.sample_index)

    def export_patch_metadata(self):
        return dict(self._patch_metadata)


class _FakeWandbImage:
    def __init__(self, data, caption=None) -> None:
        self.data = np.asarray(data)
        self.caption = caption


class _FakeWandbRun:
    def __init__(self, run_id: str) -> None:
        self.id = str(run_id)


class _FakeWandbModule:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.logs: list[dict] = []
        self.finish_calls = 0
        self.run = None
        self._counter = 0

    def init(self, **kwargs):
        self.init_calls.append(dict(kwargs))
        run_id = kwargs.get("id")
        if run_id is None:
            self._counter += 1
            run_id = f"fake-run-{self._counter}"
        self.run = _FakeWandbRun(run_id)
        return self.run

    def log(self, data, step=None):
        self.logs.append({"data": dict(data), "step": step})

    def finish(self):
        self.finish_calls += 1
        self.run = None

    def Image(self, data, caption=None):
        return _FakeWandbImage(data, caption=caption)


class _RecordingWandbModule:
    def __init__(self, rank: int, log_path: Path) -> None:
        self.rank = int(rank)
        self.log_path = Path(log_path)
        self.run = None
        self._log = {"init_calls": 0, "log_calls": 0, "finish_calls": 0}

    def init(self, **kwargs):
        self._log["init_calls"] += 1
        self._log["init_kwargs"] = dict(kwargs)
        self.run = type("_Run", (), {"id": f"rank-{self.rank}"})()
        self.log_path.write_text(json.dumps(self._log))
        return self.run

    def log(self, data, step=None):
        self._log["log_calls"] += 1
        self._log["last_step"] = step
        self.log_path.write_text(json.dumps(self._log))

    def finish(self):
        self._log["finish_calls"] += 1
        self.run = None
        self.log_path.write_text(json.dumps(self._log))

    def Image(self, data, caption=None):
        return _FakeWandbImage(data, caption=caption)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_env(rank: int, world_size: int, port: int) -> dict[str, str]:
    return {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(int(port)),
        "WORLD_SIZE": str(int(world_size)),
        "RANK": str(int(rank)),
        "LOCAL_RANK": str(int(rank)),
    }


def _ddp_training_worker(rank: int, world_size: int, port: int, tmpdir: str) -> None:
    old_env = {key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK")}
    os.environ.update(_ddp_env(rank, world_size, port))
    try:
        config = _make_cached_token_config()
        config.update(
            {
                "out_dir": str(Path(tmpdir) / "ddp_train"),
                "wandb_project": "mesh-ddp",
                "wandb_log_images": False,
                "save_final_checkpoint": False,
                "ckpt_frequency": 1,
                "val_fraction": 0.5,
                "log_frequency": 1,
            }
        )
        sys.modules["wandb"] = _RecordingWandbModule(rank, Path(tmpdir) / f"wandb_rank{rank}.json")
        result = run_autoreg_mesh_training(config, dataset=_make_training_dataset(), device="cpu", max_steps=1)
        summary = {
            "rank": int(result["rank"]),
            "world_size": int(result["world_size"]),
            "is_main_process": bool(result["is_main_process"]),
            "history_len": int(len(result["history"])),
            "checkpoint_paths": list(result["checkpoint_paths"]),
            "startup_ms": float(result["startup_ms"]),
        }
        (Path(tmpdir) / f"rank{rank}.json").write_text(json.dumps(summary))
    finally:
        sys.modules.pop("wandb", None)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _ddp_benchmark_worker(rank: int, world_size: int, port: int, tmpdir: str) -> None:
    old_env = {key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK")}
    os.environ.update(_ddp_env(rank, world_size, port))
    try:
        result = run_autoreg_mesh_benchmark(
            _make_cached_token_config(),
            dataset=_make_training_dataset(),
            model=AutoregMeshModel(_make_cached_token_config()).eval(),
            device="cpu",
            sample_count=1,
            training_steps=1,
        )
        (Path(tmpdir) / f"bench_rank{rank}.json").write_text(json.dumps(result))
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _make_training_dataset() -> _ListDataset:
    return _ListDataset([
        _make_sample_with_cached_tokens("left"),
        _make_sample_with_cached_tokens("right"),
        _make_sample_with_cached_tokens("up"),
        _make_sample_with_cached_tokens("down"),
    ])


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif key == "prompt_tokens":
            moved[key] = {inner_key: inner_value.to(device) for inner_key, inner_value in value.items()}
        else:
            moved[key] = value
    return moved


class _DummySamplingInferenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(()))
        self.offset_num_bins = (4, 4, 4)
        self.coarse_prediction_mode = "joint_pointer"
        self.coarse_grid_shape = (2, 2, 2)

    def encode_conditioning(self, volume, vol_tokens=None):
        device = volume.device
        return {
            "memory_tokens": torch.zeros((1, 8, 4), device=device, dtype=torch.float32),
            "memory_patch_centers": torch.zeros((1, 8, 3), device=device, dtype=torch.float32),
        }

    def forward_from_encoded(self, batch, *, memory_tokens, memory_patch_centers):
        del batch, memory_tokens, memory_patch_centers
        device = self._param.device
        coarse_logits = torch.tensor([[[0.0, 5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]]], device=device)
        offset_logits = torch.full((1, 1, 3, 4), -6.0, device=device)
        offset_logits[0, 0, 0, 0] = 6.0
        offset_logits[0, 0, 1, 0] = 6.0
        offset_logits[0, 0, 2, 0] = 6.0
        pred_coarse_ids = torch.tensor([[1]], dtype=torch.long, device=device)
        pred_offset_bins = torch.zeros((1, 1, 3), dtype=torch.long, device=device)
        pred_xyz = self.decode_local_xyz(pred_coarse_ids, pred_offset_bins)
        pred_refine_residual = torch.tensor([[[1.5, -0.5, 0.25]]], dtype=torch.float32, device=device)
        return {
            "coarse_logits": coarse_logits,
            "coarse_axis_logits": None,
            "offset_logits": offset_logits,
            "stop_logits": torch.full((1, 1), -8.0, dtype=torch.float32, device=device),
            "pred_coarse_ids": pred_coarse_ids,
            "pred_coarse_axis_ids": {
                "z": torch.zeros((1, 1), dtype=torch.long, device=device),
                "y": torch.zeros((1, 1), dtype=torch.long, device=device),
                "x": torch.ones((1, 1), dtype=torch.long, device=device),
            },
            "pred_offset_bins": pred_offset_bins,
            "pred_refine_residual": pred_refine_residual,
            "pred_xyz": pred_xyz,
            "pred_xyz_soft": pred_xyz + pred_refine_residual,
            "pred_xyz_refined": pred_xyz + pred_refine_residual,
            "coarse_grid_shape": self.coarse_grid_shape,
            "coarse_prediction_mode": self.coarse_prediction_mode,
        }

    def decode_local_xyz(self, coarse_ids: torch.Tensor, offset_bins: torch.Tensor) -> torch.Tensor:
        coarse = coarse_ids.to(torch.long)
        gyx = self.coarse_grid_shape[1] * self.coarse_grid_shape[2]
        z = coarse // gyx
        rem = coarse % gyx
        y = rem // self.coarse_grid_shape[2]
        x = rem % self.coarse_grid_shape[2]
        starts = torch.stack([z, y, x], dim=-1).to(torch.float32) * 8.0
        widths = torch.tensor([2.0, 2.0, 2.0], device=coarse.device, dtype=torch.float32)
        return starts + (offset_bins.to(torch.float32) + 0.5) * widths.view(1, 1, 3)


class _ModeTrackingInferenceModel(_DummySamplingInferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.training_flags: list[tuple[str, bool]] = []

    def encode_conditioning(self, volume, vol_tokens=None):
        self.training_flags.append(("encode_conditioning", bool(self.training)))
        return super().encode_conditioning(volume, vol_tokens=vol_tokens)

    def forward_from_encoded(self, batch, *, memory_tokens, memory_patch_centers):
        self.training_flags.append(("forward_from_encoded", bool(self.training)))
        return super().forward_from_encoded(
            batch,
            memory_tokens=memory_tokens,
            memory_patch_centers=memory_patch_centers,
        )


def test_autoreg_mesh_model_forward_and_losses_are_finite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    model = AutoregMeshModel(config)
    assert isinstance(model.rope, MixedRopePositionEmbedding)

    batch = autoreg_mesh_collate([_make_sample("left"), _make_sample("up")])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)
    losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
    )

    assert outputs["coarse_logits"].shape[:2] == batch["target_coarse_ids"].shape
    assert outputs["coarse_logits"].shape[-1] == 8
    assert outputs["offset_logits"].shape[:3] == (*batch["target_coarse_ids"].shape, 3)
    assert outputs["offset_logits"].shape[-1] == 4
    assert outputs["stop_logits"].shape == batch["target_coarse_ids"].shape
    assert outputs["pred_refine_residual"].shape == batch["target_xyz"].shape
    assert outputs["pred_xyz_soft"].shape == batch["target_xyz"].shape
    assert outputs["pred_xyz_refined"].shape == batch["target_xyz"].shape
    assert batch["target_valid_mask"].shape == batch["target_mask"].shape
    assert batch["target_supervision_mask"].shape == batch["target_mask"].shape
    for value in losses.values():
        assert torch.isfinite(value)
    assert "occupancy_metric" in losses
    assert "coarse_excess_nll" in losses
    assert "geometry_metric_loss" in losses
    assert "geometry_sd_loss" in losses


def test_axis_factorized_autoreg_mesh_model_forward_and_losses_are_finite() -> None:
    config = _make_factorized_cached_token_config()
    model = AutoregMeshModel(config)

    batch = autoreg_mesh_collate([_make_sample_with_cached_tokens("left"), _make_sample_with_cached_tokens("up")])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)
    losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
    )

    assert outputs["coarse_prediction_mode"] == "axis_factorized"
    assert outputs["coarse_logits"] is None
    assert outputs["coarse_axis_logits"]["z"].shape == (*batch["target_coarse_ids"].shape, 2)
    assert outputs["coarse_axis_logits"]["y"].shape == (*batch["target_coarse_ids"].shape, 2)
    assert outputs["coarse_axis_logits"]["x"].shape == (*batch["target_coarse_ids"].shape, 2)
    assert outputs["pred_coarse_axis_ids"]["z"].shape == batch["target_coarse_ids"].shape
    assert outputs["pred_coarse_axis_ids"]["y"].shape == batch["target_coarse_ids"].shape
    assert outputs["pred_coarse_axis_ids"]["x"].shape == batch["target_coarse_ids"].shape
    assert outputs["pred_coarse_ids"].shape == batch["target_coarse_ids"].shape
    assert torch.isfinite(losses["coarse_z_loss"])
    assert torch.isfinite(losses["coarse_y_loss"])
    assert torch.isfinite(losses["coarse_x_loss"])
    for value in losses.values():
        assert torch.isfinite(value)


def test_muon_autoreg_mesh_config_validates_and_training_smoke_runs(tmp_path: Path) -> None:
    config = _make_muon_cached_token_config()
    validated = validate_autoreg_mesh_config(config)
    assert validated["optimizer"]["name"] == "muon"
    assert validated["optimizer"]["momentum"] == pytest.approx(0.95)
    assert validated["optimizer"]["adamw_betas"] == [0.9, 0.95]

    config["out_dir"] = str(tmp_path / "runs_muon")
    dataset = _make_training_dataset()
    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=2)

    assert len(result["history"]) == 2
    assert torch.isfinite(torch.tensor(result["history"][-1]["loss"]))


def test_debias_autoreg_mesh_model_forward_and_losses_are_finite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_debias_config(checkpoint)
    model = AutoregMeshModel(config)

    batch = autoreg_mesh_collate([_make_sample("left"), _make_sample("up")])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)
    losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
    )

    assert model.conditioning_feature_debias_basis.shape == (48, 4)
    assert outputs["coarse_logits"].shape[-1] == 8
    for value in losses.values():
        assert torch.isfinite(value)


def test_model_train_keeps_frozen_backbone_in_eval_mode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    model = AutoregMeshModel(_make_config(checkpoint))

    assert model.backbone is not None
    assert model.backbone.training is False

    model.train()

    assert model.training is True
    assert model.backbone.training is False


def test_model_rejects_backbone_patch_size_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    config["patch_size"] = [16, 16, 16]

    with pytest.raises(ValueError, match="patch_size must match"):
        AutoregMeshModel(config)


def test_debias_projection_reduces_energy_in_learned_basis(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_debias_config(checkpoint)
    model = AutoregMeshModel(config).eval()

    volume = torch.zeros((1, 1, 16, 16, 16), dtype=torch.float32)
    with torch.no_grad():
        features = model.backbone(volume)[0]
    raw_tokens = features.flatten(2).transpose(1, 2).contiguous()
    basis = model.conditioning_feature_debias_basis

    raw_proj = torch.matmul(raw_tokens.to(torch.float32), basis).pow(2).sum(dim=-1).mean()
    debiased = model._debias_conditioning_features(raw_tokens)
    deb_proj = torch.matmul(debiased.to(torch.float32), basis).pow(2).sum(dim=-1).mean()

    assert debiased.shape == raw_tokens.shape
    assert torch.isfinite(debiased).all()
    assert deb_proj.item() < raw_proj.item()


def test_position_refine_head_is_zero_initialized() -> None:
    model = AutoregMeshModel(_make_cached_token_config())
    assert torch.count_nonzero(model.position_refine_head.weight).item() == 0
    assert torch.count_nonzero(model.position_refine_head.bias).item() == 0


def test_scheduled_sampling_uses_stripwise_replacement() -> None:
    model = AutoregMeshModel(_make_cached_token_config())
    batch = autoreg_mesh_collate([_make_sample_with_cached_tokens("left")])
    teacher_outputs = {
        "pred_coarse_ids": batch["target_coarse_ids"] + 1000,
        "pred_offset_bins": batch["target_offset_bins"] + 10,
        "pred_xyz": batch["target_xyz"] + 10.0,
        "pred_xyz_refined": batch["target_xyz"] + 20.0,
    }

    torch.manual_seed(0)
    generation_inputs = model._build_scheduled_generation_inputs(
        batch,
        teacher_outputs=teacher_outputs,
        scheduled_sampling_prob=0.5,
        scheduled_sampling_pattern="stripwise_full_strip_greedy",
        offset_feedback_enabled=True,
        refine_feedback_enabled=True,
    )

    replaced = generation_inputs["coarse_ids"] != model._build_teacher_forced_generation_inputs(batch)["coarse_ids"]
    shifted_strip_ids = torch.zeros_like(batch["target_mask"], dtype=torch.long)
    shifted_strip_ids[:, 1:] = batch["target_strip_positions"][:, :-1, 0]
    for strip_id in range(int(batch["num_strips"][0].item())):
        strip_mask = shifted_strip_ids[0] == strip_id
        strip_mask[0] = False
        if bool(strip_mask.any()):
            values = replaced[0, strip_mask]
            assert bool(torch.all(values == values[0]))


def test_scheduled_sampling_keeps_offsets_and_xyz_teacher_forced_before_offset_loss() -> None:
    model = AutoregMeshModel(_make_cached_token_config())
    batch = autoreg_mesh_collate([_make_sample_with_cached_tokens("left")])
    teacher_inputs = model._build_teacher_forced_generation_inputs(batch)
    teacher_outputs = {
        "pred_coarse_ids": batch["target_coarse_ids"] + 1000,
        "pred_offset_bins": batch["target_offset_bins"] + 10,
        "pred_xyz": batch["target_xyz"] + 10.0,
        "pred_xyz_refined": batch["target_xyz"] + 20.0,
    }

    generation_inputs = model._build_scheduled_generation_inputs(
        batch,
        teacher_outputs=teacher_outputs,
        scheduled_sampling_prob=1.0,
        scheduled_sampling_pattern="stripwise_full_strip_greedy",
        offset_feedback_enabled=False,
        refine_feedback_enabled=False,
    )

    assert torch.equal(generation_inputs["offset_bins"], teacher_inputs["offset_bins"])
    assert torch.allclose(generation_inputs["xyz"], teacher_inputs["xyz"])
    assert not torch.equal(generation_inputs["coarse_ids"], teacher_inputs["coarse_ids"])


def test_scheduled_sampling_uses_bin_center_xyz_before_refine_is_active() -> None:
    model = AutoregMeshModel(_make_cached_token_config())
    batch = autoreg_mesh_collate([_make_sample_with_cached_tokens("left")])
    teacher_outputs = {
        "pred_coarse_ids": batch["target_coarse_ids"] + 1000,
        "pred_offset_bins": batch["target_offset_bins"] + 10,
        "pred_xyz": batch["target_xyz"] + 10.0,
        "pred_xyz_refined": batch["target_xyz"] + 20.0,
    }

    generation_inputs = model._build_scheduled_generation_inputs(
        batch,
        teacher_outputs=teacher_outputs,
        scheduled_sampling_prob=1.0,
        scheduled_sampling_pattern="stripwise_full_strip_greedy",
        offset_feedback_enabled=True,
        refine_feedback_enabled=False,
    )

    assert torch.allclose(generation_inputs["xyz"][:, 1:], teacher_outputs["pred_xyz"][:, :-1])


def test_scheduled_sampling_uses_refined_xyz_after_refine_is_active() -> None:
    model = AutoregMeshModel(_make_cached_token_config())
    batch = autoreg_mesh_collate([_make_sample_with_cached_tokens("left")])
    teacher_outputs = {
        "pred_coarse_ids": batch["target_coarse_ids"] + 1000,
        "pred_offset_bins": batch["target_offset_bins"] + 10,
        "pred_xyz": batch["target_xyz"] + 10.0,
        "pred_xyz_refined": batch["target_xyz"] + 20.0,
    }

    generation_inputs = model._build_scheduled_generation_inputs(
        batch,
        teacher_outputs=teacher_outputs,
        scheduled_sampling_prob=1.0,
        scheduled_sampling_pattern="stripwise_full_strip_greedy",
        offset_feedback_enabled=True,
        refine_feedback_enabled=True,
    )

    assert torch.allclose(generation_inputs["xyz"][:, 1:], teacher_outputs["pred_xyz_refined"][:, :-1])


def test_edge_segment_on_z_slice_crossing_and_far_cases() -> None:
    crossing = _edge_segment_on_z_slice(
        np.array([4.0, 2.0, 3.0], dtype=np.float32),
        np.array([6.0, 4.0, 7.0], dtype=np.float32),
        z_slice=5,
        depth_tolerance=0.25,
    )
    far = _edge_segment_on_z_slice(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([2.0, 4.0, 7.0], dtype=np.float32),
        z_slice=10,
        depth_tolerance=0.25,
    )

    assert crossing is not None
    assert far is None


def test_choose_best_xy_slice_returns_in_bounds_slice() -> None:
    sample = _make_sample("left")
    z_slice = _choose_best_xy_slice(
        [
            sample["prompt_grid_local"].numpy(),
            sample["target_grid_local"].numpy(),
        ],
        depth=16,
        depth_tolerance=0.75,
    )
    assert 0 <= z_slice < 16


def test_rasterize_grid_on_xy_slice_draws_near_slice_and_skips_far_edges() -> None:
    grid = np.array(
        [
            [[5.0, 2.0, 2.0], [5.0, 2.0, 6.0]],
            [[12.0, 6.0, 2.0], [12.0, 6.0, 6.0]],
        ],
        dtype=np.float32,
    )
    near_mask = _rasterize_grid_on_xy_slice(
        grid[:1],
        z_slice=5,
        panel_shape=(16, 16),
        line_thickness=1,
        depth_tolerance=0.5,
    )
    far_mask = _rasterize_grid_on_xy_slice(
        grid[1:],
        z_slice=5,
        panel_shape=(16, 16),
        line_thickness=1,
        depth_tolerance=0.5,
    )

    assert float(near_mask.sum()) > 0.0
    assert float(far_mask.sum()) == pytest.approx(0.0)


def test_xy_slice_overlay_canvas_returns_rgb_image() -> None:
    sample = _make_sample("left")
    image = _make_xy_slice_overlay_canvas(
        volume=sample["volume"].numpy(),
        prompt_grid_local=sample["prompt_grid_local"].numpy(),
        target_grid_local=sample["target_grid_local"].numpy(),
        pred_grid_local=sample["target_grid_local"].numpy(),
        line_thickness=1,
        depth_tolerance=0.75,
    )

    assert image.ndim == 3
    assert image.shape[-1] == 3
    assert image.shape[0] > 16
    assert image.dtype == np.uint8


def test_autoreg_mesh_smoke_training_runs_two_steps(tmp_path: Path) -> None:
    config = _make_cached_token_config()
    config["out_dir"] = str(tmp_path / "runs")
    dataset = _make_training_dataset()
    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=2)

    assert len(result["history"]) == 2
    assert np.isfinite(result["final_metrics"]["loss"])
    assert np.isfinite(result["final_metrics"]["coarse_loss"])
    assert np.isfinite(result["final_metrics"]["offset_loss"])
    assert np.isfinite(result["final_metrics"]["stop_loss"])
    assert "val_loss" not in result["final_metrics"]
    assert result["wandb_run_id"] is None
    assert Path(result["checkpoint_paths"][0]).exists()


def test_autoreg_mesh_inference_reconstructs_lattice(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    model = AutoregMeshModel(config).eval()

    sample = _make_sample("down")
    batch = autoreg_mesh_collate([sample])
    result = infer_autoreg_mesh(
        model,
        batch,
        max_steps=int(sample["target_coarse_ids"].shape[0]),
        stop_probability_threshold=1.1,
        greedy=True,
        save_path=tmp_path,
    )

    target_shape = tuple(int(v) for v in sample["target_grid_shape"].tolist())
    conditioning_shape = tuple(int(v) for v in sample["conditioning_grid_local"].shape[:2])
    assert result["predicted_continuation_vertices_local"].shape[0] == int(sample["target_coarse_ids"].shape[0])
    assert result["predicted_bin_center_vertices_local"].shape == result["predicted_continuation_vertices_local"].shape
    assert result["continuation_grid_local"].shape == (*target_shape, 3)
    assert result["full_grid_local"].shape[-1] == 3
    assert result["full_grid_local"].shape == (conditioning_shape[0] + target_shape[0], conditioning_shape[1], 3)
    if sample["direction"] in {"left", "up"}:
        merged_cond = result["full_grid_local"][: conditioning_shape[0]]
    elif sample["direction"] in {"right", "down"}:
        merged_cond = result["full_grid_local"][-conditioning_shape[0]:]
    else:
        raise AssertionError(f"unexpected direction {sample['direction']!r}")
    np.testing.assert_allclose(merged_cond, sample["conditioning_grid_local"].numpy())
    assert result["saved_tifxyz_path"] is not None
    z_tif = tifffile.imread(Path(result["saved_tifxyz_path"]) / "z.tif")
    assert tuple(z_tif.shape) == result["full_grid_local"].shape[:2]


def test_axis_factorized_autoreg_mesh_inference_reconstructs_lattice() -> None:
    config = _make_factorized_cached_token_config()
    model = AutoregMeshModel(config).eval()

    sample = _make_sample_with_cached_tokens("down")
    result = infer_autoreg_mesh(
        model,
        sample,
        max_steps=int(sample["target_coarse_ids"].shape[0]),
        stop_probability_threshold=1.1,
        greedy=True,
    )

    target_shape = tuple(int(v) for v in sample["target_grid_shape"].tolist())
    assert result["predicted_continuation_vertices_local"].shape[0] == int(sample["target_coarse_ids"].shape[0])
    assert result["continuation_grid_local"].shape == (*target_shape, 3)
    assert "predicted_coarse_axis_ids" in result
    assert tuple(result["predicted_coarse_axis_ids"].keys()) == ("z", "y", "x")
    assert result["predicted_coarse_axis_ids"]["z"].shape[0] == int(sample["target_coarse_ids"].shape[0])


def test_infer_rejects_nonpositive_max_steps() -> None:
    config = _make_factorized_cached_token_config()
    model = AutoregMeshModel(config).eval()
    sample = _make_sample_with_cached_tokens("left")

    with pytest.raises(ValueError, match="max_steps"):
        infer_autoreg_mesh(model, sample, max_steps=0)
    with pytest.raises(ValueError, match="max_steps"):
        infer_autoreg_mesh(model, sample, max_steps=-1)


def test_infer_runs_under_eval_mode_and_restores_training_state() -> None:
    model = _ModeTrackingInferenceModel()
    model.train()
    sample = _make_sample("left")

    result = infer_autoreg_mesh(
        model,
        sample,
        max_steps=1,
        stop_probability_threshold=1.1,
        greedy=True,
    )

    assert model.training is True
    assert result["predicted_continuation_vertices_local"].shape[0] == 1
    assert model.training_flags
    assert all(flag is False for _, flag in model.training_flags)


def test_autoreg_mesh_inference_uses_sampled_xyz_for_non_greedy_rollout(monkeypatch) -> None:
    model = _DummySamplingInferenceModel().eval()
    sample = _make_sample("left")
    sampled = iter([
        torch.tensor(0, dtype=torch.long),
        torch.tensor(1, dtype=torch.long),
        torch.tensor(2, dtype=torch.long),
        torch.tensor(3, dtype=torch.long),
    ])

    monkeypatch.setattr(
        autoreg_mesh_infer,
        "_sample_from_logits",
        lambda logits, *, greedy, temperature=1.0, top_k=None, top_p=None: next(sampled),
    )

    result = infer_autoreg_mesh(
        model,
        sample,
        max_steps=1,
        stop_probability_threshold=1.1,
        greedy=False,
    )

    expected_bin_center = model.decode_local_xyz(
        torch.tensor([[0]], dtype=torch.long),
        torch.tensor([[[1, 2, 3]]], dtype=torch.long),
    )[0, 0].detach().cpu().numpy()
    expected_xyz = expected_bin_center + np.array([1.5, -0.5, 0.25], dtype=np.float32)

    np.testing.assert_allclose(result["predicted_bin_center_vertices_local"][0], expected_bin_center)
    np.testing.assert_allclose(result["predicted_continuation_vertices_local"][0], expected_xyz)


def test_autoreg_mesh_validation_metrics_are_logged_in_history(tmp_path: Path) -> None:
    config = _make_cached_token_config()
    config["out_dir"] = str(tmp_path / "runs_val")
    config["val_fraction"] = 0.5
    dataset = _make_training_dataset()

    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=1)

    assert len(result["history"]) == 1
    assert "val_loss" in result["history"][0]
    assert "val_coarse_loss" in result["history"][0]
    assert "val_coarse_excess_nll" in result["history"][0]
    assert "val_offset_loss" in result["history"][0]
    assert "val_stop_loss" in result["history"][0]
    assert "val_teacher_forced_pred_oob_fraction" in result["history"][0]
    assert "val_target_invalid_fraction" in result["history"][0]
    assert "val_coarse_constraint_keep_fraction" in result["history"][0]
    assert "val_first_strip_wrong_side_rate_refined" in result["history"][0]
    assert "rollout_val_xyz_l1_refined" in result["history"][0]
    assert "rollout_val_seam_edge_error" in result["history"][0]
    assert "rollout_val_triangle_flip_rate" in result["history"][0]
    assert "rollout_val_first_strip_wrong_side_rate" in result["history"][0]
    assert "rollout_val_pred_oob_fraction" in result["history"][0]
    assert "rollout_val_invalid_vertex_fraction" in result["history"][0]


def test_split_dataset_honors_disabled_train_volume_augmentation(tmp_path: Path) -> None:
    from vesuvius.neural_tracing.autoreg_mesh import train as train_module

    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = validate_autoreg_mesh_config(_make_config(checkpoint))
    config["volume_only_augmentation"]["enabled"] = False
    config["val_split_mode"] = "random_samples"

    dataset = _SplitCloneAutoregDataset(apply_volume_only_augmentation=False, apply_spatial_augmentation=False)
    clone_calls: list[tuple[bool, bool]] = []

    def _fake_clone(cfg, patch_metadata, *, apply_spatial_augmentation: bool, apply_volume_only_augmentation: bool):
        del cfg, patch_metadata
        clone_calls.append((bool(apply_spatial_augmentation), bool(apply_volume_only_augmentation)))
        return _SplitCloneAutoregDataset(
            apply_volume_only_augmentation=apply_volume_only_augmentation,
            apply_spatial_augmentation=apply_spatial_augmentation,
        )

    original_dataset_cls = train_module.AutoregMeshDataset
    original_clone = train_module._clone_autoreg_mesh_dataset
    try:
        train_module.AutoregMeshDataset = _SplitCloneAutoregDataset
        train_module._clone_autoreg_mesh_dataset = _fake_clone
        train_dataset, val_dataset, split_diagnostics = train_module._split_dataset(
            dataset,
            cfg=config,
            seed=0,
            val_fraction=0.5,
        )
    finally:
        train_module.AutoregMeshDataset = original_dataset_cls
        train_module._clone_autoreg_mesh_dataset = original_clone

    assert clone_calls == [(True, False), (False, False)]
    assert train_dataset is not None
    assert val_dataset is not None
    assert split_diagnostics == {}


def test_expanded_bboxes_overlap_helper_distinguishes_overlap_and_gaps() -> None:
    overlap_a = (0.0, 64.0, 0.0, 64.0, 0.0, 64.0)
    overlap_b = (0.0, 64.0, 32.0, 96.0, 0.0, 64.0)
    touching = (0.0, 64.0, 64.0, 128.0, 0.0, 64.0)
    disjoint = (0.0, 64.0, 80.0, 144.0, 0.0, 64.0)

    assert _expanded_bboxes_overlap(overlap_a, overlap_b, margin_voxels=0.0) is True
    assert _expanded_bboxes_overlap(overlap_a, touching, margin_voxels=0.0) is False
    assert _expanded_bboxes_overlap(overlap_a, touching, margin_voxels=1.0) is True
    assert _expanded_bboxes_overlap(overlap_a, disjoint, margin_voxels=0.0) is False


def test_build_spatial_split_groups_collapses_wraps_and_overlaps() -> None:
    dataset = _LeakageSplitAutoregDataset()

    groups = _build_spatial_split_groups(dataset, margin_voxels=0.0)

    assert groups == [[0, 1, 2], [3], [4]]


def test_split_dataset_spatial_groups_prevents_cross_split_bbox_overlap(tmp_path: Path) -> None:
    from vesuvius.neural_tracing.autoreg_mesh import train as train_module

    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = validate_autoreg_mesh_config(_make_config(checkpoint))
    config["val_split_mode"] = "spatial_groups"
    config["validation_leakage_margin_voxels"] = 0.0

    dataset = _LeakageSplitAutoregDataset()

    def _fake_clone(cfg, patch_metadata, *, apply_spatial_augmentation: bool, apply_volume_only_augmentation: bool):
        del cfg, apply_spatial_augmentation, apply_volume_only_augmentation
        return _LeakageSplitAutoregDataset(patch_metadata=patch_metadata)

    original_dataset_cls = train_module.AutoregMeshDataset
    original_clone = train_module._clone_autoreg_mesh_dataset
    try:
        train_module.AutoregMeshDataset = _LeakageSplitAutoregDataset
        train_module._clone_autoreg_mesh_dataset = _fake_clone
        train_dataset, val_dataset, split_diagnostics = _split_dataset(
            dataset,
            cfg=config,
            seed=0,
            val_fraction=0.4,
        )
    finally:
        train_module.AutoregMeshDataset = original_dataset_cls
        train_module._clone_autoreg_mesh_dataset = original_clone

    assert train_dataset is not None
    assert val_dataset is not None
    assert split_diagnostics["num_train_groups"] + split_diagnostics["num_val_groups"] == pytest.approx(3.0)
    assert split_diagnostics["max_group_size"] == pytest.approx(3.0)
    train_indices = [dataset.sample_index.index(item) for item in train_dataset.sample_index]
    val_indices = [dataset.sample_index.index(item) for item in val_dataset.sample_index]
    assert _find_cross_split_bbox_overlap(dataset, train_indices, val_indices, margin_voxels=0.0) is None


def test_restrict_dataset_samples_prevents_cross_split_resampling() -> None:
    train_dataset = _restrict_dataset_samples(_ResamplingSampleIndexDataset([(0, 0), (1, 0), (2, 0)]), [0, 1])
    val_dataset = _restrict_dataset_samples(_ResamplingSampleIndexDataset([(0, 0), (1, 0), (2, 0)]), [0, 2])

    assert train_dataset[0]["value"] == (1, 0)
    assert val_dataset[0]["value"] == (2, 0)


def test_occupancy_auxiliary_is_metric_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    model = AutoregMeshModel(config)

    batch = autoreg_mesh_collate([_make_sample("left")])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)

    base_losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        occupancy_loss_weight=0.0,
    )
    metric_losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        occupancy_loss_weight=1.0,
    )

    assert metric_losses["loss"].item() == pytest.approx(base_losses["loss"].item())
    assert metric_losses["occupancy_metric"].item() >= 0.0


def test_pointer_temperature_sharpens_softmax_distribution() -> None:
    raw_logits = torch.tensor([[[-0.4, 0.0, 0.8]]], dtype=torch.float32)
    probs_t1 = torch.softmax(raw_logits / 1.0, dim=-1)
    probs_t025 = torch.softmax(raw_logits / 0.25, dim=-1)
    entropy_t1 = -(probs_t1 * probs_t1.clamp(min=1e-8).log()).sum(dim=-1)
    entropy_t025 = -(probs_t025 * probs_t025.clamp(min=1e-8).log()).sum(dim=-1)

    assert probs_t025.max().item() > probs_t1.max().item()
    assert entropy_t025.item() < entropy_t1.item()


def test_coarse_excess_nll_matches_loss_minus_target_entropy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    model = AutoregMeshModel(config)

    batch = autoreg_mesh_collate([_make_sample("left")])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)
    losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
    )

    assert losses["coarse_excess_nll"].item() == pytest.approx(
        losses["coarse_loss"].item() - losses["coarse_target_entropy"].item()
    )


def test_occupancy_metric_uses_refined_predictions() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    wrong_pred = torch.zeros_like(batch["target_xyz"])
    right_pred = batch["target_xyz"].clone()
    common_outputs = {
        "coarse_logits": torch.zeros((1, batch["target_xyz"].shape[1], 8), dtype=torch.float32),
        "offset_logits": torch.zeros((1, batch["target_xyz"].shape[1], 3, 4), dtype=torch.float32),
        "stop_logits": torch.zeros((1, batch["target_xyz"].shape[1]), dtype=torch.float32),
        "pred_refine_residual": torch.zeros_like(batch["target_xyz"]),
        "pred_xyz": wrong_pred,
        "pred_xyz_soft": wrong_pred,
        "coarse_grid_shape": (2, 2, 2),
    }

    refined_losses = compute_autoreg_mesh_losses(
        {**common_outputs, "pred_xyz_refined": right_pred},
        batch,
        offset_num_bins=(4, 4, 4),
        occupancy_loss_weight=1.0,
    )
    wrong_losses = compute_autoreg_mesh_losses(
        {**common_outputs, "pred_xyz_refined": wrong_pred},
        batch,
        offset_num_bins=(4, 4, 4),
        occupancy_loss_weight=1.0,
    )

    assert refined_losses["occupancy_metric"].item() < wrong_losses["occupancy_metric"].item()


def test_masked_mean_ignores_nan_values_outside_mask() -> None:
    values = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float32)
    mask = torch.tensor([True, False, True], dtype=torch.bool)

    result = _masked_mean(values, mask)

    assert torch.isfinite(result)
    assert result.item() == pytest.approx(2.0)


def test_occupancy_metric_uses_full_grid_length_not_supervision_count() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    batch["target_supervision_mask"][0, 0] = False
    outputs = {
        "coarse_logits": torch.zeros((1, batch["target_xyz"].shape[1], 8), dtype=torch.float32),
        "offset_logits": torch.zeros((1, batch["target_xyz"].shape[1], 3, 4), dtype=torch.float32),
        "stop_logits": torch.zeros((1, batch["target_xyz"].shape[1]), dtype=torch.float32),
        "pred_refine_residual": torch.zeros_like(batch["target_xyz"]),
        "pred_xyz": batch["target_xyz"].clone(),
        "pred_xyz_soft": batch["target_xyz"].clone(),
        "pred_xyz_refined": batch["target_xyz"].clone(),
        "pred_coarse_ids": batch["target_coarse_ids"].clone(),
        "pred_coarse_axis_ids": {
            "z": torch.zeros_like(batch["target_coarse_ids"]),
            "y": torch.zeros_like(batch["target_coarse_ids"]),
            "x": torch.zeros_like(batch["target_coarse_ids"]),
        },
        "coarse_grid_shape": (2, 2, 2),
    }

    losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        occupancy_loss_weight=1.0,
    )

    assert torch.isfinite(losses["occupancy_metric"])


def test_position_refine_loss_is_gated_by_step_weight(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    model = AutoregMeshModel(config)

    batch = autoreg_mesh_collate([_make_sample("left")])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)

    off_losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        position_refine_weight_active=0.0,
        position_refine_loss_type="huber",
    )
    on_losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        position_refine_weight_active=0.05,
        position_refine_loss_type="huber",
    )

    assert off_losses["refine_loss_weight_active"].item() == pytest.approx(0.0)
    assert on_losses["refine_loss_weight_active"].item() == pytest.approx(0.05)
    assert on_losses["refine_loss"].item() >= 0.0
    assert on_losses["loss"].item() >= off_losses["loss"].item()


def test_refine_target_matches_bin_center_residual(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    model = AutoregMeshModel(config)

    sample = _make_sample("left")
    batch = autoreg_mesh_collate([sample])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)
    expected_residual = batch["target_xyz"] - batch["target_bin_center_xyz"]

    losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        position_refine_weight_active=0.05,
        position_refine_loss_type="huber",
    )

    assert torch.isfinite(expected_residual).all()
    assert losses["refine_loss"].item() >= 0.0


def test_soft_decode_matches_hard_decode_for_one_hot_logits() -> None:
    model = AutoregMeshModel(_make_cached_token_config()).eval()
    coarse_logits = torch.full((1, 1, 8), -40.0, dtype=torch.float32)
    coarse_logits[0, 0, 5] = 40.0
    offset_logits = torch.full((1, 1, 3, 4), -40.0, dtype=torch.float32)
    offset_logits[0, 0, 0, 1] = 40.0
    offset_logits[0, 0, 1, 2] = 40.0
    offset_logits[0, 0, 2, 3] = 40.0
    pred_refine_residual = torch.zeros((1, 1, 3), dtype=torch.float32)

    soft_xyz = model._soft_decode_local_xyz(coarse_logits, None, offset_logits, pred_refine_residual)
    hard_xyz = model.decode_local_xyz(
        torch.tensor([[5]], dtype=torch.long),
        torch.tensor([[[1, 2, 3]]], dtype=torch.long),
    )

    assert torch.allclose(soft_xyz, hard_xyz, atol=1e-4, rtol=1e-4)


def test_factorized_soft_decode_matches_hard_decode_for_one_hot_axis_logits() -> None:
    model = AutoregMeshModel(_make_factorized_cached_token_config()).eval()
    coarse_axis_logits = {
        "z": torch.full((1, 1, 2), -40.0, dtype=torch.float32),
        "y": torch.full((1, 1, 2), -40.0, dtype=torch.float32),
        "x": torch.full((1, 1, 2), -40.0, dtype=torch.float32),
    }
    coarse_axis_logits["z"][0, 0, 1] = 40.0
    coarse_axis_logits["y"][0, 0, 0] = 40.0
    coarse_axis_logits["x"][0, 0, 1] = 40.0
    offset_logits = torch.full((1, 1, 3, 4), -40.0, dtype=torch.float32)
    offset_logits[0, 0, 0, 1] = 40.0
    offset_logits[0, 0, 1, 2] = 40.0
    offset_logits[0, 0, 2, 3] = 40.0
    pred_refine_residual = torch.zeros((1, 1, 3), dtype=torch.float32)

    soft_xyz = model._soft_decode_local_xyz(None, coarse_axis_logits, offset_logits, pred_refine_residual)
    coarse_id = model._flatten_coarse_axis_ids(
        torch.tensor([[1]], dtype=torch.long),
        torch.tensor([[0]], dtype=torch.long),
        torch.tensor([[1]], dtype=torch.long),
    )
    hard_xyz = model.decode_local_xyz(
        coarse_id,
        torch.tensor([[[1, 2, 3]]], dtype=torch.long),
    )

    assert torch.allclose(soft_xyz, hard_xyz, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_coarse_continuation_mask_keeps_gt_and_masks_clear_wrong_side(direction: str) -> None:
    config = _make_large_cached_token_config()
    model = AutoregMeshModel(config).eval()
    batch = autoreg_mesh_collate([_make_large_sample(direction)])
    generation_inputs = model._build_teacher_forced_generation_inputs(batch)
    raw_mask = model._build_raw_coarse_continuation_mask(batch, generation_inputs)
    assert raw_mask is not None
    adjusted_mask, metrics = model._finalize_coarse_constraint_mask(
        raw_mask,
        sequence_mask=generation_inputs["sequence_mask"],
        target_coarse_ids=batch["target_coarse_ids"],
        target_supervision_mask=batch["target_supervision_mask"],
    )
    assert adjusted_mask is not None

    gt_id = int(batch["target_coarse_ids"][0, 0].item())
    assert bool(adjusted_mask[0, 0, gt_id].item())
    z, y, x = model._unflatten_coarse_ids(batch["target_coarse_ids"][:, :1])
    wrong_z = int(z[0, 0].item())
    if direction == "left":
        wrong_y = int(y[0, 0].item())
        wrong_x = 0
    elif direction == "right":
        wrong_y = int(y[0, 0].item())
        wrong_x = int(model.coarse_grid_shape[2] - 1)
    elif direction == "up":
        wrong_y = 0
        wrong_x = int(x[0, 0].item())
    else:
        wrong_y = int(model.coarse_grid_shape[1] - 1)
        wrong_x = int(x[0, 0].item())
    wrong_id = int(model._flatten_coarse_axis_ids(
        torch.tensor([[wrong_z]], dtype=torch.long),
        torch.tensor([[wrong_y]], dtype=torch.long),
        torch.tensor([[wrong_x]], dtype=torch.long),
    )[0, 0].item())
    assert bool(raw_mask[0, 0, wrong_id].item()) is False
    assert metrics["coarse_constraint_keep_fraction"].item() < 1.0


def test_joint_pointer_constraint_changes_argmax() -> None:
    model = AutoregMeshModel(_make_cached_token_config()).eval()
    with torch.no_grad():
        model.pointer_query.weight.zero_()
        model.pointer_query.bias.zero_()
        model.pointer_key.weight.zero_()
        eye = torch.eye(model.decoder_dim)
        model.pointer_query.weight.copy_(eye)
        model.pointer_key.weight.copy_(eye)

    hidden = torch.zeros((1, 1, model.decoder_dim), dtype=torch.float32)
    hidden[0, 0, 0] = 1.0
    hidden[0, 0, 1] = 0.5
    memory_tokens = torch.zeros((1, 8, model.decoder_dim), dtype=torch.float32)
    memory_tokens[0, 0, 0] = 1.0
    memory_tokens[0, 1, 1] = 1.0
    metrics = {
        "coarse_constraint_keep_fraction": hidden.new_tensor(0.25),
        "coarse_constraint_empty_rate": hidden.new_zeros(()),
        "coarse_constraint_target_outside_rate": hidden.new_zeros(()),
    }
    raw_outputs = model._compute_coarse_outputs(hidden, memory_tokens)
    masked_outputs = model._compute_coarse_outputs(
        hidden,
        memory_tokens,
        coarse_valid_mask=torch.tensor([[[False, True, False, False, False, False, False, False]]]),
        coarse_constraint_metrics=metrics,
    )

    assert int(raw_outputs["pred_coarse_ids"][0, 0].item()) == 0
    assert int(masked_outputs["pred_coarse_ids"][0, 0].item()) == 1
    assert masked_outputs["coarse_constraint_keep_fraction"].item() == pytest.approx(0.25)


def test_axis_factorized_constraint_changes_argmax() -> None:
    model = AutoregMeshModel(_make_factorized_cached_token_config()).eval()
    with torch.no_grad():
        for layer in (model.pointer_query_z, model.pointer_query_y, model.pointer_query_x):
            layer.weight.zero_()
            layer.bias.zero_()
            layer.weight.copy_(torch.eye(model.decoder_dim))
        for layer in (model.pointer_key_z, model.pointer_key_y, model.pointer_key_x):
            layer.weight.zero_()
            layer.weight.copy_(torch.eye(model.decoder_dim))

    hidden = torch.zeros((1, 1, model.decoder_dim), dtype=torch.float32)
    hidden[0, 0, 0] = 2.0
    hidden[0, 0, 2] = 2.0
    hidden[0, 0, 4] = 2.0
    hidden[0, 0, 1] = 0.5
    hidden[0, 0, 3] = 0.5
    hidden[0, 0, 5] = 0.5
    memory_tokens = torch.zeros((1, 8, model.decoder_dim), dtype=torch.float32)
    cursor = 0
    for z_idx in range(2):
        for y_idx in range(2):
            for x_idx in range(2):
                memory_tokens[0, cursor, z_idx] = 1.0
                memory_tokens[0, cursor, 2 + y_idx] = 1.0
                memory_tokens[0, cursor, 4 + x_idx] = 1.0
                cursor += 1
    metrics = {
        "coarse_constraint_keep_fraction": hidden.new_tensor(0.125),
        "coarse_constraint_empty_rate": hidden.new_zeros(()),
        "coarse_constraint_target_outside_rate": hidden.new_zeros(()),
    }
    raw_outputs = model._compute_coarse_outputs(hidden, memory_tokens)
    only_last = torch.zeros((1, 1, 8), dtype=torch.bool)
    only_last[0, 0, 7] = True
    masked_outputs = model._compute_coarse_outputs(
        hidden,
        memory_tokens,
        coarse_valid_mask=only_last,
        coarse_constraint_metrics=metrics,
    )

    assert int(raw_outputs["pred_coarse_axis_ids"]["z"][0, 0].item()) == 0
    assert int(raw_outputs["pred_coarse_axis_ids"]["y"][0, 0].item()) == 0
    assert int(raw_outputs["pred_coarse_axis_ids"]["x"][0, 0].item()) == 0
    assert int(masked_outputs["pred_coarse_axis_ids"]["z"][0, 0].item()) == 1
    assert int(masked_outputs["pred_coarse_axis_ids"]["y"][0, 0].item()) == 1
    assert int(masked_outputs["pred_coarse_axis_ids"]["x"][0, 0].item()) == 1
    assert int(masked_outputs["pred_coarse_ids"][0, 0].item()) == 7


def test_coarse_constraint_gt_union_and_empty_fallback() -> None:
    model = AutoregMeshModel(_make_cached_token_config()).eval()
    raw_mask = torch.zeros((1, 1, 8), dtype=torch.bool)
    adjusted, metrics = model._finalize_coarse_constraint_mask(
        raw_mask,
        sequence_mask=torch.tensor([[True]], dtype=torch.bool),
        target_coarse_ids=torch.tensor([[3]], dtype=torch.long),
        target_supervision_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    assert adjusted is not None
    assert bool(torch.all(adjusted).item())
    assert metrics["coarse_constraint_empty_rate"].item() == pytest.approx(1.0)
    assert metrics["coarse_constraint_target_outside_rate"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_sequence_to_grid_torch_matches_numpy_deserialize(direction: str) -> None:
    sample = _make_sample(direction)
    grid_shape = tuple(int(v) for v in sample["target_grid_shape"].tolist())
    torch_grid = _sequence_to_grid_torch(
        sample["target_xyz"],
        grid_shape=grid_shape,
        direction=direction,
    )
    numpy_grid = deserialize_continuation_grid(
        sample["target_xyz"].numpy(),
        direction=direction,
        grid_shape=grid_shape,
    )

    np.testing.assert_allclose(torch_grid.numpy(), numpy_grid)


def test_geometry_metric_loss_is_near_zero_on_isometric_copy_and_higher_on_stretch() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    translated = batch["target_xyz"] + torch.tensor([3.0, -2.0, 1.0], dtype=torch.float32).view(1, 1, 3)
    stretched = batch["target_xyz"].clone()
    stretched[..., 1] = stretched[..., 1] * 1.5

    zero_like = torch.zeros_like(batch["target_xyz"])
    exact_loss = _geometry_metric_loss(
        {"pred_xyz_soft": batch["target_xyz"].clone(), "pred_refine_residual": zero_like},
        batch,
        loss_type="huber",
        include_refine_residual=False,
    )
    translated_loss = _geometry_metric_loss(
        {"pred_xyz_soft": translated, "pred_refine_residual": zero_like},
        batch,
        loss_type="huber",
        include_refine_residual=False,
    )
    stretched_loss = _geometry_metric_loss(
        {"pred_xyz_soft": stretched, "pred_refine_residual": zero_like},
        batch,
        loss_type="huber",
        include_refine_residual=False,
    )

    assert exact_loss.item() == pytest.approx(0.0, abs=1e-8)
    assert translated_loss.item() == pytest.approx(0.0, abs=1e-8)
    assert stretched_loss.item() > translated_loss.item()


def test_geometry_sd_loss_is_near_zero_on_isometric_copy_and_higher_on_stretch() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    translated = batch["target_xyz"] + torch.tensor([3.0, -2.0, 1.0], dtype=torch.float32).view(1, 1, 3)
    stretched = batch["target_xyz"].clone()
    stretched[..., 1] = stretched[..., 1] * 1.5

    zero_like = torch.zeros_like(batch["target_xyz"])
    exact_loss = _geometry_sd_loss(
        {"pred_xyz_soft": batch["target_xyz"].clone(), "pred_refine_residual": zero_like},
        batch,
        include_refine_residual=False,
    )
    translated_loss = _geometry_sd_loss(
        {"pred_xyz_soft": translated, "pred_refine_residual": zero_like},
        batch,
        include_refine_residual=False,
    )
    stretched_loss = _geometry_sd_loss(
        {"pred_xyz_soft": stretched, "pred_refine_residual": zero_like},
        batch,
        include_refine_residual=False,
    )

    assert exact_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert translated_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert stretched_loss.item() > translated_loss.item()


def test_geometry_sd_loss_stays_finite_for_degenerate_prediction() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    zero_like = torch.zeros_like(batch["target_xyz"])
    collapsed = torch.zeros_like(batch["target_xyz"])
    loss = _geometry_sd_loss(
        {"pred_xyz_soft": collapsed, "pred_refine_residual": zero_like},
        batch,
        include_refine_residual=False,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_xyz_soft_loss_is_near_zero_on_ground_truth_and_excludes_residual_before_refine() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    outputs = {
        "coarse_logits": torch.zeros((1, batch["target_xyz"].shape[1], 8), dtype=torch.float32),
        "offset_logits": torch.zeros((1, batch["target_xyz"].shape[1], 3, 4), dtype=torch.float32),
        "stop_logits": torch.zeros((1, batch["target_xyz"].shape[1]), dtype=torch.float32),
        "pred_refine_residual": torch.full_like(batch["target_xyz"], 3.0),
        "pred_xyz": batch["target_xyz"].clone(),
        "pred_xyz_soft": batch["target_xyz"].clone() + 3.0,
        "pred_xyz_refined": batch["target_xyz"].clone() + 3.0,
        "pred_coarse_ids": batch["target_coarse_ids"].clone(),
        "pred_coarse_axis_ids": {
            "z": torch.zeros_like(batch["target_coarse_ids"]),
            "y": torch.zeros_like(batch["target_coarse_ids"]),
            "x": torch.zeros_like(batch["target_coarse_ids"]),
        },
        "coarse_grid_shape": (2, 2, 2),
    }

    off_losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        xyz_soft_loss_weight_active=1.0,
        position_refine_weight_active=0.0,
    )
    on_losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        xyz_soft_loss_weight_active=1.0,
        position_refine_weight_active=0.05,
    )

    assert off_losses["xyz_soft_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert on_losses["xyz_soft_loss"].item() > off_losses["xyz_soft_loss"].item()


def test_seam_edge_loss_is_near_zero_on_ground_truth_and_increases_when_displaced() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    gt_loss = _seam_edge_loss_from_sequence(batch["target_xyz"], batch, band_width=1, loss_type="edge_huber")
    displaced = batch["target_xyz"].clone()
    displaced[0, :3, :] = displaced[0, :3, :] + torch.tensor([0.0, 0.0, 4.0])
    displaced_loss = _seam_edge_loss_from_sequence(displaced, batch, band_width=1, loss_type="edge_huber")

    assert gt_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert displaced_loss.item() > gt_loss.item()


def test_first_strip_wrong_side_rate_refined_detects_mirrored_first_strip() -> None:
    from vesuvius.neural_tracing.autoreg_mesh.losses import _first_strip_wrong_side_rate_from_sequence

    batch = autoreg_mesh_collate([_make_sample("left")])
    gt_rate = _first_strip_wrong_side_rate_from_sequence(batch["target_xyz"], batch)
    mirrored = batch["target_xyz"].clone()
    strip_length = int(batch["strip_length"][0].item())
    cond_grid = batch["conditioning_grid_local"][0]
    frontier = cond_grid[:, -1, :]
    first_strip = mirrored[0, :strip_length]
    mirrored[0, :strip_length] = frontier - 0.5 * (first_strip - frontier)
    mirrored_rate = _first_strip_wrong_side_rate_from_sequence(mirrored, batch)

    assert gt_rate.item() == pytest.approx(0.0, abs=1e-6)
    assert mirrored_rate.item() > gt_rate.item()


def test_triangle_barrier_increases_for_flipped_continuation() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    gt_barrier = _triangle_barrier_loss_from_sequence(batch["target_xyz"], batch, margin=0.05)
    flipped = batch["target_xyz"].clone()
    target_grid = _sequence_to_grid_torch(flipped[0], grid_shape=tuple(int(v) for v in batch["target_grid_shape"][0].tolist()), direction="left")
    cond_grid = batch["conditioning_grid_local"][0]
    target_grid[:, 0, :] = cond_grid[:, -1, :] - 0.2 * (target_grid[:, 0, :] - cond_grid[:, -1, :])
    flipped[0] = torch.from_numpy(serialize_split_conditioning_example(
        cond_zyxs_local=cond_grid.numpy(),
        masked_zyxs_local=target_grid.numpy(),
        direction="left",
        volume_shape=(16, 16, 16),
        patch_size=(8, 8, 8),
        offset_num_bins=(4, 4, 4),
        frontier_band_width=4,
    )["target_xyz"]).to(torch.float32)
    flipped_barrier = _triangle_barrier_loss_from_sequence(flipped, batch, margin=0.05)

    assert gt_barrier.item() >= 0.0
    assert flipped_barrier.item() > gt_barrier.item()


def test_invalid_target_positions_are_masked_from_loss(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    model = AutoregMeshModel(config)

    sample = _make_sample("left")
    sample["target_xyz"][0] = torch.tensor([-4.0, 4.0, 8.0], dtype=torch.float32)
    sample["target_coarse_ids"][0] = -100
    sample["target_offset_bins"][0] = torch.tensor([-100, -100, -100], dtype=torch.long)
    sample["target_valid_mask"][0] = False
    sample["target_stop"][0] = 0.0

    batch = autoreg_mesh_collate([sample])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)
    losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
    )

    assert not bool(batch["target_supervision_mask"][0, 0].item())
    assert torch.isfinite(losses["loss"])


def test_target_boundary_stats_detect_invalid_frontier_and_reject() -> None:
    target_grid = _make_sample("left")["target_grid_local"].numpy().copy()
    target_grid[:, 0, 0] = -1.0

    stats = _compute_target_boundary_stats(
        target_grid,
        volume_shape=(16, 16, 16),
        direction="left",
    )

    assert stats["frontier_invalid_fraction"] > 0.0
    assert stats["touches_crop_boundary"] is True
    assert _should_reject_boundary_stats(stats) is True


def test_target_boundary_stats_reject_high_invalid_fraction() -> None:
    target_grid = _make_sample("up")["target_grid_local"].numpy().copy()
    target_grid[:] = np.array([32.0, 32.0, 32.0], dtype=np.float32)

    stats = _compute_target_boundary_stats(
        target_grid,
        volume_shape=(16, 16, 16),
        direction="up",
    )

    assert stats["target_invalid_fraction"] == pytest.approx(1.0)
    assert stats["frontier_invalid_fraction"] == pytest.approx(1.0)
    assert _should_reject_boundary_stats(stats) is True


def test_target_boundary_stats_allow_truncated_continuation_away_from_frontier() -> None:
    target_grid = _make_sample("up")["target_grid_local"].numpy().copy()
    target_grid[-1, :, :] = np.array([32.0, 32.0, 32.0], dtype=np.float32)

    stats = _compute_target_boundary_stats(
        target_grid,
        volume_shape=(16, 16, 16),
        direction="up",
    )

    assert stats["frontier_invalid_fraction"] == pytest.approx(0.0)
    assert stats["target_invalid_fraction"] > 0.0
    assert _should_reject_boundary_stats(stats) is False


def test_boundary_loss_and_oob_metrics_increase_outside_crop() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    inside = {
        "coarse_logits": torch.zeros((1, batch["target_xyz"].shape[1], 8), dtype=torch.float32),
        "offset_logits": torch.zeros((1, batch["target_xyz"].shape[1], 3, 4), dtype=torch.float32),
        "stop_logits": torch.zeros((1, batch["target_xyz"].shape[1]), dtype=torch.float32),
        "pred_refine_residual": torch.zeros_like(batch["target_xyz"]),
        "pred_xyz": batch["target_xyz"].clone(),
        "pred_xyz_soft": batch["target_xyz"].clone(),
        "pred_xyz_refined": batch["target_xyz"].clone(),
        "pred_coarse_ids": batch["target_coarse_ids"].clone(),
        "pred_coarse_axis_ids": {
            "z": torch.zeros_like(batch["target_coarse_ids"]),
            "y": torch.zeros_like(batch["target_coarse_ids"]),
            "x": torch.zeros_like(batch["target_coarse_ids"]),
        },
        "coarse_grid_shape": (2, 2, 2),
    }
    outside = dict(inside)
    outside["pred_xyz_refined"] = batch["target_xyz"].clone() + 64.0

    inside_losses = compute_autoreg_mesh_losses(
        inside,
        batch,
        offset_num_bins=(4, 4, 4),
        boundary_loss_weight_active=1.0,
    )
    outside_losses = compute_autoreg_mesh_losses(
        outside,
        batch,
        offset_num_bins=(4, 4, 4),
        boundary_loss_weight_active=1.0,
    )

    assert inside_losses["boundary_loss"].item() >= 0.0
    assert outside_losses["boundary_loss"].item() > inside_losses["boundary_loss"].item()
    assert outside_losses["pred_oob_fraction_refined"].item() > inside_losses["pred_oob_fraction_refined"].item()
    assert outside_losses["loss"].item() == pytest.approx(inside_losses["loss"].item())
    assert outside_losses["boundary_loss_weight_active"].item() == pytest.approx(0.0)
    assert torch.isfinite(outside_losses["loss"])


def test_boundary_vertices_are_not_counted_as_oob() -> None:
    batch = autoreg_mesh_collate([_make_sample("left")])
    max_coord = torch.tensor([15.9999, 15.9999, 15.9999], dtype=torch.float32).view(1, 1, 3)
    boundary_pred = max_coord.expand_as(batch["target_xyz"]).clone()
    outputs = {
        "coarse_logits": torch.zeros((1, batch["target_xyz"].shape[1], 8), dtype=torch.float32),
        "offset_logits": torch.zeros((1, batch["target_xyz"].shape[1], 3, 4), dtype=torch.float32),
        "stop_logits": torch.zeros((1, batch["target_xyz"].shape[1]), dtype=torch.float32),
        "pred_refine_residual": torch.zeros_like(batch["target_xyz"]),
        "pred_xyz": boundary_pred,
        "pred_xyz_soft": boundary_pred,
        "pred_xyz_refined": boundary_pred,
        "pred_coarse_ids": batch["target_coarse_ids"].clone(),
        "pred_coarse_axis_ids": {
            "z": torch.zeros_like(batch["target_coarse_ids"]),
            "y": torch.zeros_like(batch["target_coarse_ids"]),
            "x": torch.zeros_like(batch["target_coarse_ids"]),
        },
        "coarse_grid_shape": (2, 2, 2),
    }

    losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
    )

    assert losses["pred_oob_fraction_refined"].item() == pytest.approx(0.0)
    assert losses["teacher_forced_boundary_touch_fraction"].item() == pytest.approx(1.0)


def test_fast_prefilter_respects_frontier_band_width() -> None:
    surface_local = _make_surface(5, 5, z_offset=4.0, y_offset=2.0, x_offset=2.0)
    surface_local[:, 2, 0] = -1.0

    valid_band1, _ = _fast_prefilter_plan(
        surface_local,
        direction="left",
        conditioning_count=1,
        frontier_band_width=1,
        surface_downsample_factor=1,
        volume_shape=(16, 16, 16),
    )
    valid_band2, _ = _fast_prefilter_plan(
        surface_local,
        direction="left",
        conditioning_count=1,
        frontier_band_width=2,
        surface_downsample_factor=1,
        volume_shape=(16, 16, 16),
    )

    assert valid_band1 is True
    assert valid_band2 is False


def test_volume_only_augmentation_changes_volume_without_touching_geometry() -> None:
    sample = _make_sample("left")
    original_volume = sample["volume"].numpy()[0].copy()
    original_target = sample["target_grid_local"].clone()

    augmented = _apply_volume_only_augmentation(
        original_volume,
        {
            "enabled": True,
            "contrast_prob": 1.0,
            "mult_brightness_prob": 1.0,
            "add_brightness_prob": 1.0,
            "gamma_prob": 1.0,
            "gaussian_noise_prob": 1.0,
            "gaussian_blur_prob": 1.0,
            "slice_illumination_prob": 1.0,
            "lowres_prob": 1.0,
        },
        enabled=True,
    )

    assert augmented.shape == original_volume.shape
    assert not np.allclose(augmented, original_volume)
    assert torch.equal(sample["target_grid_local"], original_target)


def test_spatial_augmentation_mirror_transforms_volume_and_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    cond = np.array([[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]], dtype=np.float32)
    masked = np.array([[[2.0, 1.0, 0.0], [3.0, 2.0, 1.0]]], dtype=np.float32)

    def _zero_rand(*size, **kwargs):
        return torch.zeros(*size, **kwargs)

    monkeypatch.setattr(torch, "rand", _zero_rand)
    aug_volume, aug_cond, aug_masked, meta = _apply_spatial_augmentation(
        volume,
        cond_local=cond,
        masked_local=masked,
        crop_size=(4, 4, 4),
        augmentation_cfg={
            "enabled": True,
            "mirror_prob": 1.0,
            "transpose_prob": 0.0,
            "mirror_axes": [0, 2],
            "transpose_axes": [0, 1, 2],
        },
        enabled=True,
    )

    np.testing.assert_allclose(aug_volume, np.flip(volume, axis=(0, 2)))
    expected_cond = cond.copy()
    expected_masked = masked.copy()
    for arr in (expected_cond, expected_masked):
        arr[..., 0] = 3.0 - arr[..., 0]
        arr[..., 2] = 3.0 - arr[..., 2]
    np.testing.assert_allclose(aug_cond, expected_cond)
    np.testing.assert_allclose(aug_masked, expected_masked)
    assert meta["spatial_augmented"] is True
    assert meta["spatial_mirror_axes"] == [0, 2]
    assert meta["spatial_axis_order"] == [0, 1, 2]


def test_spatial_augmentation_transpose_transforms_volume_and_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    cond = np.array([[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]], dtype=np.float32)
    masked = np.array([[[2.0, 1.0, 0.0], [3.0, 2.0, 1.0]]], dtype=np.float32)

    def _zero_rand(*size, **kwargs):
        return torch.zeros(*size, **kwargs)

    monkeypatch.setattr(torch, "rand", _zero_rand)
    monkeypatch.setattr(np.random, "shuffle", lambda axes: axes.reverse())
    aug_volume, aug_cond, aug_masked, meta = _apply_spatial_augmentation(
        volume,
        cond_local=cond,
        masked_local=masked,
        crop_size=(4, 4, 4),
        augmentation_cfg={
            "enabled": True,
            "mirror_prob": 0.0,
            "transpose_prob": 1.0,
            "mirror_axes": [0, 1, 2],
            "transpose_axes": [0, 1, 2],
        },
        enabled=True,
    )

    np.testing.assert_allclose(aug_volume, np.transpose(volume, (2, 1, 0)))
    np.testing.assert_allclose(aug_cond, cond[..., [2, 1, 0]])
    np.testing.assert_allclose(aug_masked, masked[..., [2, 1, 0]])
    assert meta["spatial_augmented"] is True
    assert meta["spatial_axis_order"] == [2, 1, 0]


def test_spatial_augmentation_keeps_direction_and_strip_positions_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    cond = np.array(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 1.0]],
            [[1.0, 1.0, 0.0], [1.5, 1.0, 1.0]],
            [[2.0, 2.0, 0.0], [2.5, 2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    masked = np.array(
        [
            [[0.0, 0.0, 2.0], [0.5, 0.0, 3.0]],
            [[1.0, 1.0, 2.0], [1.5, 1.0, 3.0]],
            [[2.0, 2.0, 2.0], [2.5, 2.0, 3.0]],
        ],
        dtype=np.float32,
    )

    def _zero_rand(*size, **kwargs):
        return torch.zeros(*size, **kwargs)

    monkeypatch.setattr(torch, "rand", _zero_rand)
    monkeypatch.setattr(np.random, "shuffle", lambda axes: axes.reverse())
    _, cond_aug, masked_aug, _ = _apply_spatial_augmentation(
        np.zeros((4, 4, 4), dtype=np.float32),
        cond_local=cond,
        masked_local=masked,
        crop_size=(4, 4, 4),
        augmentation_cfg={
            "enabled": True,
            "mirror_prob": 1.0,
            "transpose_prob": 1.0,
            "mirror_axes": [0, 1, 2],
            "transpose_axes": [0, 1, 2],
        },
        enabled=True,
    )
    original = serialize_split_conditioning_example(
        cond_zyxs_local=cond,
        masked_zyxs_local=masked,
        direction="left",
        volume_shape=(16, 16, 16),
        patch_size=(8, 8, 8),
        offset_num_bins=(4, 4, 4),
        frontier_band_width=4,
    )
    augmented = serialize_split_conditioning_example(
        cond_zyxs_local=cond_aug,
        masked_zyxs_local=masked_aug,
        direction="left",
        volume_shape=(16, 16, 16),
        patch_size=(8, 8, 8),
        offset_num_bins=(4, 4, 4),
        frontier_band_width=4,
    )

    assert augmented["direction"] == original["direction"]
    np.testing.assert_allclose(augmented["conditioning_grid_local"], cond_aug)
    np.testing.assert_allclose(augmented["target_grid_local"], masked_aug)
    np.testing.assert_array_equal(augmented["target_strip_positions"], original["target_strip_positions"])
    np.testing.assert_array_equal(augmented["prompt_tokens"]["strip_positions"], original["prompt_tokens"]["strip_positions"])


def test_scheduled_sampling_probability_progression() -> None:
    from vesuvius.neural_tracing.autoreg_mesh.train import _scheduled_sampling_prob

    config = _make_cached_token_config()
    config["scheduled_sampling_enabled"] = True
    config["scheduled_sampling_max_prob"] = 0.10
    config["scheduled_sampling_start_step"] = 10
    config["scheduled_sampling_ramp_steps"] = 20

    assert _scheduled_sampling_prob(config, global_step=0) == pytest.approx(0.0)
    assert _scheduled_sampling_prob(config, global_step=10) == pytest.approx(0.0)
    assert _scheduled_sampling_prob(config, global_step=20) == pytest.approx(0.05)
    assert _scheduled_sampling_prob(config, global_step=40) == pytest.approx(0.10)


def test_scheduled_sampling_feedback_state_tracks_offset_and_refine_steps() -> None:
    config = _make_cached_token_config()
    config["offset_loss_start_step"] = 2
    config["position_refine_weight"] = 0.05
    config["position_refine_start_step"] = 5

    assert _scheduled_sampling_feedback_state(config, global_step=0) == (False, False)
    assert _scheduled_sampling_feedback_state(config, global_step=2) == (True, False)
    assert _scheduled_sampling_feedback_state(config, global_step=5) == (True, True)


def test_distributed_runtime_defaults_to_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)

    runtime = _initialize_distributed_runtime(device="cpu")

    assert runtime.is_distributed is False
    assert runtime.rank == 0
    assert runtime.world_size == 1
    assert runtime.is_main_process is True
    assert str(runtime.device) == "cpu"


def test_metric_dict_mean_across_ranks_is_noop_single_process() -> None:
    runtime = _initialize_distributed_runtime(device="cpu")
    metrics = {"loss": 1.25, "grad_norm": 3.5}

    reduced = _metric_dict_mean_across_ranks(metrics, device=torch.device("cpu"), runtime=runtime)

    assert reduced == metrics


def test_next_batch_calls_reset_callback_when_iterator_recycles() -> None:
    dataloader = torch.utils.data.DataLoader([1], batch_size=1, shuffle=False)
    iterator = iter(dataloader)
    calls = {"count": 0}

    _ = next(iterator)
    batch, iterator = _next_batch(
        iterator,
        dataloader,
        on_reset=lambda: calls.__setitem__("count", calls["count"] + 1),
    )

    assert calls["count"] == 1
    assert list(batch.numpy()) == [1]
    assert iterator is not None


def test_scheduled_sampling_enabled_smoke_train_runs(tmp_path: Path) -> None:
    config = _make_cached_token_config()
    config["out_dir"] = str(tmp_path / "runs_sched")
    config["scheduled_sampling_enabled"] = True
    config["scheduled_sampling_max_prob"] = 0.10
    config["scheduled_sampling_start_step"] = 0
    config["scheduled_sampling_ramp_steps"] = 2
    dataset = _make_training_dataset()

    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=2)

    assert len(result["history"]) == 2
    assert "scheduled_sampling_prob" in result["history"][-1]
    assert result["history"][-1]["scheduled_sampling_prob"] >= 0.0


def test_position_refine_weight_activates_after_start_step(tmp_path: Path) -> None:
    config = _make_cached_token_config()
    config["out_dir"] = str(tmp_path / "runs_refine")
    config["position_refine_enabled"] = True
    config["position_refine_weight"] = 0.05
    config["position_refine_start_step"] = 1
    dataset = _make_training_dataset()

    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=2)

    assert result["history"][0]["position_refine_weight_active"] == pytest.approx(0.0)
    assert result["history"][1]["position_refine_weight_active"] == pytest.approx(0.05)
    assert "refine_loss" in result["history"][1]


def test_geometry_metric_weight_activates_after_start_step(tmp_path: Path) -> None:
    config = _make_cached_token_config()
    config["out_dir"] = str(tmp_path / "runs_geometry")
    config["geometry_metric_enabled"] = True
    config["geometry_metric_weight"] = 0.01
    config["geometry_metric_start_step"] = 1
    dataset = _make_training_dataset()

    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=2)

    assert _geometry_metric_weight_active(config, global_step=0) == pytest.approx(0.0)
    assert _geometry_metric_weight_active(config, global_step=1) == pytest.approx(0.01)
    assert result["history"][0]["geometry_metric_weight_active"] == pytest.approx(0.0)
    assert result["history"][1]["geometry_metric_weight_active"] == pytest.approx(0.01)
    assert "geometry_metric_loss" in result["history"][1]


def test_geometry_sd_weight_activates_after_start_step(tmp_path: Path) -> None:
    config = _make_cached_token_config()
    config["out_dir"] = str(tmp_path / "runs_geometry_sd")
    config["geometry_sd_enabled"] = True
    config["geometry_sd_weight"] = 0.005
    config["geometry_sd_start_step"] = 1
    dataset = _make_training_dataset()

    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=2)

    assert _geometry_sd_weight_active(config, global_step=0) == pytest.approx(0.0)
    assert _geometry_sd_weight_active(config, global_step=1) == pytest.approx(0.005)
    assert result["history"][0]["geometry_sd_weight_active"] == pytest.approx(0.0)
    assert result["history"][1]["geometry_sd_weight_active"] == pytest.approx(0.005)
    assert "geometry_sd_loss" in result["history"][1]


def test_offset_loss_is_gated_by_active_weight(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    model = AutoregMeshModel(config)

    batch = autoreg_mesh_collate([_make_sample("left")])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)

    off_losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        offset_loss_weight_active=0.0,
    )
    on_losses = compute_autoreg_mesh_losses(
        outputs,
        batch,
        offset_num_bins=(4, 4, 4),
        offset_loss_weight_active=1.0,
    )

    assert off_losses["offset_loss_weight_active"].item() == pytest.approx(0.0)
    assert on_losses["offset_loss_weight_active"].item() == pytest.approx(1.0)
    assert off_losses["loss"].item() < on_losses["loss"].item()


def test_offset_loss_weight_activates_after_start_step(tmp_path: Path) -> None:
    config = _make_cached_token_config()
    config["out_dir"] = str(tmp_path / "runs_offset")
    config["offset_loss_weight"] = 0.2
    config["offset_loss_start_step"] = 1
    dataset = _make_training_dataset()

    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=2)

    assert result["history"][0]["offset_loss_weight_active"] == pytest.approx(0.0)
    assert result["history"][1]["offset_loss_weight_active"] == pytest.approx(0.2)


def test_distance_aware_target_builder_normalizes_and_respects_edges() -> None:
    target_ids = torch.tensor([[0, 21]], dtype=torch.long)
    mask = torch.tensor([[True, True]], dtype=torch.bool)
    neighbor_ids, target_probs = _build_distance_aware_coarse_targets(
        target_ids,
        mask,
        coarse_grid_shape=(4, 4, 4),
        radius=2,
        sigma=2.0,
    )

    assert neighbor_ids.shape[-1] == 125
    assert torch.allclose(target_probs.sum(dim=-1), torch.ones_like(target_probs.sum(dim=-1)))
    assert torch.count_nonzero(target_probs[0, 0]) < torch.count_nonzero(target_probs[0, 1])
    assert 0 in neighbor_ids[0, 0].tolist()
    assert 21 in neighbor_ids[0, 1].tolist()


def test_distance_aware_target_default_config_values() -> None:
    config = _make_cached_token_config()
    validated = validate_autoreg_mesh_config(config)
    assert validated["pointer_temperature"] == pytest.approx(0.25)
    assert validated["coarse_prediction_mode"] == "joint_pointer"
    assert validated["conditioning_feature_debias_mode"] == "none"
    assert validated["conditioning_feature_debias_basis_source"] == "zero_volume_svd"
    assert validated["conditioning_feature_debias_components"] == 16
    assert validated["scheduled_sampling_pattern"] == "stripwise_full_strip_greedy"
    assert validated["xyz_soft_loss_enabled"] is True
    assert validated["xyz_soft_loss_weight"] == pytest.approx(1.0)
    assert validated["seam_loss_enabled"] is True
    assert validated["seam_loss_weight"] == pytest.approx(0.25)
    assert validated["triangle_barrier_enabled"] is True
    assert validated["triangle_barrier_weight"] == pytest.approx(0.1)
    assert validated["triangle_barrier_margin"] == pytest.approx(0.05)
    assert validated["boundary_loss_enabled"] is False
    assert validated["boundary_loss_weight"] == pytest.approx(0.0)
    assert validated["boundary_loss_start_step"] == 0
    assert validated["rollout_val_examples_per_log"] == 1
    assert validated["rollout_val_max_steps"] is None
    assert validated["val_split_mode"] == "spatial_groups"
    assert validated["validation_leakage_margin_voxels"] == pytest.approx(0.0)
    assert validated["wandb_log_xy_slice_images"] is True
    assert validated["wandb_xy_slice_mode"] == "best_xy_slice"
    assert validated["wandb_xy_slice_line_thickness"] == 1
    assert validated["wandb_xy_slice_depth_tolerance"] == pytest.approx(0.75)
    assert validated["geometry_metric_enabled"] is True
    assert validated["geometry_metric_weight"] == pytest.approx(0.01)
    assert validated["geometry_metric_start_step"] == 2000
    assert validated["geometry_sd_enabled"] is True
    assert validated["geometry_sd_weight"] == pytest.approx(0.005)
    assert validated["geometry_sd_start_step"] == 2000
    assert validated["distance_aware_coarse_targets_enabled"] is True
    assert validated["distance_aware_coarse_target_radius"] == 1
    assert validated["distance_aware_coarse_target_sigma"] == pytest.approx(1.0)
    assert validated["coarse_continuation_constraint_enabled"] is True
    assert validated["coarse_continuation_constraint_mode"] == "hard_mask"
    assert validated["coarse_continuation_forward_scale"] == pytest.approx(4.0)
    assert validated["coarse_continuation_backward_scale"] == pytest.approx(1.5)
    assert validated["coarse_continuation_lateral_scale"] == pytest.approx(3.0)
    assert validated["coarse_continuation_min_radius_scale"] == pytest.approx(2.5)
    assert validated["coarse_continuation_empty_fallback"] == "disable_for_token"
    assert validated["spatial_augmentation"]["enabled"] is True
    assert validated["spatial_augmentation"]["mirror_prob"] == pytest.approx(0.5)
    assert validated["spatial_augmentation"]["transpose_prob"] == pytest.approx(0.25)
    assert validated["spatial_augmentation"]["mirror_axes"] == [0, 1, 2]
    assert validated["spatial_augmentation"]["transpose_axes"] == [0, 1, 2]
    assert validated["volume_only_augmentation"]["enabled"] is True
    assert validated["volume_only_augmentation"]["gamma_prob"] == pytest.approx(0.4)


def test_anisotropic_surface_downsample_factor_validates() -> None:
    config = _make_cached_token_config()
    config["surface_downsample_factor"] = [2, 1]
    validated = validate_autoreg_mesh_config(config)

    assert validated["surface_downsample_factor"] == (2, 1)


def test_distance_aware_coarse_loss_prefers_nearby_cells() -> None:
    target_ids = torch.tensor([[21]], dtype=torch.long)
    supervision_mask = torch.tensor([[True]], dtype=torch.bool)
    near_logits = torch.full((1, 1, 64), -10.0)
    far_logits = torch.full((1, 1, 64), -10.0)
    near_logits[0, 0, 22] = 10.0  # one-cell neighbor in flattened grid
    far_logits[0, 0, 63] = 10.0

    common_batch = {
        "target_coarse_ids": target_ids,
        "target_supervision_mask": supervision_mask,
        "target_offset_bins": torch.zeros((1, 1, 3), dtype=torch.long),
        "target_stop": torch.zeros((1, 1), dtype=torch.float32),
        "target_xyz": torch.zeros((1, 1, 3), dtype=torch.float32),
        "target_bin_center_xyz": torch.zeros((1, 1, 3), dtype=torch.float32),
    }
    common_outputs = {
        "offset_logits": torch.zeros((1, 1, 3, 4), dtype=torch.float32),
        "stop_logits": torch.zeros((1, 1), dtype=torch.float32),
        "pred_refine_residual": torch.zeros((1, 1, 3), dtype=torch.float32),
        "pred_xyz": torch.zeros((1, 1, 3), dtype=torch.float32),
        "coarse_grid_shape": (4, 4, 4),
    }

    near_loss = compute_autoreg_mesh_losses(
        {**common_outputs, "coarse_logits": near_logits},
        common_batch,
        offset_num_bins=(4, 4, 4),
        distance_aware_coarse_targets_enabled=True,
        distance_aware_coarse_target_radius=2,
        distance_aware_coarse_target_sigma=2.0,
        distance_aware_coarse_target_loss="soft_ce",
    )["coarse_loss"]
    far_loss = compute_autoreg_mesh_losses(
        {**common_outputs, "coarse_logits": far_logits},
        common_batch,
        offset_num_bins=(4, 4, 4),
        distance_aware_coarse_targets_enabled=True,
        distance_aware_coarse_target_radius=2,
        distance_aware_coarse_target_sigma=2.0,
        distance_aware_coarse_target_loss="soft_ce",
    )["coarse_loss"]

    assert near_loss.item() < far_loss.item()


def test_factorized_distance_aware_coarse_loss_prefers_nearby_axis_cells() -> None:
    target_ids = torch.tensor([[21]], dtype=torch.long)
    supervision_mask = torch.tensor([[True]], dtype=torch.bool)
    near_axis_logits = {
        "z": torch.full((1, 1, 4), -10.0),
        "y": torch.full((1, 1, 4), -10.0),
        "x": torch.full((1, 1, 4), -10.0),
    }
    far_axis_logits = {
        "z": torch.full((1, 1, 4), -10.0),
        "y": torch.full((1, 1, 4), -10.0),
        "x": torch.full((1, 1, 4), -10.0),
    }
    near_axis_logits["z"][0, 0, 1] = 10.0
    near_axis_logits["y"][0, 0, 1] = 10.0
    near_axis_logits["x"][0, 0, 2] = 10.0
    far_axis_logits["z"][0, 0, 3] = 10.0
    far_axis_logits["y"][0, 0, 3] = 10.0
    far_axis_logits["x"][0, 0, 3] = 10.0

    common_batch = {
        "target_coarse_ids": target_ids,
        "target_supervision_mask": supervision_mask,
        "target_offset_bins": torch.zeros((1, 1, 3), dtype=torch.long),
        "target_stop": torch.zeros((1, 1), dtype=torch.float32),
        "target_xyz": torch.zeros((1, 1, 3), dtype=torch.float32),
        "target_bin_center_xyz": torch.zeros((1, 1, 3), dtype=torch.float32),
    }
    common_outputs = {
        "coarse_prediction_mode": "axis_factorized",
        "offset_logits": torch.zeros((1, 1, 3, 4), dtype=torch.float32),
        "stop_logits": torch.zeros((1, 1), dtype=torch.float32),
        "pred_refine_residual": torch.zeros((1, 1, 3), dtype=torch.float32),
        "pred_xyz": torch.zeros((1, 1, 3), dtype=torch.float32),
        "pred_xyz_soft": torch.zeros((1, 1, 3), dtype=torch.float32),
        "coarse_grid_shape": (4, 4, 4),
    }

    near_loss = compute_autoreg_mesh_losses(
        {**common_outputs, "coarse_axis_logits": near_axis_logits},
        common_batch,
        offset_num_bins=(4, 4, 4),
        distance_aware_coarse_targets_enabled=True,
        distance_aware_coarse_target_radius=1,
        distance_aware_coarse_target_sigma=1.0,
        distance_aware_coarse_target_loss="soft_ce",
    )["coarse_loss"]
    far_loss = compute_autoreg_mesh_losses(
        {**common_outputs, "coarse_axis_logits": far_axis_logits},
        common_batch,
        offset_num_bins=(4, 4, 4),
        distance_aware_coarse_targets_enabled=True,
        distance_aware_coarse_target_radius=1,
        distance_aware_coarse_target_sigma=1.0,
        distance_aware_coarse_target_loss="soft_ce",
    )["coarse_loss"]

    assert near_loss.item() < far_loss.item()


def test_mixed_precision_config_validation() -> None:
    config = _make_cached_token_config()
    config["mixed_precision"] = "bf16"
    validated = validate_autoreg_mesh_config(config)
    assert validated["mixed_precision"] == "bf16"

    config2 = _make_cached_token_config()
    config2["mixed_precision"] = "no"
    validated2 = validate_autoreg_mesh_config(config2)
    assert validated2["mixed_precision"] == "no"

    config3 = _make_cached_token_config()
    config3["mixed_precision"] = "fp16"
    with pytest.raises(ValueError, match="mixed_precision"):
        validate_autoreg_mesh_config(config3)


def test_rope_default_config_values() -> None:
    config = _make_cached_token_config()
    validated = validate_autoreg_mesh_config(config)
    assert validated["cross_attention_use_rope"] is True
    assert validated["rope_normalize_coords"] == "separate"
    assert validated["rope_shift_coords"] == pytest.approx(0.05)
    assert validated["rope_jitter_coords"] == pytest.approx(1.05)
    assert validated["rope_rescale_coords"] == pytest.approx(2.0)


def test_batched_rope_from_coords_casts_to_rope_dtype() -> None:
    class _DtypeCheckingRope:
        def __init__(self) -> None:
            self.periods = torch.ones((1,), dtype=torch.bfloat16)
            self.seen_dtypes: list[torch.dtype] = []

        def get_embed_from_coords(self, sample_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.seen_dtypes.append(sample_coords.dtype)
            return sample_coords, sample_coords

    rope = _DtypeCheckingRope()
    coords = torch.randn(2, 4, 3, dtype=torch.float32)

    sin, cos = _batched_rope_from_coords(rope, coords)

    assert sin.shape == coords.shape
    assert cos.shape == coords.shape
    assert rope.seen_dtypes == [torch.bfloat16, torch.bfloat16]


def test_rope_config_validation_rejects_invalid_values() -> None:
    bad_normalize = _make_cached_token_config()
    bad_normalize["rope_normalize_coords"] = "weird"
    with pytest.raises(ValueError, match="rope_normalize_coords"):
        validate_autoreg_mesh_config(bad_normalize)

    bad_combo = _make_cached_token_config()
    bad_combo["rope_base"] = None
    bad_combo["rope_min_period"] = None
    bad_combo["rope_max_period"] = 10.0
    with pytest.raises(ValueError, match="rope_min_period"):
        validate_autoreg_mesh_config(bad_combo)

    bad_jitter = _make_cached_token_config()
    bad_jitter["rope_jitter_coords"] = 1.0
    with pytest.raises(ValueError, match="rope_jitter_coords"):
        validate_autoreg_mesh_config(bad_jitter)

    bad_shift = _make_cached_token_config()
    bad_shift["rope_shift_coords"] = -0.1
    with pytest.raises(ValueError, match="rope_shift_coords"):
        validate_autoreg_mesh_config(bad_shift)

    bad_cross_rope = _make_cached_token_config()
    bad_cross_rope["cross_attention_use_rope"] = "yes"
    with pytest.raises(ValueError, match="cross_attention_use_rope"):
        validate_autoreg_mesh_config(bad_cross_rope)

    bad_coarse_mode = _make_cached_token_config()
    bad_coarse_mode["coarse_prediction_mode"] = "weird"
    with pytest.raises(ValueError, match="coarse_prediction_mode"):
        validate_autoreg_mesh_config(bad_coarse_mode)

    bad_debias_mode = _make_cached_token_config()
    bad_debias_mode["conditioning_feature_debias_mode"] = "weird"
    with pytest.raises(ValueError, match="conditioning_feature_debias_mode"):
        validate_autoreg_mesh_config(bad_debias_mode)

    bad_debias_source = _make_cached_token_config()
    bad_debias_source["conditioning_feature_debias_basis_source"] = "dataset"
    with pytest.raises(ValueError, match="conditioning_feature_debias_basis_source"):
        validate_autoreg_mesh_config(bad_debias_source)

    bad_sched_pattern = _make_cached_token_config()
    bad_sched_pattern["scheduled_sampling_pattern"] = "iid"
    with pytest.raises(ValueError, match="scheduled_sampling_pattern"):
        validate_autoreg_mesh_config(bad_sched_pattern)

    bad_xy_mode = _make_cached_token_config()
    bad_xy_mode["wandb_xy_slice_mode"] = "triple"
    with pytest.raises(ValueError, match="wandb_xy_slice_mode"):
        validate_autoreg_mesh_config(bad_xy_mode)

    bad_downsample = _make_cached_token_config()
    bad_downsample["surface_downsample_factor"] = [2, 1, 1]
    with pytest.raises(ValueError, match="surface_downsample_factor"):
        validate_autoreg_mesh_config(bad_downsample)

    bad_aug_prob = _make_cached_token_config()
    bad_aug_prob["volume_only_augmentation"] = {"enabled": True, "gamma_prob": 1.5}
    with pytest.raises(ValueError, match="volume_only_augmentation.gamma_prob"):
        validate_autoreg_mesh_config(bad_aug_prob)

    bad_spatial_prob = _make_cached_token_config()
    bad_spatial_prob["spatial_augmentation"] = {"enabled": True, "mirror_prob": 1.5}
    with pytest.raises(ValueError, match="spatial_augmentation.mirror_prob"):
        validate_autoreg_mesh_config(bad_spatial_prob)

    bad_cached_spatial = _make_cached_token_config()
    bad_cached_spatial["cache_vol_tokens"] = True
    with pytest.raises(ValueError, match="cache_vol_tokens"):
        validate_autoreg_mesh_config(bad_cached_spatial)

    bad_constraint_mode = _make_cached_token_config()
    bad_constraint_mode["coarse_continuation_constraint_mode"] = "soft_bias"
    with pytest.raises(ValueError, match="coarse_continuation_constraint_mode"):
        validate_autoreg_mesh_config(bad_constraint_mode)

    bad_constraint_fallback = _make_cached_token_config()
    bad_constraint_fallback["coarse_continuation_empty_fallback"] = "none"
    with pytest.raises(ValueError, match="coarse_continuation_empty_fallback"):
        validate_autoreg_mesh_config(bad_constraint_fallback)

    bad_val_split_mode = _make_cached_token_config()
    bad_val_split_mode["val_split_mode"] = "segment_uuid"
    with pytest.raises(ValueError, match="val_split_mode"):
        validate_autoreg_mesh_config(bad_val_split_mode)

    bad_val_leakage_margin = _make_cached_token_config()
    bad_val_leakage_margin["validation_leakage_margin_voxels"] = -1.0
    with pytest.raises(ValueError, match="validation_leakage_margin_voxels"):
        validate_autoreg_mesh_config(bad_val_leakage_margin)

    bad_muon_momentum = _make_muon_cached_token_config()
    bad_muon_momentum["optimizer"]["momentum"] = 1.0
    with pytest.raises(ValueError, match="optimizer.momentum"):
        validate_autoreg_mesh_config(bad_muon_momentum)

    bad_muon_betas = _make_muon_cached_token_config()
    bad_muon_betas["optimizer"]["adamw_betas"] = [0.9]
    with pytest.raises(ValueError, match="optimizer.adamw_betas"):
        validate_autoreg_mesh_config(bad_muon_betas)

    bad_muon_bool = _make_muon_cached_token_config()
    bad_muon_bool["optimizer"]["weight_decouple"] = "yes"
    with pytest.raises(ValueError, match="optimizer.weight_decouple"):
        validate_autoreg_mesh_config(bad_muon_bool)


def test_autoreg_mesh_ddp_cpu_smoke_writes_checkpoints_on_rank_zero_only(tmp_path: Path) -> None:
    port = _find_free_port()
    mp.spawn(_ddp_training_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)

    rank0 = json.loads((tmp_path / "rank0.json").read_text())
    rank1 = json.loads((tmp_path / "rank1.json").read_text())

    assert rank0["world_size"] == 2
    assert rank1["world_size"] == 2
    assert rank0["is_main_process"] is True
    assert rank1["is_main_process"] is False
    assert rank0["history_len"] == 1
    assert rank1["history_len"] == 0
    assert len(rank0["checkpoint_paths"]) == 1
    assert rank1["checkpoint_paths"] == []
    ckpt_path = Path(rank0["checkpoint_paths"][0])
    assert ckpt_path.exists()

    wandb_rank0 = json.loads((tmp_path / "wandb_rank0.json").read_text())
    assert wandb_rank0["init_calls"] == 1
    assert wandb_rank0["log_calls"] >= 1
    assert wandb_rank0["finish_calls"] == 1
    assert not (tmp_path / "wandb_rank1.json").exists()

    resume_config = _make_cached_token_config()
    resume_config.update(
        {
            "out_dir": str(tmp_path / "resume_single"),
            "load_ckpt": str(ckpt_path),
            "save_final_checkpoint": False,
        }
    )
    resume_result = run_autoreg_mesh_training(resume_config, dataset=_make_training_dataset(), device="cpu", max_steps=2)
    assert resume_result["start_step"] == 1


def test_autoreg_mesh_benchmark_smoke_returns_expected_keys() -> None:
    config = _make_cached_token_config()
    dataset = _make_training_dataset()
    model = AutoregMeshModel(config).eval()

    result = run_autoreg_mesh_benchmark(
        config,
        dataset=dataset,
        model=model,
        device="cpu",
        sample_count=2,
    )

    assert result["dataset_length"] == len(dataset)
    assert result["sample_count"] == 2
    assert result["coarse_prediction_mode"] == "joint_pointer"
    assert result["conditioning_feature_debias_mode"] == "none"
    assert result["conditioning_feature_debias_basis_source"] == "zero_volume_svd"
    assert result["conditioning_feature_debias_components"] == 16
    assert result["coarse_grid_shape"] == [2, 2, 2]
    assert result["coarse_axis_sizes"] == {"z": 2, "y": 2, "x": 2}
    assert result["median_prompt_length"] > 0
    assert result["median_target_length"] > 0
    assert result["median_valid_prompt_tokens"] > 0
    assert result["median_valid_target_tokens"] > 0
    assert result["refine_head_present"] is True
    assert result["pointer_temperature"] == pytest.approx(0.25)
    assert result["geometry_metric_enabled"] is True
    assert result["geometry_metric_weight"] == pytest.approx(0.01)
    assert result["geometry_metric_start_step"] == 2000
    assert result["geometry_sd_enabled"] is True
    assert result["geometry_sd_weight"] == pytest.approx(0.005)
    assert result["geometry_sd_start_step"] == 2000
    assert result["distance_aware_coarse_targets_enabled"] is True
    assert result["distance_aware_coarse_target_radius"] == 1
    assert result["distance_aware_coarse_target_sigma"] == pytest.approx(1.0)
    assert result["cross_attention_use_rope"] is True
    assert result["rope_normalize_coords"] == "separate"
    assert result["rope_shift_coords"] == pytest.approx(0.05)
    assert result["rope_jitter_coords"] == pytest.approx(1.05)
    assert result["rope_rescale_coords"] == pytest.approx(2.0)
    assert result["forward_ms"] >= 0.0
    assert result["infer_ms"] >= 0.0


def test_autoreg_mesh_training_microbenchmark_smoke_returns_expected_keys() -> None:
    config = _make_cached_token_config()
    dataset = _make_training_dataset()

    result = run_autoreg_mesh_benchmark(
        config,
        dataset=dataset,
        device="cpu",
        sample_count=1,
        training_steps=1,
    )

    assert result["training_steps"] == 1
    assert result["world_size"] == 1
    assert result["per_rank_batch_size"] == int(config["batch_size"])
    assert result["effective_global_batch_size"] == int(config["batch_size"])
    assert result["startup_ms"] >= 0.0
    assert result["mean_train_step_ms"] >= 0.0
    assert result["samples_per_sec_effective"] >= 0.0


def test_autoreg_mesh_muon_benchmark_reports_optimizer_fields() -> None:
    config = _make_muon_cached_token_config()
    dataset = _make_training_dataset()

    result = run_autoreg_mesh_benchmark(
        config,
        dataset=dataset,
        device="cpu",
        sample_count=1,
        training_steps=1,
    )

    assert result["optimizer_name"] == "muon"
    assert result["muon_momentum"] == pytest.approx(0.95)
    assert result["muon_ns_steps"] == 5
    assert result["muon_adamw_lr"] == pytest.approx(3e-4)
    assert result["muon_weight_decouple"] is True
    assert result["mean_train_step_ms"] >= 0.0


def test_autoreg_mesh_ddp_cpu_benchmark_smoke(tmp_path: Path) -> None:
    port = _find_free_port()
    mp.spawn(_ddp_benchmark_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)

    rank0 = json.loads((tmp_path / "bench_rank0.json").read_text())
    rank1 = json.loads((tmp_path / "bench_rank1.json").read_text())

    assert rank0["world_size"] == 2
    assert rank1["world_size"] == 2
    assert rank0["training_steps"] == 1
    assert rank0["mean_train_step_ms"] >= 0.0
    assert rank0["samples_per_sec_effective"] >= 0.0


def test_debias_benchmark_reports_settings_and_norm_ratio(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_debias_config(checkpoint)
    dataset = _ListDataset([_make_sample("left")])
    model = AutoregMeshModel(config).eval()

    result = run_autoreg_mesh_benchmark(
        config,
        dataset=dataset,
        model=model,
        device="cpu",
        sample_count=1,
    )

    assert result["conditioning_feature_debias_mode"] == "orthogonal_project"
    assert result["conditioning_feature_debias_basis_source"] == "zero_volume_svd"
    assert result["conditioning_feature_debias_components"] == 4
    assert np.isfinite(result["conditioning_feature_debias_norm_ratio"])


def test_axis_factorized_benchmark_reports_mode_and_axis_sizes() -> None:
    config = _make_factorized_cached_token_config()
    dataset = _make_training_dataset()
    model = AutoregMeshModel(config).eval()

    result = run_autoreg_mesh_benchmark(
        config,
        dataset=dataset,
        model=model,
        device="cpu",
        sample_count=1,
    )

    assert result["coarse_prediction_mode"] == "axis_factorized"
    assert result["coarse_grid_shape"] == [2, 2, 2]
    assert result["coarse_axis_sizes"] == {"z": 2, "y": 2, "x": 2}


def test_cross_attention_rope_can_be_disabled(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    config["cross_attention_use_rope"] = False
    model = AutoregMeshModel(config).eval()

    batch = autoreg_mesh_collate([_make_sample("left")])
    batch = _move_batch(batch, torch.device("cpu"))
    outputs = model(batch)

    assert torch.isfinite(outputs["coarse_logits"]).all()


def test_autoreg_mesh_wandb_logging_includes_metrics_and_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _make_cached_token_config()
    config.update(
        {
            "out_dir": str(tmp_path / "runs_wandb"),
            "wandb_project": "mesh-mvp",
            "wandb_entity": "scroll",
            "wandb_run_name": "smoke",
            "wandb_log_images": True,
            "wandb_image_frequency": 1,
            "val_fraction": 0.5,
        }
    )
    dataset = _make_training_dataset()
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=1)

    assert result["wandb_run_id"] == "fake-run-1"
    assert fake_wandb.init_calls[0]["project"] == "mesh-mvp"
    assert fake_wandb.init_calls[0]["entity"] == "scroll"
    assert fake_wandb.init_calls[0]["name"] == "smoke"
    assert "config" in fake_wandb.init_calls[0]
    assert fake_wandb.logs[0]["step"] == 1
    logged = fake_wandb.logs[0]["data"]
    assert "train/loss" in logged
    assert "train/coarse_loss" in logged
    assert "train/coarse_excess_nll" in logged
    assert "train/xyz_soft_loss" in logged
    assert "train/seam_loss" in logged
    assert "train/triangle_barrier_loss" in logged
    assert "train/xyz_l1_refined" in logged
    assert "train/seam_edge_error_refined" in logged
    assert "train/triangle_flip_rate_refined" in logged
    assert "train/current_lr" in logged
    assert "train/grad_norm" in logged
    assert "val/loss" in logged
    assert "val/rollout_xyz_l1_refined" in logged
    assert "val/rollout_seam_edge_error" in logged
    assert "val/rollout_triangle_flip_rate" in logged
    assert "train/example" in logged
    assert "train/example_projection" in logged
    assert "train/example_xy" in logged
    assert "val/example" in logged
    assert "val/example_projection" in logged
    assert "val/example_xy" in logged
    assert isinstance(logged["train/example"], _FakeWandbImage)
    assert isinstance(logged["train/example_projection"], _FakeWandbImage)
    assert isinstance(logged["train/example_xy"], _FakeWandbImage)
    assert isinstance(logged["val/example"], _FakeWandbImage)
    assert isinstance(logged["val/example_projection"], _FakeWandbImage)
    assert isinstance(logged["val/example_xy"], _FakeWandbImage)
    assert logged["train/example"].data.ndim == 3
    assert logged["train/example_projection"].data.ndim == 3
    assert logged["train/example_xy"].data.ndim == 3
    assert logged["val/example"].data.ndim == 3
    assert logged["val/example_projection"].data.ndim == 3
    assert logged["val/example_xy"].data.ndim == 3
    assert fake_wandb.finish_calls == 1


def test_autoreg_mesh_resume_uses_checkpoint_wandb_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_training_dataset()
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    first_config = _make_cached_token_config()
    first_config.update(
        {
            "out_dir": str(tmp_path / "resume_runs"),
            "wandb_project": "mesh-mvp",
            "save_final_checkpoint": False,
        }
    )
    first_result = run_autoreg_mesh_training(first_config, dataset=dataset, device="cpu", max_steps=1)
    ckpt_path = Path(first_result["checkpoint_paths"][0])
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert saved["wandb_run_id"] == "fake-run-1"

    second_config = _make_cached_token_config()
    second_config.update(
        {
            "out_dir": str(tmp_path / "resume_runs"),
            "wandb_project": "mesh-mvp",
            "wandb_resume": True,
            "load_ckpt": str(ckpt_path),
            "save_final_checkpoint": False,
        }
    )
    second_result = run_autoreg_mesh_training(second_config, dataset=dataset, device="cpu", max_steps=2)

    assert second_result["start_step"] == 1
    assert fake_wandb.init_calls[1]["resume"] == "allow"
    assert fake_wandb.init_calls[1]["id"] == "fake-run-1"


def test_autoreg_mesh_resume_rejects_mismatched_coarse_prediction_mode(tmp_path: Path) -> None:
    dataset = _make_training_dataset()

    first_config = _make_cached_token_config()
    first_config.update(
        {
            "out_dir": str(tmp_path / "mode_resume_runs"),
            "save_final_checkpoint": False,
        }
    )
    first_result = run_autoreg_mesh_training(first_config, dataset=dataset, device="cpu", max_steps=1)
    ckpt_path = Path(first_result["checkpoint_paths"][0])

    second_config = _make_factorized_cached_token_config()
    second_config.update(
        {
            "out_dir": str(tmp_path / "mode_resume_runs"),
            "load_ckpt": str(ckpt_path),
            "save_final_checkpoint": False,
        }
    )
    with pytest.raises(ValueError, match="coarse_prediction_mode"):
        run_autoreg_mesh_training(second_config, dataset=dataset, device="cpu", max_steps=2)
