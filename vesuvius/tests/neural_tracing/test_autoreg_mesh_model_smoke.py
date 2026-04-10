from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import tifffile
from torch.utils.data import Dataset

from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.neural_tracing.autoreg_mesh.benchmark import run_autoreg_mesh_benchmark
from vesuvius.neural_tracing.autoreg_mesh.config import validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.dataset import autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.infer import infer_autoreg_mesh
from vesuvius.neural_tracing.autoreg_mesh.losses import (
    _build_distance_aware_coarse_targets,
    compute_autoreg_mesh_losses,
)
from vesuvius.neural_tracing.autoreg_mesh.model import AutoregMeshModel
from vesuvius.neural_tracing.autoreg_mesh.serialization import serialize_split_conditioning_example
from vesuvius.neural_tracing.autoreg_mesh.train import _restrict_dataset_samples, run_autoreg_mesh_training
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


def test_autoreg_mesh_model_forward_and_losses_are_finite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    model = AutoregMeshModel(config)

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
    assert outputs["pred_xyz_refined"].shape == batch["target_xyz"].shape
    assert batch["target_valid_mask"].shape == batch["target_mask"].shape
    assert batch["target_supervision_mask"].shape == batch["target_mask"].shape
    for value in losses.values():
        assert torch.isfinite(value)
    assert "occupancy_metric" in losses


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


def test_autoreg_mesh_validation_metrics_are_logged_in_history(tmp_path: Path) -> None:
    config = _make_cached_token_config()
    config["out_dir"] = str(tmp_path / "runs_val")
    config["val_fraction"] = 0.5
    dataset = _make_training_dataset()

    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=1)

    assert len(result["history"]) == 1
    assert "val_loss" in result["history"][0]
    assert "val_coarse_loss" in result["history"][0]
    assert "val_offset_loss" in result["history"][0]
    assert "val_stop_loss" in result["history"][0]


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
    assert validated["distance_aware_coarse_targets_enabled"] is True
    assert validated["distance_aware_coarse_target_radius"] == 2
    assert validated["distance_aware_coarse_target_sigma"] == pytest.approx(2.0)


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


def test_mixed_precision_is_rejected_until_supported() -> None:
    config = _make_cached_token_config()
    config["mixed_precision"] = "bf16"
    with pytest.raises(ValueError, match="does not implement mixed precision"):
        validate_autoreg_mesh_config(config)


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
    assert result["median_prompt_length"] > 0
    assert result["median_target_length"] > 0
    assert result["median_valid_prompt_tokens"] > 0
    assert result["median_valid_target_tokens"] > 0
    assert result["refine_head_present"] is True
    assert result["distance_aware_coarse_targets_enabled"] is True
    assert result["distance_aware_coarse_target_radius"] == 2
    assert result["distance_aware_coarse_target_sigma"] == pytest.approx(2.0)
    assert result["forward_ms"] >= 0.0
    assert result["infer_ms"] >= 0.0


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
    assert "loss" in logged
    assert "coarse_loss" in logged
    assert "current_lr" in logged
    assert "grad_norm" in logged
    assert "val_loss" in logged
    assert "train_example" in logged
    assert "val_example" in logged
    assert isinstance(logged["train_example"], _FakeWandbImage)
    assert isinstance(logged["val_example"], _FakeWandbImage)
    assert logged["train_example"].data.ndim == 3
    assert logged["val_example"].data.ndim == 3
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
