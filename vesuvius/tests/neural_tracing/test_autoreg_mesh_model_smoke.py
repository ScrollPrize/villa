from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.neural_tracing.autoreg_mesh.dataset import autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.infer import infer_autoreg_mesh
from vesuvius.neural_tracing.autoreg_mesh.losses import compute_autoreg_mesh_losses
from vesuvius.neural_tracing.autoreg_mesh.model import AutoregMeshModel
from vesuvius.neural_tracing.autoreg_mesh.serialization import serialize_split_conditioning_example
from vesuvius.neural_tracing.autoreg_mesh.train import run_autoreg_mesh_training


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
        frontier_band_width=1,
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
        "prompt_anchor_xyz": torch.from_numpy(serialized["prompt_anchor_xyz"]).to(torch.float32),
        "prompt_grid_local": torch.from_numpy(serialized["prompt_grid_local"]).to(torch.float32),
        "target_coarse_ids": torch.from_numpy(serialized["target_coarse_ids"]).to(torch.long),
        "target_offset_bins": torch.from_numpy(serialized["target_offset_bins"]).to(torch.long),
        "target_stop": torch.from_numpy(serialized["target_stop"]).to(torch.float32),
        "target_xyz": torch.from_numpy(serialized["target_xyz"]).to(torch.float32),
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
        "frontier_band_width": 1,
        "batch_size": 1,
        "num_workers": 0,
        "num_steps": 2,
        "optimizer": {"name": "adamw", "learning_rate": 1e-3, "weight_decay": 0.0},
    }


class _ListDataset(Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[int(idx)]


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
    for value in losses.values():
        assert torch.isfinite(value)


def test_autoreg_mesh_smoke_training_runs_two_steps(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_dinovol.pt"
    _write_local_guide_checkpoint(checkpoint)
    config = _make_config(checkpoint)
    dataset = _ListDataset([_make_sample("left"), _make_sample("right"), _make_sample("up")])
    result = run_autoreg_mesh_training(config, dataset=dataset, device="cpu", max_steps=2)

    assert len(result["history"]) == 2
    assert np.isfinite(result["final_metrics"]["loss"])
    assert np.isfinite(result["final_metrics"]["coarse_loss"])
    assert np.isfinite(result["final_metrics"]["offset_loss"])
    assert np.isfinite(result["final_metrics"]["stop_loss"])


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
    )

    target_shape = tuple(int(v) for v in sample["target_grid_shape"].tolist())
    assert result["predicted_continuation_vertices_local"].shape[0] == int(sample["target_coarse_ids"].shape[0])
    assert result["continuation_grid_local"].shape == (*target_shape, 3)
    assert result["full_grid_local"].shape[-1] == 3
    assert result["full_grid_local"].shape[0] >= result["continuation_grid_local"].shape[0]
