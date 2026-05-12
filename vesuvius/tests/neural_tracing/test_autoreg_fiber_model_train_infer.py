from __future__ import annotations

import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from vesuvius.neural_tracing.autoreg_fiber.dataset import AutoregFiberDataset, autoreg_fiber_collate
from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import FiberPath, write_fiber_cache
from vesuvius.neural_tracing.autoreg_fiber.infer import infer_autoreg_fiber
from vesuvius.neural_tracing.autoreg_fiber.losses import compute_autoreg_fiber_losses
from vesuvius.neural_tracing.autoreg_fiber.model import AutoregFiberModel
from vesuvius.neural_tracing.autoreg_fiber.train import (
    _ddp_find_unused_parameters_enabled,
    _make_projection_canvas,
    _make_xy_slice_overlay_canvas,
    _wandb_dataset_summary,
    _wandb_metrics_payload,
    load_autoreg_fiber_model_from_checkpoint,
    run_autoreg_fiber_training,
)


class _ListDataset(Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[int(idx)]


class _FakeWandbImage:
    def __init__(self, data, caption=None) -> None:
        self.data = np.asarray(data)
        self.caption = caption


class _FakeWandbTable:
    def __init__(self, columns=None, data=None) -> None:
        self.columns = list(columns or [])
        self.data = list(data or [])


class _FakeWandbRun:
    def __init__(self, run_id: str) -> None:
        self.id = str(run_id)
        self.summary = {}


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

    def Table(self, columns=None, data=None):
        return _FakeWandbTable(columns=columns, data=data)


def _tiny_config(tmp_path) -> dict:
    return {
        "dinov2_backbone": None,
        "crop_size": [16, 16, 16],
        "input_shape": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "offset_num_bins": [4, 4, 4],
        "prompt_length": 2,
        "target_length": 3,
        "point_stride": 1,
        "decoder_dim": 24,
        "decoder_depth": 1,
        "decoder_num_heads": 2,
        "decoder_dropout": 0.0,
        "max_fiber_position_embeddings": 32,
        "distance_aware_coarse_targets_enabled": False,
        "position_refine_start_step": 0,
        "position_refine_weight": 0.01,
        "xyz_soft_loss_weight": 0.1,
        "segment_vector_loss_weight": 0.01,
        "optimizer": {"name": "adamw", "learning_rate": 2e-3, "weight_decay": 0.0},
        "batch_size": 1,
        "num_workers": 0,
        "val_num_workers": 0,
        "val_fraction": 0.0,
        "num_steps": 2,
        "log_frequency": 1,
        "ckpt_frequency": 50,
        "save_final_checkpoint": True,
        "out_dir": str(tmp_path / "runs"),
    }


def _write_cache(tmp_path, points_zyx: np.ndarray):
    fiber = FiberPath(
        annotation_id="ann-a",
        tree_id="tree-a",
        target_volume="PHerc0332",
        marker="fibers_s3",
        source_points_xyz=points_zyx[:, ::-1].astype(np.float32),
        points_zyx=points_zyx.astype(np.float32),
        transform_checksum="checksum-a",
        densify_step=None,
    )
    return write_fiber_cache(fiber, tmp_path)


def _dataset_and_batch(tmp_path):
    points = np.asarray(
        [
            [3.0, 4.0, 5.0],
            [4.0, 4.0, 5.0],
            [5.0, 4.0, 5.0],
            [6.0, 4.0, 5.0],
            [7.0, 4.0, 5.0],
            [8.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    cache_path = _write_cache(tmp_path, points)
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    cfg = _tiny_config(tmp_path)
    cfg["fiber_cache_paths"] = [str(cache_path)]
    dataset = AutoregFiberDataset(cfg, volume_array=volume)
    return cfg, dataset, autoreg_fiber_collate([dataset[0]])


def _loss(model: AutoregFiberModel, batch: dict, cfg: dict) -> torch.Tensor:
    outputs = model(batch)
    return compute_autoreg_fiber_losses(
        outputs,
        batch,
        offset_num_bins=tuple(cfg["offset_num_bins"]),
        offset_loss_weight_active=1.0,
        position_refine_weight_active=float(cfg["position_refine_weight"]),
        xyz_soft_loss_weight_active=float(cfg["xyz_soft_loss_weight"]),
        segment_vector_loss_weight_active=float(cfg["segment_vector_loss_weight"]),
        distance_aware_coarse_targets_enabled=bool(cfg["distance_aware_coarse_targets_enabled"]),
    )["loss"]


def test_axis_factorized_model_forward_and_loss_are_finite(tmp_path) -> None:
    torch.manual_seed(9)
    cfg, _dataset, batch = _dataset_and_batch(tmp_path)
    cfg["coarse_prediction_mode"] = "axis_factorized"
    model = AutoregFiberModel(cfg)
    outputs = model(batch)
    losses = compute_autoreg_fiber_losses(
        outputs,
        batch,
        offset_num_bins=tuple(cfg["offset_num_bins"]),
        offset_loss_weight_active=1.0,
        position_refine_weight_active=float(cfg["position_refine_weight"]),
        xyz_soft_loss_weight_active=float(cfg["xyz_soft_loss_weight"]),
        segment_vector_loss_weight_active=float(cfg["segment_vector_loss_weight"]),
        distance_aware_coarse_targets_enabled=True,
    )

    assert outputs["coarse_prediction_mode"] == "axis_factorized"
    assert outputs["coarse_logits"] is None
    assert set(outputs["coarse_axis_logits"]) == {"z", "y", "x"}
    assert outputs["coarse_axis_logits"]["z"].shape[-1] == 2
    assert torch.isfinite(losses["loss"])
    assert "coarse_axis_loss_z" in losses


def test_ddp_find_unused_parameters_is_enabled_for_axis_and_staged_losses(tmp_path) -> None:
    cfg = _tiny_config(tmp_path)

    cfg.update(
        {
            "coarse_prediction_mode": "joint_pointer",
            "offset_loss_start_step": 0,
            "position_refine_start_step": 0,
            "xyz_soft_loss_start_step": 0,
            "segment_vector_loss_start_step": 0,
        }
    )
    assert not _ddp_find_unused_parameters_enabled(cfg)

    cfg["coarse_prediction_mode"] = "axis_factorized"
    assert _ddp_find_unused_parameters_enabled(cfg)

    cfg["ddp_find_unused_parameters"] = False
    assert not _ddp_find_unused_parameters_enabled(cfg)

    cfg["coarse_prediction_mode"] = "joint_pointer"
    cfg["ddp_find_unused_parameters"] = None
    cfg["offset_loss_start_step"] = 4000
    assert _ddp_find_unused_parameters_enabled(cfg)


def test_fiber_skeleton_projection_and_xy_canvases_are_rgb_images() -> None:
    prompt = np.asarray(
        [
            [2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    target = np.asarray(
        [
            [6.0, 7.0, 8.0],
            [8.0, 8.0, 10.0],
            [10.0, 9.0, 12.0],
        ],
        dtype=np.float32,
    )
    pred = target + np.asarray([0.0, 1.0, -1.0], dtype=np.float32)
    volume = np.zeros((16, 16, 16), dtype=np.float32)

    projection = _make_projection_canvas(
        prompt_points_local=prompt,
        target_points_local=target,
        pred_points_local=pred,
        crop_shape=(16, 16, 16),
        line_thickness=1,
    )
    xy = _make_xy_slice_overlay_canvas(
        volume=volume,
        prompt_points_local=prompt,
        target_points_local=target,
        pred_points_local=pred,
        line_thickness=1,
        depth_tolerance=1.0,
    )

    assert projection.ndim == 3
    assert projection.shape[-1] == 3
    assert projection.shape[0] >= 280
    assert projection.shape[1] >= 760
    assert projection.dtype == np.uint8
    assert int(projection.sum()) > 0
    assert xy.ndim == 3
    assert xy.shape[-1] == 3
    assert xy.shape[0] >= 530
    assert xy.shape[1] >= 512
    assert xy.dtype == np.uint8
    assert int(xy.sum()) > 0


def test_wandb_metrics_payload_splits_fiber_namespaces() -> None:
    payload = _wandb_metrics_payload(
        {
            "loss": 1.0,
            "val_loss": 2.0,
            "rollout_val_xyz_l1": 3.0,
            "num_train_fibers": 4.0,
        }
    )

    assert payload["train/loss"] == 1.0
    assert payload["val/loss"] == 2.0
    assert payload["rollout_val/xyz_l1"] == 3.0
    assert payload["data/num_train_fibers"] == 4.0


def test_wandb_dataset_summary_aggregates_autoreg_fiber_metadata(tmp_path) -> None:
    cfg, dataset, _batch = _dataset_and_batch(tmp_path)

    scalars, rows = _wandb_dataset_summary(
        dataset=dataset,
        train_dataset=dataset,
        val_dataset=None,
        max_table_rows=8,
    )

    assert scalars["num_fiber_cache_files"] == 1.0
    assert scalars["num_fibers"] == 1.0
    assert scalars["marker_fibers_s3_fibers"] == 1.0
    assert scalars["target_PHerc0332_fibers"] == 1.0
    assert rows == [
        {
            "marker": "fibers_s3",
            "target_volume": "PHerc0332",
            "fibers": 1,
            "sample_windows": len(dataset),
            "train_windows": len(dataset),
            "val_windows": 0,
            "train_fibers": 1,
            "val_fibers": 0,
            "point_count_min": 6,
            "point_count_mean": 6.0,
            "point_count_max": 6,
        }
    ]


def test_tiny_model_forward_backward_is_finite(tmp_path) -> None:
    torch.manual_seed(0)
    cfg, _dataset, batch = _dataset_and_batch(tmp_path)
    model = AutoregFiberModel(cfg)

    outputs = model(batch)
    assert outputs["coarse_logits"].shape == (1, 3, 8)
    assert outputs["offset_logits"].shape == (1, 3, 3, 4)
    assert outputs["stop_logits"].shape == (1, 3)
    loss = _loss(model, batch, cfg)
    loss.backward()

    assert torch.isfinite(loss)
    grad_values = [p.grad.detach().abs().sum() for p in model.parameters() if p.grad is not None]
    assert grad_values
    assert torch.isfinite(torch.stack(grad_values)).all()


def test_toy_fiber_loss_decreases_with_teacher_forcing(tmp_path) -> None:
    torch.manual_seed(1)
    cfg, _dataset, batch = _dataset_and_batch(tmp_path)
    model = AutoregFiberModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, batch, cfg)
        losses.append(float(loss.detach().item()))
        loss.backward()
        optimizer.step()

    assert min(losses[-3:]) < losses[0]


def test_greedy_inference_stops_and_exports(tmp_path) -> None:
    torch.manual_seed(2)
    cfg, _dataset, batch = _dataset_and_batch(tmp_path)
    model = AutoregFiberModel(cfg)
    with torch.no_grad():
        model.stop_head.bias.fill_(10.0)
    out_dir = tmp_path / "infer"

    result = infer_autoreg_fiber(
        model,
        batch,
        max_steps=3,
        stop_probability_threshold=0.5,
        min_steps=1,
        greedy=True,
        save_path=out_dir,
    )

    assert result["predicted_fiber_local_zyx"].shape == (1, 3)
    assert np.isfinite(result["predicted_fiber_local_zyx"]).all()
    assert set(result["saved_paths"]) == {"npz", "csv", "nml"}
    for path in result["saved_paths"].values():
        assert (tmp_path / "infer" / path.split("/")[-1]).exists()


def test_axis_factorized_greedy_inference_uses_axis_logits(tmp_path) -> None:
    torch.manual_seed(10)
    cfg, _dataset, batch = _dataset_and_batch(tmp_path)
    cfg["coarse_prediction_mode"] = "axis_factorized"
    model = AutoregFiberModel(cfg)
    with torch.no_grad():
        model.stop_head.bias.fill_(10.0)

    result = infer_autoreg_fiber(
        model,
        batch,
        max_steps=2,
        stop_probability_threshold=0.5,
        min_steps=1,
        greedy=True,
    )

    assert result["predicted_fiber_local_zyx"].shape == (1, 3)
    assert np.isfinite(result["predicted_fiber_local_zyx"]).all()


def test_training_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(3)
    cfg, dataset, batch = _dataset_and_batch(tmp_path)
    result = run_autoreg_fiber_training(cfg, dataset=dataset, device="cpu", max_steps=1)

    final_path = result["final_checkpoint_path"]
    assert final_path is not None
    loaded = load_autoreg_fiber_model_from_checkpoint(final_path)
    outputs = loaded(batch)

    assert outputs["pred_xyz_refined"].shape == (1, 3, 3)
    assert torch.isfinite(outputs["pred_xyz_refined"]).all()


def test_autoreg_fiber_wandb_logging_includes_skeleton_images(tmp_path, monkeypatch) -> None:
    torch.manual_seed(4)
    cfg, dataset, _batch = _dataset_and_batch(tmp_path)
    cfg.update(
        {
            "out_dir": str(tmp_path / "runs_wandb"),
            "wandb_project": "fiber-mvp",
            "wandb_entity": "scroll",
            "wandb_run_name": "smoke",
            "webknossos_api_token": "do-not-log",
            "wandb_log_images": True,
            "wandb_image_frequency": 1,
            "wandb_log_xy_slice_images": True,
            "wandb_xy_slice_image_frequency": 1,
            "val_fraction": 0.5,
            "save_final_checkpoint": False,
        }
    )
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    list_dataset = _ListDataset([dataset[0], dataset[1]])
    result = run_autoreg_fiber_training(cfg, dataset=list_dataset, device="cpu", max_steps=1)

    assert result["wandb_run_id"] == "fake-run-1"
    assert fake_wandb.init_calls[0]["project"] == "fiber-mvp"
    assert fake_wandb.init_calls[0]["entity"] == "scroll"
    assert fake_wandb.init_calls[0]["name"] == "smoke"
    assert "config" in fake_wandb.init_calls[0]
    assert fake_wandb.init_calls[0]["config"]["webknossos_api_token"] == "[redacted]"

    dataset_summary_log = next(log for log in fake_wandb.logs if log["step"] == 0)
    assert dataset_summary_log["data"]["data/dataset_length"] == 2.0

    step_log = next(log for log in fake_wandb.logs if log["step"] == 1)
    logged = step_log["data"]
    assert "train/loss" in logged
    assert "train/coarse_loss" in logged
    assert "val/loss" in logged
    assert "rollout_val/xyz_l1" in logged
    assert "train/example_projection" in logged
    assert "train/example_xy" in logged
    assert "val/example_projection" in logged
    assert "val/example_xy" in logged
    assert "train/example" not in logged
    assert "val/example" not in logged
    assert isinstance(logged["train/example_projection"], _FakeWandbImage)
    assert isinstance(logged["train/example_xy"], _FakeWandbImage)
    assert isinstance(logged["val/example_projection"], _FakeWandbImage)
    assert isinstance(logged["val/example_xy"], _FakeWandbImage)
    assert logged["train/example_projection"].data.ndim == 3
    assert logged["train/example_xy"].data.ndim == 3
    assert "skeleton" in logged["train/example_projection"].caption
    assert fake_wandb.finish_calls == 1
