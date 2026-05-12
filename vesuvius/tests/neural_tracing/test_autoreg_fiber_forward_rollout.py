"""Unit tests for ``AutoregFiberModel.forward_rollout``.

These verify the multi-step autoregressive rollout used by the optional
rollout-in-the-loop training branch (see plan B of
``sprightly-wiggling-puffin.md``):

* shapes match the existing ``forward_from_encoded`` contract,
* gradients flow back through the rollout into the model's parameters,
* when the prediction is forced to equal the ground-truth (via a
  monkey-patch of the model's coarse / offset / refine heads), the rollout
  collapses to teacher-forced behaviour — i.e. the rollout path is correct.
"""

from __future__ import annotations

import numpy as np
import torch

from vesuvius.neural_tracing.autoreg_fiber.dataset import AutoregFiberDataset, autoreg_fiber_collate
from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import FiberPath, write_fiber_cache
from vesuvius.neural_tracing.autoreg_fiber.model import AutoregFiberModel


def _tiny_config(tmp_path) -> dict:
    return {
        "dinov2_backbone": None,
        "crop_size": [16, 16, 16],
        "input_shape": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "offset_num_bins": [4, 4, 4],
        "prompt_length": 2,
        "target_length": 4,
        "point_stride": 1,
        "decoder_dim": 24,
        "decoder_depth": 2,
        "decoder_num_heads": 2,
        "decoder_dropout": 0.0,
        "max_fiber_position_embeddings": 64,
        "coarse_prediction_mode": "axis_factorized",
        "cross_attention_every_n_blocks": 1,
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
        "num_steps": 1,
        "log_frequency": 1,
        "ckpt_frequency": 50,
        "save_final_checkpoint": False,
        "out_dir": str(tmp_path / "runs"),
    }


def _write_cache(tmp_path, points_zyx: np.ndarray):
    fiber = FiberPath(
        annotation_id="ann-rollout",
        tree_id="tree-rollout",
        target_volume="PHerc0332",
        marker="fibers_s3",
        source_points_xyz=points_zyx[:, ::-1].astype(np.float32),
        points_zyx=points_zyx.astype(np.float32),
        transform_checksum="checksum-rollout",
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
    volume = np.random.default_rng(0).uniform(0.0, 1.0, size=(16, 16, 16)).astype(np.float32)
    cfg = _tiny_config(tmp_path)
    cfg["fiber_cache_paths"] = [str(cache_path)]
    dataset = AutoregFiberDataset(cfg, volume_array=volume)
    return cfg, dataset, autoreg_fiber_collate([dataset[0]])


def test_forward_rollout_shapes(tmp_path) -> None:
    torch.manual_seed(0)
    cfg, _ds, batch = _dataset_and_batch(tmp_path)
    model = AutoregFiberModel(cfg).eval()

    K = 3
    out = model.forward_rollout(batch, rollout_steps=K)
    assert out["coarse_logits"] is None
    assert isinstance(out["coarse_axis_logits"], dict)
    assert set(out["coarse_axis_logits"]) == {"z", "y", "x"}
    for axis in ("z", "y", "x"):
        assert out["coarse_axis_logits"][axis].shape[:2] == (1, K)
    assert out["offset_logits"].shape == (1, K, 3, max(cfg["offset_num_bins"]))
    assert out["stop_logits"].shape == (1, K)
    assert out["pred_xyz_soft"].shape == (1, K, 3)
    assert out["pred_xyz"].shape == (1, K, 3)
    assert out["pred_xyz_refined"].shape == (1, K, 3)
    assert out["pred_coarse_ids"].shape == (1, K)
    assert out["pred_offset_bins"].shape == (1, K, 3)
    assert out["pred_refine_residual"].shape == (1, K, 3)


def test_forward_rollout_gradient_flows(tmp_path) -> None:
    torch.manual_seed(1)
    cfg, _ds, batch = _dataset_and_batch(tmp_path)
    model = AutoregFiberModel(cfg)  # train mode
    K = 3
    out = model.forward_rollout(batch, rollout_steps=K)
    # Differentiable surrogate loss on the soft-decoded xyz.
    target = batch["target_xyz"][:, :K, :]
    loss = (out["pred_xyz_soft"] - target).pow(2).mean()
    loss.backward()
    # Sentinel parameter that participates in *every* forward path: the
    # xyz_mlp's first Linear weight. If gradient flow is broken anywhere
    # in the rollout chain it'll be None or zero.
    sentinel = model.xyz_mlp[0].weight
    assert sentinel.grad is not None
    assert torch.isfinite(sentinel.grad).all()
    assert sentinel.grad.abs().sum().item() > 0.0


def test_forward_rollout_collapses_to_teacher_forced_when_predictions_match_targets(tmp_path) -> None:
    """When the model's outputs are monkey-patched so each step predicts the
    ground-truth target, the rollout's next-step input equals what
    teacher-forcing would have fed in — so the per-position heads should
    match the teacher-forced forward (up to float tolerance, since the
    monkey-patched outputs share the same parameters per step)."""

    torch.manual_seed(2)
    cfg, _ds, batch = _dataset_and_batch(tmp_path)
    model = AutoregFiberModel(cfg).eval()
    K = int(cfg["target_length"])

    real_forward_from_encoded = AutoregFiberModel.forward_from_encoded

    def _patched(self, b, *, memory_tokens, memory_patch_centers, generation_inputs=None):
        out = real_forward_from_encoded(
            self, b, memory_tokens=memory_tokens, memory_patch_centers=memory_patch_centers,
            generation_inputs=generation_inputs,
        )
        # Override the soft / argmax predictions with the *ground-truth*
        # target so the rollout's next-step feedback matches teacher-forcing.
        T = out["pred_xyz_soft"].shape[1]
        gt_xyz = b["target_xyz"][:, :T, :].to(dtype=out["pred_xyz_soft"].dtype)
        gt_coarse = b["target_coarse_ids"][:, :T]
        gt_offset = b["target_offset_bins"][:, :T, :]
        out["pred_xyz_soft"] = gt_xyz
        out["pred_xyz"] = gt_xyz
        out["pred_coarse_ids"] = torch.where(gt_coarse < 0, torch.zeros_like(gt_coarse), gt_coarse)
        out["pred_offset_bins"] = torch.where(gt_offset < 0, torch.zeros_like(gt_offset), gt_offset)
        return out

    try:
        AutoregFiberModel.forward_from_encoded = _patched  # type: ignore[method-assign]
        rollout_out = model.forward_rollout(batch, rollout_steps=K)
    finally:
        AutoregFiberModel.forward_from_encoded = real_forward_from_encoded  # type: ignore[method-assign]

    # Compare to the un-patched teacher-forced forward (apples to apples).
    with torch.no_grad():
        tf_out = model(batch)
    for axis in ("z", "y", "x"):
        torch.testing.assert_close(
            rollout_out["coarse_axis_logits"][axis],
            tf_out["coarse_axis_logits"][axis][:, :K, :],
            rtol=1e-4,
            atol=1e-4,
        )
    torch.testing.assert_close(
        rollout_out["offset_logits"], tf_out["offset_logits"][:, :K, :, :], rtol=1e-4, atol=1e-4
    )


def test_forward_rollout_rejects_out_of_range(tmp_path) -> None:
    torch.manual_seed(3)
    cfg, _ds, batch = _dataset_and_batch(tmp_path)
    model = AutoregFiberModel(cfg).eval()
    try:
        model.forward_rollout(batch, rollout_steps=0)
    except ValueError as exc:
        assert "rollout_steps must be >= 1" in str(exc)
    else:
        raise AssertionError("expected ValueError for rollout_steps=0")

    try:
        model.forward_rollout(batch, rollout_steps=int(cfg["target_length"]) + 1)
    except ValueError as exc:
        assert "exceeds config.target_length" in str(exc)
    else:
        raise AssertionError("expected ValueError for rollout_steps > target_length")
