"""Equivalence tests for AutoregFiberModel KV-cache streaming inference.

The streaming tracer added in this branch advances the decoder one token at a
time using ``init_kv_cache`` + ``step_from_encoded_cached``. These tests
exercise the cached path against the naive single-shot inference loop
(`infer_autoreg_fiber`) on a tiny CPU-only model and assert that:

  * greedy predictions are identical token-for-token (coarse ids, offset bins,
    bin-centre XYZ, stop probability) under both prediction modes
    ('joint_pointer' and 'axis_factorized');
  * continuous logits (coarse, offset, stop) agree within a small float
    tolerance accounting for the slightly different op ordering.

This is the load-bearing correctness gate for the streaming code path.
"""

from __future__ import annotations

import numpy as np
import torch

from vesuvius.neural_tracing.autoreg_fiber.dataset import AutoregFiberDataset, autoreg_fiber_collate
from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import FiberPath, write_fiber_cache
from vesuvius.neural_tracing.autoreg_fiber.infer import _sample_from_logits, infer_autoreg_fiber
from vesuvius.neural_tracing.autoreg_fiber.model import (
    AutoregFiberModel,
    FiberKVCache,
    GENERATED_TOKEN_TYPE,
    PROMPT_TOKEN_TYPE,
    START_TOKEN_TYPE,
    build_pseudo_inference_batch,
)
from vesuvius.neural_tracing.autoreg_fiber.serialization import IGNORE_INDEX


def _tiny_config(tmp_path, *, coarse_prediction_mode: str) -> dict:
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
        "coarse_prediction_mode": coarse_prediction_mode,
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


def _dataset_and_batch(tmp_path, cfg):
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
    rng = np.random.default_rng(0)
    volume = rng.uniform(0.0, 1.0, size=(16, 16, 16)).astype(np.float32)
    cfg = dict(cfg)
    cfg["fiber_cache_paths"] = [str(cache_path)]
    cfg["spatial_augmentation_enabled"] = False
    dataset = AutoregFiberDataset(cfg, volume_array=volume)
    return cfg, dataset, autoreg_fiber_collate([dataset[0]])


def _coarse_id_from_outputs(model: AutoregFiberModel, outputs: dict, *, t_idx: int = 0) -> int:
    """Pick the greedy coarse id from the relevant logits, mirroring infer.py."""

    coarse_logits = outputs.get("coarse_logits")
    if coarse_logits is not None:
        return int(_sample_from_logits(coarse_logits[0, t_idx].float(), greedy=True).item())
    axis_logits = outputs["coarse_axis_logits"]
    z_id = _sample_from_logits(axis_logits["z"][0, t_idx].float(), greedy=True)
    y_id = _sample_from_logits(axis_logits["y"][0, t_idx].float(), greedy=True)
    x_id = _sample_from_logits(axis_logits["x"][0, t_idx].float(), greedy=True)
    return int(model._flatten_coarse_axis_ids(z_id, y_id, x_id).item())


def _greedy_trace_cached(model: AutoregFiberModel, batch: dict, *, max_steps: int) -> dict:
    """Mirror the greedy loop of ``infer_autoreg_fiber`` but using the KV-cache."""

    device = next(model.parameters()).device
    volume = batch["volume"].to(device)
    prompt_tokens = {key: value.to(device) for key, value in batch["prompt_tokens"].items()}
    prompt_anchor_xyz = batch["prompt_anchor_xyz"].to(device)
    prompt_anchor_valid = batch.get("prompt_anchor_valid", torch.ones((volume.shape[0],), dtype=torch.bool, device=device))
    if prompt_anchor_valid.ndim == 0:
        prompt_anchor_valid = prompt_anchor_valid.view(1)
    start_position = int(prompt_tokens["positions"][0, -1].item()) + 1

    encoded = model.encode_conditioning(volume)
    memory_tokens = encoded["memory_tokens"]
    memory_patch_centers = encoded["memory_patch_centers"]

    outputs, cache = model.init_kv_cache(
        prompt_tokens=prompt_tokens,
        prompt_anchor_xyz=prompt_anchor_xyz,
        prompt_anchor_valid=prompt_anchor_valid.to(device),
        target_start_position=start_position,
        memory_tokens=memory_tokens,
        memory_patch_centers=memory_patch_centers,
    )

    coarse_logits_per_step: list[torch.Tensor] = []
    offset_logits_per_step: list[torch.Tensor] = []
    stop_logits_per_step: list[torch.Tensor] = []
    coarse_ids_per_step: list[int] = []
    offset_bins_per_step: list[tuple[int, int, int]] = []
    xyz_per_step: list[torch.Tensor] = []

    for step_idx in range(max_steps):
        # Snapshot of the head logits for this step's hidden slice (shape (1, 1, ...)).
        coarse_logits_per_step.append(
            outputs["coarse_logits"][0, 0].detach().clone()
            if outputs.get("coarse_logits") is not None
            else None
        )
        offset_logits_per_step.append(outputs["offset_logits"][0, 0].detach().clone())
        stop_logits_per_step.append(outputs["stop_logits"][0, 0].detach().clone())

        # Greedy sampling (matches infer.py).
        coarse_id = _coarse_id_from_outputs(model, outputs)
        offset_bins = []
        for axis, bins in enumerate(model.offset_num_bins):
            axis_logits = outputs["offset_logits"][0, 0, axis, :bins].float()
            offset_bins.append(int(_sample_from_logits(axis_logits, greedy=True).item()))
        coarse_ids_per_step.append(coarse_id)
        offset_bins_per_step.append(tuple(offset_bins))

        coarse_tensor = torch.tensor([[coarse_id]], dtype=torch.long, device=device)
        offset_tensor = torch.tensor([offset_bins], dtype=torch.long, device=device).view(1, 1, 3)
        bin_center_xyz = model.decode_local_xyz(coarse_tensor, offset_tensor)
        xyz_per_step.append(bin_center_xyz[0, 0].detach().clone())

        if step_idx + 1 >= max_steps:
            break
        outputs, cache = model.step_from_encoded_cached(
            next_coarse_ids=coarse_tensor,
            next_offset_bins=offset_tensor,
            next_xyz=bin_center_xyz,
            next_position=torch.tensor([[start_position + step_idx + 1]], dtype=torch.long, device=device),
            cache=cache,
            memory_tokens=memory_tokens,
        )

    return {
        "coarse_ids": coarse_ids_per_step,
        "offset_bins": offset_bins_per_step,
        "xyz": torch.stack(xyz_per_step, dim=0).cpu(),
        "coarse_logits": coarse_logits_per_step,
        "offset_logits": torch.stack(offset_logits_per_step, dim=0).cpu(),
        "stop_logits": torch.stack(stop_logits_per_step, dim=0).cpu(),
        "final_cache": cache,
    }


def _greedy_trace_naive(model: AutoregFiberModel, batch: dict, *, max_steps: int) -> dict:
    """Run the existing non-cached greedy loop and capture per-step head logits."""

    coarse_logits_per_step: list[torch.Tensor | None] = []
    offset_logits_per_step: list[torch.Tensor] = []
    stop_logits_per_step: list[torch.Tensor] = []
    coarse_ids_per_step: list[int] = []
    offset_bins_per_step: list[tuple[int, int, int]] = []
    xyz_per_step: list[torch.Tensor] = []

    device = next(model.parameters()).device
    volume = batch["volume"].to(device)
    prompt_tokens = {key: value.to(device) for key, value in batch["prompt_tokens"].items()}
    prompt_anchor_xyz = batch["prompt_anchor_xyz"].to(device)
    start_position = int(prompt_tokens["positions"][0, -1].item()) + 1
    all_positions = torch.arange(start_position, start_position + max_steps, device=device, dtype=torch.long).view(1, -1)

    encoded = model.encode_conditioning(volume)
    memory_tokens = encoded["memory_tokens"]
    memory_patch_centers = encoded["memory_patch_centers"]

    buf_coarse_ids = torch.full((1, max_steps), IGNORE_INDEX, dtype=torch.long, device=device)
    buf_offset_bins = torch.full((1, max_steps, 3), IGNORE_INDEX, dtype=torch.long, device=device)
    buf_xyz = torch.zeros((1, max_steps, 3), dtype=torch.float32, device=device)
    buf_valid = torch.zeros((1, max_steps), dtype=torch.bool, device=device)

    for step_idx in range(max_steps):
        current_len = step_idx + 1
        pseudo_batch = build_pseudo_inference_batch(
            prompt_tokens=prompt_tokens,
            prompt_anchor_xyz=prompt_anchor_xyz,
            target_coarse_ids=buf_coarse_ids[:, :current_len],
            target_offset_bins=buf_offset_bins[:, :current_len],
            target_xyz=buf_xyz[:, :current_len],
            target_positions=all_positions[:, :current_len],
            target_valid_mask=buf_valid[:, :current_len],
        )
        outputs = model.forward_from_encoded(
            pseudo_batch,
            memory_tokens=memory_tokens,
            memory_patch_centers=memory_patch_centers,
        )

        coarse_logits_per_step.append(
            outputs["coarse_logits"][0, step_idx].detach().clone()
            if outputs.get("coarse_logits") is not None
            else None
        )
        offset_logits_per_step.append(outputs["offset_logits"][0, step_idx].detach().clone())
        stop_logits_per_step.append(outputs["stop_logits"][0, step_idx].detach().clone())

        coarse_id = _coarse_id_from_outputs(model, outputs, t_idx=step_idx)
        offset_bins = []
        for axis, bins in enumerate(model.offset_num_bins):
            axis_logits = outputs["offset_logits"][0, step_idx, axis, :bins].float()
            offset_bins.append(int(_sample_from_logits(axis_logits, greedy=True).item()))
        coarse_ids_per_step.append(coarse_id)
        offset_bins_per_step.append(tuple(offset_bins))

        coarse_tensor = torch.tensor([[coarse_id]], dtype=torch.long, device=device)
        offset_tensor = torch.tensor([offset_bins], dtype=torch.long, device=device).view(1, 1, 3)
        bin_center_xyz = model.decode_local_xyz(coarse_tensor, offset_tensor)
        xyz_per_step.append(bin_center_xyz[0, 0].detach().clone())

        # Feed back into the buffer for the next step (mirrors infer.py).
        buf_coarse_ids[0, step_idx] = coarse_id
        buf_offset_bins[0, step_idx] = torch.tensor(offset_bins, dtype=torch.long, device=device)
        buf_xyz[0, step_idx] = bin_center_xyz[0, 0]
        buf_valid[0, step_idx] = True

    return {
        "coarse_ids": coarse_ids_per_step,
        "offset_bins": offset_bins_per_step,
        "xyz": torch.stack(xyz_per_step, dim=0).cpu(),
        "coarse_logits": coarse_logits_per_step,
        "offset_logits": torch.stack(offset_logits_per_step, dim=0).cpu(),
        "stop_logits": torch.stack(stop_logits_per_step, dim=0).cpu(),
    }


def _assert_equivalent(naive: dict, cached: dict, *, logit_tol: float = 1e-4) -> None:
    assert naive["coarse_ids"] == cached["coarse_ids"], (naive["coarse_ids"], cached["coarse_ids"])
    assert naive["offset_bins"] == cached["offset_bins"], (naive["offset_bins"], cached["offset_bins"])
    torch.testing.assert_close(naive["xyz"], cached["xyz"], rtol=0, atol=1e-5)
    torch.testing.assert_close(naive["offset_logits"], cached["offset_logits"], rtol=logit_tol, atol=logit_tol)
    torch.testing.assert_close(naive["stop_logits"], cached["stop_logits"], rtol=logit_tol, atol=logit_tol)
    for step_idx, (nl, cl) in enumerate(zip(naive["coarse_logits"], cached["coarse_logits"], strict=True)):
        if nl is None:
            assert cl is None, f"step {step_idx}: cached produced coarse_logits but naive did not"
            continue
        torch.testing.assert_close(nl, cl, rtol=logit_tol, atol=logit_tol)


def test_kv_cache_matches_naive_joint_pointer(tmp_path) -> None:
    torch.manual_seed(123)
    cfg = _tiny_config(tmp_path, coarse_prediction_mode="joint_pointer")
    cfg, _dataset, batch = _dataset_and_batch(tmp_path, cfg)
    model = AutoregFiberModel(cfg).eval()
    max_steps = int(cfg["target_length"])

    naive = _greedy_trace_naive(model, batch, max_steps=max_steps)
    cached = _greedy_trace_cached(model, batch, max_steps=max_steps)
    _assert_equivalent(naive, cached)


def test_kv_cache_matches_naive_axis_factorized(tmp_path) -> None:
    torch.manual_seed(321)
    cfg = _tiny_config(tmp_path, coarse_prediction_mode="axis_factorized")
    cfg, _dataset, batch = _dataset_and_batch(tmp_path, cfg)
    model = AutoregFiberModel(cfg).eval()
    max_steps = int(cfg["target_length"])

    naive = _greedy_trace_naive(model, batch, max_steps=max_steps)
    cached = _greedy_trace_cached(model, batch, max_steps=max_steps)
    _assert_equivalent(naive, cached)


def test_kv_cache_seq_len_grows_one_per_step(tmp_path) -> None:
    torch.manual_seed(7)
    cfg = _tiny_config(tmp_path, coarse_prediction_mode="axis_factorized")
    cfg, _dataset, batch = _dataset_and_batch(tmp_path, cfg)
    model = AutoregFiberModel(cfg).eval()
    max_steps = int(cfg["target_length"])

    cached = _greedy_trace_cached(model, batch, max_steps=max_steps)
    final_cache: FiberKVCache = cached["final_cache"]
    prompt_len = int(batch["prompt_tokens"]["mask"].shape[1])
    # init primes (prompt + START) -> prompt_len + 1; each step adds 1.
    expected_seq_len = prompt_len + 1 + (max_steps - 1)
    assert final_cache.seq_len == expected_seq_len, (final_cache.seq_len, expected_seq_len)
    for layer_k, layer_v in final_cache.self_attn_kv:
        assert layer_k.shape[2] == expected_seq_len
        assert layer_v.shape[2] == expected_seq_len


def test_init_kv_cache_constructs_pointer_keys_lazily(tmp_path) -> None:
    torch.manual_seed(11)
    cfg = _tiny_config(tmp_path, coarse_prediction_mode="joint_pointer")
    cfg, _dataset, batch = _dataset_and_batch(tmp_path, cfg)
    model = AutoregFiberModel(cfg).eval()
    device = next(model.parameters()).device

    encoded = model.encode_conditioning(batch["volume"].to(device))
    outputs, cache = model.init_kv_cache(
        prompt_tokens={key: value.to(device) for key, value in batch["prompt_tokens"].items()},
        prompt_anchor_xyz=batch["prompt_anchor_xyz"].to(device),
        prompt_anchor_valid=torch.ones((1,), dtype=torch.bool, device=device),
        target_start_position=int(batch["prompt_tokens"]["positions"][0, -1].item()) + 1,
        memory_tokens=encoded["memory_tokens"],
        memory_patch_centers=encoded["memory_patch_centers"],
    )
    assert cache.pointer_key_norm is not None
    assert cache.axis_pointer_keys is None
    assert outputs["coarse_logits"] is not None
    assert outputs["coarse_logits"].shape[1] == 1


def test_step_rejects_wrong_shapes(tmp_path) -> None:
    torch.manual_seed(5)
    cfg = _tiny_config(tmp_path, coarse_prediction_mode="axis_factorized")
    cfg, _dataset, batch = _dataset_and_batch(tmp_path, cfg)
    model = AutoregFiberModel(cfg).eval()
    device = next(model.parameters()).device

    encoded = model.encode_conditioning(batch["volume"].to(device))
    _outputs, cache = model.init_kv_cache(
        prompt_tokens={key: value.to(device) for key, value in batch["prompt_tokens"].items()},
        prompt_anchor_xyz=batch["prompt_anchor_xyz"].to(device),
        prompt_anchor_valid=torch.ones((1,), dtype=torch.bool, device=device),
        target_start_position=int(batch["prompt_tokens"]["positions"][0, -1].item()) + 1,
        memory_tokens=encoded["memory_tokens"],
        memory_patch_centers=encoded["memory_patch_centers"],
    )
    bad_coarse = torch.zeros((1, 2), dtype=torch.long, device=device)
    bad_offset = torch.zeros((1, 1, 3), dtype=torch.long, device=device)
    bad_xyz = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
    pos = torch.zeros((1, 1), dtype=torch.long, device=device)
    try:
        model.step_from_encoded_cached(
            next_coarse_ids=bad_coarse,
            next_offset_bins=bad_offset,
            next_xyz=bad_xyz,
            next_position=pos,
            cache=cache,
            memory_tokens=encoded["memory_tokens"],
        )
    except ValueError as exc:
        assert "next_coarse_ids" in str(exc)
    else:
        raise AssertionError("expected ValueError for wrong-shape next_coarse_ids")
