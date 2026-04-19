"""Diagnostic: compare fp32 vs bf16 forward pass to identify precision-critical ops.

Usage (CPU, no GPU required):
    python -m vesuvius.neural_tracing.autoreg_mesh.diagnose_bf16

Usage (GPU):
    python -m vesuvius.neural_tracing.autoreg_mesh.diagnose_bf16 --device cuda
"""
from __future__ import annotations

import argparse
import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from vesuvius.neural_tracing.autoreg_mesh.config import validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.dataset import autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.losses import compute_autoreg_mesh_losses
from vesuvius.neural_tracing.autoreg_mesh.model import AutoregMeshModel
from vesuvius.neural_tracing.autoreg_mesh.serialization import serialize_split_conditioning_example

import numpy as np


def _make_surface(rows: int, cols: int, *, z_offset: float, y_offset: float, x_offset: float) -> np.ndarray:
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    col_axis = np.arange(cols, dtype=np.float32)[None, :]
    z = np.full((rows, cols), z_offset, dtype=np.float32) + 0.1 * row_axis + 0.05 * col_axis
    y = np.broadcast_to(y_offset + 2.0 * row_axis, (rows, cols)).astype(np.float32)
    x = np.broadcast_to(x_offset + 2.0 * col_axis, (rows, cols)).astype(np.float32)
    return np.stack([z, y, x], axis=-1)


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


def _make_config(*, decoder_dim: int = 96) -> dict:
    return validate_autoreg_mesh_config({
        "dinov2_backbone": None,
        "input_shape": [16, 16, 16],
        "patch_size": [8, 8, 8],
        "offset_num_bins": [4, 4, 4],
        "decoder_dim": decoder_dim,
        "decoder_depth": 2,
        "decoder_num_heads": 4,
        "cross_attention_every_n_blocks": 1,
        "coarse_prediction_mode": "axis_factorized",
        "pointer_temperature": 0.25,
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
    })


class _InstrumentedModel(AutoregMeshModel):
    """Captures intermediate tensors during forward pass for precision analysis."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._captures: dict[str, Tensor] = {}
        self._disable_fp32_pointer_fix = False

    def _factorized_axis_memory(self, memory_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self._disable_fp32_pointer_fix:
            gz, gy, gx = self.coarse_grid_shape
            memory_5d = memory_tokens.reshape(memory_tokens.shape[0], gz, gy, gx, memory_tokens.shape[-1])
            z_mem = memory_5d.mean(dim=(2, 3))
            y_mem = memory_5d.mean(dim=(1, 3))
            x_mem = memory_5d.mean(dim=(1, 2))
        else:
            z_mem, y_mem, x_mem = super()._factorized_axis_memory(memory_tokens)
        self._captures["axis_mem_z"] = z_mem.detach().clone()
        self._captures["axis_mem_y"] = y_mem.detach().clone()
        self._captures["axis_mem_x"] = x_mem.detach().clone()
        return z_mem, y_mem, x_mem

    def _compute_coarse_outputs(self, hidden, memory_tokens, **kwargs):
        self._captures["hidden_pre_pointer"] = hidden.detach().clone()
        self._captures["memory_tokens"] = memory_tokens.detach().clone()
        if self._disable_fp32_pointer_fix:
            result = self._compute_coarse_outputs_raw_bf16(hidden, memory_tokens, **kwargs)
        else:
            result = super()._compute_coarse_outputs(hidden, memory_tokens, **kwargs)
        if result.get("coarse_axis_logits") is not None:
            for axis in ("z", "y", "x"):
                self._captures[f"axis_logits_{axis}"] = result["coarse_axis_logits"][axis].detach().clone()
        if result.get("coarse_logits") is not None:
            self._captures["coarse_logits"] = result["coarse_logits"].detach().clone()
        self._captures["pred_coarse_ids"] = result["pred_coarse_ids"].detach().clone()
        return result

    def _compute_coarse_outputs_raw_bf16(self, hidden, memory_tokens, *, coarse_valid_mask=None, coarse_constraint_metrics=None):
        """Original pointer attention WITHOUT the fp32 fix — for comparison."""
        zeros = hidden.new_zeros(())
        coarse_constraint_metrics = coarse_constraint_metrics or {
            "coarse_constraint_keep_fraction": zeros,
            "coarse_constraint_empty_rate": zeros,
            "coarse_constraint_target_outside_rate": zeros,
        }
        z_memory, y_memory, x_memory = self._factorized_axis_memory(memory_tokens)
        z_query = F.normalize(self.pointer_query_z(hidden), dim=-1)
        y_query = F.normalize(self.pointer_query_y(hidden), dim=-1)
        x_query = F.normalize(self.pointer_query_x(hidden), dim=-1)
        z_key = F.normalize(self.pointer_key_z(z_memory), dim=-1)
        y_key = F.normalize(self.pointer_key_y(y_memory), dim=-1)
        x_key = F.normalize(self.pointer_key_x(x_memory), dim=-1)
        z_logits = torch.einsum("btd,bzd->btz", z_query, z_key) / self.pointer_temperature
        y_logits = torch.einsum("btd,byd->bty", y_query, y_key) / self.pointer_temperature
        x_logits = torch.einsum("btd,bxd->btx", x_query, x_key) / self.pointer_temperature
        if coarse_valid_mask is not None:
            gz, gy, gx = self.coarse_grid_shape
            mask_5d = coarse_valid_mask.reshape(coarse_valid_mask.shape[0], coarse_valid_mask.shape[1], gz, gy, gx)
            z_logits = z_logits.masked_fill(~mask_5d.any(dim=(3, 4)), torch.finfo(z_logits.dtype).min)
            y_logits = y_logits.masked_fill(~mask_5d.any(dim=(2, 4)), torch.finfo(y_logits.dtype).min)
            x_logits = x_logits.masked_fill(~mask_5d.any(dim=(2, 3)), torch.finfo(x_logits.dtype).min)
        gz, gy, gx = self.coarse_grid_shape
        joint = z_logits[..., :, None, None] + y_logits[..., None, :, None] + x_logits[..., None, None, :]
        if coarse_valid_mask is not None:
            joint = joint.masked_fill(~coarse_valid_mask.view(*joint.shape[:2], gz, gy, gx), torch.finfo(joint.dtype).min)
        flat = joint.reshape(*joint.shape[:2], -1).argmax(dim=-1)
        z_ids = flat // (gy * gx)
        rem = flat % (gy * gx)
        y_ids = rem // gx
        x_ids = rem % gx
        return {
            "coarse_logits": None,
            "coarse_axis_logits": {"z": z_logits, "y": y_logits, "x": x_logits},
            "pred_coarse_ids": flat,
            "pred_coarse_axis_ids": {"z": z_ids, "y": y_ids, "x": x_ids},
            "coarse_constraint_joint_valid_mask": coarse_valid_mask,
            "coarse_constraint_axis_valid_masks": None,
            **coarse_constraint_metrics,
        }

    def forward_from_encoded(self, batch, *, memory_tokens, memory_patch_centers, **kwargs):
        result = super().forward_from_encoded(batch, memory_tokens=memory_tokens, memory_patch_centers=memory_patch_centers, **kwargs)
        self._captures["pred_xyz_soft"] = result["pred_xyz_soft"].detach().clone()
        self._captures["pred_xyz_refined"] = result["pred_xyz_refined"].detach().clone()
        self._captures["offset_logits"] = result["offset_logits"].detach().clone()
        self._captures["stop_logits"] = result["stop_logits"].detach().clone()
        return result


def _compare_tensor(name: str, fp32_t: Tensor, bf16_t: Tensor, *, is_logit: bool = False) -> dict:
    fp32_f = fp32_t.float()
    bf16_f = bf16_t.float()
    max_abs_diff = (fp32_f - bf16_f).abs().max().item()
    fp32_max = fp32_f.abs().max().item()
    rel_error = max_abs_diff / max(fp32_max, 1e-8)
    cos_sim = F.cosine_similarity(fp32_f.flatten(), bf16_f.flatten(), dim=0).item() if fp32_f.numel() > 0 else 1.0
    row = {
        "name": name,
        "max_abs_diff": max_abs_diff,
        "rel_error": rel_error,
        "cos_sim": cos_sim,
        "dtype_fp32": str(fp32_t.dtype),
        "dtype_bf16": str(bf16_t.dtype),
    }
    if is_logit and fp32_t.dim() >= 2:
        fp32_argmax = fp32_f.reshape(-1, fp32_f.shape[-1]).argmax(dim=-1)
        bf16_argmax = bf16_f.reshape(-1, bf16_f.shape[-1]).argmax(dim=-1)
        match_pct = (fp32_argmax == bf16_argmax).float().mean().item() * 100
        row["argmax_match_pct"] = match_pct
    return row


def _print_table(rows: list[dict]) -> None:
    name_w = max(len(r["name"]) for r in rows)
    header = f"{'Operation':<{name_w}} | {'max_abs_diff':>12} | {'rel_error':>10} | {'cos_sim':>8} | {'argmax%':>8} | {'fp32_dtype':>12} | {'bf16_dtype':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        argmax = f"{r.get('argmax_match_pct', '-'):>8.1f}" if "argmax_match_pct" in r else f"{'-':>8}"
        print(
            f"{r['name']:<{name_w}} | {r['max_abs_diff']:>12.6f} | {r['rel_error']:>10.6f} | {r['cos_sim']:>8.6f} | {argmax} | {r['dtype_fp32']:>12} | {r['dtype_bf16']:>12}"
        )


def run_diagnostic(device: str = "cpu") -> list[dict]:
    torch.manual_seed(42)
    config = _make_config()
    model = _InstrumentedModel(config)
    model.eval()

    batch = autoreg_mesh_collate([_make_sample_with_cached_tokens("left"), _make_sample_with_cached_tokens("up")])
    batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
    if "prompt_tokens" in batch:
        batch["prompt_tokens"] = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch["prompt_tokens"].items()}

    model = model.to(device)

    # ---- fp32 pass ----
    model._captures.clear()
    with torch.no_grad():
        outputs_fp32 = model(batch)
    caps_fp32 = dict(model._captures)

    # ---- bf16 autocast pass WITH fix ----
    model._captures.clear()
    model._disable_fp32_pointer_fix = False
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        outputs_bf16_fixed = model(batch)
    caps_bf16_fixed = dict(model._captures)

    # ---- bf16 autocast pass WITHOUT fix (original bf16 path) ----
    model._captures.clear()
    model._disable_fp32_pointer_fix = True
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        outputs_bf16_raw = model(batch)
    caps_bf16_raw = dict(model._captures)
    model._disable_fp32_pointer_fix = False

    logit_keys = {"axis_logits_z", "axis_logits_y", "axis_logits_x", "coarse_logits", "offset_logits", "stop_logits"}

    # ---- compare: WITHOUT fix (the broken path) ----
    rows_raw = []
    for key in caps_fp32:
        if key in caps_bf16_raw:
            rows_raw.append(_compare_tensor(key, caps_fp32[key], caps_bf16_raw[key], is_logit=key in logit_keys))
    rows_raw.sort(key=lambda r: r["rel_error"], reverse=True)
    print("\n=== bf16 vs fp32 divergence WITHOUT fp32 pointer fix (BROKEN path) ===\n")
    _print_table(rows_raw)

    # ---- compare: WITH fix ----
    rows_fixed = []
    for key in caps_fp32:
        if key in caps_bf16_fixed:
            rows_fixed.append(_compare_tensor(key, caps_fp32[key], caps_bf16_fixed[key], is_logit=key in logit_keys))
    rows_fixed.sort(key=lambda r: r["rel_error"], reverse=True)
    print("\n=== bf16 vs fp32 divergence WITH fp32 pointer fix (FIXED path) ===\n")
    _print_table(rows_fixed)

    # ---- coarse prediction match ----
    fp32_ids = caps_fp32["pred_coarse_ids"]
    raw_ids = caps_bf16_raw["pred_coarse_ids"]
    fixed_ids = caps_bf16_fixed["pred_coarse_ids"]
    mask = batch.get("target_supervision_mask")
    if mask is not None:
        mask = mask.to(device)
        raw_match = (fp32_ids[mask] == raw_ids[mask]).float().mean().item() * 100
        fixed_match = (fp32_ids[mask] == fixed_ids[mask]).float().mean().item() * 100
    else:
        raw_match = (fp32_ids == raw_ids).float().mean().item() * 100
        fixed_match = (fp32_ids == fixed_ids).float().mean().item() * 100
    print(f"\n=== Coarse prediction match vs fp32 ===")
    print(f"  WITHOUT fix: {raw_match:.1f}% match")
    print(f"  WITH fix:    {fixed_match:.1f}% match")

    # ---- loss comparison ----
    print("\n=== Loss comparison ===\n")
    offset_num_bins = tuple(int(v) for v in config["offset_num_bins"])
    loss_fp32 = compute_autoreg_mesh_losses(outputs_fp32, batch, offset_num_bins=offset_num_bins)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss_bf16_fixed = compute_autoreg_mesh_losses(outputs_bf16_fixed, batch, offset_num_bins=offset_num_bins)
        loss_bf16_raw = compute_autoreg_mesh_losses(outputs_bf16_raw, batch, offset_num_bins=offset_num_bins)

    print(f"  {'loss_component':40s}  {'fp32':>10}  {'bf16_raw':>10}  {'bf16_fixed':>10}")
    print(f"  {'-' * 40}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for key in ["loss", "coarse_loss", "offset_loss", "stop_loss", "xyz_soft_loss"]:
        if key in loss_fp32:
            v32 = loss_fp32[key].float().item()
            vraw = loss_bf16_raw[key].float().item()
            vfix = loss_bf16_fixed[key].float().item()
            print(f"  {key:40s}  {v32:10.4f}  {vraw:10.4f}  {vfix:10.4f}")

    return rows_fixed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose bf16 precision issues in autoreg_mesh")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    args = parser.parse_args()
    run_diagnostic(device=args.device)
