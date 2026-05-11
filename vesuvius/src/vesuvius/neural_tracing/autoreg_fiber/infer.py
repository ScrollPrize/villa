from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from vesuvius.neural_tracing.autoreg_fiber.dataset import autoreg_fiber_collate
from vesuvius.neural_tracing.autoreg_fiber.model import build_pseudo_inference_batch
from vesuvius.neural_tracing.autoreg_fiber.serialization import IGNORE_INDEX


def _to_single_batch(sample_or_batch: dict) -> dict:
    volume = sample_or_batch.get("volume")
    if torch.is_tensor(volume) and volume.ndim == 4:
        return autoreg_fiber_collate([sample_or_batch])
    if torch.is_tensor(volume) and volume.ndim == 5:
        return sample_or_batch
    raise ValueError("infer_autoreg_fiber expects either a single sample dict or a collated batch dict")


def _sample_from_logits(
    logits: Tensor,
    *,
    greedy: bool,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Tensor:
    if greedy:
        return logits.argmax(dim=-1)
    scaled = logits / max(float(temperature), 1e-8)
    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), int(scaled.shape[-1]))
        topk_vals, _ = torch.topk(scaled, k, dim=-1)
        threshold = topk_vals[..., -1:]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        remove_mask = (cumulative_probs - sorted_probs) >= float(top_p)
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        scaled = torch.zeros_like(scaled).scatter(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _write_csv(path: Path, points_local: np.ndarray, points_world: np.ndarray, stop_probs: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "z_local", "y_local", "x_local", "z_world", "y_world", "x_world", "stop_probability"])
        for step, (local, world, stop_prob) in enumerate(zip(points_local, points_world, stop_probs, strict=True)):
            writer.writerow([
                int(step),
                float(local[0]),
                float(local[1]),
                float(local[2]),
                float(world[0]),
                float(world[1]),
                float(world[2]),
                float(stop_prob),
            ])


def _write_nml(path: Path, points_world_zyx: np.ndarray) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<things>",
        '  <thing id="1" name="autoreg_fiber_prediction" color.r="1" color.g="0.6" color.b="0" color.a="1">',
        "    <nodes>",
    ]
    for idx, point_zyx in enumerate(points_world_zyx, start=1):
        z, y, x = [float(v) for v in point_zyx]
        lines.append(f'      <node id="{idx}" x="{x:.6f}" y="{y:.6f}" z="{z:.6f}" radius="1" />')
    lines.append("    </nodes>")
    lines.append("    <edges>")
    for idx in range(1, int(points_world_zyx.shape[0])):
        lines.append(f'      <edge source="{idx}" target="{idx + 1}" />')
    lines.append("    </edges>")
    lines.append("  </thing>")
    lines.append("</things>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def infer_autoreg_fiber(
    model,
    sample_or_batch: dict,
    *,
    max_steps: int | None = None,
    stop_probability_threshold: float | None = 0.5,
    min_steps: int = 1,
    greedy: bool = True,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    save_path: str | Path | None = None,
) -> dict:
    was_training = bool(model.training)
    model.eval()
    try:
        batch = _to_single_batch(sample_or_batch)
        if int(batch["volume"].shape[0]) != 1:
            raise ValueError("infer_autoreg_fiber currently expects batch size 1")

        device = next(model.parameters()).device
        volume = batch["volume"].to(device)
        vol_tokens = batch.get("vol_tokens")
        if vol_tokens is not None:
            vol_tokens = vol_tokens.to(device)
        prompt_tokens = {key: value.to(device) for key, value in batch["prompt_tokens"].items()}
        prompt_anchor_xyz = batch["prompt_anchor_xyz"].to(device)
        target_positions_source = batch.get("target_positions")
        if target_positions_source is not None:
            target_positions_source = target_positions_source.to(device)

        if max_steps is None:
            if "target_lengths" in batch:
                max_steps = int(batch["target_lengths"][0].item())
            else:
                max_steps = int(getattr(model, "config", {}).get("target_length", 32))
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive or None")

        encoded = model.encode_conditioning(volume, vol_tokens=vol_tokens)
        memory_tokens = encoded["memory_tokens"]
        memory_patch_centers = encoded["memory_patch_centers"]

        buf_coarse_ids = torch.full((1, max_steps), IGNORE_INDEX, dtype=torch.long, device=device)
        buf_offset_bins = torch.full((1, max_steps, 3), IGNORE_INDEX, dtype=torch.long, device=device)
        buf_xyz = torch.zeros((1, max_steps, 3), dtype=torch.float32, device=device)
        buf_valid = torch.zeros((1, max_steps), dtype=torch.bool, device=device)
        if target_positions_source is not None:
            all_positions = target_positions_source[:, :max_steps]
            if int(all_positions.shape[1]) < max_steps:
                last = int(all_positions[0, -1].item()) if int(all_positions.shape[1]) > 0 else int(prompt_tokens["positions"][0, -1].item())
                extra = torch.arange(last + 1, last + 1 + (max_steps - int(all_positions.shape[1])), device=device, dtype=torch.long).view(1, -1)
                all_positions = torch.cat([all_positions, extra], dim=1)
        else:
            start_position = int(prompt_tokens["positions"][0, -1].item()) + 1
            all_positions = torch.arange(start_position, start_position + max_steps, device=device, dtype=torch.long).view(1, -1)

        out_coarse = np.empty(max_steps, dtype=np.int64)
        out_offsets = np.empty((max_steps, 3), dtype=np.int64)
        out_xyz = np.empty((max_steps, 3), dtype=np.float32)
        out_bin_center_xyz = np.empty((max_steps, 3), dtype=np.float32)
        out_stop_probs = np.empty(max_steps, dtype=np.float32)
        actual_steps = 0

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
            sampling_kwargs = dict(greedy=greedy, temperature=temperature, top_k=top_k, top_p=top_p)
            coarse_logits = outputs.get("coarse_logits")
            if coarse_logits is not None:
                coarse_step_logits = coarse_logits[0, current_len - 1].float()
                coarse_id = int(_sample_from_logits(coarse_step_logits, **sampling_kwargs).item())
            else:
                axis_logits = outputs.get("coarse_axis_logits")
                if axis_logits is None:
                    raise ValueError("model output must include coarse_logits or coarse_axis_logits")
                z_id = _sample_from_logits(axis_logits["z"][0, current_len - 1].float(), **sampling_kwargs)
                y_id = _sample_from_logits(axis_logits["y"][0, current_len - 1].float(), **sampling_kwargs)
                x_id = _sample_from_logits(axis_logits["x"][0, current_len - 1].float(), **sampling_kwargs)
                coarse_id = int(model._flatten_coarse_axis_ids(z_id, y_id, x_id).item())
            offset_bins_list = []
            for axis, bins in enumerate(model.offset_num_bins):
                axis_logits = outputs["offset_logits"][0, current_len - 1, axis, :bins].float()
                offset_bins_list.append(int(_sample_from_logits(axis_logits, **sampling_kwargs).item()))
            offset_tensor = torch.tensor(offset_bins_list, dtype=torch.long, device=device).view(1, 1, 3)
            coarse_tensor = torch.tensor([[coarse_id]], dtype=torch.long, device=device)
            bin_center_xyz = model.decode_local_xyz(coarse_tensor, offset_tensor)[0, 0].detach().to(torch.float32).cpu().numpy()
            refine_residual = outputs.get("pred_refine_residual")
            if refine_residual is not None:
                res = refine_residual[0, current_len - 1].detach().to(torch.float32).cpu().numpy()
                patch_diag = float(np.linalg.norm(model.patch_size))
                res_norm = float(np.linalg.norm(res))
                if res_norm > patch_diag:
                    res = res * (patch_diag / res_norm)
                sampled_xyz = bin_center_xyz + res
            else:
                sampled_xyz = bin_center_xyz
            crop_max = np.array(model.input_shape, dtype=np.float32) - 1e-4
            sampled_xyz = np.clip(sampled_xyz, 0.0, crop_max)
            stop_prob = float(torch.sigmoid(outputs["stop_logits"][0, current_len - 1].float()).item())

            buf_coarse_ids[0, step_idx] = coarse_id
            buf_offset_bins[0, step_idx] = torch.tensor(offset_bins_list, dtype=torch.long, device=device)
            buf_xyz[0, step_idx] = torch.tensor(sampled_xyz, dtype=torch.float32, device=device)
            buf_valid[0, step_idx] = True
            out_coarse[step_idx] = coarse_id
            out_offsets[step_idx] = offset_bins_list
            out_xyz[step_idx] = sampled_xyz.astype(np.float32, copy=False)
            out_bin_center_xyz[step_idx] = bin_center_xyz.astype(np.float32, copy=False)
            out_stop_probs[step_idx] = stop_prob
            actual_steps = step_idx + 1

            can_stop = (
                stop_probability_threshold is not None
                and actual_steps >= int(min_steps)
                and stop_prob >= float(stop_probability_threshold)
            )
            if can_stop:
                break

        predicted_xyz_local = out_xyz[:actual_steps].copy()
        min_corner = batch.get("min_corner")
        if min_corner is None:
            min_corner_np = np.zeros((3,), dtype=np.float32)
        else:
            min_corner_np = min_corner[0].detach().cpu().numpy().astype(np.float32, copy=False)
        predicted_xyz_world = predicted_xyz_local + min_corner_np[None, :]
        predicted_bin_center_xyz_local = out_bin_center_xyz[:actual_steps].copy()
        predicted_bin_center_xyz_world = predicted_bin_center_xyz_local + min_corner_np[None, :]
        stop_probs = out_stop_probs[:actual_steps].copy()

        saved_paths: dict[str, str] = {}
        if save_path is not None:
            out_dir = Path(save_path).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_dir / "fiber_prediction.npz"
            csv_path = out_dir / "fiber_prediction.csv"
            nml_path = out_dir / "fiber_prediction.nml"
            np.savez_compressed(
                npz_path,
                predicted_fiber_local_zyx=predicted_xyz_local,
                predicted_fiber_world_zyx=predicted_xyz_world,
                predicted_bin_center_local_zyx=predicted_bin_center_xyz_local,
                predicted_bin_center_world_zyx=predicted_bin_center_xyz_world,
                predicted_coarse_ids=out_coarse[:actual_steps].copy(),
                predicted_offset_bins=out_offsets[:actual_steps].copy(),
                stop_probabilities=stop_probs,
            )
            _write_csv(csv_path, predicted_xyz_local, predicted_xyz_world, stop_probs)
            _write_nml(nml_path, predicted_xyz_world)
            saved_paths = {"npz": str(npz_path), "csv": str(csv_path), "nml": str(nml_path)}

        return {
            "predicted_coarse_ids": out_coarse[:actual_steps].copy(),
            "predicted_offset_bins": out_offsets[:actual_steps].copy(),
            "predicted_fiber_local_zyx": predicted_xyz_local,
            "predicted_fiber_world_zyx": predicted_xyz_world,
            "predicted_bin_center_local_zyx": predicted_bin_center_xyz_local,
            "predicted_bin_center_world_zyx": predicted_bin_center_xyz_world,
            "stop_probabilities": stop_probs,
            "saved_paths": saved_paths,
        }
    finally:
        model.train(was_training)


__all__ = ["infer_autoreg_fiber"]
