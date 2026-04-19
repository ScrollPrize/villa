from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from vesuvius.neural_tracing.autoreg_mesh.dataset import autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.model import build_pseudo_inference_batch
from vesuvius.neural_tracing.autoreg_mesh.serialization import (
    IGNORE_INDEX,
    deserialize_continuation_grid,
    deserialize_full_grid,
)
from vesuvius.tifxyz import Tifxyz, write_tifxyz


def _to_single_batch(sample_or_batch: dict) -> dict:
    volume = sample_or_batch.get("volume")
    if torch.is_tensor(volume) and volume.ndim == 4:
        return autoreg_mesh_collate([sample_or_batch])
    if torch.is_tensor(volume) and volume.ndim == 5:
        return sample_or_batch
    raise ValueError("infer_autoreg_mesh expects either a single sample dict or a collated batch dict")


def _build_target_strip_coords(direction: str, grid_shape: tuple[int, int], *, device: torch.device) -> Tensor:
    h, w = int(grid_shape[0]), int(grid_shape[1])
    coords: list[tuple[float, float]] = []
    if direction in {"left", "right"}:
        strips = list(range(w)) if direction == "left" else list(range(w - 1, -1, -1))
        strip_den = max(w - 1, 1)
        within_den = max(h - 1, 1)
        for strip_idx, _col_idx in enumerate(strips):
            for within_idx in range(h):
                coords.append((float(strip_idx) / float(strip_den), float(within_idx) / float(within_den)))
    else:
        strips = list(range(h)) if direction == "up" else list(range(h - 1, -1, -1))
        strip_den = max(h - 1, 1)
        within_den = max(w - 1, 1)
        for strip_idx, _row_idx in enumerate(strips):
            for within_idx in range(w):
                coords.append((float(strip_idx) / float(strip_den), float(within_idx) / float(within_den)))
    return torch.tensor(coords, dtype=torch.float32, device=device)


def _build_target_strip_positions(direction: str, grid_shape: tuple[int, int], *, device: torch.device) -> Tensor:
    h, w = int(grid_shape[0]), int(grid_shape[1])
    positions: list[tuple[int, int]] = []
    if direction in {"left", "right"}:
        strips = list(range(w)) if direction == "left" else list(range(w - 1, -1, -1))
        for strip_idx, _col_idx in enumerate(strips):
            for within_idx in range(h):
                positions.append((strip_idx, within_idx))
    else:
        strips = list(range(h)) if direction == "up" else list(range(h - 1, -1, -1))
        for strip_idx, _row_idx in enumerate(strips):
            for within_idx in range(w):
                positions.append((strip_idx, within_idx))
    return torch.tensor(positions, dtype=torch.long, device=device)


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


def _pad_predicted_xyz(pred_xyz: np.ndarray, *, total_vertices: int) -> np.ndarray:
    if pred_xyz.shape[0] >= total_vertices:
        return pred_xyz[:total_vertices]
    padded = np.full((total_vertices, 3), np.nan, dtype=np.float32)
    padded[: pred_xyz.shape[0]] = pred_xyz
    return padded


def _build_tifxyz_from_grid(full_grid_world: np.ndarray, *, uuid: str) -> Tifxyz:
    valid = np.isfinite(full_grid_world).all(axis=-1)
    safe = np.where(valid[..., None], full_grid_world, -1.0).astype(np.float32)
    bbox = None
    if valid.any():
        valid_xyz = safe[valid]
        bbox = (
            float(valid_xyz[:, 2].min()),
            float(valid_xyz[:, 1].min()),
            float(valid_xyz[:, 0].min()),
            float(valid_xyz[:, 2].max()),
            float(valid_xyz[:, 1].max()),
            float(valid_xyz[:, 0].max()),
        )
    return Tifxyz(
        _x=safe[..., 2],
        _y=safe[..., 1],
        _z=safe[..., 0],
        uuid=uuid,
        _scale=(1.0, 1.0),
        bbox=bbox,
        _mask=valid,
        resolution="stored",
    )


@torch.no_grad()
def infer_autoreg_mesh(
    model,
    sample_or_batch: dict,
    *,
    max_steps: int | None = None,
    stop_probability_threshold: float | None = 0.5,
    greedy: bool = True,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    save_path: str | Path | None = None,
    uuid: str | None = None,
) -> dict:
    was_training = bool(model.training)
    model.eval()
    try:
        batch = _to_single_batch(sample_or_batch)
        if int(batch["volume"].shape[0]) != 1:
            raise ValueError("infer_autoreg_mesh currently expects a batch size of 1")

        device = next(model.parameters()).device
        volume = batch["volume"].to(device)
        vol_tokens = batch.get("vol_tokens")
        if vol_tokens is not None:
            vol_tokens = vol_tokens.to(device)
        prompt_tokens = {
            key: value.to(device)
            for key, value in batch["prompt_tokens"].items()
        }
        prompt_anchor_xyz = batch["prompt_anchor_xyz"].to(device)
        direction_id = batch["direction_id"].to(device)
        direction = batch["direction"][0]
        target_grid_shape = tuple(int(v) for v in batch["target_grid_shape"][0].tolist())
        total_vertices = int(target_grid_shape[0] * target_grid_shape[1])
        if max_steps is None:
            max_steps = total_vertices
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive or None")
        max_steps = min(max_steps, total_vertices)

        encoded = model.encode_conditioning(volume, vol_tokens=vol_tokens)
        memory_tokens = encoded["memory_tokens"]
        all_target_strip_coords = _build_target_strip_coords(direction, target_grid_shape, device=device)
        all_target_strip_positions = _build_target_strip_positions(direction, target_grid_shape, device=device)

        buf_coarse_ids = torch.full((1, max_steps), IGNORE_INDEX, dtype=torch.long, device=device)
        buf_offset_bins = torch.full((1, max_steps, 3), IGNORE_INDEX, dtype=torch.long, device=device)
        buf_xyz = torch.zeros((1, max_steps, 3), dtype=torch.float32, device=device)
        out_coarse = np.empty(max_steps, dtype=np.int64)
        out_coarse_axes: dict[str, np.ndarray] = {
            "z": np.empty(max_steps, dtype=np.int64),
            "y": np.empty(max_steps, dtype=np.int64),
            "x": np.empty(max_steps, dtype=np.int64),
        }
        out_offsets = np.empty((max_steps, 3), dtype=np.int64)
        out_xyz = np.empty((max_steps, 3), dtype=np.float32)
        out_bin_center_xyz = np.empty((max_steps, 3), dtype=np.float32)
        out_stop_probs = np.empty(max_steps, dtype=np.float32)
        actual_steps = 0

        for step_idx in range(max_steps):
            current_len = step_idx + 1
            target_strip_coords = all_target_strip_coords[:current_len].unsqueeze(0)

            pseudo_batch = build_pseudo_inference_batch(
                prompt_tokens=prompt_tokens,
                prompt_anchor_xyz=prompt_anchor_xyz,
                direction_id=direction_id,
                direction=[direction],
                conditioning_grid_local=[batch["conditioning_grid_local"][0].to(device)],
                strip_length=batch["strip_length"].to(device),
                num_strips=batch["num_strips"].to(device),
                target_coarse_ids=buf_coarse_ids[:, :current_len],
                target_offset_bins=buf_offset_bins[:, :current_len],
                target_xyz=buf_xyz[:, :current_len],
                target_strip_positions=all_target_strip_positions[:current_len].unsqueeze(0),
                target_strip_coords=target_strip_coords,
            )
            outputs = model.forward_from_encoded(
                pseudo_batch,
                memory_tokens=memory_tokens,
                memory_patch_centers=encoded["memory_patch_centers"],
            )

            sampling_kwargs = dict(greedy=greedy, temperature=temperature, top_k=top_k, top_p=top_p)
            if str(outputs.get("coarse_prediction_mode", getattr(model, "coarse_prediction_mode", "joint_pointer"))) == "axis_factorized":
                coarse_axis_ids = {}
                for axis_name in ("z", "y", "x"):
                    axis_logits = outputs["coarse_axis_logits"][axis_name][0, current_len - 1]
                    coarse_axis_ids[axis_name] = int(_sample_from_logits(axis_logits, **sampling_kwargs).item())
                coarse_id = int(
                    model._flatten_coarse_axis_ids(
                        torch.tensor(coarse_axis_ids["z"], dtype=torch.long, device=device),
                        torch.tensor(coarse_axis_ids["y"], dtype=torch.long, device=device),
                        torch.tensor(coarse_axis_ids["x"], dtype=torch.long, device=device),
                    ).item()
                )
            else:
                coarse_logits = outputs["coarse_logits"][0, current_len - 1]
                coarse_id = int(_sample_from_logits(coarse_logits, **sampling_kwargs).item())
                coarse_axis_ids = {
                    "z": int(outputs["pred_coarse_axis_ids"]["z"][0, current_len - 1].item()),
                    "y": int(outputs["pred_coarse_axis_ids"]["y"][0, current_len - 1].item()),
                    "x": int(outputs["pred_coarse_axis_ids"]["x"][0, current_len - 1].item()),
                }
            offset_bins_list = []
            for axis, bins in enumerate(model.offset_num_bins):
                axis_logits = outputs["offset_logits"][0, current_len - 1, axis, :bins]
                offset_bins_list.append(int(_sample_from_logits(axis_logits, **sampling_kwargs).item()))
            offset_tensor = torch.tensor(offset_bins_list, dtype=torch.long, device=device).view(1, 1, 3)
            coarse_tensor = torch.tensor([[coarse_id]], dtype=torch.long, device=device)
            bin_center_xyz = model.decode_local_xyz(coarse_tensor, offset_tensor)[0, 0].detach().cpu().numpy()
            refine_residual = outputs.get("pred_refine_residual")
            if refine_residual is not None:
                sampled_xyz = bin_center_xyz + refine_residual[0, current_len - 1].detach().cpu().numpy()
            else:
                sampled_xyz = bin_center_xyz
            stop_prob = float(torch.sigmoid(outputs["stop_logits"][0, current_len - 1]).item())

            buf_coarse_ids[0, step_idx] = coarse_id
            buf_offset_bins[0, step_idx] = torch.tensor(offset_bins_list, dtype=torch.long, device=device)
            buf_xyz[0, step_idx] = torch.tensor(sampled_xyz, dtype=torch.float32, device=device)
            out_coarse[step_idx] = coarse_id
            for axis_name in ("z", "y", "x"):
                out_coarse_axes[axis_name][step_idx] = coarse_axis_ids[axis_name]
            out_offsets[step_idx] = offset_bins_list
            out_xyz[step_idx] = sampled_xyz.astype(np.float32, copy=False)
            out_bin_center_xyz[step_idx] = bin_center_xyz.astype(np.float32, copy=False)
            out_stop_probs[step_idx] = stop_prob
            actual_steps = step_idx + 1

            if stop_probability_threshold is not None and stop_prob >= float(stop_probability_threshold):
                break

        predicted_xyz_local = out_xyz[:actual_steps].copy()
        continuation_grid_local = deserialize_continuation_grid(
            _pad_predicted_xyz(predicted_xyz_local, total_vertices=total_vertices),
            direction=direction,
            grid_shape=target_grid_shape,
        )
        conditioning_grid_local = batch["conditioning_grid_local"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        full_grid_local = deserialize_full_grid(conditioning_grid_local, continuation_grid_local, direction=direction)

        min_corner = batch["min_corner"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        predicted_xyz_world = predicted_xyz_local + min_corner[None, :]
        predicted_bin_center_xyz_local = out_bin_center_xyz[:actual_steps].copy()
        predicted_bin_center_xyz_world = predicted_bin_center_xyz_local + min_corner[None, :]
        continuation_grid_world = continuation_grid_local.copy()
        full_grid_world = full_grid_local.copy()
        finite_cont = np.isfinite(continuation_grid_world).all(axis=-1)
        continuation_grid_world[finite_cont] += min_corner
        finite_full = np.isfinite(full_grid_world).all(axis=-1)
        full_grid_world[finite_full] += min_corner

        tifxyz_path = None
        if save_path is not None:
            segment_uuid = ""
            wrap_metadata = batch["wrap_metadata"][0]
            if isinstance(wrap_metadata, dict):
                segment_uuid = str(wrap_metadata.get("segment_uuid", ""))
            surface_uuid = uuid or (segment_uuid or "autoreg_mesh_prediction")
            surface = _build_tifxyz_from_grid(full_grid_world, uuid=surface_uuid)
            tifxyz_path = write_tifxyz(Path(save_path) / surface_uuid, surface, overwrite=True)

        result = {
            "predicted_coarse_ids": out_coarse[:actual_steps].copy(),
            "predicted_offset_bins": out_offsets[:actual_steps].copy(),
            "predicted_continuation_vertices_local": predicted_xyz_local,
            "predicted_continuation_vertices_world": predicted_xyz_world,
            "predicted_bin_center_vertices_local": predicted_bin_center_xyz_local,
            "predicted_bin_center_vertices_world": predicted_bin_center_xyz_world,
            "continuation_grid_local": continuation_grid_local,
            "continuation_grid_world": continuation_grid_world,
            "full_grid_local": full_grid_local,
            "full_grid_world": full_grid_world,
            "stop_probabilities": out_stop_probs[:actual_steps].copy(),
            "saved_tifxyz_path": None if tifxyz_path is None else str(tifxyz_path),
        }
        if str(getattr(model, "coarse_prediction_mode", "joint_pointer")) == "axis_factorized":
            result["predicted_coarse_axis_ids"] = {
                axis_name: out_coarse_axes[axis_name][:actual_steps].copy()
                for axis_name in ("z", "y", "x")
            }
        return result
    finally:
        model.train(was_training)
