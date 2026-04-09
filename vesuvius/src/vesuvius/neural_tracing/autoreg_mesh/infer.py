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


def _sample_from_logits(logits: Tensor, *, greedy: bool) -> Tensor:
    if greedy:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits, dim=-1)
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
    save_path: str | Path | None = None,
    uuid: str | None = None,
) -> dict:
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
    max_steps = min(int(max_steps), total_vertices)

    encoded = model.encode_conditioning(volume, vol_tokens=vol_tokens)
    memory_tokens = encoded["memory_tokens"]
    all_target_strip_coords = _build_target_strip_coords(direction, target_grid_shape, device=device)

    generated_coarse: list[int] = []
    generated_offsets: list[list[int]] = []
    generated_xyz: list[np.ndarray] = []
    generated_bin_center_xyz: list[np.ndarray] = []
    stop_probabilities: list[float] = []

    for step_idx in range(max_steps):
        current_len = step_idx + 1
        target_strip_coords = all_target_strip_coords[:current_len].unsqueeze(0)
        target_coarse_ids = torch.full((1, current_len), IGNORE_INDEX, dtype=torch.long, device=device)
        target_offset_bins = torch.full((1, current_len, 3), IGNORE_INDEX, dtype=torch.long, device=device)
        target_xyz = torch.zeros((1, current_len, 3), dtype=torch.float32, device=device)
        if generated_coarse:
            history_len = len(generated_coarse)
            target_coarse_ids[0, :history_len] = torch.tensor(generated_coarse, dtype=torch.long, device=device)
            target_offset_bins[0, :history_len] = torch.tensor(generated_offsets, dtype=torch.long, device=device)
            target_xyz[0, :history_len] = torch.tensor(np.asarray(generated_xyz), dtype=torch.float32, device=device)

        pseudo_batch = build_pseudo_inference_batch(
            prompt_tokens=prompt_tokens,
            prompt_anchor_xyz=prompt_anchor_xyz,
            direction_id=direction_id,
            target_coarse_ids=target_coarse_ids,
            target_offset_bins=target_offset_bins,
            target_xyz=target_xyz,
            target_strip_coords=target_strip_coords,
        )
        outputs = model.forward_from_encoded(pseudo_batch, memory_tokens=memory_tokens)

        coarse_logits = outputs["coarse_logits"][0, current_len - 1]
        coarse_id = int(_sample_from_logits(coarse_logits, greedy=greedy).item())
        offset_bins = []
        for axis, bins in enumerate(model.offset_num_bins):
            axis_logits = outputs["offset_logits"][0, current_len - 1, axis, :bins]
            offset_bins.append(int(_sample_from_logits(axis_logits, greedy=greedy).item()))
        offset_tensor = torch.tensor(offset_bins, dtype=torch.long, device=device).view(1, 1, 3)
        coarse_tensor = torch.tensor([[coarse_id]], dtype=torch.long, device=device)
        bin_center_xyz = model.decode_local_xyz(coarse_tensor, offset_tensor)[0, 0].detach().cpu().numpy()
        xyz = outputs["pred_xyz_refined"][0, current_len - 1].detach().cpu().numpy()
        stop_prob = float(torch.sigmoid(outputs["stop_logits"][0, current_len - 1]).item())

        generated_coarse.append(coarse_id)
        generated_offsets.append(offset_bins)
        generated_xyz.append(xyz.astype(np.float32, copy=False))
        generated_bin_center_xyz.append(bin_center_xyz.astype(np.float32, copy=False))
        stop_probabilities.append(stop_prob)

        if stop_probability_threshold is not None and stop_prob >= float(stop_probability_threshold):
            break

    predicted_xyz_local = np.asarray(generated_xyz, dtype=np.float32)
    continuation_grid_local = deserialize_continuation_grid(
        _pad_predicted_xyz(predicted_xyz_local, total_vertices=total_vertices),
        direction=direction,
        grid_shape=target_grid_shape,
    )
    conditioning_grid_local = batch["conditioning_grid_local"][0].detach().cpu().numpy().astype(np.float32, copy=False)
    full_grid_local = deserialize_full_grid(conditioning_grid_local, continuation_grid_local, direction=direction)

    min_corner = batch["min_corner"][0].detach().cpu().numpy().astype(np.float32, copy=False)
    predicted_xyz_world = predicted_xyz_local + min_corner[None, :]
    predicted_bin_center_xyz_local = np.asarray(generated_bin_center_xyz, dtype=np.float32)
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

    return {
        "predicted_coarse_ids": np.asarray(generated_coarse, dtype=np.int64),
        "predicted_offset_bins": np.asarray(generated_offsets, dtype=np.int64),
        "predicted_continuation_vertices_local": predicted_xyz_local,
        "predicted_continuation_vertices_world": predicted_xyz_world,
        "predicted_bin_center_vertices_local": predicted_bin_center_xyz_local,
        "predicted_bin_center_vertices_world": predicted_bin_center_xyz_world,
        "continuation_grid_local": continuation_grid_local,
        "continuation_grid_world": continuation_grid_world,
        "full_grid_local": full_grid_local,
        "full_grid_world": full_grid_world,
        "stop_probabilities": np.asarray(stop_probabilities, dtype=np.float32),
        "saved_tifxyz_path": None if tifxyz_path is None else str(tifxyz_path),
    }
