from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from vesuvius.neural_tracing.autoreg_mesh.dataset import autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.model import (
    GENERATED_TOKEN_TYPE,
    START_TOKEN_TYPE,
    build_pseudo_inference_batch,
)
from vesuvius.neural_tracing.autoreg_mesh.serialization import (
    IGNORE_INDEX,
    deserialize_continuation_grid,
    deserialize_full_grid,
)
from vesuvius.tifxyz import Tifxyz, write_tifxyz


@dataclasses.dataclass
class GeometricValidationConfig:
    """Configuration for per-vertex geometric checks during inference."""

    enabled: bool = True
    oob_action: str = "resample"
    distance_action: str = "resample"
    flip_action: str = "flag"
    max_distance_scale: float = 2.0
    max_resample_attempts: int = 5


def _check_oob(xyz: np.ndarray, input_shape: tuple[int, int, int]) -> bool:
    return bool(np.all(xyz >= 0.0) and all(float(xyz[i]) < float(input_shape[i]) for i in range(3)))


def _check_distance(new_xyz: np.ndarray, prev_xyz: np.ndarray | None, *, max_dist: float) -> bool:
    if prev_xyz is None:
        return True
    return bool(float(np.linalg.norm(new_xyz - prev_xyz)) <= max_dist)


def _validate_vertex(
    xyz: np.ndarray,
    prev_xyz: np.ndarray | None,
    *,
    input_shape: tuple[int, int, int],
    max_dist: float,
    config: GeometricValidationConfig,
) -> tuple[bool, str | None]:
    if not _check_oob(xyz, input_shape):
        return False, "oob"
    if not _check_distance(xyz, prev_xyz, max_dist=max_dist):
        return False, "distance"
    return True, None


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


@torch.inference_mode()
def infer_autoreg_mesh(
    model,
    sample_or_batch: dict,
    *,
    max_steps: int | None = None,
    stop_probability_threshold: float | None = None,
    min_steps: int = 0,
    stop_only_at_strip_boundary: bool = False,
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
                    axis_logits = outputs["coarse_axis_logits"][axis_name][0, current_len - 1].float()
                    coarse_axis_ids[axis_name] = int(_sample_from_logits(axis_logits, **sampling_kwargs).item())
                coarse_id = int(
                    model._flatten_coarse_axis_ids(
                        torch.tensor([[coarse_axis_ids["z"]]], dtype=torch.long, device=device),
                        torch.tensor([[coarse_axis_ids["y"]]], dtype=torch.long, device=device),
                        torch.tensor([[coarse_axis_ids["x"]]], dtype=torch.long, device=device),
                    ).item()
                )
            else:
                coarse_logits = outputs["coarse_logits"][0, current_len - 1].float()
                coarse_id = int(_sample_from_logits(coarse_logits, **sampling_kwargs).item())
                coarse_axis_ids = {
                    "z": int(outputs["pred_coarse_axis_ids"]["z"][0, current_len - 1].item()),
                    "y": int(outputs["pred_coarse_axis_ids"]["y"][0, current_len - 1].item()),
                    "x": int(outputs["pred_coarse_axis_ids"]["x"][0, current_len - 1].item()),
                }
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
                if hasattr(model, "patch_size"):
                    patch_diag = float(np.linalg.norm(model.patch_size))
                    res_norm = float(np.linalg.norm(res))
                    if res_norm > patch_diag:
                        res = res * (patch_diag / res_norm)
                sampled_xyz = bin_center_xyz + res
            else:
                sampled_xyz = bin_center_xyz
            if hasattr(model, "input_shape"):
                crop_max = np.array(model.input_shape, dtype=np.float32) - 1e-4
                sampled_xyz = np.clip(sampled_xyz, 0.0, crop_max)
            stop_prob = float(torch.sigmoid(outputs["stop_logits"][0, current_len - 1].float()).item())

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

            strip_len = int(batch["strip_length"][0].item()) if "strip_length" in batch else 1
            is_strip_boundary = ((step_idx + 1) % max(strip_len, 1) == 0)
            can_stop = (
                stop_probability_threshold is not None
                and actual_steps >= int(min_steps)
                and stop_prob >= float(stop_probability_threshold)
                and (not stop_only_at_strip_boundary or is_strip_boundary)
            )
            if can_stop:
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


@torch.inference_mode()
def infer_autoreg_mesh_cached(
    model,
    sample_or_batch: dict,
    *,
    max_steps: int | None = None,
    stop_probability_threshold: float | None = None,
    min_steps: int = 0,
    stop_only_at_strip_boundary: bool = False,
    greedy: bool = True,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    save_path: str | Path | None = None,
    uuid: str | None = None,
    geometric_validation: GeometricValidationConfig | None = None,
) -> dict:
    was_training = bool(model.training)
    model.eval()
    try:
        batch = _to_single_batch(sample_or_batch)
        if int(batch["volume"].shape[0]) != 1:
            raise ValueError("infer_autoreg_mesh_cached currently expects a batch size of 1")

        device = next(model.parameters()).device
        volume = batch["volume"].to(device)
        vol_tokens = batch.get("vol_tokens")
        if vol_tokens is not None:
            vol_tokens = vol_tokens.to(device)
        prompt_tokens = {key: value.to(device) for key, value in batch["prompt_tokens"].items()}
        prompt_anchor_xyz = batch["prompt_anchor_xyz"].to(device)
        direction_id = batch["direction_id"].to(device)
        direction = batch["direction"][0]
        target_grid_shape = tuple(int(v) for v in batch["target_grid_shape"][0].tolist())
        total_vertices = int(target_grid_shape[0] * target_grid_shape[1])
        if max_steps is None:
            max_steps = total_vertices
        max_steps = min(int(max_steps), total_vertices)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive or None")

        encoded = model.encode_conditioning(volume, vol_tokens=vol_tokens)
        memory_tokens = encoded["memory_tokens"]
        memory_patch_centers = encoded["memory_patch_centers"]

        init_batch = {
            "prompt_tokens": prompt_tokens,
            "direction_id": direction_id,
        }
        _, cache = model.init_kv_cache(init_batch, memory_tokens=memory_tokens, memory_patch_centers=memory_patch_centers)

        all_target_strip_coords = _build_target_strip_coords(direction, target_grid_shape, device=device)
        geometric_flags: list[tuple[int, str]] = []
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

        prev_coarse_id = torch.full((1, 1), IGNORE_INDEX, dtype=torch.long, device=device)
        prev_offset_bins = torch.full((1, 1, 3), IGNORE_INDEX, dtype=torch.long, device=device)
        prev_xyz = prompt_anchor_xyz.unsqueeze(1)
        prev_valid = torch.tensor([[True]], dtype=torch.bool, device=device)

        for step_idx in range(max_steps):
            token_type = torch.full((1, 1), START_TOKEN_TYPE if step_idx == 0 else GENERATED_TOKEN_TYPE, dtype=torch.long, device=device)
            strip_coords = all_target_strip_coords[step_idx].unsqueeze(0).unsqueeze(0)

            embedding, coords = model._build_input_embeddings(
                coarse_ids=prev_coarse_id,
                offset_bins=prev_offset_bins,
                xyz=prev_xyz,
                strip_coords=strip_coords,
                direction_id=direction_id,
                token_type=token_type,
                sequence_mask=torch.ones((1, 1), dtype=torch.bool, device=device),
                geometry_valid_mask=prev_valid,
                memory_tokens=memory_tokens,
            )

            outputs, cache = model.step_from_encoded_cached(
                token_embedding=embedding,
                token_coords=coords,
                cache=cache,
                memory_tokens=memory_tokens,
            )

            sampling_kwargs = dict(greedy=greedy, temperature=temperature, top_k=top_k, top_p=top_p)
            if str(outputs.get("coarse_prediction_mode", getattr(model, "coarse_prediction_mode", "joint_pointer"))) == "axis_factorized":
                coarse_axis_ids = {}
                for axis_name in ("z", "y", "x"):
                    axis_logits = outputs["coarse_axis_logits"][axis_name][0, 0].float()
                    coarse_axis_ids[axis_name] = int(_sample_from_logits(axis_logits, **sampling_kwargs).item())
                coarse_id = int(
                    model._flatten_coarse_axis_ids(
                        torch.tensor([[coarse_axis_ids["z"]]], dtype=torch.long, device=device),
                        torch.tensor([[coarse_axis_ids["y"]]], dtype=torch.long, device=device),
                        torch.tensor([[coarse_axis_ids["x"]]], dtype=torch.long, device=device),
                    ).item()
                )
            else:
                coarse_logits = outputs["coarse_logits"][0, 0].float()
                coarse_id = int(_sample_from_logits(coarse_logits, **sampling_kwargs).item())
                coarse_axis_ids = {
                    "z": int(outputs["pred_coarse_axis_ids"]["z"][0, 0].item()),
                    "y": int(outputs["pred_coarse_axis_ids"]["y"][0, 0].item()),
                    "x": int(outputs["pred_coarse_axis_ids"]["x"][0, 0].item()),
                }

            offset_bins_list = []
            for axis, bins in enumerate(model.offset_num_bins):
                axis_logits = outputs["offset_logits"][0, 0, axis, :bins].float()
                offset_bins_list.append(int(_sample_from_logits(axis_logits, **sampling_kwargs).item()))

            offset_tensor = torch.tensor(offset_bins_list, dtype=torch.long, device=device).view(1, 1, 3)
            coarse_tensor = torch.tensor([[coarse_id]], dtype=torch.long, device=device)
            bin_center_xyz = model.decode_local_xyz(coarse_tensor, offset_tensor)[0, 0].detach().to(torch.float32).cpu().numpy()
            refine_residual = outputs.get("pred_refine_residual")
            if refine_residual is not None:
                res = refine_residual[0, 0].detach().to(torch.float32).cpu().numpy()
                patch_diag = float(np.linalg.norm(model.patch_size))
                res_norm = float(np.linalg.norm(res))
                if res_norm > patch_diag:
                    res = res * (patch_diag / res_norm)
                sampled_xyz = bin_center_xyz + res
            else:
                sampled_xyz = bin_center_xyz
            crop_max = np.array(model.input_shape, dtype=np.float32) - 1e-4
            sampled_xyz = np.clip(sampled_xyz, 0.0, crop_max)
            stop_prob = float(torch.sigmoid(outputs["stop_logits"][0, 0].float()).item())

            if geometric_validation is not None and geometric_validation.enabled:
                patch_diag = float(np.linalg.norm(model.patch_size if hasattr(model.patch_size, '__len__') else [model.patch_size]*3))
                max_dist = geometric_validation.max_distance_scale * patch_diag
                prev_xyz_np = out_xyz[step_idx - 1] if step_idx > 0 else None
                valid, reason = _validate_vertex(
                    sampled_xyz, prev_xyz_np,
                    input_shape=tuple(int(v) for v in model.input_shape),
                    max_dist=max_dist,
                    config=geometric_validation,
                )
                if not valid:
                    action = getattr(geometric_validation, f"{reason}_action", "flag")
                    if action == "stop":
                        break
                    if action == "flag":
                        geometric_flags.append((step_idx, reason))

            out_coarse[step_idx] = coarse_id
            for axis_name in ("z", "y", "x"):
                out_coarse_axes[axis_name][step_idx] = coarse_axis_ids[axis_name]
            out_offsets[step_idx] = offset_bins_list
            out_xyz[step_idx] = sampled_xyz.astype(np.float32, copy=False)
            out_bin_center_xyz[step_idx] = bin_center_xyz.astype(np.float32, copy=False)
            out_stop_probs[step_idx] = stop_prob
            actual_steps = step_idx + 1

            prev_coarse_id = coarse_tensor
            prev_offset_bins = offset_tensor
            prev_xyz = torch.tensor(sampled_xyz, dtype=torch.float32, device=device).view(1, 1, 3)
            prev_valid = torch.ones((1, 1), dtype=torch.bool, device=device)

            strip_len = int(batch["strip_length"][0].item()) if "strip_length" in batch else 1
            is_strip_boundary = ((actual_steps) % max(strip_len, 1) == 0)
            can_stop = (
                stop_probability_threshold is not None
                and actual_steps >= int(min_steps)
                and stop_prob >= float(stop_probability_threshold)
                and (not stop_only_at_strip_boundary or is_strip_boundary)
            )
            if can_stop:
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
            "geometric_flags": geometric_flags,
        }
        if str(getattr(model, "coarse_prediction_mode", "joint_pointer")) == "axis_factorized":
            result["predicted_coarse_axis_ids"] = {
                axis_name: out_coarse_axes[axis_name][:actual_steps].copy()
                for axis_name in ("z", "y", "x")
            }
        return result
    finally:
        model.train(was_training)
