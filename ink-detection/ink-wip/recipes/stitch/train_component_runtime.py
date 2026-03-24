from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data._utils.collate import default_collate

from ink.recipes.losses.reporting import resolve_train_output as resolve_train_loss_output
from ink.recipes.stitch.artifact_primitives import binary_mask_to_signed_distance_map
from ink.recipes.stitch.config import StitchData, normalize_component_key
from ink.recipes.stitch.ops import gaussian_weights, normalize_xyxy_rows, resolve_buffer_crop
from ink.recipes.stitch.runtime import _log
from ink.recipes.stitch.terms import StitchLossBatch, _requires_boundary_dist_map, compute_stitched_loss_components


def stitch_saved_tensors_context(owner, *, log: Callable[[Any, str], None]):
    if bool(owner.gradient_checkpointing):
        if bool(owner.save_on_cpu) and not bool(owner._warned_checkpoint_vs_offload):
            log(
                owner,
                "stitch save_on_cpu is disabled for stitched training because "
                "stitch.train.loss.gradient_checkpointing=true (to avoid shifting OOM to host RAM)",
            )
            owner._warned_checkpoint_vs_offload = True
        return nullcontext()
    if not bool(owner.save_on_cpu):
        return nullcontext()
    graph_mod = getattr(torch.autograd, "graph", None)
    save_on_cpu = getattr(graph_mod, "save_on_cpu", None) if graph_mod is not None else None
    if save_on_cpu is None:
        raise RuntimeError("stitch.train.loss.save_on_cpu=true requires torch.autograd.graph.save_on_cpu")
    return save_on_cpu(pin_memory=True)


def _resolve_boundary_dist_map(state, *, cache_key, stitched_targets, device):
    dist_map_np = state._boundary_dist_maps_cpu.get(cache_key)
    if dist_map_np is None:
        target_mask = stitched_targets.detach().cpu().numpy() > 0.5
        dist_map_np = binary_mask_to_signed_distance_map(target_mask)
        state._boundary_dist_maps_cpu[cache_key] = dist_map_np

    device_key = (cache_key, str(device.type), getattr(device, "index", None))
    dist_map_t = state._boundary_dist_maps_torch.get(device_key)
    if dist_map_t is None:
        dist_map_t = torch.from_numpy(dist_map_np).to(device=device, dtype=torch.float32)
        state._boundary_dist_maps_torch[device_key] = dist_map_t
    return dist_map_t


def _dataset_component_group_idx(dataset, *, component_key):
    if int(len(dataset)) <= 0:
        raise ValueError(f"train component dataset for component_key={component_key!r} produced zero patches")

    sample = dataset[0]
    if not isinstance(sample, (list, tuple)) or len(sample) < 4:
        raise ValueError("train component stitch dataset samples must be (x, y, xyxys, group_idx)")

    raw_group_idx = sample[3]
    if isinstance(raw_group_idx, torch.Tensor):
        flat = raw_group_idx.detach().reshape(-1)
        if int(flat.numel()) != 1:
            raise ValueError("train component stitch dataset group_idx sample must be scalar")
        return int(flat.item())

    raw_arr = np.asarray(raw_group_idx)
    if int(raw_arr.size) != 1:
        raise ValueError("train component stitch dataset group_idx sample must be scalar")
    return int(raw_arr.reshape(-1)[0])


def _resolve_model_device(model) -> torch.device:
    explicit_device = getattr(model, "device", None)
    if explicit_device is not None:
        try:
            return torch.device(explicit_device)
        except (RuntimeError, TypeError, ValueError):
            pass

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except (StopIteration, TypeError):
            pass

    buffers = getattr(model, "buffers", None)
    if callable(buffers):
        try:
            return next(buffers()).device
        except (StopIteration, TypeError):
            pass

    return torch.device("cpu")


def _allocate_train_loss_roi_buffers(meta, *, device):
    roi_buffers = []
    for roi in meta.get("rois", []):
        buffer_shape = tuple(int(v) for v in roi["buffer_shape"])
        roi_buffers.append(
            {
                "logits": torch.zeros(buffer_shape, device=device, dtype=torch.float32),
                "targets": torch.zeros(buffer_shape, device=device, dtype=torch.float32),
                "count": torch.zeros(buffer_shape, device=device, dtype=torch.float32),
                "offset": tuple(int(v) for v in roi["offset"]),
            }
        )
    return roi_buffers


def _run_component_patch_pass(
    owner,
    model,
    *,
    data: StitchData,
    state,
    dataset,
    component_key,
    batch_size,
    roi_buffers,
):
    components_enabled = bool(float(owner.patch_loss_weight) > 0.0)
    model_device = _resolve_model_device(model)
    patch_totals = {}
    if components_enabled:
        patch_totals = {
            "loss": torch.zeros((), device=model_device, dtype=torch.float32),
            "components": {},
        }

    patch_count = 0
    for batch in _iter_component_dataset_batches(dataset, batch_size=batch_size):
        patch_count += _run_component_batch(
            owner,
            model,
            data=data,
            state=state,
            batch=batch,
            model_device=model_device,
            roi_buffers=roi_buffers,
            patch_totals=patch_totals,
        )

    if patch_count <= 0:
        raise ValueError(f"train component dataset for component_key={component_key!r} produced zero patches")

    out = {
        "count": int(patch_count),
        "components": {},
    }
    if components_enabled:
        patch_denom = float(max(1, patch_count))
        out["components"] = {
            "loss": patch_totals["loss"] / patch_denom,
            **{
                key: value / patch_denom
                for key, value in patch_totals["components"].items()
            },
        }
    return out


def _iter_component_dataset_batches(dataset, *, batch_size):
    dataset_len = int(len(dataset))
    if dataset_len <= 0:
        return
    batch_size = int(max(1, batch_size))
    for start in range(0, dataset_len, batch_size):
        samples = [dataset[idx] for idx in range(start, min(dataset_len, start + batch_size))]
        yield default_collate(samples)


def _run_component_batch(
    owner,
    model,
    *,
    data: StitchData,
    state,
    batch,
    model_device,
    roi_buffers,
    patch_totals,
):
    x, y, xyxys, _group_idx = batch
    xyxys = normalize_xyxy_rows(xyxys)
    x = x.to(model_device, non_blocking=True)
    y = y.to(model_device, non_blocking=True)
    precision_context = getattr(owner, "precision_context", None)
    context = precision_context(device=x.device) if callable(precision_context) else nullcontext()
    with context:
        with stitch_saved_tensors_context(owner, log=_log):
            outputs = _stitch_component_forward(owner, model, x)
            _accumulate_patch_components(owner, outputs=outputs, targets=y, patch_totals=patch_totals)
            _accumulate_train_loss_roi_buffers(
                data=data,
                state=state,
                outputs=outputs,
                targets=y,
                xyxys=xyxys,
                roi_buffers=roi_buffers,
            )
    return int(outputs.shape[0])


def _stitch_component_forward(owner, model, x):
    if not bool(owner.gradient_checkpointing):
        return model(x)
    forward_impl = getattr(model, "_forward_impl", None)
    if forward_impl is None:
        forward_impl = model
    return torch_checkpoint(forward_impl, x, use_reentrant=False)


def _accumulate_patch_components(owner, *, outputs, targets, patch_totals):
    if not patch_totals:
        return
    loss_output = _compute_patch_loss_output(owner, outputs=outputs, targets=targets)
    batch_size = int(outputs.shape[0])
    patch_totals["loss"] = patch_totals["loss"] + (_component_value(loss_output.loss, device=outputs.device) * batch_size)

    component_totals = patch_totals["components"]
    for key, value in loss_output.components.items():
        key = str(key)
        if key.startswith("train/"):
            key = key.split("/", maxsplit=1)[1]
        current = component_totals.get(key)
        if current is None:
            current = torch.zeros((), device=outputs.device, dtype=torch.float32)
        component_totals[key] = current + (_component_value(value, device=outputs.device) * batch_size)


def _compute_patch_loss_output(owner, *, outputs, targets):
    patch_loss = owner.patch_loss
    if not callable(patch_loss):
        raise RuntimeError("stitched patch components require owner.patch_loss as the experiment loss recipe")
    return resolve_train_loss_output(patch_loss, outputs, targets)


def _component_value(value, *, device):
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim > 0:
        tensor = tensor.reshape(-1).mean()
    return tensor.detach()


def _accumulate_train_loss_roi_buffers(*, data: StitchData, state, outputs, targets, xyxys, roi_buffers):
    downsample = int(data.layout.downsample)
    for batch_i, xyxy in enumerate(xyxys):
        patch_logits = outputs[batch_i:batch_i + 1].to(dtype=torch.float32)
        patch_targets = targets[batch_i:batch_i + 1].to(dtype=torch.float32)
        for roi_buffer in roi_buffers:
            crop = resolve_buffer_crop(
                xyxy=xyxy,
                downsample=downsample,
                offset=roi_buffer["offset"],
                buffer_shape=roi_buffer["count"].shape,
            )
            if crop is None:
                continue

            if patch_logits.shape[-2:] != (crop["target_h"], crop["target_w"]):
                patch_logits_resized = F.interpolate(
                    patch_logits,
                    size=(crop["target_h"], crop["target_w"]),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                patch_logits_resized = patch_logits

            if patch_targets.shape[-2:] != (crop["target_h"], crop["target_w"]):
                patch_targets_resized = F.interpolate(
                    patch_targets,
                    size=(crop["target_h"], crop["target_w"]),
                    mode="nearest",
                )
            else:
                patch_targets_resized = patch_targets

            patch_weights = _torch_gaussian_weights(
                state,
                h=crop["target_h"],
                w=crop["target_w"],
                device=patch_logits_resized.device,
            )
            patch_weight_crop = patch_weights[crop["py0"]:crop["py1"], crop["px0"]:crop["px1"]]
            target_slice = (slice(crop["y1"], crop["y2"]), slice(crop["x1"], crop["x2"]))
            patch_slice = (..., slice(crop["py0"], crop["py1"]), slice(crop["px0"], crop["px1"]))

            roi_buffer["logits"][target_slice] += patch_logits_resized[patch_slice].squeeze(0).squeeze(0) * patch_weight_crop
            roi_buffer["targets"][target_slice] += patch_targets_resized[patch_slice].squeeze(0).squeeze(0) * patch_weight_crop
            roi_buffer["count"][target_slice] += patch_weight_crop


def _torch_gaussian_weights(state, *, h, w, device):
    device_key = (str(device.type), getattr(device, "index", None), int(h), int(w))
    weights = state._torch_gaussian_cache.get(device_key)
    if weights is not None:
        return weights
    weights_np = gaussian_weights(
        state._gaussian_cache,
        h=int(h),
        w=int(w),
        sigma_scale=float(state._gaussian_sigma_scale),
        min_weight=float(state._gaussian_min_weight),
    )
    weights = torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)
    state._torch_gaussian_cache[device_key] = weights
    return weights


def _summarize_train_stitched_component_losses(
    owner,
    *,
    state,
    terms,
    component_key,
    roi_buffers,
):
    component_terms = {}
    covered_px_total = 0
    component_count = 0
    needs_boundary_dist_map = bool(_requires_boundary_dist_map(terms))

    for roi_index, roi_buffer in enumerate(roi_buffers):
        valid_mask = roi_buffer["count"] > 0
        if not bool(valid_mask.any().detach().item()):
            continue

        count = roi_buffer["count"].clamp_min(1e-6)
        stitched_logits = torch.where(
            valid_mask,
            roi_buffer["logits"] / count,
            torch.zeros_like(roi_buffer["logits"]),
        )
        stitched_targets = torch.where(
            valid_mask,
            roi_buffer["targets"] / count,
            torch.zeros_like(roi_buffer["targets"]),
        )

        boundary_dist_map = None
        if needs_boundary_dist_map:
            boundary_dist_map = _resolve_boundary_dist_map(
                state,
                cache_key=("component", normalize_component_key(component_key), int(roi_index)),
                stitched_targets=stitched_targets,
                device=stitched_logits.device,
            )

        component_outputs = compute_stitched_loss_components(
            terms,
            StitchLossBatch(
                logits=stitched_logits,
                targets=stitched_targets,
                valid_mask=valid_mask,
                boundary_dist_map=boundary_dist_map,
            ),
        )
        for key, value in component_outputs.items():
            if key == "covered_px":
                continue
            component_terms.setdefault(key, []).append(value)
        covered_px_total += int(component_outputs["covered_px"])
        component_count += 1

    if component_count <= 0:
        raise ValueError(f"train component dataset for component_key={component_key!r} produced zero stitched components")

    return {
        "component_count": int(component_count),
        "covered_px": int(covered_px_total),
        "components": {key: torch.stack(values).mean() for key, values in component_terms.items()},
    }
