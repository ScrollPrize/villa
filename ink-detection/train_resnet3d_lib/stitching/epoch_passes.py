import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data._utils.collate import default_collate

from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.modeling.losses import (
    compute_binary_betti_matching_loss,
    binary_mask_to_signed_distance_map,
    compute_binary_boundary_loss,
    compute_binary_soft_cldice_loss,
    compute_region_loss_and_dice,
)
from train_resnet3d_lib.stitching.buffer_ops import gaussian_weights, resolve_buffer_crop
from train_resnet3d_lib.stitching.embedding_loss import (
    build_embedding_model_for_similarity,
    compute_stitch_embedding_similarity,
    resolve_embedding_runtime_config,
)


def _validate_train_stitch_inputs(manager):
    if not manager.train_loaders or not manager.train_segment_ids:
        return False
    if len(manager.train_loaders) != len(manager.train_segment_ids):
        raise ValueError(
            "train stitch loaders/segment_ids length mismatch "
            f"({len(manager.train_loaders)} vs {len(manager.train_segment_ids)})"
        )
    return True


def _validate_train_component_inputs(manager):
    if not manager.train_component_datasets or not manager.train_component_keys:
        return False
    if len(manager.train_component_datasets) != len(manager.train_component_keys):
        raise ValueError(
            "train component datasets/keys length mismatch "
            f"({len(manager.train_component_datasets)} vs {len(manager.train_component_keys)})"
        )
    return True


def _iter_component_dataset_batches(dataset, *, batch_size):
    dataset_len = int(len(dataset))
    if dataset_len <= 0:
        return
    batch_size = int(max(1, batch_size))
    for start in range(0, dataset_len, batch_size):
        samples = [dataset[idx] for idx in range(start, min(dataset_len, start + batch_size))]
        yield default_collate(samples)


def _should_run_train_stitch_epoch(manager, model):
    epoch = int(getattr(model, "current_epoch", 0))
    if manager.train_every_n_epochs > 1 and ((epoch + 1) % manager.train_every_n_epochs) != 0:
        return False, epoch
    return True, epoch


def _stitch_saved_tensors_context(model):
    if bool(getattr(model, "stitch_gradient_checkpointing", False)):
        if bool(getattr(model, "stitch_save_on_cpu", False)) and not bool(
            getattr(model, "_stitch_warned_checkpoint_vs_offload", False)
        ):
            log(
                "stitch save_on_cpu is disabled for stitched training because "
                "training.stitch_gradient_checkpointing=true (to avoid shifting OOM to host RAM)"
            )
            model._stitch_warned_checkpoint_vs_offload = True
        return nullcontext()
    if not bool(getattr(model, "stitch_save_on_cpu", False)):
        return nullcontext()
    graph_mod = getattr(torch.autograd, "graph", None)
    save_on_cpu = getattr(graph_mod, "save_on_cpu", None) if graph_mod is not None else None
    if save_on_cpu is None:
        raise RuntimeError("training.stitch_save_on_cpu=true requires torch.autograd.graph.save_on_cpu")
    return save_on_cpu(pin_memory=True)


def _stitch_component_forward(model, x):
    if not bool(getattr(model, "stitch_gradient_checkpointing", False)):
        return model(x)
    forward_impl = getattr(model, "_forward_impl", None)
    if forward_impl is None:
        forward_impl = model
    return torch_checkpoint(forward_impl, x, use_reentrant=False)


def _torch_gaussian_weights(manager, *, h, w, device):
    device_key = (str(device.type), getattr(device, "index", None), int(h), int(w))
    weights = manager._torch_gaussian_cache.get(device_key)
    if weights is not None:
        return weights
    weights_np = gaussian_weights(
        manager._gaussian_cache,
        h=int(h),
        w=int(w),
        sigma_scale=float(manager._gaussian_sigma_scale),
        min_weight=float(manager._gaussian_min_weight),
    )
    weights = torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)
    manager._torch_gaussian_cache[device_key] = weights
    return weights


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


def _accumulate_train_loss_roi_buffers(manager, *, outputs, targets, xyxys, roi_buffers):
    for batch_i, xyxy in enumerate(xyxys):
        patch_logits = outputs[batch_i:batch_i + 1].to(dtype=torch.float32)
        patch_targets = targets[batch_i:batch_i + 1].to(dtype=torch.float32)
        for roi_buffer in roi_buffers:
            crop = resolve_buffer_crop(
                xyxy=xyxy,
                downsample=manager.downsample,
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
                manager,
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


def _get_stitch_embedding_bundle(model):
    weight = float(getattr(model, "stitch_embedding_loss_weight", 0.0) or 0.0)
    if weight <= 0.0:
        return None

    device = model.device
    cache_key = (str(device.type), getattr(device, "index", None))
    cache = getattr(model, "_stitch_embedding_runtime_cache", None)
    if cache is None:
        cache = {}
        model._stitch_embedding_runtime_cache = cache
    bundle = cache.get(cache_key)
    if bundle is not None:
        return bundle

    config_path, checkpoint_path, config = resolve_embedding_runtime_config(
        config_path=getattr(model, "stitch_embedding_model_config_path", None),
        checkpoint_path=getattr(model, "stitch_embedding_model_checkpoint_path", None),
    )
    embedding_model, runtime_config = build_embedding_model_for_similarity(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    bundle = {
        "model": embedding_model,
        "config": runtime_config,
        "config_path": str(config_path) if config_path is not None else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "crop_size": int(runtime_config["crop_size"]),
        "downsample_factor": int(getattr(model, "stitch_embedding_downsample_factor", 8) or 1),
    }
    cache[cache_key] = bundle
    return bundle


def _compute_stitched_component_loss(
    model,
    *,
    stitched_logits,
    stitched_targets,
    valid_mask,
    boundary_dist_map=None,
    embedding_bundle=None,
):
    # Keep stitched-train objectives centralized here so topology terms can be added later
    # without scattering component-loss logic across the training loop.
    region_loss, dice, bce, dice_loss = compute_region_loss_and_dice(
        stitched_logits[None, None],
        stitched_targets[None, None],
        valid_mask=valid_mask[None, None],
        reduction_dims=(1, 2, 3),
        loss_recipe=model.loss_recipe,
        smooth_factor=model.bce_smooth_factor,
        soft_label_positive=model.soft_label_positive,
        soft_label_negative=model.soft_label_negative,
    )
    boundary_loss = torch.zeros_like(region_loss)
    boundary_weight = float(getattr(model, "stitch_boundary_loss_weight", 0.0) or 0.0)
    if boundary_weight > 0.0:
        if boundary_dist_map is None:
            raise RuntimeError("boundary_dist_map is required when stitch_boundary_loss_weight > 0")
        boundary_loss = compute_binary_boundary_loss(
            stitched_logits[None, None],
            boundary_dist_map[None, None],
            valid_mask=valid_mask[None, None],
            reduction_dims=(1, 2, 3),
        )
    cldice_loss = torch.zeros_like(region_loss)
    cldice_weight = float(getattr(model, "stitch_cldice_loss_weight", 0.0) or 0.0)
    if cldice_weight > 0.0:
        cldice_loss = compute_binary_soft_cldice_loss(
            stitched_logits[None, None],
            stitched_targets[None, None],
            valid_mask=valid_mask[None, None],
            mask_mode=getattr(model, "stitch_cldice_mask_mode", "pre_skeleton"),
            reduction_dims=(1, 2, 3),
        )
    betti_matching_loss = torch.zeros_like(region_loss)
    betti_matching_weight = float(getattr(model, "stitch_betti_matching_loss_weight", 0.0) or 0.0)
    if betti_matching_weight > 0.0:
        betti_matching_loss = compute_binary_betti_matching_loss(
            stitched_logits[None, None],
            stitched_targets[None, None],
            valid_mask=valid_mask[None, None],
            filtration_type=getattr(model, "stitch_betti_matching_filtration_type", "superlevel"),
            num_processes=getattr(model, "stitch_betti_matching_num_processes", 1),
        )
    embedding_loss = torch.zeros_like(region_loss)
    embedding_weight = float(getattr(model, "stitch_embedding_loss_weight", 0.0) or 0.0)
    if embedding_weight > 0.0:
        if embedding_bundle is None:
            raise RuntimeError("stitch embedding bundle is required when stitch_embedding_loss_weight > 0")
        embedding_loss_value = compute_stitch_embedding_similarity(
            embedding_model=embedding_bundle["model"],
            embedding_crop_size=embedding_bundle["crop_size"],
            input_downsample_factor=embedding_bundle["downsample_factor"],
            stitched_logits=stitched_logits,
            stitched_targets=stitched_targets,
            valid_mask=valid_mask,
        )
        embedding_loss = embedding_loss + embedding_loss_value
    return {
        "loss": (
            region_loss[0]
            + boundary_weight * boundary_loss[0]
            + cldice_weight * cldice_loss[0]
            + betti_matching_weight * betti_matching_loss[0]
            + embedding_weight * embedding_loss[0]
        ),
        "region_loss": region_loss[0],
        "boundary_loss": boundary_loss[0],
        "cldice_loss": cldice_loss[0],
        "betti_matching_loss": betti_matching_loss[0],
        "embedding_loss": embedding_loss[0],
        "dice": dice[0],
        "bce": bce[0],
        "dice_loss": dice_loss[0],
        "covered_px": int(valid_mask.sum().detach().item()),
    }


def _resolve_train_boundary_dist_map(manager, *, component_key, roi_index, stitched_targets, device):
    key = tuple(component_key)
    roi_i = int(roi_index)

    component_cpu_cache = manager._train_boundary_dist_maps_cpu.setdefault(key, {})
    dist_map_np = component_cpu_cache.get(roi_i)
    if dist_map_np is None:
        target_mask = stitched_targets.detach().cpu().numpy() > 0.5
        dist_map_np = binary_mask_to_signed_distance_map(target_mask)
        component_cpu_cache[roi_i] = dist_map_np

    device_key = (key, roi_i, str(device.type), getattr(device, "index", None))
    dist_map_t = manager._train_boundary_dist_maps_torch.get(device_key)
    if dist_map_t is None:
        dist_map_t = torch.from_numpy(dist_map_np).to(device=device, dtype=torch.float32)
        manager._train_boundary_dist_maps_torch[device_key] = dist_map_t
    return dist_map_t


def compute_train_stitch_loss(manager, model, *, component_key):
    if not _validate_train_component_inputs(manager):
        raise RuntimeError("train component stitch loaders are not configured")

    sid, component_idx = tuple(component_key)
    sid = str(sid)
    component_idx = int(component_idx)
    key = (sid, component_idx)
    dataset = manager.train_dataset_for_component(key)
    if dataset is None:
        raise KeyError(f"missing train component stitch dataset for component_key={key!r}")

    meta = manager._train_component_meta.get(key)
    if meta is None:
        raise KeyError(f"missing train component ROI metadata for component_key={key!r}")

    roi_buffers = _allocate_train_loss_roi_buffers(meta, device=model.device)
    patch_loss_num = torch.zeros((), device=model.device, dtype=torch.float32)
    patch_dice_num = torch.zeros((), device=model.device, dtype=torch.float32)
    patch_bce_num = torch.zeros((), device=model.device, dtype=torch.float32)
    patch_dice_loss_num = torch.zeros((), device=model.device, dtype=torch.float32)
    patch_count = 0

    component_batch_size = int(getattr(CFG, "stitch_patch_batch_size", getattr(CFG, "valid_batch_size", 1)))
    for batch in _iter_component_dataset_batches(dataset, batch_size=component_batch_size):
        x, y, xyxys, _group_idx = batch
        x = x.to(model.device, non_blocking=True)
        y = y.to(model.device, non_blocking=True)
        with _stitch_saved_tensors_context(model):
            outputs = _stitch_component_forward(model, x)
            patch_count += int(outputs.shape[0])
            if model.patch_loss_weight > 0.0:
                per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = model.compute_per_sample_loss_and_dice(
                    outputs,
                    y,
                )
                patch_loss_num = patch_loss_num + per_sample_loss.sum()
                patch_dice_num = patch_dice_num + per_sample_dice.sum()
                patch_bce_num = patch_bce_num + per_sample_bce.sum()
                patch_dice_loss_num = patch_dice_loss_num + per_sample_dice_loss.sum()
            _accumulate_train_loss_roi_buffers(
                manager,
                outputs=outputs,
                targets=y,
                xyxys=xyxys,
                roi_buffers=roi_buffers,
            )

    if patch_count <= 0:
        raise ValueError(f"train component dataset for component_key={key!r} produced zero patches")

    patch_denom = float(max(1, patch_count))
    patch_loss = patch_loss_num / patch_denom
    patch_dice = patch_dice_num / patch_denom
    patch_bce = patch_bce_num / patch_denom
    patch_dice_loss = patch_dice_loss_num / patch_denom

    stitch_loss_terms = []
    stitch_region_loss_terms = []
    stitch_boundary_loss_terms = []
    stitch_cldice_loss_terms = []
    stitch_betti_matching_loss_terms = []
    stitch_embedding_loss_terms = []
    stitch_dice_terms = []
    stitch_bce_terms = []
    stitch_dice_loss_terms = []
    covered_px_total = 0
    embedding_bundle = _get_stitch_embedding_bundle(model)
    for roi_index, roi_buffer in enumerate(roi_buffers):
        valid_mask = roi_buffer["count"] > 0
        if not bool(valid_mask.any().detach().item()):
            continue
        stitched_logits = torch.where(
            valid_mask,
            roi_buffer["logits"] / roi_buffer["count"].clamp_min(1e-6),
            torch.zeros_like(roi_buffer["logits"]),
        )
        stitched_targets = torch.where(
            valid_mask,
            roi_buffer["targets"] / roi_buffer["count"].clamp_min(1e-6),
            torch.zeros_like(roi_buffer["targets"]),
        )
        boundary_dist_map = None
        if float(getattr(model, "stitch_boundary_loss_weight", 0.0) or 0.0) > 0.0:
            boundary_dist_map = _resolve_train_boundary_dist_map(
                manager,
                component_key=key,
                roi_index=roi_index,
                stitched_targets=stitched_targets,
                device=stitched_logits.device,
            )
        component_metrics = _compute_stitched_component_loss(
            model,
            stitched_logits=stitched_logits,
            stitched_targets=stitched_targets,
            valid_mask=valid_mask,
            boundary_dist_map=boundary_dist_map,
            embedding_bundle=embedding_bundle,
        )
        stitch_loss_terms.append(component_metrics["loss"])
        stitch_region_loss_terms.append(component_metrics["region_loss"])
        stitch_boundary_loss_terms.append(component_metrics["boundary_loss"])
        stitch_cldice_loss_terms.append(component_metrics["cldice_loss"])
        stitch_betti_matching_loss_terms.append(component_metrics["betti_matching_loss"])
        stitch_embedding_loss_terms.append(component_metrics["embedding_loss"])
        stitch_dice_terms.append(component_metrics["dice"])
        stitch_bce_terms.append(component_metrics["bce"])
        stitch_dice_loss_terms.append(component_metrics["dice_loss"])
        covered_px_total += int(component_metrics["covered_px"])

    if not stitch_loss_terms:
        raise ValueError(f"train component dataset for component_key={key!r} produced zero stitched components")

    stitch_loss = torch.stack(stitch_loss_terms).mean()
    stitch_region_loss = torch.stack(stitch_region_loss_terms).mean()
    stitch_boundary_loss = torch.stack(stitch_boundary_loss_terms).mean()
    stitch_cldice_loss = torch.stack(stitch_cldice_loss_terms).mean()
    stitch_betti_matching_loss = torch.stack(stitch_betti_matching_loss_terms).mean()
    stitch_embedding_loss = torch.stack(stitch_embedding_loss_terms).mean()
    stitch_dice = torch.stack(stitch_dice_terms).mean()
    stitch_bce = torch.stack(stitch_bce_terms).mean()
    stitch_dice_loss = torch.stack(stitch_dice_loss_terms).mean()
    return {
        "component_key": key,
        "segment_id": sid,
        "component_idx": component_idx,
        "group_idx": int(model._stitch_group_idx_by_segment[sid]),
        "patch_loss": patch_loss,
        "patch_dice": patch_dice,
        "patch_bce": patch_bce,
        "patch_dice_loss": patch_dice_loss,
        "stitch_loss": stitch_loss,
        "stitch_region_loss": stitch_region_loss,
        "stitch_boundary_loss": stitch_boundary_loss,
        "stitch_cldice_loss": stitch_cldice_loss,
        "stitch_betti_matching_loss": stitch_betti_matching_loss,
        "stitch_embedding_loss": stitch_embedding_loss,
        "stitch_dice": stitch_dice,
        "stitch_bce": stitch_bce,
        "stitch_dice_loss": stitch_dice_loss,
        "component_count": int(len(stitch_loss_terms)),
        "patch_count": int(patch_count),
        "covered_px": int(covered_px_total),
    }


def run_train_stitch_pass(manager, model):
    if not manager.train_viz_enabled:
        return None
    if not _validate_train_stitch_inputs(manager):
        return None
    should_run, epoch = _should_run_train_stitch_epoch(manager, model)
    if not should_run:
        return None

    t0 = time.perf_counter()
    log(f"train stitch pass start epoch={epoch}")
    segment_viz = {}

    was_training = model.training
    precision_context = manager._precision_context(model)
    try:
        model.eval()
        with torch.inference_mode(), precision_context:
            for loader, segment_id in zip(manager.train_loaders, manager.train_segment_ids):
                sid = str(segment_id)
                meta = manager._roi_meta_by_split["train"].get(sid)
                if meta is None:
                    continue

                full_shape = tuple(int(v) for v in meta.get("full_shape", (0, 0)))
                roi_meta = list(meta.get("rois", []))
                if len(roi_meta) == 0:
                    continue

                roi_buffers = [
                    (
                        np.zeros(tuple(int(v) for v in roi["buffer_shape"]), dtype=np.float32),
                        np.zeros(tuple(int(v) for v in roi["buffer_shape"]), dtype=np.float32),
                        tuple(int(v) for v in roi["offset"]),
                    )
                    for roi in roi_meta
                ]

                for batch in loader:
                    x, _y, xyxys, _g = batch
                    x = x.to(model.device, non_blocking=True)
                    outputs = model(x)
                    for pred_buf, count_buf, offset in roi_buffers:
                        manager.accumulate_to_buffers(
                            outputs=outputs,
                            xyxys=xyxys,
                            pred_buf=pred_buf,
                            count_buf=count_buf,
                            offset=offset,
                            ds_override=manager.downsample,
                        )

                stitched, covered = manager._compose_segment_from_roi_buffers(roi_buffers, full_shape)
                covered_px = int(covered.sum())
                total_px = int(covered.size)
                coverage = float(covered_px) / float(max(1, total_px))
                if covered_px > 0:
                    vals = stitched[covered]
                    prob_mean = float(vals.mean()) if vals.size else float("nan")
                    prob_max = float(vals.max()) if vals.size else float("nan")
                else:
                    prob_mean = float("nan")
                    prob_max = float("nan")

                log(
                    f"train stitch summary segment={sid} "
                    f"coverage={coverage:.4f} covered_px={covered_px}/{total_px} "
                    f"prob_mean={prob_mean:.4f} prob_max={prob_max:.4f}"
                )
                segment_viz[sid] = {
                    "img_u8": (np.clip(stitched, 0.0, 1.0) * 255.0).astype(np.uint8),
                    "has": covered,
                    "meta": {
                        "offset": (0, 0),
                        "full_shape": full_shape,
                    },
                }
    finally:
        if was_training:
            model.train()

    log(f"train stitch pass done epoch={epoch} elapsed={time.perf_counter() - t0:.1f}s")
    return segment_viz


def _unpack_log_only_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            return batch[0], batch[1]
        if len(batch) >= 3:
            return batch[0], batch[2]
    raise ValueError("log-only stitch batch must be (x, xyxys) or (x, y, xyxys, g)")


def run_log_only_stitch_pass(manager, model):
    if not manager.log_only_loaders or not manager.log_only_segment_ids:
        return False
    if len(manager.log_only_loaders) != len(manager.log_only_segment_ids):
        raise ValueError(
            "log-only stitch loaders/segment_ids length mismatch "
            f"({len(manager.log_only_loaders)} vs {len(manager.log_only_segment_ids)})"
        )

    epoch = int(getattr(model, "current_epoch", 0))
    if manager.log_only_every_n_epochs > 1 and ((epoch + 1) % manager.log_only_every_n_epochs) != 0:
        return False

    t0 = time.perf_counter()
    log(f"log-only stitch pass start epoch={epoch}")

    manager._reset_buffers_for_split("log_only")

    was_training = model.training
    precision_context = manager._precision_context(model)
    try:
        model.eval()
        with torch.inference_mode(), precision_context:
            for loader, segment_id in zip(manager.log_only_loaders, manager.log_only_segment_ids):
                sid = str(segment_id)
                roi_buffers = manager._roi_buffers_by_split["log_only"].get(sid)
                if not roi_buffers:
                    continue
                for batch in loader:
                    x, xyxys = _unpack_log_only_batch(batch)
                    x = x.to(model.device, non_blocking=True)
                    outputs = model(x)
                    for pred_buf, count_buf, offset in roi_buffers:
                        manager.accumulate_to_buffers(
                            outputs=outputs,
                            xyxys=xyxys,
                            pred_buf=pred_buf,
                            count_buf=count_buf,
                            offset=offset,
                            ds_override=manager.log_only_downsample,
                        )
    finally:
        if was_training:
            model.train()

    log(f"log-only stitch pass done epoch={epoch} elapsed={time.perf_counter() - t0:.1f}s")
    return True
