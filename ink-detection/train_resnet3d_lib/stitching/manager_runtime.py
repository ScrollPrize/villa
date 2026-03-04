from contextlib import nullcontext

import numpy as np
import torch


def distributed_world_size(model):
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return 1
    return int(getattr(trainer, "world_size", 1) or 1)


def precision_context(model):
    precision_ctx = nullcontext()
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return precision_ctx

    strategy = getattr(trainer, "strategy", None)
    precision_plugin = getattr(strategy, "precision_plugin", None) if strategy is not None else None
    if precision_plugin is None:
        precision_plugin = getattr(trainer, "precision_plugin", None)
    if precision_plugin is not None and hasattr(precision_plugin, "forward_context"):
        return precision_plugin.forward_context()
    return precision_ctx


def reset_split_buffers(manager, split):
    buffers_by_segment = manager._roi_buffers_by_split.get(str(split))
    if not buffers_by_segment:
        return
    for roi_buffers in buffers_by_segment.values():
        for pred_buf, count_buf, _offset in roi_buffers:
            pred_buf.fill(0)
            count_buf.fill(0)


def reduce_sum_distributed(model, tensor):
    strategy = getattr(getattr(model, "trainer", None), "strategy", None)
    if strategy is None or not hasattr(strategy, "reduce"):
        raise RuntimeError("distributed stitch reduction requested but trainer.strategy.reduce is unavailable")
    return strategy.reduce(tensor, reduce_op="sum")


def sync_val_buffers_distributed(manager, model):
    if distributed_world_size(model) <= 1:
        return False

    device = model.device
    for roi_buffers in manager._roi_buffers_by_split["val"].values():
        for pred_buf, count_buf, _offset in roi_buffers:
            pred_tensor = torch.from_numpy(np.ascontiguousarray(pred_buf)).to(device=device, dtype=torch.float32)
            count_tensor = torch.from_numpy(np.ascontiguousarray(count_buf)).to(device=device, dtype=torch.float32)
            pred_tensor = reduce_sum_distributed(model, pred_tensor)
            count_tensor = reduce_sum_distributed(model, count_tensor)
            pred_buf[...] = pred_tensor.detach().cpu().numpy()
            count_buf[...] = count_tensor.detach().cpu().numpy()
    return True


def build_validation_epoch_context(manager, model):
    trainer = getattr(model, "trainer", None)
    sanity_checking = bool(trainer is not None and getattr(trainer, "sanity_checking", False))
    is_global_zero = bool(trainer is None or trainer.is_global_zero)

    train_configured = bool(manager.train_loaders) and bool(manager.train_segment_ids) and bool(
        manager._roi_meta_by_split["train"]
    )
    stitch_train_mode = bool(manager.train_enabled and train_configured)

    log_only_configured = bool(manager.log_only_loaders) and bool(manager.log_only_segment_ids) and bool(
        manager._roi_buffers_by_split["log_only"]
    )
    log_only_mode = bool(log_only_configured)

    return {
        "sanity_checking": sanity_checking,
        "is_global_zero": is_global_zero,
        "stitch_train_mode": stitch_train_mode,
        "log_only_mode": log_only_mode,
    }


def maybe_run_train_stitch(
    manager,
    model,
    *,
    stitch_train_mode,
    sanity_checking,
    is_global_zero,
):
    did_run_train_stitch = False
    train_segment_viz = {}

    if stitch_train_mode and (not is_global_zero):
        stitch_train_mode = False
    if stitch_train_mode and (not sanity_checking):
        train_segment_viz = manager.run_train_stitch_pass(model) or {}
        did_run_train_stitch = bool(train_segment_viz)
        if not did_run_train_stitch:
            stitch_train_mode = False

    return stitch_train_mode, did_run_train_stitch, train_segment_viz


def maybe_run_log_only_stitch(
    manager,
    model,
    *,
    log_only_mode,
    sanity_checking,
    is_global_zero,
):
    did_run_log_only = False

    if log_only_mode and (not is_global_zero):
        reset_split_buffers(manager, "log_only")
        log_only_mode = False
    if log_only_mode and (not sanity_checking):
        did_run_log_only = bool(manager.run_log_only_stitch_pass(model))
        if not did_run_log_only:
            log_only_mode = False

    return log_only_mode, did_run_log_only


def collect_val_segments_for_logging(manager):
    segment_to_val = {}
    segment_to_val_meta = {}

    for _loader_idx, segment_id in manager._val_loader_to_segment.items():
        sid = str(segment_id)
        roi_buffers = manager._roi_buffers_by_split["val"].get(sid)
        if not roi_buffers:
            continue
        meta = manager._roi_meta_by_split["val"].get(sid, {})
        full_shape = tuple(meta.get("full_shape", roi_buffers[0][0].shape))
        base, has = manager._compose_segment_from_roi_buffers(roi_buffers, full_shape)
        segment_to_val[sid] = (base, has)
        segment_to_val_meta[sid] = {
            "offset": (0, 0),
            "full_shape": full_shape,
        }

    return segment_to_val, segment_to_val_meta


def sync_val_buffers_and_maybe_exit_nonzero_worker(manager, model, *, is_global_zero):
    did_sync_val_buffers = manager.sync_val_buffers_distributed(model)
    if did_sync_val_buffers and (not is_global_zero):
        manager._reset_buffers_for_split("val")
        return True
    return False


def reset_epoch_end_buffers(manager, *, log_only_mode):
    manager._reset_buffers_for_split("val")
    if log_only_mode:
        manager._reset_buffers_for_split("log_only")


__all__ = [
    "distributed_world_size",
    "precision_context",
    "reset_split_buffers",
    "reduce_sum_distributed",
    "sync_val_buffers_distributed",
    "build_validation_epoch_context",
    "maybe_run_train_stitch",
    "maybe_run_log_only_stitch",
    "collect_val_segments_for_logging",
    "sync_val_buffers_and_maybe_exit_nonzero_worker",
    "reset_epoch_end_buffers",
]
