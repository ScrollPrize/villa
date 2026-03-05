from contextlib import nullcontext

import numpy as np
import torch

from train_resnet3d_lib.stitching.buffer_ops import (
    accumulate_to_buffers as _accumulate_to_buffers,
    compose_segment_from_roi_buffers as _compose_segment_from_roi_buffers,
)
from train_resnet3d_lib.stitching.epoch_passes import (
    run_log_only_stitch_pass as _run_log_only_stitch_pass,
    run_train_stitch_pass as _run_train_stitch_pass,
)
from train_resnet3d_lib.stitching.metrics_runtime import log_stitched_validation_metrics
from train_resnet3d_lib.stitching.roi_layout import allocate_segment_buffers, build_segment_roi_meta
from train_resnet3d_lib.stitching.wandb_media import log_stitched_wandb_media


def initialize_manager_state(
    manager,
    *,
    stitch_downsample,
    stitch_log_only_downsample,
    stitch_train,
    stitch_train_every_n_epochs,
    stitch_log_only_every_n_epochs,
    stitch_use_roi,
    stitch_val_bboxes,
    stitch_train_bboxes,
    stitch_log_only_bboxes,
):
    manager.downsample = max(1, int(stitch_downsample or 1))
    manager.log_only_downsample = int(stitch_log_only_downsample or manager.downsample)

    manager._roi_meta_by_split = {"val": {}, "train": {}, "log_only": {}}
    manager._roi_buffers_by_split = {"val": {}, "log_only": {}}
    manager._val_loader_to_segment = {}

    manager.train_segment_ids = []
    manager.train_loaders = []
    manager.train_enabled = bool(stitch_train)
    manager.train_every_n_epochs = max(1, int(stitch_train_every_n_epochs or 1))

    manager.log_only_segment_ids = []
    manager.log_only_loaders = []
    manager.log_only_every_n_epochs = max(1, int(stitch_log_only_every_n_epochs or 10))

    manager.borders_by_split = {"train": {}, "val": {}}
    manager.use_roi = bool(stitch_use_roi)
    manager.val_bboxes = dict(stitch_val_bboxes or {})
    manager.train_bboxes = dict(stitch_train_bboxes or {})
    manager.log_only_bboxes = dict(stitch_log_only_bboxes or {})

    manager._gaussian_cache = {}
    manager._gaussian_sigma_scale = 1.0 / 8.0
    manager._gaussian_min_weight = 1e-6


def _register_val_segments(
    manager,
    *,
    stitch_all_val,
    stitch_all_val_shapes,
    stitch_all_val_segment_ids,
    stitch_val_dataloader_idx,
    stitch_pred_shape,
    stitch_segment_id,
):
    if bool(stitch_all_val):
        if stitch_all_val_shapes is None or stitch_all_val_segment_ids is None:
            raise ValueError("stitch_all_val requires stitch_all_val_shapes and stitch_all_val_segment_ids")
        if len(stitch_all_val_shapes) != len(stitch_all_val_segment_ids):
            raise ValueError(
                "stitch_all_val_shapes and stitch_all_val_segment_ids must have the same length "
                f"(got {len(stitch_all_val_shapes)} vs {len(stitch_all_val_segment_ids)})"
            )
        for loader_idx, (segment_id, shape) in enumerate(
            zip(stitch_all_val_segment_ids or [], stitch_all_val_shapes or [])
        ):
            sid = str(segment_id)
            if sid in manager._roi_meta_by_split["val"]:
                raise ValueError(f"duplicate val stitch segment id: {sid!r}")
            bbox = manager.val_bboxes.get(sid)
            manager._register_segment("val", sid, shape, bbox, manager.downsample)
            manager._val_loader_to_segment[int(loader_idx)] = sid
        return

    if stitch_val_dataloader_idx is None or stitch_pred_shape is None:
        return

    sid = str(stitch_segment_id if stitch_segment_id is not None else stitch_val_dataloader_idx)
    bbox = manager.val_bboxes.get(sid)
    manager._register_segment("val", sid, stitch_pred_shape, bbox, manager.downsample)
    manager._val_loader_to_segment[int(stitch_val_dataloader_idx)] = sid


def _register_train_segments(manager, *, stitch_train_segment_ids, stitch_train_shapes):
    if stitch_train_shapes is None and stitch_train_segment_ids is None:
        return
    if stitch_train_shapes is None or stitch_train_segment_ids is None:
        raise ValueError("stitch_train_shapes and stitch_train_segment_ids must both be set or both be None")
    if len(stitch_train_shapes) != len(stitch_train_segment_ids):
        raise ValueError(
            "stitch_train_shapes and stitch_train_segment_ids must have the same length "
            f"(got {len(stitch_train_shapes)} vs {len(stitch_train_segment_ids)})"
        )
    for segment_id, shape in zip(stitch_train_segment_ids or [], stitch_train_shapes or []):
        sid = str(segment_id)
        if sid in manager._roi_meta_by_split["train"]:
            raise ValueError(f"duplicate train stitch segment id: {sid!r}")
        bbox = manager.train_bboxes.get(sid)
        manager._register_segment("train", sid, shape, bbox, manager.downsample)
        manager.train_segment_ids.append(sid)


def _register_log_only_segments(manager, *, stitch_log_only_segment_ids, stitch_log_only_shapes):
    if stitch_log_only_shapes is None and stitch_log_only_segment_ids is None:
        return
    if stitch_log_only_shapes is None or stitch_log_only_segment_ids is None:
        raise ValueError(
            "stitch_log_only_shapes and stitch_log_only_segment_ids must both be set or both be None"
        )
    if len(stitch_log_only_shapes) != len(stitch_log_only_segment_ids):
        raise ValueError(
            "stitch_log_only_shapes and stitch_log_only_segment_ids must have the same length "
            f"(got {len(stitch_log_only_shapes)} vs {len(stitch_log_only_segment_ids)})"
        )
    for segment_id, shape in zip(stitch_log_only_segment_ids or [], stitch_log_only_shapes or []):
        sid = str(segment_id)
        if sid in manager._roi_meta_by_split["log_only"]:
            raise ValueError(f"duplicate log-only stitch segment id: {sid!r}")
        bbox = manager.log_only_bboxes.get(sid)
        manager._register_segment("log_only", sid, shape, bbox, manager.log_only_downsample)
        manager.log_only_segment_ids.append(sid)


def register_initial_segments(
    manager,
    *,
    stitch_all_val,
    stitch_all_val_shapes,
    stitch_all_val_segment_ids,
    stitch_val_dataloader_idx,
    stitch_pred_shape,
    stitch_segment_id,
    stitch_train_shapes,
    stitch_train_segment_ids,
    stitch_log_only_shapes,
    stitch_log_only_segment_ids,
):
    _register_val_segments(
        manager,
        stitch_all_val=stitch_all_val,
        stitch_all_val_shapes=stitch_all_val_shapes,
        stitch_all_val_segment_ids=stitch_all_val_segment_ids,
        stitch_val_dataloader_idx=stitch_val_dataloader_idx,
        stitch_pred_shape=stitch_pred_shape,
        stitch_segment_id=stitch_segment_id,
    )
    _register_train_segments(
        manager,
        stitch_train_segment_ids=stitch_train_segment_ids,
        stitch_train_shapes=stitch_train_shapes,
    )
    _register_log_only_segments(
        manager,
        stitch_log_only_segment_ids=stitch_log_only_segment_ids,
        stitch_log_only_shapes=stitch_log_only_shapes,
    )

    manager.enabled = len(manager._roi_buffers_by_split["val"]) > 0


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


class StitchManager:
    def __init__(
        self,
        *,
        stitch_val_dataloader_idx=None,
        stitch_pred_shape=None,
        stitch_segment_id=None,
        stitch_all_val=False,
        stitch_downsample=1,
        stitch_all_val_shapes=None,
        stitch_all_val_segment_ids=None,
        stitch_train_shapes=None,
        stitch_train_segment_ids=None,
        stitch_use_roi=False,
        stitch_val_bboxes=None,
        stitch_train_bboxes=None,
        stitch_log_only_shapes=None,
        stitch_log_only_segment_ids=None,
        stitch_log_only_bboxes=None,
        stitch_log_only_downsample=None,
        stitch_log_only_every_n_epochs=10,
        stitch_train=False,
        stitch_train_every_n_epochs=1,
    ):
        initialize_manager_state(
            self,
            stitch_downsample=stitch_downsample,
            stitch_log_only_downsample=stitch_log_only_downsample,
            stitch_train=stitch_train,
            stitch_train_every_n_epochs=stitch_train_every_n_epochs,
            stitch_log_only_every_n_epochs=stitch_log_only_every_n_epochs,
            stitch_use_roi=stitch_use_roi,
            stitch_val_bboxes=stitch_val_bboxes,
            stitch_train_bboxes=stitch_train_bboxes,
            stitch_log_only_bboxes=stitch_log_only_bboxes,
        )

        register_initial_segments(
            self,
            stitch_all_val=stitch_all_val,
            stitch_all_val_shapes=stitch_all_val_shapes,
            stitch_all_val_segment_ids=stitch_all_val_segment_ids,
            stitch_val_dataloader_idx=stitch_val_dataloader_idx,
            stitch_pred_shape=stitch_pred_shape,
            stitch_segment_id=stitch_segment_id,
            stitch_train_shapes=stitch_train_shapes,
            stitch_train_segment_ids=stitch_train_segment_ids,
            stitch_log_only_shapes=stitch_log_only_shapes,
            stitch_log_only_segment_ids=stitch_log_only_segment_ids,
        )

    def _register_segment(self, split, segment_id, shape, bbox, ds):
        split = str(split)
        sid = str(segment_id)
        self._roi_meta_by_split[split][sid] = build_segment_roi_meta(
            shape,
            bbox,
            ds,
            use_roi=self.use_roi,
        )
        if split in self._roi_buffers_by_split:
            self._roi_buffers_by_split[split][sid] = allocate_segment_buffers(
                self._roi_meta_by_split[split][sid]
            )

    def set_borders(self, *, train_borders=None, val_borders=None):
        if train_borders is not None:
            self.borders_by_split["train"] = dict(train_borders)
        if val_borders is not None:
            self.borders_by_split["val"] = dict(val_borders)

    def set_train_loaders(self, loaders, segment_ids):
        self.train_loaders = list(loaders or [])
        self.train_segment_ids = [str(x) for x in (segment_ids or [])]

    def set_log_only_loaders(self, loaders, segment_ids):
        self.log_only_loaders = list(loaders or [])
        self.log_only_segment_ids = [str(x) for x in (segment_ids or [])]

    def _precision_context(self, model):
        return precision_context(model)

    def _reset_buffers_for_split(self, split):
        reset_split_buffers(self, split)

    def sync_val_buffers_distributed(self, model):
        return sync_val_buffers_distributed(self, model)

    def _compose_segment_from_roi_buffers(self, roi_buffers, full_shape):
        return _compose_segment_from_roi_buffers(roi_buffers, full_shape)

    def accumulate_to_buffers(self, *, outputs, xyxys, pred_buf, count_buf, offset=(0, 0), ds_override=None):
        _accumulate_to_buffers(
            outputs=outputs,
            xyxys=xyxys,
            pred_buf=pred_buf,
            count_buf=count_buf,
            downsample=int(ds_override or self.downsample),
            offset=offset,
            gaussian_cache=self._gaussian_cache,
            gaussian_sigma_scale=float(self._gaussian_sigma_scale),
            gaussian_min_weight=float(self._gaussian_min_weight),
        )

    def accumulate_val(self, *, outputs, xyxys, dataloader_idx):
        if not self.enabled:
            return

        sid = self._val_loader_to_segment.get(int(dataloader_idx))
        if sid is None:
            return

        for pred_buf, count_buf, offset in self._roi_buffers_by_split["val"].get(str(sid), []):
            self.accumulate_to_buffers(
                outputs=outputs,
                xyxys=xyxys,
                pred_buf=pred_buf,
                count_buf=count_buf,
                offset=offset,
            )

    def run_train_stitch_pass(self, model):
        return _run_train_stitch_pass(self, model)

    def run_log_only_stitch_pass(self, model):
        return _run_log_only_stitch_pass(self, model)

    def on_validation_epoch_end(self, model):
        if not self.enabled or not self._roi_buffers_by_split["val"]:
            return

        epoch_ctx = build_validation_epoch_context(self, model)
        sanity_checking = bool(epoch_ctx["sanity_checking"])
        is_global_zero = bool(epoch_ctx["is_global_zero"])

        stitch_train_mode, did_run_train_stitch, train_segment_viz = maybe_run_train_stitch(
            self,
            model,
            stitch_train_mode=bool(epoch_ctx["stitch_train_mode"]),
            sanity_checking=sanity_checking,
            is_global_zero=is_global_zero,
        )
        log_only_mode, _did_run_log_only = maybe_run_log_only_stitch(
            self,
            model,
            log_only_mode=bool(epoch_ctx["log_only_mode"]),
            sanity_checking=sanity_checking,
            is_global_zero=is_global_zero,
        )

        if sync_val_buffers_and_maybe_exit_nonzero_worker(
            self,
            model,
            is_global_zero=is_global_zero,
        ):
            return

        segment_to_val, segment_to_val_meta = collect_val_segments_for_logging(self)
        log_train_stitch = bool(stitch_train_mode and did_run_train_stitch)

        log_stitched_wandb_media(
            model=model,
            sanity_checking=sanity_checking,
            log_train_stitch=log_train_stitch,
            segment_to_val=segment_to_val,
            train_segment_viz=train_segment_viz,
            log_only_mode=bool(log_only_mode),
            log_only_segment_ids=list(self.log_only_segment_ids),
            roi_buffers_by_split=self._roi_buffers_by_split,
            roi_meta_by_split=self._roi_meta_by_split,
            borders_by_split=self.borders_by_split,
            downsample=int(self.downsample),
            log_only_downsample=int(self.log_only_downsample),
            compose_segment_from_roi_buffers=self._compose_segment_from_roi_buffers,
        )

        log_stitched_validation_metrics(
            model=model,
            sanity_checking=sanity_checking,
            segment_to_val=segment_to_val,
            segment_to_val_meta=segment_to_val_meta,
            downsample=int(self.downsample),
        )

        reset_epoch_end_buffers(self, log_only_mode=bool(log_only_mode))
