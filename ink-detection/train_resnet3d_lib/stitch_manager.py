import numpy as np

from train_resnet3d_lib.stitching.buffer_ops import (
    accumulate_to_buffers as _accumulate_to_buffers,
    compose_segment_from_roi_buffers as _compose_segment_from_roi_buffers,
)
from train_resnet3d_lib.stitching.epoch_passes import (
    run_log_only_stitch_pass as _run_log_only_stitch_pass,
    run_train_stitch_pass as _run_train_stitch_pass,
)
from train_resnet3d_lib.stitching.manager_runtime import (
    build_validation_epoch_context,
    collect_val_segments_for_logging,
    distributed_world_size,
    maybe_run_log_only_stitch,
    maybe_run_train_stitch,
    precision_context,
    reduce_sum_distributed,
    reset_epoch_end_buffers,
    reset_split_buffers,
    sync_val_buffers_and_maybe_exit_nonzero_worker,
    sync_val_buffers_distributed,
)
from train_resnet3d_lib.stitching.manager_setup import (
    initialize_manager_state,
    register_initial_segments,
)
from train_resnet3d_lib.stitching.metrics_runtime import log_stitched_validation_metrics
from train_resnet3d_lib.stitching.roi_layout import allocate_segment_buffers, build_segment_roi_meta
from train_resnet3d_lib.stitching.wandb_media import log_stitched_wandb_media


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

    def _distributed_world_size(self, model):
        return distributed_world_size(model)

    def _precision_context(self, model):
        return precision_context(model)

    def _reset_buffers_for_split(self, split):
        reset_split_buffers(self, split)

    def _reduce_sum_distributed(self, model, tensor):
        return reduce_sum_distributed(model, tensor)

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


__all__ = ["StitchManager"]
