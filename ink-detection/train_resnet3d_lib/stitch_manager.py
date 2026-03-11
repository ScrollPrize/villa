from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

import train_resnet3d_lib.config as config_runtime
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


class StitchManager:
    def __init__(
        self,
        *,
        stitch_val_dataloader_idx=None,
        stitch_pred_shape=None,
        stitch_segment_id=None,
        stitch_all_val=None,
        stitch_downsample=None,
        stitch_all_val_shapes=None,
        stitch_all_val_segment_ids=None,
        stitch_train_shapes=None,
        stitch_train_segment_ids=None,
        stitch_use_roi=None,
        stitch_val_bboxes=None,
        stitch_train_bboxes=None,
        stitch_log_only_shapes=None,
        stitch_log_only_segment_ids=None,
        stitch_log_only_bboxes=None,
        stitch_log_only_downsample=None,
        stitch_log_only_every_n_epochs=None,
        stitch_train=None,
        stitch_train_every_n_epochs=None,
        train_loaders=None,
        log_only_loaders=None,
        train_borders=None,
        val_borders=None,
    ):
        cfg = getattr(config_runtime, "CFG", None)
        self.downsample = int(
            getattr(cfg, "stitch_downsample", 1) if stitch_downsample is None else stitch_downsample
        )
        self.log_only_downsample = int(
            getattr(cfg, "stitch_log_only_downsample", getattr(cfg, "stitch_downsample", 1))
            if stitch_log_only_downsample is None
            else stitch_log_only_downsample
        )
        self.use_roi = bool(
            getattr(cfg, "stitch_use_roi", False) if stitch_use_roi is None else stitch_use_roi
        )

        self._roi_meta_by_split = {"val": {}, "train": {}, "log_only": {}}
        self._roi_buffers_by_split = {"val": {}, "log_only": {}}
        self._val_loader_to_segment = {}

        self.stitch_val_dataloader_idx = stitch_val_dataloader_idx
        self.stitch_pred_shape = stitch_pred_shape
        self.stitch_segment_id = stitch_segment_id
        self.stitch_all_val = bool(
            getattr(cfg, "stitch_all_val", False) if stitch_all_val is None else stitch_all_val
        )
        self.stitch_all_val_shapes = list(stitch_all_val_shapes or [])
        self.stitch_all_val_segment_ids = [str(x) for x in (stitch_all_val_segment_ids or [])]
        self.stitch_train_shapes = list(stitch_train_shapes or [])
        self.stitch_train_segment_ids = [str(x) for x in (stitch_train_segment_ids or [])]
        self.train_enabled = bool(getattr(cfg, "stitch_train", False) if stitch_train is None else stitch_train)
        self.train_every_n_epochs = int(
            getattr(cfg, "stitch_train_every_n_epochs", 1)
            if stitch_train_every_n_epochs is None
            else stitch_train_every_n_epochs
        )
        self.stitch_log_only_shapes = list(stitch_log_only_shapes or [])
        self.stitch_log_only_segment_ids = [str(x) for x in (stitch_log_only_segment_ids or [])]
        self.train_segment_ids = list(self.stitch_train_segment_ids)
        self.train_loaders = list(train_loaders or [])
        self.log_only_segment_ids = list(self.stitch_log_only_segment_ids)
        self.log_only_loaders = list(log_only_loaders or [])
        self.log_only_every_n_epochs = int(
            getattr(cfg, "stitch_log_only_every_n_epochs", 10)
            if stitch_log_only_every_n_epochs is None
            else stitch_log_only_every_n_epochs
        )

        self.borders_by_split = {
            "train": dict(train_borders or {}),
            "val": dict(val_borders or {}),
        }
        self.val_bboxes = dict(stitch_val_bboxes or {})
        self.train_bboxes = dict(stitch_train_bboxes or {})
        self.log_only_bboxes = dict(stitch_log_only_bboxes or {})

        self._gaussian_cache = {}
        self._gaussian_sigma_scale = 1.0 / 8.0
        self._gaussian_min_weight = 1e-6

        self._validate_segment_shape_pairs()
        self._register_initial_segments()

    def _validate_segment_shape_pairs(self) -> None:
        segment_shape_pairs = (
            ("stitch_all_val", self.stitch_all_val_segment_ids, self.stitch_all_val_shapes),
            ("stitch_train", self.stitch_train_segment_ids, self.stitch_train_shapes),
            ("stitch_log_only", self.stitch_log_only_segment_ids, self.stitch_log_only_shapes),
        )
        for name, segment_ids, shapes in segment_shape_pairs:
            if len(segment_ids) != len(shapes):
                raise ValueError(
                    f"{name} segment ids/shapes length mismatch: ids={len(segment_ids)} shapes={len(shapes)}"
                )

    def _register_initial_segments(self) -> None:
        self._register_val_segments()
        self._register_train_segments()
        self._register_log_only_segments()
        self.enabled = len(self._roi_buffers_by_split["val"]) > 0

    def _register_segment(self, split: str, segment_id: str, shape, bbox, downsample: int) -> None:
        split_name = str(split)
        sid = str(segment_id)
        self._roi_meta_by_split[split_name][sid] = build_segment_roi_meta(
            shape,
            bbox,
            downsample,
            use_roi=self.use_roi,
        )
        if split_name in self._roi_buffers_by_split:
            self._roi_buffers_by_split[split_name][sid] = allocate_segment_buffers(
                self._roi_meta_by_split[split_name][sid]
            )

    def _register_val_segments(self) -> None:
        if self.stitch_all_val:
            for loader_idx, (segment_id, shape) in enumerate(
                zip(self.stitch_all_val_segment_ids, self.stitch_all_val_shapes)
            ):
                sid = str(segment_id)
                if sid in self._roi_meta_by_split["val"]:
                    raise ValueError(f"duplicate val stitch segment id: {sid!r}")
                self._register_segment(
                    "val",
                    sid,
                    shape,
                    self.val_bboxes.get(sid),
                    self.downsample,
                )
                self._val_loader_to_segment[int(loader_idx)] = sid
            return

        if self.stitch_val_dataloader_idx is None or self.stitch_pred_shape is None:
            return

        sid = str(
            self.stitch_segment_id
            if self.stitch_segment_id is not None
            else self.stitch_val_dataloader_idx
        )
        self._register_segment(
            "val",
            sid,
            self.stitch_pred_shape,
            self.val_bboxes.get(sid),
            self.downsample,
        )
        self._val_loader_to_segment[int(self.stitch_val_dataloader_idx)] = sid

    def _register_train_segments(self) -> None:
        for segment_id, shape in zip(
            self.stitch_train_segment_ids,
            self.stitch_train_shapes,
        ):
            sid = str(segment_id)
            if sid in self._roi_meta_by_split["train"]:
                raise ValueError(f"duplicate train stitch segment id: {sid!r}")
            self._register_segment(
                "train",
                sid,
                shape,
                self.train_bboxes.get(sid),
                self.downsample,
            )

    def _register_log_only_segments(self) -> None:
        for segment_id, shape in zip(
            self.stitch_log_only_segment_ids,
            self.stitch_log_only_shapes,
        ):
            sid = str(segment_id)
            if sid in self._roi_meta_by_split["log_only"]:
                raise ValueError(f"duplicate log-only stitch segment id: {sid!r}")
            self._register_segment(
                "log_only",
                sid,
                shape,
                self.log_only_bboxes.get(sid),
                self.log_only_downsample,
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
        trainer = getattr(model, "trainer", None)
        if trainer is None:
            return nullcontext()
        strategy = getattr(trainer, "strategy", None)
        precision_plugin = getattr(strategy, "precision_plugin", None) if strategy is not None else None
        if precision_plugin is None:
            precision_plugin = getattr(trainer, "precision_plugin", None)
        if precision_plugin is not None and hasattr(precision_plugin, "forward_context"):
            return precision_plugin.forward_context()
        return nullcontext()

    def _distributed_world_size(self, model) -> int:
        trainer = getattr(model, "trainer", None)
        if trainer is None:
            return 1
        return int(getattr(trainer, "world_size", 1) or 1)

    def _reduce_sum_distributed(self, model, tensor):
        strategy = getattr(getattr(model, "trainer", None), "strategy", None)
        if strategy is None or not hasattr(strategy, "reduce"):
            raise RuntimeError("distributed stitch reduction requested but trainer.strategy.reduce is unavailable")
        return strategy.reduce(tensor, reduce_op="sum")

    def _reset_buffers_for_split(self, split: str) -> None:
        buffers_by_segment = self._roi_buffers_by_split.get(str(split))
        if not buffers_by_segment:
            return
        for roi_buffers in buffers_by_segment.values():
            for pred_buf, count_buf, _offset in roi_buffers:
                pred_buf.fill(0)
                count_buf.fill(0)

    def _compose_segment_from_roi_buffers(self, roi_buffers, full_shape):
        return _compose_segment_from_roi_buffers(roi_buffers, full_shape)

    def sync_val_buffers_distributed(self, model) -> bool:
        if self._distributed_world_size(model) <= 1:
            return False

        device = model.device
        for roi_buffers in self._roi_buffers_by_split["val"].values():
            for pred_buf, count_buf, _offset in roi_buffers:
                pred_tensor = torch.from_numpy(np.ascontiguousarray(pred_buf)).to(device=device, dtype=torch.float32)
                count_tensor = torch.from_numpy(np.ascontiguousarray(count_buf)).to(device=device, dtype=torch.float32)
                pred_tensor = self._reduce_sum_distributed(model, pred_tensor)
                count_tensor = self._reduce_sum_distributed(model, count_tensor)
                pred_buf[...] = pred_tensor.detach().cpu().numpy()
                count_buf[...] = count_tensor.detach().cpu().numpy()
        return True

    def _build_validation_epoch_context(self, model) -> dict[str, bool]:
        trainer = getattr(model, "trainer", None)
        sanity_checking = bool(trainer is not None and getattr(trainer, "sanity_checking", False))
        is_global_zero = bool(trainer is None or trainer.is_global_zero)

        train_configured = bool(self.train_loaders) and bool(self.train_segment_ids) and bool(
            self._roi_meta_by_split["train"]
        )
        log_only_configured = bool(self.log_only_loaders) and bool(self.log_only_segment_ids) and bool(
            self._roi_buffers_by_split["log_only"]
        )

        return {
            "sanity_checking": sanity_checking,
            "is_global_zero": is_global_zero,
            "stitch_train_mode": bool(self.train_enabled and train_configured),
            "log_only_mode": bool(log_only_configured),
        }

    def _maybe_run_train_stitch(self, model, *, stitch_train_mode: bool, sanity_checking: bool, is_global_zero: bool):
        did_run_train_stitch = False
        train_segment_viz = {}
        if stitch_train_mode and not is_global_zero:
            stitch_train_mode = False
        if stitch_train_mode and not sanity_checking:
            train_segment_viz = self.run_train_stitch_pass(model) or {}
            did_run_train_stitch = bool(train_segment_viz)
            if not did_run_train_stitch:
                stitch_train_mode = False
        return stitch_train_mode, did_run_train_stitch, train_segment_viz

    def _maybe_run_log_only_stitch(self, model, *, log_only_mode: bool, sanity_checking: bool, is_global_zero: bool):
        did_run_log_only = False
        if log_only_mode and not is_global_zero:
            self._reset_buffers_for_split("log_only")
            log_only_mode = False
        if log_only_mode and not sanity_checking:
            did_run_log_only = bool(self.run_log_only_stitch_pass(model))
            if not did_run_log_only:
                log_only_mode = False
        return log_only_mode

    def _collect_val_segments_for_logging(self):
        segment_to_val = {}
        segment_to_val_meta = {}
        for _loader_idx, segment_id in self._val_loader_to_segment.items():
            sid = str(segment_id)
            roi_buffers = self._roi_buffers_by_split["val"].get(sid)
            if not roi_buffers:
                continue
            meta = self._roi_meta_by_split["val"].get(sid, {})
            full_shape = tuple(meta.get("full_shape", roi_buffers[0][0].shape))
            base, has = _compose_segment_from_roi_buffers(roi_buffers, full_shape)
            segment_to_val[sid] = (base, has)
            segment_to_val_meta[sid] = {
                "offset": (0, 0),
                "full_shape": full_shape,
            }
        return segment_to_val, segment_to_val_meta

    def _sync_val_buffers_and_maybe_exit_nonzero_worker(self, model, *, is_global_zero: bool) -> bool:
        did_sync = self.sync_val_buffers_distributed(model)
        if did_sync and not is_global_zero:
            self._reset_buffers_for_split("val")
            return True
        return False

    def _reset_epoch_end_buffers(self, *, log_only_mode: bool) -> None:
        self._reset_buffers_for_split("val")
        if log_only_mode:
            self._reset_buffers_for_split("log_only")

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

        epoch_ctx = self._build_validation_epoch_context(model)
        sanity_checking = bool(epoch_ctx["sanity_checking"])
        is_global_zero = bool(epoch_ctx["is_global_zero"])

        stitch_train_mode, did_run_train_stitch, train_segment_viz = self._maybe_run_train_stitch(
            model,
            stitch_train_mode=bool(epoch_ctx["stitch_train_mode"]),
            sanity_checking=sanity_checking,
            is_global_zero=is_global_zero,
        )
        log_only_mode = self._maybe_run_log_only_stitch(
            model,
            log_only_mode=bool(epoch_ctx["log_only_mode"]),
            sanity_checking=sanity_checking,
            is_global_zero=is_global_zero,
        )

        if self._sync_val_buffers_and_maybe_exit_nonzero_worker(
            model,
            is_global_zero=is_global_zero,
        ):
            return

        segment_to_val, segment_to_val_meta = self._collect_val_segments_for_logging()
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
            compose_segment_from_roi_buffers=_compose_segment_from_roi_buffers,
        )

        log_stitched_validation_metrics(
            model=model,
            sanity_checking=sanity_checking,
            segment_to_val=segment_to_val,
            segment_to_val_meta=segment_to_val_meta,
            downsample=int(self.downsample),
        )

        self._reset_epoch_end_buffers(log_only_mode=bool(log_only_mode))
