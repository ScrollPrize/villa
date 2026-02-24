import time
import os.path as osp
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.optim import AdamW, SGD
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from metrics import StreamingBinarySegmentationMetrics

from train_resnet3d_lib.group_dro import GroupDROComputer
from warmup_scheduler import GradualWarmupScheduler
from models.resnetall import generate_model
# from models.i3dallnl import InceptionI3d
from train_resnet3d_lib.config import CFG, log


def _pick_group_norm_groups(num_channels: int, desired_groups: int) -> int:
    num_channels = int(num_channels)
    desired_groups = int(desired_groups)
    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    desired_groups = max(1, min(desired_groups, num_channels))
    for g in range(desired_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def replace_batchnorm_with_groupnorm(module: nn.Module, *, desired_groups: int = 32) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = int(child.num_features)
            groups = _pick_group_norm_groups(num_channels, desired_groups)
            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels, affine=True)
            if getattr(child, "affine", False):
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            replace_batchnorm_with_groupnorm(child, desired_groups=desired_groups)
    return module


class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale, *, norm="batch", group_norm_groups=32):
        super().__init__()
        norm = str(norm).lower()
        if norm not in {"batch", "group"}:
            raise ValueError(f"Unknown norm: {norm!r}")

        def _norm2d(num_channels: int) -> nn.Module:
            if norm == "group":
                groups = _pick_group_norm_groups(num_channels, int(group_norm_groups))
                return nn.GroupNorm(num_groups=groups, num_channels=int(num_channels))
            return nn.BatchNorm2d(int(num_channels))

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
                _norm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


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
        self.downsample = max(1, int(stitch_downsample or 1))
        self.buffers = {}
        self.segment_ids = {}
        self.buffer_meta = {}
        self.train_buffer_meta = {}
        self.train_segment_ids = []
        self.train_loaders = []
        self.train_enabled = bool(stitch_train)
        self.train_every_n_epochs = max(1, int(stitch_train_every_n_epochs or 1))
        self.log_only_buffers = {}
        self.log_only_buffer_meta = {}
        self.log_only_segment_ids = []
        self.log_only_loaders = []
        self.log_only_every_n_epochs = max(1, int(stitch_log_only_every_n_epochs or 10))
        self.log_only_downsample = int(stitch_log_only_downsample or self.downsample)
        self.borders_by_split = {"train": {}, "val": {}}
        self.use_roi = bool(stitch_use_roi)
        self.val_bboxes = dict(stitch_val_bboxes or {})
        self.train_bboxes = dict(stitch_train_bboxes or {})
        self.log_only_bboxes = dict(stitch_log_only_bboxes or {})
        self._gaussian_cache = {}
        self._gaussian_sigma_scale = 0.125
        self._gaussian_min_weight = 1e-6

        def _resolve_roi(shape, bbox, ds):
            h = int(shape[0])
            w = int(shape[1])
            ds = max(1, int(ds))
            ds_h = (h + ds - 1) // ds
            ds_w = (w + ds - 1) // ds
            if self.use_roi and bbox is not None:
                if len(bbox) != 4:
                    raise ValueError(f"stitch ROI bbox must be [y0, y1, x0, x1], got {bbox!r}")
                y0, y1, x0, x1 = [int(v) for v in bbox]
                y0 = max(0, min(y0, ds_h))
                y1 = max(0, min(y1, ds_h))
                x0 = max(0, min(x0, ds_w))
                x1 = max(0, min(x1, ds_w))
                if y1 > y0 and x1 > x0:
                    return (y0, y1, x0, x1), (ds_h, ds_w)
            return (0, ds_h, 0, ds_w), (ds_h, ds_w)

        if bool(stitch_all_val):
            if stitch_all_val_shapes is None or stitch_all_val_segment_ids is None:
                raise ValueError("stitch_all_val requires stitch_all_val_shapes and stitch_all_val_segment_ids")
            if len(stitch_all_val_shapes) != len(stitch_all_val_segment_ids):
                raise ValueError(
                    "stitch_all_val_shapes and stitch_all_val_segment_ids must have the same length "
                    f"(got {len(stitch_all_val_shapes)} vs {len(stitch_all_val_segment_ids)})"
                )

            for loader_idx, (segment_id, shape) in enumerate(zip(stitch_all_val_segment_ids, stitch_all_val_shapes)):
                bbox = self.val_bboxes.get(str(segment_id))
                (y0, y1, x0, x1), (ds_h, ds_w) = _resolve_roi(shape, bbox, self.downsample)
                buf_h = max(1, int(y1 - y0))
                buf_w = max(1, int(x1 - x0))
                self.buffers[int(loader_idx)] = (
                    np.zeros((buf_h, buf_w), dtype=np.float32),
                    np.zeros((buf_h, buf_w), dtype=np.float32),
                )
                self.segment_ids[int(loader_idx)] = str(segment_id)
                self.buffer_meta[int(loader_idx)] = {
                    "offset": (int(y0), int(x0)),
                    "full_shape": (int(ds_h), int(ds_w)),
                }
        else:
            stitch_enabled = (stitch_val_dataloader_idx is not None) and (stitch_pred_shape is not None)
            if stitch_enabled:
                idx = int(stitch_val_dataloader_idx)
                segment_id = str(stitch_segment_id or idx)
                bbox = self.val_bboxes.get(segment_id)
                (y0, y1, x0, x1), (ds_h, ds_w) = _resolve_roi(stitch_pred_shape, bbox, self.downsample)
                buf_h = max(1, int(y1 - y0))
                buf_w = max(1, int(x1 - x0))
                self.buffers[idx] = (
                    np.zeros((buf_h, buf_w), dtype=np.float32),
                    np.zeros((buf_h, buf_w), dtype=np.float32),
                )
                self.segment_ids[idx] = segment_id
                self.buffer_meta[idx] = {
                    "offset": (int(y0), int(x0)),
                    "full_shape": (int(ds_h), int(ds_w)),
                }

        if stitch_train_shapes is not None or stitch_train_segment_ids is not None:
            stitch_train_shapes = stitch_train_shapes or []
            stitch_train_segment_ids = stitch_train_segment_ids or []
            if len(stitch_train_shapes) != len(stitch_train_segment_ids):
                raise ValueError(
                    "stitch_train_shapes and stitch_train_segment_ids must have the same length "
                    f"(got {len(stitch_train_shapes)} vs {len(stitch_train_segment_ids)})"
                )

            for segment_id, shape in zip(stitch_train_segment_ids, stitch_train_shapes):
                seg_id = str(segment_id)
                bbox = self.train_bboxes.get(seg_id)
                (y0, y1, x0, x1), (ds_h, ds_w) = _resolve_roi(shape, bbox, self.downsample)
                buf_h = max(1, int(y1 - y0))
                buf_w = max(1, int(x1 - x0))
                self.train_buffer_meta[seg_id] = {
                    "offset": (int(y0), int(x0)),
                    "full_shape": (int(ds_h), int(ds_w)),
                    "buffer_shape": (int(buf_h), int(buf_w)),
                }
                self.train_segment_ids.append(seg_id)

        if stitch_log_only_shapes is not None and stitch_log_only_segment_ids is not None:
            if len(stitch_log_only_shapes) != len(stitch_log_only_segment_ids):
                raise ValueError(
                    "log-only stitch shapes/segment_ids length mismatch "
                    f"({len(stitch_log_only_shapes)} vs {len(stitch_log_only_segment_ids)})"
                )
            for segment_id, shape in zip(stitch_log_only_segment_ids, stitch_log_only_shapes):
                seg_id = str(segment_id)
                bbox = self.log_only_bboxes.get(seg_id)
                (y0, y1, x0, x1), (ds_h, ds_w) = _resolve_roi(shape, bbox, self.log_only_downsample)
                buf_h = max(1, int(y1 - y0))
                buf_w = max(1, int(x1 - x0))
                self.log_only_buffers[seg_id] = (
                    np.zeros((buf_h, buf_w), dtype=np.float32),
                    np.zeros((buf_h, buf_w), dtype=np.float32),
                )
                self.log_only_buffer_meta[seg_id] = {
                    "offset": (int(y0), int(x0)),
                    "full_shape": (int(ds_h), int(ds_w)),
                }
                self.log_only_segment_ids.append(seg_id)

        self.enabled = len(self.buffers) > 0

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
        trainer = getattr(model, "trainer", None)
        if trainer is None:
            return 1
        return int(getattr(trainer, "world_size", 1) or 1)

    def _reduce_sum_distributed(self, model, tensor):
        strategy = getattr(getattr(model, "trainer", None), "strategy", None)
        if strategy is None or not hasattr(strategy, "reduce"):
            raise RuntimeError("distributed stitch reduction requested but trainer.strategy.reduce is unavailable")
        return strategy.reduce(tensor, reduce_op="sum")

    def sync_val_buffers_distributed(self, model):
        if self._distributed_world_size(model) <= 1:
            return False
        device = model.device
        for pred_buf, count_buf in self.buffers.values():
            pred_tensor = torch.from_numpy(np.ascontiguousarray(pred_buf)).to(device=device, dtype=torch.float32)
            count_tensor = torch.from_numpy(np.ascontiguousarray(count_buf)).to(device=device, dtype=torch.float32)
            pred_tensor = self._reduce_sum_distributed(model, pred_tensor)
            count_tensor = self._reduce_sum_distributed(model, count_tensor)
            pred_buf[...] = pred_tensor.detach().cpu().numpy()
            count_buf[...] = count_tensor.detach().cpu().numpy()
        return True

    def _gaussian_weights(self, h: int, w: int) -> np.ndarray:
        h = int(h)
        w = int(w)
        if h <= 0 or w <= 0:
            raise ValueError(f"gaussian weight shape must be positive, got {(h, w)}")
        key = (h, w)
        weights = self._gaussian_cache.get(key)
        if weights is not None:
            return weights
        sigma_scale = float(self._gaussian_sigma_scale)
        if sigma_scale <= 0:
            raise ValueError(f"gaussian sigma scale must be > 0, got {sigma_scale}")
        sigma_y = max(float(h) * sigma_scale, 1.0)
        sigma_x = max(float(w) * sigma_scale, 1.0)
        y = np.arange(h, dtype=np.float32) - ((h - 1) / 2.0)
        x = np.arange(w, dtype=np.float32) - ((w - 1) / 2.0)
        wy = np.exp(-0.5 * (y / sigma_y) ** 2).astype(np.float32)
        wx = np.exp(-0.5 * (x / sigma_x) ** 2).astype(np.float32)
        weights = np.outer(wy, wx).astype(np.float32)
        peak = float(weights.max())
        if (not np.isfinite(peak)) or peak <= 0:
            raise ValueError(f"invalid gaussian peak for shape {(h, w)}: {peak}")
        weights /= peak
        min_weight = float(self._gaussian_min_weight)
        if min_weight <= 0:
            raise ValueError(f"gaussian min weight must be > 0, got {min_weight}")
        weights = np.clip(weights, min_weight, None, out=weights)
        self._gaussian_cache[key] = weights
        return weights

    def accumulate_to_buffers(self, *, outputs, xyxys, pred_buf, count_buf, offset=(0, 0), ds_override=None):
        ds = int(ds_override or self.downsample)
        y_preds = torch.sigmoid(outputs).to("cpu")
        buf_h, buf_w = pred_buf.shape[:2]
        off_y, off_x = offset
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            x1_ds = x1 // ds
            y1_ds = y1 // ds
            x2_ds = (x2 + ds - 1) // ds
            y2_ds = (y2 + ds - 1) // ds
            x1_ds -= int(off_x)
            x2_ds -= int(off_x)
            y1_ds -= int(off_y)
            y2_ds -= int(off_y)
            target_h = y2_ds - y1_ds
            target_w = x2_ds - x1_ds
            if target_h <= 0 or target_w <= 0:
                continue

            pred_patch = y_preds[i].unsqueeze(0).float()
            if pred_patch.shape[-2:] != (target_h, target_w):
                pred_patch = F.interpolate(
                    pred_patch,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            patch_weights = self._gaussian_weights(target_h, target_w)

            if x2_ds <= 0 or y2_ds <= 0 or x1_ds >= buf_w or y1_ds >= buf_h:
                continue

            y1_clamped = max(0, y1_ds)
            x1_clamped = max(0, x1_ds)
            y2_clamped = min(buf_h, y2_ds)
            x2_clamped = min(buf_w, x2_ds)

            py0 = y1_clamped - y1_ds
            px0 = x1_clamped - x1_ds
            py1 = py0 + (y2_clamped - y1_clamped)
            px1 = px0 + (x2_clamped - x1_clamped)
            if py1 <= py0 or px1 <= px0:
                continue

            pred_crop = pred_patch[..., py0:py1, px0:px1]
            weight_crop = patch_weights[py0:py1, px0:px1]
            pred_buf[y1_clamped:y2_clamped, x1_clamped:x2_clamped] += (
                pred_crop.squeeze(0).squeeze(0).numpy() * weight_crop
            )
            count_buf[y1_clamped:y2_clamped, x1_clamped:x2_clamped] += weight_crop

    def accumulate_val(self, *, outputs, xyxys, dataloader_idx):
        if not self.enabled:
            return
        idx = int(dataloader_idx)
        if idx not in self.buffers:
            return
        pred_buf, count_buf = self.buffers[idx]
        meta = self.buffer_meta.get(idx, {})
        offset = meta.get("offset", (0, 0))
        self.accumulate_to_buffers(outputs=outputs, xyxys=xyxys, pred_buf=pred_buf, count_buf=count_buf, offset=offset)

    def run_train_stitch_pass(self, model):
        if not self.train_enabled:
            return None
        if not self.train_loaders or not self.train_segment_ids:
            return None
        if len(self.train_loaders) != len(self.train_segment_ids):
            raise ValueError(
                "train stitch loaders/segment_ids length mismatch "
                f"({len(self.train_loaders)} vs {len(self.train_segment_ids)})"
            )

        epoch = int(getattr(model, "current_epoch", 0))
        if self.train_every_n_epochs > 1:
            if ((epoch + 1) % self.train_every_n_epochs) != 0:
                return None

        t0 = time.perf_counter()
        log(f"train stitch pass start epoch={epoch}")
        segment_viz = {}

        was_training = model.training
        precision_context = nullcontext()
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            strategy = getattr(trainer, "strategy", None)
            precision_plugin = getattr(strategy, "precision_plugin", None) if strategy is not None else None
            if precision_plugin is None:
                precision_plugin = getattr(trainer, "precision_plugin", None)
            if precision_plugin is not None and hasattr(precision_plugin, "forward_context"):
                precision_context = precision_plugin.forward_context()
        try:
            model.eval()
            with torch.inference_mode(), precision_context:
                for loader, segment_id in zip(self.train_loaders, self.train_segment_ids):
                    segment_id = str(segment_id)
                    meta = self.train_buffer_meta.get(segment_id)
                    if meta is None:
                        raise ValueError(f"Missing train stitch metadata for segment_id={segment_id!r}")
                    buf_h, buf_w = [int(v) for v in meta.get("buffer_shape", (0, 0))]
                    if buf_h <= 0 or buf_w <= 0:
                        raise ValueError(
                            f"Invalid train stitch buffer shape for segment_id={segment_id!r}: "
                            f"{meta.get('buffer_shape')!r}"
                        )
                    pred_buf = np.zeros((buf_h, buf_w), dtype=np.float32)
                    count_buf = np.zeros((buf_h, buf_w), dtype=np.float32)
                    offset = meta.get("offset", (0, 0))
                    for batch in loader:
                        x, _y, xyxys, _g = batch
                        x = x.to(model.device, non_blocking=True)
                        outputs = model(x)
                        self.accumulate_to_buffers(
                            outputs=outputs,
                            xyxys=xyxys,
                            pred_buf=pred_buf,
                            count_buf=count_buf,
                            offset=offset,
                            ds_override=self.downsample,
                        )
                    covered = count_buf > 0
                    covered_px = int(covered.sum())
                    total_px = int(covered.size)
                    coverage = float(covered_px) / float(max(1, total_px))
                    if covered_px > 0:
                        stitched = np.divide(
                            pred_buf.astype(np.float32),
                            count_buf.astype(np.float32),
                            out=np.zeros((buf_h, buf_w), dtype=np.float32),
                            where=covered,
                        )
                        vals = stitched[covered]
                        prob_mean = float(vals.mean()) if vals.size else float("nan")
                        prob_max = float(vals.max()) if vals.size else float("nan")
                    else:
                        stitched = np.zeros((buf_h, buf_w), dtype=np.float32)
                        prob_mean = float("nan")
                        prob_max = float("nan")
                    log(
                        f"train stitch summary segment={segment_id} "
                        f"coverage={coverage:.4f} covered_px={covered_px}/{total_px} "
                        f"prob_mean={prob_mean:.4f} prob_max={prob_max:.4f}"
                    )
                    segment_viz[segment_id] = {
                        "img_u8": (np.clip(stitched, 0.0, 1.0) * 255.0).astype(np.uint8),
                        "has": covered,
                        "meta": {
                            "offset": tuple(meta.get("offset", (0, 0))),
                            "full_shape": tuple(meta.get("full_shape", stitched.shape)),
                        },
                    }
        finally:
            if was_training:
                model.train()

        log(f"train stitch pass done epoch={epoch} elapsed={time.perf_counter() - t0:.1f}s")
        return segment_viz

    def run_log_only_stitch_pass(self, model):
        if not self.log_only_loaders or not self.log_only_segment_ids:
            return False
        if len(self.log_only_loaders) != len(self.log_only_segment_ids):
            raise ValueError(
                "log-only stitch loaders/segment_ids length mismatch "
                f"({len(self.log_only_loaders)} vs {len(self.log_only_segment_ids)})"
            )

        epoch = int(getattr(model, "current_epoch", 0))
        if self.log_only_every_n_epochs > 1:
            if ((epoch + 1) % self.log_only_every_n_epochs) != 0:
                return False

        t0 = time.perf_counter()
        log(f"log-only stitch pass start epoch={epoch}")

        for segment_id in self.log_only_segment_ids:
            if segment_id not in self.log_only_buffers:
                raise ValueError(f"Missing log-only stitch buffers for segment_id={segment_id!r}")
            pred_buf, count_buf = self.log_only_buffers[segment_id]
            pred_buf.fill(0)
            count_buf.fill(0)

        def _unpack(batch):
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    return batch[0], batch[1]
                if len(batch) >= 3:
                    return batch[0], batch[2]
            raise ValueError("log-only stitch batch must be (x, xyxys) or (x, y, xyxys, g)")

        was_training = model.training
        precision_context = nullcontext()
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            strategy = getattr(trainer, "strategy", None)
            precision_plugin = getattr(strategy, "precision_plugin", None) if strategy is not None else None
            if precision_plugin is None:
                precision_plugin = getattr(trainer, "precision_plugin", None)
            if precision_plugin is not None and hasattr(precision_plugin, "forward_context"):
                precision_context = precision_plugin.forward_context()
        try:
            model.eval()
            with torch.inference_mode(), precision_context:
                for loader, segment_id in zip(self.log_only_loaders, self.log_only_segment_ids):
                    pred_buf, count_buf = self.log_only_buffers[str(segment_id)]
                    meta = self.log_only_buffer_meta.get(str(segment_id), {})
                    offset = meta.get("offset", (0, 0))
                    for batch in loader:
                        x, xyxys = _unpack(batch)
                        x = x.to(model.device, non_blocking=True)
                        outputs = model(x)
                        self.accumulate_to_buffers(
                            outputs=outputs,
                            xyxys=xyxys,
                            pred_buf=pred_buf,
                            count_buf=count_buf,
                            offset=offset,
                            ds_override=self.log_only_downsample,
                        )
        finally:
            if was_training:
                model.train()

        log(f"log-only stitch pass done epoch={epoch} elapsed={time.perf_counter() - t0:.1f}s")
        return True

    def on_validation_epoch_end(self, model):
        if not self.enabled or not self.buffers:
            return

        sanity_checking = bool(model.trainer is not None and getattr(model.trainer, "sanity_checking", False))
        is_global_zero = bool(model.trainer is None or model.trainer.is_global_zero)
        train_configured = bool(self.train_loaders) and bool(self.train_segment_ids) and bool(self.train_buffer_meta)
        stitch_train_mode = bool(self.train_enabled and train_configured)
        log_only_configured = bool(self.log_only_loaders) and bool(self.log_only_segment_ids) and bool(self.log_only_buffers)

        did_run_train_stitch = False
        train_segment_viz = {}
        did_run_log_only = False
        if stitch_train_mode and (not is_global_zero):
            # Non-zero ranks skip the extra train stitch pass, but still need to
            # participate in distributed val-buffer reduction.
            stitch_train_mode = False
        if stitch_train_mode and (not sanity_checking):
            train_segment_viz = self.run_train_stitch_pass(model) or {}
            did_run_train_stitch = bool(train_segment_viz)
            if not did_run_train_stitch:
                # train stitching is enabled but only runs every N epochs; fall back to val-only logging.
                stitch_train_mode = False

        log_only_mode = bool(log_only_configured)
        if log_only_mode and (not is_global_zero):
            for pred_buf, count_buf in self.log_only_buffers.values():
                pred_buf.fill(0)
                count_buf.fill(0)
            log_only_mode = False
        if log_only_mode and (not sanity_checking):
            did_run_log_only = self.run_log_only_stitch_pass(model)
            if not did_run_log_only:
                log_only_mode = False

        did_sync_val_buffers = self.sync_val_buffers_distributed(model)
        if did_sync_val_buffers and (not is_global_zero):
            # Non-zero ranks are done once global stitched val buffers are reduced.
            for pred_buf, count_buf in self.buffers.values():
                pred_buf.fill(0)
                count_buf.fill(0)
            return

        log_train_stitch = bool(stitch_train_mode and did_run_train_stitch)

        segment_to_val = {}
        segment_to_val_meta = {}
        for loader_idx, (pred_buf, count_buf) in self.buffers.items():
            segment_id = self.segment_ids.get(loader_idx, str(loader_idx))
            stitched = np.divide(
                pred_buf,
                count_buf,
                out=np.zeros_like(pred_buf),
                where=count_buf != 0,
            )
            segment_id = str(segment_id)
            segment_to_val[segment_id] = (np.clip(stitched, 0, 1), (count_buf != 0))
            meta = self.buffer_meta.get(int(loader_idx), {})
            segment_to_val_meta[segment_id] = {
                "offset": meta.get("offset", (0, 0)),
                "full_shape": meta.get("full_shape", stitched.shape),
            }

        def _expand_to_full(img, has, meta):
            offset = meta.get("offset", (0, 0))
            full_shape = meta.get("full_shape", img.shape)
            if offset == (0, 0) and tuple(full_shape) == tuple(img.shape):
                return img, has
            y0, x0 = [int(v) for v in offset]
            full = np.zeros(tuple(full_shape), dtype=img.dtype)
            full_has = np.zeros(tuple(full_shape), dtype=bool)
            h, w = img.shape
            full[y0 : y0 + h, x0 : x0 + w] = img
            full_has[y0 : y0 + h, x0 : x0 + w] = has
            return full, full_has

        def _to_u8(img_float):
            return (np.clip(img_float, 0.0, 1.0) * 255.0).astype(np.uint8)

        def _add_borders_rgb(base_u8, segment_id):
            rgb = np.repeat(base_u8[..., None], 3, axis=2)
            train_border = self.borders_by_split.get("train", {}).get(str(segment_id))
            val_border = self.borders_by_split.get("val", {}).get(str(segment_id))
            if train_border is not None:
                rgb[train_border.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
            if val_border is not None:
                rgb[val_border.astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
            return rgb

        can_log_media = (
            (not sanity_checking)
            and (model.trainer is None or model.trainer.is_global_zero)
            and isinstance(model.logger, WandbLogger)
        )
        media_step = int(getattr(model.trainer, "global_step", 0)) if can_log_media else None

        if can_log_media:
            wandb_media_downsample = int(getattr(CFG, "eval_wandb_media_downsample", 1))
            if wandb_media_downsample < 1:
                raise ValueError(
                    "eval_wandb_media_downsample must be >= 1, "
                    f"got {wandb_media_downsample}"
                )

            def _downsample_for_wandb(img: np.ndarray, *, source_downsample: int) -> np.ndarray:
                source_downsample = int(source_downsample)
                if source_downsample < 1:
                    raise ValueError(f"source_downsample must be >= 1, got {source_downsample}")
                if source_downsample > 1 or wandb_media_downsample == 1:
                    return img
                return np.ascontiguousarray(img[::wandb_media_downsample, ::wandb_media_downsample])

            masks_logged = 0
            masks_log_only_logged = 0
            t0 = time.perf_counter()
            if log_train_stitch:
                segment_ids = sorted(set(segment_to_val.keys()) | set(train_segment_viz.keys()))
                for segment_id in segment_ids:
                    val_base = None
                    val_has = None
                    if segment_id in segment_to_val:
                        val_img, val_cov = segment_to_val[segment_id]
                        val_meta = segment_to_val_meta.get(segment_id, {})
                        val_base, val_has = _expand_to_full(val_img, val_cov, val_meta)
                        val_base = _to_u8(val_base)

                    train_base = None
                    train_has = None
                    if segment_id in train_segment_viz:
                        entry = train_segment_viz[segment_id]
                        train_base, train_has = _expand_to_full(entry["img_u8"], entry["has"], entry["meta"])

                    if val_base is None and train_base is None:
                        continue
                    if val_base is not None:
                        base_u8 = val_base.copy()
                    else:
                        base_u8 = train_base.copy()
                    if train_base is not None and train_has is not None:
                        base_u8[train_has] = train_base[train_has]
                    if val_base is not None and val_has is not None:
                        base_u8[val_has] = val_base[val_has]

                    image = _add_borders_rgb(base_u8, segment_id)
                    has_train = bool(train_has is not None and train_has.any())
                    has_val = bool(val_has is not None and val_has.any())
                    if has_train and has_val:
                        split_tag = "train+val"
                    elif has_train:
                        split_tag = "train"
                    elif has_val:
                        split_tag = "val"
                    else:
                        split_tag = "none"
                    image = _downsample_for_wandb(image, source_downsample=int(self.downsample))
                    safe_segment_id = str(segment_id).replace("/", "_")
                    model.logger.log_image(
                        key=f"masks/{safe_segment_id}",
                        images=[image],
                        caption=[f"{segment_id} ({split_tag} ds={self.downsample})"],
                    )
                    masks_logged += 1
            else:
                want_color = bool(self.train_enabled)
                for loader_idx, (pred_buf, count_buf) in self.buffers.items():
                    stitched = np.divide(
                        pred_buf,
                        count_buf,
                        out=np.zeros_like(pred_buf),
                        where=count_buf != 0,
                    )
                    segment_id = str(self.segment_ids.get(loader_idx, str(loader_idx)))
                    base = np.clip(stitched, 0, 1)
                    base, _ = _expand_to_full(base, count_buf != 0, segment_to_val_meta.get(segment_id, {}))
                    base_u8 = _to_u8(base)
                    if want_color:
                        image = _add_borders_rgb(base_u8, segment_id)
                    else:
                        image = base_u8
                    image = _downsample_for_wandb(image, source_downsample=int(self.downsample))
                    safe_segment_id = str(segment_id).replace("/", "_")
                    model.logger.log_image(
                        key=f"masks/{safe_segment_id}",
                        images=[image],
                        caption=[f"{segment_id} (val ds={self.downsample})"],
                    )
                    masks_logged += 1

            if log_only_mode:
                for segment_id in self.log_only_segment_ids:
                    pred_buf, count_buf = self.log_only_buffers[str(segment_id)]
                    stitched = np.divide(
                        pred_buf,
                        count_buf,
                        out=np.zeros_like(pred_buf),
                        where=count_buf != 0,
                    )
                    meta = self.log_only_buffer_meta.get(str(segment_id), {})
                    base = np.clip(stitched, 0, 1)
                    base, _ = _expand_to_full(base, count_buf != 0, meta)
                    image = _downsample_for_wandb(_to_u8(base), source_downsample=int(self.log_only_downsample))
                    safe_segment_id = str(segment_id).replace("/", "_")
                    model.logger.log_image(
                        key=f"masks_log_only/{safe_segment_id}",
                        images=[image],
                        caption=[f"{segment_id} (log-only ds={self.log_only_downsample})"],
                    )
                    masks_log_only_logged += 1

            log(
                f"wandb media done step={int(media_step)} "
                f"masks={masks_logged} masks_log_only={masks_log_only_logged} "
                f"elapsed={time.perf_counter() - t0:.2f}s"
            )

        if (not sanity_checking) and (model.trainer is None or model.trainer.is_global_zero):
            current_epoch = int(getattr(getattr(model, "trainer", None), "current_epoch", 0))
            eval_epoch = int(current_epoch + 1)
            total_epochs = int(getattr(CFG, "epochs"))
            is_final_eval_epoch = bool(eval_epoch == total_epochs)
            should_run_stitch_metrics = bool(getattr(CFG, "eval_stitch_metrics", True)) and bool(segment_to_val)
            if should_run_stitch_metrics:
                stitch_every_n_epochs = max(1, int(getattr(CFG, "eval_stitch_every_n_epochs", 1)))
                stitch_plus_one = bool(getattr(CFG, "eval_stitch_every_n_epochs_plus_one", False))
                if stitch_plus_one and stitch_every_n_epochs > 1:
                    mod = eval_epoch % stitch_every_n_epochs
                    should_run_epoch = (eval_epoch >= stitch_every_n_epochs) and (mod == 0 or mod == 1)
                else:
                    should_run_epoch = (eval_epoch % stitch_every_n_epochs) == 0
                if not should_run_epoch:
                    should_run_stitch_metrics = False
                    log(
                        f"skip stitched metrics epoch={current_epoch} "
                        f"eval_stitch_every_n_epochs={stitch_every_n_epochs} "
                        f"eval_stitch_every_n_epochs_plus_one={stitch_plus_one}"
                    )

            if should_run_stitch_metrics:
                from metrics.stitched_metrics import (
                    component_metric_specs,
                    compute_stitched_metrics,
                    summarize_component_rows,
                    write_global_component_manifest,
                )

                label_suffix = str(getattr(CFG, "val_label_suffix", "_val"))
                mask_suffix = str(getattr(CFG, "val_mask_suffix", "_val"))
                threshold = float(getattr(CFG, "eval_threshold", 0.5))
                betti_connectivity = 2
                drd_block_size = int(getattr(CFG, "eval_drd_block_size", 8))
                boundary_k = int(getattr(CFG, "eval_boundary_k", 3))
                component_worst_q = getattr(CFG, "eval_component_worst_q", 0.2)
                if isinstance(component_worst_q, str) and component_worst_q.strip().lower() in {"", "none", "null"}:
                    component_worst_q = None
                if component_worst_q is not None:
                    component_worst_q = float(component_worst_q)
                component_worst_k = getattr(CFG, "eval_component_worst_k", 2)
                if isinstance(component_worst_k, str) and component_worst_k.strip().lower() in {"", "none", "null"}:
                    component_worst_k = None
                if isinstance(component_worst_k, float) and float(component_worst_k).is_integer():
                    component_worst_k = int(component_worst_k)
                skeleton_thinning_type = str(getattr(CFG, "eval_skeleton_thinning_type", "guo_hall"))
                enable_skeleton_metrics = bool(getattr(CFG, "eval_enable_skeleton_metrics", True))
                component_min_area = int(getattr(CFG, "eval_component_min_area", 0) or 0)
                component_pad = int(getattr(CFG, "eval_component_pad", 5))
                enable_full_region_metrics = bool(getattr(CFG, "eval_stitch_full_region_metrics", False))
                save_stitch_debug_images = bool(getattr(CFG, "eval_save_stitch_debug_images", False))
                eval_topological_metrics_every_n_epochs = int(
                    getattr(CFG, "eval_topological_metrics_every_n_epochs", 1)
                )
                if eval_topological_metrics_every_n_epochs < 1:
                    raise ValueError(
                        "eval_topological_metrics_every_n_epochs must be >= 1, "
                        f"got {eval_topological_metrics_every_n_epochs}"
                    )
                eval_save_stitch_debug_images_every_n_epochs = int(
                    getattr(CFG, "eval_save_stitch_debug_images_every_n_epochs", 1)
                )
                if eval_save_stitch_debug_images_every_n_epochs < 1:
                    raise ValueError(
                        "eval_save_stitch_debug_images_every_n_epochs must be >= 1, "
                        f"got {eval_save_stitch_debug_images_every_n_epochs}"
                    )
                run_topological_metrics = bool(
                    (eval_epoch % eval_topological_metrics_every_n_epochs) == 0 or is_final_eval_epoch
                )
                save_stitch_debug_images_now = bool(
                    save_stitch_debug_images
                    and (
                        (eval_epoch % eval_save_stitch_debug_images_every_n_epochs) == 0
                        or is_final_eval_epoch
                    )
                )
                stitched_inputs_output_dir = osp.join(
                    str(getattr(CFG, "figures_dir", ".")),
                    "metrics_stitched_debug",
                )
                if not save_stitch_debug_images_now:
                    stitched_inputs_output_dir = None
                if not run_topological_metrics:
                    log(
                        f"skip topological stitched metrics epoch={current_epoch} "
                        f"eval_epoch={eval_epoch} "
                        f"eval_topological_metrics_every_n_epochs={eval_topological_metrics_every_n_epochs}"
                    )
                if save_stitch_debug_images and (not save_stitch_debug_images_now):
                    log(
                        f"skip stitched debug inputs epoch={current_epoch} "
                        f"eval_epoch={eval_epoch} "
                        "eval_save_stitch_debug_images_every_n_epochs="
                        f"{eval_save_stitch_debug_images_every_n_epochs}"
                    )

                def _parse_list(value, cast_fn):
                    if value is None:
                        return None
                    if isinstance(value, str):
                        parts = [p.strip() for p in value.replace(";", ",").split(",")]
                        return [cast_fn(p) for p in parts if p]
                    if isinstance(value, (list, tuple, np.ndarray)):
                        return [cast_fn(v) for v in value]
                    return [cast_fn(value)]

                boundary_tols = _parse_list(getattr(CFG, "eval_boundary_tols", None), float)

                threshold_grid = _parse_list(getattr(CFG, "eval_threshold_grid", None), float)
                if threshold_grid is None:
                    tmin = float(getattr(CFG, "eval_threshold_grid_min", 0.40))
                    tmax = float(getattr(CFG, "eval_threshold_grid_max", 0.70))
                    steps = int(getattr(CFG, "eval_threshold_grid_steps", 5))
                    if steps >= 2:
                        threshold_grid = np.linspace(tmin, tmax, steps).tolist()

                from train_resnet3d_lib.val_stitch_wandb import rewrite_val_stitch_metric_key

                stitch_group_idx_by_segment = getattr(model, "_stitch_group_idx_by_segment", None)
                if not isinstance(stitch_group_idx_by_segment, dict):
                    raise TypeError(
                        "model._stitch_group_idx_by_segment must be a dict mapping segment id to group index"
                    )
                stability_metric_directions = {
                    str(metric_name): bool(higher_is_better)
                    for metric_name, higher_is_better in component_metric_specs(
                        enable_skeleton_metrics=enable_skeleton_metrics,
                        include_cadenced_metrics=run_topological_metrics,
                    )
                }
                group_stability_values = {
                    metric_name: {int(group_idx): [] for group_idx in range(int(model.n_groups))}
                    for metric_name in stability_metric_directions
                }

                global_component_rows = []
                for segment_id, (pred_prob, pred_has) in segment_to_val.items():
                    segment_id_key = str(segment_id)
                    if segment_id_key not in stitch_group_idx_by_segment:
                        raise KeyError(f"missing stitched group index for segment_id={segment_id_key!r}")
                    segment_group_idx = int(stitch_group_idx_by_segment[segment_id_key])
                    if segment_group_idx < 0 or segment_group_idx >= int(model.n_groups):
                        raise ValueError(
                            f"invalid stitched group index for segment_id={segment_id_key!r}: {segment_group_idx}"
                        )

                    meta = segment_to_val_meta.get(segment_id_key, {})
                    roi_offset = meta.get("offset", (0, 0))
                    cache_max = int(max(1, len(segment_to_val))) if segment_to_val else 1
                    metrics = compute_stitched_metrics(
                        fragment_id=segment_id,
                        pred_prob=pred_prob,
                        pred_has=pred_has,
                        label_suffix=label_suffix,
                        mask_suffix=mask_suffix,
                        downsample=self.downsample,
                        roi_offset=roi_offset,
                        threshold=threshold,
                        betti_connectivity=betti_connectivity,
                        drd_block_size=drd_block_size,
                        boundary_k=boundary_k,
                        boundary_tols=boundary_tols,
                        component_worst_q=component_worst_q,
                        component_worst_k=component_worst_k,
                        component_min_area=component_min_area,
                        component_pad=component_pad,
                        skeleton_method=skeleton_thinning_type,
                        enable_skeleton_metrics=enable_skeleton_metrics,
                        include_cadenced_metrics=run_topological_metrics,
                        enable_full_region_metrics=enable_full_region_metrics,
                        threshold_grid=threshold_grid,
                        stitched_inputs_output_dir=stitched_inputs_output_dir,
                        gt_cache_max=cache_max,
                        component_rows_collector=global_component_rows,
                        eval_epoch=eval_epoch,
                    )

                    safe_segment_id = segment_id_key.replace("/", "_")
                    base_key = f"metrics/val_stitch/segments/{safe_segment_id}"
                    if not isinstance(metrics, dict):
                        raise TypeError(
                            f"compute_stitched_metrics must return a dict, got {type(metrics).__name__}"
                        )

                    for k, v in metrics.items():
                        metric_key = rewrite_val_stitch_metric_key(str(k))
                        model.log(f"{base_key}/{metric_key}", v, on_epoch=True, prog_bar=False)
                        prefix = "stability/"
                        suffix = "/mean"
                        if metric_key.startswith(prefix) and metric_key.endswith(suffix):
                            metric_name = metric_key[len(prefix) : -len(suffix)]
                            if metric_name in group_stability_values:
                                group_stability_values[metric_name][segment_group_idx].append(float(v))

                if len(model.group_names) != int(model.n_groups):
                    raise ValueError(
                        f"group_names length must match n_groups: {len(model.group_names)} vs {int(model.n_groups)}"
                    )
                for metric_name, higher_is_better in stability_metric_directions.items():
                    present_group_means = []
                    for group_idx in range(int(model.n_groups)):
                        values = np.asarray(group_stability_values[metric_name][group_idx], dtype=np.float64)
                        finite_values = values[np.isfinite(values)]
                        if finite_values.size == 0:
                            continue
                        group_mean = float(finite_values.mean())
                        present_group_means.append(group_mean)
                        safe_group_name = str(model.group_names[group_idx]).replace("/", "_")
                        model.log(
                            f"metrics/val_stitch/groups/group_{group_idx}_{safe_group_name}/stability/{metric_name}/mean",
                            group_mean,
                            on_epoch=True,
                            prog_bar=False,
                        )
                    if not present_group_means:
                        continue
                    worst_group_mean = min(present_group_means) if higher_is_better else max(present_group_means)
                    model.log(
                        f"metrics/val_stitch/worst_group/stability/{metric_name}/mean",
                        float(worst_group_mean),
                        on_epoch=True,
                        prog_bar=False,
                    )

                if global_component_rows:
                    global_stats, global_rankings = summarize_component_rows(
                        global_component_rows,
                        worst_q=component_worst_q,
                        worst_k=component_worst_k,
                        id_key="global_component_id",
                        metric_specs=component_metric_specs(
                            enable_skeleton_metrics=enable_skeleton_metrics,
                            include_cadenced_metrics=run_topological_metrics,
                        ),
                    )
                    for metric_name, stats in global_stats.items():
                        for stat_name, stat_val in stats.items():
                            model.log(
                                f"metrics/val_stitch/global/components/{metric_name}/{stat_name}",
                                float(stat_val),
                                on_epoch=True,
                                prog_bar=False,
                            )
                    manifest_path = None
                    manifest_output_dir = stitched_inputs_output_dir
                    if manifest_output_dir:
                        manifest_path = write_global_component_manifest(
                            component_rows=global_component_rows,
                            output_dir=manifest_output_dir,
                            downsample=self.downsample,
                            worst_k=component_worst_k,
                            worst_q=component_worst_q,
                            rankings=global_rankings,
                        )
                    if isinstance(model.logger, WandbLogger):
                        run = model.logger.experiment
                        if manifest_path is not None:
                            run.summary["metrics/val_stitch/global/diagnostics/components/manifest_path"] = str(
                                manifest_path
                            )
                        for metric_name, ranking in global_rankings.items():
                            k_ids = ranking["worst_k_component_ids"]
                            q_ids = ranking["worst_q_component_ids"]
                            run.summary[
                                f"metrics/val_stitch/global/diagnostics/components/{metric_name}/worst_k_component_ids"
                            ] = ",".join(str(v) for v in k_ids)
                            run.summary[
                                f"metrics/val_stitch/global/diagnostics/components/{metric_name}/worst_q_component_ids"
                            ] = ",".join(str(v) for v in q_ids)

        # reset stitch buffers
        for pred_buf, count_buf in self.buffers.values():
            pred_buf.fill(0)
            count_buf.fill(0)
        if log_only_mode:
            for pred_buf, count_buf in self.log_only_buffers.values():
                pred_buf.fill(0)
                count_buf.fill(0)

class RegressionPLModel(pl.LightningModule):
    def __init__(
        self,
        size=256,
        enc='',
        with_norm=False,
        objective="erm",
        loss_mode="batch",
        robust_step_size=None,
        group_counts=None,
        group_dro_gamma=0.1,
        group_dro_btl=False,
        group_dro_alpha=None,
        group_dro_normalize_loss=False,
        group_dro_min_var_weight=0.0,
        group_dro_adj=None,
        total_steps=780,
        n_groups=1,
        group_names=None,
        stitch_group_idx_by_segment=None,
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
        stitch_log_only_downsample=1,
        stitch_log_only_every_n_epochs=10,
        stitch_train=False,
        stitch_train_every_n_epochs=1,
        norm="batch",
        group_norm_groups=32,
        erm_group_topk=0,
    ):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()

        self.n_groups = int(n_groups)
        if group_names is None:
            group_names = [str(i) for i in range(self.n_groups)]
        self.group_names = list(group_names)
        if len(self.group_names) != self.n_groups:
            raise ValueError(f"group_names length must be {self.n_groups}, got {len(self.group_names)}")
        if stitch_group_idx_by_segment is None:
            self._stitch_group_idx_by_segment = {}
        else:
            if not isinstance(stitch_group_idx_by_segment, dict):
                raise TypeError(
                    "stitch_group_idx_by_segment must be a dict mapping segment id to group index, "
                    f"got {type(stitch_group_idx_by_segment).__name__}"
                )
            normalized_group_map = {}
            for segment_id, group_idx in stitch_group_idx_by_segment.items():
                segment_key = str(segment_id)
                group_idx_i = int(group_idx)
                if group_idx_i < 0 or group_idx_i >= self.n_groups:
                    raise ValueError(
                        f"stitch_group_idx_by_segment[{segment_key!r}]={group_idx_i} out of range [0, {self.n_groups})"
                    )
                normalized_group_map[segment_key] = group_idx_i
            self._stitch_group_idx_by_segment = normalized_group_map

        self.group_dro = None
        if str(self.hparams.objective).lower() == "group_dro":
            if robust_step_size is None:
                raise ValueError("group_dro.robust_step_size is required when training.objective is group_dro")
            if group_counts is None:
                raise ValueError("group_counts is required when training.objective is group_dro")

            self.group_dro = GroupDROComputer(
                n_groups=self.n_groups,
                group_counts=group_counts,
                alpha=group_dro_alpha,
                gamma=group_dro_gamma,
                adj=group_dro_adj,
                min_var_weight=group_dro_min_var_weight,
                step_size=robust_step_size,
                normalize_loss=group_dro_normalize_loss,
                btl=group_dro_btl,
            )

        self.erm_group_topk = int(erm_group_topk or 0)
        if self.erm_group_topk < 0:
            raise ValueError(f"erm_group_topk must be >= 0, got {self.erm_group_topk}")

        self._ema_decay = float(getattr(CFG, "ema_decay", 0.9))
        self._ema_metrics = {}

        # Evaluation-only metrics (kept separate from the optimization objective).
        self._eval_threshold = float(getattr(CFG, "eval_threshold", 0.5))
        self._val_eval_metrics = None

        self.loss_func1 = smp.losses.DiceLoss(mode="binary")
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)

        self.backbone = generate_model(
            model_depth=50,
            n_input_channels=1,
            forward_features=True,
            n_classes=1039,
        )

        norm = str(norm).lower()
        group_norm_groups = int(group_norm_groups)
        init_ckpt_path = getattr(CFG, "init_ckpt_path", None)
        if not init_ckpt_path:
            backbone_pretrained_path = getattr(CFG, "backbone_pretrained_path", "./r3d50_KM_200ep.pth")
            if not osp.exists(backbone_pretrained_path):
                raise FileNotFoundError(
                    f"Missing backbone pretrained weights: {backbone_pretrained_path}. "
                    "Either place r3d50_KM_200ep.pth next to train_resnet3d.py, set CFG.backbone_pretrained_path, "
                    "or pass --init_ckpt_path to fine-tune from a previous run."
                )
            backbone_ckpt = torch.load(backbone_pretrained_path, map_location="cpu")
            state_dict = backbone_ckpt.get("state_dict", backbone_ckpt)
            conv1_weight = state_dict["conv1.weight"]
            state_dict["conv1.weight"] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict, strict=False)

        if norm == "group":
            replace_batchnorm_with_groupnorm(self.backbone, desired_groups=group_norm_groups)

        was_training = self.backbone.training
        try:
            self.backbone.eval()
            with torch.no_grad():
                encoder_dims = [x.size(1) for x in self.backbone(torch.rand(1, 1, 20, 256, 256))]
        finally:
            if was_training:
                self.backbone.train()

        self.decoder = Decoder(encoder_dims=encoder_dims, upscale=1, norm=norm, group_norm_groups=group_norm_groups)

        if self.hparams.with_norm:
            if norm == "group":
                self.normalization = nn.GroupNorm(num_groups=1, num_channels=1)
            else:
                self.normalization = nn.BatchNorm3d(num_features=1)

        self._stitcher = StitchManager(
            stitch_val_dataloader_idx=stitch_val_dataloader_idx,
            stitch_pred_shape=stitch_pred_shape,
            stitch_segment_id=stitch_segment_id,
            stitch_all_val=bool(stitch_all_val),
            stitch_downsample=int(stitch_downsample or 1),
            stitch_all_val_shapes=stitch_all_val_shapes,
            stitch_all_val_segment_ids=stitch_all_val_segment_ids,
            stitch_train_shapes=stitch_train_shapes,
            stitch_train_segment_ids=stitch_train_segment_ids,
            stitch_use_roi=bool(stitch_use_roi),
            stitch_val_bboxes=stitch_val_bboxes,
            stitch_train_bboxes=stitch_train_bboxes,
            stitch_log_only_shapes=stitch_log_only_shapes,
            stitch_log_only_segment_ids=stitch_log_only_segment_ids,
            stitch_log_only_bboxes=stitch_log_only_bboxes,
            stitch_log_only_downsample=stitch_log_only_downsample,
            stitch_log_only_every_n_epochs=int(stitch_log_only_every_n_epochs or 10),
            stitch_train=bool(stitch_train),
            stitch_train_every_n_epochs=int(stitch_train_every_n_epochs or 1),
        )

    def set_stitch_borders(self, *, train_borders=None, val_borders=None):
        self._stitcher.set_borders(train_borders=train_borders, val_borders=val_borders)

    def set_train_stitch_loaders(self, loaders, segment_ids):
        self._stitcher.set_train_loaders(loaders, segment_ids)

    def set_log_only_stitch_loaders(self, loaders, segment_ids):
        self._stitcher.set_log_only_loaders(loaders, segment_ids)

    def _accumulate_stitch_predictions(self, *, outputs, xyxys, pred_buf, count_buf, offset=(0, 0)):
        self._stitcher.accumulate_to_buffers(
            outputs=outputs,
            xyxys=xyxys,
            pred_buf=pred_buf,
            count_buf=count_buf,
            offset=offset,
        )

    def on_train_epoch_start(self):
        device = self.device
        self._train_loss_sum = torch.tensor(0.0, device=device)
        self._train_dice_sum = torch.tensor(0.0, device=device)
        self._train_count = torch.tensor(0.0, device=device)

        self._train_group_loss_sum = torch.zeros(self.n_groups, device=device)
        self._train_group_dice_sum = torch.zeros(self.n_groups, device=device)
        self._train_group_count = torch.zeros(self.n_groups, device=device)

    def _update_train_stats(self, per_sample_loss, per_sample_dice, group_idx):
        self._train_loss_sum += per_sample_loss.sum()
        self._train_dice_sum += per_sample_dice.sum()
        self._train_count += float(per_sample_loss.numel())

        group_idx = group_idx.long()
        self._train_group_loss_sum.scatter_add_(0, group_idx, per_sample_loss)
        self._train_group_dice_sum.scatter_add_(0, group_idx, per_sample_dice)
        self._train_group_count.scatter_add_(
            0,
            group_idx,
            torch.ones_like(per_sample_loss, dtype=self._train_group_count.dtype),
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)

        return pred_mask

    def compute_per_sample_loss_and_dice(self, logits, targets):
        targets = targets.float()

        smooth_factor = 0.25
        soft_targets = (1.0 - targets) * smooth_factor + targets * (1.0 - smooth_factor)

        bce = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none")
        bce = bce.mean(dim=(1, 2, 3))

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        eps = 1e-7
        dice = (2 * intersection + eps) / (union + eps)

        dice_loss = 1.0 - dice
        per_sample_loss = 0.5 * dice_loss + 0.5 * bce
        return per_sample_loss, dice, bce, dice_loss

    def compute_group_avg(self, values, group_idx):
        group_idx = group_idx.long()
        group_map = (
            group_idx
            == torch.arange(self.n_groups, device=group_idx.device).unsqueeze(1).long()
        ).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_avg = (group_map @ values.view(-1)) / group_denom
        return group_avg, group_count

    def _update_ema_metric(self, name, value):
        decay = float(self._ema_decay)
        if torch.is_tensor(value):
            val = float(value.detach().cpu().item())
        else:
            val = float(value)
        prev = self._ema_metrics.get(name)
        if prev is None:
            ema = val
        else:
            ema = decay * prev + (1.0 - decay) * val
        self._ema_metrics[name] = ema
        self.log(f"{name}_ema", ema, on_step=False, on_epoch=True, prog_bar=False)

    def _distributed_world_size(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 1
        return int(getattr(trainer, "world_size", 1) or 1)

    def _reduce_sum_distributed(self, tensor):
        if self._distributed_world_size() <= 1:
            return tensor
        strategy = getattr(getattr(self, "trainer", None), "strategy", None)
        if strategy is None or not hasattr(strategy, "reduce"):
            raise RuntimeError("distributed validation reduction requested but trainer.strategy.reduce is unavailable")
        return strategy.reduce(tensor, reduce_op="sum")

    def _sync_validation_accumulators(self):
        if self._distributed_world_size() <= 1:
            return
        self._val_loss_sum = self._reduce_sum_distributed(self._val_loss_sum)
        self._val_dice_sum = self._reduce_sum_distributed(self._val_dice_sum)
        self._val_bce_sum = self._reduce_sum_distributed(self._val_bce_sum)
        self._val_dice_loss_sum = self._reduce_sum_distributed(self._val_dice_loss_sum)
        self._val_count = self._reduce_sum_distributed(self._val_count)
        self._val_group_loss_sum = self._reduce_sum_distributed(self._val_group_loss_sum)
        self._val_group_dice_sum = self._reduce_sum_distributed(self._val_group_dice_sum)
        self._val_group_bce_sum = self._reduce_sum_distributed(self._val_group_bce_sum)
        self._val_group_dice_loss_sum = self._reduce_sum_distributed(self._val_group_dice_loss_sum)
        self._val_group_count = self._reduce_sum_distributed(self._val_group_count)

    def training_step(self, batch, batch_idx):
        x, y, g = batch
        outputs = self(x)

        objective = str(self.hparams.objective).lower()
        loss_mode = str(self.hparams.loss_mode).lower()
        g = g.long()
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(outputs, y)

        if objective == "erm":
            if loss_mode == "batch":
                dice_loss = self.loss_func1(outputs, y)
                bce_loss = self.loss_func2(outputs, y)
                loss = 0.5 * dice_loss + 0.5 * bce_loss
                self.log("train/dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=False)
            elif loss_mode == "per_sample":
                if self.erm_group_topk > 0:
                    group_loss, group_count = self.compute_group_avg(per_sample_loss, g)
                    present = group_count > 0
                    if present.any():
                        present_losses = group_loss[present]
                        k = min(int(self.erm_group_topk), int(present_losses.numel()))
                        topk_losses, _ = torch.topk(present_losses, k, largest=True)
                        loss = topk_losses.mean()
                    else:
                        loss = per_sample_loss.mean()

                    if self.global_step % CFG.print_freq == 0:
                        if present.any():
                            worst_group_loss = group_loss[present].max()
                        else:
                            worst_group_loss = group_loss.max()
                        self.log("train/worst_group_loss", worst_group_loss, on_step=True, on_epoch=False, prog_bar=False)
                        for group_idx, group_name in enumerate(self.group_names):
                            safe_group_name = str(group_name).replace("/", "_")
                            self.log(
                                f"train/group_{group_idx}_{safe_group_name}/loss",
                                group_loss[group_idx],
                                on_step=True,
                                on_epoch=False,
                            )
                            self.log(
                                f"train/group_{group_idx}_{safe_group_name}/count",
                                group_count[group_idx],
                                on_step=True,
                                on_epoch=False,
                            )
                else:
                    loss = per_sample_loss.mean()
                self.log("train/dice", per_sample_dice.mean(), on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/dice_loss", per_sample_dice_loss.mean(), on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/bce_loss", per_sample_bce.mean(), on_step=True, on_epoch=True, prog_bar=False)
            else:
                raise ValueError(f"Unknown training.loss_mode: {self.hparams.loss_mode!r}")
        elif objective == "group_dro":
            if loss_mode != "per_sample":
                raise ValueError("GroupDRO requires training.loss_mode=per_sample")
            if self.group_dro is None:
                raise RuntimeError("GroupDRO objective was set but group_dro computer was not initialized")

            robust_loss, group_loss, group_count, _weights = self.group_dro.loss(per_sample_loss, g)
            loss = robust_loss
            self.log("train/dice", per_sample_dice.mean(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/dice_loss", per_sample_dice_loss.mean(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/bce_loss", per_sample_bce.mean(), on_step=True, on_epoch=True, prog_bar=False)

            if self.global_step % CFG.print_freq == 0:
                present = group_count > 0
                if present.any():
                    worst_group_loss = group_loss[present].max()
                else:
                    worst_group_loss = group_loss.max()
                self.log("train/worst_group_loss", worst_group_loss, on_step=True, on_epoch=False, prog_bar=False)

                for group_idx, group_name in enumerate(self.group_names):
                    safe_group_name = str(group_name).replace("/", "_")
                    self.log(
                        f"train/group_{group_idx}_{safe_group_name}/loss",
                        group_loss[group_idx],
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        f"train/group_{group_idx}_{safe_group_name}/count",
                        group_count[group_idx],
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        f"train/group_{group_idx}_{safe_group_name}/adv_prob",
                        self.group_dro.adv_probs[group_idx],
                        on_step=True,
                        on_epoch=False,
                    )
        else:
            raise ValueError(f"Unknown training.objective: {self.hparams.objective!r}")

        self._update_train_stats(per_sample_loss, per_sample_dice, g)
        if torch.isnan(loss).any():
            raise FloatingPointError("NaN loss encountered during training_step")
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        if self._train_count.item() > 0:
            avg_loss = self._train_loss_sum / self._train_count
            avg_dice = self._train_dice_sum / self._train_count
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
            avg_dice = torch.tensor(0.0, device=self.device)

        group_count = self._train_group_count
        group_loss = self._train_group_loss_sum / group_count.clamp_min(1)
        group_dice = self._train_group_dice_sum / group_count.clamp_min(1)
        worst_group_loss = group_loss.max() if group_loss.numel() else torch.tensor(0.0, device=self.device)

        self.log("train/epoch_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/epoch_avg_dice", avg_dice, on_step=False, on_epoch=True, prog_bar=False)
        self._update_ema_metric("train/total_loss", avg_loss)
        self._update_ema_metric("train/dice", avg_dice)
        self._update_ema_metric("train/worst_group_loss", worst_group_loss)
        for group_idx, group_name in enumerate(self.group_names):
            safe_group_name = str(group_name).replace("/", "_")
            self.log(
                f"train/group_{group_idx}_{safe_group_name}/epoch_loss",
                group_loss[group_idx],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"train/group_{group_idx}_{safe_group_name}/epoch_dice",
                group_dice[group_idx],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"train/group_{group_idx}_{safe_group_name}/epoch_count",
                group_count[group_idx],
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_start(self):
        device = self.device
        self._val_loss_sum = torch.tensor(0.0, device=device)
        self._val_dice_sum = torch.tensor(0.0, device=device)
        self._val_bce_sum = torch.tensor(0.0, device=device)
        self._val_dice_loss_sum = torch.tensor(0.0, device=device)
        self._val_count = torch.tensor(0.0, device=device)

        self._val_group_loss_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_dice_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_bce_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_dice_loss_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_count = torch.zeros(self.n_groups, device=device)

        self._val_eval_metrics = StreamingBinarySegmentationMetrics(
            threshold=self._eval_threshold,
            device=device,
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, xyxys, g = batch
        outputs = self(x)
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(outputs, y)

        self._val_loss_sum += per_sample_loss.sum()
        self._val_dice_sum += per_sample_dice.sum()
        self._val_bce_sum += per_sample_bce.sum()
        self._val_dice_loss_sum += per_sample_dice_loss.sum()
        self._val_count += float(per_sample_loss.numel())

        g = g.long()
        self._val_group_loss_sum.scatter_add_(0, g, per_sample_loss)
        self._val_group_dice_sum.scatter_add_(0, g, per_sample_dice)
        self._val_group_bce_sum.scatter_add_(0, g, per_sample_bce)
        self._val_group_dice_loss_sum.scatter_add_(0, g, per_sample_dice_loss)
        self._val_group_count.scatter_add_(0, g, torch.ones_like(per_sample_loss, dtype=self._val_group_count.dtype))

        if self._val_eval_metrics is not None:
            self._val_eval_metrics.update(logits=outputs, targets=y)

        self._stitcher.accumulate_val(outputs=outputs, xyxys=xyxys, dataloader_idx=dataloader_idx)

        return {"loss": per_sample_loss.mean()}

    def on_validation_epoch_end(self):
        self._sync_validation_accumulators()

        if self._val_count.item() > 0:
            avg_loss = self._val_loss_sum / self._val_count
            avg_dice = self._val_dice_sum / self._val_count
            avg_bce = self._val_bce_sum / self._val_count
            avg_dice_loss = self._val_dice_loss_sum / self._val_count
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
            avg_dice = torch.tensor(0.0, device=self.device)
            avg_bce = torch.tensor(0.0, device=self.device)
            avg_dice_loss = torch.tensor(0.0, device=self.device)

        group_count = self._val_group_count
        group_loss = self._val_group_loss_sum / group_count.clamp_min(1)
        group_dice = self._val_group_dice_sum / group_count.clamp_min(1)
        group_bce = self._val_group_bce_sum / group_count.clamp_min(1)
        group_dice_loss = self._val_group_dice_loss_sum / group_count.clamp_min(1)

        present = group_count > 0
        if present.any():
            worst_group_loss = group_loss[present].max()
            worst_group_dice = group_dice[present].min()
        else:
            worst_group_loss = group_loss.max()
            worst_group_dice = group_dice.min()

        self.log("val/avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val/worst_group_loss", worst_group_loss, on_epoch=True, prog_bar=True)
        self.log("val/avg_dice", avg_dice, on_epoch=True, prog_bar=False)
        self.log("val/worst_group_dice", worst_group_dice, on_epoch=True, prog_bar=False)
        self.log("val/avg_bce_loss", avg_bce, on_epoch=True, prog_bar=False)
        self.log("val/avg_dice_loss", avg_dice_loss, on_epoch=True, prog_bar=False)
        self._update_ema_metric("val/avg_loss", avg_loss)
        self._update_ema_metric("val/worst_group_loss", worst_group_loss)
        self._update_ema_metric("val/avg_dice", avg_dice)
        self._update_ema_metric("val/worst_group_dice", worst_group_dice)

        if self._val_eval_metrics is not None:
            eval_metrics = self._val_eval_metrics.compute()
            for k, v in eval_metrics.items():
                self.log(f"metrics/val/{k}", v, on_epoch=True, prog_bar=False)

        for group_idx, group_name in enumerate(self.group_names):
            safe_group_name = str(group_name).replace("/", "_")
            self.log(f"val/group_{group_idx}_{safe_group_name}/loss", group_loss[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/dice", group_dice[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/bce_loss", group_bce[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/dice_loss", group_dice_loss[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/count", group_count[group_idx], on_epoch=True)
        self._stitcher.on_validation_epoch_end(self)

    def configure_optimizers(self):
        lr = float(CFG.lr)
        weight_decay = float(getattr(CFG, "weight_decay", 0.0) or 0.0)
        optimizer_name = str(getattr(CFG, "optimizer", "adamw")).strip().lower()

        exclude_wd_bias_norm = bool(getattr(CFG, "exclude_weight_decay_bias_norm", False))
        use_param_groups = exclude_wd_bias_norm and weight_decay > 0
        decay_params = []
        no_decay_params = []
        if use_param_groups:
            for _, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if int(getattr(param, "ndim", 0)) < 2:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        if optimizer_name == "adamw":
            beta2 = float(getattr(CFG, "adamw_beta2", 0.999))
            eps = float(getattr(CFG, "adamw_eps", 1e-8))
            betas = (0.9, beta2)
            if use_param_groups:
                optimizer = AdamW(
                    [
                        {"params": decay_params, "weight_decay": weight_decay},
                        {"params": no_decay_params, "weight_decay": 0.0},
                    ],
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=0.0,
                )
            else:
                optimizer = AdamW(
                    self.parameters(),
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
        elif optimizer_name == "sgd":
            momentum = float(getattr(CFG, "sgd_momentum", 0.0) or 0.0)
            nesterov = bool(getattr(CFG, "sgd_nesterov", False))
            if use_param_groups:
                optimizer = SGD(
                    [
                        {"params": decay_params, "weight_decay": weight_decay},
                        {"params": no_decay_params, "weight_decay": 0.0},
                    ],
                    lr=lr,
                    momentum=momentum,
                    nesterov=nesterov,
                    weight_decay=0.0,
                )
            else:
                optimizer = SGD(
                    self.parameters(),
                    lr=lr,
                    momentum=momentum,
                    nesterov=nesterov,
                    weight_decay=weight_decay,
                )
        else:
            raise ValueError(
                f"Unsupported optimizer={CFG.optimizer!r}. Supported: 'adamw' | 'sgd'."
            )
        scheduler_name = str(getattr(CFG, "scheduler", "OneCycleLR")).lower()
        steps_per_epoch = int(self.hparams.total_steps)
        epochs = int(CFG.epochs)

        if scheduler_name == "onecyclelr":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=CFG.lr,
                pct_start=float(getattr(CFG, "onecycle_pct_start", 0.15)),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                div_factor=float(getattr(CFG, "onecycle_div_factor", 25.0)),
                final_div_factor=float(getattr(CFG, "onecycle_final_div_factor", 1e2)),
            )
            interval = "step"
        elif scheduler_name == "cosine":
            total_steps = max(1, steps_per_epoch * epochs)
            warmup_pct = float(getattr(CFG, "cosine_warmup_pct", 0.0) or 0.0)
            warmup_pct = max(0.0, min(1.0, warmup_pct))
            warmup_steps = int(round(total_steps * warmup_pct))
            warmup_steps = max(0, min(warmup_steps, total_steps - 1))

            eta_min = float(getattr(CFG, "min_lr", 0.0))

            if warmup_steps > 0:
                warmup_factor = float(getattr(CFG, "warmup_factor", 1.0) or 1.0)
                if warmup_factor <= 0:
                    raise ValueError(f"warmup_factor must be > 0, got {warmup_factor}")
                start_factor = 1.0 / warmup_factor

                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=float(start_factor),
                    end_factor=1.0,
                    total_iters=int(warmup_steps),
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(total_steps - warmup_steps),
                    eta_min=float(eta_min),
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[int(warmup_steps)],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(total_steps),
                    eta_min=float(eta_min),
            )
            interval = "step"
        elif scheduler_name == "gradualwarmupschedulerv2":
            scheduler = get_scheduler(CFG, optimizer)
            interval = "epoch"
        else:
            raise ValueError(
                f"Unsupported scheduler={CFG.scheduler!r}. Supported: 'OneCycleLR' | 'cosine' | 'GradualWarmupSchedulerV2'."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
            },
        }


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 50, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler
