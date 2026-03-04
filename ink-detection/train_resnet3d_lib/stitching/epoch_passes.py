import time

import numpy as np
import torch

from train_resnet3d_lib.config import log


def run_train_stitch_pass(manager, model):
    if not manager.train_enabled:
        return None
    if not manager.train_loaders or not manager.train_segment_ids:
        return None
    if len(manager.train_loaders) != len(manager.train_segment_ids):
        raise ValueError(
            "train stitch loaders/segment_ids length mismatch "
            f"({len(manager.train_loaders)} vs {len(manager.train_segment_ids)})"
        )

    epoch = int(getattr(model, "current_epoch", 0))
    if manager.train_every_n_epochs > 1 and ((epoch + 1) % manager.train_every_n_epochs) != 0:
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
