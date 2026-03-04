import time

import cv2
import numpy as np
from pytorch_lightning.loggers import WandbLogger

from train_resnet3d_lib.config import CFG, log


def _to_u8(img_float: np.ndarray) -> np.ndarray:
    return (np.clip(img_float, 0.0, 1.0) * 255.0).astype(np.uint8)


def _add_borders_rgb(base_u8: np.ndarray, segment_id: str, borders_by_split: dict) -> np.ndarray:
    rgb = np.repeat(base_u8[..., None], 3, axis=2)
    train_border = borders_by_split.get("train", {}).get(str(segment_id))
    val_border = borders_by_split.get("val", {}).get(str(segment_id))
    if train_border is not None:
        rgb[train_border.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
    if val_border is not None:
        rgb[val_border.astype(bool)] = np.array([0, 0, 255], dtype=np.uint8)
    return rgb


def _downsample_for_wandb(img: np.ndarray, *, source_downsample: int, wandb_media_downsample: int) -> np.ndarray:
    source_downsample = int(source_downsample)
    if source_downsample < 1:
        raise ValueError(f"source_downsample must be >= 1, got {source_downsample}")
    if source_downsample > 1 or int(wandb_media_downsample) == 1:
        return img
    in_h, in_w = img.shape[:2]
    factor = int(wandb_media_downsample)
    if (in_h % factor) == 0 and (in_w % factor) == 0:
        out_h = in_h // factor
        out_w = in_w // factor
        if img.ndim == 2:
            if img.dtype == np.uint8:
                reduced = img.reshape(out_h, factor, out_w, factor).mean(axis=(1, 3), dtype=np.float32)
            else:
                reduced = img.reshape(out_h, factor, out_w, factor).mean(axis=(1, 3))
        elif img.ndim == 3:
            channels = int(img.shape[2])
            if img.dtype == np.uint8:
                reduced = img.reshape(out_h, factor, out_w, factor, channels).mean(
                    axis=(1, 3),
                    dtype=np.float32,
                )
            else:
                reduced = img.reshape(out_h, factor, out_w, factor, channels).mean(axis=(1, 3))
        else:
            raise ValueError(f"Unsupported image ndim for W&B downsample: {img.ndim}")
        if img.dtype == np.uint8:
            reduced = np.clip(np.rint(reduced), 0.0, 255.0).astype(np.uint8, copy=False)
        else:
            reduced = reduced.astype(img.dtype, copy=False)
        return np.ascontiguousarray(reduced)
    out_h = max(1, (int(in_h) + factor - 1) // factor)
    out_w = max(1, (int(in_w) + factor - 1) // factor)
    if out_h == in_h and out_w == in_w:
        return img
    resized = cv2.resize(
        img,
        (out_w, out_h),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized)


def log_stitched_wandb_media(
    *,
    model,
    sanity_checking: bool,
    log_train_stitch: bool,
    segment_to_val: dict,
    train_segment_viz: dict,
    log_only_mode: bool,
    log_only_segment_ids: list[str],
    roi_buffers_by_split: dict,
    roi_meta_by_split: dict,
    borders_by_split: dict,
    downsample: int,
    log_only_downsample: int,
    compose_segment_from_roi_buffers,
) -> None:
    can_log_media = (
        (not bool(sanity_checking))
        and (model.trainer is None or model.trainer.is_global_zero)
        and isinstance(model.logger, WandbLogger)
    )
    if not can_log_media:
        return

    wandb_media_downsample = int(getattr(CFG, "eval_wandb_media_downsample", 1))
    if wandb_media_downsample < 1:
        raise ValueError(
            "eval_wandb_media_downsample must be >= 1, "
            f"got {wandb_media_downsample}"
        )

    masks_logged = 0
    masks_log_only_logged = 0
    media_step = int(getattr(model.trainer, "global_step", 0))
    t0 = time.perf_counter()

    if bool(log_train_stitch):
        segment_ids = sorted(set(segment_to_val.keys()) | set(train_segment_viz.keys()))
        for segment_id in segment_ids:
            val_base = None
            val_has = None
            if segment_id in segment_to_val:
                val_img, val_cov = segment_to_val[segment_id]
                val_base = _to_u8(val_img)
                val_has = val_cov

            train_base = None
            train_has = None
            if segment_id in train_segment_viz:
                entry = train_segment_viz[segment_id]
                train_base = entry["img_u8"]
                train_has = entry["has"]

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

            image = _add_borders_rgb(base_u8, str(segment_id), borders_by_split)
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
            image = _downsample_for_wandb(
                image,
                source_downsample=int(downsample),
                wandb_media_downsample=wandb_media_downsample,
            )
            safe_segment_id = str(segment_id).replace("/", "_")
            model.logger.log_image(
                key=f"masks/{safe_segment_id}",
                images=[image],
                caption=[f"{segment_id} ({split_tag} ds={downsample})"],
            )
            masks_logged += 1
    else:
        want_color = bool(getattr(CFG, "stitch_train", False))
        for segment_id, (base, _covered) in segment_to_val.items():
            base_u8 = _to_u8(np.clip(base, 0, 1))
            if want_color:
                image = _add_borders_rgb(base_u8, str(segment_id), borders_by_split)
            else:
                image = base_u8
            image = _downsample_for_wandb(
                image,
                source_downsample=int(downsample),
                wandb_media_downsample=wandb_media_downsample,
            )
            safe_segment_id = str(segment_id).replace("/", "_")
            model.logger.log_image(
                key=f"masks/{safe_segment_id}",
                images=[image],
                caption=[f"{segment_id} (val ds={downsample})"],
            )
            masks_logged += 1

    if bool(log_only_mode):
        for segment_id in log_only_segment_ids:
            sid = str(segment_id)
            roi_buffers = roi_buffers_by_split["log_only"].get(sid)
            if not roi_buffers:
                raise ValueError(f"Missing log-only stitch ROI buffers for segment_id={sid!r}")
            meta = roi_meta_by_split["log_only"].get(sid)
            if meta is None:
                raise ValueError(f"Missing log-only ROI metadata for segment_id={sid!r}")
            full_shape = tuple(meta.get("full_shape", roi_buffers[0][0].shape))
            base, _ = compose_segment_from_roi_buffers(roi_buffers, full_shape)
            image = _to_u8(np.clip(base, 0, 1))
            image = _downsample_for_wandb(
                image,
                source_downsample=int(log_only_downsample),
                wandb_media_downsample=wandb_media_downsample,
            )
            safe_segment_id = sid.replace("/", "_")
            model.logger.log_image(
                key=f"masks_log_only/{safe_segment_id}",
                images=[image],
                caption=[f"{segment_id} (log-only ds={log_only_downsample})"],
            )
            masks_log_only_logged += 1

    log(
        f"wandb media done step={int(media_step)} "
        f"masks={masks_logged} masks_log_only={masks_log_only_logged} "
        f"elapsed={time.perf_counter() - t0:.2f}s"
    )
