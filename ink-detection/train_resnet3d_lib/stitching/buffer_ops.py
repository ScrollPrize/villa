import numpy as np
import torch
import torch.nn.functional as F


def gaussian_weights(
    cache,
    *,
    h: int,
    w: int,
    sigma_scale: float,
    min_weight: float,
) -> np.ndarray:
    key = (int(h), int(w))
    weights = cache.get(key)
    if weights is not None:
        return weights

    h, w = key
    sigma_y = max(float(h) * float(sigma_scale), 1.0)
    sigma_x = max(float(w) * float(sigma_scale), 1.0)

    y = np.arange(h, dtype=np.float32) - ((h - 1) / 2.0)
    x = np.arange(w, dtype=np.float32) - ((w - 1) / 2.0)
    wy = np.exp(-0.5 * (y / sigma_y) ** 2).astype(np.float32)
    wx = np.exp(-0.5 * (x / sigma_x) ** 2).astype(np.float32)

    weights = np.outer(wy, wx).astype(np.float32)
    weights /= float(weights.max())
    weights = np.clip(weights, float(min_weight), None, out=weights)
    cache[key] = weights
    return weights


def _sigmoid_numpy(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr_clipped = np.clip(arr, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-arr_clipped))).astype(np.float32)


def stitch_prob_map(pred_buf: np.ndarray, count_buf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covered = count_buf != 0
    stitched_logits = np.divide(
        pred_buf.astype(np.float32),
        count_buf.astype(np.float32),
        out=np.zeros_like(pred_buf, dtype=np.float32),
        where=covered,
    )
    stitched_probs = np.zeros_like(stitched_logits, dtype=np.float32)
    if covered.any():
        stitched_probs[covered] = _sigmoid_numpy(stitched_logits[covered])
    return stitched_probs, covered


def compose_segment_from_roi_buffers(roi_buffers, full_shape):
    full_shape = tuple(int(v) for v in full_shape)
    full_pred = np.zeros(full_shape, dtype=np.float32)
    full_count = np.zeros(full_shape, dtype=np.float32)
    for pred_buf, count_buf, offset in roi_buffers:
        y0, x0 = [int(v) for v in offset]
        h, w = pred_buf.shape
        full_pred[y0:y0 + h, x0:x0 + w] += pred_buf
        full_count[y0:y0 + h, x0:x0 + w] += count_buf
    base, has = stitch_prob_map(full_pred, full_count)
    return np.clip(base, 0, 1), has


def accumulate_to_buffers(
    *,
    outputs,
    xyxys,
    pred_buf,
    count_buf,
    downsample,
    offset=(0, 0),
    gaussian_cache,
    gaussian_sigma_scale,
    gaussian_min_weight,
):
    ds = int(downsample)
    y_logits = outputs.detach().to("cpu", dtype=torch.float32)
    buf_h, buf_w = pred_buf.shape[:2]
    off_y, off_x = offset
    for i, (x1, y1, x2, y2) in enumerate(xyxys):
        x1_i = int(x1)
        y1_i = int(y1)
        x2_i = int(x2)
        y2_i = int(y2)

        if ds > 1 and ((x1_i % ds) or (y1_i % ds) or (x2_i % ds) or (y2_i % ds)):
            raise ValueError("stitch coordinates are not aligned with stitch_downsample")

        x1_ds = x1_i // ds
        y1_ds = y1_i // ds
        x2_ds = (x2_i + ds - 1) // ds
        y2_ds = (y2_i + ds - 1) // ds

        x1_ds -= int(off_x)
        x2_ds -= int(off_x)
        y1_ds -= int(off_y)
        y2_ds -= int(off_y)

        target_h = y2_ds - y1_ds
        target_w = x2_ds - x1_ds
        if target_h <= 0 or target_w <= 0:
            continue

        pred_patch = y_logits[i].unsqueeze(0)
        if pred_patch.shape[-2:] != (target_h, target_w):
            pred_patch = F.interpolate(
                pred_patch,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

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

        patch_weights = gaussian_weights(
            gaussian_cache,
            h=target_h,
            w=target_w,
            sigma_scale=gaussian_sigma_scale,
            min_weight=gaussian_min_weight,
        )
        pred_crop = pred_patch[..., py0:py1, px0:px1]
        weight_crop = patch_weights[py0:py1, px0:px1]

        pred_buf[y1_clamped:y2_clamped, x1_clamped:x2_clamped] += (
            pred_crop.squeeze(0).squeeze(0).numpy() * weight_crop
        )
        count_buf[y1_clamped:y2_clamped, x1_clamped:x2_clamped] += weight_crop
