from __future__ import annotations

import math
import os
import os.path as osp
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


EPS = 1e-8


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d + EPS)


def _as_bool_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={x.shape}")
    return x.astype(bool, copy=False)


def _read_gray_any(path_base: str) -> np.ndarray:
    from train_resnet3d_lib.data_ops import _read_gray

    for ext in (".png", ".tiff", ".tif"):
        arr = _read_gray(f"{path_base}{ext}")
        if arr is not None:
            return arr
    raise FileNotFoundError(f"Could not read image: {path_base}.(png|tif|tiff)")


def _downsample_bool_any(mask: np.ndarray, *, ds: int, out_hw: Tuple[int, int]) -> np.ndarray:
    """Downsample a boolean mask by OR over ds x ds blocks, matching a target shape."""
    ds = max(1, int(ds))
    out_h, out_w = [int(x) for x in out_hw]
    target_h = out_h * ds
    target_w = out_w * ds

    mask = _as_bool_2d(mask)
    h, w = mask.shape
    mask = mask[: min(h, target_h), : min(w, target_w)]
    h2, w2 = mask.shape
    pad_h = max(0, target_h - h2)
    pad_w = max(0, target_w - w2)
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)

    mask = mask.reshape(out_h, ds, out_w, ds)
    return mask.any(axis=(1, 3))


def _downsample_bool_any_roi(
    mask: np.ndarray,
    *,
    ds: int,
    out_hw: Tuple[int, int],
    offset: Tuple[int, int],
) -> np.ndarray:
    """Downsample a boolean mask by OR over ds x ds blocks, with a ds-grid offset."""
    ds = max(1, int(ds))
    out_h, out_w = [int(x) for x in out_hw]
    if out_h <= 0 or out_w <= 0:
        return np.zeros((max(1, out_h), max(1, out_w)), dtype=bool)
    off_y, off_x = [int(v) for v in offset]
    target_h = out_h * ds
    target_w = out_w * ds
    y0_full = max(0, off_y * ds)
    x0_full = max(0, off_x * ds)
    y1_full = y0_full + target_h
    x1_full = x0_full + target_w

    mask = _as_bool_2d(mask)
    h, w = mask.shape
    mask = mask[y0_full : min(h, y1_full), x0_full : min(w, x1_full)]
    h2, w2 = mask.shape
    pad_h = max(0, target_h - h2)
    pad_w = max(0, target_w - w2)
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    mask = mask.reshape(out_h, ds, out_w, ds)
    return mask.any(axis=(1, 3))


def _crop_to_mask_bbox(
    pred: np.ndarray, gt: np.ndarray, eval_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    eval_mask = _as_bool_2d(eval_mask)
    if not eval_mask.any():
        # Nothing to evaluate; return minimal arrays.
        z = np.zeros((1, 1), dtype=pred.dtype)
        zb = np.zeros((1, 1), dtype=bool)
        return z, zb, zb, (0, 1, 0, 1)

    ys, xs = np.where(eval_mask)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    pred_c = np.asarray(pred)[y0:y1, x0:x1]
    gt_c = np.asarray(gt)[y0:y1, x0:x1]
    mask_c = eval_mask[y0:y1, x0:x1]
    return pred_c, gt_c, mask_c, (y0, y1, x0, x1)


_GT_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_GT_CACHE_MAX = 8


def _gt_cache_get(key):
    val = _GT_CACHE.get(key)
    if val is not None:
        _GT_CACHE.move_to_end(key)
    return val


def _gt_cache_put(key, value):
    _GT_CACHE[key] = value
    _GT_CACHE.move_to_end(key)
    while len(_GT_CACHE) > _GT_CACHE_MAX:
        _GT_CACHE.popitem(last=False)


def _gt_cache_set_max(size: int) -> None:
    global _GT_CACHE_MAX
    size = max(0, int(size))
    _GT_CACHE_MAX = size
    while len(_GT_CACHE) > _GT_CACHE_MAX:
        _GT_CACHE.popitem(last=False)


@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int


def confusion_counts(pred: np.ndarray, gt: np.ndarray, *, mask: Optional[np.ndarray] = None) -> Confusion:
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")

    if mask is not None:
        mask = _as_bool_2d(mask)
        if mask.shape != gt.shape:
            raise ValueError(f"mask/gt shape mismatch: {mask.shape} vs {gt.shape}")
        pred = pred[mask]
        gt = gt[mask]

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def dice_from_confusion(c: Confusion) -> float:
    return _safe_div(2.0 * c.tp, 2.0 * c.tp + c.fp + c.fn)


def iou_from_confusion(c: Confusion) -> float:
    return _safe_div(c.tp, c.tp + c.fp + c.fn)


def accuracy_from_confusion(c: Confusion) -> float:
    return _safe_div(c.tp + c.tn, c.tp + c.fp + c.fn + c.tn)


def precision_from_confusion(c: Confusion) -> float:
    return _safe_div(c.tp, c.tp + c.fp)


def recall_from_confusion(c: Confusion) -> float:
    return _safe_div(c.tp, c.tp + c.fn)


def fbeta_from_confusion(c: Confusion, *, beta: float) -> float:
    beta2 = float(beta) ** 2
    numer = (1.0 + beta2) * c.tp
    denom = (1.0 + beta2) * c.tp + beta2 * c.fn + c.fp
    return _safe_div(numer, denom)


def voi_from_confusion(c: Confusion) -> float:
    total = float(c.tp + c.fp + c.fn + c.tn)
    if total <= 0:
        return float("nan")

    p11 = c.tp / total
    p10 = c.fn / total
    p01 = c.fp / total
    p00 = c.tn / total

    px1 = (c.tp + c.fn) / total
    px0 = 1.0 - px1
    py1 = (c.tp + c.fp) / total
    py0 = 1.0 - py1

    def _entropy(probs) -> float:
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        return float(-(probs * np.log2(probs)).sum())

    h_x = _entropy([px0, px1])
    h_y = _entropy([py0, py1])
    h_xy = _entropy([p00, p01, p10, p11])
    return float(2.0 * h_xy - h_x - h_y)


def _cc_structure(connectivity: int) -> np.ndarray:
    connectivity = int(connectivity)
    if connectivity == 1:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    if connectivity == 2:
        return np.ones((3, 3), dtype=bool)
    raise ValueError(f"connectivity must be 1 (4-neighborhood) or 2 (8-neighborhood), got {connectivity}")


def betti_numbers_2d(mask: np.ndarray, *, connectivity: int = 2) -> Tuple[int, int]:
    """Compute (beta0, beta1) for a 2D binary foreground mask.

    beta0: number of connected components in the foreground.
    beta1: number of holes (background components not touching the border).
    """
    from scipy.ndimage import label as cc_label

    mask = _as_bool_2d(mask)
    H, W = mask.shape
    if H == 0 or W == 0:
        return 0, 0

    struct = _cc_structure(connectivity)
    fg_lab, fg_n = cc_label(mask, structure=struct)
    beta0 = int(fg_n)

    bg_lab, bg_n = cc_label(~mask, structure=struct)
    if bg_n == 0:
        return beta0, 0

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    touching = set(np.unique(bg_lab[border])) - {0}
    all_bg = set(range(1, int(bg_n) + 1))
    holes = all_bg - touching
    beta1 = int(len(holes))
    return beta0, beta1


def betti_error(pred: np.ndarray, gt: np.ndarray, *, connectivity: int = 2) -> Dict[str, float]:
    b0p, b1p = betti_numbers_2d(pred, connectivity=connectivity)
    b0g, b1g = betti_numbers_2d(gt, connectivity=connectivity)
    return {
        "beta0_pred": float(b0p),
        "beta1_pred": float(b1p),
        "beta0_gt": float(b0g),
        "beta1_gt": float(b1g),
        "abs_beta0_err": float(abs(b0p - b0g)),
        "abs_beta1_err": float(abs(b1p - b1g)),
        "l1_betti_err": float(abs(b0p - b0g) + abs(b1p - b1g)),
    }


def euler_characteristic(mask: np.ndarray, *, connectivity: int = 2) -> int:
    b0, b1 = betti_numbers_2d(mask, connectivity=connectivity)
    return int(b0 - b1)


def psnr(pred: np.ndarray, gt: np.ndarray, *, C: float = 255.0) -> float:
    pred = _as_bool_2d(pred).astype(np.uint8) * 255
    gt = _as_bool_2d(gt).astype(np.uint8) * 255
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10((C * C) / mse))


def nrm(pred: np.ndarray, gt: np.ndarray) -> float:
    c = confusion_counts(pred, gt)
    nr_fn = _safe_div(c.fn, c.fn + c.tp)
    nr_fp = _safe_div(c.fp, c.fp + c.tn)
    return float(0.5 * (nr_fn + nr_fp))


def _boundary_mask(mask: np.ndarray, *, k: int = 3) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    mask = _as_bool_2d(mask)
    k = int(k)
    if k < 1 or (k % 2) != 1:
        raise ValueError(f"boundary kernel size k must be odd and >= 1, got {k}")
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    structure = np.ones((k, k), dtype=bool)
    er = binary_erosion(mask, structure=structure, iterations=1, border_value=0)
    return np.logical_and(mask, ~er)


def mpm(pred: np.ndarray, gt: np.ndarray, *, boundary_k: int = 3) -> float:
    """Misclassification Penalty Metric (MPM) per DIBCO-style definition."""
    from scipy.ndimage import distance_transform_edt

    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")

    gb = _boundary_mask(gt, k=boundary_k)
    dt_to_gb = distance_transform_edt((~gb).astype(np.uint8))

    fn_mask = np.logical_and(~pred, gt)
    fp_mask = np.logical_and(pred, ~gt)

    D = float(dt_to_gb[gt].sum()) + EPS
    mp_fn = float(dt_to_gb[fn_mask].sum()) / D
    mp_fp = float(dt_to_gb[fp_mask].sum()) / D
    return float(0.5 * (mp_fn + mp_fp))


def drd(pred: np.ndarray, gt: np.ndarray, *, block_size: int = 8) -> float:
    """DRD (Distance Reciprocal Distortion).

    Note: This matches the "inverse-distance 5x5 weights, normalized, center=0" variant.
    Exact contest-comparable DRD may differ in the weight matrix details.
    """
    from scipy.ndimage import convolve

    pred = _as_bool_2d(pred).astype(np.uint8)
    gt = _as_bool_2d(gt).astype(np.uint8)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    H, W = gt.shape

    # NUBN: number of non-uniform blocks in the GT.
    bs = max(1, int(block_size))
    hb = (H + bs - 1) // bs
    wb = (W + bs - 1) // bs
    pad_h = hb * bs - H
    pad_w = wb * bs - W
    gt_pad = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    gt_blocks = gt_pad.reshape(hb, bs, wb, bs)
    blk_sum = gt_blocks.sum(axis=(1, 3))
    nubn = int(np.logical_and(blk_sum > 0, blk_sum < (bs * bs)).sum())
    nubn = max(nubn, 1)

    mism = (pred != gt)
    if not mism.any():
        return 0.0

    # 5x5 inverse-distance weights, normalized, center=0.
    Wm = np.zeros((5, 5), dtype=np.float64)
    for iy, dy in enumerate(range(-2, 3)):
        for ix, dx in enumerate(range(-2, 3)):
            if dy == 0 and dx == 0:
                continue
            Wm[iy, ix] = 1.0 / math.sqrt(dy * dy + dx * dx)
    Wm /= float(Wm.sum() + EPS)

    gt_f = gt.astype(np.float64)
    gt_wsum = convolve(gt_f, Wm, mode="constant", cval=0.0)
    # If pred==0, cost is weighted GT neighborhood; if pred==1, cost is weighted (1-GT).
    drd_map = np.where(pred.astype(bool), 1.0 - gt_wsum, gt_wsum)
    return float(drd_map[mism].sum() / float(nubn))


def boundary_precision_recall_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tau: float = 1.0,
    boundary_k: int = 3,
) -> Dict[str, float]:
    """Boundary precision/recall/F1 with distance tolerance tau."""
    from scipy.ndimage import distance_transform_edt

    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    pb = _boundary_mask(pred, k=boundary_k)
    gb = _boundary_mask(gt, k=boundary_k)

    if pb.sum() == 0 and gb.sum() == 0:
        return {"b_precision": 1.0, "b_recall": 1.0, "b_f1": 1.0}
    if pb.sum() == 0 or gb.sum() == 0:
        return {"b_precision": 0.0, "b_recall": 0.0, "b_f1": 0.0}

    dt_to_gb = distance_transform_edt((~gb).astype(np.uint8))
    dt_to_pb = distance_transform_edt((~pb).astype(np.uint8))

    pb_match = (dt_to_gb[pb] <= float(tau)).sum()
    gb_match = (dt_to_pb[gb] <= float(tau)).sum()

    b_precision = _safe_div(pb_match, pb.sum())
    b_recall = _safe_div(gb_match, gb.sum())
    b_f1 = _safe_div(2.0 * b_precision * b_recall, b_precision + b_recall)
    return {"b_precision": float(b_precision), "b_recall": float(b_recall), "b_f1": float(b_f1)}


def nsd_surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tau: float = 1.0,
    boundary_k: int = 3,
) -> float:
    """Normalized Surface Dice (boundary overlap within tolerance)."""
    from scipy.ndimage import distance_transform_edt

    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    pb = _boundary_mask(pred, k=boundary_k)
    gb = _boundary_mask(gt, k=boundary_k)

    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0

    dt_to_gb = distance_transform_edt((~gb).astype(np.uint8))
    dt_to_pb = distance_transform_edt((~pb).astype(np.uint8))
    pb_match = (dt_to_gb[pb] <= float(tau)).sum()
    gb_match = (dt_to_pb[gb] <= float(tau)).sum()
    denom = float(pb.sum() + gb.sum())
    return _safe_div(pb_match + gb_match, denom)


def hausdorff_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    boundary_k: int = 3,
) -> Dict[str, float]:
    """Hausdorff (HD) and HD95 between boundaries."""
    from scipy.ndimage import distance_transform_edt

    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    pb = _boundary_mask(pred, k=boundary_k)
    gb = _boundary_mask(gt, k=boundary_k)

    if pb.sum() == 0 and gb.sum() == 0:
        return {"hd": 0.0, "hd95": 0.0, "assd": 0.0}
    if pb.sum() == 0 or gb.sum() == 0:
        # Use the image diagonal as a finite "maximal" distance penalty.
        diag = float(math.hypot(pred.shape[0], pred.shape[1]))
        return {"hd": diag, "hd95": diag, "assd": diag}

    dt_to_gb = distance_transform_edt((~gb).astype(np.uint8))
    dt_to_pb = distance_transform_edt((~pb).astype(np.uint8))
    d_pred = dt_to_gb[pb]
    d_gt = dt_to_pb[gb]

    hd = max(float(d_pred.max()), float(d_gt.max()))
    d_all = np.concatenate([d_pred, d_gt]) if d_pred.size and d_gt.size else d_pred if d_pred.size else d_gt
    hd95 = float(np.percentile(d_all, 95)) if d_all.size else float("inf")
    assd = 0.5 * (float(d_pred.mean()) + float(d_gt.mean()))
    return {"hd": hd, "hd95": hd95, "assd": assd}


def boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tau: float = 1.0,
    boundary_k: int = 3,
) -> float:
    """Boundary IoU using a tau-width band around each boundary."""
    from scipy.ndimage import distance_transform_edt

    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    pb = _boundary_mask(pred, k=boundary_k)
    gb = _boundary_mask(gt, k=boundary_k)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0

    dt_to_pb = distance_transform_edt((~pb).astype(np.uint8))
    dt_to_gb = distance_transform_edt((~gb).astype(np.uint8))
    pb_band = dt_to_pb <= float(tau)
    gb_band = dt_to_gb <= float(tau)
    inter = np.logical_and(pb_band, gb_band).sum()
    union = np.logical_or(pb_band, gb_band).sum()
    return _safe_div(inter, union)


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    mask = _as_bool_2d(mask)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    import kimimaro

    labels = mask.astype(np.uint32)
    skels = kimimaro.skeletonize(
        labels,
        parallel=1,
        progress=False,
        dust_threshold=0,
        anisotropy=(1, 1),
    )
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape

    def _draw_line(x0: int, y0: int, x1: int, y1: int) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            out[y0, x0] = True
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    for skel in skels.values():
        if skel is None:
            continue
        verts = np.asarray(getattr(skel, "vertices", None))
        if verts is None or verts.size == 0 or verts.ndim != 2 or verts.shape[1] < 2:
            continue
        xs = np.rint(verts[:, 0]).astype(np.int64)
        ys = np.rint(verts[:, 1]).astype(np.int64)
        bad = (xs < 0) | (xs >= w) | (ys < 0) | (ys >= h)
        if bad.any():
            xs2 = np.rint(verts[:, 1]).astype(np.int64)
            ys2 = np.rint(verts[:, 0]).astype(np.int64)
            bad2 = (xs2 < 0) | (xs2 >= w) | (ys2 < 0) | (ys2 >= h)
            if bad2.sum() < bad.sum():
                xs, ys = xs2, ys2
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        out[ys, xs] = True

        edges = getattr(skel, "edges", None)
        if edges is None:
            continue
        edges = np.asarray(edges)
        if edges.size == 0:
            continue
        if edges.ndim == 1:
            if edges.size == 2:
                edges = edges.reshape(1, 2)
            else:
                continue
        for edge in edges:
            if len(edge) < 2:
                continue
            i, j = int(edge[0]), int(edge[1])
            if i < 0 or j < 0 or i >= xs.size or j >= xs.size:
                continue
            _draw_line(int(xs[i]), int(ys[i]), int(xs[j]), int(ys[j]))

    return out


def skeleton_recall(pred: np.ndarray, gt: np.ndarray, *, skel_gt: Optional[np.ndarray] = None) -> float:
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if skel_gt is None:
        skel_gt = skeletonize_binary(gt)
    if skel_gt.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return _safe_div(np.logical_and(skel_gt, pred).sum(), skel_gt.sum())


def pseudo_fmeasure(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    skel_gt: Optional[np.ndarray] = None,
) -> float:
    """Pseudo F-measure (p-FM) used in H-DIBCO.

    Precision is standard; recall is computed on the GT skeleton.
    """
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if skel_gt is None:
        skel_gt = skeletonize_binary(gt)
    if skel_gt.sum() == 0:
        return 0.0

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    precision = _safe_div(tp, tp + fp)

    pre_recall = _safe_div(np.logical_and(pred, skel_gt).sum(), skel_gt.sum())
    return _safe_div(2.0 * precision * pre_recall, precision + pre_recall)


def cldice(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    skel_pred: Optional[np.ndarray] = None,
    skel_gt: Optional[np.ndarray] = None,
) -> float:
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if skel_pred is None:
        skel_pred = skeletonize_binary(pred)
    if skel_gt is None:
        skel_gt = skeletonize_binary(gt)

    tp = np.logical_and(skel_pred, gt).sum()
    ts = np.logical_and(skel_gt, pred).sum()
    tprec = _safe_div(tp, skel_pred.sum()) if skel_pred.sum() > 0 else 0.0
    tsens = _safe_div(ts, skel_gt.sum()) if skel_gt.sum() > 0 else 0.0
    return _safe_div(2.0 * tprec * tsens, tprec + tsens)


def skeleton_tube_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    radius: int = 1,
    skel_gt: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    from scipy.ndimage import distance_transform_edt

    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if skel_gt is None:
        skel_gt = skeletonize_binary(gt)
    if skel_gt.sum() == 0:
        return {"tube_precision": 0.0, "tube_recall": 0.0, "tube_f1": 0.0}

    dt = distance_transform_edt((~skel_gt).astype(np.uint8))
    tube = dt <= float(radius)

    tp = np.logical_and(pred, tube).sum()
    fp = np.logical_and(pred, ~tube).sum()
    fn = np.logical_and(~pred, tube).sum()
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec)
    return {"tube_precision": float(prec), "tube_recall": float(rec), "tube_f1": float(f1)}


def _label_components(mask: np.ndarray, *, connectivity: int = 2) -> Tuple[np.ndarray, int]:
    from scipy.ndimage import label as cc_label

    mask = _as_bool_2d(mask)
    struct = _cc_structure(connectivity)
    lab, n = cc_label(mask, structure=struct)
    return lab, int(n)


def component_iou_stats(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    connectivity: int = 2,
    worst_q: Optional[float] = 0.1,
    worst_k: Optional[int] = None,
    pred_lab: Optional[np.ndarray] = None,
    n_pred: Optional[int] = None,
    gt_lab: Optional[np.ndarray] = None,
    n_gt: Optional[int] = None,
) -> Dict[str, float]:
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if pred_lab is None or n_pred is None:
        pred_lab, n_pred = _label_components(pred, connectivity=connectivity)
    if gt_lab is None or n_gt is None:
        gt_lab, n_gt = _label_components(gt, connectivity=connectivity)

    if n_gt == 0:
        return {
            "n_gt": 0.0,
            "n_pred": float(n_pred),
            "mean_iou": float("nan"),
            "median_iou": float("nan"),
            "worst_k_mean": float("nan"),
            "worst_q_mean": float("nan"),
            "min_iou": float("nan"),
        }

    if n_pred == 0:
        ious = np.zeros(n_gt, dtype=np.float64)
    else:
        ious = np.zeros(n_gt, dtype=np.float64)
        pred_sizes = np.array([(pred_lab == (i + 1)).sum() for i in range(n_pred)], dtype=np.float64)
        for gi in range(n_gt):
            g_mask = (gt_lab == (gi + 1))
            g_size = g_mask.sum()
            if g_size == 0:
                continue
            best = 0.0
            for pj in range(n_pred):
                if pred_sizes[pj] == 0:
                    continue
                inter = np.logical_and(g_mask, pred_lab == (pj + 1)).sum()
                if inter == 0:
                    continue
                union = g_size + pred_sizes[pj] - inter
                best = max(best, float(inter) / float(union))
            ious[gi] = best

    ious_sorted = np.sort(ious)
    worst_k_mean = float("nan")
    worst_q_mean = float("nan")
    if ious_sorted.size:
        if worst_k is not None:
            k = max(1, min(int(worst_k), int(len(ious_sorted))))
            worst_k_mean = float(ious_sorted[:k].mean())
        if worst_q is not None:
            q = float(worst_q)
            q = min(max(q, 0.0), 1.0)
            kq = max(1, int(round(len(ious_sorted) * q)))
            worst_q_mean = float(ious_sorted[:kq].mean())
    return {
        "n_gt": float(n_gt),
        "n_pred": float(n_pred),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "worst_k_mean": worst_k_mean,
        "worst_q_mean": worst_q_mean,
        "min_iou": float(ious.min()) if ious.size else float("nan"),
    }


def component_dice_stats(
    pred_prob: np.ndarray,
    gt_bin: np.ndarray,
    *,
    connectivity: int = 2,
    pad: int = 0,
    worst_q: Optional[float] = 0.1,
    worst_k: Optional[int] = None,
    gt_lab: Optional[np.ndarray] = None,
    n_gt: Optional[int] = None,
) -> Dict[str, float]:
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    gt_bin = _as_bool_2d(gt_bin)
    if gt_lab is None or n_gt is None:
        gt_lab, n_gt = _label_components(gt_bin, connectivity=connectivity)

    if n_gt == 0:
        return {
            "n_gt": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "worst_k_mean": float("nan"),
            "worst_q_mean": float("nan"),
            "min": float("nan"),
        }

    pad = max(0, int(pad))
    dices = []
    eps = 1e-7
    for gi in range(1, n_gt + 1):
        mask = gt_lab == gi
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(gt_bin.shape[0], int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(gt_bin.shape[1], int(xs.max()) + 1 + pad)

        crop_pred = pred_prob[y0:y1, x0:x1]
        crop_gt = gt_bin[y0:y1, x0:x1].astype(np.float32)
        inter = float((crop_pred * crop_gt).sum())
        denom = float(crop_pred.sum() + crop_gt.sum())
        dice = (2.0 * inter + eps) / (denom + eps)
        dices.append(dice)

    if not dices:
        return {
            "n_gt": float(n_gt),
            "mean": float("nan"),
            "median": float("nan"),
            "worst_k_mean": float("nan"),
            "worst_q_mean": float("nan"),
            "min": float("nan"),
        }

    dices = np.asarray(dices, dtype=np.float64)
    dices_sorted = np.sort(dices)
    worst_k_mean = float("nan")
    worst_q_mean = float("nan")
    if dices_sorted.size:
        if worst_k is not None:
            k = max(1, min(int(worst_k), int(len(dices_sorted))))
            worst_k_mean = float(dices_sorted[:k].mean())
        if worst_q is not None:
            q = float(worst_q)
            q = min(max(q, 0.0), 1.0)
            kq = max(1, int(round(len(dices_sorted) * q)))
            worst_q_mean = float(dices_sorted[:kq].mean())

    return {
        "n_gt": float(n_gt),
        "mean": float(dices.mean()),
        "median": float(np.median(dices)),
        "worst_k_mean": worst_k_mean,
        "worst_q_mean": worst_q_mean,
        "min": float(dices.min()) if dices.size else float("nan"),
    }


def panoptic_quality(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    connectivity: int = 2,
    iou_thr: float = 0.5,
    pred_lab: Optional[np.ndarray] = None,
    n_pred: Optional[int] = None,
    gt_lab: Optional[np.ndarray] = None,
    n_gt: Optional[int] = None,
) -> Dict[str, float]:
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if pred_lab is None or n_pred is None:
        pred_lab, n_pred = _label_components(pred, connectivity=connectivity)
    if gt_lab is None or n_gt is None:
        gt_lab, n_gt = _label_components(gt, connectivity=connectivity)

    if n_gt == 0 and n_pred == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0.0, "fp": 0.0, "fn": 0.0}
    if n_gt == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0.0, "fp": float(n_pred), "fn": 0.0}
    if n_pred == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0.0, "fp": 0.0, "fn": float(n_gt)}

    pred_sizes = np.array([(pred_lab == (i + 1)).sum() for i in range(n_pred)], dtype=np.float64)
    gt_sizes = np.array([(gt_lab == (i + 1)).sum() for i in range(n_gt)], dtype=np.float64)

    pairs = []
    for gi in range(n_gt):
        g_mask = (gt_lab == (gi + 1))
        g_size = gt_sizes[gi]
        for pj in range(n_pred):
            p_size = pred_sizes[pj]
            inter = np.logical_and(g_mask, pred_lab == (pj + 1)).sum()
            if inter == 0:
                continue
            union = g_size + p_size - inter
            iou = float(inter) / float(union)
            if iou >= float(iou_thr):
                pairs.append((iou, gi, pj))
    pairs.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    sum_iou = 0.0
    for iou, gi, pj in pairs:
        if gi in matched_gt or pj in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pj)
        sum_iou += float(iou)

    tp = float(len(matched_gt))
    fp = float(n_pred) - tp
    fn = float(n_gt) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = _safe_div(sum_iou, denom) if denom > 0 else 0.0
    sq = _safe_div(sum_iou, tp) if tp > 0 else 0.0
    rq = _safe_div(tp, denom) if denom > 0 else 0.0
    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


def component_ssim_stats(
    pred_prob: np.ndarray,
    gt_bin: np.ndarray,
    *,
    connectivity: int = 2,
    pad: int = 2,
    ssim_mode: str = "prob",
    worst_q: Optional[float] = 0.1,
    worst_k: Optional[int] = None,
    gt_lab: Optional[np.ndarray] = None,
    n_gt: Optional[int] = None,
) -> Dict[str, float]:
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    gt_bin = _as_bool_2d(gt_bin)
    if gt_lab is None or n_gt is None:
        gt_lab, n_gt = _label_components(gt_bin, connectivity=connectivity)
    if n_gt == 0:
        return {
            "n_gt": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "worst_k_mean": float("nan"),
            "worst_q_mean": float("nan"),
            "min": float("nan"),
        }

    pad = max(0, int(pad))
    vals = []
    for gi in range(1, n_gt + 1):
        mask = gt_lab == gi
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(gt_bin.shape[0], int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(gt_bin.shape[1], int(xs.max()) + 1 + pad)

        crop_pred = pred_prob[y0:y1, x0:x1]
        crop_gt = gt_bin[y0:y1, x0:x1]

        if str(ssim_mode).lower() == "dist":
            from scipy.ndimage import distance_transform_edt

            pred_bin = crop_pred >= 0.5
            pred_img = distance_transform_edt(~pred_bin.astype(np.uint8)).astype(np.float32)
            gt_img = distance_transform_edt(~crop_gt.astype(np.uint8)).astype(np.float32)
            if pred_img.max() > 0:
                pred_img = pred_img / float(pred_img.max())
            if gt_img.max() > 0:
                gt_img = gt_img / float(gt_img.max())
        else:
            pred_img = crop_pred.astype(np.float32)
            gt_img = crop_gt.astype(np.float32)

        vals.append(_gaussian_ssim(pred_img, gt_img, sigma=1.5, data_range=1.0))

    if not vals:
        return {
            "n_gt": float(n_gt),
            "mean": float("nan"),
            "median": float("nan"),
            "worst_k_mean": float("nan"),
            "worst_q_mean": float("nan"),
            "min": float("nan"),
        }

    vals = np.asarray(vals, dtype=np.float64)
    vals_sorted = np.sort(vals)
    worst_k_mean = float("nan")
    worst_q_mean = float("nan")
    if vals_sorted.size:
        if worst_k is not None:
            k = max(1, min(int(worst_k), int(len(vals_sorted))))
            worst_k_mean = float(vals_sorted[:k].mean())
        if worst_q is not None:
            q = float(worst_q)
            q = min(max(q, 0.0), 1.0)
            kq = max(1, int(round(len(vals_sorted) * q)))
            worst_q_mean = float(vals_sorted[:kq].mean())
    return {
        "n_gt": float(n_gt),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "worst_k_mean": worst_k_mean,
        "worst_q_mean": worst_q_mean,
        "min": float(vals_sorted.min()),
    }


def _hist_pr_roc(
    pred_prob: np.ndarray,
    gt_bin: np.ndarray,
    *,
    num_bins: int,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    gt_bin = _as_bool_2d(gt_bin)
    num_bins = max(2, int(num_bins))

    if mask is not None:
        mask = _as_bool_2d(mask)
        pred_prob = pred_prob[mask]
        gt_bin = gt_bin[mask]

    if pred_prob.size == 0:
        return {"auprc": float("nan"), "auroc": float("nan")}

    idx = np.clip((pred_prob * float(num_bins)).astype(np.int64), 0, num_bins - 1)
    pos_hist = np.bincount(idx[gt_bin], minlength=num_bins).astype(np.float64)
    neg_hist = np.bincount(idx[~gt_bin], minlength=num_bins).astype(np.float64)

    tp = np.cumsum(pos_hist[::-1])
    fp = np.cumsum(neg_hist[::-1])
    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()

    if total_pos <= 0:
        return {"auprc": float("nan"), "auroc": float("nan")}

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(total_pos, 1.0)

    # AP (step integration over recall).
    recall0 = np.concatenate([[0.0], recall])
    prec0 = np.concatenate([[1.0], precision])
    ap = float(np.sum(np.maximum(recall0[1:] - recall0[:-1], 0.0) * prec0[1:]))

    if total_neg <= 0:
        auroc = float("nan")
    else:
        fpr = fp / np.maximum(total_neg, 1.0)
        auroc = float(np.trapz(recall, fpr))

    return {"auprc": ap, "auroc": auroc}


def component_pr_stats(
    pred_prob: np.ndarray,
    gt_bin: np.ndarray,
    *,
    connectivity: int = 2,
    pad: int = 2,
    num_bins: int = 200,
    worst_q: Optional[float] = 0.1,
    worst_k: Optional[int] = None,
    gt_lab: Optional[np.ndarray] = None,
    n_gt: Optional[int] = None,
    eval_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    gt_bin = _as_bool_2d(gt_bin)
    if gt_lab is None or n_gt is None:
        gt_lab, n_gt = _label_components(gt_bin, connectivity=connectivity)
    if n_gt == 0:
        return {
            "n_gt": 0.0,
            "auprc_mean": float("nan"),
            "auprc_worst_k_mean": float("nan"),
            "auprc_worst_q_mean": float("nan"),
            "auroc_mean": float("nan"),
            "auroc_worst_k_mean": float("nan"),
            "auroc_worst_q_mean": float("nan"),
        }

    pad = max(0, int(pad))
    auprc_vals = []
    auroc_vals = []
    for gi in range(1, n_gt + 1):
        mask = gt_lab == gi
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(gt_bin.shape[0], int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(gt_bin.shape[1], int(xs.max()) + 1 + pad)

        crop_pred = pred_prob[y0:y1, x0:x1]
        crop_gt = gt_bin[y0:y1, x0:x1]
        crop_mask = None
        if eval_mask is not None:
            crop_mask = eval_mask[y0:y1, x0:x1]
        stats = _hist_pr_roc(crop_pred, crop_gt, num_bins=num_bins, mask=crop_mask)
        auprc_vals.append(stats["auprc"])
        auroc_vals.append(stats["auroc"])

    if not auprc_vals:
        return {
            "n_gt": float(n_gt),
            "auprc_mean": float("nan"),
            "auprc_worst_k_mean": float("nan"),
            "auprc_worst_q_mean": float("nan"),
            "auroc_mean": float("nan"),
            "auroc_worst_k_mean": float("nan"),
            "auroc_worst_q_mean": float("nan"),
        }

    auprc_vals = np.asarray(auprc_vals, dtype=np.float64)
    auroc_vals = np.asarray(auroc_vals, dtype=np.float64)

    def _worst_mean(vals: np.ndarray) -> Tuple[float, float]:
        vals_sorted = np.sort(vals)
        worst_k_mean = float("nan")
        worst_q_mean = float("nan")
        if vals_sorted.size:
            if worst_k is not None:
                k = max(1, min(int(worst_k), int(len(vals_sorted))))
                worst_k_mean = float(vals_sorted[:k].mean())
            if worst_q is not None:
                q = float(worst_q)
                q = min(max(q, 0.0), 1.0)
                kq = max(1, int(round(len(vals_sorted) * q)))
                worst_q_mean = float(vals_sorted[:kq].mean())
        return worst_k_mean, worst_q_mean

    auprc_wk, auprc_wq = _worst_mean(auprc_vals)
    auroc_wk, auroc_wq = _worst_mean(auroc_vals)
    return {
        "n_gt": float(n_gt),
        "auprc_mean": float(auprc_vals.mean()),
        "auprc_worst_k_mean": auprc_wk,
        "auprc_worst_q_mean": auprc_wq,
        "auroc_mean": float(auroc_vals.mean()),
        "auroc_worst_k_mean": auroc_wk,
        "auroc_worst_q_mean": auroc_wq,
    }


def component_diagnostics(
    pred_prob: np.ndarray,
    gt_bin: np.ndarray,
    *,
    pred_lab: np.ndarray,
    n_pred: int,
    gt_lab: np.ndarray,
    n_gt: int,
    connectivity: int = 2,
    pad: int = 0,
    num_bins: int = 200,
    eval_mask: Optional[np.ndarray] = None,
    worst_q: Optional[float] = 0.1,
    worst_k: Optional[int] = None,
    ssim_mode: str = "prob",
    offset: Optional[Tuple[int, int]] = None,
) -> list[dict]:
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    gt_bin = _as_bool_2d(gt_bin)
    if n_gt <= 0:
        return []

    pad = max(0, int(pad))
    pred_sizes = np.array([(pred_lab == (i + 1)).sum() for i in range(n_pred)], dtype=np.float64)

    rows = []
    iou_vals = []
    dice_vals = []
    ssim_vals = []
    auprc_vals = []
    auroc_vals = []

    off_y, off_x = (0, 0)
    if offset is not None:
        off_y, off_x = [int(v) for v in offset]

    eps = 1e-7
    for gi in range(1, n_gt + 1):
        g_mask = (gt_lab == gi)
        if not g_mask.any():
            continue
        ys, xs = np.where(g_mask)
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1

        g_size = float(g_mask.sum())
        best_iou = 0.0
        best_pred_id = 0
        if n_pred > 0:
            cand_preds = np.unique(pred_lab[g_mask])
            cand_preds = cand_preds[cand_preds > 0]
            for pj in cand_preds:
                p_idx = int(pj) - 1
                p_size = pred_sizes[p_idx] if p_idx >= 0 else 0.0
                if p_size <= 0:
                    continue
                inter = np.logical_and(g_mask, pred_lab == int(pj)).sum()
                if inter == 0:
                    continue
                union = g_size + float(p_size) - float(inter)
                iou = float(inter) / float(union)
                if iou > best_iou:
                    best_iou = float(iou)
                    best_pred_id = int(pj)

        y0p = max(0, y0 - pad)
        y1p = min(gt_bin.shape[0], y1 + pad)
        x0p = max(0, x0 - pad)
        x1p = min(gt_bin.shape[1], x1 + pad)

        crop_pred = pred_prob[y0p:y1p, x0p:x1p]
        crop_gt = gt_bin[y0p:y1p, x0p:x1p].astype(np.float32)
        inter = float((crop_pred * crop_gt).sum())
        denom = float(crop_pred.sum() + crop_gt.sum())
        dice = (2.0 * inter + eps) / (denom + eps)

        if str(ssim_mode).lower() == "dist":
            from scipy.ndimage import distance_transform_edt

            pred_bin = crop_pred >= 0.5
            pred_img = distance_transform_edt(~pred_bin.astype(np.uint8)).astype(np.float32)
            gt_img = distance_transform_edt(~crop_gt.astype(np.uint8)).astype(np.float32)
            if pred_img.max() > 0:
                pred_img = pred_img / float(pred_img.max())
            if gt_img.max() > 0:
                gt_img = gt_img / float(gt_img.max())
        else:
            pred_img = crop_pred.astype(np.float32)
            gt_img = crop_gt.astype(np.float32)
        ssim_val = _gaussian_ssim(pred_img, gt_img, sigma=1.5, data_range=1.0)

        crop_mask = None
        if eval_mask is not None:
            crop_mask = eval_mask[y0p:y1p, x0p:x1p]
        pr_stats = _hist_pr_roc(crop_pred, crop_gt, num_bins=num_bins, mask=crop_mask)
        auprc = float(pr_stats["auprc"])
        auroc = float(pr_stats["auroc"])

        row = {
            "gt_id": int(gi),
            "gt_area": float(g_size),
            "y0": int(y0 + off_y),
            "y1": int(y1 + off_y),
            "x0": int(x0 + off_x),
            "x1": int(x1 + off_x),
            "best_iou": float(best_iou),
            "best_pred_id": int(best_pred_id),
            "dice": float(dice),
            "ssim": float(ssim_val),
            "auprc": float(auprc),
            "auroc": float(auroc),
        }
        rows.append(row)
        iou_vals.append(best_iou)
        dice_vals.append(dice)
        ssim_vals.append(ssim_val)
        auprc_vals.append(auprc)
        auroc_vals.append(auroc)

    def _worst_flags(vals):
        flags = {"wk": set(), "wq": set()}
        if not rows:
            return flags
        vals = np.asarray(vals, dtype=np.float64)
        vals = np.nan_to_num(vals, nan=1.0, posinf=1.0, neginf=1.0)
        order = np.argsort(vals)
        if worst_k is not None:
            k = max(1, min(int(worst_k), int(len(order))))
            flags["wk"] = set(int(i) for i in order[:k])
        if worst_q is not None:
            q = float(worst_q)
            q = min(max(q, 0.0), 1.0)
            kq = max(1, int(round(len(order) * q)))
            flags["wq"] = set(int(i) for i in order[:kq])
        return flags

    iou_flags = _worst_flags(iou_vals)
    dice_flags = _worst_flags(dice_vals)
    ssim_flags = _worst_flags(ssim_vals)
    auprc_flags = _worst_flags(auprc_vals)
    auroc_flags = _worst_flags(auroc_vals)

    for idx, row in enumerate(rows):
        row["worst_k_iou"] = int(idx in iou_flags["wk"])
        row["worst_q_iou"] = int(idx in iou_flags["wq"])
        row["worst_k_dice"] = int(idx in dice_flags["wk"])
        row["worst_q_dice"] = int(idx in dice_flags["wq"])
        row["worst_k_ssim"] = int(idx in ssim_flags["wk"])
        row["worst_q_ssim"] = int(idx in ssim_flags["wq"])
        row["worst_k_auprc"] = int(idx in auprc_flags["wk"])
        row["worst_q_auprc"] = int(idx in auprc_flags["wq"])
        row["worst_k_auroc"] = int(idx in auroc_flags["wk"])
        row["worst_q_auroc"] = int(idx in auroc_flags["wq"])
    return rows


def threshold_stability(
    pred_prob: np.ndarray,
    gt: np.ndarray,
    *,
    thresholds: np.ndarray,
    metric_fn,
) -> Dict[str, float]:
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    gt = _as_bool_2d(gt)
    vals = []
    for t in thresholds:
        pred_bin = pred_prob >= float(t)
        vals.append(float(metric_fn(pred_bin, gt)))
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "std": float(vals.std()) if vals.size else float("nan"),
        "min": float(vals.min()) if vals.size else float("nan"),
        "max": float(vals.max()) if vals.size else float("nan"),
    }


def _gaussian_ssim(pred: np.ndarray, gt: np.ndarray, *, sigma: float = 1.5, data_range: float = 1.0) -> float:
    from scipy.ndimage import gaussian_filter

    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1 = gaussian_filter(pred, sigma=sigma)
    mu2 = gaussian_filter(gt, sigma=sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(pred * pred, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(gt * gt, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(pred * gt, sigma=sigma) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + EPS)
    return float(np.mean(ssim_map))


def persistence_distances(
    pred_prob: np.ndarray,
    gt_bin: np.ndarray,
    *,
    dims: Tuple[int, ...] = (0, 1),
    invert: bool = True,
    downsample: int = 1,
) -> Dict[str, float]:
    try:
        import gudhi as gd
        from gudhi.hera import bottleneck_distance
        from gudhi.wasserstein import wasserstein_distance
    except Exception:
        return {
            "pd_available": 0.0,
            "bottleneck_d0": float("nan"),
            "bottleneck_d1": float("nan"),
            "wasserstein_d0": float("nan"),
            "wasserstein_d1": float("nan"),
        }

    def _diagram(img: np.ndarray, *, invert_flag: bool) -> Dict[int, np.ndarray]:
        f = (1.0 - img) if invert_flag else img
        cc = gd.CubicalComplex(top_dimensional_cells=f.astype(np.float64))
        cc.persistence()
        return {d: cc.persistence_intervals_in_dimension(d) for d in dims}

    def _downsample_mean(img: np.ndarray, factor: int) -> np.ndarray:
        factor = max(1, int(factor))
        if factor == 1:
            return img
        h, w = img.shape
        h2 = (h // factor) * factor
        w2 = (w // factor) * factor
        if h2 <= 0 or w2 <= 0:
            return img[: max(1, h), : max(1, w)]
        img = img[:h2, :w2]
        img = img.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))
        return img

    pred_prob = np.asarray(pred_prob, dtype=np.float64)
    gt_prob = np.asarray(gt_bin, dtype=np.float64)
    ds = max(1, int(downsample))
    if ds > 1:
        pred_prob = _downsample_mean(pred_prob, ds)
        gt_prob = _downsample_mean(gt_prob, ds)
    pred_dgms = _diagram(pred_prob, invert_flag=invert)
    gt_dgms = _diagram(gt_prob, invert_flag=invert)

    out = {"pd_available": 1.0}
    for d in dims:
        dgm_p = pred_dgms.get(d, np.zeros((0, 2), dtype=np.float64))
        dgm_g = gt_dgms.get(d, np.zeros((0, 2), dtype=np.float64))
        try:
            out[f"bottleneck_d{d}"] = float(bottleneck_distance(dgm_p, dgm_g))
        except Exception:
            out[f"bottleneck_d{d}"] = float("nan")
        try:
            out[f"wasserstein_d{d}"] = float(wasserstein_distance(dgm_p, dgm_g, order=1, internal_p=2))
        except Exception:
            out[f"wasserstein_d{d}"] = float("nan")
    return out


def compute_stitched_metrics(
    *,
    fragment_id: str,
    pred_prob: np.ndarray,
    pred_has: np.ndarray,
    label_suffix: str,
    mask_suffix: str,
    downsample: int,
    roi_offset: Optional[Tuple[int, int]] = None,
    threshold: float,
    fbeta: float,
    betti_connectivity: int = 2,
    drd_block_size: int = 8,
    boundary_k: int = 3,
    boundary_tols: Optional[np.ndarray] = None,
    skeleton_radius: Optional[np.ndarray] = None,
    component_iou_thr: float = 0.5,
    component_worst_q: Optional[float] = 0.1,
    component_worst_k: Optional[int] = 8,
    component_min_area: int = 0,
    component_ssim: bool = False,
    component_ssim_pad: int = 2,
    pr_num_bins: int = 200,
    threshold_grid: Optional[np.ndarray] = None,
    ssim_mode: str = "prob",
    compute_persistence: bool = False,
    persistence_downsample: int = 1,
    output_dir: Optional[str] = None,
    component_output_dir: Optional[str] = None,
    save_skeleton_images: bool = True,
    gt_cache_max: Optional[int] = None,
) -> Dict[str, float]:
    """Compute evaluation metrics on a stitched prediction for a single fragment.

    Args:
        fragment_id: segment id (used for loading GT from train_scrolls/...).
        pred_prob: stitched predicted probabilities in [0,1], shape (H', W').
        pred_has: boolean mask of where prediction is defined (count_buf != 0), shape (H', W').
    """
    seg = str(fragment_id)
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    pred_has = _as_bool_2d(pred_has)
    if pred_prob.shape != pred_has.shape:
        raise ValueError(f"pred_prob/pred_has shape mismatch: {pred_prob.shape} vs {pred_has.shape}")
    if gt_cache_max is not None:
        _gt_cache_set_max(gt_cache_max)
    pred_full_shape = tuple(pred_prob.shape)

    timings: Dict[str, float] = {}
    def _timeit(name: str, t0: float) -> None:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)

    ds = max(1, int(downsample))
    roi_key = (0, 0)
    if roi_offset is not None:
        roi_key = (int(roi_offset[0]), int(roi_offset[1]))
    cache_key = (
        seg,
        str(label_suffix),
        str(mask_suffix),
        int(ds),
        tuple(roi_key),
        tuple(pred_prob.shape),
        int(pred_has.sum()),
        int(betti_connectivity),
    )
    cache = _gt_cache_get(cache_key)
    if cache is not None:
        cache_hit = True
        gt_bin = cache["gt_bin"]
        eval_mask = cache["eval_mask"]
        gt_lab = cache["gt_lab"]
        n_gt = cache["n_gt"]
        y0, y1, x0, x1 = cache["crop"]
        pred_full_shape = tuple(cache.get("full_shape", pred_full_shape))
        pred_prob = pred_prob[y0:y1, x0:x1]
        pred_has = pred_has[y0:y1, x0:x1]
    else:
        cache_hit = False
        t0 = time.perf_counter()
        label_base = osp.join("train_scrolls", seg, f"{seg}_inklabels{label_suffix}")
        mask_base = osp.join("train_scrolls", seg, f"{seg}_mask{mask_suffix}")
        gt_gray = _read_gray_any(label_base)
        valid_gray = _read_gray_any(mask_base)

        gt_bin_full = (gt_gray.astype(np.float32) / 255.0) >= 0.5
        valid_full = valid_gray.astype(np.uint8) > 0
        if seg == "20230827161846":
            valid_full = np.flipud(valid_full)

        if roi_offset is not None and (int(roi_offset[0]) != 0 or int(roi_offset[1]) != 0):
            gt_bin = _downsample_bool_any_roi(gt_bin_full, ds=ds, out_hw=pred_prob.shape, offset=roi_offset)
            valid = _downsample_bool_any_roi(valid_full, ds=ds, out_hw=pred_prob.shape, offset=roi_offset)
        else:
            gt_bin = _downsample_bool_any(gt_bin_full, ds=ds, out_hw=pred_prob.shape)
            valid = _downsample_bool_any(valid_full, ds=ds, out_hw=pred_prob.shape)

        eval_mask = np.logical_and(pred_has, valid)
        pred_prob, gt_bin, eval_mask, crop = _crop_to_mask_bbox(pred_prob, gt_bin, eval_mask)
        y0, y1, x0, x1 = crop

        if eval_mask.shape != pred_prob.shape:
            raise ValueError("internal error: cropped shapes mismatch")
        pred_prob = pred_prob.copy()
        gt_bin = gt_bin.copy()
        pred_prob[~eval_mask] = 0.0
        gt_bin[~eval_mask] = False

        gt_lab, n_gt = _label_components(gt_bin, connectivity=betti_connectivity)
        _gt_cache_put(
            cache_key,
            {
                "gt_bin": gt_bin,
                "eval_mask": eval_mask,
                "gt_lab": gt_lab,
                "n_gt": n_gt,
                "crop": (int(y0), int(y1), int(x0), int(x1)),
                "full_shape": pred_full_shape,
            },
        )
        _timeit("load_gt", t0)

    if eval_mask.shape != pred_prob.shape:
        raise ValueError("internal error: cropped shapes mismatch")
    pred_prob = pred_prob.copy()
    pred_prob[~eval_mask] = 0.0

    t0 = time.perf_counter()
    pred_bin = pred_prob >= float(threshold)
    c = confusion_counts(pred_bin, gt_bin, mask=eval_mask)
    _timeit("confusion", t0)

    pred_bin_comp = pred_bin
    pred_prob_comp = pred_prob
    if component_min_area and int(component_min_area) > 0:
        lab, n_lab = _label_components(pred_bin, connectivity=betti_connectivity)
        if n_lab > 0:
            sizes = np.bincount(lab.ravel())[1:].astype(np.int64)
            keep = sizes >= int(component_min_area)
            keep_table = np.zeros(n_lab + 1, dtype=bool)
            keep_table[1:] = keep
            pred_bin_comp = keep_table[lab]
            remove_mask = np.logical_and(pred_bin, ~pred_bin_comp)
            if remove_mask.any():
                pred_prob_comp = pred_prob.copy()
                pred_prob_comp[remove_mask] = 0.0

    # Connected components (compute once for component-based metrics).
    pred_lab, n_pred = _label_components(pred_bin_comp, connectivity=betti_connectivity)
    if "gt_lab" not in locals():
        gt_lab, n_gt = _label_components(gt_bin, connectivity=betti_connectivity)

    t0 = time.perf_counter()
    out: Dict[str, float] = {
        "dice": dice_from_confusion(c),
        "iou": iou_from_confusion(c),
        "accuracy": accuracy_from_confusion(c),
        "precision": precision_from_confusion(c),
        "recall": recall_from_confusion(c),
        "f_beta": fbeta_from_confusion(c, beta=float(fbeta)),
        "voi": voi_from_confusion(c),
        "psnr": psnr(pred_bin, gt_bin),
        "nrm": nrm(pred_bin, gt_bin),
        "mpm": mpm(pred_bin, gt_bin, boundary_k=boundary_k),
        "drd": drd(pred_bin, gt_bin, block_size=drd_block_size),
        "eval_pixels": float(eval_mask.sum()),
        "pred_pos_rate": float(pred_bin[eval_mask].mean()) if eval_mask.any() else float("nan"),
        "gt_pos_rate": float(gt_bin[eval_mask].mean()) if eval_mask.any() else float("nan"),
    }
    _timeit("base_metrics", t0)

    t0 = time.perf_counter()
    out.update({f"betti/{k}": v for k, v in betti_error(pred_bin, gt_bin, connectivity=betti_connectivity).items()})
    out["euler_pred"] = float(euler_characteristic(pred_bin, connectivity=betti_connectivity))
    out["euler_gt"] = float(euler_characteristic(gt_bin, connectivity=betti_connectivity))
    out["abs_euler_err"] = float(abs(out["euler_pred"] - out["euler_gt"]))
    _timeit("betti_euler", t0)

    # Boundary metrics.
    t0 = time.perf_counter()
    if boundary_tols is None:
        boundary_tols = np.asarray([1.0], dtype=np.float32)
    for tau in boundary_tols:
        tau_f = float(tau)
        tau_key = str(tau_f).replace(".", "p")
        bf = boundary_precision_recall_f1(pred_bin, gt_bin, tau=tau_f, boundary_k=boundary_k)
        out[f"boundary/bf1_tau{tau_key}"] = float(bf["b_f1"])
        out[f"boundary/bprec_tau{tau_key}"] = float(bf["b_precision"])
        out[f"boundary/brec_tau{tau_key}"] = float(bf["b_recall"])
        out[f"boundary/nsd_tau{tau_key}"] = float(nsd_surface_dice(pred_bin, gt_bin, tau=tau_f, boundary_k=boundary_k))
        out[f"boundary/biou_tau{tau_key}"] = float(boundary_iou(pred_bin, gt_bin, tau=tau_f, boundary_k=boundary_k))
    _timeit("boundary_bf_nsdbiou", t0)

    t0 = time.perf_counter()
    hd = hausdorff_metrics(pred_bin, gt_bin, boundary_k=boundary_k)
    out["boundary/hd"] = float(hd["hd"])
    out["boundary/hd95"] = float(hd["hd95"])
    out["boundary/assd"] = float(hd["assd"])
    _timeit("boundary_hd", t0)

    # Skeleton / topology metrics.
    t0 = time.perf_counter()
    skel_pred = skeletonize_binary(pred_bin)
    skel_gt = skeletonize_binary(gt_bin)
    out["skeleton/recall"] = float(skeleton_recall(pred_bin, gt_bin, skel_gt=skel_gt))
    out["skeleton/cldice"] = float(
        cldice(pred_bin, gt_bin, skel_pred=skel_pred, skel_gt=skel_gt)
    )
    out["pfm"] = float(pseudo_fmeasure(pred_bin, gt_bin, skel_gt=skel_gt))
    _timeit("skeleton_base", t0)
    if skeleton_radius is None:
        skeleton_radius = np.asarray([1], dtype=np.int64)
    for r in skeleton_radius:
        t0 = time.perf_counter()
        r_i = int(r)
        r_key = str(r_i)
        tube = skeleton_tube_metrics(
            pred_bin,
            gt_bin,
            radius=r_i,
            skel_gt=skel_gt,
        )
        out[f"skeleton/tube_f1_r{r_key}"] = float(tube["tube_f1"])
        out[f"skeleton/tube_precision_r{r_key}"] = float(tube["tube_precision"])
        out[f"skeleton/tube_recall_r{r_key}"] = float(tube["tube_recall"])
        _timeit(f"skeleton_tube_r{r_key}", t0)

    # Component-level metrics.
    t0 = time.perf_counter()
    comp_stats = component_iou_stats(
        pred_bin_comp,
        gt_bin,
        connectivity=betti_connectivity,
        worst_q=component_worst_q,
        worst_k=component_worst_k,
        pred_lab=pred_lab,
        n_pred=n_pred,
        gt_lab=gt_lab,
        n_gt=n_gt,
    )
    out.update({f"components/{k}": float(v) for k, v in comp_stats.items()})
    _timeit("components_iou", t0)

    t0 = time.perf_counter()
    dice_stats = component_dice_stats(
        pred_prob_comp,
        gt_bin,
        connectivity=betti_connectivity,
        pad=component_ssim_pad,
        worst_q=component_worst_q,
        worst_k=component_worst_k,
        gt_lab=gt_lab,
        n_gt=n_gt,
    )
    out.update({f"components/dice_{k}": float(v) for k, v in dice_stats.items()})
    _timeit("components_dice", t0)

    t0 = time.perf_counter()
    pq = panoptic_quality(
        pred_bin_comp,
        gt_bin,
        connectivity=betti_connectivity,
        iou_thr=component_iou_thr,
        pred_lab=pred_lab,
        n_pred=n_pred,
        gt_lab=gt_lab,
        n_gt=n_gt,
    )
    out.update({f"pq/{k}": float(v) for k, v in pq.items()})
    _timeit("components_pq", t0)
    if component_ssim:
        t0 = time.perf_counter()
        ssim_stats = component_ssim_stats(
            pred_prob_comp,
            gt_bin,
            connectivity=betti_connectivity,
            pad=component_ssim_pad,
            ssim_mode=ssim_mode,
            worst_q=component_worst_q,
            worst_k=component_worst_k,
            gt_lab=gt_lab,
            n_gt=n_gt,
        )
        out.update({f"components/ssim_{k}": float(v) for k, v in ssim_stats.items()})
        _timeit("components_ssim", t0)

    t0 = time.perf_counter()
    comp_pr = component_pr_stats(
        pred_prob_comp,
        gt_bin,
        connectivity=betti_connectivity,
        pad=component_ssim_pad,
        num_bins=pr_num_bins,
        worst_q=component_worst_q,
        worst_k=component_worst_k,
        gt_lab=gt_lab,
        n_gt=n_gt,
        eval_mask=eval_mask,
    )
    out.update({f"components/pr_{k}": float(v) for k, v in comp_pr.items()})
    _timeit("components_pr", t0)

    if component_output_dir:
        t0 = time.perf_counter()
        try:
            os.makedirs(component_output_dir, exist_ok=True)
            rows = component_diagnostics(
                pred_prob_comp,
                gt_bin,
                pred_lab=pred_lab,
                n_pred=n_pred,
                gt_lab=gt_lab,
                n_gt=n_gt,
                connectivity=betti_connectivity,
                pad=component_ssim_pad,
                num_bins=pr_num_bins,
                eval_mask=eval_mask,
                worst_q=component_worst_q,
                worst_k=component_worst_k,
                ssim_mode=ssim_mode,
                offset=(int(roi_key[0]) + int(y0), int(roi_key[1]) + int(x0)),
            )
            safe_seg = str(fragment_id).replace("/", "_")
            if rows:
                import csv

                out_path = osp.join(component_output_dir, f"components_{safe_seg}_ds{int(ds)}.csv")
                cols = list(rows[0].keys())
                with open(out_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=cols)
                    writer.writeheader()
                    writer.writerows(rows)

            gt_path = osp.join(component_output_dir, f"components_{safe_seg}_ds{int(ds)}_gt_labels.png")
            pred_path = osp.join(component_output_dir, f"components_{safe_seg}_ds{int(ds)}_pred_labels.png")
            meta_path = osp.join(component_output_dir, f"components_{safe_seg}_ds{int(ds)}_meta.json")
            if not (osp.exists(gt_path) and osp.exists(pred_path) and osp.exists(meta_path)):
                from PIL import Image
                import json as _json

                gt_u16 = gt_lab.astype(np.uint16, copy=False)
                pred_u16 = pred_lab.astype(np.uint16, copy=False)
                Image.fromarray(gt_u16, mode="I;16").save(gt_path)
                Image.fromarray(pred_u16, mode="I;16").save(pred_path)
                off_y, off_x = roi_key
                crop_off_y, crop_off_x = int(y0), int(x0)
                meta = {
                    "segment_id": str(fragment_id),
                    "downsample": int(ds),
                    "roi_offset": [int(off_y), int(off_x)],
                    "crop_offset": [int(crop_off_y), int(crop_off_x)],
                    "full_offset": [int(off_y + crop_off_y), int(off_x + crop_off_x)],
                    "full_shape": [int(pred_full_shape[0]), int(pred_full_shape[1])],
                    "crop_shape": [int(gt_lab.shape[0]), int(gt_lab.shape[1])],
                }
                with open(meta_path, "w") as f:
                    _json.dump(meta, f)
        except Exception:
            pass
        _timeit("components_dump", t0)

    t0 = time.perf_counter()
    pr_stats = _hist_pr_roc(pred_prob, gt_bin, num_bins=pr_num_bins, mask=eval_mask)
    out["pr/auprc"] = float(pr_stats["auprc"])
    out["pr/auroc"] = float(pr_stats["auroc"])
    _timeit("pr_hist", t0)

    # Threshold stability.
    if threshold_grid is not None and len(threshold_grid):
        t0 = time.perf_counter()
        tgrid = np.asarray(threshold_grid, dtype=np.float32)
        dice_stab = threshold_stability(
            pred_prob,
            gt_bin,
            thresholds=tgrid,
            metric_fn=lambda p, g, m=eval_mask: dice_from_confusion(confusion_counts(p, g, mask=m)),
        )
        iou_stab = threshold_stability(
            pred_prob,
            gt_bin,
            thresholds=tgrid,
            metric_fn=lambda p, g, m=eval_mask: iou_from_confusion(confusion_counts(p, g, mask=m)),
        )
        fbeta_stab = threshold_stability(
            pred_prob,
            gt_bin,
            thresholds=tgrid,
            metric_fn=lambda p, g, m=eval_mask: fbeta_from_confusion(confusion_counts(p, g, mask=m), beta=float(fbeta)),
        )
        out.update({f"stability/dice_{k}": float(v) for k, v in dice_stab.items()})
        out.update({f"stability/iou_{k}": float(v) for k, v in iou_stab.items()})
        out.update({f"stability/fbeta_{k}": float(v) for k, v in fbeta_stab.items()})
        _timeit("threshold_stability", t0)

    # Save skeleton images locally (optional).
    if output_dir is not None and save_skeleton_images:
        try:
            os.makedirs(output_dir, exist_ok=True)
            safe_seg = str(fragment_id).replace("/", "_")
            thr_tag = str(float(threshold)).replace(".", "p")
            pred_path = osp.join(output_dir, f"{safe_seg}_pred_thr{thr_tag}.png")
            gt_path = osp.join(output_dir, f"{safe_seg}_gt.png")
            if (not osp.exists(pred_path)) or (not osp.exists(gt_path)):
                from PIL import Image

                pred_img = (skel_pred.astype(np.uint8) * 255)
                gt_img = (skel_gt.astype(np.uint8) * 255)
                Image.fromarray(pred_img).save(pred_path)
                Image.fromarray(gt_img).save(gt_path)
        except Exception:
            pass

    # Persistent homology distances (optional).
    if compute_persistence:
        t0 = time.perf_counter()
        out.update(
            {
                f"ph/{k}": float(v)
                for k, v in persistence_distances(
                    pred_prob,
                    gt_bin,
                    downsample=persistence_downsample,
                ).items()
            }
        )
        _timeit("persistence", t0)

    try:
        from train_resnet3d_lib.config import log as _log
    except Exception:
        _log = None
    total = sum(timings.values())
    parts = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
    msg = (
        f"metrics timing segment={seg} ds={ds} cache={'hit' if cache_hit else 'miss'} "
        f"total={total:.3f}s "
        + ", ".join([f"{k}={v:.3f}s" for k, v in parts])
    )
    if _log is None:
        print(msg, flush=True)
    else:
        _log(msg)

    return out


# Metrics implementation notes:
# Implemented: confusion-based, boundary metrics (BF1/NSD/HD/HD95/ASSD/Boundary-IoU),
# skeleton metrics (clDice, skeleton recall, tubed skeleton F1, p-FM) via scikit-image,
# component stats + PQ (worst-k and worst-q summaries),
# component-level SSIM (optional), DIBCO metrics (NRM/MPM/DRD/PSNR), Betti/Euler,
# threshold stability, SSIM, persistence distances (Gudhi optional).
# Next: consider component-wise boundary/skeleton summaries if needed, add exact DIBCO weights
# for DRD/MPM if you provide the official evaluator, and optionally add chamfer/DTW metrics.
