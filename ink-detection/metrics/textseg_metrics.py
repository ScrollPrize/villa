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


def accuracy_from_confusion(c: Confusion) -> float:
    return _safe_div(c.tp + c.tn, c.tp + c.fp + c.fn + c.tn)


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


def skeleton_chamfer(skel_pred: np.ndarray, skel_gt: np.ndarray) -> Dict[str, float]:
    from scipy.ndimage import distance_transform_edt

    skel_pred = _as_bool_2d(skel_pred)
    skel_gt = _as_bool_2d(skel_gt)

    if skel_pred.sum() == 0 and skel_gt.sum() == 0:
        return {"chamfer": 0.0, "pred_to_gt": 0.0, "gt_to_pred": 0.0}
    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        diag = float(math.hypot(skel_pred.shape[0], skel_pred.shape[1]))
        return {"chamfer": diag, "pred_to_gt": diag, "gt_to_pred": diag}

    dt_to_gt = distance_transform_edt((~skel_gt).astype(np.uint8))
    dt_to_pred = distance_transform_edt((~skel_pred).astype(np.uint8))
    pred_to_gt = float(dt_to_gt[skel_pred].mean())
    gt_to_pred = float(dt_to_pred[skel_gt].mean())
    return {
        "chamfer": float(0.5 * (pred_to_gt + gt_to_pred)),
        "pred_to_gt": pred_to_gt,
        "gt_to_pred": gt_to_pred,
    }


def _label_components(mask: np.ndarray, *, connectivity: int = 2) -> Tuple[np.ndarray, int]:
    from scipy.ndimage import label as cc_label

    mask = _as_bool_2d(mask)
    struct = _cc_structure(connectivity)
    lab, n = cc_label(mask, structure=struct)
    return lab, int(n)


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
    component_worst_q: Optional[float] = 0.1,
    component_worst_k: Optional[int] = 8,
    component_min_area: int = 0,
    component_pad: int = 2,
    threshold_grid: Optional[np.ndarray] = None,
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
        gt_beta0 = int(cache["gt_beta0"])
        gt_beta1 = int(cache["gt_beta1"])
        gt_euler = int(cache["gt_euler"])
        skel_gt = cache["skel_gt"]
        y0, y1, x0, x1 = cache["crop"]
        pred_full_shape = tuple(cache.get("full_shape", pred_full_shape))
        pred_prob = pred_prob[y0:y1, x0:x1]
        pred_has = pred_has[y0:y1, x0:x1]
    else:
        cache_hit = False
        t0 = time.perf_counter()
        from train_resnet3d_lib.config import CFG

        dataset_root = str(getattr(CFG, "dataset_root", "train_scrolls"))
        label_base = osp.join(dataset_root, seg, f"{seg}_inklabels{label_suffix}")
        mask_base = osp.join(dataset_root, seg, f"{seg}_mask{mask_suffix}")
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
        gt_beta0, gt_beta1 = betti_numbers_2d(gt_bin, connectivity=betti_connectivity)
        gt_euler = int(gt_beta0 - gt_beta1)
        skel_gt = skeletonize_binary(gt_bin)
        _gt_cache_put(
            cache_key,
            {
                "gt_bin": gt_bin,
                "eval_mask": eval_mask,
                "gt_lab": gt_lab,
                "n_gt": n_gt,
                "gt_beta0": int(gt_beta0),
                "gt_beta1": int(gt_beta1),
                "gt_euler": int(gt_euler),
                "skel_gt": skel_gt,
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
        "accuracy": accuracy_from_confusion(c),
        "f_beta": fbeta_from_confusion(c, beta=float(fbeta)),
        "voi": voi_from_confusion(c),
        "psnr": psnr(pred_bin, gt_bin),
        "mpm": mpm(pred_bin, gt_bin, boundary_k=boundary_k),
        "drd": drd(pred_bin, gt_bin, block_size=drd_block_size),
        "eval_pixels": float(eval_mask.sum()),
        "pred_pos_rate": float(pred_bin[eval_mask].mean()) if eval_mask.any() else float("nan"),
        "gt_pos_rate": float(gt_bin[eval_mask].mean()) if eval_mask.any() else float("nan"),
    }
    _timeit("base_metrics", t0)

    t0 = time.perf_counter()
    pred_beta0, pred_beta1 = betti_numbers_2d(pred_bin_comp, connectivity=betti_connectivity)
    out["betti/beta0_pred"] = float(pred_beta0)
    out["betti/beta1_pred"] = float(pred_beta1)
    out["betti/beta0_gt"] = float(gt_beta0)
    out["betti/beta1_gt"] = float(gt_beta1)
    out["betti/abs_beta0_err"] = float(abs(pred_beta0 - gt_beta0))
    out["betti/abs_beta1_err"] = float(abs(pred_beta1 - gt_beta1))
    out["betti/l1_betti_err"] = float(abs(pred_beta0 - gt_beta0) + abs(pred_beta1 - gt_beta1))
    out["euler_pred"] = float(pred_beta0 - pred_beta1)
    out["euler_gt"] = float(gt_euler)
    out["abs_euler_err"] = float(abs(out["euler_pred"] - out["euler_gt"]))
    _timeit("betti_euler", t0)

    # Boundary metrics on filtered prediction to reduce tiny-component noise effects.
    t0 = time.perf_counter()
    if boundary_tols is None:
        boundary_tols = np.asarray([1.0], dtype=np.float32)
    for tau in boundary_tols:
        tau_f = float(tau)
        tau_key = str(tau_f).replace(".", "p")
        bf = boundary_precision_recall_f1(pred_bin_comp, gt_bin, tau=tau_f, boundary_k=boundary_k)
        out[f"boundary/bf1_tau{tau_key}"] = float(bf["b_f1"])
        out[f"boundary/nsd_tau{tau_key}"] = float(nsd_surface_dice(pred_bin_comp, gt_bin, tau=tau_f, boundary_k=boundary_k))
    _timeit("boundary_bf_nsd", t0)

    t0 = time.perf_counter()
    hd = hausdorff_metrics(pred_bin_comp, gt_bin, boundary_k=boundary_k)
    out["boundary/hd"] = float(hd["hd"])
    out["boundary/hd95"] = float(hd["hd95"])
    out["boundary/assd"] = float(hd["assd"])
    _timeit("boundary_hd", t0)

    # Skeleton / topology metrics.
    t0 = time.perf_counter()
    skel_pred = skeletonize_binary(pred_bin_comp)
    out["skeleton/recall"] = float(skeleton_recall(pred_bin_comp, gt_bin, skel_gt=skel_gt))
    out["skeleton/cldice"] = float(
        cldice(pred_bin_comp, gt_bin, skel_pred=skel_pred, skel_gt=skel_gt)
    )
    out["pfm"] = float(pseudo_fmeasure(pred_bin_comp, gt_bin, skel_gt=skel_gt))
    chamfer = skeleton_chamfer(skel_pred, skel_gt)
    out["skeleton/chamfer"] = float(chamfer["chamfer"])
    out["skeleton/chamfer_pred_to_gt"] = float(chamfer["pred_to_gt"])
    out["skeleton/chamfer_gt_to_pred"] = float(chamfer["gt_to_pred"])
    _timeit("skeleton_base", t0)
    if skeleton_radius is None:
        skeleton_radius = np.asarray([1], dtype=np.int64)
    for r in skeleton_radius:
        t0 = time.perf_counter()
        r_i = int(r)
        r_key = str(r_i)
        tube = skeleton_tube_metrics(
            pred_bin_comp,
            gt_bin,
            radius=r_i,
            skel_gt=skel_gt,
        )
        out[f"skeleton/tube_f1_r{r_key}"] = float(tube["tube_f1"])
        _timeit(f"skeleton_tube_r{r_key}", t0)

    # Component-level dice summaries.
    t0 = time.perf_counter()
    dice_stats = component_dice_stats(
        pred_prob_comp,
        gt_bin,
        connectivity=betti_connectivity,
        pad=component_pad,
        worst_q=component_worst_q,
        worst_k=component_worst_k,
        gt_lab=gt_lab,
        n_gt=n_gt,
    )
    out.update({f"components/dice_{k}": float(v) for k, v in dice_stats.items()})
    out["components/n_pred"] = float(n_pred)
    _timeit("components_dice", t0)

    if component_output_dir:
        t0 = time.perf_counter()
        os.makedirs(component_output_dir, exist_ok=True)
        safe_seg = str(fragment_id).replace("/", "_")
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
        _timeit("components_dump", t0)

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
        fbeta_stab = threshold_stability(
            pred_prob,
            gt_bin,
            thresholds=tgrid,
            metric_fn=lambda p, g, m=eval_mask: fbeta_from_confusion(confusion_counts(p, g, mask=m), beta=float(fbeta)),
        )
        out.update({f"stability/dice_{k}": float(v) for k, v in dice_stab.items()})
        out.update({f"stability/fbeta_{k}": float(v) for k, v in fbeta_stab.items()})
        _timeit("threshold_stability", t0)

    # Save skeleton images locally (optional).
    if output_dir is not None and save_skeleton_images:
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

    from train_resnet3d_lib.config import log as _log
    total = sum(timings.values())
    parts = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
    msg = (
        f"metrics timing segment={seg} ds={ds} cache={'hit' if cache_hit else 'miss'} "
        f"total={total:.3f}s "
        + ", ".join([f"{k}={v:.3f}s" for k, v in parts])
    )
    _log(msg)

    return out


# Metrics implementation notes:
# Implemented: confusion-based metrics, boundary metrics (BF1/NSD/HD/HD95/ASSD),
# skeleton metrics (clDice, skeleton recall, tube F1, p-FM, chamfer),
# component-level dice summaries, DIBCO metrics (MPM/DRD/PSNR), Betti/Euler,
# and threshold stability summaries.
