from __future__ import annotations

import math
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


def _as_int_labels_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected 2D label array, got shape={x.shape}")
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"expected integer label dtype, got {x.dtype}")
    return x.astype(np.int64, copy=False)


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


def voi_split_merge_from_labels(gt_lab: np.ndarray, pred_lab: np.ndarray) -> Tuple[float, float]:
    """Compute VI split/merge from two label partitions.

    Convention:
      - split = H(Pred | GT)  (over-segmentation)
      - merge = H(GT | Pred)  (under-segmentation)

    Background handling:
      - ignore only shared background voxels where gt_lab == 0 and pred_lab == 0
      - keep (0, j) and (i, 0) overlaps so foreground/background mistakes are counted
    """
    gt_lab = _as_int_labels_2d(gt_lab)
    pred_lab = _as_int_labels_2d(pred_lab)
    if gt_lab.shape != pred_lab.shape:
        raise ValueError(f"gt_lab/pred_lab shape mismatch: {gt_lab.shape} vs {pred_lab.shape}")
    if gt_lab.size == 0:
        return float("nan"), float("nan")

    gt_flat = gt_lab.reshape(-1)
    pred_flat = pred_lab.reshape(-1)
    valid = np.logical_not(np.logical_and(gt_flat == 0, pred_flat == 0))
    if not valid.any():
        return 0.0, 0.0
    gt_flat = gt_flat[valid]
    pred_flat = pred_flat[valid]
    total = float(gt_flat.size)
    if total <= 0:
        return float("nan"), float("nan")

    gt_labels, gt_counts = np.unique(gt_flat, return_counts=True)
    pred_labels, pred_counts = np.unique(pred_flat, return_counts=True)

    pairs = np.empty(gt_flat.shape[0], dtype=[("g", gt_flat.dtype), ("p", pred_flat.dtype)])
    pairs["g"] = gt_flat
    pairs["p"] = pred_flat
    pair_labels, pair_counts = np.unique(pairs, return_counts=True)

    gt_idx = np.searchsorted(gt_labels, pair_labels["g"])
    pred_idx = np.searchsorted(pred_labels, pair_labels["p"])

    p_i = gt_counts[gt_idx].astype(np.float64) / total
    p_j = pred_counts[pred_idx].astype(np.float64) / total
    p_ij = pair_counts.astype(np.float64) / total

    split = float(np.sum(p_ij * np.log2(p_i / p_ij)))
    merge = float(np.sum(p_ij * np.log2(p_j / p_ij)))
    return split, merge


def voi_from_component_labels(gt_lab: np.ndarray, pred_lab: np.ndarray) -> float:
    split, merge = voi_split_merge_from_labels(gt_lab, pred_lab)
    return float(split + merge)


def soft_dice_from_prob(pred_prob: np.ndarray, gt_bin: np.ndarray, *, eps: float = 1e-7) -> float:
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    gt_bin = _as_bool_2d(gt_bin)
    if pred_prob.shape != gt_bin.shape:
        raise ValueError(f"pred_prob/gt_bin shape mismatch: {pred_prob.shape} vs {gt_bin.shape}")
    gt_f = gt_bin.astype(np.float32)
    inter = float((pred_prob * gt_f).sum())
    denom = float(pred_prob.sum() + gt_f.sum())
    return float((2.0 * inter + eps) / (denom + eps))


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


def _count_holes_2d(mask: np.ndarray, *, connectivity: int = 2) -> int:
    from scipy.ndimage import label as cc_label

    mask = _as_bool_2d(mask)
    H, W = mask.shape
    if H == 0 or W == 0:
        return 0

    struct = _cc_structure(connectivity)
    bg_lab, bg_n = cc_label(~mask, structure=struct)
    if bg_n == 0:
        return 0

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    touching = set(np.unique(bg_lab[border])) - {0}
    all_bg = set(range(1, int(bg_n) + 1))
    holes = all_bg - touching
    return int(len(holes))


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
    import cv2

    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "thinning"):
        raise ImportError(
            "cv2.ximgproc.thinning is required for skeleton metrics. "
            "Install OpenCV contrib (opencv-contrib-python)."
        )
    mask_u8 = (mask.astype(np.uint8) * 255)
    skel_u8 = cv2.ximgproc.thinning(mask_u8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return skel_u8 > 0


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


def _local_metrics_from_binary(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    *,
    connectivity: int,
    drd_block_size: int,
    boundary_k: int,
    gt_beta0: Optional[int] = None,
    gt_beta1: Optional[int] = None,
    skel_gt: Optional[np.ndarray] = None,
    skel_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    pred_bin = _as_bool_2d(pred_bin)
    gt_bin = _as_bool_2d(gt_bin)
    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"pred_bin/gt_bin shape mismatch: {pred_bin.shape} vs {gt_bin.shape}")

    c = confusion_counts(pred_bin, gt_bin)
    gt_lab, _ = _label_components(gt_bin, connectivity=connectivity)
    pred_lab, _ = _label_components(pred_bin, connectivity=connectivity)
    voi_split, voi_merge = voi_split_merge_from_labels(gt_lab, pred_lab)
    voi_total = float(voi_split + voi_merge)
    if gt_beta0 is None or gt_beta1 is None:
        gt_beta0_i, gt_beta1_i = betti_numbers_2d(gt_bin, connectivity=connectivity)
    else:
        gt_beta0_i = int(gt_beta0)
        gt_beta1_i = int(gt_beta1)
    pred_beta0_i, pred_beta1_i = betti_numbers_2d(pred_bin, connectivity=connectivity)
    abs_beta0_err = float(abs(pred_beta0_i - gt_beta0_i))
    abs_beta1_err = float(abs(pred_beta1_i - gt_beta1_i))
    betti_l1 = float(abs_beta0_err + abs_beta1_err)
    euler_pred = float(pred_beta0_i - pred_beta1_i)
    euler_gt = float(gt_beta0_i - gt_beta1_i)
    abs_euler_err = float(abs(euler_pred - euler_gt))

    hd = hausdorff_metrics(pred_bin, gt_bin, boundary_k=boundary_k)
    if skel_gt is None:
        skel_gt = skeletonize_binary(gt_bin)
    if skel_pred is None:
        skel_pred = skeletonize_binary(pred_bin)
    chamfer = skeleton_chamfer(skel_pred, skel_gt)

    return {
        "dice": float(dice_from_confusion(c)),
        "accuracy": float(accuracy_from_confusion(c)),
        "voi": voi_total,
        "mpm": float(mpm(pred_bin, gt_bin, boundary_k=boundary_k)),
        "drd": float(drd(pred_bin, gt_bin, block_size=drd_block_size)),
        "betti_beta0_pred": float(pred_beta0_i),
        "betti_beta1_pred": float(pred_beta1_i),
        "betti_beta0_gt": float(gt_beta0_i),
        "betti_beta1_gt": float(gt_beta1_i),
        "betti_abs_beta0_err": abs_beta0_err,
        "betti_abs_beta1_err": abs_beta1_err,
        "betti_l1": betti_l1,
        "euler_pred": euler_pred,
        "euler_gt": euler_gt,
        "abs_euler_err": abs_euler_err,
        "boundary_hd": float(hd["hd"]),
        "boundary_hd95": float(hd["hd95"]),
        "boundary_assd": float(hd["assd"]),
        "skeleton_recall": float(skeleton_recall(pred_bin, gt_bin, skel_gt=skel_gt)),
        "skeleton_cldice": float(cldice(pred_bin, gt_bin, skel_pred=skel_pred, skel_gt=skel_gt)),
        "pfm": float(pseudo_fmeasure(pred_bin, gt_bin, skel_gt=skel_gt)),
        "skeleton_chamfer": float(chamfer["chamfer"]),
        "skeleton_chamfer_pred_to_gt": float(chamfer["pred_to_gt"]),
        "skeleton_chamfer_gt_to_pred": float(chamfer["gt_to_pred"]),
    }
