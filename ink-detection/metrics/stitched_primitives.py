from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import edt
import numpy as np
from kimimaro.trace import trace


EPS = 1e-8

DEFAULT_TEASAR_PARAMS = {
    "scale": 1.5,
    "const": 300,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_acceptance_threshold": 3500,
    "soma_detection_threshold": 750,
    "soma_invalidation_const": 300,
    "soma_invalidation_scale": 2,
}


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d + EPS)


def _normalize_skeleton_method(skeleton_method: str) -> str:
    method = str(skeleton_method).strip().lower()
    if method in {"zhang_suen", "zhangsuen", "zhang-suen"}:
        return "zhang_suen"
    if method in {"guo_hall", "guohall", "guo-hall"}:
        return "guo_hall"
    if method == "kimimaro":
        return "kimimaro"
    raise ValueError(
        "skeleton_method must be one of {'zhang_suen', 'guo_hall', 'kimimaro'}, "
        f"got {skeleton_method!r}"
    )


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
    return x


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


def _cc_connectivity_cv2(connectivity: int) -> int:
    connectivity = int(connectivity)
    if connectivity == 1:
        return 4
    if connectivity == 2:
        return 8
    raise ValueError(f"connectivity must be 1 (4-neighborhood) or 2 (8-neighborhood), got {connectivity}")


def _dual_connectivity(connectivity: int) -> int:
    connectivity = int(connectivity)
    if connectivity == 1:
        return 2
    if connectivity == 2:
        return 1
    raise ValueError(f"connectivity must be 1 (4-neighborhood) or 2 (8-neighborhood), got {connectivity}")


def _hole_labels_2d(mask: np.ndarray, *, connectivity: int = 2) -> Tuple[np.ndarray, int]:
    import cv2

    mask = _as_bool_2d(mask)
    H, W = mask.shape
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.int32), 0

    bg_conn = _cc_connectivity_cv2(_dual_connectivity(connectivity))
    n_all, bg_lab = cv2.connectedComponents((~mask).astype(np.uint8, copy=False), connectivity=bg_conn)
    n_all_i = int(n_all)
    if n_all_i <= 1:
        return np.zeros_like(bg_lab, dtype=np.int32), 0

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    touching = np.unique(bg_lab[border])

    is_hole = np.ones(n_all_i, dtype=bool)
    is_hole[0] = False
    is_hole[touching] = False

    hole_ids = np.flatnonzero(is_hole).astype(np.int32, copy=False)
    n_holes = int(hole_ids.size)
    if n_holes <= 0:
        return np.zeros_like(bg_lab, dtype=np.int32), 0

    remap = np.zeros(n_all_i, dtype=np.int32)
    remap[hole_ids] = np.arange(1, n_holes + 1, dtype=np.int32)
    hole_lab = remap[bg_lab]
    return hole_lab, n_holes


def betti_numbers_2d(mask: np.ndarray, *, connectivity: int = 2) -> Tuple[int, int]:
    """Compute (beta0, beta1) for a 2D binary foreground mask.

    beta0: number of connected components in the foreground.
    beta1: number of holes (background components not touching the border).
    """
    import cv2

    mask = _as_bool_2d(mask)
    H, W = mask.shape
    if H == 0 or W == 0:
        return 0, 0

    fg_conn = _cc_connectivity_cv2(connectivity)
    fg_n_all, _fg_lab = cv2.connectedComponents(mask.astype(np.uint8, copy=False), connectivity=fg_conn)
    beta0 = int(max(0, int(fg_n_all) - 1))

    _hole_lab, beta1 = _hole_labels_2d(mask, connectivity=connectivity)
    return beta0, beta1


def _count_holes_2d(mask: np.ndarray, *, connectivity: int = 2) -> int:
    _hole_lab, n_holes = _hole_labels_2d(mask, connectivity=connectivity)
    return int(n_holes)


_BETTI_MATCHING_BACKEND = None


def _get_betti_matching_backend():
    global _BETTI_MATCHING_BACKEND
    if _BETTI_MATCHING_BACKEND is not None:
        return _BETTI_MATCHING_BACKEND

    try:
        import betti_matching as backend
    except Exception:
        try:
            from topolosses.losses.betti_matching.src import betti_matching as backend
        except Exception as exc:
            raise ImportError(
                "Strict Betti matching backend not available. Install `betti_matching` "
                "or `topolosses` with its betti-matching extension."
            ) from exc

    _BETTI_MATCHING_BACKEND = backend
    return _BETTI_MATCHING_BACKEND


def _num_features_from_coordinates(coords) -> int:
    arr = np.asarray(coords)
    if arr.size == 0:
        return 0
    if arr.ndim == 0:
        return 1
    return int(arr.shape[0])


def betti_matching_error(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
) -> Dict[str, float]:
    pred_bin_b = _as_bool_2d(pred_bin)
    gt_bin_b = _as_bool_2d(gt_bin)
    if pred_bin_b.shape != gt_bin_b.shape:
        raise ValueError(f"pred_bin/gt_bin shape mismatch: {pred_bin_b.shape} vs {gt_bin_b.shape}")

    backend = _get_betti_matching_backend()

    # Betti-matching implementations use sublevel filtration; for segmentation we
    # evaluate superlevel topology by inverting binary masks.
    pred_super = np.ascontiguousarray(1.0 - pred_bin_b.astype(np.float64))
    gt_super = np.ascontiguousarray(1.0 - gt_bin_b.astype(np.float64))

    results = backend.compute_matching([pred_super], [gt_super])
    if len(results) != 1:
        raise ValueError(f"expected one matching result, got {len(results)}")
    result = results[0]

    dims = len(result.input1_unmatched_birth_coordinates)
    err_by_dim = np.zeros((max(2, int(dims)),), dtype=np.float64)
    pair_by_dim = np.zeros((max(2, int(dims)),), dtype=np.float64)
    for dim in range(int(dims)):
        n_pred_unmatched = _num_features_from_coordinates(result.input1_unmatched_birth_coordinates[dim])
        n_gt_unmatched = _num_features_from_coordinates(result.input2_unmatched_birth_coordinates[dim])
        n_matched = _num_features_from_coordinates(result.input1_matched_birth_coordinates[dim])
        err_by_dim[dim] = float(n_pred_unmatched + n_gt_unmatched)
        pair_by_dim[dim] = float(n_matched)

    return {
        "betti_match_err": float(err_by_dim[0] + err_by_dim[1]),
        "betti_match_err_dim0": float(err_by_dim[0]),
        "betti_match_err_dim1": float(err_by_dim[1]),
    }


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
    bs = int(block_size)
    if bs < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size!r}")
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


def _draw_edge_2d(mask: np.ndarray, *, y0: float, x0: float, y1: float, x1: float) -> None:
    steps = max(int(np.ceil(abs(y1 - y0))), int(np.ceil(abs(x1 - x0))), 1)
    ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(np.int64, copy=False)
    xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(np.int64, copy=False)
    valid = (ys >= 0) & (ys < mask.shape[0]) & (xs >= 0) & (xs < mask.shape[1])
    if valid.any():
        mask[ys[valid], xs[valid]] = True


def _trace_skeleton_to_mask_2d(skel: object, *, out_shape: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(out_shape, dtype=bool)
    vertices = np.asarray(skel.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise ValueError(f"kimimaro skeleton vertices must have shape (N, 2) or (N, 3), got {vertices.shape}")
    if vertices.shape[0] == 0:
        return out

    edges = np.asarray(skel.edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"kimimaro skeleton edges must have shape (M, 2), got {edges.shape}")

    if edges.shape[0] == 0:
        y0 = int(np.rint(vertices[0, 0]))
        x0 = int(np.rint(vertices[0, 1]))
        if 0 <= y0 < out_shape[0] and 0 <= x0 < out_shape[1]:
            out[y0, x0] = True
        return out

    n_vertices = int(vertices.shape[0])
    for edge in edges:
        vi0 = int(edge[0])
        vi1 = int(edge[1])
        if vi0 < 0 or vi0 >= n_vertices or vi1 < 0 or vi1 >= n_vertices:
            raise ValueError(
                f"kimimaro edge has invalid vertex index: {(vi0, vi1)} for n_vertices={n_vertices}"
            )
        y0 = float(vertices[vi0, 0])
        x0 = float(vertices[vi0, 1])
        y1 = float(vertices[vi1, 0])
        x1 = float(vertices[vi1, 1])
        _draw_edge_2d(out, y0=y0, x0=x0, y1=y1, x1=x1)

    return out


def _skeletonize_binary_opencv(mask: np.ndarray, *, thinning_type: str) -> np.ndarray:
    import cv2

    mask = _as_bool_2d(mask)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "thinning"):
        raise ImportError(
            "cv2.ximgproc.thinning is required for OpenCV skeleton methods. "
            "Install opencv-contrib-python-headless."
        )
    thinning_method = _normalize_skeleton_method(thinning_type)
    if thinning_method == "zhang_suen":
        thinning_type_cv2 = int(cv2.ximgproc.THINNING_ZHANGSUEN)
    elif thinning_method == "guo_hall":
        thinning_type_cv2 = int(cv2.ximgproc.THINNING_GUOHALL)
    else:
        raise ValueError(
            "OpenCV thinning_type must be one of {'zhang_suen', 'guo_hall'}, "
            f"got {thinning_type!r}"
        )
    mask_u8 = mask.astype(np.uint8, copy=False) * 255
    skel_u8 = cv2.ximgproc.thinning(mask_u8, thinningType=thinning_type_cv2)
    return skel_u8 > 0


def skeletonize_binary(
    mask: np.ndarray,
    *,
    cc_labels: Optional[np.ndarray] = None,
    skeleton_method: str = "guo_hall",
) -> np.ndarray:
    mask = _as_bool_2d(mask)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    method = _normalize_skeleton_method(skeleton_method)
    if method in {"zhang_suen", "guo_hall"}:
        return _skeletonize_binary_opencv(mask, thinning_type=method)

    if cc_labels is None:
        cc_labels_i, _ = _label_components(mask, connectivity=2)
    else:
        cc_labels_i = _as_int_labels_2d(cc_labels)
        if cc_labels_i.shape != mask.shape:
            raise ValueError(f"cc_labels/mask shape mismatch: {cc_labels_i.shape} vs {mask.shape}")
        if np.any(cc_labels_i < 0):
            raise ValueError("cc_labels must be non-negative")
        if not np.array_equal(mask, cc_labels_i > 0):
            raise ValueError("cc_labels foreground (labels > 0) must exactly match mask foreground")

    skel_mask = np.zeros_like(mask, dtype=bool)
    cc_ids = np.unique(cc_labels_i)
    for cc_id in cc_ids:
        cc_id_i = int(cc_id)
        if cc_id_i == 0:
            continue
        component_mask = cc_labels_i == cc_id_i
        if not component_mask.any():
            raise ValueError(f"connected component id={cc_id_i} has no foreground pixels")

        mask3 = np.asfortranarray(component_mask[..., None].astype(np.uint8, copy=False))
        dbf = edt.edt(mask3, anisotropy=(1, 1, 1), black_border=True)
        skel = trace(
            mask3,
            dbf,
            anisotropy=(1, 1, 1),
            manual_targets_before=[],
            manual_targets_after=[],
            root=None,
            **DEFAULT_TEASAR_PARAMS,
        )
        skel_mask |= _trace_skeleton_to_mask_2d(skel, out_shape=mask.shape)

    return skel_mask


def skeleton_recall(pred: np.ndarray, gt: np.ndarray, *, skel_gt: Optional[np.ndarray] = None) -> float:
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if skel_gt is None:
        skel_gt = skeletonize_binary(gt)
    if skel_gt.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return _safe_div(np.logical_and(skel_gt, pred).sum(), skel_gt.sum())


def _pseudo_contour_mask(gt: np.ndarray) -> np.ndarray:
    """Return a GT contour mask close to the DIBCO/TIP contour extraction setup."""
    gt = _as_bool_2d(gt)
    if not gt.any():
        return np.zeros_like(gt, dtype=bool)
    try:
        import cv2
    except Exception as exc:
        raise ImportError(
            "OpenCV is required for weighted pseudo-FMeasure contour extraction; "
            "install opencv-contrib-python-headless."
        ) from exc

    gt_u8 = gt.astype(np.uint8, copy=False) * 255
    blurred = cv2.GaussianBlur(gt_u8, (0, 0), sigmaX=1.5, sigmaY=1.5, borderType=cv2.BORDER_REPLICATE)
    canny = cv2.Canny(blurred, threshold1=int(round(0.2 * 255.0)), threshold2=int(round(0.3 * 255.0)))
    contour = np.logical_and(canny > 0, gt)
    if not contour.any():
        raise ValueError("weighted pseudo-FMeasure contour extraction produced an empty contour for non-empty GT")
    return contour


def _pseudo_recall_normalizer(c_i: int) -> float:
    """Normalization N(C_i) from Ntirogiannis et al. (TIP 2013, Eq. 6)."""
    c_i = max(1, int(c_i))
    if (c_i % 2) == 1:
        n = ((float(c_i) - 1.0) / 2.0) ** 2
    else:
        half = float(c_i) / 2.0
        n = half * (half - 1.0)
    return float(max(1.0, n))


def _pseudo_weight_maps(
    gt: np.ndarray,
    *,
    connectivity: int,
    skel_gt: Optional[np.ndarray] = None,
    contour_gt: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build DIBCO-style recall/precision weight maps (GW/PW)."""
    from scipy.ndimage import distance_transform_cdt, distance_transform_edt

    gt = _as_bool_2d(gt)
    gw = np.zeros(gt.shape, dtype=np.float64)
    pw = np.ones(gt.shape, dtype=np.float64)
    if not gt.any():
        return gw, pw

    gt_lab, n_gt = _label_components(gt, connectivity=connectivity)
    if n_gt <= 0:
        return gw, pw

    if contour_gt is None:
        contour = _pseudo_contour_mask(gt)
    else:
        contour = _as_bool_2d(contour_gt)
        if contour.shape != gt.shape:
            raise ValueError(f"contour_gt/gt shape mismatch: {contour.shape} vs {gt.shape}")
        if np.logical_and(contour, ~gt).any():
            raise ValueError("contour_gt must be a subset of gt foreground")
        if gt.any() and (not contour.any()):
            raise ValueError("contour_gt is empty for non-empty GT")
    d_cm = distance_transform_cdt((~contour).astype(np.uint8), metric="chessboard").astype(np.float64)

    contour_labels = np.where(contour, gt_lab, 0).astype(np.int32, copy=False)
    _d_edt, nearest_contour_inds = distance_transform_edt((~contour).astype(np.uint8), return_indices=True)
    nearest_contour_label = contour_labels[nearest_contour_inds[0], nearest_contour_inds[1]]

    c_by_label = np.ones(int(n_gt) + 1, dtype=np.float64)
    n_by_label = np.ones(int(n_gt) + 1, dtype=np.float64)
    sw_by_label = np.ones(int(n_gt) + 1, dtype=np.float64)

    if skel_gt is None:
        try:
            skel_all = skeletonize_binary(gt)
        except Exception as exc:
            raise RuntimeError("failed to skeletonize GT while building weighted pseudo-FMeasure maps") from exc
    else:
        skel_all = _as_bool_2d(skel_gt)
        if skel_all.shape != gt.shape:
            raise ValueError(f"skel_gt/gt shape mismatch: {skel_all.shape} vs {gt.shape}")
        if np.logical_and(skel_all, ~gt).any():
            raise ValueError("skel_gt must be a subset of gt foreground")

    for label_id in range(1, int(n_gt) + 1):
        component = gt_lab == label_id
        if not component.any():
            continue
        skel_component = np.logical_and(skel_all, component)
        sw_samples = d_cm[skel_component]
        if sw_samples.size == 0:
            sw_samples = d_cm[component]
        if sw_samples.size == 0:
            continue

        c_i = max(1, int(math.ceil(float(sw_samples.max()))))
        c_by_label[label_id] = float(c_i)
        n_by_label[label_id] = _pseudo_recall_normalizer(c_i)
        sw_by_label[label_id] = max(1.0, float(2.0 * sw_samples.mean()))

    inside = gt
    d_in = d_cm[inside]
    labels_in = gt_lab[inside]
    c_in = c_by_label[labels_in]
    n_in = n_by_label[labels_in]

    w_in = np.empty_like(d_in, dtype=np.float64)
    on_contour = d_in <= 0.0
    in_core = np.logical_and(d_in > 0.0, d_in < c_in)
    in_plateau = np.logical_not(np.logical_or(on_contour, in_core))

    w_in[on_contour] = 1.0
    w_in[in_core] = d_in[in_core] / n_in[in_core]
    w_in[in_plateau] = c_in[in_plateau] / n_in[in_plateau]
    gw[inside] = w_in

    outside = np.logical_not(gt)
    sw_map = sw_by_label[nearest_contour_label]
    near = np.logical_and(outside, np.logical_and(nearest_contour_label > 0, d_cm <= sw_map))
    if near.any():
        pw[near] = 1.0 + (d_cm[near] / np.maximum(1.0, sw_map[near]))
        pw[near] = np.minimum(2.0, pw[near])

    return gw, pw


def pseudo_fmeasure_values(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    skel_gt: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Return `(pfm, pfm_nonempty)` with shared intermediate computation."""
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if skel_gt is None:
        skel_gt = skeletonize_binary(gt)
    else:
        skel_gt = _as_bool_2d(skel_gt)
        if skel_gt.shape != gt.shape:
            raise ValueError(f"skel_gt/gt shape mismatch: {skel_gt.shape} vs {gt.shape}")

    skel_count = int(skel_gt.sum())
    if skel_count == 0:
        pfm = 1.0 if int(pred.sum()) == 0 else 0.0
        return float(pfm), float("nan")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    precision = _safe_div(tp, tp + fp)
    pre_recall = _safe_div(np.logical_and(pred, skel_gt).sum(), skel_count)
    pfm = _safe_div(2.0 * precision * pre_recall, precision + pre_recall)
    return float(pfm), float(pfm)


def weighted_pseudo_fmeasure_from_weights(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    recall_weights: np.ndarray,
    recall_weights_sum: float,
    precision_weights: np.ndarray,
) -> float:
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    recall_weights = np.asarray(recall_weights, dtype=np.float64)
    precision_weights = np.asarray(precision_weights, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if recall_weights.shape != gt.shape:
        raise ValueError(f"recall_weights/gt shape mismatch: {recall_weights.shape} vs {gt.shape}")
    if precision_weights.shape != gt.shape:
        raise ValueError(f"precision_weights/gt shape mismatch: {precision_weights.shape} vs {gt.shape}")

    gw_sum = float(recall_weights_sum)
    if gw_sum <= 0.0:
        raise ValueError("invalid weighted pseudo-recall map: sum is non-positive")
    rps = float((pred.astype(np.float64) * recall_weights).sum() / gw_sum)

    bw = pred.astype(np.float64) * precision_weights
    bw_sum = float(bw.sum())
    if bw_sum <= 0.0:
        if int(pred.sum()) == 0:
            return 0.0
        raise ValueError(
            "invalid weighted pseudo-precision denominator: predicted weighted foreground sum is non-positive"
        )
    pps = float((gt.astype(np.float64) * bw).sum() / bw_sum)

    return _safe_div(2.0 * rps * pps, rps + pps)


def weighted_pseudo_fmeasure(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    connectivity: int = 2,
    skel_gt: Optional[np.ndarray] = None,
) -> float:
    """Weighted pseudo-FMeasure (Fps) from weighted pseudo-recall/precision.

    Uses the component-wise GW/PW weighting scheme described by
    Ntirogiannis et al. (TIP 2013). Empty predictions return 0.0 for weighted
    pseudo-FMeasure; invalid contour/skeleton extraction still fail-fast.
    If `skel_gt` is provided, it is reused instead of recomputing GT skeletons.
    """
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")

    if not gt.any():
        return 1.0 if not pred.any() else 0.0

    gw, pw = _pseudo_weight_maps(gt, connectivity=connectivity, skel_gt=skel_gt)
    return float(
        weighted_pseudo_fmeasure_from_weights(
            pred,
            gt,
            recall_weights=gw,
            recall_weights_sum=float(gw.sum()),
            precision_weights=pw,
        )
    )


def _label_components(mask: np.ndarray, *, connectivity: int = 2) -> Tuple[np.ndarray, int]:
    import cv2

    mask = _as_bool_2d(mask)
    cc_conn = _cc_connectivity_cv2(connectivity)
    n_all, lab = cv2.connectedComponents(mask.astype(np.uint8, copy=False), connectivity=cc_conn)
    n = int(max(0, int(n_all) - 1))
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
    gt_lab: Optional[np.ndarray] = None,
    pred_lab: Optional[np.ndarray] = None,
    pfm_weight_recall: Optional[np.ndarray] = None,
    pfm_weight_recall_sum: Optional[float] = None,
    pfm_weight_precision: Optional[np.ndarray] = None,
    enable_skeleton_metrics: bool = True,
    skeleton_method: str = "guo_hall",
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    pred_bin = _as_bool_2d(pred_bin)
    gt_bin = _as_bool_2d(gt_bin)
    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"pred_bin/gt_bin shape mismatch: {pred_bin.shape} vs {gt_bin.shape}")
    enable_skeleton_metrics = bool(enable_skeleton_metrics)
    if (not enable_skeleton_metrics) and (skel_gt is not None):
        raise ValueError("skel_gt must be None when enable_skeleton_metrics is False")

    t0 = time.perf_counter()
    c = confusion_counts(pred_bin, gt_bin)
    if timings is not None:
        timings["confusion_counts"] = timings.get("confusion_counts", 0.0) + (time.perf_counter() - t0)
    t0 = time.perf_counter()
    if gt_lab is None:
        gt_lab_i, _ = _label_components(gt_bin, connectivity=connectivity)
    else:
        gt_lab_i = _as_int_labels_2d(gt_lab)
        if gt_lab_i.shape != gt_bin.shape:
            raise ValueError(f"gt_lab/gt_bin shape mismatch: {gt_lab_i.shape} vs {gt_bin.shape}")
    if pred_lab is None:
        pred_lab_i, _ = _label_components(pred_bin, connectivity=connectivity)
    else:
        pred_lab_i = _as_int_labels_2d(pred_lab)
        if pred_lab_i.shape != pred_bin.shape:
            raise ValueError(f"pred_lab/pred_bin shape mismatch: {pred_lab_i.shape} vs {pred_bin.shape}")
    if timings is not None:
        timings["label_components"] = timings.get("label_components", 0.0) + (time.perf_counter() - t0)
    t0 = time.perf_counter()
    voi_split, voi_merge = voi_split_merge_from_labels(gt_lab_i, pred_lab_i)
    if timings is not None:
        timings["voi"] = timings.get("voi", 0.0) + (time.perf_counter() - t0)
    voi_total = float(voi_split + voi_merge)
    if gt_beta0 is None or gt_beta1 is None:
        t0 = time.perf_counter()
        gt_beta0_i, gt_beta1_i = betti_numbers_2d(gt_bin, connectivity=connectivity)
        if timings is not None:
            timings["betti_gt"] = timings.get("betti_gt", 0.0) + (time.perf_counter() - t0)
    else:
        gt_beta0_i = int(gt_beta0)
        gt_beta1_i = int(gt_beta1)
    t0 = time.perf_counter()
    pred_beta0_i, pred_beta1_i = betti_numbers_2d(pred_bin, connectivity=connectivity)
    if timings is not None:
        timings["betti_pred"] = timings.get("betti_pred", 0.0) + (time.perf_counter() - t0)
    abs_beta0_err = float(abs(pred_beta0_i - gt_beta0_i))
    abs_beta1_err = float(abs(pred_beta1_i - gt_beta1_i))
    betti_l1 = float(abs_beta0_err + abs_beta1_err)
    euler_pred = float(pred_beta0_i - pred_beta1_i)
    euler_gt = float(gt_beta0_i - gt_beta1_i)
    abs_euler_err = float(abs(euler_pred - euler_gt))

    t0 = time.perf_counter()
    betti_match = betti_matching_error(pred_bin, gt_bin)
    if timings is not None:
        timings["betti_matching"] = timings.get("betti_matching", 0.0) + (time.perf_counter() - t0)

    t0 = time.perf_counter()
    hd = hausdorff_metrics(pred_bin, gt_bin, boundary_k=boundary_k)
    if timings is not None:
        timings["boundary_hausdorff"] = timings.get("boundary_hausdorff", 0.0) + (time.perf_counter() - t0)

    pfm: Optional[float] = None
    pfm_nonempty: Optional[float] = None
    skel_recall_val: Optional[float] = None
    if enable_skeleton_metrics:
        if skel_gt is None:
            t0 = time.perf_counter()
            skel_gt = skeletonize_binary(gt_bin, skeleton_method=skeleton_method)
            if timings is not None:
                timings["skeletonize_gt"] = timings.get("skeletonize_gt", 0.0) + (time.perf_counter() - t0)
        else:
            skel_gt = _as_bool_2d(skel_gt)
            if skel_gt.shape != gt_bin.shape:
                raise ValueError(f"skel_gt/gt_bin shape mismatch: {skel_gt.shape} vs {gt_bin.shape}")
        t0 = time.perf_counter()
        pfm, pfm_nonempty = pseudo_fmeasure_values(pred_bin, gt_bin, skel_gt=skel_gt)
        if timings is not None:
            timings["pfm"] = timings.get("pfm", 0.0) + (time.perf_counter() - t0)
    pfm_weighted: Optional[float] = None
    if enable_skeleton_metrics:
        if pfm_weight_recall is None and pfm_weight_recall_sum is None and pfm_weight_precision is None:
            t0 = time.perf_counter()
            pfm_weighted = weighted_pseudo_fmeasure(
                pred_bin,
                gt_bin,
                connectivity=connectivity,
                skel_gt=skel_gt,
            )
            if timings is not None:
                timings["pfm_weighted"] = timings.get("pfm_weighted", 0.0) + (time.perf_counter() - t0)
        elif pfm_weight_recall is not None and pfm_weight_recall_sum is not None and pfm_weight_precision is not None:
            t0 = time.perf_counter()
            pfm_weighted = weighted_pseudo_fmeasure_from_weights(
                pred_bin,
                gt_bin,
                recall_weights=pfm_weight_recall,
                recall_weights_sum=float(pfm_weight_recall_sum),
                precision_weights=pfm_weight_precision,
            )
            if timings is not None:
                timings["pfm_weighted"] = timings.get("pfm_weighted", 0.0) + (time.perf_counter() - t0)
        else:
            raise ValueError(
                "pfm weighted inputs must be provided together: "
                "pfm_weight_recall, pfm_weight_recall_sum, pfm_weight_precision"
            )
    elif pfm_weight_recall is not None or pfm_weight_recall_sum is not None or pfm_weight_precision is not None:
        raise ValueError(
            "pfm weighted inputs must be None when enable_skeleton_metrics is False: "
            "pfm_weight_recall, pfm_weight_recall_sum, pfm_weight_precision"
        )

    t0 = time.perf_counter()
    mpm_val = float(mpm(pred_bin, gt_bin, boundary_k=boundary_k))
    if timings is not None:
        timings["mpm"] = timings.get("mpm", 0.0) + (time.perf_counter() - t0)
    t0 = time.perf_counter()
    drd_val = float(drd(pred_bin, gt_bin, block_size=drd_block_size))
    if timings is not None:
        timings["drd"] = timings.get("drd", 0.0) + (time.perf_counter() - t0)
    if enable_skeleton_metrics:
        if skel_gt is None:
            raise ValueError("internal error: skel_gt is required when enable_skeleton_metrics is True")
        t0 = time.perf_counter()
        skel_recall_val = float(skeleton_recall(pred_bin, gt_bin, skel_gt=skel_gt))
        if timings is not None:
            timings["skeleton_recall"] = timings.get("skeleton_recall", 0.0) + (time.perf_counter() - t0)

    out = {
        "dice": float(dice_from_confusion(c)),
        "accuracy": float(accuracy_from_confusion(c)),
        "voi": voi_total,
        "mpm": mpm_val,
        "drd": drd_val,
        "betti_beta0_pred": float(pred_beta0_i),
        "betti_beta1_pred": float(pred_beta1_i),
        "betti_beta0_gt": float(gt_beta0_i),
        "betti_beta1_gt": float(gt_beta1_i),
        "betti_abs_beta0_err": abs_beta0_err,
        "betti_abs_beta1_err": abs_beta1_err,
        "betti_l1": betti_l1,
        "betti_match_err": float(betti_match["betti_match_err"]),
        "betti_match_err_dim0": float(betti_match["betti_match_err_dim0"]),
        "betti_match_err_dim1": float(betti_match["betti_match_err_dim1"]),
        "euler_pred": euler_pred,
        "euler_gt": euler_gt,
        "abs_euler_err": abs_euler_err,
        "boundary_hd": float(hd["hd"]),
        "boundary_hd95": float(hd["hd95"]),
        "boundary_assd": float(hd["assd"]),
    }
    if enable_skeleton_metrics:
        if pfm is None or pfm_nonempty is None:
            raise ValueError("internal error: pfm metrics are required when enable_skeleton_metrics is True")
        if skel_recall_val is None:
            raise ValueError("internal error: skeleton_recall is required when enable_skeleton_metrics is True")
        if pfm_weighted is None:
            raise ValueError("internal error: pfm_weighted is required when enable_skeleton_metrics is True")
        out.update(
            {
                "skeleton_recall": float(skel_recall_val),
                "pfm": float(pfm),
                "pfm_nonempty": float(pfm_nonempty),
                "pfm_weighted": float(pfm_weighted),
            }
        )
    return out
