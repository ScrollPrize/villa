from __future__ import annotations

import hashlib
import os
import os.path as osp
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .stitched_components import (
    _build_gt_component_templates,
    component_metrics_by_gt_bbox,
    summarize_component_rows,
    write_global_component_manifest,
)
from .stitched_metric_specs import component_metric_specs
from .stitched_primitives import (
    _as_bool_2d,
    _count_holes_2d,
    _label_components,
    _local_metrics_from_binary,
    _pseudo_weight_maps,
    boundary_precision_recall_f1,
    nsd_surface_dice,
    skeletonize_binary,
    soft_dice_from_prob,
)


__all__ = [
    "component_metric_specs",
    "compute_stitched_metrics",
    "summarize_component_rows",
    "write_global_component_manifest",
]


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
        raise ValueError("eval_mask is empty after combining prediction coverage with validity mask")

    ys, xs = np.where(eval_mask)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    pred_c = np.asarray(pred)[y0:y1, x0:x1]
    gt_c = np.asarray(gt)[y0:y1, x0:x1]
    mask_c = eval_mask[y0:y1, x0:x1]
    return pred_c, gt_c, mask_c, (y0, y1, x0, x1)


def _bool_mask_digest(mask: np.ndarray) -> str:
    mask = _as_bool_2d(mask)
    packed = np.packbits(mask, axis=None)
    return hashlib.blake2b(packed.tobytes(), digest_size=16).hexdigest()


def _dataset_root_from_cfg() -> str:
    from train_resnet3d_lib.config import CFG

    if not hasattr(CFG, "dataset_root"):
        raise AttributeError("CFG.dataset_root must be set for stitched metrics evaluation")
    dataset_root = str(CFG.dataset_root).strip()
    if not dataset_root:
        raise ValueError("CFG.dataset_root must be a non-empty path")
    return dataset_root


def _load_ground_truth_masks(
    *,
    dataset_root: str,
    segment_id: str,
    label_suffix: str,
    mask_suffix: str,
    downsample: int,
    pred_shape: Tuple[int, int],
    roi_offset: Optional[Tuple[int, int]],
    timings: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    label_base = osp.join(dataset_root, segment_id, f"{segment_id}_inklabels{label_suffix}")
    mask_base = osp.join(dataset_root, segment_id, f"{segment_id}_mask{mask_suffix}")
    gt_gray = _read_gray_any(label_base)
    valid_gray = _read_gray_any(mask_base)
    if timings is not None:
        timings["read_inputs"] = timings.get("read_inputs", 0.0) + (time.perf_counter() - t0)

    t0 = time.perf_counter()
    gt_bin_full = (gt_gray.astype(np.float32) / 255.0) >= 0.5
    valid_full = valid_gray.astype(np.uint8) > 0
    if timings is not None:
        timings["binarize"] = timings.get("binarize", 0.0) + (time.perf_counter() - t0)

    t0 = time.perf_counter()
    if roi_offset is not None and (int(roi_offset[0]) != 0 or int(roi_offset[1]) != 0):
        gt_bin = _downsample_bool_any_roi(gt_bin_full, ds=downsample, out_hw=pred_shape, offset=roi_offset)
        valid = _downsample_bool_any_roi(valid_full, ds=downsample, out_hw=pred_shape, offset=roi_offset)
    else:
        gt_bin = _downsample_bool_any(gt_bin_full, ds=downsample, out_hw=pred_shape)
        valid = _downsample_bool_any(valid_full, ds=downsample, out_hw=pred_shape)
    if timings is not None:
        timings["downsample"] = timings.get("downsample", 0.0) + (time.perf_counter() - t0)
    return gt_bin, valid


def _prepare_eval_crop(
    *,
    pred_prob: np.ndarray,
    pred_has: np.ndarray,
    gt_bin: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    eval_mask = np.logical_and(pred_has, valid)
    pred_prob_c, gt_bin_c, eval_mask_c, crop = _crop_to_mask_bbox(pred_prob, gt_bin, eval_mask)
    if eval_mask_c.shape != pred_prob_c.shape:
        raise ValueError("internal error: cropped shapes mismatch")

    pred_prob_c = pred_prob_c.copy()
    gt_bin_c = gt_bin_c.copy()
    pred_prob_c[~eval_mask_c] = 0.0
    gt_bin_c[~eval_mask_c] = False
    return pred_prob_c, gt_bin_c, eval_mask_c, crop


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


def _stability_stats(values: List[float]) -> Dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "std": float(vals.std()) if vals.size else float("nan"),
        "min": float(vals.min()) if vals.size else float("nan"),
        "max": float(vals.max()) if vals.size else float("nan"),
    }


def _normalize_component_min_area(component_min_area: Any) -> int:
    value = component_min_area
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "0"}:
            return 0
        value = int(text)
    area = int(value)
    if area <= 0:
        return 0
    return area


def _remove_small_components_by_area(
    mask: np.ndarray,
    *,
    min_area: int,
    connectivity: int,
    return_labels: bool = False,
):
    import cv2

    mask_bool = _as_bool_2d(mask)
    area_i = _normalize_component_min_area(min_area)
    if area_i <= 0:
        if return_labels:
            lab, n = _label_components(mask_bool, connectivity=connectivity)
            return mask_bool, lab, int(n)
        return mask_bool

    cc_conn = 4 if int(connectivity) == 1 else 8
    n_all, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8, copy=False),
        connectivity=cc_conn,
    )
    if int(n_all) <= 1:
        if return_labels:
            pred_lab = np.zeros_like(labels)
            return mask_bool, pred_lab, 0
        return mask_bool
    sizes = stats[1:, cv2.CC_STAT_AREA].astype(np.int64, copy=False)
    keep = sizes >= int(area_i)
    keep_table = np.zeros(int(n_all), dtype=bool)
    keep_table[1:] = keep
    mask_filtered = keep_table[labels]
    if not return_labels:
        return mask_filtered
    kept_labels = np.flatnonzero(keep) + 1
    n_pred = int(kept_labels.size)
    if n_pred == 0:
        pred_lab = np.zeros_like(labels)
        return mask_filtered, pred_lab, 0
    remap = np.zeros(int(n_all), dtype=labels.dtype)
    remap[kept_labels] = np.arange(1, n_pred + 1, dtype=labels.dtype)
    pred_lab = remap[labels]
    return mask_filtered, pred_lab, n_pred


def _postprocess_prediction(
    pred_prob: np.ndarray,
    *,
    eval_mask: np.ndarray,
    threshold: float,
    connectivity: int,
    component_min_area: int,
    return_threshold_bin: bool = False,
):
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    eval_mask = _as_bool_2d(eval_mask)
    if pred_prob.shape != eval_mask.shape:
        raise ValueError(f"pred_prob/eval_mask shape mismatch: {pred_prob.shape} vs {eval_mask.shape}")

    pred_bin_threshold = np.logical_and(pred_prob >= float(threshold), eval_mask)
    pred_bin = pred_bin_threshold
    pred_bin, pred_lab, n_pred = _remove_small_components_by_area(
        pred_bin,
        min_area=int(component_min_area),
        connectivity=connectivity,
        return_labels=True,
    )

    pred_prob_clean = pred_prob.copy()
    pred_prob_clean[~pred_bin] = 0.0
    if return_threshold_bin:
        return pred_bin, pred_prob_clean, pred_lab, int(n_pred), pred_bin_threshold
    return pred_bin, pred_prob_clean, pred_lab, int(n_pred)


def _threshold_tag(threshold: float) -> str:
    thr_str = f"{float(threshold):.6f}".rstrip("0").rstrip(".")
    if not thr_str:
        thr_str = "0"
    return thr_str.replace(".", "p")


def _threshold_metric_values_from_component_stats(
    *,
    component_metric_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for metric_name, stats in component_metric_stats.items():
        for stat_name, stat_value in stats.items():
            key = metric_name if stat_name == "mean" else f"{metric_name}_{stat_name}"
            values[key] = float(stat_value)
    return values


def _compute_full_region_metrics(
    *,
    pred_bin_clean: np.ndarray,
    pred_prob_clean: np.ndarray,
    gt_bin: np.ndarray,
    betti_connectivity: int,
    drd_block_size: int,
    boundary_k: int,
    boundary_tols: Optional[np.ndarray],
    gt_beta0: int,
    gt_beta1: int,
    gt_lab: np.ndarray,
    pred_lab: np.ndarray,
    skel_gt: Optional[np.ndarray],
    pfm_weight_recall: Optional[np.ndarray],
    pfm_weight_precision: Optional[np.ndarray],
    enable_skeleton_metrics: bool = True,
    skeleton_method: str = "guo_hall",
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    enable_skeleton_metrics = bool(enable_skeleton_metrics)
    if enable_skeleton_metrics and skel_gt is None:
        raise ValueError("skel_gt is required when enable_skeleton_metrics is True")
    if (pfm_weight_recall is None) != (pfm_weight_precision is None):
        pfm_weight_recall = None
        pfm_weight_precision = None
    pfm_weight_recall_sum_i: Optional[float] = None
    if pfm_weight_recall is not None and pfm_weight_precision is not None:
        pfm_weight_recall_sum_i = float(np.asarray(pfm_weight_recall, dtype=np.float64).sum())
        if pfm_weight_recall_sum_i <= 0.0:
            pfm_weight_recall = None
            pfm_weight_precision = None
            pfm_weight_recall_sum_i = None
    full_metrics = _local_metrics_from_binary(
        pred_bin_clean,
        gt_bin,
        connectivity=betti_connectivity,
        drd_block_size=drd_block_size,
        boundary_k=boundary_k,
        gt_beta0=int(gt_beta0),
        gt_beta1=int(gt_beta1),
        gt_lab=gt_lab,
        pred_lab=pred_lab,
        skel_gt=skel_gt,
        pfm_weight_recall=pfm_weight_recall,
        pfm_weight_recall_sum=pfm_weight_recall_sum_i,
        pfm_weight_precision=pfm_weight_precision,
        enable_skeleton_metrics=enable_skeleton_metrics,
        skeleton_method=skeleton_method,
        timings=timings,
    )
    out: Dict[str, float] = {
        "dice_hard": float(full_metrics["dice"]),
        "dice_soft": float(soft_dice_from_prob(pred_prob_clean, gt_bin)),
        "accuracy": float(full_metrics["accuracy"]),
        "voi": float(full_metrics["voi"]),
        "mpm": float(full_metrics["mpm"]),
        "drd": float(full_metrics["drd"]),
        "betti/beta0_pred": float(full_metrics["betti_beta0_pred"]),
        "betti/beta1_pred": float(full_metrics["betti_beta1_pred"]),
        "betti/beta0_gt": float(full_metrics["betti_beta0_gt"]),
        "betti/beta1_gt": float(full_metrics["betti_beta1_gt"]),
        "betti/abs_beta0_err": float(full_metrics["betti_abs_beta0_err"]),
        "betti/abs_beta1_err": float(full_metrics["betti_abs_beta1_err"]),
        "betti/l1_betti_err": float(full_metrics["betti_l1"]),
        "betti/match_err": float(full_metrics["betti_match_err"]),
        "betti/match_err_dim0": float(full_metrics["betti_match_err_dim0"]),
        "betti/match_err_dim1": float(full_metrics["betti_match_err_dim1"]),
        "euler_pred": float(full_metrics["euler_pred"]),
        "euler_gt": float(full_metrics["euler_gt"]),
        "abs_euler_err": float(full_metrics["abs_euler_err"]),
        "boundary/hd": float(full_metrics["boundary_hd"]),
        "boundary/hd95": float(full_metrics["boundary_hd95"]),
        "boundary/assd": float(full_metrics["boundary_assd"]),
        "pfm_weighted": float(full_metrics["pfm_weighted"]),
    }
    if enable_skeleton_metrics:
        out.update(
            {
                "skeleton/recall": float(full_metrics["skeleton_recall"]),
                "pfm": float(full_metrics["pfm"]),
                "pfm_nonempty": float(full_metrics["pfm_nonempty"]),
            }
        )

    tau_values = np.asarray([1.0], dtype=np.float32) if boundary_tols is None else np.asarray(boundary_tols)
    for tau in tau_values:
        tau_f = float(tau)
        tau_key = str(tau_f).replace(".", "p")
        bf = boundary_precision_recall_f1(pred_bin_clean, gt_bin, tau=tau_f, boundary_k=boundary_k)
        out[f"boundary/bf1_tau{tau_key}"] = float(bf["b_f1"])
        out[f"boundary/nsd_tau{tau_key}"] = float(nsd_surface_dice(pred_bin_clean, gt_bin, tau=tau_f, boundary_k=boundary_k))
    return out


def _build_components_manifest(
    *,
    component_rows: List[Dict[str, Any]],
    full_off_y: int,
    full_off_x: int,
    enable_skeleton_metrics: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    components_manifest: List[Dict[str, Any]] = []
    rows_by_id: Dict[int, Dict[str, Any]] = {}
    for row in component_rows:
        bbox = [int(v) for v in row["bbox"]]
        bbox_full = [
            int(full_off_y + bbox[0]),
            int(full_off_y + bbox[1]),
            int(full_off_x + bbox[2]),
            int(full_off_x + bbox[3]),
        ]
        entry = {
            "component_id": int(row["component_id"]),
            "bbox_crop": bbox,
            "bbox_full": bbox_full,
            "dice_hard": float(row["dice_hard"]),
            "dice_soft": float(row["dice_soft"]),
            "accuracy": float(row["accuracy"]),
            "mpm": float(row["mpm"]),
            "drd": float(row["drd"]),
            "pfm_weighted": float(row["pfm_weighted"]),
            "voi": float(row["voi"]),
            "betti_l1": float(row["betti_l1"]),
            "abs_euler_err": float(row["abs_euler_err"]),
            "boundary_hd95": float(row["boundary_hd95"]),
            "betti_abs_beta0_err": float(row["betti_abs_beta0_err"]),
            "betti_abs_beta1_err": float(row["betti_abs_beta1_err"]),
            "betti_beta0_pred": float(row["betti_beta0_pred"]),
            "betti_beta1_pred": float(row["betti_beta1_pred"]),
            "betti_beta0_gt": float(row["betti_beta0_gt"]),
            "betti_beta1_gt": float(row["betti_beta1_gt"]),
            "betti_match_err": float(row["betti_match_err"]),
            "betti_match_err_dim0": float(row["betti_match_err_dim0"]),
            "betti_match_err_dim1": float(row["betti_match_err_dim1"]),
        }
        if enable_skeleton_metrics:
            entry.update(
                {
                    "pfm": float(row["pfm"]),
                    "pfm_nonempty": float(row["pfm_nonempty"]),
                    "skeleton_recall": float(row["skeleton_recall"]),
                }
            )
        components_manifest.append(entry)
        rows_by_id[entry["component_id"]] = row
    return components_manifest, rows_by_id


def _write_component_outputs(
    *,
    component_output_dir: str,
    fragment_id: str,
    downsample: int,
    threshold: float,
    gt_lab: np.ndarray,
    pred_lab: np.ndarray,
    pred_full_shape: Tuple[int, int],
    roi_offset: Tuple[int, int],
    crop_offset: Tuple[int, int],
    full_offset: Tuple[int, int],
    component_worst_k: Optional[int],
    component_worst_q: Optional[float],
    metric_specs: Tuple[Tuple[str, bool], ...],
    components_manifest: List[Dict[str, Any]],
    component_metric_rankings: Dict[str, Dict[str, Any]],
    save_component_debug_images: bool,
    component_debug_max_items: Optional[int],
    rows_by_id: Dict[int, Dict[str, Any]],
    pred_prob_clean: np.ndarray,
    pred_bin_clean: np.ndarray,
    gt_bin: np.ndarray,
) -> None:
    import json as _json
    from PIL import Image

    os.makedirs(component_output_dir, exist_ok=True)
    safe_seg = str(fragment_id).replace("/", "_")
    thr_tag = _threshold_tag(float(threshold))
    base_name = f"components_{safe_seg}_ds{int(downsample)}"
    threshold_base_name = f"{base_name}_thr_{thr_tag}"
    gt_path = osp.join(component_output_dir, f"{base_name}_gt_labels.png")
    pred_path = osp.join(component_output_dir, f"{threshold_base_name}_pred_labels.png")
    meta_path = osp.join(component_output_dir, f"{threshold_base_name}_meta.json")
    manifest_path = osp.join(component_output_dir, f"{threshold_base_name}_manifest.json")

    if not osp.exists(gt_path):
        gt_u16 = gt_lab.astype(np.uint16, copy=False)
        Image.fromarray(gt_u16, mode="I;16").save(gt_path)
    if not (osp.exists(pred_path) and osp.exists(meta_path)):
        pred_u16 = pred_lab.astype(np.uint16, copy=False)
        Image.fromarray(pred_u16, mode="I;16").save(pred_path)

    off_y, off_x = [int(v) for v in roi_offset]
    crop_off_y, crop_off_x = [int(v) for v in crop_offset]
    full_off_y, full_off_x = [int(v) for v in full_offset]
    meta = {
        "segment_id": str(fragment_id),
        "downsample": int(downsample),
        "threshold": float(threshold),
        "roi_offset": [off_y, off_x],
        "crop_offset": [crop_off_y, crop_off_x],
        "full_offset": [full_off_y, full_off_x],
        "full_shape": [int(pred_full_shape[0]), int(pred_full_shape[1])],
        "crop_shape": [int(gt_lab.shape[0]), int(gt_lab.shape[1])],
    }
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)

    manifest = {
        "segment_id": str(fragment_id),
        "downsample": int(downsample),
        "threshold": float(threshold),
        "roi_offset": [off_y, off_x],
        "crop_offset": [crop_off_y, crop_off_x],
        "full_offset": [full_off_y, full_off_x],
        "full_shape": [int(pred_full_shape[0]), int(pred_full_shape[1])],
        "crop_shape": [int(gt_lab.shape[0]), int(gt_lab.shape[1])],
        "worst_k": None if component_worst_k is None else int(component_worst_k),
        "worst_q": None if component_worst_q is None else float(component_worst_q),
        "components": components_manifest,
        "rankings": component_metric_rankings,
    }
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)

    if not save_component_debug_images:
        return

    selected_component_ids: List[int] = []
    for metric_name, _ in metric_specs:
        selected_component_ids.extend(component_metric_rankings[metric_name]["worst_k_component_ids"])
        selected_component_ids.extend(component_metric_rankings[metric_name]["worst_q_component_ids"])
    selected_component_ids = sorted(set(int(v) for v in selected_component_ids))
    if not selected_component_ids:
        raise ValueError("save_component_debug_images requires component_worst_k or component_worst_q to be set")
    if component_debug_max_items is not None:
        max_items = max(1, int(component_debug_max_items))
        selected_component_ids = selected_component_ids[:max_items]

    debug_dir = osp.join(component_output_dir, f"{threshold_base_name}_debug")
    os.makedirs(debug_dir, exist_ok=True)
    for component_id in selected_component_ids:
        if component_id not in rows_by_id:
            raise KeyError(f"component_id={component_id} missing from component rows")
        row = rows_by_id[component_id]
        bbox = [int(v) for v in row["bbox"]]
        by0, by1, bx0, bx1 = bbox
        crop_pred_prob = pred_prob_clean[by0:by1, bx0:bx1]
        crop_pred_bin = pred_bin_clean[by0:by1, bx0:bx1]
        crop_gt = gt_bin[by0:by1, bx0:bx1]

        cid = f"{int(component_id):05d}"
        Image.fromarray((np.clip(crop_pred_prob, 0.0, 1.0) * 255.0).astype(np.uint8)).save(
            osp.join(debug_dir, f"comp_{cid}_pred_prob.png")
        )
        Image.fromarray((crop_pred_bin.astype(np.uint8) * 255)).save(osp.join(debug_dir, f"comp_{cid}_pred_bin.png"))
        Image.fromarray((crop_gt.astype(np.uint8) * 255)).save(osp.join(debug_dir, f"comp_{cid}_gt_bin.png"))


def _save_stitched_eval_inputs(
    *,
    stitched_inputs_output_dir: str,
    fragment_id: str,
    downsample: int,
    eval_epoch: Optional[int],
    threshold: float,
    component_min_area: int,
    pred_prob: np.ndarray,
    pred_bin_threshold: np.ndarray,
    pred_bin_postprocess: np.ndarray,
    gt_bin: np.ndarray,
    eval_mask: np.ndarray,
    roi_offset: Tuple[int, int],
    crop_offset: Tuple[int, int],
    full_offset: Tuple[int, int],
    full_shape: Tuple[int, int],
    gt_skeleton: Optional[np.ndarray] = None,
    pfm_weight_recall: Optional[np.ndarray] = None,
    pfm_weight_precision: Optional[np.ndarray] = None,
) -> None:
    import json as _json
    from PIL import Image

    safe_seg = str(fragment_id).replace("/", "_")
    thr_tag = str(float(threshold)).replace(".", "p")
    crop_h, crop_w = [int(v) for v in pred_prob.shape]
    full_off_y, full_off_x = [int(v) for v in full_offset]
    crop_key = f"fulloff_y{full_off_y}_x{full_off_x}_h{crop_h}_w{crop_w}"
    segment_dir = osp.join(stitched_inputs_output_dir, safe_seg, f"ds{int(downsample)}", crop_key)
    epoch_tag = "epoch_unknown" if eval_epoch is None else f"epoch_{int(eval_epoch):04d}"
    epoch_dir = osp.join(segment_dir, epoch_tag)
    threshold_dir = osp.join(epoch_dir, "thresholds", f"thr_{thr_tag}")
    os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(epoch_dir, exist_ok=True)
    os.makedirs(threshold_dir, exist_ok=True)

    pred_prob_path = osp.join(epoch_dir, "pred_prob_eval_crop.png")
    gt_path = osp.join(segment_dir, "gt_eval_crop.png")
    eval_mask_path = osp.join(epoch_dir, "eval_mask_crop.png")
    meta_path = osp.join(segment_dir, "eval_crop_meta.json")
    epoch_meta_path = osp.join(epoch_dir, "epoch_meta.json")
    pred_bin_threshold_path = osp.join(threshold_dir, "pred_bin_threshold_eval_crop.png")
    pred_bin_postprocess_path = osp.join(threshold_dir, "pred_bin_postprocess_eval_crop.png")
    postprocess_meta_path = osp.join(threshold_dir, "postprocess_meta.json")

    pred_prob_u8 = (np.clip(pred_prob, 0.0, 1.0) * 255.0).astype(np.uint8)
    pred_bin_threshold_u8 = (pred_bin_threshold.astype(np.uint8) * 255)
    pred_bin_postprocess_u8 = (pred_bin_postprocess.astype(np.uint8) * 255)
    gt_u8 = (gt_bin.astype(np.uint8) * 255)
    eval_mask_u8 = (eval_mask.astype(np.uint8) * 255)
    if not osp.exists(pred_prob_path):
        Image.fromarray(pred_prob_u8).save(pred_prob_path)
    if not osp.exists(eval_mask_path):
        Image.fromarray(eval_mask_u8).save(eval_mask_path)
    if not osp.exists(gt_path):
        Image.fromarray(gt_u8).save(gt_path)

    if gt_skeleton is not None:
        skel_gt = _as_bool_2d(gt_skeleton)
        if skel_gt.shape != gt_bin.shape:
            raise ValueError(f"gt_skeleton/gt_bin shape mismatch: {skel_gt.shape} vs {gt_bin.shape}")
        gt_skel_path = osp.join(segment_dir, "gt_skeleton_eval_crop.png")
        if not osp.exists(gt_skel_path):
            Image.fromarray((skel_gt.astype(np.uint8) * 255)).save(gt_skel_path)

    if (pfm_weight_recall is None) != (pfm_weight_precision is None):
        raise ValueError("pfm_weight_recall and pfm_weight_precision must be provided together")
    if pfm_weight_recall is not None and pfm_weight_precision is not None:
        gw = np.asarray(pfm_weight_recall, dtype=np.float32)
        pw = np.asarray(pfm_weight_precision, dtype=np.float32)
        if gw.shape != gt_bin.shape or pw.shape != gt_bin.shape:
            raise ValueError(
                "pfm weight shape mismatch: "
                f"gw={gw.shape}, pw={pw.shape}, gt={gt_bin.shape}"
            )
        gw_vis_path = osp.join(segment_dir, "pfm_weight_recall_eval_crop.png")
        pw_vis_path = osp.join(segment_dir, "pfm_weight_precision_eval_crop.png")
        gwpw_meta_path = osp.join(segment_dir, "pfm_weights_eval_crop_meta.json")

        if (
            (not osp.exists(gw_vis_path))
            or (not osp.exists(pw_vis_path))
            or (not osp.exists(gwpw_meta_path))
        ):
            def _to_u8_heatmap(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
                arr_f = np.asarray(arr, dtype=np.float32)
                finite = np.isfinite(arr_f)
                if not finite.any():
                    return np.zeros(arr_f.shape, dtype=np.uint8), 0.0, 0.0
                vals = arr_f[finite]
                lo = float(vals.min())
                hi = float(vals.max())
                if hi <= lo:
                    return np.zeros(arr_f.shape, dtype=np.uint8), lo, hi
                norm = np.zeros(arr_f.shape, dtype=np.float32)
                norm[finite] = (arr_f[finite] - lo) / (hi - lo)
                return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8), lo, hi

            gw_u8, gw_min, gw_max = _to_u8_heatmap(gw)
            pw_u8, pw_min, pw_max = _to_u8_heatmap(pw)

            if not osp.exists(gw_vis_path):
                Image.fromarray(gw_u8).save(gw_vis_path)
            if not osp.exists(pw_vis_path):
                Image.fromarray(pw_u8).save(pw_vis_path)
            if not osp.exists(gwpw_meta_path):
                gwpw_meta = {
                    "gw_min": float(gw_min),
                    "gw_max": float(gw_max),
                    "pw_min": float(pw_min),
                    "pw_max": float(pw_max),
                    "gw_sum": float(gw.astype(np.float64).sum()),
                    "pw_sum": float(pw.astype(np.float64).sum()),
                }
                with open(gwpw_meta_path, "w") as f:
                    _json.dump(gwpw_meta, f, indent=2)

    off_y, off_x = [int(v) for v in roi_offset]
    crop_off_y, crop_off_x = [int(v) for v in crop_offset]
    if not osp.exists(meta_path):
        meta = {
            "segment_id": str(fragment_id),
            "downsample": int(downsample),
            "roi_offset": [off_y, off_x],
            "crop_offset": [crop_off_y, crop_off_x],
            "full_offset": [full_off_y, full_off_x],
            "full_shape": [int(full_shape[0]), int(full_shape[1])],
            "crop_shape": [int(pred_prob.shape[0]), int(pred_prob.shape[1])],
        }
        with open(meta_path, "w") as f:
            _json.dump(meta, f, indent=2)
    if not osp.exists(epoch_meta_path):
        epoch_meta = {
            "segment_id": str(fragment_id),
            "downsample": int(downsample),
            "epoch": None if eval_epoch is None else int(eval_epoch),
            "crop_shape": [int(pred_prob.shape[0]), int(pred_prob.shape[1])],
        }
        with open(epoch_meta_path, "w") as f:
            _json.dump(epoch_meta, f, indent=2)
    if (
        (not osp.exists(pred_bin_threshold_path))
        or (not osp.exists(pred_bin_postprocess_path))
        or (not osp.exists(postprocess_meta_path))
    ):
        Image.fromarray(pred_bin_threshold_u8).save(pred_bin_threshold_path)
        Image.fromarray(pred_bin_postprocess_u8).save(pred_bin_postprocess_path)
        postprocess_meta = {
            "threshold": float(threshold),
            "component_min_area": int(component_min_area),
        }
        with open(postprocess_meta_path, "w") as f:
            _json.dump(postprocess_meta, f, indent=2)


def _log_timing(*, segment_id: str, downsample: int, cache_hit: bool, timings: Dict[str, float]) -> None:
    from train_resnet3d_lib.config import log as _log

    if "wall" in timings:
        total = float(timings["wall"])
    else:
        total = sum(float(v) for k, v in timings.items() if "/" not in k)
        if total <= 0.0:
            total = sum(float(v) for v in timings.values())
    parts = sorted(((k, v) for k, v in timings.items() if k != "wall"), key=lambda kv: kv[1], reverse=True)
    msg = (
        f"metrics timing segment={segment_id} ds={downsample} cache={'hit' if cache_hit else 'miss'} "
        f"total={total:.3f}s "
        + ", ".join([f"{k}={v:.3f}s" for k, v in parts])
    )
    _log(msg)


def _log_empty_eval_mask(
    *,
    segment_id: str,
    downsample: int,
    pred_shape: Tuple[int, int],
    pred_has_count: int,
) -> None:
    from train_resnet3d_lib.config import log as _log

    _log(
        f"metrics skip segment={segment_id} ds={downsample} "
        f"reason=empty_eval_mask pred_shape={tuple(pred_shape)} pred_has_count={int(pred_has_count)}"
    )


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
    betti_connectivity: int = 2,
    drd_block_size: int = 8,
    boundary_k: int = 3,
    boundary_tols: Optional[np.ndarray] = None,
    component_worst_q: Optional[float] = 0.2,
    component_worst_k: Optional[int] = 2,
    component_min_area: Optional[Any] = None,
    component_pad: int = 5,
    skeleton_method: str = "guo_hall",
    enable_skeleton_metrics: bool = True,
    enable_full_region_metrics: bool = False,
    threshold_grid: Optional[np.ndarray] = None,
    component_output_dir: Optional[str] = None,
    stitched_inputs_output_dir: Optional[str] = None,
    save_component_debug_images: bool = False,
    component_debug_max_items: Optional[int] = None,
    gt_cache_max: Optional[int] = None,
    component_rows_collector: Optional[List[Dict[str, Any]]] = None,
    eval_epoch: Optional[int] = None,
) -> Dict[str, float]:
    seg = str(fragment_id)
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    pred_has = _as_bool_2d(pred_has)
    if pred_prob.shape != pred_has.shape:
        raise ValueError(f"pred_prob/pred_has shape mismatch: {pred_prob.shape} vs {pred_has.shape}")
    if gt_cache_max is not None:
        _gt_cache_set_max(gt_cache_max)
    pred_full_shape = tuple(pred_prob.shape)

    timings: Dict[str, float] = {}
    wall_t0 = time.perf_counter()

    def _timeit(name: str, t0: float) -> None:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)

    ds = max(1, int(downsample))
    pad_i = max(0, int(component_pad))
    component_min_area_i = _normalize_component_min_area(component_min_area)
    enable_skeleton_metrics = bool(enable_skeleton_metrics)
    metric_specs = component_metric_specs(enable_skeleton_metrics=enable_skeleton_metrics)
    roi_key = (0, 0)
    if roi_offset is not None:
        roi_key = (int(roi_offset[0]), int(roi_offset[1]))
    dataset_root = _dataset_root_from_cfg()
    pred_has_digest = _bool_mask_digest(pred_has)
    cache_key = (
        str(dataset_root),
        seg,
        str(label_suffix),
        str(mask_suffix),
        int(ds),
        tuple(roi_key),
        tuple(pred_prob.shape),
        str(pred_has_digest),
        int(betti_connectivity),
        bool(enable_skeleton_metrics),
        str(skeleton_method) if enable_skeleton_metrics else "__disabled__",
    )
    cache = _gt_cache_get(cache_key)
    if cache is not None:
        cache_hit = True
        cache_entry = cache
        gt_bin = cache_entry["gt_bin"]
        eval_mask = cache_entry["eval_mask"]
        gt_lab = cache_entry["gt_lab"]
        n_gt = cache_entry["n_gt"]
        gt_beta0 = int(cache_entry["gt_beta0"])
        gt_beta1 = int(cache_entry["gt_beta1"])
        skel_gt = cache_entry.get("skel_gt")
        if enable_skeleton_metrics and skel_gt is None:
            raise ValueError("cached entry missing skel_gt while enable_skeleton_metrics is True")
        y0, y1, x0, x1 = cache_entry["crop"]
        pred_full_shape = tuple(cache_entry.get("full_shape", pred_full_shape))
        pred_prob = pred_prob[y0:y1, x0:x1]
        pred_has = pred_has[y0:y1, x0:x1]
        if "gt_component_templates" not in cache_entry:
            cache_entry["gt_component_templates"] = None
        if "gt_component_pad" not in cache_entry:
            cache_entry["gt_component_pad"] = None
        if "pfm_weight_recall_full" not in cache_entry:
            cache_entry["pfm_weight_recall_full"] = None
        if "pfm_weight_precision_full" not in cache_entry:
            cache_entry["pfm_weight_precision_full"] = None
        cached_component_pad = cache_entry["gt_component_pad"]
        if cached_component_pad is not None and int(cached_component_pad) != pad_i:
            raise ValueError(
                f"component_pad changed for cached segment={seg}: "
                f"cached={int(cached_component_pad)} current={int(pad_i)}"
            )
    else:
        cache_hit = False
        t0 = time.perf_counter()
        load_gt_timings: Dict[str, float] = {}
        gt_bin_ds, valid = _load_ground_truth_masks(
            dataset_root=dataset_root,
            segment_id=seg,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
            downsample=ds,
            pred_shape=tuple(pred_prob.shape),
            roi_offset=roi_offset,
            timings=load_gt_timings,
        )
        t1 = time.perf_counter()
        eval_mask_full = np.logical_and(pred_has, valid)
        load_gt_timings["eval_mask"] = load_gt_timings.get("eval_mask", 0.0) + (time.perf_counter() - t1)
        if not eval_mask_full.any():
            _timeit("load_gt", t0)
            for key, value in load_gt_timings.items():
                timings[f"load_gt/{key}"] = timings.get(f"load_gt/{key}", 0.0) + float(value)
            _log_empty_eval_mask(
                segment_id=seg,
                downsample=ds,
                pred_shape=tuple(pred_prob.shape),
                pred_has_count=int(pred_has.sum()),
            )
            return {}
        t1 = time.perf_counter()
        pred_prob, gt_bin, eval_mask, crop = _prepare_eval_crop(
            pred_prob=pred_prob,
            pred_has=pred_has,
            gt_bin=gt_bin_ds,
            valid=valid,
        )
        load_gt_timings["prepare_eval_crop"] = load_gt_timings.get("prepare_eval_crop", 0.0) + (
            time.perf_counter() - t1
        )
        y0, y1, x0, x1 = crop
        pred_has = pred_has[y0:y1, x0:x1]

        t1 = time.perf_counter()
        gt_lab, n_gt = _label_components(gt_bin, connectivity=betti_connectivity)
        load_gt_timings["gt_label_components"] = load_gt_timings.get("gt_label_components", 0.0) + (
            time.perf_counter() - t1
        )
        gt_beta0 = int(n_gt)
        t1 = time.perf_counter()
        gt_beta1 = _count_holes_2d(gt_bin, connectivity=betti_connectivity)
        load_gt_timings["gt_holes"] = load_gt_timings.get("gt_holes", 0.0) + (time.perf_counter() - t1)
        skel_gt: Optional[np.ndarray] = None
        if enable_skeleton_metrics:
            t1 = time.perf_counter()
            skel_gt = skeletonize_binary(
                gt_bin,
                cc_labels=gt_lab,
                skeleton_method=skeleton_method,
            )
            load_gt_timings["gt_skeleton"] = load_gt_timings.get("gt_skeleton", 0.0) + (time.perf_counter() - t1)
        cache_entry = {
            "gt_bin": gt_bin,
            "eval_mask": eval_mask,
            "gt_lab": gt_lab,
            "n_gt": n_gt,
            "gt_beta0": int(gt_beta0),
            "gt_beta1": int(gt_beta1),
            "skel_gt": skel_gt,
            "crop": (int(y0), int(y1), int(x0), int(x1)),
            "full_shape": pred_full_shape,
            "gt_component_templates": None,
            "gt_component_pad": None,
            "pfm_weight_recall_full": None,
            "pfm_weight_precision_full": None,
        }
        t1 = time.perf_counter()
        _gt_cache_put(cache_key, cache_entry)
        load_gt_timings["cache_put"] = load_gt_timings.get("cache_put", 0.0) + (time.perf_counter() - t1)
        _timeit("load_gt", t0)
        for key, value in load_gt_timings.items():
            timings[f"load_gt/{key}"] = timings.get(f"load_gt/{key}", 0.0) + float(value)

    if eval_mask.shape != pred_prob.shape:
        raise ValueError("internal error: cropped shapes mismatch")
    if not eval_mask.any():
        _log_empty_eval_mask(
            segment_id=seg,
            downsample=ds,
            pred_shape=tuple(pred_prob.shape),
            pred_has_count=int(pred_has.sum()),
        )
        return {}
    pred_prob = pred_prob.copy()
    pred_prob[~eval_mask] = 0.0

    out: Dict[str, float] = {}
    gt_component_templates = cache_entry["gt_component_templates"]
    if gt_component_templates is None:
        t0 = time.perf_counter()
        gt_component_templates = _build_gt_component_templates(
            gt_bin,
            gt_lab,
            int(n_gt),
            connectivity=betti_connectivity,
            pad=pad_i,
            enable_skeleton_metrics=enable_skeleton_metrics,
            skeleton_method=skeleton_method,
        )
        cache_entry["gt_component_templates"] = gt_component_templates
        cache_entry["gt_component_pad"] = int(pad_i)
        _timeit("build_gt_component_templates", t0)

    off_y, off_x = [int(v) for v in roi_key]
    crop_off_y, crop_off_x = int(y0), int(x0)
    full_off_y = int(off_y + crop_off_y)
    full_off_x = int(off_x + crop_off_x)

    tgrid = np.asarray([float(threshold)], dtype=np.float32)
    if threshold_grid is not None and len(threshold_grid):
        tgrid = np.asarray(threshold_grid, dtype=np.float32)
    if tgrid.ndim != 1:
        raise ValueError(f"threshold grid must be 1D, got shape {tgrid.shape}")
    if tgrid.size == 0:
        raise ValueError("threshold grid must contain at least one threshold")
    if not np.isfinite(tgrid).all():
        raise ValueError("threshold grid contains non-finite threshold values")

    threshold_metric_series: Dict[str, List[float]] = {}
    threshold_grid_timings: Dict[str, float] = {}
    want_component_outputs = component_output_dir is not None
    want_component_rows = component_rows_collector is not None
    need_full_gt_pfm_maps = bool(enable_full_region_metrics or stitched_inputs_output_dir)
    pfm_weight_recall_full = cache_entry.get("pfm_weight_recall_full")
    pfm_weight_precision_full = cache_entry.get("pfm_weight_precision_full")
    if need_full_gt_pfm_maps:
        if (pfm_weight_recall_full is None) != (pfm_weight_precision_full is None):
            pfm_weight_recall_full = None
            pfm_weight_precision_full = None
            cache_entry["pfm_weight_recall_full"] = None
            cache_entry["pfm_weight_precision_full"] = None
        if pfm_weight_recall_full is None and pfm_weight_precision_full is None and bool(gt_bin.any()):
            t_build = time.perf_counter()
            gw_full, pw_full = _pseudo_weight_maps(gt_bin, connectivity=betti_connectivity)
            gw_full_sum = float(gw_full.astype(np.float64).sum())
            if gw_full_sum > 0.0:
                pfm_weight_recall_full = gw_full.astype(np.float32, copy=False)
                pfm_weight_precision_full = pw_full.astype(np.float32, copy=False)
            else:
                pfm_weight_recall_full = None
                pfm_weight_precision_full = None
            cache_entry["pfm_weight_recall_full"] = pfm_weight_recall_full
            cache_entry["pfm_weight_precision_full"] = pfm_weight_precision_full
            threshold_grid_timings["full_region_pfm_weight_cache_build"] = threshold_grid_timings.get(
                "full_region_pfm_weight_cache_build", 0.0
            ) + (time.perf_counter() - t_build)

    t0 = time.perf_counter()
    for thr in tgrid.tolist():
        thr_f = float(thr)
        thr_tag = _threshold_tag(thr_f)

        p0 = time.perf_counter()
        pred_bin_t: np.ndarray
        pred_prob_t: np.ndarray
        pred_lab_t: np.ndarray
        pred_bin_threshold_t: Optional[np.ndarray] = None
        if stitched_inputs_output_dir:
            pred_bin_t, pred_prob_t, pred_lab_t, _, pred_bin_threshold_t = _postprocess_prediction(
                pred_prob,
                eval_mask=eval_mask,
                threshold=thr_f,
                connectivity=betti_connectivity,
                component_min_area=component_min_area_i,
                return_threshold_bin=True,
            )
        else:
            pred_bin_t, pred_prob_t, pred_lab_t, _ = _postprocess_prediction(
                pred_prob,
                eval_mask=eval_mask,
                threshold=thr_f,
                connectivity=betti_connectivity,
                component_min_area=component_min_area_i,
            )
        threshold_grid_timings["postprocess"] = threshold_grid_timings.get("postprocess", 0.0) + (
            time.perf_counter() - p0
        )

        if stitched_inputs_output_dir:
            if pred_bin_threshold_t is None:
                raise ValueError("internal error: threshold mask is required for stitched input outputs")
            p0 = time.perf_counter()
            _save_stitched_eval_inputs(
                stitched_inputs_output_dir=stitched_inputs_output_dir,
                fragment_id=fragment_id,
                downsample=ds,
                eval_epoch=eval_epoch,
                threshold=thr_f,
                component_min_area=component_min_area_i,
                pred_prob=pred_prob,
                pred_bin_threshold=pred_bin_threshold_t,
                pred_bin_postprocess=pred_bin_t,
                gt_bin=gt_bin,
                eval_mask=eval_mask,
                roi_offset=(off_y, off_x),
                crop_offset=(crop_off_y, crop_off_x),
                full_offset=(full_off_y, full_off_x),
                full_shape=pred_full_shape,
                gt_skeleton=skel_gt if enable_skeleton_metrics else None,
                pfm_weight_recall=pfm_weight_recall_full,
                pfm_weight_precision=pfm_weight_precision_full,
            )
            threshold_grid_timings["save_inputs"] = threshold_grid_timings.get("save_inputs", 0.0) + (
                time.perf_counter() - p0
            )

        p0 = time.perf_counter()
        component_metric_timings_t: Dict[str, float] = {}
        component_rows_t, component_metric_stats_t, component_metric_rankings_t = component_metrics_by_gt_bbox(
            pred_prob_t,
            pred_bin_t,
            gt_bin,
            connectivity=betti_connectivity,
            drd_block_size=drd_block_size,
            boundary_k=boundary_k,
            pad=pad_i,
            worst_q=component_worst_q,
            worst_k=component_worst_k,
            enable_skeleton_metrics=enable_skeleton_metrics,
            gt_lab=gt_lab,
            pred_lab=pred_lab_t,
            n_gt=n_gt,
            gt_component_templates=gt_component_templates,
            skeleton_method=skeleton_method,
            timings=component_metric_timings_t,
        )
        threshold_grid_timings["component_metrics"] = threshold_grid_timings.get("component_metrics", 0.0) + (
            time.perf_counter() - p0
        )
        for key, value in component_metric_timings_t.items():
            agg_key = f"component_metrics/{key}"
            threshold_grid_timings[agg_key] = threshold_grid_timings.get(agg_key, 0.0) + float(value)

        threshold_metric_values = _threshold_metric_values_from_component_stats(
            component_metric_stats=component_metric_stats_t,
        )
        for metric_base, metric_value in threshold_metric_values.items():
            metric_key = f"thresholds/{metric_base}/thr_{thr_tag}"
            if metric_key in out:
                raise ValueError(f"duplicate threshold metric key: {metric_key}")
            out[metric_key] = float(metric_value)
            threshold_metric_series.setdefault(metric_base, []).append(float(metric_value))

        if enable_full_region_metrics:
            p0 = time.perf_counter()
            full_region_timings_t: Dict[str, float] = {}
            full_region_out_t = _compute_full_region_metrics(
                pred_bin_clean=pred_bin_t,
                pred_prob_clean=pred_prob_t,
                gt_bin=gt_bin,
                betti_connectivity=betti_connectivity,
                drd_block_size=drd_block_size,
                boundary_k=boundary_k,
                boundary_tols=boundary_tols,
                gt_beta0=int(gt_beta0),
                gt_beta1=int(gt_beta1),
                gt_lab=gt_lab,
                pred_lab=pred_lab_t,
                skel_gt=skel_gt,
                pfm_weight_recall=pfm_weight_recall_full,
                pfm_weight_precision=pfm_weight_precision_full,
                enable_skeleton_metrics=enable_skeleton_metrics,
                skeleton_method=skeleton_method,
                timings=full_region_timings_t,
            )
            threshold_grid_timings["full_region_metrics"] = threshold_grid_timings.get("full_region_metrics", 0.0) + (
                time.perf_counter() - p0
            )
            for key, value in full_region_timings_t.items():
                agg_key = f"full_region_metrics/{key}"
                threshold_grid_timings[agg_key] = threshold_grid_timings.get(agg_key, 0.0) + float(value)
            for metric_name, metric_value in full_region_out_t.items():
                metric_base = f"full_region/{str(metric_name)}"
                metric_key = f"thresholds/{metric_base}/thr_{thr_tag}"
                if metric_key in out:
                    raise ValueError(f"duplicate threshold metric key: {metric_key}")
                out[metric_key] = float(metric_value)
                threshold_metric_series.setdefault(metric_base, []).append(float(metric_value))

        components_manifest_t: Optional[List[Dict[str, Any]]] = None
        rows_by_id_t: Optional[Dict[int, Dict[str, Any]]] = None
        if want_component_outputs or want_component_rows:
            components_manifest_t, rows_by_id_t = _build_components_manifest(
                component_rows=component_rows_t,
                full_off_y=full_off_y,
                full_off_x=full_off_x,
                enable_skeleton_metrics=enable_skeleton_metrics,
            )

        if want_component_rows:
            if components_manifest_t is None:
                raise ValueError("internal error: component manifest was not built")
            for entry in components_manifest_t:
                global_entry = dict(entry)
                global_entry["segment_id"] = str(fragment_id)
                global_entry["downsample"] = int(ds)
                global_entry["threshold"] = float(thr_f)
                global_entry["threshold_tag"] = str(thr_tag)
                global_entry["global_component_id"] = (
                    f"{global_entry['segment_id']}:{global_entry['threshold_tag']}:{int(global_entry['component_id'])}"
                )
                component_rows_collector.append(global_entry)

        if want_component_outputs:
            if components_manifest_t is None:
                raise ValueError("internal error: component manifest is required for component outputs")
            if rows_by_id_t is None:
                raise ValueError("internal error: component rows lookup is required for component outputs")
            p0 = time.perf_counter()
            _write_component_outputs(
                component_output_dir=component_output_dir,
                fragment_id=fragment_id,
                downsample=ds,
                threshold=thr_f,
                gt_lab=gt_lab,
                pred_lab=pred_lab_t,
                pred_full_shape=pred_full_shape,
                roi_offset=(off_y, off_x),
                crop_offset=(crop_off_y, crop_off_x),
                full_offset=(full_off_y, full_off_x),
                component_worst_k=component_worst_k,
                component_worst_q=component_worst_q,
                metric_specs=metric_specs,
                components_manifest=components_manifest_t,
                component_metric_rankings=component_metric_rankings_t,
                save_component_debug_images=save_component_debug_images,
                component_debug_max_items=component_debug_max_items,
                rows_by_id=rows_by_id_t,
                pred_prob_clean=pred_prob_t,
                pred_bin_clean=pred_bin_t,
                gt_bin=gt_bin,
            )
            threshold_grid_timings["component_outputs"] = threshold_grid_timings.get("component_outputs", 0.0) + (
                time.perf_counter() - p0
            )
    _timeit("threshold_stability", t0)
    for key, value in threshold_grid_timings.items():
        timings[f"threshold_stability/{key}"] = timings.get(f"threshold_stability/{key}", 0.0) + float(value)

    for metric_base, metric_values in threshold_metric_series.items():
        finite_vals = [float(v) for v in metric_values if np.isfinite(v)]
        stab = _stability_stats(finite_vals)
        out.update({f"stability/{metric_base}_{k}": float(v) for k, v in stab.items()})

    timings["wall"] = max(0.0, float(time.perf_counter() - wall_t0))
    _log_timing(segment_id=seg, downsample=ds, cache_hit=cache_hit, timings=timings)

    return out
