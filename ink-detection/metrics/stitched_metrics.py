from __future__ import annotations

import hashlib
import os
import os.path as osp
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .stitched_components import (
    COMPONENT_METRIC_SPECS,
    _build_gt_component_templates,
    component_metrics_by_gt_bbox,
    summarize_component_rows,
    write_global_component_manifest,
)
from .stitched_primitives import (
    _as_bool_2d,
    _count_holes_2d,
    _label_components,
    _local_metrics_from_binary,
    betti_numbers_2d,
    boundary_precision_recall_f1,
    confusion_counts,
    dice_from_confusion,
    nsd_surface_dice,
    pseudo_fmeasure_values,
    skeletonize_binary,
    skeleton_tube_metrics,
    soft_dice_from_prob,
    voi_from_component_labels,
    weighted_pseudo_fmeasure_from_weights,
)


__all__ = [
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
) -> Tuple[np.ndarray, np.ndarray]:
    label_base = osp.join(dataset_root, segment_id, f"{segment_id}_inklabels{label_suffix}")
    mask_base = osp.join(dataset_root, segment_id, f"{segment_id}_mask{mask_suffix}")
    gt_gray = _read_gray_any(label_base)
    valid_gray = _read_gray_any(mask_base)

    gt_bin_full = (gt_gray.astype(np.float32) / 255.0) >= 0.5
    valid_full = valid_gray.astype(np.uint8) > 0

    if roi_offset is not None and (int(roi_offset[0]) != 0 or int(roi_offset[1]) != 0):
        gt_bin = _downsample_bool_any_roi(gt_bin_full, ds=downsample, out_hw=pred_shape, offset=roi_offset)
        valid = _downsample_bool_any_roi(valid_full, ds=downsample, out_hw=pred_shape, offset=roi_offset)
    else:
        gt_bin = _downsample_bool_any(gt_bin_full, ds=downsample, out_hw=pred_shape)
        valid = _downsample_bool_any(valid_full, ds=downsample, out_hw=pred_shape)
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


def _component_metric_means_for_stability(
    pred_prob_clean: np.ndarray,
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    *,
    gt_component_templates: List[Dict[str, Any]],
    connectivity: int,
) -> Dict[str, float]:
    pred_prob_clean = np.asarray(pred_prob_clean, dtype=np.float32)
    pred_bin = _as_bool_2d(pred_bin)
    gt_bin = _as_bool_2d(gt_bin)
    if pred_prob_clean.shape != gt_bin.shape:
        raise ValueError(f"pred_prob_clean/gt_bin shape mismatch: {pred_prob_clean.shape} vs {gt_bin.shape}")
    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"pred_bin/gt_bin shape mismatch: {pred_bin.shape} vs {gt_bin.shape}")

    dice_hard_vals: List[float] = []
    dice_soft_vals: List[float] = []
    voi_vals: List[float] = []
    pfm_vals: List[float] = []
    pfm_nonempty_vals: List[float] = []
    pfm_weighted_vals: List[float] = []
    betti_l1_vals: List[float] = []

    for template in gt_component_templates:
        y0, y1, x0, x1 = [int(v) for v in template["bbox"]]
        crop_pred_prob = pred_prob_clean[y0:y1, x0:x1]
        crop_pred_bin = pred_bin[y0:y1, x0:x1]
        crop_gt = gt_bin[y0:y1, x0:x1]

        c = confusion_counts(crop_pred_bin, crop_gt)
        crop_gt_lab = template["gt_lab"]
        if crop_gt_lab.shape != crop_gt.shape:
            raise ValueError(
                f"template gt_lab shape mismatch for component {template['component_id']}: "
                f"{crop_gt_lab.shape} vs {crop_gt.shape}"
            )
        crop_pred_lab, _ = _label_components(crop_pred_bin, connectivity=connectivity)
        dice_hard_vals.append(float(dice_from_confusion(c)))
        dice_soft_vals.append(float(soft_dice_from_prob(crop_pred_prob, crop_gt)))
        voi_vals.append(float(voi_from_component_labels(crop_gt_lab, crop_pred_lab)))
        pfm_i, pfm_nonempty_i = pseudo_fmeasure_values(crop_pred_bin, crop_gt, skel_gt=template["gt_skel"])
        pfm_vals.append(float(pfm_i))
        pfm_nonempty_vals.append(float(pfm_nonempty_i))
        pfm_weighted_vals.append(
            float(
                weighted_pseudo_fmeasure_from_weights(
                    crop_pred_bin,
                    crop_gt,
                    recall_weights=template["pfm_weight_recall"],
                    recall_weights_sum=float(template["pfm_weight_recall_sum"]),
                    precision_weights=template["pfm_weight_precision"],
                )
            )
        )

        pred_beta0, pred_beta1 = betti_numbers_2d(crop_pred_bin, connectivity=connectivity)
        gt_beta0 = int(template["gt_beta0"])
        gt_beta1 = int(template["gt_beta1"])
        betti_l1_vals.append(float(abs(int(pred_beta0) - gt_beta0) + abs(int(pred_beta1) - gt_beta1)))

    def _mean_or_nan(vals: List[float]) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        return float(arr.mean()) if arr.size else float("nan")

    def _nanmean_or_nan(vals: List[float]) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return float("nan")
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    return {
        "dice_hard": _mean_or_nan(dice_hard_vals),
        "dice_soft": _mean_or_nan(dice_soft_vals),
        "voi": _mean_or_nan(voi_vals),
        "pfm": _mean_or_nan(pfm_vals),
        "pfm_nonempty": _nanmean_or_nan(pfm_nonempty_vals),
        "pfm_weighted": _mean_or_nan(pfm_weighted_vals),
        "betti_l1": _mean_or_nan(betti_l1_vals),
    }


def _postprocess_prediction(
    pred_prob: np.ndarray,
    *,
    eval_mask: np.ndarray,
    threshold: float,
    connectivity: int,
    component_min_area: int,
    return_labels: bool = False,
):
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    eval_mask = _as_bool_2d(eval_mask)
    if pred_prob.shape != eval_mask.shape:
        raise ValueError(f"pred_prob/eval_mask shape mismatch: {pred_prob.shape} vs {eval_mask.shape}")

    pred_bin = np.logical_and(pred_prob >= float(threshold), eval_mask)
    if component_min_area and int(component_min_area) > 0:
        lab, n_lab = _label_components(pred_bin, connectivity=connectivity)
        if n_lab > 0:
            sizes = np.bincount(lab.ravel())[1:].astype(np.int64)
            keep = sizes >= int(component_min_area)
            keep_table = np.zeros(n_lab + 1, dtype=bool)
            keep_table[1:] = keep
            pred_bin = keep_table[lab]

    pred_prob_clean = pred_prob.copy()
    pred_prob_clean[~pred_bin] = 0.0
    if return_labels:
        pred_lab, n_pred = _label_components(pred_bin, connectivity=connectivity)
        return pred_bin, pred_prob_clean, pred_lab, int(n_pred)
    return pred_bin, pred_prob_clean


def _compute_full_region_metrics(
    *,
    pred_bin_clean: np.ndarray,
    pred_prob_clean: np.ndarray,
    gt_bin: np.ndarray,
    betti_connectivity: int,
    drd_block_size: int,
    boundary_k: int,
    boundary_tols: Optional[np.ndarray],
    skeleton_radius: Optional[np.ndarray],
    gt_beta0: int,
    gt_beta1: int,
    skel_gt: np.ndarray,
    skeleton_thinning_type: str,
    timings: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    skel_pred = skeletonize_binary(pred_bin_clean, thinning_type=skeleton_thinning_type)
    full_metrics = _local_metrics_from_binary(
        pred_bin_clean,
        gt_bin,
        connectivity=betti_connectivity,
        drd_block_size=drd_block_size,
        boundary_k=boundary_k,
        gt_beta0=int(gt_beta0),
        gt_beta1=int(gt_beta1),
        skel_gt=skel_gt,
        skel_pred=skel_pred,
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
        "euler_pred": float(full_metrics["euler_pred"]),
        "euler_gt": float(full_metrics["euler_gt"]),
        "abs_euler_err": float(full_metrics["abs_euler_err"]),
        "boundary/hd": float(full_metrics["boundary_hd"]),
        "boundary/hd95": float(full_metrics["boundary_hd95"]),
        "boundary/assd": float(full_metrics["boundary_assd"]),
        "skeleton/recall": float(full_metrics["skeleton_recall"]),
        "skeleton/cldice": float(full_metrics["skeleton_cldice"]),
        "pfm": float(full_metrics["pfm"]),
        "pfm_nonempty": float(full_metrics["pfm_nonempty"]),
        "pfm_weighted": float(full_metrics["pfm_weighted"]),
        "skeleton/chamfer": float(full_metrics["skeleton_chamfer"]),
        "skeleton/chamfer_pred_to_gt": float(full_metrics["skeleton_chamfer_pred_to_gt"]),
        "skeleton/chamfer_gt_to_pred": float(full_metrics["skeleton_chamfer_gt_to_pred"]),
    }

    tau_values = np.asarray([1.0], dtype=np.float32) if boundary_tols is None else np.asarray(boundary_tols)
    for tau in tau_values:
        tau_f = float(tau)
        tau_key = str(tau_f).replace(".", "p")
        bf = boundary_precision_recall_f1(pred_bin_clean, gt_bin, tau=tau_f, boundary_k=boundary_k)
        out[f"boundary/bf1_tau{tau_key}"] = float(bf["b_f1"])
        out[f"boundary/nsd_tau{tau_key}"] = float(nsd_surface_dice(pred_bin_clean, gt_bin, tau=tau_f, boundary_k=boundary_k))

    radius_values = np.asarray([1], dtype=np.int64) if skeleton_radius is None else np.asarray(skeleton_radius)
    for radius in radius_values:
        r_i = int(radius)
        r_key = str(r_i)
        tube = skeleton_tube_metrics(pred_bin_clean, gt_bin, radius=r_i, skel_gt=skel_gt)
        out[f"skeleton/tube_f1_r{r_key}"] = float(tube["tube_f1"])
    return out, skel_pred


def _build_components_manifest(
    *,
    component_rows: List[Dict[str, Any]],
    full_off_y: int,
    full_off_x: int,
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
            "pfm": float(row["pfm"]),
            "pfm_nonempty": float(row["pfm_nonempty"]),
            "pfm_weighted": float(row["pfm_weighted"]),
            "voi": float(row["voi"]),
            "betti_l1": float(row["betti_l1"]),
            "abs_euler_err": float(row["abs_euler_err"]),
            "boundary_hd95": float(row["boundary_hd95"]),
            "skeleton_recall": float(row["skeleton_recall"]),
            "skeleton_cldice": float(row["skeleton_cldice"]),
            "skeleton_chamfer": float(row["skeleton_chamfer"]),
            "betti_abs_beta0_err": float(row["betti_abs_beta0_err"]),
            "betti_abs_beta1_err": float(row["betti_abs_beta1_err"]),
            "betti_beta0_pred": float(row["betti_beta0_pred"]),
            "betti_beta1_pred": float(row["betti_beta1_pred"]),
            "betti_beta0_gt": float(row["betti_beta0_gt"]),
            "betti_beta1_gt": float(row["betti_beta1_gt"]),
        }
        components_manifest.append(entry)
        rows_by_id[entry["component_id"]] = row
    return components_manifest, rows_by_id


def _write_component_outputs(
    *,
    component_output_dir: str,
    fragment_id: str,
    downsample: int,
    gt_lab: np.ndarray,
    pred_lab: np.ndarray,
    pred_full_shape: Tuple[int, int],
    roi_offset: Tuple[int, int],
    crop_offset: Tuple[int, int],
    full_offset: Tuple[int, int],
    component_worst_k: Optional[int],
    component_worst_q: Optional[float],
    components_manifest: List[Dict[str, Any]],
    component_metric_rankings: Dict[str, Dict[str, Any]],
    save_component_debug_images: bool,
    component_debug_max_items: Optional[int],
    gt_component_templates: List[Dict[str, Any]],
    rows_by_id: Dict[int, Dict[str, Any]],
    pred_prob_clean: np.ndarray,
    pred_bin_clean: np.ndarray,
    gt_bin: np.ndarray,
) -> None:
    import json as _json
    from PIL import Image

    os.makedirs(component_output_dir, exist_ok=True)
    safe_seg = str(fragment_id).replace("/", "_")
    gt_path = osp.join(component_output_dir, f"components_{safe_seg}_ds{int(downsample)}_gt_labels.png")
    pred_path = osp.join(component_output_dir, f"components_{safe_seg}_ds{int(downsample)}_pred_labels.png")
    meta_path = osp.join(component_output_dir, f"components_{safe_seg}_ds{int(downsample)}_meta.json")
    manifest_path = osp.join(component_output_dir, f"components_{safe_seg}_ds{int(downsample)}_manifest.json")

    if not (osp.exists(gt_path) and osp.exists(pred_path) and osp.exists(meta_path)):
        gt_u16 = gt_lab.astype(np.uint16, copy=False)
        pred_u16 = pred_lab.astype(np.uint16, copy=False)
        Image.fromarray(gt_u16, mode="I;16").save(gt_path)
        Image.fromarray(pred_u16, mode="I;16").save(pred_path)

    off_y, off_x = [int(v) for v in roi_offset]
    crop_off_y, crop_off_x = [int(v) for v in crop_offset]
    full_off_y, full_off_x = [int(v) for v in full_offset]
    meta = {
        "segment_id": str(fragment_id),
        "downsample": int(downsample),
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

    template_by_id = {int(t["component_id"]): t for t in gt_component_templates}
    selected_component_ids: List[int] = []
    for metric_name, _ in COMPONENT_METRIC_SPECS:
        selected_component_ids.extend(component_metric_rankings[metric_name]["worst_k_component_ids"])
        selected_component_ids.extend(component_metric_rankings[metric_name]["worst_q_component_ids"])
    selected_component_ids = sorted(set(int(v) for v in selected_component_ids))
    if not selected_component_ids:
        raise ValueError("save_component_debug_images requires component_worst_k or component_worst_q to be set")
    if component_debug_max_items is not None:
        max_items = max(1, int(component_debug_max_items))
        selected_component_ids = selected_component_ids[:max_items]

    debug_dir = osp.join(component_output_dir, f"components_{safe_seg}_ds{int(downsample)}_debug")
    os.makedirs(debug_dir, exist_ok=True)
    for component_id in selected_component_ids:
        if component_id not in rows_by_id:
            raise KeyError(f"component_id={component_id} missing from component rows")
        if component_id not in template_by_id:
            raise KeyError(f"component_id={component_id} missing from GT templates")
        row = rows_by_id[component_id]
        if "_pred_skel" not in row:
            raise KeyError(f"component_id={component_id} missing cached pred skeleton")
        bbox = [int(v) for v in row["bbox"]]
        by0, by1, bx0, bx1 = bbox
        crop_pred_prob = pred_prob_clean[by0:by1, bx0:bx1]
        crop_pred_bin = pred_bin_clean[by0:by1, bx0:bx1]
        crop_gt = gt_bin[by0:by1, bx0:bx1]
        skel_pred_comp = row["_pred_skel"]
        skel_gt_comp = template_by_id[component_id]["gt_skel"]

        cid = f"{int(component_id):05d}"
        Image.fromarray((np.clip(crop_pred_prob, 0.0, 1.0) * 255.0).astype(np.uint8)).save(
            osp.join(debug_dir, f"comp_{cid}_pred_prob.png")
        )
        Image.fromarray((crop_pred_bin.astype(np.uint8) * 255)).save(osp.join(debug_dir, f"comp_{cid}_pred_bin.png"))
        Image.fromarray((crop_gt.astype(np.uint8) * 255)).save(osp.join(debug_dir, f"comp_{cid}_gt_bin.png"))
        Image.fromarray((skel_pred_comp.astype(np.uint8) * 255)).save(
            osp.join(debug_dir, f"comp_{cid}_pred_skeleton.png")
        )
        Image.fromarray((skel_gt_comp.astype(np.uint8) * 255)).save(osp.join(debug_dir, f"comp_{cid}_gt_skeleton.png"))


def _compute_threshold_stability_metrics(
    *,
    threshold_grid: np.ndarray,
    pred_prob: np.ndarray,
    eval_mask: np.ndarray,
    betti_connectivity: int,
    component_min_area: int,
    gt_bin: np.ndarray,
    gt_component_templates: List[Dict[str, Any]],
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    tgrid = np.asarray(threshold_grid, dtype=np.float32)
    dice_hard_vals: List[float] = []
    dice_soft_vals: List[float] = []
    voi_vals: List[float] = []
    pfm_vals: List[float] = []
    pfm_nonempty_vals: List[float] = []
    pfm_weighted_vals: List[float] = []
    betti_l1_vals: List[float] = []
    for t in tgrid:
        t0 = time.perf_counter()
        pred_bin_t, pred_prob_t = _postprocess_prediction(
            pred_prob,
            eval_mask=eval_mask,
            threshold=float(t),
            connectivity=betti_connectivity,
            component_min_area=component_min_area,
            return_labels=False,
        )
        if timings is not None:
            timings["postprocess"] = timings.get("postprocess", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        component_means_t = _component_metric_means_for_stability(
            pred_prob_t,
            pred_bin_t,
            gt_bin,
            gt_component_templates=gt_component_templates,
            connectivity=betti_connectivity,
        )
        if timings is not None:
            timings["component_means"] = timings.get("component_means", 0.0) + (time.perf_counter() - t0)
        dice_hard_vals.append(float(component_means_t["dice_hard"]))
        dice_soft_vals.append(float(component_means_t["dice_soft"]))
        voi_vals.append(float(component_means_t["voi"]))
        pfm_vals.append(float(component_means_t["pfm"]))
        pfm_nonempty_vals.append(float(component_means_t["pfm_nonempty"]))
        pfm_weighted_vals.append(float(component_means_t["pfm_weighted"]))
        betti_l1_vals.append(float(component_means_t["betti_l1"]))

    out: Dict[str, float] = {}
    dice_hard_stab = _stability_stats(dice_hard_vals)
    dice_soft_stab = _stability_stats(dice_soft_vals)
    voi_stab = _stability_stats(voi_vals)
    pfm_stab = _stability_stats(pfm_vals)
    pfm_nonempty_arr = np.asarray(pfm_nonempty_vals, dtype=np.float64)
    pfm_nonempty_arr = pfm_nonempty_arr[np.isfinite(pfm_nonempty_arr)]
    pfm_nonempty_stab = _stability_stats(pfm_nonempty_arr.tolist())
    pfm_weighted_stab = _stability_stats(pfm_weighted_vals)
    betti_l1_stab = _stability_stats(betti_l1_vals)
    out.update({f"stability/dice_hard_{k}": float(v) for k, v in dice_hard_stab.items()})
    out.update({f"stability/dice_soft_{k}": float(v) for k, v in dice_soft_stab.items()})
    out.update({f"stability/voi_{k}": float(v) for k, v in voi_stab.items()})
    out.update({f"stability/pfm_{k}": float(v) for k, v in pfm_stab.items()})
    out.update({f"stability/pfm_nonempty_{k}": float(v) for k, v in pfm_nonempty_stab.items()})
    out.update({f"stability/pfm_weighted_{k}": float(v) for k, v in pfm_weighted_stab.items()})
    out.update({f"stability/betti_l1_{k}": float(v) for k, v in betti_l1_stab.items()})
    return out


def _save_skeleton_images(
    *,
    output_dir: str,
    fragment_id: str,
    threshold: float,
    skel_gt: np.ndarray,
    pred_bin_clean: np.ndarray,
    skel_pred: Optional[np.ndarray],
    skeleton_thinning_type: str,
) -> Optional[np.ndarray]:
    os.makedirs(output_dir, exist_ok=True)
    safe_seg = str(fragment_id).replace("/", "_")
    thr_tag = str(float(threshold)).replace(".", "p")
    pred_path = osp.join(output_dir, f"{safe_seg}_pred_thr{thr_tag}.png")
    gt_path = osp.join(output_dir, f"{safe_seg}_gt.png")
    if (not osp.exists(pred_path)) or (not osp.exists(gt_path)):
        from PIL import Image

        if skel_pred is None:
            skel_pred = skeletonize_binary(pred_bin_clean, thinning_type=skeleton_thinning_type)
        pred_img = (skel_pred.astype(np.uint8) * 255)
        gt_img = (skel_gt.astype(np.uint8) * 255)
        Image.fromarray(pred_img).save(pred_path)
        Image.fromarray(gt_img).save(gt_path)
    return skel_pred


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

    total = sum(timings.values())
    parts = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
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
    skeleton_radius: Optional[np.ndarray] = None,
    component_worst_q: Optional[float] = 0.2,
    component_worst_k: Optional[int] = 2,
    component_min_area: int = 0,
    component_pad: int = 5,
    skeleton_thinning_type: str = "zhang_suen",
    enable_full_region_metrics: bool = False,
    threshold_grid: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
    component_output_dir: Optional[str] = None,
    stitched_inputs_output_dir: Optional[str] = None,
    save_skeleton_images: bool = True,
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

    def _timeit(name: str, t0: float) -> None:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)

    ds = max(1, int(downsample))
    pad_i = max(0, int(component_pad))
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
        str(skeleton_thinning_type),
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
        skel_gt = cache_entry["skel_gt"]
        y0, y1, x0, x1 = cache_entry["crop"]
        pred_full_shape = tuple(cache_entry.get("full_shape", pred_full_shape))
        pred_prob = pred_prob[y0:y1, x0:x1]
        pred_has = pred_has[y0:y1, x0:x1]
        if "gt_component_templates" not in cache_entry:
            cache_entry["gt_component_templates"] = None
        if "gt_component_pad" not in cache_entry:
            cache_entry["gt_component_pad"] = None
        cached_component_pad = cache_entry["gt_component_pad"]
        if cached_component_pad is not None and int(cached_component_pad) != pad_i:
            raise ValueError(
                f"component_pad changed for cached segment={seg}: "
                f"cached={int(cached_component_pad)} current={int(pad_i)}"
            )
    else:
        cache_hit = False
        t0 = time.perf_counter()
        gt_bin_ds, valid = _load_ground_truth_masks(
            dataset_root=dataset_root,
            segment_id=seg,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
            downsample=ds,
            pred_shape=tuple(pred_prob.shape),
            roi_offset=roi_offset,
        )
        eval_mask_full = np.logical_and(pred_has, valid)
        if not eval_mask_full.any():
            _log_empty_eval_mask(
                segment_id=seg,
                downsample=ds,
                pred_shape=tuple(pred_prob.shape),
                pred_has_count=int(pred_has.sum()),
            )
            return {}
        pred_prob, gt_bin, eval_mask, crop = _prepare_eval_crop(
            pred_prob=pred_prob,
            pred_has=pred_has,
            gt_bin=gt_bin_ds,
            valid=valid,
        )
        y0, y1, x0, x1 = crop
        pred_has = pred_has[y0:y1, x0:x1]

        gt_lab, n_gt = _label_components(gt_bin, connectivity=betti_connectivity)
        gt_beta0 = int(n_gt)
        gt_beta1 = _count_holes_2d(gt_bin, connectivity=betti_connectivity)
        skel_gt = skeletonize_binary(gt_bin, thinning_type=skeleton_thinning_type)
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
        }
        _gt_cache_put(cache_key, cache_entry)
        _timeit("load_gt", t0)

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

    t0 = time.perf_counter()
    pred_bin_clean, pred_prob_clean, pred_lab, n_pred = _postprocess_prediction(
        pred_prob,
        eval_mask=eval_mask,
        threshold=float(threshold),
        connectivity=betti_connectivity,
        component_min_area=component_min_area,
        return_labels=True,
    )
    _timeit("postprocess_and_confusion", t0)
    pred_bin_threshold = np.logical_and(pred_prob >= float(threshold), eval_mask)

    skel_pred: Optional[np.ndarray] = None

    out: Dict[str, float] = {}
    if enable_full_region_metrics:
        t0 = time.perf_counter()
        full_region_timings: Dict[str, float] = {}
        full_region_out, skel_pred = _compute_full_region_metrics(
            pred_bin_clean=pred_bin_clean,
            pred_prob_clean=pred_prob_clean,
            gt_bin=gt_bin,
            betti_connectivity=betti_connectivity,
            drd_block_size=drd_block_size,
            boundary_k=boundary_k,
            boundary_tols=boundary_tols,
            skeleton_radius=skeleton_radius,
            gt_beta0=int(gt_beta0),
            gt_beta1=int(gt_beta1),
            skel_gt=skel_gt,
            skeleton_thinning_type=skeleton_thinning_type,
            timings=full_region_timings,
        )
        out.update(full_region_out)
        _timeit("full_region_metrics", t0)
        for key, value in full_region_timings.items():
            timings[f"full_region_metrics/{key}"] = timings.get(f"full_region_metrics/{key}", 0.0) + float(value)

    t0 = time.perf_counter()
    gt_component_templates = cache_entry["gt_component_templates"]
    if gt_component_templates is None:
        gt_component_templates = _build_gt_component_templates(
            gt_bin,
            gt_lab,
            int(n_gt),
            connectivity=betti_connectivity,
            pad=pad_i,
            skeleton_thinning_type=skeleton_thinning_type,
        )
        cache_entry["gt_component_templates"] = gt_component_templates
        cache_entry["gt_component_pad"] = int(pad_i)
    component_metric_timings: Dict[str, float] = {}
    component_rows, component_metric_stats, component_metric_rankings = component_metrics_by_gt_bbox(
        pred_prob_clean,
        pred_bin_clean,
        gt_bin,
        connectivity=betti_connectivity,
        drd_block_size=drd_block_size,
        boundary_k=boundary_k,
        pad=pad_i,
        worst_q=component_worst_q,
        worst_k=component_worst_k,
        keep_pred_skeleton=save_component_debug_images,
        gt_lab=gt_lab,
        n_gt=n_gt,
        gt_component_templates=gt_component_templates,
        skeleton_thinning_type=skeleton_thinning_type,
        timings=component_metric_timings,
    )
    for metric_name, stats in component_metric_stats.items():
        out.update({f"components/{metric_name}_{k}": float(v) for k, v in stats.items()})
    out["components/n_pred"] = float(n_pred)
    if not enable_full_region_metrics:
        alias_map = {
            "dice_hard": "dice_hard",
            "dice_soft": "dice_soft",
            "accuracy": "accuracy",
            "voi": "voi",
            "mpm": "mpm",
            "drd": "drd",
            "pfm": "pfm",
            "pfm_nonempty": "pfm_nonempty",
            "pfm_weighted": "pfm_weighted",
            "betti/l1_betti_err": "betti_l1",
            "abs_euler_err": "abs_euler_err",
            "boundary/hd95": "boundary_hd95",
            "skeleton/recall": "skeleton_recall",
            "skeleton/cldice": "skeleton_cldice",
            "skeleton/chamfer": "skeleton_chamfer",
        }
        for out_key, comp_metric in alias_map.items():
            out[out_key] = float(out[f"components/{comp_metric}_mean"])
    _timeit("components_metrics", t0)
    for key, value in component_metric_timings.items():
        timings[f"components_metrics/{key}"] = timings.get(f"components_metrics/{key}", 0.0) + float(value)

    off_y, off_x = [int(v) for v in roi_key]
    crop_off_y, crop_off_x = int(y0), int(x0)
    full_off_y = int(off_y + crop_off_y)
    full_off_x = int(off_x + crop_off_x)

    if stitched_inputs_output_dir:
        t0 = time.perf_counter()
        _save_stitched_eval_inputs(
            stitched_inputs_output_dir=stitched_inputs_output_dir,
            fragment_id=fragment_id,
            downsample=ds,
            eval_epoch=eval_epoch,
            threshold=float(threshold),
            component_min_area=component_min_area,
            pred_prob=pred_prob,
            pred_bin_threshold=pred_bin_threshold,
            pred_bin_postprocess=pred_bin_clean,
            gt_bin=gt_bin,
            eval_mask=eval_mask,
            roi_offset=(off_y, off_x),
            crop_offset=(crop_off_y, crop_off_x),
            full_offset=(full_off_y, full_off_x),
            full_shape=pred_full_shape,
        )
        _timeit("stitched_inputs_dump", t0)

    components_manifest, rows_by_id = _build_components_manifest(
        component_rows=component_rows,
        full_off_y=full_off_y,
        full_off_x=full_off_x,
    )

    if component_rows_collector is not None:
        for entry in components_manifest:
            global_entry = dict(entry)
            global_entry["segment_id"] = str(fragment_id)
            global_entry["downsample"] = int(ds)
            global_entry["global_component_id"] = (
                f"{global_entry['segment_id']}:{int(global_entry['component_id'])}"
            )
            component_rows_collector.append(global_entry)

    if component_output_dir:
        t0 = time.perf_counter()
        _write_component_outputs(
            component_output_dir=component_output_dir,
            fragment_id=fragment_id,
            downsample=ds,
            gt_lab=gt_lab,
            pred_lab=pred_lab,
            pred_full_shape=pred_full_shape,
            roi_offset=(off_y, off_x),
            crop_offset=(crop_off_y, crop_off_x),
            full_offset=(full_off_y, full_off_x),
            component_worst_k=component_worst_k,
            component_worst_q=component_worst_q,
            components_manifest=components_manifest,
            component_metric_rankings=component_metric_rankings,
            save_component_debug_images=save_component_debug_images,
            component_debug_max_items=component_debug_max_items,
            gt_component_templates=gt_component_templates,
            rows_by_id=rows_by_id,
            pred_prob_clean=pred_prob_clean,
            pred_bin_clean=pred_bin_clean,
            gt_bin=gt_bin,
        )
        _timeit("components_dump", t0)

    if threshold_grid is not None and len(threshold_grid):
        t0 = time.perf_counter()
        threshold_stability_timings: Dict[str, float] = {}
        out.update(
            _compute_threshold_stability_metrics(
                threshold_grid=np.asarray(threshold_grid, dtype=np.float32),
                pred_prob=pred_prob,
                eval_mask=eval_mask,
                betti_connectivity=betti_connectivity,
                component_min_area=component_min_area,
                gt_bin=gt_bin,
                gt_component_templates=gt_component_templates,
                timings=threshold_stability_timings,
            )
        )
        _timeit("threshold_stability", t0)
        for key, value in threshold_stability_timings.items():
            timings[f"threshold_stability/{key}"] = timings.get(f"threshold_stability/{key}", 0.0) + float(value)

    if output_dir is not None and save_skeleton_images:
        skel_pred = _save_skeleton_images(
            output_dir=output_dir,
            fragment_id=fragment_id,
            threshold=float(threshold),
            skel_gt=skel_gt,
            pred_bin_clean=pred_bin_clean,
            skel_pred=skel_pred,
            skeleton_thinning_type=skeleton_thinning_type,
        )

    _log_timing(segment_id=seg, downsample=ds, cache_hit=cache_hit, timings=timings)

    return out
