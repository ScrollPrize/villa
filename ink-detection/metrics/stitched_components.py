from __future__ import annotations

import os
import os.path as osp
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .stitched_primitives import (
    _as_bool_2d,
    _label_components,
    _local_metrics_from_binary,
    _pseudo_precision_weights,
    _pseudo_recall_weights,
    betti_numbers_2d,
    skeletonize_binary,
    soft_dice_from_prob,
)


def _build_gt_component_templates(
    gt_bin: np.ndarray,
    gt_lab: np.ndarray,
    n_gt: int,
    *,
    connectivity: int,
    pad: int,
    skeleton_thinning_type: str = "zhang_suen",
) -> List[Dict[str, Any]]:
    gt_bin = _as_bool_2d(gt_bin)
    gt_lab = np.asarray(gt_lab)
    if gt_lab.shape != gt_bin.shape:
        raise ValueError(f"gt_lab/gt_bin shape mismatch: {gt_lab.shape} vs {gt_bin.shape}")

    pad = max(0, int(pad))
    templates: List[Dict[str, Any]] = []
    for gi in range(1, int(n_gt) + 1):
        component_mask = gt_lab == gi
        if not component_mask.any():
            continue

        ys, xs = np.where(component_mask)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(gt_bin.shape[0], int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(gt_bin.shape[1], int(xs.max()) + 1 + pad)
        crop_gt = gt_bin[y0:y1, x0:x1]
        crop_gt_lab, _ = _label_components(crop_gt, connectivity=connectivity)
        gt_beta0, gt_beta1 = betti_numbers_2d(crop_gt, connectivity=connectivity)
        gt_skel = skeletonize_binary(crop_gt, thinning_type=skeleton_thinning_type)
        pfm_weight_recall = _pseudo_recall_weights(crop_gt).astype(np.float32, copy=False)
        pfm_weight_recall_sum = float(pfm_weight_recall.sum(dtype=np.float64))
        if crop_gt.any() and pfm_weight_recall_sum <= 0.0:
            raise ValueError(f"invalid weighted pseudo-recall sum for component {gi}: {pfm_weight_recall_sum}")
        pfm_weight_precision = _pseudo_precision_weights(crop_gt, connectivity=connectivity).astype(
            np.float32, copy=False
        )
        templates.append(
            {
                "component_id": int(gi),
                "bbox": [int(y0), int(y1), int(x0), int(x1)],
                "gt_beta0": int(gt_beta0),
                "gt_beta1": int(gt_beta1),
                "gt_lab": crop_gt_lab,
                "gt_skel": gt_skel,
                "pfm_weight_recall": pfm_weight_recall,
                "pfm_weight_recall_sum": float(pfm_weight_recall_sum),
                "pfm_weight_precision": pfm_weight_precision,
            }
        )
    return templates


COMPONENT_METRIC_SPECS: Tuple[Tuple[str, bool], ...] = (
    ("dice_hard", True),
    ("dice_soft", True),
    ("accuracy", True),
    ("pfm", True),
    ("pfm_nonempty", True),
    ("pfm_weighted", True),
    ("skeleton_cldice", True),
    ("skeleton_recall", True),
    ("voi", False),
    ("mpm", False),
    ("drd", False),
    ("betti_l1", False),
    ("abs_euler_err", False),
    ("boundary_hd95", False),
    ("skeleton_chamfer", False),
)


def _component_metric_summary(
    rows: List[Dict[str, Any]],
    *,
    key: str,
    higher_is_better: bool,
    worst_q: Optional[float],
    worst_k: Optional[int],
    id_key: str = "component_id",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not rows:
        return (
            {
                "n_gt": 0.0,
                "mean": float("nan"),
                "worst_k_mean": float("nan"),
                "worst_q_mean": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            },
            {
                "sort_order": "asc" if higher_is_better else "desc",
                "component_ids_sorted_worst_first": [],
                "worst_k_component_ids": [],
                "worst_q_component_ids": [],
            },
        )

    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    component_ids = [row[id_key] for row in rows]
    order = np.argsort(values) if higher_is_better else np.argsort(-values)
    sorted_values = values[order]
    sorted_ids = [component_ids[int(i)] for i in order.tolist()]

    worst_k_mean = float("nan")
    worst_q_mean = float("nan")
    worst_k_ids: List[Any] = []
    worst_q_ids: List[Any] = []
    if sorted_values.size:
        if worst_k is not None:
            k = max(1, min(int(worst_k), int(len(sorted_values))))
            worst_k_mean = float(sorted_values[:k].mean())
            worst_k_ids = list(sorted_ids[:k])
        if worst_q is not None:
            q = min(max(float(worst_q), 0.0), 1.0)
            kq = max(1, int(round(len(sorted_values) * q)))
            worst_q_mean = float(sorted_values[:kq].mean())
            worst_q_ids = list(sorted_ids[:kq])

    return (
        {
            "n_gt": float(len(rows)),
            "mean": float(values.mean()),
            "worst_k_mean": worst_k_mean,
            "worst_q_mean": worst_q_mean,
            "min": float(values.min()) if values.size else float("nan"),
            "max": float(values.max()) if values.size else float("nan"),
        },
        {
            "sort_order": "asc" if higher_is_better else "desc",
            "component_ids_sorted_worst_first": list(sorted_ids),
            "worst_k_component_ids": worst_k_ids,
            "worst_q_component_ids": worst_q_ids,
        },
    )


def summarize_component_rows(
    rows: List[Dict[str, Any]],
    *,
    worst_q: Optional[float],
    worst_k: Optional[int],
    id_key: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
    metric_stats: Dict[str, Dict[str, float]] = {}
    metric_rankings: Dict[str, Dict[str, Any]] = {}
    for metric_name, higher_is_better in COMPONENT_METRIC_SPECS:
        stats, rankings = _component_metric_summary(
            rows,
            key=metric_name,
            higher_is_better=higher_is_better,
            worst_q=worst_q,
            worst_k=worst_k,
            id_key=id_key,
        )
        metric_stats[metric_name] = stats
        metric_rankings[metric_name] = rankings
    return metric_stats, metric_rankings


def write_global_component_manifest(
    *,
    component_rows: List[Dict[str, Any]],
    component_output_dir: str,
    downsample: int,
    worst_k: Optional[int],
    worst_q: Optional[float],
    rankings: Dict[str, Dict[str, Any]],
) -> str:
    import json as _json

    os.makedirs(component_output_dir, exist_ok=True)
    out_path = osp.join(component_output_dir, f"components_global_ds{int(downsample)}_manifest.json")
    manifest = {
        "downsample": int(downsample),
        "worst_k": None if worst_k is None else int(worst_k),
        "worst_q": None if worst_q is None else float(worst_q),
        "n_components": int(len(component_rows)),
        "components": component_rows,
        "rankings": rankings,
    }
    with open(out_path, "w") as f:
        _json.dump(manifest, f, indent=2)
    return out_path


def component_metrics_by_gt_bbox(
    pred_prob: np.ndarray,
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    *,
    connectivity: int = 2,
    drd_block_size: int = 8,
    boundary_k: int = 3,
    pad: int = 0,
    worst_q: Optional[float] = 0.2,
    worst_k: Optional[int] = 2,
    keep_pred_skeleton: bool = False,
    gt_lab: Optional[np.ndarray] = None,
    n_gt: Optional[int] = None,
    gt_component_templates: Optional[List[Dict[str, Any]]] = None,
    skeleton_thinning_type: str = "zhang_suen",
    timings: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    pred_bin = _as_bool_2d(pred_bin)
    gt_bin = _as_bool_2d(gt_bin)
    if pred_prob.shape != gt_bin.shape:
        raise ValueError(f"pred_prob/gt_bin shape mismatch: {pred_prob.shape} vs {gt_bin.shape}")
    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"pred_bin/gt_bin shape mismatch: {pred_bin.shape} vs {gt_bin.shape}")
    if gt_component_templates is None:
        if gt_lab is None or n_gt is None:
            gt_lab, n_gt = _label_components(gt_bin, connectivity=connectivity)
        gt_component_templates = _build_gt_component_templates(
            gt_bin,
            gt_lab,
            int(n_gt),
            connectivity=connectivity,
            pad=pad,
            skeleton_thinning_type=skeleton_thinning_type,
        )

    rows: List[Dict[str, Any]] = []
    for template in gt_component_templates:
        gi = int(template["component_id"])
        bbox = [int(v) for v in template["bbox"]]
        y0, y1, x0, x1 = bbox

        crop_pred_prob = pred_prob[y0:y1, x0:x1]
        crop_pred_bin = pred_bin[y0:y1, x0:x1]
        crop_gt = gt_bin[y0:y1, x0:x1]
        crop_gt_lab = template["gt_lab"]
        if crop_gt_lab.shape != crop_gt.shape:
            raise ValueError(
                f"template gt_lab shape mismatch for component {gi}: "
                f"{crop_gt_lab.shape} vs {crop_gt.shape}"
            )

        t0 = time.perf_counter()
        crop_pred_skel = skeletonize_binary(crop_pred_bin, thinning_type=skeleton_thinning_type)
        if timings is not None:
            timings["skeletonize_pred"] = timings.get("skeletonize_pred", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        crop_pred_lab, _ = _label_components(crop_pred_bin, connectivity=connectivity)
        if timings is not None:
            timings["label_pred"] = timings.get("label_pred", 0.0) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        local_metric_timings: Dict[str, float] = {}
        local_metrics = _local_metrics_from_binary(
            crop_pred_bin,
            crop_gt,
            connectivity=connectivity,
            drd_block_size=drd_block_size,
            boundary_k=boundary_k,
            gt_beta0=int(template["gt_beta0"]),
            gt_beta1=int(template["gt_beta1"]),
            skel_gt=template["gt_skel"],
            skel_pred=crop_pred_skel,
            gt_lab=crop_gt_lab,
            pred_lab=crop_pred_lab,
            pfm_weight_recall=template["pfm_weight_recall"],
            pfm_weight_recall_sum=float(template["pfm_weight_recall_sum"]),
            pfm_weight_precision=template["pfm_weight_precision"],
            timings=local_metric_timings,
        )
        if timings is not None:
            timings["local_metrics"] = timings.get("local_metrics", 0.0) + (time.perf_counter() - t0)
            for key, value in local_metric_timings.items():
                timings[f"local_metrics/{key}"] = timings.get(f"local_metrics/{key}", 0.0) + float(value)
        t0 = time.perf_counter()
        dice_hard = float(local_metrics["dice"])
        dice_soft = float(soft_dice_from_prob(crop_pred_prob, crop_gt))
        if timings is not None:
            timings["soft_dice"] = timings.get("soft_dice", 0.0) + (time.perf_counter() - t0)

        row = {
            "component_id": int(gi),
            "bbox": bbox,
            "dice_hard": dice_hard,
            "dice_soft": dice_soft,
            "accuracy": float(local_metrics["accuracy"]),
            "voi": float(local_metrics["voi"]),
            "mpm": float(local_metrics["mpm"]),
            "drd": float(local_metrics["drd"]),
            "pfm": float(local_metrics["pfm"]),
            "pfm_nonempty": float(local_metrics["pfm_nonempty"]),
            "pfm_weighted": float(local_metrics["pfm_weighted"]),
            "betti_l1": float(local_metrics["betti_l1"]),
            "abs_euler_err": float(local_metrics["abs_euler_err"]),
            "boundary_hd95": float(local_metrics["boundary_hd95"]),
            "skeleton_recall": float(local_metrics["skeleton_recall"]),
            "skeleton_cldice": float(local_metrics["skeleton_cldice"]),
            "skeleton_chamfer": float(local_metrics["skeleton_chamfer"]),
            "betti_abs_beta0_err": float(local_metrics["betti_abs_beta0_err"]),
            "betti_abs_beta1_err": float(local_metrics["betti_abs_beta1_err"]),
            "betti_beta0_pred": float(local_metrics["betti_beta0_pred"]),
            "betti_beta1_pred": float(local_metrics["betti_beta1_pred"]),
            "betti_beta0_gt": float(local_metrics["betti_beta0_gt"]),
            "betti_beta1_gt": float(local_metrics["betti_beta1_gt"]),
        }
        if keep_pred_skeleton:
            row["_pred_skel"] = crop_pred_skel
        rows.append(row)

    metric_stats, metric_rankings = summarize_component_rows(
        rows,
        worst_q=worst_q,
        worst_k=worst_k,
        id_key="component_id",
    )

    return rows, metric_stats, metric_rankings
