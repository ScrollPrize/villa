from __future__ import annotations

from typing import Dict

WANDB_VAL_STITCH_METRIC_SUMMARIES: Dict[str, str] = {
    "metrics/val_stitch/global/components/dice_hard/mean": "max",
    "metrics/val_stitch/global/components/dice_hard/worst_k_mean": "max",
    "metrics/val_stitch/global/components/dice_hard/worst_q_mean": "max",
    "metrics/val_stitch/global/components/dice_soft/mean": "max",
    "metrics/val_stitch/global/components/dice_soft/worst_k_mean": "max",
    "metrics/val_stitch/global/components/dice_soft/worst_q_mean": "max",
    "metrics/val_stitch/global/components/pfm/mean": "max",
    "metrics/val_stitch/global/components/pfm/worst_k_mean": "max",
    "metrics/val_stitch/global/components/pfm/worst_q_mean": "max",
    "metrics/val_stitch/global/components/pfm_nonempty/mean": "max",
    "metrics/val_stitch/global/components/pfm_nonempty/worst_k_mean": "max",
    "metrics/val_stitch/global/components/pfm_nonempty/worst_q_mean": "max",
    "metrics/val_stitch/global/components/pfm_weighted/mean": "max",
    "metrics/val_stitch/global/components/pfm_weighted/worst_k_mean": "max",
    "metrics/val_stitch/global/components/pfm_weighted/worst_q_mean": "max",
    "metrics/val_stitch/global/components/voi/mean": "min",
    "metrics/val_stitch/global/components/voi/worst_k_mean": "min",
    "metrics/val_stitch/global/components/voi/worst_q_mean": "min",
    "metrics/val_stitch/global/components/mpm/mean": "min",
    "metrics/val_stitch/global/components/mpm/worst_k_mean": "min",
    "metrics/val_stitch/global/components/mpm/worst_q_mean": "min",
    "metrics/val_stitch/global/components/drd/mean": "min",
    "metrics/val_stitch/global/components/drd/worst_k_mean": "min",
    "metrics/val_stitch/global/components/drd/worst_q_mean": "min",
    "metrics/val_stitch/global/components/betti_l1/mean": "min",
    "metrics/val_stitch/global/components/betti_l1/worst_k_mean": "min",
    "metrics/val_stitch/global/components/betti_l1/worst_q_mean": "min",
    "metrics/val_stitch/global/components/betti_match_err/mean": "min",
    "metrics/val_stitch/global/components/betti_match_err/worst_k_mean": "min",
    "metrics/val_stitch/global/components/betti_match_err/worst_q_mean": "min",
    "metrics/val_stitch/global/components/abs_euler_err/mean": "min",
    "metrics/val_stitch/global/components/abs_euler_err/worst_k_mean": "min",
    "metrics/val_stitch/global/components/abs_euler_err/worst_q_mean": "min",
    "metrics/val_stitch/global/components/boundary_hd95/mean": "min",
    "metrics/val_stitch/global/components/boundary_hd95/worst_k_mean": "min",
    "metrics/val_stitch/global/components/boundary_hd95/worst_q_mean": "min",
    "metrics/val_stitch/global/components/skeleton_cldice/mean": "max",
    "metrics/val_stitch/global/components/skeleton_cldice/worst_k_mean": "max",
    "metrics/val_stitch/global/components/skeleton_cldice/worst_q_mean": "max",
    "metrics/val_stitch/global/components/skeleton_chamfer/mean": "min",
    "metrics/val_stitch/global/components/skeleton_chamfer/worst_k_mean": "min",
    "metrics/val_stitch/global/components/skeleton_chamfer/worst_q_mean": "min",
    "metrics/val_stitch/segments/*/stability/dice_hard/mean": "max",
    "metrics/val_stitch/segments/*/stability/dice_soft/mean": "max",
    "metrics/val_stitch/segments/*/stability/accuracy/mean": "max",
    "metrics/val_stitch/segments/*/stability/pfm/mean": "max",
    "metrics/val_stitch/segments/*/stability/pfm_nonempty/mean": "max",
    "metrics/val_stitch/segments/*/stability/pfm_weighted/mean": "max",
    "metrics/val_stitch/segments/*/stability/voi/mean": "min",
    "metrics/val_stitch/segments/*/stability/mpm/mean": "min",
    "metrics/val_stitch/segments/*/stability/drd/mean": "min",
    "metrics/val_stitch/segments/*/stability/betti_l1/mean": "min",
    "metrics/val_stitch/segments/*/stability/betti_match_err/mean": "min",
    "metrics/val_stitch/segments/*/stability/betti_abs_beta0_err/mean": "min",
    "metrics/val_stitch/segments/*/stability/betti_abs_beta1_err/mean": "min",
    "metrics/val_stitch/segments/*/stability/abs_euler_err/mean": "min",
    "metrics/val_stitch/segments/*/stability/boundary_hd95/mean": "min",
    "metrics/val_stitch/segments/*/stability/skeleton_recall/mean": "max",
    "metrics/val_stitch/segments/*/stability/skeleton_cldice/mean": "max",
    "metrics/val_stitch/segments/*/stability/skeleton_chamfer/mean": "min",
}


def rewrite_val_stitch_metric_key(key: str) -> str:
    if key.startswith("stability/"):
        rest = key[len("stability/") :]
        for suffix in ("_mean", "_std", "_min", "_max"):
            if rest.endswith(suffix):
                metric_name = rest[: -len(suffix)]
                if not metric_name:
                    raise ValueError(f"invalid stability metric key: {key!r}")
                stat_name = suffix[1:]
                return f"stability/{metric_name}/{stat_name}"
        raise ValueError(f"unrecognized stability metric key: {key!r}")
    if key.startswith("thresholds/"):
        rest = key[len("thresholds/") :]
        metric_name, sep, threshold_tag = rest.rpartition("/")
        if not sep or not metric_name or not threshold_tag.startswith("thr_"):
            raise ValueError(f"unrecognized thresholds metric key: {key!r}")
        return key
    raise ValueError(f"unrecognized stitched metric key: {key!r}")
