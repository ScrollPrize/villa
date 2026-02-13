from __future__ import annotations

from typing import Dict

WANDB_VAL_STITCH_METRIC_SUMMARIES: Dict[str, str] = {
    "metrics/val_stitch/segments/*/summary/dice_hard": "max",
    "metrics/val_stitch/segments/*/summary/dice_soft": "max",
    "metrics/val_stitch/segments/*/summary/drd": "min",
    "metrics/val_stitch/segments/*/summary/mpm": "min",
    "metrics/val_stitch/segments/*/summary/voi": "min",
    "metrics/val_stitch/segments/*/summary/betti/l1_betti_err": "min",
    "metrics/val_stitch/segments/*/summary/abs_euler_err": "min",
    "metrics/val_stitch/segments/*/summary/boundary/hd95": "min",
    "metrics/val_stitch/segments/*/summary/skeleton/cldice": "max",
    "metrics/val_stitch/segments/*/summary/skeleton/chamfer": "min",
    "metrics/val_stitch/segments/*/summary/pfm": "max",
    "metrics/val_stitch/segments/*/summary/pfm_nonempty": "max",
    "metrics/val_stitch/segments/*/summary/pfm_weighted": "max",
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
    "metrics/val_stitch/segments/*/stability/pfm/mean": "max",
    "metrics/val_stitch/segments/*/stability/pfm_nonempty/mean": "max",
    "metrics/val_stitch/segments/*/stability/pfm_weighted/mean": "max",
    "metrics/val_stitch/segments/*/stability/voi/mean": "min",
    "metrics/val_stitch/segments/*/stability/betti_l1/mean": "min",
}


def rewrite_val_stitch_metric_key(key: str) -> str:
    if key.startswith("components/"):
        rest = key[len("components/") :]
        if rest == "n_pred":
            return "components/n_pred"
        for suffix in ("_worst_k_mean", "_worst_q_mean", "_n_gt", "_mean", "_min", "_max"):
            if rest.endswith(suffix):
                metric_name = rest[: -len(suffix)]
                if not metric_name:
                    raise ValueError(f"invalid components metric key: {key!r}")
                stat_name = suffix[1:]
                return f"components/{metric_name}/{stat_name}"
        raise ValueError(f"unrecognized components metric key: {key!r}")
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
        metric_name, sep, threshold_tag = rest.partition("/")
        if not sep or not metric_name or not threshold_tag.startswith("thr_"):
            raise ValueError(f"unrecognized thresholds metric key: {key!r}")
        return key
    return f"summary/{key}"
