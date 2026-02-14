from __future__ import annotations

from typing import Dict

from metrics.stitched_metric_specs import component_metric_specs, component_metric_supports_worst


def get_wandb_val_stitch_metric_summaries(*, enable_skeleton_metrics: bool) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for metric_name, higher_is_better in component_metric_specs(enable_skeleton_metrics=enable_skeleton_metrics):
        summary_mode = "max" if bool(higher_is_better) else "min"
        stat_names = ("mean",)
        if component_metric_supports_worst(metric_name=metric_name):
            stat_names = ("mean", "worst_k_mean", "worst_q_mean")
        for stat_name in stat_names:
            out[f"metrics/val_stitch/global/components/{metric_name}/{stat_name}"] = summary_mode
        out[f"metrics/val_stitch/segments/*/stability/{metric_name}/mean"] = summary_mode
    return out


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
