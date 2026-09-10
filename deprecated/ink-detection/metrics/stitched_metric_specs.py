from __future__ import annotations

from typing import Set, Tuple


COMPONENT_METRIC_SPECS_ALWAYS: Tuple[Tuple[str, bool], ...] = (
    ("dice_hard", True),
    ("dice_soft", True),
    ("accuracy", True),
    ("mpm", False),
    ("drd", False),
)

COMPONENT_METRIC_SPECS_CADENCED: Tuple[Tuple[str, bool], ...] = (
    ("voi", False),
    ("betti_l1", False),
    ("betti_match_err", False),
    ("betti_match_err_dim0", False),
    ("betti_match_err_dim1", False),
    ("betti_abs_beta0_err", False),
    ("betti_abs_beta1_err", False),
    ("betti_beta0_pred", False),
    ("betti_beta1_pred", False),
    ("betti_beta0_gt", False),
    ("betti_beta1_gt", False),
    ("abs_euler_err", False),
    ("boundary_hd95", False),
)

COMPONENT_METRIC_SPECS_SKELETON: Tuple[Tuple[str, bool], ...] = (
    ("pfm_weighted", True),
    ("pfm", True),
    ("pfm_nonempty", True),
    ("skeleton_recall", True),
)


def component_metric_specs(
    *,
    enable_skeleton_metrics: bool,
    include_cadenced_metrics: bool = True,
) -> Tuple[Tuple[str, bool], ...]:
    specs = COMPONENT_METRIC_SPECS_ALWAYS
    if bool(include_cadenced_metrics):
        specs = specs + COMPONENT_METRIC_SPECS_CADENCED
    if bool(enable_skeleton_metrics):
        specs = specs + COMPONENT_METRIC_SPECS_SKELETON
    return specs


COMPONENT_DESCRIPTOR_METRICS: Tuple[str, ...] = (
    "betti_beta0_pred",
    "betti_beta1_pred",
    "betti_beta0_gt",
    "betti_beta1_gt",
)

_COMPONENT_DESCRIPTOR_METRIC_SET: Set[str] = set(COMPONENT_DESCRIPTOR_METRICS)


def component_metric_supports_worst(*, metric_name: str) -> bool:
    return str(metric_name) not in _COMPONENT_DESCRIPTOR_METRIC_SET
