import os.path as osp

import numpy as np

from train_resnet3d_lib.config import CFG, log


def _parse_list(value, cast_fn):
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
        return [cast_fn(p) for p in parts if p]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [cast_fn(v) for v in value]
    return [cast_fn(value)]


def _resolve_component_worst_q():
    component_worst_q = getattr(CFG, "eval_component_worst_q", 0.2)
    if isinstance(component_worst_q, str) and component_worst_q.strip().lower() in {"", "none", "null"}:
        component_worst_q = None
    if component_worst_q is not None:
        component_worst_q = float(component_worst_q)
    return component_worst_q


def _resolve_component_worst_k():
    component_worst_k = getattr(CFG, "eval_component_worst_k", 2)
    if isinstance(component_worst_k, str) and component_worst_k.strip().lower() in {"", "none", "null"}:
        component_worst_k = None
    if isinstance(component_worst_k, float) and float(component_worst_k).is_integer():
        component_worst_k = int(component_worst_k)
    return component_worst_k


def _resolve_threshold_grid():
    threshold_grid = _parse_list(getattr(CFG, "eval_threshold_grid", None), float)
    if threshold_grid is not None:
        return threshold_grid

    tmin = float(getattr(CFG, "eval_threshold_grid_min", 0.40))
    tmax = float(getattr(CFG, "eval_threshold_grid_max", 0.70))
    steps = int(getattr(CFG, "eval_threshold_grid_steps", 5))
    if steps >= 2:
        return np.linspace(tmin, tmax, steps).tolist()
    return None


def should_run_stitch_metrics(*, current_epoch, eval_epoch, segment_to_val):
    should_run = bool(getattr(CFG, "eval_stitch_metrics", True)) and bool(segment_to_val)
    if not should_run:
        return False

    stitch_every_n_epochs = max(1, int(getattr(CFG, "eval_stitch_every_n_epochs", 1)))
    stitch_plus_one = bool(getattr(CFG, "eval_stitch_every_n_epochs_plus_one", False))
    if stitch_plus_one and stitch_every_n_epochs > 1:
        mod = eval_epoch % stitch_every_n_epochs
        should_run_epoch = (eval_epoch >= stitch_every_n_epochs) and (mod == 0 or mod == 1)
    else:
        should_run_epoch = (eval_epoch % stitch_every_n_epochs) == 0

    if should_run_epoch:
        return True

    log(
        f"skip stitched metrics epoch={current_epoch} "
        f"eval_stitch_every_n_epochs={stitch_every_n_epochs} "
        f"eval_stitch_every_n_epochs_plus_one={stitch_plus_one}"
    )
    return False


def _resolve_topology_schedule(*, current_epoch, eval_epoch, is_final_eval_epoch):
    save_stitch_debug_images = bool(getattr(CFG, "eval_save_stitch_debug_images", False))

    eval_topological_metrics_every_n_epochs = int(
        getattr(CFG, "eval_topological_metrics_every_n_epochs", 1)
    )
    if eval_topological_metrics_every_n_epochs < 1:
        raise ValueError(
            "eval_topological_metrics_every_n_epochs must be >= 1, "
            f"got {eval_topological_metrics_every_n_epochs}"
        )

    eval_save_stitch_debug_images_every_n_epochs = int(
        getattr(CFG, "eval_save_stitch_debug_images_every_n_epochs", 1)
    )
    if eval_save_stitch_debug_images_every_n_epochs < 1:
        raise ValueError(
            "eval_save_stitch_debug_images_every_n_epochs must be >= 1, "
            f"got {eval_save_stitch_debug_images_every_n_epochs}"
        )

    run_topological_metrics = bool(
        (eval_epoch % eval_topological_metrics_every_n_epochs) == 0 or is_final_eval_epoch
    )
    save_stitch_debug_images_now = bool(
        save_stitch_debug_images
        and (
            (eval_epoch % eval_save_stitch_debug_images_every_n_epochs) == 0
            or is_final_eval_epoch
        )
    )

    stitched_inputs_output_dir = osp.join(
        str(getattr(CFG, "figures_dir", ".")),
        "metrics_stitched_debug",
    )
    if not save_stitch_debug_images_now:
        stitched_inputs_output_dir = None

    if not run_topological_metrics:
        log(
            f"skip topological stitched metrics epoch={current_epoch} "
            f"eval_epoch={eval_epoch} "
            f"eval_topological_metrics_every_n_epochs={eval_topological_metrics_every_n_epochs}"
        )
    if save_stitch_debug_images and (not save_stitch_debug_images_now):
        log(
            f"skip stitched debug inputs epoch={current_epoch} "
            f"eval_epoch={eval_epoch} "
            "eval_save_stitch_debug_images_every_n_epochs="
            f"{eval_save_stitch_debug_images_every_n_epochs}"
        )

    return {
        "run_topological_metrics": run_topological_metrics,
        "stitched_inputs_output_dir": stitched_inputs_output_dir,
    }


def resolve_metrics_options(*, current_epoch, eval_epoch, is_final_eval_epoch):
    topology_schedule = _resolve_topology_schedule(
        current_epoch=current_epoch,
        eval_epoch=eval_epoch,
        is_final_eval_epoch=is_final_eval_epoch,
    )

    return {
        "label_suffix": str(getattr(CFG, "val_label_suffix", "_val")),
        "mask_suffix": str(getattr(CFG, "val_mask_suffix", "_val")),
        "threshold": float(getattr(CFG, "eval_threshold", 0.5)),
        "betti_connectivity": 2,
        "drd_block_size": int(getattr(CFG, "eval_drd_block_size", 8)),
        "boundary_k": int(getattr(CFG, "eval_boundary_k", 3)),
        "boundary_tols": _parse_list(getattr(CFG, "eval_boundary_tols", None), float),
        "component_worst_q": _resolve_component_worst_q(),
        "component_worst_k": _resolve_component_worst_k(),
        "skeleton_thinning_type": str(getattr(CFG, "eval_skeleton_thinning_type", "guo_hall")),
        "enable_skeleton_metrics": bool(getattr(CFG, "eval_enable_skeleton_metrics", True)),
        "component_min_area": int(getattr(CFG, "eval_component_min_area", 0) or 0),
        "component_pad": int(getattr(CFG, "eval_component_pad", 5)),
        "enable_full_region_metrics": bool(getattr(CFG, "eval_stitch_full_region_metrics", False)),
        "run_topological_metrics": bool(topology_schedule["run_topological_metrics"]),
        "threshold_grid": _resolve_threshold_grid(),
        "stitched_inputs_output_dir": topology_schedule["stitched_inputs_output_dir"],
    }


__all__ = [
    "should_run_stitch_metrics",
    "resolve_metrics_options",
]
