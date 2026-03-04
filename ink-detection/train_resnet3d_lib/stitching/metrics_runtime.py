import numpy as np

from train_resnet3d_lib.config import CFG
from train_resnet3d_lib.stitching.metrics_runtime_aggregation import (
    collect_segment_stability_metrics,
    log_global_component_metrics,
    log_group_stability_metrics,
    segment_group_index,
)
from train_resnet3d_lib.stitching.metrics_runtime_config import (
    resolve_metrics_options,
    should_run_stitch_metrics,
)


def log_stitched_validation_metrics(
    *,
    model,
    sanity_checking: bool,
    segment_to_val: dict[str, tuple[np.ndarray, np.ndarray]],
    segment_to_val_meta: dict[str, dict],
    downsample: int,
) -> bool:
    if bool(sanity_checking):
        return False
    if not (model.trainer is None or model.trainer.is_global_zero):
        return False

    current_epoch = int(getattr(getattr(model, "trainer", None), "current_epoch", 0))
    eval_epoch = int(current_epoch + 1)
    total_epochs = int(getattr(CFG, "epochs"))
    is_final_eval_epoch = bool(eval_epoch == total_epochs)

    if not should_run_stitch_metrics(
        current_epoch=current_epoch,
        eval_epoch=eval_epoch,
        segment_to_val=segment_to_val,
    ):
        return False

    from metrics.stitched_metrics import (
        _threshold_tag,
        component_metric_specs,
        compute_stitched_metrics,
        summarize_component_rows,
        write_global_component_manifest,
    )
    from train_resnet3d_lib.val_stitch_wandb import rewrite_val_stitch_metric_key

    options = resolve_metrics_options(
        current_epoch=current_epoch,
        eval_epoch=eval_epoch,
        is_final_eval_epoch=is_final_eval_epoch,
    )
    threshold_tag_main = _threshold_tag(options["threshold"])

    stitch_group_idx_by_segment = getattr(model, "_stitch_group_idx_by_segment", None)
    if not isinstance(stitch_group_idx_by_segment, dict):
        raise TypeError(
            "model._stitch_group_idx_by_segment must be a dict mapping segment id to group index"
        )

    stability_metric_directions = {
        str(metric_name): bool(higher_is_better)
        for metric_name, higher_is_better in component_metric_specs(
            enable_skeleton_metrics=bool(options["enable_skeleton_metrics"]),
            include_cadenced_metrics=bool(options["run_topological_metrics"]),
        )
    }
    group_stability_values = {
        metric_name: {int(group_idx): [] for group_idx in range(int(model.n_groups))}
        for metric_name in stability_metric_directions
    }

    global_component_rows = []
    for segment_id, (pred_prob, pred_has) in segment_to_val.items():
        segment_id_key = str(segment_id)
        segment_group_idx = segment_group_index(
            model=model,
            stitch_group_idx_by_segment=stitch_group_idx_by_segment,
            segment_id_key=segment_id_key,
        )

        meta = segment_to_val_meta.get(segment_id_key, {})
        roi_offset = meta.get("offset", (0, 0))
        cache_max = int(max(1, len(segment_to_val))) if segment_to_val else 1
        metrics = compute_stitched_metrics(
            fragment_id=segment_id,
            pred_prob=pred_prob,
            pred_has=pred_has,
            label_suffix=options["label_suffix"],
            mask_suffix=options["mask_suffix"],
            downsample=int(downsample),
            roi_offset=roi_offset,
            threshold=float(options["threshold"]),
            betti_connectivity=int(options["betti_connectivity"]),
            drd_block_size=int(options["drd_block_size"]),
            boundary_k=int(options["boundary_k"]),
            boundary_tols=options["boundary_tols"],
            component_worst_q=options["component_worst_q"],
            component_worst_k=options["component_worst_k"],
            component_min_area=int(options["component_min_area"]),
            component_pad=int(options["component_pad"]),
            skeleton_method=options["skeleton_thinning_type"],
            enable_skeleton_metrics=bool(options["enable_skeleton_metrics"]),
            include_cadenced_metrics=bool(options["run_topological_metrics"]),
            enable_full_region_metrics=bool(options["enable_full_region_metrics"]),
            threshold_grid=options["threshold_grid"],
            stitched_inputs_output_dir=options["stitched_inputs_output_dir"],
            gt_cache_max=cache_max,
            component_rows_collector=global_component_rows,
            eval_epoch=eval_epoch,
        )

        safe_segment_id = segment_id_key.replace("/", "_")
        base_key = f"metrics/val_stitch/segments/{safe_segment_id}"
        collect_segment_stability_metrics(
            model=model,
            base_key=base_key,
            segment_group_idx=segment_group_idx,
            metrics=metrics,
            group_stability_values=group_stability_values,
            rewrite_key=rewrite_val_stitch_metric_key,
        )

    log_group_stability_metrics(
        model=model,
        stability_metric_directions=stability_metric_directions,
        group_stability_values=group_stability_values,
    )

    if global_component_rows:
        log_global_component_metrics(
            model=model,
            global_component_rows=global_component_rows,
            threshold_tag_main=threshold_tag_main,
            threshold=float(options["threshold"]),
            downsample=int(downsample),
            stitched_inputs_output_dir=options["stitched_inputs_output_dir"],
            component_worst_q=options["component_worst_q"],
            component_worst_k=options["component_worst_k"],
            enable_skeleton_metrics=bool(options["enable_skeleton_metrics"]),
            run_topological_metrics=bool(options["run_topological_metrics"]),
            component_metric_specs=component_metric_specs,
            summarize_component_rows=summarize_component_rows,
            write_global_component_manifest=write_global_component_manifest,
        )

    return True


__all__ = ["log_stitched_validation_metrics"]
