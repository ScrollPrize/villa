import numpy as np
from pytorch_lightning.loggers import WandbLogger


def segment_group_index(*, model, stitch_group_idx_by_segment, segment_id_key):
    if segment_id_key not in stitch_group_idx_by_segment:
        raise KeyError(f"missing stitched group index for segment_id={segment_id_key!r}")
    segment_group_idx = int(stitch_group_idx_by_segment[segment_id_key])
    if segment_group_idx < 0 or segment_group_idx >= int(model.n_groups):
        raise ValueError(
            f"invalid stitched group index for segment_id={segment_id_key!r}: {segment_group_idx}"
        )
    return segment_group_idx


def collect_segment_stability_metrics(*, model, base_key, segment_group_idx, metrics, group_stability_values, rewrite_key):
    if not isinstance(metrics, dict):
        raise TypeError(
            f"compute_stitched_metrics must return a dict, got {type(metrics).__name__}"
        )

    for k, v in metrics.items():
        metric_key = rewrite_key(str(k))
        model.log(f"{base_key}/{metric_key}", v, on_epoch=True, prog_bar=False)
        prefix = "stability/"
        suffix = "/mean"
        if metric_key.startswith(prefix) and metric_key.endswith(suffix):
            metric_name = metric_key[len(prefix): -len(suffix)]
            if metric_name in group_stability_values:
                group_stability_values[metric_name][segment_group_idx].append(float(v))


def log_group_stability_metrics(*, model, stability_metric_directions, group_stability_values):
    if len(model.group_names) != int(model.n_groups):
        raise ValueError(
            f"group_names length must match n_groups: {len(model.group_names)} vs {int(model.n_groups)}"
        )

    for metric_name, higher_is_better in stability_metric_directions.items():
        present_group_means = []
        for group_idx in range(int(model.n_groups)):
            values = np.asarray(group_stability_values[metric_name][group_idx], dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue
            group_mean = float(finite_values.mean())
            present_group_means.append(group_mean)
            safe_group_name = str(model.group_names[group_idx]).replace("/", "_")
            model.log(
                f"metrics/val_stitch/groups/group_{group_idx}_{safe_group_name}/stability/{metric_name}/mean",
                group_mean,
                on_epoch=True,
                prog_bar=False,
            )
        if not present_group_means:
            continue
        worst_group_mean = min(present_group_means) if higher_is_better else max(present_group_means)
        model.log(
            f"metrics/val_stitch/worst_group/stability/{metric_name}/mean",
            float(worst_group_mean),
            on_epoch=True,
            prog_bar=False,
        )


def _group_component_rows_by_threshold(global_component_rows):
    global_rows_by_threshold = {}
    threshold_value_by_tag = {}
    for entry in global_component_rows:
        threshold_tag = entry.get("threshold_tag")
        if not isinstance(threshold_tag, str) or not threshold_tag:
            raise KeyError(
                "global component row is missing a valid threshold_tag; "
                f"entry keys={sorted(entry.keys())!r}"
            )

        threshold_value = float(entry["threshold"])
        existing_threshold_value = threshold_value_by_tag.get(threshold_tag)
        if existing_threshold_value is None:
            threshold_value_by_tag[threshold_tag] = threshold_value
        elif not np.isclose(float(existing_threshold_value), threshold_value):
            raise ValueError(
                f"inconsistent threshold values for threshold_tag={threshold_tag!r}: "
                f"{existing_threshold_value} vs {threshold_value}"
            )

        global_rows_by_threshold.setdefault(threshold_tag, []).append(entry)

    return global_rows_by_threshold, threshold_value_by_tag


def log_global_component_metrics(
    *,
    model,
    global_component_rows,
    threshold_tag_main,
    threshold,
    downsample,
    stitched_inputs_output_dir,
    component_worst_q,
    component_worst_k,
    enable_skeleton_metrics,
    run_topological_metrics,
    component_metric_specs,
    summarize_component_rows,
    write_global_component_manifest,
):
    global_rows_by_threshold, threshold_value_by_tag = _group_component_rows_by_threshold(global_component_rows)
    if threshold_tag_main not in global_rows_by_threshold:
        available_tags = sorted(global_rows_by_threshold.keys())
        raise ValueError(
            "eval_threshold is missing from stitched threshold rows; "
            f"eval_threshold={threshold} threshold_tag={threshold_tag_main!r} "
            f"available_threshold_tags={available_tags!r}"
        )

    global_metric_specs = component_metric_specs(
        enable_skeleton_metrics=enable_skeleton_metrics,
        include_cadenced_metrics=run_topological_metrics,
    )
    global_component_rows_main = global_rows_by_threshold[threshold_tag_main]
    global_stats, global_rankings = summarize_component_rows(
        global_component_rows_main,
        worst_q=component_worst_q,
        worst_k=component_worst_k,
        id_key="global_component_id",
        metric_specs=global_metric_specs,
    )

    for metric_name, stats in global_stats.items():
        for stat_name, stat_val in stats.items():
            model.log(
                f"metrics/val_stitch/global/components/{metric_name}/{stat_name}",
                float(stat_val),
                on_epoch=True,
                prog_bar=False,
            )

    sorted_threshold_tags = sorted(
        global_rows_by_threshold.keys(),
        key=lambda tag: (float(threshold_value_by_tag[tag]), str(tag)),
    )
    for threshold_tag in sorted_threshold_tags:
        threshold_rows = global_rows_by_threshold[threshold_tag]
        threshold_stats, _ = summarize_component_rows(
            threshold_rows,
            worst_q=component_worst_q,
            worst_k=component_worst_k,
            id_key="global_component_id",
            metric_specs=global_metric_specs,
        )
        for metric_name, stats in threshold_stats.items():
            for stat_name, stat_val in stats.items():
                model.log(
                    "metrics/val_stitch/global/thresholds/components/"
                    f"{metric_name}/{stat_name}/thr_{threshold_tag}",
                    float(stat_val),
                    on_epoch=True,
                    prog_bar=False,
                )

    manifest_path = None
    if stitched_inputs_output_dir:
        manifest_path = write_global_component_manifest(
            component_rows=global_component_rows_main,
            output_dir=stitched_inputs_output_dir,
            downsample=int(downsample),
            worst_k=component_worst_k,
            worst_q=component_worst_q,
            rankings=global_rankings,
        )

    if isinstance(model.logger, WandbLogger):
        run = model.logger.experiment
        if manifest_path is not None:
            run.summary["metrics/val_stitch/global/diagnostics/components/manifest_path"] = str(manifest_path)
        for metric_name, ranking in global_rankings.items():
            k_ids = ranking["worst_k_component_ids"]
            q_ids = ranking["worst_q_component_ids"]
            run.summary[
                f"metrics/val_stitch/global/diagnostics/components/{metric_name}/worst_k_component_ids"
            ] = ",".join(str(v) for v in k_ids)
            run.summary[
                f"metrics/val_stitch/global/diagnostics/components/{metric_name}/worst_q_component_ids"
            ] = ",".join(str(v) for v in q_ids)


