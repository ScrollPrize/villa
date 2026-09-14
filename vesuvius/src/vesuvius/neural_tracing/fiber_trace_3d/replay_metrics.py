"""Complete reference-distance partitions; no inference or traced-path changes."""
from __future__ import annotations

import copy
import math
from collections import defaultdict

METRIC_DEFINITION = "restart-episodes-80-base-voxels-v1"
RESTART_EPISODE_DISTANCE = 80.0
MEAN_DEFINITION = "combined-directed-distance-per-restarts-plus-one-v1"


def _episode_indices(arcs, distance):
    return [i for i, x in enumerate(arcs) if i == 0 or x - arcs[i-1] >= distance]


def reference_segments(case: dict, scale: float = 1.0, *, episode_distance: float = RESTART_EPISODE_DISTANCE) -> list[float]:
    """Partition [0,L] at directional failure arcs, retaining the terminal span.

    Reset advances belong to reference-distance accounting, not to credited
    successful tracing. Suppress events less than80 base voxels after the
    preceding actual event; never suppress the first event of a directed run.
    Directional arcs already increase from the start even for reverse traces.
    """
    length = float(case["reference_length_base_voxels"])
    if not math.isfinite(length) or length <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("reference length and scale must be finite and positive")
    evaluated = float(case["evaluated_through_base_voxels"])
    if not case["evaluation_complete"] or not math.isfinite(evaluated) or evaluated < length - 1e-7:
        raise ValueError("complete directional evaluation required")
    if not math.isfinite(episode_distance) or episode_distance < 0:
        raise ValueError("episode distance must be finite and nonnegative")
    events = [float(f["directional_reference_arc_base_voxels"]) for f in case["failures"]]
    arcs = [0.0] + events + [length]
    if any(not math.isfinite(x) for x in arcs) or any(b < a for a, b in zip(arcs, arcs[1:])):
        raise ValueError("failure arcs must be finite, ordered, and within the reference")
    retained = [events[i] for i in _episode_indices(events, episode_distance)]
    arcs = [0.0] + retained + [length]
    return [(b - a) * scale for a, b in zip(arcs, arcs[1:])]


def segment_statistics(segments: list[float], directions: int) -> dict:
    values = sorted(segments)
    count = len(values)
    if directions < 0 or count < directions or (count and not directions):
        raise ValueError("invalid segment/direction count")
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("invalid segment length")

    def quantile(q):
        if not values:
            return 0.0
        index = (count - 1) * q
        lo = int(index)
        return values[lo] + (values[min(lo + 1, count - 1)] - values[lo]) * (index - lo)

    length = math.fsum(values)
    denominator = count - directions + 1 if count else 0
    return dict(metric_definition=METRIC_DEFINITION, directions=directions,
                failures=count - directions, reliability_segments=count,
                mean_definition=MEAN_DEFINITION, mean_denominator=denominator,
                directed_reference_mm=length,
                mean_segment_length_mm=length / denominator if denominator else 0.0,
                median_segment_length_mm=quantile(0.5),
                p05_segment_length_mm=quantile(0.05),
                p95_segment_length_mm=quantile(0.95),
                zero_length_segments=sum(v == 0 for v in values))


def complete_report(summary: dict, *, episode_distance: float = RESTART_EPISODE_DISTANCE) -> dict:
    if not summary["completed"]:
        raise ValueError("incomplete replay set")
    scrolls, collections = defaultdict(list), {}
    for source in summary["sources"]:
        scale = source["summary"]["base_voxel_size_um"] / 1000
        rows = [reference_segments(c, scale, episode_distance=episode_distance) for c in source["cases"]]
        collections[source["source_id"]] = rows
        scrolls[source["backend_config"].get("coordinate_space", source["source_id"]).split("/")[0]].extend(rows)

    def aggregate(rows):
        value = segment_statistics([v for row in rows for v in row], len(rows))
        value["metric_definition"] = METRIC_DEFINITION if episode_distance == RESTART_EPISODE_DISTANCE else "complete-reference-partition-v1" if episode_distance == 0 else "restart-episodes-custom-v1"
        return value

    total = aggregate([row for rows in collections.values() for row in rows])
    if total["directions"] != summary["cases"]:
        raise ValueError("case/failure totals disagree with saved summary")
    cases = [case for source in summary["sources"] for case in source["cases"]]
    expected_failures = (sum(len(reference_segments(case))-1 for case in cases)
        if summary.get("metric_definition") == METRIC_DEFINITION
        else sum(len(case["failures"]) for case in cases))
    if expected_failures != summary["failures"]:
        raise ValueError("failure total disagrees with saved events and metric definition")
    if not math.isclose(total["directed_reference_mm"], summary["directed_reference_length_mm"], rel_tol=1e-9):
        raise ValueError("reference length disagrees with saved summary")
    total["directed_runs"] = total["directions"]
    return dict(metric_definition=total["metric_definition"],
                mean_definition=MEAN_DEFINITION,
                restart_episode_distance_base_voxels=episode_distance,
                reference_selection_policy=summary.get("reference_selection_policy", "whole_explicit_fibers_no_crop"),
                global_result=total,
                scrolls={k: aggregate(v) for k, v in scrolls.items()},
                collections={k: aggregate(v) for k, v in collections.items()})


def correct_summary(summary: dict) -> dict:
    """Return corrected metadata; input cases and raw failure data stay intact."""
    report = complete_report(summary)
    result = copy.deepcopy(summary)
    result.update(version=2, metric_definition=METRIC_DEFINITION)
    total = report["global_result"]
    result["restart_episode_distance_base_voxels"] = RESTART_EPISODE_DISTANCE
    result["failures"] = total["failures"]
    result["failures_per_directed_mm"] = total["failures"] / total["directed_reference_mm"] if total["directed_reference_mm"] else None
    result["mean_definition"] = MEAN_DEFINITION
    result["mean_denominator"] = total["mean_denominator"]
    result["mean_segment_length_percent"] = 100.0 / total["mean_denominator"] if total["mean_denominator"] else 0.0
    for key in ("reliability_segments", "mean_segment_length_mm", "median_segment_length_mm",
                "p05_segment_length_mm", "p95_segment_length_mm", "zero_length_segments"):
        result[key] = total[key]
    if not total["directions"]:
        result["mean_segment_length_mm"] = None
    for source in result["sources"]:
        source["version"] = 5
        source["metric_definition"] = METRIC_DEFINITION
        stats = source["summary"]
        new = report["collections"][source["source_id"]]
        count = new["reliability_segments"]
        denominator = new["mean_denominator"]
        for key in ("failures", "total_failures"):
            if key in stats:
                stats[key] = new["failures"]
        if "failures_per_directed_mm" in stats:
            stats["failures_per_directed_mm"] = new["failures"] / new["directed_reference_mm"] if new["directed_reference_mm"] else None
        for case in source["cases"]:
            values = reference_segments(case, stats["base_voxel_size_um"] / 1000)
            arcs = [float(f["directional_reference_arc_base_voxels"]) for f in case["failures"]]
            starts = _episode_indices(arcs, RESTART_EPISODE_DISTANCE)
            case["restart_episodes"] = [{"first_event_index": a, "last_event_index": b-1,
                "begin_arc_base_voxels": arcs[a], "end_arc_base_voxels": arcs[b-1],
                "span_base_voxels": arcs[b-1]-arcs[a]} for a, b in zip(starts, starts[1:]+[len(arcs)])]
            case.update(restart_count=len(values)-1, reliability_segments=len(values),
                        mean_segment_length_mm=math.fsum(values)/len(values))
        # Legacy failed/seeded-span means count retry events; they are not the
        # reported episode partition and must not survive as competing metrics.
        for key in list(stats):
            if key.startswith(("mean_failed_span_", "mean_seeded_span_")) or key == "seeded_spans":
                del stats[key]
        reasons = defaultdict(int)
        for case in source["cases"]:
            for episode in case["restart_episodes"]:
                event = case["failures"][episode["first_event_index"]]
                reasons[event.get("reason", "unknown")] += 1
        if "failure_reasons" in stats:
            stats["failure_reasons"] = dict(reasons)
        stats.update(metric_definition=METRIC_DEFINITION, reliability_segments=count,
                     mean_definition=MEAN_DEFINITION, mean_denominator=denominator,
                     mean_segment_length_base_voxels=stats["directed_reference_length_base_voxels"] / denominator if denominator else 0.0,
                     mean_segment_length_percent=100.0 / denominator if denominator else 0.0)
        for key in ("mean_segment_length_mm", "median_segment_length_mm", "p05_segment_length_mm",
                    "p95_segment_length_mm", "zero_length_segments"):
            stats[key] = new[key]
        if not new["directions"]:
            stats["mean_segment_length_mm"] = None
    return result
