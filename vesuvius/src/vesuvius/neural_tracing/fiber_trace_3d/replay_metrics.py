"""Complete reference-distance partitions; no inference or traced-path changes."""
from __future__ import annotations

import copy
import math
from collections import defaultdict

METRIC_DEFINITION = "complete-reference-partition-v1"


def reference_segments(case: dict, scale: float = 1.0) -> list[float]:
    """Partition [0,L] at directional failure arcs, retaining the terminal span.

    Reset advances belong to this reference-distance accounting, not to credited
    successful tracing. Repeated/endpoint failures intentionally create zeros.
    Directional arcs already increase from the start even for reverse traces.
    """
    length = float(case["reference_length_base_voxels"])
    if not math.isfinite(length) or length <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("reference length and scale must be finite and positive")
    evaluated = float(case["evaluated_through_base_voxels"])
    if not case["evaluation_complete"] or not math.isfinite(evaluated) or evaluated < length - 1e-7:
        raise ValueError("complete directional evaluation required")
    arcs = [0.0] + [float(f["directional_reference_arc_base_voxels"]) for f in case["failures"]] + [length]
    if any(not math.isfinite(x) for x in arcs) or any(b < a for a, b in zip(arcs, arcs[1:])):
        raise ValueError("failure arcs must be finite, ordered, and within the reference")
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
    return dict(metric_definition=METRIC_DEFINITION, directions=directions,
                failures=count - directions, reliability_segments=count,
                directed_reference_mm=length,
                mean_segment_length_mm=length / count if count else 0.0,
                median_segment_length_mm=quantile(0.5),
                p05_segment_length_mm=quantile(0.05),
                p95_segment_length_mm=quantile(0.95),
                zero_length_segments=sum(v == 0 for v in values))


def complete_report(summary: dict) -> dict:
    if not summary["completed"]:
        raise ValueError("incomplete replay set")
    scrolls, collections = defaultdict(list), {}
    for source in summary["sources"]:
        scale = source["summary"]["base_voxel_size_um"] / 1000
        rows = [reference_segments(c, scale) for c in source["cases"]]
        collections[source["source_id"]] = rows
        scrolls[source["backend_config"]["coordinate_space"].split("/")[0]].extend(rows)

    def aggregate(rows):
        return segment_statistics([v for row in rows for v in row], len(rows))

    total = aggregate([row for rows in collections.values() for row in rows])
    if total["directions"] != summary["cases"] or total["failures"] != summary["failures"]:
        raise ValueError("case/failure totals disagree with saved summary")
    if not math.isclose(total["directed_reference_mm"], summary["directed_reference_length_mm"], rel_tol=1e-9):
        raise ValueError("reference length disagrees with saved summary")
    return dict(metric_definition=METRIC_DEFINITION, global_result=total,
                scrolls={k: aggregate(v) for k, v in scrolls.items()},
                collections={k: aggregate(v) for k, v in collections.items()})


def correct_summary(summary: dict) -> dict:
    """Return corrected metadata; input cases and raw failure data stay intact."""
    report = complete_report(summary)
    result = copy.deepcopy(summary)
    result.update(version=2, metric_definition=METRIC_DEFINITION)
    total = report["global_result"]
    result["mean_segment_length_percent"] = 100.0 / total["reliability_segments"] if total["reliability_segments"] else 0.0
    for key in ("reliability_segments", "mean_segment_length_mm", "median_segment_length_mm",
                "p05_segment_length_mm", "p95_segment_length_mm", "zero_length_segments"):
        result[key] = total[key]
    for source in result["sources"]:
        source["version"] = 5
        source["metric_definition"] = METRIC_DEFINITION
        stats = source["summary"]
        new = report["collections"][source["source_id"]]
        count = new["reliability_segments"]
        stats.update(metric_definition=METRIC_DEFINITION, reliability_segments=count,
                     mean_segment_length_base_voxels=stats["directed_reference_length_base_voxels"] / count if count else 0.0,
                     mean_segment_length_percent=100.0 / count if count else 0.0)
        for key in ("mean_segment_length_mm", "median_segment_length_mm", "p05_segment_length_mm",
                    "p95_segment_length_mm", "zero_length_segments"):
            stats[key] = new[key]
    return result
