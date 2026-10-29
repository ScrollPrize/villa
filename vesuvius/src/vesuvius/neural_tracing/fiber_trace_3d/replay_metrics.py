"""Complete reference-distance partitions; no inference or traced-path changes."""
from __future__ import annotations

import copy
import math
from collections import defaultdict

METRIC_DEFINITION = "restart-episodes-80-base-voxels-v1"
RESTART_EPISODE_DISTANCE = 80.0
MEAN_DEFINITION = "combined-directed-distance-per-restarts-plus-one-v1"


def filter_output_segments(document: dict, minimum_base_voxels: float, *, confidence_reasons=()) -> dict:
    """Output-only pruning; never alter tracing/reseeding or saved input data.

    Gate on traced polyline arc, account coverage on GT arc as the replay does.
    GT error stops already define some boundaries: not a GT-free filter score.
    """
    import numpy as np
    if not math.isfinite(minimum_base_voxels) or minimum_base_voxels < 0:
        raise ValueError('minimum length must be finite and nonnegative')
    replay = document['replay']
    scale = float(document['provenance']['base_voxel_size_um']) / 1000
    begin, end = replay['reference_begin_arc_base'], replay['reference_end_arc_base']
    if not (scale > 0 and end > begin and replay['completed_reference_arc_base'] >= end - 1e-7):
        raise ValueError('complete valid replay required')
    segments = replay['segments']
    failures = replay['failures']
    failure_indices = [int(f['segment_index']) for f in failures]
    if len(set(failure_indices)) != len(failure_indices) or any(i < 0 or i >= len(segments) for i in failure_indices):
        raise ValueError('invalid failure-to-segment mapping')
    confidence_failures = [f for f in failures if segments[int(f['segment_index'])]['termination_reason'] in confidence_reasons]
    failures = [f for f in failures if segments[int(f['segment_index'])]['termination_reason'] not in confidence_reasons]
    failure_indices = [int(f['segment_index']) for f in failures]
    records = []
    for i, s in enumerate(segments):
        points = np.asarray(s['trace_points_base_xyz'], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('invalid traced polyline')
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        covered = max(0., s['end_reference_arc_base'] - s['matches'][0]['search_begin_arc_base']) if s['matches'] else 0.
        reason = s['termination_reason']
        records.append(dict(segment_index=i, trace_length_base=length, trace_length_mm=length*scale,
            reference_mm=covered*scale, keep=length >= minimum_base_voxels,
            error=i in failure_indices, presence_stop=reason=='presence_filtered',
            reverse_stop=reason=='reverse_consistency_filtered', confidence_stop=reason in confidence_reasons))
    if sum(r['presence_stop']+r['reverse_stop'] for r in records) != len(replay['filtered_stops']):
        raise ValueError('filtered-stop mapping mismatch')
    arcs = [float(f['reference_arc_base']) for f in failures if records[int(f['segment_index'])]['keep']]
    if any(b<a for a,b in zip(arcs,arcs[1:])):
        raise ValueError('failure arcs must be ordered')
    return dict(source=document['provenance']['source_id'],
        case_index=document['provenance']['case_index'], minimum_base_voxels=minimum_base_voxels,
        reference_mm=(end-begin)*scale, records=records,
        ordinary_restarts=len(_episode_indices(arcs, RESTART_EPISODE_DISTANCE)),
        original_error_events=len(failures), original_filtered_stops=len(replay['filtered_stops'])+len(confidence_failures))


def filtered_output_statistics(cases: list[dict]) -> dict:
    """Aggregate surviving output pieces, exposing discarded coverage and events."""
    import numpy as np
    records = [r for c in cases for r in c['records']]
    kept = [r for r in records if r['keep']]
    discarded = [r for r in records if not r['keep']]
    retained = math.fsum(r['reference_mm'] for r in kept)
    reference = math.fsum(c['reference_mm'] for c in cases)
    error_events = sum(r['error'] for r in kept)
    presence = sum(r['presence_stop'] for r in kept)
    reverse = sum(r['reverse_stop'] for r in kept)
    confidence = sum(r.get('confidence_stop', False) for r in kept)
    stops = error_events + presence + reverse + confidence
    quantiles = np.quantile([r['reference_mm'] for r in kept], [.05,.5,.95]).tolist() if kept else [None]*3
    return dict(directions=len(cases), ordinary_restarts=sum(c['ordinary_restarts'] for c in cases),
        error_events=error_events, presence_filtered=presence, reverse_filtered=reverse,
        confidence_filtered=confidence, filtered_stops=presence+reverse+confidence, actual_stops=stops,
        original_actual_stops=sum(c['original_error_events']+c['original_filtered_stops'] for c in cases),
        kept_segments=len(kept), discarded_segments=len(discarded),
        discarded_error_events=sum(r['error'] for r in discarded),
        reference_mm=reference, retained_mm=retained,
        discarded_mm=math.fsum(r['reference_mm'] for r in discarded),
        retained_percent=100*retained/reference if reference else 0,
        mean_mm=retained/(stops+1) if kept else None,
        piece_mean_mm=retained/len(kept) if kept else None,
        trace_piece_mean_base=math.fsum(r['trace_length_base'] for r in kept)/len(kept) if kept else None,
        trace_piece_mean_mm=math.fsum(r.get('trace_length_mm', 0.) for r in kept)/len(kept) if kept else None,
        p05_mm=quantiles[0], median_mm=quantiles[1], p95_mm=quantiles[2])


def training_retained_metrics(documents: list[dict]) -> dict[str, float]:
    """Shared directed replay accounting, including both directions in numerator
    and denominator. This is not the union of forward/reverse accepted spans.
    Empty output has zero lengths and an explicit zero piece count for TB.
    """
    result = {}
    for minimum, prefix in ((0, 'retained'), (512, 'retained_min512')):
        cases = [filter_output_segments(doc, minimum) for doc in documents]
        groups = {'': cases}
        for case in cases:
            groups.setdefault('/'+case['source'], []).append(case)
        for suffix, group in groups.items():
            values = filtered_output_statistics(group)
            for key in ('retained_percent', 'retained_mm', 'kept_segments',
                        'piece_mean_mm', 'trace_piece_mean_base', 'trace_piece_mean_mm',
                        'reverse_filtered', 'presence_filtered'):
                metric = key.removeprefix('retained_')
                result[f'{prefix}_{metric}{suffix}'] = values[key] if values[key] is not None else 0.
    return result


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
    evaluated_cases = summary.get("evaluated_cases", summary["cases"])
    if total["directions"] != evaluated_cases:
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


def split_report(summary: dict, fiber_splits: dict) -> dict:
    """Apply the same complete episode metrics to predefined whole-fiber groups."""
    groups = defaultdict(list)
    for source in summary['sources']:
        for case in source['cases']:
            fiber = source['backend_config']['fibers'][case['source_index']]
            split = fiber_splits[fiber]['split']
            groups[split].append(reference_segments(case, source['summary']['base_voxel_size_um']/1000))
    return {key: segment_statistics([value for row in rows for value in row], len(rows))
            for key, rows in groups.items()}


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
    if any(case.get('reverse_checks', 0) for source in result['sources'] for case in source['cases']):
        result['consistency'] = consistency_report(result)
    return result


def consistency_report(summary):
    """Separate GT failures from model consistency rejection; never credit gaps."""
    rows = {}
    for source in summary['sources']:
        scale = source['summary']['base_voxel_size_um'] / 1000
        cases = source['cases']
        skipped = 0.0
        for case in cases:
            last = 0.0
            length = case['reference_length_base_voxels']
            for begin, detected, end in case.get('filtered_stops', []):
                if not (last <= begin <= detected <= end <= length + 1e-7):
                    raise ValueError('invalid/overlapping filtered interval')
                skipped += (end - begin) * scale
                last = end
        reference = sum(c['reference_length_base_voxels'] for c in cases) * scale
        retained = sum(c['traced_length_base_voxels'] for c in cases) * scale
        if retained + skipped > reference + 1e-6:
            raise ValueError('retained and skipped coverage overlap')
        ordinary = sum(c['restart_count'] for c in cases)
        filtered = sum(len(c.get('filtered_stops', [])) for c in cases)
        rows[source['source_id']] = dict(ordinary_restarts=ordinary,
            filtered_stops=filtered, reverse_checks=sum(c.get('reverse_checks', 0) for c in cases),
            reference_mm=reference, retained_mm=retained, filtered_skipped_mm=skipped,
            other_untraced_mm=max(0., reference-retained-skipped),
            directions=len(cases))
    total = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
    for row in [*rows.values(), total]:
        row['retained_percent'] = 100 * row['retained_mm']/row['reference_mm'] if row['reference_mm'] else 0
        row['mean_retained_mm_per_termination'] = row['retained_mm']/(row['ordinary_restarts']+row['filtered_stops']+row['directions'])
    return dict(protocol='periodic-reverse256-20x40-reseed10x20-angle25-v1',
                mean_note='retained distance / (ordinary restart episodes + filtered stops + directed runs); legacy means elsewhere count reference gaps',
                scrolls=rows, combined=total)
