"""Read-only cached-vector analysis of GT and baseline trace failure regions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from .embedding_consistency import distribution


def arc_lengths(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def line_point(points, arcs, at):
    if at < 0 or at > arcs[-1]:
        raise ValueError("reference query outside line")
    return np.array([np.interp(at, arcs, points[:, d]) for d in range(3)])


def history_indices(arcs, index, spacing=32., window=256., count=8):
    """Replay the tracer's append-at-spacing history, excluding candidate itself."""
    selected = []
    for j in range(index):
        selected = [k for k in selected if arcs[j] - arcs[k] <= window]
        if not selected or arcs[j] - arcs[selected[-1]] >= spacing:
            selected.append(j)
        selected = selected[-count:]
    return [k for k in selected if arcs[index] - arcs[k] <= window]


class Queries:
    def __init__(self):
        self.points, self.lookup = [], {}

    def add(self, point):
        key = tuple(float(v) for v in np.asarray(point, dtype=np.float32))
        if key not in self.lookup:
            self.lookup[key] = len(self.points)
            self.points.append(key)
        return self.lookup[key]


def cosine(a, b):
    return float(np.clip(np.dot(a, b), -1, 1))


def contrast(rows, true, taken, references):
    if rows[true]["status"] != "ok" or rows[taken]["status"] != "ok":
        return None
    # Identical valid reference subset for true/taken; never separate averages.
    refs = [r for r in references if rows[r]["status"] == "ok"]
    if not refs:
        return None
    a = np.mean([cosine(rows[true]["embedding"], rows[r]["embedding"]) for r in refs])
    b = np.mean([cosine(rows[taken]["embedding"], rows[r]["embedding"]) for r in refs])
    return dict(true=float(a), taken=float(b), gap=float(a-b), references=len(refs),
                requested_references=len(references), complete=len(refs) == len(references))


def stats(values):
    result = distribution(values)
    if values:
        result.update(p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)),
                      fraction_gt_09=float(np.mean(np.asarray(values) > .9)))
    return result


def analyze(settings):
    root = Path(settings["pilot"])
    report = json.loads((root / "report.json").read_text())
    off = next(v for v in report["variants"] if v["weight"] == 0)
    on = next(v for v in report["variants"] if v["weight"] > 0)
    baseline = Path(off["output"])
    inferred = json.loads((Path(on["output"]) / "inference_summary.json").read_text())
    caches = list(inferred["cache_roots"].values())
    if len(caches) != 1:
        raise ValueError("analysis expects exactly one source")
    manifest = Path(caches[0]) / "inference.lasagna.json"
    identity = json.loads((Path(caches[0]) / "cache_identity.json").read_text())
    if identity["identity"]["model"]["checkpoint_sha256"] != report["checkpoint_sha256"]:
        raise ValueError("embedding cache checkpoint does not match pilot")
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=False)
    queries, events, lag_pairs = Queries(), [], {}
    omitted_offsets = Counter()
    cases = [json.loads(p.read_text()) for p in sorted((baseline / "cases").glob("*/trace.json"))]
    for ci, case in enumerate(cases):
        gt = np.asarray(case["reference_points_base_xyz"], dtype=float)
        ga = arc_lengths(gt)
        replay = case["replay"]
        if ci == 0:
            for lag in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                # Anchor grid32, independently interpolated endpoints for lags8/16.
                lag_pairs[lag] = [(queries.add(line_point(gt, ga, a)), queries.add(line_point(gt, ga, a+lag)))
                    for a in np.arange(0, ga[-1]-lag, 32.)]
        for failure in replay["failures"]:
            si = failure["segment_index"]
            segment = replay["segments"][si]
            path = np.asarray(segment["trace_points_base_xyz"], dtype=float)
            arcs = arc_lengths(path)
            fi = failure["segment_point_index"]
            matches = {m["trace_point_index"]: m for m in segment["matches"]}
            if fi is None or not 0 <= fi < len(path):
                raise ValueError("failure lacks valid trace point index")
            for offset in [0, 32, 64, 128]:
                at = arcs[fi] - offset
                if at < 0:
                    omitted_offsets[offset] += 1
                    continue
                index = int(np.searchsorted(arcs, at, side="right") - 1)
                if index not in matches:
                    omitted_offsets[offset] += 1
                    continue
                match = matches[index]
                ref_arc = match["matched_reference_arc_base"]
                gt_ref_arcs = np.arange(segment["start_reference_arc_base"], ref_arc, 32.)
                gt_ref_arcs = gt_ref_arcs[(gt_ref_arcs < ref_arc) & (gt_ref_arcs >= ref_arc-256)][-8:]
                refs = [queries.add(line_point(gt, ga, a)) for a in gt_ref_arcs]
                actual_refs = [queries.add(path[j]) for j in history_indices(arcs, index)]
                events.append(dict(case=ci, reverse=case["provenance"]["reverse"],
                    failure=failure["index"], segment=si, offset=offset,
                    actual_offset=float(arcs[fi]-arcs[index]), reference_arc=ref_arc,
                    threshold_error_ratio=match["threshold_error_ratio"],
                    euclidean_error_base_voxels=match["euclidean_error_base_voxels"],
                    true=queries.add(match["matched_reference_point_base_xyz"]), taken=queries.add(path[index]),
                    gt_history=refs, route_history=actual_refs,
                    lag_references={str(d): queries.add(line_point(gt, ga, ref_arc-d))
                                    for d in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                                    if offset == 0 and ref_arc >= d},
                    seed=[queries.add(line_point(gt, ga, segment["start_reference_arc_base"]))]))
    point_file = output / "points.json"
    point_file.write_text(json.dumps(queries.points))
    started = time.monotonic()
    with (output / "queries.json").open("w") as stdout, (output / "native.log").open("w") as stderr:
        subprocess.run([settings["native_binary"], "sample-embeddings", str(manifest), str(point_file)],
                       stdout=stdout, stderr=stderr, check=True, timeout=240)
    rows = json.loads((output / "queries.json").read_text())
    if len(rows) != len(queries.points):
        raise ValueError("native result count mismatch")
    lags = {}
    for lag, pairs in lag_pairs.items():
        valid = [(a,b) for a,b in pairs if rows[a]["status"] == rows[b]["status"] == "ok"]
        lags[lag] = dict(requested=len(pairs), **stats([cosine(rows[a]["embedding"], rows[b]["embedding"]) for a,b in valid]))
    for event in events:
        a, b = event["true"], event["taken"]
        event["coverage"] = [rows[a]["status"], rows[b]["status"]]
        for mode in ["gt_history", "route_history", "seed"]:
            event[mode + "_contrast"] = contrast(rows, a, b, event[mode])
        event["lag_contrasts"] = {lag: contrast(rows, a, b, [ref]) for lag, ref in event["lag_references"].items()}
        if event["coverage"] == ["ok", "ok"]:
            event.update(pair_cosine=cosine(rows[a]["embedding"], rows[b]["embedding"]),
                gt_presence=rows[a]["presence"], taken_presence=rows[b]["presence"])
    stages = {}
    for offset in [0, 32, 64, 128]:
        group = [e for e in events if e["offset"] == offset]
        stage = dict(requested=len(group), omitted_before_segment=omitted_offsets[offset],
            covered=sum("pair_cosine" in e for e in group))
        for field in ["pair_cosine", "gt_presence", "taken_presence", "threshold_error_ratio", "actual_offset"]:
            stage[field] = stats([e[field] for e in group if field in e])
        for mode in ["gt_history", "route_history", "seed"]:
            contrasts = [e[mode+"_contrast"] for e in group if e[mode+"_contrast"] is not None]
            stage[mode] = {field: stats([v[field] for v in contrasts]) for field in ["true", "taken", "gap"]}
            stage[mode]["true_preferred_fraction"] = float(np.mean([v["gap"] > 0 for v in contrasts])) if contrasts else None
            stage[mode]["complete_histories"] = sum(v["complete"] for v in contrasts)
        stages[offset] = stage
    failure_events = [e for e in events if e["offset"] == 0]
    failure_lags = {}
    for lag in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        values = [e["lag_contrasts"].get(str(lag)) for e in failure_events]
        values = [v for v in values if v is not None]
        failure_lags[lag] = dict(n=len(values), true=stats([v["true"] for v in values]),
            taken=stats([v["taken"] for v in values]), gap=stats([v["gap"] for v in values]),
            true_preferred_fraction=float(np.mean([v["gap"] > 0 for v in values])) if values else None)
    presence_groups = {}
    for name, predicate in [("both_high", lambda e: min(e.get("gt_presence", 0),e.get("taken_presence", 0)) >= .5),
                            ("gt_low", lambda e: e.get("gt_presence", 1) < .5),
                            ("taken_high", lambda e: e.get("taken_presence", 0) >= .5)]:
        group = [e for e in failure_events if "pair_cosine" in e and predicate(e)]
        presence_groups[name] = dict(n=len(group), pair_cosine=stats([e["pair_cosine"] for e in group]),
            route_gap=stats([e["route_history_contrast"]["gap"] for e in group if e["route_history_contrast"] is not None]))
    result = dict(checkpoint=report["checkpoint"], checkpoint_sha256=report["checkpoint_sha256"],
        step=report["step"], fiber=report["fiber"], manifest=str(manifest),
        notes="Existing enabled cache queried for baseline path; CPU/native only, no new inference. GT-history is an oracle diagnostic. Error region need not be a verified different fiber. Lags in base voxels; one event per failure/direction/offset.",
        query_seconds=time.monotonic()-started, coverage=dict(Counter(r["status"] for r in rows)),
        lags=lags, stages=stages, failure_lags=failure_lags, presence_groups=presence_groups, events=events)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(output=str(output), coverage=result["coverage"], query_seconds=result["query_seconds"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    args = parser.parse_args()
    analyze(json.loads(Path(args.control).read_text())["embedding_route_analysis"])


if __name__ == "__main__":
    main()
