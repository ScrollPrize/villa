"""Read-only TensorBoard audit: python -m ...fiber_trace_3d.analyze_training RUN."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def distribution(values):
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return None
    position = (len(values) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return {
        "count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": values[lower] + (values[upper] - values[lower]) * (position - lower),
        "min": values[0], "max": values[-1],
    }


def summarize(events, *, threshold=1.0, warmup_steps=100):
    """Accept original TensorBoard Event protobufs without tensor/image decoding."""
    series = {}
    config = None
    start = None
    for event in events:
        if event.wall_time > 0:
            start = event.wall_time if start is None else min(start, event.wall_time)
        for value in event.summary.value:
            if value.HasField("simple_value"):
                series.setdefault(value.tag, []).append({
                    "step": event.step, "wall_time": event.wall_time,
                    "value": value.simple_value,
                })
            elif value.tag == "config/json/text_summary" and value.tensor.string_val:
                config = json.loads(value.tensor.string_val[0])
    if start is None:
        raise ValueError("No timestamped TensorBoard events found")
    for rows in series.values():
        rows.sort(key=lambda row: row["wall_time"])
        for row in rows:
            row["elapsed_seconds"] = row["wall_time"] - start
    tests = series.get("test/loss_total", [])
    finite_tests = [row for row in tests if math.isfinite(row["value"])]
    crossing = next((row for row in finite_tests if row["value"] < threshold), None)
    # A paired gap includes rank-0 logging/scheduling; it is NOT a CUDA timer.
    train_by_step = {}
    for row in series.get("train/loss_total", []):
        train_by_step.setdefault(row["step"], []).append(row)
    gaps = []
    for test in tests:
        candidates = [row for row in train_by_step.get(test["step"], [])
                      if row["wall_time"] <= test["wall_time"]]
        if candidates:
            gaps.append(test["wall_time"] - candidates[-1]["wall_time"])
    return {
        "event_start_unix_seconds": start,
        "elapsed_origin": "first event (excludes process startup before writer creation)",
        "threshold": threshold, "first_threshold_crossing": crossing,
        "best_test": min(finite_tests, key=lambda row: row["value"], default=None),
        "last_test": tests[-1] if tests else None,
        "test_curve": tests,
        "quality_curves": {tag: rows for tag, rows in series.items()
                           if tag.startswith("test/")},
        "config": config,
        "timing_sample_filter": f"logged steps > {warmup_steps}; not every-step measurements",
        "timing_ms": {tag: distribution(row["value"] for row in rows
                                       if row["step"] > warmup_steps)
                      for tag, rows in series.items()
                      if tag.startswith("timing/") and tag.endswith("_ms")},
        "paired_train_to_test_gap_seconds": distribution(gaps),
        "paired_train_to_test_gap_sum_seconds": sum(gaps),
        "gap_caveat": "Approximate rank-0 timestamp gaps; includes logging and scheduling. "
                      "Do not sum sparse per-step timers to infer total training time.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.warmup_steps < 0:
        parser.error("threshold must be finite and warmup-steps nonnegative")
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

    files = sorted(args.run.glob("events.out.tfevents.*"))
    if not files:
        parser.error(f"no event files in {args.run}")
    if len(files) != 1:
        parser.error("expected one event file; resumed/multiple writers need explicit session analysis")
    result = summarize(LegacyEventFileLoader(str(files[0])).Load(),
                       threshold=args.threshold, warmup_steps=args.warmup_steps)
    result["event_file"] = str(files[0].resolve())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
