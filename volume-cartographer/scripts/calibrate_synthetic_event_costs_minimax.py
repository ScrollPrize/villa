#!/usr/bin/env python3
"""Compare soft-L1 and minimax fits using synthetic event-cost work only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path

from calibrate_synthetic_event_costs import (
    FrequencyMonitor,
    MONITOR_CPU,
    WorkCase,
    all_frequency_reports_valid,
    collect_case,
    evaluate,
    fit,
    fit_minimax,
    validate_fixed_frequency,
    validate_topology,
    write_json,
)


MAX_HOLDOUT_MEDIAN_ERROR_PERCENT = 20.0
MAX_MEDIAN_REGRESSION_PERCENTAGE_POINTS = 2.0
MAX_RMS_REGRESSION_PERCENTAGE_POINTS = 2.0
MAX_FAMILY_REGRESSION_PERCENTAGE_POINTS = 5.0
MAX_COEFFICIENT_INTERVAL_WIDTH = 0.20

FRESH_HOLDOUT_CASES = tuple(
    WorkCase(kind, size, iterations, 9, 3, "minimax_holdout")
    for kind, iterations in (
        ("branch", 163_000),
        ("stream-read", 917_000),
        ("stream-write", 973_000),
        ("cache-read", 337_000),
        ("grid-sample", 251_000),
        ("mixed-grid-phase", 263_000),
        ("mixed-grid-random", 277_000),
    )
    for size in (98_304, 1_572_864, 6_291_456)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(benchmark: Path) -> dict[str, str]:
    scripts = Path(__file__).resolve().parent
    paths = (
        benchmark.resolve(),
        Path(__file__).resolve(),
        scripts / "calibrate_synthetic_event_costs.py",
        scripts / "passive_event_model.py",
    )
    return {str(path): sha256(path) for path in paths}


def case_signature(record_or_case: object) -> tuple[object, ...]:
    value = record_or_case
    if isinstance(value, dict):
        value = value["case"]
        return (
            value["kind"], value["working_set_bytes"], value["iterations"],
            value["rounds"], value["warmup_rounds"],
        )
    return (
        value.kind, value.working_set_bytes, value.iterations,
        value.rounds, value.warmup_rounds,
    )


def opened_records(observations: dict[str, object]) -> list[dict[str, object]]:
    records = []
    for key in (
        "fit", "cache_fit", "holdout", "cache_holdout", "mixed_fit",
        "mixed_holdout", "serialization_fit", "serialization_holdout",
        "diagnostic",
    ):
        records.extend(observations.get(key, []))
    return records


def fit_records(observations: dict[str, object]) -> list[dict[str, object]]:
    records = (
        list(observations.get("fit", []))
        + list(observations.get("cache_fit", []))
        + list(observations.get("mixed_fit", []))
    )
    rejected = {
        "pointer", "alu", "fp", "mixed", "cache-write",
    }
    result = [record for record in records if record["case"]["kind"] not in rejected]
    expected = {
        "branch", "stream-read", "stream-write", "cache-read", "grid-sample",
        "mixed-grid-phase", "mixed-grid-random",
    }
    if {record["case"]["kind"] for record in result} != expected:
        raise RuntimeError("synthetic fit records do not contain the expected seven kinds")
    return result


def relative_native_mad_percent(record: dict[str, object]) -> float:
    samples = list(map(float, record["native_work_samples_ns"]))
    median = statistics.median(samples)
    return 100.0 * statistics.median(abs(value - median) for value in samples) / median


def by_kind(model: dict[str, object], records: list[dict[str, object]]) -> dict[str, object]:
    return {
        kind: evaluate(model, [record for record in records if record["case"]["kind"] == kind])
        for kind in sorted({record["case"]["kind"] for record in records})
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--base-observations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-trials", type=int, default=5)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.native_trials < 5:
        raise RuntimeError("minimax holdout requires at least five native trials")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(args.base_observations.read_text())
    if base.get("renderer_inputs_used") is not False:
        raise RuntimeError("base observations are not synthetic-only")
    opened = {case_signature(record) for record in opened_records(base)}
    fresh = {case_signature(case) for case in FRESH_HOLDOUT_CASES}
    if len(fresh) != len(FRESH_HOLDOUT_CASES) or not fresh.isdisjoint(opened):
        raise RuntimeError("fresh minimax holdout overlaps an opened synthetic case")

    freeze = {
        "schema_version": 1,
        "renderer_inputs_used": False,
        "case_manifest": [asdict(case) for case in FRESH_HOLDOUT_CASES],
        "source_hashes": source_hashes(args.benchmark),
        "base_observations_sha256": sha256(args.base_observations),
        "acceptance": {
            "maximum_holdout_median_error_percent": MAX_HOLDOUT_MEDIAN_ERROR_PERCENT,
            "maximum_median_regression_percentage_points": MAX_MEDIAN_REGRESSION_PERCENTAGE_POINTS,
            "maximum_rms_regression_percentage_points": MAX_RMS_REGRESSION_PERCENTAGE_POINTS,
            "maximum_family_regression_percentage_points": MAX_FAMILY_REGRESSION_PERCENTAGE_POINTS,
            "maximum_coefficient_interval_width": MAX_COEFFICIENT_INTERVAL_WIDTH,
            "minimum_maximum_improvement": "twice maximum per-case relative native-sample MAD",
        },
    }
    freeze_path = args.output_dir / "freeze.json"
    observations_path = args.output_dir / "observations.json"
    if args.reuse:
        if json.loads(freeze_path.read_text()) != freeze:
            raise RuntimeError("frozen minimax experiment inputs have changed")
        observations = json.loads(observations_path.read_text())
    else:
        write_json(freeze_path, freeze)
        validate_topology()
        os.sched_setaffinity(0, {MONITOR_CPU})
        state, policies = validate_fixed_frequency()
        monitor = FrequencyMonitor(policies)
        records = []
        with monitor:
            for index, case in enumerate(FRESH_HOLDOUT_CASES, 1):
                print(f"[minimax-holdout] {index}/{len(FRESH_HOLDOUT_CASES)} {case.case_id}", flush=True)
                records.append(collect_case(args.benchmark, args.output_dir, case, args.native_trials))
        postflight, _ = validate_fixed_frequency()
        observations = {
            "schema_version": 1,
            "renderer_inputs_used": False,
            "fresh_holdout": records,
            "frequency": {**state, **monitor.report(int(state["target_khz"])), "postflight": postflight},
            "freeze_sha256": sha256(freeze_path),
        }
        write_json(observations_path, observations)

    records = fit_records(base)
    holdout = list(observations["fresh_holdout"])
    fixed_process_ns = float(base["startup"]["fixed_process_ns"])
    baseline = fit(records, fixed_process_ns, allow_overlap=False, include_serial_pressure=False)
    candidate = fit_minimax(records, fixed_process_ns, baseline["coefficients_ns"])
    baseline["fit"] = evaluate(baseline, records)
    candidate["fit"] = evaluate(candidate, records)
    baseline["fresh_holdout"] = evaluate(baseline, holdout)
    candidate["fresh_holdout"] = evaluate(candidate, holdout)
    baseline["fresh_holdout_by_workload"] = by_kind(baseline, holdout)
    candidate["fresh_holdout_by_workload"] = by_kind(candidate, holdout)

    maximum_native_mad = max(relative_native_mad_percent(record) for record in holdout)
    improvement = (
        baseline["fresh_holdout"]["maximum_absolute_error_percent"]
        - candidate["fresh_holdout"]["maximum_absolute_error_percent"]
    )
    median_regression = (
        candidate["fresh_holdout"]["median_absolute_error_percent"]
        - baseline["fresh_holdout"]["median_absolute_error_percent"]
    )
    rms_regression = (
        candidate["fresh_holdout"]["rms_error_percent"]
        - baseline["fresh_holdout"]["rms_error_percent"]
    )
    family_regression = max(
        candidate["fresh_holdout_by_workload"][kind]["maximum_absolute_error_percent"]
        - baseline["fresh_holdout_by_workload"][kind]["maximum_absolute_error_percent"]
        for kind in candidate["fresh_holdout_by_workload"]
    )
    maximum_interval_width = max(
        value["normalized_width"] for value in candidate["coefficient_intervals"]
    )
    selection = {
        "renderer_inputs_used_for_selection": False,
        "frequency_reports_valid": all_frequency_reports_valid(observations["frequency"]),
        "maximum_native_sample_mad_percent": maximum_native_mad,
        "required_maximum_improvement_percentage_points": 2.0 * maximum_native_mad,
        "maximum_improvement_percentage_points": improvement,
        "median_regression_percentage_points": median_regression,
        "rms_regression_percentage_points": rms_regression,
        "maximum_family_regression_percentage_points": family_regression,
        "maximum_coefficient_interval_width": maximum_interval_width,
    }
    candidate["selection"] = selection
    candidate["schema_version"] = 3
    candidate["source"] = "generic_synthetic_work_only"
    candidate["renderer_inputs_used"] = False
    candidate["observations_sha256"] = sha256(observations_path)
    candidate["freeze_sha256"] = sha256(freeze_path)
    candidate["synthetic_calibration_valid"] = bool(
        candidate["jacobian_rank"] == candidate["event_parameter_count"]
        and candidate["bound_hit_count"] == 0
        and maximum_interval_width <= MAX_COEFFICIENT_INTERVAL_WIDTH
        and candidate["fresh_holdout"]["median_absolute_error_percent"] <= MAX_HOLDOUT_MEDIAN_ERROR_PERCENT
        and improvement > 2.0 * maximum_native_mad
        and median_regression <= MAX_MEDIAN_REGRESSION_PERCENTAGE_POINTS
        and rms_regression <= MAX_RMS_REGRESSION_PERCENTAGE_POINTS
        and family_regression <= MAX_FAMILY_REGRESSION_PERCENTAGE_POINTS
        and selection["frequency_reports_valid"]
    )
    baseline.update({
        "schema_version": 3,
        "source": "generic_synthetic_work_only",
        "renderer_inputs_used": False,
        "matched_soft_l1_baseline": True,
        "synthetic_calibration_valid": False,
        "observations_sha256": candidate["observations_sha256"],
        "freeze_sha256": candidate["freeze_sha256"],
    })
    candidate["matched_soft_l1_baseline"] = baseline
    write_json(args.output_dir / "soft_l1_model.json", baseline)
    write_json(args.output_dir / "model.json", candidate)
    print(json.dumps(candidate, indent=2, sort_keys=True))
    return 0 if candidate["synthetic_calibration_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
