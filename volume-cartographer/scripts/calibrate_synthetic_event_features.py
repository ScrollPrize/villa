#!/usr/bin/env python3
"""Evaluate named passive event bases using synthetic work only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np

from calibrate_synthetic_event_costs import (
    FrequencyMonitor,
    MONITOR_CPU,
    WorkCase,
    all_frequency_reports_valid,
    collect_case,
    evaluate,
    features,
    fit,
    validate_fixed_frequency,
    validate_topology,
    write_json,
)
from passive_event_model import (
    DATA_READ_FEATURE_NAMES,
    DATA_READ_SPLIT_CACHE_FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    SPLIT_CACHE_FEATURE_NAMES,
)


MAX_PARAMETER_CORRELATION = 0.98
MAX_STABILITY_MOVEMENT_PERCENT = 20.0
MAX_HOLDOUT_MEDIAN_ERROR_PERCENT = 20.0
MAX_MEDIAN_REGRESSION_PERCENTAGE_POINTS = 2.0
MAX_RMS_REGRESSION_PERCENTAGE_POINTS = 2.0
MAX_FAMILY_REGRESSION_PERCENTAGE_POINTS = 5.0
MAX_DENSITY_NATIVE_RANGE_PERCENT = 10.0

CANDIDATE_SCHEMAS = {
    "data_reads": DATA_READ_FEATURE_NAMES,
    "split_cache": SPLIT_CACHE_FEATURE_NAMES,
    "data_reads_split_cache": DATA_READ_SPLIT_CACHE_FEATURE_NAMES,
}

DENSITY_FIT_CASES = tuple(
    WorkCase(kind, size, base * multiplier, 5 + level * 2, level, role)
    for kind, role, base in (
        ("read-four", "read_four_fit", 720_000),
        ("read-eight", "read_eight_fit", 440_000),
        ("write-eight", "write_eight_fit", 520_000),
    )
    for size in (16_384, 262_144, 4_194_304, 12_582_912)
    for level, multiplier in enumerate((1, 2, 4))
)

ACCESS_SEPARATION_FIT_CASES = tuple(
    WorkCase(kind, size, base * multiplier, 5 + level * 2, level, role)
    for kind, role, base in (
        ("line-read-one", "line_read_one_fit", 1_200_000),
        ("line-read-eight", "line_read_eight_fit", 360_000),
        ("line-write-one", "line_write_one_fit", 1_200_000),
        ("line-write-eight", "line_write_eight_fit", 360_000),
    )
    for size in (16_384, 262_144, 4_194_304, 12_582_912)
    for level, multiplier in enumerate((1, 2, 4))
)

CROSSED_ACCESS_FIT_CASES = tuple(
    WorkCase(kind, size, base * multiplier, 5 + level * 2, level, role)
    for kind, role, base in (
        ("line-r1-w1", "line_r1_w1_fit", 750_000),
        ("line-r8-w1", "line_r8_w1_fit", 300_000),
        ("line-r1-w8", "line_r1_w8_fit", 300_000),
        ("line-r8-w8", "line_r8_w8_fit", 180_000),
    )
    for size in (16_384, 262_144, 4_194_304, 12_582_912)
    for level, multiplier in enumerate((1, 2, 4))
)

ALL_DENSITY_FIT_CASES = (
    *DENSITY_FIT_CASES,
    *ACCESS_SEPARATION_FIT_CASES,
    *CROSSED_ACCESS_FIT_CASES,
)
DENSITY_FAMILY_ROLES = tuple(
    sorted({case.role for case in ALL_DENSITY_FIT_CASES})
)

FRESH_HOLDOUT_CASES = tuple(
    WorkCase(kind, size, iterations, 10, 4, "feature_holdout")
    for kind, iterations in (
        ("branch", 211_000),
        ("stream-read", 1_037_000),
        ("stream-write", 1_091_000),
        ("cache-read", 379_000),
        ("grid-sample", 293_000),
        ("mixed-grid-phase", 307_000),
        ("mixed-grid-random", 319_000),
        ("read-four", 227_000),
        ("read-eight", 139_000),
        ("write-eight", 151_000),
        ("line-read-one", 251_000),
        ("line-read-eight", 157_000),
        ("line-write-one", 263_000),
        ("line-write-eight", 163_000),
        ("line-r1-w1", 193_000),
        ("line-r8-w1", 127_000),
        ("line-r1-w8", 131_000),
        ("line-r8-w8", 101_000),
    )
    for size in (24_576, 393_216, 3_932_160, 15_728_640)
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
        scripts.parent / "core/test/bench_thread_pool_dispatch.cpp",
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
    for value in observations.values():
        if not isinstance(value, list):
            continue
        records.extend(
            record for record in value
            if isinstance(record, dict) and "case" in record
        )
    return records


def base_fit_records(observations: dict[str, object]) -> list[dict[str, object]]:
    records = (
        list(observations.get("fit", []))
        + list(observations.get("cache_fit", []))
        + list(observations.get("mixed_fit", []))
    )
    expected = {
        "branch", "stream-read", "stream-write", "cache-read", "grid-sample",
        "mixed-grid-phase", "mixed-grid-random",
    }
    result = [record for record in records if record["case"]["kind"] in expected]
    if {record["case"]["kind"] for record in result} != expected:
        raise RuntimeError("base fit observations do not contain seven expected kinds")
    return result


def matrix_diagnostics(
    records: list[dict[str, object]], feature_names: tuple[str, ...]
) -> dict[str, object]:
    matrix = np.asarray([features(record, feature_names) for record in records])
    measured = np.asarray(
        [record["native_work_median_ns"] for record in records], dtype=float
    )
    relative = matrix / measured[:, np.newaxis]
    norms = np.linalg.norm(relative, axis=0)
    if np.any(norms <= np.finfo(float).eps):
        return {"rank": 0, "parameter_count": len(feature_names)}
    normalized = relative / norms
    singular = np.linalg.svd(normalized, compute_uv=False)
    covariance = np.linalg.pinv(normalized.T @ normalized)
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = covariance / np.outer(diagonal, diagonal)
    maximum_correlation = max(
        abs(float(correlation[i, j]))
        for i in range(len(feature_names)) for j in range(i)
    )
    return {
        "rank": int(np.linalg.matrix_rank(normalized)),
        "parameter_count": len(feature_names),
        "normalized_singular_values": list(map(float, singular)),
        "normalized_condition_number": float(singular[0] / singular[-1]),
        "maximum_absolute_parameter_correlation": maximum_correlation,
    }


def stability(
    records: list[dict[str, object]],
    fixed_process_ns: float,
    feature_names: tuple[str, ...],
    full_model: dict[str, object],
) -> dict[str, object]:
    full = np.asarray(full_model["coefficients_ns"], dtype=float)
    rows = []
    for role in DENSITY_FAMILY_ROLES:
        reduced = [record for record in records if record["case"]["role"] != role]
        diagnostics = matrix_diagnostics(reduced, feature_names)
        if diagnostics["rank"] != len(feature_names):
            rows.append({"omitted_family": role, "rank_lost": True})
            continue
        model = fit(
            reduced,
            fixed_process_ns,
            allow_overlap=False,
            feature_names=feature_names,
        )
        coefficients = np.asarray(model["coefficients_ns"], dtype=float)
        movement = 100.0 * np.abs(coefficients - full) / np.maximum(np.abs(full), 1e-6)
        rows.append({
            "omitted_family": role,
            "rank_lost": False,
            "maximum_coefficient_movement_percent": float(np.max(movement)),
            "coefficient_movement_percent": list(map(float, movement)),
        })
    rank_lost = any(row["rank_lost"] for row in rows)
    maximum = max(
        (row.get("maximum_coefficient_movement_percent", float("inf")) for row in rows),
        default=float("inf"),
    )
    return {
        "cases": rows,
        "rank_lost": rank_lost,
        "maximum_coefficient_movement_percent": maximum,
    }


def relative_native_mad_percent(record: dict[str, object]) -> float:
    samples = list(map(float, record["native_work_samples_ns"]))
    median = statistics.median(samples)
    return 100.0 * statistics.median(abs(value - median) for value in samples) / median


def relative_native_range_percent(record: dict[str, object]) -> float:
    samples = list(map(float, record["native_work_samples_ns"]))
    return 100.0 * (max(samples) - min(samples)) / statistics.median(samples)


def by_kind(model: dict[str, object], records: list[dict[str, object]]) -> dict[str, object]:
    return {
        kind: evaluate(model, [record for record in records if record["case"]["kind"] == kind])
        for kind in sorted({record["case"]["kind"] for record in records})
    }


def fit_models(
    base: dict[str, object], density: dict[str, object]
) -> dict[str, dict[str, object]]:
    records = base_fit_records(base) + list(density["density_fit"])
    fixed_process_ns = float(base["startup"]["fixed_process_ns"])
    schemas = {"matched_legacy": LEGACY_FEATURE_NAMES, **CANDIDATE_SCHEMAS}
    result = {}
    for name, schema in schemas.items():
        model = fit(
            records,
            fixed_process_ns,
            allow_overlap=False,
            feature_names=schema,
        )
        model["fit"] = evaluate(model, records)
        model["matrix_diagnostics"] = matrix_diagnostics(records, schema)
        model["density_family_stability"] = stability(
            records, fixed_process_ns, schema, model
        )
        model["fit_record_count"] = len(records)
        result[name] = model
    return result


def fit_gate(model: dict[str, object]) -> bool:
    diagnostics = model["matrix_diagnostics"]
    stability_report = model["density_family_stability"]
    return bool(
        diagnostics["rank"] == diagnostics["parameter_count"]
        and diagnostics["maximum_absolute_parameter_correlation"]
        < MAX_PARAMETER_CORRELATION
        and model["bound_hit_count"] == 0
        and not stability_report["rank_lost"]
        and stability_report["maximum_coefficient_movement_percent"]
        <= MAX_STABILITY_MOVEMENT_PERCENT
    )


def collect_cases(
    benchmark: Path,
    output_dir: Path,
    cases: tuple[WorkCase, ...],
    trials: int,
    label: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    state, policies = validate_fixed_frequency()
    monitor = FrequencyMonitor(policies)
    records = []
    with monitor:
        for index, case in enumerate(cases, 1):
            print(f"[{label}] {index}/{len(cases)} {case.case_id}", flush=True)
            records.append(collect_case(benchmark, output_dir, case, trials))
    postflight, _ = validate_fixed_frequency()
    frequency = {
        **state,
        **monitor.report(int(state["target_khz"])),
        "postflight": postflight,
    }
    return records, frequency


def artifact_hashes(output_dir: Path, records: list[dict[str, object]]) -> dict[str, str]:
    result = {}
    for record in records:
        path = output_dir / f"callgrind.{record['case_id']}"
        result[str(path.resolve())] = sha256(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fit", "holdout"), required=True)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--base-observations", required=True, type=Path)
    parser.add_argument("--opened-observations", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-trials", type=int, default=5)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.native_trials < 5:
        raise RuntimeError("feature calibration requires at least five native trials")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_topology()
    os.sched_setaffinity(0, {MONITOR_CPU})

    base = json.loads(args.base_observations.read_text())
    if base.get("renderer_inputs_used") is not False:
        raise RuntimeError("base observations are not synthetic-only")
    density_path = args.output_dir / "fit_observations.json"
    if args.phase == "fit":
        if args.reuse:
            density = json.loads(density_path.read_text())
        else:
            records, frequency = collect_cases(
                args.benchmark, args.output_dir, ALL_DENSITY_FIT_CASES,
                args.native_trials, "feature-fit",
            )
            density = {
                "schema_version": 1,
                "renderer_inputs_used": False,
                "density_fit": records,
                "frequency": frequency,
                "artifact_hashes": artifact_hashes(args.output_dir, records),
            }
            write_json(density_path, density)
        models = fit_models(base, density)
        report = {
            "schema_version": 1,
            "renderer_inputs_used": False,
            "density_observations_sha256": sha256(density_path),
            "models": models,
            "density_measurements": {
                "maximum_native_range_percent": max(
                    relative_native_range_percent(record)
                    for record in density["density_fit"]
                ),
                "frequency_reports_valid": all_frequency_reports_valid(
                    density["frequency"]
                ),
            },
            "fit_gates": {name: fit_gate(model) for name, model in models.items()},
        }
        report["density_measurements_valid"] = bool(
            report["density_measurements"]["maximum_native_range_percent"]
            <= MAX_DENSITY_NATIVE_RANGE_PERCENT
            and report["density_measurements"]["frequency_reports_valid"]
        )
        write_json(args.output_dir / "fit_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["density_measurements_valid"] and any(
            report["fit_gates"][name] for name in CANDIDATE_SCHEMAS
        ) else 2

    density = json.loads(density_path.read_text())
    models = fit_models(base, density)
    candidate_fit_gates = {
        name: fit_gate(models[name]) for name in CANDIDATE_SCHEMAS
    }
    maximum_density_range = max(
        relative_native_range_percent(record) for record in density["density_fit"]
    )
    if (
        maximum_density_range > MAX_DENSITY_NATIVE_RANGE_PERCENT
        or not all_frequency_reports_valid(density["frequency"])
    ):
        raise RuntimeError("density calibration measurements failed stability gates")
    if not any(candidate_fit_gates.values()):
        raise RuntimeError("no candidate passes opened-fit gates")

    opened = {case_signature(record) for record in opened_records(base)}
    opened.update(case_signature(record) for record in density["density_fit"])
    opened_hashes = {str(args.base_observations): sha256(args.base_observations)}
    for path in args.opened_observations:
        observations = json.loads(path.read_text())
        opened.update(case_signature(record) for record in opened_records(observations))
        opened_hashes[str(path)] = sha256(path)
    fresh = {case_signature(case) for case in FRESH_HOLDOUT_CASES}
    if len(fresh) != len(FRESH_HOLDOUT_CASES) or not fresh.isdisjoint(opened):
        raise RuntimeError("fresh feature holdout overlaps opened observations")

    freeze = {
        "schema_version": 1,
        "renderer_inputs_used": False,
        "case_manifest": [asdict(case) for case in FRESH_HOLDOUT_CASES],
        "candidate_schemas": {name: list(value) for name, value in CANDIDATE_SCHEMAS.items()},
        "source_hashes": source_hashes(args.benchmark),
        "opened_observation_hashes": opened_hashes,
        "density_observations_sha256": sha256(density_path),
        "fit_gates": candidate_fit_gates,
        "frozen_fit_models": {
            name: {
                "feature_names": model["feature_names"],
                "coefficients_ns": model["coefficients_ns"],
                "matrix_diagnostics": model["matrix_diagnostics"],
                "density_family_stability": model["density_family_stability"],
            }
            for name, model in models.items()
        },
        "acceptance": {
            "maximum_parameter_correlation": MAX_PARAMETER_CORRELATION,
            "maximum_stability_movement_percent": MAX_STABILITY_MOVEMENT_PERCENT,
            "maximum_holdout_median_error_percent": MAX_HOLDOUT_MEDIAN_ERROR_PERCENT,
            "maximum_median_regression_percentage_points": MAX_MEDIAN_REGRESSION_PERCENTAGE_POINTS,
            "maximum_rms_regression_percentage_points": MAX_RMS_REGRESSION_PERCENTAGE_POINTS,
            "maximum_family_regression_percentage_points": MAX_FAMILY_REGRESSION_PERCENTAGE_POINTS,
            "maximum_density_native_range_percent": MAX_DENSITY_NATIVE_RANGE_PERCENT,
            "minimum_maximum_improvement": "twice maximum per-case relative native-sample MAD",
        },
    }
    freeze_path = args.output_dir / "holdout_freeze.json"
    holdout_path = args.output_dir / "holdout_observations.json"
    if args.reuse:
        if json.loads(freeze_path.read_text()) != freeze:
            raise RuntimeError("frozen feature experiment inputs have changed")
        holdout_observations = json.loads(holdout_path.read_text())
    else:
        write_json(freeze_path, freeze)
        records, frequency = collect_cases(
            args.benchmark, args.output_dir, FRESH_HOLDOUT_CASES,
            args.native_trials, "feature-holdout",
        )
        holdout_observations = {
            "schema_version": 1,
            "renderer_inputs_used": False,
            "fresh_holdout": records,
            "frequency": frequency,
            "freeze_sha256": sha256(freeze_path),
            "artifact_hashes": artifact_hashes(args.output_dir, records),
        }
        write_json(holdout_path, holdout_observations)

    holdout = list(holdout_observations["fresh_holdout"])
    baseline = models["matched_legacy"]
    baseline["fresh_holdout"] = evaluate(baseline, holdout)
    baseline["fresh_holdout_by_workload"] = by_kind(baseline, holdout)
    maximum_native_mad = max(relative_native_mad_percent(record) for record in holdout)
    outputs = {}
    for name in CANDIDATE_SCHEMAS:
        model = models[name]
        model["fresh_holdout"] = evaluate(model, holdout)
        model["fresh_holdout_by_workload"] = by_kind(model, holdout)
        improvement = (
            baseline["fresh_holdout"]["maximum_absolute_error_percent"]
            - model["fresh_holdout"]["maximum_absolute_error_percent"]
        )
        median_regression = (
            model["fresh_holdout"]["median_absolute_error_percent"]
            - baseline["fresh_holdout"]["median_absolute_error_percent"]
        )
        rms_regression = (
            model["fresh_holdout"]["rms_error_percent"]
            - baseline["fresh_holdout"]["rms_error_percent"]
        )
        family_regression = max(
            model["fresh_holdout_by_workload"][kind]["maximum_absolute_error_percent"]
            - baseline["fresh_holdout_by_workload"][kind]["maximum_absolute_error_percent"]
            for kind in model["fresh_holdout_by_workload"]
        )
        selection = {
            "fit_gate_passed": candidate_fit_gates[name],
            "renderer_inputs_used_for_selection": False,
            "frequency_reports_valid": all_frequency_reports_valid(
                holdout_observations["frequency"]
            ),
            "maximum_native_sample_mad_percent": maximum_native_mad,
            "required_maximum_improvement_percentage_points": 2.0 * maximum_native_mad,
            "maximum_improvement_percentage_points": improvement,
            "median_regression_percentage_points": median_regression,
            "rms_regression_percentage_points": rms_regression,
            "maximum_family_regression_percentage_points": family_regression,
        }
        model["selection"] = selection
        model["schema_version"] = 4
        model["source"] = "generic_synthetic_work_only"
        model["renderer_inputs_used"] = False
        model["freeze_sha256"] = sha256(freeze_path)
        model["observations_sha256"] = sha256(holdout_path)
        model["synthetic_calibration_valid"] = bool(
            candidate_fit_gates[name]
            and model["fresh_holdout"]["median_absolute_error_percent"] <= MAX_HOLDOUT_MEDIAN_ERROR_PERCENT
            and improvement > 2.0 * maximum_native_mad
            and median_regression <= MAX_MEDIAN_REGRESSION_PERCENTAGE_POINTS
            and rms_regression <= MAX_RMS_REGRESSION_PERCENTAGE_POINTS
            and family_regression <= MAX_FAMILY_REGRESSION_PERCENTAGE_POINTS
            and selection["frequency_reports_valid"]
        )
        outputs[name] = model
        write_json(args.output_dir / f"model_{name}.json", model)

    baseline.update({
        "schema_version": 4,
        "source": "generic_synthetic_work_only",
        "renderer_inputs_used": False,
        "matched_soft_l1_baseline": True,
        "synthetic_calibration_valid": False,
        "freeze_sha256": sha256(freeze_path),
        "observations_sha256": sha256(holdout_path),
    })
    write_json(args.output_dir / "model_matched_legacy.json", baseline)
    report = {
        "schema_version": 1,
        "renderer_inputs_used": False,
        "matched_legacy": baseline,
        "candidates": outputs,
    }
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if any(
        model["synthetic_calibration_valid"] for model in outputs.values()
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
