#!/usr/bin/env python3
"""Fit passive synchronization replay using only generic synthetic work."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from calibrate_thread_dispatch_shared import (
    FrequencyMonitor,
    MONITOR_CPU,
    validate_fixed_frequency,
    validate_topology,
)
from run_thread_sync_replay import (
    assign_costs,
    read_event_stream,
    simulate_adjusted,
)
from passive_event_model import modeled_thread_costs_ns


CALIBRATION_CPUS = tuple(range(8))
MAX_PARAMETER_CORRELATION = 0.95
MAX_HOLDOUT_ERROR_PERCENT = 20.0
MAX_HOLDOUT_SPEEDUP_ERROR_PERCENT = 30.0
MAX_STABILITY_MOVEMENT_PERCENT = 30.0


@dataclass(frozen=True)
class Case:
    workers: int
    tasks: int
    work_iterations: int
    warmup_rounds: int
    rounds: int
    role: str
    work_kind: str = "grid-sample"
    working_set_bytes: int = 1_048_576
    task_iteration_skew: int = 0
    pair: str = ""

    @property
    def case_id(self) -> str:
        return (
            f"{self.role}-w{self.workers}-t{self.tasks}"
            f"-i{self.work_iterations}-s{self.task_iteration_skew}"
            f"-u{self.warmup_rounds}-r{self.rounds}"
            f"-{self.work_kind}-b{self.working_set_bytes}"
        )


BASE_FIT_CASES = (
    Case(2, 1, 0, 0, 8, "fit"),
    Case(4, 4, 0, 0, 64, "fit"),
    Case(6, 3, 0, 32, 16, "fit"),
    Case(2, 2, 400_000, 0, 16, "fit"),
    Case(4, 1, 1_600_000, 8, 8, "fit"),
    Case(6, 6, 400_000, 0, 32, "fit"),
    Case(2, 2, 0, 0, 96, "fit"),
    Case(4, 2, 0, 16, 32, "fit"),
    Case(6, 1, 0, 0, 128, "fit"),
    Case(4, 1, 0, 0, 8, "fit"),
    Case(6, 1, 0, 0, 8, "fit"),
    Case(2, 2, 0, 8, 16, "fit"),
)

MIXED_FIT_CASES = (
    Case(2, 2, 120_000, 0, 24, "fit", "mixed-grid-phase", 65_536),
    Case(
        4, 4, 80_000, 0, 24, "fit", "mixed-grid-random", 1_048_576, 20_000
    ),
    Case(6, 6, 180_000, 4, 12, "fit", "mixed-grid-random", 8_388_608),
    Case(
        4, 2, 240_000, 8, 18, "fit", "mixed-grid-phase", 262_144, 80_000
    ),
)

FIT_CASES = (*BASE_FIT_CASES, *MIXED_FIT_CASES)

BASE_HOLDOUT_CASES = (
    Case(1, 1, 7_500, 0, 13, "holdout"),
    Case(3, 3, 0, 4, 72, "holdout"),
    Case(5, 3, 400_000, 0, 11, "holdout"),
    Case(7, 1, 350_000, 9, 15, "holdout"),
    Case(3, 2, 1_600_000, 0, 5, "holdout"),
    Case(5, 5, 0, 20, 28, "holdout"),
)

MIXED_HOLDOUT_CASES = (
    Case(
        3, 3, 100_000, 3, 19, "holdout", "mixed-grid-random", 131_072, 25_000
    ),
    Case(5, 5, 160_000, 0, 13, "holdout", "mixed-grid-phase", 2_097_152),
    Case(
        7, 4, 220_000, 6, 9, "holdout", "mixed-grid-random", 8_388_608, 55_000
    ),
)

HOLDOUT_CASES = (*BASE_HOLDOUT_CASES, *MIXED_HOLDOUT_CASES)

DEPENDENCY_FIT_CASES = (
    Case(2, 2, 240_000, 2, 8, "dependency_fit", "mixed-grid-phase", 65_536,
         pair="fit_a"),
    Case(2, 2, 60_000, 2, 32, "dependency_fit", "mixed-grid-phase", 65_536,
         pair="fit_a"),
    Case(4, 4, 160_000, 2, 10, "dependency_fit", "mixed-grid-random", 1_048_576,
         pair="fit_b"),
    Case(4, 4, 40_000, 2, 40, "dependency_fit", "mixed-grid-random", 1_048_576,
         pair="fit_b"),
    Case(6, 3, 200_000, 2, 10, "dependency_fit", "mixed-grid-random", 8_388_608,
         pair="fit_c"),
    Case(6, 3, 50_000, 2, 40, "dependency_fit", "mixed-grid-random", 8_388_608,
         pair="fit_c"),
    Case(4, 4, 120_000, 2, 10, "dependency_fit", "mixed-grid-phase", 262_144,
         40_000, "fit_d"),
    Case(4, 4, 30_000, 2, 40, "dependency_fit", "mixed-grid-phase", 262_144,
         10_000, "fit_d"),
)

DEPENDENCY_VALIDATION_CASES = (
    Case(3, 3, 150_000, 2, 12, "dependency_validation", "mixed-grid-phase",
         131_072, pair="validation_a"),
    Case(3, 3, 50_000, 2, 36, "dependency_validation", "mixed-grid-phase",
         131_072, pair="validation_a"),
    Case(5, 5, 100_000, 2, 12, "dependency_validation", "mixed-grid-random",
         2_097_152, pair="validation_b"),
    Case(5, 5, 30_000, 2, 40, "dependency_validation", "mixed-grid-random",
         2_097_152, pair="validation_b"),
)

DEPENDENCY_HOLDOUT_CASES = (
    Case(7, 4, 140_000, 2, 8, "dependency_holdout", "mixed-grid-random",
         8_388_608, 35_000, "holdout"),
    Case(7, 4, 40_000, 2, 28, "dependency_holdout", "mixed-grid-random",
         8_388_608, 10_000, "holdout"),
)


@dataclass
class ReplayPoint:
    case: Case
    events: list
    core_counts: tuple[int, int]
    native_medians: dict[int, float]
    native_samples: dict[int, list[float]]
    result_path: str
    fixed_process_ns: float = 0.0
    replay_cache: dict[tuple[int, float], dict[str, float]] = field(
        default_factory=dict
    )


def runner_command(
    runner: Path,
    benchmark: Path,
    output_dir: Path,
    case: Case,
    trace_trials: int,
    native_trials: int,
    event_cost_model: Path,
) -> list[str]:
    return [
        sys.executable,
        str(runner),
        "--workload",
        "synthetic",
        "--benchmark",
        str(benchmark),
        "--output-dir",
        str(output_dir),
        "--mode",
        "futures",
        "--workers",
        str(case.workers),
        "--tasks",
        str(case.tasks),
        "--work-iterations",
        str(case.work_iterations),
        "--task-iteration-skew",
        str(case.task_iteration_skew),
        "--work-kind",
        case.work_kind,
        "--working-set-bytes",
        str(case.working_set_bytes),
        "--warmup-rounds",
        str(case.warmup_rounds),
        "--rounds",
        str(case.rounds),
        "--trace-trials",
        str(trace_trials),
        "--native-trials",
        str(native_trials),
        "--cpu-affinity",
        ",".join(str(cpu) for cpu in CALIBRATION_CPUS),
        "--event-cost-model",
        str(event_cost_model),
    ]


def collect_case(
    runner: Path,
    benchmark: Path,
    output_dir: Path,
    case: Case,
    trace_trials: int,
    native_trials: int,
    event_cost_model: Path,
) -> Path:
    command = runner_command(
        runner, benchmark, output_dir, case, trace_trials, native_trials,
        event_cost_model,
    )
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    summary = json.loads(completed.stdout)
    return Path(summary["result_path"])


def startup_command(benchmark: Path, workers: int) -> list[str]:
    return [
        str(benchmark), "--mode", "futures", "--workers", str(workers),
        "--tasks", "1", "--work-iterations", "0", "--work-kind", "grid-sample",
        "--working-set-bytes", "512", "--rounds", "1", "--warmup-rounds", "0",
    ]


def collect_worker_startup(
    benchmark: Path, trials: int
) -> tuple[dict[str, object], dict[str, object]]:
    state, policies = validate_fixed_frequency()
    monitor = FrequencyMonitor(policies)
    records = []
    with monitor:
        for workers in range(1, 8):
            command = startup_command(benchmark, workers)
            external = []
            internal = []
            affinity = ",".join(str(cpu) for cpu in CALIBRATION_CPUS[: workers + 1])
            for _ in range(max(100, trials)):
                start = time.perf_counter_ns()
                completed = subprocess.run(
                    ["taskset", "-c", affinity, *command],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                external.append(float(time.perf_counter_ns() - start))
                internal.append(
                    float(json.loads(completed.stdout)["wall_seconds"]) * 1e9
                )
            overhead = [
                max(0.0, process - work)
                for process, work in zip(external, internal, strict=True)
            ]
            records.append(
                {
                    "workers": workers,
                    "command": command,
                    "trials": len(external),
                    "external_samples_ns": external,
                    "internal_samples_ns": internal,
                    "overhead_samples_ns": overhead,
                    "median_overhead_ns": statistics.median(overhead),
                }
            )
    postflight, _ = validate_fixed_frequency()
    frequency = monitor.report(int(state["target_khz"]))
    frequency["postflight"] = postflight
    return {"records": records}, {**state, **frequency}


def fit_worker_startup(startup: dict[str, object]) -> dict[str, object]:
    records = startup["records"]
    base = float(records[0]["median_overhead_ns"])
    workers = np.asarray([record["workers"] for record in records[1:]], dtype=float)
    measured = np.asarray(
        [record["median_overhead_ns"] for record in records[1:]], dtype=float
    )

    def residuals(values: np.ndarray) -> np.ndarray:
        return (base + (workers - 1.0) * values[0]) / measured - 1.0

    result = least_squares(
        residuals,
        np.asarray([100_000.0]),
        bounds=(1e-6, 5_000_000.0),
        loss="soft_l1",
        f_scale=0.02,
    )
    if not result.success:
        raise RuntimeError(f"worker startup fit failed: {result.message}")
    errors = residuals(result.x)
    return {
        "base_process_ns": base,
        "per_worker_startup_ns": float(result.x[0]),
        "maximum_median_error_percent": 100.0 * float(np.max(np.abs(errors))),
        "rms_median_error_percent": 100.0
        * float(np.sqrt(np.mean(errors**2))),
        "cases": [
            {
                "workers": int(worker_count),
                "measured_ns": float(observed),
                "predicted_ns": base + (worker_count - 1.0) * float(result.x[0]),
                "error_percent": 100.0 * float(error),
            }
            for worker_count, observed, error in zip(
                workers, measured, errors, strict=True
            )
        ],
    }


def load_point(
    case: Case, result_path: Path, event_model: dict[str, object]
) -> ReplayPoint:
    result = json.loads(result_path.read_text())
    if result["case"]["workload"] != "synthetic":
        raise RuntimeError(f"non-synthetic observation rejected: {result_path}")
    if result["trace"]["dependency_source"] != "drd_vector_clocks":
        raise RuntimeError(f"synthetic observation lacks DRD dependencies: {result_path}")
    if result["trace"]["unresolved_happens_before"] != 0:
        raise RuntimeError(f"synthetic observation has unresolved dependencies: {result_path}")

    events = read_event_stream(Path(result["trace"]["event_path"]))
    profiles = {
        int(thread): {name: int(value) for name, value in profile.items()}
        for thread, profile in result["profiles"].items()
    }
    event_costs = modeled_thread_costs_ns(profiles, event_model)
    assign_costs(
        events,
        event_costs,
        0.5,
        "equal",
    )
    core_counts = (
        1,
        int(result["case"].get("parallel_cores", int(result["case"]["workers"]) + 1)),
    )
    native = result["native_whole_process"]
    return ReplayPoint(
        case=case,
        events=events,
        core_counts=core_counts,
        native_medians={
            cores: float(native[str(cores)]["measured_work_ns"]["median"])
            for cores in core_counts
        },
        native_samples={
            cores: [
                float(value)
                for value in native[str(cores)]["measured_work_ns"]["samples"]
            ]
            for cores in core_counts
        },
        result_path=str(result_path),
        fixed_process_ns=0.0,
    )


def parameter_names(model: str) -> tuple[str, ...]:
    if model == "handoff":
        return ("cross_thread_release_ns",)
    if model == "dependency":
        return ("dependency_excess_scale",)
    raise ValueError(f"unknown model {model}")


def predict(
    point: ReplayPoint,
    cores: int,
    model: str,
    values: np.ndarray,
    fixed_handoff: float | None = None,
) -> float:
    parameters = dict(zip(parameter_names(model), values, strict=True))
    if model == "dependency" and fixed_handoff is None:
        raise RuntimeError("dependency model requires a frozen handoff latency")
    handoff = (
        float(parameters["cross_thread_release_ns"])
        if model == "handoff"
        else float(fixed_handoff)
    )
    dependency_scale = float(parameters.get("dependency_excess_scale", 1.0))
    key = (cores, handoff)
    replay = point.replay_cache.get(key)
    if replay is None:
        replay = simulate_adjusted(
            point.events,
            cores,
            "fifo",
            cross_thread_latency=handoff,
            replay_idle_scale=1.0,
        )
        point.replay_cache[key] = replay
    adjusted = (
        replay["hard_schedule_lower_bound"]
        + dependency_scale * replay["dependency_excess"]
        + replay["raw_replay_excess"]
    )
    return (
        point.fixed_process_ns
        + float(adjusted)
    )


def fit_model(
    model: str,
    points: list[ReplayPoint],
    fixed_handoff: float | None = None,
) -> dict[str, object]:
    names = parameter_names(model)
    initial = {
        "cross_thread_release_ns": 5_000.0,
        "dependency_excess_scale": 0.5,
    }
    upper = {
        "cross_thread_release_ns": 2_000_000.0,
        "dependency_excess_scale": 1.0,
    }
    x0 = np.asarray([initial[name] for name in names])
    lower = {
        "cross_thread_release_ns": 1e-6,
        "dependency_excess_scale": 0.0,
    }
    bounds = (
        np.asarray([lower[name] for name in names]),
        np.asarray([upper[name] for name in names]),
    )

    def residuals(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                predict(point, cores, model, values, fixed_handoff)
                / point.native_medians[cores]
                - 1.0
                for point in points
                for cores in point.core_counts
            ]
        )

    result = least_squares(
        residuals,
        x0,
        bounds=bounds,
        x_scale="jac",
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=300,
    )
    if not result.success:
        raise RuntimeError(f"{model} fit failed: {result.message}")
    jacobian = np.asarray(result.jac)
    rank = int(np.linalg.matrix_rank(jacobian))
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = covariance / np.outer(diagonal, diagonal)
    off_diagonal = [
        abs(float(correlation[row, column]))
        for row in range(len(names))
        for column in range(row)
        if math.isfinite(float(correlation[row, column]))
    ]
    errors = residuals(result.x)
    bound_hits = [
        name
        for index, name in enumerate(names)
        if math.isclose(result.x[index], bounds[0][index], rel_tol=0.0, abs_tol=1e-4)
        or math.isclose(result.x[index], bounds[1][index], rel_tol=1e-5)
    ]
    return {
        "name": model,
        "parameter_names": list(names),
        "parameters": {
            name: float(value) for name, value in zip(names, result.x, strict=True)
        },
        "jacobian_rank": rank,
        "parameter_count": len(names),
        "maximum_absolute_parameter_correlation": max(off_diagonal, default=0.0),
        "parameters_hit_bounds": bound_hits,
        "maximum_fit_error_percent": 100.0 * float(np.max(np.abs(errors))),
        "rms_fit_error_percent": 100.0 * float(np.sqrt(np.mean(errors**2))),
        "values": result.x,
    }


def evaluate(
    model: str,
    values: np.ndarray,
    points: list[ReplayPoint],
    fixed_handoff: float | None = None,
) -> dict[str, object]:
    cases = []
    speedups = []
    all_median_errors = []
    all_individual_errors = []
    all_speedup_errors = []
    for point in points:
        predictions = {
            cores: predict(point, cores, model, values, fixed_handoff)
            for cores in point.core_counts
        }
        for cores in point.core_counts:
            predicted = predictions[cores]
            measured = point.native_medians[cores]
            median_error = predicted / measured - 1.0
            individual = [
                predicted / sample - 1.0 for sample in point.native_samples[cores]
            ]
            all_median_errors.append(median_error)
            all_individual_errors.extend(individual)
            cases.append(
                {
                    "case_id": point.case.case_id,
                    "cores": cores,
                    "predicted_ns": predicted,
                    "measured_median_ns": measured,
                    "median_error_percent": 100.0 * median_error,
                    "maximum_individual_error_percent": 100.0
                    * max(abs(value) for value in individual),
                }
            )
        serial_cores, parallel_cores = point.core_counts
        predicted_speedup = (
            predictions[serial_cores] / predictions[parallel_cores]
        )
        measured_speedup = (
            point.native_medians[serial_cores]
            / point.native_medians[parallel_cores]
        )
        speedup_error = predicted_speedup / measured_speedup - 1.0
        all_speedup_errors.append(speedup_error)
        speedups.append(
            {
                "case_id": point.case.case_id,
                "parallel_cores": parallel_cores,
                "predicted_speedup": predicted_speedup,
                "measured_speedup": measured_speedup,
                "speedup_error_percent": 100.0 * speedup_error,
                "absolute_speedup_error_percent": 100.0 * abs(speedup_error),
            }
        )
    return {
        "cases": cases,
        "speedups": speedups,
        "median_absolute_error_percent": 100.0
        * statistics.median(abs(value) for value in all_median_errors),
        "rms_median_error_percent": 100.0
        * math.sqrt(statistics.fmean(value * value for value in all_median_errors)),
        "maximum_median_error_percent": 100.0
        * max(abs(value) for value in all_median_errors),
        "maximum_individual_error_percent": 100.0
        * max(abs(value) for value in all_individual_errors),
        "median_absolute_speedup_error_percent": 100.0
        * statistics.median(abs(value) for value in all_speedup_errors),
        "rms_speedup_error_percent": 100.0
        * math.sqrt(statistics.fmean(value * value for value in all_speedup_errors)),
        "maximum_absolute_speedup_error_percent": 100.0
        * max(abs(value) for value in all_speedup_errors),
    }


def print_speedup_report(report: dict[str, object]) -> None:
    print("[sync-calibration] holdout speedup comparison")
    print(
        f"{'case':<42} {'cores':>5} {'estimated':>10} "
        f"{'achieved':>10} {'abs error':>10}"
    )
    for result in report["speedups"]:
        print(
            f"{result['case_id']:<42} {result['parallel_cores']:>5} "
            f"{result['predicted_speedup']:>9.3f}x "
            f"{result['measured_speedup']:>9.3f}x "
            f"{result['absolute_speedup_error_percent']:>9.2f}%"
        )
    print(
        "[sync-calibration] speedup absolute error: "
        f"median={report['median_absolute_speedup_error_percent']:.2f}% "
        f"rms={report['rms_speedup_error_percent']:.2f}% "
        f"max={report['maximum_absolute_speedup_error_percent']:.2f}%"
    )


def maximum_pair_regression(
    baseline: dict[str, object],
    candidate: dict[str, object],
    points: list[ReplayPoint],
) -> float:
    baseline_cases = {
        (case["case_id"], case["cores"]): abs(case["median_error_percent"])
        for case in baseline["cases"]
    }
    candidate_cases = {
        (case["case_id"], case["cores"]): abs(case["median_error_percent"])
        for case in candidate["cases"]
    }
    regressions = []
    for pair in dict.fromkeys(point.case.pair for point in points):
        keys = [
            (point.case.case_id, cores)
            for point in points
            if point.case.pair == pair
            for cores in point.core_counts
        ]
        regressions.append(
            max(candidate_cases[key] for key in keys)
            - max(baseline_cases[key] for key in keys)
        )
    return max(regressions, default=0.0)


def leave_one_pair_out_stability(
    model: str,
    points: list[ReplayPoint],
    baseline: dict[str, float],
    fixed_handoff: float | None = None,
) -> dict[str, object]:
    cases = []
    maximum_change = 0.0
    pair_names = tuple(dict.fromkeys(point.case.pair for point in points))
    for omitted_pair in pair_names:
        retained = [point for point in points if point.case.pair != omitted_pair]
        fitted = fit_model(
            model,
            retained,
            fixed_handoff,
        )
        changes = {
            name: 100.0 * (float(value) / baseline[name] - 1.0)
            for name, value in fitted["parameters"].items()
        }
        maximum_change = max(
            maximum_change, *(abs(value) for value in changes.values())
        )
        cases.append(
            {
                "omitted_pair": omitted_pair,
                "parameter_change_percent": changes,
                "jacobian_rank": fitted["jacobian_rank"],
                "maximum_absolute_parameter_correlation": fitted[
                    "maximum_absolute_parameter_correlation"
                ],
            }
        )
    return {
        "pairs": cases,
        "maximum_absolute_parameter_change_percent": maximum_change,
    }


def dependency_profile(
    points: list[ReplayPoint], fixed_handoff: float
) -> dict[str, object]:
    samples = []
    for scale in np.linspace(0.0, 1.0, 101):
        values = np.asarray([scale])
        residuals = [
            predict(point, cores, "dependency", values, fixed_handoff)
            / point.native_medians[cores]
            - 1.0
            for point in points
            for cores in point.core_counts
        ]
        rms = math.sqrt(statistics.fmean(value * value for value in residuals))
        samples.append({"scale": float(scale), "rms_error_percent": 100.0 * rms})
    best = min(samples, key=lambda sample: sample["rms_error_percent"])
    return {"best_grid_point": best, "samples": samples}


def collect_block(
    runner: Path,
    benchmark: Path,
    output_dir: Path,
    cases: tuple[Case, ...],
    trace_trials: int,
    native_trials: int,
    event_cost_model: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    state, policies = validate_fixed_frequency()
    monitor = FrequencyMonitor(policies)
    records = []
    with monitor:
        for index, case in enumerate(cases, 1):
            print(f"[sync-calibration] {index}/{len(cases)} {case.case_id}", flush=True)
            path = collect_case(
                runner,
                benchmark,
                output_dir,
                case,
                trace_trials,
                native_trials,
                event_cost_model,
            )
            records.append({"case": asdict(case), "result_path": str(path)})
    postflight, _ = validate_fixed_frequency()
    frequency = monitor.report(int(state["target_khz"]))
    frequency["postflight"] = postflight
    return records, {**state, **frequency}


def load_points(
    records: list[dict[str, object]], event_model: dict[str, object]
) -> list[ReplayPoint]:
    return [
        load_point(
            Case(**record["case"]), Path(record["result_path"]), event_model
        )
        for record in records
    ]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-trials", type=int, default=3)
    parser.add_argument("--native-trials", type=int, default=5)
    parser.add_argument("--event-cost-model", required=True, type=Path)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--base-observations", type=Path)
    parser.add_argument("--refresh-startup", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topology = validate_topology()
    os.sched_setaffinity(0, {MONITOR_CPU})

    observations_path = args.output_dir / "observations.json"
    if args.reuse and args.base_observations is not None:
        raise RuntimeError("--reuse and --base-observations are mutually exclusive")
    if args.reuse:
        observations = json.loads(observations_path.read_text())
        if args.refresh_startup:
            startup, startup_frequency = collect_worker_startup(
                args.benchmark, args.native_trials
            )
            observations["startup"] = startup
            observations["frequency"]["startup"] = startup_frequency
            write_json(observations_path, observations)
    elif args.refresh_startup:
        raise RuntimeError("--refresh-startup requires --reuse")
    elif args.base_observations is not None:
        observations = json.loads(args.base_observations.read_text())
        if observations.get("renderer_inputs_used") is not False:
            raise RuntimeError("base observations are not synthetic-only")
        dependency_fit, dependency_fit_frequency = collect_block(
            args.runner,
            args.benchmark,
            args.output_dir,
            DEPENDENCY_FIT_CASES,
            args.trace_trials,
            args.native_trials,
            args.event_cost_model,
        )
        dependency_validation, dependency_validation_frequency = collect_block(
            args.runner,
            args.benchmark,
            args.output_dir,
            DEPENDENCY_VALIDATION_CASES,
            args.trace_trials,
            args.native_trials,
            args.event_cost_model,
        )
        dependency_holdout, dependency_holdout_frequency = collect_block(
            args.runner,
            args.benchmark,
            args.output_dir,
            DEPENDENCY_HOLDOUT_CASES,
            args.trace_trials,
            args.native_trials,
            args.event_cost_model,
        )
        observations["schema_version"] = 3
        observations["dependency_fit"] = dependency_fit
        observations["dependency_validation"] = dependency_validation
        observations["dependency_holdout"] = dependency_holdout
        observations["frequency"] = {
            **observations.get("frequency", {}),
            "dependency_fit": dependency_fit_frequency,
            "dependency_validation": dependency_validation_frequency,
            "dependency_holdout": dependency_holdout_frequency,
        }
        write_json(observations_path, observations)
    else:
        startup, startup_frequency = collect_worker_startup(
            args.benchmark, args.native_trials
        )
        fit, fit_frequency = collect_block(
            args.runner,
            args.benchmark,
            args.output_dir,
            FIT_CASES,
            args.trace_trials,
            args.native_trials,
            args.event_cost_model,
        )
        holdout, holdout_frequency = collect_block(
            args.runner,
            args.benchmark,
            args.output_dir,
            HOLDOUT_CASES,
            args.trace_trials,
            args.native_trials,
            args.event_cost_model,
        )
        dependency_fit, dependency_fit_frequency = collect_block(
            args.runner,
            args.benchmark,
            args.output_dir,
            DEPENDENCY_FIT_CASES,
            args.trace_trials,
            args.native_trials,
            args.event_cost_model,
        )
        dependency_validation, dependency_validation_frequency = collect_block(
            args.runner,
            args.benchmark,
            args.output_dir,
            DEPENDENCY_VALIDATION_CASES,
            args.trace_trials,
            args.native_trials,
            args.event_cost_model,
        )
        dependency_holdout, dependency_holdout_frequency = collect_block(
            args.runner,
            args.benchmark,
            args.output_dir,
            DEPENDENCY_HOLDOUT_CASES,
            args.trace_trials,
            args.native_trials,
            args.event_cost_model,
        )
        observations = {
            "schema_version": 3,
            "source": "bench_thread_pool_dispatch_only",
            "renderer_inputs_used": False,
            "startup": startup,
            "fit": fit,
            "holdout": holdout,
            "dependency_fit": dependency_fit,
            "dependency_validation": dependency_validation,
            "dependency_holdout": dependency_holdout,
            "frequency": {
                "startup": startup_frequency,
                "fit": fit_frequency,
                "holdout": holdout_frequency,
                "dependency_fit": dependency_fit_frequency,
                "dependency_validation": dependency_validation_frequency,
                "dependency_holdout": dependency_holdout_frequency,
            },
            "topology": topology,
        }
        write_json(observations_path, observations)

    event_model = json.loads(args.event_cost_model.read_text())
    startup_fit = fit_worker_startup(observations["startup"])
    fit_points = load_points(observations["fit"], event_model)
    holdout_points = load_points(observations["holdout"], event_model)
    dependency_fit_points = load_points(observations["dependency_fit"], event_model)
    dependency_validation_points = load_points(
        observations["dependency_validation"], event_model
    )
    dependency_holdout_points = load_points(
        observations["dependency_holdout"], event_model
    )

    baseline = fit_model("handoff", fit_points)
    baseline_values = baseline.pop("values")
    fixed_handoff = float(baseline["parameters"]["cross_thread_release_ns"])
    fitted = fit_model("dependency", dependency_fit_points, fixed_handoff)
    values = fitted.pop("values")

    baseline_existing = evaluate("handoff", baseline_values, holdout_points)
    baseline_dependency_fit = evaluate(
        "handoff", baseline_values, dependency_fit_points
    )
    baseline_validation = evaluate(
        "handoff", baseline_values, dependency_validation_points
    )
    baseline_holdout = evaluate(
        "handoff", baseline_values, dependency_holdout_points
    )
    candidate_existing = evaluate(
        "dependency", values, holdout_points, fixed_handoff
    )
    candidate_dependency_fit = evaluate(
        "dependency", values, dependency_fit_points, fixed_handoff
    )
    candidate_validation = evaluate(
        "dependency", values, dependency_validation_points, fixed_handoff
    )
    candidate_holdout = evaluate(
        "dependency", values, dependency_holdout_points, fixed_handoff
    )
    diagnostics = {
        "handoff_only": {
            "fit": baseline,
            "existing_holdout": baseline_existing,
            "dependency_fit": baseline_dependency_fit,
            "dependency_validation": baseline_validation,
            "dependency_holdout": baseline_holdout,
        },
        "dependency_excess": {
            "fit": fitted,
            "existing_holdout": candidate_existing,
            "dependency_fit": candidate_dependency_fit,
            "dependency_validation": candidate_validation,
            "dependency_holdout": candidate_holdout,
        },
    }

    candidate = diagnostics["dependency_excess"]
    fit_report = candidate["fit"]
    stability = leave_one_pair_out_stability(
        "dependency",
        dependency_fit_points,
        fit_report["parameters"],
        fixed_handoff,
    )
    fit_report["leave_one_pair_out"] = stability
    fit_report["profile"] = dependency_profile(
        dependency_fit_points, fixed_handoff
    )
    pair_regression = maximum_pair_regression(
        baseline_validation,
        candidate_validation,
        dependency_validation_points,
    )
    runtime_rms_improvement = 1.0 - (
        candidate_validation["rms_median_error_percent"]
        / baseline_validation["rms_median_error_percent"]
    )
    speedup_rms_regression = (
        candidate_validation["rms_speedup_error_percent"]
        - baseline_validation["rms_speedup_error_percent"]
    )
    candidate_valid = (
        fit_report["jacobian_rank"] == fit_report["parameter_count"]
        and not fit_report["parameters_hit_bounds"]
        and stability["maximum_absolute_parameter_change_percent"]
        <= MAX_STABILITY_MOVEMENT_PERCENT
        and candidate_validation["median_absolute_error_percent"]
        <= MAX_HOLDOUT_ERROR_PERCENT
        and candidate_validation["rms_speedup_error_percent"]
        <= MAX_HOLDOUT_ERROR_PERCENT
        and candidate_validation["maximum_absolute_speedup_error_percent"]
        <= MAX_HOLDOUT_SPEEDUP_ERROR_PERCENT
        and runtime_rms_improvement >= 0.10
        and speedup_rms_regression <= 2.0
        and pair_regression <= 5.0
        and candidate_existing["rms_median_error_percent"]
        <= baseline_existing["rms_median_error_percent"] + 5.0
        and candidate_existing["rms_speedup_error_percent"]
        <= baseline_existing["rms_speedup_error_percent"] + 2.0
        and bool(observations["frequency"]["fit"]["within_tolerance"])
        and bool(observations["frequency"]["holdout"]["within_tolerance"])
        and bool(observations["frequency"]["startup"]["within_tolerance"])
        and bool(
            observations["frequency"]["dependency_fit"]["within_tolerance"]
        )
        and bool(
            observations["frequency"]["dependency_validation"]["within_tolerance"]
        )
        and bool(
            observations["frequency"]["dependency_holdout"]["within_tolerance"]
        )
        and startup_fit["maximum_median_error_percent"]
        <= MAX_HOLDOUT_ERROR_PERCENT
    )
    candidate["selection"] = {
        "runtime_rms_relative_improvement": runtime_rms_improvement,
        "speedup_rms_regression_percentage_points": speedup_rms_regression,
        "maximum_pair_regression_percentage_points": pair_regression,
        "sealed_holdout_used_for_selection": False,
    }
    model_valid = bool(
        event_model.get("synthetic_calibration_valid", False)
    ) and candidate_valid
    selected_name = "dependency_excess" if candidate_valid else "handoff_only"
    selected_holdout_report = (
        candidate_validation if candidate_valid else baseline_validation
    )
    combined_parameters = {
        f"{name}_ns": float(value)
        for name, value in zip(
            event_model["feature_names"],
            event_model["coefficients_ns"],
            strict=True,
        )
    }
    combined_parameters["fixed_process_ns"] = float(
        startup_fit["base_process_ns"]
    )
    combined_parameters["per_worker_startup_ns"] = float(
        startup_fit["per_worker_startup_ns"]
    )
    combined_parameters.update(baseline["parameters"])
    if candidate_valid:
        combined_parameters.update(fit_report["parameters"])
    model = {
        "schema_version": 3,
        "source": "bench_thread_pool_dispatch_only",
        "renderer_inputs_used": False,
        "tested_candidate": "dependency_excess",
        "candidate_accepted": candidate_valid,
        "selected_candidate": selected_name,
        "parameter_count": len(combined_parameters),
        "parameters": combined_parameters,
        "event_cost_model": event_model,
        "diagnostics": diagnostics,
        "worker_startup": startup_fit,
        "synthetic_calibration_valid": model_valid,
        "timing_claims_enabled": False,
    }
    write_json(args.output_dir / "model.json", model)
    print_speedup_report(selected_holdout_report)
    print(json.dumps(model, indent=2, sort_keys=True))
    return 0 if model_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
