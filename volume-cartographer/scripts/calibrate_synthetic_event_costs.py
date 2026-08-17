#!/usr/bin/env python3
"""Calibrate passive Callgrind event costs using generic synthetic work only."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, linprog

from calibrate_thread_dispatch_shared import (
    FrequencyMonitor,
    MONITOR_CPU,
    validate_fixed_frequency,
    validate_topology,
)
from run_render_callgrind import CACHE_GEOMETRY
from run_thread_sync_replay import parse_thread_profiles
from passive_event_model import (
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    features_for_model,
    modeled_feature_cost_ns,
)


EVENT_NAMES = (
    "Ir", "Dr", "Dw", "Bc", "Bi", "I1mr", "D1mr", "D1mw",
    "ILmr", "DLmr", "DLmw", "Bcm", "Bim",
)
CALIBRATION_CPU = 0
MAX_HOLDOUT_MEDIAN_ERROR_PERCENT = 20.0
MAX_MIXED_HOLDOUT_ERROR_PERCENT = 30.0
MAX_SERIALIZATION_HOLDOUT_ERROR_PERCENT = 30.0
MAX_CONTROL_REGRESSION_PERCENTAGE_POINTS = 5.0


@dataclass(frozen=True)
class WorkCase:
    kind: str
    working_set_bytes: int
    iterations: int
    rounds: int
    warmup_rounds: int
    role: str

    @property
    def case_id(self) -> str:
        return (
            f"{self.role}-{self.kind}-s{self.working_set_bytes}"
            f"-i{self.iterations}-r{self.rounds}-u{self.warmup_rounds}"
        )


def cases_for(role: str) -> tuple[WorkCase, ...]:
    if role == "fit":
        kinds = {
            "branch": 40_000, "stream-read": 250_000,
            "stream-write": 250_000, "grid-sample": 60_000,
        }
        sizes = (16_384, 65_536, 262_144, 1_048_576, 4_194_304, 12_582_912)
        specs = tuple(
            (kind, size, base * multiplier, 5 + level * 2, level)
            for kind, base in kinds.items()
            for size in sizes
            for level, multiplier in enumerate((1, 3, 7))
        )
    elif role == "cache_fit":
        sizes = (
            16_384, 65_536, 262_144, 1_048_576,
            4_194_304, 12_582_912, 33_554_432,
        )
        specs = tuple(
            (kind, size, 80_000 * multiplier, 5 + level * 2, level)
            for kind in ("cache-read",)
            for size in sizes
            for level, multiplier in enumerate((1, 3, 7))
        )
    elif role == "holdout":
        kinds = {
            "branch": 180_000, "stream-read": 1_125_000,
            "stream-write": 1_125_000, "grid-sample": 270_000,
        }
        sizes = (32_768, 131_072, 524_288, 2_097_152, 8_388_608)
        specs = tuple(
            (kind, size, iterations, 8, 2)
            for kind, iterations in kinds.items()
            for size in sizes
        )
    elif role == "cache_holdout":
        specs = tuple(
            (kind, size, 360_000, 8, 2)
            for kind in ("cache-read",)
            for size in (
                32_768, 131_072, 524_288, 2_097_152,
                8_388_608, 20_971_520,
            )
        )
    elif role == "mixed_fit":
        sizes = (16_384, 65_536, 262_144, 1_048_576, 4_194_304, 12_582_912)
        specs = tuple(
            (kind, size, 60_000 * multiplier, 5 + level * 2, level)
            for kind in ("mixed-grid-phase", "mixed-grid-random")
            for size in sizes
            for level, multiplier in enumerate((1, 3, 7))
        )
    elif role == "mixed_holdout":
        specs = tuple(
            (kind, size, 270_000, 8, 2)
            for kind in ("mixed-grid-phase", "mixed-grid-random")
            for size in (32_768, 131_072, 524_288, 2_097_152, 8_388_608)
        )
    elif role == "serialization_holdout":
        specs = tuple(
            ("pointer", size, 190_000, 8, 2)
            for size in (49_152, 196_608, 786_432, 3_145_728, 10_485_760)
        ) + tuple(
            (kind, size, 190_000, 8, 2)
            for kind in (
                "cache-read", "grid-sample",
                "mixed-grid-phase", "mixed-grid-random",
            )
            for size in (49_152, 3_145_728)
        )
    else:
        kinds = {
            "alu": 450_000, "fp": 450_000,
            "pointer": 270_000, "mixed": 135_000,
        }
        sizes = (32_768, 131_072, 524_288, 2_097_152, 8_388_608)
        specs = tuple(
            (kind, size, iterations, 8, 2)
            for kind, iterations in kinds.items()
            for size in sizes
        )
    return tuple(WorkCase(*spec, role) for spec in specs)


FIT_CASES = cases_for("fit")
CACHE_FIT_CASES = cases_for("cache_fit")
HOLDOUT_CASES = cases_for("holdout")
CACHE_HOLDOUT_CASES = cases_for("cache_holdout")
MIXED_FIT_CASES = cases_for("mixed_fit")
MIXED_HOLDOUT_CASES = cases_for("mixed_holdout")
SERIALIZATION_HOLDOUT_CASES = cases_for("serialization_holdout")
DIAGNOSTIC_CASES = cases_for("diagnostic")
STARTUP_CASE = WorkCase("alu", 512, 0, 1, 0, "startup")


def benchmark_command(binary: Path, case: WorkCase) -> list[str]:
    return [
        str(binary), "--mode", "serial", "--workers", "1", "--tasks", "1",
        "--work-kind", case.kind, "--working-set-bytes", str(case.working_set_bytes),
        "--work-iterations", str(case.iterations), "--rounds", str(case.rounds),
        "--warmup-rounds", str(case.warmup_rounds),
    ]


def run(command: list[str], cpu: int = CALIBRATION_CPU) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["taskset", "-c", str(cpu), *command], check=True, text=True, capture_output=True
    )


def collect_case(binary: Path, output_dir: Path, case: WorkCase, trials: int) -> dict[str, object]:
    command = benchmark_command(binary, case)
    native_process = []
    native_work = []
    checksums = set()
    for _ in range(trials):
        start = time.perf_counter_ns()
        completed = run(command)
        native_process.append(float(time.perf_counter_ns() - start))
        output = json.loads(completed.stdout)
        native_work.append(float(output["wall_seconds"]) * 1e9)
        checksums.add(int(output["checksum"]))
    if len(checksums) != 1:
        raise RuntimeError(f"nondeterministic checksum for {case.case_id}")
    callgrind_path = output_dir / f"callgrind.{case.case_id}"
    callgrind_command = [
        "valgrind", "--tool=callgrind", "--instr-atstart=no",
        "--collect-systime=no", "--cache-sim=yes", "--separate-threads=yes",
        "--branch-sim=yes", f"--I1={CACHE_GEOMETRY['I1']}",
        f"--D1={CACHE_GEOMETRY['D1']}", f"--LL={CACHE_GEOMETRY['LL']}",
        f"--callgrind-out-file={callgrind_path}", *command,
    ]
    run(callgrind_command)
    profiles = parse_thread_profiles(callgrind_path)
    return {
        "case": asdict(case), "case_id": case.case_id,
        "native_process_samples_ns": native_process,
        "native_process_median_ns": statistics.median(native_process),
        "native_work_samples_ns": native_work,
        "native_work_median_ns": statistics.median(native_work),
        "profiles": {
            str(thread): {name: int(events[name]) for name in EVENT_NAMES}
            for thread, events in profiles.items()
        },
        "command": command, "callgrind_command": callgrind_command,
    }


def collect_startup(binary: Path, trials: int) -> dict[str, object]:
    command = benchmark_command(binary, STARTUP_CASE)
    process_samples = []
    work_samples = []
    overhead_samples = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        completed = run(command)
        process_ns = float(time.perf_counter_ns() - start)
        work_ns = float(json.loads(completed.stdout)["wall_seconds"]) * 1e9
        process_samples.append(process_ns)
        work_samples.append(work_ns)
        overhead_samples.append(max(0.0, process_ns - work_ns))
    return {
        "case": asdict(STARTUP_CASE),
        "command": command,
        "trials": trials,
        "process_samples_ns": process_samples,
        "work_samples_ns": work_samples,
        "overhead_samples_ns": overhead_samples,
        "fixed_process_ns": statistics.median(overhead_samples),
    }


def features(
    record: dict[str, object], feature_names: tuple[str, ...] = FEATURE_NAMES
) -> np.ndarray:
    per_thread = [
        features_for_model(profile, feature_names)
        for profile in record["profiles"].values()
    ]
    if not per_thread:
        raise RuntimeError(f"case {record['case_id']} has no thread profiles")
    return np.sum(per_thread, axis=0)


def fit(
    records: list[dict[str, object]],
    fixed_process_ns: float,
    allow_overlap: bool = True,
    include_serial_pressure: bool = False,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if feature_names is not None and include_serial_pressure:
        raise RuntimeError("explicit features and serialization pressure are exclusive")
    if allow_overlap and include_serial_pressure:
        raise RuntimeError("serialization pressure cannot be fit with overlap")
    if feature_names is None:
        feature_names = FEATURE_NAMES if include_serial_pressure else LEGACY_FEATURE_NAMES
    if allow_overlap and feature_names != LEGACY_FEATURE_NAMES:
        raise RuntimeError("extended feature bases require zero overlap")
    matrix = np.asarray([features(record, feature_names) for record in records])
    measured = np.asarray(
        [record["native_work_median_ns"] for record in records], dtype=float
    )
    family_counts = {
        role: sum(record["case"]["role"] == role for record in records)
        for role in {record["case"]["role"] for record in records}
    }
    weights = np.asarray(
        [1.0 / math.sqrt(family_counts[record["case"]["role"]]) for record in records]
    )
    weights /= np.mean(weights)
    scales = np.maximum(np.median(matrix, axis=0), 1.0)

    parameter_count = matrix.shape[1] + int(allow_overlap)

    def residuals(values: np.ndarray) -> np.ndarray:
        coefficients = values[: matrix.shape[1]] / scales
        overlap = float(values[-1]) if allow_overlap else 0.0
        predicted = np.asarray(
            [modeled_feature_cost_ns(row, coefficients, overlap) for row in matrix]
        )
        data = weights * (predicted / measured - 1.0)
        ridge = 0.03 * values[: matrix.shape[1]] / statistics.median(measured)
        return np.concatenate((data, ridge))

    initial = np.full(matrix.shape[1], statistics.median(measured) / matrix.shape[1])
    lower = np.zeros(parameter_count)
    upper = np.full(parameter_count, np.inf)
    if allow_overlap:
        initial = np.append(initial, 0.5)
        upper[-1] = 1.0
    result = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        loss="soft_l1", f_scale=0.03, x_scale="jac", max_nfev=2000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    coefficients = result.x[: matrix.shape[1]] / scales
    overlap = float(result.x[-1]) if allow_overlap else 0.0
    jacobian = np.asarray(result.jac)[: len(records)]
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = covariance / np.outer(diagonal, diagonal)
    return {
        "feature_names": list(feature_names),
        "basis": (
            "passive event costs with L1 miss serialization pressure"
            if include_serial_pressure
            else (
                "legacy passive event costs"
                if feature_names == LEGACY_FEATURE_NAMES
                else "named passive event costs"
            )
        ),
        "coefficients_ns": list(map(float, coefficients)),
        "stall_overlap_fraction": overlap,
        "feature_scales": list(map(float, scales)),
        "fixed_process_ns": float(fixed_process_ns),
        "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
        "event_parameter_count": len(result.x),
        "parameter_count": len(result.x) + 1,
        "bound_hit_count": int(
            np.count_nonzero(result.x <= 1e-12)
            + np.count_nonzero(np.isfinite(upper) & np.isclose(result.x, upper))
        ),
        "maximum_absolute_parameter_correlation": max(
            abs(float(correlation[i, j]))
            for i in range(len(result.x)) for j in range(i)
        ),
    }


def fit_minimax(
    records: list[dict[str, object]],
    fixed_process_ns: float,
    reference_coefficients: list[float],
) -> dict[str, object]:
    """Fit the legacy basis by minimizing maximum per-case relative error."""
    matrix = np.asarray(
        [features(record, LEGACY_FEATURE_NAMES) for record in records], dtype=float
    )
    measured = np.asarray(
        [record["native_work_median_ns"] for record in records], dtype=float
    )
    relative_matrix = matrix / measured[:, np.newaxis]
    column_norms = np.linalg.norm(relative_matrix, axis=0)
    if np.any(column_norms <= np.finfo(float).eps):
        raise RuntimeError("minimax fit has an empty feature column")
    normalized = relative_matrix / column_norms
    feature_count = normalized.shape[1]
    objective = np.zeros(feature_count + 1)
    objective[-1] = 1.0
    upper = np.column_stack((normalized, -np.ones(len(records))))
    lower = np.column_stack((-normalized, -np.ones(len(records))))
    constraints = np.vstack((upper, lower))
    limits = np.concatenate((np.ones(len(records)), -np.ones(len(records))))
    optimum = linprog(
        objective,
        A_ub=constraints,
        b_ub=limits,
        bounds=[(0.0, None)] * (feature_count + 1),
        method="highs",
    )
    if not optimum.success:
        raise RuntimeError(f"minimax fit failed: {optimum.message}")
    optimum_error = float(optimum.x[-1])
    face_tolerance = max(1e-10, 1e-8 * max(1.0, optimum_error))
    face_limits = np.concatenate(
        (
            np.full(len(records), 1.0 + optimum_error + face_tolerance),
            np.full(len(records), -(1.0 - optimum_error - face_tolerance)),
        )
    )
    face_constraints = np.vstack((normalized, -normalized))
    coefficient_bounds = [(0.0, None)] * feature_count

    intervals = []
    for index in range(feature_count):
        direction = np.zeros(feature_count)
        direction[index] = 1.0
        minimum = linprog(
            direction,
            A_ub=face_constraints,
            b_ub=face_limits,
            bounds=coefficient_bounds,
            method="highs",
        )
        maximum = linprog(
            -direction,
            A_ub=face_constraints,
            b_ub=face_limits,
            bounds=coefficient_bounds,
            method="highs",
        )
        if not minimum.success or not maximum.success:
            raise RuntimeError("failed to bound the minimax optimal face")
        intervals.append((float(minimum.fun), float(-maximum.fun)))

    reference = np.asarray(reference_coefficients, dtype=float)
    if reference.shape != (feature_count,):
        raise RuntimeError("minimax reference coefficient count does not match")
    reference_normalized = reference * column_norms
    lexicographic_constraints = face_constraints.copy()
    lexicographic_limits = face_limits.copy()
    for index in range(feature_count):
        deviation_objective = np.zeros(feature_count + 1)
        deviation_objective[-1] = 1.0
        deviation_constraints = np.pad(
            lexicographic_constraints, ((0, 0), (0, 1))
        )
        positive = np.zeros(feature_count + 1)
        positive[index], positive[-1] = 1.0, -1.0
        negative = np.zeros(feature_count + 1)
        negative[index], negative[-1] = -1.0, -1.0
        deviation_constraints = np.vstack(
            (deviation_constraints, positive, negative)
        )
        deviation_limits = np.concatenate(
            (
                lexicographic_limits,
                [reference_normalized[index], -reference_normalized[index]],
            )
        )
        deviation = linprog(
            deviation_objective,
            A_ub=deviation_constraints,
            b_ub=deviation_limits,
            bounds=[*coefficient_bounds, (0.0, None)],
            method="highs",
        )
        if not deviation.success:
            raise RuntimeError("minimax lexicographic deviation became infeasible")
        deviation_tolerance = max(
            1e-10, 1e-8 * max(1.0, abs(float(deviation.fun)))
        )
        positive = np.zeros(feature_count)
        positive[index] = 1.0
        negative = -positive
        lexicographic_constraints = np.vstack(
            (lexicographic_constraints, positive, negative)
        )
        lexicographic_limits = np.concatenate(
            (
                lexicographic_limits,
                [
                    reference_normalized[index] + deviation.fun + deviation_tolerance,
                    -reference_normalized[index] + deviation.fun + deviation_tolerance,
                ],
            )
        )
        direction = np.zeros(feature_count)
        direction[index] = 1.0
        lower_tie_break = linprog(
            direction,
            A_ub=lexicographic_constraints,
            b_ub=lexicographic_limits,
            bounds=coefficient_bounds,
            method="highs",
        )
        if not lower_tie_break.success:
            raise RuntimeError("minimax lower-coefficient tie-break became infeasible")
        coefficient_tolerance = max(
            1e-10, 1e-8 * max(1.0, abs(float(lower_tie_break.fun)))
        )
        lexicographic_constraints = np.vstack(
            (lexicographic_constraints, direction)
        )
        lexicographic_limits = np.concatenate(
            (lexicographic_limits, [lower_tie_break.fun + coefficient_tolerance])
        )
    feasible = linprog(
        np.zeros(feature_count),
        A_ub=lexicographic_constraints,
        b_ub=lexicographic_limits,
        bounds=coefficient_bounds,
        method="highs",
    )
    if not feasible.success:
        raise RuntimeError("minimax lexicographic tie-break became infeasible")
    selected = feasible.x / column_norms
    coefficient_intervals = [
        {
            "minimum_ns": minimum / column_norms[index],
            "maximum_ns": maximum / column_norms[index],
            "normalized_width": (
                (maximum - minimum)
                / max(abs(feasible.x[index]), abs(reference_normalized[index]), 1e-12)
            ),
        }
        for index, (minimum, maximum) in enumerate(intervals)
    ]

    singular_values = np.linalg.svd(normalized, compute_uv=False)
    covariance = np.linalg.pinv(normalized.T @ normalized)
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = covariance / np.outer(diagonal, diagonal)
    maximum_correlation = max(
        abs(float(correlation[i, j]))
        for i in range(feature_count) for j in range(i)
    )
    return {
        "feature_names": list(LEGACY_FEATURE_NAMES),
        "basis": "legacy passive event costs",
        "coefficients_ns": list(map(float, selected)),
        "stall_overlap_fraction": 0.0,
        "feature_scales": list(map(float, 1.0 / column_norms)),
        "fixed_process_ns": float(fixed_process_ns),
        "jacobian_rank": int(np.linalg.matrix_rank(normalized)),
        "event_parameter_count": feature_count,
        "parameter_count": feature_count + 1,
        "bound_hit_count": int(np.count_nonzero(selected <= 1e-12)),
        "maximum_absolute_parameter_correlation": maximum_correlation,
        "normalized_singular_values": list(map(float, singular_values)),
        "normalized_condition_number": float(singular_values[0] / singular_values[-1]),
        "objective": "unweighted maximum absolute per-case relative error",
        "optimal_maximum_relative_error": optimum_error,
        "optimal_face_tolerance": face_tolerance,
        "coefficient_intervals": coefficient_intervals,
        "tie_break": (
            "lexicographic minimum normalized absolute deviation from matched "
            "soft-L1 coefficients in feature order; lower coefficient resolves "
            "equal-distance ties"
        ),
    }


def evaluate(model: dict[str, object], records: list[dict[str, object]]) -> dict[str, object]:
    coefficients = np.asarray(model["coefficients_ns"])
    feature_names = tuple(model["feature_names"])
    rows = []
    for record in records:
        predicted = modeled_feature_cost_ns(
            features(record, feature_names),
            coefficients,
            float(model.get("stall_overlap_fraction", 0.0)),
        )
        measured = float(record["native_work_median_ns"])
        error = 100.0 * (predicted / measured - 1.0)
        rows.append({"case_id": record["case_id"], "predicted_ns": predicted,
                     "measured_ns": measured, "error_percent": error,
                     "absolute_error_percent": abs(error)})
    absolute = [row["absolute_error_percent"] for row in rows]
    return {"cases": rows, "median_absolute_error_percent": statistics.median(absolute),
            "rms_error_percent": math.sqrt(statistics.fmean(row["error_percent"] ** 2 for row in rows)),
            "maximum_absolute_error_percent": max(absolute)}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def promote_pointer_fit_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    promoted = []
    for record in records:
        if record["case"]["kind"] != "pointer":
            continue
        value = copy.deepcopy(record)
        value["source_case_id"] = value["case_id"]
        value["case"]["role"] = "serialization_fit"
        value["case_id"] = value["case_id"].replace(
            "diagnostic-pointer", "serialization_fit-pointer", 1
        )
        promoted.append(value)
    if not promoted:
        raise RuntimeError("opened pointer diagnostics are required for fitting")
    return promoted


def all_frequency_reports_valid(value: object) -> bool:
    reports = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if "within_tolerance" in node:
                reports.append(bool(node["within_tolerance"]))
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return bool(reports) and all(reports)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-trials", type=int, default=5)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--base-observations", type=Path)
    parser.add_argument(
        "--extension", choices=("mixed", "serialization"), default="mixed"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_topology()
    os.sched_setaffinity(0, {MONITOR_CPU})
    observations_path = args.output_dir / "observations.json"
    if args.reuse and args.base_observations is not None:
        raise RuntimeError("--reuse and --base-observations are mutually exclusive")
    if args.reuse:
        observations = json.loads(observations_path.read_text())
    elif args.base_observations is not None:
        observations = json.loads(args.base_observations.read_text())
        if observations.get("renderer_inputs_used") is not False:
            raise RuntimeError("base observations are not synthetic-only")
        state, policies = validate_fixed_frequency()
        monitor = FrequencyMonitor(policies)
        if args.extension == "mixed":
            mixed_fit_records = []
            mixed_holdout_records = []
            all_cases = (*MIXED_FIT_CASES, *MIXED_HOLDOUT_CASES)
            with monitor:
                for index, case in enumerate(all_cases, 1):
                    print(
                        f"[event-calibration] {index}/{len(all_cases)} {case.case_id}",
                        flush=True,
                    )
                    target = (
                        mixed_fit_records
                        if case.role == "mixed_fit"
                        else mixed_holdout_records
                    )
                    target.append(
                        collect_case(
                            args.benchmark, args.output_dir, case, args.native_trials
                        )
                    )
        else:
            if not observations.get("mixed_fit") or not observations.get(
                "mixed_holdout"
            ):
                raise RuntimeError(
                    "serialization extension requires existing mixed observations"
                )
            serialization_holdout_records = []
            all_cases = SERIALIZATION_HOLDOUT_CASES
            with monitor:
                for index, case in enumerate(all_cases, 1):
                    print(
                        f"[event-calibration] {index}/{len(all_cases)} {case.case_id}",
                        flush=True,
                    )
                    serialization_holdout_records.append(
                        collect_case(
                            args.benchmark, args.output_dir, case, args.native_trials
                        )
                    )
        postflight, _ = validate_fixed_frequency()
        extension_frequency = {
            **state,
            **monitor.report(int(state["target_khz"])),
            "postflight": postflight,
        }
        if args.extension == "mixed":
            observations["schema_version"] = 2
            observations["mixed_fit"] = mixed_fit_records
            observations["mixed_holdout"] = mixed_holdout_records
            observations["frequency"] = {
                "base": observations.get("frequency"),
                "mixed_extension": extension_frequency,
            }
        else:
            observations["schema_version"] = 3
            observations["serialization_fit"] = promote_pointer_fit_records(
                observations.get("diagnostic", [])
            )
            observations["serialization_holdout"] = (
                serialization_holdout_records
            )
            observations["frequency"] = {
                "base_fit": observations.get("frequency"),
                "serialization_holdout": extension_frequency,
            }
        write_json(observations_path, observations)
    else:
        state, policies = validate_fixed_frequency()
        monitor = FrequencyMonitor(policies)
        fit_records = []
        cache_fit_records = []
        holdout_records = []
        cache_holdout_records = []
        mixed_fit_records = []
        mixed_holdout_records = []
        serialization_holdout_records = []
        with monitor:
            startup = collect_startup(args.benchmark, max(20, args.native_trials))
            diagnostic_records = []
            all_cases = (
                *FIT_CASES, *CACHE_FIT_CASES, *HOLDOUT_CASES,
                *CACHE_HOLDOUT_CASES, *MIXED_FIT_CASES,
                *MIXED_HOLDOUT_CASES, *DIAGNOSTIC_CASES,
                *SERIALIZATION_HOLDOUT_CASES,
            )
            for index, case in enumerate(all_cases, 1):
                print(f"[event-calibration] {index}/{len(all_cases)} {case.case_id}", flush=True)
                target = {
                    "fit": fit_records,
                    "cache_fit": cache_fit_records,
                    "holdout": holdout_records,
                    "cache_holdout": cache_holdout_records,
                    "mixed_fit": mixed_fit_records,
                    "mixed_holdout": mixed_holdout_records,
                    "diagnostic": diagnostic_records,
                    "serialization_holdout": serialization_holdout_records,
                }[case.role]
                target.append(collect_case(args.benchmark, args.output_dir, case, args.native_trials))
        postflight, _ = validate_fixed_frequency()
        frequency = {**state, **monitor.report(int(state["target_khz"])), "postflight": postflight}
        observations = {
            "schema_version": 3,
            "source": "generic_synthetic_work_only",
            "renderer_inputs_used": False,
            "startup": startup,
            "fit": fit_records,
            "cache_fit": cache_fit_records,
            "holdout": holdout_records,
            "cache_holdout": cache_holdout_records,
            "mixed_fit": mixed_fit_records,
            "mixed_holdout": mixed_holdout_records,
            "serialization_fit": promote_pointer_fit_records(
                diagnostic_records
            ),
            "serialization_holdout": serialization_holdout_records,
            "diagnostic": diagnostic_records,
            "frequency": frequency,
        }
        write_json(observations_path, observations)
    baseline_fit_records = [
        record for record in observations["fit"]
        if not record["case"]["kind"].startswith("cache-")
    ]
    cache_fit_records = observations.get("cache_fit") or [
        record for record in observations["fit"]
        if record["case"]["kind"].startswith("cache-")
    ]
    baseline_holdout_records = [
        record for record in observations["holdout"]
        if not record["case"]["kind"].startswith("cache-")
    ]
    cache_holdout_records = observations.get("cache_holdout") or [
        record for record in observations["holdout"]
        if record["case"]["kind"].startswith("cache-")
    ]
    mixed_fit_records = observations.get("mixed_fit", [])
    mixed_holdout_records = observations.get("mixed_holdout", [])
    if not mixed_fit_records or not mixed_holdout_records:
        raise RuntimeError("mixed fit and holdout observations are required")
    serialization_fit_records = observations.get("serialization_fit") or (
        promote_pointer_fit_records(observations.get("diagnostic", []))
    )
    serialization_holdout_records = observations.get(
        "serialization_holdout", []
    )
    if not serialization_holdout_records:
        raise RuntimeError("fresh serialization holdout observations are required")
    fit_records = (
        baseline_fit_records + cache_fit_records + mixed_fit_records
        + serialization_fit_records
    )
    holdout_records = (
        baseline_holdout_records + cache_holdout_records + mixed_holdout_records
        + serialization_holdout_records
    )
    startup = observations["startup"]
    model = fit(
        fit_records,
        float(startup["fixed_process_ns"]),
        allow_overlap=False,
        include_serial_pressure=True,
    )
    baseline = fit(
        fit_records,
        float(startup["fixed_process_ns"]),
        allow_overlap=False,
        include_serial_pressure=False,
    )
    model["fit"] = evaluate(model, fit_records)
    model["holdout"] = evaluate(model, holdout_records)
    model["fit_by_workload_family"] = {
        "baseline_work": evaluate(model, baseline_fit_records),
        "cache_stress": evaluate(model, cache_fit_records),
        "mixed_grid": evaluate(model, mixed_fit_records),
        "serialization": evaluate(model, serialization_fit_records),
    }
    model["holdout_by_workload_family"] = {
        "baseline_work": evaluate(model, baseline_holdout_records),
        "cache_stress": evaluate(model, cache_holdout_records),
        "mixed_grid": evaluate(model, mixed_holdout_records),
        "serialization": evaluate(model, serialization_holdout_records),
    }
    model["startup"] = {
        "case": startup["case"],
        "trials": startup["trials"],
        "fixed_process_ns": startup["fixed_process_ns"],
        "minimum_overhead_ns": min(startup["overhead_samples_ns"]),
        "maximum_overhead_ns": max(startup["overhead_samples_ns"]),
        "median_absolute_deviation_ns": statistics.median(
            abs(value - startup["fixed_process_ns"])
            for value in startup["overhead_samples_ns"]
        ),
    }
    model["measurement_scope"] = {
        "native_target": "benchmark internal measured work loop",
        "callgrind_target": "same measured work loop via client requests",
        "process_startup": "separate zero-work native executions",
    }
    model["diagnostic"] = evaluate(model, observations.get("diagnostic", []))
    fresh_by_kind = {
        kind: [
            record for record in serialization_holdout_records
            if record["case"]["kind"] == kind
        ]
        for kind in {record["case"]["kind"] for record in serialization_holdout_records}
    }
    model["fresh_holdout_by_workload"] = {
        kind: evaluate(model, records) for kind, records in fresh_by_kind.items()
    }
    baseline["fit"] = evaluate(baseline, fit_records)
    baseline["holdout"] = evaluate(baseline, holdout_records)
    baseline["holdout_by_workload_family"] = {
        "baseline_work": evaluate(baseline, baseline_holdout_records),
        "cache_stress": evaluate(baseline, cache_holdout_records),
        "mixed_grid": evaluate(baseline, mixed_holdout_records),
        "serialization": evaluate(baseline, serialization_holdout_records),
    }
    baseline["fresh_holdout_by_workload"] = {
        kind: evaluate(baseline, records)
        for kind, records in fresh_by_kind.items()
    }
    model["six_feature_baseline"] = baseline
    model["schema_version"] = 3
    model["source"] = "generic_synthetic_work_only"
    model["renderer_inputs_used"] = False
    control_kinds = set(fresh_by_kind) - {"pointer"}
    maximum_control_regression = max(
        model["fresh_holdout_by_workload"][kind][
            "maximum_absolute_error_percent"
        ]
        - baseline["fresh_holdout_by_workload"][kind][
            "maximum_absolute_error_percent"
        ]
        for kind in control_kinds
    )
    pointer_maximum_improvement = (
        baseline["fresh_holdout_by_workload"]["pointer"][
            "maximum_absolute_error_percent"
        ]
        - model["fresh_holdout_by_workload"]["pointer"][
            "maximum_absolute_error_percent"
        ]
    )
    serial_coefficient = float(model["coefficients_ns"][-1])
    model["selection"] = {
        "maximum_control_regression_percentage_points": (
            maximum_control_regression
        ),
        "pointer_maximum_improvement_percentage_points": (
            pointer_maximum_improvement
        ),
        "serialization_coefficient_ns": serial_coefficient,
        "renderer_inputs_used_for_selection": False,
        "frequency_reports_valid": all_frequency_reports_valid(
            observations.get("frequency")
        ),
    }
    model["synthetic_calibration_valid"] = bool(
        model["jacobian_rank"] == model["event_parameter_count"]
        and model["maximum_absolute_parameter_correlation"] < 0.98
        and model["bound_hit_count"] == 0
        and serial_coefficient > 1e-9
        and model["holdout"]["median_absolute_error_percent"]
        <= MAX_HOLDOUT_MEDIAN_ERROR_PERCENT
        and model["fresh_holdout_by_workload"]["pointer"][
            "median_absolute_error_percent"
        ] <= MAX_HOLDOUT_MEDIAN_ERROR_PERCENT
        and model["fresh_holdout_by_workload"]["pointer"][
            "maximum_absolute_error_percent"
        ] <= MAX_SERIALIZATION_HOLDOUT_ERROR_PERCENT
        and pointer_maximum_improvement > 0.0
        and maximum_control_regression
        <= MAX_CONTROL_REGRESSION_PERCENTAGE_POINTS
        and model["selection"]["frequency_reports_valid"]
    )
    standalone_baseline = {
        **baseline,
        "schema_version": 3,
        "source": "generic_synthetic_work_only",
        "renderer_inputs_used": False,
        "matched_control_only": True,
        "synthetic_calibration_valid": False,
        "measurement_scope": model["measurement_scope"],
        "startup": model["startup"],
    }
    write_json(args.output_dir / "six_feature_model.json", standalone_baseline)
    write_json(args.output_dir / "model.json", model)
    print(json.dumps(model, indent=2, sort_keys=True))
    return 0 if model["synthetic_calibration_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
