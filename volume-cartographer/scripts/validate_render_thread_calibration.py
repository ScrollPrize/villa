#!/usr/bin/env python3
"""Validate the dispatch calibration on production renderer workloads."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_thread_dispatch_shared as dispatch_calibration


SCENARIO_LEVELS = {
    "full_res": 1,
    "fallback_1": 2,
    "fallback_3": 4,
    "mixed_correlated": 4,
    "mixed_shuffled": 4,
}
VALIDATION_WORKERS = tuple(range(1, 8))
MAX_ERROR_PERCENT = 20.0
EXPECTED_PARAMETERS = {
    "work_ns_per_iteration",
    "fixed_dispatch_ns",
    "per_future_dispatch_ns",
}


@dataclass(frozen=True)
class RenderCase:
    scenario: str
    levels: int
    workers: int
    repetitions: int

    @property
    def case_id(self) -> str:
        return f"renderer-{self.scenario}-l{self.levels}-w{self.workers}"


def load_model(path: Path) -> dict[str, object]:
    model = json.loads(path.read_text())
    if model.get("schema_version") != 3:
        raise RuntimeError("renderer validation requires dispatch model schema 3")
    if model.get("parameter_count") != 3:
        raise RuntimeError("renderer validation requires exactly three parameters")
    parameters = model.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != EXPECTED_PARAMETERS:
        raise RuntimeError("dispatch model parameter set is not the accepted schema")
    if not model.get("synthetic_calibration_valid"):
        raise RuntimeError("synthetic dispatch calibration did not pass")
    domain = model.get("domain", {})
    if domain.get("mode") != "futures" or int(domain.get("maximum_workers", 0)) < 7:
        raise RuntimeError("dispatch model does not cover renderer validation domain")
    return model


def dispatch_nanoseconds(parameters: dict[str, float], workers: int) -> float:
    return (
        float(parameters["fixed_dispatch_ns"])
        + workers * float(parameters["per_future_dispatch_ns"])
    )


def predict_render_nanoseconds(
    parameters: dict[str, float],
    levels: int,
    workers: int,
    one_worker_nanoseconds: float,
) -> float:
    one_worker_dispatch = levels * dispatch_nanoseconds(parameters, 1)
    serial_work = one_worker_nanoseconds - one_worker_dispatch
    if serial_work <= 0:
        raise RuntimeError("measured one-worker render is no larger than dispatch")
    return (
        serial_work / workers
        + levels * dispatch_nanoseconds(parameters, workers)
    )


def command_for(binary: Path, case: RenderCase) -> list[str]:
    command = [
        "taskset",
        "-c",
        dispatch_calibration.affinity(case.workers),
        str(binary),
        "--fixture",
        "parallel",
        "--scenario",
        case.scenario,
        "--repetitions",
        str(case.repetitions),
        "--native-trials",
        "1",
    ]
    if case.workers > 1:
        command.append("--require-parallel-execution")
    return command


def run_case(binary: Path, case: RenderCase) -> dict[str, object]:
    env = os.environ.copy()
    env["VC_RENDER_SAMPLER_THREADS"] = str(case.workers)
    result = subprocess.run(
        command_for(binary, case),
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    output = json.loads(result.stdout)
    if output.get("scenario") != case.scenario:
        raise RuntimeError(f"scenario mismatch for {case.case_id}")
    if output.get("fixture") != "parallel":
        raise RuntimeError(f"fixture mismatch for {case.case_id}")
    if int(output.get("worker_override", 0)) != case.workers:
        raise RuntimeError(f"worker override mismatch for {case.case_id}")
    if output.get("build_type") != "Release":
        raise RuntimeError("renderer validation requires a Release benchmark")
    if output.get("architecture_target") != "x86-64-v3":
        raise RuntimeError("renderer validation requires the x86-64-v3 build")
    observed_threads = int(output.get("observed_threads", 0))
    if case.workers == 1 and observed_threads != 1:
        raise RuntimeError("one-worker renderer used an unexpected thread count")
    if case.workers > 1 and observed_threads < 2:
        raise RuntimeError(f"parallel execution was not observed for {case.case_id}")
    seconds = float(output["native_median_seconds"])
    if seconds <= 0:
        raise RuntimeError(f"nonpositive renderer time for {case.case_id}")
    return {
        "case": asdict(case),
        "case_id": case.case_id,
        "nanoseconds_per_render": seconds * 1e9 / case.repetitions,
        "output": output,
        "command": command_for(binary, case),
    }


def choose_repetitions(
    binary: Path, scenario: str, levels: int, target_seconds: float
) -> int:
    pilot = RenderCase(scenario, levels, max(VALIDATION_WORKERS), 2)
    sample = run_case(binary, pilot)
    nanoseconds = float(sample["nanoseconds_per_render"])
    return max(1, min(16_384, math.ceil(target_seconds * 1e9 / nanoseconds)))


def make_cases(repetitions: dict[str, int]) -> list[RenderCase]:
    return [
        RenderCase(scenario, levels, workers, repetitions[scenario])
        for scenario, levels in SCENARIO_LEVELS.items()
        for workers in VALIDATION_WORKERS
    ]


def collect_cases(
    binary: Path, cases: list[RenderCase], trials: int
) -> list[dict[str, object]]:
    samples = {case.case_id: [] for case in cases}
    total = len(cases) * trials
    completed = 0
    started = time.monotonic()
    for trial in range(trials):
        ordered = cases[trial % len(cases) :] + cases[: trial % len(cases)]
        if trial % 2:
            ordered.reverse()
        for case in ordered:
            samples[case.case_id].append(run_case(binary, case))
            completed += 1
            elapsed = time.monotonic() - started
            eta = elapsed * (total - completed) / completed
            print(
                f"[renderer-validation] {completed}/{total} {case.case_id} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
    return [
        {
            "case": asdict(case),
            "case_id": case.case_id,
            "samples": samples[case.case_id],
        }
        for case in cases
    ]


def summarize(
    records: list[dict[str, object]], parameters: dict[str, float]
) -> tuple[list[dict[str, object]], bool]:
    by_scenario = {
        (record["case"]["scenario"], int(record["case"]["workers"])): record
        for record in records
    }
    reports = []
    valid = True
    for scenario, levels in SCENARIO_LEVELS.items():
        baseline = by_scenario[scenario, 1]
        baseline_values = [
            float(sample["nanoseconds_per_render"])
            for sample in baseline["samples"]
        ]
        one_worker_median = statistics.median(baseline_values)
        for workers in VALIDATION_WORKERS:
            record = by_scenario[scenario, workers]
            values = [
                float(sample["nanoseconds_per_render"])
                for sample in record["samples"]
            ]
            median = statistics.median(values)
            predicted = predict_render_nanoseconds(
                parameters, levels, workers, one_worker_median
            )
            sample_errors = [100.0 * (predicted / value - 1.0) for value in values]
            median_error = 100.0 * (predicted / median - 1.0)
            maximum_individual = max(abs(value) for value in sample_errors)
            accepted = workers == 1 or (
                abs(median_error) <= MAX_ERROR_PERCENT
                and maximum_individual <= MAX_ERROR_PERCENT
            )
            valid = valid and accepted
            reports.append(
                {
                    "case_id": record["case_id"],
                    "scenario": scenario,
                    "levels": levels,
                    "workers": workers,
                    "sample_nanoseconds_per_render": values,
                    "minimum_nanoseconds_per_render": min(values),
                    "median_nanoseconds_per_render": median,
                    "maximum_nanoseconds_per_render": max(values),
                    "mad_nanoseconds_per_render": statistics.median(
                        abs(value - median) for value in values
                    ),
                    "predicted_nanoseconds_per_render": predicted,
                    "median_runtime_error_percent": median_error,
                    "maximum_individual_runtime_error_percent": maximum_individual,
                    "accepted": accepted,
                }
            )
    return reports, valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--target-seconds", type=float, default=0.5)
    parser.add_argument("--reuse-observations", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.target_seconds <= 0:
        raise RuntimeError("trials and target seconds must be positive")

    model = load_model(args.model)
    parameters = {
        key: float(value) for key, value in model["parameters"].items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topology = dispatch_calibration.validate_topology()
    os.sched_setaffinity(0, {dispatch_calibration.MONITOR_CPU})
    observations_path = args.output_dir / "renderer-observations.json"

    if args.reuse_observations:
        payload = json.loads(observations_path.read_text())
        records = payload["observations"]
        frequency = payload["frequency"]
        repetitions = payload["repetitions"]
        topology = payload["topology"]
    else:
        state, policies = dispatch_calibration.validate_fixed_frequency()
        repetitions = {
            scenario: choose_repetitions(
                args.benchmark, scenario, levels, args.target_seconds
            )
            for scenario, levels in SCENARIO_LEVELS.items()
        }
        monitor = dispatch_calibration.FrequencyMonitor(policies)
        with monitor:
            records = collect_cases(
                args.benchmark, make_cases(repetitions), args.trials
            )
        postflight, _ = dispatch_calibration.validate_fixed_frequency()
        frequency = {
            **state,
            **monitor.report(int(state["target_khz"])),
            "postflight": postflight,
        }
        dispatch_calibration.write_json(
            observations_path,
            {
                "schema_version": 1,
                "repetitions": repetitions,
                "observations": records,
                "frequency": frequency,
                "topology": topology,
            },
        )

    reports, timing_valid = summarize(records, parameters)
    validation_valid = timing_valid and bool(frequency["within_tolerance"])
    result = {
        "schema_version": 1,
        "dispatch_model_schema_version": model["schema_version"],
        "matrix": {
            "scenarios": SCENARIO_LEVELS,
            "workers": list(VALIDATION_WORKERS),
            "trials": args.trials,
            "repetitions": repetitions,
            "maximum_error_percent": MAX_ERROR_PERCENT,
        },
        "formula": {
            "description": (
                "subtract one-worker dispatch per level, divide remaining work "
                "by workers, then add worker-count dispatch per level"
            ),
            "fitted_renderer_parameters": 0,
        },
        "cases": reports,
        "frequency": frequency,
        "topology": topology,
        "timing_valid": timing_valid,
        "renderer_validation_valid": validation_valid,
    }
    dispatch_calibration.write_json(
        args.output_dir / "renderer-validation.json", result
    )

    validated_model = dict(model)
    validated_model["domain"] = {
        **model["domain"],
        "actual_renderer_validation_complete": True,
    }
    validated_model["renderer_validation_file"] = "renderer-validation.json"
    validated_model["renderer_validation_valid"] = validation_valid
    validated_model["timing_claims_enabled"] = bool(
        model["synthetic_calibration_valid"] and validation_valid
    )
    dispatch_calibration.write_json(
        args.output_dir / "validated-model.json", validated_model
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if validation_valid else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"validate_render_thread_calibration.py: {error}", file=sys.stderr)
        raise SystemExit(1)
