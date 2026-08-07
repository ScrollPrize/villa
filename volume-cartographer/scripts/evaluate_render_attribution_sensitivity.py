#!/usr/bin/env python3
"""Evaluate passive renderer work-attribution policies without fitting them."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from passive_event_model import modeled_thread_costs_ns
from run_thread_sync_replay import (
    assign_costs,
    parse_drd_trace,
    parse_thread_profiles,
    replay_scales_for_model,
    simulate_adjusted,
)


SCENARIOS = (
    "full_res",
    "fallback_1",
    "fallback_3",
    "mixed_correlated",
    "mixed_shuffled",
)
WORKERS = tuple(range(1, 8))
BASELINE_POLICY = "equal/residual0.5"


@dataclass(frozen=True)
class Policy:
    placement: str
    residual_window_weight: float

    @property
    def policy_id(self) -> str:
        return f"{self.placement}/residual{self.residual_window_weight:g}"


POLICIES = tuple(
    Policy(placement, residual)
    for placement in ("front", "equal", "back")
    for residual in (0.0, 0.5, 1.0)
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_cases(path: Path) -> list[dict[str, object]]:
    manifest = json.loads(path.read_text())
    if int(manifest.get("schema_version", 0)) != 1:
        raise RuntimeError("attribution case manifest must use schema 1")
    cases = list(manifest.get("cases", []))
    identities = {
        (str(case["scenario"]), int(case["workers"])) for case in cases
    }
    expected = {(scenario, workers) for scenario in SCENARIOS for workers in WORKERS}
    if identities != expected or len(cases) != len(expected):
        raise RuntimeError("case manifest must contain every scenario/worker exactly once")
    return cases


def validate_event_model(
    sync_model: dict[str, object], event_model: dict[str, object]
) -> None:
    if sync_model.get("event_cost_model") != event_model:
        raise RuntimeError(
            "supplied event model does not match synchronization model"
        )


def case_artifacts(case: dict[str, object]) -> list[Path]:
    result_path = Path(str(case["result_path"]))
    result = json.loads(result_path.read_text())
    paths = [result_path]
    paths.extend(Path(path) for path in result["trace"]["raw_paths"])
    prefix = case.get("callgrind_prefix")
    if prefix:
        paths.extend(Path(path) for path in sorted(glob.glob(f"{prefix}-*")))
    return paths


def freeze_inputs(
    case_manifest: Path,
    model_path: Path,
    event_model_path: Path,
    cases: list[dict[str, object]],
    extra_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    paths = {
        case_manifest.resolve(),
        model_path.resolve(),
        event_model_path.resolve(),
        Path(__file__).resolve(),
        (SCRIPT_DIR / "run_thread_sync_replay.py").resolve(),
        *(path.resolve() for path in extra_paths),
    }
    for case in cases:
        paths.update(path.resolve() for path in case_artifacts(case))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR.parent.parent,
    ).stdout.strip()
    return {
        "schema_version": 1,
        "repository_revision": revision,
        "artifacts": {
            str(path): sha256_file(path) for path in sorted(paths)
        },
        "configuration": {
            "baseline_policy": BASELINE_POLICY,
            "policies": [asdict(policy) for policy in POLICIES],
            "scheduler": "fifo",
            "renderer_inputs_used_for_fit": False,
        },
    }


def verify_frozen_inputs(provenance: dict[str, object]) -> None:
    for name, expected in provenance["artifacts"].items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen attribution input changed: {path}")


def assert_cost_preservation(
    events: list,
    thread_costs: dict[int, float],
) -> None:
    actual: dict[int, float] = {}
    for event in events:
        actual[event.thread] = actual.get(event.thread, 0.0) + event.duration
    for thread in {event.thread for event in events}:
        expected = float(thread_costs[thread])
        if not math.isclose(
            actual.get(thread, 0.0), expected, rel_tol=1e-12, abs_tol=1e-6
        ):
            raise RuntimeError(f"attribution changed total cost for thread {thread}")


def profiles_for_case(
    case: dict[str, object], result: dict[str, object]
) -> dict[int, dict[str, int]]:
    prefix = case.get("callgrind_prefix")
    if prefix:
        return parse_thread_profiles(Path(str(prefix)))
    return {
        int(thread): {name: int(value) for name, value in events.items()}
        for thread, events in result["profiles"].items()
    }


def predict_case(
    case: dict[str, object],
    sync_model: dict[str, object],
    comparison_models: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    result = json.loads(Path(str(case["result_path"])).read_text())
    result_case = result["case"]
    if result_case.get("workload") != "renderer":
        raise RuntimeError("attribution sensitivity accepts renderer results only")
    if (
        result_case["scenario"] != case["scenario"]
        or int(result_case["workers"]) != int(case["workers"])
    ):
        raise RuntimeError("renderer result identity does not match case manifest")

    profiles = profiles_for_case(case, result)
    event_model = sync_model["event_cost_model"]
    costs = modeled_thread_costs_ns(profiles, event_model)
    parameters = sync_model["parameters"]
    handoff = float(parameters["cross_thread_release_ns"])
    replay_idle_scale, dependency_excess_scale = replay_scales_for_model(sync_model)
    predictions = {policy.policy_id: [] for policy in POLICIES}
    comparison_models = comparison_models or {}
    comparison_costs = {
        name: modeled_thread_costs_ns(profiles, model["event_cost_model"])
        for name, model in comparison_models.items()
    }
    comparison_predictions = {name: [] for name in comparison_models}
    repetitions = int(result_case["repetitions"])
    cores = int(result_case["parallel_cores"])

    for raw_path in result["trace"]["raw_paths"]:
        parsed = parse_drd_trace(Path(raw_path))
        for policy in POLICIES:
            assign_costs(
                parsed.events,
                costs,
                policy.residual_window_weight,
                policy.placement,
            )
            assert_cost_preservation(parsed.events, costs)
            replay = simulate_adjusted(
                parsed.events,
                cores,
                "fifo",
                cross_thread_latency=handoff,
                replay_idle_scale=replay_idle_scale,
                dependency_excess_scale=dependency_excess_scale,
            )
            predictions[policy.policy_id].append(
                float(replay["modeled_makespan"]) / repetitions
            )
        for name, model in comparison_models.items():
            assign_costs(
                parsed.events,
                comparison_costs[name],
                0.5,
                "equal",
            )
            assert_cost_preservation(parsed.events, comparison_costs[name])
            model_parameters = model["parameters"]
            model_idle_scale, model_dependency_scale = replay_scales_for_model(
                model
            )
            replay = simulate_adjusted(
                parsed.events,
                cores,
                "fifo",
                cross_thread_latency=float(
                    model_parameters["cross_thread_release_ns"]
                ),
                replay_idle_scale=model_idle_scale,
                dependency_excess_scale=model_dependency_scale,
            )
            comparison_predictions[name].append(
                float(replay["modeled_makespan"]) / repetitions
            )

    measured = float(
        result["native_whole_process"][str(cores)][
            "measured_render_nanoseconds_per_call"
        ]["median"]
    )
    return {
        "scenario": str(case["scenario"]),
        "workers": int(case["workers"]),
        "measured_ns": measured,
        "predicted_ns": {
            policy: statistics.median(samples)
            for policy, samples in predictions.items()
        },
        "comparison_predicted_ns": {
            name: statistics.median(samples)
            for name, samples in comparison_predictions.items()
        },
    }


def summarize_model_comparisons(
    predictions: list[dict[str, object]], names: tuple[str, ...]
) -> dict[str, object]:
    reports = {}
    for name in names:
        rows = []
        for prediction in predictions:
            predicted = float(prediction["comparison_predicted_ns"][name])
            measured = float(prediction["measured_ns"])
            rows.append(
                {
                    "scenario": prediction["scenario"],
                    "workers": prediction["workers"],
                    "predicted_ns": predicted,
                    "measured_ns": measured,
                    "runtime_error_percent": 100.0 * (predicted / measured - 1.0),
                }
            )
        for scenario in SCENARIOS:
            scenario_rows = [row for row in rows if row["scenario"] == scenario]
            one = next(row for row in scenario_rows if row["workers"] == 1)
            for row in scenario_rows:
                predicted_speedup = one["predicted_ns"] / row["predicted_ns"]
                measured_speedup = one["measured_ns"] / row["measured_ns"]
                row["speedup_error_percent"] = 100.0 * (
                    predicted_speedup / measured_speedup - 1.0
                )

        def maximum(selected: list[dict[str, object]], field: str) -> dict[str, object]:
            row = max(selected, key=lambda value: abs(float(value[field])))
            return {
                "scenario": row["scenario"],
                "workers": row["workers"],
                "signed_error_percent": row[field],
                "absolute_error_percent": abs(float(row[field])),
            }

        reports[name] = {
            "rows": rows,
            "maximum_runtime_error_workers_1": maximum(
                [row for row in rows if row["workers"] == 1],
                "runtime_error_percent",
            ),
            "maximum_runtime_error_workers_2_to_7": maximum(
                [row for row in rows if row["workers"] >= 2],
                "runtime_error_percent",
            ),
            "maximum_runtime_error_by_worker": {
                str(workers): maximum(
                    [row for row in rows if row["workers"] == workers],
                    "runtime_error_percent",
                )
                for workers in WORKERS
            },
            "maximum_speedup_error": maximum(rows, "speedup_error_percent"),
        }
    return reports


def summarize(
    predictions: list[dict[str, object]],
) -> dict[str, object]:
    reports = {}
    for policy in (policy.policy_id for policy in POLICIES):
        rows = []
        for prediction in predictions:
            predicted = float(prediction["predicted_ns"][policy])
            measured = float(prediction["measured_ns"])
            rows.append(
                {
                    "scenario": prediction["scenario"],
                    "workers": prediction["workers"],
                    "predicted_ns": predicted,
                    "measured_ns": measured,
                    "runtime_error_percent": 100.0 * (predicted / measured - 1.0),
                }
            )
        for scenario in SCENARIOS:
            scenario_rows = [row for row in rows if row["scenario"] == scenario]
            one = next(row for row in scenario_rows if row["workers"] == 1)
            for row in scenario_rows:
                predicted_speedup = one["predicted_ns"] / row["predicted_ns"]
                measured_speedup = one["measured_ns"] / row["measured_ns"]
                row["speedup_error_percent"] = 100.0 * (
                    predicted_speedup / measured_speedup - 1.0
                )
        runtime = [abs(row["runtime_error_percent"]) for row in rows]
        speedup = [abs(row["speedup_error_percent"]) for row in rows]
        reports[policy] = {
            "rows": rows,
            "summary": {
                "runtime_median_absolute_error_percent": statistics.median(runtime),
                "runtime_rms_error_percent": math.sqrt(
                    statistics.fmean(value * value for value in runtime)
                ),
                "runtime_maximum_absolute_error_percent": max(runtime),
                "runtime_cases_within_20_percent": sum(value <= 20.0 for value in runtime),
                "speedup_median_absolute_error_percent": statistics.median(speedup),
                "speedup_rms_error_percent": math.sqrt(
                    statistics.fmean(value * value for value in speedup)
                ),
                "speedup_maximum_absolute_error_percent": max(speedup),
                "speedup_cases_within_20_percent": sum(value <= 20.0 for value in speedup),
            },
        }

    baseline = reports[BASELINE_POLICY]["summary"]
    for report in reports.values():
        summary = report["summary"]
        summary["runtime_rms_delta_from_baseline_points"] = (
            summary["runtime_rms_error_percent"]
            - baseline["runtime_rms_error_percent"]
        )
        summary["speedup_rms_delta_from_baseline_points"] = (
            summary["speedup_rms_error_percent"]
            - baseline["speedup_rms_error_percent"]
        )
    diagnostic_order = sorted(
        reports,
        key=lambda policy: reports[policy]["summary"]["runtime_rms_error_percent"],
    )
    return {
        "baseline_policy": BASELINE_POLICY,
        "policy_selected": False,
        "diagnostic_order_by_runtime_rms": diagnostic_order,
        "policies": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--event-model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare",
        nargs=3,
        action="append",
        metavar=("NAME", "SYNC_MODEL", "EVENT_MODEL"),
        default=[],
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    sync_model = json.loads(args.model.read_text())
    event_model = json.loads(args.event_model.read_text())
    if sync_model.get("renderer_inputs_used") is not False:
        raise RuntimeError("synchronization model is not synthetic-only")
    validate_event_model(sync_model, event_model)
    comparison_models = {}
    comparison_paths = []
    for name, sync_name, event_name in args.compare:
        if name in comparison_models:
            raise RuntimeError(f"duplicate comparison model {name}")
        sync_path = Path(sync_name)
        event_path = Path(event_name)
        comparison = json.loads(sync_path.read_text())
        comparison_event = json.loads(event_path.read_text())
        if comparison.get("renderer_inputs_used") is not False:
            raise RuntimeError(f"comparison model {name} is not synthetic-only")
        validate_event_model(comparison, comparison_event)
        comparison_models[name] = comparison
        comparison_paths.extend((sync_path, event_path))
    provenance = freeze_inputs(
        args.cases,
        args.model,
        args.event_model,
        cases,
        tuple(comparison_paths),
    )
    provenance["configuration"]["comparison_models"] = list(comparison_models)
    verify_frozen_inputs(provenance)
    predictions = []
    for index, case in enumerate(cases, 1):
        print(
            f"[attribution] {index}/{len(cases)} "
            f"{case['scenario']} workers={case['workers']}",
            flush=True,
        )
        predictions.append(
            predict_case(case, sync_model, comparison_models)
        )
    report = {
        "schema_version": 1,
        "renderer_inputs_used_for_fit": False,
        **summarize(predictions),
        "model_comparisons": summarize_model_comparisons(
            predictions, tuple(comparison_models)
        ),
    }
    verify_frozen_inputs(provenance)
    write_json(args.output, report)
    provenance["output"] = {
        "path": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
    }
    provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    write_json(provenance_path, provenance)
    for policy in report["diagnostic_order_by_runtime_rms"]:
        summary = report["policies"][policy]["summary"]
        print(
            f"{policy:18} runtime_rms={summary['runtime_rms_error_percent']:.2f}% "
            f"speedup_rms={summary['speedup_rms_error_percent']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
