#!/usr/bin/env python3
"""Fit the synthetic rendering Callgrind-to-native calibration model."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SCHEMA_VERSION = 1
MODEL_VERSION = 2
FIXED_IPC = 4.0
FIXED_LAST_LEVEL_MISS_CYCLES = 35.0
PARAMETER_BOUNDS = {
    "l1_miss_cycles": (1.0, 20.0),
    "branch_mispredict_cycles": (5.0, 40.0),
    "parallel_2_effective_workers": (1.0, 2.0),
    "parallel_4_effective_workers": (1.0, 4.0),
}
REQUIRED_EVENTS = (
    "Ir",
    "I1mr",
    "D1mr",
    "D1mw",
    "ILmr",
    "DLmr",
    "DLmw",
    "Bcm",
    "Bim",
)
TRAIN_MAX_ERROR = 0.08
HELD_OUT_MAX_ERROR = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_observations(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported rendering calibration observation schema")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("rendering calibration has no observations")
    for case in cases:
        missing = set(REQUIRED_EVENTS).difference(case.get("events_per_pixel", {}))
        if missing:
            raise RuntimeError(f"{case.get('name')} is missing events {sorted(missing)}")
        samples = case.get("native_ns_per_pixel_samples", [])
        if len(samples) < 5 or any(float(sample) <= 0 for sample in samples):
            raise RuntimeError(f"{case.get('name')} requires at least five native samples")
        if case.get("role") not in {"training", "held_out"}:
            raise RuntimeError(f"{case.get('name')} has an invalid calibration role")
    return value


def modeled_cycles_per_pixel(case: dict[str, object], parameters: list[float]) -> float:
    l1_cycles, branch_cycles, parallel_2, parallel_4 = parameters
    events = case["events_per_pixel"]
    l1_misses = events["I1mr"] + events["D1mr"] + events["D1mw"]
    last_level_misses = events["ILmr"] + events["DLmr"] + events["DLmw"]
    branch_misses = events["Bcm"] + events["Bim"]
    work_cycles = (
        events["Ir"] / FIXED_IPC
        + l1_cycles * l1_misses
        + FIXED_LAST_LEVEL_MISS_CYCLES * last_level_misses
        + branch_cycles * branch_misses
    )
    workers = int(case["workers"])
    effective = {1: 1.0, 2: parallel_2, 4: parallel_4}.get(workers)
    if effective is None:
        raise RuntimeError(f"unsupported calibration worker count {workers}")
    return work_cycles / effective if case["fixture"] == "parallel" else work_cycles


def native_median(case: dict[str, object]) -> float:
    return statistics.median(float(value) for value in case["native_ns_per_pixel_samples"])


def fit_scale(cases: list[dict[str, object]], parameters: list[float]) -> float:
    ratios = [modeled_cycles_per_pixel(case, parameters) / native_median(case) for case in cases]
    return sum(ratios) / sum(value * value for value in ratios)


def objective(cases: list[dict[str, object]], parameters: list[float]) -> float:
    scale = fit_scale(cases, parameters)
    errors = [
        scale * modeled_cycles_per_pixel(case, parameters) / native_median(case) - 1.0
        for case in cases
    ]
    return sum(error * error for error in errors) / len(errors)


def minimize_coordinate(
    cases: list[dict[str, object]], parameters: list[float], index: int, low: float, high: float
) -> float:
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left = low
    right = high
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)

    def evaluate(value: float) -> float:
        candidate = parameters.copy()
        candidate[index] = value
        return objective(cases, candidate)

    y1 = evaluate(x1)
    y2 = evaluate(x2)
    for _ in range(80):
        if y1 < y2:
            right, x2, y2 = x2, x1, y1
            x1 = right - golden * (right - left)
            y1 = evaluate(x1)
        else:
            left, x1, y1 = x1, x2, y2
            x2 = left + golden * (right - left)
            y2 = evaluate(x2)
    return (left + right) / 2.0


def fit(cases: list[dict[str, object]], initial: list[float]) -> tuple[list[float], float]:
    parameters = initial.copy()
    bounds = list(PARAMETER_BOUNDS.values())
    for _ in range(100):
        previous = objective(cases, parameters)
        for index, (low, high) in enumerate(bounds):
            value = minimize_coordinate(cases, parameters, index, low, high)
            if abs(value - low) < 1e-9:
                value = low
            elif abs(value - high) < 1e-9:
                value = high
            parameters[index] = value
        if abs(previous - objective(cases, parameters)) < 1e-16:
            break
    return parameters, fit_scale(cases, parameters)


def error_summary(
    cases: list[dict[str, object]], parameters: list[float], nanoseconds_per_cycle: float
) -> dict[str, float]:
    errors = [
        nanoseconds_per_cycle * modeled_cycles_per_pixel(case, parameters) / native_median(case) - 1.0
        for case in cases
    ]
    return {
        "max_absolute_relative_error": max(abs(error) for error in errors),
        "rms_relative_error": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def case_result(
    case: dict[str, object], parameters: list[float], nanoseconds_per_cycle: float
) -> dict[str, object]:
    samples = [float(value) for value in case["native_ns_per_pixel_samples"]]
    median = statistics.median(samples)
    cycles = modeled_cycles_per_pixel(case, parameters)
    predicted = cycles * nanoseconds_per_cycle
    return {
        "name": case["name"],
        "role": case["role"],
        "native_ns_per_pixel_min": min(samples),
        "native_ns_per_pixel_median": median,
        "native_ns_per_pixel_max": max(samples),
        "native_ns_per_pixel_mad": statistics.median(abs(value - median) for value in samples),
        "modeled_cycles_per_pixel": cycles,
        "estimated_ns_per_pixel": predicted,
        "relative_error": predicted / median - 1.0,
    }


def main() -> int:
    args = parse_args()
    observations = load_observations(args.observations)
    cases = observations["cases"]
    training = [case for case in cases if case["role"] == "training"]
    held_out = [case for case in cases if case["role"] == "held_out"]
    if not training or not held_out:
        raise RuntimeError("calibration requires training and held-out observations")

    starts = (
        [4.0, 15.0, 1.2, 1.8],
        [1.0, 5.0, 1.0, 1.0],
        [20.0, 40.0, 2.0, 4.0],
    )
    fits = [fit(training, start) for start in starts]
    parameters, nanoseconds_per_cycle = min(
        fits, key=lambda item: objective(training, item[0])
    )
    predictions = [
        [modeled_cycles_per_pixel(case, fit_parameters) * fit_scale for case in cases]
        for fit_parameters, fit_scale in fits
    ]
    sensitivity = max(
        max(values) - min(values) for values in zip(*predictions, strict=True)
    )
    training_error = error_summary(training, parameters, nanoseconds_per_cycle)
    held_out_error = error_summary(held_out, parameters, nanoseconds_per_cycle)
    if training_error["max_absolute_relative_error"] > TRAIN_MAX_ERROR:
        raise RuntimeError(f"training error exceeds {TRAIN_MAX_ERROR:.0%}")
    if held_out_error["max_absolute_relative_error"] > HELD_OUT_MAX_ERROR:
        raise RuntimeError(f"held-out error exceeds {HELD_OUT_MAX_ERROR:.0%}")

    model = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "reference": observations["reference"],
        "formula": {
            "modeled_ipc": FIXED_IPC,
            "l1_miss_cycles": parameters[0],
            "last_level_miss_cycles": FIXED_LAST_LEVEL_MISS_CYCLES,
            "branch_mispredict_cycles": parameters[1],
        },
        "effective_parallelism": {
            "1": 1.0,
            "2": parameters[2],
            "4": parameters[3],
        },
        "nanoseconds_per_modeled_cycle": nanoseconds_per_cycle,
        "fit": {
            "objective": "mean squared relative error over case medians",
            "parameter_bounds": PARAMETER_BOUNDS,
            "training": training_error,
            "held_out": held_out_error,
            "multi_start_max_prediction_delta_ns_per_pixel": sensitivity,
            "acceptance": {
                "training_max_absolute_relative_error": TRAIN_MAX_ERROR,
                "held_out_max_absolute_relative_error": HELD_OUT_MAX_ERROR,
            },
        },
        "cases": [
            case_result(case, parameters, nanoseconds_per_cycle)
            for case in sorted(cases, key=lambda value: value["name"])
        ],
    }
    args.output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps(model, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"calibrate_render_model.py: {error}")
        raise SystemExit(1)
