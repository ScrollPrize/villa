#!/usr/bin/env python3
"""Maintain the C++ synthetic-renderer Callgrind performance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from render_valgrind_common import SCENARIOS

SCHEMA_VERSION = 1
NATIVE_EVALUATION_SCHEMA_VERSIONS = {2, 3}
DEFAULT_TOLERANCE = 0.05
DATA_READ_FEATURE_NAMES = (
    "non_data_instructions",
    "data_reads",
    "data_writes",
    "l1_data_misses",
    "last_level_data_misses",
    "branch_misses",
    "branch_weighted_l1_misses",
)
REPLAY_CONFIGURATION = {
    "workers": 4,
    "cores": 5,
    "tie_policy": "fifo",
    "split_policy": "equal",
    "residual_fraction": 0.5,
    "wake_latency_ns": 0.0,
    "replay_idle_scale": 1.0,
    "dependency_excess_scale": 1.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_tolerance(value: object) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise RuntimeError("modeled-runtime tolerance must be finite and in [0, 1)")
    return tolerance


def validate_score(value: object, source: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{source} modeled-runtime score must be finite and positive"
        ) from error
    if not math.isfinite(score) or score <= 0.0:
        raise RuntimeError(
            f"{source} modeled-runtime score must be finite and positive"
        )
    return score


def freeze_reference(args: argparse.Namespace) -> None:
    tolerance = validate_tolerance(args.tolerance)
    model_path = args.model.resolve()
    model_hash = sha256_file(model_path)
    model_id = json.loads(model_path.read_text()).get("model_id")
    if not model_id:
        raise RuntimeError("render model has no model_id")
    cases: dict[str, object] = {}
    for path in args.results:
        result = json.loads(path.resolve().read_text())
        if result.get("kind") != "evaluation":
            raise RuntimeError(
                f"expected evaluation artifact, got {result.get('kind')!r}"
            )
        schema = result.get("schema_version")
        if schema == SCHEMA_VERSION:
            if result.get("model_sha256") != model_hash:
                raise RuntimeError(f"evaluation {path} used a different model")
        elif schema in NATIVE_EVALUATION_SCHEMA_VERSIONS:
            if result.get("model_id") != model_id:
                raise RuntimeError(f"evaluation {path} used a different model")
        else:
            raise RuntimeError(f"unsupported evaluation schema in {path}")
        if result["case"] in cases:
            raise RuntimeError(f"duplicate evaluation case {result['case']}")
        case = {
            "checksum": result["checksum"],
            "modeled_runtime_score_ns": validate_score(
                result["modeled_runtime_score_ns"], "evaluation"
            ),
        }
        if "identity" in result:
            case["identity"] = result["identity"]
        cases[result["case"]] = case

    expected = {
        f"{fixture}/{scenario}"
        for fixture in ("serial", "parallel")
        for scenario in SCENARIOS
    }
    if set(cases) != expected:
        raise RuntimeError(
            f"reference case set mismatch: missing={sorted(expected - set(cases))}, "
            f"extra={sorted(set(cases) - expected)}"
        )
    write_json_atomic(
        args.output.resolve(),
        {
            "schema_version": SCHEMA_VERSION,
            "model_sha256": model_hash,
            "tolerance": tolerance,
            "cases": dict(sorted(cases.items())),
        },
    )


def freeze_model(args: argparse.Namespace) -> None:
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("renderer_inputs_used") is not False:
        raise RuntimeError("calibration is not synthetic-only")
    if not calibration.get("candidate_accepted", False) and not args.allow_unpromoted:
        raise RuntimeError(
            "calibration was not accepted; pass --allow-unpromoted only after "
            "explicitly reviewing and approving the experimental model"
        )
    event_model = calibration.get("event_cost_model", {})
    if tuple(event_model.get("feature_names", ())) != DATA_READ_FEATURE_NAMES:
        raise RuntimeError("calibration does not use the required data-read basis")
    coefficients = event_model.get("coefficients_ns", ())
    if len(coefficients) != len(DATA_READ_FEATURE_NAMES):
        raise RuntimeError("calibration has an invalid event coefficient count")
    parameters = calibration.get("parameters", {})
    if "cross_thread_release_ns" not in parameters:
        raise RuntimeError("calibration has no cross-thread release parameter")
    model = {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "source": "synthetic thread-pool calibration only",
        "renderer_inputs_used": False,
        "timing_claims_enabled": False,
        "score_semantics": "relative modeled runtime, not absolute wall time",
        "cross_thread_release_ns": float(parameters["cross_thread_release_ns"]),
        "event_cost_model": {
            "feature_names": list(DATA_READ_FEATURE_NAMES),
            "coefficients_ns": [float(value) for value in coefficients],
            "stall_overlap_fraction": float(
                event_model.get("stall_overlap_fraction", 0.0)
            ),
        },
        "replay": REPLAY_CONFIGURATION,
    }
    write_json_atomic(args.output.resolve(), model)


def set_tolerance(args: argparse.Namespace) -> None:
    reference_path = args.reference.resolve()
    reference = json.loads(reference_path.read_text())
    if reference.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported modeled-runtime reference schema")
    if not isinstance(reference.get("cases"), dict) or len(reference["cases"]) != 8:
        raise RuntimeError("modeled-runtime reference must contain eight cases")
    reference["tolerance"] = validate_tolerance(args.tolerance)
    write_json_atomic((args.output or reference_path).resolve(), reference)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-reference")
    freeze.add_argument("--model", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    freeze.add_argument("results", nargs="+", type=Path)
    freeze.set_defaults(function=freeze_reference)

    freeze_model_parser = subparsers.add_parser("freeze-model")
    freeze_model_parser.add_argument("--calibration", required=True, type=Path)
    freeze_model_parser.add_argument("--model-id", required=True)
    freeze_model_parser.add_argument("--output", required=True, type=Path)
    freeze_model_parser.add_argument("--allow-unpromoted", action="store_true")
    freeze_model_parser.set_defaults(function=freeze_model)

    set_tolerance_parser = subparsers.add_parser("set-tolerance")
    set_tolerance_parser.add_argument("--reference", required=True, type=Path)
    set_tolerance_parser.add_argument("--tolerance", required=True, type=float)
    set_tolerance_parser.add_argument("--output", type=Path)
    set_tolerance_parser.set_defaults(function=set_tolerance)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"run_render_valgrind_ci.py: {error}", file=sys.stderr)
        raise SystemExit(1)
