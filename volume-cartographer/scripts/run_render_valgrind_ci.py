#!/usr/bin/env python3
"""Collect and gate the complete synthetic-renderer Valgrind CI matrix."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from native_thread_sync_replay import (
    NativeReplayEngine,
    attribution_request,
    replay_request,
)
from render_valgrind_common import (
    CACHE_GEOMETRY,
    SCENARIOS,
    callgrind_command,
    drd_command,
    load_renderer_metadata,
    parse_thread_profiles,
    require_supported_host,
    valgrind_version,
)
from thread_sync_trace import parse_drd_trace, write_event_stream

SCHEMA_VERSION = 1
NATIVE_EVALUATION_SCHEMA_VERSION = 2
BENCHMARK_METADATA_SCHEMA = 1
DEFAULT_REPETITIONS = 1
DEFAULT_QUANTUM = 10000
DEFAULT_DRD_ATTEMPTS = 3
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
IDENTITY_FIELDS = (
    "compiler_id",
    "compiler_version",
    "build_type",
    "architecture_target",
)
WORKLOAD_FIELDS = (
    "fixture",
    "scenario",
    "width",
    "height",
    "tile_size",
    "repetitions",
    "measured_pixels",
    "worker_override",
)


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


def load_manifest(path: Path, kind: str) -> dict[str, object]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported artifact schema in {path}")
    if value.get("kind") != kind:
        raise RuntimeError(f"expected {kind} artifact, got {value.get('kind')!r}")
    return value


def _workers(fixture: str) -> int:
    return 1 if fixture == "serial" else int(REPLAY_CONFIGURATION["workers"])


def _environment(fixture: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["VC_RENDER_SAMPLER_THREADS"] = str(_workers(fixture))
    return environment


def _validate_metadata(
    metadata: dict[str, object], fixture: str, scenario: str, repetitions: int
) -> None:
    expected = {
        "fixture": fixture,
        "scenario": scenario,
        "repetitions": repetitions,
        "worker_override": _workers(fixture),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(
                f"benchmark metadata {name} changed: {metadata.get(name)!r} != {value!r}"
            )
    expected_pixels = int(metadata["width"]) * int(metadata["height"]) * repetitions
    if metadata["measured_pixels"] != expected_pixels:
        raise RuntimeError("benchmark measured-pixel count is inconsistent")
    observed = int(metadata["observed_threads"])
    if fixture == "serial" and observed != 1:
        raise RuntimeError("serial fixture executed on multiple threads")
    if fixture == "parallel" and observed <= 1:
        raise RuntimeError("parallel fixture did not execute on multiple threads")


def collect_callgrind(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "callgrind.out"
    metadata_path = output_dir / "metadata.callgrind.json"
    for stale in glob.glob(f"{prefix}*"):
        Path(stale).unlink()
    metadata_path.unlink(missing_ok=True)
    args.artifact.unlink(missing_ok=True)

    command = callgrind_command(
        args.benchmark.resolve(),
        args.fixture,
        args.scenario,
        metadata_path,
        prefix,
        args.repetitions,
        separate_threads=True,
    )
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=_environment(args.fixture),
    )
    metadata = load_renderer_metadata(metadata_path)
    _validate_metadata(metadata, args.fixture, args.scenario, args.repetitions)
    profiles = parse_thread_profiles(prefix)
    profile_paths = sorted(Path(name).resolve() for name in glob.glob(f"{prefix}-*"))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_metadata_schema": BENCHMARK_METADATA_SCHEMA,
        "kind": "callgrind",
        "case": f"{args.fixture}/{args.scenario}",
        "valgrind_version": valgrind_version(),
        "cache_geometry": CACHE_GEOMETRY,
        "metadata": metadata,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "profile_prefix": str(prefix),
        "profiles": {
            str(thread): events for thread, events in sorted(profiles.items())
        },
        "profile_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in profile_paths
        ],
        "command": command,
        "stdout": completed.stdout.strip(),
    }
    write_json_atomic(args.artifact.resolve(), manifest)
    print(f"{manifest['case']}: collected {len(profiles)} Callgrind profiles")


def _trace_is_complete(parsed: object) -> bool:
    return (
        parsed.unmatched_waits == 0
        and parsed.unresolved_happens_before == 0
        and bool(parsed.events)
    )


def collect_drd(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "drd.log"
    event_path = output_dir / "events.jsonl"
    metadata_path = output_dir / "metadata.drd.json"
    args.artifact.unlink(missing_ok=True)
    selected = None
    command: list[str] = []
    stdout = ""
    for attempt in range(1, args.attempts + 1):
        trace_path.unlink(missing_ok=True)
        event_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        command = drd_command(
            args.benchmark.resolve(),
            args.scenario,
            metadata_path,
            trace_path,
            args.repetitions,
            args.quantum,
        )
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            env=_environment("parallel"),
        )
        stdout = completed.stdout.strip()
        parsed = parse_drd_trace(trace_path)
        if _trace_is_complete(parsed):
            selected = (attempt, parsed)
            break
    if selected is None:
        raise RuntimeError(
            f"DRD trace remained incomplete after {args.attempts} attempts"
        )
    attempt, parsed = selected
    metadata = load_renderer_metadata(metadata_path)
    _validate_metadata(metadata, "parallel", args.scenario, args.repetitions)
    write_event_stream(event_path, parsed.events)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_metadata_schema": BENCHMARK_METADATA_SCHEMA,
        "kind": "drd",
        "case": f"parallel/{args.scenario}",
        "valgrind_version": valgrind_version(),
        "metadata": metadata,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "event_path": str(event_path),
        "event_sha256": sha256_file(event_path),
        "attempt": attempt,
        "trace": {
            "events": len(parsed.events),
            "blocking_futex_waits": parsed.blocking_waits,
            "matched_futex_waits": parsed.matched_waits,
            "unmatched_futex_waits": parsed.unmatched_waits,
            "happens_before_edges": parsed.happens_before_edges,
            "unresolved_happens_before": parsed.unresolved_happens_before,
            "scheduler_quantum_basic_blocks": args.quantum,
        },
        "command": command,
        "stdout": stdout,
    }
    write_json_atomic(args.artifact.resolve(), manifest)
    print(
        f"{manifest['case']}: collected complete DRD graph ({len(parsed.events)} events)"
    )


def load_model(path: Path) -> dict[str, object]:
    model = json.loads(path.read_text())
    if model.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported modeled-runtime score model schema")
    if model.get("renderer_inputs_used") is not False:
        raise RuntimeError("modeled-runtime score model is not synthetic-only")
    if model.get("timing_claims_enabled") is not False:
        raise RuntimeError(
            "relative score model must not enable absolute timing claims"
        )
    if model.get("replay") != REPLAY_CONFIGURATION:
        raise RuntimeError("modeled-runtime replay configuration changed")
    return model


def _verify_manifest_files(manifest: dict[str, object]) -> None:
    metadata_path = Path(str(manifest["metadata_path"]))
    if sha256_file(metadata_path) != manifest["metadata_sha256"]:
        raise RuntimeError("benchmark metadata changed after collection")
    if manifest["kind"] == "callgrind":
        for profile in manifest["profile_files"]:
            if sha256_file(Path(str(profile["path"]))) != profile["sha256"]:
                raise RuntimeError("Callgrind profile changed after collection")
        parsed = {
            str(thread): events
            for thread, events in sorted(
                parse_thread_profiles(Path(str(manifest["profile_prefix"]))).items()
            )
        }
        if parsed != manifest["profiles"]:
            raise RuntimeError("Callgrind manifest does not match its raw profiles")
    else:
        for key in ("trace", "event"):
            if (
                sha256_file(Path(str(manifest[f"{key}_path"])))
                != manifest[f"{key}_sha256"]
            ):
                raise RuntimeError(f"DRD {key} changed after collection")


def _validate_pair(callgrind: dict[str, object], drd: dict[str, object] | None) -> None:
    _verify_manifest_files(callgrind)
    if drd is None:
        return
    _verify_manifest_files(drd)
    if callgrind["case"] != drd["case"]:
        raise RuntimeError("Callgrind and DRD artifacts describe different cases")
    if drd["trace"]["unmatched_futex_waits"] != 0:
        raise RuntimeError("DRD trace has unmatched blocking waits")
    if drd["trace"]["unresolved_happens_before"] != 0:
        raise RuntimeError("DRD trace has unresolved happens-before dependencies")
    for name in (*IDENTITY_FIELDS, *WORKLOAD_FIELDS, "checksum"):
        if callgrind["metadata"].get(name) != drd["metadata"].get(name):
            raise RuntimeError(f"Callgrind/DRD metadata mismatch for {name}")
    if callgrind["valgrind_version"] != drd["valgrind_version"]:
        raise RuntimeError("Callgrind and DRD used different Valgrind versions")


def estimate_score(
    callgrind: dict[str, object],
    drd: dict[str, object] | None,
    model: dict[str, object],
    replay_engine: Path,
) -> tuple[float, dict[str, object] | None]:
    profiles = {
        int(thread): {str(name): int(value) for name, value in events.items()}
        for thread, events in callgrind["profiles"].items()
    }
    repetitions = int(callgrind["metadata"]["repetitions"])
    with NativeReplayEngine(replay_engine.resolve()) as engine:
        costs, total_cost = engine.model_profile_costs(
            profiles, model["event_cost_model"]
        )
        if drd is None:
            return total_cost / repetitions, None
        replay = model["replay"]
        event_count = engine.load_graph("renderer", Path(str(drd["event_path"])))
        if event_count != int(drd["trace"]["events"]):
            raise RuntimeError("native replay loaded a different DRD event count")
        engine.register_attributions(
            "renderer",
            [
                attribution_request(
                    "modeled-runtime",
                    costs,
                    float(replay["residual_fraction"]),
                    str(replay["split_policy"]),
                )
            ],
        )
        result = engine.replay_batch(
            "renderer",
            [
                replay_request(
                    "parallel",
                    "modeled-runtime",
                    int(replay["cores"]),
                    str(replay["tie_policy"]),
                    float(replay["wake_latency_ns"]),
                    float(model["cross_thread_release_ns"]),
                    float(replay["replay_idle_scale"]),
                    float(replay["dependency_excess_scale"]),
                )
            ],
        )["parallel"]
        engine_info = engine.info()
    return float(result["modeled_makespan"]) / repetitions, {
        "result": result,
        "engine": engine_info,
    }


def _case_identity(manifest: dict[str, object]) -> dict[str, object]:
    metadata = manifest["metadata"]
    return {
        **{name: metadata[name] for name in IDENTITY_FIELDS},
        **{name: metadata[name] for name in WORKLOAD_FIELDS},
        "valgrind_version": manifest["valgrind_version"],
        "cache_geometry": manifest["cache_geometry"],
        "benchmark_metadata_schema": manifest["benchmark_metadata_schema"],
    }


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


def check_reference(
    result: dict[str, object],
    reference: dict[str, object],
    tolerance: float | None = None,
) -> None:
    if reference.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported modeled-runtime reference schema")
    reference_tolerance = validate_tolerance(reference.get("tolerance", -1.0))
    if tolerance is None:
        tolerance = reference_tolerance
    else:
        tolerance = validate_tolerance(tolerance)
    case = reference.get("cases", {}).get(result["case"])
    if case is None:
        raise RuntimeError(f"reference has no case {result['case']}")
    reference_identity = case.get("identity", {})
    observed_identity = result.get("identity", {})
    reference_version = (
        reference_identity.get("valgrind_version")
        if isinstance(reference_identity, dict)
        else None
    )
    observed_version = (
        observed_identity.get("valgrind_version")
        if isinstance(observed_identity, dict)
        else None
    )
    result["reference_valgrind_version"] = reference_version
    result["observed_valgrind_version"] = observed_version
    result["valgrind_version_changed"] = reference_version != observed_version
    observed = validate_score(result["modeled_runtime_score_ns"], "observed")
    expected = validate_score(case["modeled_runtime_score_ns"], "reference")
    ratio = observed / expected
    result["reference_modeled_runtime_score_ns"] = expected
    result["reference_ratio"] = ratio
    result["relative_error"] = ratio - 1.0
    if ratio > 1.0 + tolerance:
        raise RuntimeError(
            f"modeled-runtime score for {result['case']} is {ratio:.3f}x reference; "
            f"required <= {1.0 + tolerance:.2f}x"
        )


def evaluate(args: argparse.Namespace) -> None:
    output_path = args.output.resolve()
    failed_path = output_path.with_suffix(".failed.json")
    output_path.unlink(missing_ok=True)
    failed_path.unlink(missing_ok=True)
    callgrind = load_manifest(args.callgrind.resolve(), "callgrind")
    fixture = str(callgrind["metadata"]["fixture"])
    drd = load_manifest(args.drd.resolve(), "drd") if args.drd else None
    if fixture == "parallel" and drd is None:
        raise RuntimeError("parallel evaluation requires a DRD artifact")
    if fixture == "serial" and drd is not None:
        raise RuntimeError("serial evaluation must not use a DRD artifact")
    _validate_pair(callgrind, drd)
    model = load_model(args.model.resolve())
    score, replay = estimate_score(callgrind, drd, model, args.replay_engine.resolve())
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "evaluation",
        "case": callgrind["case"],
        "model_id": model["model_id"],
        "model_sha256": sha256_file(args.model.resolve()),
        "score_semantics": "relative_modeled_runtime_not_absolute_wall_time",
        "modeled_runtime_score_ns": score,
        "modeled_nanoseconds_per_pixel": score
        / (int(callgrind["metadata"]["width"]) * int(callgrind["metadata"]["height"])),
        "checksum": callgrind["metadata"]["checksum"],
        "identity": _case_identity(callgrind),
        "replay": replay,
    }
    failure = None
    if args.reference:
        try:
            check_reference(
                result, json.loads(args.reference.read_text()), args.tolerance
            )
        except RuntimeError as error:
            failure = error
    result["status"] = "failed" if failure else "passed"
    write_json_atomic(failed_path if failure else output_path, result)
    print(
        f"{result['case']}: {score:.3f} modeled ns/call"
        + (
            f", {result['reference_ratio']:.3f}x reference"
            if "reference_ratio" in result
            else ""
        )
    )
    if failure:
        raise failure


def freeze_reference(args: argparse.Namespace) -> None:
    tolerance = validate_tolerance(args.tolerance)
    model_path = args.model.resolve()
    model_hash = sha256_file(model_path)
    model_id = json.loads(model_path.read_text()).get("model_id")
    if not model_id:
        raise RuntimeError("render model has no model_id")
    cases: dict[str, object] = {}
    for path in args.results:
        result_path = path.resolve()
        result = json.loads(result_path.read_text())
        if result.get("kind") != "evaluation":
            raise RuntimeError(
                f"expected evaluation artifact, got {result.get('kind')!r}"
            )
        schema = result.get("schema_version")
        if schema == SCHEMA_VERSION:
            if result.get("model_sha256") != model_hash:
                raise RuntimeError(f"evaluation {path} used a different model")
        elif schema == NATIVE_EVALUATION_SCHEMA_VERSION:
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

    callgrind = subparsers.add_parser("callgrind")
    callgrind.add_argument("--benchmark", required=True, type=Path)
    callgrind.add_argument("--fixture", required=True, choices=("serial", "parallel"))
    callgrind.add_argument("--scenario", required=True, choices=SCENARIOS)
    callgrind.add_argument("--output-dir", required=True, type=Path)
    callgrind.add_argument("--artifact", required=True, type=Path)
    callgrind.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    callgrind.set_defaults(function=collect_callgrind)

    drd = subparsers.add_parser("drd")
    drd.add_argument("--benchmark", required=True, type=Path)
    drd.add_argument("--scenario", required=True, choices=SCENARIOS)
    drd.add_argument("--output-dir", required=True, type=Path)
    drd.add_argument("--artifact", required=True, type=Path)
    drd.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    drd.add_argument("--quantum", type=int, default=DEFAULT_QUANTUM)
    drd.add_argument("--attempts", type=int, default=DEFAULT_DRD_ATTEMPTS)
    drd.set_defaults(function=collect_drd)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--callgrind", required=True, type=Path)
    evaluation.add_argument("--drd", type=Path)
    evaluation.add_argument("--model", required=True, type=Path)
    evaluation.add_argument("--reference", type=Path)
    evaluation.add_argument("--replay-engine", required=True, type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument("--tolerance", type=float)
    evaluation.set_defaults(function=evaluate)

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
    require_supported_host()
    if getattr(args, "repetitions", 1) <= 0:
        raise RuntimeError("repetitions must be positive")
    if getattr(args, "attempts", 1) <= 0:
        raise RuntimeError("DRD attempts must be positive")
    args.function(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"run_render_valgrind_ci.py: {error}", file=sys.stderr)
        raise SystemExit(1)
