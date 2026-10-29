#!/usr/bin/env python3
"""Maintain the C++ synthetic-renderer Callgrind performance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCENARIOS = (
    "full_res",
    "fallback_3",
    "mixed_correlated",
    "mixed_shuffled",
)
SCHEMA_VERSION = 1
NATIVE_EVALUATION_SCHEMA_VERSIONS = {2, 3}
DEFAULT_TOLERANCE = 0.05


def collect_evaluate(args: argparse.Namespace) -> None:
    """Allow two fresh retries for capture, replay, or reference-check failures."""
    case = args.case_dir.resolve()
    capture = case / "callgrind"
    output = case / "evaluation.json"
    case.mkdir(parents=True, exist_ok=True)
    reference = getattr(args, "reference", None)
    checked = case / "checked.json"
    checked.unlink(missing_ok=True)
    for attempt in range(1, 4):
        reuse = (attempt == 1 and getattr(args, "reuse_capture", False)
                 and (capture / "collected.stamp").is_file() and output.is_file())
        if not reuse:
            output.unlink(missing_ok=True)
            shutil.rmtree(capture, ignore_errors=True)
            capture.mkdir()
        print(f"{args.fixture}/{args.scenario}: capture attempt {attempt}/3", flush=True)
        scheduler = capture / "scheduler.log"
        collect = [
            str(args.valgrind), "--tool=callgrind", "--instr-atstart=no",
            "--fair-sched=yes", "--scheduling-quantum=10000",
            "--separate-threads=yes", "--dump-every-bb=10000",
            "--combine-dumps=no", "--collect-systime=no", "--cache-sim=yes",
            "--branch-sim=yes", "--I1=32768,8,64", "--D1=32768,8,64",
            "--LL=8388608,16,64",
        ]
        if args.fixture == "parallel":
            collect += ["--trace-sched=yes", "--trace-syscalls=yes", "--time-stamp=yes",
                        f"--log-file={scheduler}"]
        collect += [
            f"--callgrind-out-file={capture / 'callgrind.out'}", str(args.benchmark),
            "--fixture", args.fixture, "--scenario", args.scenario,
            "--native-trials", "1", "--metadata", str(capture / "metadata.json"),
            "--repetitions", "1", "--callgrind",
        ]
        evaluate = [
            str(args.replay), "evaluate-render", "--fixture", args.fixture,
            "--scenario", args.scenario, "--callgrind-prefix", str(capture / "callgrind.out"),
            "--callgrind-metadata", str(capture / "metadata.json"),
            "--model", str(args.model), "--output", str(output),
        ]
        if args.fixture == "parallel":
            evaluate += ["--callgrind-scheduler", str(scheduler)]
        env = dict(os.environ, VC_RENDER_SAMPLER_THREADS="4" if args.fixture == "parallel" else "1")
        result = None if reuse else subprocess.run(collect, env=env, check=False)
        if not reuse and result.returncode == 0:
            result = subprocess.run(evaluate, check=False)
        valid = (reuse or result.returncode == 0) and output.is_file()
        if valid and reference is not None:
            gate = list(evaluate)
            gate[gate.index("--output") + 1] = str(checked)
            result = subprocess.run(gate + ["--reference", str(reference)], check=False)
            valid = result.returncode == 0 and checked.is_file()
        if valid:
            (capture / "collected.stamp").touch()
            return
        prefix = "failed-gate-attempt" if reference is not None else "failed-attempt"
        failed = case / f"{prefix}-{attempt}"
        shutil.rmtree(failed, ignore_errors=True)
        capture.rename(failed)
        for artifact in (output, checked, case / "checked.failed.json"):
            if artifact.exists():
                artifact.rename(failed / artifact.name)
        print(f"Capture/replay/reference attempt {attempt} failed; retained files in {failed}",
              file=sys.stderr, flush=True)
    raise RuntimeError(f"{args.fixture}/{args.scenario}: capture/replay/reference failed after 3 attempts")


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

    collect = subparsers.add_parser("collect-evaluate")
    for option in ("valgrind", "benchmark", "replay", "model", "case-dir"):
        collect.add_argument(f"--{option}", required=True, type=Path)
    collect.add_argument("--fixture", choices=("serial", "parallel"), required=True)
    collect.add_argument("--scenario", choices=SCENARIOS, required=True)
    collect.add_argument("--reference", type=Path)
    collect.add_argument("--reuse-capture", action="store_true")
    collect.set_defaults(function=collect_evaluate)

    freeze = subparsers.add_parser("freeze-reference")
    freeze.add_argument("--model", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    freeze.add_argument("results", nargs="+", type=Path)
    freeze.set_defaults(function=freeze_reference)

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
