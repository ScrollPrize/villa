#!/usr/bin/env python3
"""Run one synthetic rendering case under a fixed Callgrind model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 2
MODEL_VERSION = 2
CALIBRATION_SCHEMA_VERSION = 1
REQUIRED_MODEL_EVENTS = (
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
CACHE_GEOMETRY = {
    "I1": "32768,8,64",
    "D1": "32768,8,64",
    "LL": "8388608,16,64",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--fixture", required=True, choices=("serial", "parallel"))
    parser.add_argument(
        "--scenario",
        required=True,
        choices=(
            "full_res",
            "fallback_3",
            "mixed_correlated",
            "mixed_shuffled",
            "full_res_shuffled",
            "fallback_3_shuffled",
            "full_res_cache_stress",
            "full_res_cache_stress_shuffled",
        ),
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--repetitions", type=int)
    return parser.parse_args()


def require_supported_host() -> None:
    machine = platform.machine().lower()
    if sys.platform != "linux" or machine not in {"x86_64", "amd64"}:
        print("render Callgrind benchmark supports Linux amd64 only", file=sys.stderr)
        raise SystemExit(77)


def valgrind_version() -> str:
    result = subprocess.run(
        ["valgrind", "--version"], check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def parse_callgrind(path: Path) -> dict[str, int]:
    events: list[str] | None = None
    totals: list[int] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("events:"):
            events = line.split()[1:]
        elif line.startswith(("totals:", "summary:")):
            totals = [int(value) for value in line.split()[1:]]
    if not events or totals is None or len(totals) > len(events):
        raise RuntimeError(f"cannot parse named Callgrind summary from {path}")
    totals.extend(0 for _ in range(len(events) - len(totals)))
    return dict(zip(events, totals, strict=True))


def load_calibration(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise RuntimeError("unsupported rendering calibration schema")
    if value.get("model_version") != MODEL_VERSION:
        raise RuntimeError("rendering calibration model version changed")
    return value


def modeled_work_cycles(
    events: dict[str, int], calibration: dict[str, object]
) -> float:
    missing = set(REQUIRED_MODEL_EVENTS).difference(events)
    if missing:
        raise RuntimeError(f"Callgrind output is missing events {sorted(missing)}")
    formula = calibration["formula"]
    return (
        events["Ir"] / float(formula["modeled_ipc"])
        + float(formula["l1_miss_cycles"])
        * (events["I1mr"] + events["D1mr"] + events["D1mw"])
        + float(formula["last_level_miss_cycles"])
        * (events["ILmr"] + events["DLmr"] + events["DLmw"])
        + float(formula["branch_mispredict_cycles"]) * (events["Bcm"] + events["Bim"])
    )


def calibrated_estimate(
    events: dict[str, int],
    pixels: int,
    fixture: str,
    workers: int,
    calibration: dict[str, object],
) -> dict[str, float]:
    effective = (
        1.0
        if fixture == "serial"
        else calibration["effective_parallelism"].get(str(workers))
    )
    if effective is None:
        raise RuntimeError(
            f"calibration has no effective parallelism for {workers} workers"
        )
    work_cycles = modeled_work_cycles(events, calibration)
    wall_cycles = work_cycles / float(effective)
    cycles_per_pixel = wall_cycles / pixels
    nanoseconds_per_cycle = float(calibration["nanoseconds_per_modeled_cycle"])
    nanoseconds_per_pixel = cycles_per_pixel * nanoseconds_per_cycle
    return {
        "modeled_work_cycles": work_cycles,
        "modeled_wall_cycles": wall_cycles,
        "modeled_cycles_per_pixel": cycles_per_pixel,
        "nanoseconds_per_modeled_cycle": nanoseconds_per_cycle,
        "estimated_nanoseconds_per_pixel": nanoseconds_per_pixel,
        "estimated_mpx_per_second": 1000.0 / nanoseconds_per_pixel,
    }


def load_metadata(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    required = {
        "scenario",
        "fixture",
        "measured_pixels",
        "checksum",
        "observed_threads",
    }
    missing = required.difference(value)
    if missing:
        raise RuntimeError(f"benchmark metadata is missing {sorted(missing)}")
    return value


def check_baseline(
    result: dict[str, object], baseline_path: Path, case_name: str
) -> None:
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported rendering benchmark baseline schema")
    if baseline.get("model_version") != MODEL_VERSION:
        raise RuntimeError("rendering benchmark cost-model version changed")
    case = baseline.get("cases", {}).get(case_name)
    if not case:
        raise RuntimeError(f"baseline has no case {case_name}")

    expected_checksum = case.get("checksum")
    if expected_checksum is not None and result["checksum"] != expected_checksum:
        raise RuntimeError(
            f"checksum changed: {result['checksum']} != {expected_checksum}"
        )

    observed = float(result["modeled_cycles_per_pixel"])
    maximum = float(case["modeled_cycles_per_pixel_max"])
    if observed > maximum:
        raise RuntimeError(
            f"modeled cost regression for {case_name}: {observed:.3f} > {maximum:.3f}"
        )


def main() -> int:
    args = parse_args()
    require_supported_host()
    if args.check and (not args.baseline or not args.calibration):
        raise RuntimeError("--check requires --baseline and --calibration")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_name = f"{args.fixture}/{args.scenario}"
    default_workers = 1 if args.fixture == "serial" else 4
    workers = args.workers or default_workers
    worker_suffix = f"-w{workers}" if args.workers is not None else ""
    stem = f"{args.fixture}{worker_suffix}-{args.scenario}"
    raw_path = args.output_dir / f"callgrind.{stem}.out"
    metadata_path = args.output_dir / f"metadata.{stem}.json"
    result_path = args.output_dir / f"result.{stem}.json"

    env = os.environ.copy()
    env["VC_RENDER_SAMPLER_THREADS"] = str(workers)
    command = [
        "valgrind",
        "--tool=callgrind",
        "--instr-atstart=no",
        "--collect-systime=no",
        "--cache-sim=yes",
        "--branch-sim=yes",
        f"--I1={CACHE_GEOMETRY['I1']}",
        f"--D1={CACHE_GEOMETRY['D1']}",
        f"--LL={CACHE_GEOMETRY['LL']}",
        f"--callgrind-out-file={raw_path}",
        str(args.benchmark),
        "--callgrind",
        "--fixture",
        args.fixture,
        "--scenario",
        args.scenario,
        "--metadata",
        str(metadata_path),
    ]
    if args.repetitions:
        command.extend(("--repetitions", str(args.repetitions)))
    subprocess.run(command, check=True, env=env)

    metadata = load_metadata(metadata_path)
    events = parse_callgrind(raw_path)
    pixels = int(metadata["measured_pixels"])
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "case": case_name,
        "valgrind_version": valgrind_version(),
        "cache_geometry": CACHE_GEOMETRY,
        "worker_override": int(env["VC_RENDER_SAMPLER_THREADS"]),
        **metadata,
        "events": events,
        "events_per_pixel": {
            name: value / pixels for name, value in sorted(events.items())
        },
        "command": command,
    }
    if args.calibration:
        calibration = load_calibration(args.calibration)
        result.update(
            calibrated_estimate(events, pixels, args.fixture, workers, calibration)
        )
        result["calibration_reference"] = calibration["reference"]
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))

    if args.check:
        check_baseline(result, args.baseline, case_name)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"run_render_callgrind.py: {error}", file=sys.stderr)
        raise SystemExit(1)
