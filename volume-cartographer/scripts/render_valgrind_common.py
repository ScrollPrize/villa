#!/usr/bin/env python3
"""Shared helpers for deterministic synthetic-renderer Valgrind runs."""

from __future__ import annotations

import glob
import json
import platform
import subprocess
import sys
from pathlib import Path

CACHE_GEOMETRY = {
    "I1": "32768,8,64",
    "D1": "32768,8,64",
    "LL": "8388608,16,64",
}
SCENARIOS = (
    "full_res",
    "fallback_3",
    "mixed_correlated",
    "mixed_shuffled",
)


def require_supported_host() -> None:
    machine = platform.machine().lower()
    if sys.platform != "linux" or machine not in {"x86_64", "amd64"}:
        raise RuntimeError("render Valgrind benchmark supports Linux amd64 only")


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


def parse_thread_profiles(prefix: Path) -> dict[int, dict[str, int]]:
    profiles: dict[int, dict[str, int]] = {}
    for name in sorted(glob.glob(f"{prefix}-*")):
        path = Path(name)
        thread = None
        for line in path.read_text().splitlines():
            if line.startswith("thread:"):
                thread = int(line.split()[1])
                break
        if thread is None:
            raise RuntimeError(f"Callgrind profile has no thread ID: {path}")
        if thread in profiles:
            raise RuntimeError(f"duplicate Callgrind profile for thread {thread}")
        profiles[thread] = parse_callgrind(path)
    if not profiles:
        raise RuntimeError(f"no separate-thread Callgrind profiles at {prefix}-*")
    return profiles


def load_renderer_metadata(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    required = {
        "scenario",
        "fixture",
        "width",
        "height",
        "tile_size",
        "repetitions",
        "measured_pixels",
        "checksum",
        "observed_threads",
        "worker_override",
        "compiler_id",
        "compiler_version",
        "build_type",
        "architecture_target",
    }
    missing = required.difference(value)
    if missing:
        raise RuntimeError(f"benchmark metadata is missing {sorted(missing)}")
    return value


def renderer_command(
    benchmark: Path,
    fixture: str,
    scenario: str,
    metadata: Path,
    repetitions: int | None,
    *,
    callgrind: bool,
) -> list[str]:
    command = [
        str(benchmark),
        "--fixture",
        fixture,
        "--scenario",
        scenario,
        "--native-trials",
        "1",
        "--metadata",
        str(metadata),
    ]
    if repetitions is not None:
        command.extend(("--repetitions", str(repetitions)))
    if callgrind:
        command.append("--callgrind")
    return command


def callgrind_command(
    benchmark: Path,
    fixture: str,
    scenario: str,
    metadata: Path,
    output: Path,
    repetitions: int | None,
    *,
    separate_threads: bool,
    fair_scheduler: bool = True,
) -> list[str]:
    command = [
        "valgrind",
        "--tool=callgrind",
        "--instr-atstart=no",
    ]
    if fair_scheduler:
        command.append("--fair-sched=yes")
    if separate_threads:
        command.append("--separate-threads=yes")
    command.extend(
        [
            "--collect-systime=no",
            "--cache-sim=yes",
            "--branch-sim=yes",
            f"--I1={CACHE_GEOMETRY['I1']}",
            f"--D1={CACHE_GEOMETRY['D1']}",
            f"--LL={CACHE_GEOMETRY['LL']}",
            f"--callgrind-out-file={output}",
            *renderer_command(
                benchmark,
                fixture,
                scenario,
                metadata,
                repetitions,
                callgrind=True,
            ),
        ]
    )
    return command
