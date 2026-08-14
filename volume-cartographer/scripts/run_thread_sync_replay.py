#!/usr/bin/env python3
"""Extract and replay passive Valgrind scheduler/futex traces."""

from __future__ import annotations

import argparse
import glob
import heapq
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from native_thread_sync_replay import (
    NativeReplayEngine,
    attribution_request,
    replay_request,
    resolve_replay_engine,
)
from render_valgrind_common import CACHE_GEOMETRY, parse_thread_profiles
from thread_sync_trace import (
    ParsedTrace,
    TraceEvent,
    parse_core_trace,
    parse_drd_trace,
    read_event_stream,  # noqa: F401
    write_event_stream,
)

SCHEMA_VERSION = 3


def assign_costs(
    events: list[TraceEvent],
    thread_costs: dict[int, float],
    residual_fraction: float,
    split_policy: str,
) -> None:
    by_thread: dict[int, list[TraceEvent]] = {}
    for event in events:
        event.duration = 0.0
        by_thread.setdefault(event.thread, []).append(event)

    for thread, thread_events in by_thread.items():
        if thread not in thread_costs:
            raise RuntimeError(f"trace thread {thread} has no Callgrind profile")
        windows: list[tuple[list[TraceEvent], float]] = []
        current: list[TraceEvent] = []
        blocked = False
        candidates: list[TraceEvent] = []
        for event in thread_events:
            current.append(event)
            if event.kind == "futex_resume":
                blocked = False
            if event.kind not in {"thread_start", "futex_resume"} and not blocked:
                candidates.append(event)
            if event.kind == "futex_wait" and bool(event.detail.get("blocked")):
                blocked = True
            if event.kind == "work_quantum":
                windows.append((candidates, 1.0))
                current = []
                candidates = []
        if current or candidates:
            windows.append((candidates, residual_fraction))

        eligible_windows = [
            (candidates, units) for candidates, units in windows if candidates
        ]
        thread_cost = float(thread_costs[thread])
        if not eligible_windows:
            if thread_cost != 0.0:
                raise RuntimeError(
                    f"trace thread {thread} has positive cost but no eligible event"
                )
            continue
        total_units = sum(units for _, units in eligible_windows)
        if total_units <= 0.0:
            if len(eligible_windows) != 1:
                raise RuntimeError(
                    f"trace thread {thread} has no positive attribution weight"
                )
            eligible_windows = [(eligible_windows[0][0], 1.0)]
            total_units = 1.0
        unit_cost = thread_cost / total_units
        for candidates, units in eligible_windows:
            if split_policy == "equal":
                share = unit_cost * units / len(candidates)
                for event in candidates:
                    event.duration += share
            elif split_policy == "front":
                candidates[0].duration += unit_cost * units
            elif split_policy == "back":
                candidates[-1].duration += unit_cost * units
            else:
                raise RuntimeError(f"unknown split policy {split_policy}")


def simulate(
    events: list[TraceEvent],
    cores: int,
    tie_policy: str,
    wake_latency: float = 0.0,
    cross_thread_latency: float = 0.0,
) -> dict[str, float]:
    if cores <= 0:
        raise RuntimeError("simulated core count must be positive")
    successors: list[list[int]] = [[] for _ in events]
    remaining = [0] * len(events)
    dependency_finish = [0.0] * len(events)
    dependency_kind: list[list[tuple[int, str]]] = [[] for _ in events]
    for event in events:
        unique: dict[int, str] = {}
        for predecessor, kind in event.dependencies:
            unique[predecessor] = kind
        dependency_kind[event.sequence] = list(unique.items())
        remaining[event.sequence] = len(unique)
        for predecessor in unique:
            successors[predecessor].append(event.sequence)

    initially_ready: set[int] = {
        sequence for sequence, count in enumerate(remaining) if count == 0
    }
    ready: set[int] = set(initially_ready) if tie_policy != "fifo" else set()
    ready_by_release: list[tuple[float, int]] = []
    eligible_fifo: list[int] = []
    if tie_policy == "fifo":
        ready_by_release = [
            (dependency_finish[sequence], sequence) for sequence in initially_ready
        ]
        heapq.heapify(ready_by_release)
    core_available = [0.0] * cores
    finish = [0.0] * len(events)
    start = [0.0] * len(events)
    last_thread = 0
    max_thread = max((event.thread for event in events), default=0)
    scheduled = 0
    idle = 0.0
    sync_delay = 0.0

    def has_ready() -> bool:
        return bool(ready or ready_by_release or eligible_fifo)

    while has_ready():
        core = min(range(cores), key=lambda index: (core_available[index], index))
        core_time = core_available[core]

        def candidate_key(sequence: int) -> tuple[float, int, int]:
            event = events[sequence]
            possible = max(core_time, dependency_finish[sequence])
            if tie_policy == "fifo":
                tie = event.sequence
            elif tie_policy == "round_robin":
                tie = (event.thread - last_thread - 1) % max_thread
            else:
                raise RuntimeError(f"unknown tie policy {tie_policy}")
            return possible, tie, event.sequence

        if tie_policy == "fifo":
            while ready_by_release and ready_by_release[0][0] <= core_time:
                _, candidate = heapq.heappop(ready_by_release)
                heapq.heappush(eligible_fifo, candidate)
            if eligible_fifo:
                sequence = heapq.heappop(eligible_fifo)
            else:
                _, sequence = heapq.heappop(ready_by_release)
        else:
            sequence = min(ready, key=candidate_key)
            ready.remove(sequence)
        event = events[sequence]
        release = dependency_finish[sequence]
        for predecessor, kind in dependency_kind[sequence]:
            edge_finish = finish[predecessor]
            if (
                kind in {"drd_happens_before", "futex_wake"}
                and events[predecessor].thread != event.thread
            ):
                edge_finish += cross_thread_latency
            if (
                kind == "futex_wake"
                and events[predecessor].thread == 1
                and event.thread != 1
            ):
                edge_finish += wake_latency
            release = max(release, edge_finish)
        event_start = max(core_time, release)
        start[sequence] = event_start
        finish[sequence] = event_start + event.duration
        idle += event_start - core_time
        program_finish = max(
            (
                finish[predecessor]
                for predecessor, kind in dependency_kind[sequence]
                if kind == "program_order"
            ),
            default=0.0,
        )
        sync_delay += max(0.0, release - program_finish)
        core_available[core] = finish[sequence]
        last_thread = event.thread
        scheduled += 1
        for successor in successors[sequence]:
            remaining[successor] -= 1
            dependency_finish[successor] = max(
                dependency_finish[successor], finish[sequence]
            )
            if remaining[successor] == 0:
                if tie_policy == "fifo":
                    heapq.heappush(
                        ready_by_release,
                        (dependency_finish[successor], successor),
                    )
                else:
                    ready.add(successor)

    if scheduled != len(events):
        blocked = [index for index, count in enumerate(remaining) if count > 0]
        raise RuntimeError(f"dependency graph is cyclic at events {blocked[:8]}")
    makespan = max(finish, default=0.0)
    work = sum(event.duration for event in events)
    return {
        "modeled_work": work,
        "modeled_makespan": makespan,
        "simulated_core_idle": idle,
        "logical_sync_delay": sync_delay,
        "utilization": work / (cores * makespan) if makespan else 0.0,
    }


def dependency_critical_path(
    events: list[TraceEvent],
    wake_latency: float = 0.0,
    cross_thread_latency: float = 0.0,
    excluded_dependency_kinds: frozenset[str] = frozenset(),
) -> float:
    successors: list[list[int]] = [[] for _ in events]
    remaining = [0] * len(events)
    dependencies: list[list[tuple[int, str]]] = [[] for _ in events]
    for event in events:
        unique = {
            predecessor: kind
            for predecessor, kind in event.dependencies
            if kind not in excluded_dependency_kinds
        }
        dependencies[event.sequence] = list(unique.items())
        remaining[event.sequence] = len(unique)
        for predecessor in unique:
            successors[predecessor].append(event.sequence)

    ready = {index for index, count in enumerate(remaining) if count == 0}
    finish = [0.0] * len(events)
    completed = 0
    while ready:
        sequence = min(ready)
        ready.remove(sequence)
        event = events[sequence]
        release = 0.0
        for predecessor, kind in dependencies[sequence]:
            edge_finish = finish[predecessor]
            if (
                kind in {"drd_happens_before", "futex_wake"}
                and events[predecessor].thread != event.thread
            ):
                edge_finish += cross_thread_latency
            if (
                kind == "futex_wake"
                and events[predecessor].thread == 1
                and event.thread != 1
            ):
                edge_finish += wake_latency
            release = max(release, edge_finish)
        finish[sequence] = release + event.duration
        completed += 1
        for successor in successors[sequence]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.add(successor)

    if completed != len(events):
        raise RuntimeError("dependency graph is cyclic")
    return max(finish, default=0.0)


def simulate_adjusted(
    events: list[TraceEvent],
    cores: int,
    tie_policy: str,
    wake_latency: float = 0.0,
    cross_thread_latency: float = 0.0,
    replay_idle_scale: float = 1.0,
    dependency_excess_scale: float = 1.0,
) -> dict[str, float]:
    if not 0.0 <= replay_idle_scale <= 1.0:
        raise RuntimeError("replay idle scale must be between zero and one")
    if not 0.0 <= dependency_excess_scale <= 1.0:
        raise RuntimeError("dependency excess scale must be between zero and one")
    result = simulate(
        events,
        cores,
        tie_policy,
        wake_latency,
        cross_thread_latency,
    )
    critical_path = dependency_critical_path(
        events,
        wake_latency,
        cross_thread_latency,
    )
    hard_critical_path = dependency_critical_path(
        events,
        wake_latency,
        cross_thread_latency,
        frozenset({"drd_happens_before"}),
    )
    work_bound = result["modeled_work"] / cores
    hard_lower_bound = max(hard_critical_path, work_bound)
    lower_bound = max(hard_lower_bound, critical_path)
    raw_makespan = result["modeled_makespan"]
    tolerance = 1e-9 * max(1.0, raw_makespan, lower_bound)
    if raw_makespan + tolerance < lower_bound:
        raise RuntimeError("replay makespan is below its scheduling lower bound")
    lower_bound = min(lower_bound, raw_makespan)
    replay_excess = max(0.0, raw_makespan - lower_bound)
    dependency_excess = max(0.0, lower_bound - hard_lower_bound)
    adjusted = (
        hard_lower_bound
        + dependency_excess_scale * dependency_excess
        + replay_idle_scale * replay_excess
    )
    result.update(
        {
            "raw_replay_makespan": raw_makespan,
            "dependency_critical_path": critical_path,
            "hard_dependency_critical_path": hard_critical_path,
            "work_per_core_lower_bound": work_bound,
            "hard_schedule_lower_bound": hard_lower_bound,
            "schedule_lower_bound": lower_bound,
            "dependency_excess": dependency_excess,
            "dependency_excess_scale": dependency_excess_scale,
            "raw_replay_excess": replay_excess,
            "replay_idle_scale": replay_idle_scale,
            "raw_simulated_core_idle": result["simulated_core_idle"],
            "modeled_makespan": adjusted,
            "simulated_core_idle": max(0.0, cores * adjusted - result["modeled_work"]),
            "utilization": (
                result["modeled_work"] / (cores * adjusted) if adjusted else 0.0
            ),
        }
    )
    return result


def replay_scales_for_model(loaded_model: dict[str, object]) -> tuple[float, float]:
    schema = int(loaded_model.get("schema_version", 1))
    parameters = loaded_model.get("parameters", {})
    if schema <= 2:
        if "dependency_excess_scale" in parameters:
            raise RuntimeError(
                "schema 1-2 models cannot contain dependency_excess_scale"
            )
        return float(parameters.get("replay_idle_scale", 1.0)), 1.0
    if "replay_idle_scale" in parameters:
        raise RuntimeError("schema 3+ models cannot contain replay_idle_scale")
    return 1.0, float(parameters.get("dependency_excess_scale", 1.0))


def _parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for part in value.split(","):
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            cpus.extend(range(first, last + 1))
        else:
            cpus.append(int(part))
    if not cpus or len(set(cpus)) != len(cpus):
        raise argparse.ArgumentTypeError("CPU affinity must contain unique CPUs")
    return cpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--workload",
        default="dispatch",
        choices=("dispatch", "synthetic", "renderer"),
    )
    parser.add_argument("--mode", default="futures", choices=("futures", "batch"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--work-iterations", type=int, default=400000)
    parser.add_argument("--task-iteration-skew", type=int, default=0)
    parser.add_argument(
        "--work-kind",
        default="alu",
        choices=(
            "alu",
            "fp",
            "branch",
            "pointer",
            "stream-read",
            "stream-write",
            "cache-read",
            "cache-write",
            "grid-sample",
            "mixed-grid-phase",
            "mixed-grid-random",
            "mixed",
        ),
    )
    parser.add_argument("--working-set-bytes", type=int, default=256 * 1024)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--warmup-rounds", type=int, default=32)
    parser.add_argument("--quantum", type=int, default=10000)
    parser.add_argument("--native-trials", type=int, default=5)
    parser.add_argument("--trace-trials", type=int, default=3)
    parser.add_argument(
        "--scenario",
        default="full_res",
        choices=(
            "full_res",
            "fallback_1",
            "fallback_3",
            "mixed_correlated",
            "mixed_shuffled",
        ),
    )
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--native-repetitions", type=int)
    parser.add_argument("--cpu-affinity", type=_parse_cpu_list)
    parser.add_argument("--wake-latency", type=float, default=0.0)
    parser.add_argument("--cross-thread-latency", type=float)
    parser.add_argument(
        "--cost-model",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "core/test/data/render_callgrind_model.json",
    )
    parser.add_argument("--event-cost-model", type=Path)
    parser.add_argument("--replay-engine", type=Path)
    return parser.parse_args()


def _require_supported_host() -> None:
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("thread sync replay currently supports Linux amd64 only")


def _benchmark_args(
    args: argparse.Namespace, repetitions: int | None = None
) -> list[str]:
    if args.workload == "renderer":
        repetitions = args.repetitions if repetitions is None else repetitions
        return [
            str(args.benchmark),
            "--fixture",
            "parallel",
            "--scenario",
            args.scenario,
            "--repetitions",
            str(repetitions),
            "--native-trials",
            "1",
        ]
    return [
        str(args.benchmark),
        "--mode",
        args.mode,
        "--workers",
        str(args.workers),
        "--tasks",
        str(args.tasks),
        "--work-iterations",
        str(args.work_iterations),
        "--task-iteration-skew",
        str(args.task_iteration_skew),
        "--work-kind",
        args.work_kind,
        "--working-set-bytes",
        str(args.working_set_bytes),
        "--rounds",
        str(args.rounds),
        "--warmup-rounds",
        str(args.warmup_rounds),
    ]


def _callgrind_benchmark_args(args: argparse.Namespace) -> list[str]:
    command = _benchmark_args(args)
    if args.workload == "renderer":
        command.append("--callgrind")
    return command


def _run(
    command: list[str],
    affinity: list[int] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if affinity:
        command = ["taskset", "-c", ",".join(str(cpu) for cpu in affinity), *command]
    return subprocess.run(command, check=True, text=True, capture_output=True, env=env)


def _native_trials(
    command: list[str],
    cpus: list[int],
    core_count: int,
    trials: int,
    env: dict[str, str] | None = None,
) -> tuple[list[float], list[float]]:
    external = []
    internal = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        completed = _run(command, cpus[:core_count], env)
        external.append(float(time.perf_counter_ns() - start))
        internal.append(float(json.loads(completed.stdout)["wall_seconds"]) * 1e9)
    return external, internal


def _native_renderer_trials(
    command: list[str],
    cpus: list[int],
    core_count: int,
    trials: int,
    env: dict[str, str],
    repetitions: int,
) -> tuple[list[float], list[float]]:
    external = []
    internal = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        run = _run(command, cpus[:core_count], env)
        external.append(float(time.perf_counter_ns() - start))
        output = json.loads(run.stdout)
        internal.append(float(output["native_median_seconds"]) * 1e9 / repetitions)
    return external, internal


def _valgrind_scheduler_options(workload: str) -> list[str]:
    return ["--fair-sched=yes"] if workload in {"synthetic", "renderer"} else []


def _valgrind_trace_tool_options(workload: str) -> list[str]:
    if workload not in {"synthetic", "renderer"}:
        return ["--tool=none"]
    return [
        "--tool=drd",
        "--trace-segment=yes",
        "--trace-csw=yes",
    ]


def _parallel_core_count(workload: str, workers: int) -> int:
    return workers + 1 if workload in {"synthetic", "renderer"} else workers


def main() -> int:
    args = parse_args()
    _require_supported_host()
    if args.workers <= 0 or args.quantum <= 0:
        raise RuntimeError("workers, tasks, rounds, and quantum must be positive")
    if args.workload in {"dispatch", "synthetic"} and (
        args.tasks <= 0 or args.rounds <= 0
    ):
        raise RuntimeError("synthetic tasks and rounds must be positive")
    if args.workload == "renderer" and args.repetitions <= 0:
        raise RuntimeError("renderer repetitions must be positive")
    if args.native_repetitions is not None and args.native_repetitions <= 0:
        raise RuntimeError("native repetitions must be positive")
    if (
        args.work_iterations < 0
        or args.task_iteration_skew < 0
        or args.warmup_rounds < 0
        or args.native_trials <= 0
        or args.trace_trials <= 0
    ):
        raise RuntimeError("work iterations must be nonnegative and trials positive")
    parallel_cores = _parallel_core_count(args.workload, args.workers)
    cpus = args.cpu_affinity or list(range(parallel_cores))
    if len(cpus) < parallel_cores:
        raise RuntimeError(
            "CPU affinity must contain the workers plus a caller CPU"
            if args.workload in {"synthetic", "renderer"}
            else "CPU affinity must contain at least --workers CPUs"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_engine_path = resolve_replay_engine(args.replay_engine, args.benchmark)
    if args.workload == "renderer":
        stem = f"renderer-{args.scenario}-w{args.workers}-r{args.repetitions}"
    else:
        prefix = "synthetic-" if args.workload == "synthetic" else ""
        stem = (
            f"{prefix}{args.mode}-w{args.workers}-t{args.tasks}"
            f"-i{args.work_iterations}-s{args.task_iteration_skew}"
            f"-u{args.warmup_rounds}-r{args.rounds}"
        )
        if args.workload == "synthetic":
            stem += f"-k{args.work_kind}-b{args.working_set_bytes}"
    trace_path = args.output_dir / f"core-trace.{stem}.trial0.log"
    event_path = args.output_dir / f"sync-events.{stem}.jsonl"
    callgrind_prefix = args.output_dir / f"callgrind-sync.{stem}"
    result_path = args.output_dir / f"sync-replay.{stem}.json"
    benchmark_command = _benchmark_args(args)
    callgrind_benchmark_command = _callgrind_benchmark_args(args)
    native_repetitions = args.native_repetitions or args.repetitions
    native_benchmark_command = _benchmark_args(args, native_repetitions)
    benchmark_env = os.environ.copy()
    if args.workload == "renderer":
        benchmark_env["VC_RENDER_SAMPLER_THREADS"] = str(args.workers)

    trace_commands: list[list[str]] = []
    trace_outputs: list[str] = []
    parsed_trials: list[ParsedTrace] = []
    trace_paths: list[Path] = []
    for trial in range(args.trace_trials):
        trial_path = args.output_dir / f"core-trace.{stem}.trial{trial}.log"
        trace_command = [
            "valgrind",
            *_valgrind_trace_tool_options(args.workload),
            *_valgrind_scheduler_options(args.workload),
            "--time-stamp=yes",
            f"--scheduling-quantum={args.quantum}",
            "--trace-sched=yes",
            "--trace-syscalls=yes",
            f"--log-file={trial_path}",
            *benchmark_command,
        ]
        trace_run = _run(trace_command, cpus, benchmark_env)
        trace_commands.append(trace_command)
        trace_outputs.append(trace_run.stdout.strip())
        trace_paths.append(trial_path)
        parsed_trials.append(
            parse_drd_trace(trial_path)
            if args.workload in {"synthetic", "renderer"}
            else parse_core_trace(trial_path)
        )
    parsed = parsed_trials[0]
    event_paths = [
        (
            event_path
            if trial == 0
            else args.output_dir / f"sync-events.{stem}.trial{trial}.jsonl"
        )
        for trial in range(args.trace_trials)
    ]
    for trial_path, trial in zip(event_paths, parsed_trials, strict=True):
        write_event_stream(trial_path, trial.events)

    for old_path in glob.glob(f"{callgrind_prefix}*"):
        Path(old_path).unlink()
    callgrind_command = [
        "valgrind",
        "--tool=callgrind",
        "--instr-atstart=no",
        *_valgrind_scheduler_options(args.workload),
        "--separate-threads=yes",
        "--collect-systime=no",
        "--cache-sim=yes",
        "--branch-sim=yes",
        f"--I1={CACHE_GEOMETRY['I1']}",
        f"--D1={CACHE_GEOMETRY['D1']}",
        f"--LL={CACHE_GEOMETRY['LL']}",
        f"--callgrind-out-file={callgrind_prefix}",
        *callgrind_benchmark_command,
    ]
    callgrind_run = _run(callgrind_command, cpus, benchmark_env)
    profiles = parse_thread_profiles(callgrind_prefix)
    instruction_costs = {
        thread: float(events["Ir"]) for thread, events in profiles.items()
    }
    cost_inputs: list[tuple[str, dict[int, float]]] = [
        ("instructions", instruction_costs)
    ]
    event_model = None
    event_costs = None
    fixed_process_ns = 0.0
    base_process_ns = 0.0
    per_worker_startup_ns = 0.0
    cross_thread_latency = args.cross_thread_latency or 0.0
    replay_idle_scale = 1.0
    dependency_excess_scale = 1.0
    if args.event_cost_model is not None:
        from passive_event_model import modeled_thread_costs_ns

        loaded_model = json.loads(args.event_cost_model.read_text())
        event_model = loaded_model.get("event_cost_model", loaded_model)
        loaded_parameters = loaded_model.get("parameters", {})
        if args.cross_thread_latency is None:
            cross_thread_latency = float(
                loaded_parameters.get("cross_thread_release_ns", 0.0)
            )
        replay_idle_scale, dependency_excess_scale = replay_scales_for_model(
            loaded_model
        )
        if event_model.get("renderer_inputs_used") is not False:
            raise RuntimeError("event-cost model is not synthetic-only")
        event_costs = modeled_thread_costs_ns(profiles, event_model)
        base_process_ns = float(
            loaded_model.get("parameters", {}).get(
                "fixed_process_ns", event_model["fixed_process_ns"]
            )
        )
        per_worker_startup_ns = float(
            loaded_model.get("parameters", {}).get("per_worker_startup_ns", 0.0)
        )
        fixed_process_ns = (
            base_process_ns + max(0, args.workers - 1) * per_worker_startup_ns
        )
        cost_inputs.append(("synthetic_event_ns", event_costs))
    elif args.workload != "synthetic":
        from run_render_callgrind import modeled_work_cycles

        model = json.loads(args.cost_model.read_text())
        cost_inputs.append(
            (
                "modeled_events",
                {
                    thread: modeled_work_cycles(events, model)
                    for thread, events in profiles.items()
                },
            )
        )

    simulations: dict[str, object] = {}
    with NativeReplayEngine(replay_engine_path) as replay_engine:
        for trial, trial_event_path in enumerate(event_paths):
            event_count = replay_engine.load_graph(f"trial{trial}", trial_event_path)
            if event_count != len(parsed_trials[trial].events):
                raise RuntimeError("native replay loaded a different event count")

        attributions = []
        jobs = []
        job_destinations: dict[str, tuple[str, str]] = {}
        for cost_name, costs in cost_inputs:
            for residual in (0.0, 0.5, 1.0):
                for split in ("front", "equal", "back"):
                    attribution_id = f"{cost_name}/residual{residual:g}/{split}"
                    attributions.append(
                        attribution_request(attribution_id, costs, residual, split)
                    )
                    for tie in ("fifo", "round_robin"):
                        key = f"{attribution_id}/{tie}"
                        simulations[key] = {}
                        for cores in (1, parallel_cores):
                            job_id = f"{key}/cores{cores}"
                            jobs.append(
                                replay_request(
                                    job_id,
                                    attribution_id,
                                    cores,
                                    tie,
                                    args.wake_latency,
                                    cross_thread_latency,
                                    replay_idle_scale,
                                    dependency_excess_scale,
                                )
                            )
                            job_destinations[job_id] = (key, str(cores))
        replay_engine.register_attributions("trial0", attributions)
        batch = replay_engine.replay_batch("trial0", jobs)
        for job_id, value in batch.items():
            key, cores = job_destinations[job_id]
            simulations[key][cores] = value

        if event_costs is not None:
            nominal_cost_name = "synthetic_event_ns"
            nominal_costs = event_costs
        elif args.workload == "synthetic":
            nominal_cost_name = "instructions"
            nominal_costs = instruction_costs
        else:
            nominal_cost_name = "modeled_events"
            nominal_costs = cost_inputs[-1][1]
        nominal_attribution = f"{nominal_cost_name}/residual0.5/equal"
        trace_trial_speedups = []
        trace_trial_parallel_ns = []
        for trial in range(args.trace_trials):
            graph_id = f"trial{trial}"
            attribution_id = nominal_attribution
            if trial != 0:
                replay_engine.register_attributions(
                    graph_id,
                    [attribution_request(attribution_id, nominal_costs, 0.5, "equal")],
                )
            trial_results = replay_engine.replay_batch(
                graph_id,
                [
                    replay_request(
                        "serial",
                        attribution_id,
                        1,
                        "fifo",
                        args.wake_latency,
                        cross_thread_latency,
                        replay_idle_scale,
                        dependency_excess_scale,
                    ),
                    replay_request(
                        "parallel",
                        attribution_id,
                        parallel_cores,
                        "fifo",
                        args.wake_latency,
                        cross_thread_latency,
                        replay_idle_scale,
                        dependency_excess_scale,
                    ),
                ],
            )
            serial_trial = trial_results["serial"]["modeled_makespan"]
            parallel_trial = trial_results["parallel"]["modeled_makespan"]
            trace_trial_speedups.append(serial_trial / parallel_trial)
            trace_trial_parallel_ns.append(parallel_trial)
        replay_engine_diagnostics = {
            "executable": str(replay_engine_path),
            "protocol_version": 1,
            "startup_seconds": replay_engine.startup_seconds,
            "phases": replay_engine.timings,
        }

    native: dict[str, object] = {}
    for cores in (1, parallel_cores):
        if args.workload == "renderer":
            samples, internal = _native_renderer_trials(
                native_benchmark_command,
                cpus,
                cores,
                args.native_trials,
                benchmark_env,
                native_repetitions,
            )
        else:
            samples, internal = _native_trials(
                benchmark_command, cpus, cores, args.native_trials
            )
        native[str(cores)] = {
            "samples_ns": samples,
            "minimum_ns": min(samples),
            "median_ns": statistics.median(samples),
            "maximum_ns": max(samples),
        }
        if internal:
            native[str(cores)]["measured_work_ns"] = {
                "samples": internal,
                "minimum": min(internal),
                "median": statistics.median(internal),
                "maximum": max(internal),
            }
            if args.workload == "renderer":
                native[str(cores)]["measured_render_nanoseconds_per_call"] = native[
                    str(cores)
                ]["measured_work_ns"]

    one_worker_renderer = None
    if args.workload == "renderer" and args.workers != 1:
        one_worker_env = {**benchmark_env, "VC_RENDER_SAMPLER_THREADS": "1"}
        external, internal = _native_renderer_trials(
            native_benchmark_command,
            cpus,
            len(cpus),
            args.native_trials,
            one_worker_env,
            native_repetitions,
        )
        one_worker_renderer = {
            "whole_process_samples_ns": external,
            "whole_process_median_ns": statistics.median(external),
            "measured_render_nanoseconds_per_call": {
                "samples": internal,
                "median": statistics.median(internal),
            },
        }

    nominal_key = f"{nominal_cost_name}/residual0.5/equal/fifo"
    nominal = simulations[nominal_key]
    if args.workload == "renderer":
        native_modeled_ns = {
            cores: native[str(cores)]["measured_work_ns"]["median"] * args.repetitions
            for cores in (1, parallel_cores)
        }
        native_scope = "measured render calls"
    elif args.workload == "synthetic":
        native_modeled_ns = {
            cores: native[str(cores)]["measured_work_ns"]["median"]
            for cores in (1, parallel_cores)
        }
        native_scope = "benchmark internal measured work loop"
    else:
        native_modeled_ns = {
            cores: native[str(cores)]["median_ns"] for cores in (1, parallel_cores)
        }
        native_scope = "whole process"
    predicted_speedup = (
        nominal["1"]["modeled_makespan"]
        / nominal[str(parallel_cores)]["modeled_makespan"]
    )
    observed_speedup = native_modeled_ns[1] / native_modeled_ns[parallel_cores]
    local_ns_per_modeled_work = native_modeled_ns[1] / nominal["1"]["modeled_makespan"]
    locally_estimated_parallel_ns = (
        nominal[str(parallel_cores)]["modeled_makespan"] * local_ns_per_modeled_work
    )
    predicted_speedups = [
        value["1"]["modeled_makespan"] / value[str(parallel_cores)]["modeled_makespan"]
        for value in simulations.values()
    ]
    complete = all(
        trial.unmatched_waits == 0 and trial.unresolved_happens_before == 0
        for trial in parsed_trials
    )
    unmatched_total = sum(trial.unmatched_waits for trial in parsed_trials)
    unresolved_hb_total = sum(
        trial.unresolved_happens_before for trial in parsed_trials
    )
    trace_median_speedup = statistics.median(trace_trial_speedups)
    trace_median_error = trace_median_speedup / observed_speedup - 1.0
    trace_median_parallel_ns = statistics.median(trace_trial_parallel_ns)
    trace_median_runtime_error = (
        trace_median_parallel_ns / native_modeled_ns[parallel_cores] - 1.0
    )
    if unresolved_hb_total:
        status = "trace_incomplete"
        reason = (
            f"{unresolved_hb_total} DRD happens-before references lack a "
            "traced predecessor"
        )
    elif not complete:
        status = "trace_incomplete"
        reason = f"{unmatched_total} blocking futex waits lack a traced wake"
    elif event_model is not None and abs(trace_median_runtime_error) > 0.20:
        status = "validation_failed"
        reason = (
            "median replay runtime differs from native by "
            f"{trace_median_runtime_error:+.1%}, beyond the 20% feasibility bound"
        )
    elif event_model is None and abs(trace_median_error) > 0.20:
        status = "validation_failed"
        reason = (
            "median replay speedup differs from native by "
            f"{trace_median_error:+.1%}, beyond the 20% feasibility bound"
        )
    else:
        status = "experimental"
        reason = (
            "renderer validation remains experimental until every frozen case passes"
            if args.workload == "renderer"
            else "validation matrix and dispatch-specific calibration are not complete"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "timing_claims_enabled": False,
        "reason": reason,
        "case": (
            {
                "workload": "renderer",
                "scenario": args.scenario,
                "workers": args.workers,
                "parallel_cores": parallel_cores,
                "repetitions": args.repetitions,
                "native_repetitions": native_repetitions,
                "whole_process": True,
            }
            if args.workload == "renderer"
            else {
                "workload": args.workload,
                "mode": args.mode,
                "workers": args.workers,
                "parallel_cores": parallel_cores,
                "tasks": args.tasks,
                "work_iterations": args.work_iterations,
                "task_iteration_skew": args.task_iteration_skew,
                "work_kind": args.work_kind,
                "working_set_bytes": args.working_set_bytes,
                "rounds": args.rounds,
                "warmup_rounds": args.warmup_rounds,
                "whole_process": True,
            }
        ),
        "trace": {
            "dependency_source": (
                "drd_vector_clocks"
                if args.workload in {"synthetic", "renderer"}
                else "core_futex_syscalls"
            ),
            "scheduler_quantum_basic_blocks": args.quantum,
            "events": len(parsed.events),
            "full_quanta_by_thread": parsed.full_quanta,
            "blocking_futex_waits": parsed.blocking_waits,
            "matched_futex_waits": parsed.matched_waits,
            "unmatched_futex_waits": parsed.unmatched_waits,
            "nonfutex_waitsys": parsed.nonfutex_waitsys,
            "happens_before_edges": parsed.happens_before_edges,
            "unresolved_happens_before": parsed.unresolved_happens_before,
            "raw_path": str(trace_path),
            "raw_paths": [str(path) for path in trace_paths],
            "event_path": str(event_path),
            "event_paths": [str(path) for path in event_paths],
        },
        "profiles": {str(thread): events for thread, events in profiles.items()},
        "event_costs_ns_by_thread": (
            {str(thread): cost for thread, cost in event_costs.items()}
            if event_costs is not None
            else None
        ),
        "simulations": simulations,
        "native_replay": replay_engine_diagnostics,
        "validation": {
            "native_scope": native_scope,
            "nominal_simulation": nominal_key,
            "predicted_speedup": predicted_speedup,
            "observed_speedup": observed_speedup,
            "relative_speedup_error": predicted_speedup / observed_speedup - 1.0,
            "case_local_ns_per_modeled_work": local_ns_per_modeled_work,
            "locally_estimated_parallel_ns": locally_estimated_parallel_ns,
            "relative_parallel_runtime_error": (
                locally_estimated_parallel_ns / native_modeled_ns[parallel_cores] - 1.0
            ),
            "sensitivity_speedup_min": min(predicted_speedups),
            "sensitivity_speedup_max": max(predicted_speedups),
            "trace_trial_predicted_speedups": trace_trial_speedups,
            "trace_trial_speedup_min": min(trace_trial_speedups),
            "trace_trial_speedup_median": trace_median_speedup,
            "trace_trial_speedup_max": max(trace_trial_speedups),
            "trace_trial_median_relative_error": trace_median_error,
            "trace_trial_predicted_parallel_ns": trace_trial_parallel_ns,
            "trace_trial_parallel_ns_median": trace_median_parallel_ns,
            "trace_trial_median_relative_runtime_error": trace_median_runtime_error,
        },
        "native_whole_process": native,
        "native_one_worker_renderer": one_worker_renderer,
        "worker_wake_latency_modeled_work": args.wake_latency,
        "cross_thread_latency_modeled_work": cross_thread_latency,
        "replay_idle_scale": replay_idle_scale,
        "dependency_excess_scale": dependency_excess_scale,
        "fixed_process_ns": fixed_process_ns,
        "base_process_ns": base_process_ns,
        "per_worker_startup_ns": per_worker_startup_ns,
        "event_cost_model": (
            str(args.event_cost_model) if args.event_cost_model else None
        ),
        "commands": {
            "trace": trace_commands,
            "callgrind": callgrind_command,
            "benchmark": benchmark_command,
            "native_benchmark": native_benchmark_command,
        },
        "benchmark_outputs": {
            "trace": trace_outputs,
            "callgrind": callgrind_run.stdout.strip(),
        },
        "limitations": [
            "Valgrind timestamps are retained for diagnostics but never used as work cost.",
            "Callgrind and scheduler traces are separate deterministic executions.",
            "Partial scheduler quanta are evaluated at zero, half, and one full quantum.",
            "Native comparisons include process startup, warmup, measurement, and teardown.",
            "Simulated idle is a model output, not an observed native scheduler state.",
            "Renderer replay acceptance compares whole-process trace and native boundaries.",
            "Internal renderer timings are diagnostic because the passive trace has no matching marker.",
            "Renderer dependencies come from DRD vector clocks, including userspace synchronization that does not block in a syscall.",
            "Renderer workers run with one additional caller CPU; seven workers therefore occupy all eight physical CPUs in CCD0.",
        ],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "reason": result["reason"],
                "result_path": str(result_path),
                "trace": result["trace"],
                "validation": result["validation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"run_thread_sync_replay.py: {error}", file=sys.stderr)
        raise SystemExit(1)
