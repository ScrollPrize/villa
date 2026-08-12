#!/usr/bin/env python3
"""Compare Python reference replay with the persistent native engine."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from native_thread_sync_replay import (
    NativeReplayEngine,
    attribution_request,
    replay_request,
    resolve_replay_engine,
)
from run_thread_sync_replay import (
    assign_costs,
    read_event_stream,
    simulate_adjusted,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": percentile(values, 0.95),
        "maximum_seconds": max(values),
    }


def equivalent(actual: dict[str, float], expected: dict[str, float]) -> bool:
    return actual.keys() == expected.keys() and all(
        math.isclose(actual[name], expected[name], rel_tol=1e-12, abs_tol=1e-6)
        for name in expected
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--replay-engine", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cores", type=int)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise RuntimeError("repetitions must be positive")

    source = json.loads(args.result.read_text())
    costs = {
        int(thread): float(value)
        for thread, value in source["event_costs_ns_by_thread"].items()
    }
    cores = args.cores or int(source["case"]["parallel_cores"])
    wake_latency = float(source.get("worker_wake_latency_modeled_work", 0.0))
    cross_thread_latency = float(source.get("cross_thread_latency_modeled_work", 0.0))
    replay_idle_scale = float(source.get("replay_idle_scale") or 1.0)
    dependency_excess_scale = float(source.get("dependency_excess_scale") or 1.0)

    parse_times = []
    events = None
    for _ in range(args.repetitions):
        start = time.perf_counter()
        events = read_event_stream(args.events)
        parse_times.append(time.perf_counter() - start)
    assert events is not None

    attribution_times = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        assign_costs(events, costs, 0.5, "equal")
        attribution_times.append(time.perf_counter() - start)

    python_times = []
    expected = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        result = simulate_adjusted(
            events,
            cores,
            "fifo",
            wake_latency=wake_latency,
            cross_thread_latency=cross_thread_latency,
            replay_idle_scale=replay_idle_scale,
            dependency_excess_scale=dependency_excess_scale,
        )
        python_times.append(time.perf_counter() - start)
        expected.append(result)

    engine_path = resolve_replay_engine(args.replay_engine)
    with NativeReplayEngine(engine_path) as engine:
        info = engine.info()
        load_start = time.perf_counter()
        event_count = engine.load_graph("benchmark", args.events)
        load_wall = time.perf_counter() - load_start
        attribution_start = time.perf_counter()
        engine.register_attributions(
            "benchmark",
            [attribution_request("nominal", costs, 0.5, "equal")],
        )
        attribution_wall = time.perf_counter() - attribution_start
        native_times = []
        actual = []
        for repetition in range(args.repetitions):
            start = time.perf_counter()
            value = engine.replay_batch(
                "benchmark",
                [
                    replay_request(
                        f"repetition{repetition}",
                        "nominal",
                        cores,
                        "fifo",
                        wake_latency,
                        cross_thread_latency,
                        replay_idle_scale,
                        dependency_excess_scale,
                    )
                ],
            )[f"repetition{repetition}"]
            native_times.append(time.perf_counter() - start)
            actual.append(value)
        startup = engine.startup_seconds

    parity = [
        equivalent(native, python)
        for native, python in zip(actual, expected, strict=True)
    ]
    if not all(parity):
        raise RuntimeError("native replay differs from the Python reference")
    report = {
        "input": str(args.events.resolve()),
        "event_count": event_count,
        "cores": cores,
        "repetitions": args.repetitions,
        "request": {
            "residual_fraction": 0.5,
            "split_policy": "equal",
            "tie_policy": "fifo",
            "wake_latency": wake_latency,
            "cross_thread_latency": cross_thread_latency,
            "replay_idle_scale": replay_idle_scale,
            "dependency_excess_scale": dependency_excess_scale,
        },
        "native_build": info,
        "python": {
            "parse": summary(parse_times),
            "attribution": summary(attribution_times),
            "replay": summary(python_times),
        },
        "native": {
            "process_startup_seconds": startup,
            "load_wall_seconds": load_wall,
            "attribution_wall_seconds": attribution_wall,
            "warm_replay": summary(native_times),
        },
        "warm_replay_speedup": (
            statistics.median(python_times) / statistics.median(native_times)
        ),
        "all_repetitions_equivalent": all(parity),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
