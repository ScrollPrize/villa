#!/usr/bin/env python3
"""Persistent client for the native passive synchronization replay engine."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

PROTOCOL_VERSION = 2
ENVIRONMENT_VARIABLE = "VC_THREAD_SYNC_REPLAY_BIN"


def resolve_replay_engine(
    explicit: Path | None = None,
    benchmark: Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    configured = os.environ.get(ENVIRONMENT_VARIABLE)
    if configured:
        candidates.append(Path(configured))
    if benchmark is not None:
        candidates.append(benchmark.resolve().parent / "bench_thread_sync_replay")
    discovered = shutil.which("bench_thread_sync_replay")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates) or "no candidates"
    raise RuntimeError(
        "native thread-sync replay executable was not found; pass "
        f"--replay-engine or set {ENVIRONMENT_VARIABLE} (checked {rendered})"
    )


class NativeReplayEngine:
    def __init__(self, executable: Path):
        self.executable = executable.resolve()
        self._request_id = 0
        self._closed = False
        self.timings: list[dict[str, object]] = []
        start = time.perf_counter()
        self._process = subprocess.Popen(
            [str(self.executable), "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.startup_seconds = time.perf_counter() - start

    def __enter__(self) -> NativeReplayEngine:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _request(self, command: str, **values: object) -> dict[str, object]:
        if self._closed or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("native replay engine is closed")
        self._request_id += 1
        request = {
            "schema_version": PROTOCOL_VERSION,
            "request_id": self._request_id,
            "command": command,
            **values,
        }
        start = time.perf_counter()
        try:
            self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        except BrokenPipeError as error:
            line = ""
            pipe_error = error
        else:
            pipe_error = None
        if not line:
            return_code = self._process.poll()
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().strip()
            detail = f"exit={return_code}, stderr={stderr!r}"
            raise RuntimeError(
                f"native replay engine terminated unexpectedly ({detail})"
            ) from pipe_error
        response = json.loads(line)
        if response.get("schema_version") != PROTOCOL_VERSION:
            raise RuntimeError("native replay response has an unsupported schema")
        if response.get("request_id") != self._request_id:
            raise RuntimeError("native replay response request ID is out of order")
        if response.get("status") != "ok":
            raise RuntimeError(
                str(response.get("error", "native replay request failed"))
            )
        self.timings.append(
            {
                "command": command,
                "wall_seconds": time.perf_counter() - start,
                **{
                    str(name): float(value)
                    for name, value in response.items()
                    if str(name).startswith("native_")
                },
            }
        )
        return response

    def load_graph(self, graph_id: str, event_path: Path) -> int:
        response = self._request(
            "load_graph", graph_id=graph_id, event_path=str(event_path.resolve())
        )
        return int(response["event_count"])

    def info(self) -> dict[str, str]:
        response = self._request("info")
        return {
            name: str(response[name])
            for name in ("compiler", "compiler_version", "build_type", "architecture")
        }

    def model_profile_costs(
        self,
        profiles: dict[int, dict[str, int]],
        event_cost_model: dict[str, object],
    ) -> tuple[dict[int, float], float]:
        response = self._request(
            "model_profile_costs",
            profiles={
                str(thread): {str(name): int(value) for name, value in events.items()}
                for thread, events in sorted(profiles.items())
            },
            event_cost_model=event_cost_model,
        )
        costs = {
            int(thread): float(cost)
            for thread, cost in response["thread_costs"].items()
        }
        return costs, float(response["total_cost"])

    def register_attributions(
        self,
        graph_id: str,
        attributions: list[dict[str, object]],
    ) -> None:
        self._request(
            "register_attributions",
            graph_id=graph_id,
            attributions=attributions,
        )

    def replay_batch(
        self,
        graph_id: str,
        jobs: list[dict[str, object]],
    ) -> dict[str, dict[str, float]]:
        response = self._request("replay_batch", graph_id=graph_id, jobs=jobs)
        results: dict[str, dict[str, float]] = {}
        for item in response["results"]:
            job_id = str(item["job_id"])
            if job_id in results:
                raise RuntimeError(f"native replay returned duplicate job ID {job_id}")
            results[job_id] = {
                str(name): float(value) for name, value in item["result"].items()
            }
        if list(results) != [str(job["job_id"]) for job in jobs]:
            raise RuntimeError("native replay returned jobs out of request order")
        return results

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                self._request("stop")
        finally:
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
            if self._process.stderr is not None:
                self._process.stderr.close()
            self._process.wait(timeout=5)


def attribution_request(
    attribution_id: str,
    thread_costs: dict[int, float],
    residual_fraction: float,
    split_policy: str,
) -> dict[str, object]:
    return {
        "attribution_id": attribution_id,
        "thread_costs": {
            str(thread): float(cost) for thread, cost in thread_costs.items()
        },
        "residual_fraction": float(residual_fraction),
        "split_policy": split_policy,
    }


def replay_request(
    job_id: str,
    attribution_id: str,
    cores: int,
    tie_policy: str,
    wake_latency: float = 0.0,
    cross_thread_latency: float = 0.0,
    replay_idle_scale: float = 1.0,
    dependency_excess_scale: float = 1.0,
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "attribution_id": attribution_id,
        "cores": cores,
        "tie_policy": tie_policy,
        "wake_latency": wake_latency,
        "cross_thread_latency": cross_thread_latency,
        "replay_idle_scale": replay_idle_scale,
        "dependency_excess_scale": dependency_excess_scale,
    }
