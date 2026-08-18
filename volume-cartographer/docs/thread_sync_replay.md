# Native Thread-Synchronization Replay

The regular rendering estimate is native C++ end to end after Valgrind writes
its raw files. `bench_thread_sync_replay evaluate-render` parses periodic
separate-thread Callgrind profiles and the scheduler stream from that same run,
locates the measured DRD window from the benchmark's existing clock calls,
reconstructs vector-clock dependencies, evaluates all logical-worker
assignments supported by both scheduler traces, attributes chronological costs,
and replays the graph conservatively. Python remains only in offline
calibration and compatibility tooling.

Python starts one persistent process and exchanges one JSON object per line.
Every request and response uses `schema_version: 2` and an ordered
`request_id`. Supported commands are:

- `info`: return compiler, build type, and architecture metadata.
- `model_profile_costs`: validate raw per-thread Callgrind events and evaluate
  the frozen event-cost model in deterministic numeric-thread order.
- `load_graph`: load and validate one JSONL event stream under a `graph_id`.
- `register_attributions`: assign per-thread costs to named attribution IDs.
- `replay_batch`: execute ordered jobs against cached graph/attribution pairs.
- `stop`: return a final response and exit successfully.

Each replay job specifies its job and attribution IDs, core count, tie policy,
wake and cross-thread latencies, replay-idle scale, and dependency-excess
scale. Responses preserve job order and contain the same fields as Python's
reference `simulate_adjusted` result. Invalid graphs, references, costs,
policies, and numeric ranges return a request-scoped error response. Protocol
I/O and internal process failures terminate the client operation.

Callers locate the executable in this order: explicit `--replay-engine`,
`VC_THREAD_SYNC_REPLAY_BIN`, beside the benchmark executable, then `PATH`.
There is no Python fallback.

The rendering CMake/Ninja graph invokes Valgrind and the native evaluator
directly. NumPy and the standard-library Python compatibility client are not
part of production gate execution.

Build and run the compatibility tests:

```bash
cmake --build build-release \
  --target bench_thread_sync_replay test_thread_sync_replay_native -j32
VC_THREAD_SYNC_REPLAY_BIN=build-release/bin/bench_thread_sync_replay \
  python3 -m unittest core/test/test_run_thread_sync_replay.py
```

Measure a saved trace against the Python oracle:

```bash
python3 scripts/benchmark_thread_sync_replay.py \
  --events /path/to/sync-events.jsonl \
  --result /path/to/sync-replay.json \
  --replay-engine build-release/bin/bench_thread_sync_replay \
  --repetitions 5
```
