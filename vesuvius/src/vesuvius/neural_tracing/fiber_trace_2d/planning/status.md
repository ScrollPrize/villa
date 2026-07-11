# Load-Only Parallelism Diagnostics Status

- [x] Capture current user task in `planning/task.md`.
- [x] Create focused diagnostic task plan.
- [x] Add real process CPU timing to benchmark profile rows and summary.
- [x] Compile-check changed Python.
- [x] Run load-only profile benchmark.
- [x] Update docs/specs and task log with results.

Result: load-only profile now reports both synthetic loader worker factor and
real process CPU factor. On the 100-batch load-only profile run,
`loader_thread_factor=29.951` but `process_cpu_factor=3.817`, matching the
observed low system CPU utilization much better than the old table did.
