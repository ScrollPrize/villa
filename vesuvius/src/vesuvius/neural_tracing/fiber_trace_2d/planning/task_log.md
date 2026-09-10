# Manager merge and exclusive tracer — 2026-09-10

- User paused queue instrumentation; partial implementation and remaining work in docs/live_inference_concurrency_20260910.md.51 precommit tests pass.
- Checkpoint commits:f31412abc native tracing,2c88b8ffd shared demand/accumulator,ef16a59e5 Fiber runners/results/configs. Working tree clean before merge.
- Merging local fiber-training-manager e7bde77d5 into fiber-training-retrain; common ancestor6009bb7f4. No rebase needed.7 conflicts:5 planning files,train.py,volume-cartographer/README.md. C++ implementations merged automatically.
- train.py combines sample-axis progress with manager handoff/shared snapshot helpers. Preserve sample metadata in both durable snapshots and RAM handoff; retain resumable-test dedup guards. README retains HiGHS docs and test diagnostics; changelog preserves both histories.
- Manager protocol v1 and capability amendment read; no protocol changes authorized/needed. Fixed evaluation uses existing low-level Client with pausable=false/adaptable=false and explicit priority, exclusive=true. Cooperative registered training can yield; arbitrary GPU processes cannot be safely evicted.
- Independent merge/plan review approved; corrected stale draft/pending wording
  in merged specs to match locked protocol and scoped reference validation.
- Tests explicitly force villa2 import roots:25 training-state/gradient/sample
  progress tests pass (12CUDA cases not run),54 manager/scheduling tests pass.
  Initial unqualified invocation resolved one module from the environment's
  villa checkout; discarded that result and reran with explicit villa2 paths.
  Gloo local sockets required the approved pytest command outside sandbox.
