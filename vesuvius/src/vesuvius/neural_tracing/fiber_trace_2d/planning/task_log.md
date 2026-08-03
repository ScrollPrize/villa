# Task log: process-parallel rolling-mmap flush

## Findings

- Representative eight-GPU crop: synchronous flush completed neural inference
  in 178.8 s; one background Python thread took 305.4 s, a 126.6 s / 70.8%
  regression. The thread did not parallelize chunks and could contend for the
  GIL, memory bandwidth, and process-global native thread settings.
- `_CircularZBand` already exposes a filesystem-backed mmap per accumulator
  channel. Spawned workers can reopen these by path and reconstruct wrapped
  logical reads without copying a band or sharing Python objects.
- Fiber and Lasagna model adapters store picklable configuration/product data;
  their finalizers are CPU/NumPy operations. `OmeZarrOutputAdapter` stores
  product/path metadata and atomic writes target distinct chunk files.
- A process worker needs only one output chunk's denominator/raw/finalized
  arrays. RAM therefore scales with flush worker count, not band or volume Z.
- GPU worker processes must start before the flush pool, and the flush pool
  before reader threads, to keep portable spawn lifecycle deterministic.

## Plan review

- Use explicit spawned processes with bounded task/result queues and a
  coordinator-pumped `2 * workers` IPC window; executor/pool feeder threads and
  eager whole-batch submission are prohibited.
- Zero workers must restore the old immediate-release planner and lifecycle,
  providing a genuine synchronous baseline. The normal automatic default is
  the available CPU count capped at 64 processes.
- Mmap descriptors are absolute path-only metadata; coordinator validates
  frozen generations, workers open read-only mappings, and no forced mmap flush
  occurs.
- Production adapters are spawn-pickle validated before traversal. Flush
  processes receive no model weights/tensors and never initialize CUDA.
- Batch/task IDs, exit-code polling, bounded queue pumping, and ordered shutdown
  cover Python errors, hard exits, result backpressure, and interrupts without
  releasing frozen slots.
- Each task exclusively owns all dirty products for one scale/chunk origin.
  Concurrent coarse-pyramid invalidation must be race tolerant.

## Implementation and validation

- Replaced the flush `ThreadPoolExecutor` with explicit persistent spawn
  processes. Bounded queues carry immutable task/path descriptors; workers
  cache read-only memmaps and never receive ndarray payloads.
- Added whole-batch completion gating, generation validation, task/batch result
  validation, hard-exit polling, process-first shutdown, and mmap cleanup.
- Added `--flush-workers` to both front ends (available CPU count capped at 64
  by default); zero preserves the synchronous immediate-release planner.
- Focused `unittest` coverage passed for process overlap, two-worker distinct
  execution, task failure, hard exit, mmap cleanup, planner sizing, serial vs
  multi-GPU exactness, and Lasagna CLI forwarding. The two-worker controlled
  test processed four 0.1 s writes across two distinct PIDs with 0.40 s
  aggregate worker work. Spawn startup dominated its 1.6-1.7 s flush wall
  time, as expected for this intentionally tiny test.
- Python compilation passed for the shared runner, both front ends, and touched
  tests. Fiber `--help` exposes the new option. The environment lacks pytest,
  so the pytest-based Fiber forwarding test was compiled but not executed.
- A pre-existing Lasagna legacy CLI assertion expects `ome_chunk=64` despite
  explicitly passing `--ome-chunk 32`; that unrelated assertion still fails.
  The full test-module run also stalled in an existing Zarr-heavy test, so
  validation used named focused cases.
- Representative eight-GPU post-change timing remains to be collected on the
  user's full-volume workload. The known baselines are 178.8 s synchronous and
  305.4 s for the removed Python-thread version.
- Corrected spawn-time native thread limiting by reusing the pyramid worker's
  `_single_threaded_native_runtime()` around both GPU-worker and flush-worker
  `Process.start()` calls. Limiting only inside a spawned worker is too late:
  spawn re-imports the main module, including NumPy/OpenBLAS, before invoking
  the worker target. Parent environment and runtime limits are restored after
  startup.
