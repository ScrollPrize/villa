# Plan: parallel native accumulation

## Mechanical refactor

1. Separate current `_accumulate_group` into coordinator-side chunk planning
   and an execution primitive. Preserve exact clipping, resume masks, shared
   weight semantics, product selection, activity accounting, and canonical
   ordering. Validate this synchronous refactor before enabling processes.
2. Describe an accumulation task entirely with bounded descriptors: tile/result
   slot, scale, global chunk origin, logical ring destination slices, result
   source slices, product names, weight/product mmap descriptors, and task IDs.

## Native kernel

3. Add a pybind11 `accumulator_add` extension and package it with Lasagna. It
   accepts strided 3D float16/float32 destination views and float32 source
   views, releases the GIL, and performs in-place add with the same per-add
   rounding implied by destination dtype.
4. On GCC/Clang x86, compile an isolated target-attributed AVX-512F+F16C row
   kernel that converts half to float32, adds float32, and rounds-to-nearest-even
   back to half; F16C is not treated as half arithmetic. Runtime-dispatch only
   when matching CPU support is present. Keep the module
   baseline free of global `-mavx512*` flags. Provide portable IEEE-half scalar
   conversion/add and float32 scalar/vector-friendly fallback for macOS,
   arm64, older x86, unsupported compilers, and missing extension. Report the
   selected backend once; never fail inference merely because SIMD is absent.

## Process pipeline

5. Add persistent spawn-context accumulator workers after GPU workers and
   before TensorStore context creation. Workers attach existing result shared
   slots, lazily reopen accumulator mmaps, set native libraries to one thread,
   and invoke the native primitive. Queues carry descriptors only.
6. Deterministically map `(scale, chunk_z, chunk_y, chunk_x)` to one worker with
   a stable integer formula (never Python hash) and one bounded FIFO per worker.
   The coordinator dispatches tiles canonically, calls ring `ensure` before
   dispatch, and only that owner updates a chunk, preserving per-chunk order
   without locks or races. Weight and all missing products for one chunk form
   one task.
7. Split dispatch from canonical completion. Reference-count every GPU result
   slot by its accumulation tasks; release it only after all acknowledgements.
   Consume completed events canonically before progress/row flush. Do not allow
   dispatch to reserve more rolling Z generations than the existing active-row
   plus frozen-flush capacity. Flush waits only for tasks that precede its
   frontier by construction.
   Queue submission is nonblocking and pumps acknowledgements under backpressure.
   Activity becomes committed only after successful acknowledgements; failures
   invalidate the event and cannot expose provisional data to flush.
8. Add `--accumulator-workers` to both frontends and shared API: default
   `min(cpu_count, 32)`, zero restores synchronous accumulation. Keep queues and
   outstanding tasks bounded by result slots and a small per-worker window.
   Add startup, throughput, queue-wait, work-sum, and backend diagnostics.
9. On errors/interrupts, stop GPU assignment, drain/cancel no new work,
   terminate accumulator workers if necessary, then stop flush workers and
   clean queues/shared memory/mmaps without semaphore leaks or hangs.
   Worker mmap/shared-memory caches close on every exit; normal sentinels,
   timeout escalation, termination, and queue feeder joins are explicit.

## Testing and measurement

10. Native tests: exhaustive/representative half boundaries, subnormals,
    signed cancellation, overflow/NaN/Inf behavior, arbitrary row strides,
    float32 exactness, backend reporting, portable fallback, and native-vs-
    NumPy per-add results. Require contiguous X rows for SIMD, support unaligned
    rows/YZ strides/wrap pieces/tails, validate writable/nonoverlapping buffers,
    and expose forceable scalar/SIMD selection. Compile without global ISA flags.
11. Pipeline tests: synchronous mechanical equivalence, deterministic chunk
    ownership/order, multi-process speed/concurrency, result-slot lifetime,
    rolling wrap, sparse/resume, multi-product shared weight, async flush
    overlap, worker exception/hard-exit/interrupt cleanup, and serial plus
    multi-device Fiber/Lasagna output comparisons for FP16 and FP32.
12. Benchmark native-kernel throughput separately from process-pipeline
    throughput using identical fixtures for NumPy FP16/FP32, native scalar/SIMD, and
    1/8/32 accumulation workers. Report command, CPU, backing, iterations,
    mean/median/min/max, effective task concurrency, and end-to-end controlled
    crop if the environment permits. Do not claim speedup without results.

## Specs/docs/changelog/workflow

13. Update specs for deterministic process-owned chunks, native runtime
    dispatch/portable fallback, slot/frontier lifecycle, boundedness, numerical
    behavior, and CLI defaults. Document architecture, tuning, installation,
    diagnostics, limitations, and benchmarking. Update status, task log, and
    changelog, explicitly recording any deferred test or platform limitation.
