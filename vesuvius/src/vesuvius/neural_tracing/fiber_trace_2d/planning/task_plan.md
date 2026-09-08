# Plan: overlap Fiber3D batch preparation with training

## Design

1. Add a bounded CUDA-preparation wrapper around the existing deterministic
   `FiberTrace3DTrainingBatchPipeline`; do not create another sample loader.
2. A single background coordinator consumes and collates ordered CPU batches,
   copies them into allocator-owned pinned host batches, releases their shared-memory
   sample leases, submits nonblocking H2D plus target materialization on a
   dedicated CUDA stream, and records a completion event.
3. Keep exactly two future prepared batches. Startup prepares N and N+1; once
   batch N is taken, admit N+2, giving it approximately one model step to
   finish. Current plus queued batches bound live pinned/device storage to three
   batches per rank. Do not reuse device buffers explicitly; allocator-owned
   tensors plus `record_stream` provide correct cross-stream lifetime.
4. The training stream waits on the prepared batch event without a device-wide
   synchronization. Record the training stream on returned device tensors so a
   staging slot cannot be reused while computation still consumes it.
5. Propagate worker/preparation exceptions at the next ordered consumption and
   close/join all preparation resources on normal and exceptional teardown.
6. Make the path an explicit training option layered on
   `sample_pipeline_workers`; retain the synchronous and CPU paths unchanged.
   It is CUDA-only, defaults off, is incompatible with replay modes, and uses a
   concrete depth setting defaulting to two queued future batches.
7. Quiesce the preparation stream before evaluation/checkpoint phases so
   training target kernels cannot overlap evaluation. A rank-local preparation
   failure is raised immediately; torchrun remains responsible for terminating
   sibling ranks that may already be inside DDP.

## Profiling and validation

- Add focused tests for ordering across more than one depth wrap, bounded depth,
  exception propagation, early cleanup, and config validation. Exercise actual
  CUDA event/stream behavior in the two-GPU benchmark.
- Run the focused Fiber3D/profile/ordered-pipeline test suites.
- Run at least two uninstrumented 2-GPU GroupNorm/output-stride-4 trials with
  the same 30-step warmup and 90-step measurement window.
- Report mean throughput and iteration mean/median/p95, H2D/preparation wait,
  peak allocated/reserved device memory and calculated live pinned-host bytes,
  and compare with 83.812 samples/s real-pipeline and 95.708 samples/s
  GPU-replay references. Separate enqueue time from exposed event wait; do not
  enable synchronizing detailed profiling in throughput trials.
- Confirm canonical stream indices, source hashes, and losses against the
  synchronous shared pipeline for the same seed across several queue wraps.

## Spec update

Document the optional CUDA-only bounded two-batch-ahead pinned/CUDA preparation
stage, allocator/event stream lifetime, exact live-batch bound, lease lifetime,
evaluation quiescing, failure behavior, and synchronous compatibility path in
`planning/specs.md`.

## Docs update

Append commands, configuration, benchmark artifacts, and before/after results
to `docs/training_profile_2gpu.md`.

## Changelog and task log

Add a concise changelog entry and record design decisions, failed experiments,
commands, artifacts, and measurements in `planning/task_log.md`.
