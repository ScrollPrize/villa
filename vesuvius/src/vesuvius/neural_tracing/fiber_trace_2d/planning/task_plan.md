# Plan: profile the stride-4 full-pipeline gap

## Measurement

1. Preserve the committed stride-4 implementation as the profiling baseline.
2. Run a contemporaneous, repeated 2x2 GroupNorm matrix: stride 1/4 by prepared-GPU-batch replay/real pipeline. Keep seed, sample stream, eager mode, batch size, warmup, measured window, and cache state matched.
3. Measure uninstrumented throughput separately from diagnostic runs and quantify loader-profiling overhead.
4. Add an intermediate raw-batch replay that retains preparation, target construction, and H2D while removing ongoing producer variability.
5. Report command, dataset, build/runtime mode, hardware/software identity, steps, mean/median/p95 timings, throughput, target shape/bytes, and run-to-run variation.

## Attribution

6. Separate coordinator wait, sample loading, dependency discovery, coordinate generation, interpolation/value transforms, augmentation, batch assembly, target materialization, H2D, forward/loss, backward, optimizer, and DDP synchronization using existing per-sample/per-step instrumentation.
7. Measure queue-ready depth and empty duration, producer active span/effective concurrency, per-rank supply, and slowest-rank wait. Treat overlapped critical paths as a maximum, not a sum of worker CPU times.
8. Use CUDA events or synchronized window boundaries for asynchronous CUDA stages while preserving unbiased aggregate throughput from the slowest DDP rank.
9. Benchmark the actual trainer producer topology/config in isolation to distinguish inadequate aggregate capacity from queue/integration stalls.
10. If existing measurements merge materially different stages, add minimal timing instrumentation without changing numerical behavior.
11. Validate causality by comparing prepared replay, raw replay, and real production; then perturb only the implicated queue/worker capacity if needed.

## Outcome

12. Record the limiting stage and next concrete optimization. Implement it only if profiling identifies a clear, local change during this task.

## Spec update

No behavioral specification change is intended. If profiling adds durable instrumentation, document the additional timing fields in `planning/specs.md`.

## Documentation update

Add paired replay/full-pipeline measurements and bottleneck attribution to `docs/training_profile_2gpu.md`.

## Tests

- Unit-test new timing aggregation or configuration behavior.
- Run focused Fiber3D profiling tests.
- Run `git diff --check`.

## Changelog

Add an entry only if code or durable profiling instrumentation changes.
