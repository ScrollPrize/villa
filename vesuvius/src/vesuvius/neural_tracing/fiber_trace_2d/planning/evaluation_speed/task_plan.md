# Plan

Build a bounded evaluation-only benchmark calling the existing training/evaluation
code. Use the approved profile launcher with a new explicit evaluation mode and
exclusive managed ownership, no calibration required for this bounded benchmark.
Measure startup separately from per-rank loading, target preparation, forward/loss
and synchronization. Compare identical deterministic sample groups and BF16 model
initialization; no optimizer steps, normal checkpoints or TensorBoard needed.

First measure current evaluation with a short subset and a five-minute budget.
Use measured bottlenecks to choose changes, prioritizing reuse of the optimized
shared per-sample loading pipeline over another implementation. Test unmanaged
DDP evaluation and preserve managed boundary/restart semantics; report any path
not optimized or hardware run not completed explicitly.

## Spec update
No changes to test membership, reductions or model math. Benchmark-only timing
and stop controls must not affect ordinary runs. No new manager protocol.

## Docs updates
Record command, input, CPU/GPU counts, timings and limitations in a dedicated
evaluation benchmark document and task log. Keep other uncommitted work intact.

## Tests
CPU regression tests for evaluation-only stop/timing and sample order; bounded
managed GPU comparison on idle devices, with cleanup before reservation release.

## Changelog
Record measured results and actual optimization, not projected speedups.
