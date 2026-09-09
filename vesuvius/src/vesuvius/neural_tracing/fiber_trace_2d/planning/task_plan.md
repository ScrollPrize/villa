# Plan: remove unconditional Fiber3D scalar synchronization

1. Define one rank-consistent predicate for scalar metric consumption from the
   configured interval and initial verbose steps.
2. Run distributed metric reduction and GPU-to-host scalar conversion only when
   that predicate is true. All ranks must make the same collective decision.
3. Preserve default first-100-step and interval logging. Add a configurable
   initial verbose-step count so throughput benchmarks can set it to zero.
4. Allow profiling rows without `loss_total` on non-metric steps rather than
   introducing a local `.item()` synchronization.
5. Add regression tests for due-step selection and retained metric values, then
   rerun the approved two-GPU batch-28 benchmark twice if it improves.

## Spec update

Document rank-consistent on-demand scalar reduction and the initial log-step
configuration while preserving default output behavior.

## Documentation update

Record before/after batch-28 throughput, DDP critical-path iteration timing,
and artifacts in the training profiling document.

## Changelog

Add a concise entry if the production training loop changes.
