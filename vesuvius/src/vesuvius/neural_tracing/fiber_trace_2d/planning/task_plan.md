# Plan: distribution-invariant Fiber3D losses

1. Inventory every Fiber3D loss and diagnostic reduction, including direction,
   positive/negative presence, angle, branch fractions, and selected score.
2. Represent each reduction by its local differentiable numerator and local
   denominator. During DDP training, sum denominators and detached numerators
   across ranks, while scaling each rank's differentiable local numerator so
   DDP's gradient averaging yields the exact global-ratio gradient.
3. Preserve the existing per-patch/per-branch balancing semantics: synchronize
   sums of already formed nonempty group means and their group counts rather
   than flattening them into a voxel mean. Preserve single-process behavior and
   avoid collectives during ordinary unit-test/inference loss calls.
4. Make two-branch quota routing independent of rank partitioning by applying
   its deterministic score-based repair independently per stable stream/sample,
   rather than across the rank-local batch. This intentionally corrects the
   old multi-sample routing semantics while leaving one-sample behavior intact.
5. Thread one explicit loss-reduction context through the shared training loss
   entrypoint. Retain and sum local sufficient statistics across all logical
   microbatches, then perform exactly one packed collective and one backward at
   the optimizer-update boundary. Use the differentiable surrogate
   `global_value.detach() + world_size * (local_sum-local_sum.detach()) /
   global_count`; do not also apply the former sample-based microbatch scaling.
   This supports unequal logical work without mismatched collectives.
6. Keep distributed dense-test evaluation on its existing non-lockstep local
   loss plus row-gather path; training-only reduction must not add test-time
   collectives.
7. Add regression tests with deliberately uneven sparse-target placement and
   compare global objective values and parameter gradients against an
   equivalent single-batch reference. Cover zero-positive ranks, conditioned
   and legacy losses, quota-trigger routing, ordinary CPU/Gloo DDP, and unequal
   virtual-microbatch accumulation.
8. Benchmark the established two-GPU S1a training workload before and after the
   synchronized reductions using the same command/data/settings and repeated
   measured steps. Report global samples/s, step mean/p50/p95, and the dedicated
   `loss_stat_sync` stage.
9. Update task log, status, specification, code documentation, profiling notes,
   and changelog with semantics, commands, results, and limitations.

## Spec update

Require all optimization losses to use global sufficient-statistic reductions
and therefore be invariant to physical-rank and logical-microbatch partitioning.
Reconcile obsolete statements that Fiber3D has no internal microbatching.

## Documentation update

Document the reduction/gradient-scaling contract and add the measured two-GPU
collective overhead to the training profiling report.

## Changelog

Add a concise entry for globally normalized Fiber3D DDP losses.
