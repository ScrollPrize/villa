# Plan: Deterministic Distributed Dense Evaluation

## Implementation

1. Extend the existing `_FiberTrace3DBatchDataset`/`_make_batch_dataloader`
   indexing with an explicit sample offset so test batches can retain their
   configured global sample indices while ranks independently select global
   batch indices `rank, rank + world_size, ...`.
2. Construct one persistent prefetched test DataLoader per rank using the same
   worker count, prefetch factor, worker device, and multiprocessing context as
   training. Reuse it across test intervals.
3. Run dense evaluation on every DDP rank. Each global test batch is evaluated
   exactly once, including one correctly sliced final partial batch. Gather
   tiny `(global_batch_index, metric_row)` records to rank 0, restore global
   order, and apply the previous Python-float unweighted per-batch mean exactly.
4. Keep stdout, TensorBoard, sample visualization, Trace2CP, best-checkpoint
   selection, and snapshot writes rank-0-only. Non-main ranks wait at the
   existing post-test metric broadcasts after contributing dense metrics.
5. Measure the full configured-test wall time on rank 0 through dense testing,
   rank-0 visualization, and Trace2CP (excluding the final TensorBoard flush),
   print `test_timing step=... total_seconds=...`, and write
   `timing/test_total_seconds`. Retain detailed diagnostics.
6. Keep the test worker pool alive across intervals and clean it up before DDP
   teardown. Report that train and test each retain their configured worker
   pool. Rank-0-only post-dense phases remain required to finish within the
   process-group timeout; diagnostics continue to cover them.

## Testing

- Unit-test deterministic disjoint global-batch assignment, non-batch-aligned
  sample offsets, zero-work ranks, full coverage, and final partial sizing.
- Unit-test offset-aware DataLoader indexing and globally ordered gathered-row
  reconstruction against serial batch-mean semantics.
- Exercise single-process fallback so non-DDP training remains compatible.
- Run focused Fiber 3D tests, Python compilation, and diff checks. A real
  eight-GPU timing measurement is not feasible inside the agent test run; the
  new stdout/TensorBoard timing will measure the next user training test.

## Spec update

Specify deterministic DDP test sharding, exact batch-mean reduction,
persistent prefetch workers, rank-0-only side effects, and test timing output.

## Docs update

Document dense-test ownership, sharding/indexing, worker reuse, reduction, and
the stdout/TensorBoard timing field.

## Changelog

Record distributed prefetched dense evaluation and explicit test timing.
