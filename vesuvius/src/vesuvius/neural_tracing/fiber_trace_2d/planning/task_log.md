# Task Log: Distributed Prefetched Fiber 3D Dense Tests

## Findings and baseline

- Watchdog diagnostics proved step 95800 spent 662.742 seconds in 16 serial
  rank-0 dense-test `load_batch` calls and only 1.194 seconds in GPU forwards.
  NCCL timed out at 600 seconds while rank 0 continued making loader progress.
- The preceding step 95700 spent 532.544 seconds in dense-test loads and
  588.176 seconds in the complete configured-test routine, already close to the
  process-group timeout.
- Training uses persistent process-worker DataLoader prefetch, while dense
  testing previously bypassed it and synchronously loaded all 123 samples on
  rank 0.
- Independent review required global-batch sharding, literal non-aligned sample
  offsets, exact historical per-batch metric order/weighting, all-rank initial
  and interval collective participation, and explicit worker-pool ownership.

## Implementation

- Every rank now builds the held-out loader and retains a test DataLoader using
  the same worker/prefetch/device/context settings as training.
- Global test batch IDs are deterministically assigned as
  `rank, rank + WORLD_SIZE, ...`; the configured start is applied as a literal
  sample offset and only the global final batch is sliced.
- Ranks gather `(global_batch_index, float_metric_row)` records. Rank 0 restores
  global order and applies the prior unweighted Python per-batch mean, avoiding
  a floating-point reduction-order change to best-checkpoint selection.
- All ranks enter both step-0 and interval dense tests. TensorBoard, stdout,
  visualization, Trace2CP, and snapshots remain rank-0-only. Rank 0 reuses its
  already evaluated global batch zero for visualization.
- Rank 0 prints `test_timing step=... total_seconds=...` and logs
  `timing/test_total_seconds` before the configured-test TensorBoard flush.

## Validation

- Python compilation and `git diff --check` pass.
- A direct helper smoke test covered 123 samples over eight ranks, exact
  disjoint global-batch coverage, final partial sizing, ordered metric means,
  and the stdout/TensorBoard timing helper.
- Added pytest coverage for non-batch-aligned offsets, `N < batch_size`,
  non-divisible counts, zero-work ranks, mocked gathered-row ordering, and
  timing output/scalar logging.
- The existing `.venv_las` does not contain pytest, so the pytest cases could
  not be executed without installing a dependency. No install was performed.
- A real two-rank Gloo smoke could not initialize because the sandbox forbids
  local transport sockets (`Operation not permitted`). The helper-level mocked
  collective test remains in the suite.
- No eight-GPU training run was started. The next user run will provide the
  required after-change wall time in stdout and TensorBoard; no measured
  speedup is claimed yet.

## Known limitation

- Rank-0-only visualization and Trace2CP run after distributed dense evaluation
  while other ranks wait for the final result broadcast. Diagnostics still
  cover these phases, and they must remain below the 600-second process-group
  timeout. They were not the bottleneck in the captured failure.
