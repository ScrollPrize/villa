# Task log

- 2026-09-11: Confirmed JSON CP-only versus NML dense targets. User approved dense selected-fiber supervision and scroll/arc-length balanced train and test. No GPU or active-run changes.
- Specs are in planning/specs.md; local_development.md is absent.
- Independent plan/code review by review_dense_sampling approved the design;
  requested exact whole-line re-entry coverage and vectorized segment rejection.
  Implemented both. No deviations from requested sampling/label semantics.
- New suite initially exposed two fixture errors (unsupported backend spelling
  and CP absent from synthetic line vertices); fixtures corrected. 13 CPU tests
  then passed in 16.87s, including parsing all 996 real collection files.
- Broader tests exposed two constructor-bypassing prefetch fixtures missing the
  new sampler field; initialized it explicitly (no production fallback).
- Approved command: /home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests
  with suites arc-sampling and sampling-regression, CUDA hidden, one native
  thread. Run 20260911_175944_023065: 15 passed (16.27s), then 312 passed,
  5 skipped (24.28s). Skips are unavailable CUDA paths; no GPU workload started.
  Evidence: /ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260911_175944_023065/.
- Selected-line labels match NML targets exactly on synthetic straight traces
  at strides 1 and 4; tests also cover re-entry and stream/data seed separation.
  Real GPU throughput/convergence is not measured in this correctness task.
  Active training and checkpoints are untouched; restart is needed to apply.
- Final exact-worktree rerun: 20260911_180055_592744, 15 passed in 16.58s plus
  312 passed/5 CUDA skips in 23.46s. git diff --check clean. The calibration
  fingerprint includes the new sampling/dataset fields, so an old memory
  report is not silently reused with new supervision.
- User requested focused CUDA integration instead of relying on CPU-only
  coverage. Added cuda-arc suite to the existing manager-gated fixed launcher;
  review_dense_sampling approved the bounded plan extension. Uses the real
  collection architecture on synthetic imagery/selected-fiber geometry, not
  a real-volume throughput or convergence benchmark. One GPU, batch 1, 128^3,
  output stride 4; FP32/BF16 sequentially under a 4 GiB allocator cap.
- First managed run 20260911_181845_985896: both CUDA cases passed in 5.89s.
  CPU/CUDA targets match; finite nonzero gradients and train/eval losses.
  Exclusive task 8b9b3b43-2107-4173-9898-89a0524a5ee5 released after test exit.
- Final run 20260911_181929_975715 additionally asserts an actual parameter
  update and exactly repeatable no-grad evaluation: 2 passed in 5.79s.
  Manager task 5d4b0403-7fad-49a1-a2b2-8c8a5ba03e11 released; diff check clean.
