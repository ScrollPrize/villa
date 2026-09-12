# embedding task history before merge (2026-09-12)

## task.md

# Scroll-balanced dense collection supervision

Enable dense polyline supervision per JSON dataset for the five-scroll collection, train and test. Supervise only the selected fiber. Choose scroll uniformly, then continuous distance uniformly across its fiber collection. Fixed repeatable test locations. Preserve fast shared queue, model, losses, LR and manager.

## task_plan.md

# Plan

1. Add dataset dense_supervision boolean (JSON default false, NML stays dense). Add loader sampling_strategy=scroll_balanced_arclength (default control_points) and dataset scroll_id. Require explicit IDs, dense labels and positive finite fiber lengths for balanced mode; combine entries sharing an ID.
2. Precompute segment and per-scroll fiber length CDFs. Seed draws from global sample index independent of worker/rank. Use a typed continuous location for crop, tangent, targets and prefetch, not a fabricated CP. Preserve flat CP traversal for trace2cp and legacy defaults.
3. Opt-in dense JSON supervises the selected complete polyline, including re-entry, via existing segment clipping/target materialization. NML behavior stays unchanged.
4. Enable all train/test collection entries, grouping both Paris4 folders, and correct PHerc1229 folder identity to PHerc1299. Keep fixed test seed/index range and existing sample budget (not an exhaustive traversal under this strategy).
5. CPU tests: distribution, continuous interpolation, repeated vertices, invalid data, reordered deterministic requests, dense selected-only supervision at stride1/4, re-entry, legacy behavior, test config inheritance and prefetch alignment. Use approved configurable test launcher.
6. Focused GPU integration: acquire exclusive non-pausable/non-adaptable manager allocation through the same launcher. Compare CPU/CUDA dense targets and run the collection U-Net at stride 4 on a continuous sampled patch, BF16 and FP32, forward/backward/update and evaluation with finite gradients/loss. Single GPU, batch 1, capped at 4 GiB allocator memory. No checkpoint or production-run edits; manager handles pause/resume.

## Spec update

Update planning/specs.md for dense JSON opt-in, continuous scroll-balanced strategy and deterministic evaluation; preserve compatibility.

## Docs updates

Document config and changed scientific meaning of collection metrics in docs. Update status/task_log/changelog. No checkpoint edits.

## status.md

# Status

- [x] Inspect implementation and record plan.
- [x] Independent plan and implementation review.
- [x] Implement sampler, dense switch and collection config.
- [x] CPU regression tests and docs.
- [x] Manager-exclusive CUDA dense-target integration, FP32/BF16 forward/backward/update/evaluation.

## task_log.md

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
