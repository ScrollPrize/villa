# Merge `fiber2d-tweaks` Conflict Resolution Plan

## Conflict Inventory

- Auto-merged code/docs:
  - `runner.py`
  - `docs/code_structure.md`
  - `planning/specs.md`
- Conflicted files:
  - `planning/changelog.md`
  - `planning/status.md`
  - `planning/task.md`
  - `planning/task_log.md`
  - `planning/task_plan.md`
  - `vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Relevant Specs To Preserve

- Preserve the current BatchNorm2d model default, full transformed training
  centerline visualization, deterministic sample ordering, skip semantics,
  prefetch semantics, and current training/pipeline/cache config keys.
- Preserve the `fiber2d-exp` merge requirements: analytic Lasagna two-channel
  decoding, concrete paired geometric maps, direct reference-to-TTA lookup,
  coordinate-sampled TTA patches, and bidirectional Trace2CP scoring.
- Preserve strict coordinate-only geometric augmentation for training,
  augment-vis, line tracing, Trace2CP, labels, TTA, and all shared loader paths.
- Accept the `fiber2d-tweaks` behavior as a narrow runner/debug exception:
  `--dir-vis` may apply pixel-perfect image-space identity/flip/90-degree
  rotation variants to an already sampled center patch for checkpoint
  robustness inspection, and `--dbg-dirs` may paste the unaugmented center into
  transformed diagnostic contexts. This exception must not be reused by
  training, labels, augment-vis, line tracing, Trace2CP, or TTA.

## Resolution Plan

- Replace conflicted per-task planning docs with this merge task state.
- Resolve `planning/changelog.md` by keeping durable entries from both sides:
  the current loader/training/pipeline/Trace2CP history and the incoming
  `--dir-vis` image-space diagnostic entries.
- Resolve the test import conflict by keeping current direct-map helpers and
  incoming dir-vis helpers, while omitting stale helpers that must not exist:
  `_nearest_tta_point_for_reference` and `_line_trace_tta_entries`.
- Narrow the no-image-space-geometric regression test so it still rejects
  training/TTA image-space warp helpers but excludes the explicit
  `_dir_vis_image_space_augmentations` diagnostic helper.
- Clarify `planning/specs.md` and `docs/code_structure.md` with the same
  `--dir-vis` diagnostic exception.
- Review auto-merged code against the merged specs before staging:
  - `runner.py` provides `--dir-vis` identity/flip/rot90/rot180/rot270 panels,
    natural-size strip output, and `--dbg-dirs` pasted-center debug row.
  - `runner.py` still has no `_nearest_tta_point_for_reference`,
    `_line_trace_tta_entries`, image-space line/Trace2CP TTA warp helper, or
    dense nearest-grid TTA lookup.
  - `direction.py` still uses analytic decode.
  - `model.py` still uses `BatchNorm2d`.
  - `train.py` still requests `include_line_xy=True` for training
    visualization.
- Stage resolved files only after conflict markers are gone and semantic checks
  pass.

## Spec Update

- Add a narrow `--dir-vis` diagnostic exception to the image-space geometric
  prohibition.
- Keep the coordinate-only rule unchanged for training, labels, augment-vis,
  line tracing, Trace2CP, TTA, and shared loader paths.
- Keep incoming `--dir-vis` output semantics: one labeled `dir_vis.jpg`, native
  model inference per variant, 4x nearest-neighbor display scaling, 8x8 display
  cells, 6-pixel anti-aliased direction segments, and optional `--dbg-dirs`
  second row.

## Docs Updates

- Update `docs/code_structure.md` to document the `--dir-vis` exception while
  keeping the core augmentation/loader paths coordinate-only.
- Keep `planning/task_log.md` limited to this merge task only.

## Tests

- Compile-check touched Python:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/augmentation.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/direction.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run focused tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog Update

- Keep existing durable changelog entries from both branches.
- Add no extra changelog line for the merge mechanics alone.
