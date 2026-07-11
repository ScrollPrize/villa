# Merge `fiber2d-exp` Conflict Resolution Plan

## Conflict Inventory

- Code and tests auto-merged:
  - `augmentation.py`
  - `direction.py`
  - `loader.py`
  - `runner.py`
  - `docs/code_structure.md`
  - `planning/specs.md`
  - `test_fiber_trace_2d_loader.py`
- Conflicted files are task-state docs:
  - `planning/changelog.md`
  - `planning/status.md`
  - `planning/task.md`
  - `planning/task_log.md`
  - `planning/task_plan.md`

## Relevant Specs To Preserve

- Keep the current BatchNorm2d default for the V0 direction model.
- Keep full transformed centerline availability for training visualizations.
- Keep deterministic sample ordering, skip semantics, prefetch semantics, and
  current training/pipeline/cache config keys.
- Keep `volume_cache_memory_mib`, `volume_io_threads`, and
  `training.pipeline_isolated_loaders` because they are still documented in
  `specs.md` and implemented by the current branch.
- Keep `--load-only` benchmark behavior using the configured batch pipeline
  when requested by current specs.
- Accept `fiber2d-exp` Trace2CP/TTA requirements:
  - analytic Lasagna two-channel direction decoding;
  - no image-space geometric augmentation helpers;
  - paired concrete forward/backward geometric maps;
  - direct reference-to-TTA map lookup instead of dense nearest scans;
  - coordinate-sampled TTA patches;
  - bidirectional Trace2CP scoring and optional `--vis-tta` debug outputs.

## Resolution Plan

- Replace stale per-task planning conflicts with this merge task state.
- Resolve `planning/changelog.md` by keeping both durable 2026-07-11 entries:
  BatchNorm and the `fiber2d-exp` Trace2CP/TTA/decoder entries.
- Review auto-merged code for spec regressions before staging:
  - `train.py` still calls `load_batch(... include_line_xy=True, include_coords=False)`.
  - `model.py` still uses `nn.BatchNorm2d` and has no GroupNorm fallback.
  - `loader.py` still parses and forwards `volume_cache_memory_mib` and
    `volume_io_threads`.
  - `train.py` still supports `pipeline_isolated_loaders`.
  - `direction.py` uses analytic decoding and no `bins` argument.
  - `runner.py` no longer contains image-space geometric TTA warp helpers.
  - TTA patch builders return both `source_xy_grid` and
    `reference_to_tta_xy_grid`.
- If any auto-merged code contradicts `specs.md`, fix the code rather than
  weakening the spec.
- Stage resolved files only after conflict markers are gone and the semantic
  checks pass.

## Spec Update

- No new spec behavior is required beyond the auto-merged union currently in
  `planning/specs.md`.
- During resolution, keep the union of both branches' specs rather than taking
  either side wholesale.
- If implementation review finds a mismatch, update `planning/specs.md` only to
  clarify the intended behavior, not to remove current requirements.

## Docs Updates

- Keep `docs/code_structure.md` aligned with the merged implementation:
  - BatchNorm model documentation;
  - analytic direction decoder;
  - coordinate-only TTA;
  - bidirectional Trace2CP;
  - retained loader pipeline/cache knobs.
- Do not preserve old task logs in `planning/task_log.md`; it should describe
  this merge task only.

## Tests

- Compile-check touched Python:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/augmentation.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/direction.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run focused tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog Update

- Keep both existing 2026-07-11 changelog entries from the merged histories.
- Add no extra changelog line for conflict planning alone.
