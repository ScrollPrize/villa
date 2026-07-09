# Task Log

## Prefetch Debug Profiling

- Added temporary prefetch profiling output in `loader.py`.
- Prefetch now prints a short column legend, then for each patch:
  - source-construction stage start/done lines for descriptor lookup,
    line-window construction, Lasagna-normal sampling, and source-coordinate
    generation, so stalls before the first patch row are visible;
  - a `prefetch start ...` line before the potentially blocking sampler call;
  - a completed timing row with source total, descriptor, line-window,
    Lasagna-normal, source-coordinate, envelope-coordinate, sampler/cache, and
    total patch milliseconds;
  - sampler stats for valid pixels, dependency count, downloaded chunks, bytes,
    and sampler prefetch mode.
- Existing shared coordinate/cache behavior is unchanged; the patch only adds
  timings and debug output around it.

Validation:

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - Result: passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result after inner source-stage debug print fix: `34 passed in 2.73s`.
- `PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace_2d.train --help`
  - Result: passed.

## Vectorized Strip And Line Coordinate Generation

- Added a torch-backed dense side-strip grid builder in `strip_geometry.py`. It keeps the existing Python/Numpy VC3D/Lasagna frame transport and roll smoothing, then vectorizes the per-pixel arc-coordinate, cubic Hermite, and normal interpolation work.
- Added CP-local source geometry reuse for augment visualization. The runner now builds descriptor/window/Lasagna normals/source strip coordinates once for the selected sample and reuses that source grid for all contact-sheet cells.
- Replaced affine line-coordinate generation with a direct vectorized inverse mapping. Smooth-offset cases use a chunked tensor nearest search over the geometric source-coordinate grid instead of a Python loop over every line sample.
- Kept VC3D blocking volume sampling and coordinate-space geometric augmentation unchanged.
- Added tests comparing the torch strip grid against the existing NumPy path and checking vectorized shifted line coordinates.

Validation:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `26 passed in 1.96s`.
- Augment-vis command:
  - `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.runner /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --export-dir /tmp/fiber_trace_aug_debug_blocking --sample-index 1 --augment-vis`
  - Result: command completed and wrote `augment_contact_sheet.jpg` plus `augment_summary.txt`.
  - Current timing sample: first `unaugmented` row total `2486.4 ms`, dominated by `volume_sample=2204.9 ms`; one-time `strip_coords=103.0 ms`; affine repeated cells typically `3.6..4.8 ms`; smooth/combined line-coordinate rows mostly `1.8..2.4 ms` after the vectorized fallback.

Environment note:

- The same command with `PYTHONNOUSERSITE=1` failed before reaching the loader because that environment resolved a zarr install where `zarr.storage.Store` is missing. The normal local command above matches the currently working checkout environment.

## Complete Current-State Documentation

- Replaced the active task and task plan with the docs TODO from `planning/todo.md`.
- Expanded `docs/code_structure.md` from a short module list into current-state documentation covering module responsibilities, config keys, dataset construction, deterministic sample selection, scale semantics, Lasagna normal handling, batch shapes, prefetch, augmentation contact sheets, runner commands, tests, and local caveats.
- Added a documentation-maintenance expectation to `planning/specs.md`: `docs/code_structure.md` describes current implementation details while `planning/specs.md` remains the normative behavior source.
- Marked the docs TODO as complete in `planning/todo.md`.

Validation:

- Docs-only pass; no runtime behavior changed.
- `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_plan.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/status.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/task_log.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/todo.md`
  - Result: passed.

## Planned V0 2D Fiber-Strip Direction Training

- Moved the remaining training TODO from `planning/todo.md` into `planning/task.md`.
- Replaced the docs task plan with a scoped V0 training plan in `planning/task_plan.md`.
- The plan preserves the current VC3D/Lasagna strip loader, coordinate-space geometric augmentation, GPU value augmentation, deterministic sample selection, and fake/local-array testing constraints.
- Added `planning/status.md` checklist entries for planning, implementation, spec/docs updates, and validation.
- Refined the training plan to require the Lasagna ambiguous two-cos-channel direction encoding, `0.5 + 0.5*cos(2*theta)` and `0.5 + 0.5*cos(2*theta + pi/4)`, instead of raw signed vector regression or an `abs(dot)` loss.

## V0 2D Fiber-Strip Direction Training Implementation

- Added `direction.py` with Lasagna two-cos-channel direction encoding, an approximate visualization decoder, augmented-line CP/tangent extraction, and CP-local eight-neighbor supervision gathering.
- Added `model.py` with a small V0 2D CNN that predicts exactly two encoded direction channels per pixel.
- Added `train.py` with JSON `training` config parsing, deterministic sample-index stepping, `[control_point, strip_z]` batch flattening, encoded MSE loss, TensorBoard scalar/text/image logging, and `snapshots/current.pt` / `snapshots/best.pt`.
- Updated `configs/loader_example.json` to enable the current augmentation settings and include the default four-control-point/16-strip-offset training config.
- Updated `planning/specs.md`, `docs/code_structure.md`, `planning/local_development.md`, `planning/changelog.md`, and `planning/status.md` for the new training path.
- Added focused tests for double-angle direction ambiguity, CP-local supervision, 64-patch batch assembly, and one-step local-array training smoke.

Validation:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result before final docs/status edits: `30 passed in 3.57s`.
  - Final result after explicit transformed-CP coordinate and docs/status edits: `30 passed in 2.75s`.

## Planned Full-Augmentation Training Prefetch

- Updated `planning/task.md` with the active prefetch-hardening task.
- Replaced `planning/task_plan.md` with a focused plan for training-oriented
  prefetch and full-augmentation request/load coordinate regression tests.
- Current code inspection indicates `chunk_requests_for_sample_index` already
  uses augmentation padding, final coordinate-space augmentation, deterministic
  per-offset augmentation parameters, and all strip-z offsets.
- The plan therefore focuses on adding a convenient `train.py --prefetch`
  command, documenting the command, and adding tests that prove prefetch
  coordinates match actual augmented loading coordinates.
- Refined the plan so `--prefetch-steps 0` means prefetch all configured
  training steps, and negative step counts are rejected.

## Full-Augmentation Training Prefetch Implementation

- Added `train.py --prefetch` mode. It maps training steps to the same
  deterministic control-point sample-index range used by training, calls
  `FiberStrip2DLoader.prefetch`, prints a compact summary, and exits before
  model, TensorBoard, run-directory, or snapshot setup.
- Added `--prefetch-steps`; omitted or `0` means all configured
  `training.max_steps`, and negative values are rejected.
- Added `--prefetch-start-step` for resuming prefetch ranges by 1-based
  training step.
- Added a regression test that records final augmented coordinates seen by
  `chunk_requests_for_coords` and `sample_coords`, proving prefetch and loading
  use the same per-strip-z coordinates.
- Added training prefetch tests for explicit step counts, all-steps mode, and
  negative-step validation.
- Updated specs and local docs with the train prefetch command and
  base-volume-only/final-coordinate semantics.

## Planned Augment-Vis Path Unification

- Replaced the active task with the broader mismatch-removal task: augment-vis
  source/patch handling is the intended behavior and should be the shared path
  for training, runner loading, and prefetch.
- Wrote a new task plan that removes the current differences:
  - training/prefetch using old NumPy strip-coordinate generation;
  - augment-vis-only source-geometry reuse;
  - prefetch using a cache-incompatible Python chunk-store wrapper;
  - opaque prefetch discovery with no progress.
- The plan keeps the only intended difference for training: multiple strip-z
  offsets, derived from one CP-local source geometry by offsetting along the
  strip normals/frame direction.
- Refined the prefetch requirement: prefetch must cover the configured maximum
  augmentation envelope for each CP/offset instead of depending on one sampled
  random augmentation draw.

## Augment-Vis Path Unification Implementation

- Added offset-axis data to `FiberStripGrid` so strip-z offsets can be derived
  from one CP-local torch-vectorized source grid.
- Refactored `FiberStrip2DLoader` around `build_strip_source` and
  `build_strip_patch_from_source`; training, center-strip loading, and
  augment-vis now delegate to this shared source/patch path.
- Replaced production `build_sample` use of the old NumPy strip-grid builder
  with shared torch source geometry and per-offset coordinate derivation.
- Changed prefetch to process the shared source-envelope coordinates per
  CP/strip-z offset and call `CoordinateSampler.prefetch_coords`.
- Added sampler-level prefetch. VC3D prefetch uses the same blocking
  `sample_coords` path as loading and discards values; the NumPy test sampler
  keeps a local chunk-request implementation.
- Kept `chunk_requests_for_sample_index` only as a compatibility/test helper
  derived from the same prefetch envelope.
- Updated specs, code docs, and local development notes for canonical
  augment-vis-style loading and augmentation-envelope prefetch.
- Added regression tests for sampler-level prefetch, envelope coverage of
  augmented load dependencies, and failure if training calls the old NumPy
  strip builder.

Validation:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `34 passed in 2.74s`.
- `PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace_2d.train --help`
  - Result: passed.
- `PYTHONPATH=vesuvius/src:. python -m vesuvius.neural_tracing.fiber_trace_2d.runner --help`
  - Result: passed.
