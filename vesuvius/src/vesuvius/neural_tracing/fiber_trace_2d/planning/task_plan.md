# 3D Augmentation Stream And 2D Spec Parity Plan

## Current Findings

- 3D currently uses one `sample_index` for both CP/data selection and
  `_sample_augment_params(...)`. In the simple unbounded training path this
  happens to vary by training step, but it does not implement the 2D invariant
  that bounded CP reuse and augmentation seeding are separate streams.
- 3D has no implemented `training.max_sample_index` equivalent. The shared spec
  requires this bounded-prefix reuse behavior and explicitly says augmentation
  seeds must remain unbounded when CP/data samples wrap.
- Dense 3D test loaders are currently built by copying the train config and
  replacing `datasets` with `test_datasets`, so top-level train augmentation
  settings are inherited by tests.
- 3D currently treats `training.max_steps <= 0` as one finite dataset pass.
  The shared 2D training contract says `max_steps: 0` means the deterministic
  sample stream repeats indefinitely.
- 3D dense-test evaluation currently expands `training.test_control_points <= 0`
  to the full held-out count, but still loads those samples in random order.
  The 2D sentinel behavior is full held-out evaluation in flat CP order from
  sample index zero.
- 3D prefetch currently defaults `--prefetch-steps` to `1`, so omitting the
  flag does not follow 2D's "use config max_steps" behavior. It also has no
  `max_sample_index` interaction yet.
- 2D progress wording currently says first 100 deterministic sample indices.
  Per the current task, the spec should be changed to first 100 training steps
  for both 2D and 3D.
- Current JSON/non-NML 3D samples have been partly moved toward carrying line
  segment metadata, but this task will make the intended rule explicit: loaders
  always provide full transformed fiber-line context, while target mode decides
  whether loss supervision is dense line or CP-only.
- The default-looking predicted direction in test visualization is still model
  inference. If raw direction logits are near-degenerate, the analytic Lasagna
  ambiguous-direction decoder can produce a stable arbitrary axis. That should
  be interpreted as an untrained/ambiguous output unless raw channel inspection
  proves a loader fallback is being used.

## Relevant 2D Specs Not Fully Carried To 3D

- Bounded deterministic-prefix reuse: `training.max_sample_index` should bound
  CP/data selection while keeping augmentation seeds on the unbounded training
  stream.
- Deterministic CP order semantics: changing training step count or batch size
  should only truncate/extend the deterministic stream prefix.
- Test loader separation: `test_datasets` should define fixed deterministic
  evaluation data, not inherit training augmentations unless explicitly enabled.
- Full-test sentinel: `training.test_control_points: 0` should use every
  held-out CP in flat order from zero.
- Unbounded training sentinel: `training.max_steps: 0` should mean indefinite
  deterministic stream repetition, not one finite pass.
- Prefetch CLI semantics: explicit positive `--prefetch-steps` overrides
  config; explicit `0` means all selected CPs; omitted uses `training.max_steps`;
  negative values are invalid; `training.max_sample_index` bounds the training
  prefix being prefetched.
- Full context with loss-mode separation: loader sample context and target
  materialization must be distinct so visualization/context segments cannot
  accidentally become dense supervision.
- Console progress semantics: print the first 100 training steps, then fall
  back to `training.scalar_log_interval`.
- Spec/docs synchronization: non-dimensional 2D requirements such as run/test
  determinism, config keys, and TensorBoard test behavior must be documented for
  3D when supported.

## Implementation

1. Add explicit 3D sample-index separation.
   - Introduce `data_sample_index`/`augmentation_sample_index` naming in the
     loader path.
   - Keep descriptor lookup keyed by the bounded data sample index.
   - Key `_rng(..., "augment3d")`, smooth displacement controls, value noise,
     anisotropic blur randomness, shift, rotation, scale, and flips by the
     unbounded augmentation sample index.
   - Preserve existing deterministic results when no limit is configured by
     passing the same integer as both indices.

2. Add 3D `training.max_sample_index`.
   - Parse and validate it in `fiber_trace_3d/train.py`.
   - In direct loader and PyTorch DataLoader paths, compute:
     - `raw_sample_index = global_step_sample_index + batch_offset`
     - `data_sample_index = raw_sample_index % max_sample_index` when the limit
       is positive, otherwise `raw_sample_index`
     - `augmentation_sample_index = raw_sample_index`
   - Use the same bounded data index for prefetch dependency generation so
     prefetch and training consume the same CP/data prefix.
   - Keep progress/reporting explicit about raw versus bounded sample index
     where needed.

3. Fix dense test augmentation behavior.
   - When constructing the `test_datasets` loader, set top-level
     `augment_enabled=false` by default.
   - Add `training.test_augment_enabled` as the only opt-in to preserve train
     augmentation settings for tests.
   - Keep test sample selection deterministic and fixed.
   - Implement `training.test_control_points: 0` as full held-out evaluation in
     flat CP order from sample index zero, ignoring `test_start_sample_index`.

4. Add 3D CLI resume support.
   - Add `--resume PATH` to `fiber_trace_3d.train`.
   - Change `run_training(...)` to accept an optional CLI resume checkpoint.
   - Resolve resume precedence explicitly: CLI `--resume` overrides any
     existing `training.resume`/top-level `resume` key.
   - Reuse the existing 3D snapshot loader so model and optimizer state are
     restored and `start_step` comes from the checkpoint.
   - Keep the existing fresh timestamped run-directory behavior for resumed
     runs, matching the 2D trainer behavior.
   - Print a clear resume line with checkpoint path, checkpoint step, next step,
     and new run dir.
   - Include the effective CLI resume path in the TensorBoard config text or an
     adjacent text field so the run record shows how it was started.
   - Validate that configured `max_steps` is greater than the resumed
     checkpoint step; otherwise fail clearly instead of silently exiting.

5. Normalize full-line context handling.
   - Build transformed full-fiber segment metadata for every 3D sample where a
     fiber line is available.
   - For NML/dense samples, target materialization may consume the line segments
     for dense supervision.
   - For JSON/non-NML samples, keep `_TARGET_MODE_CP_ONLY`; use the full-line
     segments only for visualization/context, while CP-only direction/presence
     losses remain at the CP neighborhood.
   - Ensure target materialization continues filtering dense rasterization by
     target mode.

6. Match 2D run-length and prefetch sentinels.
   - Preserve `training.max_steps: 0` as unbounded training repetition instead
     of converting it to a finite pass. Because this cannot naturally terminate,
     require explicit user interruption or an external run manager for that
     mode.
   - Keep bounded CP reuse controlled only by `training.max_sample_index`; do
     not use `max_steps` as a hidden sample bound.
   - Change 3D prefetch argument parsing so omitted `--prefetch-steps` uses
     `training.max_steps`; explicit positive values override config; explicit
     `0` prefetches all selected CPs; negative values fail clearly.
   - When `training.max_sample_index` is positive, prefetch that deterministic
     bounded training prefix for train data. When `test_datasets` is configured,
     also prefetch held-out test CPs once where applicable.

7. Align console progress semantics.
   - Keep 3D's existing first-100-step printing behavior.
   - Update the shared 2D spec from "first 100 deterministic sample indices" to
     "first 100 training steps" so both trainers use the same documented
     behavior.

8. Add tests.
   - Test that repeated bounded data samples receive different deterministic
     augmentation parameters when their raw training stream indices differ.
   - Test that the same raw stream index remains reproducible.
   - Test that dense test loader construction disables augmentation by default
     and respects `training.test_augment_enabled=true`.
   - Test that dense `test_control_points: 0` selects flat held-out CP order
     from zero.
   - Test that `--resume`/CLI resume precedence is wired for 3D training, at
     least through argument parsing and resume-path resolution.
   - Test 3D prefetch step-count resolution for omitted, positive, zero, and
     negative `--prefetch-steps`.
   - Test that JSON samples can carry full-line segments without dense
     materialization, and NML samples keep dense line supervision.

9. Validation commands.
   - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
   - `PYTHONPATH=vesuvius/src:. python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/targets.py`
   - `git diff --check`

## Spec Update

- Add/adjust shared specs to state that 3D follows the same raw training-stream
  augmentation seeding semantics as 2D.
- Document 3D `training.max_sample_index` and its interaction with prefetch and
  augmentation seeding.
- Document `training.test_augment_enabled` and the default non-augmented dense
  test loader behavior.
- Document 3D `--resume PATH`, its precedence over config resume keys, and the
  fresh-run-dir behavior.
- Document the 3D full-line-context versus target-mode-supervision split.
- Change the shared console-progress spec to first 100 training steps for both
  2D and 3D.
- Document 3D `training.max_steps: 0`, dense-test `test_control_points: 0`, and
  3D prefetch step-count resolution.

## Docs Updates

- Update `docs/code_structure.md` in the 3D loader/trainer sections to describe
  raw sample index, bounded data index, augmentation index, and test loader
  augmentation defaults.
- Document the exact 3D resume command form in local development notes.
- Update local development notes if the train/debug commands mention
  `max_sample_index`, `max_steps: 0`, prefetch step-count behavior, or test
  augmentation behavior.

## Changelog

- Add a 2026-07-15 entry for 3D augmentation-index parity, CLI resume support,
  dense test augmentation default, and full-line visualization context.

## Deviations Or Deferrals

- None planned. Any requirement that cannot be implemented directly must be
  logged in `planning/task_log.md` before continuing.
