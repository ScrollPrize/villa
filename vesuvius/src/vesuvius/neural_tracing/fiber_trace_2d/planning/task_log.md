# 3D Augmentation Stream And 2D Spec Parity Task Log

## Notes

- Planning only so far.
- The shared 2D spec already states that `training.max_sample_index` bounds
  CP/data selection but does not bound augmentation seeding.
- Current 3D loader uses the same `sample_index` for descriptor lookup and
  augmentation parameter generation, so it lacks the explicit 2D raw/data index
  split.
- Current 3D test-loader construction copies train augmentation settings unless
  changed.
- Current 3D training can resume from JSON config keys, but lacks the requested
  CLI `--resume` path. The 3D `--checkpoint` flag is only for Trace2CP
  visualization and must not be treated as training resume.
- Follow-up spec scan found additional non-dimensional 2D parity items to
  integrate before implementation: `training.max_steps: 0` as unbounded
  training repetition, dense-test `test_control_points: 0` as full flat-order
  held-out evaluation, prefetch step-count resolution matching 2D, and console
  progress wording. The console progress requirement is intentionally changed
  to first 100 training steps for both 2D and 3D per user instruction.

## Deviations Or Deferrals

- None.

## Validation

- Not run yet; implementation has not started.
