# Task Plan: Augment-Vis Timing Cleanup

## Scope

Exclude one-time deterministic sample-order cache construction from the
augment-vis per-sample timing table while keeping the actual per-sample
descriptor lookup timing visible.

## Implementation

- In `_export_augment_contact_sheet`, perform an unprofiled descriptor lookup
  before starting the source-build timer.
- Leave `build_strip_source` profiling unchanged so training/profile paths keep
  their current instrumentation.

## Spec Update

No spec change required; this is a diagnostic measurement cleanup.

## Docs Updates

Update `planning/status.md` and `planning/task_log.md`.

## Tests

- Run syntax validation for `runner.py`.
- Run the focused 2D loader test suite.
