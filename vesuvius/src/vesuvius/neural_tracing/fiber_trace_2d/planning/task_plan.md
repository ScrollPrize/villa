# Trace2CP Vertical Space Doubling Plan

## Scope

- Change only Trace2CP segment patch loading height.
- Keep CP-local training patches, regular strip sampling, metric semantics,
  TTA behavior, and tracer logic unchanged.
- Apply to all Trace2CP users because they all call
  `build_trace2cp_segment_patch`.

## Implementation

1. Update Trace2CP segment height calculation in `loader.py`.
   - Replace the current `2 * patch_shape_hw[0]` height with
     `4 * patch_shape_hw[0]`.
   - Prefer a named local multiplier so the intent is obvious.

2. Update tests.
   - Rename the existing double-height regression to four-times-height.
   - Change expected image/valid/coords heights from `10` to `20` when the
     configured patch height is `5`.

## Spec Update

- Update the Trace2CP segment loading spec from "twice configured patch
  height" to "four times configured patch height".

## Docs Updates

- Update `docs/code_structure.md` Trace2CP description to say four times the
  configured patch height.
- Update changelog, status, and task log.

## Validation

- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run `git diff --check` on touched files.
