# Task Plan: Opt-In Augment-Vis Profiling

## Implementation

1. Add `--augment-profile` for use with `--augment-vis`.
2. Keep normal `--augment-vis` behavior focused on export only:
   - no timing profile dicts;
   - no timing tables;
   - no output timing line.
3. When `--augment-profile` is set:
   - build the same contact sheet and summary as before;
   - run the same augmentation entries a second time without collecting output
     images;
   - print one timing table for pass 1 and one for pass 2;
   - keep `total/no-first` and `avg/no-first` rows in each table.
4. Do not change augmentation parameter generation, image output, cache behavior,
   or training paths.

## Spec Update

Update `planning/specs.md` so timing diagnostics are tied to
`--augment-profile`, and document the two-pass cold/warm behavior.

## Docs Updates

Update `docs/code_structure.md` to mention `--augment-profile` and the two
profile passes.

## Tests

- Extend runner/export tests so default augment-vis does not print timing rows.
- Add a profile-mode export test that asserts pass 1 and pass 2 timing tables
  are printed.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

No durable changelog entry is needed; this is a debug-output control change.
