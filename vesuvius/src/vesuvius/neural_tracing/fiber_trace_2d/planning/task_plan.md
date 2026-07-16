# Native 3D Trace2CP Product Candidate Scoring Plan

## Implementation

- Replace native 3D Trace2CP candidate selection with the requested product
  score:
  `dot(current_dir, step_dir) * dot(candidate_dir, step_dir) * candidate_presence`.
- Align sign-ambiguous sampled direction axes before each dot product.
- Keep invalid candidate rejection unchanged.
- Remove obsolete native 3D Trace2CP additive candidate-selection CLI/config
  knobs: `--direction-weight` and `--presence-weight`.
- Keep `--step-voxels` as the step-size control; default remains `4.0`
  selected-level voxels.

## Spec Update

- Update `planning/specs.md` native 3D Trace2CP candidate-selection text to
  document product scoring instead of weighted additive loss.
- State that native 3D Trace2CP candidate selection is not controlled by
  direction/presence weights.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP section with the product
  score and removed candidate-weight knobs.

## Tests

- Add/update a focused native 3D Trace2CP candidate-scoring regression test.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d_trace2cp"`.
- Run `py_compile` for the native 3D Trace2CP tool.
- Run `git diff --check`.

## Changelog

- Add a 2026-07-16 changelog line for product candidate scoring.
