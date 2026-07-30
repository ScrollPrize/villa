# Plan: Native 3D Trace2CP Refined Presence Visualization

## Implementation

1. Add a strip-plane alignment helper in the Python native 3D Trace2CP tool.
   - Derive local column and row axes from the actual sampled strip coordinate
     grid.
   - Compute a sign-invariant plane-alignment weight for each valid voxel:
     `sqrt((dir dot column_axis)^2 + (dir dot row_axis)^2)`.
   - Keep invalid/degenerate grid locations invalid for the modulation.

2. Extend `_sample_presence_on_strip`.
   - Keep the existing raw behavior by default.
   - Add an opt-in flag that multiplies sampled presence by the strip-plane
     alignment weight while reusing the already sampled direction tensor.

3. Wire visualization callers.
   - Single-pair fused side/top presence panels use plane-modulated presence.
   - Whole-fiber regenerated side/top presence panels use plane-modulated
     presence.
   - Original side/top presence panels remain raw presence.

## Spec Update

- Add the visualization-only requirement that refined/fused/regenerated native
  3D Trace2CP presence slices display presence multiplied by ambiguous
  direction alignment to the strip plane, not a signed dot with a single
  tangent vector.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP visualization notes with
  the raw-vs-modulated presence panel distinction.

## Tests

- Add/adjust unit tests for `_sample_presence_on_strip` to verify raw behavior
  remains unchanged and opt-in plane modulation scales by projection into the
  slice plane.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'strip_presence'`
- Run:
  `PYTHONPATH=vesuvius/src:. python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check`.

## Changelog

- Add a short 2026-07-30 changelog entry for modulated refined native 3D
  Trace2CP presence visualization.
