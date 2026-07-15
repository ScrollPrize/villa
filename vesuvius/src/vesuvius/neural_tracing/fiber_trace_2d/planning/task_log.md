# 3D Fiber Follow-Up Task Log

## Implementation Notes

- Current 3D affine augmentation is present:
  - `augment_shift_zyx` controls CP placement inside the output patch;
  - `augment_rotation_degrees` samples a 3D axis-angle rotation;
  - isotropic scale and axis flips are also wired.
- Refactored 3D geometry into explicit paired maps:
  - `backward_source_zyx` samples the CP-centered 3D patch from selected-level
    source-volume coordinates;
  - `forward_map_zyx` maps source-volume coordinates back to output-patch
    coordinates for fiber line/control-point lookup and target generation.
- Labels now transform line-window source points through `forward_map_zyx`
  instead of multiplying by the affine matrix directly.
- Added opt-in smooth displacement config and deterministic per-sample control
  lattices:
  - `1d`: one direct component offset as a function of one coordinate;
  - `2d`: one direct component offset from a 2D lattice;
  - `3d`: explicitly invertible triangular coupling stages.
- Smooth displacement is baked into both maps during construction. No runtime
  path searches, solves, or derives one map direction from the other.
- Added opt-in anisotropic 3D blur as a torch value augmentation after volume
  sampling.
- Added `fiber_trace_3d.trace2cp_bridge`, which samples dense 3D checkpoint
  output at Trace2CP strip coordinates, projects Lasagna 3x2 directions into
  the strip frame, carries presence, and calls the existing 2D Trace2CP scorer.
- Deliberate boundary: this task adds the low-level projection/scoring bridge.
  It does not yet add a train-loop `test_trace2cp_enabled` best-checkpoint
  switch, because that needs an explicit 2D metric-loader config/coordinate
  ownership path rather than guessing from the 3D training config.
- Added inert new config keys to the checked-in 3D example configs.
- Shear/skew and ringing remain unsupported and still fail loudly when non-zero.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - Result: `11 passed in 2.55s`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/tests/neural_tracing/test_fiber_trace.py`
  - Result: `330 passed in 9.80s`.
