# Trace2CP Side Z-Axis Correction Task Log

## Implementation Notes

- Added optional `side_axis_xyz/side_axis_zyx` fields to side-strip grids.
  Existing `offset_axis_xyz/offset_axis_zyx` remains the side image-y Lasagna
  row-normal axis.
- Updated side-strip coordinate generation to interpolate both the row-normal
  axis and the VC3D frame side axis.
- Added Trace2CP-only side-z grid helpers in the loader. Z-search layers now
  offset center segment coordinates along `side_axis_*` by
  `layer * z_step_voxels * volume_spacing_base`.
- Updated `_Trace2CpZPlaneCache`, z-corrected surface reconstruction, traced
  top-strip reconstruction, and refined Trace2CP source construction to treat
  traced `z_voxels` as side out-of-plane offset, not side image-y offset.
- Updated the side/top-z experiment to sample row-normal and side-z axes
  separately: row-normal lifts side-view direction, side-z applies the
  out-of-plane offset and local top-patch row axis. The side-z axis is
  orthogonalized against the sampled tangent and row-normal before use.
- Restored production of side-presence z-pillar top panels in Trace2CP top
  debug output. The panel drawing path still existed, but `_evaluate_trace2cp_pair`
  was no longer populating original, fused-center, or z-corrected presence
  z-pillars after the earlier debug cleanup.
- Updated specs, code-structure docs, changelog, and task status.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/strip_geometry.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result after side-z correction: `249 passed in 10.27s`
  - Result after restoring presence z-pillars: `250 passed in 10.81s`

## Deviations

- Kept `strip_z_offset` on `sample_trace2cp_segment_source()` as the existing
  row-axis offset API to preserve non-z Trace2CP and regular strip behavior.
  Added dedicated side-z methods for Trace2CP z-search instead of changing the
  meaning of that existing argument.
