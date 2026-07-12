# Trace2CP Z-Search Layer TIFF Export Plan

## Scope

- Add a runner CLI flag for Trace2CP z-search layer export.
- Preserve current z-search tracing, scoring, z-corrected JPG panels, and
  public metric semantics.

## Implementation

- Add `--trace2cp-z-layers-tif`.
- Reject the flag unless `--trace2cp-z-search` is active.
- Build a compact uint8 debug stack from the existing `_Trace2CpZPlaneCache`
  after tracing has inferred its layers:
  - sorted layer slice images first;
  - sorted presence maps second when the checkpoint exposes presence output.
- Store the stack and page labels in `_Trace2CpZTraceDebug`.
- Single-pair mode writes `trace2cp_z_layers.tif`.
- Whole-fiber mode writes pair-local TIFFs under `trace2cp_z_layers/`.
- Add summary/stdout lines with the output path(s), page count, and page labels.

## Spec Update

- Document the new flag and non-interleaved page ordering.
- Clarify that the export uses cached/inferred z-search layers and does not
  re-sample or interpolate.

## Docs Updates

- Update `docs/code_structure.md` runner section.
- Update changelog, status, and task log for this task.

## Tests

- Add a unit test for z-layer TIFF stack ordering.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
