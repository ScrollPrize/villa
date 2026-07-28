# Task Log: VC3D-Only Physical Units For Native 3D Trace2CP

## Implementation Notes

- Started from the existing length-normalized whole-fiber Trace2CP metric.
- Removed the previous broad metadata probing helpers for dataset config,
  record metadata, alternate voxel-size keys, unit parsing, and OME multiscales.
- Added a single VC3D-only helper that reads
  `record.sampler.volume.metadata["voxelsize"]`, interprets it as micrometers,
  and converts to meters.
- Physical reference length is now the original base-coordinate fiber arc
  length multiplied by the VC3D voxel size in meters.
- Per-meter stdout/progress/summary output remains conditional: if VC3D does
  not expose a finite positive `voxelsize`, physical units are omitted.
- Updated tests to use fake VC3D sampler metadata and added a regression that
  dataset-config voxel-size keys are ignored.
- Updated specs, code-structure docs, and changelog wording to document the
  VC3D-only source.
- Removed VC3D's `discoverPublicSamplePixelSize` flag so remote volume
  metadata normalization always discovers
  `scan/tomo/acquisition/detector/samplePixelSize` when no explicit positive
  `voxelsize` exists.
- Rebuilt `volume-cartographer/build/python-bindings` target `vc_volume`.
- Refreshed the editable/local `volume-cartographer` pip install with
  `python -m pip install -e volume-cartographer --no-deps --break-system-packages`.
- Shortened whole-fiber human progress/stdout metrics to `err/kvx` and
  `err/m` with three decimals, and removed `physical_unit=m` plus reference
  length fields from human progress output. JSON summary fields remain
  full-precision and explicitly named.
- Restored native progress output to carriage-return line updates for
  non-terminal states, with a newline only at completion.

## Deviations / Deferred Items

- None so far.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "whole_fiber_trace"`
  passed: 5 passed, 142 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "whole_fiber_progress or whole_fiber_trace"`
  passed: 6 passed, 141 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 145 passed, 2 skipped.
- `git diff --check` passed.
- `cmake --build volume-cartographer/build/python-bindings --target vc_volume -j 8`
  passed.
- `python -m pip install -e volume-cartographer --no-deps --break-system-packages`
  passed.
- `python -c "import vc.volume as v; print(v.Volume)"` passed.
