# Plan: VC3D-Only Physical Units For Native 3D Trace2CP

## Goals

- Remove the previous custom physical voxel-size readers.
- Use only `record.sampler.volume.metadata["voxelsize"]` from the VC3D Python
  `Volume` binding.
- Treat `voxelsize` as micrometers and convert to meters with `1e-6`.
- Omit per-meter stdout/progress/summary values when the VC3D metadata path is
  unavailable or invalid.
- Use compact human stdout/progress labels `err/kvx` and `err/m`, rounded to
  three decimals, without physical unit or reference length fields.
- Restore single-line live progress updates with carriage returns instead of
  printing one line per whole-fiber segment.

## Non-Goals

- Do not change the primary `restarts_per_kvx` metric.
- Do not change tracing, restart detection, visualization, or cache behavior.
- Do not parse metadata files directly in fiber code.
- Do not add fallback keys, filename inference, or dataset-config physical-unit
  handling.

## Implementation Steps

1. Replace `_selected_voxel_size_xyz_m` and its helper stack with one
   VC3D-only helper that reads `record.sampler.volume.metadata["voxelsize"]`.
2. Compute physical reference length from original base-coordinate fiber line
   arc length multiplied by the VC3D voxel size in meters.
3. Keep per-meter progress/stdout/summary output conditional on the helper
   returning a finite positive value.
4. Update tests to attach fake VC3D sampler metadata instead of dataset-config
   voxel-size keys.
5. Replace docs/spec wording that described dataset config, record metadata,
   OME multiscales, or other custom fallback readers as accepted sources with
   explicit language rejecting those paths.
6. Remove VC3D's `discoverPublicSamplePixelSize` gate so remote
   `metadata.json` values at
   `scan/tomo/acquisition/detector/samplePixelSize` are always normalized into
   `metadata["voxelsize"]` when no explicit positive `voxelsize` exists.
7. Rebuild the local VC3D Python binding and refresh the editable/local pip
   install with `--no-deps`.
8. Shorten whole-fiber human progress/stdout metric labels and precision while
   keeping full-precision JSON summary fields.
9. Update `_emit_native_progress()` to use carriage-return progress updates for
   non-terminal states and a newline only at completion.

## Spec Update

- Native 3D whole-fiber Trace2CP physical units come only from VC3D volume
  metadata key `voxelsize` in micrometers.
- If VC3D does not supply a valid positive `voxelsize`, per-meter output is
  omitted.
- VC3D remote volume metadata normalization always discovers the public
  `samplePixelSize` field when needed; this is independent of base-scale
  rebasing.
- Human whole-fiber progress/stdout uses `err/kvx` and `err/m` labels with
  three decimal places and omits reference length and physical unit tokens.
- Live progress uses carriage returns to update one line and must not emit one
  newline per segment.

## Docs Updates

- Update `docs/code_structure.md` native 3D whole-fiber metric description to
  reference VC3D `volume.metadata["voxelsize"]` only.

## Validation Commands

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "whole_fiber_trace"`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- `git diff --check`

## Changelog Update

- Update the existing native 3D whole-fiber Trace2CP changelog entry to state
  that per-meter reporting uses VC3D `voxelsize` metadata only.
