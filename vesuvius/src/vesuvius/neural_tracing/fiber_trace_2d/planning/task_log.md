# Base-Voxel Fiber Acceptance Threshold Task Log

## Findings

- VC3D currently requires positive physical voxel-size metadata only because
  its `50 um` acceptance threshold is converted to trace voxels. Trace geometry
  itself does not need physical metadata.
- The configured local base volume has no trustworthy positive physical size:
  its generated VC metadata reports zero and its OME-Zarr transforms are
  dimensionless. The filename is intentionally not parsed as metadata.
- The C++ metric CLI currently compares its default threshold directly in
  trace-grid voxels. The Python native CLI compares in selected-volume voxels.
  Both therefore need explicit working-to-base conversion.

## Deviations

- Independent agent review is not permitted by the active runtime policy
  unless the user explicitly requests delegation. The implementation is being
  reviewed directly against the task, specifications, and call sites.

## Validation

Implemented:

- The shared C++ segment result now distinguishes trace-grid and base-grid
  endpoint errors. Acceptance compares the base-grid error to `20` base
  voxels; optional micrometer error is populated only from a finite positive
  base-voxel size.
- VC3D no longer rejects a segment action when `Volume::voxelSize()` is absent
  or zero. Rejection and success reports always name the base-voxel endpoint
  error and append micrometers only when available.
- `vc_fiber_trace_metric` now exposes
  `--error-threshold-base-voxels` with default `20`; its working trace error is
  multiplied by the manifest-derived trace-to-base scale before acceptance.
- The Python native CLI now exposes
  `--whole-fiber-error-threshold-base-voxels` with default `20`; it divides the
  public threshold by `volume_spacing_base` for internal target-plane search
  and records both selected-grid and base-grid endpoint errors.
- The former ambiguous CLI option/config names were removed rather than kept
  as aliases. They described working-grid voxels and would preserve the wrong
  scale-dependent contract.

Commands:

```bash
cmake --build volume-cartographer/build/ci-tests-clang-systemdeps \
  --target test_fiber_trace3d vc_fiber_trace_metric VC3D -j32
volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_trace3d
volume-cartographer/build/ci-tests-clang-systemdeps/bin/vc_fiber_trace_metric --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src \
  python -m pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py
python -m py_compile \
  vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py \
  vesuvius/tests/neural_tracing/test_fiber_trace_3d.py
git diff --check
```

Results:

- `test_fiber_trace3d`: 27 passed, including a nonzero-error case proving a
  trace error below 20 is rejected when its converted base error exceeds 20,
  and a missing-physical-size case proving physical metadata is optional.
- Python `test_fiber_trace_3d.py`: 179 passed, 2 skipped, including a
  `volume_spacing_base=4` boundary case where 5 working voxels equals 20 base
  voxels.
- `vc_fiber_trace_metric --help`: passed and lists the new base-voxel option
  with default 20.
- Focused C++ targets built successfully with `-j32`. Ninja repeatedly emitted
  its existing recoverable `premature end of file` warning and rebuilt more
  dependencies than expected. VC3D emitted existing unrelated Qt deprecation
  warnings.

The first Python pytest attempt inherited an installed pytest plugin that
requires unavailable `zarr.testing`; disabling third-party plugin autoload
used the repository tests without installing or changing dependencies.

## Limitations

- The interactive VC3D segment action was not run in this non-GUI session.
  Compilation and shared-core regressions pass, but the user's project remains
  the required manual end-to-end check.
