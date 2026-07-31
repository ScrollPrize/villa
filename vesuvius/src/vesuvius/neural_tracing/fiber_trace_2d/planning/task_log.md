# Remove Unpublished Fiber Version 2

## Implementation

- Restricted VC3D, Atlas, the shared fiber tracer, and the Lasagna line probe
  to top-level `vc3d_fiber` versions 1 and 3.
- Kept version-1 numeric control points and version-3 object control points.
- Removed the VC3D and shared-reader migrations for segment metadata schemas
  `(1, 1)` and `(2, 2)`; version 3 accepts only the current `(3, 2)` schema.
- Restricted the Python training reader and sync merger to the same contract.
- Removed version-2 fallback-diagnostic normalization from sync because v2 is
  now rejected before merging.
- Replaced v2 migration tests with explicit file-version and descriptor-schema
  rejection tests.
- Updated the format docs, implementation map, specification, and changelog.

## Validation

- `python -m py_compile volume-cartographer/scripts/fiber_merge.py volume-cartographer/scripts/vc_sync.py vesuvius/src/vesuvius/neural_tracing/fiber_trace/fiber_json.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src python -m pytest -q vesuvius/tests/neural_tracing/test_fiber_trace.py volume-cartographer/scripts/tests/test_fiber_merge.py volume-cartographer/scripts/tests/test_vc_sync_helpers.py`
  - 194 passed.
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_line_annotation_generated_views test_atlas -j32`
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_line_annotation_generated_views`
  - 56 test cases passed.
- `cmake --build volume-cartographer/build --target VC3D vc_lasagna_line_probe vc_fiber_trace_metric -j32`
  - All production targets built successfully.

## Known Test Fixture Failure

- The full `test_atlas` binary still reports three pre-existing pred-snap
  fixture failures because its Lasagna `nx` channel is not a 3D `(Z,Y,X)`
  zarr. The target compiles, and the failure is unrelated to fiber file-version
  parsing.

## Compatibility Decision

- Top-level file version 2 was never published and has no compatibility path.
- `tracer_version: 2` remains required inside current version-3 segment
  descriptors; it is a tracer-schema version, not the removed file version.
