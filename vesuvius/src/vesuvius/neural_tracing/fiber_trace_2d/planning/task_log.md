# Native VC3D 3D Fiber Tracer Task Log

## Implemented

- Added an explicit agent rule in root `AGENTS.md` and
  `fiber_trace_2d/AGENTS.md`: do not copy existing implementations into new
  locations; extract shared helpers/libs and port callers.
- Extracted shared Lasagna compact-channel sampling helpers into
  `vc/lasagna/ChannelSampler.hpp` and `src/lasagna/ChannelSampler.cpp`.
- Ported `LasagnaNormalSampler.cpp` to use the shared helper, removing the
  duplicated private compact-channel implementation.
- Added `vc_fiber_tracer`, a Qt-free native C++ core library for persisted
  fiber inference fields and CP-to-CP segment tracing.
- Added project-level `fiber_inference_datasets` and
  `selected_fiber_inference_dataset` storage to `VolumePkg`.
- Added a focused C++ smoke test,
  `volume-cartographer/core/test/test_fiber_trace3d.cpp`, covering a straight
  CP-to-CP trace, bidirectional acceptance, fusion, and CP preservation.
- Added a Ctrl-right-click generated-line context menu action,
  "Optimize segment with native fiber tracer", which dispatches a background
  native trace task through the existing line-optimization busy/save/view
  update path.
- Updated `planning/specs.md`, `docs/code_structure.md`, and
  `planning/changelog.md`.

## Validation

- `cmake --build volume-cartographer/build/python-bindings --target vc_lasagna`
  passed before GUI integration.
- `cmake --build volume-cartographer/build/python-bindings --target vc_fiber_tracer`
  passed before GUI integration.
- `cmake -S volume-cartographer -B volume-cartographer/build/ci-tests-clang-systemdeps`
  passed/reconfigured the existing CI test build tree.
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_fiber_trace3d`
  passed.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_trace3d`
  passed: `1 test case(s) passed`.
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target VC3D`
  passed. Existing unrelated Qt deprecation warnings were emitted in
  `InkDetectionOverlayController.cpp` and `CPointCollectionWidget.cpp`.

## Deviations / Remaining Work

- The native C++ tracer currently performs one-step beam expansion/pruning.
  The config carries `beamLookaheadSteps`, but full Python-equivalent
  multi-step lookahead is not implemented yet.
- Persisted tracer-optimized segment metadata, invalidation, and regular
  reoptimization protection are not implemented yet. Accepted GUI traces are
  applied and saved as ordinary optimized line geometry for now.
- GUI progress is the existing busy overlay plus worker log messages, not a
  numeric progress overlay in the line annotation dialog.
- The native core smoke test uses a deterministic fake prediction source.
  Real-data parity against the Python reference command has not been run.
