# VC3D Native Tracing For New Fibers

## Findings

- New `LineAnnotationSession` instances and the GUI correctly default to
  `NativeFiberTrace3d`.
- Initial seed placement nevertheless called the generic Lasagna
  `startOptimization` path unconditionally. It did not inspect the global mode
  or open fiber-inference data.
- The existing fiber-mode optimizer already supports one control point: it
  rebuilds a Lasagna baseline and replaces both open tails with native traces.
  The initial creation path simply never dispatched to it.
- Applying the initial Lasagna task normally also materializes and saves it, so
  chaining after ordinary completion would persist a mislabeled intermediate
  result.

## Plan Review

- The plan reuses the existing native single-CP implementation and does not add
  a second extrapolation algorithm.
- It preserves Lasagna creation when no fiber inference is configured, while a
  configured but invalid dataset remains a visible error.
- The internal Lasagna baseline is applied only as transient session input and
  is neither displayed nor saved.
- Independent review was not used because delegation is prohibited unless the
  user explicitly requests subagents; the plan was reviewed locally instead.

## Implementation

- Added a shared, Qt-free seed-dispatch decision that selects native seed
  tracing only for native global mode plus either a selected inference dataset
  or exactly one attached inference dataset. Multiple unselected datasets do
  not force a file picker.
- Seed placement opens configured fiber inference before mutating the new
  fiber. A configured but invalid source therefore reports the existing error
  and does not silently start a Lasagna fiber.
- Added a session continuation flag. Successful Lasagna seed completion applies
  its line only as transient input, suppressing generated-view materialization,
  deferred dialog presentation, success callbacks, and fiber persistence.
- Completion immediately dispatches the existing single-control-point native
  optimizer with full retracing. That optimizer reuses the baseline tangent and
  replaces both open tails using established native extrapolation/fallback
  semantics.
- Added command-line information logs for both the native continuation and the
  explicit no-selected/unique-inference Lasagna path.
- Updated the specification, VC3D fiber documentation, and changelog.

## Validation

- `cmake --build volume-cartographer/build -j32 --target VC3D`: passed.
- `cmake --build volume-cartographer/build/ci-coverage-clang-systemdeps -j32
  --target test_line_annotation_generated_views`: passed.
- `test_line_annotation_generated_views`: 57 cases passed, including the new
  selected/unique-dataset dispatch matrix and the existing single-CP native
  extrapolation/fallback cases.
- No automated Qt interaction fixture currently places a real GUI seed against
  an attached inference manifest. The production dispatch and native core are
  covered separately; an interactive VC3D smoke test remains appropriate.
