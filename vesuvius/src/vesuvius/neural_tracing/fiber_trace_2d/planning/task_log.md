# Task Log: VC3D Fiber-Global Tracing Mode

## 2026-07-30 - Discovery And Planning

- The existing GUI exposes only per-span Ctrl-right-click native tracing; normal
  CP edits and the full-reoptimization button always dispatch to Lasagna.
- Successful native interpolations already persist on their owning CP as
  `segment_to_next`, and shared Lasagna reinitialization protects those spans.
- The shared reinitializer solves outward from a seed span and passes the dense
  endpoint direction of each completed/protected span into the neighboring
  span's continuation candidates. This already provides the requested
  native-neighbor direction behavior when native spans are protected during
  fallback.
- Lasagna reinitialization already generates both open tails from the endpoint
  CP and inward dense-line direction. The native tracer exposes the shared
  one-way beam path but requires target planes, so bounded extrapolation will
  reuse it with an artificial distance plane rather than duplicate stepping.
- The global mode belongs on the fiber/session, while `segment_to_next` remains
  per-span outcome/provenance. Missing mode metadata will default to Lasagna.
- The existing `Length` spin box controls total construction length from a
  single seed, not explicit tail distance. A separate extrapolation spin box is
  required.

## Workflow Deviation

- `AGENTS.md` asks for an independent plan review. Runtime policy prohibits
  sub-agent delegation unless the user explicitly requests it, so a direct
  review was performed and the independent-review checklist item remains open.

## 2026-07-30 - Implementation

- Added the persisted fiber-wide `optimization_mode` contract with `lasagna`
  and `native_fiber_trace3d` values. Missing mode metadata defaults to Lasagna.
- Added the line-annotation mode combo and a persisted base-voxel
  extrapolation-distance spin box. Mode changes are transactional and restore
  the prior mode and segment records if the rebuild cannot start or fails.
- Exported native endpoint extrapolation through `vc_fiber_tracer`; it builds a
  distance target plane and reuses the shared one-way beam tracer.
- Added mixed whole-fiber orchestration. Native CP spans are protected,
  unsuccessful native spans are rebuilt by the shared Lasagna reinitializer,
  and native endpoint extrapolation falls back independently to each Lasagna
  tail.
- Routed full rebuilds, automatic CP edits, mode changes, and synchronous save
  finalization through the fiber-wide mode. Manual per-span trace/revert remains
  independent of the global mode.
- Updated the specifications, code-structure documentation, VC3D fiber JSON
  documentation, and changelog.

## 2026-07-30 - Validation

- Built VC3D and the affected C++ test targets with 32 build jobs.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_trace3d`:
  39 test cases passed.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_line_annotation_generated_views`:
  47 test cases passed.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_lasagna_line_optimizer`:
  29 test cases passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src python -m pytest vesuvius/tests/neural_tracing/test_fiber_trace.py volume-cartographer/scripts/tests/test_fiber_merge.py volume-cartographer/scripts/tests/test_vc_sync_helpers.py`:
  185 tests passed.
- `git diff --check` passed.

## Test Limitation

- The current focused test target does not instantiate `LineAnnotationDialog`,
  and no standalone headless Qt dialog harness exists in this area. The new
  controls and signal wiring are compile-checked through the VC3D target; their
  interactive layout and behavior still require a GUI smoke test.
