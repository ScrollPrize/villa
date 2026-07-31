# Main Line-Annotation GUI Merge

## Discovery

- Started a plain `git merge main` at `fc25b4d1f`.
- Git merged the controller, settings, viewer lifecycle, and unrelated main
  changes automatically.
- Textual conflicts are limited to `LineAnnotationDialog.cpp` and
  `LineAnnotationDialog.hpp` around toolbar controls and their members.
- Identified an unmarked semantic conflict: main replaced `lineSurface` with
  the schematic overview and now assumes one rendered strip, while the target
  requires the overview plus both rendered strips.
- Confirmed the desired top-view strip is the branch's ordinary interactive
  `lineSurface` viewer, not the older fixed-height/fit-to-width implementation.

## Plan Review

- Reviewed the plan against `planning/specs.md`, `planning/plan.md`, the
  clarified user requirement, and the merged `#1286`/`#1289` lifecycle code.
- The plan preserves persisted per-span labels and all fiber interpolation
  behavior while adopting main's toolbar, schematic overview, in-place view
  refresh, and teardown fixes.
- Independent subagent review was not used because the active collaboration
  policy prohibits delegation unless the user explicitly requests subagents;
  this local review is the only process deviation.

## Implementation Notes

- Resolved the toolbar conflict with main's Annotation popup and action-backed
  auto-reoptimization while retaining the fiber-global Lasagna/Fiber model
  combo and extrapolation-distance control.
- Kept main's intentional removal of the seed-direction and current-cut
  shift-scroll-mode selectors.
- Added the rendered `lineSurface` pointer to `GeneratedViews` so held overlays
  can use the exact prior top-strip geometry during in-place replacement.
- Kept the schematic overview as a separate full-width widget and restored the
  two-entry interactive rendered-strip loop in `lineSurface`, `lineSideSlice`
  order.
- Restored independent camera state and equal splitter participation for both
  rendered strips. The outer cut/strip default remains 2:1 so two rendered
  strips retain useful height.
- Replaced the one-strip pending flag with per-strip state. Static overlays,
  current-position markers, and span labels now select held/current geometry
  and descriptor data independently for each rendered strip.
- Kept main's surface-epoch gating, placement focus, immediate schematic-map
  feedback, and clear-viewer-references-before-container-deletion behavior.
- Kept pause and optimization badges on the bottom `lineSideSlice` viewer.
- Updated the specification, VC3D fiber documentation, implementation map, and
  changelog with the merged layout contract.

## Validation

- `git diff --check`
  - Passed.
- `git diff --name-only --diff-filter=U`
  - Empty after staging; both merge conflicts are resolved.
- Targeted `git diff main --check` over the resolution and documentation files
  - Passed. The complete cached merge diff still reports trailing whitespace
    already present in main's new `scripts/spiral/track_graph_testing.py`; the
    merge resolution leaves that unrelated incoming file unchanged.
- `cmake --build volume-cartographer/build --target VC3D vc_lasagna_line_probe vc_fiber_trace_metric -j32`
  - Passed with the existing Qt deprecation/SFINAE warnings.
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_line_annotation_generated_views -j32`
  - Passed. Ninja reported and recovered from a premature build-file EOF before
    CMake regenerated the build.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_line_annotation_generated_views`
  - 56 test cases passed.
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target VC3D -j32`
  - Passed under Clang with existing Qt deprecation warnings.

## Deviations And Limitations

- No implementation requirements were simplified or deferred.
- Interactive visual verification of the final Qt layout was not run in this
  non-GUI session; both GCC and Clang compiled the complete VC3D target.
