# Task Log

## 2026-09-28 Commit and integrate main

- Committed signed CP direction annotations/reset menus as 160ff1ac6 and
  project-name label as 1be8594cc. Unrelated untracked experiments excluded.
- Merged origin/main (f4570bfa6). Resolved controller, documentation and Python
  merge conflicts by retaining upstream version-4 span tags and nearest donor
  matching alongside our display normal, direction and width metadata.
- Extended direction merge regression coverage to version 4 with a damaged
  span tag. VC3D and all seven focused test executables rebuilt successfully.
- Post-merge validation: six CTest targets passed (generated views, trace3d,
  Lasagna optimizer/view surfaces, QuadSurface, fiber slice geometry); direct
  test_fiber_global_layout passed all 36 cases; Python fiber_merge/vc_sync helper
  suites passed all 296 tests. Diff against origin/main passes whitespace checks.
- No new live GUI session or real-volume trace performed for this merge.
  No push or PR creation until the user approves the proposed title/description.

## 2026-09-27 Correction reset menu scope

- Context-menu reset now addresses the selected CP and appears at the bottom.
  Whole-fiber reset moved to the annotation window menu.
- Both use one controller callback with optional CP scope. Peer panes match CPs
  by position. Only removed directions dirty their adjacent spans; normal-only
  reset does not request geometric reoptimization.
- Added regression coverage for field reset, neighboring CP preservation and
  span metadata preservation.
- Independent review unavailable; local review used. Live GUI validation not
  performed. VC3D and the focused test target built successfully.
- Build: cmake --build volume-cartographer/build --target VC3D test_line_annotation_generated_views -j32
- Passed: ctest --test-dir volume-cartographer/build -R '^test_line_annotation_generated_views$' --output-on-failure
- git diff --check passed. No commit made.
