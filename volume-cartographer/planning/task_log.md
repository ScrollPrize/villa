# Task Log

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
