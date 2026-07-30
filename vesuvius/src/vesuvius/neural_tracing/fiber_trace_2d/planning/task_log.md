# Task Log: Native Diagnostic Refresh And Edge-Truncated Extrapolation

## 2026-07-30 - Findings

- Loading works because `materializeGeneratedViews()` constructs labels from the
  persisted CP metadata.
- Successful optimization materializes those labels again, but then calls
  `refreshBranchLineViews()`. That replaces branch/control overlays through
  `setGeneratedBranchOverlayData()`, which clears all span metrics and does not
  repopulate them. The metadata itself remains intact in fiber mode.
- The Lasagna-mode Reoptimize button separately clears every
  `segment_to_next` record before invoking a reinitializer that already supports
  protected native spans. That conflicts with per-span persistence and is only
  appropriate during an explicit transition to Lasagna mode.
- `traceOneWayCore()` returns the initial state when final-frontier pruning finds
  no valid candidates, discarding earlier valid extrapolation steps.
- `replaceOpenTailsWithNative()` accepts only target-plane completion, so a
  retained edge-truncated path would still be replaced by the Lasagna tail
  without an extrapolation-specific acceptance rule.

## Plan Review

- Confirmed label repopulation uses current session controls, preventing stale
  label positions after overlay replacement.
- Limited edge truncation to `no_valid_candidates`; budget exhaustion remains a
  failure.
- Limited the changed acceptance semantics to extrapolated tails. CP-pair and
  whole-fiber tracing retain their current rules.
- No separate review agent was used for this small follow-up; the plan was
  checked directly against the active specs and both affected call paths.

## Deviations

- None before implementation.

## Implementation

- Branch/control overlay replacement now takes the current span diagnostics in
  the same generated-view state update. Reoptimize, branch refresh, link edits,
  and no-reoptimization CP edits all supply metrics from current session
  metadata instead of clearing them in a later overlay pass.
- Same-mode Lasagna Reoptimize no longer clears accepted CP-owned records before
  calling the existing protected-span full reinitializer. Explicit mode change
  to Lasagna still clears them.
- `traceOneWayCore()` retains the best state from its latest nonempty frontier
  and returns it when the next generation has no valid candidates.
- VC3D accepts a retained two-or-more-point extrapolation whose reason starts
  with `no_valid_candidates`; the missing-target-plane diagnostic suffix is
  preserved. Other incomplete reasons still fall back to Lasagna.

## Test Iteration

- The first focused run showed that one-way reasons append missing-plane
  context (`no_valid_candidates:missing_target_planes=extrapolation_distance`).
  The extrapolation-only classifier now matches the stable reason prefix rather
  than discarding that useful suffix.
- `test_fiber_trace3d` now proves a valid first step is retained when the next
  direction is invalid.
- `test_line_annotation_generated_views` now proves a fiber-mode tail ends at
  the last valid prediction sample instead of the longer Lasagna endpoint.

## Validation

- `test_fiber_trace3d`: 42 passed.
- `test_line_annotation_generated_views`: 50 passed, including atomic branch
  overlay replacement with a supplied native meeting-error metric.
- `test_lasagna_line_optimizer`: 35 passed.
- `cmake --build volume-cartographer/build --target VC3D -j32`: succeeded with
  the existing Qt SFINAE incomplete-type warnings.
- The interactive label repaint was not exercised by launching the GUI in the
  headless test environment. The generated-view state replacement is covered by
  the focused regression and the complete VC3D target compiles and links.
- `git diff --check` passed. The final audit confirmed that
  `no_valid_candidates` is recognized only by the extrapolation tail adapter;
  CP-pair and whole-fiber acceptance remain unchanged.
