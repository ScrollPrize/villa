# Plan: benchmark experiment-step plots

## Plot semantics

1. Preserve input order among experiments with the same algorithm completion
   date and assign the resulting sequence integer steps starting at one.
2. Define the historical Pareto frontier as strict measured best-so-far score
   improvements. Draw its monotone step line and annotate only those points.
3. Give every non-frontier result its own distinct marker and named legend
   entry. Keep unmeasured assumed-floor controls individually identifiable and
   exclude them from the measured frontier.
4. Rotate frontier annotations 30 degrees toward the upper left and put a
   compact legend below the axes.
5. Show integer step numbers as the primary x-axis labels and add completion
   dates only at spaced representative steps.
6. Replace the censored replay score with mean segment length over total tested
   length, `100/(failures+1)`, and use matching names in C++, JSON, CLI output,
   plot data, and documentation. Bump replay JSON to version 4, emit
   `reliability_segments` and `mean_segment_length_*`, and remove the deprecated
   distance-per-failure and zero-failure-convention fields.
7. Give every algorithm variant a stable method identifier and base label.
   Validate base-label consistency across plots, and derive BP labels only by
   appending ` + BP` without changing the base wording or order.

## Tests

1. Add focused Python tests for stable date ordering, strict frontier
   selection, and same-date experiment steps.
2. Validate benchmark JSON, run the plotting tests, regenerate all SVGs, and
   inspect their tick, marker, annotation, and legend content.
3. Update and run the focused C++ replay-summary tests for zero, one, and
   multiple failures plus the versioned JSON output.
4. Test that conflicting base labels for one method are rejected and validate
   the generated cross-plot labels.

## Spec update

- Update the existing benchmark-plot contract with stable integer steps,
  measured-frontier semantics, annotation and marker rules, and legend
  placement. No tracing behavior changes are required.

## Docs updates

- Document the experiment-step x-axis, sparse completion-date ticks, frontier
  labels, and non-frontier markers in the benchmark results page.
- Recompute displayed historical replay values from archived total lengths and
  failure counts while retaining original JSON hashes and clearly identifying
  their deprecated version-2/3 fields; do not imply that the runs were repeated.

## Changelog update

- Record the benchmark plot readability change.
