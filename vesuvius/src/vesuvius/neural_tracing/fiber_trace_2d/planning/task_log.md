# Task log: benchmark experiment-step plots

## 2026-09-04

- The existing plot uses calendar dates directly and sorts same-date points by
  algorithm name, collapsing and reordering experiments that should occupy
  distinct steps.
- The revised plot will use stable date ordering, strict measured best-so-far
  points as the historical Pareto frontier, and will not allow unmeasured floor
  assumptions to establish that frontier.
- Independent review required integer step numbers to remain primary labels,
  clarified that the measured envelope is absent before its first measurement,
  and identified the existing benchmark specification as requiring an update.
- Implemented one-based experiment coordinates, date-change tick metadata,
  strict measured frontier selection, frontier-only 30-degree annotations,
  diamond non-frontier markers, separately marked assumptions, and a legend
  below each plot. The initial render exposed a clipped first annotation and a
  redundant frontier legend entry; both were corrected before validation.
- Corrected an initial misinterpretation that grouped all dominated measured
  points under `Other measured result`: every non-frontier result now receives
  an individual marker and named bottom-legend entry.
- Replaced the censored `100/max(failures,1)` replay metric with mean segment
  length `100/(failures+1)`. The benchmark treats the complete tested corpus as
  one interval, so `N` failures create `N+1` segments. Renamed the C++ summary,
  JSON fields, CLI headings, and plot metric accordingly; JSON is now version 4.
- Independent review confirmed the aggregation and required the version-4
  schema transition to be explicit. Added negative JSON assertions for every
  removed legacy summary field and documented that historical values were
  recomputed without rerunning or rewriting archived JSON provenance.
- Replaced ad hoc plot labels with stable method IDs and base labels. BP-derived
  plots now append only ` + BP`; specifically, the staged method is consistently
  `Fiberlet + staged filtering` or `Fiberlet + staged filtering + BP`, never
  reordered as `Staged Fiberlet`.
- Validation: seven focused Python tests pass with external pytest plugin
  autoload disabled; all nine focused C++ replay benchmark tests pass in the
  Release build; benchmark JSON validation succeeds; two consecutive SVG
  generations have identical SHA-256 digests; targeted `git diff --check`
  passes; and all three rasterized SVGs were visually inspected.
