# Task log: retain the main BP constraint component

- The current BP-only cohort retains all extracted pieces after constraint
  selection, so isolated and smaller factor components receive independent
  solver gauges.
- Component filtering must happen before reference/BP cross extraction;
  otherwise benchmark piece indices no longer correspond to the solved cohort.
- The reference/reference table currently prints during setup and the
  reference-to-BP table prints inside each solver run. Both need buffered
  formatting so they can be emitted at the command boundary without delaying
  ordinary progress output.
- Independent review rejected orientation-factor connectivity because it can
  retain multiple effective winding gauges. Selection must reuse the exact
  prepared winding graph and, for fixed-orientation runs, the prepass Mixed
  exclusions. It also requires explicit source-trace remapping and command-wide
  diagnostic storage.
- User clarification: constraints ending at final Mixed/Defect or invalid
  winding pieces must be omitted from the benchmark entirely, not counted as
  wrong and not used for offset calibration.

## Implementation

- Added a reusable constraint-report subset/remap operation. It retains only
  selected pieces, remaps source traces and constraint endpoints, and
  recomputes constraint counters.
- Added largest effective-winding-component selection using the same prepared
  winding factors, orientation state, signed-winding rules, and parallel
  winding cutoff as the solver.
- BP-only runs iteratively keep the largest effective winding component. With
  fixed orientations, each reduced cohort reruns the orientation prepass until
  the component set is stable. Equal-size components prefer the crop-central
  piece, then the lowest original piece index.
- Reference/reference diagnostics and each reference-to-BP benchmark are now
  buffered and printed together at the end of the command.
- Reference-to-BP observations whose solved BP endpoint is Mixed/Defect or has
  invalid winding state produce no candidate. Candidate-free observations are
  excluded before gauge offset calibration and from all accuracy totals.

## Validation

- Built `vc_fiber_trace_chunk`, `test_fiber_trace_winding_bp`, and
  `test_fiberlet_crop_trace` with 32 jobs.
- Focused CTest result: 2/2 tests passed in 1.15 seconds.
- A real 1024-crop BP run reduced the cohort from 500 to 465 traces and from
  19,389 to 16,428 constraints, reported one final winding component, and
  printed all reference diagnostics after the solver output.
- The active-only reference benchmark contained 510 usable cross constraints:
  409 right and 101 wrong (80.196%). Mixed/Defect and invalid-winding endpoints
  were absent from both this total and gauge calibration.
- Reproduced the split-fiber command with `--piece-length 512`. Component
  filtering can remove an interior piece from a source fiber, so the subset
  operation now splits retained noncontiguous runs into separate represented
  traces rather than pretending the remaining arc intervals are consecutive.
  The full command completed with 1,250 retained pieces, one winding gauge, and
  1,237 usable active-only reference constraints.
