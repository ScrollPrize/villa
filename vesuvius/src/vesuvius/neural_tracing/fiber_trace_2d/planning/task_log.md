# Native 3D Trace Fusion Pairwise Meeting Log

## Planning Notes

- Current native 3D fusion still searches over straight CP-axis progress and
  interpolates each trace at that progress.
- The requested change replaces that primary meeting search with pairwise
  scoring over traced arc length.
- The plan keeps `gap_factor = 1.0` exactly as requested for the first
  implementation.
- The previous accidental native Trace2CP startup stdout change should be
  removed during implementation because it was not part of the requested image
  behavior fix.

## Deviations / Deferred Items

- No implementation has been done for this task yet.

## Validation

- Pending implementation.
