# Trace2CP DP Local-Angle Semantics Plan

## Implementation

- Change the side Trace2CP DP transition step constant from 32 px to 4 px.
- Remove the code that converts `max_direction_angle_degrees` into a hard
  vertical move cap.
- Keep a separate broad DP compute search band, independent of candidate angle,
  to avoid unbounded DP states while still allowing steep local fiber slopes.
- Reduce shared DP second-order smoothing defaults by 10x:
  `dy=0.005`, `dz=0.01`.
- Keep candidate angle as a local direction-field angular excess penalty only.

## Spec Update

- Update specs to state that side DP candidate angle is local to the sampled
  direction field and must not cap global horizontal slope.
- Document the 4 px side DP transition step and reduced smoothing defaults.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP section with the revised DP
  transition step and local-angle semantics.

## Tests

- Add a regression test where a steep local direction path connects despite a
  small candidate-angle setting.
- Run focused Trace2CP DP tests and py-compile the runner.

## Changelog

- Add a dated note for the Trace2CP DP local-angle/smoothing adjustment.
