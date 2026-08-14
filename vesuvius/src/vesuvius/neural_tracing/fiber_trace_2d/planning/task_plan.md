# Plan: bounded fiberlet replay comparison

## Implementation

1. Parse replay-only `--length N` as an optional finite positive number of base
   voxels. Select one `ForwardPolylineArcInterval` at the CLI boundary so
   omission means reference end and overshoot clamps there; pass its effective
   absolute begin/end to extraction and both replay engines.
2. Add an optional reference-end bound to the greedy replay request. Clamp
   forward matching, trace budget, target point, reset progression, completion,
   and failure fractions to the effective interval.
3. Add the same optional bound to graph replay. Bound seed projection, matching,
   reset progression, completion, and fractions; stop evaluating route points
   once the selected reference end is reached. If that happens inside a
   fiberlet, publish an explicit partial terminal edge: keep the selected edge's
   full identity/cost accounting, retain only route samples through the bound,
   set no terminal anchor node, and mark the segment as partial.
4. Feed the one CLI-selected interval to tube extraction and both engines.
   Persist only its exact sliced reference geometry plus requested/effective
   interval metadata in replay outputs. Validate engine intervals, failure arcs,
   fractions, and published geometry before writing. Keep `--along`
   visualization windows clipped to this interval, with a backward-only window
   when a failure occurs at the selected end.
5. Preserve full-reference behavior when `--length` is omitted. No compatibility
   layer is required for the experimental replay artifact; serialize the
   effective graph replay end explicitly.

## Tests

1. Extend greedy replay tests with a bounded interval ending inside the source
   reference and assert completion/end/fractions never use the full tail.
2. Extend graph replay tests with the same bounded interval, including an edge
   that crosses the boundary. Verify both boundary outcomes: an over-threshold
   endpoint is a failure at fraction one, while an accepted endpoint completes;
   neither evaluates later edge samples.
3. Retain shared interval tests for omission, explicit limits, and overshoot.
4. Add preprocessing/publication tests proving the omitted reference tail is
   absent, plus regression coverage that omission still selects the full tail.
5. Build `vc_fiberlets`, `test_fiber_trace3d`, `test_fiberlet_paths`, and
   `test_fiber_replay` with `-j32`; run the focused tests and verify help text.

## Spec Update

- Specify `--length` units, origin, clamping, shared-engine semantics, boundary
  precedence, partial terminal edges, and artifact interval behavior. Replace
  replay wording that assumes the selected interval is always the full fiber.

## Documentation Updates

- Update the replay command and behavior in `volume-cartographer/docs/fiberlets.md`.
- Record review, implementation, validation, and the change in planning docs.
