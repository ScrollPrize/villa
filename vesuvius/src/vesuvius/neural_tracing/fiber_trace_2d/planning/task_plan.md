# Plan: Direction-label MILP diagnostic

## Contract

- Add a separate `direction-diagnostic` command to
  `vc_fiber_trace_chunk`. It reads an existing crop-trace artifact and a
  compatible Lasagna normal manifest; it does not retrace fibers.
- Run the unchanged gradual two-direction classifier over every stored trace at
  the selected `--direction-dominance` value. Preserve original trace indices.
- Remove mixed traces before piece construction, spatial search, constraint
  scoring, optional strength pruning, and optimization. Write the complete
  initial direction OBJ family from the unfiltered stored traces with an
  `_initial` basename so the removed set remains inspectable. Then create the
  retained-line vector and explicit filtered-to-original trace-index map; hard
  continuity is built only from that retained vector.
- Reuse the canonical constraint extractor, batched winding sampler, optional
  `--constraints-per-fiber`, constraint OBJ writer, HiGHS solver, and label OBJ
  writer. The diagnostic uses the ordinary discrete H/V-plus-broken MILP with
  parity disabled because the reference contains only H/V information.
- Reject LP relaxation, exact-continuous H/V, parity-dependent, and
  solver-only edge-filter ablations in this diagnostic rather than comparing
  unlike models. Broken cost, MIP gap, extraction geometry, winding cutoff,
  pruning, threads, and cache controls remain configurable. The diagnostic
  forces `hvOnly=true`; redundant explicit `--hv-only` is rejected.
  `--mip-gap` is an allowed diagnostic solver option and
  `--no-winding-cutoff` is allowed because parity is disabled.
- Compare each optimized piece with its retained source trace's initial
  direction. H/V has an arbitrary binary gauge in every active connected
  component, so choose the component flip that minimizes active-piece
  disagreement; ties keep direction 1 mapped to H. Build components from the
  exact post-pruning constraint vector passed to the solver, including hard
  continuity, after removing edges incident to final decoded Broken pieces.
  This matches the solver's discrete canonicalization graph, including its
  conversion of zero-degree pieces to Broken.
- Count an active piece whose gauge-aligned H/V differs from its initial
  direction as an orientation error. Count every optimized broken piece as a
  broken error. Report both piece and unique-source-trace error totals.
- Print a compact gauge-aligned confusion table by initial direction, separate
  raw solver-canonical H/V/broken totals, active-component and flipped-component
  counts, aggregate error rates, and one stable row for every erroneous piece
  with original trace ID, trace-local piece ID, arc interval, initial direction,
  raw optimized label, component flip, aligned label, and error kind. Piece
  rate divides by retained extracted pieces; trace rate divides by retained
  source traces represented by at least one piece, with each erroneous source
  trace counted once across both error kinds.
- If every trace is mixed, still write the complete initial family plus valid
  empty constraint and label families and report an empty optimal diagnostic;
  do not silently skip outputs or treat mixed traces as solver pieces.
- For output base `<base>`, write the existing full-trace direction family as
  `<base>_initial.obj`, `<base>_initial_dir1.obj`,
  `<base>_initial_dir2.obj`, `<base>_initial_mixed.obj`, and matching anchors.
  Existing constraint connector suffixes and piece-label suffixes continue to
  use `<base>` and cannot collide. Initial artifacts contain full traces; MILP
  artifacts contain extracted pieces. The default base is sibling
  `<trace-stem>_direction_diagnostic`.

## Implementation

1. Add reusable core comparison types and a deterministic comparison function
   beside the existing HiGHS labeling API.
2. Extend CLI parsing and mode validation for `direction-diagnostic`, sharing
   existing extraction and solve paths rather than copying their behavior.
3. Filter mixed traces with an explicit filtered-to-original index map, emit
   the initial direction artifacts, run extraction/pruning/H/V MILP, then print
   the comparison.
4. Add focused tests for mixed removal mapping, component gauge flips including
   a broken middle piece, gauge ties, zero-degree Broken behavior, broken and
   opposite-label errors, disconnected components, multi-piece source traces,
   stable error ordering, unique-trace union/denominators, invalid/non-discrete
   inputs, exact output names, all-mixed outputs, and CLI option acceptance and
   rejection.
5. Build with `-j32`, run `test_fiberlet_crop_trace`, and exercise the
   centered-384 stored trace artifact at dominance 0.90.

## Spec Update

Document the diagnostic pipeline, mixed-trace exclusion point, H/V-only model,
component-gauge comparison, error definitions, stable report fields, and
supported/rejected options in `planning/specs.md`.

## Docs Updates

Add a runnable centered-crop example and explain the initial direction layers,
MILP layers, and comparison table in
`volume-cartographer/docs/fiber_chunk_tracing.md`.

## Testing

- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`.
- Run the full focused crop-trace test binary.
- Run the centered-384 diagnostic with the existing normal manifest and report
  retained/mixed fibers, constraints, solve result, and labeling errors.
- Run `git diff --check`.

## Changelog

Record the new direction-reference MILP diagnostic command.
