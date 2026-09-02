# Task Plan

## Reference artifacts

1. Retain the aggregate tagged-reference OBJ and add deterministic sibling
   files `<base>_reference_hs_<half_step_index>.obj`, one source per file.
2. Use the existing ordered-polyline writer and reference header. Stage the
   aggregate and all indexed files first, then replace the previous artifact
   family. Clean up staged/published files on failure and remove stale indexed
   siblings on fewer-source or reference-free runs without matching unrelated
   paths.
3. Keep filename order authoritative: file index `i` represents virtual
   winding `i/2`, with solver navigation slot `i/2` rounded down.

## Viewer

1. Discover indexed reference siblings independently of the aggregate file.
   Reject malformed matching names and require exactly one nonempty ordered
   polyline whose `reference_<i>_...` ordinal matches the file index.
2. Add one Napari layer per indexed reference fiber while retaining the
   aggregate layer.
3. Maintain a complete two-half-step reference visibility grid across the
   contiguous union of solver and reference winding slots. Next, Previous, and
   animation rotate every reference bit by two half-steps, preserving parity,
   hidden bits, and wraparound without depending on which files exist. Slot
   association is navigation-only and does not assert cross-component gauge
   equivalence.
4. Add mutually exclusive `Aggregate`, `Selected`, and `Hidden` controls.
   Aggregate is initially visible with indexed layers hidden. Selected hides
   aggregate geometry and initializes its persistent indexed mask from the
   currently visible solver winding slots. The selected mask continues to
   rotate while Aggregate or Hidden is displayed.

## Diagnostics

1. Preserve the Ceres `raw_w` and calibrated `est_w` columns.
2. Keep the BP `est_w` selection unchanged in the globally calibrated
   reference coordinate. Do not run an independent candidate selection for
   the displayed `raw_w`.
3. Determine the unique gauge and orientation component from all admitted,
   candidate-bearing evidence used by the final `all` scorer, not from the
   independently selected raw calibration votes.
4. Inverse-map the exact final candidate as
   `rawLatent = globalSign * est_w + gaugeOffset`. Convert latent coordinate to
   integer `mapWinding` by subtracting the solved class offset for the
   reference H/V class, component phase sign, and selected phase. Validate the
   result is integral, then add the shared output winding offset used by
   generated solver OBJ layers and print this published index as `raw_w`.
5. Print `NA` when there is no final estimate, no unique contributing gauge,
   or no unique compatible orientation component. An off-grid inverse is an
   invariant error. Keep the lower-level raw per-gauge inference helper because
   gauge calibration itself still requires it.

## Validation

- Add Python unit tests for indexed artifact discovery/validation, exact
  half-step naming, aggregate/indexed mutual exclusion, union-range handling,
  and parity-preserving arbitrary-mask rotation with missing layers.
- Extend C++ winding benchmark tests to prove that `raw_w` is the inverse
  transform of the exact selected `est_w`, not a separately selected
  half-step, and that ambiguous multi-gauge values remain absent.
- Cover both global and component phase signs, both H/V offsets, negative
  relative windings, and a nonzero publication shift. The reported value must
  be an integer matching the generated OBJ winding index.
- Build Release `vc_fiber_trace_chunk`; run focused C++ and Python tests.
- Run one existing 1024 diagnostic command to regenerate real indexed
  reference artifacts and inspect the resulting filenames/table headings.

## Spec Update

- Replace the conflicting independent-reference viewer requirement with the
  indexed artifact and navigation invariants in `planning/specs.md`.
- Correct the reference benchmark table invariant in
  `volume-cartographer/planning/spec.md`.

## Docs Updates

- Document aggregate and indexed reference layers, controls, navigation
  grouping, and the BP/Ceres raw estimate columns in
  `volume-cartographer/docs/fiber_chunk_tracing.md`.

## Changelog Update

- Record indexed reference visualization and raw BP estimate reporting.
