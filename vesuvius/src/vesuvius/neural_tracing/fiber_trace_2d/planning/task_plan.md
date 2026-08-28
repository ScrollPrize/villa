# Plan: retain the main BP constraint component

## Semantics

1. Expose a shared selector that reuses the winding solver's exact prepared
   factor graph, including hard continuity, merged pair measurements, signed
   perpendicular availability, quantized target multipliers, parallel cutoff,
   and fixed-Mixed exclusions. Find its effective-winding components and retain
   only the largest. Isolated pieces are one-piece components. Select
   deterministically by piece count, then prefer the component containing the
   crop-central piece, then the lowest original piece index.
2. Add a shared constraint-report subset transform. Remap retained source
   traces, pieces, per-trace piece indices, and constraint endpoints
   consistently. Preserve source arc provenance and extraction
   timing/candidate counters while recomputing subset constraint counters.
   Return old trace and piece indices so CLI-owned lines, original IDs,
   directions, and masks are subset explicitly rather than inferred from
   geometry.
3. Prepare the unfiltered topology to obtain its existing crop-central piece.
   When fixed orientations are requested, solve the orientation prepass, use
   its exact fixed classes for winding-component selection, subset, and repeat
   monotonically until the represented cohort is one effective-winding
   component. Reuse the final orientation prepass rather than solving it again.
   Without fixed orientations, select once using the empty fixed-class set.
   Rebuild topology after every subset and only then run final winding BP,
   reference/BP cross extraction, and output. Reference fibers must not
   participate in component selection. Print one compact cumulative filter
   summary per solver run.
4. Convert reference/reference and reference-to-BP diagnostic printers into
   formatters. Accumulate their complete strings during the run and emit them,
   in deterministic execution order, immediately before the
   `direction-ablation` command returns. Leave progress and ordinary solver
   reports streaming as they do now.
5. Treat a reference/BP observation with no valid final active H/V winding
   candidate as outside the benchmark population. Such constraints contribute
   neither offset-calibration intervals nor right/wrong/total counts. Keep the
   raw cross extraction unchanged so this is explicitly a post-solve benchmark
   population rule.

## Testing

- Unit-test largest effective-winding component retention, isolated pieces,
  fixed-Mixed exclusions, parallel-cutoff and missing-sign splits, merged
  pair-factor connectivity, deterministic equal-component selection,
  constraint/source remapping, provenance, and counter recomputation.
- Unit-test that final Mixed/Defect and invalid-winding endpoints are excluded
  from calibration and every class total.
- Add a CLI-level formatter/output-order test where practical; otherwise keep
  formatters independently testable and validate a real reference-fiber run.
- Build `vc_fiber_trace_chunk`, `test_fiberlet_crop_trace`, and
  `test_fiber_trace_winding_bp`; run the focused tests and `git diff --check`.

## Spec update

Document exact effective-winding-component filtering, fixed-orientation
prepass iteration, deterministic selection, reference independence, remapping
requirements, and end-of-command placement of both reference diagnostic
sections.

## Docs updates

Update the chunk tracing documentation with the main-component filter summary
and the deferred reference diagnostic output order.

## Changelog

Add one entry covering BP main-component retention and deferred reference
diagnostics.
