# Plan: retain complete reference fibers in diagnostics

## Implementation

1. Remove crop clipping from the reference artifact/diagnostic construction.
2. Preserve deterministic filename ordering and one source ID per selected
   JSON fiber; export and diagnose each complete dense line exactly once.
3. Keep crop bounds and all traced/BP fiber handling unchanged.
4. Adjust status output and error text so they describe complete selected
   fibers rather than retained crop runs.

## Validation

1. Build the Release `vc_fiber_trace_chunk` target.
2. Run focused existing fiber JSON and winding tests.
3. Run the real 1024 diagnostic against the 2026-09-01 reference stack and
   verify all 26 selected fibers are retained, including the two annotations
   outside the trace crop.
4. Summarize adjacent-pair dominant relation and measured winding step,
   including the final annotations.
5. Run `git diff --check`.

## Spec update

Update `planning/specs.md` to require complete selected reference geometry for
reference visualization and diagnostics.

## Docs update

Update `volume-cartographer/docs/fiber_chunk_tracing.md` to remove the clipping
contract and document that reference annotations may extend outside the trace
artifact crop.

## Changelog

Record that reference diagnostics now retain complete tagged VC3D fibers.
