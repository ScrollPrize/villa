# Plan: report central versus non-central BP states

## Implementation

- Capture the same source-piece cohort used for the prior comparison before
  component filtering: central means original `source_piece == 1`. Carry this
  bit through every subset mapping so later piece renumbering cannot move it.
- Count final H, V, active H/V, Defect, total pieces, and Defect percentage from
  post-projection orientation plus winding validity. Invalid winding is Defect.
- Format the result as a compact aligned table and defer it with the reference
  diagnostics so it appears immediately before that BP execution's reference
  benchmark. Print it directly when no reference benchmark was requested.

## Tests

- Extract the aggregation/formatting into a deterministic helper.
- Add focused coverage for both cohorts, invalid projected states, zero-sized
  cohorts, and exact row/total invariants.
- Build `vc_fiber_trace_chunk` and the focused test target with 32 jobs.
- Rerun the established 1024 diagnostic and verify output placement and counts.

## Spec update

- Specify the central/non-central summary and its final-state semantics.

## Docs updates

- Document the table, central cohort definition, and tuning purpose.

## Changelog

- Record the new BP state-distribution diagnostic.
