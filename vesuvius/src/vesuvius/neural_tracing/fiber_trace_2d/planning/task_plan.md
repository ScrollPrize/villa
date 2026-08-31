# Plan: normal-alignment progress output

## Implementation

- Add a shared, optional progress callback to binary pairwise BP. Emit one
  event after every completed message iteration and a terminal event, outside
  numerical loops except for the existing serial iteration boundary. Catch a
  callback exception inside the OpenMP region, stop at the synchronized
  boundary, and rethrow it after leaving the region. Exclude callback time from
  BP phase timing attribution.
- Add a normal-alignment progress event with explicit sampling, factor-build,
  component-build, message-passing, and finalize phases. Thread the callback
  through lattice sampling/alignment and map binary-BP iterations into the
  message phase.
- Emit bounded factor/component progress from the existing deterministic loops.
  Sampling is one opaque batch read, so report its start and completion without
  inventing intermediate completion counts. Factors count lattice sites
  scanned. Component preparation counts normalization items, factor adjacency
  insertions, and visited nodes. Finalization counts retained nodes. Core
  preparation/finalization callbacks occur only at phase boundaries and every
  65,536 work items; BP emits at most one event per configured iteration.
- Add a rate-limited CLI formatter. Print phase, completed/total, percent,
  elapsed time, ETA when derivable, and BP residual. ETAs are explicitly
  phase-local; the message ETA is labeled as time to the configured iteration
  limit rather than expected convergence. Sampling reports no intermediate
  ETA. Always print phase transitions, phase completion, and one success-only
  terminal completion event.
- Keep every callback optional so all existing callers retain behavior and
  avoid reporting overhead beyond an empty callback check.

## Tests

- Extend binary-BP tests above the real OpenMP factor threshold to verify
  serialized monotonic callbacks, early/message-limit terminal state, safe
  callback exception propagation, and exact numerical equality excluding
  timing fields.
- Extend normal-alignment tests to verify ordered phase transitions, terminal
  completion, exact work totals on a lattice with holes/multiple components,
  bounded event count, and exact output equivalence with and without the
  callback. Preserve factor and component traversal order.
- Build `vc_fiber_trace_chunk` and the focused BP/alignment tests, run them,
  and run `git diff --check`.

## Spec update

- Document the optional, observational normal-alignment progress contract and
  its phase/completion semantics.

## Docs updates

- Document that the crop CLI now reports progress during normal-volume
  sampling, factor/component preparation, and normal-sign BP.

## Changelog

- Record live normal-alignment progress with message residual and ETA.
