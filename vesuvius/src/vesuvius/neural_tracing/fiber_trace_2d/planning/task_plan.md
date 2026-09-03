# Plan

1. Reuse the existing deterministic replay restart path after failures while
   retaining the endpoint-window requirement for the first seed.
2. Replace first-failure-only benchmark outcomes with all failure events and
   actual seeded replay intervals over the complete directional reference run.
   Credited length is the union of `[first match search begin, segment end]`
   intervals for seeded segments; missing-seed gaps and unseeded tails receive
   no credit. A failed span is the corresponding seeded interval ending at a
   failure; an unseeded failure has length zero.
3. Report total failures, failure density per directed millimeter, directions
   with and without failures, failure reasons, credited traced length, mean
   failed-span length, and mean seeded-span length in base voxels and
   millimeters. Exclude a successful terminal suffix from mean failed-span
   length but include it in mean seeded-span length. Preserve separate whole-run
   evaluation-complete and failure-free semantics in JSON.
4. Add focused aggregation and replay regression tests proving that ordered
   multiple failures in one direction are retained, reverse source arcs are
   correct, missing gaps are not credited, terminal suffixes are handled, and
   replay reaches the reference end.
5. Emit incompatible benchmark schema version 2. Serialize every failure's
   reason, directional and source-oriented arc/fraction, reference/evaluator
   points, and anisotropic threshold measurement.
6. Update the spec, CLI documentation, benchmark documentation, task log, and
   changelog. Build and run focused Release tests, then rerun and record the
   endpoint benchmark on the frozen PHercParis4 1024 crop.

## Spec Update

Replace first-failure termination with deterministic continuation through the
entire directional run. Specify all-failure serialization and failure-free span
metrics without changing ordinary replay behavior.

## Docs Update

Document continuation, the distinction between a completed whole run and a
failure-free direction, and the revised result fields. Supersede the prior
first-failure benchmark row with a new reproducible run record.

## Changelog

Record whole-run reference replay and multi-failure benchmark accounting.

## Validation

Build the Release CLI and focused tests; run unit tests for multiple failures,
aggregation, and JSON; execute the 1024-crop endpoint benchmark and record its
command, revision, artifact identities, timing, and results.

Create the canonical external-data record only after committing the measured
implementation. Keep the historical first-failure record unchanged and mark it
as superseded in the result index.
