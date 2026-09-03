# Plan

1. Extract reusable reference-run preparation and directed replay aggregation
   into the Fiber tracer core. Clip tagged reference polylines to the half-open
   crop, preserve every contiguous run, and generate forward and reversed
   directed cases without duplicating replay or threshold logic.
2. Add a `reference-replay-benchmark` mode to `vc_fiber_trace_chunk`. Reuse the
   crop graph materialization and `traceFiberletGraphReplay`, expose the crop,
   reference selection, base-voxel size, replay search settings, and anisotropic
   normal threshold, and default worker counts to host CPUs.
3. Add opt-in shared replay controls to require the initial seed in the first
   ordinary seed window and to stop at the first failure. A valid seed's
   threshold-checked endpoint offset is credited; failure to find that seed is
   a zero-length result. Preserve ordinary restart behavior by default.
4. Define deterministic aggregation: each directed run stops at its first
   failure; full success contributes the complete run length. Report direction
   counts, failures by reason, total/full/traced base length, mean credited
   length over all directions, mean first-failure length over failed directions,
   binary completion rate, millimeters, and length-weighted `100*traced/full`
   success.
5. Emit a versioned JSON summary. Add focused core and CLI tests for clipping,
   maximum-face closure, exit/re-entry, bidirectional accounting, reversed
   failure arcs, no endpoint seed, non-distance failures, physical conversion,
   empty inputs, and threshold metadata.
6. Add a repository benchmark runner that consumes JSON rather than parsing
   human output and records each invocation as Markdown
   with Git revision, exact command, effective configuration, artifact
   identities, clean/dirty state and diff checksum, host/build metadata, cache
   state, repetition, timing, raw output, and parsed results. It must run both the new
   endpoint benchmark and existing oracle-pruning benchmark without rerunning
   prediction, Fiberlet generation, or crop tracing.
7. Add a separate benchmark-results Markdown table linking the individual run
   records. Run both benchmarks on the frozen PHercParis4 1024 crop and record
   their measured results.

## Spec Update

Specify reference clipping, directed endpoint cases, first-failure length,
anisotropic threshold reuse, aggregate success denominator, required physical
voxel size, deterministic ordering, machine-readable results, and Markdown run
provenance. Mark these external-data evaluations as manual scientific
benchmarks rather than CI performance gates.

## Docs Update

Document the new CLI mode and benchmark-record workflow in the Fiberlet crop
tracing documentation. Keep the benchmark result index separate from the
reproduction guide and per-run records.

## Changelog

Record the reference-endpoint replay benchmark and reproducible Markdown
benchmark recording workflow.

## Validation

Build Release targets, run focused unit/CLI tests, validate Markdown and shell
commands, execute both benchmarks against the frozen PHercParis4 1024-crop
artifacts, and report exact
commands, inputs, build type, timing, and results.
