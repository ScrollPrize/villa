# Plan

1. Generalize reference replay so Fiberlet, direct greedy, and Lasagna use the
   same directed cases, forward-reference matcher, anisotropic threshold,
   failure accounting, completion requirement, and distance-per-failure
   summary. Preserve each backend's native seed and reset increment, but record
   these differences explicitly.
2. Add an explicit replay tracer selection to `vc_fiber_trace_chunk`. Keep the
   current Fiberlet backend as the default; require a Fiber prediction manifest
   only for direct greedy replay. Lasagna replay uses the normal manifest.
3. Reuse the existing direct greedy `traceFiberReplay` implementation with a
   synthetic two-control-point input for each clipped directed run. Extract the
   existing Lasagna normal-transport step into a public helper, port its current
   caller to that helper, and drive it through the same reset-capable replay
   implementation. Lasagna starts exactly at the reference endpoint tangent,
   keeps the previous direction across invalid normal samples, and is reported
   as a normal-transport control.
4. Version benchmark JSON to version 3 so it records tracer identity, common
   evaluator settings, actual evaluation/reset spacing, backend-specific
   effective settings, and only the inputs consumed by that backend. Keep
   headline metrics identical across backends and retain version-2 Fiberlet
   records as historical artifacts.
5. Add focused synthetic tests for direct-result adapter semantics, constant
   and invalid-normal Lasagna replay, antipodal normal transport, and JSON
   tracer provenance. Re-run the existing greedy replay and Lasagna optimizer
   tests, then smoke-test CLI validation and both direct backends on the frozen
   external dataset.
6. Build Release targets, run focused tests, then run and record the greedy and
   Lasagna reference benchmarks on the frozen PHercParis4 1024 crop if the
   required local prediction input is discoverable.

The crop-pruning command and its benchmark schema are intentionally unchanged.

## Spec Update

Specify the three reference replay backends, common metric/evaluation policy,
backend inputs, and the fact that crop pruning remains Fiberlet-only.

## Docs Update

Document commands and effective settings for Fiberlet, greedy, and Lasagna
reference replay. Add reproducible run records and index rows for completed
external-data runs.

## Changelog

Record direct greedy and Lasagna reference replay support.

## Validation

Run the focused C++ tests and the frozen 1024-crop replay command for both new
backends. Record wall/CPU time, source revision, artifact identities, failure
count, mean distance per failure, and percentage.
