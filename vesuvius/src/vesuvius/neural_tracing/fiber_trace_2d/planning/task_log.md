# Task log: promote the best 1024 winding tune

## Inputs

- Tuning artifact: `data/workdir3/crop_traces.zarr` (1024 crop)
- Normal manifest: `data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json`
- Reference directory: `data/test_datasets/2026-09-01_fiber_stack2`
- Reference tag: `hendrik_crop1`
- Complete-reference implementation: `f89346f33`
- Starting winding weights: `0.5,0,1,2,1`
- Starting sign weights: `0.5,1`
- Quality fraction: `0.25`
- Piece length: `512` base voxels

## Plan review

- Use the existing zero-aware local search so zero coordinates can become
  positive and positive coordinates can move to zero, `/2`, or `*2`.
- Run one process only. The command reuses pre-winding constraint extraction and
  topology across all candidate weight scenarios.
- Rank with the existing deterministic objective: convergence, exact reference
  estimates, missing/wrong references, correct items, evaluated items, wrong
  items, residual, then tuple.

## Deviations

- An independent subagent review was not run because the active orchestration
  instruction prohibits spawning subagents unless the user explicitly requests
  delegation. The plan was reviewed locally against the current specification,
  previous tuning log, and CLI validation rules.

## 1024 result

The initial local search was run against the established 1024 trace artifact.
It reached a local optimum after 129 scenarios and six accepted moves:

- Start: winding `0.5,0,1,2,1`, sign `0.5,1`, exact references `13/26`,
  constraint accuracy `8016/10713` (`74.825%`).
- Selected: winding `0,0,0.5,4,1`, sign `1,0.5`, exact references `16/26`,
  constraint accuracy `8009/10703` (`74.829%`).

The selected 1024 tuple is promoted at the user's request for direct checking.
The fixed reference denominator includes one missing estimate because that
reference has no usable cross constraint in this crop.

## Implementation

- Shared class defaults: `0,0,0.5,4,1`.
- Shared sign defaults: `1,0.5`.
- CLI options continue to override both tuples explicitly.
- Updated CLI help, focused regression expectations, user documentation,
  specification, and changelog.

## Deviation

Before the promotion request, a 2048 baseline and part of a 2048 local-search
neighborhood were run after misinterpreting the intended workload. The user
stopped that work; no 2048 candidate was promoted, and this change contains
only the completed 1024 optimum.

## Validation

- Built optimized targets `vc_fiber_trace_chunk` and
  `test_fiber_trace_winding_bp` with `cmake --build volume-cartographer/build
  --target vc_fiber_trace_chunk test_fiber_trace_winding_bp -j 16`.
- `volume-cartographer/build/bin/test_fiber_trace_winding_bp`: 70 cases passed.
- `volume-cartographer/build/bin/vc_fiber_trace_chunk --help` reports winding
  defaults `[0,0,0.5,4,1]` and sign defaults `[1,0.5]`.
- `git diff --check` passed.
