# Task log: retain complete reference fibers in diagnostics

## Starting point

- `updateReferenceFiberArtifact` clips every selected dense annotation line to
  the stored trace artifact's half-open crop.
- The clipped runs are used by both `<base>_reference.obj` and all reference
  constraint/benchmark diagnostics.
- On the 2026-09-01 stack this retained 24 of 26 selected fibers; annotations
  `...000010.json` and `...000026.json` were outside the 1024 crop, while two
  other annotations appeared as short boundary fragments.

## Plan review

- The complete JSON lines are already validated and deterministically sorted by
  `loadTaggedVc3dFiberJsonDirectory`; no parser or ordering change is needed.
- Reference geometry is diagnostic-only. Keeping it complete does not expand
  or alter the traced/BP fiber population.
- The existing constraint extractor and full-volume normal sampler can consume
  points outside the stored trace crop.

## Deviations

- An independent subagent review was not run because the active orchestration
  instruction prohibits spawning subagents unless the user explicitly requests
  delegation. The plan was reviewed locally against `AGENTS.md`, `specs.md`,
  the reference CLI flow, and the current documentation.

## Implementation

- `updateReferenceFiberArtifact` now emits one OBJ line and one diagnostic trace
  from every selected JSON fiber's complete dense line.
- Removed the crop bounds from that helper's interface and renamed its status
  field from `retained_runs` to `retained_fibers`.
- Kept the trace artifact crop bounds for BP topology and all crop-produced
  fibers; only diagnostic reference geometry changed.

## Validation

- Release build completed for `vc_fiber_trace_chunk`, `test_fiber_json`, and
  `test_fiber_trace_winding_bp`.
- `test_fiber_json`: 2 cases passed.
- `test_fiber_trace_winding_bp`: 70 cases passed.
- The real 2026-09-01 stack retained all 26 selected fibers and all 7,797 dense
  points. The previous crop-clipped path retained only 24 sources.
- With reference splitting disabled, all 25 adjacent filename pairs produced a
  dominant perpendicular constraint. After the standard scale-first factor
  `0.822`, 23 quantized to the expected `0.5` step. Pair `3.5->4.0` measured
  `1.272` after scaling and quantized to `1.5`; pair `6.5->7.0` measured
  `1.004` and crossed the same boundary by only `0.004`.
- The final `12.0->12.5` pair is perpendicular, with raw step `0.601`, scaled
  step `0.494`, and canonical step `0.5`. The preceding `11.5->12.5`
  skip-one relation is parallel with raw step `1.094`, consistent with the
  same orientation one winding apart rather than the same winding.

## Execution note

- The first complete-line validation used `direction-ablation`, whose existing
  control flow continued into an unnecessary BP solve after reference
  extraction. No BP result was used for the adjacent-pair evaluation, and no
  further optimizer run was started after the user correction.
