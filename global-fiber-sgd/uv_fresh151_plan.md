# Fresh 151-winding solve

User explicitly rejects warm expansion: restart the global U/V solve with 0–150
candidate indices. Preserve upstream normal/H/V/direction results and constraints.
Do not use any previous U/V state, reference surface evidence, or expanded logits.

## Implementation

- Generalize the existing joint Adam solver's count, initial parameters, priors
  and state metadata. Shared helper returns fresh zero offsets/shared anchors,
  radius-based Gaussian winding logits, 1% broken probability, and radius-based
  widths across the requested range. For K=100 preserve existing initialization.
- Reuse immutable geometry/constraint tensors (including radius), not their
  cached 100-candidate initialization when K differs. Scale the original radial
  initialization's 5..90 indices proportionally to (K-1)/99. This is only a prior,
  not anatomical/reference winding evidence. No repeated extraction required.
- Run original sequence independently for CCW/CW: fresh joint Adam 10,000 steps;
  sign-charge16 refinement 4,000 steps from this newly created state; then current
  sign-charge4 categorical update for 10 full sweeps. Only within-experiment
  stage handoffs load checkpoints. No checkpoint from any previous experiment.
- Keep original losses, batch4096, learning rates and stage settings. Reserve
  48GiB per GPU for the larger joint solver, then12GiB/batch2048 for categorical
  sweeps. Use separate GPUs through existing manager/approved runner.
- Preserve earlier results. New outputs under data/uv_fresh151_v1. Do not claim
  zero errors guarantees correct winding placement; report retention too.

## Spec update

Configurable winding_count applies to fresh joint initialization and optimization,
not only categorical continuation. Cache remains physical evidence; initialization
is count-dependent and recorded in each solve's configuration/summary.

## Tests / validation

Regression tests for old100 initialization identity, new151 shape/support,
zero offsets, 1% broken probability, radial initialization reaching added range,
and geometry/width gradients/updates. Run existing solver/renderer regressions.
Independent plan/code review. Monitor memory and stage progress. Full persisted
audit of final151 results. Render using established1000px/2px opaque TIFF/SVG maps
and validated reference projection cache (diagnostic only).

## Docs updates

Archive previous task log, record current task/plan/status and stage commands,
runtime/results/limitations in task_log and fresh results document. Changelog
notes corrected fresh initialization. No commit requested.
