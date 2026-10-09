# H-direction alignment (new stage3)

## Task

Pause global U/V assignment. Add whole-H-fiber forward/reverse sign alignment
after accepted normal and H/V stages, before global assignment. Use existing
parallel constraints and weak umbilicus circulation prior. Only global circulation
sign is a user choice; global assignment will later test both CW/CCW.

## Implementation

- Exclude earlier rejected fibers and all V fibers. H/V labels and normal signs
  immutable. No new rejection variables in this direction-only pass; report
  remaining inconsistencies instead of silently deleting fibers.
- Preserve original AFV point order as each fiber's tangent gauge. Estimate signed
  tangent at each parallel event arc with centered±16basevx secant (clip at ends).
  Reuse/generalize existing tangent_at to a batched helper; no new duplicate sampler.
  Dot of two tangents gives the signed pair observation. Degenerate secants counted
  and excluded from pair loss, not replaced with made-up directions.
- Arc-weighted local circulation prior: sample original segments at<=16basevx,
  compare forward tangent with normalize(axis_tangent × outward_radial), where
  axis points along increasing native Z. Canonical positive circulation CCW as
  seen from positive-axis end looking toward negative. Frame-check catalogue
  umbilicus with shared validator. Clockwise negates prior/signs globally.
- One scalar sign logit per retained H fiber. Pair objective is shared normal-sign
  alignment term (no broken gates); weak prior .001, optional late binary entropy
  .001, Adam CUDA lr.05/10000steps. Deterministic seed and GPU-manager ownership.
- Canonical CCW result plus exactly negated CW equivalent (no second optimization
  required); identical pair costs. Save all source IDs, eligible mask, H signs,
  probabilities/logits and connected-component IDs. Non-H/rejected sign=0 and
  explicit mask; do not interpret that as an assignment. Tangents at constraint
  points have no trainable parameters.
- Source/config/code fingerprints, cached fixed arrays and immutable results.
  No integration into U/V, no V-direction alignment, no fiber cutting in this pass.

## Validation

Shared tangent scalar/batch equivalence, polyline reversal covariance, clockwise
gauge symmetry, pair-loss extraction equivalence to existing normal stage,
synthetic connected directions with reversed source gauges, invalid tangent and
prior handling, frozen exclusions and saved-array/source audit.
Run approved pytest then one full managed GPU experiment, independently recompute
hard disagreement and report isolated/components/ambiguous/sign changes vs weak
umbilicus-only baseline. No anatomical correctness claim.

## Spec update

New third pass changes one whole-fiber tangent gauge only. Retain normal/HV
outputs, keep U/V work explicitly paused pending more user constraints.

## Docs updates / changelog

Document h_direction.md/results/config and exact approved runner. Replace active
task/task_log/status, archive U/V task notes separately, durable changelog entry.
