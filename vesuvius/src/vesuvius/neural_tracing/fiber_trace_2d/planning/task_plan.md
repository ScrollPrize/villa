# Training-feedback fiber refinement

## Approved continuation launch
Resume raw current.pt from 5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436
(update 11800), preserve optimizer/LR 0.01, data, EMA and batch settings; enable
refinement and add existing per-scroll normal manifests. Use a new run name and
the approved production-start launcher with automatic manager admission. Check
startup for errors and record the launch/log. No protocol or semantic changes.

1. Inspect queue transport, coordinate transforms, native normal sampler and
   snapshot/cooperative lifetime. Disabled behavior stays unchanged.
2. Validated opt-in settings and shared versioned per-component geometry store;
   immutable reference, mutable curves, preserved eligibility/source identity.
3. Fixed-budget semi-implicit solve: uniform segment quadrature, baseline tether,
   small displacement smoothness, current tangent-plane projection, tapered
   ellipse step cap, equal-distance resampling after each step. Existing native
   normal interpolation and geometric transforms must be reused.
4. Carry lightweight geometry in the existing queued sample payload; update after
   actual model forwards (raw train/test only, never EMA observations). Existing
   prefetched batches remain immutable; reject stale conflicting proposals.
5. Save all train/test geometry beside checkpoints, restore matching geometry on
   resume and keep it across GPU cohort changes. Never overwrite input annotations.
6. CPU geometry/integration/restore regression tests and small runtime measurements
   through approved runner; manager allocation for any CUDA checks. No long run.

## Spec update
Add the opt-in contract to specs.md including moving test targets and bounded
lookahead. No manager protocol changes.

## Docs updates
Add training_refinement.md with settings, solver and snapshot inspection/restore.
Brief task log and changelog; independent plan/integration review.

## Validation
Approved cooperative test caller; numerical shape, quadrature, clamp/taper,
resampling, augmented coordinate roundtrip, persistence/stale proposal tests.
Report measured overhead and limitations rather than promise a percentage.
