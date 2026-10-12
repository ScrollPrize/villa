# Coordinate-gradient reformulation

Implement an opt-in combined anchor-q -> physical-U -> endpoint-q operator.
Preserve the objective, trainable widths/state, extrapolation, microbatch updates,
sampling and existing reference path. No loss simplifications or frozen widths.

Use explicit derivatives: U=C[s]+(a-s)w[s]+d; q'=t+(U-C[t])/w[t].
Combine cancelling boundary derivatives before reduction. Return physical U for
continuations. For d=0, q'=a; for s=t, q'=a+d/w[s]. Implement deterministic
block-local width-gradient reductions, then a fixed reduction of partials, with
no global atomic scatter. CPU reference fallback and optional CUDA acceleration.
Factor shared implementation rather than duplicate solver behavior.

## Validation
Independent plan/code review. Float64 gradcheck and dense-autograd comparisons
for q and width gradients, mixed U/q losses, negative/positive distances, V,
same/cross buckets, extrapolation and boundary conventions; FP32 tolerance tests.
Exclusive approved-runner benchmarks B4096/16384, baseline and optimized; full
update timing p50/p95, memory and profiler. Matched 4032-update B16384 LR.06
quality run (seed0, sampled ~10 pass-equivalents), compare retained length,
offset-corrected reference L1/within2, sign errors and winding ranges. Preserve
baseline outputs. Do not promote unless speed and quality justify it.

## Spec update
No objective/default change. FP32 reduction order can change trajectories;
mathematical equivalence, not bitwise identity, is the contract. Disclose any
unsupported path or uncompleted validation explicitly.

Follow-up user authorization: validate precise division, then use ordinary
faster division if quality is preserved. Keep both modes explicitly selectable
and document their comparison. This is the only authorized arithmetic relaxation.

## Docs updates
Persist derivation, exact configs/commands, tests, profiles and final results in
uv_coordinate_results.md and README; update current task/status/log/changelog.
