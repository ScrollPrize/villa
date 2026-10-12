# Combined coordinate mapping: derivation and experiment

Status: complete; standard division selected for the new path as requested. `coordinate_backend`
defaults to `reference`; `combined` opts in for shared-anchor layout only.

## Why the existing path is expensive

Each endpoint/hypothesis goes from anchor turn to physical U, adds physical
arc distance, then goes back to endpoint turn. Generic autograd scatters many
gradients into ~151 shared winding widths and cumulative boundaries, repeatedly
for pair endpoints, three unary points and continuation endpoints. Prior B4096
profile: complete update22.392ms, backward15.589ms, indexed accumulation10.212ms
(nested within backward), fused pair forward/backward1.718ms. These cannot be
added. This experiment targets width-coordinate backward, not arbitrary fiber
state gathers or loss changes.

## Exact mathematical reformulation

Let widths w be positive, C=[0,cumsum(w)], anchor turn a, physical horizontal
distance d (zero for V fibers), s=clamp(floor(a)), and
t=clamp(searchsorted(C,U,right=True)-1). Then:

```
f = a-s
U = C[s] + f*w[s] + d
q = t + (U-C[t])/w[t]
```

For same bucket s=t, q=a+d/w[s]. In particular d=0 gives q=a.
Clamped bucket IDs preserve linear extrapolation beyond either endpoint; they
do not clamp U or q. General crossing retains the exact cumulative-width map.

Given upstream gU,gq, set h=gq/w[t], g=gU+h. Derivatives (away from seams):

```
da = g*w[s]
dd = g
dw[j] = gU * [j<s]
      + h * ([j<s]-[j<t])
      + g*f * [j=s]
      - h*(q-t) * [j=t]
```

The same-bucket width derivative simplifies to:

```
dw[j] = gU*[j<s] + (gU*f - gq*d/w[s]^2)*[j=s]
```

Critically, physical U still has its full prefix-width derivative. Continuation
losses need it; deleting it would change the optimization. Boundaries are a
derived input, not independent parameters: their derivative is incorporated
directly into dw, so the custom operator returns no separate boundary gradient.
The existing softplus(raw_width)+100 parameterization remains outside the
operator and receives dw normally. No variable is frozen or detached from its
proper derivative; no hypothesis or loss removed.

## Implementation and numerical contract

- Shared operator: `fiber_uv_coordinates.py`; CUDA helper:
  `fiber_uv_coordinates_cuda.py`. Solver uses this same helper for event,
  unary and continuation coordinates. Existing audit remains independent.
- Forward uses existing bucket/search conventions and algebraically cancels
  same-bucket roundtrips. CPU/double explicit backward is a small-test reference.
- CUDA FP32 backward: tiles of256 points ×16 widths reduce to one partial per
  tile/width; second fixed-order reduction produces width gradients. Each output
  has one writer; no floating-point atomics. This deliberately still evaluates
  width masks, rather than introducing a nondeterministic scatter/histogram.
- First-order autograd only. Triton imported lazily for CUDA FP32. CPU/double
  fallback materializes points×widths and is not intended for production-scale
  CPU/double solving. Production solver is CUDA FP32.
- Mathematics is unchanged, but FP32 cancellation/reduction order is different.
  Bitwise equality with old trajectories is NOT promised. Same-code repeat
  determinism, gradient tolerances and solve quality are tested separately.
- Reusing anchor calculations across three unary points is deferred until this
  isolated change is measured; current timings do not claim that optimization.

## Reproduction

Existing approved runner, exclusive GPU5, sequential same-GPU comparisons:

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_coord_matrix.json
```

Matrix children are the committed `run_uv_coord_{reference,combined}_*.json`.
B4096/B16384:105updates,5warmup excluded,100CUDA-event timing samples. Separate
26-update profiler per backend. Shared AFV constraint tensor cache,151CCW
hypotheses,seed0,full quadratic compiled pair loss,random sampled microbatches,
fresh initialization,LR.06,widthLR2. Profilers are not memory measurements.
Compare exact paired configs; earlier B4096 LR.03 timings are historical context.

Tests:9CPU coordinate cases passed (float32/64 value/gradient comparisons,
double gradcheck, U-only/q-only/mixed outputs, seams/extrapolation/zero distance,
noncontiguous/broadcast inputs). Initial broader regression47passed, superseded
by101passed after adding full-state/softplus-chain coverage;1CUDA test deselected.
Managed combined runs execute CUDA preflight then reset the original seed.
Independent plan/code review found no blocker; complete timings/results below
will determine whether the experiment is useful.

## Preliminary production-update measurements (superseded pending RN rerun)

Review of generated PTX found ordinary Triton `/` lowering to `div.full.f32`,
not explicit round-to-nearest. These v1 combined measurements used that form.
The implementation now supports `coordinate_division=precise` (`tl.div_rn`)
and `standard` (ordinary Triton `/`). User explicitly authorized returning to
the faster ordinary form after precise validation. Baseline measurements remain
valid; precise timings and quality are rerun as v2. Original v1combined data is
the ordinary form, not round-to-nearest. No other fast-math options are enabled.

H100 80GB, GPU UUID GPU-09395a0d-28a4-4b5a-cc3e-d3e024d05c36,
sequential same-GPU tests under lease0885c108-355d-4187-93a7-701484d973f2.
FP32; FMA disabled, but the above division caveat applies to v1.
Times exclude compile warmup, initial width-boundary calculation and diagnostics.

| Batch | Backend | Mean ms | p50 ms | p95 ms | Peak allocated GiB |
|---|---|---:|---:|---:|---:|
|4096|reference|23.415|23.374|23.498|3.985|
|4096|combined|14.390|14.388|14.490|3.974|
|16384|reference|60.273|60.271|60.453|7.429|
|16384|combined|21.492|21.487|21.600|7.392|

Speedups:1.627× at4096;2.804× at16384. About38.55%/64.34% less update time.
This is update throughput, not complete-run wall time (which includes audits).

| Batch | Backend | Coordinates ms | Pair forward ms | Other forward ms | Backward ms | Coalescing+optimizer ms |
|---|---|---:|---:|---:|---:|---:|
|4096|reference|1.247|0.534|4.404|15.872|1.359|
|4096|combined|1.321|0.517|4.397|6.906|1.248|
|16384|reference|0.975|1.110|3.187|52.803|2.197|
|16384|combined|1.150|1.118|3.482|13.538|2.204|

CUDA preflight passed for both combined timings. Production regression:
101passed,1CUDA case deselected (same CUDA helper executed under managed GPU).
Existing sparse-invariant warning only. No new dependencies installed.

## Hotspot confirmation (separate B4096 profiled update)

| Operation | Reference | Combined |
|---|---:|---:|
| Generic indexed accumulation |27calls /10.312ms|3calls /0.182ms|
| Combined coordinate backward |—|7calls /0.983ms|
| Of which tiled partial kernel |—|0.954ms|

The24removed indexed accumulations account for nearly all of the old hotspot.
The tiled-partial time is nested inside combined backward, not additional.
SparseAdam (~1.399ms in the combined profiled update), other backward arithmetic,
forward launch overhead and the pair loss now matter more. Profiled timings are
diagnostics only; the unprofiled table above is the performance comparison.

## Matched solve

Queued/started under exclusive lease3067633e-45dd-4daf-8b7b-036e1d1a4fbf,
same GPU5.4032 sampled updates, B16384, LR.06, widthLR2, fresh seed0 initialization.
Compare saved `data/uv_lr16k_0.06_v1` control. Same objective and sampling budget;
no reference initialization. Reproduction:

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_coord_quality.json
```

The control ran exclusively on another H100; use the same-GPU short trials above
for speed claims. This solve comparison primarily checks trajectory/quality.

Standard division completed4032updates exit0. Initial data/metrics exactly match
control. Final full-graph audit (6,606,100events), no reduced diagnostic sample:

| Mode | Reference L1 | Within ±2 % | Retained length % | Sign violations % | Recorded solver s |
|---|---:|---:|---:|---:|---:|
| Reference .06 control |11.660019|13.292521|65.485704|3.165387|414.581|
| Combined, standard division |11.659781|13.287325|65.490114|3.195105|213.333|
| Combined, precise division |11.660387|13.263416|65.485030|3.179760|217.483|

Both fit winding q10/q90=49/122; matched reference42/125; globaloffset+12.
The~.030percentage-point sign difference and~.005pp within2difference reflect
nonbitwise trajectories; not an exact final-assignment match or proof of better
quality. Reference metrics use retained matched original fibers, equal weighting,
so small population differences matter. Recorded solver time includes audits
but excludes final reference calculation/save and manager wait.

Precise validation uses `run_uv_coord_matrix_rn.json`, same approved command as
above replacing the config path. Its children include B4096/B16384 timing,
B4096profile and matched quality. GPU5 exclusive lease
874fda5a-7c4b-420d-b3d4-6f1bb7349f53. Short manager inventory wait between leases
was a60second driver-monitor refresh after release, not a compute bottleneck.

## Division-mode comparison

Only CUDA FP32 backward divisions are selected by `coordinate_division`.
Forward mapping and CPU/double reference do not change. Both modes retain
fixed-order reductions and disable FMA in the custom kernels. `standard` is
ordinary Triton division; `precise` explicitly requests round-to-nearest.

| Batch | Reference ms | Combined standard ms | Combined precise ms |
|---|---:|---:|---:|
|4096|23.415|14.390|14.121|
|16384|60.273|21.492|21.994|

At16k ordinary division is~2.3% faster than precise in these runs; at4k precise
was~1.9% faster. Do not claim universal division-speed superiority from this
single sequential trial. Both massively improve over reference at16k. Precise
p50/p95:14.113/14.209ms at4k,21.983/22.091ms at16k. Peak allocation unchanged
between division variants:3.974/7.392GiB. No precision change to stored state.

## Decision

Both full4032-update runs completed with comparable aggregate quality. Keep
ordinary `coordinate_division=standard` as the new path's default per user;
`precise` remains selectable. The overall `coordinate_backend=reference` default
is unchanged; opt in with `coordinate_backend=combined`. No freezing, lost
losses or hypothesis reduction. H100 tested only, one seed, not bitwise solve
equivalence. CPU/double small-test fallback; accelerated first derivatives only.

Remaining reuse of the three unary anchors is deferred: this isolated change
already removed the measured hotspot. Next user-authorized experiments are
larger batches and a real multi-GPU joint solve, documented separately.


