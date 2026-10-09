# Whole-fiber H/V stage

## Recommended config and inspection

`run_hv_recommended.json` runs only degree_mix/penalty0.1/entropy0.01, with the same
10000steps/lr0.05/seed0/prior0.001 and frozen stage1 input as the sweep winner.
It writes to a separate output root; the existing winner has not been rerun.

Visualization: `data/hv_stage_v1/visualization_1000/` (44.4MiB).
Import these separate layers, with `fibers.mtl` beside the OBJ files:

- `h_fibers.obj` + `h_normals.ply`:400long H fibers, blue.
- `v_fibers.obj` + `v_normals.ply`:400long V fibers, green.
- `broken_fibers.obj` + `broken_normals.ply`:200long rejected fibers, red.
- `umbilicus.ply`:thick orange axis.

The broken sample includes66normal-stage and134new H/V-stage rejections. This is
deliberately stratified and length-biased, not population proportions. Whole curves
are retained. `index.json` maps each original ID, final H/V/broken probabilities,
rejection stage, normal-stage probabilities, and independent layer vertex ranges.
Arrows point outward; stored normals inward, length640/radius20basevoxels. Normal
directions on rejected fibers remain the model output and are not validated.
The combined `fibers.obj`/`signed_normals.ply` is optional; do not import both it
and all separate layers unless you want duplicate geometry.

Reproduce with the approved crop-benchmark runner and `run_hv_visualization.json`.
16Python tests pass; all separate OBJ edges and PLY indices validated programmatically.

Completed comparison: [hv_stage_results.md](hv_stage_results.md).
Recommended candidate for inspection: degree_mix, penalty0.1, entropy0.01.
It rejects20838additional fibers (4.0835%of stage1retained),23240total (4.5329%of
original), retaining251658H/237800V. Preserves95.2522%of active-input arc and
90.9436%of active events, with0.05826%retained-event contradictions. No assignment
has max ternary probability<0.9. This confidence is optimizer certainty, not
calibrated anatomical correctness. Full result stored in
`data/hv_stage_v1/adam_sweep/solution_degree_mix_b0.1_e0.01.npz`.

Late entropy removed ambiguity with nearly unchanged consistency/deletion rates.
Degree weighting removes slightly more fibers but less total length and fewer
constraints than uniform penalty0.1. Penalty1 retains more fibers but leaves
1.21–1.59%contradictions, versus0.058–0.085%at penalty0.1. User's informal3–10%
expected rejection was noted after low-penalty runs; not an optimization target.
All8saved results passed ID/frozen-mask/probability/independent violation-count
checks. One managed shared H100,88.85–89.68s/variant,662.4MiBpeak tensors; GPU
grant released successfully.143native and15Python tests pass.

Stage1 normal-sign solve is reused, not rerun: penalty0.1 inward solution,
2402excluded fibers (0.4685%). They are fixed broken for stage2. The510k remaining
fibers are never cut: if a fiber changes family mid-curve, conflicting relations
can make rejecting the whole fiber favorable. Rejection is model inconsistency,
not proven anatomical crossover.

## Cached input

Native `afv-constraints` supports `axial_prior_output`, optionally
`axial_prior_only:true` for backfilling existing constraints without repeating
search. The same code is also invoked before ordinary constraint extraction.
The additive sidecar stores one scalar per source fiber, ordered IDs, AFV hash/UUID,
umbilicus hash/frame and algorithm parameters. Original AFV/SQLite remain untouched.

Compute arc-weighted mean of `2*(t dot u)^2-1`, where t is the original polyline
segment tangent and u is the local umbilicus-axis tangent, NOT radial direction.
Midpoint quadrature uses intervals <=16basevoxels. u uses dense frame-checked axis
centers at z±8 (nearest integer z); intervals clamp at volume endpoints. Geometry
zero-length segments contribute no weight; entirely degenerate fibers fail.
The underlying umbilicus XY interpolation is clamped outside its annotation span.
Positive favors V, negative H; no separate tangent-sign ambiguity.

512698fibers:31.26s native process wall including AFV loading; feature computation
0.084s, JSON-inclusive0.151s. RelWithDebInfo/GCC13; one run, not a performance study.
143native tests passed.42.2096% fibers have positive axial scores.

## Optimization

510296active fibers,7277943active events (5978037parallel,1299906perpendicular);
21147active fibers have no remaining events and are determined only by weak prior.
Stage1-deleted events and fibers contribute no trainable losses.

Two logits per active fiber: h (H/V) and b (new rejection). Let p=sigmoid(h),
q=sigmoid(b). The ternary probabilities are `((1-q)(1-p),(1-q)p,q)` for H,V,broken.
Hard labels use argmax with deterministic H,V,broken tie order. Labels in saved
arrays:0=H,1=V,2=broken. Inherited broken entries are exactly[0,0,1].

Pair loss averages expected incompatible assignments over the FIXED active-event
count: `(1-qi)(1-qj)*(1-r*(2pi-1)*(2pj-1))/2`; r=+1parallel,-1perpendicular.
Each event has equal weight; repeated encounters are separate evidence. Winding
distance and signed normal orientation are NOT used to infer family labels.
Prior is .001mean((1-q)*(1-a*(2p-1))/2). Rejection mean(c*q), where c is either
penalty or penalty*(.5+.5degree/meanDegree). Degree is computed once after stage1,
including isolated fibers in the mean. Positive ternary entropy is minimized:
zero first50%, linear ramp to full at80%, then constant.

Matrix: cost uniform/degree_mix × penalty0.1/1 × entropy0/0.01.10000Adamsteps,
lr0.05,seed0,deterministic Torch, one manager-allocated shared H100 with4GiB quota.
Reported runtime excludes queue/preparation, includes diagnostic readouts. Source
hashes and frozen config are recorded. Fixed arrays prepared once(37.08s).

## Commands

Build: `bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build`

Use the approved runner with each config, in this order (existing completed
outputs are not overwritten):

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_axial_features.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_hv_stage.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_hv_report.json
```

Outputs: `data/hv_stage_v1/axial_features.json`, `tensors.npz`, and
`adam_sweep/{config,summary,history_*,solution_*}`. Existing inward normal arrays
are unchanged. Results report is `hv_stage_results.md`.

Tests: approved pytest command, `--noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src`
with `vesuvius/tests/test_fiber_hv_stage.py`, normal-sign and visualization tests.
Tests cover scalar formula, gradients, global family swap, synthetic inconsistent
fiber rejection, frozen exclusions/degree, empty metrics and hardening.

## Limits

No ground-truth H/V labels in Qual AFV; consistency and deletion tradeoffs only.
Weak axis prior may misidentify strongly deformed components; no global optimum
claim. Fixed10000steps do not establish convergence. The additive feature cache is
a sidecar rather than a schema migration. Old constraint databases lacked source
content hashes; current AFV hash and per-fiber lengths are checked against source
provenance, alongside stage1's immutable database fingerprint.
