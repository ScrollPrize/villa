# Current reviewed-tracing baseline — 2026-10-02

Preferred policy: **full-prefix forward/backward consistency every512 base voxels**, including terminal endpoints, **no direction-history initialization and no short-sample filtering**. Reverse acceptance ellipse:20normal/40tangential base voxels. Beam8, lookahead2, step4trace voxels, normal/tangent smoothness0.1/10, embedding disabled.

Snapshot: best **raw u40500** (5,184,000training samples), training run `5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_sideband_reviewed32_128_sd1_16x8_ema_20261001_203634`; selected by test/loss_total.

Frozen checkpoint: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/halfsigma_sideband_best_raw_20261002/frozen/live_inference/_models/9d3bb1c0cd5b0ee58d16d9b2bdcec931faec675421e6e2c707f7fd10ce059f7e/snapshot.pt` (directory name is checkpoint SHA256).

| Policy, all91pieces /182directions | Ordinary error restarts | Filtered stops | Retained distance |
|---|---:|---:|---:|
| No reverse filter |16|0|100%|
| Full-prefix512, no history — selected |5|66|94.7436%|

Selection prioritizes low error count and few filtered stops, not retained distance. All samples, including those shorter than1000base voxels, remain included. This is a tuned-cohort result, not an independent holdout claim.

Workload and full output: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/reverse_prefix_40500_v1/prefix512_terminal/`.

Native replay flags: `--replay-reverse-interval-base 512 --replay-reverse-all-endpoints --replay-reverse-to-start`; history defaults to0. The existing runner is `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, action `reviewed_reverse_prefix` runs the documented comparison matrix.

The reverse verifier itself does not use GT. Reference replay does use GT for error detection, terminal scheduling and reseeding after filtered stops. A GT-free dense-crop application must cut/reseed generated traces without reference assistance; its results must be measured separately, not assumed equivalent.

Detailed experiments: [prefix sweep](reviewed_reverse_prefix_40500_20261002.md), [history sweep](reviewed_reverse_history_40500_20261002.md).

## Current BP crop benchmark baseline — selected 2026-10-04

Use **post-fw/bw presence-ranked whole-piece deduplication**, ellipse10normal/
20tangential base voxels,0%threshold (any positive sampled overlap),45degree
unoriented direction gate, midpoint samples at most4base voxels apart. Reverse
filter splits are independent pieces. Higher arc-weighted mean presence wins;
whole losing pieces are removed, not clipped. No GT is used in this selection.

Then legacy-fine min256 splitting, median±64/blend1, partitioned length cost100
per128voxels, boundary10, magnitude weights **.2,.2,0,.0375,0** (perpendicular
near/far, parallel same/one/two), hard signs both, scale.822/phase.5,500messages.
Same rawu40500 beam8/full-prefix512/**isotropic20** source tracing as before.

| Non-oracle accuracy | Observed shift | Constraint-disable score | Oracle accuracy |
|---:|---:|---:|---:|
|65.3846%|91.6667%|−55.1216%|88.4615%|

Selected by user, accuracy first then shift then disable score.17/26correct,
missing4/12; shift covers24references.501retained traces/774BPpieces/8420constraints.
Exact repeated winner; tuned on1024crop, not a held-out claim or global runtime
default change. Removal is worse than the former no-dedup baseline below.

Selected trial `presence-10x20-f0-2d33708bd653` under
`/ephemeral/hendrik/fiber_crop_bench/20261003_iso20_overlap_costs_v1/`.
Exact `-ind.config.json`, `-oracle.config.json`, run hashes and report:
`volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-overlap-costs.md/json`.
Matched2048runner action: `iso20_overlap2048` via
`bash /home/hendrik/villa3/scripts/run_training_failure_mining`.

## Previous no-dedup BP control — selected 2026-10-03

Historical iterative winner, retained as a comparison: legacy-fine splitting,
minimum256 base voxels, partitioned length disabling cost100 per128 voxels,
boundary penalty10, parallel-one magnitude weight0.0375, all other magnitude
weights0, hard signs both, median probes±64/blend1, fixed scale0.822/phase0.5,
500 message iterations. This supersedes disabling150/boundary0/P1=.05.

| Non-oracle accuracy | Observed shift score | Constraint-disable score | Oracle accuracy |
|---:|---:|---:|---:|
|61.5385%|87.5000%|−28.9207%|80.7692%|

Rank accuracy first, shift second, disable score third. Accuracy/shift did not
improve; the disable score improved from−29.5030%. Winner repeats matched exactly.
Accuracy includes26 reference labels; observed shift covers24 (missing4/12).
This is a tuned1024³ crop result, not a held-out claim or runtime-default change.

The BP crop uses the same raw u40500 snapshot but **isotropic20** reverse
acceptance. The91-fiber counts above are for **ellipse20/40**, not a full-set
isotropic20 evaluation. Keep these two protocols distinct.

Full125-configuration sweep and exact artifact/config provenance:
`volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-iterative-accuracy.md`
(matching JSON). Approved runner action `iso20_iterative`; report-only mode
regenerates the table without running benchmarks.
