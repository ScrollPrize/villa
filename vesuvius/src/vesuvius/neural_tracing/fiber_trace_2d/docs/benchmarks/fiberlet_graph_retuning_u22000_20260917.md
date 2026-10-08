# Fiberlet graph retuning on refinement EMA999 u22000

## Outcome

Keep the existing production fiberlet graph defaults: beam width 16, decision
step 48 base voxels, and lookahead 384 base voxels. A staged 43-setting search
on the corrected five-error benchmark did not eliminate a failure. Every
completed setting had 5 failures and 100% traced coverage. The best summed
first-failure movement was only +22.762 base voxels against a predeclared
+160-base promotion threshold, with one case regressing by 3.089 base voxels.
There was therefore no credible candidate to promote to short200.

This result retunes only graph replay. It holds the production anchor extractor,
fiberlet construction, model, inference cache, benchmark, ellipse, grace, and
restart grouping fixed. The unsuccessful weighted-anchor experiment is not
mixed into this comparison.

## Fixed identities and protocol

- Model: refinement EMA999 u22000, checkpoint SHA
  `67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`.
- Benchmark: corrected mini5 v3, manifest SHA
  `ed031e2f07dafa32b0393186ae531422aee0909898d1e1f621fba69301ddddbe`,
  5 directed cases and 16.8873457254 mm.
- Fiberlets: production anchor extraction, storage chunks 128 base voxels and
  224 construction workers.
- Evaluation: 40-base normal / 80-base tangential ellipse, 160-base excursion
  grace, and 80-base restart episode grouping inherited unchanged from the
  locked evaluation.
- Hardware/build: workspace native evaluation build; all eight GPUs were
  available through the exclusive manager request, but the sweep reused cached
  inference (`tiles=0`) and was graph/CPU work.
- Candidate cap: 180 seconds. The sole timeout was retained as a failed immutable
  attempt rather than deleted.

Each stage explicitly prewarmed the production graph cache and then repeated the
legacy setting. Prewarm rows are excluded below. `First total` is the sum, over
the five cases, of the first failure reference arc or the case end when failure
free. `Delta` and `min delta` are relative to that stage's repeated legacy row.
The combined mean/median are identical for all completed rows (2.814558 mm and
1.319450 mm) because all retain the same five restart episodes; the first-failure
diagnostic resolves the otherwise hidden small route changes.

## Complete measured matrix

| Stage | Variant | B | Step | Look | Fail | Coverage % | First total | Delta | Min delta | Wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | b16_s016_l096 | 16 | 16 | 96 | 5 | 100.0000 | 6736.803 | -10.250 | -7.237 | 17.239 |
| S1 | b16_s016_l192 | 16 | 16 | 192 | 5 | 100.0000 | 6747.159 | +0.105 | +0.000 | 14.712 |
| S1 | b16_s016_l384 | 16 | 16 | 384 | 5 | 100.0000 | 6747.068 | +0.015 | -0.014 | 14.112 |
| S1 | b16_s016_l768 | 16 | 16 | 768 | timeout | — | — | — | — | — |
| S1 | b16_s024_l096 | 16 | 24 | 96 | 5 | 100.0000 | 6768.691 | +21.638 | -3.089 | 14.637 |
| S1 | b16_s024_l192 | 16 | 24 | 192 | 5 | 100.0000 | 6747.039 | -0.014 | -0.014 | 14.080 |
| S1 | b16_s024_l384 | 16 | 24 | 384 | 5 | 100.0000 | 6747.039 | -0.014 | -0.014 | 14.798 |
| S1 | b16_s024_l768 | 16 | 24 | 768 | 5 | 100.0000 | 6747.054 | +0.000 | +0.000 | 62.538 |
| S1 | b16_s032_l096 | 16 | 32 | 96 | 5 | 100.0000 | 6768.691 | +21.638 | -3.089 | 13.776 |
| S1 | b16_s032_l192 | 16 | 32 | 192 | 5 | 100.0000 | 6747.068 | +0.015 | -0.014 | 14.825 |
| S1 | b16_s032_l384 | 16 | 32 | 384 | 5 | 100.0000 | 6747.068 | +0.015 | -0.014 | 14.329 |
| S1 | b16_s032_l768 | 16 | 32 | 768 | 5 | 100.0000 | 6747.068 | +0.015 | -0.014 | 30.832 |
| S1 | b16_s048_l096 | 16 | 48 | 96 | 5 | 100.0000 | 6769.816 | +22.762 | -3.089 | 14.040 |
| S1 | b16_s048_l192 | 16 | 48 | 192 | 5 | 100.0000 | 6747.054 | +0.000 | +0.000 | 14.317 |
| S1 | baseline_b16_s048_l384 | 16 | 48 | 384 | 5 | 100.0000 | 6747.054 | +0.000 | +0.000 | 14.660 |
| S1 | b16_s048_l768 | 16 | 48 | 768 | 5 | 100.0000 | 6747.054 | +0.000 | +0.000 | 30.529 |
| S1 | b16_s064_l096 | 16 | 64 | 96 | 5 | 100.0000 | 6761.680 | +14.626 | -3.089 | 14.591 |
| S1 | b16_s064_l192 | 16 | 64 | 192 | 5 | 100.0000 | 6747.068 | +0.015 | -0.014 | 14.067 |
| S1 | b16_s064_l384 | 16 | 64 | 384 | 5 | 100.0000 | 6747.068 | +0.015 | -0.014 | 15.113 |
| S1 | b16_s064_l768 | 16 | 64 | 768 | 5 | 100.0000 | 6747.068 | +0.015 | -0.014 | 29.536 |
| S1 | b16_s096_l096 | 16 | 96 | 96 | 5 | 100.0000 | 6766.155 | +19.102 | -3.089 | 13.493 |
| S1 | b16_s096_l192 | 16 | 96 | 192 | 5 | 100.0000 | 6747.039 | -0.014 | -0.014 | 15.505 |
| S1 | b16_s096_l384 | 16 | 96 | 384 | 5 | 100.0000 | 6747.039 | -0.014 | -0.014 | 14.132 |
| S1 | b16_s096_l768 | 16 | 96 | 768 | 5 | 100.0000 | 6747.039 | -0.014 | -0.014 | 28.921 |
| S2 | baseline_b16_s048_l384 | 16 | 48 | 384 | 5 | 100.0000 | 6747.054 | +0.000 | +0.000 | 14.433 |
| S2 | b08_s048_l096 | 8 | 48 | 96 | 5 | 100.0000 | 6766.409 | +19.355 | -3.407 | 14.410 |
| S2 | b32_s048_l096 | 32 | 48 | 96 | 5 | 100.0000 | 6769.688 | +22.634 | -3.089 | 13.846 |
| S2 | b64_s048_l096 | 64 | 48 | 96 | 5 | 100.0000 | 6762.790 | +15.736 | -3.089 | 14.367 |
| S2 | b08_s024_l096 | 8 | 24 | 96 | 5 | 100.0000 | 6758.306 | +11.253 | -3.388 | 14.030 |
| S2 | b32_s024_l096 | 32 | 24 | 96 | 5 | 100.0000 | 6761.680 | +14.626 | -3.089 | 14.357 |
| S2 | b64_s024_l096 | 64 | 24 | 96 | 5 | 100.0000 | 6761.680 | +14.626 | -3.089 | 14.492 |
| S2 | b08_s032_l096 | 8 | 32 | 96 | 5 | 100.0000 | 6765.299 | +18.245 | -3.407 | 13.632 |
| S2 | b32_s032_l096 | 32 | 32 | 96 | 5 | 100.0000 | 6761.680 | +14.626 | -3.089 | 14.020 |
| S2 | b64_s032_l096 | 64 | 32 | 96 | 5 | 100.0000 | 6761.680 | +14.626 | -3.089 | 15.321 |
| S3 | baseline_b16_s048_l384 | 16 | 48 | 384 | 5 | 100.0000 | 6747.054 | +0.000 | +0.000 | 14.932 |
| S3 | b16_s040_l096 | 16 | 40 | 96 | 5 | 100.0000 | 6752.965 | +5.911 | -19.346 | 14.775 |
| S3 | b16_s056_l096 | 16 | 56 | 96 | 5 | 100.0000 | 6737.823 | -9.231 | -7.237 | 16.000 |
| S3 | b16_s040_l160 | 16 | 40 | 160 | 5 | 100.0000 | 6745.445 | -1.609 | -1.595 | 14.401 |
| S3 | b16_s048_l160 | 16 | 48 | 160 | 5 | 100.0000 | 6763.264 | +16.211 | -1.595 | 13.997 |
| S3 | b16_s056_l160 | 16 | 56 | 160 | 5 | 100.0000 | 6747.068 | +0.015 | -0.014 | 14.231 |
| S3 | b32_s040_l096 | 32 | 40 | 96 | 5 | 100.0000 | 6752.965 | +5.911 | -19.346 | 15.028 |
| S3 | b32_s056_l096 | 32 | 56 | 96 | 5 | 100.0000 | 6768.691 | +21.638 | -3.089 | 14.360 |
| S3 | b32_s040_l160 | 32 | 40 | 160 | 5 | 100.0000 | 6747.039 | -0.014 | -0.014 | 14.493 |
| S3 | b32_s048_l160 | 32 | 48 | 160 | 5 | 100.0000 | 6765.327 | +18.274 | +0.000 | 13.729 |
| S3 | b32_s056_l160 | 32 | 56 | 160 | 5 | 100.0000 | 6747.039 | -0.014 | -0.014 | 13.729 |

## Interpretation

Short lookahead 96 changes route decisions slightly; lookahead 192 and 384 are
nearly identical, while 768 is slower and can become impractical at the finest
step. Beam widths 8–64 do not improve failure count, and beam32 only shifts the
same failure locations by a few base voxels. The local step/lookahead grid also
does not reveal a stable basin: `b16_s040_l096` gains 5.911 base voxels in total
but regresses one case by 19.346. This is parameter jitter, not a useful quality
improvement.

The historical short200 production-fiberlet result remains 67 failures,
9.736615 mm mean, 1.473118 mm median, 0.144118 mm P05 and 7.847040 mm P95 over
662.089848 directed mm. The regular u22000 beam tracer remains better on that
benchmark at 62 failures and 10.509363 mm mean. No short200 rerun was made because
no mini5 candidate passed the documented promotion rule.

## Artifacts and reproduction

Complete immutable candidate metrics, per-case diagnostics, rankings, and
attempt provenance are under:

- `/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_fiberlet_graph_stage1_20260916_v1`
- `/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_fiberlet_graph_stage2_20260917_v1`
- `/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_fiberlet_graph_stage3_20260917_v1`

Run each checked-in config through the same approved entry point:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_graph_stage1_20260916_v1.json
```

Replace `stage1_20260916` with `stage2_20260917` or `stage3_20260917` for the
other stages. Rerunning validates every completed row and skips it; the verified
stage-3 resume completed without recomputation. Regression validation used:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
```

Native replay suites passed (23, 85 and 17 cases); the Python inspection suite
passed 269 tests with 8 optional skips. No native implementation or numerical
tracing behavior changed. The added Python orchestration is resumable and binds
results to model, benchmark, native binary, and ordered candidate hashes.
