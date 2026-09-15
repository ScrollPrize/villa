# Embedding integration experiments — 2026-09-15

Frozen u11800 presence/direction + paired current u42000 embeddings. No anchors. No default promotion.
50 fibers / five scrolls / 180 eligible run-directions. Split:30 fit,9 selection,11 calibration-heldout fibers.
Mean = combined directed reference distance / (counted restarts +1). Median/P05/P95 include terminal episodes. All distances below are mm.

## Full cohort

| Method | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| Classic | 292 | 23.439 | 5.823 | 0.176 | 62.408 |
| u11800 embeddings off | 113 | 60.242 | 9.337 | 0.233 | 94.960 |
| Original EMA .25/512 | 111 | 61.318 | 10.512 | 0.236 | 96.233 |
| Previous fitted EMA | 110 | 61.871 | 10.506 | 0.232 | 89.634 |
| angular | 115 | 59.204 | 10.507 | 0.262 | 102.700 |
| dual | 113 | 60.242 | 9.984 | 0.237 | 108.852 |
| ema | 117 | 58.200 | 10.011 | 0.232 | 102.469 |
| gate | 110 | 61.871 | 10.685 | 0.228 | 96.830 |
| window | 111 | 61.318 | 10.490 | 0.251 | 89.563 |

## PHerc0139

| Method | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| Classic | 108 | 10.776 | 2.235 | 0.079 | 25.657 |
| u11800 embeddings off | 50 | 23.030 | 3.722 | 0.221 | 45.707 |
| Original EMA .25/512 | 49 | 23.491 | 3.898 | 0.217 | 47.174 |
| Previous fitted EMA | 46 | 24.991 | 3.476 | 0.216 | 48.308 |
| angular | 51 | 22.588 | 3.191 | 0.222 | 47.049 |
| dual | 49 | 23.491 | 3.191 | 0.220 | 47.327 |
| ema | 51 | 22.588 | 3.191 | 0.220 | 47.320 |
| gate | 50 | 23.030 | 3.596 | 0.214 | 47.506 |
| window | 49 | 23.491 | 3.707 | 0.219 | 47.680 |

## PHerc0332

| Method | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| Classic | 9 | 42.516 | 16.777 | 0.923 | 84.676 |
| u11800 embeddings off | 4 | 85.032 | 22.429 | 7.075 | 129.835 |
| Original EMA .25/512 | 4 | 85.032 | 22.419 | 7.216 | 129.789 |
| Previous fitted EMA | 4 | 85.032 | 22.426 | 7.140 | 129.785 |
| angular | 4 | 85.032 | 22.433 | 7.114 | 129.853 |
| dual | 4 | 85.032 | 22.425 | 7.122 | 129.801 |
| ema | 4 | 85.032 | 22.426 | 7.141 | 129.783 |
| gate | 4 | 85.032 | 22.426 | 7.121 | 129.784 |
| window | 3 | 106.290 | 28.078 | 15.340 | 132.288 |

## PHerc1299

| Method | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| Classic | 26 | 15.078 | 6.361 | 0.318 | 35.558 |
| u11800 embeddings off | 7 | 50.888 | 29.134 | 1.411 | 40.666 |
| Original EMA .25/512 | 9 | 40.711 | 21.474 | 1.587 | 40.531 |
| Previous fitted EMA | 7 | 50.888 | 29.347 | 1.411 | 40.674 |
| angular | 7 | 50.888 | 27.647 | 1.472 | 40.677 |
| dual | 8 | 45.234 | 22.720 | 1.587 | 40.614 |
| ema | 6 | 58.158 | 32.759 | 1.385 | 40.746 |
| gate | 8 | 45.234 | 24.605 | 1.529 | 40.613 |
| window | 7 | 50.888 | 29.079 | 1.518 | 40.677 |

## PHerc1451

| Method | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| Classic | 2 | 16.123 | 11.053 | 1.746 | 11.348 |
| u11800 embeddings off | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| Original EMA .25/512 | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| Previous fitted EMA | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| angular | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| dual | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| ema | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| gate | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| window | 0 | 48.369 | 12.092 | 11.129 | 13.056 |

## PHercParis4

| Method | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| Classic | 147 | 32.516 | 8.894 | 0.196 | 85.487 |
| u11800 embeddings off | 52 | 90.801 | 17.375 | 0.271 | 112.570 |
| Original EMA .25/512 | 49 | 96.249 | 17.301 | 0.282 | 113.838 |
| Previous fitted EMA | 53 | 89.119 | 17.469 | 0.270 | 112.370 |
| angular | 53 | 89.119 | 14.774 | 0.289 | 112.565 |
| dual | 52 | 90.801 | 16.208 | 0.286 | 113.965 |
| ema | 56 | 84.429 | 17.216 | 0.271 | 112.494 |
| gate | 48 | 98.213 | 17.571 | 0.283 | 112.681 |
| window | 52 | 90.801 | 16.616 | 0.271 | 112.370 |

## Trace subset: fit

| Method | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| Classic | 149 | 22.678 | 8.228 | 0.197 | 48.304 |
| u11800 embeddings off | 59 | 56.696 | 13.211 | 0.205 | 87.481 |
| Original EMA .25/512 | 55 | 60.745 | 14.490 | 0.228 | 87.481 |
| Previous fitted EMA | 56 | 59.679 | 14.978 | 0.198 | 87.481 |
| angular | 56 | 59.679 | 13.885 | 0.200 | 87.991 |
| dual | 56 | 59.679 | 13.298 | 0.232 | 87.988 |
| ema | 58 | 57.656 | 13.862 | 0.201 | 87.981 |
| gate | 53 | 62.995 | 14.797 | 0.199 | 87.481 |
| window | 57 | 58.651 | 14.373 | 0.201 | 86.324 |

## Trace subset: selection

| Method | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| Classic | 85 | 20.195 | 5.006 | 0.088 | 66.985 |
| u11800 embeddings off | 36 | 46.940 | 7.095 | 0.235 | 112.370 |
| Original EMA .25/512 | 35 | 48.244 | 7.213 | 0.235 | 112.370 |
| Previous fitted EMA | 34 | 49.622 | 6.485 | 0.241 | 112.370 |
| angular | 37 | 45.705 | 4.857 | 0.236 | 112.370 |
| dual | 35 | 48.244 | 5.597 | 0.235 | 112.370 |
| ema | 38 | 44.533 | 6.210 | 0.236 | 112.370 |
| gate | 37 | 45.705 | 6.666 | 0.227 | 112.370 |
| window | 35 | 48.244 | 7.016 | 0.240 | 112.370 |

## Trace subset: test

| Method | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| Classic | 58 | 29.307 | 4.674 | 0.027 | 75.103 |
| u11800 embeddings off | 18 | 91.007 | 10.993 | 0.293 | 113.071 |
| Original EMA .25/512 | 21 | 78.597 | 10.001 | 0.271 | 114.467 |
| Previous fitted EMA | 20 | 82.339 | 11.129 | 0.284 | 109.915 |
| angular | 22 | 75.179 | 11.127 | 0.416 | 109.763 |
| dual | 22 | 75.179 | 10.542 | 0.279 | 112.577 |
| ema | 21 | 78.597 | 9.337 | 0.271 | 112.725 |
| gate | 20 | 82.339 | 10.993 | 0.284 | 112.894 |
| window | 19 | 86.456 | 11.129 | 0.296 | 110.063 |

## All fitted candidates

Parameter vectors: EMA=[weight,half-life]; window=[weight]; gate=[weight,half-life,bias,presence coefficient,distance coefficient]; angular=[weight,half-life,cap degrees per8basevoxels]; dual=[weight,short half-life,long-minus-short half-life,mix bias,mix slope].
Spacing/range are base voxels. Asterisk marks each selection-set winner. Heldout values never select candidates.

| Family | Spacing | Window | Start | Fitted parameters | Iteration | Selection loss | Heldout loss |
|---|---:|---:|---|---|---:|---:|---:|
| ema* | 8 | — | 0.0454013, 64 | 0.0441175, 64.9586 | 1 | 4.2407527 | 5.7113791 |
| ema | 8 | — | 0.0454013, 128 | 0.0454013, 128 | 0 | 4.2409725 | 5.7106252 |
| ema | 8 | — | 0.0454013, 256 | 0.0454013, 256 | 0 | 4.2409105 | 5.7101178 |
| ema | 8 | — | 0.0454013, 512 | 0.0454013, 512 | 0 | 4.2409391 | 5.7109723 |
| ema | 8 | — | 0.0454013, 1024 | 0.0454013, 1024 | 0 | 4.2409778 | 5.7122006 |
| ema | 16 | — | 0.0454013, 64 | 0.0400028, 71.3335 | 7 | 4.2409263 | 5.7065697 |
| ema | 16 | — | 0.0454013, 128 | 0.0454013, 128 | 0 | 4.2412744 | 5.7103677 |
| ema | 16 | — | 0.0454013, 256 | 0.0454013, 256 | 0 | 4.2408342 | 5.7108488 |
| ema | 16 | — | 0.0454013, 512 | 0.0454013, 512 | 0 | 4.2408433 | 5.7119284 |
| ema | 16 | — | 0.0454013, 1024 | 0.0454013, 1024 | 0 | 4.2409053 | 5.7128820 |
| ema | 32 | — | 0.0454013, 64 | 0.0480403, 441.685 | 150 | 4.2423224 | 5.7149072 |
| ema | 32 | — | 0.0454013, 128 | 0.0479782, 470.623 | 133 | 4.2422614 | 5.7149439 |
| ema | 32 | — | 0.0454013, 256 | 0.0454013, 256 | 0 | 4.2421427 | 5.7122455 |
| ema | 32 | — | 0.0454013, 512 | 0.0454013, 512 | 0 | 4.2414546 | 5.7129617 |
| ema | 32 | — | 0.0454013, 1024 | 0.0454013, 1024 | 0 | 4.2412381 | 5.7136545 |
| ema | 64 | — | 0.0454013, 64 | 0.0496396, 399.878 | 150 | 4.2427645 | 5.7128868 |
| ema | 64 | — | 0.0454013, 128 | 0.0494614, 414.195 | 101 | 4.2426877 | 5.7128234 |
| ema | 64 | — | 0.0454013, 256 | 0.0454013, 256 | 0 | 4.2420039 | 5.7095256 |
| ema | 64 | — | 0.0454013, 512 | 0.0454013, 512 | 0 | 4.2413864 | 5.7102575 |
| ema | 64 | — | 0.0454013, 1024 | 0.0454013, 1024 | 0 | 4.2412062 | 5.7119007 |
| window* | 8 | 256 | 0.0454013 | 0.0362782 | 12 | 4.2371397 | 5.7079558 |
| window | 8 | 512 | 0.0454013 | 0.0407358 | 6 | 4.2388716 | 5.7090240 |
| window | 8 | 1024 | 0.0454013 | 0.0431729 | 3 | 4.2399082 | 5.7104564 |
| window | 8 | 2048 | 0.0454013 | 0.0440116 | 2 | 4.2404690 | 5.7111912 |
| window | 16 | 256 | 0.0454013 | 0.0340917 | 15 | 4.2393589 | 5.7110639 |
| window | 16 | 512 | 0.0454013 | 0.0406581 | 6 | 4.2410150 | 5.7132101 |
| window | 16 | 1024 | 0.0454013 | 0.0430803 | 3 | 4.2413349 | 5.7135253 |
| window | 16 | 2048 | 0.0454013 | 0.044442 | 4 | 4.2418237 | 5.7142496 |
| window | 32 | 256 | 0.0454013 | 0.0366486 | 11 | 4.2475944 | 5.7162175 |
| window | 32 | 512 | 0.0454013 | 0.0431154 | 3 | 4.2462149 | 5.7204561 |
| window | 32 | 1024 | 0.0454013 | 0.045288 | 4 | 4.2443252 | 5.7214022 |
| window | 32 | 2048 | 0.0454013 | 0.0454013 | 0 | 4.2431693 | 5.7209015 |
| window | 64 | 256 | 0.0454013 | 0.0454013 | 0 | 4.2465219 | 5.7371054 |
| window | 64 | 512 | 0.0454013 | 0.0454013 | 0 | 4.2436380 | 5.7224131 |
| window | 64 | 1024 | 0.0454013 | 0.0454013 | 0 | 4.2414980 | 5.7182751 |
| window | 64 | 2048 | 0.0454013 | 0.0454013 | 0 | 4.2406163 | 5.7147193 |
| gate* | 8 | — | 0.0454013, 64, 3, 2, 2 | 0.0464859, 78.9032, 1.67657, 3.314, 2.68291 | 13 | 4.2406368 | 5.7111392 |
| gate | 8 | — | 0.0454013, 256, 3, 2, 2 | 0.0454013, 256, 3, 2, 2 | 0 | 4.2408800 | 5.7100668 |
| gate | 8 | — | 0.0454013, 1024, 3, 2, 2 | 0.0452009, 963.682, 3.39966, 1.54601, 1.80948 | 4 | 4.2409210 | 5.7120032 |
| angular | 8 | — | 0.0454013, 64, 1 | 0.0454013, 64, 1 | 0 | 4.2410131 | 5.7100210 |
| angular* | 8 | — | 0.0454013, 256, 5 | 0.0454013, 256, 5 | 0 | 4.2409105 | 5.7101178 |
| angular | 8 | — | 0.0454013, 1024, 15 | 0.0454013, 1024, 15 | 0 | 4.2409778 | 5.7122006 |
| dual | 8 | — | 0.0454013, 64, 512, 0, 2 | 0.0454013, 64, 512, 0, 2 | 0 | 4.2410636 | 5.7107315 |
| dual | 8 | — | 0.0454013, 128, 1024, 0, 2 | 0.0454013, 128, 1024, 0, 2 | 0 | 4.2409692 | 5.7103629 |
| dual* | 8 | — | 0.0454013, 256, 2048, 0, 2 | 0.0454013, 256, 2048, 0, 2 | 0 | 4.2408724 | 5.7104778 |

## Runtime (not a controlled speed benchmark)

| Family | Native seconds | Coordinator seconds |
|---|---:|---:|
| angular | 286.323 | 355.875 |
| dual | 286.845 | 357.936 |
| ema | 285.595 | 359.027 |
| gate | 285.558 | 356.595 |
| window | 300.391 | 371.988 |

RelWithDebInfo GCC13, one full180direction evaluation per family; exclusive64CPU/all8GPU manager allocations. Same cached source/inference data, with additional inference for newly visited off-route chunks where needed. Timings include differing cache misses and are not decoder microbenchmarks.

## Verification and limitations

Native/PyTorch parity: 270 real paths, max absolute weighted-score error 2.12950414e-07.
Accepted windows: 148; exclusions: `{'reference_end': 4, 'no_shared_near_reference_start': 22, 'outside_prediction_cone': 34, 'short_segment': 55, 'connector_curvature': 28}`.
GPU fitting:150 Adam updates per family,45 candidates,LR.03; candidates within each family are vectorized. Weight and half-life are jointly fitted (finite window fits only weight). Common warm-up up to4096basevoxels is recomputed and detached; branches are32steps/~256basevoxels.
Spacing controls admission of selected observations; every candidate is still scored. Gate rejection consumes the admission interval. Finite history expires at the candidate position. Angular cap scales with elapsed physical distance. Dual continuously updates two EMAs and adaptively mixes distances, not anchors.
Fixed-path surrogate is not beam-search rollout: connected GT candidates need not lie on the discrete beam grid, finite warm-up resets ancestry, and truncated backprop ignores warm-up gradients. This cohort was used for earlier tuning; heldout is not pristine unseen data. Small full-eval count changes are not evidence of significance.
Opt-in flags leave old default behavior unchanged. No rendering requested.

## Full evaluation artifacts

- Classic: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_thickness40_grace160_20260914_v2/full/summary.json`
- u11800 embeddings off: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/summary.json`
- Original EMA .25/512: `/ephemeral/hendrik/fiber3d_train2/evaluations/mixed_current_ema_only_20260915_v1/full/summary.json`
- Previous fitted EMA: `/ephemeral/hendrik/fiber3d_train2/evaluations/fixed_path_ema_linear_20260915_v1/full/summary.json`
- angular: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_integration_angular_20260915_v1/full/summary.json`
- dual: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_integration_dual_20260915_v1/full/summary.json`
- ema: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_integration_ema_20260915_v1/full/summary.json`
- gate: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_integration_gate_20260915_v1/full/summary.json`
- window: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_integration_window_20260915_v1/full/summary.json`

## Exact commands

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-extract --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_fit_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py integration-fit --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_fit_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py integration-audit --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_fit_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py integration-report --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_fit_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_angular_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_angular_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_angular_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_dual_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_dual_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_dual_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_ema_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_ema_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_ema_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_gate_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_gate_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_gate_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_window_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_window_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_embedding_integration_window_20260915_v1.json
```

## Outcome

Best new full-cohort count: 110 restarts (gate); previous fitted EMA: 110.
No new family improves the previous full-cohort restart count.
Compare per-scroll and heldout tables before interpreting an aggregate tie. No automatic default change.

Pending: []
