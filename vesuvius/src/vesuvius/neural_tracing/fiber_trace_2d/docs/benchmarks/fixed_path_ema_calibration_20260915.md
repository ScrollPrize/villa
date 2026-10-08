# Fixed-path EMA calibration — 2026-09-15

Frozen u11800 presence/direction + paired current u42000 embeddings. No anchors or neural-model training.
Distances in mm; means = combined directed reference distance / (counted restarts + 1).
Whole-fiber split:30 fit /9 model-selection /11 calibration-heldout. This cohort was used historically for tuning; heldout is not pristine unseen test data.

## Full cohort

| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Classic | 292 | 23.439 | 5.823 | 0.176 | 62.408 |
| u11800 embedding off | 113 | 60.242 | 9.337 | 0.233 | 94.960 |
| u11800 + u42000 EMA baseline | 111 | 61.318 | 10.512 | 0.236 | 96.233 |
| Fitted linear EMA | 110 | 61.871 | 10.506 | 0.232 | 89.634 |
| Fitted smooth EMA | 117 | 58.200 | 9.689 | 0.264 | 92.592 |

## Trace results: fit

| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Classic | 149 | 22.678 | 8.228 | 0.197 | 48.304 |
| u11800 embedding off | 59 | 56.696 | 13.211 | 0.205 | 87.481 |
| u11800 + u42000 EMA baseline | 55 | 60.745 | 14.490 | 0.228 | 87.481 |
| Fitted linear EMA | 56 | 59.679 | 14.978 | 0.198 | 87.481 |
| Fitted smooth EMA | 55 | 60.745 | 13.381 | 0.200 | 88.024 |

## Trace results: selection

| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Classic | 85 | 20.195 | 5.006 | 0.088 | 66.985 |
| u11800 embedding off | 36 | 46.940 | 7.095 | 0.235 | 112.370 |
| u11800 + u42000 EMA baseline | 35 | 48.244 | 7.213 | 0.235 | 112.370 |
| Fitted linear EMA | 34 | 49.622 | 6.485 | 0.241 | 112.370 |
| Fitted smooth EMA | 38 | 44.533 | 5.314 | 0.246 | 112.370 |

## Trace results: test

| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Classic | 58 | 29.307 | 4.674 | 0.027 | 75.103 |
| u11800 embedding off | 18 | 91.007 | 10.993 | 0.293 | 113.071 |
| u11800 + u42000 EMA baseline | 21 | 78.597 | 10.001 | 0.271 | 114.467 |
| Fitted linear EMA | 20 | 82.339 | 11.129 | 0.284 | 109.915 |
| Fitted smooth EMA | 24 | 69.165 | 10.485 | 0.285 | 109.763 |

## PHerc0139

| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Classic | 108 | 10.776 | 2.235 | 0.079 | 25.657 |
| u11800 embedding off | 50 | 23.030 | 3.722 | 0.221 | 45.707 |
| u11800 + u42000 EMA baseline | 49 | 23.491 | 3.898 | 0.217 | 47.174 |
| Fitted linear EMA | 46 | 24.991 | 3.476 | 0.216 | 48.308 |
| Fitted smooth EMA | 52 | 22.161 | 3.197 | 0.223 | 45.680 |

## PHerc0332

| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Classic | 9 | 42.516 | 16.777 | 0.923 | 84.676 |
| u11800 embedding off | 4 | 85.032 | 22.429 | 7.075 | 129.835 |
| u11800 + u42000 EMA baseline | 4 | 85.032 | 22.419 | 7.216 | 129.789 |
| Fitted linear EMA | 4 | 85.032 | 22.426 | 7.140 | 129.785 |
| Fitted smooth EMA | 4 | 85.032 | 22.427 | 7.097 | 129.780 |

## PHerc1299

| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Classic | 26 | 15.078 | 6.361 | 0.318 | 35.558 |
| u11800 embedding off | 7 | 50.888 | 29.134 | 1.411 | 40.666 |
| u11800 + u42000 EMA baseline | 9 | 40.711 | 21.474 | 1.587 | 40.531 |
| Fitted linear EMA | 7 | 50.888 | 29.347 | 1.411 | 40.674 |
| Fitted smooth EMA | 9 | 40.711 | 17.756 | 1.664 | 40.539 |

## PHerc1451

| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Classic | 2 | 16.123 | 11.053 | 1.746 | 11.348 |
| u11800 embedding off | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| u11800 + u42000 EMA baseline | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| Fitted linear EMA | 0 | 48.369 | 12.092 | 11.129 | 13.056 |
| Fitted smooth EMA | 0 | 48.369 | 12.092 | 11.129 | 13.056 |

## PHercParis4

| Model | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| Classic | 147 | 32.516 | 8.894 | 0.196 | 85.487 |
| u11800 embedding off | 52 | 90.801 | 17.375 | 0.271 | 112.570 |
| u11800 + u42000 EMA baseline | 49 | 96.249 | 17.301 | 0.282 | 113.838 |
| Fitted linear EMA | 53 | 89.119 | 17.469 | 0.270 | 112.370 |
| Fitted smooth EMA | 52 | 90.801 | 17.563 | 0.285 | 112.370 |

## Fitted parameters

| Mapping | Weight | Half-life base voxels | Threshold | Temperature | Selection iteration |
|---|---:|---:|---:|---:|---:|
| Baseline linear | 0.25 | 512 | — | — | — |
| Linear | 0.045401283 | 510.432190 | — | — | 242 |
| Smooth | 0.227174774 | 513.778931 | 0.8921006321907043 | 0.04869288206100464 | 300 |

## Fixed-path objective

GPU Adam,300steps/family,LR0.03. Learn bounded weight[.001,2], half-life[32,2048], optional threshold[0,1] and temperature[.01,.5].
Mean per-step native base loss + embedding penalty, pairwise softplus ranking (margin.005,temperature.05); successful controls preserve baseline relative cost. Log-weight/half-life regularizer. Inverse-total-window-per-fiber weighting, separately normalized failure/control losses.
Smooth penalty = weight*temperature*[softplus((distance-threshold)/temperature)-softplus(-threshold/temperature)], so penalty(0)=0.
Shared1024-base-voxel maximum warm-up, detached after recomputation at each current half-life;32-step/256-base-voxel branches, including native parent-before-update normalized EMA.
Native vs GPU exported-score max absolute discrepancy: 2.68220901e-07.

| Mapping | Subset | Failure windows | GT preferred % | Ranking loss |
|---|---|---:|---:|---:|
| Baseline | fit | 12 | 16.67 | 3.911473 |
| Baseline | selection | 6 | 0.00 | 4.300224 |
| Baseline | test | 11 | 0.00 | 5.870987 |
| Linear | fit | 12 | 16.67 | 3.833037 |
| Linear | selection | 6 | 0.00 | 4.241023 |
| Linear | test | 11 | 0.00 | 5.711046 |
| Smooth | fit | 12 | 16.67 | 3.816257 |
| Smooth | selection | 6 | 0.00 | 4.228208 |
| Smooth | test | 11 | 0.00 | 5.676226 |

## Coverage and limitations

Accepted 148 windows; exclusions: `{'reference_end': 4, 'no_shared_near_reference_start': 22, 'outside_prediction_cone': 34, 'short_segment': 55, 'connector_curvature': 28}`.
Actual branch distances (base voxels): `{'good': {'min': 255.63870334625244, 'median': 255.97105479240417, 'max': 255.99918127059937}, 'bad': {'min': 255.99629640579224, 'median': 255.99989891052246, 'max': 256.0020456314087}}`.
No windows were replaced with zeros. Native extraction is CPU-only with existing local cache and readiness checks.
Hermite GT connector is continuous and limited by curvature/circumscribed cone checks, NOT guaranteed on the discrete beam grid. Equal sample counts are not exactly equal Euclidean lengths.
Finite warm-up reseeds embedding and smoothness state; parity is against native exported finite-prefix scoring, not original full ancestry. Long-window sums are not depth2 beam search. Report also contains first-two-step diagnostics.
Initial short dataset accidentally recorded null library hash (wrong lib/ location); current native core library hash is included in comparison.json, and subsequent long extraction records it correctly. Binary/scoring implementation did not change between those exports.
Long-window diagnostic, when present in comparison.json, applies the same learned parameters without fitting. It mines512-base-voxel windows independently, so it is not a strictly paired truncation ablation.

## Artifacts

- Classic: /ephemeral/hendrik/fiber3d_train2/evaluations/classic_thickness40_grace160_20260914_v2/full/summary.json
- u11800 embedding off: /ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/summary.json
- u11800 + u42000 EMA baseline: /ephemeral/hendrik/fiber3d_train2/evaluations/mixed_current_ema_only_20260915_v1/full/summary.json
- Fitted linear EMA: /ephemeral/hendrik/fiber3d_train2/evaluations/fixed_path_ema_linear_20260915_v1/full/summary.json
- Fitted smooth EMA: /ephemeral/hendrik/fiber3d_train2/evaluations/fixed_path_ema_smooth_20260915_v1/full/summary.json

Pending evaluations: []
No automatic default change. No visualization run.

## Conclusion

The simple fitted linear weight gives110restarts versus111baseline, but smooth mapping gives117 despite lower surrogate loss. No default promoted. Heldout counts: embeddings-off18, baseline21, fitted-linear20, smooth24. Linear gains in0139 and1299 trade off against Paris4. This small single-evaluation difference does not establish statistical significance. Larger512basevoxel windows preserve zero heldout GT-preference wins for both fitted models; loss improvements alone do not reflect beam outcomes.

The current fixed-path failure windows have positive mean GT-minus-error base costs and embedding costs (fit:0.168649+0.004420,selection:0.227589+0.009000,test:0.235930+0.010791). Thus reducing embedding influence is understandable in this objective; it does not learn to discriminate the GT branch. The nonlinear near-suppression reduces this loss but performs worse in actual tracing. A future exact-beam-candidate dataset would address the surrogate mismatch; it was not part of this first implementation.

## Verification

Workspace RelWithDebInfo GCC13 native build.103native tests and219Python tests passed;8optional skips. Both full runs completed180directions/50fibers and6867.627253directedmm, outer commands exited0, effective EMA-only parameters verified. Native walls287.813s linear/288.463s smooth, coordinator358.877s/358.022s; not controlled throughput benchmarks. GPU fitting used one exclusive manager-granted GPU, native full runs all8exclusive. No active evaluation remains from this experiment.

## Exact commands

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-extract --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_fit_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-fit --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_fit_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-extract --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_fit_long_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-fit --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_fit_long_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_linear_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_linear_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_linear_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_smooth_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_smooth_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_smooth_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py fixed-path-report --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fixed_path_ema_fit_20260915_v1.json
```
