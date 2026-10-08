# Final EMA999 u11800 — full evaluation and visualization

Merge: afb14b400 (83dad0268 + be93b129a). Final stopped training snapshot: current EMA0.999, update11800 / samples1510400, test loss0.29991902176998864. SHA256 `4d7c3b7ed06a29a344f8f21175158bf40c42ade440943ff68aa2775ee0cc1d82`.

Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1`.

Same50fibers/five scrolls/six collections; 90 eligible runs/180directions and 6867.627253 directedmm. CP-bounded/no Lasagna spans; thickness40, width80, excursion grace160; restart episode gap80 reference/base voxels; embedding off. Mean=combined directed distance/(reported restarts+1). Percentiles use actual segments; all distances mm. Snapshot ages differ: not a controlled EMA-decay comparison.

## Commands and validation

Approved runner used for freeze-snapshot, prepare, validate-full, full:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_eligible_ema999_final_vis_20260914_v1.json
```

All8GPUs exclusively via manager, priority100/64logical CPUs. Workspace RelWithDebInfo native build. One full run; no training. Existing CT/normal downloads reused; new checkpoint-specific inference cache. Build/tests after merge: 23 replay/render +79 native tracer cases; 206Python passed/8skipped. Live lease reacquisition not exercised by CPU tests; this run uses existing exclusive whole-run allocation.

180trace and180render metadata artifacts present. Sample case45 visually inspected: combined CT/presence, distinct top/side strips, restart IDs and aligned transverse sections. Full summary metrics validated through shared complete_report; identical coverage to baselines. Native wall time including rendering: 1413.538s (not comparable to prior trace-only timings).

## Combined

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 292 | 23.439 | 5.823 | 0.176 | 62.408 |
| EMA99 u1800 | 174 | 39.244 | 7.626 | 0.197 | 87.637 |
| EMA99 u5200 | 148 | 46.091 | 8.202 | 0.143 | 88.438 |
| EMA999 u10200 | 119 | 57.230 | **9.759** | **0.268** | 90.082 |
| EMA999 final u11800 | **113** | **60.242** | 9.337 | 0.233 | **94.960** |

## PHerc0139

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 108 | 10.776 | 2.235 | 0.079 | 25.657 |
| EMA99 u1800 | 73 | 15.872 | 2.535 | 0.090 | 31.510 |
| EMA99 u5200 | 66 | 17.531 | 2.842 | 0.090 | 39.896 |
| EMA999 u10200 | 55 | 20.974 | 3.324 | **0.227** | **46.612** |
| EMA999 final u11800 | **50** | **23.030** | **3.722** | 0.221 | 45.707 |

## PHerc0332

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 9 | 42.516 | 16.777 | 0.923 | 84.676 |
| EMA99 u1800 | **2** | **141.720** | 22.319 | **15.340** | **152.532** |
| EMA99 u5200 | **2** | **141.720** | 22.354 | **15.340** | 152.481 |
| EMA999 u10200 | **2** | **141.720** | **22.467** | **15.340** | 152.156 |
| EMA999 final u11800 | 4 | 85.032 | 22.429 | 7.075 | 129.835 |

## PHerc1299

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 26 | 15.078 | 6.361 | 0.318 | 35.558 |
| EMA99 u1800 | 8 | 45.234 | 15.613 | 0.663 | **68.910** |
| EMA99 u5200 | 11 | 33.925 | 15.406 | 0.847 | 39.372 |
| EMA999 u10200 | **5** | **67.851** | **36.132** | **1.607** | 44.400 |
| EMA999 final u11800 | 7 | 50.888 | 29.134 | 1.411 | 40.666 |

## PHerc1451

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 2 | 16.123 | 11.053 | 1.746 | 11.348 |
| EMA99 u1800 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| EMA99 u5200 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| EMA999 u10200 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| EMA999 final u11800 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |

## PHercParis4

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 147 | 32.516 | 8.894 | 0.196 | 85.487 |
| EMA99 u1800 | 91 | 52.309 | 11.289 | **0.271** | 89.582 |
| EMA99 u5200 | 69 | 68.749 | 11.903 | 0.212 | 109.763 |
| EMA999 u10200 | 57 | 82.973 | 15.545 | 0.271 | **112.997** |
| EMA999 final u11800 | **52** | **90.801** | **17.375** | **0.271** | 112.570 |

## Equal-scroll scores

| Model | Classic-normalized score ↑ | Arithmetic scroll mean mm ↑ |
|---|---:|---:|
| Classic | 23.401875 | 23.401875 |
| EMA99 u1800 | 54.904589 | 60.701007 |
| EMA99 u5200 | 55.845764 | 62.058971 |
| EMA999 u10200 | 69.041084 | 72.377558 |
| EMA999 final u11800 | 61.052591 | 59.624176 |

Normalized score = Classic arithmetic scroll mean × geometric mean of five model/Classic mean ratios (20% each); not a measured distance. Final model improves combined restarts (119→113; −5.04%) and mean (+5.26%) over u10200, but regresses on0332/1299 and the equal-scroll score. Combined median/P05 also decrease; improvement is not uniform.

## Source summaries

- Classic: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_thickness40_grace160_20260914_v2/full/summary.json`
- EMA99 u1800: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_thickness40_grace160_20260914_v2/full/summary.json`
- EMA99 u5200: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_current_20260914_1345/full/summary.json`
- EMA999 u10200: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_current_20260914_1555/full/summary.json`
- EMA999 final u11800: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/summary.json`

