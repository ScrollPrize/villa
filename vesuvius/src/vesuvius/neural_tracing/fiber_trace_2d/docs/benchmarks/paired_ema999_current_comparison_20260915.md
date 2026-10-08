# Paired EMA999 current u42000 — embedding off versus tuned

All50fibers/180directions complete in both runs; same6867.627253directedmm,40/80ellipse,grace160,restart80referencevx andCP-bounded/noLasagna eligibility. Same frozen current paired-contrastive EMA999u42000, SHA7ee3f7c19d12188a297b87c7ab02dde0de01792dcea97e4c691384240cc17a15. Tuned=K2meanLRU/spacing512/novelty.10/weight.10/nearest/noEMAmix. Embedding16 emitted in both runs; same cache reused. All prior rows use embedding scoring off. Mean=directed reference/(reported restarts+1); median/P05/P95 actual segments. Distances mm.

Current off205restarts vs tuned208; combined mean33.338→32.859mm.0139 improves60→59,0332 worsens77→81,other scroll counts unchanged. Native elapsed1057.095s off (new inference),309.850s tuned (warmcache plus additional inferred chunks); not a fair policy throughput comparison. Best EMA999u10800 off launched next; best off/on results pending. Publication implementation deliberately unchanged per user.

## Combined

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 292 | 23.439 | 5.823 | 0.176 | 62.408 |
| EMA99 u1800 | 174 | 39.244 | 7.626 | 0.197 | 87.637 |
| EMA99 u5200 | 148 | 46.091 | 8.202 | 0.143 | 88.438 |
| EMA999 u10200 | 119 | 57.230 | **9.759** | **0.268** | 90.082 |
| EMA999 final u11800 | **113** | **60.242** | 9.337 | 0.233 | **94.960** |
| Paired EMA999 u42000 off | 205 | 33.338 | 5.191 | 0.200 | 87.839 |
| Paired EMA999 u42000 tuned | 208 | 32.859 | 5.021 | 0.198 | 87.772 |

## PHerc0139

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 108 | 10.776 | 2.235 | 0.079 | 25.657 |
| EMA99 u1800 | 73 | 15.872 | 2.535 | 0.090 | 31.510 |
| EMA99 u5200 | 66 | 17.531 | 2.842 | 0.090 | 39.896 |
| EMA999 u10200 | 55 | 20.974 | 3.324 | **0.227** | **46.612** |
| EMA999 final u11800 | **50** | **23.030** | 3.722 | 0.221 | 45.707 |
| Paired EMA999 u42000 off | 60 | 19.255 | 3.757 | 0.079 | 30.695 |
| Paired EMA999 u42000 tuned | 59 | 19.576 | **3.884** | 0.091 | 30.728 |

## PHerc0332

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 9 | 42.516 | 16.777 | 0.923 | 84.676 |
| EMA99 u1800 | **2** | **141.720** | 22.319 | **15.340** | **152.532** |
| EMA99 u5200 | **2** | **141.720** | 22.354 | **15.340** | 152.481 |
| EMA999 u10200 | **2** | **141.720** | **22.467** | **15.340** | 152.156 |
| EMA999 final u11800 | 4 | 85.032 | 22.429 | 7.075 | 129.835 |
| Paired EMA999 u42000 off | 77 | 5.451 | 2.774 | 0.521 | 16.777 |
| Paired EMA999 u42000 tuned | 81 | 5.185 | 2.764 | 0.428 | 16.777 |

## PHerc1299

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 26 | 15.078 | 6.361 | 0.318 | 35.558 |
| EMA99 u1800 | 8 | 45.234 | 15.613 | 0.663 | **68.910** |
| EMA99 u5200 | 11 | 33.925 | 15.406 | 0.847 | 39.372 |
| EMA999 u10200 | **5** | **67.851** | **36.132** | 1.607 | 44.400 |
| EMA999 final u11800 | 7 | 50.888 | 29.134 | 1.411 | 40.666 |
| Paired EMA999 u42000 off | 10 | 37.010 | 20.206 | 0.722 | 40.480 |
| Paired EMA999 u42000 tuned | 10 | 37.010 | 20.206 | **1.849** | 40.482 |

## PHerc1451

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 2 | 16.123 | 11.053 | 1.746 | 11.348 |
| EMA99 u1800 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| EMA99 u5200 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| EMA999 u10200 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| EMA999 final u11800 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Paired EMA999 u42000 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Paired EMA999 u42000 tuned | **0** | **48.369** | **12.092** | **11.129** | **13.056** |

## PHercParis4

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 147 | 32.516 | 8.894 | 0.196 | 85.487 |
| EMA99 u1800 | 91 | 52.309 | 11.289 | **0.271** | 89.582 |
| EMA99 u5200 | 69 | 68.749 | 11.903 | 0.212 | 109.763 |
| EMA999 u10200 | 57 | 82.973 | 15.545 | 0.271 | 112.997 |
| EMA999 final u11800 | **52** | **90.801** | **17.375** | **0.271** | 112.570 |
| Paired EMA999 u42000 off | 58 | 81.567 | 11.636 | 0.174 | **113.023** |
| Paired EMA999 u42000 tuned | 58 | 81.567 | 14.660 | 0.174 | 112.370 |

## Equal-scroll scores

| Model | Classic-normalized score | Arithmetic scroll mean mm |
|---|---:|---:|
| Classic | 23.401875 | 23.401875 |
| EMA99 u1800 | 54.904589 | 60.701007 |
| EMA99 u5200 | 55.845764 | 62.058971 |
| EMA999 u10200 | 69.041084 | 72.377558 |
| EMA999 final u11800 | 61.052591 | 59.624176 |
| Paired EMA999 u42000 off | 31.228526 | 38.330285 |
| Paired EMA999 u42000 tuned | 31.020111 | 38.341290 |

## Commands

Same approved runner for prepare/validate-full/full, snapshot frozen once:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_paired_ema999_current_tuned_20260915_v1.json
```

- Classic: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_thickness40_grace160_20260914_v2/full/summary.json`
- EMA99 u1800: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_thickness40_grace160_20260914_v2/full/summary.json`
- EMA99 u5200: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_current_20260914_1345/full/summary.json`
- EMA999 u10200: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_current_20260914_1555/full/summary.json`
- EMA999 final u11800: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_final_vis_20260914_v1/full/summary.json`
- Paired EMA999 u42000 off: `/ephemeral/hendrik/fiber3d_train2/evaluations/paired_ema999_current_off_20260915_v1/full/summary.json`
- Paired EMA999 u42000 tuned: `/ephemeral/hendrik/fiber3d_train2/evaluations/paired_ema999_current_tuned_20260915_v1/full/summary.json`

