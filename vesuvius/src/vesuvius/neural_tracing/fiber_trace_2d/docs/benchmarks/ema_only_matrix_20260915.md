# EMA-only embedding matrix — completed 2026-09-15

All four requested runs completed successfully: **50 fibers / 180 eligible directions / 6867.627253 directed reference mm** each. Same CP-bounded/no-Lasagna selection, ellipse thickness40/width80basevoxels, excursion grace160 and restart filtering80referencevoxels. No visualization or retuning.

Policy: **plain EMA**, weight0.25, half-life512basevoxels; no anchors, gating, retained-seed mixture or delay. This is the previously recorded69-restart tuning winner, not the overall Anchor-LRU winner. EMA999 in snapshot labels refers to model-weight averaging, a separate mechanism. Older incorrectly labelled 'tuned' rows are explicitly **Anchor-LRU** below.

Mean = combined directed reference distance / (reported restarts + 1), for each aggregation. Median/P05/P95 use actual segments including terminal segments. All lengths are mm. Best/tied values are bold. Single evaluations on one held-out cohort: small differences are not statistical significance claims.

## Findings

- Paired best u10800: EMA-only133restarts vs135off (Anchor-LRU139).
- Paired current u42000: EMA-only211 vs205off (Anchor-LRU208).
- u11800 predictions + best u10800 embeddings:123, worse than u11800 alone113.
- u11800 predictions + current u42000 embeddings:111, slightly better than u11800 alone113; combined mean61.318vs60.242mm. This is the lowest combined restart count in this matched table, not a win on every metric/scroll.

## Combined

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 292 | 23.439 | 5.823 | 0.176 | 62.408 |
| EMA99 u1800 | 174 | 39.244 | 7.626 | 0.197 | 87.637 |
| EMA99 u5200 | 148 | 46.091 | 8.202 | 0.143 | 88.438 |
| EMA999 u10200 | 119 | 57.230 | 9.759 | **0.268** | 90.082 |
| EMA999 final u11800 | 113 | 60.242 | 9.337 | 0.233 | 94.960 |
| Paired EMA999 u42000 off | 205 | 33.338 | 5.191 | 0.200 | 87.839 |
| Paired EMA999 u42000 Anchor-LRU | 208 | 32.859 | 5.021 | 0.198 | 87.772 |
| Paired EMA999 best u10800 off | 135 | 50.497 | 8.912 | 0.184 | 90.116 |
| Paired EMA999 best u10800 Anchor-LRU | 139 | 49.054 | 8.920 | 0.197 | 90.098 |
| Paired best u10800 EMA-only | 133 | 51.251 | 8.922 | 0.204 | 90.124 |
| Paired current u42000 EMA-only | 211 | 32.394 | 4.800 | 0.203 | 87.674 |
| u11800 + best u10800 embeddings EMA-only | 123 | 55.384 | 9.337 | 0.238 | **103.367** |
| u11800 + current u42000 embeddings EMA-only | **111** | **61.318** | **10.512** | 0.236 | 96.233 |

## PHerc0139

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 108 | 10.776 | 2.235 | 0.079 | 25.657 |
| EMA99 u1800 | 73 | 15.872 | 2.535 | 0.090 | 31.510 |
| EMA99 u5200 | 66 | 17.531 | 2.842 | 0.090 | 39.896 |
| EMA999 u10200 | 55 | 20.974 | 3.324 | **0.227** | 46.612 |
| EMA999 final u11800 | 50 | 23.030 | 3.722 | 0.221 | 45.707 |
| Paired EMA999 u42000 off | 60 | 19.255 | 3.757 | 0.079 | 30.695 |
| Paired EMA999 u42000 Anchor-LRU | 59 | 19.576 | 3.884 | 0.091 | 30.728 |
| Paired EMA999 best u10800 off | 59 | 19.576 | 3.191 | 0.091 | 46.279 |
| Paired EMA999 best u10800 Anchor-LRU | 64 | 18.070 | 3.191 | 0.127 | 46.192 |
| Paired best u10800 EMA-only | 62 | 18.644 | 3.455 | 0.100 | 46.227 |
| Paired current u42000 EMA-only | 55 | 20.974 | 3.191 | 0.144 | 33.909 |
| u11800 + best u10800 embeddings EMA-only | 53 | 21.751 | 3.191 | 0.203 | 45.686 |
| u11800 + current u42000 embeddings EMA-only | **49** | **23.491** | **3.898** | 0.217 | **47.174** |

## PHerc0332

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 9 | 42.516 | 16.777 | 0.923 | 84.676 |
| EMA99 u1800 | **2** | **141.720** | 22.319 | **15.340** | **152.532** |
| EMA99 u5200 | **2** | **141.720** | 22.354 | **15.340** | 152.481 |
| EMA999 u10200 | **2** | **141.720** | **22.467** | **15.340** | 152.156 |
| EMA999 final u11800 | 4 | 85.032 | 22.429 | 7.075 | 129.835 |
| Paired EMA999 u42000 off | 77 | 5.451 | 2.774 | 0.521 | 16.777 |
| Paired EMA999 u42000 Anchor-LRU | 81 | 5.185 | 2.764 | 0.428 | 16.777 |
| Paired EMA999 best u10800 off | **2** | **141.720** | 22.446 | **15.340** | 152.238 |
| Paired EMA999 best u10800 Anchor-LRU | **2** | **141.720** | 22.446 | **15.340** | 152.184 |
| Paired best u10800 EMA-only | **2** | **141.720** | 22.446 | **15.340** | 152.224 |
| Paired current u42000 EMA-only | 84 | 5.002 | 2.889 | 0.576 | 16.906 |
| u11800 + best u10800 embeddings EMA-only | 4 | 85.032 | 22.431 | 7.134 | 129.775 |
| u11800 + current u42000 embeddings EMA-only | 4 | 85.032 | 22.419 | 7.216 | 129.789 |

## PHerc1299

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 26 | 15.078 | 6.361 | 0.318 | 35.558 |
| EMA99 u1800 | 8 | 45.234 | 15.613 | 0.663 | **68.910** |
| EMA99 u5200 | 11 | 33.925 | 15.406 | 0.847 | 39.372 |
| EMA999 u10200 | **5** | **67.851** | **36.132** | 1.607 | 44.400 |
| EMA999 final u11800 | 7 | 50.888 | 29.134 | 1.411 | 40.666 |
| Paired EMA999 u42000 off | 10 | 37.010 | 20.206 | 0.722 | 40.480 |
| Paired EMA999 u42000 Anchor-LRU | 10 | 37.010 | 20.206 | 1.849 | 40.482 |
| Paired EMA999 best u10800 off | 11 | 33.925 | 16.752 | 0.542 | 40.413 |
| Paired EMA999 best u10800 Anchor-LRU | 9 | 40.711 | 23.673 | **1.869** | 39.502 |
| Paired best u10800 EMA-only | 9 | 40.711 | 23.671 | 1.761 | 39.483 |
| Paired current u42000 EMA-only | 9 | 40.711 | 16.729 | 0.796 | 44.491 |
| u11800 + best u10800 embeddings EMA-only | 10 | 37.010 | 15.988 | 1.671 | 40.482 |
| u11800 + current u42000 embeddings EMA-only | 9 | 40.711 | 21.474 | 1.587 | 40.531 |

## PHerc1451

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 2 | 16.123 | 11.053 | 1.746 | 11.348 |
| EMA99 u1800 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| EMA99 u5200 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| EMA999 u10200 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| EMA999 final u11800 | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Paired EMA999 u42000 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Paired EMA999 u42000 Anchor-LRU | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Paired EMA999 best u10800 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Paired EMA999 best u10800 Anchor-LRU | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Paired best u10800 EMA-only | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Paired current u42000 EMA-only | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| u11800 + best u10800 embeddings EMA-only | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| u11800 + current u42000 embeddings EMA-only | **0** | **48.369** | **12.092** | **11.129** | **13.056** |

## PHercParis4

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 147 | 32.516 | 8.894 | 0.196 | 85.487 |
| EMA99 u1800 | 91 | 52.309 | 11.289 | 0.271 | 89.582 |
| EMA99 u5200 | 69 | 68.749 | 11.903 | 0.212 | 109.763 |
| EMA999 u10200 | 57 | 82.973 | 15.545 | 0.271 | 112.997 |
| EMA999 final u11800 | 52 | 90.801 | **17.375** | 0.271 | 112.570 |
| Paired EMA999 u42000 off | 58 | 81.567 | 11.636 | 0.174 | 113.023 |
| Paired EMA999 u42000 Anchor-LRU | 58 | 81.567 | 14.660 | 0.174 | 112.370 |
| Paired EMA999 best u10800 off | 63 | 75.194 | 13.677 | 0.197 | 112.370 |
| Paired EMA999 best u10800 Anchor-LRU | 64 | 74.037 | 14.681 | 0.271 | 112.242 |
| Paired best u10800 EMA-only | 60 | 78.892 | 14.169 | 0.260 | 112.693 |
| Paired current u42000 EMA-only | 63 | 75.194 | 14.450 | 0.197 | 112.370 |
| u11800 + best u10800 embeddings EMA-only | 56 | 84.429 | 15.128 | **0.296** | 113.284 |
| u11800 + current u42000 embeddings EMA-only | **49** | **96.249** | 17.301 | 0.282 | **113.838** |

## Equal-scroll scores

These give each scroll equal weight, unlike combined restart totals. The Classic-normalized score is the Classic arithmetic mean multiplied by the geometric mean of five model/Classic per-scroll mean ratios; it is not an observed distance.

| New run | Classic-normalized score | Arithmetic mean of scroll means (mm) | Coordinator wall (s) |
|---|---:|---:|---:|
| Paired best u10800 EMA-only | 60.274 | 65.667 | 319.361 |
| Paired current u42000 EMA-only | 31.313 | 38.050 | 386.710 |
| u11800 + best u10800 embeddings EMA-only | 55.816 | 55.318 | 363.870 |
| u11800 + current u42000 embeddings EMA-only | 59.307 | 58.770 | 356.715 |

Wall times include cache reuse and different amounts of extra inference; they are not controlled throughput comparisons. Runs used one all-eight-GPU exclusive manager allocation at priority100, sequentially. Mixed roles partition the GPUs4+4. Original cache namespaces and numerical inference settings are preserved, including256tile/maxbatch8/bf16. No full-volume inference regeneration.

## Exact reproduction commands

Each config is pinned to immutable checkpoints and SHA256 digests. Existing completed output directories must not be overwritten; select new output directories for reruns while retaining cache namespaces. The original commands were:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_paired_best_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_paired_best_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_paired_best_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_paired_current_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_paired_current_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_paired_current_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mixed_best_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mixed_best_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mixed_best_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mixed_current_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mixed_current_ema_only_20260915_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mixed_current_ema_only_20260915_v1.json
```

## Validation and provenance

All runners exited0; all native and coordinator summaries completed without incomplete markers. Verified effective mode/weight/half-life,180cases, equal reference coverage and each underlying cache's checkpoint SHA. For mixed runs, every selected channel path was verified against its intended role cache. Native executable/shared-library and live/mixed runner hashes are in the companion JSON. No code changes during these four evaluations. Prior implementation validation:215Python tests and102native cases passed,8optional tests skipped.

| Run | Presence/nx/ny SHA256 | Embedding SHA256 | Summary |
|---|---|---|---|
| Paired best u10800 EMA-only | `440d8689455023b03993b1f1c8aa5cf47e0b7f02f4523104b85ab6114e8e9dcf` | `440d8689455023b03993b1f1c8aa5cf47e0b7f02f4523104b85ab6114e8e9dcf` | `/ephemeral/hendrik/fiber3d_train2/evaluations/paired_best_ema_only_20260915_v1/full/summary.json` |
| Paired current u42000 EMA-only | `7ee3f7c19d12188a297b87c7ab02dde0de01792dcea97e4c691384240cc17a15` | `7ee3f7c19d12188a297b87c7ab02dde0de01792dcea97e4c691384240cc17a15` | `/ephemeral/hendrik/fiber3d_train2/evaluations/paired_current_ema_only_20260915_v1/full/summary.json` |
| u11800 + best u10800 embeddings EMA-only | `4d7c3b7ed06a29a344f8f21175158bf40c42ade440943ff68aa2775ee0cc1d82` | `440d8689455023b03993b1f1c8aa5cf47e0b7f02f4523104b85ab6114e8e9dcf` | `/ephemeral/hendrik/fiber3d_train2/evaluations/mixed_best_ema_only_20260915_v1/full/summary.json` |
| u11800 + current u42000 embeddings EMA-only | `4d7c3b7ed06a29a344f8f21175158bf40c42ade440943ff68aa2775ee0cc1d82` | `7ee3f7c19d12188a297b87c7ab02dde0de01792dcea97e4c691384240cc17a15` | `/ephemeral/hendrik/fiber3d_train2/evaluations/mixed_current_ema_only_20260915_v1/full/summary.json` |
