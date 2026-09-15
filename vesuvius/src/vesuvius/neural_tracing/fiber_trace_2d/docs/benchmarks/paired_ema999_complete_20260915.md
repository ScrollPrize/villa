# Paired EMA999 matrix — completed 2026-09-15

All four runs completed: 50 fibers, 180 directions, 6867.627253 directed mm. Same CP-bounded/no-Lasagna protocol, ellipse thickness40/width80, grace160 and restart gap80 reference voxels. Mean=combined directed reference distance/(reported restarts+1); quantiles include terminal segments. Distances in mm. Tuned=2 anchors, mean novelty-LRU, spacing512, novelty0.10, weight0.10, nearest touch, no EMA mix. No retuning.

Best paired snapshot u10800 improves substantially over current u42000, but tuned embedding does not improve combined restart counts on either snapshot. Previous non-embedding u11800 remains best combined. 'Best' checkpoint means minimum validation total loss, not trace-score selection. Inference outputs shared within each off/tuned pair, with additional off-path inference as needed. Timings are not directly comparable cold versus warm.

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
| Paired EMA999 best u10800 off | 135 | 50.497 | 8.912 | 0.184 | 90.116 |
| Paired EMA999 best u10800 tuned | 139 | 49.054 | 8.920 | 0.197 | 90.098 |

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
| Paired EMA999 best u10800 off | 59 | 19.576 | 3.191 | 0.091 | 46.279 |
| Paired EMA999 best u10800 tuned | 64 | 18.070 | 3.191 | 0.127 | 46.192 |

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
| Paired EMA999 best u10800 off | **2** | **141.720** | 22.446 | **15.340** | 152.238 |
| Paired EMA999 best u10800 tuned | **2** | **141.720** | 22.446 | **15.340** | 152.184 |

## PHerc1299

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 26 | 15.078 | 6.361 | 0.318 | 35.558 |
| EMA99 u1800 | 8 | 45.234 | 15.613 | 0.663 | **68.910** |
| EMA99 u5200 | 11 | 33.925 | 15.406 | 0.847 | 39.372 |
| EMA999 u10200 | **5** | **67.851** | **36.132** | 1.607 | 44.400 |
| EMA999 final u11800 | 7 | 50.888 | 29.134 | 1.411 | 40.666 |
| Paired EMA999 u42000 off | 10 | 37.010 | 20.206 | 0.722 | 40.480 |
| Paired EMA999 u42000 tuned | 10 | 37.010 | 20.206 | 1.849 | 40.482 |
| Paired EMA999 best u10800 off | 11 | 33.925 | 16.752 | 0.542 | 40.413 |
| Paired EMA999 best u10800 tuned | 9 | 40.711 | 23.673 | **1.869** | 39.502 |

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
| Paired EMA999 best u10800 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Paired EMA999 best u10800 tuned | **0** | **48.369** | **12.092** | **11.129** | **13.056** |

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
| Paired EMA999 best u10800 off | 63 | 75.194 | 13.677 | 0.197 | 112.370 |
| Paired EMA999 best u10800 tuned | 64 | 74.037 | 14.681 | **0.271** | 112.242 |

## Reproduction

Use the approved runner with each of the four eval_paired_ema999_{current,best}_{off,tuned}_20260915_v1.json configs, modes prepare, validate-full, full. Checkpoints/SHA hashes, source mappings and tuned arguments are pinned in those configs. All eight GPUs, exclusive manager priority100. No visualization. Full machine-readable metrics and original summary paths accompany this document.
