# Refinement run212757 — best and latest EMA999

Frozen both before evaluation from `5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260915_212757`: best u22000/2816000samples (denseEMA999loss0.2077809360), current u37100/4748800samples(loss0.2406114705). Both EMAdecay0.999 independently verified. Best is dense-validation selected, not tracing-selected.

Full50originalGT fibers/90eligible spans/180directions/6867.627253directedmm each. No embeddings/rendering. Same CP-bounded/no-Lasagna selection,40/80ellipse,160grace,80restartgap. Mean=combined directed distance/(counted restarts+1); terminal spans included in quantiles. Distances mm. Refined training archives not used as GT.

Bestu22000:99restarts/68.676273mean/11.128854median; new best overall restarts,mean,median in recorded history. Currentu37100:105/64.788936/9.837204. Current vsbest:0139−4restarts,Paris4+10,allothersunchanged. Compared previous leader EMA999u16500:best−4restarts and+4% mean. Single deterministic evaluation per checkpoint; not significance or causal claim.

Timing best:342.674s inference/coordinator,298.286s native. Current:345.486s inference/coordinator,299.883s native. Phases overlap, not additive; outer setup/cleanup excluded. Sequential all8GPUs exclusive manager; cached source data, fresh checkpoint-specific inference. Shared namespace remains isolated by model hash.

best: SHA256 `67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`, output `/ephemeral/hendrik/fiber3d_train2/evaluations/refinement_212757_best_ema999_off_20260916_v1`.

Command: `/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_refinement_212757_best_ema999_off_20260916_v1.json` (freeze-snapshot/prepare preceded full; full validates provenance).

current: SHA256 `60d3b3aee466907d566894ea755ef0015ed314bea4c86541bb90621518661f4b`, output `/ephemeral/hendrik/fiber3d_train2/evaluations/refinement_212757_current_ema999_off_20260916_v1`.

Command: `/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_refinement_212757_current_ema999_off_20260916_v1.json` (freeze-snapshot/prepare preceded full; full validates provenance).

## Combined

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 292 | 23.439 | 5.823 | 0.176 | 62.408 |
| EMA99 u1800 | 174 | 39.244 | 7.626 | 0.197 | 87.637 |
| EMA99 u5200 | 148 | 46.091 | 8.202 | 0.143 | 88.438 |
| EMA999 u10200 | 119 | 57.230 | 9.759 | **0.268** | 90.082 |
| EMA999 final u11800 | 113 | 60.242 | 9.337 | 0.233 | 94.960 |
| Paired EMA999 u42000 off | 205 | 33.338 | 5.191 | 0.200 | 87.839 |
| Paired EMA999 u42000 tuned | 208 | 32.859 | 5.021 | 0.198 | 87.772 |
| Paired EMA999 best u10800 off | 135 | 50.497 | 8.912 | 0.184 | 90.116 |
| Paired EMA999 best u10800 tuned | 139 | 49.054 | 8.920 | 0.197 | 90.098 |
| Refinement EMA999 u14400 off | 111 | 61.318 | 10.512 | 0.211 | 109.668 |
| Refinement 131854 EMA999 u16500 off | 103 | 66.035 | 10.504 | 0.197 | **109.763** |
| Refinement 131854 EMA999 u24200 off | 114 | 59.718 | 9.627 | 0.201 | 104.759 |
| Refinement 185710 best raw u13300 off | 126 | 54.076 | 9.653 | 0.219 | 89.520 |
| Refinement 185710 best EMA99 u17800 off | 106 | 64.183 | 10.856 | 0.220 | 91.374 |
| Refinement 212757 best EMA999 u22000 off | **99** | **68.676** | **11.129** | 0.227 | 100.802 |
| Refinement 212757 current EMA999 u37100 off | 105 | 64.789 | 9.837 | 0.230 | 101.758 |

## PHerc0139

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 108 | 10.776 | 2.235 | 0.079 | 25.657 |
| EMA99 u1800 | 73 | 15.872 | 2.535 | 0.090 | 31.510 |
| EMA99 u5200 | 66 | 17.531 | 2.842 | 0.090 | 39.896 |
| EMA999 u10200 | 55 | 20.974 | 3.324 | **0.227** | 46.612 |
| EMA999 final u11800 | 50 | 23.030 | 3.722 | 0.221 | 45.707 |
| Paired EMA999 u42000 off | 60 | 19.255 | 3.757 | 0.079 | 30.695 |
| Paired EMA999 u42000 tuned | 59 | 19.576 | 3.884 | 0.091 | 30.728 |
| Paired EMA999 best u10800 off | 59 | 19.576 | 3.191 | 0.091 | 46.279 |
| Paired EMA999 best u10800 tuned | 64 | 18.070 | 3.191 | 0.127 | 46.192 |
| Refinement EMA999 u14400 off | 52 | 22.161 | 3.495 | 0.170 | 46.727 |
| Refinement 131854 EMA999 u16500 off | 45 | 25.534 | **4.346** | 0.115 | 48.050 |
| Refinement 131854 EMA999 u24200 off | 51 | 22.588 | 4.205 | 0.148 | 37.379 |
| Refinement 185710 best raw u13300 off | 59 | 19.576 | 3.179 | 0.091 | 39.682 |
| Refinement 185710 best EMA99 u17800 off | 46 | 24.991 | 4.165 | 0.190 | 47.859 |
| Refinement 212757 best EMA999 u22000 off | 44 | 26.101 | 3.875 | 0.157 | 48.072 |
| Refinement 212757 current EMA999 u37100 off | **40** | **28.648** | 4.269 | 0.094 | **48.689** |

## PHerc0332

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 9 | 42.516 | 16.777 | 0.923 | 84.676 |
| EMA99 u1800 | **2** | **141.720** | 22.319 | **15.340** | **152.532** |
| EMA99 u5200 | **2** | **141.720** | 22.354 | **15.340** | 152.481 |
| EMA999 u10200 | **2** | **141.720** | 22.467 | **15.340** | 152.156 |
| EMA999 final u11800 | 4 | 85.032 | 22.429 | 7.075 | 129.835 |
| Paired EMA999 u42000 off | 77 | 5.451 | 2.774 | 0.521 | 16.777 |
| Paired EMA999 u42000 tuned | 81 | 5.185 | 2.764 | 0.428 | 16.777 |
| Paired EMA999 best u10800 off | **2** | **141.720** | 22.446 | **15.340** | 152.238 |
| Paired EMA999 best u10800 tuned | **2** | **141.720** | 22.446 | **15.340** | 152.184 |
| Refinement EMA999 u14400 off | **2** | **141.720** | 22.416 | **15.340** | 152.399 |
| Refinement 131854 EMA999 u16500 off | 3 | 106.290 | 16.777 | 6.354 | 152.190 |
| Refinement 131854 EMA999 u24200 off | **2** | **141.720** | 22.420 | **15.340** | 152.383 |
| Refinement 185710 best raw u13300 off | 3 | 106.290 | **28.101** | **15.340** | 132.317 |
| Refinement 185710 best EMA99 u17800 off | 3 | 106.290 | 16.777 | 6.403 | 152.271 |
| Refinement 212757 best EMA999 u22000 off | 3 | 106.290 | 16.777 | 6.426 | 152.175 |
| Refinement 212757 current EMA999 u37100 off | 3 | 106.290 | 16.777 | 6.407 | 152.179 |

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
| Paired EMA999 best u10800 tuned | 9 | 40.711 | 23.673 | 1.869 | 39.502 |
| Refinement EMA999 u14400 off | **5** | **67.851** | 28.974 | **7.102** | 41.432 |
| Refinement 131854 EMA999 u16500 off | **5** | **67.851** | 29.409 | 1.844 | 41.425 |
| Refinement 131854 EMA999 u24200 off | 7 | 50.888 | 27.694 | 0.261 | 40.672 |
| Refinement 185710 best raw u13300 off | 6 | 58.158 | 26.519 | 1.944 | 41.264 |
| Refinement 185710 best EMA99 u17800 off | **5** | **67.851** | 29.553 | 2.443 | 40.792 |
| Refinement 212757 best EMA999 u22000 off | 7 | 50.888 | 27.712 | 0.955 | 40.664 |
| Refinement 212757 current EMA999 u37100 off | 7 | 50.888 | 27.681 | 1.681 | 40.670 |

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
| Refinement EMA999 u14400 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Refinement 131854 EMA999 u16500 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Refinement 131854 EMA999 u24200 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Refinement 185710 best raw u13300 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Refinement 185710 best EMA99 u17800 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Refinement 212757 best EMA999 u22000 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |
| Refinement 212757 current EMA999 u37100 off | **0** | **48.369** | **12.092** | **11.129** | **13.056** |

## PHercParis4

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 147 | 32.516 | 8.894 | 0.196 | 85.487 |
| EMA99 u1800 | 91 | 52.309 | 11.289 | 0.271 | 89.582 |
| EMA99 u5200 | 69 | 68.749 | 11.903 | 0.212 | 109.763 |
| EMA999 u10200 | 57 | 82.973 | 15.545 | 0.271 | 112.997 |
| EMA999 final u11800 | 52 | 90.801 | 17.375 | 0.271 | 112.570 |
| Paired EMA999 u42000 off | 58 | 81.567 | 11.636 | 0.174 | 113.023 |
| Paired EMA999 u42000 tuned | 58 | 81.567 | 14.660 | 0.174 | 112.370 |
| Paired EMA999 best u10800 off | 63 | 75.194 | 13.677 | 0.197 | 112.370 |
| Paired EMA999 best u10800 tuned | 64 | 74.037 | 14.681 | 0.271 | 112.242 |
| Refinement EMA999 u14400 off | 52 | 90.801 | 15.345 | **0.284** | 112.370 |
| Refinement 131854 EMA999 u16500 off | 50 | 94.362 | 15.564 | 0.223 | **113.742** |
| Refinement 131854 EMA999 u24200 off | 54 | 87.499 | 14.620 | 0.244 | 113.546 |
| Refinement 185710 best raw u13300 off | 58 | 81.567 | 15.608 | 0.271 | 109.975 |
| Refinement 185710 best EMA99 u17800 off | 52 | 90.801 | 16.969 | 0.271 | 112.370 |
| Refinement 212757 best EMA999 u22000 off | **45** | **104.618** | **17.398** | 0.281 | 112.145 |
| Refinement 212757 current EMA999 u37100 off | 55 | 85.936 | 11.638 | 0.271 | 112.370 |
