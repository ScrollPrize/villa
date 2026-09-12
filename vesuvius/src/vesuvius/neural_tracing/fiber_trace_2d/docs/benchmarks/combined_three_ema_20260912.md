# Equal-scroll tracing comparison

Overall score: arithmetic mean of the five per-scroll means, **20% each**.
Lengths are mm. Best/tied values are bold; old pooled overall scores are omitted.
Classic is longer-trained; earlier modern rows use best checkpoints by600k.
New EMA rows are frozen from the later training run; checkpoint ages are listed below.

## Overall

| Model | Equal-scroll score ↑ |
|---|---:|
| Classic | 5.723 |
| NML raw | 2.482 |
| NML EMA | 3.054 |
| JSON raw | 4.930 |
| JSON EMA | 6.308 |
| Combined raw | 5.368 |
| Combined EMA | 6.202 |
| New current999 | 6.240 |
| New best999 | 6.180 |
| New best99 | **6.389** |

## PHerc0139 — 10 fibers

20 directions; 1392.347mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 564 | 2.384 | **0.175** | 0.0071 | 11.631 | 8 |
| NML raw | 902 | 1.510 | 0.078 | 0.0068 | 9.693 | 10 |
| NML EMA | 750 | 1.808 | 0.082 | 0.0053 | 9.358 | 8 |
| JSON raw | 485 | 2.757 | 0.097 | 0.0052 | 15.315 | 8 |
| JSON EMA | 437 | 3.047 | 0.106 | 0.0050 | 15.933 | 13 |
| Combined raw | 448 | 2.975 | 0.133 | 0.0069 | 15.984 | **4** |
| Combined EMA | 459 | 2.907 | 0.094 | 0.0061 | 15.596 | 7 |
| New current999 | 426 | 3.122 | 0.081 | 0.0064 | 16.317 | 5 |
| New best999 | 416 | 3.193 | 0.093 | 0.0061 | 17.727 | 7 |
| New best99 | **399** | **3.323** | 0.099 | **0.0074** | **18.464** | 7 |

## PHerc0332 — 3 fibers

6 directions; 471.222mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 27 | 14.279 | 4.206 | 0.0074 | 48.805 | 1 |
| NML raw | 69 | 6.283 | 2.909 | **0.0249** | 22.058 | **0** |
| NML EMA | 52 | 8.125 | 0.250 | 0.0079 | 35.816 | **0** |
| JSON raw | 26 | 14.726 | 1.999 | 0.0071 | 60.832 | **0** |
| JSON EMA | **17** | **20.488** | 8.486 | 0.0092 | 69.482 | **0** |
| Combined raw | 23 | 16.249 | 2.528 | 0.0057 | 58.601 | **0** |
| Combined EMA | 19 | 18.849 | **10.735** | 0.0037 | 67.689 | **0** |
| New current999 | 20 | 18.124 | 5.760 | 0.0071 | 68.505 | **0** |
| New best999 | 20 | 18.124 | 3.823 | 0.0087 | 68.494 | **0** |
| New best99 | 19 | 18.849 | 2.099 | 0.0112 | **69.754** | 1 |

## PHerc1299 — 5 fibers

10 directions; 599.748mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | **159** | **3.549** | **0.803** | **0.0315** | **17.573** | 1 |
| NML raw | 344 | 1.694 | 0.260 | 0.0068 | 9.602 | 5 |
| NML EMA | 318 | 1.828 | 0.248 | 0.0092 | 9.318 | 6 |
| JSON raw | 225 | 2.552 | 0.223 | 0.0095 | 11.192 | 1 |
| JSON EMA | 233 | 2.468 | 0.217 | 0.0119 | 11.649 | 2 |
| Combined raw | 250 | 2.307 | 0.184 | 0.0094 | 11.731 | 3 |
| Combined EMA | 216 | 2.654 | 0.257 | 0.0116 | 14.332 | 3 |
| New current999 | 202 | 2.829 | 0.324 | 0.0082 | 13.871 | 2 |
| New best999 | 209 | 2.739 | 0.282 | 0.0086 | 13.839 | **0** |
| New best99 | 194 | 2.940 | 0.297 | 0.0136 | 14.849 | 3 |

## PHerc1451 — 2 fibers

4 directions; 94.430mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | **19** | **4.106** | **1.862** | **0.0707** | 17.222 | **0** |
| NML raw | 83 | 1.085 | 0.079 | 0.0087 | 5.855 | 2 |
| NML EMA | 72 | 1.243 | 0.156 | 0.0097 | 6.246 | **0** |
| JSON raw | 43 | 2.009 | 0.217 | 0.0111 | 11.336 | **0** |
| JSON EMA | 37 | 2.303 | 0.268 | 0.0192 | 11.598 | **0** |
| Combined raw | 36 | 2.361 | 0.436 | 0.0452 | 11.234 | **0** |
| Combined EMA | 39 | 2.196 | 0.262 | 0.0285 | 11.056 | **0** |
| New current999 | 30 | 2.777 | 0.802 | 0.0117 | 13.449 | **0** |
| New best999 | 36 | 2.361 | 0.211 | 0.0105 | **17.278** | **0** |
| New best99 | 38 | 2.248 | 0.193 | 0.0106 | 11.171 | **0** |

## PHercParis4 — 30 fibers

60 directions; 5523.811mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 1226 | 4.295 | **0.185** | 0.0128 | 23.435 | 12 |
| NML raw | 2943 | 1.839 | 0.092 | 0.0105 | 9.102 | 19 |
| NML EMA | 2380 | 2.264 | 0.061 | 0.0087 | 11.655 | 25 |
| JSON raw | 2059 | 2.607 | 0.034 | **0.0132** | 10.385 | **10** |
| JSON EMA | 1649 | 3.232 | 0.048 | 0.0119 | 13.225 | **10** |
| Combined raw | 1814 | 2.948 | 0.066 | 0.0125 | 14.619 | **10** |
| Combined EMA | 1194 | 4.405 | 0.145 | 0.0112 | 27.513 | 14 |
| New current999 | 1211 | 4.346 | 0.105 | 0.0087 | 25.911 | 13 |
| New best999 | 1172 | 4.484 | 0.118 | 0.0091 | **28.163** | 17 |
| New best99 | **1145** | **4.584** | 0.148 | 0.0088 | 26.157 | 13 |

## New checkpoint provenance

| Label | Step | Samples | Decay | Test loss |
|---|---:|---:|---:|---:|
| New current999 | 11300 | 1446400 | 0.999 | 0.386195 |
| New best999 | 9800 | 1254400 | 0.999 | 0.385559 |
| New best99 | 9800 | 1254400 | 0.99 | 0.381921 |

## Evaluation sources

- Classic: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v3_heap`
- NML raw: `/ephemeral/hendrik/fiber3d_train2/evaluations/compare_s1a_600k_best_raw_20260912`
- NML EMA: `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/s1a_600k_ema`
- JSON raw: `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/json_600k_raw`
- JSON EMA: `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/json_600k_ema`
- Combined raw: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_600k_20260912/raw`
- Combined EMA: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_600k_20260912/ema`
- New current999: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_three_ema_20260912/current999`
  SHA256 `3e87efbcb793371b7c6a047081503bc1c9cbfd2315fea2d72447fa4f7a9d9efb`.
- New best999: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_three_ema_20260912/best999`
  SHA256 `7a4ac5025f2184ef1803790d08c7dc84ee14b33e3cea8a2e34c2c9ff5ab07d42`.
- New best99: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_three_ema_20260912/best99`
  SHA256 `ab66f474550eaf27d60e63b29b2ccb3717f95ded0fd5c9d1724927ee24164fd9`.

Small-scroll support is limited (0332:3fibers;1451:2fibers).
No statistical significance claim. Training is not stopped or restarted by evaluation.
