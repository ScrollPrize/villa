# INCOMPLETE MATRIX

Pending: reference6500_off, best9900_tuned, best9900_off, current32000_tuned, current32000_off

# Equal-scroll tracing comparison

Overall: Classic arithmetic mean × geometric mean of five model/Classic scroll ratios (**20% each**).
Classic scale = 5.722681385mm. This normalized score is not an observed mean distance.
Lengths are mm. Best/tied values are bold; old pooled overall scores are omitted.
Classic is longer-trained; modern checkpoint ages are recorded below or in their source reports.
This comparison does not match training budgets.

## Overall

| Model | Classic-normalized score (mm) ↑ |
|---|---:|
| Classic | 5.723 |
| NML raw (best test loss; test=2.388740) — 358,400 samples / u2,800 | 2.473 |
| NML EMA99 (best test loss; test=2.491809) — 384,000 samples / u3,000 | 2.935 |
| JSON raw (best test loss; test=0.405784) — 537,600 samples / u4,200 | 4.353 |
| JSON EMA99 (best test loss; test=0.389835) — 588,800 samples / u4,600 | 5.056 |
| Combined raw (best test loss; test=0.412379) — 588,800 samples / u4,600 | 4.676 |
| Combined EMA99 (best test loss; test=0.400296) — 588,800 samples / u4,600 | 5.267 |
| Combined EMA999 (latest/current; test=0.386195) — 1,446,400 samples / u11,300 | 5.612 |
| Combined EMA999 (best test loss; test=0.385559) — 1,254,400 samples / u9,800 | 5.456 |
| Combined EMA99 (best test loss; test=0.381921) — 1,254,400 samples / u9,800 | 5.592 |
| Combined EMA99 (best test loss; test=0.373058) — 1,881,600 samples / u14,700 | 5.701 |
| Combined EMA999 (best test loss; test=0.380882) — 1,920,000 samples / u15,000 | **5.790** |
| Combined EMA999 (latest/current; test=0.390237) — 2,649,600 samples / u20,700 | 5.686 |
| Combined EMA99 (latest/current; test=0.387144) — 2,649,600 samples / u20,700 | 5.514 |
| Combined raw (latest/current; test=0.403531) — 2,649,600 samples / u20,700 | 4.728 |
| Combined finetune EMA999 (latest/current; test=0.454956) — 6,336,000 samples / u49,500 | 5.724 |
| Combined finetune EMA99 (latest/current; test=0.455648) — 6,336,000 samples / u49,500 | 5.665 |
| Combined finetune raw (latest/current; test=0.468491) — 6,336,000 samples / u49,500 | 5.360 |
| Combined finetune raw (best test loss; test=0.378411) — 2,764,800 samples / u21,600 | 5.589 |
| Combined finetune EMA99 (best test loss; test=0.383839) — 2,803,200 samples / u21,900 | 5.324 |
| Combined finetune EMA999 (best test loss; test=0.388823) — 2,995,200 samples / u23,400 | 5.486 |
| Collection17k | 3.566 |
| Collection3k | 4.311 |
| Collection5k | 4.419 |
| Collection28k | 4.872 |
| Collection11k | 5.347 |
| Embedding EMA99 reference6500_tuned | 4.695 |

## PHerc0139 — 10 fibers

20 directions; 1392.347mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 564 | 2.384 | **0.175** | 0.0071 | 11.631 | 8 |
| NML raw (best test loss; test=2.388740) — 358,400 samples / u2,800 | 902 | 1.510 | 0.078 | 0.0068 | 9.693 | 10 |
| NML EMA99 (best test loss; test=2.491809) — 384,000 samples / u3,000 | 750 | 1.808 | 0.082 | 0.0053 | 9.358 | 8 |
| JSON raw (best test loss; test=0.405784) — 537,600 samples / u4,200 | 485 | 2.757 | 0.097 | 0.0052 | 15.315 | 8 |
| JSON EMA99 (best test loss; test=0.389835) — 588,800 samples / u4,600 | 437 | 3.047 | 0.106 | 0.0050 | 15.933 | 13 |
| Combined raw (best test loss; test=0.412379) — 588,800 samples / u4,600 | 448 | 2.975 | 0.133 | 0.0069 | 15.984 | 4 |
| Combined EMA99 (best test loss; test=0.400296) — 588,800 samples / u4,600 | 459 | 2.907 | 0.094 | 0.0061 | 15.596 | 7 |
| Combined EMA999 (latest/current; test=0.386195) — 1,446,400 samples / u11,300 | 426 | 3.122 | 0.081 | 0.0064 | 16.317 | 5 |
| Combined EMA999 (best test loss; test=0.385559) — 1,254,400 samples / u9,800 | 416 | 3.193 | 0.093 | 0.0061 | 17.727 | 7 |
| Combined EMA99 (best test loss; test=0.381921) — 1,254,400 samples / u9,800 | 399 | 3.323 | 0.099 | 0.0074 | 18.464 | 7 |
| Combined EMA99 (best test loss; test=0.373058) — 1,881,600 samples / u14,700 | 391 | 3.388 | 0.096 | 0.0065 | 16.007 | 3 |
| Combined EMA999 (best test loss; test=0.380882) — 1,920,000 samples / u15,000 | 427 | 3.115 | 0.085 | 0.0040 | 15.627 | 7 |
| Combined EMA999 (latest/current; test=0.390237) — 2,649,600 samples / u20,700 | 392 | 3.379 | 0.097 | 0.0070 | 16.037 | 9 |
| Combined EMA99 (latest/current; test=0.387144) — 2,649,600 samples / u20,700 | 430 | 3.094 | 0.072 | 0.0061 | 15.872 | 5 |
| Combined raw (latest/current; test=0.403531) — 2,649,600 samples / u20,700 | 485 | 2.757 | 0.082 | 0.0052 | 12.244 | 9 |
| Combined finetune EMA999 (latest/current; test=0.454956) — 6,336,000 samples / u49,500 | 368 | 3.589 | 0.100 | 0.0057 | 18.549 | 4 |
| Combined finetune EMA99 (latest/current; test=0.455648) — 6,336,000 samples / u49,500 | 368 | 3.589 | 0.100 | 0.0060 | 15.885 | 5 |
| Combined finetune raw (latest/current; test=0.468491) — 6,336,000 samples / u49,500 | **363** | **3.635** | 0.101 | 0.0057 | **19.800** | 6 |
| Combined finetune raw (best test loss; test=0.378411) — 2,764,800 samples / u21,600 | 471 | 2.836 | 0.056 | 0.0073 | 15.833 | 9 |
| Combined finetune EMA99 (best test loss; test=0.383839) — 2,803,200 samples / u21,900 | 431 | 3.087 | 0.079 | 0.0040 | 15.856 | 9 |
| Combined finetune EMA999 (best test loss; test=0.388823) — 2,995,200 samples / u23,400 | 431 | 3.087 | 0.074 | 0.0062 | 14.008 | 5 |
| Collection17k | 523 | 2.564 | 0.072 | 0.0042 | 11.683 | 11 |
| Collection3k | 548 | 2.451 | 0.077 | **0.0083** | 14.483 | 8 |
| Collection5k | 476 | 2.807 | 0.115 | 0.0062 | 14.985 | 4 |
| Collection28k | 439 | 3.033 | 0.078 | 0.0082 | 16.510 | **2** |
| Collection11k | 451 | 2.956 | 0.096 | 0.0059 | 15.611 | 6 |
| Embedding EMA99 reference6500_tuned | 544 | 2.469 | 0.065 | 0.0046 | 14.125 | 11 |

## PHerc0332 — 3 fibers

6 directions; 471.222mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 27 | 14.279 | 4.206 | 0.0074 | 48.805 | 1 |
| NML raw (best test loss; test=2.388740) — 358,400 samples / u2,800 | 69 | 6.283 | 2.909 | **0.0249** | 22.058 | **0** |
| NML EMA99 (best test loss; test=2.491809) — 384,000 samples / u3,000 | 52 | 8.125 | 0.250 | 0.0079 | 35.816 | **0** |
| JSON raw (best test loss; test=0.405784) — 537,600 samples / u4,200 | 26 | 14.726 | 1.999 | 0.0071 | 60.832 | **0** |
| JSON EMA99 (best test loss; test=0.389835) — 588,800 samples / u4,600 | 17 | 20.488 | 8.486 | 0.0092 | 69.482 | **0** |
| Combined raw (best test loss; test=0.412379) — 588,800 samples / u4,600 | 23 | 16.249 | 2.528 | 0.0057 | 58.601 | **0** |
| Combined EMA99 (best test loss; test=0.400296) — 588,800 samples / u4,600 | 19 | 18.849 | 10.735 | 0.0037 | 67.689 | **0** |
| Combined EMA999 (latest/current; test=0.386195) — 1,446,400 samples / u11,300 | 20 | 18.124 | 5.760 | 0.0071 | 68.505 | **0** |
| Combined EMA999 (best test loss; test=0.385559) — 1,254,400 samples / u9,800 | 20 | 18.124 | 3.823 | 0.0087 | 68.494 | **0** |
| Combined EMA99 (best test loss; test=0.381921) — 1,254,400 samples / u9,800 | 19 | 18.849 | 2.099 | 0.0112 | 69.754 | 1 |
| Combined EMA99 (best test loss; test=0.373058) — 1,881,600 samples / u14,700 | 19 | 18.849 | 8.471 | 0.0034 | 107.402 | **0** |
| Combined EMA999 (best test loss; test=0.380882) — 1,920,000 samples / u15,000 | 17 | 20.488 | 7.823 | 0.0116 | 115.866 | 1 |
| Combined EMA999 (latest/current; test=0.390237) — 2,649,600 samples / u20,700 | 19 | 18.849 | 4.019 | 0.0080 | 107.369 | **0** |
| Combined EMA99 (latest/current; test=0.387144) — 2,649,600 samples / u20,700 | 16 | 21.419 | 4.263 | 0.0049 | 120.057 | **0** |
| Combined raw (latest/current; test=0.403531) — 2,649,600 samples / u20,700 | **15** | **22.439** | 2.185 | 0.0041 | 124.398 | **0** |
| Combined finetune EMA999 (latest/current; test=0.454956) — 6,336,000 samples / u49,500 | 20 | 18.124 | 11.149 | 0.0030 | 100.355 | 1 |
| Combined finetune EMA99 (latest/current; test=0.455648) — 6,336,000 samples / u49,500 | 18 | 19.634 | 11.140 | 0.0028 | 95.742 | 1 |
| Combined finetune raw (latest/current; test=0.468491) — 6,336,000 samples / u49,500 | 20 | 18.124 | 8.509 | 0.0046 | 89.157 | **0** |
| Combined finetune raw (best test loss; test=0.378411) — 2,764,800 samples / u21,600 | 16 | 21.419 | 1.383 | 0.0031 | 120.075 | 1 |
| Combined finetune EMA99 (best test loss; test=0.383839) — 2,803,200 samples / u21,900 | 16 | 21.419 | 7.696 | 0.0035 | 73.563 | **0** |
| Combined finetune EMA999 (best test loss; test=0.388823) — 2,995,200 samples / u23,400 | 17 | 20.488 | 10.923 | 0.0051 | 115.824 | **0** |
| Collection17k | 34 | 11.781 | 1.092 | 0.0086 | 53.930 | 1 |
| Collection3k | 23 | 16.249 | 5.662 | 0.0057 | 55.703 | 1 |
| Collection5k | 20 | 18.124 | 3.140 | 0.0022 | 102.431 | **0** |
| Collection28k | **15** | **22.439** | **15.340** | 0.0080 | 105.638 | 1 |
| Collection11k | 17 | 20.488 | 9.718 | 0.0087 | **139.852** | **0** |
| Embedding EMA99 reference6500_tuned | 19 | 18.849 | 10.669 | 0.0067 | 69.759 | **0** |

## PHerc1299 — 5 fibers

10 directions; 599.748mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | **159** | **3.549** | **0.803** | **0.0315** | 17.573 | 1 |
| NML raw (best test loss; test=2.388740) — 358,400 samples / u2,800 | 344 | 1.694 | 0.260 | 0.0068 | 9.602 | 5 |
| NML EMA99 (best test loss; test=2.491809) — 384,000 samples / u3,000 | 318 | 1.828 | 0.248 | 0.0092 | 9.318 | 6 |
| JSON raw (best test loss; test=0.405784) — 537,600 samples / u4,200 | 225 | 2.552 | 0.223 | 0.0095 | 11.192 | 1 |
| JSON EMA99 (best test loss; test=0.389835) — 588,800 samples / u4,600 | 233 | 2.468 | 0.217 | 0.0119 | 11.649 | 2 |
| Combined raw (best test loss; test=0.412379) — 588,800 samples / u4,600 | 250 | 2.307 | 0.184 | 0.0094 | 11.731 | 3 |
| Combined EMA99 (best test loss; test=0.400296) — 588,800 samples / u4,600 | 216 | 2.654 | 0.257 | 0.0116 | 14.332 | 3 |
| Combined EMA999 (latest/current; test=0.386195) — 1,446,400 samples / u11,300 | 202 | 2.829 | 0.324 | 0.0082 | 13.871 | 2 |
| Combined EMA999 (best test loss; test=0.385559) — 1,254,400 samples / u9,800 | 209 | 2.739 | 0.282 | 0.0086 | 13.839 | **0** |
| Combined EMA99 (best test loss; test=0.381921) — 1,254,400 samples / u9,800 | 194 | 2.940 | 0.297 | 0.0136 | 14.849 | 3 |
| Combined EMA99 (best test loss; test=0.373058) — 1,881,600 samples / u14,700 | 206 | 2.777 | 0.257 | 0.0137 | 13.963 | 2 |
| Combined EMA999 (best test loss; test=0.380882) — 1,920,000 samples / u15,000 | 214 | 2.677 | 0.238 | 0.0073 | 15.534 | 3 |
| Combined EMA999 (latest/current; test=0.390237) — 2,649,600 samples / u20,700 | 210 | 2.726 | 0.276 | 0.0088 | 14.019 | 1 |
| Combined EMA99 (latest/current; test=0.387144) — 2,649,600 samples / u20,700 | 197 | 2.897 | 0.321 | 0.0085 | 15.366 | 3 |
| Combined raw (latest/current; test=0.403531) — 2,649,600 samples / u20,700 | 218 | 2.630 | 0.284 | 0.0114 | 14.744 | 1 |
| Combined finetune EMA999 (latest/current; test=0.454956) — 6,336,000 samples / u49,500 | 180 | 3.157 | 0.387 | 0.0094 | **19.955** | 3 |
| Combined finetune EMA99 (latest/current; test=0.455648) — 6,336,000 samples / u49,500 | 178 | 3.190 | 0.346 | 0.0095 | 17.286 | 1 |
| Combined finetune raw (latest/current; test=0.468491) — 6,336,000 samples / u49,500 | 185 | 3.076 | 0.269 | 0.0085 | 19.743 | 4 |
| Combined finetune raw (best test loss; test=0.378411) — 2,764,800 samples / u21,600 | 210 | 2.726 | 0.255 | 0.0077 | 14.011 | 3 |
| Combined finetune EMA99 (best test loss; test=0.383839) — 2,803,200 samples / u21,900 | 197 | 2.897 | 0.283 | 0.0059 | 17.313 | 3 |
| Combined finetune EMA999 (best test loss; test=0.388823) — 2,995,200 samples / u23,400 | 201 | 2.842 | 0.256 | 0.0068 | 15.732 | 5 |
| Collection17k | 359 | 1.625 | 0.084 | 0.0097 | 10.229 | 1 |
| Collection3k | 261 | 2.213 | 0.211 | 0.0068 | 12.068 | 4 |
| Collection5k | 224 | 2.563 | 0.226 | 0.0134 | 15.594 | 1 |
| Collection28k | 250 | 2.307 | 0.134 | 0.0113 | 10.763 | 3 |
| Collection11k | 212 | 2.702 | 0.235 | 0.0134 | 13.616 | 4 |
| Embedding EMA99 reference6500_tuned | 195 | 2.926 | 0.348 | 0.0109 | 14.511 | 2 |

## PHerc1451 — 2 fibers

4 directions; 94.430mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | **19** | **4.106** | **1.862** | **0.0707** | 17.222 | **0** |
| NML raw (best test loss; test=2.388740) — 358,400 samples / u2,800 | 83 | 1.085 | 0.079 | 0.0087 | 5.855 | 2 |
| NML EMA99 (best test loss; test=2.491809) — 384,000 samples / u3,000 | 72 | 1.243 | 0.156 | 0.0097 | 6.246 | **0** |
| JSON raw (best test loss; test=0.405784) — 537,600 samples / u4,200 | 43 | 2.009 | 0.217 | 0.0111 | 11.336 | **0** |
| JSON EMA99 (best test loss; test=0.389835) — 588,800 samples / u4,600 | 37 | 2.303 | 0.268 | 0.0192 | 11.598 | **0** |
| Combined raw (best test loss; test=0.412379) — 588,800 samples / u4,600 | 36 | 2.361 | 0.436 | 0.0452 | 11.234 | **0** |
| Combined EMA99 (best test loss; test=0.400296) — 588,800 samples / u4,600 | 39 | 2.196 | 0.262 | 0.0285 | 11.056 | **0** |
| Combined EMA999 (latest/current; test=0.386195) — 1,446,400 samples / u11,300 | 30 | 2.777 | 0.802 | 0.0117 | 13.449 | **0** |
| Combined EMA999 (best test loss; test=0.385559) — 1,254,400 samples / u9,800 | 36 | 2.361 | 0.211 | 0.0105 | **17.278** | **0** |
| Combined EMA99 (best test loss; test=0.381921) — 1,254,400 samples / u9,800 | 38 | 2.248 | 0.193 | 0.0106 | 11.171 | **0** |
| Combined EMA99 (best test loss; test=0.373058) — 1,881,600 samples / u14,700 | 32 | 2.623 | 0.278 | 0.0137 | 17.268 | **0** |
| Combined EMA999 (best test loss; test=0.380882) — 1,920,000 samples / u15,000 | 28 | 2.951 | 0.544 | 0.0139 | 17.270 | **0** |
| Combined EMA999 (latest/current; test=0.390237) — 2,649,600 samples / u20,700 | 34 | 2.485 | 0.283 | 0.0131 | 14.152 | **0** |
| Combined EMA99 (latest/current; test=0.387144) — 2,649,600 samples / u20,700 | 37 | 2.303 | 0.115 | 0.0063 | 11.439 | 1 |
| Combined raw (latest/current; test=0.403531) — 2,649,600 samples / u20,700 | 51 | 1.717 | 0.094 | 0.0040 | 11.093 | **0** |
| Combined finetune EMA999 (latest/current; test=0.454956) — 6,336,000 samples / u49,500 | 40 | 2.146 | 0.362 | 0.0160 | 11.001 | **0** |
| Combined finetune EMA99 (latest/current; test=0.455648) — 6,336,000 samples / u49,500 | 41 | 2.098 | 0.380 | 0.0149 | 10.965 | **0** |
| Combined finetune raw (latest/current; test=0.468491) — 6,336,000 samples / u49,500 | 46 | 1.889 | 0.240 | 0.0059 | 10.952 | **0** |
| Combined finetune raw (best test loss; test=0.378411) — 2,764,800 samples / u21,600 | 34 | 2.485 | 0.235 | 0.0102 | 12.389 | **0** |
| Combined finetune EMA99 (best test loss; test=0.383839) — 2,803,200 samples / u21,900 | 36 | 2.361 | 0.209 | 0.0069 | 11.588 | **0** |
| Combined finetune EMA999 (best test loss; test=0.388823) — 2,995,200 samples / u23,400 | 34 | 2.485 | 0.299 | 0.0080 | 12.329 | **0** |
| Collection17k | 61 | 1.453 | 0.231 | 0.0133 | 8.586 | **0** |
| Collection3k | 64 | 1.389 | 0.092 | 0.0067 | 9.410 | **0** |
| Collection5k | 65 | 1.369 | 0.116 | 0.0078 | 9.365 | **0** |
| Collection28k | 43 | 2.009 | 0.176 | 0.0143 | 9.265 | **0** |
| Collection11k | 32 | 2.623 | 0.864 | 0.0145 | 11.192 | 1 |
| Embedding EMA99 reference6500_tuned | 59 | 1.499 | 0.147 | 0.0084 | 9.319 | **0** |

## PHercParis4 — 30 fibers

60 directions; 5523.811mm directed reference.
Total segments = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 1226 | 4.295 | **0.185** | 0.0128 | 23.435 | 12 |
| NML raw (best test loss; test=2.388740) — 358,400 samples / u2,800 | 2943 | 1.839 | 0.092 | 0.0105 | 9.102 | 19 |
| NML EMA99 (best test loss; test=2.491809) — 384,000 samples / u3,000 | 2380 | 2.264 | 0.061 | 0.0087 | 11.655 | 25 |
| JSON raw (best test loss; test=0.405784) — 537,600 samples / u4,200 | 2059 | 2.607 | 0.034 | **0.0132** | 10.385 | 10 |
| JSON EMA99 (best test loss; test=0.389835) — 588,800 samples / u4,600 | 1649 | 3.232 | 0.048 | 0.0119 | 13.225 | 10 |
| Combined raw (best test loss; test=0.412379) — 588,800 samples / u4,600 | 1814 | 2.948 | 0.066 | 0.0125 | 14.619 | 10 |
| Combined EMA99 (best test loss; test=0.400296) — 588,800 samples / u4,600 | 1194 | 4.405 | 0.145 | 0.0112 | 27.513 | 14 |
| Combined EMA999 (latest/current; test=0.386195) — 1,446,400 samples / u11,300 | 1211 | 4.346 | 0.105 | 0.0087 | 25.911 | 13 |
| Combined EMA999 (best test loss; test=0.385559) — 1,254,400 samples / u9,800 | 1172 | 4.484 | 0.118 | 0.0091 | 28.163 | 17 |
| Combined EMA99 (best test loss; test=0.381921) — 1,254,400 samples / u9,800 | 1145 | 4.584 | 0.148 | 0.0088 | 26.157 | 13 |
| Combined EMA99 (best test loss; test=0.373058) — 1,881,600 samples / u14,700 | 1169 | 4.495 | 0.127 | 0.0112 | 27.594 | 9 |
| Combined EMA999 (best test loss; test=0.380882) — 1,920,000 samples / u15,000 | 1173 | 4.480 | 0.113 | 0.0096 | 25.209 | 10 |
| Combined EMA999 (latest/current; test=0.390237) — 2,649,600 samples / u20,700 | 1095 | 4.783 | 0.134 | 0.0098 | **28.273** | 9 |
| Combined EMA99 (latest/current; test=0.387144) — 2,649,600 samples / u20,700 | 1321 | 4.000 | 0.082 | 0.0106 | 22.394 | 11 |
| Combined raw (latest/current; test=0.403531) — 2,649,600 samples / u20,700 | 1822 | 2.935 | 0.049 | 0.0107 | 13.562 | 17 |
| Combined finetune EMA999 (latest/current; test=0.454956) — 6,336,000 samples / u49,500 | **1081** | **4.841** | 0.139 | 0.0110 | 28.144 | 9 |
| Combined finetune EMA99 (latest/current; test=0.455648) — 6,336,000 samples / u49,500 | 1226 | 4.295 | 0.103 | 0.0101 | 24.316 | 8 |
| Combined finetune raw (latest/current; test=0.468491) — 6,336,000 samples / u49,500 | 1317 | 4.011 | 0.087 | 0.0089 | 23.202 | 15 |
| Combined finetune raw (best test loss; test=0.378411) — 2,764,800 samples / u21,600 | 1141 | 4.599 | 0.127 | 0.0094 | 26.565 | 13 |
| Combined finetune EMA99 (best test loss; test=0.383839) — 2,803,200 samples / u21,900 | 1623 | 3.282 | 0.049 | 0.0075 | 16.699 | 20 |
| Combined finetune EMA999 (best test loss; test=0.388823) — 2,995,200 samples / u23,400 | 1370 | 3.863 | 0.072 | 0.0103 | 22.618 | 11 |
| Collection17k | 1908 | 2.807 | 0.061 | 0.0099 | 10.403 | 13 |
| Collection3k | 1248 | 4.223 | 0.165 | 0.0121 | 25.485 | 12 |
| Collection5k | 1626 | 3.276 | 0.077 | 0.0112 | 17.007 | 11 |
| Collection28k | 1769 | 3.020 | 0.040 | 0.0111 | 12.310 | 7 |
| Collection11k | 1502 | 3.536 | 0.069 | 0.0116 | 18.143 | **6** |
| Embedding EMA99 reference6500_tuned | 1364 | 3.879 | 0.111 | 0.0096 | 20.228 | 11 |

## Checkpoint provenance

| Label | Step | Samples | Decay | Selected because | Test metric | Test loss |
|---|---:|---:|---:|---|---|---:|
| NML raw (best test loss; test=2.388740) — 358,400 samples / u2,800 | 2800 | 358400 | unknown | best test loss | test/loss_total | 2.388740 |
| NML EMA99 (best test loss; test=2.491809) — 384,000 samples / u3,000 | 3000 | 384000 | unknown | best test loss | test_ema_0p99/loss_total | 2.491809 |
| JSON raw (best test loss; test=0.405784) — 537,600 samples / u4,200 | 4200 | 537600 | unknown | best test loss | test/loss_total | 0.405784 |
| JSON EMA99 (best test loss; test=0.389835) — 588,800 samples / u4,600 | 4600 | 588800 | unknown | best test loss | test_ema_0p99/loss_total | 0.389835 |
| Combined raw (best test loss; test=0.412379) — 588,800 samples / u4,600 | 4600 | 588800 | unknown | best test loss | test/loss_total | 0.412379 |
| Combined EMA99 (best test loss; test=0.400296) — 588,800 samples / u4,600 | 4600 | 588800 | unknown | best test loss | test_ema_0p99/loss_total | 0.400296 |
| Combined EMA999 (latest/current; test=0.386195) — 1,446,400 samples / u11,300 | 11300 | 1446400 | 0.999 | latest/current | test_ema_0p999/loss_total | 0.386195 |
| Combined EMA999 (best test loss; test=0.385559) — 1,254,400 samples / u9,800 | 9800 | 1254400 | 0.999 | best test loss | test_ema_0p999/loss_total | 0.385559 |
| Combined EMA99 (best test loss; test=0.381921) — 1,254,400 samples / u9,800 | 9800 | 1254400 | 0.99 | best test loss | test_ema_0p99/loss_total | 0.381921 |
| Combined EMA99 (best test loss; test=0.373058) — 1,881,600 samples / u14,700 | 14700 | 1881600 | 0.99 | best test loss | test_ema_0p99/loss_total | 0.373058 |
| Combined EMA999 (best test loss; test=0.380882) — 1,920,000 samples / u15,000 | 15000 | 1920000 | 0.999 | best test loss | test_ema_0p999/loss_total | 0.380882 |
| Combined EMA999 (latest/current; test=0.390237) — 2,649,600 samples / u20,700 | 20700 | 2649600 | 0.999 | latest/current | test_ema_0p999/loss_total | 0.390237 |
| Combined EMA99 (latest/current; test=0.387144) — 2,649,600 samples / u20,700 | 20700 | 2649600 | 0.99 | latest/current | test_ema_0p99/loss_total | 0.387144 |
| Combined raw (latest/current; test=0.403531) — 2,649,600 samples / u20,700 | 20700 | 2649600 | raw | latest/current | test/loss_total | 0.403531 |
| Combined finetune EMA999 (latest/current; test=0.454956) — 6,336,000 samples / u49,500 | 49500 | 6336000 | 0.999 | latest/current | test_ema_0p999/loss_total | 0.454956 |
| Combined finetune EMA99 (latest/current; test=0.455648) — 6,336,000 samples / u49,500 | 49500 | 6336000 | 0.99 | latest/current | test_ema_0p99/loss_total | 0.455648 |
| Combined finetune raw (latest/current; test=0.468491) — 6,336,000 samples / u49,500 | 49500 | 6336000 | raw | latest/current | test/loss_total | 0.468491 |
| Combined finetune raw (best test loss; test=0.378411) — 2,764,800 samples / u21,600 | 21600 | 2764800 | raw | best test loss | test/loss_total | 0.378411 |
| Combined finetune EMA99 (best test loss; test=0.383839) — 2,803,200 samples / u21,900 | 21900 | 2803200 | 0.99 | best test loss | test_ema_0p99/loss_total | 0.383839 |
| Combined finetune EMA999 (best test loss; test=0.388823) — 2,995,200 samples / u23,400 | 23400 | 2995200 | 0.999 | best test loss | test_ema_0p999/loss_total | 0.388823 |
| Embedding EMA99 reference6500_tuned | 6500 | 832000 | 0.99 | unspecified | test_ema_0p99/loss_total | 1.056982 |

## Evaluation sources

- Classic: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v3_heap`
- NML raw (best test loss; test=2.388740) — 358,400 samples / u2,800: `/ephemeral/hendrik/fiber3d_train2/evaluations/compare_s1a_600k_best_raw_20260912`
  SHA256 `5b359ed861fcccb491d3cf35fd159795aabce762f46ac54237bfdd1dfd6c00c9`.
- NML EMA99 (best test loss; test=2.491809) — 384,000 samples / u3,000: `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/s1a_600k_ema`
  SHA256 `4f7b875e116332202a5eb762c2a7229280a8b7631326e3340f898a76c523cdf7`.
- JSON raw (best test loss; test=0.405784) — 537,600 samples / u4,200: `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/json_600k_raw`
  SHA256 `8e4ed6d8794c6070335c33572fd55bfc2e657cd2e3dcc1cee4885ddce236cdb4`.
- JSON EMA99 (best test loss; test=0.389835) — 588,800 samples / u4,600: `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/json_600k_ema`
  SHA256 `c2c5053f91a4c3ccccfe362795035a1c9c0e18e1fdf1b125984d96e71bffe4d8`.
- Combined raw (best test loss; test=0.412379) — 588,800 samples / u4,600: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_600k_20260912/raw`
  SHA256 `5f7bccacb390174da74feafd76a3edfbfd880a59bee97efac573f56633f42314`.
- Combined EMA99 (best test loss; test=0.400296) — 588,800 samples / u4,600: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_600k_20260912/ema`
  SHA256 `3c9766fc5d9f4e41686cbb6d4d780369c4a865ea7c13b4c6889636e015558124`.
- Combined EMA999 (latest/current; test=0.386195) — 1,446,400 samples / u11,300: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_three_ema_20260912/current999`
  SHA256 `3e87efbcb793371b7c6a047081503bc1c9cbfd2315fea2d72447fa4f7a9d9efb`.
- Combined EMA999 (best test loss; test=0.385559) — 1,254,400 samples / u9,800: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_three_ema_20260912/best999`
  SHA256 `7a4ac5025f2184ef1803790d08c7dc84ee14b33e3cea8a2e34c2c9ff5ab07d42`.
- Combined EMA99 (best test loss; test=0.381921) — 1,254,400 samples / u9,800: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_three_ema_20260912/best99`
  SHA256 `ab66f474550eaf27d60e63b29b2ccb3717f95ded0fd5c9d1724927ee24164fd9`.
- Combined EMA99 (best test loss; test=0.373058) — 1,881,600 samples / u14,700: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_best_pair_20260912_19m/best99-retry1`
  SHA256 `dffd46a25b0ecca7f54373736b412cfbb92119f5c26c936ef7e3d968b4d99abf`.
- Combined EMA999 (best test loss; test=0.380882) — 1,920,000 samples / u15,000: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_best_pair_20260912_19m/best999`
  SHA256 `a55fd90653780ba7e37b3e838c1186bb62ca1e7884d4ada94de18a941eabf409`.
- Combined EMA999 (latest/current; test=0.390237) — 2,649,600 samples / u20,700: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_current_three_20260912_26m/current999`
  SHA256 `a09b0ae4e25a5845096132a19e01c02cf6d8b276f1e26ba076a9d336a2b8d91e`.
- Combined EMA99 (latest/current; test=0.387144) — 2,649,600 samples / u20,700: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_current_three_20260912_26m/current99`
  SHA256 `0b6aab1354b963a48709126a40d282d41472be7efc016a285989ec21d3831c73`.
- Combined raw (latest/current; test=0.403531) — 2,649,600 samples / u20,700: `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_current_three_20260912_26m/current`
  SHA256 `276c4993788a73c176deb6897bf0ee0dab4578c2b5e39dc4e5a412e19a2f17c3`.
- Combined finetune EMA999 (latest/current; test=0.454956) — 6,336,000 samples / u49,500: `/ephemeral/hendrik/fiber3d_train2/comparisons/finetune_four_20260913/current999`
  SHA256 `141c57afbe1eff8fbc52211cc6b1ac6a02205e7660bd78a5b83d6380f949d810`.
- Combined finetune EMA99 (latest/current; test=0.455648) — 6,336,000 samples / u49,500: `/ephemeral/hendrik/fiber3d_train2/comparisons/finetune_four_20260913/current99`
  SHA256 `31eae592cefa2b670f5bf6a18e2610757484ae9bc1e58f694d2c82451564621c`.
- Combined finetune raw (latest/current; test=0.468491) — 6,336,000 samples / u49,500: `/ephemeral/hendrik/fiber3d_train2/comparisons/finetune_four_20260913/current`
  SHA256 `65a2db4f72c6e1ec849df3d66e71c21243971c1a54c7372b2cf6918b358e143f`.
- Combined finetune raw (best test loss; test=0.378411) — 2,764,800 samples / u21,600: `/ephemeral/hendrik/fiber3d_train2/comparisons/finetune_four_20260913/best`
  SHA256 `7f400b4ea63fdc5db83cf7f73a9c612308079dde720cf1e9fcd408e1d7d8e2c5`.
- Combined finetune EMA99 (best test loss; test=0.383839) — 2,803,200 samples / u21,900: `/ephemeral/hendrik/fiber3d_train2/comparisons/finetune_best_emas_20260913/best99`
  SHA256 `cf18a37c5216361d55c57564f72e44e940c9435cbaa728bad149dede08779a97`.
- Combined finetune EMA999 (best test loss; test=0.388823) — 2,995,200 samples / u23,400: `/ephemeral/hendrik/fiber3d_train2/comparisons/finetune_best_emas_20260913/best999`
  SHA256 `5080f743c5f4f2d0473414f07b3c52ff6b4f1d741580db30284b688771293a34`.
- Collection17k: `/ephemeral/hendrik/fiber3d_train2/evaluations/collection_best17k_full256_b8_20260911_v1`
- Collection3k: `/ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best3k_full256_b8_20260911_v1`
- Collection5k: `/ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best5k_full256_b8_20260911_v1`
- Collection28k: `/ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_current28k_full256_b8_20260912_v1`
- Collection11k: `/ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best11k_full256_b8_20260912_v1`
- Embedding EMA99 reference6500_tuned: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_full_reference6500_tuned_20260914_v2`
  SHA256 `0a2651b711e7e5ee6d445ac8c68ff482c1951798ced2f44896c25602e1c50417`.

Small-scroll support is limited (0332:3fibers;1451:2fibers).
No statistical significance claim. Training is not stopped or restarted by evaluation.

## Execution caveats

New runs use eight shared20GiB GPU grants,96CPU,batch1,float32 accumulation; historical full runs generally used exclusive8GPU,batch8,float16 accumulation. No isolated runtime comparison or bit-identical inference claim. The hard tuning fiber is included in the full cohort. All actual failure positions and trace geometry remain in each new run's full/cases directory.
