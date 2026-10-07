# Neural followers: full-prefix FW/BW consistency

Same reviewed 91 fibers / 182 directions, native reference acceptance40/80base vx, grace160 and restart grouping80. Confidence and reverse-consistency stops are counted separately from ordinary errors. GT supplies restart locations and traversal sign, not the inferred frame or history. FW/BW rows enable full-prefix checks every512 base voxels plus all terminal endpoints, ellipse20/40, no reverse history. Controls are unfiltered runs. Neural models have no width modulation. The regular EMA999 u12100 control keeps cosine width40% modulation, post-modulated presence cutoff0.4, step6 trace voxels (12base), beam8/lookahead2, with FW/BW disabled.

Flow v5 u26000: confidence0.5, n_commit8, gatefull. Unified supplied checkpoint is u35000 (S3 has no u33000): confidence0.7, n_commit8, gatefull. Both CTlevel2/grid4, tracegrid8, maxlen100000 trace voxels, BF16, seed0. Unified maximum ready-forward batch32, Flow16. Upstream853dcf536 with native-reader injection and commit observer. Noise keys use per-trace seed position/decision, not scheduling; BF16 batch-dependent roundoff remains possible. Historical memory u27000 and Flow v4 u20000 are recomputed from saved traces, not rerun; their original policies/revisions remain in the companion JSON.

Collection-wide seed dedup16 and CP export spacing800 are CLI collection/export operations, not applied across independent benchmark references. Full min0 results respect min-length0; min512 is additional output-only analysis to match the tuned baseline. This filtering is GT-assisted because geometric errors already split outputs. Report confidence stops and coverage, not errors alone.

Neural confidence restarts use the existing native GT restart advancement; the tuned baseline uses its presence-aligned restart policy. This is a comparison of complete policies, not an isolated model substitution. Neural FW/BW filtered stops also restart on the GT arc, not a dense-presence maximum. Reverse confidence exhaustion before reaching the prefix origin fails verification; terminal confidence may therefore become a consistency stop. Reverse seeding uses the actual forward endpoint and tangent only for traversal sign, with a fresh CT heading/frame. No reference path is used in reverse verification.

Mean = retained GT distance / (all surviving individual stops +1). Piece mean = retained GT distance / kept pieces; median is retained GT span per piece. P05/P95 and complete machine-readable metrics are in companion JSON.

| Model | Min vx | Errors | Confidence stops | Presence stops | FW/BW stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned (FW/BW off) | 0 | 0 | 0 | 417 | 0 | 586 | 89.864 | 2.980 | 2.126 | 0.665 |
| EMA999 u12100 tuned (FW/BW off) | 512 | 0 | 0 | 86 | 0 | 233 | 82.086 | 13.079 | 4.884 | 3.814 |
| Neural memory u27000 | 0 | 12 | 1369 | 0 | 0 | 1561 | 99.051 | 0.992 | 0.880 | 0.308 |
| Neural memory u27000 | 512 | 6 | 226 | 0 | 0 | 336 | 76.542 | 4.554 | 3.158 | 2.199 |
| Neural Flow v4 u20000 | 0 | 43 | 772 | 0 | 0 | 1006 | 99.615 | 1.668 | 1.373 | 0.643 |
| Neural Flow v4 u20000 | 512 | 21 | 224 | 0 | 0 | 356 | 85.700 | 4.829 | 3.337 | 2.742 |
| Flow v5 u26000 | 0 | 26 | 2559 | 0 | 0 | 2766 | 97.172 | 0.520 | 0.487 | 0.000 |
| Flow v5 u26000 | 512 | 14 | 140 | 0 | 0 | 296 | 84.840 | 7.587 | 3.973 | 2.771 |
| Unified u35000 | 0 | 24 | 3034 | 0 | 0 | 3234 | 96.320 | 0.436 | 0.413 | 0.000 |
| Unified u35000 | 512 | 13 | 132 | 0 | 0 | 285 | 86.662 | 8.228 | 4.215 | 3.263 |
| Flow v5 + FW/BW | 0 | 8 | 2776 | 0 | 454 | 3394 | 71.448 | 0.306 | 0.292 | 0.000 |
| Flow v5 + FW/BW | 512 | 5 | 78 | 0 | 74 | 254 | 62.789 | 5.509 | 3.427 | 2.485 |
| Unified + FW/BW | 0 | 5 | 3812 | 0 | 391 | 4355 | 70.919 | 0.234 | 0.226 | 0.000 |
| Unified + FW/BW | 512 | 3 | 75 | 0 | 60 | 239 | 65.357 | 6.518 | 3.791 | 2.795 |
| Memory matched control | 0 | 14 | 1369 | 0 | 0 | 1564 | 99.042 | 0.990 | 0.878 | 0.308 |
| Memory matched control | 512 | 6 | 225 | 0 | 0 | 340 | 77.053 | 4.604 | 3.141 | 2.200 |
| Flow v4 matched control | 0 | 43 | 754 | 0 | 0 | 988 | 99.622 | 1.707 | 1.398 | 0.686 |
| Flow v4 matched control | 512 | 23 | 214 | 0 | 0 | 352 | 85.831 | 4.999 | 3.380 | 2.772 |
| Memory + FW/BW | 0 | 3 | 1137 | 0 | 546 | 1829 | 63.940 | 0.525 | 0.485 | 0.000 |
| Memory + FW/BW | 512 | 1 | 84 | 0 | 153 | 287 | 48.426 | 2.809 | 2.339 | 1.711 |
| Flow v4 + FW/BW | 0 | 3 | 528 | 0 | 518 | 1189 | 69.397 | 0.916 | 0.809 | 0.000 |
| Flow v4 + FW/BW | 512 | 1 | 90 | 0 | 148 | 309 | 61.773 | 3.568 | 2.771 | 2.461 |

## PHerc0139

| Model | Min vx | Errors | Confidence stops | Presence stops | FW/BW stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned (FW/BW off) | 0 | 0 | 0 | 234 | 0 | 273 | 91.032 | 1.105 | 0.951 | 0.350 |
| EMA999 u12100 tuned (FW/BW off) | 512 | 0 | 0 | 32 | 0 | 60 | 66.395 | 5.739 | 3.156 | 2.066 |
| Neural memory u27000 | 0 | 3 | 482 | 0 | 0 | 529 | 97.961 | 0.575 | 0.528 | 0.000 |
| Neural memory u27000 | 512 | 1 | 49 | 0 | 0 | 71 | 70.639 | 3.951 | 2.838 | 2.170 |
| Neural Flow v4 u20000 | 0 | 17 | 238 | 0 | 0 | 303 | 99.630 | 1.093 | 0.938 | 0.322 |
| Neural Flow v4 u20000 | 512 | 4 | 51 | 0 | 0 | 79 | 75.642 | 3.853 | 2.731 | 2.460 |
| Flow v5 u26000 | 0 | 9 | 981 | 0 | 0 | 1036 | 94.581 | 0.272 | 0.260 | 0.000 |
| Flow v5 u26000 | 512 | 2 | 28 | 0 | 0 | 61 | 73.072 | 6.723 | 3.417 | 2.732 |
| Unified u35000 | 0 | 9 | 848 | 0 | 0 | 899 | 94.994 | 0.316 | 0.301 | 0.000 |
| Unified u35000 | 512 | 3 | 28 | 0 | 0 | 66 | 81.691 | 7.281 | 3.530 | 2.721 |
| Flow v5 + FW/BW | 0 | 2 | 832 | 0 | 128 | 1001 | 65.654 | 0.194 | 0.187 | 0.000 |
| Flow v5 + FW/BW | 512 | 0 | 16 | 0 | 7 | 45 | 55.370 | 6.580 | 3.509 | 2.732 |
| Unified + FW/BW | 0 | 1 | 897 | 0 | 97 | 1031 | 67.243 | 0.193 | 0.186 | 0.000 |
| Unified + FW/BW | 512 | 0 | 15 | 0 | 12 | 51 | 60.555 | 6.168 | 3.387 | 3.263 |
| Memory matched control | 0 | 5 | 509 | 0 | 0 | 559 | 97.791 | 0.541 | 0.499 | 0.000 |
| Memory matched control | 512 | 1 | 49 | 0 | 0 | 71 | 69.898 | 3.909 | 2.808 | 2.186 |
| Flow v4 matched control | 0 | 19 | 225 | 0 | 0 | 291 | 99.651 | 1.146 | 0.977 | 0.352 |
| Flow v4 matched control | 512 | 6 | 50 | 0 | 0 | 80 | 75.810 | 3.793 | 2.703 | 2.454 |
| Memory + FW/BW | 0 | 0 | 401 | 0 | 135 | 574 | 62.514 | 0.332 | 0.311 | 0.000 |
| Memory + FW/BW | 512 | 0 | 15 | 0 | 30 | 58 | 46.279 | 2.870 | 2.276 | 1.938 |
| Flow v4 + FW/BW | 0 | 1 | 123 | 0 | 135 | 295 | 66.365 | 0.728 | 0.642 | 0.000 |
| Flow v4 + FW/BW | 512 | 0 | 19 | 0 | 31 | 67 | 54.198 | 3.031 | 2.307 | 2.230 |

## PHerc0332

| Model | Min vx | Errors | Confidence stops | Presence stops | FW/BW stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned (FW/BW off) | 0 | 0 | 0 | 7 | 0 | 8 | 31.915 | 0.444 | 0.444 | 0.333 |
| EMA999 u12100 tuned (FW/BW off) | 512 | 0 | 0 | 0 | 0 | 0 | 0.000 | — | — | — |
| Neural memory u27000 | 0 | 0 | 6 | 0 | 0 | 8 | 100.000 | 1.590 | 1.391 | 1.351 |
| Neural memory u27000 | 512 | 0 | 4 | 0 | 0 | 4 | 80.011 | 1.781 | 2.226 | 2.304 |
| Neural Flow v4 u20000 | 0 | 0 | 3 | 0 | 0 | 5 | 100.000 | 2.782 | 2.226 | 2.773 |
| Neural Flow v4 u20000 | 512 | 0 | 3 | 0 | 0 | 4 | 93.244 | 2.594 | 2.594 | 2.782 |
| Flow v5 u26000 | 0 | 0 | 2 | 0 | 0 | 4 | 100.000 | 3.710 | 2.782 | 2.782 |
| Flow v5 u26000 | 512 | 0 | 2 | 0 | 0 | 4 | 100.000 | 3.710 | 2.782 | 2.782 |
| Unified u35000 | 0 | 0 | 1 | 0 | 0 | 3 | 100.000 | 5.564 | 3.710 | 3.531 |
| Unified u35000 | 512 | 0 | 1 | 0 | 0 | 3 | 100.000 | 5.564 | 3.710 | 3.531 |
| Flow v5 + FW/BW | 0 | 0 | 2 | 0 | 3 | 6 | 72.463 | 1.344 | 1.344 | 1.573 |
| Flow v5 + FW/BW | 512 | 0 | 1 | 0 | 2 | 4 | 72.463 | 2.016 | 2.016 | 2.148 |
| Unified + FW/BW | 0 | 0 | 1 | 0 | 1 | 4 | 88.582 | 3.286 | 2.465 | 2.250 |
| Unified + FW/BW | 512 | 0 | 1 | 0 | 1 | 4 | 88.582 | 3.286 | 2.465 | 2.250 |
| Memory matched control | 0 | 0 | 6 | 0 | 0 | 8 | 100.000 | 1.590 | 1.391 | 1.502 |
| Memory matched control | 512 | 0 | 5 | 0 | 0 | 5 | 87.204 | 1.617 | 1.941 | 1.900 |
| Flow v4 matched control | 0 | 0 | 3 | 0 | 0 | 5 | 100.000 | 2.782 | 2.226 | 2.773 |
| Flow v4 matched control | 512 | 0 | 3 | 0 | 0 | 4 | 90.015 | 2.504 | 2.504 | 2.782 |
| Memory + FW/BW | 0 | 0 | 4 | 0 | 4 | 10 | 67.870 | 0.839 | 0.755 | 0.942 |
| Memory + FW/BW | 512 | 0 | 0 | 0 | 2 | 2 | 22.198 | 0.823 | 1.235 | 1.235 |
| Flow v4 + FW/BW | 0 | 0 | 1 | 0 | 2 | 5 | 90.556 | 2.519 | 2.016 | 2.460 |
| Flow v4 + FW/BW | 512 | 0 | 1 | 0 | 2 | 4 | 80.940 | 2.252 | 2.252 | 2.468 |

## PHerc1299

| Model | Min vx | Errors | Confidence stops | Presence stops | FW/BW stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned (FW/BW off) | 0 | 0 | 0 | 2 | 0 | 24 | 99.807 | 49.048 | 6.131 | 5.033 |
| EMA999 u12100 tuned (FW/BW off) | 512 | 0 | 0 | 1 | 0 | 23 | 99.807 | 73.572 | 6.398 | 5.033 |
| Neural memory u27000 | 0 | 3 | 66 | 0 | 0 | 91 | 99.987 | 2.106 | 1.620 | 1.088 |
| Neural memory u27000 | 512 | 3 | 24 | 0 | 0 | 39 | 77.529 | 4.082 | 2.931 | 2.189 |
| Neural Flow v4 u20000 | 0 | 9 | 40 | 0 | 0 | 71 | 100.000 | 2.949 | 2.076 | 2.109 |
| Neural Flow v4 u20000 | 512 | 8 | 30 | 0 | 0 | 50 | 89.615 | 3.388 | 2.642 | 2.391 |
| Flow v5 u26000 | 0 | 6 | 41 | 0 | 0 | 69 | 100.000 | 3.071 | 2.137 | 1.870 |
| Flow v5 u26000 | 512 | 6 | 23 | 0 | 0 | 47 | 92.065 | 4.524 | 2.888 | 2.195 |
| Unified u35000 | 0 | 5 | 42 | 0 | 0 | 68 | 99.857 | 3.067 | 2.165 | 1.560 |
| Unified u35000 | 512 | 5 | 14 | 0 | 0 | 35 | 90.223 | 6.651 | 3.800 | 3.443 |
| Flow v5 + FW/BW | 0 | 4 | 33 | 0 | 33 | 90 | 75.981 | 1.578 | 1.245 | 0.777 |
| Flow v5 + FW/BW | 512 | 4 | 13 | 0 | 10 | 37 | 64.989 | 3.422 | 2.589 | 2.115 |
| Unified + FW/BW | 0 | 2 | 20 | 0 | 36 | 77 | 74.785 | 1.869 | 1.432 | 0.837 |
| Unified + FW/BW | 512 | 2 | 10 | 0 | 9 | 33 | 68.373 | 4.582 | 3.055 | 2.343 |
| Memory matched control | 0 | 4 | 61 | 0 | 0 | 87 | 99.987 | 2.233 | 1.694 | 1.109 |
| Memory matched control | 512 | 3 | 29 | 0 | 0 | 42 | 80.395 | 3.592 | 2.822 | 2.180 |
| Flow v4 matched control | 0 | 8 | 38 | 0 | 0 | 68 | 100.000 | 3.137 | 2.168 | 2.152 |
| Flow v4 matched control | 512 | 8 | 28 | 0 | 0 | 50 | 90.613 | 3.610 | 2.672 | 2.454 |
| Memory + FW/BW | 0 | 1 | 29 | 0 | 49 | 98 | 67.682 | 1.247 | 1.018 | 0.789 |
| Memory + FW/BW | 512 | 1 | 10 | 0 | 18 | 37 | 53.498 | 2.629 | 2.132 | 1.587 |
| Flow v4 + FW/BW | 0 | 1 | 22 | 0 | 42 | 86 | 68.469 | 1.529 | 1.174 | 1.013 |
| Flow v4 + FW/BW | 512 | 1 | 12 | 0 | 21 | 41 | 57.056 | 2.403 | 2.052 | 1.557 |

## PHerc1451

| Model | Min vx | Errors | Confidence stops | Presence stops | FW/BW stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned (FW/BW off) | 0 | 0 | 0 | 3 | 0 | 5 | 95.320 | 2.501 | 2.001 | 1.271 |
| EMA999 u12100 tuned (FW/BW off) | 512 | 0 | 0 | 2 | 0 | 3 | 83.980 | 2.938 | 2.938 | 3.099 |
| Neural memory u27000 | 0 | 0 | 4 | 0 | 0 | 6 | 100.000 | 2.099 | 1.749 | 1.205 |
| Neural memory u27000 | 512 | 0 | 2 | 0 | 0 | 3 | 78.479 | 2.746 | 2.746 | 2.989 |
| Neural Flow v4 u20000 | 0 | 0 | 1 | 0 | 0 | 3 | 100.000 | 5.248 | 3.499 | 4.381 |
| Neural Flow v4 u20000 | 512 | 0 | 1 | 0 | 0 | 2 | 91.740 | 4.814 | 4.814 | 4.814 |
| Flow v5 u26000 | 0 | 0 | 2 | 0 | 0 | 4 | 100.000 | 3.499 | 2.624 | 2.624 |
| Flow v5 u26000 | 512 | 0 | 1 | 0 | 0 | 3 | 93.947 | 4.930 | 3.287 | 3.761 |
| Unified u35000 | 0 | 0 | 0 | 0 | 0 | 2 | 100.000 | 10.496 | 5.248 | 5.248 |
| Unified u35000 | 512 | 0 | 0 | 0 | 0 | 2 | 100.000 | 10.496 | 5.248 | 5.248 |
| Flow v5 + FW/BW | 0 | 0 | 2 | 0 | 3 | 7 | 63.828 | 1.117 | 0.957 | 0.635 |
| Flow v5 + FW/BW | 512 | 0 | 1 | 0 | 1 | 2 | 47.682 | 1.668 | 2.502 | 2.502 |
| Unified + FW/BW | 0 | 0 | 0 | 0 | 0 | 2 | 100.000 | 10.496 | 5.248 | 5.248 |
| Unified + FW/BW | 512 | 0 | 0 | 0 | 0 | 2 | 100.000 | 10.496 | 5.248 | 5.248 |
| Memory matched control | 0 | 0 | 2 | 0 | 0 | 4 | 100.000 | 3.499 | 2.624 | 2.624 |
| Memory matched control | 512 | 0 | 2 | 0 | 0 | 4 | 100.000 | 3.499 | 2.624 | 2.624 |
| Flow v4 matched control | 0 | 0 | 1 | 0 | 0 | 3 | 100.000 | 5.248 | 3.499 | 4.381 |
| Flow v4 matched control | 512 | 0 | 1 | 0 | 0 | 2 | 91.740 | 4.814 | 4.814 | 4.814 |
| Memory + FW/BW | 0 | 0 | 1 | 0 | 3 | 6 | 64.126 | 1.346 | 1.122 | 1.229 |
| Memory + FW/BW | 512 | 0 | 1 | 0 | 3 | 4 | 60.113 | 1.262 | 1.577 | 1.314 |
| Flow v4 + FW/BW | 0 | 0 | 1 | 0 | 1 | 4 | 87.950 | 3.077 | 2.308 | 1.992 |
| Flow v4 + FW/BW | 512 | 0 | 1 | 0 | 1 | 3 | 79.689 | 2.788 | 2.788 | 2.461 |

## PHercParis4

| Model | Min vx | Errors | Confidence stops | Presence stops | FW/BW stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned (FW/BW off) | 0 | 0 | 0 | 171 | 0 | 276 | 88.564 | 4.798 | 2.990 | 1.618 |
| EMA999 u12100 tuned (FW/BW off) | 512 | 0 | 0 | 51 | 0 | 147 | 85.044 | 15.241 | 5.391 | 4.737 |
| Neural memory u27000 | 0 | 6 | 811 | 0 | 0 | 927 | 99.214 | 1.128 | 0.997 | 0.314 |
| Neural memory u27000 | 512 | 2 | 147 | 0 | 0 | 219 | 78.130 | 4.854 | 3.325 | 2.218 |
| Neural Flow v4 u20000 | 0 | 17 | 490 | 0 | 0 | 624 | 99.541 | 1.798 | 1.487 | 0.574 |
| Neural Flow v4 u20000 | 512 | 9 | 139 | 0 | 0 | 221 | 88.001 | 5.504 | 3.711 | 2.988 |
| Flow v5 u26000 | 0 | 11 | 1533 | 0 | 0 | 1653 | 97.452 | 0.586 | 0.549 | 0.000 |
| Flow v5 u26000 | 512 | 6 | 86 | 0 | 0 | 181 | 87.016 | 8.719 | 4.480 | 3.296 |
| Unified u35000 | 0 | 10 | 2143 | 0 | 0 | 2262 | 96.081 | 0.415 | 0.396 | 0.000 |
| Unified u35000 | 512 | 5 | 89 | 0 | 0 | 179 | 87.310 | 8.565 | 4.546 | 3.427 |
| Flow v5 + FW/BW | 0 | 2 | 1907 | 0 | 287 | 2290 | 72.577 | 0.308 | 0.295 | 0.000 |
| Flow v5 + FW/BW | 512 | 1 | 47 | 0 | 54 | 166 | 64.767 | 5.860 | 3.636 | 2.506 |
| Unified + FW/BW | 0 | 2 | 2894 | 0 | 257 | 3241 | 70.895 | 0.209 | 0.204 | 0.000 |
| Unified + FW/BW | 512 | 1 | 49 | 0 | 38 | 149 | 65.681 | 6.877 | 4.108 | 2.808 |
| Memory matched control | 0 | 5 | 791 | 0 | 0 | 906 | 99.253 | 1.158 | 1.021 | 0.315 |
| Memory matched control | 512 | 2 | 140 | 0 | 0 | 218 | 78.334 | 5.105 | 3.349 | 2.238 |
| Flow v4 matched control | 0 | 16 | 487 | 0 | 0 | 621 | 99.545 | 1.812 | 1.494 | 0.618 |
| Flow v4 matched control | 512 | 9 | 132 | 0 | 0 | 216 | 88.024 | 5.777 | 3.798 | 3.101 |
| Memory + FW/BW | 0 | 2 | 702 | 0 | 355 | 1141 | 63.736 | 0.560 | 0.521 | 0.000 |
| Memory + FW/BW | 512 | 0 | 58 | 0 | 100 | 186 | 48.462 | 2.840 | 2.428 | 1.668 |
| Flow v4 + FW/BW | 0 | 1 | 381 | 0 | 338 | 799 | 70.011 | 0.905 | 0.817 | 0.000 |
| Flow v4 + FW/BW | 512 | 0 | 57 | 0 | 93 | 194 | 64.408 | 3.975 | 3.094 | 2.483 |

## Runtime

| Model | Seconds to last completed case |
|---|---:|
| Neural memory u27000 | 103.78 |
| Neural Flow v4 u20000 | 108.80 |
| Flow v5 u26000 | 211.96 |
| Unified u35000 | 227.46 |
| Flow v5 + FW/BW | 364.24 |
| Unified + FW/BW | 432.38 |
| Memory matched control | 101.30 |
| Flow v4 matched control | 102.92 |
| Memory + FW/BW | 212.42 |
| Flow v4 + FW/BW | 224.90 |

Times include worker/model initialization and CT reads, exclude manager wait, reference/hash preflight and shutdown. Progress timestamps preserve original measurements even if cached summary analysis is rerun.

## Reproduce

Set `scripts/training_failure_mining_control.json` to `{"action":"neural_follow","pilot":false,"reverse":true}` then run:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Artifacts/configs/checkpoint hashes: `/ephemeral/hendrik/fiber3d_train2/evaluations/neural_follow_reverse_91_20261007_v1`. GPUs0–5 via manager. Native RelWithDebInfo build. One run per model; timing depends on CT cache state, not an isolated speed comparison.

## Validation

149 native regression cases and38 Python tests passed through the same approved runner (control: action neural_follow, tests true). Native verifier extraction preserves existing history0/history4 and frozen-width behavior. Callback tests cover disabled parity, correct/wrong-way/lateral/duplicate/confidence reverse paths, interval/terminal checks, rollback, reseed advancement and expired callbacks. Two-direction GPU pilots validate nested inference before full runs; configuration tests preserve historical policies and batching.

This is an online filter: rejected prefixes change subsequent restart locations and later traces, so confidence counts need not remain constant. Min512 analysis discards saved pieces only; no tracing changes. Historical memory/Flowv4 rows are retained; matched-control and +FW/BW rows are new full evaluations. No default promotion.

| FW/BW model | Reverse checks |
|---|---:|
| Flow v5 + FW/BW | 1533 |
| Unified + FW/BW | 1392 |
| Memory + FW/BW | 1525 |
| Flow v4 + FW/BW | 1499 |

## Memory and Flow v4 follow-up

The current upstream loader rejects the old memory/identity architecture. These two models therefore use the frozen working runtime from ba86eea12231ff770324d2adb7fceb709e93a3ea (upstream9200d51a9), checked against saved implementation hashes. The shared current native replay/filter and GPU queue are reused; models/rollout are not reimplemented or simplified.

Original settings retained: memory confidence.4/full gate/commit8 with3refinement steps; Flow v4 checkpoint-recorded confidence.5/prefix gate/commit16 with0refinement steps. BF16, batch8,16case workers/GPU,24input/I/O workers. Original forward length bound is3×remaining reference length. Fresh matched unfiltered controls distinguish FW/BW changes from integration/batching variation. Our own tuned EMA999u12100 row is explicitly WITHOUT FW/BW.

To reproduce old-model runs use models ["memory","flow"], pilot false, reverse false for matched controls or true for FW/BW, through the same approved runner. Run operation inspect_older first to materialize and verify the frozen runtime.
