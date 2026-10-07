# Neural followers: Flow v5 and unified scoring

Same reviewed 91 fibers / 182 directions, native reference acceptance40/80base vx, grace160 and restart grouping80. Confidence terminations are counted separately; all other nonterminal reasons remain errors. GT supplies restart locations and traversal sign, not the inferred frame or history. No FW/BW or width modulation added to neural models.

Flow v5 u26000: confidence0.5, n_commit8, gatefull. Unified supplied checkpoint is u35000 (S3 has no u33000): confidence0.7, n_commit8, gatefull. Both CTlevel2/grid4, tracegrid8, maxlen100000 trace voxels, BF16, seed0. Unified maximum ready-forward batch32, Flow16. Upstream853dcf536 with native-reader injection and commit observer. Noise keys use per-trace seed position/decision, not scheduling; BF16 batch-dependent roundoff remains possible. Historical memory u27000 and Flow v4 u20000 are recomputed from saved traces, not rerun; their original policies/revisions remain in the companion JSON.

Collection-wide seed dedup16 and CP export spacing800 are CLI collection/export operations, not applied across independent benchmark references. Full min0 results respect min-length0; min512 is additional output-only analysis to match the tuned baseline. This filtering is GT-assisted because geometric errors already split outputs. Report confidence stops and coverage, not errors alone.

Neural confidence restarts use the existing native GT restart advancement; the tuned baseline uses its presence-aligned restart policy. This is a comparison of complete policies, not an isolated model substitution.

Mean = retained GT distance / (all surviving individual stops +1). Piece mean = retained GT distance / kept pieces; median is retained GT span per piece. P05/P95 and complete machine-readable metrics are in companion JSON.

| Model | Min vx | Errors | Confidence/presence stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned | 0 | 0 | 417 | 586 | 89.864 | 2.980 | 2.126 | 0.665 |
| EMA999 u12100 tuned | 512 | 0 | 86 | 233 | 82.086 | 13.079 | 4.884 | 3.814 |
| Neural memory u27000 | 0 | 12 | 1369 | 1561 | 99.051 | 0.992 | 0.880 | 0.308 |
| Neural memory u27000 | 512 | 6 | 226 | 336 | 76.542 | 4.554 | 3.158 | 2.199 |
| Neural Flow v4 u20000 | 0 | 43 | 772 | 1006 | 99.615 | 1.668 | 1.373 | 0.643 |
| Neural Flow v4 u20000 | 512 | 21 | 224 | 356 | 85.700 | 4.829 | 3.337 | 2.742 |
| Flow v5 u26000 | 0 | 26 | 2559 | 2766 | 97.172 | 0.520 | 0.487 | 0.000 |
| Flow v5 u26000 | 512 | 14 | 140 | 296 | 84.840 | 7.587 | 3.973 | 2.771 |
| Unified u35000 | 0 | 24 | 3034 | 3234 | 96.320 | 0.436 | 0.413 | 0.000 |
| Unified u35000 | 512 | 13 | 132 | 285 | 86.662 | 8.228 | 4.215 | 3.263 |

## PHerc0139

| Model | Min vx | Errors | Confidence/presence stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned | 0 | 0 | 234 | 273 | 91.032 | 1.105 | 0.951 | 0.350 |
| EMA999 u12100 tuned | 512 | 0 | 32 | 60 | 66.395 | 5.739 | 3.156 | 2.066 |
| Neural memory u27000 | 0 | 3 | 482 | 529 | 97.961 | 0.575 | 0.528 | 0.000 |
| Neural memory u27000 | 512 | 1 | 49 | 71 | 70.639 | 3.951 | 2.838 | 2.170 |
| Neural Flow v4 u20000 | 0 | 17 | 238 | 303 | 99.630 | 1.093 | 0.938 | 0.322 |
| Neural Flow v4 u20000 | 512 | 4 | 51 | 79 | 75.642 | 3.853 | 2.731 | 2.460 |
| Flow v5 u26000 | 0 | 9 | 981 | 1036 | 94.581 | 0.272 | 0.260 | 0.000 |
| Flow v5 u26000 | 512 | 2 | 28 | 61 | 73.072 | 6.723 | 3.417 | 2.732 |
| Unified u35000 | 0 | 9 | 848 | 899 | 94.994 | 0.316 | 0.301 | 0.000 |
| Unified u35000 | 512 | 3 | 28 | 66 | 81.691 | 7.281 | 3.530 | 2.721 |

## PHerc0332

| Model | Min vx | Errors | Confidence/presence stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned | 0 | 0 | 7 | 8 | 31.915 | 0.444 | 0.444 | 0.333 |
| EMA999 u12100 tuned | 512 | 0 | 0 | 0 | 0.000 | — | — | — |
| Neural memory u27000 | 0 | 0 | 6 | 8 | 100.000 | 1.590 | 1.391 | 1.351 |
| Neural memory u27000 | 512 | 0 | 4 | 4 | 80.011 | 1.781 | 2.226 | 2.304 |
| Neural Flow v4 u20000 | 0 | 0 | 3 | 5 | 100.000 | 2.782 | 2.226 | 2.773 |
| Neural Flow v4 u20000 | 512 | 0 | 3 | 4 | 93.244 | 2.594 | 2.594 | 2.782 |
| Flow v5 u26000 | 0 | 0 | 2 | 4 | 100.000 | 3.710 | 2.782 | 2.782 |
| Flow v5 u26000 | 512 | 0 | 2 | 4 | 100.000 | 3.710 | 2.782 | 2.782 |
| Unified u35000 | 0 | 0 | 1 | 3 | 100.000 | 5.564 | 3.710 | 3.531 |
| Unified u35000 | 512 | 0 | 1 | 3 | 100.000 | 5.564 | 3.710 | 3.531 |

## PHerc1299

| Model | Min vx | Errors | Confidence/presence stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned | 0 | 0 | 2 | 24 | 99.807 | 49.048 | 6.131 | 5.033 |
| EMA999 u12100 tuned | 512 | 0 | 1 | 23 | 99.807 | 73.572 | 6.398 | 5.033 |
| Neural memory u27000 | 0 | 3 | 66 | 91 | 99.987 | 2.106 | 1.620 | 1.088 |
| Neural memory u27000 | 512 | 3 | 24 | 39 | 77.529 | 4.082 | 2.931 | 2.189 |
| Neural Flow v4 u20000 | 0 | 9 | 40 | 71 | 100.000 | 2.949 | 2.076 | 2.109 |
| Neural Flow v4 u20000 | 512 | 8 | 30 | 50 | 89.615 | 3.388 | 2.642 | 2.391 |
| Flow v5 u26000 | 0 | 6 | 41 | 69 | 100.000 | 3.071 | 2.137 | 1.870 |
| Flow v5 u26000 | 512 | 6 | 23 | 47 | 92.065 | 4.524 | 2.888 | 2.195 |
| Unified u35000 | 0 | 5 | 42 | 68 | 99.857 | 3.067 | 2.165 | 1.560 |
| Unified u35000 | 512 | 5 | 14 | 35 | 90.223 | 6.651 | 3.800 | 3.443 |

## PHerc1451

| Model | Min vx | Errors | Confidence/presence stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned | 0 | 0 | 3 | 5 | 95.320 | 2.501 | 2.001 | 1.271 |
| EMA999 u12100 tuned | 512 | 0 | 2 | 3 | 83.980 | 2.938 | 2.938 | 3.099 |
| Neural memory u27000 | 0 | 0 | 4 | 6 | 100.000 | 2.099 | 1.749 | 1.205 |
| Neural memory u27000 | 512 | 0 | 2 | 3 | 78.479 | 2.746 | 2.746 | 2.989 |
| Neural Flow v4 u20000 | 0 | 0 | 1 | 3 | 100.000 | 5.248 | 3.499 | 4.381 |
| Neural Flow v4 u20000 | 512 | 0 | 1 | 2 | 91.740 | 4.814 | 4.814 | 4.814 |
| Flow v5 u26000 | 0 | 0 | 2 | 4 | 100.000 | 3.499 | 2.624 | 2.624 |
| Flow v5 u26000 | 512 | 0 | 1 | 3 | 93.947 | 4.930 | 3.287 | 3.761 |
| Unified u35000 | 0 | 0 | 0 | 2 | 100.000 | 10.496 | 5.248 | 5.248 |
| Unified u35000 | 512 | 0 | 0 | 2 | 100.000 | 10.496 | 5.248 | 5.248 |

## PHercParis4

| Model | Min vx | Errors | Confidence/presence stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA999 u12100 tuned | 0 | 0 | 171 | 276 | 88.564 | 4.798 | 2.990 | 1.618 |
| EMA999 u12100 tuned | 512 | 0 | 51 | 147 | 85.044 | 15.241 | 5.391 | 4.737 |
| Neural memory u27000 | 0 | 6 | 811 | 927 | 99.214 | 1.128 | 0.997 | 0.314 |
| Neural memory u27000 | 512 | 2 | 147 | 219 | 78.130 | 4.854 | 3.325 | 2.218 |
| Neural Flow v4 u20000 | 0 | 17 | 490 | 624 | 99.541 | 1.798 | 1.487 | 0.574 |
| Neural Flow v4 u20000 | 512 | 9 | 139 | 221 | 88.001 | 5.504 | 3.711 | 2.988 |
| Flow v5 u26000 | 0 | 11 | 1533 | 1653 | 97.452 | 0.586 | 0.549 | 0.000 |
| Flow v5 u26000 | 512 | 6 | 86 | 181 | 87.016 | 8.719 | 4.480 | 3.296 |
| Unified u35000 | 0 | 10 | 2143 | 2262 | 96.081 | 0.415 | 0.396 | 0.000 |
| Unified u35000 | 512 | 5 | 89 | 179 | 87.310 | 8.565 | 4.546 | 3.427 |

## Runtime

| Model | Seconds to last completed case |
|---|---:|
| Neural memory u27000 | 103.78 |
| Neural Flow v4 u20000 | 108.80 |
| Flow v5 u26000 | 211.96 |
| Unified u35000 | 227.46 |

Times include worker/model initialization and CT reads, exclude manager wait, reference/hash preflight and shutdown. Progress timestamps preserve original measurements even if cached summary analysis is rerun.

## Reproduce

Set `scripts/training_failure_mining_control.json` to `{"action":"neural_follow","pilot":false}` then run:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

Artifacts/configs/checkpoint hashes: `/ephemeral/hendrik/fiber3d_train2/evaluations/neural_follow_91_20261007_v1`. GPUs0–5 via manager. Native RelWithDebInfo build. One run per model; timing depends on CT cache state, not an isolated speed comparison.
