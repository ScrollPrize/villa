# Equal-scroll comparison: classic and 600k raw/EMA

Each scroll contributes exactly **20%** to the overall score:

`overall score = (mean0139 + mean0332 + mean1299 + mean1451 + meanParis4) / 5`

Per-scroll mean = directed reference length / (restarts + directions), including
terminal and zero-length segments. All lengths are in **mm**. The old pooled
overall score is intentionally omitted. Percentiles are reported per scroll,
not averaged into invented overall percentiles. **Bold marks the best value
within each metric and scroll; tied best values are all bold.**

All seven evaluations cover the same 50 fibers / 100 directions across five
scrolls. Classic is the previously saved, longer-trained baseline—not a matched
600k training run. Its 100 raw trace artifacts were re-audited without rerunning
inference. Modern runs reached 601,600 samples, selecting their best raw and EMA
by test loss at or before that cutoff. Every selected EMA has decay 0.99.

## Overall score: equal scroll weights

| Model | Equal-scroll mean ↑ |
|---|---:|
| Classic | 5.723 |
| NML raw | 2.482 |
| NML EMA | 3.054 |
| JSON raw | 4.930 |
| JSON EMA | **6.308** |
| Combined raw | 5.368 |
| Combined EMA | 6.202 |

## PHerc0139 — 10 fibers

20 directions; 1392.347 mm directed reference.
Total segment count = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 564 | 2.384 | **0.175** | **0.0071** | 11.631 | 8 |
| NML raw | 902 | 1.510 | 0.078 | 0.0068 | 9.693 | 10 |
| NML EMA | 750 | 1.808 | 0.082 | 0.0053 | 9.358 | 8 |
| JSON raw | 485 | 2.757 | 0.097 | 0.0052 | 15.315 | 8 |
| JSON EMA | **437** | **3.047** | 0.106 | 0.0050 | 15.933 | 13 |
| Combined raw | 448 | 2.975 | 0.133 | 0.0069 | **15.984** | **4** |
| Combined EMA | 459 | 2.907 | 0.094 | 0.0061 | 15.596 | 7 |

## PHerc0332 — 3 fibers

6 directions; 471.222 mm directed reference.
Total segment count = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 27 | 14.279 | 4.206 | 0.0074 | 48.805 | 1 |
| NML raw | 69 | 6.283 | 2.909 | **0.0249** | 22.058 | **0** |
| NML EMA | 52 | 8.125 | 0.250 | 0.0079 | 35.816 | **0** |
| JSON raw | 26 | 14.726 | 1.999 | 0.0071 | 60.832 | **0** |
| JSON EMA | **17** | **20.488** | 8.486 | 0.0092 | **69.482** | **0** |
| Combined raw | 23 | 16.249 | 2.528 | 0.0057 | 58.601 | **0** |
| Combined EMA | 19 | 18.849 | **10.735** | 0.0037 | 67.689 | **0** |

## PHerc1299 — 5 fibers

10 directions; 599.748 mm directed reference.
Total segment count = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | **159** | **3.549** | **0.803** | **0.0315** | **17.573** | **1** |
| NML raw | 344 | 1.694 | 0.260 | 0.0068 | 9.602 | 5 |
| NML EMA | 318 | 1.828 | 0.248 | 0.0092 | 9.318 | 6 |
| JSON raw | 225 | 2.552 | 0.223 | 0.0095 | 11.192 | **1** |
| JSON EMA | 233 | 2.468 | 0.217 | 0.0119 | 11.649 | 2 |
| Combined raw | 250 | 2.307 | 0.184 | 0.0094 | 11.731 | 3 |
| Combined EMA | 216 | 2.654 | 0.257 | 0.0116 | 14.332 | 3 |

## PHerc1451 — 2 fibers

4 directions; 94.430 mm directed reference.
Total segment count = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | **19** | **4.106** | **1.862** | **0.0707** | **17.222** | **0** |
| NML raw | 83 | 1.085 | 0.079 | 0.0087 | 5.855 | 2 |
| NML EMA | 72 | 1.243 | 0.156 | 0.0097 | 6.246 | **0** |
| JSON raw | 43 | 2.009 | 0.217 | 0.0111 | 11.336 | **0** |
| JSON EMA | 37 | 2.303 | 0.268 | 0.0192 | 11.598 | **0** |
| Combined raw | 36 | 2.361 | 0.436 | 0.0452 | 11.234 | **0** |
| Combined EMA | 39 | 2.196 | 0.262 | 0.0285 | 11.056 | **0** |

## PHercParis4 — 30 fibers

60 directions; 5523.811 mm directed reference.
Total segment count = restarts + directions.

| Model | Restarts ↓ | Mean ↑ | Median ↑ | P05 ↑ | P95 ↑ | Zero ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 1226 | 4.295 | **0.185** | 0.0128 | 23.435 | 12 |
| NML raw | 2943 | 1.839 | 0.092 | 0.0105 | 9.102 | 19 |
| NML EMA | 2380 | 2.264 | 0.061 | 0.0087 | 11.655 | 25 |
| JSON raw | 2059 | 2.607 | 0.034 | **0.0132** | 10.385 | **10** |
| JSON EMA | 1649 | 3.232 | 0.048 | 0.0119 | 13.225 | **10** |
| Combined raw | 1814 | 2.948 | 0.066 | 0.0125 | 14.619 | **10** |
| Combined EMA | **1194** | **4.405** | 0.145 | 0.0112 | **27.513** | 14 |

## Interpretation

JSON EMA has the highest equal-scroll score: 6.308mm versus Combined EMA6.202mm
(1.7% higher) and Classic5.723mm (10.2% higher). Combined EMA leads on Paris4;
JSON EMA leads on0139 and0332; Classic leads on1299 and1451 (by mean length).
PHerc0332 has just three evaluated fibers and PHerc1451 two, so equal weighting
does not imply equal statistical certainty. The arithmetic mean also remains
sensitive to scrolls with large absolute trace lengths; it is not a relative-
improvement or rank score. No significance claim is made.

Training was not changed by this report-only reweighting.

## Checkpoint selection

| Modern model | Step | Samples at selected checkpoint |
|---|---:|---:|
| NML raw | 2800 | 358400 |
| NML EMA | 3000 | 384000 |
| JSON raw | 4200 | 537600 |
| JSON EMA | 4600 | 588800 |
| Combined raw | 4600 | 588800 |
| Combined EMA | 4600 | 588800 |

## Evidence

Classic audited input:
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v3_heap/full`

Classic summary SHA256: `04ca86dadbd21f0b7aaf5cffbf235ced9698bf54d79f4fed763c97e2f78c98a8`.

Modern reports (each contains selection, frozen-checkpoint hash and full metrics):

- `/ephemeral/hendrik/fiber3d_train2/comparisons/nml_json_combined_20260912/comparison.json` (NML and JSON rows).
- `/ephemeral/hendrik/fiber3d_train2/comparisons/combined_600k_20260912/report.json` (combined600k rows).

Validated identical scroll IDs, per-scroll direction counts and reference lengths;
recomputed every per-scroll mean from length/(restarts+directions), then averaged
the five means with equal weights. No GPU computation or training changes.
