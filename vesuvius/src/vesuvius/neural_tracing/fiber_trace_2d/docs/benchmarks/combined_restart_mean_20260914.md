# Corrected aggregate means — current evaluation protocol

2026-09-14. User-requested formula at every aggregation scope:

**Mean = combined directed evaluated distance / (reported restarts + 1).**

The old mean divided by restarts + directed runs. Reliability segment counts and median/P05/P95 still describe actual segments and are unchanged. Consequently this new combined-distance metric need not fall within the segment percentiles. Distances include both directions exactly as before; no inference/tracing rerun. All three models:50fibers/90eligible runs/180directions,6867.627252833mm; thickness40,width80,grace160,episode80,no embedding. PHerc0332:425.160903676mm, two restarts =>141.720301225mm.

## Combined scores

| Model | Classic-normalized equal-scroll score | Arithmetic mean of scroll means (mm) | Combined distance/(restarts+1) (mm) |
|---|---:|---:|---:|
| Classic | 23.401875 | 23.401875 | 23.439001 |
| EMA99 u1800 | 54.904589 | 60.701007 | 39.243584 |
| EMA99 u5200 | 55.845764 | 62.058971 | 46.091458 |

Normalized score = Classic arithmetic scroll mean × geometric mean of five model/Classic ratios, equal20% weights. Scores recalculated with the new means for every model.

## All combined

| Model | Restarts | Combined mean mm | Median mm | P05 mm | P95 mm | Denominator |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 292 | 23.439001 | 5.822666 | 0.176021 | 62.408360 | 293 |
| EMA99 u1800 | 174 | 39.243584 | 7.625640 | 0.197488 | 87.637392 | 175 |
| EMA99 u5200 | 148 | 46.091458 | 8.202461 | 0.143040 | 88.437892 | 149 |

## PHerc0139

| Model | Restarts | Combined mean mm | Median mm | P05 mm | P95 mm | Denominator |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 108 | 10.775725 | 2.235374 | 0.079368 | 25.656991 | 109 |
| EMA99 u1800 | 73 | 15.872351 | 2.534936 | 0.090248 | 31.510496 | 74 |
| EMA99 u5200 | 66 | 17.530657 | 2.842227 | 0.089569 | 39.895663 | 67 |

## PHerc0332

| Model | Restarts | Combined mean mm | Median mm | P05 mm | P95 mm | Denominator |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 9 | 42.516090 | 16.777453 | 0.923118 | 84.675506 | 10 |
| EMA99 u1800 | 2 | 141.720301 | 22.319331 | 15.339546 | 152.532149 | 3 |
| EMA99 u5200 | 2 | 141.720301 | 22.353526 | 15.339546 | 152.480602 | 3 |

## PHerc1299

| Model | Restarts | Combined mean mm | Median mm | P05 mm | P95 mm | Denominator |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 26 | 15.077982 | 6.361196 | 0.317741 | 35.558385 | 27 |
| EMA99 u1800 | 8 | 45.233947 | 15.613233 | 0.662523 | 68.909987 | 9 |
| EMA99 u5200 | 11 | 33.925461 | 15.406374 | 0.847329 | 39.371548 | 12 |

## PHerc1451

| Model | Restarts | Combined mean mm | Median mm | P05 mm | P95 mm | Denominator |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 2 | 16.123110 | 11.053252 | 1.746046 | 11.347567 | 3 |
| EMA99 u1800 | 0 | 48.369329 | 12.092332 | 11.128854 | 13.055811 | 1 |
| EMA99 u5200 | 0 | 48.369329 | 12.092332 | 11.128854 | 13.055811 | 1 |

## PHercParis4

| Model | Restarts | Combined mean mm | Median mm | P05 mm | P95 mm | Denominator |
|---|---:|---:|---:|---:|---:|---:|
| Classic | 147 | 32.516470 | 8.893962 | 0.195724 | 85.486571 | 148 |
| EMA99 u1800 | 91 | 52.309103 | 11.289367 | 0.271443 | 89.582314 | 92 |
| EMA99 u5200 | 69 | 68.749107 | 11.903306 | 0.211525 | 109.762664 | 70 |

## Implementation and reproduction

Shared replay_metrics.segment_statistics/complete_report/correct_summary now expose mean_definition and mean_denominator separately from the unchanged episode metric. Summary voxel/percent means and visualization aggregate consumers use the new denominator. Comparison assertions and labels updated. Read original summaries below with complete_report to reproduce; raw traces and original summaries retained, corrected report JSON saved beside this Markdown.

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
```

Regressions cover0332-style multi-direction totals, zero restarts, single direction, empty input, legacy filtered summaries/idempotence, and per-fiber/direction aggregation. No native algorithm changes or new dependencies.

- Classic source: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_thickness40_grace160_20260914_v2/full/summary.json`
- EMA99 u1800 source: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_thickness40_grace160_20260914_v2/full/summary.json`
- EMA99 u5200 source: `/ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_current_20260914_1345/full/summary.json`
