# Task Log: 1024 node-unary Mixed BP evaluation

## Inputs

- Trace dataset: `data/workdir3/crop_traces.zarr`
- Output basename: `data/workdir3/fiber-crop-1024/fibers`
- Normal manifest: `data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json`
- Direction dominance: `0.9`
- Piece length: no split (`1000000000`)
- Constraints: perpendicular only

## Results

- Fixed `T=2.5` and swept the committed node-unary model:

  | Unary | Confident H/V | Initial D1/D2 agreement | Cross-group assignment | Mixed argmax | Count coverage | Count mismatch | Strength coverage | Strength mismatch | Churn from prior |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 2 | 1 | 224 | 0 | 78 | 13.56% | 0.00% | 16.42% | 0.00% | n/a |
  | 4 | 72 | 337 | 0 | 50 | 46.12% | 0.44% | 53.92% | 0.26% | 141 |
  | 6 | 266 | 357 | 0 | 38 | 63.95% | 1.17% | 71.61% | 0.57% | 32 |
  | 8 | 369 | 359 | 2 | 29 | 71.91% | 1.83% | 79.78% | 0.85% | 16 |
  | 10 | 401 | 359 | 3 | 15 | 78.32% | 3.72% | 84.86% | 1.78% | 16 |
  | 12 | 413 | 359 | 3 | 11 | 81.79% | 4.75% | 87.56% | 2.43% | 4 |
  | 16 | 427 | 359 | 3 | 8 | 85.23% | 5.84% | 89.71% | 3.15% | 3 |

- `Confident H/V` means a unique H or V top state with probability at least
  0.75. Initial Direction1/Direction2 agreement is diagnostic only because the
  geometric reference contains known errors.
- Binary sum-product at `T=2.5` confidently orients 440 fibers, resolving
  90.77% of factors by count and 93.26% by strength, but its mismatch is much
  higher: 8.03% by count and 4.30% by strength.
- Selected unary cost `8` as the knee of the coverage/consistency frontier. It
  gains 103 confident H/V assignments over cost 6 while retaining less than 1%
  strength-weighted mismatch. Cost 10 gains only 32 more confident assignments
  while more than doubling weighted mismatch.
- The selected run converged in 159 message iterations and 0.17 solver seconds.
  It has five components: one 496-fiber seeded factor component and four
  isolates. All four exact ties are therefore structural isolate/gauge cases.
- Selected argmax partition: 226 V, 30 Mixed, 240 H, and 4 ties. Against the
  noisy geometric grouping, Direction1 is `1 V / 0 Mixed / 205 H / 3 ties`,
  Direction2 is `154 V / 1 Mixed / 1 H / 1 tie`, and initial Mixed is
  `71 V / 29 Mixed / 34 H / 0 ties`. `P(Mixed)` AUROC is 0.683998.
- Regenerated the main short OBJ layers and consistency CSV under
  `data/workdir3/fiber-crop-1024/` with `--bp-mixed-cost 8`. Object counts in
  `fibers_bp_v.obj`, `fibers_bp_mixed.obj`, `fibers_bp_h.obj`, and
  `fibers_bp_tie.obj` are 226, 30, 240, and 4, respectively, partitioning all
  500 fibers exactly.

## Plan Review

- Independent review required a predeclared sweep, confidence-based H/V
  coverage, resolved-factor coverage alongside mismatch, neighboring-setting
  churn, gauge-aware component reporting, a binary baseline, and exact output
  partition verification. The plan now includes each item.

## Deviations

- None.
