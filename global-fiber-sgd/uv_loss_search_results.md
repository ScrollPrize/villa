# Adaptive loss-weight search — stopped; new baseline

Fresh seed0, 151 hypotheses, CCW, same reference subset IDs14–20 / base Z30k–50k.
Each trial runs10080 minibatch updates (~10k sampled pass equivalents), joint
optimization from the start, six GPUs/globalbatch24576. StateLR0.07348469228349533,
widthLR2 fixed. Reference L1 is offset-corrected and computed on retained fibers;
reference is tuning data, not held out. Retention changes the evaluated cohort.

Code/plan committed dba4ffa6d. Thirteen CPU tests passed. Each native GPU trial
runs distributed gradient and coordinate checks. No intermediate1k diagnostics.
Objective below is whole-run online mean (not final frozen-state loss); different
loss weights make its magnitude non-comparable between rows.

## Accepted changes

| Change | Ref L1 | Within ±2 % | Retained length % | Mean objective | Width factor | Bucket-sign violations % |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.967458 | 68.7458 | 50.1774 | 11.47031 | 0.913733 | 0 |
| position_distance_weight → 2 | 1.936389 | 70.3830 | 49.6564 | 11.78460 | 0.915169 | 0.00000 |
| perpendicular_relative_weight → 2.5 | 1.870103 | 72.9897 | 49.2143 | 11.32355 | 0.916587 | 0.00000 |
| rejection_cost → 2.5 | 1.688906 | 77.1364 | 43.5020 | 3.43410 | 0.911669 | 0.00000 |
| distance_weight → 4 | 1.592174 | 79.4783 | 35.3044 | 4.67702 | 0.912237 | 0.00000 |
| perpendicular_relative_weight → 10 | 1.591549 | 80.2817 | 38.1802 | 4.83693 | 0.912836 | 0.00000 |
| perpendicular_relative_weight → 40 | 1.568019 | 79.8727 | 40.5607 | 4.84404 | 0.985141 | 0.00000 |
| sign_charge_multiplier → 8 | 1.429992 | 82.4790 | 42.2991 | 3.21709 | 0.985852 | 0.12610 |
| prior_weight → 0.0025 | 1.416348 | 82.8113 | 42.4245 | 3.21620 | 0.986126 | 0.12478 |
| rejection_cost → 40 | 1.402994 | 81.6594 | 56.8878 | 22.84536 | 1.004732 | 0.58252 |
| perpendicular_relative_weight → 2.5 | 1.374684 | 83.1013 | 56.3910 | 22.32153 | 0.879230 | 0.00000 |
| distance_weight → 0.25 | 1.259795 | 85.0512 | 63.3334 | 17.88251 | 0.909152 | 0.00000 |
| perpendicular_relative_weight → 10 | 1.197234 | 87.6127 | 63.2974 | 18.60773 | 0.900014 | 0.04943 |
| sign_weight → 40 | 1.156342 | 87.8466 | 65.3013 | 18.52733 | 0.899681 | 0.01146 |
| position_distance_weight → 0.125 | 1.123602 | 87.0512 | 65.9041 | 18.43135 | 0.981746 | 0.00000 |
| position_distance_weight → 0.5 | 1.048824 | 88.3529 | 65.7749 | 18.45137 | 0.978318 | 0.00000 |
| offset_soft_weight → 0.00025 | 1.038235 | 88.4118 | 65.7749 | 18.45162 | 0.978573 | 0.00000 |

## Reproduction and full records

Approved command:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_loss_search.json
```

Output: `data/uv_loss_search_v1/`. See `results.json` for every attempted
configuration, `acceptances.json` for the chain, `best.json` for the incumbent,
`active_trial.json` for current work, and individual `trials/*/config.json`,
`summary.json`, `solution.npz`. The final root `summary.json` is written only
when every coordinate alternative in the declared fine grid has been checked
without improvement. The search was stopped by user after **96 completed trials / 16 acceptances**;
this final certificate does not exist. Fine-grid/no-improvement completion was
not reached. Interrupted trial 8ed0c7e316d39355c587 is excluded.
Manager98ec0fe8-2d8a-4627-bc12-73a62350072a is terminal, held resources zero.

The selected result is now the current benchmark baseline, explicitly configured
in [run_uv_current_baseline.json](run_uv_current_baseline.json). Historical configs
and solver fallback defaults remain untouched; future comparative experiments
should derive from this canonical config, not historical files. The solution is
`data/uv_loss_search_v1/trials/eb142efc153b00cd9e83/solution.npz`.
Committed compact records: [all trials](uv_loss_search_trials.json) and
[baseline provenance/metrics](uv_current_baseline_result.json). Large solver
arrays remain in the ignored data directory.

Baseline reproduction (NOT launched as part of stopping):
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_current_baseline.json
```

Compared with search start: L1 **1.967458 → 1.038235** (47.23% lower), within2
**68.7458% → 88.4118%**, retained length **50.1774% → 65.7749%** (+15.5974pp),
width factor **.913733 → .978573**, signs **0% → 0%**. Retained cohort grows
from1475 to1700 matched original fibers. This is training/reference-tuning
performance, not independently held-out generalization.

Selected solve time170.432s. Full-history loss18.45162 differs from baseline
11.47031 because coefficients changed; do not interpret that as a comparable
accuracy measure. Optimizer/data semantics unchanged. All trials use same fresh
seed0, no checkpoint continuation or reference initialization.

Changed loss weights vs search start:
- rejection_cost:10 →40
- distance_weight:1 →.25
- sign_charge_multiplier:32 →8
- sign_weight:10 →40
- prior_weight:.01 →.0025
- offset_soft_weight:.001 →.00025

Other weights remain at search-start values, including position .5,
perpendicular ratio10, entropy .05, metric .01, Vprior .01, offsetlimit10000,
continuation10, widthprior .1 and widthsmooth .01.

Validation after stopping:13 tests passed2.73s; independent baseline review
approved. Selected weights and all non-runtime solver settings match the winning
trial; selected L1 is the minimum among96 completed trials; saved arrays finite.
The zero-sign statistic is bucket-sign geometry, not an assertion of zero
physical-layout errors. No GPU solve rerun during finalization.
