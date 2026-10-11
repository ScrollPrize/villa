# Corrected quadratic fused solve

Completed with exit0; implementation c468354dd, configuration86d2a4d8d.
Output: data/uv_squared_vertical_split_v1/{history.json,summary.json,solution.npz}.

## Command

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_squared_vertical_split.json
```

Paris4 cached CCW constraints,151 hypotheses,4096 random-with-replacement
minibatches,16130 joint updates. All variables active from start, fresh radial
initialization; reference only used for metrics. Shared managed GPU1/24GiB.
81 regression tests previously passed. No new solver code in this experiment.

## Progress

Pass-equivalent is event count, not exhaustive epoch. L1 is globally offset
corrected (offset+12 throughout). Reference quantiles use retained matched fibers.
Intermediate sign diagnostics sample100k events; final audits all6,606,100.
Objective is interval-average online objective, not frozen-state global loss.

| Pass-equivalent | Seconds | Objective | Ref L1 | Within2 % | Retained length % | Sign violations % | Fit q10/90 | Ref q10/90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.0 | — | 12.2423 | 12.9912 | 100.0000 | 40.9860 | 49/122 | 42/124 |
| 1 | 61.7 | 80.1353 | 11.9613 | 13.0534 | 84.0409 | 14.9976 | 49/122 | 42/124 |
| 2 | 121.2 | 12.5317 | 11.8688 | 13.1195 | 72.1916 | 9.4528 | 49/122 | 42/124 |
| 3 | 208.0 | 6.0513 | 11.8192 | 13.1045 | 68.3073 | 7.0389 | 49/122 | 42/124 |
| 4 | 260.7 | 4.4235 | 11.7946 | 13.1336 | 66.8569 | 5.7019 | 49/122 | 42/125 |
| 5 | 334.4 | 3.7037 | 11.7763 | 13.1455 | 66.2903 | 4.8610 | 49/122 | 42/125 |
| 6 | 408.1 | 3.3139 | 11.7614 | 13.2094 | 66.0293 | 4.3583 | 49/122 | 42/125 |
| 7 | 481.6 | 3.0555 | 11.7493 | 13.1921 | 65.9431 | 4.0353 | 49/122 | 42/125 |
| 8 | 555.7 | 2.8699 | 11.7398 | 13.2133 | 65.9467 | 3.9094 | 49/122 | 42/125 |
| 9 | 628.9 | 2.7250 | 11.7304 | 13.2342 | 65.9608 | 3.4041 | 49/122 | 42/125 |
| 10 | 699.3 | 2.6192 | 11.7233 | 13.2134 | 66.0169 | 3.3308 | 49/122 | 42/125 |

## Comparison

| Metric | Earlier fused quadratic | Corrected quadratic |
|---|---:|---:|
| Solver elapsed seconds |712.58|699.28|
| Ref L1 |11.723623|11.723346|
| Within2 % |13.209575|13.213415|
| Retained length % |66.022779|66.016892|
| Sign violations % |3.331115|3.330811|

1.9% shorter measured solver time; shared GPU runs are not isolated timing
comparisons. Timings include optimizer and audits/compilation, exclude manager
queue. Existing isolated fusion benchmark measured1.823x complete-update speedup
(53.345ms eager vs29.258ms fused) on the older likelihood formulation, not a
new measurement for corrected squared. See uv_pair_fusion_results.md.

Corrected model removes pairwise wrapped phase and invented CT-Z displacement,
and separates unary V≈Z prior. Results practically unchanged: no meaningful
improvement in global winding resolution. This does not test moment-based O(K)
acceleration; still compiled all-to-all evaluation.
