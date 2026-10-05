# Seven-snapshot evaluation — 2026-10-05

- Output `/ephemeral/hendrik/fiber3d_train2/comparisons/sideband1_222625_seven_20261005`.
- Existing template `.fiber_eval_active.json` SHA256
  `aa5857cfbaddd8a4a8825c49ce01e4ebeddc96c30a966eabf0a4e1ea48d07e49`.
- Frozen selection uses both raw best choices and current; both EMAs best/current.
- Previous task log archived as evaluation_results_20261004.md.
- Independent plan review approved. Frozen samples: best_trace 6,604,800,
  best raw 7,948,800, current raw/EMA99/EMA999 9,318,400,
  best EMA99 9,228,800, best EMA999 9,203,200. Selection hashes/metrics
  preserved in series.json. Best_trace remains the earlier 14-restart reviewed32
  selection; its full91 rerun is a separate measurement.

## Completed results

Every evaluation completed 182 directions with no infrastructure failures.
Reported benchmark wall time 227–233 seconds each (excludes preparation/queueing).
Native binary SHA256 stayed
`78e14240bd2fbd475eb3622a16116debf7b61179de153a3c570ee503f09c8fe0`;
template hash unchanged before every entry. No precision/tracer/code changes.

| Snapshot | Samples | 0139 | 0332 | 1299 | 1451 | Paris4 | Total | Implied mm | Previous total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw best by restarts | 6,604,800 | 5 | 0 | 3 | 2 | 9 | 19 | 69.31 | 19 |
| Raw best by loss | 7,948,800 | 6 | 0 | 3 | 0 | 9 | 18 | 72.96 | 19 |
| Raw current | 9,318,400 | 8 | 0 | 2 | 0 | 10 | 20 | 66.01 | 25 |
| EMA99 best | 9,228,800 | 9 | 0 | 4 | 0 | 10 | 23 | 57.76 | 20 |
| EMA99 current | 9,318,400 | 11 | 0 | 3 | 0 | 11 | 25 | 53.31 | 18 |
| EMA999 best | 9,203,200 | 9 | 0 | 2 | 0 | 8 | 19 | 69.31 | 19 |
| EMA999 current | 9,318,400 | 8 | 0 | 5 | 0 | 10 | 23 | 57.76 | 18 |

Previous totals are from the October4 selected checkpoints (different sample
counts except unchanged raw best-by-restarts). Implied mm uses directed total
1386.175605/(restarts+1), not equal-scroll weighting. Small differences should
not be overinterpreted given previously observed directed-trace variation.
Unchanged best-by-restarts reproduced all per-scroll counts from the prior run.

Executed existing approved `run_fiber3d_experiment trace-series-prepare`, then
seven sequential `trace-series-next` invocations. Selection JSON contains exact
snapshot SHA256 and dense-loss or reviewed-restarts selection metric. Every
series entry verified the original training root alive after evaluation;
training reinitializes its workers after final grant release. GPUs6–7 untouched.
Detailed results under output directory above, `series.json`, each variant's
`trace_audit.json`, `full/summary.json`, and `results_after_current999.json`.
Final live check: manager generation29 running on GPU0–5; training progressed
through samples 9,323,520 after evaluation, not merely an alive idle root.
