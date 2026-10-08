# N2 ordinary-only correctness and shift scores — 2026-09-28

## Result

Same548 original source fibers and29,597 original pieces as the fixed-reference N2 experiment. Original source and piece ID sets were checked exactly, not just counts.

**The winding solve includes no reference nodes, cross factors, fixed states, or reference-conditioned initialization.** References are used to select the requested neighborhood and, after the ordinary solve is frozen, for the existing independent winding readout. Existing post-split reference attachments are unchanged.

All components were retained. Without reference anchors there are28 integer gauges with reference observations. Gauge0 is the **only** gauge with observations for all26 reference fibers and accounts for2826/2861 reference observations (**98.78%**). The other27 gauges cover partial ladders, often with unbounded/tied solutions. No aggregate global ladder is manufactured from incompatible gauges.

The table reports the **full-coverage gauge0's deterministic rank-1 readout**, not a silently selected best-scoring component. All28 gauge readouts are preserved in the adjacent JSON and raw report.

| Disable / step cost | Exact references | Correctness | Offset-change count K | Shift score | Tied/unbounded best reference rows |
|---|---:|---:|---:|---:|---:|
| 3 / 10 | 4/26 | 15.38% | 21 | 19.23% | 10/26 (38.46%) |
| 10 / 3 | 2/26 | 7.69% | 22 | 15.38% | 14/26 (53.85%) |
| 10 / 10 | 4/26 | 15.38% | 17 | 34.62% | 7/26 (26.92%) |

Correctness uses the established calibrated exact half-winding match. Shift score is `100*(1-K/26)`, where K counts changes in predicted-minus-reference offset between consecutive references; a global offset is ignored. Larger is better. This is not a unit-winding displacement error or a percentage of correctly traced voxels.

Tie/unbounded counts are shown because the historical deterministic rank-1 rule selects one candidate even when the optimum is nonunique. They are not all unique confident predictions.

## Interpretation

These are poor ordinary-only results; the earlier conditioned defect percentages did not imply a nearly correct reference-free ladder.10/10 gives the best shift score of these three, but only4/26 references exactly match.

Every reference in gauge0 has nonzero minimum hard-sign infringements in its independent readout, for all three settings. Thus this is not merely an arbitrary global winding offset. The ordinary assignments conflict with the held-out reference evidence, while the earlier reference-fixed solve could instead move/disable ordinary pieces to accommodate that evidence.

References are used only for post-solve gauge sign/offset calibration in the established benchmark. This is an unconditioned **solve**, not reference-free metric calibration. Neighborhood selection also uses reference geometry by design.

The amount removed by a reference-based filtering/oracle evaluation is **not** measured here. Neither these scores nor the earlier conditioned disabled-piece percentages are that filtering metric.

## Full-coverage ladder

| Reference | 3/10 | 10/3 | 10/10 |
|---:|---:|---:|---:|
| 0 | 13 | 4.5 | 6 |
| 0.5 | 12.5 | 2 | 5.5 |
| 1 | 13 | 4.5 | 6 |
| 1.5 | 9 | 2 | 5.5 |
| 2 | 14 | 2 | 7 |
| 2.5 | 14 | 2 | 6.5 |
| 3 | 12 | 2 | 5 |
| 3.5 | 14 | 2 | 7 |
| 4 | 9 | 6.5 | 2.5 |
| 4.5 | 13 | 5.5 | 4.5 |
| 5 | 12 | 5 | 5 |
| 5.5 | 12.5 | 5 | 5.5 |
| 6 | 12 | 4.5 | 6 |
| 6.5 | 13 | 4.5 | 5.5 |
| 7 | 13 | 6.5 | 6 |
| 7.5 | 13 | 2 | 6.5 |
| 8 | 12 | 2 | 6 |
| 8.5 | 12 | 5.5 | 4.5 |
| 9 | 9 | 6.5 | 2 |
| 9.5 | 12 | 5.5 | 4.5 |
| 10 | 9 | 6 | 2 |
| 10.5 | 14 | 2.5 | 5.5 |
| 11 | 13 | 3 | 6 |
| 11.5 | 11.5 | 6 | 4.5 |
| 12 | 12 | 5.5 | 5 |
| 12.5 | 12.5 | 6 | 4.5 |

## Per-gauge coverage (no solver filtering)

| Run | Gauge | References observed /26 | Observations | Exact /26 | Shift score |
|---|---:|---:|---:|---:|---:|
| n2-d3-c10 | 0 | 26 | 2826 | 4 | 19.23% |
| n2-d3-c10 | 1 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 2 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 3 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 4 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 5 | 1 | 2 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 6 | 2 | 2 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 7 | 1 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 8 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 9 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 10 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 11 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 12 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 13 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 14 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 15 | 3 | 3 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 16 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 17 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 18 | 2 | 2 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 19 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 20 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 21 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 22 | 2 | 2 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 23 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 24 | 1 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 25 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 26 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d3-c10 | 27 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 0 | 26 | 2826 | 2 | 15.38% |
| n2-d10-c3 | 1 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 2 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 3 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 4 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 5 | 1 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 6 | 2 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 7 | 1 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 8 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 9 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 10 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 11 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 12 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 13 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 14 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 15 | 3 | 3 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 16 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 17 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 18 | 2 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 19 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 20 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 21 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 22 | 2 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 23 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 24 | 1 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 25 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 26 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c3 | 27 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 0 | 26 | 2826 | 4 | 34.62% |
| n2-d10-c10 | 1 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 2 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 3 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 4 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 5 | 1 | 2 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 6 | 2 | 2 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 7 | 1 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 8 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 9 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 10 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 11 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 12 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 13 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 14 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 15 | 3 | 3 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 16 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 17 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 18 | 2 | 2 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 19 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 20 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 21 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 22 | 2 | 2 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 23 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 24 | 1 | 2 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 25 | 1 | 1 | 1 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 26 | 1 | 1 | 0 | unavailable (partial/ambiguous) |
| n2-d10-c10 | 27 | 1 | 1 | 1 | unavailable (partial/ambiguous) |

Partial-gauge correctness retains the denominator26 rather than scoring isolated references as a misleading100%; their shift scores are unavailable. The top-level all-gauge correctness/shift fields deliberately remain null because the existing metric requires a unique reference ladder.

## Reproducibility

Frozen2048 Paris4 beam8/40x80/45-degree/reverse-consistent trace artifact:
`/ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1/beam-crop-new2048-coverage40x80-angle45-reverse-v2.zarr`.

Inference basis: EMA999 best u22000 from `5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260915_212757`, prediction `7ff0ce6c`; no new inference or tracing. Same source-certified local orthogonal events and positive physical-winding L1 continuation as the conditioned tests. Parallel factors off; joint H/V/winding/defect states; fixed phase0.5/scale0.822;64workers per run;500-message cap. All three completed at that cap; no cap extension.

Solver graph:548 sources,29,597 ordinary pieces,42,102 ordinary factors.0 reference nodes and0 reference-cross factors. Selection/readout graph is separate. RelWithDebInfo. One run each, three runs concurrent.

| Run | Command wall seconds |
|---|---:|
| n2-d3-c10 | 117.28 |
| n2-d10-c3 | 115.59 |
| n2-d10-c10 | 109.19 |

### Approved commands

Build:
```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
```

Run each config (replace the suffix with `n2-d10-c3` or `n2-d10-c10` for the other two):
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_n2_unconditioned_n2-d3-c10_20260928.json
```

Analyze existing outputs without rerunning solves:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_n2_unconditioned_report_20260928.json
```

Validation:123 native winding tests and27 Python report/runner/sweep tests passed. New tests cover cohort identity, reference-leakage declarations, existing correctness/shift definitions, incomplete ladders, multiple gauges, and preserved ambiguity flags. Independent plan and code review found no wiring blocker. Cohort IDs additionally verified against full conditioned artifacts and hashes in all three real runs.

### Files

Output root:
`/ephemeral/hendrik/fiber_crop_bench/20260928_n2_unconditioned_v1/`

- `summary.json`: all gauge/candidate/ambiguity rows, full cohort IDs, input comparison hashes and run provenance.
- `<run>_neighborhood_cohort.json`: selected original source/piece IDs and graph separation audit.
- `<run>.log`: full native diagnostics and independent reference candidate tables.
- `<run>_consistency.csv`, `<run>_winding_factors.csv`: ordinary states/factors.
- `<run>.run.json`, `<run>.source.patch`, `<run>.resources.txt`: exact command, binary/input hashes, code patch, timing/resources.

Adjacent JSON retains all per-gauge metrics/readout summaries and provenance; large ID arrays and the full candidate alternatives remain in the raw summary. No default tuning promotion.
