# Moving-window perpendicular refinement

## Change

Perpendicular alternating projections now search B around its current arc,
update B, then search A around its current arc using the updated B. Each search
still covers plus/minus64base voxels, clipped to physical fiber endpoints, but
its center advances on subsequent iterations. Initial seeds no longer impose a
fixed total travel limit. Parallel projection is unchanged; its reciprocal
checks remain off by default.

The shared configuration defaults `localMovingProjectionWindows=true`;
`false` restores old seed-fixed behavior. AFV JSON uses
`moving_projection_windows`. Historical configs explicitly select false.
The iteration cap remains configurable (default16); all distance, orientation,
support and winding tests are unchanged. Refinement remains local, not a global
closest-pair solver, and the final reciprocal test checks A only.

## Measurement protocol

Same full512,698fiber Paris4_v1 AFV and unsigned Lasagna normal sampler as the
[fixed-window experiment](parallel_reciprocal_results.md). Whole curves retained;
no inference, alignment, BP solve or constraint database rewrite.
RelWithDebInfo/GCC13,128threads, one run per4/16/64iteration setting. Timings are
single observations, not statistically established performance improvements.

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_moving_projection.json
/home/hendrik/.venv_las/bin/python global-fiber-sgd/summarize_reciprocal_sweep.py --moving
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_moving_validation.json
```

Raw results, binary hashes, input hashes, command logs and resource measurements:
`data/moving_projection_v1/`; aggregate:`moving_projection_results.json`.

Reciprocal rejection percentages use perpendicular attempts that actually reach
that check, not all candidates or unique final events. Adaptive refinement and
deduplication change denominators across runs. With moving windows, iterations
can also reach different encounters farther along the curves. The boundary
counter now means the **last projection window**, not the initial seed window;
the original meaning remains applicable to fixed-mode runs.

## Full-volume results

All rows below have parallel reciprocal checks disabled.

| Window mode | Iterations | Perpendicular checks | Reciprocal failures | Rejected % of checked | Accepted perpendicular | Accepted parallel | Accepted total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed seed (previous) | 4 | 11,860,359 | 2,670,276 | 22.514293% | 1,153,778 | 3,061,068 | 4,214,846 |
| Moving | 4 | 11,069,428 | 46 | 0.000416% | 483,446 | 3,061,441 | 3,544,887 |
| Moving | 16 | 11,069,459 | 0 | 0% | 483,271 | 3,062,208 | 3,545,479 |
| Moving | 64 | 11,069,487 | 0 | 0% | 483,271 | 3,064,310 | 3,547,581 |

| Window mode / iterations | All candidate attempts | Deduplicated attempts | Extraction seconds | Native wall seconds | Runner wall seconds |
|---|---:|---:|---:|---:|---:|
| Fixed / 4 | 32,514,051 | 9,436,284 | 271.08 | 301.60 | 303.78 |
| Moving / 4 | 31,811,719 | 12,608,376 | 237.63 | 269.10 | 271.67 |
| Moving / 16 | 31,811,442 | 12,608,760 | 222.44 | 252.51 | 255.10 |
| Moving / 64 | 31,812,783 | 12,608,766 | 223.47 | 253.79 | 256.37 |

Moving4 has43of46failures on its last artificial window boundary; no physical
endpoint failures. Moving16/64 have none. This resolves the observed reciprocal
rejection problem without disabling the perpendicular check.

**New default: moving windows,16iterations.** This eliminates the residual46
failures seen at4 while64offers no improvement in this check or in the accepted
perpendicular count. This is a practical cap, not proof of exact convergence:
16/64outputs are not identical; the shared adaptive refinement schedule creates
different candidates and accepted parallel counts differ by2,102(0.069%). Exact
event positions/weights were not exported for an identity comparison.

The perpendicular event count falls while deduplication rises: multiple seeds
can now reach the same encounter. The existing deduplicator merges perpendicular
events within8base voxels on both curves. Other rejection gates and refinement
candidate counts also change. These totals do NOT demonstrate either improved
or reduced spatial coverage; that would need an event/pair coverage comparison.
The historical exported database is intentionally unchanged.

## Validation details

140native tests passed. New off-grid crossing fixture checks fixed-window and
one-iteration failure versus successful moving-window refinement in both input
orders. Existing strict endpoint tests still pass. Explicit legacy mode exactly
matches all19,889saved pilot measurements. The first attempted fixture was too
grid-aligned to exercise window escape; its failed assertion was corrected
before any full experiment. No extraction guard was relaxed to pass tests.
After selecting the16default, rebuilt and reran140tests plus legacy pilot;
both passed again. Parallel-only regression compares fixed/moving output arc
positions and winding measurements exactly. No macOS/arm64 run was performed;
the change uses existing portable scalar C++ geometry operations.
