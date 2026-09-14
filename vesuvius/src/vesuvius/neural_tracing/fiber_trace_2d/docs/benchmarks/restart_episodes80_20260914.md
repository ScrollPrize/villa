# Restart episode reporting — 2026-09-14

80 base voxel gap from the preceding actual reset; first event always counted; exact80 starts another episode. Per directed eligible run. Raw events/visual IDs preserved. Reported restarts below are filtered only. Means include terminal spans.

Classic: completed CPU-only no-vis rerun of50 fibers/180 eligible directed runs in34.721s. No inference worker pool, zero inferred tiles/published chunks.

Current EMA99 u32000 tuned: CPU-only rerun stopped on missing Paris4 cache chunk[213,57,78]. Saved original50-fiber/100-direction traces recomputed instead; they use whole fibers without current CP/Lasagna exclusions. NOT a directly comparable model ranking against Classic. Need missing inference before a matched eligible-span rerun.

## Classic eligible-span rerun

Distances mm, before → after. Only filtered restart counts shown.

| Scroll | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| PHerc0139 | 166 | 3.694 → 5.152 | 0.611 → 1.890 | 0.013 → 0.112 | 16.779 → 23.261 |
| PHerc0332 | 12 | 13.286 → 23.620 | 2.543 → 17.667 | 0.007 → 0.910 | 48.820 → 57.132 |
| PHerc1299 | 44 | 6.674 → 7.539 | 4.260 → 5.386 | 0.116 → 0.505 | 18.079 → 18.912 |
| PHerc1451 | 3 | 6.910 → 6.910 | 9.548 → 9.548 | 0.770 → 0.770 | 11.396 → 11.396 |
| PHercParis4 | 285 | 9.984 → 12.565 | 2.176 → 4.459 | 0.013 → 0.197 | 48.230 → 51.581 |
| All | 510 | 7.631 → 9.953 | 1.466 → 3.454 | 0.013 → 0.179 | 35.858 → 42.754 |

## Current32000 tuned — historical whole-fiber saved traces only

Distances mm, before → after. Only filtered restart counts shown.

| Scroll | Restarts | Mean | Median | P05 | P95 |
|---|---:|---:|---:|---:|---:|
| PHerc0139 | 169 | 3.129 → 7.367 | 0.115 → 2.139 | 0.006 → 0.226 | 15.729 → 30.697 |
| PHerc0332 | 10 | 17.453 → 29.451 | 2.452 → 16.058 | 0.002 → 0.334 | 98.885 → 124.415 |
| PHerc1299 | 109 | 2.677 → 5.040 | 0.201 → 1.334 | 0.010 → 0.213 | 13.825 → 35.235 |
| PHerc1451 | 30 | 2.098 → 2.777 | 0.447 → 0.673 | 0.045 → 0.156 | 10.709 → 13.339 |
| PHercParis4 | 439 | 4.194 → 11.070 | 0.087 → 1.471 | 0.011 → 0.218 | 23.990 → 75.049 |
| All | 757 | 3.927 → 9.430 | 0.106 → 1.497 | 0.009 → 0.217 | 22.139 → 50.990 |

## Reproduction

Approved runner `/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py`:

- `cached-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_restart80_cached_20260914.json` (completed).
- Same mode with `eval_current32000_restart80_cached_20260914.json` (explicit cache miss; preserved incomplete output).
- `report-metrics` with either config recomputes saved metrics without tracing/inference.

Outputs are under `/ephemeral/hendrik/fiber3d_train2/evaluations/{classic,current32000}_restart80_cached_20260914/`. Existing outputs are never overwritten by cached-full. No GPU manager allocation: current exclusive launcher requires GPUs; direct CPU-only mode is deliberate, no isolated performance claim.
