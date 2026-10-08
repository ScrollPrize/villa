# Short200: Classic and fiberlet comparison

Same immutable200forward pieces,662.089848mm as short200_best_u22000_20260916.md.
Classic completed; initial fiberlet u22000storage128run stopped for parallelism
correction (no complete quality result). Scroll-localv2run in progress.

## Classic

Snapshot `/home/hendrik/runs/s1a_128_1_single_8x8_20260801_084232/snapshots/best91_5k.pt`,
SHA256 `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17`.
Historical binding `classic_thickness40_grace160_20260914_v2/full`.
Output `/ephemeral/hendrik/fiber3d_train2/evaluations/short200_classic_20260916_v1`.

| Scroll | Pieces | Restarts | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 77 | 55 | 3.780 | 0.814 | 0.048 | 4.306 |
| PHerc0332 | 1 | 1 | 1.199 | 1.199 | 0.208 | 2.191 |
| PHerc1299 | 25 | 20 | 3.655 | 0.965 | 0.204 | 6.166 |
| PHerc1451 | 1 | 1 | 1.080 | 1.080 | 0.342 | 1.817 |
| PHercParis4 | 96 | 76 | 4.793 | 0.628 | 0.140 | 7.877 |
| All | 200 | 153 | 4.299 | 0.720 | 0.099 | 6.484 |

Agreed aggregate mean=combined reference distance/(restarts+1). Terminal spans
included in quantiles; same80voxel restart grouping/40normal/80tangent/160grace.
30.403s measured setup/replay/cleanup excluding grant wait and prior hashing;
8new inference tiles required (1publishedchunk),13.493s coordinator phase.
54/200pieces finish without reported restart versus141/200for u22000beam.
One timing sample, not a cold benchmark.
u22000beam reference62restarts/10.509mm; Classic153restarts/4.299mm.

## Fiberlet configuration

Same frozen u22000 and existing inference cache. Construction storage128basevoxels,
24workers/source, beam16/lookahead384/commit48, production anchor/path kernels.
Changing storage invalidates construction cache only, not inference cache.
Old beam summary predates canonical fiberlet fields; shared binder reconstructs
these from validated stored inference identity, no prediction mutation.
Use graph credited coverage alongside restarts; missing seeds can skip distance.
Output `/ephemeral/hendrik/fiber3d_train2/evaluations/short200_fiberlet_u22000_128_20260916_v1`.

## Commands

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_classic_20260916_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_fiberlet_u22000_128_20260916_v1.json
```

Both exclusive manager8GPU/128CPU. Immutableoutput; new directory on repeat.
Validated200cases and identical length for Classic.2focused binding tests pass;
git diff --check passes. Native binary unchanged, existing production implementation.

## Scroll-local correction (v2, in progress)

Initialv1stopped after531s (4931newtiles/2961publications last observed), roughly
7300–8000%CPU; outputpartial and published caches preserved. v2uses224principal
CPU permits, scroll-by-scroll sources, construction_workers224instead of24/source,
exact-owner paired generation rather than64hashlocks, and empty-owner memoization.
No kernel numerical changes. All8GPUs available through demand-onlymanager lease,
30second idle release, no whole-run GPUhold. Construction/inference caches reused;
the v2timing is consequently warmer, not a clean cold-vs-cold speedup measurement.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_fiberlet_u22000_128_scroll_20260916_v2.json
```

Outputs `short200_fiberlet_u22000_128_scroll_20260916_v2/scrolls/00`etc preserve
native localcaseIDs; finalcombined full/summary.json links scroll_artifacts.
Build/tests via approved runners:23replay/85trace/11reference,258Pythonpassed8skipped.


## Completed scroll-local v2 result

All200 pieces completed; wall1485.621s (24m46s),67reported failures,99.846% credited reference coverage. This run reused earlier inference/partial graph caches and filled additional misses; not a cold throughput comparison. Search-only time was not separately recorded.

| Scroll | Failures | Mean mm | Median mm | P05 mm | P95 mm |
|---|---:|---:|---:|---:|---:|
| All | 67 | 9.737 | 1.473 | 0.144 | 7.847 |
| PHerc0139 | 33 | 6.226 | 1.209 | 0.124 | 5.932 |
| PHerc0332 | 0 | 2.399 | 2.399 | 2.399 | 2.399 |
| PHerc1299 | 11 | 6.396 | 1.795 | 0.119 | 6.597 |
| PHerc1451 | 0 | 2.159 | 2.159 | 2.159 | 2.159 |
| PHercParis4 | 23 | 15.379 | 1.920 | 0.205 | 10.416 |

Comparison on same662.089848mm: Classic153failures/4.299mm, bestbeam62/10.509mm, fiberlets67/9.737mm. All values use combineddistance/(failures+1). Two graph failures in Paris4 were graph_exhausted/no_usable_seed_for_remaining_reference. Remaining failures were distance threshold events.
