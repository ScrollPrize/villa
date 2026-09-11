# Collection run20260911_182306 best3k: full five-scroll test

Completed2026-09-11 on code`44cdc1311`; no production or tracer changes.
All50locked test fibers/100directions, all5actualscrolls/6collections.
The new checkpoint improves over the previous collection checkpoint but does not
beat the classic globally on failures/mean span. All metrics below are explicitly
paired; no checkpoint is selected based on this new trace result.

## Global comparison

| Metric | Classic best91.5k | Previous collection best17k | New collection best3k |
|---|---:|---:|---:|
| Mean before restart mm | 3.428 | 2.486 | 3.299 |
| Median mm | 0.180 | 0.062 | 0.137 |
| P05 mm | 0.00117 | 0.00056 | 0.00173 |
| P95 mm | 17.670 | 10.406 | 17.414 |
| Failures | 1995 | 2885 | 2144 |
| Zero-length failures | 65 | 99 | 65 |
| Failure-free directions | 4 | 4 | 4 |
| Actual attempts | 2095 | 2984 | 2240 |
| Terminal suffixes | 100 | 99 | 96 |
| Mean all attempts mm | 3.856 | 2.706 | 3.606 |
| Mean positive seeded spans mm | 3.979 | 2.799 | 3.714 |
| Historical corpus reliability mm | 4.049 | 2.800 | 3.768 |
| Failed polyline mean mm | 3.493 | 2.544 | 3.358 |
| Coordinator seconds | 516.699 | 400.888 | 404.521 |
| Native seconds incl. readiness | 469.562 | 353.194 | 356.295 |
| Inferred 256³ tiles | 34441 | 34162 | 34216 |
| Model calls | 4760 | 4531 | 4512 |
| Summed worker seconds | 3402.995 | 2702.610 | 2716.120 |
| Peak allocated GiB | 35.005 | 36.677 | 36.677 |
| Peak reserved GiB | 54.672 | 46.648 | 46.648 |

Distances are millimeters. Primary distribution is reference progress during
failure-ended attempts, retaining zeros and excluding successful terminal
suffixes. Mean is pooled over spans, not an unweighted mean of scrolls.
P05/P95 use inclusive linear interpolation. Historical corpus reliability is
full directed reference length/(failures+1), not empirical mean before restart.

## Per-scroll classic versus new

| Scroll | Classic mean | New mean | Classic median | New median | Classic P05 | New P05 | Classic P95 | New P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 2.233 | 2.257 | 0.146 | 0.069 | 0.00014 | 0.00000 | 11.232 | 13.359 |
| PHerc0332 | 12.994 | 17.088 | 0.309 | 0.115 | 0.00000 | 0.00000 | 48.898 | 66.665 |
| PHerc1299 | 3.418 | 2.153 | 0.790 | 0.204 | 0.02598 | 0.00404 | 17.738 | 13.344 |
| PHerc1451 | 2.341 | 1.017 | 0.762 | 0.083 | 0.06766 | 0.00335 | 10.069 | 8.071 |
| PHercParis4 | 3.785 | 3.859 | 0.162 | 0.163 | 0.00171 | 0.00644 | 20.964 | 24.449 |
| Global | 3.428 | 3.299 | 0.180 | 0.137 | 0.00117 | 0.00173 | 17.670 | 17.414 |

## Per-scroll previous collection versus new

| Scroll | Prev17k mean | New mean | Prev17k median | New median | Prev17k P05 | New P05 | Prev17k P95 | New P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 2.376 | 2.257 | 0.066 | 0.069 | 0.00000 | 0.00000 | 11.551 | 13.359 |
| PHerc0332 | 11.884 | 17.088 | 0.813 | 0.115 | 0.00000 | 0.00000 | 59.412 | 66.665 |
| PHerc1299 | 1.503 | 2.153 | 0.077 | 0.204 | 0.00098 | 0.00404 | 10.188 | 13.344 |
| PHerc1451 | 1.149 | 1.017 | 0.216 | 0.083 | 0.00413 | 0.00335 | 5.814 | 8.071 |
| PHercParis4 | 2.577 | 3.859 | 0.058 | 0.163 | 0.00143 | 0.00644 | 9.489 | 24.449 |
| Global | 2.486 | 3.299 | 0.062 | 0.137 | 0.00056 | 0.00173 | 10.406 | 17.414 |

## Counts

| Scroll | Fibers | Classic failures | Prev17k failures | New3k failures | Classic zero | Prev17k zero | New3k zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 10 | 564 | 523 | 548 | 26 | 34 | 33 |
| PHerc0332 | 3 | 27 | 34 | 23 | 6 | 4 | 3 |
| PHerc1299 | 5 | 159 | 359 | 261 | 1 | 10 | 5 |
| PHerc1451 | 2 | 19 | 61 | 64 | 0 | 0 | 0 |
| PHercParis4 | 30 | 1226 | 1908 | 1248 | 32 | 51 | 24 |
| Global | 50 | 1995 | 2885 | 2144 | 65 | 99 | 65 |

Both Paris4 collections pooled here; per-collection data retained in JSON.
Stored labelPHerc1229 maps to actualPHerc1299. All three evaluate the same
8081.558059625directed reference mm. Small groups and correlated forward/reverse
traces mean these are descriptive statistics, not confidence intervals.

## Snapshot capture and model

Original training file:
`/ephemeral/hendrik/fiber3d_train2/runs/collection_20260909_5scroll_128_sd1_16x8_20260911_182306/snapshots/best.pt`.
Training was ongoing; captured best at step3000,384000samples,
metric`test/loss_total`=0.43289151958754807. SHA256:
`ee4595cd702029ad1f2edbbd56ebaf8006991b57601f3bbc7499b0e1a4970b18`.
This selected dense test-loss best at request time, not tracing-best or a later
moving best.pt. Existing freeze_checkpoint helper staged immutable weights at:
`/tmp/fiber3d_snapshot_20260911_182306/live_inference/_models/ee4595cd702029ad1f2edbbd56ebaf8006991b57601f3bbc7499b0e1a4970b18/snapshot.pt`.
The live runner subsequently froze the same digest at the durable global cache:
`/ephemeral/hendrik/vc3d_cache/live_inference/_models/ee4595cd702029ad1f2edbbd56ebaf8006991b57601f3bbc7499b0e1a4970b18/snapshot.pt`.
Canonical file hash rechecked after completion. Original best.pt may subsequently
change as training continues.

Authoritative checkpoint configuration: UNet base16/depth5, GroupNorm,
pixelshuffle, unconditioned single branch,7raw channels,native output_stride4,
zscore input, bf16 inference. Same model-output geometry as previous collection:
256³input→64³native output; requested total factor4/native4=1. No second
downsample. InputL1→outputL3/base scale8. Classic has a dense head then filtered
downsample; same output lattice does not imply identical model math.

No GT-heading correction or additional GT conditioning applied; the earlier
[initialization issue](classic_full_5scroll_20260911.md) remains logged for later.

## Data/config/commands

Same split:
`/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910`.
Same CT/normal mappings and all6collections as previous full run:
0139(10),0332(3),1299(5),1451(2),Paris4_eval_fibers(24),Paris4_fibers(6).
Workload differs from eval_collection_best17k_full.json only in checkpoint path,
hash, output_root, inference_cache_namespace. Inputs fetched from existing cache;
all5inference stores report warm_chunks=0.

From /home/hendrik/villa2, unchanged approved runner:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best3k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best3k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best3k_full.json
```

Fresh output namespace:
`/ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best3k_full256_b8_20260911_v1/full`.
Prepared full.json and full_bindings.json live one directory above.
Saved summary/config/binding digests are in companion JSON. Original measured
workload references temporary staged weights; for a later rerun create a NEW
namespace/config using the durable canonical snapshot above. Do not overwrite
prepared evidence or reuse finished inference.

Exclusive manager priority100/all8H10080GBs/224logicalCPUs, task
`f9b57683-156c-4147-84c2-61ff86d6ded3`.
Pausable training yielded before grant. At completion evaluator is terminal,
finished/completed, held/grantedCPU0/GPUs{}. Training task
`58fd84af-60cb-47b3-8fa4-c9acdb4bf251` returned to applied/grant_available,
unpaused with128CPUs/all8GPUs. No manual training pause/resume or settings changes.

## Timing/validation

Same256tile96overlap32border32outputchunk,batchcap8,128readers,256inputslots,
128resultslots,4publicationworkers,bf16model,float16chunk-major accumulation;
GCC13RelWithDebInfo binary /tmp/vc-live-native-cmake/bin/vc_fiber_trace_metric.
Native beam8/lookahead2,trace step4/base scale2,normalweight0.1/tangent10,
ellipsoid radii20/80basevox,matchrefine1 unchanged.

One full run per snapshot. Different weights/paths/inferred work and source-cache
warmth prevent an isolated architectural speedup claim. Coordinator time includes
setup/tracing/inference/drain but excludes manager waiting, initial validation and
finalcleanup. Native includes readiness waits; worker seconds summed over workers
include transfers/preprocessing/output work. Batch timing mean/min/p50/p95/max
and histogram retained in JSON. Bulk rate about100tiles256³/s.

- Prepare and validate-full verified snapshot/config, every locked annotation,
  membership and normal bindings.
- Full command exited0;100/100complete directions.
- All100saved artifacts,2144failure spans, source units/identities, credited length
  and actual segment counts independently checked by existing analysis script.
- 96terminal suffixes:4directions end on failure/reset at endpoint, so no fictitious
  terminal segment added. Actual2240attempts,65zero-length failures.
- All requested/read/GPU/accumulation/publication queues drained; input/result slots
  returned.132044incidental partial chunks are unrequested, not missing work.
- 34216tiles/4512modelcalls,98.4569%tiles in full8batches,10988published spatial chunks
 (threechannels each); peak publication backlog18.
- Native-stride code unchanged and already regression-tested previously; no new
  unit-test rerun or GPU training performed. Actual managed full execution and
  artifact assertions validate this configuration-only experiment.
- Independent plan/config review passed; no dropped scope or new runtime failures.

## Read-only analysis reproduction

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py /ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best3k_full256_b8_20260911_v1/full
```

[Machine-readable result](collection_182306_best3k_full_20260911.json) retains
per-scroll/collection values, checkpoint metadata, timing comparison and manager
release/training-resumption evidence. Prior records:
[classic](classic_full_5scroll_20260911.md),
[previous collection](collection_best17k_full_20260911.md).
