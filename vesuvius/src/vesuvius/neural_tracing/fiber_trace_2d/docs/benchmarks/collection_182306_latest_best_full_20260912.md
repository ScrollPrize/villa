# Latest28k and best11k — full five-scroll comparison, 2026-09-12

## Outcome

Both requested evaluations completed all50locked test fibers /100directions across5actualscrolls. Latest28k ran first, then the separately frozen best11k. Each used fresh inference with shared source fetches and an exclusive8GPU manager grant. Classic and the last best5k are saved baselines, not rerun.

Latest28k:2516restarts,mean3.089281mm,median0.049606mm. Best11k:2214restarts,mean3.492462mm,median0.085944mm. Best11k improves mean8.51% over lastbest5k and13.05% over latest28k, but remains9.46% below classic. Best11k's median is slightly below best5k despite the better mean. Individual distribution statistics need not move together.

All distances below include terminal segments. **Bold marks the best within each scroll/metric**: fewer restarts, larger distances. Full-reference mean=L/(F+C); percentiles pool the same complete intervals, not per-fiber averages. Both Paris4 collections are pooled; PHerc1229 labels refer to actualPHerc1299.

## Global

| Model | Restarts ↓ | Mean mm ↑ | Median mm ↑ | P05 mm ↑ | P95 mm ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | **1995** | **3.858** | **0.210** | **0.01122** | **19.465** |
| Last best5k | 2411 | 3.218 | 0.093 | 0.00951 | 16.819 |
| Latest28k | 2516 | 3.089 | 0.050 | 0.01045 | 15.583 |
| Best11k | 2214 | 3.492 | 0.086 | 0.00943 | 17.115 |

## PHerc0139

| Model | Restarts ↓ | Mean mm ↑ | Median mm ↑ | P05 mm ↑ | P95 mm ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 564 | 2.384 | **0.175** | 0.00709 | 11.631 |
| Last best5k | 476 | 2.807 | 0.115 | 0.00623 | 14.985 |
| Latest28k | **439** | **3.033** | 0.078 | **0.00819** | **16.510** |
| Best11k | 451 | 2.956 | 0.096 | 0.00592 | 15.611 |

## PHerc0332

| Model | Restarts ↓ | Mean mm ↑ | Median mm ↑ | P05 mm ↑ | P95 mm ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | 27 | 14.279 | 4.206 | 0.00743 | 48.805 |
| Last best5k | 20 | 18.124 | 3.140 | 0.00222 | 102.431 |
| Latest28k | **15** | **22.439** | **15.340** | 0.00799 | 105.638 |
| Best11k | 17 | 20.488 | 9.718 | **0.00875** | **139.852** |

## PHerc1299

| Model | Restarts ↓ | Mean mm ↑ | Median mm ↑ | P05 mm ↑ | P95 mm ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | **159** | **3.549** | **0.803** | **0.03149** | **17.573** |
| Last best5k | 224 | 2.563 | 0.226 | 0.01340 | 15.594 |
| Latest28k | 250 | 2.307 | 0.134 | 0.01126 | 10.763 |
| Best11k | 212 | 2.702 | 0.235 | 0.01335 | 13.616 |

## PHerc1451

| Model | Restarts ↓ | Mean mm ↑ | Median mm ↑ | P05 mm ↑ | P95 mm ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | **19** | **4.106** | **1.862** | **0.07072** | **17.222** |
| Last best5k | 65 | 1.369 | 0.116 | 0.00781 | 9.365 |
| Latest28k | 43 | 2.009 | 0.176 | 0.01429 | 9.265 |
| Best11k | 32 | 2.623 | 0.864 | 0.01447 | 11.192 |

## PHercParis4

| Model | Restarts ↓ | Mean mm ↑ | Median mm ↑ | P05 mm ↑ | P95 mm ↑ |
|---|---:|---:|---:|---:|---:|
| Classic | **1226** | **4.295** | **0.185** | **0.01276** | **23.435** |
| Last best5k | 1626 | 3.276 | 0.077 | 0.01124 | 17.007 |
| Latest28k | 1769 | 3.020 | 0.040 | 0.01112 | 12.310 |
| Best11k | 1502 | 3.536 | 0.069 | 0.01157 | 18.143 |

## Runtime comparison

One full run per snapshot; same geometry/batching/tracing settings. Coordinator time is measured before final context teardown and excludes manager admission; native time includes readiness waits. These are not pure model-inference timings.

| Model | Coordinator s | Native s | Tiles | Model calls | Worker sum s | Batch mean / median / P95 s | Full8-tile fraction % | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Classic | 516.699 | 469.562 | 34441 | 4760 | 3402.995 | 0.7149 / 0.7311 / 1.3188 | 98.116 | 35.005 / 54.672 |
| Last best5k | 402.933 | 354.757 | 34090 | 4561 | 2713.625 | 0.5950 / 0.5943 / 0.8621 | 98.234 | 36.677 / 46.648 |
| Latest28k | 399.916 | 350.331 | 33857 | 4440 | 2668.878 | 0.6011 / 0.5897 / 0.8707 | 98.556 | 36.677 / 46.648 |
| Best11k | 398.684 | 351.003 | 33939 | 4511 | 2686.838 | 0.5956 / 0.5920 / 0.8789 | 98.082 | 36.677 / 46.648 |

The new runs take about6m40s for evaluation, essentially unchanged from lastbest5k and about23% less than classic. Different trajectories cause small differences in inferred tile count. All requested work queues drained and warm prediction chunks=0 in both runs. Unrequested incidental candidates need not drain. Manager root-start to final contact:latest494.511s,best529.976s,including admission/setup/teardown; distinct from measured evaluation time.

## Checkpoints and provenance

Run: `/ephemeral/hendrik/fiber3d_train2/runs/collection_20260909_5scroll_128_sd1_16x8_20260911_182306`.

Both files were frozen at task start (2026-09-12 around05:52UTC) with the existing live_evaluate.freeze_checkpoint helper, before either GPU run. No chasing mutable snapshot files during execution.

| Selection | Saved step | Samples seen | Dense test/loss_total | SHA256 |
|---|---:|---:|---:|---|
| current.pt | 28000 | 3584000 | 0.38799891914012 | 114a13003b9a33f78b07cc61a83abd710a2f4bf8408ea16909c249857018817b |
| best.pt | 11000 | 1408000 | 0.3603664571877386 | ca5349cd785b447109bc8be46f211501f1c18d1575e5e7e5c3998f33e0924b34 |

best.pt is selected by lowest dense test/loss_total, not tracing performance. Both models remain UNetbase16/depth5,GroupNorm,pixelshuffle,native output_stride4. Original/frozen paths and full metadata in individual JSON reports. Staging root /tmp/fiber3d_snapshot_20260912_0552; canonical immutable copies in /ephemeral/hendrik/vc3d_cache/live_inference/_models/<SHA>/snapshot.pt. Both canonical hashes rechecked after their evaluations.

Code clean at starting revision b32f31a5538a09a5600890ee35b17e256a759e6c. No production code changes. Native GCC13 RelWithDebInfo binary /tmp/vc-live-native-cmake/bin/vc_fiber_trace_metric SHA2567a08b63d9cac8f2afd057af16326decabfd09cd36e1802dca52702d06e89c3a2.

## Data, settings and validation

Locked split /ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910.50fibers:0139=10,0332=3,1299=5,1451=2,Paris4eval=24+fibers=6. Total8081.558059625directed reference mm. Exact annotation/normal/source hashes and mappings in each output root's full_bindings.json.

Both new configs differ from lastbest5k only in snapshot/hash/output/namespace. Same inputL1,totalfactor4→baseL3,256tile96overlap32border32outchunks,bf16,cap8,float16chunk-major accumulator,128readers256inputslots128resultslots,64I/O8decode threads. Same beam8/lookahead2,step4atbasefactor2,normalweight0.1/tangent10,ellipse20normal/80tangentialbasevox,matchrefine1. Native stride4 avoids additional decimation. No GT-heading/reset/threshold changes.

Both prepare and validate-full passed locked membership/config/manifests. Both full runs completed100/100 with no incomplete marker; existing analysis script audited100raw artifacts each. Current schema source5/replay-set2 uses complete-reference-partition-v1. Split at directional failure arcs over[0,L], including final/repeated zero intervals; linear percentile interpolation at(n−1)q. Forced reset advance belongs to full-reference accounting, not successfully traced physical length. Legacy physical/failure-only diagnostics remain separate in JSON.

Independent reviewer approved plan, exact config deltas and hashes. Existing successful pilot reused as prerequisite, not substituted for full test. Prior metric regression tests already passed; no new production changes. No installs or training changes.

## Reproduction

From /home/hendrik/villa2, using the existing approved entrypoint:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_current28k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_current28k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_current28k_full.json
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py /ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_current28k_full256_b8_20260912_v1/full
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best11k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best11k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best11k_full.json
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py /ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best11k_full256_b8_20260912_v1/full
```

Output/namespace are single-use; change them for an intentional rerun. No inference reuse. Manager priority100,exclusive224logicalCPU/8H100. Latest task2a9f6e3b-29e3-4c46-8d63-9f8b9853ab1a;best task8243bed6-1a8a-40fc-9ea6-6b13f276f844. First released before second started. Both commands exited0; both manager tasks terminal finished/completed with0heldCPU/0GPU. Training task58fd84af-60cb-47b3-8fa4-c9acdb4bf251 restored to applied128CPU/8GPU. Final manager/exit records stored in accompanying JSON; training yielding/restoration done by manager only.

## Limitations and records

One run each; no significance claim. Forward/reverse observations are correlated and0332/1451groups small. Asynchronous float16 accumulation and batch kernels may affect low-order inference/trajectory values. Wall time is not a controlled pure-compute throughput measurement. Baselines retain their original implementations/timings with corrected metric summaries. No deferred scope or omitted collection.

[Latest28k exact report](collection_182306_current28k_full_20260912.json), [Best11k exact report](collection_182306_best11k_full_20260912.json), [combined machine-readable comparison](collection_182306_latest_best_full_20260912.json).
