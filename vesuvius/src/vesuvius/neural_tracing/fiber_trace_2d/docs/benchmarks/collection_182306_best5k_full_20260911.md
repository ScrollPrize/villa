# Current best5k: full five-scroll evaluation — 2026-09-11

## Outcome

Completed all 50 locked test fibers / 100 directions across five actual scrolls (six collections), with fresh inference and prefetched sources. Best5k regresses relative to best3k from the same run: mean 3.601407 → 3.218462 mm (10.63% shorter), restarts 2144 → 2411. Paris4 drives the regression (+378 restarts); 0139, 0332 and 1299 improve; 1451 worsens by one restart. Dense test/loss_total improved 0.43289152 → 0.40834693 without improving full tracing overall. One descriptive trial, not a significance claim.

## Global

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 1995 | 2095 | 3.857546 | 0.210496 | 0.011224 | 19.465051 | 22 |
| Previous run best17k | 2885 | 2985 | 2.707390 | 0.067985 | 0.009190 | 11.575651 | 26 |
| Same run best3k | 2144 | 2244 | 3.601407 | 0.148167 | 0.009164 | 18.480183 | 25 |
| New best5k | 2411 | 2511 | 3.218462 | 0.092753 | 0.009512 | 16.819130 | 16 |

## PHerc0139

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 564 | 584 | 2.384157 | 0.175258 | 0.007086 | 11.630505 | 8 |
| Previous run best17k | 523 | 543 | 2.564176 | 0.072169 | 0.004188 | 11.682571 | 11 |
| Same run best3k | 548 | 568 | 2.451316 | 0.077146 | 0.008295 | 14.483141 | 8 |
| New best5k | 476 | 496 | 2.807152 | 0.114909 | 0.006231 | 14.984857 | 4 |

## PHerc0332

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 27 | 33 | 14.279445 | 4.205966 | 0.007430 | 48.804954 | 1 |
| Previous run best17k | 34 | 40 | 11.780542 | 1.092106 | 0.008632 | 53.930331 | 1 |
| Same run best3k | 23 | 29 | 16.249024 | 5.661654 | 0.005702 | 55.702890 | 1 |
| New best5k | 20 | 26 | 18.123911 | 3.140176 | 0.002224 | 102.430566 | 0 |

## PHerc1299

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 159 | 169 | 3.548804 | 0.803244 | 0.031485 | 17.572927 | 1 |
| Previous run best17k | 359 | 369 | 1.625333 | 0.083672 | 0.009702 | 10.228828 | 1 |
| Same run best3k | 261 | 271 | 2.213092 | 0.211262 | 0.006759 | 12.067869 | 4 |
| New best5k | 224 | 234 | 2.563025 | 0.226469 | 0.013400 | 15.594383 | 1 |

## PHerc1451

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 19 | 23 | 4.105658 | 1.861616 | 0.070723 | 17.222278 | 0 |
| Previous run best17k | 61 | 65 | 1.452771 | 0.231442 | 0.013292 | 8.585655 | 0 |
| Same run best3k | 64 | 68 | 1.388679 | 0.092141 | 0.006701 | 9.409508 | 0 |
| New best5k | 65 | 69 | 1.368553 | 0.116398 | 0.007805 | 9.365389 | 0 |

## PHercParis4

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 1226 | 1286 | 4.295343 | 0.185032 | 0.012760 | 23.435173 | 12 |
| Previous run best17k | 1908 | 1968 | 2.806814 | 0.060791 | 0.009892 | 10.403026 | 13 |
| Same run best3k | 1248 | 1308 | 4.223097 | 0.165054 | 0.012118 | 25.485190 | 12 |
| New best5k | 1626 | 1686 | 3.276282 | 0.077321 | 0.011240 | 17.007384 | 11 |

## Metric and audit

Complete-reference-partition-v1: partition each directed reference at failure arcs from 0 to L; include terminal and zero intervals. Mean = total reference length/(restarts+directed cases); quantiles use the same segments with linear interpolation at (n−1)q. Total 8081.558059625 mm / 2511 segments. Reset advances belong to reference accounting, not successful physical tracing. Credited/path/failure-only diagnostics remain separate in JSON.

All 100 saved artifacts audited against failures/credited geometry. Source schema5/replay-set2. Snapshot hash rechecked in canonical cache; locked annotation and manifest hashes validated. Warm inference chunks=0; all requested work drained. Unrequested incidental candidate chunks need not drain. Forward/reverse directions are correlated. Asynchronous float16 accumulation can affect low-order values between fresh trials; settings unchanged.

## Runtime

| Metric | New best5k |
|---|---:|
| Coordinator seconds, before context teardown | 402.933 |
| Native seconds, including readiness waits | 354.757 |
| Inferred 256³ tiles | 34090 |
| Model calls | 4561 |
| Summed worker seconds | 2713.625 |
| Tiles in full8-tile batches (%) | 98.234 |
| Batch time mean / median / P95 seconds | 0.5950 / 0.5943 / 0.8621 |
| Peak allocated / reserved GiB | 36.677 / 46.648 |

Previous coordinator timings: classic516.699s, previous17k400.888s, same-run3k404.521s. Excludes admission and context teardown; not total command wall time.

## Snapshot and inputs

Original selection: `/ephemeral/hendrik/fiber3d_train2/runs/collection_20260909_5scroll_128_sd1_16x8_20260911_182306/snapshots/best.pt`.
Captured best step5000, 640000 samples, global batch128, test/loss_total=0.4083469252361626. Selected by lowest dense test loss, not tracing score or moving current.pt.
SHA256 `2823f5175be6d9ecc072bffd9079b6a2d984aa7bd014ab8c72d6b195992d101e`.
Staging: `/tmp/fiber3d_snapshot_20260911_2101/live_inference/_models/2823f5175be6d9ecc072bffd9079b6a2d984aa7bd014ab8c72d6b195992d101e/snapshot.pt`.
Canonical immutable copy: `/ephemeral/hendrik/vc3d_cache/live_inference/_models/2823f5175be6d9ecc072bffd9079b6a2d984aa7bd014ab8c72d6b195992d101e/snapshot.pt`.
UNetbase16/depth5, GroupNorm, pixelshuffle, native output_stride4. InputCTlevel1,totalfactor4→baselevel3; no double downsampling.

Locked split: `/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910`.
Sources: 0139(10 fibers),0332(3),PHerc1229-labelled actual1299(5),1451(2),Paris4 eval24+fibers6. Exact hashes/CT/normal mappings in `/ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best5k_full256_b8_20260911_v1/full_bindings.json`.

Only snapshot/hash/output/namespace differ from best3k: same 256tile/96overlap/32border/32outputchunk, bf16,cap8,128readers/256inputslots/128resultslots,float16chunk-major accumulator,beam8/lookahead2,step4atbasefactor2,normalweight0.1/tangent10,ellipse20normal/80tangentialbasevox,matchrefine1. No GT-heading/reset changes or training edits.

## Commands and evidence

From /home/hendrik/villa2, existing approved runner:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best5k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best5k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best5k_full.json
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py /ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best5k_full256_b8_20260911_v1/full
```

Reruns require a new output/namespace (intentionally single-use). Existing successful pilot reused as prerequisite; full set actually executed. No installs.

Code HEAD6517d82e688e91c51334a5500cd3d8cd871e6950 plus prior uncommitted metric correction. No further production changes. GCC13 RelWithDebInfo native binary /tmp/vc-live-native-cmake/bin/vc_fiber_trace_metric:
SHA256 7a08b63d9cac8f2afd057af16326decabfd09cd36e1802dca52702d06e89c3a2.
Native production diff SHA256 e70a2234ff159cf92a927a8b059e0f73bd044a64730fedab09387754444dd003.
Prior correction passed11native+16CLI+9Python tests. This configuration passed prepare/validate-full and complete execution/audit.

Manager task3153232e-55bf-4b8f-89d9-3d3d5676e953,exclusive224logicalCPU/8H100,priority100. Command exit0; terminal finished/completed,held0CPU/0GPU. Training task58fd84af-60cb-47b3-8fa4-c9acdb4bf251 restored to applied128CPU/8GPU. Final states recorded in accompanying JSON. Manager root-start to final contact496.747s (8m16.7s), including admission/setup/teardown; this is distinct from402.933s measured evaluation.
[Exact quality, performance and provenance](collection_182306_best5k_full_20260911.json).
