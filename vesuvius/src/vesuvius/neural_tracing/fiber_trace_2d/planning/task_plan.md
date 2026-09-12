# Plan: latest28k then best11k

1. Read workflow, relevant specs and overarching plan; independently review. Archive best5k task log. Starting clean at b32f31a55; native corrected-metric binary SHA7a08b63d9cac8f2afd057af16326decabfd09cd36e1802dca52702d06e89c3a2. Config-only task, no production edits planned.
2. Freeze both original files with existing live_evaluate.freeze_checkpoint before any GPU work. Captured current28k/3584000samples/dense loss0.38799891914012/SHA114a13003b9a33f78b07cc61a83abd710a2f4bf8408ea16909c249857018817b and best11k/1408000samples/loss0.3603664571877386/SHAca5349cd785b447109bc8be46f211501f1c18d1575e5e7e5c3998f33e0924b34. Both native stride4/base16/depth5/GroupNorm/pixelshuffle. Original and frozen/canonical paths retained; training may advance but snapshots stay fixed.
3. Clone prior best5k config twice; only checkpoint/hash/output/namespace change. Identical50fibers100directions6collections5scrolls and source/normal mappings; inputL1,totalfactor4,256tile96overlap32border32outchunks,bf16,cap8,float16chunk-major accumulator,128readers256inputslots128resultslots. Same beam8/lookahead2/step4/ellipse thresholds/GT initialization. Prepare and validate-full both with approved runner. Existing successful pilot remains prerequisite; no subset rerun.
4. Run current28k first via existing approved full command under exclusive manager priority100,224logicalCPU/8GPU, never bypass. Follow to100/100, artifact audit, drained requested queues, immutable hash, warmchunks0, and manager release. Report first iteration vs classic/best5k while preparing second; then run best11k sequentially with distinct inference namespace and same checks. No prediction cache reuse, shared source fetches permitted. Continue until both complete and training resources restored.
5. Reuse trace_restart_analysis_20260911.py for terminal-inclusive L/(F+C), pooled median/P05/P95, per-scroll/per-collection counts and legacy diagnostics. Record coordinator/native time, tiles,batches,batch mean/median/P95 and peaks; distinguish pre-teardown time from manager lifecycle. Compare all metrics side by side, highlight max distances/min restarts; preserve baseline snapshots/results (no baseline rerun requested).
6. Persist each config and combined benchmark JSON/MD with commands,source/snapshot/config/binary hashes,code revision,manager evidence; update index,changelog,status/task log. No new training,installs or tracing changes. Note single-trial/correlated directions/asynchronous accumulation caveats.

## Spec update

None. Current complete-reference-partition-v1, snapshot-authoritative native-stride handling and exclusive fresh-inference specs apply unchanged.

## Docs updates

New20260912latest28k/best11k comparison vsclassic/lastbest5k, reproducible commands and metadata; archive prior log, update benchmark index/changelog and current progress log. Existing baselines remain intact.
