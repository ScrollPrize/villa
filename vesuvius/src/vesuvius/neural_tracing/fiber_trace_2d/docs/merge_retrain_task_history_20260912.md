# retrain task history before merge (2026-09-12)

## task.md

# Evaluate current best snapshot

Run all50locked test fibers across five actual scrolls in both directions against new best.pt from collection_20260909_5scroll_128_sd1_16x8_20260911_182306. Freeze selection; fresh inference with source caches reused and exclusive8GPU manager. Compare complete-segment mean/median/P05/P95 and restart counts against saved baselines. Continue to completion and persist results.

## task_plan.md

# Plan: current best5k full evaluation

1. Read workflow/specs/plan and independently review. Archive preceding correction log; preserve its uncommitted code/docs/tests. Corrected native binary already rebuilt/tested (11native+16CLI); no implementation changes expected.
2. Freeze current best using existing live_evaluate.freeze_checkpoint into /tmp staging, then canonical model cache in runner. Captured best5k,640000samples,test/loss_total0.4083469252361626,SHA2823f5175be6d9ecc072bffd9079b6a2d984aa7bd014ab8c72d6b195992d101e. Keep original path. best means minimum dense test/loss_total, not latest current.pt or tracing-selected.
3. Clone best3k workload changing only frozen snapshot/hash and distinct output/inference namespace. Same50fibers100directions6collections5scrolls,source mappings,native stride handling,256tile96overlap32border32outputchunk,bf16,cap8,128readers256inputslots128resultslots,canonical tracer parameters. No GT-heading/reset changes. Use approved run_classic_fiber_evaluation.py prepare,validate-full,full; manager exclusive8GPUs priority100/224logicalCPUs; no unmanaged bypass.
4. Monitor to100/100 and successful cleanup. Reuse source fetches, never inference. Existing completed pilot satisfies prerequisite; no new subset trial. Audit snapshot hash,warmchunks0,membership,F+C mean. Run existing saved-artifact audit on100cases; compare global/per-scroll full metrics with classic,previous17k,earlier3k. Record wall/tiles/batches and timing caveats.
5. Persist config and benchmark JSON/MD with command/data/checkpoint/build/revision provenance; link index, update log/status/changelog. No training changes/installs. Record uncommitted corrected-metric patch provenance; do not claim clean historical code revision.

## Spec update

None. Apply complete-reference-partition-v1 (L/(F+C), terminals included in quantiles), authoritative checkpoint/native stride and exclusive fresh-inference specs.

## Docs updates

New best5k comparison and metadata; archive prior task log, update progress/log/changelog. Preserve historical benchmarks.

## status.md

# Status

- [x] Read workflow/specs and freeze current best5k.
- [x] Write plan and archive previous log.
- [x] Independent review approved; only snapshot/hash/output/namespace differ.
- [x] Prepare/validate full workload:50fibers100directions and all manifests verified.
- [x] Exclusive fresh-inference100-direction run:completed402.933s,exit0.
- [x] Audit all100artifacts,compare full metrics; manager released,training resumed128CPU/8GPU.
- [x] Persisted best5k config,benchmark JSON/MD,provenance,index,changelog and log.

## task_log.md

# Current best5k full evaluation

- Full50fiber/5scroll/100direction scope, no repeated subset questions.
- Frozen with existing live_evaluate.freeze_checkpoint: step5000,640000samples,test/loss_total0.4083469252361626,SHA2823f5175be6d9ecc072bffd9079b6a2d984aa7bd014ab8c72d6b195992d101e. Runcollection_20260909_5scroll_128_sd1_16x8_20260911_182306; staging/tmp/fiber3d_snapshot_20260911_2101.
- Prior correction changes remain uncommitted/preserved; native binary rebuilt/tested. Old log archived in docs/complete_segment_correction_task_log_20260911.md. No tracer/inference changes this task.
- Independent reviewer current_best5k_plan_review approved plan and verified config delta restricted to snapshot/hash/output/namespace. Prepare and validate-full passed50fibers100directions.
- Code HEAD6517d82e688e91c51334a5500cd3d8cd871e6950 plus previous correction patch. Native binary SHA2567a08b63d9cac8f2afd057af16326decabfd09cd36e1802dca52702d06e89c3a2; native production diff SHA256e70a2234ff159cf92a927a8b059e0f73bd044a64730fedab09387754444dd003. GCC13RelWithDebInfo.
- Launched full via approved runner and eval_collection_182306_best5k_full.json. Manager inventory224logicalCPU/8GPU; prior training task58fd84af-60cb-47b3-8fa4-c9acdb4bf251 active/pausable. Manager handles yielding/restoration; no manual training mutation.
- Evaluation task3153232e-55bf-4b8f-89d9-3d3d5676e953 acquired exclusive224CPU/8GPU; training held0CPU/0GPU while waiting. Snapshot native output_stride4,base16/depth5,GroupNorm,pixelshuffle confirmed; total inference factor4 therefore no extra spatial decimation.
- At167.9s:11229tiles,3289published chunks, roughly100tiles/s recent throughput. All100directed traces active; fresh source-keyed outputs created for all5scans, no error. GPU snapshot7/8at100%,one60%; utilization instantaneous, not an averaged bottleneck profile.

## Completed result and validation

- Full run exit0,all100directions complete; no incomplete markers. Coordinator402.932717s,native354.757123s,34090tiles/4561model calls,2713.625487summed worker seconds,98.2341%tiles in8tile batches. Peak allocated36.676594GiB/reserved46.648438GiB. Warmchunks0; requested queues drained. Same inference/tracer settings, no re-runs or numerical changes.
- Corrected global mean3.218461991mm,median0.092752871mm,P05/P95 0.009512263/16.819130439mm;2411restarts,2511segments,16zero complete-reference intervals,8081.558059625mm. Earlier same-run best3k mean3.601407335→3.218461991 (10.63%shorter),2144→2411restarts. Paris4+378restarts dominates regression;0139,0332,1299 improve and1451 worsens by1.
- Per-scroll mean/restarts:0139 2.807152212mm/476;0332 18.123911129mm/20;1299 2.563025142mm/224;1451 1.368552756mm/65;Paris4 3.276281643mm/1626. Full paired mean/median/P05/P95/counts vsclassic,previous17k,and same-run3k in docs/benchmarks/collection_182306_best5k_full_20260911.md/.json.
- Existing trace_restart_analysis_20260911.py audited all100artifacts. Summary SHA256f99eea64f0dc8582b612084161660dfce108e8c6cfcc2d39dd6ae9797db361c2; preparedconfig03651c7a2a2b2b35e260e36ed1e90fac14a1344211f6e86b33c6aecaf6d4f196; bindingsfc15edf01804747ac7097595a97f5ee18e2dfbeed272a2ebfa9825ebbae29357. Frozen canonical snapshot hash rechecked unchanged.
- Manager task terminal finished/completed,held0CPU/0GPU; training restored applied128CPU/8GPU. Rootstart-to-lastcontact496.747s includes admission/setup/teardown, distinct from measured402.933s evaluation. No force termination; cleanup completed normally.

## Exact commands

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best5k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best5k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_182306_best5k_full.json
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py /ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best5k_full256_b8_20260911_v1/full
```

## Limitations/deviations

- No scope omissions. One full trial; asynchronous accumulation can affect low-order values, and forward/reverse observations are correlated. No statistical significance claim or causal explanation for regression. best.pt uses dense test loss, not this tracing metric.
- Previous correction work remains uncommitted and preserved; benchmark records exact native binary/production diff hashes. No installations or training/model changes. Config-only task verified by prepare/validate/full/artifact audit; prior metric regression suite already passed36tests. No additional test suite rerun needed.
- GPU-utilization read inside sandbox could not reach driver; repeated outside sandbox successfully. No driver/system changes. Final process check found evaluated PIDs already exited; cleanup was not modified.
