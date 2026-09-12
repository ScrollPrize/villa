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
