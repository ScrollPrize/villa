# Refinement EMA999 evaluation

1. Freeze current_ema_0p999.pt via existing approved runner; record hash and update count.
2. Reuse latest no-embedding workload: all 50 original test fibers, eight GPUs under exclusive high-priority manager lease, no embedding output/weight or rendering. Preserve ellipse40/80, grace160, restart filtering80 and eligible spans. Do not use refined crop archives as GT.
3. Independent plan review, prepare/validate provenance, run full evaluation through approved runner and monitor until completion or genuine blocker.
4. Report combined and per-scroll restarts, mean, median/P05/P95 versus matched baselines; persist provenance/results.

## Spec update

Visualization follow-up: clone the completed workload into a separate output directory, changing only render=true and manager label. Keep frozen checkpoint and cache namespace identical. Prepare/validate and run through approved runner; allow on-demand inference for any additional visualization coverage. Verify renders and per-fiber index/metrics, compare rerun metrics with completed111-restart baseline.

None; locked evaluation protocol unchanged.

## Docs updates

Record commands, snapshot and results in task_log and benchmark docs; changelog result. Previous merge task preserved in commit0577d1d8f.
