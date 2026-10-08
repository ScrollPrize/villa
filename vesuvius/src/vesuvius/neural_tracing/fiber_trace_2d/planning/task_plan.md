# Plan

1. Prior work committed in 956a81c1e and 510339f25. Independent plan review.
2. Extract native reverse-path monotone ellipse verification into a shared observer used by both existing beam tracing and callback tracing. Native replay retains interval/rollback/restart bookkeeping.
3. Extend callback binding with optional reverse runner, full-prefix checks every512 base voxels and all terminal endpoints, ellipse20/40, no history. Reverse follows actual traced prefix, not GT. Neural reverse uses a distinct tracer sharing the forward batch queue and loading pool, same checkpoint/policy; confidence exhaustion before reaching prefix start fails verification.
4. Neural models have no dense presence field for aligned-local-maximum reseeding: explicitly retain their existing GT-arc seed progression after a filtered stop. Document this difference from regular tracer; do not invent presence values.
5. Regression tests: disabled path, successful/failed/short reverse, terminal check, expired callbacks. Approved runner builds/tests, pilot then full182 directions for both models. Preserve batching/cache and run namespace identity. Report min0/min512 with saved neural and tuned regular controls.

6. Follow-up: add memory/Flowv4 using saved configurations (batch8,16case workers/GPU, original confidence/commit/gate and scales). Inspect checkpoints with approved runner to select compatible legacy loader. No changes to shared filter or scheduling. Independently review extension.
7. Because historical upstream9200d51a9 differs from current853dcf536, run matched current-code unfiltered controls as well as FW/BW full182 cases for both old models, with pilots first. Audit checkpoint, heading, policy and references; retain historical rows distinctly. Always include tuned regular EMA999u12100 without FW/BW.
8. Extend reusable report and tests to cover old configuration preservation and control selection. Update same benchmark docs/changelog/log, separate confidence/FW/error counts, min0/min512 and per-scroll lengths/coverage. No default promotion inferred.

Compatibility finding: current loader rejects memory/identity features. Use the frozen ba86eea12 model/rollout runtime for both old models, verified against181 saved source hashes, while sharing current replay/queue/native reader. This is an explicit versioned runtime artifact, not a copied/reimplemented tracer. Preserve original forward cap3×remaining reference distance and old abort-hook API. Matched controls also isolate dependencies outside that frozen package.

## Spec update
Optional neural consistency uses shared native verification and replay. Separate confidence/consistency/error counts; explicit neural reseed difference.

## Docs updates
Benchmark MD/JSON, exact policy/checkpoints/commands, per-scroll results, task log/status/changelog.
