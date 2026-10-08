# Plan

1. Freeze best.pt and record metadata/hash; verify raw weights, not EMA. Best refers to saved validation selection, not tracing selection.
2. Independent review; prepare/validate full50originalGT/noembedding/norender40/80ellipse160grace80restart protocol, all8GPUs exclusive manager. Reuse approved runner, no source modifications.
3. Complete full evaluation, compare combined/per-scroll all metrics to Classic and priorEMA checkpoints, especially bestu16500. Persist timing/provenance/results.

## Spec update

None.

## Docs updates

Archive previous active task under refinement_u24200; update benchmark docs, task_log,changelog.
