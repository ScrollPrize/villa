# Plan: full collection snapshot comparison

1. Preserve current classic baseline: add p05/p95 to reproducible saved analysis, retain means/counts and metric definitions. Link detailed docs/benchmarks record from existing docs/native_trace2cp_benchmarks.md (older paul4 tests differ in cohort/method). Record current committed implementation revisions, actual full command/config, locked split, source mappings, checkpoint/hash and output path. Log GT tangent currently selects/alines model direction, not enforced seed heading; reset advances separately. No tracer behavior changes; investigation deferred explicitly.
2. Archive previous active log and update workflow. Independent plan review against task/specs/plan. Commit baseline docs before evaluation.
3. Inspect requested snapshots; use best.pt selected by saved test/loss_total, step17000/samples2176000, SHA2563fecbcd6806242baa7bfc7ffcd7ec27988b28ea43e87c589e6eacc7c280a6a7c. Embedded config authoritative, output_stride4. Verify shared adapter/worker native shape, blend weights, remaining_scaledown=total/native. Add a focused stride4 batched-worker regression if missing. Reuse shared code; no duplicate inference or unnecessary production edit. No installs/training.
4. New full-set workload cloned from measured cap8 heap config; identical6collections/5scrolls50fibers100directions, input L1, total output factor4 relative input,256tiles,96overlap32border,32output chunks,bf16,cap8,exclusive8GPUs/managerpriority100, existing trace/error settings. Only snapshot and fresh inference namespace/output differ. Reuse fetched inputs but never prior inferred chunks. Preserve 3600srequest timeout. Prepare/validate via approved runner, run full; monitor progress and memory, report failure rather than silent cap/scale changes.
5. Run focused CPU contract tests and real managed full evaluation. Inspect all completion/queue/artifact counts and zero-warm inference provenance. Compare same metrics/counts per scroll with classic; report time plus native stride/batch arithmetic differences. Record actual command/checkpoint/config hashes/results/failures/limitations. Do not stop at launch; continue through completion or genuine input blocker.

## Spec update

Existing native output stride and whole-test evaluation specs already apply. No runtime change planned; add only missing regression coverage. GT initialization change explicitly later.

## Docs updates

Add benchmark entry and detailed record under docs/benchmarks; link existing native_trace2cp_benchmarks.md without mixing older methodology. Persist baseline quantiles and follow-up result, task/status/log/changelog. Lasagna scope: existing Python/PyTorch worker, standard pytest and tiny CPU inference-only fixtures; production validation CUDA managed exclusively. No installation or unrelated subproject work.
