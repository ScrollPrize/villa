# Paired versus single-patch embeddings: current evaluation

Old raw: update5400 /691200 patches, SHA
`7ea0f9c89e4bd4ce8c48dbae920b7ed24c810c178ab34576767e30b930057b35`.
Paired raw: update3700 /473600 patches, SHA
`c412ccf6dd9a95562af370b99a92b7595aadd343f689d90bea088cee8415f505`.
This is **not matched training age**: user requested immediate evaluation.

Same held-out PHerc1299 (dataset label PHerc1229) fiber
`lt_20260825T075316065_000069.json`, both directions. Identical GT reference
coordinates verified directly; same lag sampling and complete counts in every
row. Paired cache:4057/4057 valid native queries (includes failure-route queries).
Input level1, BF16256-cube tiles, border32/overlap96, signed FP32 embeddings,
same native interpolation and normalization. No training stop or model changes.

## GT cosine versus arclength separation

Distances are base voxels (2.399um each). At4096 this is9.827mm along GT.

| Separation | Pairs | Old mean | Paired mean | Old median | Paired median |
|---:|---:|---:|---:|---:|---:|
|8|466|0.944|0.847|0.981|0.936|
|16|465|0.948|0.861|0.979|0.960|
|32|465|0.926|0.874|0.967|0.948|
|64|464|0.878|0.870|0.943|0.944|
|128|462|0.795|0.870|0.882|0.936|
|256|458|0.691|0.873|0.801|0.933|
|512|450|0.651|0.868|0.788|0.935|
|1024|434|0.585|0.874|0.674|0.934|
|2048|402|0.512|0.870|0.587|0.937|
|4096|338|0.338|0.892|0.325|0.943|

Distance-dependent decay is largely absent on this fiber in the paired model,
but local similarity is lower. Higher GT similarity alone is not evidence of
better identity discrimination: wrong-route embeddings can also be similar.

## Tracing and false-route ambiguity

Same default history: spacing32, window256, count8, weight0.25.

| Model | Failures without embedding | With embedding |
|---|---:|---:|
|Old5400|96|109|
|Paired3700|140|133|

At paired baseline failure locations, GT/taken cosine mean0.812, median0.883;
47.1% exceed0.9. For44 both-high-presence cases (both>=0.5),56.8% exceed0.9.
Mean actual-route-history GT-minus-taken gap is only0.0186. Long-distance
oracle GT-reference gaps remain approximately zero or negative. Thus global
continuity improved descriptively, while disambiguation remains weak and tracing
overall is still worse than the older, more-trained model on this one fiber.
Old and new failure sets differ; those are not matched-event comparisons.

## Reproduction and outputs

Approved runner subcommands, in order: `embedding-freeze-current`,
`embedding-trace-pilot`, `embedding-route-analysis`. Settings in
`vesuvius/scripts/fiber3d_experiment.json`. GPU work manager-nonexclusive,
oneGPU20GiB/16CPU; training retained. Off/on wall87.0s/130.7s includes inference,
tracing and concurrency effects, not a standalone model-speed benchmark.
Cached native query/analysis0.598s, CPU only. Both commands exited0.

Under `/ephemeral/hendrik/fiber3d_train2/evaluations/`:
- `embedding_trace_paired_current_20260913_v2/report.json`: frozen identity,
  full trace results and parameters.
- `embedding_route_paired_current_20260913_v2/report.json`: all lag distributions,
  failure metrics and coverage; points.json/queries.json preserve queries/vectors.
- `embedding_route_analysis_20260913_v3/report.json`: old range reference.

The previous5400 watcher exited on a checkpoint-read error; no watcher remains.
This immediate evaluation used a validated frozen snapshot, not the live file.
