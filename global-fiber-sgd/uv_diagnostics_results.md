# Diagnostic acceleration

## Exactness and CPU timings

Same Paris4 initial/final states, 489,197 original fibers, 151 hypotheses.
Three repetitions per backend, full edge audit. Reference timing also receives
the new static grouping cache, so the isolated speedup is conservative.

| State | Reference mean / p50 / p95 (s) | Optimized mean / p50 / p95 (s) |
|---|---:|---:|
| Initial | 11.516 / 11.514 / 11.548 | 4.822 / 5.018 / 5.114 |
| Final | 11.404 / 11.263 / 11.710 | 5.118 / 5.102 / 5.223 |

Every per-source median matches exactly, including NaNs. Complete audit and
reference-metric dictionaries match exactly. Constant-group shortcut avoids
sorting without changing weights or tie conventions; only 10,775 initial groups
(2.20%) and 7,030 final groups (1.44%) use the canonical median fallback.
Shared geometry avoids repeated softmax/coordinate reconstruction. Static source
grouping and arc offsets are cached only for immutable input arrays.

## Six-GPU end-to-end comparison

336 updates / 66,060,288 samples (~10 pass-equivalents), global batch196608,
32768/GPU, same initialization and LR, all variables trained. Twelve diagnostic
checkpoints, same sampled intermediate/full final populations, no cProfile.

| Diagnostics | Solve wall seconds |
|---|---:|
| Original synchronous | 152.743 |
| Exact optimized + two asynchronous CPU workers | 34.847 |

**4.38× faster with reporting retained.** All12 checkpoint metric/loss dictionaries
and final solution arrays were exactly equal. Only timestamps/scope labels differ.
Measured from inside solver after input/model setup through final solution save;
not manager wait, process startup or initial data preparation.

Async diagnostic CPU work totals54.108s (including baseline), overlapping the
optimizer and other diagnostic worker. Queue backpressure plus final draining
took8.082s. Do not subtract summed concurrent CPU work from wall time. Each worker
handles one snapshot; this is not per-fiber parallelism. Two is a starting point,
not proven optimal. Snapshot queue counts running+queued work and is bounded to2;
each full state is~0.832GiB, plus temporary arrays. Moreworkerscanbe measured but
128 concurrent snapshots would not make one snapshot128timesfaster.

Main thread copies immutable CPU snapshots and loss scalars, workers never touch
CUDA/model state. Results are published in optimizer-step order; no snapshots
dropped, final queue drained before save. Errors propagate and workers shut down.
Async mode rejects active sampling/early-stop protocols needing current metrics.
`seconds` now identifies snapshot capture; `solve_wall_seconds` includes completion
of all diagnostics and save. `seconds_scope` makes this explicit in history.

## Commands and tests

All use the preapproved runner:
```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_diagnostic_benchmark.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_distributed_32k_6gpu_fastdiag.json
```
97 targeted tests passed,2 separate CUDA tests deselected; real6GPU run executes
CUDA coordinate and NCCL gradient preflights. Tests cover empty/invalid groups,
ties,multiplepieces,orderedboundedpublication and workerfailure propagation.

GPU coordinate backend remains opt-in. CPU diagnostic backend defaults optimized;
`diagnostic_backend=reference` retains the canonical median loop for validation.
`diagnostic_workers=0` remains the synchronous default; these experiments opt into2.
