# Corrected complete-segment tracing comparison — 2026-09-11

This is the current comparison, superseding the failure-only and whole-corpus F+1 figures in the earlier reports. **No inference or tracing was rerun.** Raw trace paths and failure events are unchanged.

## Definition

For each directed fiber, partition `[0, full reference length]` at its recorded directional restart arcs. Keep every interval, including the successful terminal interval and any zero intervals. Thus one10mm trace with one restart has mean5mm, regardless of where the restart occurred. Across the benchmark, mean=`total directed reference mm / (restarts + directed cases)`. Quantiles use these same complete partitions, with linear interpolation at `(n−1)q`. Do not average per-fiber means.

Forced reset advances are included in this full-reference accounting; they are **not** asserted to be successfully traced distance. Credited traced progress and physical polyline length remain separate legacy diagnostics in JSON. Repeated/endpoint failure positions can still yield zero intervals. A final failed reset reaching the endpoint still contributes its final partition interval, even if no further physical trace segment was emitted.

All models:50locked test fibers,100directed cases,8081.558059625mm total reference length. Six collections pooled into five actual scrolls; PHerc1229-labelled annotations refer to PHerc1299, and both Paris4 collections are pooled. Forward/reverse cases are correlated; descriptive statistics, not independent-sample significance estimates.

## Global

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 1995 | 2095 | 3.857546 | 0.210496 | 0.011224 | 19.465051 | 22 |
| Collection 09-10 best17k | 2885 | 2985 | 2.707390 | 0.067985 | 0.009190 | 11.575651 | 26 |
| Collection 09-11 best3k | 2144 | 2244 | 3.601407 | 0.148167 | 0.009164 | 18.480183 | 25 |

## PHerc0139

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 564 | 584 | 2.384157 | 0.175258 | 0.007086 | 11.630505 | 8 |
| Collection 09-10 best17k | 523 | 543 | 2.564176 | 0.072169 | 0.004188 | 11.682571 | 11 |
| Collection 09-11 best3k | 548 | 568 | 2.451316 | 0.077146 | 0.008295 | 14.483141 | 8 |

## PHerc0332

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 27 | 33 | 14.279445 | 4.205966 | 0.007430 | 48.804954 | 1 |
| Collection 09-10 best17k | 34 | 40 | 11.780542 | 1.092106 | 0.008632 | 53.930331 | 1 |
| Collection 09-11 best3k | 23 | 29 | 16.249024 | 5.661654 | 0.005702 | 55.702890 | 1 |

## PHerc1299

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 159 | 169 | 3.548804 | 0.803244 | 0.031485 | 17.572927 | 1 |
| Collection 09-10 best17k | 359 | 369 | 1.625333 | 0.083672 | 0.009702 | 10.228828 | 1 |
| Collection 09-11 best3k | 261 | 271 | 2.213092 | 0.211262 | 0.006759 | 12.067869 | 4 |

## PHerc1451

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 19 | 23 | 4.105658 | 1.861616 | 0.070723 | 17.222278 | 0 |
| Collection 09-10 best17k | 61 | 65 | 1.452771 | 0.231442 | 0.013292 | 8.585655 | 0 |
| Collection 09-11 best3k | 64 | 68 | 1.388679 | 0.092141 | 0.006701 | 9.409508 | 0 |

## PHercParis4

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 1226 | 1286 | 4.295343 | 0.185032 | 0.012760 | 23.435173 | 12 |
| Collection 09-10 best17k | 1908 | 1968 | 2.806814 | 0.060791 | 0.009892 | 10.403026 | 13 |
| Collection 09-11 best3k | 1248 | 1308 | 4.223097 | 0.165054 | 0.012118 | 25.485190 | 12 |

## Reproduction and historical correction

From `/home/hendrik/villa2`:

```bash
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python vesuvius/scripts/correct_saved_replay_metrics.py /ephemeral/hendrik/fiber3d_train2/evaluations
# Add --apply to migrate; the default is read-only. Already-correct files are unchanged.
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py /ephemeral/hendrik/fiber3d_train2/evaluations/collection_182306_best3k_full256_b8_20260911_v1/full
```

Migration corrected25completed replay-set summaries (including older timing/pilot runs); validated counts, ordered arcs, completion and reference length before writing. Immutable `summary.before-complete-segments.json` files retain exact original bytes. Only summary metadata was replaced atomically; no case artifacts/cache/checkpoints modified. Immediate repeated migration was a no-op. Incomplete runs with no final summary were not assigned results. Older separately scoped crop/replay progress plots retain their legacy convention and are not part of this five-scroll comparison.

Each of the three main reports was separately audited against all100raw `cases/*/trace.json` artifacts. Current source schema5/replay-set schema2 identify `complete-reference-partition-v1`. All old failure-only figures in rewritten JSON live under `legacy_diagnostics`; headline global/scroll/collection fields now use complete segments. Existing snapshot/performance metadata is retained.

[Migration hashes and records](complete_segments_migration_20260911.json). [Classic provenance](classic_full_5scroll_20260911.md), [previous collection provenance](collection_best17k_full_20260911.md), [latest collection provenance](collection_182306_best3k_full_20260911.md).

## Validation

- Native GCC13 RelWithDebInfo: rebuilt `test_fiber_reference_replay_benchmark` and `vc_fiber_trace_metric` in `/tmp/vc-live-native-cmake`;11native tests passed.
- Python metric/migration regression suite:9tests passed, covering10mm/one restart, no failures, mixed scales/cases, reverse/repeated/endpoint failures, invalid/incomplete input and idempotent backups.
- Native CLI end-to-end synthetic suite:16tests passed (including corrected two-direction zero-restart mean), outside sandbox after a sandbox Zarr async-thread setup stall. No training/evaluation dataset rerun.
