# Classic full-test trace-length analysis — 2026-09-11

Saved full cap8/heap run; no inference, tracing or benchmark rerun. Input:
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v3_heap/full`.
Summary SHA256: `7ded270d794461c558ee6ca2f4a7ea8091db7a382dec231c130dedc7285c9d3c`.
Classic best91_5k.pt, 50 locked test fibers / 100 directions; same settings as
[full batching report](full_dynamic_inference_batching_20260911.md).

## Results

| Scroll | Fibers / directions | Failures | Mean before restart mm | Median mm | Mean incl. terminal/zero attempts mm | Failed polyline mean mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 10 / 20 | 564 | 2.232873 | 0.145997 | 2.381535 | 2.314644 |
| PHerc0332 | 3 / 6 | 27 | 12.994266 | 0.309232 | 14.274223 | 13.059089 |
| PHerc1299 | 5 / 10 | 159 | 3.417562 | 0.790378 | 3.548305 | 3.462410 |
| PHerc1451 | 2 / 4 | 19 | 2.341021 | 0.761576 | 4.105658 | 2.390917 |
| PHercParis4 | 30 / 60 | 1226 | 3.785202 | 0.161655 | 4.293710 | 3.845984 |
| Global | 50 / 100 | 1995 | 3.427927 | 0.179913 | 3.855690 | 3.493324 |

Both Paris4 collections are pooled by failure/segment count, not an unweighted mean
of collection means. Collection-label PHerc1229 maps to actual PHerc1299 using
coordinate-space provenance. Paris4 eval_fibers separately: 4.751982 mm before
restart (783 failures / 48 directions); Paris4 fibers: 2.076424 mm (443 / 12).
These are small and unequal test samples, especially PHerc1451 and PHerc0332;
forward/reverse evaluations of one fiber are not independent biological samples.

## Definitions and audit

- **Mean before restart:** sum saved failure.traced_span_base_voxels converted to
  mm using each source's base voxel size, divided by number of failures.
  This is matched reference-fiber progress, not geometric traced-polyline length.
  Include all 65 zero-length failures; exclude successful terminal suffixes.
  Global numerator 6838.715007309 mm / 1995 failures = 3.427927322 mm.
- **All attempts:** include one terminal segment per directed case and zero-length
  failures: credited reference distance 8077.670214917 mm / 2095 segments =
  3.855689840 mm. All artifacts were checked to contain failures+1 segments; their
  reference-span sum matches credited length, with no overlap double counting.
- **Positive seeded spans:** existing mean_seeded_span_length_mm excludes
  zero-length spans; global 8077.670214917 / 2030 = 3.979147889 mm.
- **Historical whole-corpus reliability:** existing mean_segment_length_mm is
  intentionally full directed reference length/(failures+1), treating the corpus
  as one interval: 8081.558059625 / 1996 = 4.048876783 mm. This is not a bug and
  was not changed. Do not call it the empirical mean before restart.
- **Polyline cross-check:** sum Euclidean distances between consecutive saved
  trace_points_base_xyz within each segment only, never across reset jumps.
  Failure-ended polylines average 3.493324400 mm; all segments 3.917995731 mm.
  Main comparison uses reference progress so deviations/wiggles do not inflate it.

All 100 cases completed. All 1995 failure spans were independently reconstructed
from their segment's first search_begin_arc_base and failure reference_arc_base.
Artifact provenance/fiber/direction keys were matched uniquely to summary cases;
source means, global length, failure totals and reliability match saved summaries.
All failures have reason distance_above_threshold. Four directions were failure-free.
The 0.179913 mm global failure-span median shows strong skew toward many short
segments and a smaller number of long ones. It does not by itself establish why
short failures occur.

## Throughput interpretation

The completed profile supports inference as the main **sustained** bottleneck:
about 82 actual 256³ tiles/s across eight H100s, ready input queues full,
98.1% of tiles in full batches of eight, maximum publication backlog 11.
Main coordinator thread used 10.19 CPU seconds in a 60-second observation.
Batch1 to cap8 on the same heap scheduler reduced coordinator wall
860.380 → 516.699 seconds (39.95%). Sparse final tails may be input-fetch- or
small-batch-limited. This is not a claim of 100% GPU utilization throughout;
summed worker timings include transfer/normalization/output/synchronization,
not only GPU kernels. See the linked performance report for exact methodology.

## Reproduce the analysis only

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py
```

This standard-library-only script reads saved JSON and prints the derived report;
it does not start models, tracing, a manager job, or write evaluation artifacts.
The exact executed calculation is retained alongside this report; machine-readable
results are in [trace_restart_analysis_20260911.json](trace_restart_analysis_20260911.json).
Independent plan/metric review independently reproduced all per-scroll means.
No runtime metric/threshold changes or deferred implementation work.

## Committed implementation state

Before this analysis: native cache/tracing 75013f6cc, shared inference a238c4598,
integration/configs/results 175e3645b. Prior validation: 106 relevant inference/
runner/profile tests plus 64 preprocess tests and six subtests passed; completed
full cap8 run covered all 100 directions. This analysis only adds documentation
and the read-only reproducibility calculation, with arithmetic/artifact assertions.
