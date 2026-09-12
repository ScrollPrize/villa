# Classic full five-scroll baseline — 2026-09-11

## Current corrected metrics

The complete-segment comparison below supersedes all failure-only/corpus-F+1 quality numbers in the historical record that follows. Traces, checkpoints and timings are unchanged. See [full corrected per-scroll comparison](complete_segments_20260911.md).

| Model | Restarts | Segments | Mean mm | Median mm | P05 mm | P95 mm | Zero segments |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic best91.5k | 1995 | 2095 | 3.857546 | 0.210496 | 0.011224 | 19.465051 | 22 |
| Collection 09-10 best17k | 2885 | 2985 | 2.707390 | 0.067985 | 0.009190 | 11.575651 | 26 |
| Collection 09-11 best3k | 2144 | 2244 | 3.601407 | 0.148167 | 0.009164 | 18.480183 | 25 |

## Historical record (superseded quality metrics; preserved run provenance)


## Result and committed implementation

This is the completed current-code baseline before evaluating the collection
snapshot. Implementation committed as native `75013f6cc`, shared GPU inference
`a238c4598`, integrated runner/configs `175e3645b`; saved-artifact audit
`faa197d20`. These commits capture the measured working-tree implementation;
the benchmark itself ran before that code was committed. No GT-heading fix is
included. GCC13 RelWithDebInfo native binary:
`/tmp/vc-live-native-cmake/bin/vc_fiber_trace_metric`.

| Scroll | Fibers | Failures | P05 mm | Median mm | P95 mm | Mean mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 10 | 564 | 0.00014 | 0.146 | 11.232 | 2.233 |
| PHerc0332 | 3 | 27 | 0.00000 | 0.309 | 48.898 | 12.994 |
| PHerc1299 | 5 | 159 | 0.02598 | 0.790 | 17.738 | 3.418 |
| PHerc1451 | 2 | 19 | 0.06766 | 0.762 | 10.069 | 2.341 |
| PHercParis4 | 30 | 1226 | 0.00171 | 0.162 | 20.964 | 3.785 |
| Global | 50 | 1995 | 0.00117 | 0.180 | 17.670 | 3.428 |

Distances are matched reference progress during failure-ended attempts, including
zero spans and excluding successful terminal segments. P05/P95 use inclusive
linear interpolation (Python statistics.quantiles method='inclusive'). Pool spans,
not scroll means. Both directions are evaluated but are not independent samples.
PHerc1229 is a collection label for actual PHerc1299; Paris4 collections pooled.

Global: 1995 failures,65 zero spans,100 complete directions. Failed-span mean
3.427927322mm; all2095attempts including terminal/zero spans3.855689840mm.
Historical whole-corpus reliability remains4.048876783mm, defined as
8081.558059625 directed reference mm/(1995+1), a different metric.
[Detailed definitions and artifact audit](../trace_restart_analysis_20260911.md).
[Exact metrics](../trace_restart_analysis_20260911.json) and
[read-only reproduction script](../trace_restart_analysis_20260911.py).

## Checkpoint and data

Snapshot:
`/home/hendrik/runs/s1a_128_1_single_8x8_20260801_084232/snapshots/best91_5k.pt`.
SHA256 `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17`.

Locked test split:
`/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910`.
Original annotations under its sibling `all/`; test symlinks and hashes checked.
Exact selected fibers, coordinates, CT URLs and normal-manifest hashes are in
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v3_heap/full_bindings.json`.
Six source collections, five actual scans:
0139/20260102150214 (10 fibers),0332/20251211183505 (3),
1299/20260309130042 (5),1451/20260319101107 (2),
Paris4/20260411134726 (24 eval_fibers +6 fibers).
CT selected level1; sources are vesuvius-challenge-open-data as listed in the
committed workload. Same approved sheet-normal bindings as the full comparison.

## Exact command and settings

From `/home/hendrik/villa2`, using the existing approved runner:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_full_dynamic256_b8_heap.json
```

Preparation used the same command with `prepare` instead of `full`.
The config is pinned by implementation commit175e3645b. Exclusive8H100s via
GPU manager priority100; model batch cap8;256³input tiles, overlap96,border32;
total inference downsample4 relative to input L1 (stored base scale8),32³output
chunks; bf16 model, float16 chunk-major accumulation. Native trace step4 at
trace-to-base2, beam8/lookahead2, smoothness normal0.1/tangent10, match refine1,
ellipsoid normal radius20basevox/tangential radius80basevox. Both directions.
Full config and effective tracer options also persist in saved summary.json.

Raw results:
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v3_heap/full`.
Summary SHA256 `7ded270d794461c558ee6ca2f4a7ea8091db7a382dec231c130dedc7285c9d3c`.
No rerun for quantiles: all100saved trace artifacts were audited.
Coordinator516.699s/native469.562s,34441tiles/4760modelcalls.
[Full timing methodology and batch1 comparison](../full_dynamic_inference_batching_20260911.md).
Older entries in native_trace2cp_benchmarks.md use different cohorts/protocols and
must not be interpreted as directly comparable scores.

## Deferred diagnostic: GT heading and zero reference progress

The wrapper supplies the local GT tangent on every initial start/restart, but
traceOneWayCore selects a sign-aligned MODEL prediction and uses it for
previousStepDirection,currentSampleDirection,historyDirection. No minimum
alignment threshold; GT direction is not enforced as the seed heading. Candidate
cone is centered on that model prediction. This may contribute to short/zero
matched spans; causal contribution not yet measured.

Forward matching can stay at the start despite physical trace movement.
Saved case4,segment29 had9matched steps and zero reference progress;
start/failure arc32644.555431143188, next start32652.555431143188basevox.
Reset=max(failure arc,old start+nominal step), clamped to end, so it advanced8basevox
(~0.0192mm) separately. No endless retry at the same start.
Future work: audit start GT/model/candidate angles and compare explicit GT-heading
initialization under the same benchmark; inspect local matching on failures.
Do not implement this while changing checkpoints, so the comparison stays matched.
