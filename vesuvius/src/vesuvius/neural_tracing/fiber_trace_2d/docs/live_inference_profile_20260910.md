# Live inference profile and configuration cleanup — 2026-09-10

Outcome: GPU computation is much faster than the observed demand pipeline.
Serialized result accumulation and mmap allocation prevent GPU workers from
receiving enough work. No performance optimization or full-evaluation restart
was performed in this task; the600s request-timeout problem is still outstanding.

## Measured run

Artifacts: `/ephemeral/hendrik/fiber3d_train2/evaluations/inference_profiles/20260910T113753450009/`.
Files: `report.json`, `workload.json`, `coordinator.prof`, `coordinator.txt`, isolated `demand/` cache.

Classic checkpoint SHA256 `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17`.
PHerc0139 scan20260102150214, CT L1, requested chunks intersecting the10 locked
test fibers. Native reader/shared model adapter/shared DemandStore and worker
pool; no replica inference code. Source metadata, reference paths/hashes,
implementation hashes and runtime versions are in the report. Existing native
tracer build is GCC13 RelWithDebInfo; installed native input sampler used as in
the previous run. Linux amd64, H10080GB, bf16 model, f16 accumulators.

Nominal measured windows10s+10s+40s; actual61.58s. Total119.56s includes metadata,
real tile loading, model warmup,33.87s pipeline warmup and cleanup. All8workers
returned a result before the pipeline window (confirmed in console). A
synchronous completion overran the40s cutoff by1.49s. No hard realtime deadline;
native input-thread shutdown may still wait, as explicitly deferred earlier.

| Measurement | GPUs | Batch | Iterations / completed tiles | Measured wall | Tiles/s | Input Mvox/s | 128³-equivalent tiles/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model only512³ | 1 | 1 | 13 | 10.006s | 1.299 | 174.38 | 83.15 |
| Model only128³ | 1 | 16 | 75batches/1200tiles | 10.088s | 118.95 | 249.46 | 118.95 |
| Instrumented demand pipeline512³ | 8 | 1/worker | 9 | 41.485s | 0.217 | 29.12 | 13.88 |

Training uses128³ patches while the existing inference uses512³ tiles (64x input
voxels/tile). Model-only tests repeat a real, preloaded/preprocessed CT tensor
with two warmup forwards; adapter FP32 output conversion/clamp remains included.
128³/batch16 is a model-cost comparison, not equivalent tiling/output quality or
a proposed production setting. No eight-GPU model-only scaling measurement.

Model-only512³ mean/p50/p95 CUDA latency:769.63/769.68/770.57ms.
Model-only128³ batch16:134.46/134.56/134.88ms per batch.

## Pipeline breakdown

Counts/phase measurements below cover the41.485s window. Phase durations are
inclusive and overlap; they must NOT be summed into an exclusive wall-time pie.
The pipeline is instrumented with cProfile and wrappers, so its throughput is
not an uninstrumented performance claim. cProfile's multi-threaded inclusive
totals can exceed wall time; use explicit phase wall timers for durations.

| Phase | Count | Mean | p50 | p95 | Inclusive sum |
|---|---:|---:|---:|---:|---:|
| GPU model CUDA time | 9 | 0.7712s | 0.7712s | 0.7718s | 6.9407s across workers |
| Entire GPU worker | 9 | 0.9314s | 0.9323s | 0.9337s | 8.3827s across workers |
| Result completion/CPU staging + accumulation | 9 | 3.9005s | 4.0944s | 4.6428s | 35.1043s |
| Mmap chunk allocation (within completion) | 261 | 0.05563s | 0.03938s | 0.13952s | 14.5183s |
| Final chunk encoding/writing + publication | 11 | 0.03629s | 0.03668s | 0.04043s | 0.3992s |
| Native input tile reads, concurrent | 16 | 4.3263s | 4.4917s | 5.3416s | 69.2201s across threads |
| Completed worker result waiting for coordinator | 9 | 8.1321s | 6.3638s | 20.3309s | 73.1886s across results |

CUDA transfer/conversion/preprocess averages: H2D19.77ms, conversion0.91ms,
preprocess5.66ms, output/D2H event133.39ms. The reported output/D2H *wall* call
also waits for previously submitted kernels, so it is not pure transfer time.

Profiler observations:

-9result completions touched1050chunk-advance calls and allocated261chunks.
-2,138,112`pwrite` calls and2,154`fsync` calls: `_allocate_chunk` writes each row
  of every raw/weight mmap and syncs each array. Only small portions of enormous
  logical arrays are physically allocated, but this allocation method is costly.
-1,008NPZ member reads: `_advance` reopens staged tiles and `values[name]`
  materializes an entire member before slicing to the32³ output chunk. For
  current128³ output tiles, each raw7-channel member is56MiB and weight8MiB.
  If grouped as the observed paired accesses,504pairs imply31.5GiB of member
  materialization for9new64MiB tile payloads (576MiB). This byte amplification is
  inferred from member shapes/access counts, not physical-device I/O counters;
  OS page cache can serve reads. CRC scanning and repeated array copies remain.
-`DemandWorkerPool.pump` calls `store.complete` synchronously before freeing the
  paired input/result slot. Processing several results delays both queue reads
  and submission of new GPU work. Measured result queue delays explain idle GPUs.

Only11requested output chunks became ready in the window;774tiles were still
queued and7input reads pending at cutoff. This is an intentionally partial
profile, not a complete volume or tracing score. Chunk side32 gives360,448output
voxels as an upper bound (edge extents can be smaller). Startup/warmup publication
is not included in window counts. No source scan or whole-volume download.

## Historical suggestions from the initial profile (superseded)

The later user clarification rejects canonical accumulation order and scratch
durability: neither is required for demand inference. Suggestions1–3 below are
retained as history, not current guidance. The implemented correction removes
NPZ entirely and uses existing shared-RAM accumulator workers, without scratch
preallocation/sync. See live_inference_accumulation_fix_20260910.md.

1. Stop re-reading full NPZ members per destination chunk: memory-map uncompressed
   staged arrays or retain a bounded decoded-tile cache, preserving canonical
   per-add order and late-demand semantics.
2. Replace per-row reservation with efficient checked allocation while preserving
   ENOSPC safety and never overwriting adjacent live regions. Do not simply drop
   durability guarantees.
3. Move staging/accumulation off the scheduling thread with bounded queues and
   ordered per-source ownership; release GPU slots once output is safely staged.
4. Profile the native read path separately; it currently returns a float32 tile
   that Python converts back to uint8/uint16. Input timing is already a secondary
   bottleneck. This run does not attribute read time to network vs sampling.
5. Correct fixed per-request600s timeout independently before a full restart;
   active lower-priority requests must not be treated as dead just for queue age.

## Config-driven runner

`fiber_trace_3d/configs/eval_classic_collection.json` now owns checkpoint/digest,
split/output/native/catalogue paths, devices, pilot source/result, source mappings,
normal selectors, evaluation parameters and profile windows. Each source maps a
collection ID to sample_id, volume_id, CT URL/input_level, expected test count and
normal selection (`catalogue` or `manager-run`, both explicit). Different
collection and sample names are ordinary data, not a CLI special case.

The runner has no PHerc-specific code or confirmation flag. It retains prepare,
pilot, full and validation modes plus optional `--config`. Original split
membership/content, prepared config/workload digests and normal hashes are checked.
New prepared files go to `classic_s1_91500_20260910_config_v2`, preserving the
previous failed/full and successful/pilot artifacts. The old successful pilot is
explicitly configured as pilot_result; no full run was started by preparation.

Exact reusable approved command (existing environment, no activation/install):

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
```

Other modes use the same prefix: `prepare`, `validate-full`; then `full` only when
ready to restart. Re-running profile creates a new timestamped isolated directory.
`--config /path/to/workload.json` selects another workload explicitly.

Validation: native/live smoke62tracer+1cache+25Python passed11.90s after shared
store-setup extraction. Worker profiling off/on identity tests2passed5.49s and
preserve exact chunk values. Generic config/normal/membership and timing-summary
tests pass. Final independent review fixed CUDA event device selection and
capturing warmed-worker IDs before measurement; current measured run used GPU0
and all8were already warm, so those corrections do not change these results.
