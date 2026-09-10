# Chunk-major accumulator experiment — 2026-09-10

Observed completed throughput improved **2.84×**, from113.589 to322.899
equivalent128³ input tiles/s. The change is scratch addressing only: each output
chunk is contiguous inside each existing per-channel mmap file. Production
default remains `flat`; this is an opt-in, measured implementation.

## Implementation

- `lasagna/tiled_predict3d.py`: `_MmapBandDescriptor.shape_zyx` remains logical;
  `storage_shape` is grid-ZYX × local-ZYX when `chunk_side > 0`. A shared region
  iterator supplies ordinary3D views to both the existing native addition kernel
  and finalizer reads. Flat circular-Z addressing is unchanged.
- `lasagna/demand_predict3d.py`: `accumulator_layout="chunk-major"` creates
  sparse per-channel files with padded edge bricks and charges full touched
  bricks against the scratch budget. Layout participates in cache identity.
- `live_evaluate.py` passes the inference option; `live_inference_profile.py`
  allows a profile-only override. `configs/eval_classic_collection.json` selects
  chunk-major only in its profile section, alongside the existing128 result slots.
- No additional files per chunk, staging copies, eager zero writes, prefaulting,
  madvise, allocation reservation or scratch sync. Output Zarr and readiness
  semantics are unchanged. The32³ tasks, eight owners and existing native
  AVX-512 addition kernel are unchanged.

At32³, a raw f16 channel brick is64KiB and the f32 weight brick128KiB. Across
seven raw channels and one weight that is576KiB per output brick, in eight
separate contiguous ranges. The former full-volume row strides no longer
scatter a chunk's small rows across a large number of pages. The logical output
extent remains unchanged; only scratch boundary bricks include padding.

## Matched benchmark

Exact reusable command (existing environment, no rebuild/install):

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
```

Input: ten locked PHerc0139 test fibers, CT20260102150214 at L1,1599 requested
output chunks. Classic `best91_5k.pt` checkpoint SHA256
`f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17`.
Eight H10080GB GPUs,512³ input tiles, overlap96, border32, output scale4,
32³ output chunks, bf16 model/f16 raw accumulation/f32 weights. Eight input
slots,128 result slots, eight accumulator processes, four publication threads.
Shared slots total10GiB in both runs. Native sampler/kernel unchanged;
existing GCC13 RelWithDebInfo build, Ubuntu/Linux amd64. NumPy fallback remains
portable; macOS/arm64 were not benchmarked.

Artifacts under `/ephemeral/hendrik/fiber3d_train2/evaluations/inference_profiles/`:

| Metric | Flat `20260910T122023690046` | Chunk-major `20260910T130746099412` |
|---|---:|---:|
| Measured window |40.004s|40.037s|
| Fully accumulated512³ tiles |71|202|
| Completed512³ tiles/s |1.775|5.045|
| Completed128³-equivalent tiles/s |113.589|322.899|
| GPU results received in window |170|196|
| Requested chunks published |43|334|
| Accumulation backlog, start → end |11 →110|6 →0|
| Acknowledged accumulator tasks |8125|21437|
| Task mean |36.392ms|2.730ms|
| Task p50 |2.513ms|0.837ms|
| Task p95 |20.672ms|7.716ms|
| Task max |5065.245ms|117.969ms|
| Worker user CPU, sum |9.24s|14.42s|
| Worker kernel CPU, sum |289.17s|43.56s|
| Minor faults |19,290,464|2,614,227|
| Major faults |22,644|5,794|
| Kernel-accounted storage read bytes |0|0|
| Kernel-accounted storage write bytes |9,613,385,728|6,087,655,424|

All21437 acknowledged chunk-major tasks used AVX-512. Mean task time improved
13.33×. Despite processing more work, aggregate kernel CPU fell84.94% and minor
faults86.45%. These observations support the scattered-page mapping overhead
diagnosis; they do not identify individual kernel functions. `write_bytes`
accounts dirtied storage, not completed disk traffic; major faults here do not
imply disk reads. Eight workers accumulated43.56 kernel CPU seconds in a40s
window, not43.56 seconds of serial latency.

Chunk-major sampled states:425 running,1944 sleeping,23 D across2392 samples;
`pipe_read` accounted1945 wait samples. Accumulators now often wait for work.
Final backlog:0 accumulation,0 publication,2 GPU queued/active,8 input loads,
126 free result slots. The last part of this window was input-starved rather
than accumulator-bound. Native read mean0.842s, p500.624s, p951.829s,
max6.136s (190 completed reads). This is evidence for profiling loader/dispatch
next, not proof of a single remaining bottleneck. Publication mean0.384s across
334 chunks runs in background; overlapping timings cannot be added together.

GPU-only same-run controls, one GPU:512³ batch1 at1.299 tiles/s (13 measured
iterations, mean769.630ms, p50769.718ms, p95770.660ms);128³ batch16 at118.892
tiles/s (75 iterations, mean134.559ms, p50134.615ms, p95134.905ms per batch).
These are model-only geometry comparisons, not end-to-end pipeline rates.

Both layouts use warmup, cProfile and OS sampling with approximately40-second
measurement windows. Eight workers warmed before measurement; chunk-major
warmup9.10s.202 completed versus196 received includes the six results already
accumulating at measurement start. Rates are input-work equivalents, not unique
output voxels after overlap. This is one chunk-major window compared with the
previous matched flat128+OS window; cache warmth and the spatial region reached
differ. No repeated-run confidence interval or fixed-work microbenchmark is
claimed. Raw timing distributions and identities are retained in report.json.

## Validation

```sh
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -q lasagna/tests/test_demand_predict3d.py
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -q lasagna/tests/test_preprocess_cos_omezarr.py -k 'native_accumulator or shared_parallel_pipeline_matches_serial_output_exactly or shared_worker_uses_descriptor_queues or shared_parallel_pipeline_profile'
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests /tmp/vc-live-native-cmake /home/hendrik/.venv_las/bin/python
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -q vesuvius/tests/neural_tracing/test_live_inference_profile.py vesuvius/tests/scripts/test_classic_fiber_evaluation.py
```

Results:27 demand tests in16.99s;4 selected shared tests in10.17s; native wrapper
62 tracer+1 cache tests and42 Python tests in22.33s;5 profile/config tests
in1.56s. Counts overlap between commands. Independent plan/code review passed.
Exact raw/final values match the same-arrival-order oracle for f16/f32 and
scales1/2. Cross-brick writes, padding, circular-Z wrap, sparse budget rejection,
multiple workers/sources, late publication and no scratch sync are covered.

No model precision/arithmetic change. Live arrival scheduling may still change
floating-point addition order, as already explicitly authorized; separate live
runs are not promised bitwise-identical. Sequential order/layout stays unchanged.

No full tracing restart,8-buffer follow-up or commit in this task. The optional
8-buffer follow-up was not needed to establish the matched128-buffer layout
gain; do not infer that128 is necessary. Production stays flat pending promotion.
Temporary owned benchmark scratch was cleaned by the existing runner; finalized
partial Zarr chunks and reports remain. Scratch is rebuildable, not resume truth.

## Follow-up: writable prefaulting

User then requested a quick prefault trial. Added optional
`inference.accumulator_prefault` (default false), with a profile-only override.
It requires chunk-major layout. On first accumulation into each brick, each
owner/channel mapping calls Linux `MADV_POPULATE_WRITE` on the entire brick,
even when the incoming tile covers only a fragment. Ranges are page-aligned
and clipped at mapping end; successful offsets are remembered in mapping-local
RAM. Failed advice raises without marking success. The read/finalizer path
never advises. Addition math remains unchanged.

This is writable population, not `MADV_WILLNEED` or read-prefetch. It can
allocate sparse-file backing and page-cache memory, without Python zero writes
or scratch durability synchronization. It needs Linux5.14+ support; the Linux
UAPI value23 is used if Python's header build omitted the named constant.
Unsupported platforms/kernels fail explicitly; no silent fallback. See the
[Linux madvise manual](https://man7.org/linux/man-pages/man2/madvise.2.html).

Same approved command and workload as above, only
`profile.accumulator_prefault: true`. Artifact `20260910T131528402311` records
the enabled flag and implementation hashes. Advice runs inside the existing
accumulator task timer, not an untimed preparatory pass. Standard pipeline
warmup remains, as in all controls; there is no full-workload prefault before
the measured window. Eight workers warm in9.305s; total setup/profile/cleanup
97.27s, including the two model-only controls and40.076s pipeline window.

| Metric | Flat | Chunk-major | Chunk-major + prefault |
|---|---:|---:|---:|
| Completed512³ tiles |71|202|211|
| Completed128³-equivalent tiles/s |113.589|322.899|336.960|
| Accumulator task mean |36.392ms|2.730ms|2.797ms|
| Accumulator task p50 |2.513ms|0.837ms|0.780ms|
| Accumulator task p95 |20.672ms|7.716ms|8.010ms|
| Accumulator task max |5065.245ms|117.969ms|104.476ms|
| Acknowledged tasks |8125|21437|22527|
| Worker kernel CPU, sum |289.17s|43.56s|47.84s|
| Worker user CPU, sum |9.24s|14.42s|15.20s|
| Minor faults |19,290,464|2,614,227|2,718,858|
| Major faults |22,644|5,794|0|
| Kernel-accounted read bytes |0|0|0|
| Kernel-accounted write bytes |9,613,385,728|6,087,655,424|7,059,734,528|
| Requested chunks published |43|334|285|
| Final queued/in-progress publications |see baseline report|0/0|48/4|

No convincing prefault benefit in this quick test. Accumulated throughput was
4.35% higher, but mean task latency and kernel CPU increased slightly; published
output was lower with52 publications still queued/in progress at cutoff. Work
count and cache/scheduling differ, so neither a4% throughput win nor a precise
publication regression is established by single windows. All22527 acknowledged
tasks used AVX-512; prefault did not hide a backend switch. Major faults falling
to zero alone is not evidence of less work: advice moves population into an
explicit syscall, whose total cost is what matters.

Prefault backlog:4→0 accumulating,207 newly received results and211 completed;
the four extra completions were already accumulating at window start. Final
state2 GPU queued/active,8 input loads,126 free result slots. Native read mean
0.754s, p500.624s, p950.986s, max5.403s; publication mean0.467s. Input/dispatch
and publication timing now matter more than accumulation saturation.

Decision: retain prefault as an experiment, **disabled** in the checked-in
profile config and in production defaults. Profile stays chunk-major; production
layout stays flat pending promotion. No8-buffer rerun or full tracing restart.
Independent review passed.38 demand tests passed in22.40s, including real Linux
prefault f16/f32 oracle tests and spawned multiworker use, plus spy tests for
whole-brick ranges, page alignment/end clipping, repeated advice suppression,
failed-advice retry, no read-side advice and unsupported-platform rejection.
After measuring, only the prefault exception diagnostic was improved to include
the underlying OS error (no successful-path change), and the profile flag was
reset to false. The saved benchmark workload retains its true flag.

Final regression rerun with the same commands above: native wrapper62 tracer
+1 cache tests and53 Python tests in28.01s;4 selected shared tests in10.69s;
5 profile/config tests in1.85s. Counts overlap the38 demand cases. No build or
dependency changes. `git diff --check` passes.
