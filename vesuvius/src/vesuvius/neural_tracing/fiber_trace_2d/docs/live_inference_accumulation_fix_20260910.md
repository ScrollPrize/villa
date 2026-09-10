# Demand accumulation simplification — 2026-09-10

Outcome: removed NPZ staging, canonical-order waiting, scratch row-zero writes
and scratch fsync. The corrected eight-GPU profile handled71 tiles in40.003s,
versus9 in41.485s before (1.775 versus0.217 tiles/s, observed8.18x). This is an
instrumented bounded pipeline result, not full-tracing throughput or an isolated
CPU speedup: native input reads were considerably warmer on the repeat.

## Current path

`GPU shared-RAM result → existing chunk-owner accumulator processes → sparse mmap → background Zarr publication → coordinator ready acknowledgement`

- Reuse `tiled_predict3d._accumulate_process_main` directly; its old arithmetic
  body is extracted into `_execute_accumulate_process_task`, also used by the
  synchronous tiny-test adapter. Sequential inference arithmetic/order unchanged.
- Demand updates arrive in GPU result order, explicitly permitted by the user.
  One owner processes each chunk's weight and all raw channels without races;
  neither locks on files nor disk synchronization are required for visibility.
- Hash chunk-grid indices, not voxel origins: coordinates divisible by32 would
  otherwise collapse to one owner modulo8. Final benchmark recorded all eight
  owners active with nearly equal task counts.
- Shared result slots remain leased until every touched chunk acknowledges its
  contribution. Input slots release independently after H2D, allowing concurrent
  loading without waiting for accumulation. No array payloads in control queues.
- Each chunk tracks pending/applied contributor sets in coordinator RAM. Queued,
  inflight, accumulating and consumed tile state remains in SQLite; incomplete
  scratch/tiles are discarded on restart. Canonical edge ordinals are removed.
- Apply incidental neighbors without scheduling their missing tiles. A late
  request for a fully accumulated incidental chunk only queues publication.
- Bounded per-owner queues32 tasks; pending descriptors bounded by result-slot
  count times tile footprint. Default one accumulator process per GPU, four
  publication threads. Workload overrides: `inference.accumulator_workers` and
  `inference.flush_workers`. SQLite stays coordinator-only. Finalizers run only
  after all contributors acknowledge; results/errors return before ready commit.
- Fresh full-volume sparse files already read as zeros. No dense initialization,
  NPZ files, explicit scratch `pwrite`, scratch `fsync` or predecessor staging.
  Final Zarr file/directory durability and ready-ledger commits are retained.
- Logical touched-byte/allocated-block/free-space checks are advisory. File-backed
  mmap still incurs page faults/writeback, and concurrent disk exhaustion may
  kill a worker (including SIGBUS). Worker exit aborts, never marks data ready.
  Cleanup stops workers and joins publishers before removing scratch/slots.

## Reproducible benchmark

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
```

No environment installation or training changes. Same classic best91_5k
checkpoint SHA256 `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17`,
PHerc0139/20260102150214 CT L1, chunks intersecting10 locked test fibers,
1599 requested output chunks, eight H10080GB GPUs. Input512³, overlap96,
border32, output chunk32³, bf16 model and f16 raw accumulators, eight input
workers, one result slot/GPU. Native sampler/build unchanged: GCC13
RelWithDebInfo/Linux amd64. Python/Torch runtime, effective config, source,
reference hashes and implementation hashes are recorded in each report.

Artifacts under `/ephemeral/hendrik/fiber3d_train2/evaluations/inference_profiles/`:

- Original baseline: `20260910T113753450009/`.
- Intermediate asynchronous implementation, before correcting owner hash:
  `20260910T115606476956/`:71 completed tiles/40.0049s (1.7748/s),43 published;
  only65 GPU results received inside window, six prior results finished
  accumulation during window. Not a valid eight-accumulator scaling measurement.
- Final corrected owner distribution: `20260910T115814021990/`.

Each run has `report.json`, workload, cProfile files and isolated output cache;
temporary scratch is removed on shutdown. No full-volume download or full
evaluation restart. Final total96.378s includes setup, model warmup,9.822s GPU
worker warmup and cleanup. Measured windows approximately10+10+40s; all8 GPU
workers returned a result before the40s pipeline window. Accumulator cold-page
faults are NOT excluded by that GPU warmup criterion.

| Metric | Original | Corrected |
|---|---:|---:|
| Pipeline window | 41.4852s | 40.0029s |
| GPU results received in window | 9 | 71 |
| Tiles fully accumulated in window | not separately measured | 71 |
| Reported pipeline tiles/s | 0.216945 | 1.774870 |
| Input Mvox/s | 29.1178 | 238.2190 |
| Requested chunks published | 11 | 43 |
| Model-only512³ tiles/s, one GPU | 1.29923 | 1.29936 |
| Coordinator result wait, mean | 8.1321s | 0.07214s |
| Coordinator result wait, p50 / p95 | 6.3638 / 20.3309s | 0.04054 / 0.20842s |
| Worker model CUDA, mean | 0.77119s | 0.78751s |
| Worker model CUDA, p50 / p95 | 0.77122 / 0.77182s | 0.77101 / 0.87625s |
| Native input read, mean | 4.3263s | 0.73088s |
| Native input read, p50 / p95 | 4.4917 / 5.3416s | 0.63752 / 0.87415s |

Original tile handling included NPZ staging/attempted canonical accumulation;
corrected completed-tile counter releases only after all consumers acknowledge.
The final run also received71 GPU results inside its window, so receive-count
throughput gives the same comparison without differing completion semantics.

Corrected coordinator planning (`begin_complete`):71 calls, mean36.479ms,
p5019.609ms, p95148.238ms. Original synchronous completion was mean3.9005s,
p504.0944s/p954.6428s; these are deliberately different work scopes because
accumulation/publication have moved out of the coordinator.

Corrected accumulator tasks:8100 acknowledgements, mean32.470ms,
p502.527ms/p9520.834ms, maximum5.099s. Per-worker counts:
1033,1019,1001,993,992,1005,1022,1035. The long tail means mmap-backed work
still stalls; this profile alone does not distinguish filesystem allocation,
page faults, writeback, or other host contention. No claim of GPU saturation.
Publication43 tasks: mean671.904ms, p50645.981ms/p95989.263ms, executed off
the coordinator. Phase durations overlap and must not be summed as wall time.

All runs use coordinator cProfile plus explicit wall timers. cProfile cumulative
times can overlap/misattribute threaded native work; explicit timers and worker
events are the quantitative phase evidence. Cache warmth/source tiles processed
and system contention differ across runs; observed8.18x is not solely attributable
to the arithmetic worker change. Existing tests verify no scratch NPZ/pwrite/fsync.

## Validation

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests /tmp/vc-live-native-cmake /home/hendrik/.venv_las/bin/python
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -q lasagna/tests/test_demand_predict3d.py vesuvius/tests/neural_tracing/test_live_inference_profile.py vesuvius/tests/scripts/test_classic_fiber_evaluation.py
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -q lasagna/tests/test_preprocess_cos_omezarr.py -k 'native_accumulator or shared_parallel_pipeline_matches_serial_output_exactly or shared_worker_uses_descriptor_queues or shared_parallel_pipeline_profile'
```

- Native replay wrapper:62 native tracer cases,1 cache case,30 Python tests pass.
- Expanded demand/profile/config suite:19 pass; demand alone15 pass.
- Shared sequential/native accumulator regressions:4 pass,59 deselected. Exact
  sequential-vs-parallel output regression unchanged and passes.
- Demand f16/f32 at scales1/2: raw/weight accumulators exactly match an independent
  arrival-order NumPy oracle. Identity-model final uint8 output differs from the
  sequential-order control by at most1 in these fixtures (not a general bound
  for arbitrary models/traces).
- Tests cover reverse arrivals applied immediately, no scratch IO calls,
  duplicate acknowledgements, late incidental demand without extra inference,
  multiple sources/two workers, queue pressure, retained RAM slot ownership,
  publication readiness gate, publication failure and accumulator hard exit.
- Independent plan/code review completed; caught and corrected owner-hash
  collapse before final benchmark. `git diff --check` passes.

## Remaining limitations

- No real full-trace quality rerun; arrival order may slightly change final
  quantized outputs and potentially trace decisions. Precision is unchanged.
- File-backed accumulation is not RAM-only: cold-page/writeback tails remain.
- Native input still has the existing float32 sampling then integer conversion;
  no loader/model/batch changes in this task.
- The independent600s native request timeout remains unchanged; fix before
  another long queued full evaluation. Stuck native-read cancellation remains
  user-deferred. No full run was restarted and no commit was requested.
- Tested on Linux amd64; shared worker/fallback paths are portable but macOS/arm64
  were not exercised here. Disk-space checks are advisory rather than reservations.
