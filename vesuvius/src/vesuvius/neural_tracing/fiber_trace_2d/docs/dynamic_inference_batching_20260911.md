# Dynamic per-GPU inference batching — 2026-09-11

## Full-set continuation

The pilot below is historical, not the end of the acceleration task. Full-set
cap8 runs now completed100/100 directions:684.879s with repeated queue scans and
516.699s with the exact-order heap scheduler. The matching heap batch1 control
completed in860.380s: dynamic cap8 uses39.95% less time (1.67× faster). The normal classic workload uses the validated256³/cap8 setting; generic
inference still defaults to1. See [the full-set report](full_dynamic_inference_batching_20260911.md)
and its JSON companion for current results, configuration, tests and limitations.

## Four-fiber pilot outcome and scope (historical)

Dynamic batching is implemented in the existing shared GPU worker. On the four-fiber implementation/memory pilot, cap8 completed with35.01GiB peak allocated GPU memory and processed tiles at74.48 128³-equivalents per summed worker second, versus44.21 at cap1 (+68.48%). Coordinator wall was25.094s versus28.372s (-11.55%). Cap16 failed with CUDA OOM before any tile/chunk completed; no silent smaller-cap retry or partial-success claim.

**This is not full-set tuning.** The user explicitly notes that the full fiber collection supplies more queued work and smaller proportional tails. Stop the small-set sweep; do not choose a production optimum or extrapolate full-run speedup from these wall times. Cap8 is a successfully tested candidate for the full-set comparison, not a new global default. Batching defaults to1 for compatibility. Caps128–1024 are accepted integer settings but were not attempted: they are not plausible simultaneous256³ batches with this model on80GB GPUs under the measured memory behavior.16 is an observed failure, not proof that every intermediate cap or different allocator policy fails.

| Max batch / GPU | Total coordinator s | Summed worker s | Tiles / model calls | 128³ equivalents / worker s | Peak allocated / reserved GiB (max across GPUs) | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.372 | 98.623 | 545 / 545 | 44.21 | 4.53 / 5.65 | 26 |
| 2 | 26.224 | 75.072 | 545 / 288 | 58.08 | 8.88 / 12.65 | 25 |
| 4 | 28.712 | 67.053 | 546 / 170 | 65.14 | 23.60 / 27.79 | 22 |
| 8 | 25.094 | 58.001 | 540 / 103 | 74.48 | 35.01 / 54.67 | 22 |
| 16 | OOM, incomplete | — | 0 completed tiles | — | allocation failed | — |

| Cap | Actual batch size: number of calls | Mean actual batch | Batch wall min / median / max s | Native wall s | Published spatial chunks |
|---:|---|---:|---:|---:|---:|
| 1 | 1: 545 | 1.000 | 0.171 / 0.175 / 0.440 | 19.454 | 142 |
| 2 | 1: 31; 2: 257 | 1.892 | 0.172 / 0.255 / 0.887 | 17.093 | 143 |
| 4 | 1: 44; 2: 1; 4: 125 | 3.212 | 0.172 / 0.434 / 1.055 | 19.219 | 143 |
| 8 | 1: 38; 2: 2; 3: 1; 7: 1; 8: 61 | 5.243 | 0.172 / 0.734 / 1.122 | 15.896 | 140 |

One run per cap, ordered1,2,4,8,16. Histograms are actual model-call sizes, not queue depth. Each successful histogram's weighted total equals completed tile count; the sum of batch_seconds equals worker_seconds. No warm inference chunks in any trial; all successful runs completed8/8 directions. Published counts are spatial chunk groups, each containing all3channels. Batch wall min/median/max mixes actual batch sizes and includes first-call effects; it is not repeated-run variability.

The planned fresh cap1/cap8 repeats were prepared but **not run**, following the user's clarification to reserve tuning for the full collection. Their *_repeat.json configs and prepared roots remain labeled for audit; no inference was performed there.

## What changed

- `lasagna/tiled_predict3d.py`: shared `_multi_gpu_worker_main` accepts max_batch_size (default1) and optionally reports batches. It blocks only for the first descriptor, then consumes immediately available descriptors up to the cap. A partial batch runs immediately; no timer waits to fill it. It consumes at most one shutdown sentinel, finishing already collected items.
- Each item retains its uint8/uint16 interpretation, coordinates, requested products and input/result slot IDs. One B,1,D,H,W tensor goes through one model call; output batch/spatial dimensions are validated before publishing any result descriptors. The existing adapter keeps per-tile CT intensity normalization: no statistics shared across tiles, and no change to normal-vector encoding.
- Batch1 retains the same shared code path without an extra batch-copy. Larger batches allocate a combined input tensor once and copy converted tiles into it, avoiding a list of B float tiles plus torch.cat duplication. Synchronous pageable H2D finishes reading each shared input before its slot is released. GPU conversion uses GPU-owned data.
- Per-item weighted/downsampled outputs go into original result slots and through the SAME accumulator/finalizer. No new NPZ, disk staging, database, explicit filesystem synchronization, inference cache or tracer implementation.
- `lasagna/demand_predict3d.py`: positive integer cap validation, forwarding, actual batch counts/histogram/durations and per-GPU CUDA peak allocation/reservation telemetry. Existing per-tile worker timing receives amortized batch cost so it sums correctly. Optional detailed CUDA H2D/conversion timing sums disjoint per-input event pairs.
- `fiber_trace_3d/live_evaluate.py`: reads/validates `inference.max_batch_size`, forwards it, prints it and includes it in cache identity. `live_inference_profile.py` also forwards the setting, with optional profile override. Existing full-volume callers use the same worker with default batch1 (no new standalone CLI batching flag in this task).
- No model weight, precision, tile geometry, overlap, normalizer or trace parameter change. Batch-dependent GPU kernels and existing unordered float16 accumulation can change low-order values/trajectories; exact replay equality is not claimed for fresh runs. Tiny CPU fixtures verify exact output mapping/arithmetic through the batch path.

## Timing and bottleneck interpretation

Coordinator wall includes source/store/model-worker setup, native tracing waits, accumulation/publication and any reference-prefetch drain. It excludes manager waiting, preliminary validation/namespace/checkpoint-freeze setup and final process cleanup. Native wall starts later and includes readiness waiting. Worker time starts after dequeue, excludes model loading/queue wait, and includes transfer, conversion, adapter preprocessing, model forward, output weighting/downsampling/transfer and synchronization; **not CUDA-kernel-only time**. Summed worker seconds overlap across GPUs and cannot be added to coordinator wall.

Normalized worker throughput is `tiles × (256/128)^3 / summed_worker_seconds`. Total-coordinator normalized rates for caps1/2/4/8 were153.67/166.26/152.13/172.15 128³-equivalents/s. Work volume differs slightly with trace paths (9.144/9.144/9.160/9.060 Gvox, including overlap/padding). Mean worker time per tile is lower with batching; n=1 wall ordering is not a reliable tuning criterion.

The cap4 GPU-worker sum improves over cap2, but its full wall is worse. Its final queue samples show zero GPU jobs with6 then2 outstanding input reads at26.8–27.8s, and native log mentions uncached0139 normal data. That is a concrete input tail, not evidence that batch4 inference is slower. Both CT and normal fetch caches are shared/reused, not guaranteed entirely prefetched for new off-reference trajectories. Warmth and run order remain confounders.

Buffers stay256 input slots/128 result slots (9GiB total) across all trials.128 submitted GPU jobs means queued/executing/uncollected tiles, not128 simultaneous model calls. Cap8 had61 full8-tile calls (488/540 tiles); smaller calls handle ready-work/partial-tail conditions. No artificial full-batch wait.128 result slots may constrain very large caps, independently of VRAM. Full-set tuning should check sustained actual batches, queue/readiness profiles, steady throughput and tail separately.

### Cap16 failure

Manager task345fb261-898f-4b23-9ff6-a578e8cc818d admitted all8GPUs exclusively. A worker failed an8GiB allocation with5.50GiB free; PyTorch reported53.83GiB allocated plus18.72GiB reserved-but-unallocated (73.16GiB process use including non-PyTorch). This suggests allocator/workspace pressure as well as live tensors. No allocator tuning or retry was silently enabled. The error propagated to native/coordinator, all owned workers exited and GPU memory was released. Incomplete summary records0 inferred tiles and0 published chunks for all3sources. Keep that evidence; it is not a valid timing/quality row.

## Workload, commands and evidence

Same original four fibers/eight directions as [the tile-size pilot](four_fiber_fresh_inference_20260911.md): one PHerc0139, two PHerc0332, one PHercParis4_fibers. Same classic best91_5k.pt SHA256 f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17. bf16,256³ inference, overlap96/border32, scaledown_power2,32³ published chunks, float16 chunk-major accumulators,128 readers,64 I/O/8 decode threads, all8H100s exclusive at manager priority100/224logical CPUs. Native GCC13 RelWithDebInfo build at /tmp/vc-live-native-cmake. No C++ changes in this task.

Manager task IDs by cap:1=429721a7-ea05-49b2-a74a-c4767e821164;2=291cf13d-cf70-4da8-9c9a-c112ad304700;4=d2b780e1-fe8e-428b-ab3a-c0cefc83ab4e;8=211348bb-7457-4e52-8925-9f86a814d395;16=345fb261-898f-4b23-9ff6-a578e8cc818d.

Run from /home/hendrik/villa2. The same approved runner was used throughout, with prepare then full for each config (full includes validation; baseline also explicitly validate-full):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_dynamic256_b1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_dynamic256_b2.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_dynamic256_b4.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_dynamic256_b8.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_dynamic256_b16.json
```

To prepare, use the exact same command with `prepare` replacing `full`. These namespaces are single-use: any future repetition needs a NEW namespace/output root and prepare/validation; do not delete old results to rerun. The prior baseline result29.144s is historical, not substituted for the current cap1 baseline28.372s.

Evidence for capN: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_four_dynamic256_bN_20260911_v1`, with full.json/full_bindings.json, full/inference_summary.json, summary.json, inference_queues.jsonl, native.log and cases/0..7/{trace.json,profile.json,reference.obj,trace.obj}; cap16 has inference_incomplete.json instead. Prediction namespaces under /ephemeral/hendrik/vc3d_cache/live_inference/_namespaces/classic_four_dynamic256_bN_20260911_v1. No inference outputs deleted/reused.

## Tests and review

```bash
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' -q lasagna/tests/test_dynamic_inference_batching.py lasagna/tests/test_demand_predict3d.py vesuvius/tests/scripts/test_classic_fiber_evaluation.py vesuvius/tests/neural_tracing/test_live_inference_cache_namespace.py
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/lasagna /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' -q -x --tb=short lasagna/tests/test_preprocess_cos_omezarr.py -k 'shared_worker_uses_descriptor or shared_parallel_pipeline or shared_result_layout or calibrated_worker'
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' -q vesuvius/tests/neural_tracing/test_live_inference_profile.py
git diff --check
```

97 focused tests passed29.37s and6 legacy shared-worker tests passed12.09s. Covers batch1, actual multi-item calls, partial tails including cap128 with only5ready tiles, one-sentinel consumption, mixed uint8/uint16 tiles, distinct/permuted output slots, differing product requests, independent sample preprocessing, profile on/off, invalid B rejection before publication, spawned accumulation/backpressure and default shared-worker compatibility. These are CPU correctness fixtures, not CPU training; actual production trials all use CUDA under the manager. Independent plan/implementation review caught a coordinator variable wiring issue before benchmarks; fixed and retested.

The isolated-profiler tests also passed5/5 in1.64s (108 targeted tests total). Independent final-results review verified raw summaries, histograms, timings and OOM/scope claims.

At the pilot stage the broader legacy suite failed because its CLI fixture globally mocked builtins.open as PermissionError, blocking checkpoint/provenance reads. This has since been fixed with real temporary artifacts and a narrowly scoped procfs mock; explicit Zarr-v2 input fixtures and the final asynchronous-flush progress report were corrected too. The full continuation's latest preprocessing run passed64tests/6subtests. An initial broad long traceback had also segfaulted while formatting a released mmap array; subsequent diagnostics used short tracebacks. Pytest pythonpath entries are space-separated, and top-level legacy imports require repo/lasagna explicitly. No installs were made.

Portability: shared Python/PyTorch path retains CPU/non-CUDA behavior and gates CUDA telemetry; no OS-specific feature added. Performance/memory limits were measured only on this Linux/H100 setup. Full-collection benchmarking is covered by the continuation above; allocator tuning, intermediate caps9–15 and prepared small-run repetitions remain unexecuted.
