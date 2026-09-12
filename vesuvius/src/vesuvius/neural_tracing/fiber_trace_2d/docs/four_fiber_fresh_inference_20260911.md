# Four-fiber fresh inference: 512³ versus 256³

## Outcome

Both configurations completed all four fibers in both directions (8/8 cases), under separate exclusive eight-H100 GPU-manager grants. No inference chunks were reused. On this single short trial per configuration, 256³ reduced coordinator wall time by 15.97% (34.682→29.144s), while processing 22.85% fewer input voxels. It did **not** improve inference throughput per processed voxel. Defaults have not been changed.

| Measurement | 512³ baseline | 256³ comparison |
|---|---:|---:|
| Coordinator wall, including source/store/model-worker setup and publication | 34.681649s | 29.144405s |
| Native tracing wall, including readiness waits | 25.096036s | 20.300160s |
| Inferred tiles | 87 | 537 |
| Processed input volume, including overlaps/padding | 11.676942 Gvox | 9.009365 Gvox |
| 128³-equivalent tiles / coordinator second | 160.546 | 147.404 |
| Summed GPU-worker tile wall | 100.818513s | 97.962473s |
| Mean worker wall / tile | 1.158833s | 0.182425s |
| 128³-equivalent tiles / worker second | 55.228 | 43.854 |
| Published spatial chunks (three channels each) | 143 | 139 |
| Reused inference chunks | 0 | 0 |
| Shared input/result buffer capacity | 72 GiB | 9 GiB |
| Completed directed cases | 8 | 8 |
| Trace threshold failures/restarts | 23 | 22 |
| Direction elapsed min / median / max | 11.585 / 19.988 / 25.095s | 18.051 / 19.088 / 20.299s |

One iteration per configuration, in the order512 then256; no repeat-run mean/p95 or confidence interval is claimed. Direction latency statistics describe eight different cases, not repeated measurements. A 512³ tile equals64 128³ tiles; a256³ tile equals8. Worker wall includes input transfer/conversion, adapter/model inference and output transfer/weighting, excluding queue wait and model loading; it is **not GPU-kernel-only time**. Summed concurrent timers cannot be added to coordinator wall. Manager queuing, preliminary validation, initial namespace/checkpoint-freeze/config/precision setup and final process teardown are outside coordinator wall.

## Workload and invariant settings

- Branch fiber-training-retrain; native build `/tmp/vc-live-native-cmake`, GCC13, RelWithDebInfo. Existing canonical batched tracer, one candidate worker per direction,8 concurrent directions. No tracer or inference implementation/numeric changes in this task.
- Classic snapshot `/home/hendrik/runs/s1a_128_1_single_8x8_20260801_084232/snapshots/best91_5k.pt`, SHA256 `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17`.
- GPU manager priority100, eight H100s exclusively,224 logical CPU allocation. Tasks512: `bb06b4e4-c997-416d-917b-751e3d0e9496`;256: `b85b89a7-bce6-4bcb-a142-60e59c4f8ed4`. Training cooperatively released resources; no bypass. The256 request waited for release outside its measured interval.
- bf16 inference; float16 chunk-major accumulators; scaledown_power2; border32; overlap96; output Zarr chunks32³; Blosc-Zstd;128 input workers,256 input slots,128 result slots,64 input I/O threads,8 decode threads;8 persistent model workers.
- Native beam8, lookahead2, step4, normalweight0.1, tangentweight10; ellipse error threshold20 base voxels, tangential factor4, reference refinement1; rendering disabled. Shared decoded cache216408444518 bytes.
- Sources and locked original filenames:

| Collection / scan | Fiber |
|---|---|
| PHerc0139 /20260102150214 | lt_20260909T071741261_000094.json |
| PHerc0332 /20251211183505 | kb_20260822T033517433_000058.json |
| PHerc0332 /20251211183505 | kb_20260822T164832186_000062.json |
| PHercParis4_fibers /20260411134726 | kb_20260729T182039882_000211.json |

Original split `/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910`. The new optional source.test_fibers selector preserves original filename ordering/content and only filters after complete locked collection count/hash/membership validation. Prepared full_bindings.json records the selected files and provenance. Default behavior without a selector remains the whole source.

## Bottleneck evidence and limitations

-512: first queue sample9.671s; initial81 source reads all submitted to GPUs by13.7s; first results around18.7s, bulk work mostly done29s. GPU utilization sampled in the burst was[100,100,53,100,100,100,94,100]%. At30.8–32.8s two late input reads were running with no GPU work submitted; the last two GPU tiles appeared33.8s. This is a startup + inference burst + input-limited tail, not sustained eight-GPU saturation throughout.
-256: first queue sample8.924s;128 submitted/uncollected GPU jobs and a loaded-input backlog sustained from~11.9–24.0s, then a small tail to29.1s. “Submitted” includes queued jobs, not128 simultaneous GPU kernels. The utilization query arrived after completion and cannot establish its GPU utilization percentage. Ready-input backlog argues against loader starvation during the bulk phase.
- Summed direction readiness waits141.005s/144.829s account for94.4%/94.2% of summed direction elapsed time. Remaining summed elapsed8.333s/8.856s also includes cache I/O and scheduling, not just trace CPU. Tracing compute is no longer the dominant end-to-end cost.
- Fixed absolute overlap/border changes stride416→160 and trimmed interior448→192. Smaller tiles reduce excess surrounding volume in this sparse request workload despite more overlap amplification. Aggregate worker processing time barely changes100.8→98.0s. The observed overall speedup is consistent with reduced excess inference and a shorter tail, not a faster model kernel; one run cannot isolate these effects.
- Source CT/normal caches were reused, not cleared or guaranteed fully warm. Both native logs mention streaming uncached0332 normal data. The second run may benefit from warmer source/page caches. No network-byte timing or pure-model CUDA event breakdown was collected in this short end-to-end trial.
- Tile-size changes alter receptive-context/blending and therefore trajectories: failures per case512=`[0,2,2,0,1,0,5,13]`,256=`[1,1,1,0,0,1,7,11]`. Published spatial chunk counts differ as well. This is not a fixed-residual/numerically identical comparison and does not establish quality superiority. Small cohort results must not be extrapolated into a guaranteed full50-fiber speedup.
- Reference-intersection prefetch116 spatial chunks in both configurations, lower priority than live trace demand. Both complete queue records show no pending work; no measurable post-trace drain interval was sampled.

## Reproduction and artifacts

Run from `/home/hendrik/villa2`, using the existing approved runner:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_fresh_tile512.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_fresh_tile512.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_fresh_tile512.json

/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_fresh_tile256.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_fresh_tile256.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_four_fresh_tile256.json
```

These are the executed commands, not blindly reusable fresh-output paths: a further repetition must first use new output roots and new single-use inference_cache_namespace values, then prepare/validate again. Keep the approved runner unchanged. Existing namespaces intentionally reject reuse; do not delete the benchmark evidence to rerun.

Evidence roots:

- `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_four_fresh_tile512_20260911_v1`
- `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_four_fresh_tile256_20260911_v1`

Each contains full.json/full_bindings.json and full/{summary.json,inference_summary.json,inference_queues.jsonl,native.log,cases/}. Case artifacts include trace.json, profile.json, reference.obj and trace.obj. Prediction namespaces under `/ephemeral/hendrik/vc3d_cache/live_inference/_namespaces/` are `classic_four_tile512_20260911_v1` and `classic_four_tile256_20260911_v1`.

Focused validation:

```bash
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src -q vesuvius/tests/scripts/test_classic_fiber_evaluation.py vesuvius/tests/neural_tracing/test_live_inference_cache_namespace.py
git diff --check
```

23 tests passed. Both real workloads passed checkpoint/split/config/normal provenance validation, exited0, completed8 cases and produced nonzero inferred tiles with zero warm chunks. No full100-direction evaluation or tuning beyond the requested tile-size comparison was performed. No platform-specific implementation changes; performance is measured only on this Linux/H100 host.
