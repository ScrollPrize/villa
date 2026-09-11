# Collection best17k — full five-scroll evaluation, 2026-09-11

## Outcome

Completed all50locked test fibers/100directions across5scrolls (6collections).
No training fibers added. Compared with the
[classic current-code baseline](classic_full_5scroll_20260911.md), the collection
checkpoint is faster but worse overall on this tracing metric:
mean failure-ended reference span3.427927 →2.486310mm
(27.47% shorter),
failures1995 →2885 (+44.61%).
Coordinator516.699 →400.888s (22.41% less time).
Do not replace the classic quality baseline with this checkpoint.
No GT-heading/threshold/predictor-search change was made.

| Scroll | Classic failures | Collection failures | Classic mean mm | Collection mean mm | Classic median mm | Collection median mm |
|---|---:|---:|---:|---:|---:|---:|
| PHerc0139 | 564 | 523 | 2.233 | 2.376 | 0.146 | 0.066 |
| PHerc0332 | 27 | 34 | 12.994 | 11.884 | 0.309 | 0.813 |
| PHerc1299 | 159 | 359 | 3.418 | 1.503 | 0.790 | 0.077 |
| PHerc1451 | 19 | 61 | 2.341 | 1.149 | 0.762 | 0.216 |
| PHercParis4 | 1226 | 1908 | 3.785 | 2.577 | 0.162 | 0.058 |
| Global | 1995 | 2885 | 3.428 | 2.486 | 0.180 | 0.062 |

### New checkpoint distribution

| Scroll | P05 mm | Median mm | P95 mm | Mean mm | Zero spans |
|---|---:|---:|---:|---:|---:|
| PHerc0139 | 0.00000 | 0.066 | 11.551 | 2.376 | 34 |
| PHerc0332 | 0.00000 | 0.813 | 59.412 | 11.884 | 4 |
| PHerc1299 | 0.00098 | 0.077 | 10.188 | 1.503 | 10 |
| PHerc1451 | 0.00413 | 0.216 | 5.814 | 1.149 | 0 |
| PHercParis4 | 0.00143 | 0.058 | 9.489 | 2.577 | 51 |
| Global | 0.00056 | 0.062 | 10.406 | 2.486 | 99 |

All distances are matched reference progress during failure-ended attempts,
including zero-length attempts; successful terminal suffixes excluded.
Percentiles are inclusive linearly interpolated sample quantiles. Both Paris4
collections pooled by spans, actual1299 identity recovered from PHerc1229 label.
99zero failed spans,4failure-free directions. Small scroll samples and correlated
forward/reverse observations make these descriptive statistics, not independent
confidence intervals. The low training test/loss_total does not guarantee good
downstream tracing; this experiment does not establish the cause of regression.

Other global metrics, explicitly distinct:
- All actual attempts including terminal/zero spans:2.705719mm (2984segments).
- Positive seeded span mean:2.798567mm.
- Historical whole-corpus reliability:2.800263mm versus classic4.048877mm,
  computed8081.558059625 directed reference mm/(2885+1).
- Actual failed traced-polyline mean (no reset jumps):2.544466mm.
- Per-collection metrics retained in [JSON](collection_best17k_full_20260911.json).

## Snapshot, data and implementation

Checkpoint:
`/ephemeral/hendrik/fiber3d_train2/runs/collection_20260909_5scroll_128_sd1_16x8_20260910_162446/snapshots/best.pt`.
SHA256 `3fecbcd6806242baa7bfc7ffcd7ec27988b28ea43e87c589e6eacc7c280a6a7c`.
Embedded step17000, samples_seen2176000, optimizer_global_batch_size128.
Selected by saved test/loss_total=0.3195892147353438, not by this tracing test.
Checkpoint config is authoritative: UNet base16/depth5, GroupNorm,
pixelshuffle, unconditioned single branch,7rawchannels, native output_stride4.

Baseline docs and plan committed before running as `af1a59c67`, atop
`faa197d20` / native`75013f6cc` / shared inference`a238c4598` /
integration`175e3645b`. No production code needed changing.
New workload only changes snapshot path/hash and fresh output/cache namespace
relative to eval_classic_full_dynamic256_b8_heap.json.
Focused native-stride regression and workload file were present during run.

Same locked50fiber split and all CT/normal/source bindings as linked baseline:
`/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/split_95_5_length_seed20260910`.
Actual scrolls:0139(10fibers),0332(3),1299(5),1451(2),Paris4(30).
All fiber content/membership and normal/checkpoint hashes verified by prepare and
validate-full. Reused fetched source data only; every source reports warm_chunks=0.
Fresh output:
`/ephemeral/hendrik/fiber3d_train2/evaluations/collection_best17k_full256_b8_20260911_v1/full`.
Complete binding audit one directory above:full_bindings.json.
JSON below retains raw summary/config/binding hashes and manager terminal record.

Native factor4 already supported end-to-end:256input→64output per model tile.
Requested total inference factor4/native4=1, so there is no second output
downsample. Blend weights are put on native lattice. InputL1 +factor4 gives
outputL3/base scale8, identical output lattice to classic (which inferred dense
output then filtered down). New native-head/target semantics versus classic dense
head are part of the checkpoint change; equal lattice does not imply equal math.
GT tangent still only selects/sign-aligns model seed direction; investigation
remains deferred per request.

## Exact commands

From /home/hendrik/villa2, same approved runner, in this order:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_best17k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_best17k_full.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_collection_best17k_full.json
```

The config still references the old classic pilot solely as the existing
infrastructure-success gate; no new-model pilot is claimed. The actual new-model
run is full50fibers, not a short subset. Output namespace is single-use; future
runs need fresh namespaces rather than reuse completed predictions.

Exclusive manager task`bc3bdf5b-74bf-4271-a1ed-e79869b91321`, priority100,
224logicalCPUs/all8H10080GBs. No training run. Manager reports finished/completed,
terminal=true, held/grantedCPU0 and GPUs{}, shell exit0.

## Performance

| Setting | Classic best91.5k | Collection best17k |
|---|---:|---:|
| Coordinator seconds | 516.699 | 400.888 |
| Native seconds (includes readiness waits) | 469.562 | 353.194 |
| Inferred 256³ tiles | 34441 | 34162 |
| Model calls | 4760 | 4531 |
| Summed worker seconds | 3402.995 | 2702.610 |
| Full8batch tiles % | 98.12 | 98.33 |
| Peak allocated GiB | 35.005 | 36.677 |
| Peak reserved GiB | 54.672 | 46.648 |
| Published spatial chunks (3channels each) | 11169 | 10987 |
| Peak publication backlog | 11 | 13 |

Same256tile/96overlap/32border/32outputchunk, batchcap8,128readers,
256inputslots,128resultslots,4publication workers, chunk-major float16accumulation,
bf16inference; same GCC13 RelWithDebInfo native tracer with beam8/lookahead2,
trace-step4,trace-to-base2,normalweight0.1,tangent10,ellipsoid20/80basevox.
At one busy snapshot7GPUs showed100% utilization and one64%; sustained processing
about100tiles/s late in the bulk phase. No exact CUDA-only utilization fraction
claimed. New-model batch histogram and mean/min/p50/p95/max timings are retained
in JSON. Final requested/read/GPU/accumulation/publication queues all zero,
buffers returned.131660incidental partial chunks were unrequested, not missing
required work.

One full run per checkpoint; source cache/order and changed trajectories affect
workload. New run inferred0.81% fewer tiles. Coordinator time excludes manager
waiting/initial validation/finalcleanup; native time includes readiness waits.
Worker time is summed across overlapping processes and includes preprocessing,
transfer and output work, not just GPU kernels. No claim of a controlled
architecture-only speedup.

## Validation and analysis corrections

- 83 dynamic-batching/demand/runner tests passed in28.49s before launch.
- 2 additional native-output/model-scale tests passed after evaluation in2.04s
 (282deselected). Initial attempt failed collection because tests.zarr_utils
 required vesuvius project path; corrected command below. No test failure hidden.
- Independent plan/implementation review passed.
- 100 saved artifacts independently checked against summary, including all2885
 failure spans, provenance pairings, full reference extent, segment counts and
 credited-length sums. No inference rerun for analysis.
- Classic saved artifacts reaudited with the same script; existing numbers unchanged.
- Analysis initially assumed every direction had a terminal suffix, as all100did in
 classic. One new direction failed/reset at the endpoint and has no terminal suffix.
 Read-only analysis now counts actual segments (99terminal), retaining all failures
 and validating complete extent; no change to runtime reset rules.
- GT-heading investigation intentionally deferred, not silently fixed.
- No installs or production-code changes; no hyperparameter tuning.

```bash
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' -q --tb=short lasagna/tests/test_dynamic_inference_batching.py lasagna/tests/test_demand_predict3d.py vesuvius/tests/scripts/test_classic_fiber_evaluation.py
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/vesuvius/src /home/hendrik/villa2/vesuvius /home/hendrik/villa2' -q --tb=short vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'native_output_stride_truncates_decoder or adapter_checkpoint or total_stride_below_native_stride'
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py /ephemeral/hendrik/fiber3d_train2/evaluations/collection_best17k_full256_b8_20260911_v1/full
```

The last command only reads saved artifacts and prints JSON. Omitting its path
reaudits the classic baseline.
