# Embedding branch integration and exclusive bbox profile

## Merge and conflict resolution

Merged `fiber-training-embedding` tip `15b4005db` into `fiber-training-retrain`
without rebase/history rewriting. Existing work was first checkpointed separately:
`6f5d23cc9` manager, `530f28435` native bbox, `f6a2d18d1` shared inference,
`3caeab7e9` Fiber callers/tests/planning.

- Both branches added `gpu_manager/exclusive.py` and tests. Consolidated into
  one `ExclusiveTask` lease used by calibration cohorts and `run_exclusive`'s
  gated command adapter. Kept both regression suites, command watchdog,
  retained-token reconnect, SIGTERM/HUP cancellation, and verified worker join
  before release. Failed/uncertain cleanup retains ownership.
- Complete CPU/GPU/VRAM grants are checked before authorization. Heterogeneous
  device budgets use the locked scheduler's proportional CPU request arithmetic;
  the actual CPU grant must still equal the requested inventory capacity.
- `train.py` had overlapping sample-progress fixes. Incoming full handoff
  metadata plus explicit optimizer batch-size metadata preserve both cumulative
  sample logging and embedding/cooperative training. Kept its matching test.
- Active planning remains this task. Incoming task/status/log/plan were archived
  as `embedding_incoming_20260911_*` documents before merging. Both feature
  specs/changelog histories were preserved.
- Incoming collection-config test assumed the old S1a config was unchanged.
  Our S1a experiment remains LR0.1/2Msteps; new collection remains LR0.01/20ksteps.
  Neither config was edited. Test explicitly asserts collection LR/duration and
  compares the remaining shared settings, excluding experiment name/LR/duration.

## Manager changes reviewed

Incoming automatic bootstrap/shared socket discovery, logical-CPU inventory,
idle retirement, and full-process-identity service exclusion were retained.
Only terminal, zero-held/zero-grant ledgers may adopt a corrected CPU inventory;
live reservations are not reinterpreted. Manager service/monitor descendants
must not keep an unrelated workload alive. No protocol schema changes.

Classic and standalone inference launch through the shared exclusive lease,
priority100. Exact admission occurs before the command imports/uses CUDA.
Preauthorization rejects existing external GPU compute processes. This is
cooperative ownership, not OS enforcement against unmanaged jobs starting later.
No manager journal edits, forced task release, or unmanaged profile fallback.

## Reproducible commands and validation

Rebuilt the shared extension, which had reverted to the old villa installation:

```bash
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 \
  uv pip install --python /home/hendrik/.venv_las/bin/python --no-deps --reinstall -e ./volume-cartographer
```

Build43.79s; installed `vc.volume` is from this checkout (GCC13, normal
scikit-build Release default). Native standalone replay build is RelWithDebInfo.
Extension SHA256 `eac740fa5c4da55f7af43038204bdcb822a877e59dfea79e3871db4bf0957982`.
New keyword-only `with_mask/with_stats` API confirmed, removed `sample_zyx_block`
absent. No reduced precision or fast-math change introduced.

```bash
GPU_MANAGER_SCHEMA_PYTHON=/usr/bin/python3 /home/hendrik/.venv_las/bin/python \
  -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src' -q gpu_manager/tests

/home/hendrik/.venv_las/bin/python -m pytest --noconftest \
  -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' \
  -q gpu_manager/tests/test_exclusive.py vesuvius/tests/neural_tracing/test_managed_evaluation.py

/home/hendrik/.venv_las/bin/python -m pytest --noconftest \
  -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' \
  -q vesuvius/tests/neural_tracing/test_native_bbox_read.py \
  lasagna/tests/test_demand_predict3d.py lasagna/tests/test_inference_profiling.py \
  vesuvius/tests/neural_tracing/test_live_inference_profile.py

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/hendrik/.venv_las/bin/python -m pytest --noconftest \
  -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2/vesuvius /home/hendrik/villa2' \
  -q vesuvius/tests/neural_tracing/test_cooperative_integration.py \
  vesuvius/tests/neural_tracing/test_fiber_embeddings.py \
  vesuvius/tests/neural_tracing/test_fiber_trace_3d_sample_progress.py \
  vesuvius/tests/neural_tracing/test_memory_calibration.py \
  vesuvius/tests/neural_tracing/test_managed_launch.py \
  vesuvius/tests/neural_tracing/test_collection_training_config.py

bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests \
  /tmp/vc-live-native-cmake /home/hendrik/.venv_las/bin/python
```

Results:121 manager tests27.36s;24 exclusive/launch4.87s;80 native bbox/demand/
profiling25.77s;57 training/embedding33.16s,1 explicitly opt-in CUDA test skipped;
native wrapper62tracer+1cache cases and66Python tests29.85s. Initial failures were
the missing optional venv jsonschema dependency (used existing system validator,
no install) and the collection config assumption above. Independent integration
review found no blocking issue. CPU tests completed before real GPU measurement.

## Exclusive performance run

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
```

Report:
`/ephemeral/hendrik/fiber3d_train2/evaluations/inference_profiles/20260911T100545242010/report.json`.
Workload JSON beside it freezes parameters; report records source/checkpoint/
implementation hashes and per-tile/queue samples.

Manager auto-started normally at the home socket. Task
`b1229340-8414-4343-a00e-c5bd20997a2a` held224logical CPUs and all8H100s,
84,447,068,160managed VRAM bytes per GPU, exclusive priority100,
non-pausable/non-adaptable. Status checks showed this as the sole live task.
External-compute preflight passed. During two GPU samples six/seven devices
reported100% utilization; others60/64% and24%, respectively (point samples,
not utilization averages). Afterwards task finished with zero-held/zero-granted
resources, no active tasks, and all GPUs returned to idle baseline memory.

Input:10locked PHerc0139 held-out reference fibers,1599intersected output chunks,
source `PHerc0139/20260102150214` atL1, classic `best91_5k.pt`. This is the shared
inference pipeline driven by reference prefetch, **not** the100-direction native
trace evaluation. Eight GPUs;512cubed inputs,border32,overlap96,scaledown4,
32cubed output chunks;bf16 model,f16 raw/f32 weights;chunk-major,no prefault;
128readers,256input/128result slots,native64IO/8decode. cProfile disabled.

One40.00085s post-warmup window; all8workers warmed in9.46s. Entire command's
profile body including setup/model controls/cleanup148.66s. Historical controls
use the same workload/queue/precision settings; caches were not reset, and code
now includes the embedding branch, so this is not a randomized isolated A/B.

| Run | Fully accumulated512³ tiles | Window seconds | Equivalent128³ tiles/s |
|---|---:|---:|---:|
| Previous RAM-only,20260910T151437745034 |113|40.6604|177.8633|
| Previous repeat,20260910T151946615783 |119|40.3890|188.5661|
| Unified native bbox,20260911T100545242010 |327|40.00085|523.1888|

Observed2.77–2.94x versus those historical windows. Received324tile results and
published689output chunks.327completed includes6tiles already accumulating at
window start, with3left at end; these are throughput-window counts, not a fresh
327-tile end-to-end latency test.

## Bottleneck evidence

| Stage | Mean | p50 | p95 | Count |
|---|---:|---:|---:|---:|
| GPU model inference |794.30ms|770.60ms|956.85ms|324|
| GPU output processing/transfer event span |155.56ms|146.63ms|190.01ms|324|
| H2D CUDA span |24.07ms|23.42ms|28.28ms|324|
| Whole inference worker |980.73ms|948.97ms|1172.98ms|324|
| Accumulator task |2.973ms|0.761ms|11.183ms|34520|
| Completed native bbox read |1962.85ms|123.39ms|22349.90ms|123|
| Native reader calling-thread CPU |145.29ms|121.32ms|334.29ms|123|
| Shared input-slot copy |11.706ms|11.345ms|16.945ms|123|

Model-only512batch1 control:13iterations,1.29916tiles/s on one GPU;
CUDA mean/p50/p950.76967/0.77001/0.77100s.128batch16control75iterations,
118.979tiles/s on one GPU. Do not compare128native tiles with512tile equivalents
without accounting for different model efficiency/overlap/outputs.

Coordinator admission/background/event wall totals dropped from
6.518/19.378/14.358s to0.0926/1.1689/1.6755s. Executor submission dropped
6.439s→0.00435s. The prior GIL/allocation/casting bottleneck no longer dominates.
Worker time is now approximately model+output+H2D;523equivalent tiles/s matches
eight workers taking roughly0.98s per512tile. Accumulation is not backlogged:
sampled mean0.92tiles,p500,p954,maximum6.

Important remaining limit: prepared reads drained to zero near30s; GPU submitted
backlog fell121→48 by window end while128readers remained running. Completed
late native calls spent up to24.39s off-CPU (no major faults recorded on their
calling threads). Median native read is123ms, but long-tail input/cache waits
are real. This instrumentation does not distinguish cache locks, queued decode,
remote fetch, or underlying storage waits; do not assert a network diagnosis.
The40s window was still buffered and GPU-fed. Longer sustained throughput and
the cause of that native wait tail require follow-up profiling.

No further performance change, full100-direction evaluation, training run,
macOS/ARM validation, or long-run steadiness claim is part of this merge.
