# Full-set dynamic inference batching — 2026-09-11

## Scope and current state

All three full comparisons completed100/100. With the same heap scheduler,
dynamic cap8 reduced coordinator time from860.380s to516.699s: **39.95% less time,
1.67× faster** than batch1. Normalized model-worker throughput increased78.05%.
The normal classic workload now uses the validated256³/cap8 settings with a fresh
output namespace; the generic model API retains default batch1. This is a continuation
of the [four-fiber implementation/memory pilot](dynamic_inference_batching_20260911.md),
not a claim that the pilot completed the acceleration task.

All runs use the locked **50 test fibers / 100 directions**, across six collections
and five scans. PHerc1229 remains the collection label but uses its embedded
PHerc1299 scan identity. Both Paris4 collections share their approved normal
source. No training fibers or full source-volume scan is added.

## Method

- Same classic `best91_5k.pt`, SHA256
  `f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17`.
- Eight H100 80GB GPUs, exclusive manager priority100, 224 logical CPU allocation.
- 256³ inference tiles, overlap96, border32, inference scale4, output32³ chunks.
  bf16 model inference and the existing float16 chunk-major accumulation.
- 128 readers, 256 input slots, 128 result slots, four publication workers.
  Same native tracer, parameters, reference thresholds and per-tile normalization.
- Native GCC13 RelWithDebInfo build at `/tmp/vc-live-native-cmake`.
- Fresh inference namespace/output root for every trial. CT and normal fetch
  caches are shared; off-reference trajectories can still cause new fetches.
- One full trial per configuration. Batch timing distributions describe model
  calls, not repeated full-run uncertainty. Batch-dependent kernels and existing
  arrival-order accumulation can slightly change trace paths and work counts.

Coordinator wall includes source/store/model-worker setup, inference,
tracing/readiness waits and required prefetch drain, but excludes manager waiting,
initial validation/checkpoint freezing and final process cleanup. Native
wall includes readiness waits. Their difference is **not** labeled entirely setup.
Worker time is summed across overlapping GPU workers, including transfer,
normalization, forward, output processing and synchronization; it is not CUDA-only
kernel time. Normalize work as `tiles × 8` for 128³ tile equivalents.

## Full-set results

| Setting | Coordinator seconds | Native seconds | Tiles / model calls | Summed worker seconds | Trace failures |
|---|---:|---:|---:|---:|---:|
| Batch1, priority heaps | 860.380 | 812.407 | 34,540 / 34,540 | 6,076.468 | 2,011 |
| Dynamic cap8, repeated queue scans | 684.879 | 637.367 | 34,394 / 6,865 | 3,603.703 | 1,990 |
| Dynamic cap8, priority heaps | 516.699 | 469.562 | 34,441 / 4,760 | 3,402.995 | 1,995 |

The same-scheduler batch1→cap8 comparison isolates batching without reintroducing
the slow queue implementation. Equivalent128³ tiles per summed worker second:
45.474→80.966 (+78.05%); per coordinator second321.160→533.246. Sustained loaded
rates were approximately45 versus82 actual256³ tiles/s across all8GPUs, not the
lower whole-run averages including setup/tails. Cap8 processed0.29% fewer tiles
because trajectories differ slightly; normalized worker throughput accounts for
that work difference. All runs cover the same8,081.558 directed reference mm.
Failure-count variation is recorded, not claimed as an accuracy improvement.

| Setting | Admission-stage seconds | Maximum publication backlog | Peak allocated / reserved GiB (max across GPUs) | Published spatial chunks |
|---|---:|---:|---:|---:|
| Batch1, heaps | 1.620 | 2 | 4.531 / 5.654 | 11,227 |
| Cap8, scans | 313.128 | 4,647 | 35.005 / 54.672 | 11,143 |
| Cap8, heaps | 1.351 | 11 | 35.005 / 54.672 | 11,169 |

Batch1 model-call duration mean/min/p50/p95/max:
0.175926/0.170549/0.171742/0.206244/0.940296s (34,540calls).
Cap8-heaps:0.714915/0.170998/0.731054/1.318776/2.589941s (4,760calls,
mixed actual sizes). Full distributions/counters are in the
[machine-readable summary](full_dynamic_inference_batching_20260911.json).

Control evidence:
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b1_20260911_v3_heap/full`.
Manager task `14dd7407-9c58-4984-8179-503e38bd910b`.

The heap run completed in24.56% less coordinator time. Its admission-stage total
was1.351s versus313.128s, and peak publication backlog11 versus4,647. Actual
batch histogram: `1:487, 2:18, 3:15, 4:5, 5:6, 6:4, 7:1, 8:4224`;
98.1% of tiles were processed in full8-tile calls. Maximum allocated35.005GiB and
reserved54.672GiB across GPUs remain within the observed cap8 envelope (seven
workers reserved40.663GiB; one reserved54.672GiB). Batch duration
min/median/max0.170998/0.731054/2.589941s. All100directions completed, allwarm0 and
all requested work queues drained. Source-cache warmth and slight path/work-count
differences remain caveats; the same-code batch1 control ran afterward.

Heap evidence:
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v3_heap/full`.
Manager task `669b56d8-9404-45e6-847f-280b14cf76f6`.

The following detailed reference counters describe the **old-scheduler row**:

100/100 directions completed; all warm inference counts were zero. Published
11,143 spatial chunks (three channels each). Mean actual model batch5.010;
histogram: `1:2628, 2:85, 3:54, 4:45, 5:59, 6:208, 7:577, 8:3209`.
Per-GPU peak allocation35.005GiB, peak reservation54.672GiB. Model-call durations
min/median/max0.170723/0.668824/2.373327s, mixing actual sizes and startup effects.

The coordinator spent313.128s in the admission stage (selection plus reservation
and executor submission),1.299s dispatching,37.647s processing result events and
41.396s pumping accumulation/publication. Publication backlog peaked at4,647.
These are stage wall timings, **not** disjoint exclusive CPU function costs.
The 60-second observer found47.27 main-thread CPU seconds, with ready input reads
remaining plentiful. Final requested work, reads, GPU jobs, accumulation and
publication queues were drained. Unrequested incidental partial chunks are not
unfinished requested work.

Evidence:
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_full_dynamic256_b8_20260911_v2/full`.
Manager task `ad90cbb2-3e21-4c66-9930-18f2eba4cb71`.

## Queue fix

The former admission selected globally by scanning all queued tiles, then scanned
the winning store again. A lazy per-store min-heap now retains the identical key:
foreground before background, decreasing remaining fiber length, then source name
and tile coordinates. Only new/promoted pending priorities add entries. Stale and
already-admitted entries are discarded. Global selection compares store heads.
Coordinator ownership, foreground/background policy, loader, accumulator and
publication semantics are unchanged. Detailed once-per-second queue accounting
still scans metadata; not every scan was removed.

Synthetic admission benchmark: five stores,32,000 pending tiles,256 admissions,
three trials; queue construction excluded. Old scans took
1.475868/1.539756/1.594855s; heap took0.001490/0.001446/0.001439s. Every selected
source/tile matched exactly. This measures admission only, not full-run speedup.
The seeded regression additionally interleaves promotions, duplicate/decreasing
priorities, background/foreground changes, source ties, requests after admission
and exhaustion of stale entries.

The heap run's60-second observer measured10.19 main-thread CPU seconds versus
47.27 for the old-scan run. Both kept ready input queues full; the heap run
sustained about82tiles256³/s without accumulating a publication backlog. Whole
process CPU includes loader/publication threads and is not interchangeable with
the coordinator main-thread time. Observer reports and raw queues remain beside
the respective full-run outputs.

The final batch1 observer used6.15 main-thread CPU seconds in60.173s. Ready inputs
remained full in both heap runs. GPU inference is now the sustained bottleneck;
the sparse final single-trace tail can be limited by new input reads and naturally
small batches. No extra buffer/worker sweep was necessary after the backlog was
removed. This is a measured useful setting, not an exhaustive global optimum.

## Reproduction

Use the existing approved runner, from `/home/hendrik/villa2`:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_full_dynamic256_b8_heap.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_full_dynamic256_b1_heap.json
```

Both were prepared with the same command using `prepare` instead of `full`.
These result namespaces are single-use; create a fresh output/namespace for any
future timing repeat rather than reusing finished inference. The cap1 control runs
after cap8, giving the control the warmer source cache. No installs or environment
changes were made.

## Failures and validation

The first full batch1 run (`classic_full_dynamic256_b1_20260911_v1`) timed out on
the native600-second chunk-request deadline while inference continued normally.
Longest-remaining-first priority can leave a short trace waiting that long. It
completed only2/100 directions and is **not** a valid complete performance row.
Its26,916 inferred tiles/8,429 published chunks and failure artifacts are retained.
Matched subsequent configurations explicitly use a3,600-second request timeout;
the priority policy was not changed. Prepared but unrun configurations include
the old cap8v1 and superseded cap1longwaitv2.

The standard `eval_classic_collection.json` now matches the measured heap-cap8
inference settings and full source membership. A regression compares those fields
against the measured configuration. Its separate, prepared output namespace is
`classic_s1_91500_dynamic256_20260911_v1`; it does not reuse benchmark predictions.
The standard command remains the same runner with that config (or its default).

The legacy CLI failure came from a test mocking *all* file opens as permission
errors, including checkpoint/provenance reads. Fixtures now use real tiny temporary
artifacts and mock only the `/proc/self/oom_score_adj` write and expensive inference.
The obsolete download helper mock and Zarr-v2 input fixture format were corrected.
The broader suite also exposed a pre-existing final progress-display gap:
asynchronous publication completed after the last tile report. A forced final
progress refresh now follows acknowledged flush completion; no inference math
changed.64 preprocess tests and6 subtests passed.106 batching/demand/evaluation/
profile tests passed; cached-only adapter head assertions passed separately.

Zarr fixtures use `zarr_format=2`, matching production. A suggested older
`zarr_version=2` spelling was tested and rejected by the installed Zarr3 runtime,
then reverted. This task does not claim Zarr2 runtime validation. GDB attachment
was denied by host ptrace policy; no host security setting was altered.

The cap16 pilot OOM remains a recorded limit, not an attempted full-set run.
Caps9–15, allocator tuning and caps128–1024 were not trialed. The measured cap8
configuration does not imply a portable maximum for other models/GPUs.

## Validation commands

```bash
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' -q --tb=short lasagna/tests/test_dynamic_inference_batching.py lasagna/tests/test_demand_predict3d.py vesuvius/tests/scripts/test_classic_fiber_evaluation.py vesuvius/tests/neural_tracing/test_live_inference_cache_namespace.py vesuvius/tests/neural_tracing/test_live_inference_profile.py vesuvius/tests/neural_tracing/test_cached_trace_profile.py
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/lasagna /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' -q --tb=short lasagna/tests/test_preprocess_cos_omezarr.py
git diff --check
```

Exact synthetic admission comparison, using the removed scan algorithm as the
historical control and the actual new production methods (no GPU or files):

```bash
/home/hendrik/.venv_las/bin/python - <<'PY'
import heapq, json, statistics, time
from lasagna.demand_predict3d import DemandStore, DemandWorkerPool

def setup():
    pool = DemandWorkerPool.__new__(DemandWorkerPool)
    pool.stores = {}
    for source in range(5):
        store = DemandStore.__new__(DemandStore)
        store.priorities = {(i, 0, 0): (i % 2, float(i % 127))
                            for i in range(source * 6400, (source + 1) * 6400)}
        store.tiles = {}
        store._pending_priorities = [(-p[0], -p[1], tile)
                                     for tile, p in store.priorities.items()]
        heapq.heapify(store._pending_priorities)
        pool.stores[str(source)] = store
    return pool

def trial(use_heap):
    pool, order = setup(), []
    start = time.perf_counter()
    for _ in range(256):
        if use_heap:
            _, _, name, tile = pool._next_admission()
            assert pool.stores[name].next_tile() == tile
        else:
            choices = [(-p[0], -p[1], name, tile)
                       for name, store in pool.stores.items()
                       for tile, p in store.priorities.items() if tile not in store.tiles]
            _, _, name, _ = min(choices)
            store = pool.stores[name]
            pending = [tile for tile in store.priorities if tile not in store.tiles]
            tile = min(pending, key=lambda t: (-store.priorities[t][0],
                                              -store.priorities[t][1], t))
            store.tiles[tile] = {"state": "inflight"}
        order.append((name, tile))
    return time.perf_counter() - start, order

times, reference = {False: [], True: []}, None
for _ in range(3):
    for use_heap in (False, True):
        elapsed, order = trial(use_heap)
        times[use_heap].append(elapsed)
        if reference is None:
            reference = order
        assert order == reference
print(json.dumps({"scan_seconds": times[False], "heap_seconds": times[True],
                  "median_speedup": statistics.median(times[False]) /
                                    statistics.median(times[True]),
                  "identical_order": True}, indent=2))
PY
```
