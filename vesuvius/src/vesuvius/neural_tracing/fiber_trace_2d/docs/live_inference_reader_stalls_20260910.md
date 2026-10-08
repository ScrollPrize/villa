# Native reader stalls: profiling diagnosis

## Outcome

Confirmed blocking source: the native C++ `sampleZYXBlockTyped` initializes its
output and validity vectors **before** entering `nb::gil_scoped_release`.
For this uint8 source and512³ tiles that is256MiB of output/validity buffers.
The Python coordinator cannot run while a reader holds the GIL through that
setup. This is C++ input loading, not Python Zarr source-volume reading.

A second cost is the Python wrapper's full-volume dtype round trip:
`Vc3dCoordinateSampler.sample_block_zyx` converts the native uint8 array to
float32; `NativeTileReader` converts it back to uint8 for the shared input slot.
The native result was already in the required integer type. GPU normalization
and model precision need not change to remove this round trip.

This task profiles only: allocation boundaries and conversion behavior are
**not fixed yet**. No speculative throughput improvement is claimed.

## Controlled measurements

Run `20260910T152402140226`, report beneath
`/ephemeral/hendrik/fiber3d_train2/evaluations/inference_profiles/`.
Same PHerc0139 source/level, real512³ reference tile. Twelve serial operations
per case, with an independent Python thread requesting a heartbeat every2ms.
Operation and returned-buffer destruction are timed separately; resource counters
are for the calling thread. No GPU work overlaps these isolated controls.

The allocation-only controls call the real native sampler entirely outside the
volume and require `requested_chunks == 0`. They allocate the same output/validity
vectors but perform **no source chunk fetching, decompression or copying**.
Their branch never reaches the GIL-release scope. Small binding/metadata work is
included, so this is not a separately instrumented `memset` timer.

| Control | Operation mean / p50 / p95, ms | Heartbeat maximum gap, ms | Heartbeat intervals |
|---|---:|---:|---:|
| Idle30ms sleep | 30.057 / 30.057 / 30.062 | 2.076 | 176 |
| Allocation-only128³ | 0.192 / 0.163 / 0.320 | 2.238 | 2 |
| Allocation-only512³ | 63.703 / 63.497 / 64.780 | 65.354 | 25 |
| Prewarmed native512³ read | 167.903 / 167.917 / 168.018 | 64.753 | 625 |
| Prewarmed full reader wrapper512³ | 397.865 / 398.399 / 398.874 | 69.750 | 1895 |

The512³ allocation control used63.693ms calling-thread CPU per operation,
32,769 minor faults and zero major faults. Its returned-buffer destruction added
4.972ms mean. Native read result destruction added7.360ms mean. The small control
finishes too quickly for meaningful heartbeat percentile statistics (two samples).
Allocation behavior/page reuse differs by size; do not infer a linear scaling law.

The warmed native versus wrapper difference is about230ms per tile. This includes
casts, validity-mask conversion, wrapper checks and intermediate-buffer cleanup;
it is not a timer exclusively around the two dtype casts. Earlier real-pipeline
stage timers separately identify integer cast and shared-slot copy costs.
One preliminary native call warms the region; no assertion claims every subsequent
real read has zero storage I/O. Only the outside-volume controls guarantee no
requested chunks.

Heartbeat gaps alone include OS scheduling, but the repeatable no-I/O control,
near-equal CPU/wall operation time, idle control and source GIL boundary together
establish the GIL-held native setup as a real blocking source. The experiment
does not establish its exclusive percentage of end-to-end wall time or prove it
is the only remaining bottleneck.

## Pipeline evidence

Unmodified RAM-only windows before enabling additional diagnostics reached
177.8633 and188.5661 fully accumulated128³-equivalent tiles/s (40.6604/40.3890s).
These are short windows with different initial/final backlogs, not steady-state
whole-volume completion measurements. The model-only512³ control was about
1.299tiles/s on one H100, suggesting about665equivalent128³/s across eight GPUs
before loading, normalization, transfer, accumulation and publication overhead.
That extrapolation is not measured eight-GPU end-to-end throughput.

The diagnostic cProfile window reached156.8730equivalents/s in40.3893s,
99fully accumulated512³ tiles versus196GPU results received. It ended with
118tiles accumulating,125input reads admitted,109publications queued and only
2free result slots. Backlogs and coordinator stalls persist; adding input queues
alone cannot solve them. Accumulator compute per chunk task was mean2.395ms,
p500.884ms,p956.003ms.

The cProfile call graph unexpectedly includes reader/background/async-Zarr
activity and overlapping cumulative times greater than the wall window. Do not
treat `coordinator.txt` cumulative totals as a coordinator-only time partition.
Explicit phase timers are the useful evidence: admission10.011s,
background7.152s,dispatch2.660s,event handling17.219s in this diagnostic window.
In the earlier clean RAM-only run, `executor.submit` had6.439s wall but0.0054s
calling-thread CPU. The delays appear at operations that wait/release/reacquire
execution, consistent with starvation by GIL-held native work. Removing SQLite
eliminated that particular waiting site, not the underlying blocking source.

## Reproduction, identity and limitations

Temporarily set `profile.reader_responsiveness_probe=true` and `profile.cprofile=true`
in `configs/eval_classic_collection.json`, then reuse:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
```

Both flags are restored tofalse after this task. Model/buffer/concurrency settings
are unchanged:8H100,128reader threads,256input/128result slots,chunk-major
accumulation,bf16model,float16raw/float32weight. No build/install performed.
CPython3.14.6 with GIL enabled. Loaded extension:

`/home/hendrik/.venv_las/lib/python3.14/site-packages/vc/volume.cpython-314-x86_64-linux-gnu.so`

SHA256 `219ae6451fb282af1947d4785a8b24da5a1c359be97dbca769a98831049309b9`.
This installed binary was reused, not rebuilt; its exact compiler flags were not
recovered. The separate native tracer build is RelWithDebInfo but is not proof of
the installed extension's build flags. Module identity is stored in the report.

External GDB attachment was denied by the host ptrace policy. No native live
stacks were obtained and no security policy was changed. Independent review
approved the fallback controls and their limits. Both managed runs completed,
with all GPU grants/holdings released; no training started.

Profiling regression command:

```bash
/home/hendrik/.venv_las/bin/python -m pytest --noconftest \
  -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' \
  -q lasagna/tests/test_inference_profiling.py vesuvius/tests/neural_tracing/test_live_inference_profile.py
```

Ten tests passed, including helper cleanup on errors. No full100-direction trace
evaluation was run. Timing/fault counters are platform-dependent and optional.

## Targeted next changes (not implemented)

1. Allocate/zero the C++ vectors within the GIL-released scope, keeping Python
   object construction under the GIL. Audit buffer destruction as well.
2. Expose/reuse the native integer block result in inference without the float32
   round trip; retain strict selected-level/validity semantics and exact values.
3. Repeat the same controls and clean pipeline profile before further concurrency,
   buffer reuse or queue changes. Do not promise a speedup until measured.
