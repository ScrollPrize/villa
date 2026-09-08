# Fiber3D per-sample pipeline benchmark

The first bounded two-stage benchmark used the saved S1a reference loader on a
warm local cache. It measured loading and value augmentation only; it did not
run the model or alter normal training.

## Command

```bash
AGENTS_AGENT_MODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=/home/hendrik/villa/vesuvius /home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.pipeline_benchmark vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/training_baseline_s1a_20260729.json --output-root /ephemeral/hendrik/fiber3d_train2 --samples 256 --warmup-samples 32 --verify-samples 2 --load-workers 16 --augment-workers 32 --raw-queue-capacity 64 --result-queue-capacity 64 --max-in-flight 128 --cache-state warm
```

Result:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_095101_262442/result.json`

## Result

- Completion throughput: 22.67 samples/s (2.83 batches/s at batch 8).
- Deterministic ordered-delivery throughput: 25.63 samples/s.
- Cold lifecycle throughput, including 32 warmup items: 18.72 samples/s.
- Startup: 7.39 s.
- Processes: 16 load + 32 augmentation; measured summed worker CPU was 51.1
  cores because VC3D/native load work can use internal CPU beyond the ordinary
  one-thread BLAS/OpenMP limits.
- Capacity: raw queue 64, result queue 64, global in-flight credits 128. A raw
  128-cubed float32 volume plus mask is about 10 MiB; the conservative two-copy
  tensor bound is about 2.5 GiB before metadata/runtime overhead.
- Reorder high-water: 43 samples. This remained below the global credit bound.

| Measurement | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| raw load service | 664.0 ms | 532.0 ms | 945.0 ms | 1115.5 ms |
| value augmentation service | 1204.2 ms | 401.1 ms | 2379.3 ms | 2618.4 ms |
| raw queue residence | 44.0 ms | 15.6 ms | 185.5 ms | 292.7 ms |
| result queue residence | 16.4 ms | 11.5 ms | 48.8 ms | 130.4 ms |
| reorder wait | 1146.2 ms | 1554.5 ms | 2484.5 ms | 2660.3 ms |
| batch delivery wait | 261.1 ms | 218.4 ms | 645.0 ms | 748.9 ms |
| batch tensor stack | 44.8 ms | 52.4 ms | 65.7 ms | 68.7 ms |

The queues are not the throughput bottleneck: their residence times are small
relative to both stage service times, and every configured worker participated.
Per-sample scheduling removes whole-batch assignment stalls and lets later work
continue while an early sample is slow. Strict deterministic consumption can
still make a batch wait for that early sample; its high reorder wait does not
stall workers because the coordinator drains later results.

At the measured means, idealized capacity is about 24 samples/s for 16 load
workers and 27 samples/s for 32 augmentation workers. Merely scaling the same
kernels to 221 samples/s would require roughly 147 load plus 267 augmentation
workers, before contention. The next optimization must therefore split and
reduce the geometry/volume-load and value-augmentation kernels rather than only
increase queue depth or process count. The large augmentation p50/p95 gap is a
priority for substage profiling; this run does not yet attribute it to a
specific transform.

Two synchronous samples were compared field-for-field against the split path,
and all tensors and identities matched exactly. Synthetic tests also cover a
slow earliest sample with queues of capacity one, bounded reordering, stage
exceptions, and nonzero and zero-code hard worker exits.

## VC3D concurrency comparison

Three later matched repeats compared VC3D defaults with source-read and decode
admission both fixed at one. Persistent-cache probing remained parallel at its
fixed 32-worker capacity.

| Configuration | throughput range | median run | raw wall mean | raw CPU mean | CPU/wall |
|---|---:|---:|---:|---:|---:|
| VC3D defaults | 20.85–22.61 samples/s | 21.75 samples/s | 680.9 ms | 1065.4 ms | 1.56 |
| read=1, decode=1 | 21.94–22.32 samples/s | 22.16 samples/s | 668.4 ms | 1025.4 ms | 1.53 |

The constrained setting was about 1.9% faster by median run, but the ranges
overlap and this is not evidence of a meaningful throughput change. It did not
make raw loading single-core. Source inspection subsequently showed that
`sample_coords` independently dispatched interpolation to up to eight native
workers; the aggregate process timing did not justify the earlier attribution
to persistent-cache probes. Value augmentation continued to have CPU/wall near
1.0.

## Serial coordinate-sampler profile

The stable launcher was rerun with source-read, decode, and coordinate-sampling
limits all set to one. The fresh-process/no-warmup result is:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_112340_869943/result.json`.
The matched 32-warmup run is:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_112117_273980/result.json`.

| Stage | Native service rate | Equivalent 128^3 samples/s | summed service | effective host concurrency |
|---|---:|---:|---:|---:|
| coordinate-map generation | 10.40 MP/s | 4.96 | 51.63 s | 3.24 |
| dependency discovery | 9.02 MP/s | 4.30 | 59.52 s | 3.73 |
| coordinate interpolation | 5.83 MP/s | 2.78 | 92.16 s | 5.78 |
| chunk resolution/pinning | 682 chunks/s | 63.11 | 4.06 s | 0.25 |
| source codec calls | 1.61M chunks/s, 3.23M MiB/s output | 161,109 | 0.00159 s | 0.00010 |

The codec result is real but not a useful optimization target: this source uses
uncompressed 2 MiB chunks, so `decode_chunk_payload` is effectively validation
and ownership transfer. Filesystem/cache access is represented by the separate
chunk-resolution row. Across 256 samples, 2,767 chunks were resolved and 2,566
went through the pass-through codec path.

Every request reported one interpolation worker. Raw load CPU and wall means
were 927.6 and 963.3 ms respectively in the fresh-process run, confirming that
each loader process is approximately single-core with some blocking wait. Host
completion throughput was 16.00 samples/s (16.69 after 32 warmups), versus the
prior read=1/decode=1 median of 22.16 samples/s. Serial interpolation therefore
reduces host throughput by about 24.7%, while revealing its single-worker rate.
No sampling values or augmentation behavior changed.

## Requested-level locality optimization

Dependency discovery and pinned trilinear interpolation were profiled in
isolation on the same S1a 128-cubed workload, with one load worker and all VC3D
source-read, decode, and sampling limits set to one. Both modes used the same
Release (`-O3 -DNDEBUG`, LTO, x86-64-v3) binary; only the runtime locality fast
path changed. No GPU was visible to either run.

| Stage | Reference mean (p50/p95) | Optimized mean (p50/p95) | Rate before | Rate after | Speedup |
|---|---:|---:|---:|---:|---:|
| dependency discovery | 240.32 ms (239.36/245.56) | 45.01 ms (44.85/46.58) | 4.16 samples/s | 22.22 samples/s | 5.34x |
| pinned interpolation | 377.04 ms (366.82/454.55) | 53.26 ms (52.76/59.14) | 2.65 samples/s | 18.78 samples/s | 7.08x |

The combined service time fell from 617.36 ms to 98.28 ms, an effective serial
capacity of 10.18 training samples/s. Dependency discovery exceeds its
20-samples/s target. Interpolation remains 6.1% below its separate
20-samples/s target, although the requested combined 10-samples/s capacity is
met.

The dependency path reuses the previous coordinate's compact chunk footprint;
96.75% of points reuse it on this workload. Chunk-key resolutions fell from
1,073,741,824 to 5,974,848 and no hash probes were needed. The interpolation
path gathers 97.70% of points directly from one pinned chunk. A worker-local
eight-entry pointer cache reduces pinned-map lookups to 1,361 for the full
64-sample run. Non-power-of-two chunk dimensions use exact division rather than
a bit-shift approximation.

Reference artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_123107_188904/result.json`.
Optimized artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_123308_640082/result.json`.

Exact dependency sets, output bytes, coverage, missing-data behavior, level
transforms, non-power-of-two chunks, boundary coordinates and non-finite
coordinates are covered by the C++ reference-versus-optimized tests. The
ordinary training default enables the fast path but not its diagnostic
counters; the benchmark can disable either independently.

## 128-process raw-loading capacity

The benchmark's `--raw-only` mode uses the same stage-one workers, global
credits, control pipes and ordered coordinator but launches no augmentation
processes. It materializes and transfers each complete `FiberTrace3DRawSample`
through the bounded output queue, then discards it. Thus queue/shared-memory IPC
is included, while value augmentation and batch stacking are excluded.

Production-mode command arguments were 2,048 measured samples after 256 warmup,
128 load workers, output queue 128, global in-flight limit 256, source/decode/
sampling limits 1, warm cache, and locality diagnostics disabled. GPUs were
hidden. The result is
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_125817_240977/result.json`.

- End-to-end throughput: 93.76 raw samples/s.
- Completion-span throughput: 95.21 samples/s.
- All 128 workers produced measured samples; mean measured CPU utilization was
  116.1 cores and startup took 11.73 s.
- Raw service: mean 1,205 ms, p50 971 ms, p95 2,484 ms.
- Output queue/IPC: mean 36.8 ms, p50 27.0 ms, p95 91.4 ms.
- Reorder high-water: 236 of the 256 globally bounded in-flight samples.

The result is well below the 221 samples/s training target. Adding processes is
not sufficient: coordinate-map generation alone consumed 1,468 summed worker
seconds and slowed to 1.40 samples/service-second under host-wide contention.
It accounts for about 59% of total raw worker service in this run and is the
next loading optimization target.

## Coordinate-generation substage profile

A one-worker raw-only run profiled 128 measured S1a samples after 32 warmup
with all native limits at one and GPUs hidden. Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_132508_129751/result.json`.
The eager generator sustained 6.17 coordinate maps/service-second and the full
raw loader sustained 2.88 samples/s. Smooth displacement is selected for 50%
of this stream, producing a bimodal geometry distribution: geometry p50 was
36.6ms while p95 was 320.0ms. Smooth interpolation cost 136.0ms/sample on
average; the later full-map validity scan cost another 44.3ms/sample.

An exact cache-local candidate split the Z dimension into 16 eight-slice tiles
while retaining the same Torch operations and final coordinate map. Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_133140_083423/result.json`.
It was bitwise-identical to eager generation in tests for smooth modes
none/1d/2d/3d and tile depths 1/3/8. Geometry capacity improved from 6.17 to
8.23 maps/s, and full raw loading from 2.88 to 3.23 samples/s (+12.0%). This is
a safe intermediate result, not yet the planned implicit native fusion.

The matched 128-loader depth-eight run reached 111.71 samples/s end-to-end,
versus 93.76 samples/s eager (+19.1%). Mean geometry time fell from 716.7ms to
541.9ms and mean complete raw service from 1,205.4ms to 1,050.5ms. All 128
workers participated. Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_133508_256805/result.json`.
The remaining 221-samples/s target requires deeper fusion; tiling alone does
not remove the final coordinate map, scale pass or validity scan.

Fusing the unchanged NumPy selected-to-base multiplication and validity checks
into each hot eight-slice tile reached 116.53 samples/s (+4.3% over tiling,
+24.3% over eager). It eliminated the later full-map conversion/validity
passes; validity service fell from 126.3ms to 84.8ms under 128-way contention.
Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_133912_525705/result.json`.
The final base-coordinate map is still materialized for the current sampler;
removing that map requires the planned shared implicit coordinate provider.

## Native fused coordinate generation

The next implementation replaces the full-resolution Torch coordinate-map
pipeline with `vc.generate_fiber3d_coords`. The scalar C++ loop releases the
GIL and directly evaluates the affine transform plus optional low-resolution
1D/2D/3D smooth controls for each output point. It emits only the final
contiguous float32 base-coordinate map and uint8 validity mask; sampling still
uses the shared VC3D requested-level sampler. `geometry_backend=native` selects
it explicitly, `eager` retains the reference implementation, and `auto` uses
native generation when the installed binding and input contract allow it.

All runs below used the real S1a 128-cubed configuration, warm local data,
Release GCC 13 VC3D bindings, one source-read/decode/sampling native thread per
loader, and no visible GPUs. The one-worker runs measured 32 samples after 8
warmup samples:

| Backend | Raw samples/s | Geometry mean (p50/p95) | Coordinate maps/s |
|---|---:|---:|---:|
| eager cache-local | 3.04 | 177.57 ms (264.88/270.72) | 5.63 |
| native fused | 4.72 | 56.07 ms (93.49/93.61) | 17.83 |

The native artifact is
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_141944_055271/result.json`;
the matched eager artifact is
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_141905_331255/result.json`.
Smooth displacement is selected for half of the deterministic stream, so the
native distribution remains bimodal: approximately 93 ms with smooth controls
and 13 ms without them.

The production-topology native run used 128 loader processes, 2,048 measured
samples after 256 warmup samples, output queue 128, global in-flight limit 256,
and disabled locality diagnostics. It reached **227.33 samples/s** while using
73.47 summed CPU cores. This is 1.95x the previous cache-local fused Torch result
of 116.53 samples/s and exceeds the 221 samples/s target. Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_142050_133015/result.json`.
The exact benchmark invocation was:

```bash
/home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.pipeline_benchmark \
  /home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/training_baseline_s1a_20260729.json \
  --output-root /ephemeral/hendrik/fiber3d_train2 \
  --samples 2048 --warmup-samples 256 --verify-samples 0 \
  --load-workers 128 --augment-workers 1 --raw-only \
  --raw-queue-capacity 64 --result-queue-capacity 128 --max-in-flight 256 \
  --cache-state warm --volume-io-threads 1 --volume-decode-threads 1 \
  --volume-sample-threads 1 --disable-locality-stats \
  --geometry-backend native --geometry-tile-depth 8
```

Deterministic none/1D/2D/3D fixtures differ from the eager coordinates by at
most `3.8147e-6` base voxel (mean `4.5e-8` to `5.43e-8`), with identical
validity masks. Focused tests also require no more than `1e-4` sampled-value
mismatch and maximum uint8 difference one.

Low-resolution 3D precomposition was considered but not added. Direct lookup
of the already-small controls is simpler and the measured end-to-end loader
capacity now exceeds the target. Precomposition should be reconsidered only if
a future profile shows smooth-control evaluation dominating a new workload.

## Full value-augmentation pipeline after native generation

The stable CPU-only launcher was run without `--raw-only` using 256 measured
samples after 32 warmup samples, 16 load workers, 32 value-augmentation
workers, queue capacities 64, global in-flight limit 128, and one native I/O,
decode, and sampling thread per loader:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_pipeline_benchmark
```

It sustained 24.02 samples/s over the measured production span and 22.49
samples/s end to end. Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_143832_106216/result.json`.

| Stage | Mean | p50 | p95 | Worker count |
|---|---:|---:|---:|---:|
| raw loading | 230.30 ms | 247.54 ms | 281.99 ms | 16 |
| value augmentation | 1,102.59 ms | 344.28 ms | 2,205.08 ms | 32 |
| raw inter-stage queue wait | 1,705.34 ms | 1,813.98 ms | 2,871.93 ms | — |
| batch stack | 49.33 ms/batch | 49.33 ms | 62.45 ms | coordinator |

The raw queue wait proves that augmentation backpressures loading in this
topology. Augmentation alone consumes 1.105 CPU-seconds/sample on average, so
221 samples/s would require about 244 fully utilized CPU cores before loading,
IPC, ordering, and batching. Subsequent operation-resolved timing confirmed the
50%-probability anisotropic blur as the source of the bimodal latency.

## Native anisotropic value augmentation

The instrumented Torch reference run used the same launcher and 16-loader /
32-augmentation-worker topology. Oriented blur was enabled for 48.4% of the
deterministic samples and consumed about 1.82 seconds whenever enabled. Its
884ms all-sample contribution dwarfed isotropic blur (94ms), gamma (64ms),
normalization (57ms), contrast (50ms), and noise (11ms). The measured pipeline
rate was 22.65 samples/s. Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_145403_816239/result.json`.

The replacement `vc.volume.anisotropic_blur_3d` is a GIL-free scalar C++
kernel. It preserves the principal/perpendicular/perpendicular pass order,
Gaussian kernels, stable basis and replicated borders. For each pass it
precomputes the constant trilinear stencil represented by all Gaussian taps,
merges duplicate offsets, and traverses the volume once. Two 128-cubed float32
buffers replace the Torch implementation's repeated full coordinate grids and
17 `grid_sample` calls. It uses the calling thread only; parallelism remains in
the process-level augmentation pool. `value_augmentation_backend=auto|native|torch`
controls selection, with `auto` defaulting to native for eligible CPU tensors
and narrowly falling back for an absent binding or unsupported tensor contract.

The matched native run reached **58.81 samples/s** production and 52.07
samples/s end to end. Full value augmentation fell from 1,164.97ms to 494.43ms
mean, 358.62ms to 373.88ms median, and 2,243.86ms to 747.30ms p95. Enabled
native oriented blur averaged about 339ms, a 5.4x improvement, and complete
augmentation sustains **2.02 samples/service-second per worker**. Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_150249_214899/result.json`.

The constant stencil intentionally omits Torch's position-dependent float32
voxel-to-normalized-to-voxel coordinate round trip. On a full 128-cubed random
fixture, maximum error was `6.18e-5` and mean error `1.64e-6`; ramp error was at
most `3.10e-6`. Tests use scale-relative bounds and cover full-size random,
non-power-of-two, impulse, ramp, high-contrast, singleton, basis-threshold and
border cases.

A host-scale run with 64 loaders and 128 augmentation workers used 2,048
measured samples after 256 warmup samples. It reached 76.80 samples/s while
augmentation retained 1.44 samples/service-second under contention. All 192
workers participated, but the single coordinator became the next bottleneck:
batch stacking averaged 66.39ms per eight samples, result-queue wait averaged
2.10s, and only 78.5 CPU cores were effectively busy. Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260907_150407_151591/result.json`.
This does not indicate an augmentation scaling failure; it establishes that
coordinator delivery/batch construction must be addressed separately before
the global pipeline can approach 221 samples/s.

## 128-worker continuation

Using exactly 32 loading and 96 augmentation workers, the committed native
oriented-blur baseline delivered 81.54 ordered samples/s. Reusing that kernel
for axis-aligned isotropic blur reached 85.75/s. Fusing valid-only z-score,
contrast, brightness and gamma into one native reduction/output traversal then
reached 131.01 ordered samples/s (123.52/s worker production); value
augmentation fell to 322.6ms/sample. The native fused transform differs from
the Torch sequence by at most `2.15e-6` in tested random/partial/empty/constant
mask cases and retains the exact seeded Torch noise draw.

A 56/72 split and a benchmark-only 128-worker fused pool were both slower. The
fused pool's completed samples waited 1.02s on average to be received, proving
that large tensor queue transport and ordered materialization now suppress
worker utilization.

The bounded transport now stores each sample's float32 volume and boolean mask
in a generation-tagged shared slot. Queues carry only small metadata, and the
coordinator retains the pipeline credit until the consuming batch releases its
slot. Consecutive samples are exposed as a zero-copy batch view; a stale
generation, wrapped batch, duplicate release, or mismatched stream index fails
explicitly. With 256 slots (2.5 GiB), queue lag fell below 1 ms/sample and batch
assembly below 0.05 ms/batch.

Two exact native-loop improvements then removed the final augmentation
bottleneck. Oblique stencils use a precomputed linear-offset interior path and
tap-major contiguous row updates, while clamped border voxels retain the scalar
reference path and every voxel retains the original tap accumulation order.
Enabled anisotropic blur fell from about 530 ms to about 77 ms under the final
host load; all native/Torch tolerance tests remain unchanged.

The confirmed target run used 87 loading and 41 augmentation workers, one
native thread per worker, 512 shared slots (5 GiB), 128 warmup samples, and
4,096 measured S1a 128-cubed samples. The fixed launcher was:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_pipeline_benchmark
```

| metric | result |
|---|---:|
| measured production | **229.75 samples/s** |
| full process start through drain | **223.95 samples/s** |
| ordered completion span | 234.45 samples/s |
| aggregate worker CPU use | 127.45 logical cores |
| reorder high-water | 88 / 512 |
| load service mean / median / p95 | 377.67 / 366.49 / 532.46 ms |
| augmentation service mean / median / p95 | 172.37 / 170.38 / 288.04 ms |
| result queue mean / batch view mean | 1.09 / 0.060 ms |

Artifact:
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_090019_097829/result.json`.
This establishes loader-plus-augmentation capacity; the normal trainer still
uses its existing per-rank `DataLoader`, so connecting the reusable pipeline to
DDP training and measuring quality-per-hour remain separate work.

## Shared worker pool

The split benchmark now supports `--shared-workers`. It creates one process
population whose workers can run either stage without routing sample payloads
through a Python scheduler. Load workers atomically claim monotonically
admitted indices and augmentation workers consume the existing bounded raw
queue directly. Soft stage affinity preserves loader/cache and augmentation
working sets. A loader helps augmentation only when publishing to the raw queue
would block, or after every load has been dispatched. Stage operations are
initialized lazily, so augmentation-affine workers do not allocate unused VC
loader/cache state.

Admission is distinct from the hard shared-slot bound. The configured initial
frontier is 16 samples; each completed load advances that frontier by two. The
512 outstanding credits and generation-owned tensor slots remain the absolute
bound, and a modulo slot cannot be reused until its exact earlier owner is
delivered or explicitly released. Output remains reordered by deterministic
stream index. Exceptions identify stage/index, clean exits and hard exits are
failures, and the fixed-pool and stage-one-only modes remain available.

The stable launcher now selects 128 shared workers with a soft 87-load/41-
augmentation starting affinity:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_pipeline_benchmark
```

All measurements used the saved S1a 128-cubed reference, warm cache, Release
GCC 13 VC bindings, one native I/O/decode/sample thread per worker, 4,096
measured samples after 128 warmup samples, and no visible GPUs.

| topology / metric | production samples/s | full start/drain samples/s |
|---|---:|---:|
| committed fixed reference | 229.75 | 223.95 |
| contemporaneous fixed control | 232.20 | 226.22 |
| shared trial 1 | 224.68 | 220.58 |
| shared trial 2 | 227.44 | 223.03 |
| shared trial 3 | 224.56 | 219.87 |
| shared mean / median / approximate p95 | **225.56 / 224.68 / 227.16** | **221.16 / 220.58 / 222.79** |

Final artifacts are:

- `/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_102016_709985/result.json`
- `/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_102112_977641/result.json`
- `/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_102157_405229/result.json`

After the control pipe was narrowed to its final one-way form, an exact-source
confirmation run reached 227.18 production samples/s and 222.04 samples/s over
the full start/drain span. Its artifact is
`/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_102823_682491/result.json`.

The shared implementation exceeds the 221 samples/s production target and
retains 97.1% of the contemporaneous fixed control on the three-run mean. It
uses exactly 128 sample workers and the existing 5 GiB shared tensor bank.
Observed production CPU demand was 124.3-125.2 logical cores. Peak RSS was not
captured, which remains a measurement limitation. This is still benchmark-only;
the normal DDP trainer has not yet been switched to this reusable pipeline.
