# Benchmarks

Usage, methodology, and full results for the [package benchmark summary](../README.md#performance).
Run all commands below from the package root (`vc_reader/`).
Generated JSON reports under `docs/benchmarks/` are kept locally and ignored by
Git. Report paths below refer to those local files; benchmark input fixtures
remain versioned.

Current results were refreshed **2026-10-17** at `0fd629ebd`, with the same
Release extensions for every suite. The refresh manifest (`docs/benchmarks/refresh.json`)
records report hashes and coverage. Current results are in the PyTorch, resident
geometry, augmentation catalog, and reader/pipeline sections. The two optimization
history sections are explicitly marked historical and retain their original data.


## PyTorch CUDA batches

`bench_torch.py` measures GPU-ready batches using the same image data and
per-sample seeds across three routes:

- `serial`: ordinary `DataLoader(num_workers=0)` with per-sample CUDA processing.
- `batch`: the public `Pipeline.batch()` API after the CPU loader.
- `prefetch`: the public `CudaLoader` wrapper, with one batch queued ahead.

The chain is min-max normalization followed by the same 12 augmentations, with
identical parameters and order, as the [single-crop benchmark](#single-crop-latency-12-augmentations-cpu-output).
Both harnesses use `heavy_transforms()` from `bench_pipeline.py` to keep the
recipe consistent. The loader uses one combined pipeline with `compact=False`;
the single-crop benchmark separates normalization to measure its stage cost.
All routes end with stacked CUDA tensors. Uploads, GPU
casts, reading, augmentation, collation, and output disposal are included in
epoch throughput. Batch latency measures `next()` through completion on the
consumer stream; final producer-stream completion is included in epoch timing.
There is no model, backward pass, or simulated training delay. Prefetch benefit
during actual training must be measured with that model separately.

The first full epoch validates finite output/shape and exact crop hashes across
routes and warms readers, CUDA, and persistent workers. It is excluded from
timing. Backend order rotates between repetitions. OS/reader caches remain
warm; this is not a cold-storage benchmark. JSON records raw measurements,
mean/p50/p95/min/max, epoch throughput, seeds, configuration, extension hashes,
build flags, and environment. `--modes` limits the routes for profiling.

Use an existing Release CUDA build and an environment with PyTorch and NumPy:

```bash
cmake --build /tmp/vc-reader-cuda --parallel 4

# Small generated-input smoke check; creates and removes a temporary raw Zarr.
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --shape 16 16 16 \
  --batches 4 --reps 1 --output /tmp/torch-smoke.json

# Real-data comparison: 3 × 64 batches of four 128³ crops per route.
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --array /path/to/volume.zarr --level 2 \
  --batches 64 --batch-size 4 --reps 3 --output docs/benchmarks/torch.json

# Regression tests: real CUDA when available, CPU configuration checks otherwise.
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python tests/test_torch.py -v

# A separate profiling run, excluded from performance comparisons.
AGENTS_AGENT_MODE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=/tmp/torch-loader-profile .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --modes prefetch --shape 128 128 128 \
  --batches 4 --reps 1 --output /tmp/torch-profile-timing.json
```

The regression suite checks per-sample equivalence, image/label routing,
metadata, source removal, variable/pyramid outputs, dtypes, CUDA streams,
training backward, epoch seeds, spawn workers, partial batches, early iterator
closure, and validation errors. `VC_READER_NATIVE_DIR` is inherited by spawned
test/benchmark workers so they load the same extensions as the parent. Without
that variable, tests use the installed extensions.

Measured on PHercParis4 level 2 with RTX 5090, PyTorch 2.14.1+cu130 and a
Release CUDA build: three repetitions of 64 batches of four 128³ crops after
one validation/warmup epoch per route. Four spawned CPU loader workers and four
native reader threads per worker serve the batch routes. All 256 crop hashes
matched exactly across routes. The complete report (`docs/benchmarks/torch.json`)
contains the exact command, input, build hashes and raw measurements.

| Route | Mean batch ms | p50 ms | p95 ms | Crops/s |
| --- | ---: | ---: | ---: | ---: |
| Per-sample CUDA, `num_workers=0` | 7.135 | 6.920 | 8.294 | 560.35 |
| `Pipeline.batch()` | 5.158 | 4.874 | 5.943 | 775.21 |
| `CudaLoader`, one batch ahead | 5.187 | 4.774 | 5.876 | 770.86 |

The public prefetch loader delivered 37.6% higher throughput than the serial route.
These are loader-only results. The native executor still loops over samples within each submitted batch.

### Current kernel profile

A separate Nsight Systems trace of the same 12-augmentation chain covers four
warmup and four measured batches (32 crops). The sum of GPU kernel time is
31.732 ms. This excludes host work,
transfers and waiting; instrumented wall timings are not performance comparisons.
The largest kernel groups in that trace, amortized per crop, are:

| Kernel group | µs/crop | Share of kernel time |
| --- | ---: | ---: |
| Convolution | 275.8 | 27.8% |
| Trilinear resampling | 117.7 | 11.9% |
| Statistics collection | 98.8 | 10.0% |
| Local-field expansion | 82.5 | 8.3% |
| Median filter | 56.5 | 5.7% |
| Nearest-neighbor resizing | 51.4 | 5.2% |

Convolution now accounts for the largest kernel group. These groups aggregate
shared kernels across transforms; they are not per-augmentation timings.
[Kernel/API report](../docs/benchmarks/current-profile.csv) ·
Profile configuration (`docs/benchmarks/current-profile.json`).

```bash
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
AGENTS_AGENT_MODE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/vc-readme-refresh-profile \
  .venv/bin/python benchmarks/bench_torch.py --native-dir /tmp/vc-reader-cuda \
  --modes prefetch --array "$VC_BENCH_VOLUME" --level 2 \
  --batches 4 --batch-size 4 --reps 1 --output docs/benchmarks/current-profile.json
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  /tmp/vc-readme-refresh-profile.nsys-rep > docs/benchmarks/current-profile.csv
```

## Resident CUDA geometry

Current float32 128³ random-image timings: 200 synchronous calls per case after
20 warmups, rotating case order. Upload, download and output destruction are
excluded; public-call input preservation, validation and field construction are
included. Three seeds validate exact same-backend replay and label outputs.

| Operation | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| flip_x | 0.099 | 0.097 | 0.150 |
| flip_all | 0.105 | 0.078 | 0.213 |
| elastic | 0.250 | 0.234 | 0.378 |

The image-only Elastic case uses sigma 12/magnitude 2; the README gallery uses a
stronger sigma 32/magnitude 8 recipe and transforms both image and label arrays.
Complete resident geometry report (`docs/benchmarks/geometry.json`).

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_cuda_geometry.py \
  --native-dir /tmp/vc-reader-cuda --warmup 20 --reps 200 \
  --output docs/benchmarks/geometry.json
```

## README augmentation timings

[`bench_augmentations.py`](bench_augmentations.py) measures all **61** README
recipes (59 volume examples and two 2D companions) on CPU, CUDA with host
inputs/outputs, and CUDA with resident inputs/outputs. Recipe parsing and sample
construction are shared with preview generation in `scripts/augment_examples.py`;
there is no second hand-maintained transform list. This command updates only
timing rows, preserving recipes, descriptions, images and the Elastic video.

The real PHercParis4 level-2 image starts at `(9000, 3600, 3600)`, is 128³, and
is divided by 255 to float32. The reference image starts 384 voxels farther along
Z. Masks are int16 CT > 0.5; geometric cases route both images and labels,
ChannelMisalignment uses two identical image channels, and the two companions
use a central slice or CT-derived coordinate surface. These match the README
preview helpers and the previous catalog's workload construction.

One CPU thread, `compact=True`, one warmup and five measured calls per route,
seeds 123–127; case and backend orders are deterministically shuffled. All calls
complete synchronously. Reading, setup, validation/comparison, output disposal,
initial resident upload and final resident download are excluded. API input
copies and validation are included. Uploaded CUDA includes host/device transfers.
The short five-call protocol matches the old tables; inspect raw samples for
jitter rather than treating speedup ratios as confidence intervals.

Full report (`docs/benchmarks/augmentations.json`) records recipes, input hashes,
seeds, build hashes, commands, mean/p50/p95/min/max, every timing and numerical
comparison. All floating outputs pass `atol=5e-5, rtol=1e-5`. Strong Elastic has
8 CPU/CUDA label differences across 10,485,760 voxels, consistent with float32
nearest-neighbor boundary rounding; all other labels match exactly. CUDA host
and resident results match exactly for every case. `--allow-differences` records
numerical mismatches explicitly; it never bypasses finite, shape, dtype,
source-preservation or CUDA-route checks. Review the report when using it on a
new build. The existing geometry/precision tests cover the field's RMS/gradient
limits independently.

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --parallel 4
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_augmentations.py \
  --native-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" --level 2 \
  --start 9000 3600 3600 --size 128 --warmup 1 --reps 5 --allow-differences \
  --output docs/benchmarks/augmentations.json --update-readme
AGENTS_AGENT_MODE=1 .venv/bin/python tests/test_augment_benchmark.py -v
```

Use `--only Elastic Gamma` for a subset, and omit `--update-readme` to inspect a
report first. NumPy and the existing reader/augmentation extensions suffice;
preview rendering still uses Pillow. The new harness tests verify numerical
failure detection, exact large integer comparisons, safe recipe parsing, input
routing, and timing-row updates without modifying recipes/images.
The shared native-build loader is idempotent so combined unittest discovery
does not re-register native types. Run the complete regression suite with:

```bash
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python -m unittest discover -s tests -v
```

## CUDA geometry work reduction

**Historical optimization comparison (0fd629ebd).** These are the original
before/after measurements, not the current refresh. Current standalone results
are above; current reader and full-pipeline results are below.

The next pass after `f0f71d65b` removes repeated work in elastic construction and
resampling. It keeps the same lattice, seeds, image/field dtypes, cubic-spline
and linear interpolants, requested RMS strength, and gradient cap. Changes are in
`cpp/augment/cuda/elastic.cu` and `cpp/augment/cuda/geometry.cu`:

- Expand the outer B-spline axis once per voxel, sharing its four weights and
  index calculations across displacement components.
- Decompose output coordinates once and skip multiplication by an identity
  matrix. Specialize ordinary float32 2D/3D sampling with bounded 32-bit indices.
- Factor interior bilinear/trilinear interpolation into three/seven axis blends,
  avoiding repeated corner weights, address construction and bounds checks.
  Boundary, validity, wider-coordinate and exceptional-value paths remain available.
- Fuse elastic centering into energy/gradient collection and the final scaling
  pass, removing a full field write/read. Cap reduction partials at 1024 instead
  of 8192 for 128³ crops, and reuse neighbor tests across field components.
  Energy still uses centered values, avoiding cancellation from subtracting
  squared means. Reduction order and floating-point grouping change.

Linux x86-64, RTX 5090, Release without fast-math; PHercParis4 level 2 at the
path in the commands below. These are loader/augmentation measurements without
a model or backward pass. Warm host/OS caches, identical crops and seeds, no
changes to batch size or transform settings. Baseline extensions were preserved
from `f0f71d65b` in `/tmp/vc-work-reduction-before` before rebuilding.

| GPU-ready loader | Mean batch ms | p50 ms | p95 ms | Crops/s |
| --- | ---: | ---: | ---: | ---: |
| Before | 4.695 | 4.342 | 5.059 | 851.65 |
| After | 2.432 | 2.082 | 2.904 | 1643.53 |

Three timed epochs of 64 batches × four 128³ crops, after a validation/warmup
epoch: **93.0% higher loader throughput**. Epoch startup and disposal remain
included; the JSON contains all samples and min/max. Run-to-run scheduling and
desktop GPU activity contribute jitter; these are not confidence intervals.

| Resident elastic, random float32 128³ | Mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| Before | 0.858 | 0.837 | 0.988 |
| After | 0.243 | 0.231 | 0.346 |

200 synchronous calls per case after 20 warmups, rotating case order:
**3.53× faster mean elastic time**. Flip kernels are unchanged; their short-call
wall times are dominated by host overhead and jitter.

The separate 12-transform heavy pipeline uses three real CT crops × three
repetitions, one warmup, host outputs and cleared CUDA chunk caches:

| Route | Build | Augment mean ms | p50 ms | p95 ms | Total mean ms |
| --- | --- | ---: | ---: | ---: | ---: |
| CUDA | Before | 2.222 | 2.174 | 2.465 | 3.929 |
| CUDA | After | 1.438 | 1.368 | 1.749 | 3.136 |
| Hybrid | Before | 2.306 | 2.241 | 2.536 | 3.544 |
| Hybrid | After | 1.337 | 1.330 | 1.391 | 2.504 |

The before (`docs/benchmarks/work-reduction-ct-before.json`) and
after (`docs/benchmarks/work-reduction-ct-after.json`) reports include CPU
controls, stage distributions, input coordinates and raw samples. Image
normalization and reader code are unchanged. CPU/CUDA maximum absolute output
difference on these crops was 9.54e-7; hybrid and CUDA outputs matched exactly.

Separate Nsight Systems profiles cover 32 crops (four warmup and four measured
batches). Total GPU kernel time fell from **30.016 to 10.276 ms**; these sums
exclude host work/transfers and are not end-to-end latency.

| Kernel group | Before total ms | After total ms |
| --- | ---: | ---: |
| Resampling | 10.072 | 1.650 |
| B-spline expansion (all axes) | 9.243 | 1.505 |
| Field energy/gradient | 2.981 | 0.478 |
| Final energy/gradient reduction | 0.943 | 0.154 |
| Median filter | 1.799 | 1.803 |
| Integer reflection | 1.322 | 1.325 |

Median filtering now accounts for 17.5% of kernel time, resampling 16.1%, and
reflection 12.9%. See the full [before](../docs/benchmarks/work-reduction-profile-before.csv)
and [after](../docs/benchmarks/work-reduction-profile-after.csv) kernel/API reports,
with before (`docs/benchmarks/work-reduction-profile-before.json`) and
after (`docs/benchmarks/work-reduction-profile-after.json`) profiling configurations.
Instrumented wall timings are excluded from the performance comparisons.

Across three random 128³ reference images, maximum change from the old CUDA
elastic output was 7.99e-6, with RMSE at most 1.59e-7. All 6,291,456 sampled label
voxels matched in this comparison; this is not a guarantee at every nearest
boundary. New plans replay exactly on the same backend. The VCA2 format and
field size are unchanged, so old VCA2 records still load, but replayed image
interpolation may round differently. This does not measure model quality.

Reproduce using existing configured builds and dependencies, from `vc_reader/`:

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --parallel 4
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr

AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_cuda_geometry.py \
  --native-dir /tmp/vc-work-reduction-before --warmup 20 --reps 200 \
  --snapshot /tmp/work-reduction-reference.npz --output /tmp/work-reduction-geometry-before.json
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_cuda_geometry.py \
  --native-dir /tmp/vc-reader-cuda --warmup 20 --reps 200 \
  --reference /tmp/work-reduction-reference.npz --allow-differences \
  --output /tmp/work-reduction-geometry-after.json

# Repeat with the preserved baseline directory and a separate output filename.
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --modes prefetch --array "$VC_BENCH_VOLUME" \
  --level 2 --batches 64 --batch-size 4 --reps 3 --output /tmp/work-reduction-loader-after.json
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_pipeline.py \
  --suite pipeline --backend both --native-dir /tmp/vc-reader-cuda \
  --array "$VC_BENCH_VOLUME" --level 2 --start 9000 3600 3600 \
  --shape 128 128 128 --crops 3 --jitter 256 --warmup 1 --reps 3 \
  --threads 8 --memory-cache-mib 0 --cuda-cache clear --augmentation heavy \
  --pipeline-read box --include-hybrid --output /tmp/work-reduction-ct-after.json

AGENTS_AGENT_MODE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/work-reduction-profile-after \
  .venv/bin/python benchmarks/bench_torch.py --native-dir /tmp/vc-reader-cuda \
  --modes prefetch --array "$VC_BENCH_VOLUME" --level 2 \
  --batches 4 --batch-size 4 --reps 1 --output /tmp/work-reduction-profile-after.json
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  /tmp/work-reduction-profile-after.nsys-rep > /tmp/work-reduction-profile-after.csv

AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python tests/test_precision.py -v
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python tests/test_torch.py -v
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  .venv/bin/python tests/test_precision.py \
  PrecisionTests.test_elastic_field_strength_and_gradient \
  PrecisionTests.test_geometry_multilinear_and_boundaries -v
```

Nine precision tests pass on both CPU-only and CUDA builds, and all 18 loader
tests pass with CUDA (CPU-only: two pass, 16 skip). Compute Sanitizer reports zero
memory errors for the two added geometry/field tests. The CPU-only check used
the existing augmentation-only `/tmp/vc-volume-cpu` build with the CPU reader
extension from `build/cp314-cp314-linux_x86_64`, staged together in
`/tmp/vc-work-reduction-cpu` for `VC_READER_NATIVE_DIR`. New checks cover
elastic mean/RMS/gradient limits, singleton/non-cubic 2D/3D grids, noninteger
knot spacing, capped reductions, multichannel interpolation against analytic
multilinear functions, boundary fills, validity masks, int64 labels and replay.
macOS/arm64 and other GPU architectures were not measured. The pre-existing
intermittent Python 3.14 semaphore cleanup warning also occurred in this loader run.

## CUDA geometry optimizations

**Historical optimization comparison (f0f71d65b).** These are the original
before/after measurements, not the current refresh. Current standalone results
are above; current reader and full-pipeline results are below.

The precision pass builds on exact integer flips and float32 elastic construction.
Ordinary crop geometry, resident fields, CPU image workspaces and new replay
fields now use float32. Wider cases and replay-format changes are explained in
the [precision policy](../docs/precision.md). Nearest resizing uses shared integer
indices to resolve half-voxel ties consistently on CPU and CUDA.

Measurements use an RTX 5090, Linux x86-64, Release builds without fast-math,
and the PHercParis4 volume below at level 2. The loader chain is normalization,
elastic, median, flip and gamma; the separate heavy pipeline has 12 transforms.
Loader throughput includes reading, uploads, casts, augmentation and collation,
with no model/backward work. This is not a training-throughput measurement.

Reproduce with existing configured builds and dependencies. The saved baseline
was the build at `5d2eda6f9` (already containing integer flips and float32 elastic
construction). `--native-dir` selects its preserved extensions; the output JSON
records extension hashes, compiler configuration, input and raw measurements.

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --parallel 4
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr

# Resident float32 random 128³ image, 20 warmups and 200 timed calls per case.
# Run against the baseline first to create the reference snapshot.
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_cuda_geometry.py \
  --native-dir /tmp/vc-precision-before --warmup 20 --reps 200 \
  --snapshot /tmp/geometry-reference.npz --output /tmp/geometry-before.json
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_cuda_geometry.py \
  --native-dir /tmp/vc-reader-cuda --warmup 20 --reps 200 \
  --reference /tmp/geometry-reference.npz --allow-differences \
  --output /tmp/geometry-after.json

# 3 timed epochs of 64 batches × four 128³ crops, after a validation/warmup epoch.
# Repeat with --native-dir /tmp/vc-precision-before for the baseline.
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --modes prefetch \
  --array "$VC_BENCH_VOLUME" --level 2 --batches 64 --batch-size 4 --reps 3 \
  --output /tmp/precision-loader.json

# Synchronous full pipeline, 3 crops × 3 repetitions, one warmup.
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_pipeline.py \
  --suite pipeline --backend both --native-dir /tmp/vc-reader-cuda \
  --array "$VC_BENCH_VOLUME" --level 2 --start 9000 3600 3600 \
  --shape 128 128 128 --crops 3 --jitter 256 --warmup 1 --reps 3 \
  --threads 8 --memory-cache-mib 0 --cuda-cache clear \
  --augmentation heavy --pipeline-read box --include-hybrid \
  --output /tmp/precision-ct.json

# Separate profile; exclude this run from timing comparisons.
AGENTS_AGENT_MODE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/precision-profile \
  .venv/bin/python benchmarks/bench_torch.py --native-dir /tmp/vc-reader-cuda \
  --modes prefetch --array "$VC_BENCH_VOLUME" --level 2 \
  --batches 4 --batch-size 4 --reps 1 --output /tmp/precision-profile.json
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  /tmp/precision-profile.nsys-rep > /tmp/precision-profile.csv
```

### Measured results

GPU-ready batch latency and epoch throughput (192 timed batches per build):

| Build | Mean batch ms | p50 ms | p95 ms | Crops/s |
| --- | ---: | ---: | ---: | ---: |
| Original batch pipeline | 7.622 | 7.261 | 8.352 | 524.64 |
| Before this precision pass | 5.627 | 5.260 | 6.114 | 710.47 |
| Float32 execution and fields | 4.694 | 4.360 | 4.991 | 851.88 |

This precision pass raised loader throughput by 19.9%; the full optimization
sequence raised it by 62.4%. These are separate runs on the same machine/input,
not confidence intervals. The final build retains overflow/range handling.

Resident microbenchmarks (200 calls per case, 20 warmups, synchronous wall time):

| Operation | Build | Mean ms | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: |
| flip_x | Before this pass | 0.097 | 0.080 | 0.215 |
| flip_all | Before this pass | 0.085 | 0.075 | 0.148 |
| elastic | Before this pass | 1.099 | 1.071 | 1.271 |
| flip_x | After | 0.099 | 0.099 | 0.105 |
| flip_all | After | 0.090 | 0.078 | 0.103 |
| elastic | After | 0.844 | 0.838 | 0.854 |

Elastic mean time fell 23.2% in this pass. Flip timings are dominated by short-call
host overhead and jitter; the integer flip kernel was already present in this
baseline. Original geometry timings (`docs/benchmarks/geometry-original.json`),
pre-pass timings (`docs/benchmarks/geometry-before-precision.json`), and
final timings/numerical comparisons (`docs/benchmarks/geometry-float32.json`)
include all raw samples and min/max values.

The separate heavy pipeline uses three CT crops × three timed repetitions per
backend, host outputs, and one warmup per backend. Augmentation excludes
normalization and reading; total includes reading, normalization and download:

| Route | Build | Augment mean ms | p50 ms | p95 ms | Total mean ms |
| --- | --- | ---: | ---: | ---: | ---: |
| cpu | Before | 171.738 | 172.064 | 172.523 | 177.630 |
| cpu | After | 148.328 | 148.716 | 149.640 | 151.708 |
| cuda | Before | 2.759 | 2.730 | 2.888 | 4.521 |
| cuda | After | 2.222 | 2.187 | 2.422 | 4.016 |
| hybrid | Before | 2.855 | 2.704 | 3.435 | 4.046 |
| hybrid | After | 2.153 | 2.144 | 2.202 | 3.496 |

CPU normalization also fell from 5.392 to 2.823 ms mean. Full stage distributions,
commands and raw samples are in the before (`docs/benchmarks/precision-ct-before.json`)
and after (`docs/benchmarks/precision-ct-after.json`) reports. Final CPU/CUDA
maximum absolute difference across these three crops was 1.014e-6; hybrid and
CUDA outputs matched exactly. This comparison is between current backends,
not a claim of unchanged outputs from the old implementation.

### Remaining GPU costs

A separate Nsight Systems trace covered 32 crops (four warmup and four measured
batches of four). The kernel-time sum was 29.390 ms, versus 36.398 ms
before this precision pass and 49.847 ms before the whole optimization sequence.
These sums exclude host work and transfers and are not end-to-end latency.

| Kernel group | Final GPU kernel time |
| --- | ---: |
| Elastic resampling | 34.1% |
| B-spline expansion | 29.5% |
| Field energy/gradient | 10.2% |
| Median filter | 6.1% |
| Integer reflection | 4.5% |

Geometry and field expansion remain the largest costs. Gamma fell from 2.267 ms
total to .463 ms; normalization application fell from .879 to .332 ms.
See the [final kernel/API report](../docs/benchmarks/precision-profile.csv),
[pre-pass report](../docs/benchmarks/geometry-profile.csv), and
[original report](../docs/benchmarks/original-profile.csv). The
profile configuration (`docs/benchmarks/precision-profile.json`) identifies the
build/input; its instrumented timings are excluded from performance comparisons.

Validation: seven precision tests pass on both CPU-only and CUDA builds; all 18
loader tests pass on CUDA (CPU-only: two pass, 16 GPU tests skip). Tests cover
53 transforms in both float32/float64, exact current-backend replay, derived
fields, integer resize ties, large-coordinate cancellation and extreme values.
New replay fields halve the 128³ elastic record from 50,331,945 to 25,166,121 bytes.
Older records must be regenerated. Float32 geometry can choose an adjacent label
at a nearest-neighbor boundary; the three random 128³ comparison images changed
31 label voxels out of 6,291,456 and had maximum image difference 7.451e-6.

Some Python 3.14 multiprocessing runs emitted a shutdown resource-tracker warning
about an already-removed semaphore; tests and benchmark validation completed.
This precision pass does not resolve that intermittent cleanup warning. These
measurements cover Linux/RTX 5090; macOS/arm64 were not tested.

## Reader and pipeline benchmarks

[`bench_pipeline.py`](bench_pipeline.py) has two suites:

- `--suite readers` (default): reading/sampling only, comparing vc_reader CPU,
  vc_reader CUDA, TensorStore, and zarr-python. Workloads are batches of ML crops,
  tifxyz surface tiles, axis-aligned slabs, and rotated 3-D boxes. Flat oblique
  planes remain available for synthetic smoke checks.
- `--suite pipeline`: vc_reader CPU versus vc_reader CUDA only, from affine
  fetching/sampling through normalization and augmentation. The default heavy
  chain contains 12 transforms, including elastic deformation and median filtering.
  `--pipeline-read box --include-hybrid` compares axis-aligned reads and adds
  CPU reading followed by CUDA normalization/augmentation.

Both accept local, HTTP, or S3 **3-D uint8 Zarr v2 arrays**. Reader comparisons
need the `bench` extra (TensorStore, zarr-python, SciPy, tifffile, and remote-store support).
The full pipeline needs only the package's NumPy dependency. There are no
TensorStore or zarr-python stages in the full-pipeline comparison.

```bash
# Small synthetic smoke check for all CPU readers.
python benchmarks/bench_pipeline.py --suite readers --shape 16 16 16 \
  --plane-size 32 32 --slab-depth 4 --batch-size 2 --crops 2 --reps 2

# Reader comparison using the bundled PHercParis4 surface and its matching volume.
python benchmarks/bench_pipeline.py --suite readers --backend both \
  --array /path/to/PHercParis4-volume.zarr --level 2 --shape 128 128 128 \
  --workloads ml_batch surface slab rotated_box --surface-size 96 512 \
  --plane-size 512 512 --batch-size 8 --output /tmp/readers.json

# Heavy training pipeline: vc_reader CPU versus CUDA.
python benchmarks/bench_pipeline.py --suite pipeline --backend both \
  --array /path/to/volume.zarr --level 2 --augmentation heavy \
  --shape 128 128 128 --output /tmp/pipeline.json
```

Use matching Release reader/augmentation extensions; CUDA requires a build with
`VC_READER_ENABLE_CUDA=ON`. `--native-dir /path/to/cmake-build` selects both
extensions from an existing build without installing them. `--backend cpu`,
`cuda`, or `both` selects vc_reader modes; `--baselines tensorstore zarr` selects
the reader baselines, and `--baselines` without names omits them. `--workloads`
selects reader workloads. `--start Z Y X`, `--jitter`, `--angle`, `--threads`,
and cache budgets control the requests. With no `--array`, the script generates
a temporary seeded input; use real CT data for performance decisions.

Reports include mean/p50/p95/min/max latency, throughput, speedups, and raw
measurements, plus geometry, seeds, versions, platform, extension hashes, and
cache counters. Validation and warmup are excluded from timing. Direct reads
must match exactly; interpolated and augmented outputs must agree within
`atol=5e-5, rtol=1e-5`. Requested baselines are never silently skipped.

TensorStore and zarr-python fetch enclosing regions and use the same
[SciPy trilinear sampler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html)
for surfaces, oblique planes, and rotated boxes. ROI planning and interpolation
are timed; float32 coordinates follow vc_reader's sampling contract. Affine
coordinate construction is timed; tifxyz coordinate grids are prepared before
timing for every backend, as described below. vc_reader
samples directly from chunks. ML requests use vc_reader's `read_many`, concurrent
TensorStore futures, or zarr-python thread-pool reads. Batch outputs are lists of
crops; no backend pays for a final stack. Baseline planning/read/sampling timings
are also recorded in JSON; vc_reader's fused read/sample time cannot be split.

The `cuda` mode uses device chunks with no automatic CPU sampling fallback.
The optional `hybrid` pipeline mode explicitly uses CPU reading/sampling followed
by an upload and CUDA normalization/augmentation. Host output is
the default, including the final download. `--cuda-output device` instead measures
a GPU-consumer endpoint. Calls complete before timing stops. Host/OS caches are
not flushed; `--memory-cache-mib 0` disables vc_reader/TensorStore decoded caches
(zarr-python has no additional decoded cache). `--cuda-cache clear` drops source
chunks before every request, including uploads in the timing; `warm` reuses them
subject to its budget. These are sequential request timings, not cold-storage or
overlapped training-loader measurements.

The full pipeline reports separate normalization and augmentation stages.
`--augmentation light` selects the original four-transform smoke workload;
`heavy` selects the training workload described in the results below. The
normalization defaults to min-max and can be changed with `--normalization`.
`--pipeline-read box` selects integer axis-aligned crops instead of affine
sampling (ignoring `--angle` and flooring jittered origins). `--include-hybrid`
adds the third route when `--backend` is `both` or `cuda`: the public CUDA reader's
`source="cpu"` path uploads the uint8 box crop and casts it to float32 on the GPU.
For affine sampling, the hybrid route instead samples float32 on CPU and uploads
that result. Transfer time is included in the fetching/sampling stage.

## Reusable tifxyz surface

[`fixtures/fiber_132487`](fixtures/fiber_132487) is an unchanged copy from
`/mnt/raid_nvme/paris4_sheet_strips_v2/all_strips/fiber_132487`. It is the longest
of the 118,481 strips marked family `H` in `fibers.tsv`: **183.244 mm** by that
index. The fixture includes `x.tif`, `y.tif`, `z.tif`, `mask.tif`, `meta.json`,
`trace.json`, and `expected_error.tif` (about 2 MB total).
[`provenance.json`](fixtures/fiber_132487/provenance.json) records the source,
selection, original index entry, and SHA-256 hashes. No source files were changed.

The stored coordinate grid is **25 × 4800**, with UV scale `(0.25, 0.2)`.
XYZ values are in **9.6 µm voxels**, matching **level 2** of the supplied volume;
they must not be divided by four again. `--surface-coordinate-scale` defaults
to 1; set it explicitly when mapping to another resolution (4 for level 0 of
this volume). `--surface /path/to/tifxyz` selects another surface with matching
volume coordinates. The benchmark does not register surfaces to other volumes.

`--workloads surface` prepares **96 × 512 native-UV-pixel tiles** along the
strip. It uses `meta.json` scale to bilinearly interpolate the coordinate TIFFs,
converts XYZ to ZYX, and selects tiles with valid mask coverage. Tile selection
is deterministic and spread across the strip; `--crops` controls the number.
It does not use `--start`, `--jitter`, or `--angle`. Tile geometry, source hashes,
and coordinate hashes are recorded in the JSON. `--surface-size H W` changes
the tile dimensions; oversized tiles or coordinates outside the volume fail.

Each request samples **one layer on the curved surface**, using vc_reader's
general `sample` API, rather than approximating it with an affine plane.
TensorStore/zarr-python fetch the enclosing voxel region and use SciPy to
sample exactly the same point grid. Geometry stays on the host before each
request; CUDA coordinate/source uploads and final output downloads are timed.
TIFF loading, coordinate-field interpolation, mask filtering, and tile selection
are excluded for every backend. This measures the volume-reading portion of
surface rendering; normal-offset layers, projections, and image writing are
not part of the workload. The `plane` workload remains available separately
and is still included in the default synthetic smoke run.

## Reading/sampling timing comparisons

Refreshed 2026-10-17 at `0fd629ebd` on the supplied PHercParis4 volume:

```text
/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
```

These results use **level 2**, shape `(18946, 8174, 8174)`, uint8, 128³ chunks,
and the **local uncompressed cache** (`compressor: null`). They measure warm
filesystem-cache access, not remote S3 transfers or compressed-data decoding.
The machine has an **Intel Core Ultra 7 270K Plus** and an **NVIDIA GeForce RTX
5090** (driver 595.84), running Linux x86-64 and Python 3.14. The vc_reader
extensions are a **Release** CUDA build; the CPU column uses only its CPU path.
TensorStore is 0.1.85, zarr-python 3.4.1, and SciPy 1.18.1.

The reader/concurrency setting is **8**. TensorStore uses 8 file-I/O/data-copy
workers; zarr-python uses an 8-worker request pool and async/threading limits of
8. Decoded host caches are disabled for vc_reader and TensorStore; no decoded
cache is added to zarr-python. **CUDA source chunks are cleared before every
request, and final results are downloaded to host**, so upload and download
costs are included. Raw ML/slab outputs are uint8; interpolated surface/box outputs
are float32. CUDA has no CPU sampling fallback in these runs.

There are **6 requests × 10 repetitions = 60 measured requests per
workload/backend**, after validation and 2 warmup passes. Each ML request holds
8 independently positioned crops. Origins start at ZYX `(9000, 3600, 3600)` and
use up to ±256 voxels of seeded jitter for the synthetic box/slab geometry.
Rotated boxes use 10° rotations in two coordinate planes; slab orientations
cycle through all three axes. Surface requests use six real curved tiles from
the fixture above, each 96 × 512 pixels.
All workloads use the same geometry across readers. Backend order rotates each
repetition.

Timing cells are **mean / p50 / p95 milliseconds per request**, including
planning and sampling. TensorStore/zarr-python sampling includes enclosing-ROI
reads, coordinate preparation as described above, and SciPy interpolation;
those columns are not claims of built-in interpolation in either storage library.

| Workload | zarr-python | TensorStore | vc_reader CPU | vc_reader CUDA |
| --- | ---: | ---: | ---: | ---: |
| ML batch: 8 × 128³ crops | 13.914 / 13.553 / 15.472 | 4.839 / 4.867 / 5.773 | 1.153 / 1.136 / 1.316 | 4.791 / 4.734 / 5.633 |
| Curved tifxyz surface: 96 × 512 | 10.933 / 10.462 / 14.651 | 6.576 / 6.503 / 8.391 | 0.845 / 0.832 / 1.189 | 1.832 / 1.796 / 2.175 |
| Axis-aligned slab: 16 × 512² | 6.177 / 6.134 / 7.157 | 2.789 / 2.730 / 3.202 | 0.491 / 0.471 / 0.806 | 1.987 / 2.104 / 2.488 |
| Rotated box: 128³ | 83.160 / 69.232 / 99.469 | 73.106 / 65.362 / 97.817 | 14.776 / 14.870 / 18.662 | 1.670 / 1.592 / 1.927 |

Throughput is **requests/s** over the sum of measured latencies; one ML request
is an entire 8-crop batch. JSON also reports crops/s, Mvoxels/s, output GiB/s,
stage times, min/max, cache counters, and every individual measurement.

| Workload | zarr-python | TensorStore | vc_reader CPU | vc_reader CUDA |
| --- | ---: | ---: | ---: | ---: |
| ML batch: 8 × 128³ crops | 71.9 | 206.7 | 867.6 | 208.7 |
| Curved tifxyz surface: 96 × 512 | 91.5 | 152.1 | 1183.9 | 545.8 |
| Axis-aligned slab: 16 × 512² | 161.9 | 358.6 | 2037.6 | 503.2 |
| Rotated box: 128³ | 12.0 | 13.7 | 67.7 | 598.9 |

Speedups divide baseline p50 by the corresponding vc_reader p50. Values below
1× mean vc_reader is slower for that comparison.

| Workload | CPU vs zarr | CPU vs TensorStore | CUDA vs zarr | CUDA vs TensorStore | CUDA vs CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| ML batch: 8 × 128³ crops | 11.94× | 4.29× | 2.86× | 1.03× | 0.24× |
| Curved tifxyz surface: 96 × 512 | 12.58× | 7.82× | 5.83× | 3.62× | 0.46× |
| Axis-aligned slab: 16 × 512² | 13.04× | 5.80× | 2.92× | 1.30× | 0.22× |
| Rotated box: 128³ | 4.66× | 4.40× | 43.48× | 41.05× | 9.34× |

CPU is the faster vc_reader path for direct ML batches, axis-aligned slabs,
and the real surface tiles in this host-output workload. CUDA benefits the
rotated-box sampling most.
Direct reads matched exactly; the maximum discrepancy across sampled outputs
was 0.0000458 intensity units. These results describe this dataset, geometry,
and cache policy; other chunking, codecs, storage, and locality can change them.

Full reading/sampling report (`docs/benchmarks/readers.json`).

## Single-crop latency (12 augmentations, CPU output)

Axis-aligned crops are the default workload; rotated sampling is a separate
special case below. Both benchmarks measure input preparation for one crop at
a time and return the result to CPU memory, with no model forward or backward
pass. The PyTorch loader benchmark uses the same axis-aligned reads,
normalization and 12 augmentations, but measures batches with warm reader caches
and GPU outputs.

These comparisons use the same volume, level, hardware, Release build, thread
setting, cache policy, and 60-call measurement protocol above. Each request
fetches one 128³ crop, applies min-max normalization, then executes this
**12-transform heavy augmentation chain** in order:

```python
A.Elastic(sigma=12, magnitude=2, border=True)
A.MedianFilter(3)  # no quantization
A.Flip(axes="x", p_axis=1)
A.GaussianBlur(0.7)
A.BrightnessMultiply(1.1)
A.BrightnessAdd(0.03)
A.Contrast(1.15)
A.Gamma(0.9)
A.LocalContrast(12, loc=0.5, factor=1.1)
A.LocalSmoothing(12, loc=0.5, strength=0.3, sigma=1)
A.Sharpen(0.1, p_clamp=0)
A.LowResolution(0.85)
```

The same fixed recipe and per-crop seed are used across routes, with all
transforms active. Normalization and augmentation are separate calls for stage
measurement; augmentation uses `compact=False`.
Timing includes the fresh source upload and final result download for CUDA.
The listed chain is a reproducible heavy workload, not a prescribed training recipe.

### Default: axis-aligned boxes

All three routes fetch the same integer axis-aligned 128³ boxes, without
interpolation during reading. The hybrid route reads uint8 crops on CPU,
uploads only the crop, casts to float32 on GPU, and executes normalization and
all augmentations on CUDA. The CUDA-chunks route fetches source chunks onto the
GPU before extracting each box. Both include the final host download.

Cells are **mean / p50 / p95 milliseconds per crop**.

| Stage | CPU throughout | CUDA chunks + CUDA pipeline | CPU read + CUDA pipeline |
| --- | ---: | ---: | ---: |
| Fetching + conversion/upload | 0.545 / 0.528 / 0.844 | 1.077 / 1.103 / 1.350 | 0.540 / 0.573 / 0.745 |
| Normalization | 2.783 / 2.776 / 2.870 | 0.126 / 0.126 / 0.134 | 0.096 / 0.095 / 0.120 |
| 12 augmentations | 148.213 / 148.094 / 150.146 | 1.395 / 1.371 / 1.630 | 1.363 / 1.336 / 1.508 |
| Final host download | 0.000 / 0.000 / 0.000 | 0.554 / 0.549 / 0.600 | 0.537 / 0.523 / 0.585 |
| End-to-end | 151.542 / 151.408 / 153.622 | 3.152 / 3.169 / 3.520 | 2.537 / 2.548 / 2.933 |

| Metric | CPU throughout | CUDA chunks + CUDA pipeline | CPU read + CUDA pipeline |
| --- | ---: | ---: | ---: |
| Crops/s | 6.60 | 317.23 | 394.24 |
| Mvoxels/s | 13.84 | 665.27 | 826.79 |
| Speedup over CPU (p50) | 1.00× | 47.78× | 59.41× |

For these boxes, CPU reading followed by CUDA processing was **59.41× faster
than staying on CPU** and **1.24× faster than the CUDA-chunks route** end to end
(p50). Hybrid and CUDA-chunks outputs matched bit-for-bit for all six crops;
the maximum difference from CPU was 1.1e-06. This measures fresh uploads with
warm filesystem access; reusing GPU source chunks can change the tradeoff.
Axis-aligned hybrid report (`docs/benchmarks/pipeline-box-hybrid.json`).

### Special case: rotated-crop sampling

This comparison samples a 128³ crop with a 10° YX rotation using trilinear
interpolation. It compares CPU and CUDA reading and processing with the same
normalization, augmentation chain and measurement protocol as the axis-aligned
benchmark above.

Cells are **mean / p50 / p95 milliseconds per crop**. Speedups use p50.

| Stage | vc_reader CPU | vc_reader CUDA | CUDA speedup |
| --- | ---: | ---: | ---: |
| Fetching + sampling | 14.037 / 13.781 / 17.895 | 1.176 / 1.168 / 1.404 | 11.80× |
| Normalization | 2.682 / 2.639 / 2.953 | 0.126 / 0.123 / 0.156 | 21.48× |
| 12 augmentations | 144.051 / 144.002 / 146.046 | 1.405 / 1.353 / 1.703 | 106.43× |
| Final host download | 0.000 / 0.000 / 0.000 | 0.551 / 0.537 / 0.632 | — |
| End-to-end | 160.769 / 161.036 / 164.651 | 3.257 / 3.177 / 3.897 | 50.69× |

| Throughput | vc_reader CPU | vc_reader CUDA |
| --- | ---: | ---: |
| Crops/s | 6.22 | 307.03 |
| Mvoxels/s | 13.04 | 643.89 |

CPU/CUDA outputs passed the same tolerance check; maximum absolute difference
was 1.01e-06 after the full chain. Full pipeline report (`docs/benchmarks/pipeline.json`).
Setup, validation, warmup, cache clearing, and final output destruction are
outside both suites' timed intervals. Their JSON reports contain exact commands,
versions, origins, seeds, extension hashes, and raw samples.

Reproduce these runs using the existing Release CUDA build (adjust its path if
needed):

```bash
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
NATIVE_BUILD=/tmp/vc-reader-cuda

AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_pipeline.py --suite readers --backend both \
  --native-dir "$NATIVE_BUILD" --array "$VC_BENCH_VOLUME" --level 2 --start 9000 3600 3600 \
  --workloads ml_batch surface slab rotated_box --surface-size 96 512 \
  --shape 128 128 128 --plane-size 512 512 --slab-depth 16 --batch-size 8 \
  --crops 6 --jitter 256 --warmup 2 --reps 10 --threads 8 \
  --memory-cache-mib 0 --cuda-cache clear --output docs/benchmarks/readers.json

AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_pipeline.py --suite pipeline --backend both \
  --native-dir "$NATIVE_BUILD" --array "$VC_BENCH_VOLUME" --level 2 --start 9000 3600 3600 \
  --shape 128 128 128 --crops 6 --jitter 256 --warmup 2 --reps 10 --threads 8 \
  --memory-cache-mib 0 --cuda-cache clear --augmentation heavy \
  --output docs/benchmarks/pipeline.json

AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_pipeline.py --suite pipeline --backend both \
  --native-dir "$NATIVE_BUILD" --array "$VC_BENCH_VOLUME" --level 2 --start 9000 3600 3600 \
  --shape 128 128 128 --crops 6 --jitter 256 --warmup 2 --reps 10 --threads 8 \
  --memory-cache-mib 0 --cuda-cache clear --augmentation heavy \
  --pipeline-read box --include-hybrid \
  --output docs/benchmarks/pipeline-box-hybrid.json
```

