# Benchmarks

Usage, methodology, and full results for the [package benchmark summary](../README.md#performance).
Run all commands below from the package root (`vc_reader/`).

## PyTorch CUDA batches

`bench_torch.py` measures GPU-ready batches using the same image data and
per-sample seeds across four routes:

- `serial`: ordinary `DataLoader(num_workers=0)` with per-sample CUDA processing.
- `prototype`: the earlier async native batch prototype, with fixed-shape,
  PyTorch-owned output buffers and four CPU loader workers by default.
- `batch`: the public `Pipeline.batch()` API after the CPU loader.
- `prefetch`: the public `CudaLoader` wrapper, with one batch queued ahead.

The chain is min-max normalization, elastic deformation, median filtering,
flipping, and gamma. All routes end with stacked CUDA tensors. Uploads, GPU
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
  --batches 64 --batch-size 4 --reps 3 --output /tmp/torch-loader.json

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

Measured on PHercParis4 level 2 with an RTX 5090, PyTorch 2.14.1+cu130 and a
Release CUDA build: three repetitions of 64 batches of four 128³ crops, after
one validation/warmup epoch per route. Four spawned CPU loader workers and four
native reader threads per worker were used for the batch routes. All 256 crop
hashes matched exactly across routes. The [complete report](../docs/benchmarks/torch.json)
contains the exact volume path, command, build provenance and raw timings.

| Route | Mean batch ms | p50 ms | p95 ms | Crops/s |
| --- | ---: | ---: | ---: | ---: |
| Per-sample CUDA, `num_workers=0` | 8.765 | 8.631 | 9.478 | 456.18 |
| Fixed-shape async prototype | 7.497 | 6.878 | 8.482 | 533.31 |
| `Pipeline.batch()` | 7.744 | 7.340 | 8.597 | 516.33 |
| `CudaLoader`, one batch ahead | 7.539 | 7.157 | 8.540 | 530.35 |

The public loader delivered 16.3% higher throughput than the per-sample route,
and was within 0.6% of the fixed-shape prototype. These are loader-only results,
not a measured training speedup. Public outputs use the PyTorch allocator and
support target routing and shape changes; no float precision or transform
parameters were changed. The native executor still loops over samples within
the submitted batch; this change exposes and schedules it, rather than adding
new batched kernels.

A separate Nsight Systems trace of the public prefetch route (four warmup plus
four measured batches on the same volume) attributed 46.1% of GPU kernel time
to `geometry_kernel`, 19.5% to `expand_controls`, and 10.4% to
`field_energy_gradient`. These existing geometry/elastic kernels remain the
main GPU costs. Profiling includes warmup and is excluded from the timing table.

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

Measured on the supplied PHercParis4 volume:

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
| ML batch: 8 × 128³ crops | 13.965 / 13.911 / 15.278 | 4.890 / 4.910 / 5.747 | 1.153 / 1.140 / 1.291 | 4.701 / 4.671 / 5.374 |
| Curved tifxyz surface: 96 × 512 | 10.789 / 10.606 / 13.900 | 6.730 / 6.601 / 8.192 | 0.801 / 0.775 / 1.149 | 1.775 / 1.745 / 1.986 |
| Axis-aligned slab: 16 × 512² | 6.161 / 6.197 / 7.155 | 2.678 / 2.679 / 2.985 | 0.494 / 0.470 / 0.823 | 2.010 / 2.197 / 2.642 |
| Rotated box: 128³ | 68.466 / 68.116 / 71.360 | 64.874 / 64.903 / 65.886 | 13.921 / 13.682 / 17.377 | 1.617 / 1.593 / 1.879 |

Throughput is **requests/s** over the sum of measured latencies; one ML request
is an entire 8-crop batch. JSON also reports crops/s, Mvoxels/s, output GiB/s,
stage times, min/max, cache counters, and every individual measurement.

| Workload | zarr-python | TensorStore | vc_reader CPU | vc_reader CUDA |
| --- | ---: | ---: | ---: | ---: |
| ML batch: 8 × 128³ crops | 71.6 | 204.5 | 867.4 | 212.7 |
| Curved tifxyz surface: 96 × 512 | 92.7 | 148.6 | 1248.4 | 563.3 |
| Axis-aligned slab: 16 × 512² | 162.3 | 373.4 | 2023.7 | 497.5 |
| Rotated box: 128³ | 14.6 | 15.4 | 71.8 | 618.5 |

Speedups divide baseline p50 by the corresponding vc_reader p50. Values below
1× mean vc_reader is slower for that comparison.

| Workload | CPU vs zarr | CPU vs TensorStore | CUDA vs zarr | CUDA vs TensorStore | CUDA vs CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| ML batch: 8 × 128³ crops | 12.20× | 4.31× | 2.98× | 1.05× | 0.24× |
| Curved tifxyz surface: 96 × 512 | 13.68× | 8.51× | 6.08× | 3.78× | 0.44× |
| Axis-aligned slab: 16 × 512² | 13.18× | 5.70× | 2.82× | 1.22× | 0.21× |
| Rotated box: 128³ | 4.98× | 4.74× | 42.76× | 40.74× | 8.59× |

CPU is the faster vc_reader path for direct ML batches, axis-aligned slabs,
and the real surface tiles in this host-output workload. CUDA benefits the
rotated-box sampling most.
Direct reads matched exactly; the maximum discrepancy across sampled outputs
was 0.0000458 intensity units. These results describe this dataset, geometry,
and cache policy; other chunking, codecs, storage, and locality can change them.

[Full reading/sampling report](../docs/benchmarks/readers.json).

## Full training-pipeline timing comparison

This comparison uses **vc_reader CPU versus vc_reader CUDA only**. It uses the
same volume, level, hardware, Release build, thread setting, cache policy, and
60-call measurement protocol above. Each request fetches and trilinearly samples
one 128³ crop, applies min-max normalization, then executes this **12-transform
heavy augmentation chain** in order:

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

The same fixed recipe and per-crop seed are used on both backends, with all
transforms active. Sampling uses a 10° YX rotation. Normalization and augmentation
are separate calls for stage measurement; augmentation uses `compact=False`.
Timing includes the fresh source upload and final result download for CUDA.
The listed chain is a reproducible heavy workload, not a prescribed training recipe.

Cells are **mean / p50 / p95 milliseconds per crop**. Speedups use p50.

| Stage | vc_reader CPU | vc_reader CUDA | CUDA speedup |
| --- | ---: | ---: | ---: |
| Fetching + sampling | 15.124 / 15.199 / 19.100 | 1.104 / 1.076 / 1.333 | 14.12× |
| Normalization | 5.237 / 5.230 / 5.341 | 0.138 / 0.136 / 0.142 | 38.49× |
| 12 augmentations | 169.853 / 169.915 / 171.951 | 3.150 / 3.115 / 3.372 | 54.54× |
| Final host download | 0.000 / 0.000 / 0.000 | 0.543 / 0.529 / 0.588 | — |
| End-to-end | 190.215 / 189.966 / 195.157 | 4.935 / 4.881 / 5.374 | 38.92× |

| Throughput | vc_reader CPU | vc_reader CUDA |
| --- | ---: | ---: |
| Crops/s | 5.26 | 202.63 |
| Mvoxels/s | 11.03 | 424.94 |

CPU/CUDA outputs passed the same tolerance check; maximum absolute difference
was 3.58e-7 after the full chain. [Full pipeline report](../docs/benchmarks/pipeline.json).
Setup, validation, warmup, cache clearing, and final output destruction are
outside both suites' timed intervals. Their JSON reports contain exact commands,
versions, origins, seeds, extension hashes, and raw samples.

### Axis-aligned boxes: CPU read followed by CUDA augmentation

This additional run uses the same volume, level 2, Release build, 128³ shape,
12 augmentations, cache policy, and 60 calls per mode. All three routes fetch
the same integer axis-aligned boxes; these timings are a separate comparison
from the rotated crops above. The hybrid route reads uint8 crops on CPU,
uploads only the crop, casts to float32 on GPU, and executes normalization and
all augmentations on CUDA. The CUDA-chunks route fetches source chunks onto the
GPU before extracting each box. Both include the final host download.

Cells are **mean / p50 / p95 milliseconds per crop**.

| Stage | CPU throughout | CUDA chunks + CUDA pipeline | CPU read + CUDA pipeline |
| --- | ---: | ---: | ---: |
| Fetching + conversion/upload | 0.532 / 0.485 / 0.791 | 1.010 / 0.990 / 1.270 | 0.552 / 0.541 / 0.780 |
| Normalization | 5.423 / 5.389 / 5.787 | 0.139 / 0.137 / 0.146 | 0.119 / 0.113 / 0.142 |
| 12 augmentations | 167.493 / 167.573 / 169.183 | 3.165 / 3.127 / 3.383 | 3.164 / 3.099 / 3.503 |
| Final host download | 0.000 / 0.000 / 0.000 | 0.568 / 0.557 / 0.675 | 0.558 / 0.546 / 0.668 |
| End-to-end | 173.448 / 173.431 / 175.089 | 4.881 / 4.812 / 5.312 | 4.393 / 4.371 / 4.946 |

| Metric | CPU throughout | CUDA chunks + CUDA pipeline | CPU read + CUDA pipeline |
| --- | ---: | ---: | ---: |
| Crops/s | 5.77 | 204.86 | 227.66 |
| Mvoxels/s | 12.09 | 429.62 | 477.43 |
| Speedup over CPU (p50) | 1.00× | 36.04× | 39.68× |

For these boxes, CPU reading followed by CUDA processing was **39.68× faster
than staying on CPU** and **1.10× faster than the CUDA-chunks route** end to end
(p50). Hybrid and CUDA-chunks outputs matched bit-for-bit for all six crops;
the maximum difference from CPU was 3.58e-7. This measures fresh uploads with
warm filesystem access; reusing GPU source chunks can change the tradeoff.
[Axis-aligned hybrid report](../docs/benchmarks/pipeline-box-hybrid.json).

Reproduce these runs using the existing Release CUDA build (adjust its path if
needed):

```bash
VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
NATIVE_BUILD=/tmp/vc-reader-cuda

.venv/bin/python benchmarks/bench_pipeline.py --suite readers --backend both \
  --native-dir "$NATIVE_BUILD" --array "$VOLUME" --level 2 --start 9000 3600 3600 \
  --workloads ml_batch surface slab rotated_box --surface-size 96 512 \
  --shape 128 128 128 --plane-size 512 512 --slab-depth 16 --batch-size 8 \
  --crops 6 --jitter 256 --warmup 2 --reps 10 --threads 8 \
  --memory-cache-mib 0 --cuda-cache clear --output docs/benchmarks/readers.json

.venv/bin/python benchmarks/bench_pipeline.py --suite pipeline --backend both \
  --native-dir "$NATIVE_BUILD" --array "$VOLUME" --level 2 --start 9000 3600 3600 \
  --shape 128 128 128 --crops 6 --jitter 256 --warmup 2 --reps 10 --threads 8 \
  --memory-cache-mib 0 --cuda-cache clear --augmentation heavy \
  --output docs/benchmarks/pipeline.json

.venv/bin/python benchmarks/bench_pipeline.py --suite pipeline --backend both \
  --native-dir "$NATIVE_BUILD" --array "$VOLUME" --level 2 --start 9000 3600 3600 \
  --shape 128 128 128 --crops 6 --jitter 256 --warmup 2 --reps 10 --threads 8 \
  --memory-cache-mib 0 --cuda-cache clear --augmentation heavy \
  --pipeline-read box --include-hybrid \
  --output docs/benchmarks/pipeline-box-hybrid.json
```

