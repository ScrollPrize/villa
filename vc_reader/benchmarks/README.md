# Benchmarks

Run commands from the `vc_reader/` package directory. Use matching Release
reader and transform extensions; CUDA measurements require a CUDA-enabled
build. Generated reports in `benchmarks/results/` and `docs/benchmarks/` are
local artifacts ignored by Git. Input fixtures remain versioned.

| Suite | Measures |
| --- | --- |
| [`bench_pipeline.py`](bench_pipeline.py) | CPU/CUDA reading, sampling, normalization and transforms |
| [`bench_transforms.py`](bench_transforms.py) | Individual transform recipes, including transfers and resident CUDA arrays |
| [`bench_torch.py`](bench_torch.py) | Per-sample processing, CUDA batches and loader prefetch |
| [`fiber_training/bench_fiber_trainer.py`](fiber_training/bench_fiber_trainer.py) | Full model and optimizer updates with the fiber trainer |

## Fiber trainer comparison

[`fiber_training/bench_fiber_trainer.py`](fiber_training/bench_fiber_trainer.py) runs real sampled batches
through the model and optimizer. The table uses `--steps 500 --warmup 20
--grad-steps 1 --lookahead 0 --timeline`; initialization and warmup are excluded.
Use `--backend existing` for the trainer pipeline, or `--backend vc-reader
--reader-device cpu|cuda --heading tensor --geometry vc-reader` for vc_reader.
The CUDA run also uses `--heading-stream separate`. Supply `--checkpoint`,
`--location-weights-dir`, and a fresh `--output` directory; `--help` lists the
environment and build-selection options.

| Input pipeline | Reading / sampling | Transforms | Geometry targets | Heading |
| --- | --- | --- | --- | --- |
| Original trainer | Trainer CPU code | Trainer CPU code | Trainer CPU code | Learned heading model and trainer CT fallback |
| vc_reader CPU | C++ CPU, loader workers | C++ CPU, loader workers | C++ CPU, loader workers | CPU structure tensor |
| vc_reader CUDA | CUDA | CUDA batches | CUDA in batch prefetch; worker plane targets on CPU | CUDA structure tensor |


Measured **2026-10-19** on an **Intel Core Ultra 7 270K Plus / RTX 5090**, with PyTorch **2.12.1+cu130**,
compiled BF16 model execution, ten loader workers and four trainer CPU threads.
Each row is one fresh 500-update run, in the order shown.

| Pipeline | it/s | Total runtime | MB/s | Mvox/s |
| --- | ---: | ---: | ---: | ---: |
| Original trainer | 3.65 | **137.173 s** | 726.67 | 181.67 |
| vc_reader CPU | 3.65 | **136.826 s** | 728.52 | 182.13 |
| vc_reader CUDA | 10.43 | **47.943 s** | 2079.15 | 519.79 |

Throughput counts each **144×104×104** single-channel crop once: **24,920,064,000 voxels**
and **99,680,256,000 bytes** across 16,000 float32 model-input crops. MB/s uses
decimal megabytes (10⁶ bytes); Mvox/s uses 10⁶ voxels.


## Reader and pipeline benchmarks

[`bench_pipeline.py`](bench_pipeline.py) compares reading/sampling or complete
sampling/transform pipelines. It accepts local, HTTP or S3 3-D uint8 Zarr v2
arrays. Reader comparisons include vc_reader CPU/CUDA, TensorStore and
zarr-python. The pipeline suite compares vc_reader backends and optionally
CPU reading followed by CUDA transforms.

```bash
.venv/bin/python benchmarks/bench_pipeline.py --suite readers --backend both \
  --array /path/to/volume.zarr --level 2 --shape 128 128 128 \
  --workloads ml_batch plane surface slab rotated_box --surface-size 96 512 \
  --plane-size 512 512 --batch-size 8 --min-unique-chunks 5000 \
  --xy-radius 1024 --output /tmp/readers.json

.venv/bin/python benchmarks/bench_pipeline.py --suite pipeline --backend both \
  --array /path/to/volume.zarr --level 2 --transform heavy \
  --shape 128 128 128 --output /tmp/pipeline.json
```

`--build-dir` selects an existing build. `--cuda-output device` measures a GPU
consumer endpoint; host output includes downloads. `--memory-cache-mib` and
`--cuda-cache warm|clear` control decoded/device cache reuse. Reports record
configuration, versions, hashes, validation, raw samples and latency percentiles.
Each reader workload expands to at least `--min-unique-chunks` distinct storage
chunks (default **5,000**), counted across the union of requests, including
nonzero-weight trilinear neighbors. `--crops` sets the number of geometry
templates and minimum requests, rather than fixing the final request count.
Templates sweep through Z; additional XY columns stay within `--xy-radius`
voxels of each template. Whole-chunk translations preserve within-chunk alignment.
`--start` and `--jitter` choose the initial box/plane templates. Surface requests
translate the bundled real curved geometry; translated coordinates do not claim
to follow the original physical sheet.

Before timing, every voxel in each candidate's enclosing region must be nonzero.
This excludes zero-valued volume padding, including interpolation neighbors.
Arrays with valid zero-valued data may need a different input for this benchmark.
The planner fails if it cannot meet coverage within its candidate limit; increase
`--xy-radius` or select better template locations with `--start`/`--surface`.
Reports contain exact sampled-chunk and enclosing-region unions, per-request
counts, translated geometry, and padding rejection counts. Validation, warmup,
and timed repetitions reuse the accepted request list. Planning and padding
checks are excluded from timing. Synthetic input uses nonzero seeded uint8 data.

Throughput is total output voxels or bytes divided by summed timed request
seconds: **MB/s = 10⁶ output bytes/s**, **Mvox/s = 10⁶ output voxels/s**. Integer
box/slab outputs are uint8; interpolated plane/surface/rotated outputs are float32.
These rates measure delivered outputs, not storage bytes or distinct chunks per
second. Host CUDA output includes downloads.

Reader baselines fetch enclosing regions and use the same SciPy trilinear
sampler; vc_reader samples directly from chunks. These synchronous request
measurements do not include a model or represent an overlapped trainer.

## Reader results

Measured **2026-10-19**, Intel Core Ultra 7 270K Plus / RTX 5090, Release CUDA
build, local uncompressed uint8 PHercParis4 level 2, source shape
**18,946 × 8,174 × 8,174**, storage chunks **128³**. Every accepted enclosing
region was checked to contain no zero-valued voxels. Each workload has its own
coverage union; repeated accesses do not increase the unique count.

| Workload | Requests per repetition | Unique sampled chunks | Unique enclosing-read chunks |
| --- | ---: | ---: | ---: |
| Batch of 8 × 128³ crops | 442 | 5003 | 5003 |
| Oblique plane, 512 × 512 | 386 | 5021 | 7559 |
| Curved surface, 96 × 512 | 784 | 5007 | 7837 |
| Slab, 16 × 512 × 512 | 403 | 5005 | 5005 |
| Rotated box, 128³ | 812 | 5001 | 5564 |

| Workload | Reader | Total s | Median ms/request | MB/s | Mvox/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Batch of 8 × 128³ crops | zarr-python | 32.237 | 14.428 | 1150.17 | 1150.17 |
| Batch of 8 × 128³ crops | TensorStore | 13.256 | 5.914 | 2796.94 | 2796.94 |
| Batch of 8 × 128³ crops | vc_reader CPU | 2.738 | 1.209 | 13539.44 | 13539.44 |
| Batch of 8 × 128³ crops | vc_reader CUDA | 11.738 | 5.164 | 3158.81 | 3158.81 |
| Oblique plane, 512 × 512 | zarr-python | 59.458 | 30.893 | 34.04 | 8.51 |
| Oblique plane, 512 × 512 | TensorStore | 34.737 | 17.905 | 58.26 | 14.56 |
| Oblique plane, 512 × 512 | vc_reader CPU | 3.631 | 1.837 | 557.34 | 139.34 |
| Oblique plane, 512 × 512 | vc_reader CUDA | 3.578 | 1.780 | 565.56 | 141.39 |
| Curved surface, 96 × 512 | zarr-python | 47.532 | 11.856 | 16.21 | 4.05 |
| Curved surface, 96 × 512 | TensorStore | 27.506 | 6.875 | 28.02 | 7.00 |
| Curved surface, 96 × 512 | vc_reader CPU | 3.091 | 0.767 | 249.36 | 62.34 |
| Curved surface, 96 × 512 | vc_reader CUDA | 7.342 | 1.834 | 104.98 | 26.24 |
| Slab, 16 × 512 × 512 | zarr-python | 12.356 | 6.060 | 684.01 | 684.01 |
| Slab, 16 × 512 × 512 | TensorStore | 5.968 | 2.930 | 1416.19 | 1416.19 |
| Slab, 16 × 512 × 512 | vc_reader CPU | 0.910 | 0.368 | 9283.84 | 9283.84 |
| Slab, 16 × 512 × 512 | vc_reader CUDA | 4.504 | 2.313 | 1876.41 | 1876.41 |
| Rotated box, 128³ | zarr-python | 275.765 | 67.529 | 123.50 | 30.88 |
| Rotated box, 128³ | TensorStore | 263.496 | 64.845 | 129.25 | 32.31 |
| Rotated box, 128³ | vc_reader CPU | 12.612 | 2.900 | 2700.43 | 675.11 |
| Rotated box, 128³ | vc_reader CUDA | 6.853 | 1.632 | 4969.78 | 1242.45 |

All four readers passed output validation on every accepted request. Integer
outputs match exactly; interpolation uses `atol=5e-5, rtol=1e-5`. Geometry and
padding checks, validation, one warmup pass, and cache clearing are untimed.
Total seconds sum all measured request durations across the five repetitions,
after validation and warmup; MB/s and Mvox/s use that same total. Five timed
repetitions reuse the request list, rotating reader execution order.
The OS cache remains warm; host decoded-cache budgets are zero, CUDA has a
512 MiB budget cleared before each request, and final host downloads are timed.
These are synchronous request measurements with no model or I/O overlap.

Reproduction (`VOLUME` points to the local PHercParis4 masked Zarr group):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_pipeline.py --suite readers --backend both \
  --build-dir /tmp/vc-reader-benchmark-large --array "$VOLUME" --level 2 \
  --shape 128 128 128 --start 9000 3600 3600 \
  --workloads ml_batch plane surface slab rotated_box \
  --surface-size 96 512 --plane-size 512 512 --batch-size 8 \
  --crops 6 --jitter 256 --min-unique-chunks 5000 --xy-radius 1024 \
  --reps 5 --warmup 1 --threads 8 --memory-cache-mib 0 \
  --cuda-cache clear --cuda-output host --output docs/benchmarks/readers-5000.json
```

The local JSON report records request geometry, template hashes, coverage,
padding rejections, raw timings, validation errors, versions, build settings,
and extension hashes. It is generated and ignored by Git.

## Replaying reader workloads through transforms

Use `--suite pipeline --requests-from` to run the exact accepted request lists
from a reader report through vc_reader CPU/CUDA normalization and transforms.
It restores saved shapes, origins, and axes, and verifies the surface fixture
and local-coordinate hashes before restoring each recorded translation. The
array location, level, shape, chunks, and dtype must match the reader report.
Coverage and the nonzero-region guarantee come from those unchanged requests.
TensorStore and zarr-python are not invoked by this replay path.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_pipeline.py --suite pipeline \
  --requests-from docs/benchmarks/readers-5000.json --backend cuda \
  --build-dir /tmp/vc-reader-benchmark-large \
  --workloads plane surface slab rotated_box ml_batch --min-unique-chunks 5000 \
  --transform heavy --normalization minmax --reps 1 --warmup 1 --threads 8 \
  --memory-cache-mib 0 --cuda-cache clear --cuda-output host --cuda-host-memory pinned \
  --output docs/benchmarks/pipeline-5000-batched-pinned.json
```

To measure CUDA with individual per-crop transform calls, repeat this command
with `--cuda-transforms unbatched` and
`--output docs/benchmarks/pipeline-5000-unbatched-pinned.json`. The default is
`--cuda-transforms batched`. Reading, geometry, seeds, warmup, and timing
boundaries are identical; only transform execution and output assembly differ.

Host output defaults to `--cuda-host-memory pinned`: each workload reuses pinned
buffers populated during warmup, and both CUDA transform modes copy into those
buffers with one completion wait per request. Host views are consumed before
the next request reuses their storage. `--cuda-host-memory allocating` measures
fresh host allocations instead. Buffer allocations on first use are included
in warmup; any new output
shape encountered during timing includes its allocation cost.

The heavy recipe is min-max normalization followed by Elastic, MedianFilter,
Flip, GaussianBlur, BrightnessMultiply, BrightnessAdd, Contrast, Gamma,
LocalContrast, LocalSmoothing, Sharpen, and LowResolution. Each backend uses one
combined `Pipeline` with `compact=False`. CUDA uses `Pipeline.batch()` once per
reader request: eight crops for the ML batch and one crop for other workloads.
Device batch assembly is timed, and uniform outputs download in one batch.
CPU applies transforms sequentially with `inplace=True` on fresh float32 reader
buffers. The same per-crop seeds are used on CPU and CUDA.
Planes and surfaces use 2D transforms; other crops use 3D transforms.
This measures synchronous requests, without loader workers, model execution,
or prefetch overlap.

The first full warmup pass also validates every request's shape, dtype, finite
values, and CPU/CUDA agreement (`atol=5e-5`, `rtol=1e-5`). Validation and warmup
are excluded from totals. Timing includes reads/sampling, float32 conversion,
normalization, all twelve transforms, allocation, and final CUDA host downloads.
Cache clearing and output disposal are untimed. Output is float32 for every
workload, so **MB/s = 4 × Mvox/s**. Both rates use total output divided by the
same summed request duration shown in the total-seconds column. Reports retain
the source report hash, exact geometry, coverage, per-request validation,
raw stage timings, total seconds, median/p95 latency, and throughput. Results
are saved after each completed timed repetition.

## Pipeline results

Measured **2026-10-19** on the Intel Core Ultra 7 270K Plus / RTX 5090.
Both CUDA modes use the same optimized Release transform build (`-O3 -DNDEBUG`)
and reuse pinned host output buffers. Exact request geometry, chunk coverage,
input volume, eight reader threads, disabled host decoded caches, per-request
CUDA cache clearing, and host output endpoint are unchanged.
**One timed repetition** follows **one complete warmup pass** per configuration.
Each CUDA mode was measured in a CUDA-only run. The CPU row retains the complete
matching CPU measurement; CPU transforms process crops sequentially and its
eight threads apply to reading. Batched CUDA uses `Pipeline.batch()` per request,
including device batch assembly and host downloads. Unbatched CUDA calls
`Pipeline` separately for each crop. All outputs are float32.

| Workload | Requests | Unique chunks | Backend | Total s | Median ms/request | MB/s | Mvox/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Batch, 8 × 128³ | 442 | 5003 | vc_reader CUDA batched | 4.818 | 10.892 | 6156.13 | 1539.03 |
| Batch, 8 × 128³ | 442 | 5003 | vc_reader CUDA unbatched | 5.376 | 12.247 | 5517.35 | 1379.34 |
| Batch, 8 × 128³ | 442 | 5003 | vc_reader CPU | 515.514 | 1166.110 | 57.54 | 14.38 |
| Plane, 512² | 386 | 5021 | vc_reader CUDA batched | 0.747 | 1.900 | 542.09 | 135.52 |
| Plane, 512² | 386 | 5021 | vc_reader CUDA unbatched | 0.759 | 1.907 | 532.96 | 133.24 |
| Plane, 512² | 386 | 5021 | vc_reader CPU | 8.563 | 22.122 | 47.27 | 11.82 |
| Surface, 96 × 512 | 784 | 5007 | vc_reader CUDA batched | 1.585 | 2.010 | 97.27 | 24.32 |
| Surface, 96 × 512 | 784 | 5007 | vc_reader CUDA unbatched | 1.565 | 1.963 | 98.48 | 24.62 |
| Surface, 96 × 512 | 784 | 5007 | vc_reader CPU | 3.989 | 5.018 | 38.64 | 9.66 |
| Slab, 16 × 512² | 403 | 5005 | vc_reader CUDA batched | 2.074 | 5.150 | 3260.33 | 815.08 |
| Slab, 16 × 512² | 403 | 5005 | vc_reader CUDA unbatched | 2.147 | 5.294 | 3149.51 | 787.38 |
| Slab, 16 × 512² | 403 | 5005 | vc_reader CPU | 134.556 | 312.901 | 50.25 | 12.56 |
| Rotated box, 128³ | 812 | 5001 | vc_reader CUDA batched | 1.687 | 2.030 | 4038.76 | 1009.69 |
| Rotated box, 128³ | 812 | 5001 | vc_reader CUDA unbatched | 1.666 | 1.984 | 4089.18 | 1022.30 |
| Rotated box, 128³ | 812 | 5001 | vc_reader CPU | 120.389 | 148.090 | 56.58 | 14.14 |

The CUDA-only warmup checks shape, dtype, and finite outputs for every request.
CPU/CUDA numerical agreement with pinned host outputs was checked on small samples
of all five workloads, including all slab orientations. The focused comparison
also checks batched/unbatched agreement on 32 real crops. CUDA batch correctness,
stream ownership, and benchmark tests passed: 34 tests and 93 subtests.

Raw timings, validation, stage breakdowns, exact geometry, and source-report hashes
are saved in `docs/benchmarks/pipeline-5000-batched-pinned.json` and
`docs/benchmarks/pipeline-5000-unbatched-pinned.json`; CPU values come from
`docs/benchmarks/pipeline-5000.json`. Generated JSON reports are local and ignored
by Git. Total seconds exclude validation/warmup and sum measured request times;
throughput uses that same duration. Reader-only totals span five repetitions,
so compare throughput rather than unnormalized totals between the suites.

## Focused CUDA batching comparison

[`bench_cuda_batch.py`](bench_cuda_batch.py) isolates batching using the first
four recorded ML requests (32 real crops), GPU-resident float32 input, and
the same min-max normalization plus twelve transforms. Both routes preserve
their input and download into the same reused pinned host destination. Reading,
upload, input collation, validation, and output disposal are excluded. Seeds
match between modes; execution order alternates. `--stages` also measures each
transform separately. `--crop-size 64` selects centered subcrops for scaling
comparisons; the default uses the complete 128³ crops.

Measured on the RTX 5090 with an optimized Release CUDA build (`-O3 -DNDEBUG`),
**30 repetitions after two warmup passes**, or 120 timed batches per mode.
Each cell lists mean transform time per batch, mean download time per batch,
then total timed seconds, MB/s, and Mvox/s. Totals include transforms and downloads.

| Reader | 8 × 128³ | 8 × 64³ |
| --- | ---: | ---: |
| CUDA batched | 4.195 ms transform<br>2.834 ms download<br>0.844 s total<br>9,546.9 MB/s<br>2,386.7 Mvox/s | 0.992 ms transform<br>0.347 ms download<br>0.161 s total<br>6,264.2 MB/s<br>1,566.0 Mvox/s |
| CUDA unbatched | 5.823 ms transform<br>2.768 ms download<br>1.031 s total<br>7,811.8 MB/s<br>1,952.9 Mvox/s | 2.048 ms transform<br>0.357 ms download<br>0.289 s total<br>3,487.9 MB/s<br>872.0 Mvox/s |

Transform speedups are **1.39× at 128³** and **2.06× at 64³**. Including host
downloads, speedups are **1.22×** and **1.80×**. Batched/unbatched outputs match
exactly at 128³; maximum difference at 64³ is 2.39e-7. Reports are saved in
`docs/benchmarks/cuda-batch-128.json` and `cuda-batch-64.json`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_cuda_batch.py \
  --build-dir /tmp/vc-reader-benchmark-large \
  --requests-from docs/benchmarks/readers-5000.json --requests 4 \
  --reps 30 --warmup 2 --output docs/benchmarks/cuda-batch-128.json
```

Repeat with `--crop-size 64` and a different output path for the scaling case.
These focused measurements cover a small subset, not the full 5,000-chunk workload.

## Transform benchmarks

[`bench_transforms.py`](bench_transforms.py) measures all README recipes
on CPU, CUDA with transfers, and CUDA with resident arrays. Recipes and input
construction live in [`transform_cases.py`](transform_cases.py), shared
with [`generate_transform_previews.py`](generate_transform_previews.py).
`--update-readme` changes timing rows in the package
README without altering examples or images.

```bash
VC_BENCH_VOLUME=/path/to/PHercParis4-volume.zarr
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_transforms.py \
  --build-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" --level 2 \
  --start 9000 3600 3600 --size 128 --warmup 1 --reps 5 \
  --output /tmp/transforms.json
```

Use `--only Elastic Gamma` for selected recipes. Preparation, validation and
initial resident upload/final download are outside timed resident calls;
public-call copies and validation remain included. Inspect raw measurements
when using a short repetition count. CPU/CUDA interpolation can differ slightly
at nearest-neighbor label boundaries; numerical differences are reported rather
than treated as model-quality results.

## PyTorch CUDA batches

[`bench_torch.py`](bench_torch.py) compares per-sample processing,
`Pipeline.batch()`, and `CudaLoader` with one prefetched batch. Routes use the
same images and per-sample seeds; a validation/warmup epoch precedes timing.
Reading, transfers, casting, transform and collation are included. There is
no forward/backward pass; use the fiber trainer comparison for actual training.

```bash
.venv/bin/python benchmarks/bench_torch.py \
  --build-dir /tmp/vc-reader-cuda --array /path/to/volume.zarr --level 2 \
  --batches 64 --batch-size 4 --reps 3 --output /tmp/torch-batches.json
```

Use `--help` for workload and profiling options. Correctness coverage is in `tests/test_torch.py` and
`tests/test_cuda_batch.py`.


## Reporting

Use `--help` for each harness. Keep validation/profiling separate from throughput
runs, retain JSON reports with commands and build hashes, and report the input
shape, dtype, batch size, worker/thread counts and cache policy with timings.
