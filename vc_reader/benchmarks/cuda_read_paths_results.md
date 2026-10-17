# CUDA follow-up: integer gathers, crops and batches

Measured 2026-10-16. These experiments cover the remaining promising CPU read
paths: integer `read_points`, axis-aligned `read`/`read_many`, and batched affine
sampling. Together with the [point/affine sampling experiments](cuda_sampling_results.md),
these provide the initial implementation scope for 3-D uint8 CT.

The strongest additional opportunity is **batched affine sampling**. Eight
128³ crops take 67.8 ms on the float32 CPU path versus 3.1 ms with fresh chunk
reads and uploads when overlapping, or 66.9 versus 12.0 ms when separated.
Resident chunks reduce both GPU cases to about 1.4 ms. Axis-aligned crops also
benefit from residency, but fresh requests often favor reading just the uint8
crop on CPU, uploading it, then casting/normalizing on GPU. Integer gathers
benefit primarily when their source chunks are already resident.

These are benchmark prototypes, not a public CUDA reader backend. No
normalization/augmentation chain or training step is included in the timings.

## Protocol and comparisons

Same hardware, source and Release build as the first report: RTX 5090, driver
595.84, Intel Core Ultra 7 270K Plus, 16 reader threads, GCC 15 `-O3 -DNDEBUG -g`,
nvcc 13.3 `-O3 -lineinfo -std=c++20`, native `sm_120`, default CUDA FMA and no
fast math. Python 3.14.4 and NumPy 2.5.4. BLAS/OpenMP thread counts are one.

Real uint8 CT input, 128³ source chunks:
`/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2`.
All requests lie in the stored 1024³ region starting at ZYX `(9000,3600,3600)`.
The integer requests round the established random, patch and local point sets
to int64: 1M, 4,194,304 and 1M points respectively. No integer index is converted
to float in the gather kernel.

Single boxes are aligned or offset by `(17,33,49)` from a chunk boundary.
Batches contain 32 × 32³, 16 × 64³, or 8 × 128³ crops. Origins vary by 4 voxels
per arrangement step for overlapping requests, or 192 for separated requests.
Affine grids rotate by 0.35 radians in the YX plane with a quarter-voxel origin
offset. Both CPU and GPU use integer global origins and local float32 affine
coordinates, interpolation weights and accumulation.

Each case uses three warmups and eleven measured repetitions, with modes
shuffled using seed 901. No builds, tests or other benchmarks ran concurrently.
Source reads use warm local mmap and no decoded memory cache. Every timed output
is checked; downloads, comparisons, hashing and output destruction are untimed.
All timed calls finish synchronously. Device buffers are reused; CPU output
allocation is included. CPU batches use one contiguous output with views per crop.

- **CPU** returns host output. Boxes call the real `read_many` (also the
  implementation of `read`), integer gathers call `read_points`.
- **CPU → device** adds upload of the completed result and synchronization.
- **CPU raw → device** reads uint8 crops on CPU, uploads only those crops,
  then casts to float32 on GPU. Its final result matches the other float32
  box modes. This avoids uploading four-byte float32 voxels.
- **GPU read+upload** reads the union of required chunks, packs/uploads it once,
  uploads request descriptors, and samples. Output stays on GPU.
- **GPU upload** starts with already-packed host chunks, including the same
  transfers and sampling. Its full statistics are in the raw JSON.
- **GPU resident** starts with chunks and lookup table on GPU. It still uploads
  indices or batch descriptors every call. Integer indices are int64 unless
  explicitly labeled int32. Input narrowing is checked but excluded from timing;
  the int32 result assumes the caller already provides valid int32 indices.
- **GPU all resident** also reuses integer indices. An untimed upload and gather
  restores int64 storage after other modes overwrite it with int32 indices.
  This is a best-case repeated-request measurement with warm input caches.
- **GPU serial** uses one launch per crop, sharing the same source and a single
  descriptor upload. The ordinary batch mode uses one kernel launch for the batch.

The float32 CPU affine baseline invokes the shared native sampler once per grid.
The existing public `read_affine_many` likewise loops over grids natively; it
still uses float64 internally and is not the float32 baseline. The prototype
baseline incurs a Python/native crossing per grid, so the reported speedups
include that overhead and are not pure interpolation-kernel ratios. CPU affine
latency, especially small batches, also varies substantially; see mean/p95 below.

Source discovery/rebasing, descriptor construction, lookup-table upload, device
allocation and initial GPU startup are excluded. The one-time Python source
plan took **413–1824 ms for integer points**, **0.063–0.096 ms for single
boxes**, **0.108–0.236 ms for box batches**, and **0.200–0.538 ms for affine
batches**. Integer planning needs native integration; its current NumPy unique
operation is unsuitable for per-request use. Grid plans use corners and chunk
unions, without dense coordinate enumeration. Even their cost matters beside
a 0.04 ms resident crop, so these resident numbers require reusable or efficient
native planning. CPU timings include the native reader's own request planning.

## Integer gathers

Times below and in subsequent tables are **mean / p50 / p95 milliseconds**.
The int32 and all-resident variants only change GPU inputs, not output values.

| Workload | Source MiB | CPU | CPU → device | GPU read+upload | GPU resident, int64 | GPU resident, int32 | GPU all resident |
|---|---:|---|---|---|---|---|---|
| integer_random | 1458 | 4.472 / 4.620 / 4.915 | 4.814 / 4.676 / 5.635 | 176.307 / 177.033 / 178.383 | 1.432 / 1.428 / 1.562 | 0.783 / 0.753 / 0.879 | 0.079 / 0.080 / 0.082 |
| integer_patch | 42 | 7.260 / 6.889 / 8.845 | 8.031 / 8.112 / 9.379 | 10.462 / 10.340 / 11.206 | 5.723 / 5.551 / 6.422 | 2.896 / 2.822 / 3.277 | 0.072 / 0.071 / 0.078 |
| integer_local | 16 | 2.737 / 2.693 / 3.298 | 2.799 / 2.776 / 3.285 | 2.888 / 2.802 / 3.487 | 1.363 / 1.341 / 1.731 | 0.749 / 0.694 / 0.919 | 0.027 / 0.026 / 0.031 |

Random points touch 1458 MiB of chunks to produce less than 1 MiB of output:
177.0 ms with fresh read/upload versus 4.7 ms for CPU gather plus output upload.
Resident chunks reverse the result (1.43 ms with int64 or 0.75 ms with int32
indices). For the patch, indices alone occupy 96 MiB as int64 or 48 MiB as
int32. Prefer procedural geometry for regular patches, as in the first report.

## Single axis-aligned crops

| Workload | Source MiB | CPU | CPU → device | CPU raw → device | GPU read+upload | GPU resident |
|---|---:|---|---|---|---|---|
| box128_aligned_uint8 | 2 | 0.124 / 0.122 / 0.146 | 0.221 / 0.215 / 0.263 | — | 0.251 / 0.253 / 0.298 | 0.050 / 0.034 / 0.096 |
| box128_unaligned_uint8 | 16 | 0.360 / 0.408 / 0.451 | 0.544 / 0.547 / 0.739 | — | 1.652 / 1.670 / 1.948 | 0.051 / 0.040 / 0.100 |
| box128_aligned_float32 | 2 | 0.590 / 0.523 / 0.801 | 0.990 / 0.965 / 1.117 | 0.369 / 0.356 / 0.424 | 0.558 / 0.414 / 0.988 | 0.083 / 0.060 / 0.120 |
| box128_unaligned_float32 | 16 | 0.490 / 0.458 / 0.639 | 1.118 / 1.127 / 1.324 | 0.680 / 0.662 / 0.936 | 1.888 / 1.832 / 2.248 | 0.084 / 0.076 / 0.127 |
| box256_unaligned_uint8 | 54 | 1.483 / 1.289 / 2.371 | 1.955 / 1.934 / 2.158 | — | 7.328 / 7.354 / 7.763 | 0.212 / 0.213 / 0.238 |
| box256_unaligned_float32 | 54 | 3.705 / 3.734 / 3.954 | 7.617 / 7.311 / 8.730 | 2.278 / 2.223 / 2.703 | 6.951 / 6.697 / 7.764 | 0.223 / 0.222 / 0.238 |

For unaligned 256³ float32 output, GPU source upload is only modestly faster
than CPU float32 output upload (6.70 versus 7.31 ms), and loses to CPU uint8
crop upload followed by GPU conversion (2.22 ms). The resident crop is 0.22 ms.
That argues for retaining the crop-upload route when the source is not resident.
GPU normalization could consume the uploaded uint8 crop directly in a future
implementation; this experiment measures only conversion, not normalization.

## Batched axis-aligned crops

All outputs are float32. Chunk unions are deduplicated across the batch.

| Workload | Source MiB | CPU | CPU → device | CPU raw → device | GPU read+upload | GPU resident | GPU serial, resident |
|---|---:|---|---|---|---|---|---|
| box32x32_overlap | 2 | 0.473 / 0.488 / 0.504 | 0.690 / 0.701 / 0.727 | 0.369 / 0.350 / 0.435 | 0.291 / 0.279 / 0.359 | 0.040 / 0.040 / 0.047 | 0.098 / 0.088 / 0.148 |
| box32x32_spread | 144 | 0.570 / 0.518 / 0.847 | 0.824 / 0.800 / 1.057 | 0.552 / 0.508 / 0.803 | 18.113 / 17.882 / 19.190 | 0.063 / 0.050 / 0.109 | 0.111 / 0.104 / 0.147 |
| box16x64_overlap | 4 | 1.397 / 1.536 / 1.888 | 2.265 / 2.238 / 2.607 | 1.116 / 1.105 / 1.472 | 0.738 / 0.644 / 1.024 | 0.108 / 0.097 / 0.143 | 0.134 / 0.125 / 0.168 |
| box16x64_spread | 72 | 0.864 / 0.860 / 0.923 | 1.860 / 1.760 / 2.214 | 0.846 / 0.829 / 0.958 | 9.076 / 9.199 / 9.713 | 0.120 / 0.135 / 0.147 | 0.142 / 0.153 / 0.171 |
| box8x128_overlap | 16 | 3.353 / 3.303 / 3.760 | 6.972 / 6.963 / 7.722 | 2.266 / 2.333 / 2.793 | 2.039 / 1.845 / 2.611 | 0.209 / 0.217 / 0.227 | 0.234 / 0.234 / 0.242 |
| box8x128_spread | 84 | 4.266 / 4.264 / 4.693 | 7.981 / 7.958 / 8.648 | 2.250 / 2.165 / 2.629 | 10.814 / 10.581 / 11.760 | 0.224 / 0.225 / 0.233 | 0.239 / 0.237 / 0.254 |

The 32 small overlapping crops share one 2 MiB chunk. The same batch spread
across the region requires 144 MiB: fresh read/upload increases from 0.28 to
17.88 ms, while CPU crop upload costs 0.35 and 0.51 ms. With resident sources,
both GPU batches complete in 0.04–0.05 ms. Overlap and source reuse matter far
more than launch count once transfer volume is large.

## Batched affine crops

| Workload | Source MiB | CPU float32 | CPU → device | GPU read+upload | GPU resident | GPU serial, resident |
|---|---:|---|---|---|---|---|
| affine32x32_overlap | 2 | 30.249 / 25.240 / 43.576 | 29.394 / 26.418 / 41.674 | 0.447 / 0.428 / 0.515 | 0.155 / 0.148 / 0.199 | 0.226 / 0.229 / 0.262 |
| affine32x32_spread | 144 | 34.641 / 33.749 / 45.601 | 35.628 / 35.143 / 43.713 | 17.791 / 17.667 / 18.584 | 0.193 / 0.205 / 0.230 | 0.328 / 0.314 / 0.363 |
| affine16x64_overlap | 4 | 23.086 / 19.315 / 33.047 | 22.728 / 24.299 / 28.320 | 0.976 / 0.960 / 1.112 | 0.453 / 0.456 / 0.466 | 0.605 / 0.609 / 0.613 |
| affine16x64_spread | 72 | 24.042 / 23.512 / 32.026 | 23.286 / 24.254 / 27.971 | 9.379 / 9.330 / 10.070 | 0.439 / 0.444 / 0.456 | 0.534 / 0.546 / 0.560 |
| affine8x128_overlap | 16 | 68.046 / 67.756 / 70.882 | 69.770 / 68.697 / 75.232 | 3.100 / 3.112 / 3.335 | 1.391 / 1.391 / 1.399 | 1.493 / 1.493 / 1.504 |
| affine8x128_spread | 84 | 66.443 / 66.937 / 70.799 | 70.473 / 70.443 / 75.257 | 12.030 / 11.969 / 12.898 | 1.387 / 1.385 / 1.397 | 1.469 / 1.467 / 1.480 |

Fresh read/upload wins for all six affine batches. Relative to CPU host-output
p50, speedups range from 1.9× for separated 32³ crops to 59.0× for overlapping
32³ crops; for eight 128³ crops they are 5.6× separated and 21.8× overlapping.
These comparisons include CPU planning and per-grid dispatch, not just math.

One GPU launch per batch is helpful, but not the principal source of the large
affine improvement. For eight 128³ crops, batching reduces resident latency only
from about 1.49 to 1.39 ms (overlap). For 32 small box crops it reduces 0.088 to
0.040 ms. CUDA-event medians for those box kernels are 0.065 versus 0.016 ms;
for 32 overlapping affine crops they are 0.140 versus 0.087 ms. Descriptor
transfer, synchronization and Python/ctypes dispatch explain part of the gap
between kernel duration and completed-call latency.

## Implementation implications

1. Implement float32 point/affine sampling, with procedural affine grids and
   integer global origins, as established in the first experiment. Add a batch
   interface with shared source chunks and contiguous output views.
2. Support exact integer gathers, accepting int32 where safe and int64 otherwise.
   Reuse GPU indices or generate procedural coordinates on device where possible.
3. Provide axis-aligned uint8/float32 crop kernels for resident sources, while
   retaining CPU uint8 crop upload for fresh requests with little source reuse.
4. Integrate native chunk planning and a bounded device chunk cache. Deduplicate
   uploads across requests/batches; avoid repacking and uploading the same source
   for every crop. Choose the route using residency and transferred bytes, not
   simply whether CUDA is available.
5. Expose device buffers and stream ownership so sampling, normalization and
   augmentation can compose without intermediate host copies. Reuse allocation
   and staging buffers; benchmark pinned transfers/overlap during implementation.

The experiment does not establish production dispatch thresholds or cache sizes.
It also does not measure compressed/remote/cold I/O, training contention, other
GPUs, or macOS/arm64. All input data is uint8; other stored dtypes require their
own correctness and transfer-cost coverage. Heterogeneous/empty grids are
correctness-tested, while performance batches use uniform shapes.

## Validation and profiling

All 21 real-data workloads passed every output comparison in every mode.
Integer and box outputs matched exactly, including float32 box outputs.
Maximum affine CPU/GPU difference was **0.00004578 raw CT units**, or 1.8e-7
on a 0–1 scale, passing atol=3.1e-5 plus rtol=1e-6. No lowered precision or
fast-math option was used. The shared single-point/affine kernel still passes
its earlier fixture after extraction into a helper used by the batch kernel.

The focused suite passed **7 tests**, covering chunk union deduplication,
int64 extremes, float32 origin invariance, and optional GPU contracts. Compute
Sanitizer reported **0 errors** for integer32/64, partial and missing chunks,
outside coordinates, reflected/sheared affine grids, mixed/empty grids, serial
and batched execution, and CPU result upload/GPU conversion.

A separate Nsight Systems run used integer_local, box32x32_overlap and
affine32x32_overlap with one warmup and two repetitions, plus startup validation.
It recorded 26.29 ms H2D copies, 8.56 ms D2H verification copies and 1.72 ms of
kernels across the entire trace. Affine batch kernels accounted for 67.1% of
kernel time, float32 box kernels 13.6%, and int64 gathers 12.6%. These aggregate
figures include validation and both batch/serial modes; verification downloads
are excluded from benchmark timings. The first cudaMalloc incurred 138.8 ms
of startup, also excluded. Instrumented timings are not used in the tables.

## Reproduce

Use the existing native Release build and development venv; no installation is
required. From the repository root:

```bash
export AGENTS_AGENT_MODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=vc_reader/src
/usr/local/cuda/bin/nvcc -O3 -lineinfo -std=c++20 \
  -gencode arch=compute_120,code=sm_120 -shared -Xcompiler=-fPIC \
  vc_reader/benchmarks/bench_cuda_sample.cu -o /tmp/vc-sampler-cuda/libvc_reads.so
vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  vc_reader/build/cp314-cp314-linux_x86_64/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_cuda_reads.py /tmp/vc-sampler-cuda/libvc_reads.so \
  --reps 11 --warmup 3 --threads 16 \
  --output vc_reader/benchmarks/results/cuda-read-paths.json
VC_READER_CUDA_SAMPLE_LIB=/tmp/vc-sampler-cuda/libvc_reads.so \
  vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  vc_reader/build/cp314-cp314-linux_x86_64/_native.cpython-314-x86_64-linux-gnu.so \
  -m pytest -q vc_reader/tests/test_cuda_read_benchmark.py vc_reader/tests/test_sampling_float32.py
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  vc_reader/build/cp314-cp314-linux_x86_64/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_cuda_reads.py /tmp/vc-sampler-cuda/libvc_reads.so --validate-only
/usr/local/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  -o /tmp/vc-sampler-cuda/read-paths-profile \
  vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  vc_reader/build/cp314-cp314-linux_x86_64/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_cuda_reads.py /tmp/vc-sampler-cuda/libvc_reads.so \
  --only integer_local,box32x32_overlap,affine32x32_overlap --reps 2 --warmup 1 \
  --output /tmp/vc-sampler-cuda/read-paths-profile.json
/usr/local/cuda/bin/nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  /tmp/vc-sampler-cuda/read-paths-profile.nsys-rep
```

The previous report includes the native build command. Change the CUDA
architecture flag for another GPU. GPU commands require access outside the
session's restricted sandbox. CPU-only tests skip the optional GPU contract
unless `VC_READER_CUDA_SAMPLE_LIB` is set. `--only` selects case names from the
tables. Raw JSON in the ignored results directory records all observations,
mean/p50/p95/min/max, source/output hashes, binary hashes, errors, source bytes,
coordinate bytes, planning costs and CUDA-event timings.
