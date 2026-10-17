# Optional CUDA CT reader

The CUDA backend supports **3-D uint8 CT**: axis-aligned `read`/`read_many`,
integer `read_points`, explicit-point nearest/trilinear `sample`, and single or
batched nearest/trilinear `read_affine`. Outputs are owned device arrays, ready
for CUDA normalization/augmentation or a DLPack consumer such as PyTorch.

```python
import numpy as np
from vc_reader.reader import ZarrReader
from vc_reader import augment as A

pipeline = A.Pipeline([
    A.Normalize("minmax"),
    A.GaussianNoise(.01),
], backend="cuda")

with ZarrReader.open("/path/to/volume.zarr/2") as reader:
    with reader.cuda(cache_bytes=512 << 20, device=0) as gpu:
        patch = gpu.read_affine(
            (128, 128, 128), (.25, .5, .75), np.eye(3, dtype=np.float32),
            coordinate_origin=(9000, 3600, 3600),
        )
        result = pipeline({"image": patch.reshape(1, 128, 128, 128)},
                          seed=123, inplace=True)["image"]
        # No image download between sampling, normalization and augmentation.
        import torch
        tensor = torch.from_dlpack(result)  # shares CUDA storage and ownership
        # result.numpy() is an explicit host copy, if one is needed.
```

`reader.cuda()` requires a CUDA-enabled build and a visible device. The ordinary
reader API remains available without CUDA. `vc_reader.cuda.is_available()` checks
reader availability; `vc_reader.augment.cuda.is_available()` checks augmentation.
No Torch/CuPy runtime dependency is required by the backend.

## Requests and routing

The adapter methods mirror the corresponding reader operations, returning
`DeviceArray` (or a list of arrays for batches). They accept `source=`:

- `"auto"` (default) samples from cached/uploaded source bricks (see below). For
  axis-aligned reads it instead crops uint8 on CPU and uploads the crop if missing
  source bricks occupy more bytes than the raw crop output. Any request exceeding
  the cache budget uses CPU sampling/cropping followed by upload. Floating box
  conversion happens on GPU. Execution errors are propagated, never retried.
- `"chunks"` requires GPU sampling and sufficient cache budget for the request's
  brick union. It raises before source loading if that working set exceeds the
  budget. Use this to warm the cache deliberately.
- `"cpu"` explicitly samples/crops on CPU and uploads its result. The returned
  device-array contract is identical.

The routing rule compares transferred bytes; it is not a universal latency
predictor. Fresh, sparse requests can favor CPU processing even when they fit
in the device cache. Use explicit routes when the workload is known.

The private per-adapter LRU cache holds **bricks**: blocks of at most 32 voxels
per axis that evenly divide the storage chunks (128³ chunks hold 4³ bricks of
32³; chunk axes of 32 or fewer, or odd extents, are not subdivided). It is bounded
by `cache_bytes`, counting whole bricks, including padding past the volume edge.
Planning keeps only bricks the request can read: boxes use their brick overlap;
affine crops test every brick in their bounding box against the sampled
parallelepiped with a conservative separating-axis test (dilated by the
interpolation neighbourhood and float32 rounding). Oblique crops therefore load
roughly their own footprint instead of every chunk their bounding box touches.
Native planning deduplicates bricks across crop batches. Only misses are read
through the existing reader (local mmap/slab, codecs, remote/disk/memory caches)
and uploaded. Raw local chunks are copied brick by brick straight into staging
without being read whole or inserted into the host memory cache; memory-cache
hits and compressed chunks still use it.

Bricks live in fixed slots of device segments (up to 256 MiB each, allocated on
demand within the budget and freed by `clear_cache()` and `close()`). Host staging
reuses two page-locked buffers of at most 32 MiB each, or one brick per buffer if
a brick is larger, plus device staging of the same size. Each staging batch is one
`cudaMemcpyAsync` followed by a scatter kernel into the bricks' slots, on the
default stream, overlapping CPU reads of the next batch. Misses are uploaded in
order of the first crop needing them, and each crop's sampling kernel is enqueued
as soon as its bricks' transfers are, so sampling overlaps the remaining reads.
Events prevent staging reuse before a transfer finishes; errors drain queued work
and discard potentially incomplete cache entries. Public calls still finish before
returning. Buffers grow lazily, survive `clear_cache()`, and are released when the
adapter closes. Pinning has an initial allocation cost; reuse the adapter across
requests. Pinned-memory allocation errors propagate as CUDA errors. CPU
sampling/crop-upload routes are unchanged. Outputs, coordinate buffers, staging
and lookup scratch are additional memory, outside the source-cache budget. CUDA's
allocator may also retain freed memory. No GPU decompression or direct storage I/O
is claimed.

`gpu.stats` reports resident cache bytes/budget, cache hits, uploads, uploaded
bytes, and CPU/GPU route counts. Hits and `chunk_uploads` count bricks (the key
name is kept for compatibility). `gpu.clear_cache()` drops cached bricks.
Source volumes are treated as immutable; clear the GPU cache after changing data.
Closing the adapter frees its cache and scratch, but leaves returned outputs valid.
The source reader must remain open while issuing new requests.

Examples:

```python
boxes = gpu.read_many([((10, 20, 30), (64, 64, 64)),
                       ((20, 30, 40), (32, 64, 64))], dtype=np.float32)
crops = gpu.read_affine_many([
    ((64, 64, 64), (10.25, 20.5, 30.75), np.eye(3)),
    ((32, 64, 64), (11.25, 21.5, 31.75), np.eye(3)),
])
values = gpu.read_points(np.array([[10, 20, 30], [11, 21, 31]], dtype=np.int32))
interpolated = gpu.sample(np.array([[.25, .5, .75]], dtype=np.float32),
                          coordinate_origin=(9000, 3600, 3600))

# Repeated explicit requests: plan once and keep coordinates on device.
request = gpu.prepare_points(points, order=1)  # integer=True for int32/int64 gathers
values = request.read()
```

Prepared requests own a snapshot of host/device coordinates and the native chunk
plan. They reuse indices even after source eviction; subsequent reads reload only
missing source chunks. Regular planes/crops should use affine geometry instead
of storing dense coordinates. Shape-changing or heterogeneous crop batches return
nonoverlapping views into one contiguous device output allocation. Empty crops,
missing chunks, out-of-volume neighbors and fill values follow CPU semantics.
Integer gathers never convert their indices to floating point.

## Precision

The default **3-D uint8 CPU path and CUDA path use float32** geometry, weights
and interpolation. Absolute caller coordinates are split into an integer origin
and local float32 coordinates before narrowing. Supplying `coordinate_origin`
explicitly avoids constructing large absolute floating coordinates at all.
Already-rounded absolute float32 inputs cannot recover lost precision.

Affine coordinates use separate float32 multiply/add rounding on both backends;
interpolation permits ordinary CUDA FMA. Nearest uses ties-to-even in global
integer space, including odd origins. Compared with the previous float64 default,
coordinates near half-voxel ties can select a different voxel. This is a discrete
sampling change, not a small intensity perturbation. Explicit `dtype=np.float64`
on the CPU preserves the legacy reference for linear interpolation/affine reads;
CUDA accepts uint8 nearest/crop output or float32 output, with linear output
always float32. Other source dtypes keep their existing CPU implementation.

Source dimensions and explicit integer coordinate origins are limited to 2**30;
individual output-axis extents are limited to 2**24, keeping output indices exactly
representable in float32. Integer point values may use the full int32/int64 range;
outside values return fill. Nonfinite explicit coordinates produce NaN for linear
sampling and fill for nearest. Nonfinite affine definitions are rejected.

## Device arrays, pipelines and streams

`DeviceArray.shape`, `.dtype`, `.device`, `.size`, `.nbytes` and `.ndim` describe
storage. `.reshape(...)` returns a view. `.astype(np.float32)` converts uint8 on
GPU. `vc_reader.cuda.asarray(numpy_array, device=0)` uploads host arrays.
`.numpy()` explicitly downloads; implicit NumPy conversion raises.

CUDA pipelines accept a sample whose array targets are all `DeviceArray` values
on the same device. Image/scalar targets must be float32/float64, label and valid
mask dtypes are preserved, and metadata passes through. Use `.astype(np.float32)`
for raw uint8 reader output before normalization. Inputs are preserved by default
using a device copy; `inplace=True` permits modification and avoids that copy.
Always use returned arrays, since spatial operations can replace storage.
Replay/plan recording also accepts device samples. Sparse/directional/local
coordinate targets retain the existing host API and are rejected by the resident
API, which prevents hidden full-image downloads.

Calls use the selected device's default stream and **complete before returning**.
There is no arbitrary producer-stream/asynchronous-call API in this version.
DLPack exports can be consumed on any stream because production is complete.
External writes through shared DLPack views must finish before handing those
arrays back to vc_reader or calling `.numpy()`. Final destruction of exported
storage synchronizes the device to protect work queued by external consumers.
Small validation/statistics flags may transfer to host inside augmentation;
image/label volumes remain resident. GPU outputs remain valid after their reader,
cache or pipeline has been released.

CUDA adapters, prepared requests and arrays cannot be pickled or used after fork.
Construct them inside each worker; the ordinary `ZarrReader` retains its existing
fork/spawn support. Concurrent calls on one CUDA reader serialize cache mutation
and sampling. Different adapters have independent caches.

## Brick residency and resident-pipeline performance

Measured 2026-10-17 on the RTX 5090 / Core Ultra 7 270K Plus host, Release
`sm_120`, against the previous implementation (commit `61aefe673`). Workload: the
Paris4 sheet-strip trainer capture (`fiber_strip_results.md`), 32 oblique
144×128×128 crops per batch. Times are **mean / p50 / p95 ms per batch**.

| Measurement | Before | After |
|---|---:|---:|
| `read_affine_many` only, cache cleared per batch (8 batches × 5) | 81.5 / 80.0 / 91.4 | 16.0 / 15.8 / 17.0 |
| Uploaded per batch | 969 MiB | 164 MiB |
| Replay: read → `Normalize("zscore")` → augmentation, on device (16 batches) | 100.1 / 99.3 / 115.6 | 25.9 / 25.5 / 28.4 |
| — read stage | 86.3 | 16.2 |
| — normalize + augment stage | 13.0 | 8.8 |
| 14-worker trainer loader, crops/s (56 measured batches) | 137.8 | 190.2 |

Every affine output has the same SHA-256 as before, and CUDA augmentation outputs
match bit for bit across 280 input/pipeline cases (real crops, zeros, constants,
subnormal, tiny and huge magnitudes, float64). The loader is now limited by its
CPU workers (mean wait 92 ms per batch), not by the CUDA image path.

Where the time went, and what changed:

- **Bytes.** Whole-chunk residency moved ~6× more data than sampling reads;
  bricks plus the separating-axis footprint reduce uploads to the 32³-granular
  footprint. Copying raw chunks straight into staging (rather than through the
  host memory cache) removes a full extra chunk copy per miss. Brick sizes from
  16³ to 32×32×128 were measured; 32³ was fastest. The CPU brick copy is now the
  main read cost and is close to this host's memory bandwidth for 32-byte rows.
- **Sampling kernel** (7.0 → 2.7 ms per batch): 32-bit indices and shifts for
  power-of-two bricks, and register-gathered corners for cells crossing a brick
  edge. Previously nearly every warp took the generic per-corner path. The
  per-case arithmetic is unchanged: `trilinear()` within a brick and the
  generic k=3 sum otherwise.
- **Normalization** (5.3 → 2.2 ms per batch): consumer GPUs run float64
  arithmetic *and* float32↔float64 conversions at 1/64 rate. The z-score bounds
  pass runs in float32 with an integer count (exact: order-free, zeros excluded).
  Loops are versioned on the uniform scale test, which an in-loop division had
  defeated. Float32 outputs divide by multiplying with RN(1/d) plus one Markstein
  correction; results within one float64 ulp of a float32 rounding midpoint (or
  outside the normal float32 range) fall back to true division, so the float32
  result is identical. This was verified over all finite float32 inputs for
  three (center, divisor) pairs and 2.1 billion more inputs across 500
  adversarial pairs: 14.9 billion cases, zero mismatches. Float64 outputs keep
  true division.
- **Conversions.** Normal-range float32↔float64 conversions in the hot kernels
  (normalization, statistics, contrast, brightness, noise) are integer bit moves
  with round-to-nearest-even. Verified exhaustively for widening and on 60 billion
  doubles around every float32 value and rounding midpoint for narrowing.
  Zero, subnormal, infinite and NaN values use the hardware conversion.

Remaining per-crop costs are mostly host-side: each augmentation call validates
inputs with a device readback and completes before returning, so CPU work for the
next crop cannot overlap GPU work. Gaussian blur (~1 ms per batch) is the largest
remaining augmentation kernel.

Reproduce with `benchmarks/bench_cuda_strip_reads.py` (read-only, hash-compared)
and the fiber-strip replay/loader commands in `fiber_strip_results.md`.

## Upload performance

The sections below describe the previous whole-chunk implementation.

The upload comparison uses the local PHercParis4 78 keV level-2 decoded cache,
128³ uint8 chunks, a GeForce RTX 5090, and Release builds targeting `sm_120`.
The original implementation is commit `a51c9f1ba`.
`bench_cuda_uploads.py` uses 16 reader threads, no decoded memory cache, warm
filesystem pages, five warmups and 31 measured repetitions. Each call includes
planning, device allocation, CPU reading, transfer, sampling and synchronization.
GPU source-cache clearing, verification downloads and output destruction are
outside timing. Staging allocations are warmed and reused. Modes are shuffled
with seed 901; builds are measured separately without concurrent tests/builds.

Times are **mean / p50 / p95 milliseconds**. Chunk cases gather a few values but
force full source-chunk uploads to isolate cache loading; ordinary sparse gathers
should still consider `source="cpu"`. Affine cases produce full float32 crops.

| Request | Uploaded MiB | Original pageable synchronous | Pinned synchronous | Pinned async with preallocation |
|---|---:|---:|---:|---:|
| 1 chunk | 2 | 0.189 / 0.189 / 0.193 | 0.177 / 0.176 / 0.179 | 0.177 / 0.177 / 0.179 |
| 8 chunks | 16 | 1.560 / 1.561 / 1.780 | 0.956 / 0.939 / 1.176 | 0.986 / 0.937 / 1.317 |
| 32 chunks | 64 | 7.745 / 7.514 / 8.729 | 3.986 / 3.904 / 4.498 | 3.704 / 3.657 / 4.090 |
| 128 chunks | 256 | 27.202 / 27.329 / 27.963 | 16.172 / 16.122 / 17.097 | 15.558 / 15.521 / 16.746 |
| Affine 128³ | 16 | 2.281 / 2.214 / 2.806 | 1.482 / 1.506 / 1.871 | 1.441 / 1.402 / 1.909 |
| Affine 256³ | 54 | 8.396 / 8.188 / 9.467 | 5.301 / 5.241 / 5.702 | 5.210 / 5.197 / 5.650 |

Pinning accounts for most of the improvement. Async double buffering with device
allocations interleaved among copies initially took 16.725 ms p50 for 256 MiB;
allocating destinations before submission reduced this to 15.521 ms. This is a
small additional gain over pinned synchronous copies, not a further large speedup.
Resident gather p50 remains 0.015–0.022 ms and resident affine p50 0.238–1.527 ms.
Every measured output has the same SHA-256 as the original CUDA implementation;
the CPU reference is also checked with the existing float32 tolerance. Sampling,
normalization and augmentation arithmetic and RNG order are unchanged.

An independent Nsight trace of the 256 MiB case (one warmup, one repetition)
attributes 84.4% of baseline CUDA API time to synchronous `cudaMemcpy` (51.10 ms
across all setup/verification/measured calls), versus 0.028 ms total gather kernel
time. The first async trace reduced copy submission time to 1.39 ms but increased
`cudaMallocAsync` time from 6.87 to 19.67 ms, motivating preallocation. These trace
totals include untimed setup and validation; they are not the latency table.
In the final trace, `cudaMallocAsync` takes 5.35 ms. GPU copies overlap the CPU
staging intervals between an event wait returning and submission of the next
batch's copies. Initial pinning of the two 32 MiB buffers takes 18.56 ms in that
trace, reinforcing the need to reuse the adapter rather than recreate it per crop.

The existing `bench_reader_cuda.py` end-to-end workload also improves. It uses
different crop coordinates from the transfer-focused harness, three warmups,
11 measured repetitions, and the same real volume, thread count and build flags.
Fresh GPU-cache times are **mean / p50 / p95 milliseconds**:

| Request | Original | Final |
|---|---:|---:|
| Affine 128³ | 3.641 / 3.585 / 4.014 | 2.347 / 2.278 / 2.612 |
| Affine 256³ | 10.796 / 10.865 / 11.475 | 6.405 / 6.319 / 6.919 |
| Eight overlapping affine crops | 4.386 / 4.400 / 4.738 | 3.158 / 3.064 / 3.589 |
| Box 256³, forced chunk route | 7.429 / 7.386 / 8.232 | 4.201 / 4.146 / 4.579 |
| Affine 128³ → minmax → brightness → noise | 3.879 / 3.782 / 4.373 | 2.556 / 2.484 / 2.956 |

Automatic routing still chooses CPU cropping for the box case: its p50 remains
2.84–2.86 ms. The resident pipeline remains 0.43–0.44 ms. Routing is unchanged.

The results describe repeated requests through a reused adapter on this host.
They do not measure storage-cold or remote I/O, first-use pinning latency, or other
GPUs/platforms. Each active adapter can retain 64 MiB of pinned host memory
(two full chunks for oversized chunks), separate from its GPU cache budget.
CPU-only builds retain their existing path.

Reproduce after building as below; use `run_native.py` to select preserved builds
without installation. `--compare` enforces identical CUDA output hashes:

```bash
export AGENTS_AGENT_MODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=vc_reader/src
vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_cuda_uploads.py --reps 31 --warmup 5 \
  --build-dir /tmp/vc-reader-cuda --compare /tmp/vc-upload-baseline/uploads.json \
  --output vc_reader/benchmarks/results/uploads/async-final.json

vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_reader_cuda.py \
  --only affine128,affine256,box256,affine_batch8,pipeline --reps 11 --warmup 3 \
  --output vc_reader/benchmarks/results/uploads/public-final.json

DEBUGINFOD_URLS='' /usr/local/cuda/bin/nsys profile --trace=cuda \
  --sample=none --cpuctxsw=none --resolve-symbols=false \
  -o /tmp/vc-upload-final-profile \
  vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_cuda_uploads.py --only chunks128 --reps 1 --warmup 1 \
  --output /tmp/vc-upload-final-profile.json
/usr/local/cuda/bin/nsys stats --report cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_kern_sum \
  /tmp/vc-upload-final-profile.nsys-rep
```

Omit `--compare` for the initial baseline. The default input is
`/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2`;
`--array` overrides it. JSON includes raw timings, source path, command, binary
hash, environment and CMake configuration. The pinned-only experiment retains
the original single 64 MiB batch and synchronous copy loop. The async experiment
uses two 32 MiB batches. Raw reports are under ignored `benchmarks/results/uploads/`.

## Build and validate

Use the existing development environment. `VC_READER_ENABLE_CUDA=OFF` is still
the default; CPU builds require neither the CUDA toolkit nor runtime. CUDA builds
now compile both reader and augmentation support. The build dependency is
nanobind >= 2.10.2 for reusable array-API/DLPack wrappers
([nanobind changelog](https://nanobind.readthedocs.io/en/latest/changelog.html#version-2-10-1-dec-8-2025));
this implementation was tested with nanobind 3.1.0.

Commands used from the repository root, reusing already-present dependencies:

```bash
export AGENTS_AGENT_MODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=vc_reader/src
cmake -S vc_reader -B /tmp/vc-reader-cuda \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DVC_READER_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_VC_DELTA3D="$PWD/vc_reader/build/cp314-cp314-linux_x86_64/_deps/vc_delta3d-src" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/vc-reader-cuda -j 3
vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_augment.cpython-314-x86_64-linux-gnu.so \
  -m pytest -q vc_reader/tests
```

Adjust Python ABI paths, cached codec location and CUDA architecture for the
local environment. Tests that use local HTTP fixtures require localhost socket
access; CUDA tests require a GPU and skip in CPU-only builds. Performance and
memory-check commands/results are in the public-backend benchmark report.

The final upload implementation passes 1,616 tests (one skipped, two live-network
tests deselected). CPU-only sampling checks pass 81 tests with 26 CUDA skips;
the two local HTTP cases require socket access outside the sandbox. Compute
Sanitizer reports zero errors for four staging reuse/error recovery,
eviction/concurrency and resident-pipeline cases:

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=vc_reader/src OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --target-processes all \
  --error-exitcode 99 vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_augment.cpython-314-x86_64-linux-gnu.so \
  -m pytest -q vc_reader/tests/test_reader_cuda.py \
  -k 'upload_staging or cache_budget or resident_pipeline'
```
