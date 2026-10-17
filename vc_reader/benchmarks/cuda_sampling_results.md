# Float32 CUDA CT sampling feasibility

Measured 2026-10-16 on real PHercParis4 CT. **CUDA sampling is worth pursuing
for oriented crops and repeated sampling from resident source chunks.** With
fresh warm chunk reads and uploads, 128³/256³ trilinear crops were 4.5×/7.3×
faster than the float32 CPU candidate. Random samples over a large region need
source reuse: uploading 1.42 GiB every call is substantially slower than CPU.

The [follow-up experiments](cuda_read_paths_results.md) cover integer gathers,
axis-aligned crops and batched affine sampling, including CPU crop upload as
an alternative to uploading whole source chunks.

This is a feasibility experiment, not a public CUDA reader backend or a complete
sampling → normalization → augmentation benchmark. GPU-output modes leave the
sample buffer on device, ready for such a chain; downstream operations are not
included. The existing public sampler remains an accuracy reference. Private
native float32 entries share its chunk planner, cache resolution and interpolation
code, rather than copying the CPU sampler into the benchmark.

## Precision and geometry

Both candidates use float32 coordinates, weights, source-value conversions and
accumulators. Global origins are integers; float32 coordinates are relative to
them. The GPU has no float64 sampling path. Float64 appears only in reference
runs and one-time conversion of the benchmark's original coordinate definitions.
Already-local float32 inputs need no such conversion. Converting absolute points
to float32 before rebasing would lose precision that rebasing cannot recover.

CUDA uses ordinary FMA for interpolation, without fast math. Affine coordinate
multiply/add operations use separate float32 rounding to match this CPU build;
otherwise contraction can move nearest-neighbor samples across half-voxel ties.
Ties-to-even accounts for the integer origin's parity.

On the chunk workloads, maximum float32 CPU versus float64-reference trilinear
error was **0.006134 raw CT units**, or **0.0000241 on a 0–1 scale**. The maximum
CUDA versus float32 CPU difference was **0.00004578 raw units** (1.8e-7 scaled).
Each timed GPU result is checked at atol=3.1e-5, rtol=1e-6 in raw units; nearest
outputs must equal the float32 CPU result exactly. Repeated GPU results are exact.

Nearest sampling is discontinuous: changing to float32 can select a different
voxel near a half-integer even with rebasing. Compared with the original float64
geometry, changed outputs were 61/1,000,000 random points, 122/4,194,304 explicit
patch points, 14/1,000,000 local points, 0 for 64³/128³ affine crops, 476/16,777,216
for 256³, and 160/4,194,304 for the affine plane. Individual changed CT values can
differ substantially (maximum 79), so these are not small intensity errors.
CPU and GPU using the same float32 geometry agree exactly for nearest sampling.

## Protocol

- Host: Intel Core Ultra 7 270K Plus, 24 logical CPUs, Linux x86-64; 16 reader
  threads. GPU: RTX 5090, driver 595.84. Python 3.14.4, NumPy 2.5.4.
- CPU: CMake Release, GCC 15, `-O3 -DNDEBUG -g`, no fast math. CUDA: nvcc 13.3,
  `-O3 -lineinfo -std=c++20`, native `sm_120` code, ordinary FMA.
- Input: uint8 PHercParis4 level 2 at
  `/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2`.
  Region origin ZYX `(9000,3600,3600)`, edge 1024. Warm local mmap reads, no
  decoded memory cache. No input files are changed.
- Random/local sets contain 1,000,000 points. The patch and affine plane describe
  the same physical 16×512×512 grid, using the established `bench_sample.py`
  geometry. Their coordinate evaluation order differs slightly. Cubes are
  rotated 64³, 128³ and 256³ crops centered in the same region.
- Three warmups, eleven measured repetitions. Mode ordering is shuffled with a
  fixed seed. All timed calls complete synchronously; verification, hashing and
  output destruction are outside timing. GPU allocations are reused.
- **Read+upload** includes reading already-known source boxes/chunks into host
  buffers, H2D transfer and sampling. **Upload** starts with a prepared host
  source. **Resident** reuses GPU source data but still uploads explicit point
  coordinates every call. These three modes leave outputs on device. Verification
  downloads happen after the timer. A separate host-output mode includes D2H.
- Source discovery and input rebasing are separately recorded, excluded from
  timings for both candidates. The NumPy touched-chunk planner is deliberately
  a prototype: it takes 44–2791 ms for these workloads and is unsuitable for
  per-call use. A production chunk path must reuse/integrate native planning.
  Compact affine-box bounds require only eight corners (0.03–0.04 ms here),
  providing a useful first implementation without enumerating points.
- No claims about cold/remote I/O, compressed sources, cache eviction, pinned
  staging, transfer overlap, GPU contention with training, or full training
  throughput. The GPU also served the desktop. No macOS/arm64 measurements.

## Chunk-packed results

Times are **mean / p50 / p95 milliseconds**. GPU speedups compare float32 CPU
p50 with read+upload p50. GPU kernels are timed separately with CUDA events.
Source chunk packing uses existing `read_many` with contiguous output views;
only conservatively touched chunks are uploaded, including padded edge chunks.

| Trilinear workload | Source MiB | CPU float64 reference | CPU float32 | GPU upload | GPU read+upload | GPU resident | Speedup |
|---|---:|---|---|---|---|---|---:|
| random | 1458.00 | 12.326 / 11.928 / 14.407 | 10.835 / 10.469 / 12.985 | 84.578 / 84.353 / 86.423 | 174.231 / 173.464 / 177.941 | 1.032 / 1.008 / 1.148 | 0.06× |
| patch | 42.00 | 15.652 / 16.245 / 22.014 | 13.413 / 12.851 / 18.138 | 5.667 / 5.407 / 6.405 | 8.196 / 7.993 / 9.275 | 3.058 / 3.007 / 3.331 | 1.61× |
| local | 16.00 | 5.956 / 5.950 / 6.233 | 5.139 / 5.224 / 5.615 | 1.771 / 1.767 / 1.868 | 2.158 / 2.117 / 2.454 | 0.857 / 0.862 / 0.981 | 2.47× |
| affine64 | 8.00 | 1.295 / 1.294 / 1.509 | 1.284 / 1.324 / 1.445 | 0.511 / 0.488 / 0.620 | 0.805 / 0.729 / 1.049 | 0.068 / 0.054 / 0.130 | 1.82× |
| affine128 | 16.00 | 9.812 / 8.724 / 12.548 | 9.865 / 8.729 / 14.768 | 1.196 / 1.177 / 1.277 | 2.010 / 1.922 / 2.416 | 0.212 / 0.218 / 0.285 | 4.54× |
| affine256 | 60.00 | 61.258 / 60.254 / 66.763 | 63.711 / 63.752 / 68.933 | 4.079 / 4.056 / 4.194 | 8.690 / 8.677 / 9.606 | 0.658 / 0.659 / 0.667 | 7.35× |
| plane | 42.00 | 14.833 / 14.202 / 17.723 | 15.215 / 14.688 / 18.477 | 2.884 / 2.684 / 3.518 | 5.487 / 5.421 / 6.195 | 0.259 / 0.265 / 0.271 | 2.71× |

Float32 alone is not a consistent CPU speedup. It improves these random/local
trilinear workloads, but planning, grouping, memory traffic and integer-origin
handling dominate other cases. Nearest CPU sampling is often slower in this
candidate. These are reasons to measure against the float32 CPU implementation,
not infer a CUDA advantage from the earlier double-precision baseline.

Explicit patch coordinates cost 48 MiB per request even in float32. Representing
the same plane procedurally removes that transfer: resident GPU trilinear p50 is
3.007 ms with explicit points versus 0.265 ms with an affine grid. The kernels
alone take 0.189 and 0.203 ms respectively; coordinate transfer dominates the
explicit case. Keep geometry procedural where possible.

## Compact-box alternative

These rows read the tight enclosing source box instead of full chunks. This
reduces transferred bytes for compact crops. Timings have the same meaning as
above; host output additionally downloads the sampled result. Each row's CPU
comparison comes from that same paired run, not a different layout's timings.

| Trilinear workload | Source MiB | CPU float32 | GPU read+upload, device output | GPU read+upload, host output | GPU resident |
|---|---:|---|---|---|---|
| patch | 27.67 | 14.752 / 16.333 / 18.604 | 6.555 / 6.501 / 6.934 | 7.558 / 7.464 / 8.241 | 2.966 / 2.960 / 2.997 |
| local | 0.84 | 5.015 / 5.113 / 5.353 | 1.371 / 1.371 / 1.547 | 1.611 / 1.584 / 1.863 | 1.013 / 0.979 / 1.123 |
| affine64 | 0.43 | 1.305 / 1.318 / 1.467 | 0.269 / 0.232 / 0.399 | 0.227 / 0.225 / 0.268 | 0.053 / 0.030 / 0.115 |
| affine128 | 3.35 | 10.681 / 9.666 / 14.403 | 0.864 / 0.835 / 1.121 | 1.335 / 1.351 / 1.413 | 0.166 / 0.173 / 0.186 |
| affine256 | 26.53 | 65.575 / 67.318 / 69.600 | 3.509 / 3.435 / 3.954 | 12.697 / 12.677 / 13.116 | 0.473 / 0.474 / 0.487 |
| plane | 27.67 | 17.232 / 19.246 / 22.503 | 3.767 / 3.705 / 4.162 | 4.690 / 4.727 / 4.838 | 0.217 / 0.221 / 0.233 |

For 256³, returning output to host raises median latency from 3.435 ms to
12.677 ms in this run. Keeping sampling, normalization and augmentation on the
GPU is therefore a material design advantage. It is not yet a measured full-chain
speedup. For scattered random points, uploading whole source chunks on every
request loses badly (173.5 ms versus 10.5 ms CPU); resident reuse reverses that
comparison (1.01 ms including point upload).

## Profiling and verification

`gprofng` clock sampling of 100 repetitions each of the original random/patch
trilinear paths attributed exclusive CPU time to point interpolation callbacks
31.61%, point classification/planning 24.38%, run interpolation 11.72%,
`cell_of` 10.51%, and chunk-ID lookup 5.31%. These are sampled CPU-time fractions,
not wall-time fractions. The source extraction leaves the public reference's
nine real-data output hashes unchanged.

The float32 candidate and reference reader suites passed **213 tests**, including
new translation-invariance tests at integer origins up to three million,
odd-origin nearest ties, boundaries, missing chunks, and malformed inputs.
Compute Sanitizer reported **0 errors** for both source layouts, point and
affine geometry, nearest and trilinear interpolation. Every measured real-data
CPU/GPU pair passed validation. CUDA disassembly contains no `DADD`, `DMUL`,
`DFMA`, or `DSETP` instructions. A separate Nsight Systems trace verifies kernel
and transfer activity; its instrumented timings are excluded from the tables.

## Reproduce

Use the existing development environment and native build; no dependencies are
installed. From the repository root, build the shared native math and optional
experiment (change the CUDA architecture to match another GPU):

```bash
export AGENTS_AGENT_MODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=vc_reader/src
cmake --build vc_reader/build/cp314-cp314-linux_x86_64 --target _native -j 2
mkdir -p /tmp/vc-sampler-cuda
/usr/local/cuda/bin/nvcc -O3 -lineinfo -std=c++20 \
  -gencode arch=compute_120,code=sm_120 -shared -Xcompiler=-fPIC \
  vc_reader/benchmarks/bench_cuda_sample.cu -o /tmp/vc-sampler-cuda/libvc_sample_float32.so
vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  vc_reader/build/cp314-cp314-linux_x86_64/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_cuda_sample.py /tmp/vc-sampler-cuda/libvc_sample_float32.so \
  --allow-fma --layout chunks --outputs nearest,float32 \
  --modes cpu,cpu64,upload_device,resident_device,read_upload_device \
  --reps 11 --warmup 3 --output vc_reader/benchmarks/results/cuda-sample-float32-chunks.json
```

For the box table, use `--layout box --outputs float32
--only local,patch,affine64,affine128,affine256,plane
--modes cpu,cpu64,upload_device,resident_device,read_upload_device,read_upload`
and output `results/cuda-sample-float32-box.json`. Those JSON files (in the
ignored results directory) contain every observation, hashes, precision errors,
source/coordinate bytes, kernel times, and preparation costs.

```bash
vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  vc_reader/build/cp314-cp314-linux_x86_64/_native.cpython-314-x86_64-linux-gnu.so \
  -m pytest -q vc_reader/tests/test_sampling_float32.py vc_reader/tests/test_points.py \
  vc_reader/tests/test_affine.py vc_reader/tests/test_zarr_reader.py vc_reader/tests/test_normalization.py
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  vc_reader/.venv/bin/python vc_reader/benchmarks/run_native.py \
  vc_reader/build/cp314-cp314-linux_x86_64/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_cuda_sample.py /tmp/vc-sampler-cuda/libvc_sample_float32.so \
  --allow-fma --validate-only
```

Tests with HTTP fixtures require localhost socket access. CUDA runs require GPU
access; neither is available inside this session's restricted sandbox. The
successful runs used the existing host environment with those permissions.

The implementation is limited to 3-D uint8 CT. Before making it a public path,
provide a device-buffer/stream interface that composes with normalization and
augmentation, and integrate native source planning plus bounded chunk residency.
Start with affine crops, whose geometry and bounds are compact. A production
float32 CPU path should use the same integer-origin representation and publish
its precision contract; the private benchmark entries are not a supported API.
