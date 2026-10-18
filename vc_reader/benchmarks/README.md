# Benchmarks

Usage, methodology, and full results for the [package benchmark summary](../README.md#performance).
Run all commands below from the package root (`vc_reader/`).
Generated JSON reports under `docs/benchmarks/` are kept locally and ignored by
Git. Report paths below refer to those local files; benchmark input fixtures
remain versioned.

For metadata-only discovery, `bench_discovery.py` measures indexing and six
scroll-volume navigation requests against a direct metadata scan. It records
hashes, mean/p50/p95 latency, peak allocation, and optional cProfile output with
no array requests. `bench_paired.py` compares six real PHerc0009B CT/ink/mask
reads against independent array readers, with CPU and CUDA output options.
See [commands and measurements](../docs/scroll-reading-benchmarks.md).

For local label reading, `bench_source_readers.py` validates and times six fixed
TIFF crops against equivalent tifffile reads, using the bundled PHercParis4
geometry mask by default. See [commands, measurements, and limitations](../docs/source-reading.md#run-and-validate).

`bench_source_storage.py` adds fresh-cache and warm-crop acceptance for published
sharded Zarr v3 ink labels and image formats, with content hashes, I/O counters,
allocation measurements, and profiles. `prepare_source_formats.py` prepares
documented image derivatives of the bundled real mask. See the
[label-reader protocol and results](../docs/source-reading-benchmarks.md).

Catalog-wide results were refreshed **2026-10-17** at `46b908a31`, with the same
Release extensions for every suite. The refresh manifest (`docs/benchmarks/refresh.json`)
records report hashes and coverage. Current results are in the PyTorch, resident
geometry, augmentation catalog, and reader/pipeline sections. The two optimization
history sections are explicitly marked historical and retain their original data.
The 2D support, median, statistics, resize, Gaussian and compact-field experiments below
retain their original before/after measurements.

## 2D augmentation support (2026-10-17)

The 28 remaining image augmentations now support `(C,Y,X)` as well as
`(C,Z,Y,X)` on CPU and CUDA. Shared grid, neighborhood, field, histogram and
Fourier helpers specialize on spatial dimension at compile time. Host dispatch
selects the specialization; pixel loops do not branch on dimension. 2D uses
four interpolation corners, disk neighborhoods, two separable passes and 2D
FFT plans as appropriate. Explicit spatial tuples must match the image dimension;
omitted CLAHE/dropout sizes and dropout axes adapt to it. The tracing and
representation-conversion operations retain their specialized contracts.

CPU and CUDA now share channel-statistics requirements. The shared CPU pixel mapper
skips unused statistics and computes only bounds when moments are unnecessary.
CPU ringing filters normalize their short weight arrays once, removing a
per-output division pass. This permits small rounding differences; random
sampling and 3D neighborhood definitions are unchanged.

Validation: all 56 tests pass with CUDA; the CPU-only build passes with 20 CUDA
skips. Tests cover both float dtypes, singleton/odd dimensions, selected channels,
validity masks, large integer labels, replay, parameter validation, independent
NumPy filter/Fourier references, batched resident tensors, nondefault streams,
retained outputs and training backward. All 61 catalog recipes pass on a real
512×512 CT slice. On the saved 64³ CT crop, all 28 CUDA outputs and 27 CPU outputs
are bit-identical to baseline; CPU RingingOvershoot differs by at most
2.98e-7. This is numerical validation, not a model-quality evaluation.

The baseline is the saved `ba1184aeb` Release extension. Both builds use the same
compiler settings, on Linux amd64, Intel Core Ultra 7 270K Plus / RTX 5090,
CUDA architecture 120. Inputs are PHercParis4 volume
`20260411134726-2.400um-0.2m-78keV-masked.zarr`, level 2, origin
`(9000,3600,3600)`, float32 divided by 255. References start 384 voxels farther
along Z. The 3D crop is 128³; the 2D crop is a single 512² Z plane.

Each public-call route has five warmups and 20 measurements, seeds 123–142,
with shuffled route order and CPU affinity fixed to core 0. Builds run
sequentially. Reading, preparation, validation comparisons, disposal and initial
resident upload/final download are excluded. Public-call copies and input/output
validation are included; CUDA uploaded includes transfers. Earlier unpinned and
intermediate-build measurements are excluded from the tables below. CPU allocator
state and public-call overhead cause variability; kernel profiling and isolated
CPU repeats supplement these measurements.

Cells are **mean / p50 / p95 milliseconds**.

| 128³ transform | CPU before | CPU after | CUDA resident before | CUDA resident after |
| --- | ---: | ---: | ---: | ---: |
| AdditiveNoise | 21.268 / 21.277 / 21.540 | 15.651 / 15.710 / 15.958 | 0.226 / 0.218 / 0.290 | 0.240 / 0.235 / 0.299 |
| BiasField | 12.188 / 11.966 / 13.191 | 11.635 / 11.656 / 11.921 | 0.320 / 0.320 / 0.376 | 0.333 / 0.316 / 0.431 |
| CLAHE | 59.217 / 59.182 / 59.444 | 52.773 / 52.847 / 53.030 | 0.550 / 0.527 / 0.612 | 0.559 / 0.557 / 0.616 |
| ChannelMisalignment | 62.119 / 62.100 / 62.829 | 62.128 / 62.150 / 62.619 | 0.288 / 0.259 / 0.450 | 0.271 / 0.259 / 0.368 |
| Defocus | 17.488 / 17.545 / 17.692 | 15.992 / 16.032 / 16.369 | 0.771 / 0.774 / 0.836 | 0.782 / 0.793 / 0.844 |
| Equalize | 24.662 / 24.652 / 24.831 | 19.460 / 19.465 / 19.703 | 0.372 / 0.408 / 0.427 | 0.357 / 0.357 / 0.422 |
| FDA | 626.328 / 626.271 / 629.860 | 622.657 / 624.581 / 628.723 | 4.538 / 4.448 / 8.093 | 3.741 / 2.220 / 8.239 |
| ForegroundCrop | 2.273 / 2.282 / 2.428 | 2.305 / 2.319 / 2.413 | 0.471 / 0.447 / 0.565 | 0.491 / 0.458 / 0.735 |
| ForegroundDropout | 10.105 / 12.249 / 12.670 | 6.619 / 6.577 / 6.826 | 0.345 / 0.300 / 0.468 | 0.350 / 0.365 / 0.454 |
| GibbsRinging | 428.767 / 428.752 / 429.866 | 429.047 / 429.656 / 437.309 | 0.397 / 0.394 / 0.511 | 0.393 / 0.376 / 0.508 |
| GridDropout | 44.729 / 44.765 / 44.934 | 38.858 / 38.893 / 39.025 | 0.155 / 0.135 / 0.235 | 0.146 / 0.121 / 0.204 |
| HistogramMatching | 97.836 / 97.891 / 98.137 | 87.569 / 87.517 / 87.987 | 4.405 / 4.941 / 5.243 | 1.933 / 1.894 / 2.040 |
| Illumination | 40.065 / 40.027 / 40.552 | 28.062 / 28.083 / 28.422 | 0.166 / 0.184 / 0.219 | 0.150 / 0.143 / 0.210 |
| MotionBlur | 250.077 / 249.985 / 250.617 | 243.999 / 243.983 / 244.581 | 1.356 / 1.370 / 1.412 | 1.355 / 1.355 / 1.432 |
| MultiplicativeNoise | 12.244 / 12.211 / 12.512 | 8.338 / 8.360 / 8.570 | 0.146 / 0.130 / 0.232 | 0.139 / 0.149 / 0.205 |
| ObjectDropout | 22.267 / 22.272 / 22.731 | 22.681 / 22.632 / 23.045 | 0.426 / 0.473 / 0.519 | 0.541 / 0.442 / 0.781 |
| OrientedBlur | 16.395 / 16.436 / 16.698 | 7.628 / 7.586 / 8.005 | 0.762 / 0.731 / 0.976 | 0.724 / 0.726 / 0.899 |
| PixelDistributionAdaptation | 24.639 / 24.580 / 24.943 | 18.500 / 18.495 / 18.722 | 4.047 / 4.479 / 5.339 | 1.477 / 1.435 / 1.672 |
| PlasmaBrightnessContrast | 31.437 / 31.380 / 31.750 | 31.655 / 31.696 / 31.943 | 0.664 / 0.655 / 0.735 | 0.680 / 0.708 / 0.734 |
| PlasmaShadow | 25.268 / 25.319 / 25.524 | 25.349 / 25.379 / 25.617 | 0.593 / 0.593 / 0.643 | 0.591 / 0.597 / 0.641 |
| RandomToneCurve | 14.036 / 14.028 / 14.304 | 11.824 / 11.796 / 12.088 | 0.138 / 0.114 / 0.201 | 0.172 / 0.137 / 0.332 |
| RingingOvershoot | 18.136 / 18.157 / 18.382 | 13.991 / 13.973 / 14.348 | 4.415 / 4.295 / 4.942 | 4.348 / 4.302 / 4.741 |
| SaltAndPepper | 18.603 / 18.567 / 18.966 | 10.074 / 10.104 / 10.285 | 0.151 / 0.132 / 0.218 | 0.162 / 0.153 / 0.217 |
| ShotNoise | 71.223 / 71.237 / 71.426 | 67.767 / 67.833 / 68.103 | 1.228 / 1.212 / 1.293 | 1.240 / 1.256 / 1.294 |
| StripeMasking | 39.556 / 39.548 / 39.735 | 39.719 / 39.697 / 39.871 | 0.154 / 0.188 / 0.200 | 0.155 / 0.170 / 0.208 |
| UnsharpMask | 16.188 / 16.096 / 16.535 | 10.227 / 10.239 / 10.585 | 0.351 / 0.384 / 0.455 | 0.339 / 0.386 / 0.437 |
| Vignetting | 40.332 / 40.305 / 40.829 | 29.106 / 28.276 / 29.359 | 0.173 / 0.173 / 0.237 | 0.168 / 0.205 / 0.224 |
| ZoomBlur | 267.789 / 267.794 / 268.140 | 261.578 / 261.590 / 261.900 | 1.879 / 1.860 / 1.939 | 1.911 / 1.931 / 2.036 |

Isolated CPU repeats use the same 128³ source, core 0, ten warmups and 100
measurements per transform, without CUDA calls interleaved. These checks were
used to investigate timing changes in the initial implementation.

| CPU transform | Before mean / p50 / p95 ms | After mean / p50 / p95 ms |
| --- | ---: | ---: |
| MultiplicativeNoise | 17.624 / 17.679 / 17.914 | 13.695 / 13.694 / 14.078 |
| RingingOvershoot | 17.756 / 17.741 / 18.215 | 13.037 / 13.044 / 13.394 |
| UnsharpMask | 21.624 / 21.730 / 21.949 | 15.467 / 15.513 / 15.828 |

Nsight Systems traces use the same 28 3D recipes, one warmup and two measured
calls per route; instrumented public-call timings are excluded above. Leading
kernel groups are ringing, zoom, motion and shot noise. Kernel statistics below
are **mean / p50 / p95 microseconds per launch**.

| CUDA kernel | Before | After |
| --- | ---: | ---: |
| RingingOvershoot | 1384.750 / 1365.130 / 1420.454 | 1424.170 / 1365.386 / 1710.550 |
| ZoomBlur | 1734.899 / 1735.193 / 1736.240 | 1735.026 / 1735.255 / 1736.360 |
| MotionBlur | 1195.377 / 1195.643 / 1196.827 | 1195.339 / 1195.467 / 1197.043 |
| ShotNoise | 1093.340 / 1093.148 / 1099.092 | 1094.438 / 1093.995 / 1098.411 |
| Defocus | 625.363 / 625.261 / 626.230 | 624.781 / 624.877 / 625.429 |
| OrientedBlur | 193.212 / 136.159 / 307.919 | 193.535 / 136.192 / 308.775 |
| PlasmaBrightnessContrast | 465.742 / 465.550 / 466.351 | 465.459 / 465.406 / 465.630 |
| PlasmaShadow | 458.350 / 456.798 / 462.862 | 457.353 / 456.382 / 461.302 |

The trace contains 879 kernel launches before and after, with unchanged register
counts for the measured augmentation kernels. Their medians differ by less than
0.2%, except the small UnsharpMask pointwise kernel (4.560 → 4.704 µs); ringing
shows tail variability in the instrumented trace. The largest CPU public-call
p50 increase is 1.64% for ForegroundCrop (0.037 ms); the two confirmed prototype
regressions are faster than baseline in both catalog and isolated measurements.

CPU hardware-counter profiling was unavailable because the host sets
`perf_event_paranoid=4`. macOS and arm64 were not runtime-tested; the CPU path
adds no platform-specific instructions.

Representative 512² results, **mean / p50 / p95 milliseconds**:

| 2D transform | CPU | CUDA uploaded | CUDA resident |
| --- | ---: | ---: | ---: |
| BiasField | 1.072 / 1.070 / 1.091 | 0.226 / 0.209 / 0.290 | 0.079 / 0.062 / 0.184 |
| CLAHE | 5.315 / 5.313 / 5.361 | 0.476 / 0.463 / 0.529 | 0.320 / 0.306 / 0.387 |
| Defocus | 0.506 / 0.513 / 0.546 | 0.273 / 0.243 / 0.430 | 0.093 / 0.079 / 0.152 |
| FDA | 63.468 / 63.454 / 63.733 | 0.480 / 0.508 / 0.613 | 0.289 / 0.238 / 0.459 |
| MotionBlur | 19.301 / 19.311 / 19.417 | 0.386 / 0.381 / 0.460 | 0.236 / 0.187 / 0.381 |
| OrientedBlur | 0.625 / 0.622 / 0.648 | 0.241 / 0.234 / 0.292 | 0.099 / 0.084 / 0.106 |
| RingingOvershoot | 0.703 / 0.703 / 0.717 | 0.588 / 0.586 / 0.601 | 0.442 / 0.436 / 0.450 |
| GridDropout | 3.287 / 3.285 / 3.314 | 0.216 / 0.196 / 0.272 | 0.108 / 0.051 / 0.251 |

Local ignored reports `docs/benchmarks/2d-{volume-before,volume-after,plane,numerics,summary}.json`
contain exact commands, extension hashes, raw timings, comparisons and profile
samples. The summary also contains the isolated CPU samples and measurement
script source. Reproduce with existing builds and environment (no installation);
`taskset` is the Linux timing
control used here and can be omitted on other platforms:

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --target _augment --parallel 4
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-volume-cpu --target _augment --parallel 2
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
VC_2D_TRANSFORMS=(AdditiveNoise BiasField ChannelMisalignment CLAHE Defocus Equalize FDA
  ForegroundCrop ForegroundDropout GibbsRinging GridDropout HistogramMatching Illumination
  MotionBlur MultiplicativeNoise ObjectDropout OrientedBlur PixelDistributionAdaptation
  PlasmaBrightnessContrast PlasmaShadow RandomToneCurve RingingOvershoot SaltAndPepper
  ShotNoise StripeMasking UnsharpMask Vignetting ZoomBlur)
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 .venv/bin/python benchmarks/bench_augmentations.py \
  --native-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" \
  --only "${VC_2D_TRANSFORMS[@]}" --warmup 5 --reps 20 --output /tmp/2d-volume-complete.json
# Use /tmp/vc-2d-before and a distinct output path for the saved baseline.
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 .venv/bin/python benchmarks/bench_augmentations.py \
  --native-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" \
  --plane --size 512 --warmup 5 --reps 20 --output /tmp/2d-plane-complete.json
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/2d-profile-complete-pinned \
  .venv/bin/python benchmarks/bench_augmentations.py --native-dir /tmp/vc-reader-cuda \
  --array "$VC_BENCH_VOLUME" --only "${VC_2D_TRANSFORMS[@]}" --warmup 1 --reps 2 \
  --output /tmp/2d-profile-complete-pinned.json
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/2d-profile-complete-pinned.nsys-rep
# Isolated CPU script: source is retained in 2d-summary.json, cpu_isolated_script.
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 .venv/bin/python /tmp/profile_planar_3d_cpu.py /tmp/vc-reader-cuda \
  RingingOvershoot,MultiplicativeNoise,UnsharpMask /tmp/2d-cpu-isolated-complete.json
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python -m unittest discover -s tests -v
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-volume-cpu \
  .venv/bin/python -m unittest discover -s tests -v
```

## Reuse overlapping median neighborhoods (2026-10-17)

CUDA size-3 median filtering sorts three-value columns once per thread and
shares them with adjacent X outputs through warp shuffles. A 3D interior output
loads nine source values instead of 27; warp endpoints load missing columns
explicitly. A shared compile-time odd-even merge generator selects the exact
middle rank from the sorted columns. The 2D path uses the same mechanism for
nine neighbors, replacing repeated full-neighborhood IEEE-key searches.
Reflection, channel selection, quantization options, and output dtypes are
unchanged. Single-channel CUDA outputs replace the working buffer, removing an
8 MiB copy per 128³ float32 crop; multiple channels retain per-channel copies.
Index arithmetic uses uint32 when the volume fits and size_t otherwise.

Measured against the saved clean `c169c451e` extension on Linux amd64 / RTX 5090,
Release, CUDA architecture 120. The loader uses PHercParis4 level-2 128³ CT crops,
min-max normalization, and the shared 12-transform `heavy_transforms()` chain.
There are four spawned workers, four reader threads per worker, and four crops
per batch. Each route has a validation/warmup epoch and three repetitions of
64 batches with rotating route order and warm caches. Two runs per
build give **384 measurements per route/build**. These results use the two after
runs followed by two baseline runs; the earlier exploratory baseline is excluded.
Reads, transfers, casts, augmentation, collation, disposal and stream completion
are included; model computation is excluded. All 256 validation crop hashes
match across routes, runs, and builds. Values are **mean / p50 / p95 milliseconds**.

| Loader route | Before | After | Crops/s before → after |
| --- | ---: | ---: | ---: |
| `Pipeline.batch()` | 3.580 / 3.117 / 4.363 | 3.404 / 2.977 / 3.868 | 1116.37 → 1174.31 (+5.2%) |
| `CudaLoader` prefetch | 3.398 / 3.009 / 3.909 | 3.274 / 2.899 / 3.739 | 1176.49 → 1220.92 (+3.8%) |

Separate Nsight Systems traces cover four validation/warmup and four measured
batches (32 crops). Baseline leading groups are median filtering (10.4% of kernel
time), strided Gaussian convolution (9.6%), and geometry sampling (9.5%). Median
kernel mean / p50 / p95 time falls **56.459 / 56.384 / 56.811 µs →
31.338 / 31.312 / 31.616 µs**, a 44.5% reduction in mean time. Device-to-device
copies fall **160 → 128**, removing **256 MiB** across the trace. The leading
remaining groups are sharpening, strided convolution, and geometry sampling;
sharpening includes a scheduling outlier. Instrumented timings are excluded
from the throughput measurements. Kernel summaries are in local ignored
`docs/benchmarks/median-profile-{before,after}.csv`; traces are
`/tmp/median-profile-before.nsys-rep` and `/tmp/median-prototype.nsys-rep`.

Isolated `MedianFilter(3)` calls use the float32 CT crop at `(9000, 3600, 3600)`,
divided by 255, with one CPU thread. The 3D input is 128³; the 2D input is a
512×512 Z plane. Each route has five warmups followed by 60 measurements with
seeds 123–182 and deterministically shuffled route order. The 3D results combine
two after/before blocks (120 measurements); 2D uses one before/after block (60).
Public-call input preservation and validation are included. Reads, preparation,
comparisons, output disposal, and initial resident upload/final download are
excluded; uploaded CUDA includes transfers. CPU/CUDA outputs match exactly.

| Input / route | Before mean / p50 / p95 ms | After mean / p50 / p95 ms |
| --- | ---: | ---: |
| 128³ CPU | 16.158 / 16.105 / 16.679 | 16.001 / 15.953 / 16.200 |
| 128³ CUDA uploaded | 1.477 / 1.470 / 1.774 | 1.416 / 1.392 / 1.744 |
| 128³ CUDA resident | 0.182 / 0.176 / 0.231 | 0.159 / 0.164 / 0.208 |
| 512² CPU | 5.428 / 5.444 / 5.488 | 5.469 / 5.468 / 5.495 |
| 512² CUDA uploaded | 0.512 / 0.507 / 0.561 | 0.202 / 0.191 / 0.248 |
| 512² CUDA resident | 0.370 / 0.365 / 0.432 | 0.069 / 0.051 / 0.149 |

The 2D resident p50 improves 7.2×. Host overhead and timing variability limit
isolated 3D gains; CPU timings are essentially unchanged. These are throughput
and numerical checks, not model-quality measurements. All 43 tests pass with
CUDA; all 18 precision tests pass in the CPU-only build. Independent NumPy rank
references cover 2D/3D, singleton axes, row/warp/block boundaries, partial warps,
float32/float64, selected channels, exact palettes, quantization, finite extremes,
signed zeros, even windows, validity masks, input preservation and pickle replay.
Batch tests cover retained outputs, nondefault streams and training backward.
Signed-zero tie signs are not guaranteed. macOS/arm64 and volumes needing
64-bit CUDA indexing were not runtime-tested. One loader run emitted the existing
Python resource-tracker semaphore cleanup warning after completing successfully.

Raw local reports are `docs/benchmarks/median-{before-repeat,before-clean,after,after-repeat}.json`,
`median-plane-{before,after}.json`, and `median-loader-{before-repeat,before-clean,after,after-repeat}.json`.
`median-summary.json` combines the samples and checks validation hashes. Reports
retain commands, configuration, extension hashes, raw timings and comparisons.

Reproduce with the existing environment/builds (no installation). Use
`/tmp/vc-median-before` for the saved baseline, and distinct report paths for
repetitions. `--plane` reads a single Z plane and adapts the catalog recipes
and their targets to 2D; it cannot update the cubic-volume README timing tables.

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --target _augment --parallel 4
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-volume-cpu --target _augment --parallel 2
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_augmentations.py \
  --native-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" \
  --only MedianFilter --warmup 5 --reps 60 --output /tmp/median-after.json
# Add --plane --size 512 and use a distinct output path for the 2D measurement.
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --modes batch prefetch \
  --array "$VC_BENCH_VOLUME" --level 2 --batches 64 --batch-size 4 --reps 3 \
  --output /tmp/median-loader-after.json
AGENTS_AGENT_MODE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/median-profile-after \
  .venv/bin/python benchmarks/bench_torch.py --native-dir /tmp/vc-reader-cuda \
  --modes prefetch --array "$VC_BENCH_VOLUME" --level 2 \
  --batches 4 --batch-size 4 --reps 1 --output /tmp/median-profile-after.json
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum --format csv /tmp/median-profile-after.nsys-rep
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python -m unittest discover -s tests -v
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-volume-cpu \
  .venv/bin/python -m unittest discover -s tests -p test_precision.py -v
```

## Remove unused CUDA statistics (2026-10-17)

Filter overflow guards now evaluate a boolean magnitude predicate directly,
without image sums, extrema, or a second reduction kernel. Synchronous calls
read a single flag to choose promotion; asynchronous batches publish into their
existing error flags. Rounding the threshold downward to float32 preserves the
original double comparison for finite float32 inputs. CPU guards already use
a direct predicate. Unclipped CUDA sharpening skips its unused statistics scan.
Gamma without retained statistics, clipped sharpening, local gamma, clipped
local brightness, sheet compression, and median quantization share the existing
channel bounds reduction, which computes only extrema in the input dtype.
CPU algorithms, image dtypes, probabilities, and random draws are unchanged.

Measured on Linux amd64 / RTX 5090, Release, CUDA architecture 120, against the
saved extension from the clean `cae66d562` workspace. The full loader uses
PHercParis4 level-2 128³ CT crops, min-max normalization, and the shared
12-transform `heavy_transforms()` chain. There are four spawned loader workers,
four reader threads per worker, and four crops per batch. Each run has a
validation/warmup epoch and three repetitions of 64 batches per route, with
rotating route order and warm caches. Two runs per build, in before/after then
after/before order, provide **384 measurements per route/build**. Reads,
transfers, casts, augmentation, collation, disposal and stream completion are
included; model computation is excluded. All 256 validation crop hashes match
across routes, runs, and builds. Values are **mean / p50 / p95 milliseconds**.

| Loader route | Before | After | Crops/s before → after |
| --- | ---: | ---: | ---: |
| `Pipeline.batch()` | 3.900 / 3.429 / 4.720 | 3.489 / 3.109 / 4.138 | 1024.95 → 1145.71 (+11.8%) |
| `CudaLoader` prefetch | 3.794 / 3.416 / 4.476 | 3.349 / 2.964 / 3.856 | 1053.73 → 1192.79 (+13.2%) |

Separate Nsight Systems traces cover four validation/warmup and four measured
batches (32 crops). General statistics collection initially accounts for 12.5%
of kernel time, its final reduction 3.2%, median filtering 9.0%, and strided
Gaussian convolution 8.3%. General statistics scans fall from **192 to 32**;
the remaining scans serve contrast. Gamma adds 32 bounds-only scans. Direct
magnitude checks replace the 96 guard reductions, and 32 unused sharpening
scans disappear. Across these affected groups, launches fall **480 → 224**
and kernel time falls **3.197 → 1.106 ms (65.4% less)**. Each crop avoids one
8 MiB image read and eight launches. Total traced kernel time is 20.036 →
16.828 ms, but includes unrelated scheduling outliers; use the affected-group
comparison to assess this change. The remaining leading groups are median
filtering (10.8%), strided convolution (9.9%), and geometry sampling (9.7%).
Instrumented timings are excluded from throughput measurements.
[Before profile](../docs/benchmarks/statistics-profile-before.csv) and
[after profile](../docs/benchmarks/statistics-profile-after.csv) retain the kernel summaries.

Isolated README recipes use the level-2 128³ float32 CT crop at
`(9000, 3600, 3600)`, divided by 255, with one CPU thread. Each route has five
warmups and 60 measurements at seeds 123–182, with deterministic shuffled
case/route order. These synchronous public calls include input preservation
and validation; reads, output comparison/disposal, and initial resident upload
and final download are excluded. The clean measurements ran after/before,
after compilation finished. An initial baseline overlapping compilation is
retained locally but excluded from this table.

| Transform, resident CUDA | Before | After |
| --- | ---: | ---: |
| Gamma | 0.152 / 0.146 / 0.202 | 0.147 / 0.144 / 0.218 |
| LocalGamma | 0.153 / 0.157 / 0.213 | 0.153 / 0.149 / 0.210 |
| GaussianBlur | 0.202 / 0.206 / 0.255 | 0.203 / 0.196 / 0.270 |
| LocalSmoothing | 0.383 / 0.410 / 0.544 | 0.425 / 0.400 / 0.578 |
| Sharpen | 0.263 / 0.214 / 0.429 | 0.193 / 0.187 / 0.244 |

Synchronous gains are smaller and noisy; LocalSmoothing's mean regresses despite
a slightly lower median. These results do not establish a speedup for every
isolated transform. Raw reports also include CPU and uploaded CUDA measurements.
All measured seeds pass CPU/CUDA comparisons, with maximum absolute image
difference 1.19e-7. This is a numerical check, not model-quality evidence.

All 40 tests pass with CUDA; all 16 precision tests pass in the CPU-only build.
Added NumPy gamma/sharpen references cover 2D/3D, singleton and partial-block
sizes, two channels, float32/float64, inversion, retained moments, clipping,
input preservation and pickle replay. A batch regression checks both sides of
exact and rounded overflow thresholds, positive/negative exceptional voxels,
nondefault streams and recovery after rejection. macOS/arm64 and arrays
requiring 64-bit indexing were not runtime-tested.

Local ignored reports `statistics-loader-{before,after}{,-repeat}.json` and
`statistics-summary.json` retain loader measurements. The isolated table uses
`statistics-before-repeat.json` and `statistics-after.json`. Reports record
commands, input/configuration, extension hashes, and numerical checks. Nsight
traces are `/tmp/statistics-profile-{before,after}.nsys-rep`.

Reproduce using the existing builds and environment (no installation). Use
`/tmp/vc-statistics-before` for the saved baseline and distinct report paths;
run the loader twice per build in reversed order for the combined results.

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --target _augment --parallel 4
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --modes batch prefetch \
  --array "$VC_BENCH_VOLUME" --level 2 --batches 64 --batch-size 4 --reps 3 \
  --output /tmp/statistics-loader-after.json
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_augmentations.py \
  --native-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" \
  --only Gamma GaussianBlur Sharpen LocalGamma LocalSmoothing \
  --warmup 5 --reps 60 --output /tmp/statistics-after.json
AGENTS_AGENT_MODE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/statistics-profile-after \
  .venv/bin/python benchmarks/bench_torch.py --native-dir /tmp/vc-reader-cuda \
  --modes prefetch --array "$VC_BENCH_VOLUME" --level 2 \
  --batches 4 --batch-size 4 --reps 1 --output /tmp/statistics-profile-after.json
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/statistics-profile-after.nsys-rep
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python -m unittest discover -s tests -v
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-volume-cpu \
  .venv/bin/python -m unittest discover -s tests -p test_precision.py -v
```

## Composed low-resolution sampling (2026-10-17)

LowResolution composes nearest reduction with linear reconstruction using one
lookup table per axis. Both CPU and CUDA sample directly from the original
image, eliminating the reduced image and its sampling pass. Single-channel
outputs replace the working buffer: CUDA also removes the initial channel copy
and final copy, and CPU removes the final copy. Multiple selected channels keep
a final per-channel copy so unselected channels remain intact. Identity scales
skip sampling. A 128³ float32 CUDA crop uses a 4.5 KiB table, with 32-bit offsets;
larger volumes retain 64-bit offsets. Shared CPU/CUDA helpers construct the
tables and factor interpolation into seven blends in 3D or three in 2D.

Measured on Linux amd64 / RTX 5090, Release, CUDA architecture 120, against the
saved build of `641e562e6`. The isolated recipe is `LowResolution(scale=0.5)` on
a PHercParis4 level-2 128³ float32 CT crop at `(9000, 3600, 3600)`, divided by 255.
There is one CPU thread. Results combine two blocks of 60 measurements per route,
each after five warmups, with seeds 123–182 and deterministically shuffled route
order. Repeated blocks measure variability on the same input. Public-call input
preservation and validation are included; reads, output comparisons/disposal,
and initial resident upload/final download are excluded. Uploaded CUDA includes
transfers. Values are **mean / p50 / p95 milliseconds**.

| LowResolution route | Before | After |
| --- | ---: | ---: |
| CPU | 10.534 / 12.090 / 12.721 | 7.173 / 7.162 / 7.366 |
| CUDA uploaded | 2.189 / 1.632 / 4.136 | 1.315 / 1.320 / 1.466 |
| CUDA resident | 0.319 / 0.354 / 0.428 | 0.132 / 0.122 / 0.301 |

The full loader uses min-max normalization and `heavy_transforms()`'s shared
12-transform chain (including `LowResolution(.85)`), four spawned loader workers
with four reader threads each, and batches of four 128³ crops. Each route has a
validation/warmup epoch followed by three repetitions of 64 batches, with rotating
route order and warm caches. Reading, transfers, casts, augmentation, collation,
disposal, and stream completion are included; model computation is excluded.
All 256 validation crop hashes match across routes within each build; rounding
changes mean hashes differ between builds.

| Loader route | Before mean / p50 / p95 ms | After mean / p50 / p95 ms | Crops/s before → after |
| --- | ---: | ---: | ---: |
| `Pipeline.batch()` | 4.311 / 3.972 / 4.898 | 3.937 / 3.672 / 4.680 | 927.27 → 1015.36 (+9.5%) |
| `CudaLoader` prefetch | 4.185 / 3.746 / 4.754 | 3.714 / 3.332 / 4.374 | 955.44 → 1076.61 (+12.7%) |

Separate Nsight Systems traces cover four warmup and four measured batches
(32 crops). Baseline leading kernel groups are general resampling (15.1%),
statistics collection (11.1%), and median filtering (8.1%); nearest resizing
alone takes another 7.3%. The change removes all 32 nearest-resize launches,
reduces general-resampling launches from 64 to 32, and adds 32 composed-sampling
launches averaging **10.24 µs**. Device-to-device copies fall from 224 to 160,
removing 512 MiB of copied data across the trace. Total kernel time falls from
22.397 to 18.905 ms. The leading remaining groups are statistics collection
(13.2%), median filtering (9.6%), and strided convolution (8.8%). Instrumented
timings are excluded from throughput results; trace totals include scheduling
outliers.

All 38 tests pass with CUDA; the separate CPU-only precision suite passes all
15 tests. Independent NumPy references materialize the two resize stages and
cover 2D/3D, singleton axes, identity scales, multiple and selected channels,
float32/float64, exact nearest ties, overflow-prone finite values, validity masks,
input preservation, and pickle replay. A batch test checks retained outputs,
nondefault CUDA streams, and training backward after buffer replacement.
Comparing before/after outputs on the real crop for scales .5, .85, and randomly
sampled per-axis scales at seeds 123–125 gives a maximum absolute difference of
**1.79e-7** across CPU/CUDA. This is a numerical check, not model-quality evidence.
macOS/arm64 and arrays requiring CUDA's 64-bit offset path were not runtime-tested.
Some loader runs emit Python resource-tracker semaphore cleanup warnings at
process shutdown; validation and measured epochs complete successfully.

Local ignored reports `docs/benchmarks/resize-{before,after}{,-repeat}.json`,
`resize-loader-{before,after}.json`, and `resize-profile-{before,after}.json`
record commands, configuration, extension hashes, and raw measurements.
`resize-summary.json` combines the 120 samples per route. Nsight traces and
CSV summaries are under `/tmp/resize-profile-{before,after}.*`.

Reproduce with the existing environments and builds (no installation):

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --target _augment --parallel 4
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
# Use /tmp/vc-resize-before and a distinct output path for the saved baseline.
# Run twice with distinct report paths to reproduce the two measurement blocks.
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_augmentations.py \
  --native-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" \
  --only LowResolution --warmup 5 --reps 60 --output /tmp/resize-after.json
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --modes batch prefetch \
  --array "$VC_BENCH_VOLUME" --level 2 --batches 64 --batch-size 4 --reps 3 \
  --output /tmp/resize-loader-after.json
AGENTS_AGENT_MODE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/resize-profile-after \
  .venv/bin/python benchmarks/bench_torch.py --native-dir /tmp/vc-reader-cuda \
  --modes prefetch --array "$VC_BENCH_VOLUME" --level 2 \
  --batches 4 --batch-size 4 --reps 1 --output /tmp/resize-profile-after.json
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv \
  /tmp/resize-profile-after.nsys-rep
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python -m unittest discover -s tests -v
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-volume-cpu --target _augment --parallel 4
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-volume-cpu \
  .venv/bin/python -m unittest discover -s tests -p test_precision.py -v
```

## Compact local fields (2026-10-17)

BrightnessGradient, LocalContrast, LocalGamma and LocalSmoothing now consume
separable fields directly on CPU and CUDA. Each field stores four normalization
scalars and one float32 vector per axis: **1,552 bytes for 128³**, eliminating
the 8 MiB dense weight array and its expansion pass. CPU loops reuse each row's
prefix product. Centered brightness uses the product of axis means; nearly
constant fields use an allocation-free scan to avoid amplifying product rounding.
Unclipped brightness also skips unused image bounds. CUDA LocalContrast computes
only the weighted mean it needs, without collecting unused image bounds.
Sampling and field evaluation are shared between backends. No compiler precision
flags, image dtypes, transform probabilities, or random draw order changed.

Measured on Linux amd64 / RTX 5090, Release CUDA architecture 120. README recipes
use a PHercParis4 level-2 128³ float32 CT crop at `(9000, 3600, 3600)`, divided
by 255, with one CPU thread. Results combine **two blocks of 30 measurements**
per case/route, each after five warmups (seeds 123–152 in each block). Case and
route orders are deterministically shuffled. Public-call validation and input
preservation are included; setup, reads, comparison, output disposal, and initial
resident upload/final download are excluded. Values are **mean / p50 / p95 ms**.
The repeat blocks check timing variability; they are not independent datasets.

| Transform / route | Before | After |
| --- | ---: | ---: |
| BrightnessGradient, CPU | 13.667 / 14.005 / 14.554 | 2.590 / 2.577 / 2.796 |
| BrightnessGradient, resident CUDA | 0.243 / 0.218 / 0.442 | 0.136 / 0.133 / 0.210 |
| LocalContrast, CPU | 9.371 / 9.396 / 9.779 | 4.189 / 4.191 / 4.338 |
| LocalContrast, resident CUDA | 0.250 / 0.220 / 0.435 | 0.151 / 0.138 / 0.216 |
| LocalGamma, CPU | 18.068 / 18.064 / 18.376 | 12.307 / 12.329 / 12.515 |
| LocalGamma, resident CUDA | 0.245 / 0.221 / 0.436 | 0.160 / 0.161 / 0.219 |
| LocalSmoothing, CPU | 19.115 / 19.081 / 19.740 | 7.132 / 7.100 / 7.532 |
| LocalSmoothing, resident CUDA | 0.416 / 0.444 / 0.596 | 0.389 / 0.393 / 0.451 |

The loader uses min-max normalization and `heavy_transforms()`'s shared
12-transform chain, four spawned loader workers with four reader threads each,
and batches of four 128³ crops. Each route has a full validation/warmup epoch,
then three repetitions of 64 batches in rotating route order, with warm caches.
Reading, transfers, casts, augmentation, collation, disposal and stream completion
are included; there is no model computation. All 256 validation crop hashes
match between batch and prefetch routes within each build.

| Loader route | Before mean / p50 / p95 ms | After mean / p50 / p95 ms | Crops/s before → after |
| --- | ---: | ---: | ---: |
| `Pipeline.batch()` | 4.491 / 4.088 / 4.972 | 4.254 / 3.956 / 4.909 | 890.13 → 939.86 (+5.6%) |
| `CudaLoader` prefetch | 4.458 / 4.026 / 4.984 | 4.183 / 3.869 / 4.628 | 896.90 → 955.77 (+6.6%) |

A separate Nsight Systems trace covers four warmup and four measured batches
(32 crops). The baseline's leading kernel groups were resampling (13.7%),
statistics collection (12.9%) and local-field expansion (10.7%). All **64 dense
local-field expansion launches disappear**, removing 2.633 ms of kernel work.
Consumers now evaluate their weights, so the net change in total kernel time is
24.607 → 22.723 ms (7.7% less). The largest remaining groups are resampling
(14.9%), statistics collection (11.0%) and strided convolution (9.0%).
Instrumented timings are excluded from the throughput tables; the trace includes
occasional scheduling outliers. [Before profile](../docs/benchmarks/separable-profile-before.csv)
and [after profile](../docs/benchmarks/separable-profile-after.csv) retain the data.

The CUDA regression suite passes all 35 tests; the separate CPU-only precision
suite passes all 13 tests. Independent NumPy references cover each transform,
2D/3D data, singleton axes, narrow/off-image/near-constant fields, two channels,
float32/float64 inputs, source preservation, and exact same-backend pickle replay.
A storage regression keeps each 128³ single-channel local-transform plan below
2 KiB. The real-data catalog checks pass for every measured seed. A separate
comparison of the four recipes at seeds 123–125 gives maximum before/after
absolute error **5.96e-8** across CPU and CUDA. This is a numerical comparison,
not a model-quality measurement. macOS/arm64 and arrays requiring the 64-bit
indexing fallback were not runtime-tested. See the [precision policy](../docs/precision.md).

Local JSON reports `docs/benchmarks/separable-{before,after}{,-repeat}.json`
contain commands, raw samples, input hashes, extension hashes and numerical
checks. `separable-summary.json` combines their 60 samples per route;
`separable-loader-{before,after}.json` and `separable-profile-{before,after}.json`
record loader/profiler configurations, and `separable-numerics.json` records the
separate numerical comparison.

Reproduce using the existing build/environment (no setup required). To compare
the saved baseline, use `/tmp/vc-separable-before` and a separate output path.
Run the augmentation command twice with distinct report paths for the two blocks.

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --target _augment --parallel 4
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_augmentations.py \
  --native-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" \
  --only BrightnessGradient LocalContrast LocalGamma LocalSmoothing \
  --warmup 5 --reps 30 --output /tmp/separable-after.json
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --modes batch prefetch \
  --array "$VC_BENCH_VOLUME" --level 2 --batches 64 --batch-size 4 --reps 3 \
  --output /tmp/separable-loader-after.json
AGENTS_AGENT_MODE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/separable-profile-after \
  .venv/bin/python benchmarks/bench_torch.py --native-dir /tmp/vc-reader-cuda \
  --modes prefetch --array "$VC_BENCH_VOLUME" --level 2 \
  --batches 4 --batch-size 4 --reps 1 --output /tmp/separable-profile-after.json
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/separable-profile-after.nsys-rep
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python -m unittest discover -s tests -v
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-volume-cpu --target _augment --parallel 4
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-volume-cpu \
  .venv/bin/python -m unittest discover -s tests -p test_precision.py -v
```

## Gaussian filter work reduction (2026-10-17)

The real training-loader profile identified Gaussian convolution as 28.4% of
GPU kernel time. The CUDA implementation now extends shared-memory tiles once
and reuses their samples across outputs, instead of fetching and boundary-mapping
every tap independently. It uses 128-element rows for the contiguous axis and
eight positions on 32 adjacent lines for other axes. Small-array tile arithmetic
uses 32-bit indices with headroom for inactive threads; large arrays retain
64-bit indices. Radii above 32 retain the general convolution implementation.
CPU convolution already extends rows and shares boundary tables.

LocalSmoothing and UnsharpMask now call the same blur implementation with
separate input/output buffers. Choosing buffers by pass parity eliminates the
initial image copy and the final result copy for a three-axis blur: two fewer
8 MiB copies per 128³ float32 crop. Identity kernels also skip filtering and
scratch allocation. Coefficients, tap order, boundary modes, image dtypes,
overflow handling, and random draws remain unchanged; CUDA FMA remains enabled.

Measured on RTX 5090, Release CUDA architecture 120, PHercParis4 level 2,
128³ crops. The single-transform cases use the README recipes, crop origin
`(9000, 3600, 3600)`, float32 CT / 255, one CPU thread, five warmups and 30
measurements per route (seeds 123–152). Cases/routes are deterministically
shuffled. Resident timings include public-call copies/validation and exclude
initial upload, final download, and output disposal. Values below are
**mean / p50 / p95 milliseconds**; timing tails remain noisy.

| Resident CUDA transform | Before | After |
| --- | ---: | ---: |
| GaussianBlur | 0.278 / 0.272 / 0.341 | 0.181 / 0.183 / 0.241 |
| LocalSmoothing | 0.492 / 0.529 / 0.597 | 0.430 / 0.441 / 0.702 |
| UnsharpMask | 0.460 / 0.489 / 0.672 | 0.370 / 0.407 / 0.563 |
| SheetCompression | 1.113 / 1.087 / 1.268 | 0.928 / 0.888 / 1.034 |

The full loader uses min-max normalization and the shared 12-transform chain,
four spawned CPU loader workers, four reader threads per worker, and batches of
four crops. Each route has one validation/warmup epoch, then three repetitions
of 64 batches with rotating route order and warm caches. Timings include reads,
upload/casts, augmentation, collation, output disposal, and stream completion;
they exclude model execution. All 256 validation crop hashes matched across
routes **and across the before/after builds**.

| Loader route | Before mean / p50 / p95 ms | After mean / p50 / p95 ms | Crops/s before → after |
| --- | ---: | ---: | ---: |
| `Pipeline.batch()` | 5.310 / 4.883 / 5.889 | 4.461 / 4.053 / 4.900 | 752.93 → 896.23 (+19.0%) |
| `CudaLoader` prefetch | 5.153 / 4.772 / 5.736 | 4.393 / 3.980 / 4.701 | 775.96 → 910.19 (+17.3%) |

Separate Nsight Systems runs cover four warmup and four measured batches
(32 crops). Convolution kernel time fell from **8.808 to 2.553 ms (3.45×)**;
total kernel time fell from 31.052 to 24.946 ms. Device-to-device copies fell
from 288 to 224, confirming two eliminated copies per crop. The largest remaining
kernel groups are trilinear resampling (13.6%), statistics collection (12.7%),
and dense local-field expansion (10.6%). Instrumented wall times are excluded
from performance comparisons. [Before profile](../docs/benchmarks/gaussian-profile-before.csv)
and [after profile](../docs/benchmarks/gaussian-profile-after.csv) retain the data.

The local JSON reports `docs/benchmarks/gaussian-{before,after}.json`,
`gaussian-loader-{before,after}.json`, and `gaussian-profile-{before,after}.json`
record commands, input hashes, extension hashes, raw measurements, and validation.
Reproduce with the existing environments/builds (no setup required):

```bash
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-reader-cuda --target _augment --parallel 4
VC_BENCH_VOLUME=/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
# Use /tmp/vc-filter-baseline and a separate output path for the saved before build.
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_augmentations.py \
  --native-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" \
  --only GaussianBlur LocalSmoothing UnsharpMask SheetCompression \
  --warmup 5 --reps 30 --output /tmp/filter-final-after.json
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --modes batch prefetch \
  --array "$VC_BENCH_VOLUME" --level 2 --batches 64 --batch-size 4 --reps 3 \
  --output /tmp/filter-loader-after.json
AGENTS_AGENT_MODE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/filter-profile-final \
  .venv/bin/python benchmarks/bench_torch.py --native-dir /tmp/vc-reader-cuda \
  --modes prefetch --array "$VC_BENCH_VOLUME" --level 2 \
  --batches 4 --batch-size 4 --reps 1 --output /tmp/filter-profile-final.json
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv \
  /tmp/filter-profile-final.nsys-rep
AGENTS_AGENT_MODE=1 VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python -m unittest discover -s tests -v
```

All 33 tests pass with CUDA, including stream, replay, backward, and spawn-worker
coverage. New numerical tests compare an independent NumPy separable reference
over signed float32/float64 data, partial tiles, singleton axes, all three edge
modes, identity filters, and radii 32/33. Separate-output tests cover even/odd
pass counts and preservation of source images. The CPU-only precision suite
also passes. Real CT CPU/CUDA differences stay below 5.01e-6 for all four
affected recipes. Performance was measured on Linux amd64/RTX 5090 only;
macOS/arm64 and arrays requiring 64-bit indexing were not runtime-tested.

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
| Per-sample CUDA, `num_workers=0` | 4.873 | 4.714 | 5.933 | 820.23 |
| `Pipeline.batch()` | 3.638 | 3.306 | 4.022 | 1098.67 |
| `CudaLoader`, one batch ahead | 3.281 | 2.927 | 3.935 | 1218.34 |

The public prefetch loader delivered 48.5% higher throughput than the serial route.
These are loader-only results. The native executor still loops over samples within each submitted batch.

### Current kernel profile

A separate Nsight Systems trace of the same 12-augmentation chain covers four
warmup and four measured batches (32 crops). The sum of GPU kernel time is
16.325 ms. This excludes host work,
transfers and waiting; instrumented wall timings are not performance comparisons.
The largest kernel groups in that trace, amortized per crop, are:

| Kernel group | µs/crop | Share of kernel time |
| --- | ---: | ---: |
| Convolution | 79.8 | 15.6% |
| Trilinear resampling | 51.2 | 10.0% |
| Sharpening | 43.2 | 8.5% |
| Flipping | 41.9 | 8.2% |
| Pointwise transforms | 35.1 | 6.9% |
| Median filter | 31.3 | 6.1% |

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
| flip_x | 0.100 | 0.097 | 0.122 |
| flip_all | 0.092 | 0.078 | 0.111 |
| elastic | 0.246 | 0.233 | 0.321 |

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

Refreshed 2026-10-17 at `46b908a31` on the supplied PHercParis4 volume:

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
| ML batch: 8 × 128³ crops | 14.283 / 14.158 / 15.602 | 4.916 / 4.894 / 6.026 | 1.155 / 1.147 / 1.297 | 4.785 / 4.796 / 5.274 |
| Curved tifxyz surface: 96 × 512 | 11.128 / 10.824 / 14.702 | 6.687 / 6.645 / 8.035 | 0.901 / 0.868 / 1.325 | 1.784 / 1.792 / 1.961 |
| Axis-aligned slab: 16 × 512² | 6.189 / 6.263 / 7.391 | 2.936 / 2.900 / 3.499 | 0.488 / 0.401 / 0.809 | 2.197 / 2.314 / 2.829 |
| Rotated box: 128³ | 72.803 / 69.288 / 78.310 | 81.512 / 65.259 / 72.842 | 14.051 / 13.792 / 17.319 | 1.637 / 1.629 / 1.873 |

Throughput is **requests/s** over the sum of measured latencies; one ML request
is an entire 8-crop batch. JSON also reports crops/s, Mvoxels/s, output GiB/s,
stage times, min/max, cache counters, and every individual measurement.

| Workload | zarr-python | TensorStore | vc_reader CPU | vc_reader CUDA |
| --- | ---: | ---: | ---: | ---: |
| ML batch: 8 × 128³ crops | 70.0 | 203.4 | 865.5 | 209.0 |
| Curved tifxyz surface: 96 × 512 | 89.9 | 149.5 | 1109.3 | 560.4 |
| Axis-aligned slab: 16 × 512² | 161.6 | 340.6 | 2050.6 | 455.2 |
| Rotated box: 128³ | 13.7 | 12.3 | 71.2 | 610.7 |

Speedups divide baseline p50 by the corresponding vc_reader p50. Values below
1× mean vc_reader is slower for that comparison.

| Workload | CPU vs zarr | CPU vs TensorStore | CUDA vs zarr | CUDA vs TensorStore | CUDA vs CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| ML batch: 8 × 128³ crops | 12.35× | 4.27× | 2.95× | 1.02× | 0.24× |
| Curved tifxyz surface: 96 × 512 | 12.48× | 7.66× | 6.04× | 3.71× | 0.48× |
| Axis-aligned slab: 16 × 512² | 15.63× | 7.24× | 2.71× | 1.25× | 0.17× |
| Rotated box: 128³ | 5.02× | 4.73× | 42.52× | 40.05× | 8.46× |

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
| Fetching + conversion/upload | 0.536 / 0.492 / 0.836 | 1.167 / 1.110 / 1.518 | 0.554 / 0.540 / 0.897 |
| Normalization | 2.821 / 2.797 / 3.138 | 0.132 / 0.126 / 0.183 | 0.099 / 0.092 / 0.127 |
| 12 augmentations | 146.387 / 146.465 / 148.219 | 0.720 / 0.697 / 0.949 | 0.689 / 0.673 / 0.764 |
| Final host download | 0.000 / 0.000 / 0.000 | 0.549 / 0.528 / 0.667 | 0.528 / 0.522 / 0.579 |
| End-to-end | 149.745 / 149.723 / 151.629 | 2.567 / 2.468 / 3.248 | 1.870 / 1.833 / 2.215 |

| Metric | CPU throughout | CUDA chunks + CUDA pipeline | CPU read + CUDA pipeline |
| --- | ---: | ---: | ---: |
| Crops/s | 6.68 | 389.54 | 534.70 |
| Mvoxels/s | 14.00 | 816.92 | 1121.35 |
| Speedup over CPU (p50) | 1.00× | 60.65× | 81.68× |

For these boxes, CPU reading followed by CUDA processing was **81.68× faster
than staying on CPU** and **1.35× faster than the CUDA-chunks route** end to end
(p50). Hybrid and CUDA-chunks outputs matched bit-for-bit for all six crops;
the maximum difference from CPU was 1.07e-06. This measures fresh uploads with
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
| Fetching + sampling | 15.125 / 15.164 / 19.092 | 1.189 / 1.186 / 1.535 | 12.79× |
| Normalization | 2.601 / 2.586 / 2.776 | 0.132 / 0.125 / 0.165 | 20.65× |
| 12 augmentations | 143.319 / 143.225 / 145.493 | 0.726 / 0.695 / 0.970 | 206.12× |
| Final host download | 0.000 / 0.000 / 0.000 | 0.538 / 0.525 / 0.635 | — |
| End-to-end | 161.045 / 160.279 / 165.771 | 2.585 / 2.544 / 3.113 | 63.01× |

| Throughput | vc_reader CPU | vc_reader CUDA |
| --- | ---: | ---: |
| Crops/s | 6.21 | 386.87 |
| Mvoxels/s | 13.02 | 811.32 |

CPU/CUDA outputs passed the same tolerance check; maximum absolute difference
was 1.07e-06 after the full chain. Full pipeline report (`docs/benchmarks/pipeline.json`).
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


## Aligned sampling

`bench_sampling.py` compares direct and controlled affine crops of an existing
real Zarr volume with low-level reads. Supply `--array`, `--start Z Y X`,
`--native-dir`, and `--output`; `--size` defaults to 64 cubed. `bench_sampling_fields.py`
measures real XYZ field values from the bundled fixture, re-encoded temporarily
as CZYX TIFF without surface projection. Both use six fixed requests, two warmup
passes and ten measured passes, plus separate cold-reader, allocation and
cProfile runs. Existing dependencies/builds are required; no installation occurs.

See [measurement commands and results](../docs/aligned-sampling-benchmarks.md).

## Surface layers and label-patch projection

`bench_surface.py` runs six real requests, two warmups and ten repetitions,
recording first requests, initialization, profiles, cache/transfer counters and
output hashes. The default uses the bundled PHercParis4 tifxyz and the existing
local level-2 volume. It independently verifies coordinates with TIFF/SciPy
and CT samples against direct volume reads.

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_surface.py \
  --native-dir /tmp/vc-reader-cuda --backend both --output /tmp/surface-layers.json

AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_surface.py \
  --native-dir /tmp/vc-reader-cuda --backend both \
  --label-source benchmarks/fixtures/surface_projection/label.json \
  --reference-source benchmarks/fixtures/surface_projection/reference.json \
  --start 18576 7984 --size 16 16 --output /tmp/surface-projection.json
```

The second command uses a recorded real PHerc0009B label/geometry pair and an
explicit render-convention registration. It compares the first zero-thickness
projection to `surface_reference.py`, an independent float64 scalar oracle, and
checks CT crops directly. It benchmarks a 2.4 µm label band with 2.4 µm context.
The initial run fetches only geometry files and requested label/volume chunks;
subsequent runs can add `--offline`. See
[registration evidence and measurements](../docs/surface-sampling-benchmarks.md).

`bench_pipeline.py` now uses `SurfaceGeometry` for its curved point workload.
Tifxyz `scale` is XY, so the unequal scales in the bundled strip are applied to
columns and rows respectively. Previous surface timing tables used the opposite
axis interpretation; their selected UV requests and hashes are not comparable
to new runs. The volume reader workload still samples one surface layer.

## Composable training workflows

`bench_training.py` shares the example-owned Dataset and input recipes from
`scripts/training_workflows.py` and `scripts/training_inputs.py`. It compares
reader and Dataset overhead, then measures default collation, host pinning,
upload, resident augmentation and persistent-worker epochs separately. Each
workload has six fixed requests, two warmup passes and ten measured passes.
Use `--cuda` to add GPU stages; `--torch-threads` defaults to one. Real projection
requires `--batch-size 1` because geometry determines its output bounds.

Run `--help` for workflow names and options. Exact real IDs, independent
comparisons, reproduction commands, measured costs and real volume-transfer
acceptance are recorded in
[training workflow acceptance](../docs/training-workflow-benchmarks.md).
The [usage guide](../docs/training-workflows.md) documents cache budgets, field
semantics, layer validity, seeds, process cleanup and optional dependencies.
