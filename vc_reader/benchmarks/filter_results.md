# Median and Gaussian optimization — 2026-10-16

This pass builds on [the first optimization pass](augmentation_results.md). On the same 128³ CT crop, median size 3 fell from **176.18 to 44.07 ms** and Gaussian sigma 1 from **20.24 to 13.42 ms** (p50). The unchanged Python transforms took **172.46 and 70.19 ms**, respectively. Including a fresh reader call and normalization, native median takes **46.37 ms** and blur **16.74 ms**.

## Implementation

- `cpp/augment/intensity.hpp`: an exact palette of up to 256 distinct channel values feeds a two-level sliding histogram. Each step replaces one column of the window instead of selecting from an entire new window. Palette values can be arbitrary doubles; there is no quantization. A bounded probe rejects typical continuous-valued input early. The full encoding pass also checks the limit before modifying the input. Larger palettes, NaNs, and mixed signed zeros retain the original selection path.
- `cpp/augment/core.hpp`: Gaussian convolution accumulates 16 adjacent outputs before writing them once. Reflected row padding and precomputed strided offsets remove boundary work from the tap loop. Kernel generation, tap order, double intermediates, and boundary rules are unchanged. The shared helper also benefits local smoothing and deformation fields.
- No new transform classes, Python filter kernels, dependencies, RNG changes, CPU-specific intrinsics, or fast-math flags.
- `benchmarks/bench_augment.py` adds filter-width sweeps and a fixed noisy-input option. `profile_augment.cpp` accepts an iteration count. Regression tests cover exact palettes, continuous values, constant/signed-zero inputs, palette overflow, and wide kernels on small dimensions.

## Measurement setup

- Linux x86-64, Intel Core Ultra 7 270K Plus, CPU affinity 0; one Torch/OMP/MKL/OpenBLAS thread. Python 3.14.4, Torch 2.14.1+cu130 on CPU, NumPy 2.5.4, SciPy 1.18.1.
- Release build: `-O3 -DNDEBUG -std=gnu++20`, no native-CPU or fast-math flags. Baseline source commit: `08b060713c0070c9b06d06f1061f0567f4924317`.
- Baseline extension SHA-256: `01d0f90829bc113ab75b1a7c7af3955c186831f90dd43783038ae60c08dc3339`.
- Updated extension SHA-256: `af1cd99f5eb9c374c409faded8bff6d16ad70d1472467a379815b714eed356a4`. Updated reports confirm the imported extension matches the CMake output. Baseline reports deliberately load the saved baseline binary.
- 128³: 3 warmups, 15 observations per backend. 64³ sweeps: 2 warmups, 5 observations. Backend order alternates. Seed 123. Timings include input preservation, sampling, and adapters; reads are excluded except in the reader table.
- Reader mode uses `--read-each-call --inplace`: both backends receive a fresh float32 read, divide by 255, and augment. Each report records 76 disk hits and zero remote fetches. These are warm local reads, not full DataLoader or cold-storage throughput.
- Gaussian Python reference uses its direct convolution with autotuning disabled. Native retains double intermediates, while Torch generally uses float32. All measured filter outputs pass the harness comparison; median outputs match exactly.

Input array:

`/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2`

One float32 channel from uint8 CT divided by 255. 128³ starts at `(1280,2688,3200)` (29.54% nonzero); 64³ starts at `(1312,2752,3232)` (59.03% nonzero). The noisy sweep adds fixed float32 Gaussian noise with standard deviation .01 before timing, exercising the general median path.

## Results

Times are **mean / p50 / p95 milliseconds**. Improvement is baseline native p50 / updated native p50. Python numbers come from the updated run. These are local measurements on a shared host; p95 is approximate with these sample counts.

### 128³, normal input

| Transform | Before C++ | After C++ | Python | Improvement |
| --- | ---: | ---: | ---: | ---: |
| GaussianBlur | 23.002 / 20.241 / 27.764 | 16.922 / 13.419 / 21.494 | 67.789 / 70.192 / 70.932 | 1.51× |
| MedianFilter | 176.281 / 176.176 / 176.998 | 44.046 / 44.066 / 44.346 | 172.547 / 172.462 / 173.249 | 4.00× |
| LocalSmoothing | 48.119 / 45.115 / 52.960 | 31.566 / 31.373 / 32.289 | 27.567 / 27.569 / 28.057 | 1.44× |
| IntensityPipeline | 28.616 / 28.746 / 29.214 | 32.985 / 32.963 / 33.488 | 92.894 / 92.754 / 93.652 | 0.87× |

The companion intensity pipeline was sensitive to run context: its original baseline p50 was 28.746 ms, a second baseline run was 40.212 ms, and the updated run was 32.963 ms. The repeated baseline blur and median p50s remained 20.086 and 177.234 ms. We therefore reran the pipeline alone in fresh processes (4 warmups, 20 observations); this showed the expected blur benefit, with Python timings stable across the pair. The isolated results are below; they do not erase the mixed-case variation above.

| Isolated intensity pipeline | Before C++ | After C++ | Python | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Brightness → gamma → blur | 26.678 / 26.469 / 27.454 | 20.769 / 20.659 / 21.418 | 87.743 / 87.668 / 88.430 | 1.28× |

### 128³, fresh reader + normalization + augmentation

| Transform | Before C++ | After C++ | Python | Improvement |
| --- | ---: | ---: | ---: | ---: |
| GaussianBlur | 25.677 / 24.023 / 29.015 | 19.054 / 16.737 / 22.081 | 69.330 / 70.962 / 71.642 | 1.44× |
| MedianFilter | 188.038 / 187.807 / 189.219 | 46.214 / 46.366 / 46.585 | 172.805 / 172.735 / 173.897 | 4.05× |

### 64³, normal input

| Transform | Before C++ | After C++ | Python | Improvement |
| --- | ---: | ---: | ---: | ---: |
| GaussianBlur | 1.445 / 1.448 / 1.451 | 1.109 / 1.100 / 1.145 | 1.709 / 1.709 / 1.722 | 1.32× |
| MedianFilter | 36.048 / 36.059 / 36.142 | 7.274 / 7.259 / 7.331 | 37.497 / 37.459 / 37.727 | 4.97× |
| GaussianBlurSigma0.5 | 0.892 / 0.893 / 0.896 | 0.932 / 0.930 / 0.938 | 0.964 / 0.968 / 0.981 | 0.96× |
| GaussianBlurSigma2 | 2.519 / 2.515 / 2.551 | 1.459 / 1.440 / 1.510 | 2.832 / 2.793 / 2.978 | 1.75× |
| GaussianBlurSigma4 | 5.836 / 5.811 / 5.932 | 2.412 / 2.404 / 2.463 | 5.766 / 5.658 / 6.305 | 2.42× |
| MedianFilterSize4 | 74.700 / 74.843 / 74.954 | 9.734 / 9.749 / 9.779 | 76.763 / 76.732 / 76.903 | 7.68× |
| MedianFilterSize5 | 140.873 / 137.143 / 152.045 | 13.087 / 13.115 / 13.146 | 139.568 / 139.544 / 139.864 | 10.46× |
| MedianFilterSize7 | 346.880 / 347.078 / 347.173 | 21.629 / 21.618 / 21.691 | 351.498 / 350.997 / 352.402 | 16.06× |

### 64³, continuous-valued noisy input

| Transform | Before C++ | After C++ | Python | Improvement |
| --- | ---: | ---: | ---: | ---: |
| GaussianBlur | 1.497 / 1.495 / 1.523 | 1.080 / 1.082 / 1.091 | 1.813 / 1.815 / 1.828 | 1.38× |
| MedianFilter | 62.859 / 62.875 / 63.062 | 63.241 / 63.228 / 63.283 | 66.227 / 66.220 / 66.301 | 0.99× |
| GaussianBlurSigma0.5 | 0.896 / 0.896 / 0.898 | 0.933 / 0.933 / 0.935 | 0.965 / 0.969 / 0.980 | 0.96× |
| GaussianBlurSigma2 | 2.535 / 2.539 / 2.554 | 1.446 / 1.442 / 1.478 | 2.880 / 2.878 / 2.949 | 1.76× |
| GaussianBlurSigma4 | 5.957 / 5.954 / 6.005 | 2.390 / 2.400 / 2.426 | 5.359 / 5.341 / 5.479 | 2.48× |
| MedianFilterSize4 | 132.012 / 131.787 / 132.650 | 133.009 / 132.858 / 133.614 | 138.001 / 137.980 / 138.152 | 0.99× |
| MedianFilterSize5 | 238.101 / 238.068 / 238.327 | 239.927 / 239.896 / 240.198 | 249.138 / 249.133 / 249.319 | 0.99× |
| MedianFilterSize7 | 599.815 / 599.384 / 601.190 | 608.852 / 608.859 / 608.967 | 628.780 / 628.713 / 629.188 | 0.98× |

The histogram benefit depends on the values reaching median: normalized uint8 CT and many pointwise transforms retain a small palette, while noise or interpolation generally do not. The unchanged general selector is within 2% of baseline here. The smallest Gaussian kernel (sigma .5) is about 4% slower than baseline at 64³, although still slightly faster than the Python reference. Wider kernels show larger gains.

## Native profiles

The standalone workload used the raw 128³ chunk `.../2/10/21/25`, with 100 blur calls or 30 median calls, pinned to CPU 0. Builds used `-O2 -g -pg`; these profiles locate costs and are separate from the uninstrumented Release latency measurements. Hardware `perf` sampling is unavailable on this host (`perf_event_paranoid=4`).

| Workload | Before: top sampled costs | After: top sampled costs |
| --- | --- | --- |
| Blur, 100 calls | Gaussian convolution 1.72 s (92.5%); validation .05 s (2.7%) | Gaussian convolution 1.00 s (83.3%); channel writeback .09 s (7.5%); validation .06 s (5.0%) |
| Median, 30 calls | `std::__introselect` 4.60 s (84.2%); median window assembly .80 s (14.7%) | histogram body .69 s (69%); hash lookup .18 s (18%); encoding helper .05 s (5%) |

Total sampled native time: blur 1.86 → 1.20 s, median 5.46 → 1.00 s. Sampling resolution is 10 ms; shared-library time is not represented in these totals. GNU gprof labels compiler-generated selection/hash clones as `frame_dummy`; mapping histogram PCs to `nm -n -C` symbols identifies the functions above. Resolved symbols are retained in `results/filters-profile-symbols.txt`.

A separate Release kernel experiment compared accumulator blocks of 4/8/16/32 outputs (3 warmups, 20 observations). For 128³ sigma 3, p50 was 22.818 / 19.541 / 18.388 / 27.879 ms; for sigma 1 it was 12.147 / 11.164 / 10.927 / 14.856 ms. This selected 16. The experiment reads the raw chunk directly; its 64³ subcase uses the first 64³ bytes, not the populated crop used in the main tables. Raw observations are in `results/filters-blur-blocks.txt`.

## Build and reproduce

From the repository root, with the existing dependencies:

```sh
AGENTS_AGENT_MODE=1 cmake -S vc_reader -B /tmp/vc-augment-build \
  -DVC_READER_BUILD_READER=OFF \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-augment-build --target _augment -j 2
cp /tmp/vc-augment-build/_augment.cpython-314-x86_64-linux-gnu.so vc_reader/src/vc_reader/

export AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=vc_reader/src:vesuvius/src:vc_reader/.venv/lib/python3.14/site-packages
taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/bench_augment.py \
  --only GaussianBlur,MedianFilter,LocalSmoothing,IntensityPipeline \
  --reps 15 --warmup 3 --output vc_reader/benchmarks/results/filters-after-128.json
taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/bench_augment.py \
  --only GaussianBlur,MedianFilter --read-each-call --inplace \
  --reps 15 --warmup 3 --output vc_reader/benchmarks/results/filters-reader-after-128.json
taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/bench_augment.py \
  --filter-sweep --start 1312 2752 3232 --shape 64 64 64 --reps 5 --warmup 2 \
  --output vc_reader/benchmarks/results/filters-after-64.json
taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/bench_augment.py \
  --filter-sweep --input-noise-std .01 --start 1312 2752 3232 --shape 64 64 64 \
  --reps 5 --warmup 2 --output vc_reader/benchmarks/results/filters-noisy-after-64.json
```

`-S` and the explicit site-packages path bypass a stale local editable-reader import hook. Normally the usual `PYTHONPATH=vc_reader/src:vesuvius/src` suffices. `taskset` is Linux-specific measurement configuration; the library does not require it. The ABI suffix in the copy command is specific to this Python environment.

The baseline binary was saved before editing in `/tmp/vc-filter-baseline/`. The same commands were run with `/tmp/vc-filter-baseline/run_before.py` in place of `bench_augment.py`, and `before` output filenames. This loader imports the saved extension as `vc_reader._augment` and executes the current harness. The later baseline repeat is retained as `filters-before-repeat-128.json`. The isolated pipeline runs used `--only IntensityPipeline --reps 20 --warmup 4`, saved to `filters-pipeline-before-128.json` and `filters-pipeline-after-128.json`.

Profile commands (the before build uses saved baseline headers in `/tmp/vc-filter-baseline/before` instead):

```sh
AGENTS_AGENT_MODE=1 c++ -std=c++20 -O2 -g -pg -I vc_reader/cpp/augment \
  vc_reader/benchmarks/profile_augment.cpp -o /tmp/vc-filter-baseline/after/profile
cd /tmp/vc-filter-baseline/after
taskset -c 0 ./profile \
  /mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2/10/21/25 blur 100
gprof -b -p ./profile gmon.out
# Run again with "median 30" for the median profile.
```

The profiling source and augmentation benchmark harness are tracked; generated JSON, profiles, binary snapshots and comparison arrays are local artifacts under the ignored `benchmarks/results/` and `/tmp/vc-filter-baseline/` directories. Tables here retain the summary statistics.

## Correctness and limitations

- **772 tests passed**, 2 network tests deselected, 3 existing multiprocessing/fork warnings. This includes the reader suite and Vesuvius dataset integration.
- **300 additional filter arrays match the baseline bit-for-bit**, covering float32/float64, 2D/3D, odd/singleton shapes, mixed signed zeros, palette overflow, median widths 2/3/4/5/7, Gaussian sigmas 0/.3/1/2/4, and both boundary modes.
- **1,068 saved regression arrays** from before the first optimization pass still match exactly in array values, including random transforms and consumers of the shared Gaussian helper.
- The 75 added tests use SciPy references for palette and continuous-valued median, Gaussian boundaries, constant-value bit preservation, and overflow after the initial palette probe. Median references compare exactly; Gaussian SciPy comparisons use `rtol=atol=2e-15`. Native-before/native-after comparisons use exact bytes without tolerance.
- No change to filter radius, RNG, arithmetic precision, tap accumulation order, or input ownership semantics. The histogram returns original palette values. Its general fallback preserves the prior selection behavior, including signed-zero ties.
- Linux x86-64 was tested. Code uses portable C++20 and compiler vectorization; macOS/arm64 performance and runtime validation remain untested. The 16-output block size was tuned on this CPU.
- The large median speedup applies only to small palettes. Continuous-valued median remains dominated by selection, and tiny Gaussian kernels have little work over which to amortize setup. The tables retain those slower cases.

Verification commands from the repository root:

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=vc_reader/src:vesuvius/src \
  spiral-fitting/.venv/bin/python -m pytest vc_reader/tests \
  vesuvius/tests/test_native_augmentation.py -m 'not network' -q --tb=short
AGENTS_AGENT_MODE=1 PYTHONPATH=vc_reader/src vc_reader/.venv/bin/python \
  /tmp/vc-filter-baseline/compare_filters.py /tmp/vc-filter-baseline/outputs.npz
AGENTS_AGENT_MODE=1 PYTHONPATH=vc_reader/src vc_reader/.venv/bin/python \
  vc_reader/benchmarks/results/check_regression.py \
  vc_reader/benchmarks/results/augment-regression-before.npz
```
