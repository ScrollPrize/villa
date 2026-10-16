# Augmentation performance after optimization — 2026-10-16

This records the first optimization pass. See the [filter follow-up](filter_results.md)
for subsequent median and Gaussian changes; the binaries and timings below are
the historical baseline for that work.

The native brightness → gamma → blur pipeline now takes **27.53 ms**, down from **244.21 ms**, on a real 128³ CT crop. The Python pipeline takes **89.07 ms** in the final comparison. Including a fresh `vc_reader` float32 read, normalization, and augmentation, native takes **27.06 ms** versus Python's **87.65 ms** with `inplace=True`.

All 38 timed native cases improved over the original native implementation at 128³. This does **not** make every transform faster than Torch/SciPy: native wins 12 of 36 individual comparisons at 128³ and 14 at 64³. Noise generation, several local intensity operations, and some simple operations still trail the Python libraries. Median is approximately tied at 128³ after a 12× improvement. Timings below include the remaining slower cases.

## Changes

- `core.hpp`: eliminate error-message construction and repeated shape work inside validation loops; vectorizable finite checks without fast-math; borrowed input buffers with copy-on-write; avoid clearing allocations immediately overwritten; contiguous Gaussian tap loops with unchanged accumulation order.
- `spatial.hpp`: direct row copies/reversals for exact geometry; dtype and dimensionality dispatch outside interpolation loops; direct displacement-grid access; precomputed resize coordinates and weights.
- `intensity.hpp`: direct typed pointwise operations; precomputed median neighbors and smear addresses; contiguous sharpening loops; partial percentile selection; rectangle work limited to the affected box; cheaper local field construction.
- `labels.hpp`: padded binary neighborhoods, precomputed connected-component adjacency, typed region output and bulk one-hot copies.
- Binding and dataset adapter: explicit `inplace=True` for fresh reader buffers, ownership retained through output views, rejection of readonly/unaligned/overlapping inputs. Default calls still preserve inputs and return independent storage.

The RNG, scalar distributions, image precision, and reduction order are preserved. No fast-math, reduced precision, native-CPU compiler flags, extra dependencies, or thread pools were introduced. The Gaussian, median and interpolation kernels remain shared helpers used by their existing callers.

## Measurement setup

- Intel Core Ultra 7 270K Plus, Linux x86-64, Python 3.14.4. All final comparisons and the repeated 128³ baseline are pinned to CPU 0, a performance core. The host is shared; these are local measurements, not cross-platform guarantees.
- Torch 2.14.1+cu130 on CPU, NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0.
- Uninstrumented Release extension, `-O3 -DNDEBUG -std=gnu++20`; one Torch/OMP/MKL/OpenBLAS thread. Two warm-ups, ten observations per backend, alternating backend order; fixed seed 123 reset outside timing.
- Original extension SHA-256: `f799cd4cec4b0bdd69e747643775e5f9d4288e1bf7586e8c7d364b7adc90d3e6`; source baseline commit `bf561bf5fa7d8e2f8225e4a01e60dece94029665`.
- Final extension SHA-256: `01d0f90829bc113ab75b1a7c7af3955c186831f90dd43783038ae60c08dc3339`. All three final reports verify it matches the CMake output.
- One float32 image channel, uint8 CT divided by 255. Label cases use synthetic int16 threshold masks, not annotations. Selected cases add an ignore region.
- Standard mode includes sampling, owning outputs, and NumPy/Torch adaptation. Python clones inputs inside the timer; native preserves inputs internally. Reads, construction, RNG reset, and comparisons are excluded.
- Reader mode includes fresh reads through the same reader, float32 conversion, division by 255, synthetic labels when needed, and augmentation. Python mutates its fresh tensors directly; native receives `inplace=True`. Reader counters: 989 disk hits, zero remote fetches. These are warm local reads with OS caching, not cold I/O or full DataLoader throughput.
- The Python classes execute unchanged. Fixed flip/transpose parameters prevent random no-ops; Gaussian blur uses direct convolution with autotuning disabled. Native uses double intermediates where the original native implementation did; Torch generally uses float32.

Input:

`/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2`

128³ start: `(1280, 2688, 3200)`; 64³ start: `(1312, 2752, 3232)`. Their nonzero fractions are 29.54% and 59.03%; synthetic foreground fractions are 1.44% and 3.48%.

## Build and run

From the repository root, using the existing environments; no installs are needed:

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
  --reps 10 --warmup 2 --output vc_reader/benchmarks/results/augment-final-128.json
taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/bench_augment.py \
  --start 1312 2752 3232 --shape 64 64 64 --reps 10 --warmup 2 \
  --output vc_reader/benchmarks/results/augment-final-64.json
taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/bench_augment.py \
  --read-each-call --inplace --reps 10 --warmup 2 \
  --output vc_reader/benchmarks/results/augment-reader-final-128.json
```

`-S` with an explicit site-packages path avoids this environment's stale editable-reader import hook, ensuring the reader extension in `vc_reader/src` is loaded. This is a local environment workaround; normally `PYTHONPATH=vc_reader/src:vesuvius/src` suffices. `taskset` is Linux-specific benchmark configuration, not a library requirement.

The original binary and Python adapter were saved before optimization in `/tmp/vc-augment-baseline/`. The pinned baseline used the same current harness through a loader selecting those saved modules:

```sh
# Same environment exports as above.
taskset -c 0 vc_reader/.venv/bin/python -S /tmp/vc-augment-baseline/run_baseline.py \
  --reps 10 --warmup 2 \
  --output vc_reader/benchmarks/results/augment-baseline-pinned-128.json
```

The baseline's `native_matches_build=false` is expected: it deliberately loads the saved original extension while the build directory contains the optimized extension. Its SHA above matches the original benchmark's Release binary.

## 128³ before and after

Times are **mean / p50 / p95 milliseconds**. Speedup is original native p50 divided by final native p50. Python timings are from the final run. Ten samples provide only an approximate p95; consult raw samples for variation.

| Transform | Original C++ | Final C++ | Python | Native improvement | Output comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| CopyOnly | 38.283 / 38.275 / 39.116 | 0.749 / 0.748 / 0.772 | 0.511 / 0.511 / 0.512 | 51.14× | matched |
| BrightnessMultiply | 21.056 / 21.005 / 21.582 | 0.935 / 0.936 / 0.983 | 0.576 / 0.579 / 0.592 | 22.43× | matched |
| BrightnessAdd | 21.770 / 21.739 / 22.028 | 1.060 / 1.048 / 1.177 | 0.601 / 0.594 / 0.662 | 20.74× | matched |
| Contrast | 24.394 / 24.357 / 24.706 | 4.905 / 4.858 / 5.109 | 2.821 / 2.829 / 2.932 | 5.01× | matched |
| Gamma | 31.893 / 31.929 / 32.248 | 11.295 / 11.401 / 11.624 | 20.644 / 20.618 / 20.927 | 2.80× | matched |
| Invert | 22.161 / 22.158 / 22.559 | 1.966 / 2.009 / 2.152 | 1.293 / 1.281 / 1.396 | 11.03× | matched |
| GaussianNoise | 60.212 / 60.128 / 63.430 | 41.737 / 41.790 / 44.657 | 11.786 / 10.706 / 13.840 | 1.44× | independent RNG |
| RicianNoise | 93.249 / 93.389 / 93.608 | 76.205 / 76.250 / 76.681 | 16.230 / 16.255 / 16.483 | 1.22× | independent RNG |
| ClipOutliers | 50.734 / 50.795 / 53.646 | 17.540 / 17.540 / 20.672 | 10.283 / 11.629 / 11.945 | 2.90× | matched |
| GaussianBlur | 237.981 / 237.848 / 242.349 | 24.523 / 24.627 / 29.450 | 69.833 / 71.171 / 72.709 | 9.66× | matched |
| MedianFilter | 2167.680 / 2167.618 / 2168.187 | 176.901 / 176.896 / 177.667 | 172.884 / 172.859 / 174.494 | 12.25× | matched |
| Sharpen | 59.670 / 58.957 / 64.269 | 23.748 / 23.707 / 24.048 | 98.175 / 98.247 / 98.796 | 2.49× | matched |
| BrightnessGradient | 56.960 / 56.955 / 57.327 | 15.312 / 15.390 / 15.598 | 6.233 / 6.174 / 6.582 | 3.70× | matched |
| LocalContrast | 59.575 / 59.373 / 62.654 | 10.786 / 10.812 / 11.504 | 7.506 / 7.519 / 7.733 | 5.49× | matched |
| LocalGamma | 59.574 / 59.596 / 59.838 | 27.174 / 27.101 / 28.308 | 14.970 / 15.664 / 15.914 | 2.20× | matched |
| LocalSmoothing | 324.633 / 324.602 / 325.044 | 41.129 / 41.192 / 42.212 | 30.307 / 30.424 / 30.639 | 7.88× | matched |
| BlankRectangle | 96.891 / 96.868 / 97.110 | 0.757 / 0.725 / 0.922 | 0.806 / 0.774 / 0.976 | 133.55× | independent RNG |
| SliceIllumination | 29.631 / 29.714 / 29.935 | 1.033 / 1.025 / 1.152 | 1.128 / 1.151 / 1.171 | 29.00× | independent RNG |
| Smear | 72.179 / 72.136 / 72.358 | 4.025 / 3.993 / 4.222 | 3.580 / 3.584 / 3.611 | 18.07× | matched |
| Flip | 231.046 / 231.065 / 231.468 | 1.113 / 1.144 / 1.371 | 1.051 / 1.035 / 1.111 | 202.06× | matched |
| Rotate90 | 232.204 / 232.227 / 233.042 | 1.865 / 1.888 / 2.015 | 1.222 / 1.203 / 1.318 | 123.00× | matched |
| Transpose | 230.937 / 230.899 / 231.330 | 1.745 / 1.783 / 1.845 | 1.961 / 1.916 / 2.142 | 129.50× | matched |
| CropPad | 197.921 / 198.035 / 198.955 | 0.661 / 0.662 / 0.685 | 0.545 / 0.545 / 0.551 | 299.02× | matched |
| Affine | 254.625 / 254.737 / 255.230 | 92.109 / 92.158 / 92.388 | 84.273 / 84.277 / 84.851 | 2.76× | small FP difference |
| Elastic | 2375.248 / 2374.475 / 2395.657 | 343.143 / 343.201 / 345.301 | 622.847 / 621.375 / 627.188 | 6.92× | independent RNG |
| SheetCompression | 1297.276 / 1298.031 / 1302.529 | 148.874 / 148.767 / 149.911 | 224.084 / 223.900 / 225.624 | 8.73× | small FP difference |
| LowResolution | 194.445 / 194.261 / 195.386 | 7.803 / 7.755 / 7.970 | 6.971 / 6.982 / 7.024 | 25.05× | matched |
| ThickSlice | 331.104 / 329.361 / 339.365 | 2.170 / 2.159 / 2.471 | 4.385 / 4.346 / 4.578 | 152.52× | matched |
| Morphology | 257.108 / 257.031 / 258.109 | 34.746 / 34.827 / 36.416 | 15.043 / 15.916 / 16.038 | 7.38× | matched |
| RemoveComponents | 47.326 / 47.194 / 48.198 | 7.381 / 7.375 / 7.935 | 13.859 / 13.834 / 14.130 | 6.40× | independent RNG |
| ReplaceLabel | 20.268 / 20.251 / 20.344 | 1.748 / 1.694 / 1.936 | 1.135 / 1.132 / 1.239 | 11.96× | matched |
| Regions | 51.953 / 51.968 / 52.063 | 4.039 / 4.037 / 4.060 | 2.873 / 2.863 / 3.076 | 12.87× | matched |
| OneHot | 124.488 / 124.274 / 125.097 | 6.286 / 6.282 / 6.444 | 5.376 / 5.347 / 5.740 | 19.78× | matched |
| MaskImage | 38.853 / 38.857 / 38.925 | 2.200 / 2.187 / 2.300 | 1.537 / 1.518 / 1.704 | 17.77× | matched |
| DeepSupervision | 138.089 / 138.112 / 138.444 | 0.848 / 0.846 / 0.858 | 0.932 / 0.940 / 0.973 | 163.31× | matched |
| MedialSurface | 591.457 / 591.493 / 592.084 | 45.943 / 45.883 / 47.208 | 41.754 / 42.241 / 42.502 | 12.89× | algorithm comparison |
| Pseudo2DRoundTrip | — | — | — | — | Python error |
| RandomBinaryOperator | 196.855 / 196.765 / 201.224 | 25.645 / 25.662 / 27.169 | 45.626 / 46.253 / 46.824 | 7.67× | matched |
| IntensityPipeline | 244.191 / 244.207 / 244.511 | 27.622 / 27.529 / 28.358 | 88.972 / 89.067 / 89.407 | 8.87× | matched |

## 64³ final comparisons

Times are **mean / p50 / p95 milliseconds**. Python/C++ greater than one means native is faster.

| Transform | Python | C++ | Python/C++ | Output comparison |
| --- | ---: | ---: | ---: | --- |
| CopyOnly | 0.049 / 0.049 / 0.053 | 0.076 / 0.077 / 0.080 | 0.64× | matched |
| BrightnessMultiply | 0.074 / 0.078 / 0.086 | 0.096 / 0.096 / 0.103 | 0.82× | matched |
| BrightnessAdd | 0.072 / 0.076 / 0.082 | 0.107 / 0.107 / 0.115 | 0.71× | matched |
| Contrast | 0.320 / 0.325 / 0.332 | 0.608 / 0.608 / 0.618 | 0.53× | matched |
| Gamma | 1.868 / 1.869 / 1.878 | 1.590 / 1.589 / 1.596 | 1.18× | matched |
| Invert | 0.118 / 0.120 / 0.126 | 0.208 / 0.209 / 0.219 | 0.57× | matched |
| GaussianNoise | 1.303 / 1.193 / 1.497 | 5.069 / 5.064 / 5.387 | 0.24× | independent RNG |
| RicianNoise | 1.710 / 1.698 / 1.776 | 9.369 / 9.366 / 9.399 | 0.18× | independent RNG |
| ClipOutliers | 1.717 / 1.861 / 1.874 | 1.962 / 1.979 / 2.261 | 0.94× | matched |
| GaussianBlur | 1.719 / 1.726 / 1.736 | 1.467 / 1.461 / 1.486 | 1.18× | matched |
| MedianFilter | 37.468 / 37.435 / 37.617 | 36.139 / 36.132 / 36.280 | 1.04× | matched |
| Sharpen | 2.418 / 2.386 / 2.561 | 1.129 / 1.120 / 1.171 | 2.13× | matched |
| BrightnessGradient | 0.637 / 0.638 / 0.658 | 1.666 / 1.666 / 1.670 | 0.38× | matched |
| LocalContrast | 0.766 / 0.764 / 0.780 | 1.008 / 1.007 / 1.013 | 0.76× | matched |
| LocalGamma | 1.219 / 1.215 / 1.238 | 2.313 / 2.313 / 2.320 | 0.53× | matched |
| LocalSmoothing | 3.118 / 3.116 / 3.147 | 2.936 / 2.895 / 3.046 | 1.08× | matched |
| BlankRectangle | 0.115 / 0.116 / 0.121 | 0.207 / 0.208 / 0.214 | 0.56× | independent RNG |
| SliceIllumination | 0.249 / 0.250 / 0.257 | 0.106 / 0.108 / 0.116 | 2.31× | independent RNG |
| Smear | 0.941 / 0.943 / 0.960 | 0.523 / 0.518 / 0.550 | 1.82× | matched |
| Flip | 0.122 / 0.122 / 0.125 | 0.107 / 0.107 / 0.113 | 1.14× | matched |
| Rotate90 | 0.165 / 0.165 / 0.168 | 0.179 / 0.178 / 0.185 | 0.93× | matched |
| Transpose | 0.199 / 0.199 / 0.216 | 0.175 / 0.175 / 0.181 | 1.14× | matched |
| CropPad | 0.087 / 0.086 / 0.093 | 0.080 / 0.080 / 0.086 | 1.09× | matched |
| Affine | 8.236 / 8.218 / 8.383 | 11.831 / 11.828 / 11.913 | 0.69× | matched |
| Elastic | 23.212 / 23.319 / 23.506 | 40.668 / 40.634 / 40.810 | 0.57× | independent RNG |
| SheetCompression | 13.427 / 13.455 / 13.785 | 18.277 / 18.261 / 18.391 | 0.74× | matched |
| LowResolution | 0.920 / 0.920 / 0.932 | 0.939 / 0.939 / 0.945 | 0.98× | matched |
| ThickSlice | 0.522 / 0.528 / 0.532 | 0.286 / 0.287 / 0.290 | 1.84× | matched |
| Morphology | 1.957 / 1.964 / 1.972 | 2.400 / 2.408 / 2.427 | 0.82× | matched |
| RemoveComponents | 2.212 / 2.202 / 2.254 | 1.501 / 1.501 / 1.521 | 1.47× | independent RNG |
| ReplaceLabel | 0.143 / 0.143 / 0.150 | 0.230 / 0.229 / 0.233 | 0.62× | matched |
| Regions | 0.320 / 0.322 / 0.326 | 0.599 / 0.598 / 0.614 | 0.54× | matched |
| OneHot | 0.644 / 0.643 / 0.650 | 0.811 / 0.811 / 0.819 | 0.79× | matched |
| MaskImage | 0.188 / 0.188 / 0.196 | 0.265 / 0.263 / 0.278 | 0.72× | matched |
| DeepSupervision | 0.112 / 0.114 / 0.123 | 0.130 / 0.131 / 0.140 | 0.87× | matched |
| MedialSurface | 5.701 / 5.696 / 5.745 | 4.030 / 4.027 / 4.064 | 1.41× | algorithm comparison |
| Pseudo2DRoundTrip | — | — | — | Python error |
| RandomBinaryOperator | 3.658 / 3.656 / 3.714 | 1.131 / 1.126 / 1.150 | 3.25× | matched |
| IntensityPipeline | 3.642 / 3.648 / 3.669 | 2.913 / 2.909 / 2.946 | 1.25× | matched |

## 128³ fresh reader + augmentation

Times are **mean / p50 / p95 milliseconds**. Python/C++ greater than one means native is faster.

| Transform | Python | C++ | Python/C++ | Output comparison |
| --- | ---: | ---: | ---: | --- |
| CopyOnly | 1.202 / 1.198 / 1.216 | 1.417 / 1.409 / 1.452 | 0.85× | matched |
| BrightnessMultiply | 0.654 / 0.654 / 0.661 | 0.950 / 0.949 / 0.956 | 0.69× | matched |
| BrightnessAdd | 0.647 / 0.646 / 0.653 | 1.029 / 1.030 / 1.035 | 0.63× | matched |
| Contrast | 2.836 / 2.821 / 2.889 | 4.849 / 4.834 / 4.929 | 0.58× | matched |
| Gamma | 20.684 / 20.671 / 20.826 | 11.050 / 11.053 / 11.403 | 1.87× | matched |
| Invert | 1.331 / 1.316 / 1.402 | 1.824 / 1.820 / 1.842 | 0.72× | matched |
| GaussianNoise | 5.625 / 5.578 / 5.923 | 38.114 / 38.084 / 38.249 | 0.15× | independent RNG |
| RicianNoise | 17.725 / 17.703 / 18.010 | 75.336 / 75.420 / 75.727 | 0.23× | independent RNG |
| ClipOutliers | 9.373 / 10.082 / 10.472 | 18.689 / 18.689 / 20.146 | 0.54× | matched |
| GaussianBlur | 69.952 / 71.184 / 72.046 | 25.789 / 25.837 / 29.256 | 2.76× | matched |
| MedianFilter | 172.661 / 172.779 / 173.742 | 179.124 / 177.525 / 187.510 | 0.97× | matched |
| Sharpen | 98.189 / 98.190 / 98.788 | 23.581 / 23.405 / 24.200 | 4.20× | matched |
| BrightnessGradient | 8.316 / 9.582 / 9.666 | 23.392 / 23.379 / 25.496 | 0.41× | matched |
| LocalContrast | 7.523 / 7.566 / 7.637 | 10.753 / 10.700 / 11.069 | 0.71× | matched |
| LocalGamma | 14.375 / 14.327 / 14.652 | 28.050 / 28.024 / 28.402 | 0.51× | matched |
| LocalSmoothing | 29.166 / 29.992 / 30.301 | 49.790 / 49.866 / 52.361 | 0.60× | matched |
| BlankRectangle | 0.827 / 0.826 / 0.832 | 0.778 / 0.776 / 0.788 | 1.06× | independent RNG |
| SliceIllumination | 1.174 / 1.164 / 1.221 | 0.978 / 0.983 / 0.999 | 1.18× | independent RNG |
| Smear | 3.621 / 3.626 / 3.686 | 3.895 / 3.866 / 4.003 | 0.94× | matched |
| Flip | 1.759 / 1.756 / 1.773 | 2.017 / 2.012 / 2.060 | 0.87× | matched |
| Rotate90 | 1.842 / 1.834 / 1.892 | 2.898 / 2.892 / 2.962 | 0.63× | matched |
| Transpose | 2.608 / 2.603 / 2.676 | 2.858 / 2.847 / 2.896 | 0.91× | matched |
| CropPad | 1.265 / 1.261 / 1.291 | 1.877 / 1.862 / 1.945 | 0.68× | matched |
| Affine | 108.112 / 108.077 / 108.521 | 93.083 / 93.071 / 93.443 | 1.16× | small FP difference |
| Elastic | 630.586 / 628.453 / 635.921 | 365.459 / 365.381 / 372.080 | 1.72× | independent RNG |
| SheetCompression | 265.145 / 263.282 / 275.102 | 165.146 / 165.035 / 168.561 | 1.60× | small FP difference |
| LowResolution | 7.025 / 6.999 / 7.125 | 7.674 / 7.669 / 7.704 | 0.91× | matched |
| ThickSlice | 4.972 / 4.944 / 5.132 | 3.298 / 3.313 / 3.517 | 1.49× | matched |
| Morphology | 16.136 / 16.911 / 17.027 | 37.062 / 37.141 / 39.168 | 0.46× | matched |
| RemoveComponents | 15.623 / 15.611 / 15.798 | 9.398 / 9.424 / 9.715 | 1.66× | independent RNG |
| ReplaceLabel | 2.210 / 2.185 / 2.328 | 2.707 / 2.698 / 2.760 | 0.81× | matched |
| Regions | 4.205 / 4.185 / 4.400 | 5.471 / 5.429 / 5.686 | 0.77× | matched |
| OneHot | 6.053 / 6.087 / 6.291 | 7.429 / 7.404 / 7.658 | 0.82× | matched |
| MaskImage | 2.149 / 2.166 / 2.180 | 2.753 / 2.740 / 2.799 | 0.79× | matched |
| DeepSupervision | 2.106 / 2.132 / 2.194 | 2.184 / 2.188 / 2.223 | 0.97× | matched |
| MedialSurface | 42.622 / 42.638 / 42.942 | 47.768 / 47.794 / 48.183 | 0.89× | algorithm comparison |
| Pseudo2DRoundTrip | — | — | — | Python error |
| RandomBinaryOperator | 46.833 / 47.585 / 47.873 | 26.658 / 26.690 / 28.160 | 1.78× | matched |
| IntensityPipeline | 87.562 / 87.652 / 88.066 | 27.043 / 27.057 / 27.494 | 3.24× | matched |

## Native profiling

The standalone `profile_augment.cpp` workload ran five median calls on the raw chunk `.../2/10/21/25`, using `-O2 -g -pg`. Before optimization, the median transform body accounted for 92.31% of sampled time (10.09 s), with `std::nth_element` accounting for 6.86% (0.75 s). Per-neighbor division, reflection, and coordinate reconstruction dominated the transform body.

After precomputing neighbor addresses, sampled native CPU time fell from 10.93 s to 0.87 s for the five calls. The median body accounts for 20.69% (0.18 s); the selection helper `std::__introselect` accounts for 79.31% (0.69 s). GNU gprof labels the compiler-generated selection clone as `frame_dummy`; mapping the histogram PCs to `nm -n -C` symbols identifies the actual helper. This profile has only 87 samples after optimization; use the Release latency tables for performance claims.

Commands:

```sh
AGENTS_AGENT_MODE=1 c++ -std=c++20 -O2 -g -pg -I vc_reader/cpp/augment \
  vc_reader/benchmarks/profile_augment.cpp -o /tmp/vc-augment-baseline/profile-after
(cd /tmp/vc-augment-baseline && taskset -c 0 ./profile-after \
  /mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2/10/21/25 median)
gprof -b -p /tmp/vc-augment-baseline/profile-after /tmp/vc-augment-baseline/gmon.out
```

## Correctness and limitations

- **697 passed, 2 network tests deselected** across the reader suite and dataset integration. Tests include replay, multiprocessing workers, large integer labels, float32/float64, 2D/3D, edge padding, odd/singleton dimensions, inplace equivalence, buffer lifetimes, and rejected readonly/unaligned/overlapping arrays. Three existing fork warnings were emitted.
- **1,068 saved output arrays** from the original binary compare exactly in array values with the optimized binary, including random transforms and additional geometry/filter edge cases. No RNG sequence or precision change was used to obtain these gains.
- Existing Python/native differences remain visible: at 128³ affine and sheet-compression image differences reach 5.424e-6 and 4.828e-6 respectively; their labels match exactly. Medial-surface skeletonization differs at 94 of 2,097,152 voxels and remains an algorithm comparison. Independent RNG cases compare workloads, not identical draws.
- The Python pseudo-2D round-trip still raises `AttributeError: 'int' object has no attribute 'shape'` on its own `nchannels` metadata. It has no performance ratio. Native pseudo-2D is covered by tests.
- Specialized tracing transforms and standalone probability wrappers are outside this benchmark's coverage.
- Noise remains expensive with the preserved scalar RNG/transcendental sequence. Local transforms and simple scalar operations retain double calculations and validation that Torch's float32 kernels do not share. The report does not claim every native transform is faster.
- Only Linux x86-64 was exercised here. The changes use standard C++20 and compiler vectorization with no architecture-specific intrinsics; macOS/arm64 runtime validation remains outstanding.

Verification commands:

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=vc_reader/src:vesuvius/src \
  spiral-fitting/.venv/bin/python -m pytest vc_reader/tests \
  vesuvius/tests/test_native_augmentation.py -m 'not network' -q --tb=short
AGENTS_AGENT_MODE=1 PYTHONPATH=vc_reader/src vc_reader/.venv/bin/python \
  vc_reader/benchmarks/results/check_regression.py \
  vc_reader/benchmarks/results/augment-regression-before.npz
```

The full suite needed local HTTP/IPC socket access outside the sandbox. Test fixtures remained local; no packages were installed. The regression snapshot and checker are retained in the ignored `results/` directory.

Raw observations and metadata are retained locally in `results/augment-baseline-pinned-128.json`, `augment-final-128.json`, `augment-final-64.json`, and `augment-reader-final-128.json`. The original unpinned `augment-128.json` and `augment-64.json` are preserved. Native profiles are in `augment-median-before.gprof.txt`, `augment-median-after.gprof.txt`, and `augment-median-after-symbols.txt`.
