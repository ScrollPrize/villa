# Augmentation optimization follow-up — 2026-10-16

This pass follows the [median/Gaussian work](filter_results.md). On a real 128³
PHercParis4 CT crop, affine resampling fell from **91.53 to 44.54 ms**, sheet
compression from **135.02 to 94.37 ms**, morphology from **20.09 to 12.67 ms**,
region encoding from **4.04 to 0.89 ms**, and one-hot encoding from **6.31 to
2.58 ms** (medians). The corresponding Python implementations took **107.37,
268.10, 13.98, 2.77 and 5.27 ms** in the final run. All 38 successful real-data
benchmark cases retain byte-identical native outputs at both measured sizes.

## Changes and scope

- `cpp/augment/spatial.hpp`: iterate coordinates directly instead of dividing
  every voxel index; use a bounds-check-free interpolation stencil when all
  corners are inside. The boundary path, corner order, weight multiplication
  order, double accumulators, nearest ties, and integer label copies are intact.
  Affine, elastic, sheet compression, and the shared tracing geometry benefit.
- `cpp/augment/intensity.hpp`: build local Gaussian fields in a single pass,
  retaining the original axis multiplication order. Gaussian/Rician noise and
  local brightness/contrast/gamma operate on typed channel views instead of
  staging entire double volumes; each voxel still uses double arithmetic and
  one final cast. Constant blank rectangles fill contiguous rows without index
  and value buffers. Statistical rectangles retain their original reductions.
- `cpp/augment/labels.hpp`: small morphology footprints combine contiguous byte
  rows, permitting compiler vectorization; footprints larger than 32 entries
  retain per-voxel early termination. Move the working volume into morphology
  and write its binary result times the original channel maximum directly.
  Region/one-hot encoding specializes singleton label groups and avoids copies
  and initialization of buffers that are entirely overwritten.
- `cpp/augment/core.hpp`: share checked scalar conversion and dispatch integer
  channel writeback once per channel. Integer range checks remain in the generic
  writeback path. Morphology's specialized output is provably either zero or the
  already representable original maximum, and preserves floating signed zero.
- Benchmarks can load a saved `_augment` build through `run_native.py`, record
  exact output hashes, and profile the additional expensive native kernels.
  `tests/test_augment_optimization.py` adds independent SciPy checks for 2D/3D
  interpolation, small/large footprints, borders, integer/float output, signed
  zero and separable local fields.

The pass reviewed and measured the complete existing 39-case models harness.
Filters, simple brightness/contrast/gamma, copying, exact geometry, resolution
simulation, components, masking and target utilities are retained in the tables
as controls even when unchanged. No RNG algorithm, precision, accumulation
order, dependency, internal thread, SIMD intrinsic or fast-math flag was added.
Probability wrappers have no separate voxel kernel. Specialized tracing APIs
were reviewed but are not independently benchmarked with this CT-only workload;
surface perturbation/crop planning code is unchanged.

## Workload and method

- Array: `/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2`.
- 128³ starts at ZYX `(1280,2688,3200)`; 64³ starts at `(1312,2752,3232)`.
  One float32 CT channel, uint8 values divided by 255. Label workloads use
  `CT > .5` int16 masks, **not ground-truth annotations**.
- Linux x86-64, Intel Core Ultra 7 270K Plus, affinity CPU 0, GCC 15.2.0.
  Release `-O3 -DNDEBUG -std=gnu++20`, portable default architecture flags.
  Python 3.14.4, Torch 2.14.1+cu130 on CPU, NumPy 2.5.4, SciPy 1.18.1.
- Main 128³ and 64³ comparisons: 3 warmups, 11 observations per backend/case.
  Reader comparison: 2 warmups, 7 observations. Isolated follow-ups: 4 warmups,
  21 observations. One Torch/OMP/MKL/OpenBLAS thread. Backend order alternates
  within each run; baseline and optimized runs execute sequentially.
- Fixed native seed 123, with Python/NumPy/Torch RNG reset outside each timer.
  Transform construction, volume reads, output comparisons and hashing are
  untimed except for the explicitly labeled reader workloads. Python receives
  a timed input clone; native preserves inputs internally. Both return owning
  results. Python blur uses the existing direct convolution, autotuning off.
- Raw JSON retains all observations, commands, build flags, versions, extension
  hashes, differences, and output hashes in the ignored local directory
  `benchmarks/results/optimization-pass-20261016/`.

Baseline source: `01fcdcc5689f35e1b4b3502006a63a16603241ba`. Baseline extension SHA-256:
`af1cd99f5eb9c374c409faded8bff6d16ad70d1472467a379815b714eed356a4`. Optimized extension SHA-256:
`cdeb85fa6ce723bea855bb0a97d2b0139039948d17b88e1f9b6999d67c8beeda`. The final harness confirms the imported extension
matches the Release build. The baseline is the extension saved before editing.

## 128³, all workloads

Times are **mean / p50 / p95 milliseconds**. Ratios compare p50; values above
one indicate faster optimized native execution. Python columns use the final
run. The two builds' full measurements remain separate in the raw JSON.

| Transform | Before C++ | After C++ | Python | Native improvement | Python / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| CopyOnly | 0.765 / 0.769 / 0.803 | 0.747 / 0.753 / 0.769 | 0.511 / 0.511 / 0.513 | 1.02× | 0.68× |
| BrightnessMultiply | 0.977 / 0.999 / 1.053 | 0.943 / 0.961 / 0.994 | 0.571 / 0.575 / 0.586 | 1.04× | 0.60× |
| BrightnessAdd | 1.033 / 1.045 / 1.113 | 1.016 / 1.017 / 1.084 | 0.572 / 0.575 / 0.598 | 1.03× | 0.57× |
| Contrast | 4.856 / 4.880 / 4.908 | 4.854 / 4.830 / 5.028 | 2.787 / 2.764 / 2.916 | 1.01× | 0.57× |
| Gamma | 11.202 / 11.273 / 11.409 | 11.154 / 11.249 / 11.479 | 20.529 / 20.566 / 20.724 | 1.00× | 1.83× |
| Invert | 1.990 / 2.027 / 2.200 | 1.982 / 2.036 / 2.182 | 1.285 / 1.275 / 1.412 | 1.00× | 0.63× |
| GaussianNoise | 41.576 / 43.906 / 44.217 | 37.389 / 37.395 / 37.605 | 14.149 / 14.272 / 14.487 | 1.17× | 0.38× |
| RicianNoise | 75.155 / 75.221 / 75.682 | 73.372 / 73.435 / 73.518 | 16.241 / 16.290 / 16.472 | 1.02× | 0.22× |
| ClipOutliers | 17.450 / 15.214 / 20.689 | 17.251 / 14.904 / 20.549 | 10.052 / 11.657 / 11.812 | 1.02× | 0.78× |
| GaussianBlur | 17.756 / 14.504 / 22.461 | 17.769 / 14.650 / 22.267 | 71.098 / 73.929 / 74.287 | 0.99× | 5.05× |
| MedianFilter | 44.138 / 44.192 / 44.395 | 43.972 / 44.080 / 44.229 | 172.102 / 171.870 / 173.067 | 1.00× | 3.90× |
| Sharpen | 22.243 / 21.538 / 24.002 | 22.280 / 21.759 / 24.950 | 99.629 / 100.272 / 101.853 | 0.99× | 4.61× |
| BrightnessGradient | 15.103 / 15.172 / 15.506 | 11.182 / 11.199 / 11.487 | 6.111 / 6.112 / 6.177 | 1.35× | 0.55× |
| LocalContrast | 10.566 / 10.672 / 10.970 | 9.383 / 11.889 / 12.229 | 10.197 / 9.306 / 11.733 | 0.90× | 0.78× |
| LocalGamma | 17.981 / 17.999 / 18.211 | 14.189 / 14.269 / 14.690 | 10.221 / 10.125 / 10.642 | 1.26× | 0.71× |
| LocalSmoothing | 31.116 / 30.829 / 31.780 | 41.605 / 41.565 / 41.898 | 31.967 / 32.007 / 32.222 | 0.74× | 0.77× |
| BlankRectangle | 0.768 / 0.749 / 0.914 | 0.571 / 0.573 / 0.604 | 0.754 / 0.761 / 0.770 | 1.31× | 1.33× |
| SliceIllumination | 1.086 / 1.162 / 1.191 | 1.066 / 1.133 / 1.172 | 1.111 / 1.147 / 1.164 | 1.03× | 1.01× |
| Smear | 4.221 / 4.183 / 4.521 | 3.963 / 3.937 / 4.187 | 3.605 / 3.598 / 3.670 | 1.06× | 0.91× |
| Flip | 1.031 / 1.080 / 1.184 | 1.057 / 1.108 / 1.197 | 1.112 / 1.115 / 1.130 | 0.97× | 1.01× |
| Rotate90 | 1.935 / 1.904 / 2.137 | 1.867 / 1.903 / 2.023 | 1.195 / 1.174 / 1.366 | 1.00× | 0.62× |
| Transpose | 1.824 / 1.837 / 2.013 | 1.824 / 1.825 / 2.044 | 1.968 / 1.978 / 2.060 | 1.01× | 1.08× |
| CropPad | 0.682 / 0.677 / 0.752 | 0.656 / 0.665 / 0.683 | 0.552 / 0.555 / 0.557 | 1.02× | 0.83× |
| Affine | 91.565 / 91.534 / 91.827 | 44.641 / 44.539 / 45.079 | 107.538 / 107.366 / 108.331 | 2.06× | 2.41× |
| Elastic | 281.611 / 281.597 / 282.308 | 244.160 / 245.767 / 247.420 | 643.593 / 643.299 / 647.982 | 1.15× | 2.62× |
| SheetCompression | 135.173 / 135.023 / 135.966 | 94.365 / 94.372 / 94.849 | 268.167 / 268.101 / 270.614 | 1.43× | 2.84× |
| LowResolution | 7.832 / 7.743 / 8.061 | 7.772 / 7.757 / 7.842 | 6.971 / 6.989 / 7.046 | 1.00× | 0.90× |
| ThickSlice | 2.215 / 2.262 / 2.557 | 2.170 / 2.277 / 2.299 | 4.290 / 4.289 / 4.494 | 0.99× | 1.88× |
| Morphology | 20.172 / 20.093 / 20.515 | 12.513 / 12.669 / 13.072 | 13.889 / 13.981 / 14.076 | 1.59× | 1.10× |
| RemoveComponents | 7.209 / 7.245 / 7.573 | 7.513 / 7.518 / 7.980 | 13.869 / 13.907 / 14.136 | 0.96× | 1.85× |
| ReplaceLabel | 1.755 / 1.695 / 1.912 | 1.693 / 1.677 / 1.781 | 1.153 / 1.131 / 1.378 | 1.01× | 0.67× |
| Regions | 4.043 / 4.039 / 4.074 | 0.905 / 0.887 / 1.026 | 2.775 / 2.766 / 2.857 | 4.55× | 3.12× |
| OneHot | 6.266 / 6.315 / 6.471 | 2.501 / 2.583 / 2.747 | 5.267 / 5.274 / 5.424 | 2.45× | 2.04× |
| MaskImage | 2.214 / 2.195 / 2.380 | 2.146 / 2.146 / 2.249 | 1.497 / 1.492 / 1.529 | 1.02× | 0.69× |
| DeepSupervision | 0.889 / 0.887 / 0.959 | 0.861 / 0.866 / 0.885 | 0.929 / 0.925 / 0.962 | 1.02× | 1.07× |
| MedialSurface | 46.324 / 46.253 / 46.838 | 43.968 / 44.014 / 44.607 | 41.080 / 41.191 / 41.289 | 1.05× | 0.94× |
| Pseudo2DRoundTrip | — | — | Python error | — | — |
| RandomBinaryOperator | 13.127 / 15.418 / 15.749 | 11.522 / 13.886 / 14.328 | 41.952 / 41.152 / 44.081 | 1.11× | 2.96× |
| IntensityPipeline | 20.957 / 20.958 / 21.181 | 21.176 / 20.890 / 22.174 | 88.496 / 88.640 / 89.318 | 1.00× | 4.24× |

The local transforms are sensitive to run context. LocalContrast and
LocalSmoothing were slower in the mixed-case 128³ table, so they were repeated
individually in fresh processes. LocalSmoothing is approximately unchanged
when isolated; LocalContrast improves there, but its Python timing also varies.
These runs do **not** establish a robust single 128³ speedup for either transform.
The 64³ and reader results below retain their own observed outcomes.

| Transform | Before C++ | After C++ | Python | Native improvement | Python / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| LocalContrast | 22.322 / 21.869 / 23.692 | 9.404 / 12.065 / 12.439 | 10.080 / 9.658 / 11.182 | 1.81× | 0.80× |

| Transform | Before C++ | After C++ | Python | Native improvement | Python / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| LocalSmoothing | 42.353 / 41.739 / 43.523 | 42.074 / 41.257 / 43.950 | 32.764 / 33.402 / 34.001 | 1.01× | 0.81× |

## 64³, changed kernels and shared consumers

The full 39-case runs are in `before-64.json` and `after-64.json`; the table
below includes changed kernels plus filter/pipeline controls.

| Transform | Before C++ | After C++ | Python | Native improvement | Python / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| GaussianNoise | 5.035 / 5.294 / 5.322 | 4.592 / 4.592 / 4.610 | 1.539 / 1.529 / 1.596 | 1.15× | 0.33× |
| RicianNoise | 9.221 / 9.225 / 9.243 | 9.093 / 9.088 / 9.113 | 1.718 / 1.706 / 1.756 | 1.01× | 0.19× |
| GaussianBlur | 1.101 / 1.101 / 1.118 | 1.094 / 1.095 / 1.100 | 1.706 / 1.702 / 1.750 | 1.00× | 1.55× |
| MedianFilter | 7.239 / 7.241 / 7.297 | 7.270 / 7.275 / 7.333 | 37.501 / 37.508 / 37.633 | 1.00× | 5.16× |
| BrightnessGradient | 1.662 / 1.662 / 1.673 | 1.326 / 1.327 / 1.330 | 0.637 / 0.637 / 0.654 | 1.25× | 0.48× |
| LocalContrast | 1.006 / 1.005 / 1.013 | 0.705 / 0.707 / 0.710 | 0.756 / 0.756 / 0.763 | 1.42× | 1.07× |
| LocalGamma | 2.321 / 2.314 / 2.348 | 2.076 / 2.073 / 2.102 | 1.205 / 1.204 / 1.216 | 1.12× | 0.58× |
| LocalSmoothing | 2.031 / 2.010 / 2.109 | 1.869 / 1.863 / 1.903 | 2.860 / 2.869 / 2.916 | 1.08× | 1.54× |
| BlankRectangle | 0.204 / 0.211 / 0.215 | 0.065 / 0.072 / 0.077 | 0.113 / 0.114 / 0.119 | 2.93× | 1.58× |
| Affine | 11.783 / 11.792 / 11.838 | 5.834 / 5.834 / 5.935 | 8.195 / 8.180 / 8.322 | 2.02× | 1.40× |
| Elastic | 34.326 / 34.312 / 34.449 | 27.072 / 27.088 / 27.115 | 22.668 / 22.828 / 23.152 | 1.27× | 0.84× |
| SheetCompression | 16.495 / 16.519 / 16.651 | 9.865 / 9.874 / 10.009 | 13.606 / 13.535 / 14.150 | 1.67× | 1.37× |
| Morphology | 2.420 / 2.413 / 2.470 | 1.312 / 1.314 / 1.326 | 1.959 / 1.960 / 1.973 | 1.84× | 1.49× |
| Regions | 0.601 / 0.606 / 0.613 | 0.119 / 0.121 / 0.123 | 0.314 / 0.315 / 0.319 | 5.03× | 2.61× |
| OneHot | 0.806 / 0.809 / 0.812 | 0.243 / 0.245 / 0.249 | 0.639 / 0.639 / 0.644 | 3.31× | 2.61× |
| MedialSurface | 3.952 / 3.940 / 4.008 | 3.383 / 3.362 / 3.472 | 5.706 / 5.704 / 5.730 | 1.17× | 1.70× |
| RandomBinaryOperator | 1.130 / 1.126 / 1.152 | 0.876 / 0.873 / 0.901 | 3.767 / 3.769 / 3.803 | 1.29× | 4.32× |
| IntensityPipeline | 2.614 / 2.602 / 2.652 | 2.619 / 2.607 / 2.668 | 3.573 / 3.571 / 3.616 | 1.00× | 1.37× |

## 128³, fresh reader → normalization → augmentation

Both implementations read fresh float32 buffers, divide by 255 and construct
any threshold labels inside the timer. Native uses `inplace=True`; Python
mutates its fresh tensors. The final run records **260 disk hits, zero remote
fetches**, and no reader errors. These are warm local reads, not cold storage
or full training/DataLoader throughput.

| Transform | Before C++ | After C++ | Python | Native improvement | Python / after |
| --- | ---: | ---: | ---: | ---: | ---: |
| GaussianNoise | 38.241 / 38.271 / 38.334 | 36.966 / 36.968 / 37.015 | 5.640 / 5.618 / 5.845 | 1.04× | 0.15× |
| RicianNoise | 75.056 / 75.300 / 75.379 | 73.148 / 73.246 / 73.279 | 17.931 / 17.905 / 18.143 | 1.03× | 0.24× |
| BrightnessGradient | 23.167 / 21.585 / 26.043 | 10.855 / 10.800 / 11.150 | 6.076 / 6.070 / 6.163 | 2.00× | 0.56× |
| LocalContrast | 18.042 / 15.629 / 21.474 | 6.217 / 6.124 / 6.890 | 7.523 / 7.479 / 7.722 | 2.55× | 1.22× |
| LocalGamma | 24.936 / 22.512 / 28.590 | 14.487 / 14.648 / 15.001 | 10.574 / 10.592 / 10.730 | 1.54× | 0.72× |
| LocalSmoothing | 31.593 / 31.434 / 32.189 | 38.671 / 36.420 / 42.236 | 27.035 / 27.714 / 27.847 | 0.86× | 0.76× |
| BlankRectangle | 0.781 / 0.779 / 0.793 | 0.644 / 0.643 / 0.653 | 0.842 / 0.842 / 0.857 | 1.21× | 1.31× |
| Affine | 93.372 / 93.416 / 93.504 | 45.935 / 45.997 / 46.245 | 106.626 / 106.618 / 107.105 | 2.03× | 2.32× |
| Elastic | 298.829 / 298.877 / 299.754 | 226.116 / 226.036 / 226.708 | 596.494 / 597.363 / 598.514 | 1.32× | 2.64× |
| SheetCompression | 149.367 / 149.488 / 149.783 | 95.592 / 95.348 / 96.145 | 222.109 / 222.165 / 223.286 | 1.57× | 2.33× |
| Morphology | 21.481 / 21.497 / 21.937 | 13.405 / 13.427 / 13.769 | 15.206 / 15.263 / 15.527 | 1.60× | 1.14× |
| Regions | 5.340 / 5.367 / 5.427 | 2.172 / 2.195 / 2.235 | 4.080 / 4.056 / 4.317 | 2.44× | 1.85× |
| OneHot | 7.344 / 7.368 / 7.510 | 3.658 / 3.662 / 3.817 | 5.967 / 6.052 / 6.163 | 2.01× | 1.65× |

## Profiling

GNU gprof sampled separate standalone C++ builds (`-O2 -g -pg`), 20 calls per
operation, pinned to CPU 0 on the raw 128³ uint8 chunk `.../2/10/21/25`.
These instrumented profiles locate work; the Release tables above provide
latency measurements. Baseline profiles used the saved original headers.

| Workload | Baseline sampled hotspots | Optimized sampled hotspots |
| --- | --- | --- |
| Affine | coordinate/resample loop 1.02 s (58%); interpolation .64 s (36%) | resample .26 s (41%); interpolation .21 s (33%) |
| Elastic | Gaussian convolution 1.33 s (31%); resample 1.00 s (23%); field sampling .92 s (22%); interpolation .81 s (19%) | Gaussian 1.31 s (44%); sampling .81 s (27%); resample .36 s (12%); interpolation .30 s (10%) |
| Sheet compression | resample 1.29 s (41%); interpolation .65 s (21%); Gaussian .50 s (16%) | Gaussian .47 s (26%); interpolation .36 s (20%); field initialization .34 s (19%); resample .21 s (12%) |
| Local gamma | application .19 s (38%); local field construction .17 s (34%) | field construction .09 s (50%); application .05 s (28%) |
| Morphology | stencil .17 s (52%); application .11 s (33%) | stencil .17 s (52%); application .12 s (36%) |
| Gaussian noise | application .24 s (80%) | application .29 s (94%) |

Total sampled executable CPU time: affine 1.76 → .63 s, elastic 4.26 → 2.99 s,
sheet 3.11 → 1.78 s, local gamma .50 → .18 s, morphology .33 → .33 s,
noise .30 → .31 s. gprof excludes time in shared-library math routines, so
these totals do not measure total noise latency or attribute time to `log`/`cos`.
Its 10 ms sampling resolution makes the smaller profiles coarse. The standalone
morphology profile uses a float binary image, whereas the latency harness uses
int16 labels and also benefits from the specialized integer writeback.
The profile does not establish a morphology or noise speedup on its own.
Checksums match between the two profiling executables for all six workloads.

Commands from the repository root (use saved headers for the baseline):

```sh
mkdir -p /tmp/vcr-aug-pass/after
AGENTS_AGENT_MODE=1 c++ -std=c++20 -O2 -g -pg -I vc_reader/cpp/augment \
  vc_reader/benchmarks/profile_augment.cpp -o /tmp/vcr-aug-pass/after/profile
cd /tmp/vcr-aug-pass/after
for op in affine elastic sheet local morphology noise; do
  taskset -c 0 ./profile \
    /mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2/10/21/25 \
    "$op" 20 > "$op.checksum"
  gprof -b -p ./profile gmon.out > "$op.gprof.txt"
done
```

Raw profiles are retained with `before-`/`after-` prefixes in the results
artifact directory. The compiler-generated variant wrapper names in the
original gprof output correspond to the application functions listed above.

## Correctness and limits

- **1,005 tests passed**, 2 network tests deselected; three existing fork warnings.
  This includes reader tests, augmentation replay/ownership/worker tests and
  Vesuvius dataset integration. The full suite required local socket access for
  loopback HTTP fixtures and IPC. No dependencies were installed.
- **1,068 saved regression arrays match byte-for-byte**, including float32/float64,
  2D/3D, odd/singleton grids, multiple channels, random transforms and geometry.
  All **38 successful cases at each crop size**, and all **13 reader workloads**,
  also have identical native-before/native-after output hashes.
- Python/native differences predate this pass. At 128³, affine and sheet
  compression differ by at most 5.424e-6 and 4.828e-6 in image values; their
  labels are exact. Native output is unchanged. RNG-based cases use independent
  draws across implementations, so their ratios compare work, not identical
  realizations. Medial-surface algorithms differ at 94 of 2,097,152 voxels.
- Python's pseudo-2D round trip still raises
  `AttributeError: 'int' object has no attribute 'shape'` on metadata; no ratio
  is reported. Native pseudo-2D behavior remains covered by tests.
- Gaussian and Rician noise remain substantially slower than Python. Removing
  staging buffers helps Gaussian noise; preserved scalar RNG/transcendental
  work still dominates, especially for Rician noise. Elastic remains slower
  than Python at 64³ despite the native improvement. Several small operations
  and local transforms are also slower than Python; these results are retained.
- Timings depend on crop size, allocator/cache state and the shared host. The
  isolated local-transform observations demonstrate this variation. With 7–21
  samples, p95 is descriptive, not a statistically precise tail estimate.
- Only Linux x86-64 was exercised. The implementation is standard C++20 with
  compiler vectorization and general fallbacks; macOS/arm64 runtime and
  performance remain unverified.

## Build, test and reproduce

From the repository root with the existing dependencies (no installation):

```sh
export AGENTS_AGENT_MODE=1
mkdir -p /tmp/vcr-aug-pass/before
# Save before editing/rebuilding; do not overwrite an existing baseline.
cp -n vc_reader/src/vc_reader/_augment*.so /tmp/vcr-aug-pass/before/
cmake -S vc_reader -B /tmp/vc-augment-build \
  -DVC_READER_BUILD_READER=OFF \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/vc-augment-build --target _augment -j 2
cp /tmp/vc-augment-build/_augment.cpython-314-x86_64-linux-gnu.so vc_reader/src/vc_reader/
PYTHONPATH=vc_reader/src:vesuvius/src spiral-fitting/.venv/bin/python -m pytest \
  vc_reader/tests vesuvius/tests/test_native_augmentation.py -m 'not network' -q --tb=short
```

The ABI suffix is specific to this environment. The actual build reused the
already configured Release CMake directory. For the measured benchmark setup:

```sh
export AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=vc_reader/src:vesuvius/src:vc_reader/.venv/lib/python3.14/site-packages
for version in before after; do
  if [ "$version" = before ]; then
    aug_binary=/tmp/vcr-aug-pass/before/_augment.cpython-314-x86_64-linux-gnu.so
  else
    aug_binary=/tmp/vc-augment-build/_augment.cpython-314-x86_64-linux-gnu.so
  fi
  taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/run_native.py \
    "$aug_binary" vc_reader/benchmarks/bench_augment.py --reps 11 --warmup 3 \
    --output "/tmp/vcr-aug-pass/$version-128.json"
done
```

The recorded 128³ files are named `baseline128.json` and `final128.json`.
`-S` plus the explicit site-packages path bypasses a stale local editable import
hook; the regular source `PYTHONPATH` normally suffices. The benchmark environment
contains the Python reference dependencies; the test command uses the existing
`spiral-fitting` interpreter only, without modifying or building that subproject.
`taskset` is Linux-specific measurement configuration, not a library requirement.

Use the same before/after loop with these additional arguments for other runs:

```sh
# 64³ full comparison, 11 observations / 3 warmups:
--start 1312 2752 3232 --shape 64 64 64
# Fresh-reader comparison, 7 observations / 2 warmups:
--read-each-call --inplace --reps 7 --warmup 2 \
  --only Affine,Elastic,SheetCompression,GaussianNoise,RicianNoise,BrightnessGradient,LocalContrast,LocalGamma,LocalSmoothing,BlankRectangle,Morphology,Regions,OneHot
# Isolated local transforms: separate processes for each transform and build.
--only LocalContrast --reps 21 --warmup 4
--only LocalSmoothing --reps 21 --warmup 4
```

The original snapshot/checker remains in the ignored results directory. Exact
byte checking reuses its workload generation rather than duplicating kernels:

```sh
PYTHONPATH=vc_reader/src vc_reader/.venv/bin/python \
  vc_reader/benchmarks/results/check_regression.py \
  /tmp/vcr-aug-pass/regression.npz \
  /tmp/vcr-aug-pass/before/_augment.cpython-314-x86_64-linux-gnu.so
PYTHONPATH=vc_reader/src vc_reader/.venv/bin/python - <<'PYTHON'
import runpy, sys, numpy as np
sys.argv = ['check_regression.py', '/tmp/vcr-aug-pass/regression.npz']
state = runpy.run_path('vc_reader/benchmarks/results/check_regression.py')
with np.load(sys.argv[1]) as expected:
    for key, value in state['arrays'].items():
        assert value.dtype == expected[key].dtype
        assert value.shape == expected[key].shape
        assert value.tobytes() == expected[key].tobytes(), key
PYTHON
```

The tracked independent references run as part of pytest. To compare real-data
hashes from any matching pair of benchmark JSON files:

```python
import json
before = json.load(open('/tmp/vcr-aug-pass/baseline128.json'))
after = json.load(open('/tmp/vcr-aug-pass/final128.json'))
old = {row['name']: row for row in before['cases']}
for row in after['cases']:
    if 'native_output_sha256' in row:
        assert row['native_output_sha256'] == old[row['name']]['native_output_sha256']
```
