# C++ augmentation ranking on real CT

Measured 2026-10-16, source commit `b8fdd7c62d66482423ea65b469ceba56d5931b2d`.
This benchmarks the current vc_reader C++ package only. No augmentation kernels
were changed and no other augmentation implementation was run.

For the selected parameters across all six 128³ crops, the slowest median
calls are SheetCompression, Elastic, MedianFilter and RicianNoise. At 64³,
MedianFilter is slowest among the measured augmentations.
On CT perturbed with fixed noise, MedianFilter rises to **495.32 ms** at 128³
and becomes the strongest optimization candidate among the measured cases.

## Workload and measurement

- Intel Core Ultra 7 270K Plus, Linux x86-64, pinned to CPU 0. One augmentation
  thread, no concurrent builds/tests/benchmark jobs. Python 3.14.4; GCC 15.2.0.
- Fresh Release build: `-O3 -g -DNDEBUG`, C++20, no fast-math. The loader selects
  that exact extension without replacing the installed package. Both main
  reports verify the extension hash matches the recorded build.
- Six real PHercParis4 level-2 CT regions, each measured at 128³ and at a
  centered 64³ subcrop. One float32 image channel, uint8 CT divided by 255.
  The local array is read only:
  `/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2`.
- Label operations use int16 masks derived from `CT > 127.5`, **not annotations**.
  Geometry operates on image plus label; image-only operations receive just the
  image. Mask-corruption augmentations receive a float32 binary mask.
- 26 individual augmentations, three warmups per case/crop, 20 measured repetitions per crop:
  **120 observations per case and size**. Seed is `123 + repetition`, shared
  across crops/cases. Case order is deterministically shuffled per repetition.
- Timed: the public NumPy API including adaptation, native validation, parameter
  sampling, kernels and owning outputs. Untimed: reads, normalization, pipeline
  construction, validation/hashing and output destruction. Default ownership
  preserves inputs; every call starts from the original crop.
- Ranking uses pooled p50 across equally weighted crops. P95 includes both
  timing variation and crop differences; it is not a confidence interval.
  Per-crop observations and mean/p50/p95/min/max are retained in JSON.
  The main samples were collected with two subsequently excluded cases
  (RemoveComponents and ClipOutliers) in the interleaving. Their rows were
  removed without retiming the remaining
  cases; JSON records this collection detail. New runs use only the 26 cases.
- Parameters are fixed workloads, not a sweep of defaults or training
  application probabilities. Notably: Elastic `sigma=3, magnitude=1` (current
  cubic B-spline implementation); SheetCompression `strength=.2,
  intensity_sigma=1, smoothing=3, axes='z'`; MedianFilter `size=3`; RicianNoise
  and GaussianNoise `std=.05`; GaussianBlur `sigma=1`; Affine `scale=1/1.1`.
  Every native constructor is recorded in JSON.

## Full ranking

Times are **mean / p50 / p95 milliseconds per call**, ordered by 128³ p50.

| Case | Inputs | 128³ ms | 64³ ms |
| --- | --- | ---: | ---: |
| SheetCompression | both | 103.019 / 103.434 / 104.858 | 9.323 / 9.349 / 9.514 |
| Elastic | both | 97.700 / 97.630 / 98.791 | 9.494 / 9.499 / 9.607 |
| MedianFilter | image | 87.965 / 88.441 / 89.693 | 10.462 / 10.568 / 10.779 |
| RicianNoise | image | 73.152 / 72.954 / 73.129 | 9.072 / 9.071 / 9.105 |
| Affine | both | 44.431 / 44.287 / 45.229 | 5.767 / 5.765 / 5.805 |
| LocalSmoothing | image | 43.371 / 44.009 / 44.638 | 1.886 / 1.871 / 2.009 |
| GaussianNoise | image | 36.975 / 36.922 / 37.065 | 4.579 / 4.577 / 4.620 |
| RandomBinaryOperator | binary | 27.853 / 28.366 / 30.178 | 1.308 / 1.280 / 1.515 |
| Sharpen | image | 23.273 / 24.011 / 24.993 | 1.141 / 1.138 / 1.171 |
| GaussianBlur | image | 20.795 / 21.807 / 22.391 | 1.088 / 1.081 / 1.136 |
| LocalGamma | image | 21.969 / 21.315 / 27.006 | 2.587 / 2.565 / 2.662 |
| Gamma | image | 17.986 / 17.289 / 23.114 | 2.044 / 2.028 / 2.121 |
| BrightnessGradient | image | 12.528 / 11.861 / 17.824 | 1.367 / 1.365 / 1.399 |
| LowResolution | image | 8.215 / 7.894 / 10.830 | 0.927 / 0.923 / 0.950 |
| LocalContrast | image | 6.704 / 6.340 / 9.277 | 0.686 / 0.679 / 0.714 |
| Contrast | image | 5.901 / 5.558 / 7.012 | 0.710 / 0.664 / 0.888 |
| Smear | image | 4.752 / 4.093 / 9.308 | 0.441 / 0.430 / 0.479 |
| ThickSlice | both | 2.179 / 2.066 / 2.342 | 0.276 / 0.271 / 0.304 |
| Rotate90 | both | 2.086 / 1.878 / 4.476 | 0.183 / 0.181 / 0.213 |
| Invert | image | 1.987 / 1.842 / 2.185 | 0.200 / 0.195 / 0.224 |
| Transpose | both | 1.983 / 1.816 / 2.959 | 0.180 / 0.174 / 0.208 |
| Flip | both | 1.519 / 1.369 / 2.235 | 0.123 / 0.116 / 0.155 |
| BrightnessAdd | image | 1.094 / 1.061 / 1.268 | 0.098 / 0.092 / 0.117 |
| SliceIllumination | image | 1.143 / 0.990 / 1.366 | 0.092 / 0.085 / 0.115 |
| BrightnessMultiply | image | 1.073 / 0.978 / 1.292 | 0.092 / 0.086 / 0.115 |
| BlankRectangle | image | 0.750 / 0.655 / 0.899 | 0.059 / 0.053 / 0.080 |

The `binary` input is a float32 version of the derived mask.

## Continuous-valued filter input

The same six real 128³ CT crops were perturbed once before timing with fixed
NumPy Gaussian noise, seed 123, std 0.01 in normalized intensity units. The
augmentation algorithm and build were unchanged. These are 120 observations
per case and three warmups. A matched two-case control uses the same case order
and seeds without the perturbation. This controls for the different cache and
allocation history of a filter-only run versus the full suite; its GaussianBlur
median is 21.43 ms on original CT and 21.93 ms with the perturbation.

| Filter | Matched original CT control, mean / p50 / p95 ms | CT + fixed noise, mean / p50 / p95 ms |
| --- | ---: | ---: |
| MedianFilter | 88.488 / 88.504 / 93.733 | 493.260 / 495.316 / 499.311 |
| GaussianBlur | 20.463 / 21.428 / 22.066 | 21.933 / 21.929 / 22.411 |

MedianFilter is therefore the first candidate to investigate in pipelines where
noise or interpolation precedes it: its exact continuous-value selection path
is much more expensive than the histogram path on quantized CT. This experiment
measures Gaussian-perturbed CT specifically; it does not establish identical
timings for every interpolation or normalization scheme.

## Crop details

| Group | Size | ZYX start | Nonzero voxels | Derived foreground |
| --- | --- | --- | ---: | ---: |
| Original | 128³ | `(8960, 4480, 3200)` | 100.000% | 21.85% |
| Original | 128³ | `(5120, 3840, 3840)` | 100.000% | 10.25% |
| Original | 128³ | `(5120, 2816, 5120)` | 100.000% | 25.93% |
| Added dense | 128³ | `(4608, 5760, 5888)` | 99.999% | 72.75% |
| Added dense | 128³ | `(4480, 5760, 5248)` | 100.000% | 57.83% |
| Added dense | 128³ | `(4224, 5888, 5120)` | 100.000% | 47.86% |
| Original | 64³ | `(8992, 4512, 3232)` | 100.000% | 20.33% |
| Original | 64³ | `(5152, 3872, 3872)` | 100.000% | 8.15% |
| Original | 64³ | `(5152, 2848, 5152)` | 100.000% | 23.06% |
| Added dense | 64³ | `(4640, 5792, 5920)` | 100.000% | 79.48% |
| Added dense | 64³ | `(4512, 5792, 5280)` | 100.000% | 67.71% |
| Added dense | 64³ | `(4256, 5920, 5152)` | 100.000% | 54.73% |

All twelve inputs contain CT signal; none are empty/fill-only crops.
Input and seeded output SHA-256 hashes are included in the raw reports.

The original three regions are retained, followed by three denser regions.
Discovery sampled 256 stored 128³ chunks with NumPy RNG seed 20261016 and
chunk-index bounds `[8,12,12]` (inclusive) to `[140,52,52]` (exclusive), then
1,024 unique stored neighbors around the eight highest-scoring candidates
with seed 20261017 and offsets `[-8,-5,-5]` to `[9,6,6]` (exclusive).
Candidates were ranked by the minimum foreground fraction of their 128³ crop
and centered 64³ subcrop; the top three were selected. Their 128³ foreground
fractions are 72.8%, 57.8%, and 47.9%, compared with 10.3–25.9% in the original
regions. They are nonoverlapping, nonconstant crops. Foreground here means
`CT > 127.5`, not an anatomical annotation or simply nonzero intensity.
The complete 1,280-candidate record is saved in
`results/native-crop-candidates.json`; central slice/mask previews are saved
in `results/native-dense-crops.png`.

Native mode now defaults to these six centers. Explicit `--start` and
`--additional-start` override the preset.

## Reproduce

Run from the repository root with the existing development environment:

```bash
export AGENTS_AGENT_MODE=1
cmake -S vc_reader -B /tmp/vc-augment-ranking \
  -DVC_READER_BUILD_READER=OFF \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O3 -g -DNDEBUG' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/vc-augment-ranking --target _augment -j 2

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=vc_reader/src:vc_reader/.venv/lib/python3.14/site-packages
bench() {
  taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/run_native.py \
    /tmp/vc-augment-ranking/_augment.cpython-314-x86_64-linux-gnu.so \
    vc_reader/benchmarks/bench_augment.py --native-only \
    --build-dir /tmp/vc-augment-ranking --reps 20 --warmup 3 "$@"
}
bench --shape 128 128 128 --profile all \
  --output vc_reader/benchmarks/results/native-ranking-128.json
bench --shape 64 64 64 --profile all \
  --output vc_reader/benchmarks/results/native-ranking-64.json
bench --shape 128 128 128 \
  --only MedianFilter,GaussianBlur --input-noise-std .01 \
  --output vc_reader/benchmarks/results/native-ranking-continuous-128.json
bench --shape 128 128 128 \
  --only MedianFilter,GaussianBlur \
  --output vc_reader/benchmarks/results/native-ranking-filter-control-128.json
```

No packages were installed. `-S` plus explicit site-packages avoids a stale
editable-reader import hook in this environment. Normally `PYTHONPATH=vc_reader/src`
with a working environment suffices. `taskset` and the extension filename above
are specific to this Linux/Python host; the harness itself also supports macOS.

## Native function profile

A separate `gprofng collect app -p hi` run used the four slowest 128³ cases,
10 repetitions on each of the six crops, and three warmups. It includes startup,
validation and hashing. These are exclusive CPU sample percentages of the
whole profiling process, **not latency fractions for individual transforms**.
The main timing tables do not use profiled observations.

| Native function | Exclusive CPU samples |
| --- | ---: |
| `median_histogram` | 21.46% |
| `linear_value<3>` (image interpolation) | 11.53% |
| `resample_impl<3>` for int16 labels | 10.99% |
| `apply<RicianNoise>` | 8.56% |
| `sample(Elastic)` | 6.33% |
| `gaussian_blur` | 4.41% |
| `sample(SheetCompression)` | 2.92% |

There is also 5.52% in an unresolved libc address and 11.10% in an unresolved
libm address; these are not assigned to particular algorithms here. The profile
supports investigating median histogram maintenance, geometric resampling and
noise generation.

For symbols, the same Release objects were relinked using the generated link
command with `-Wl,-s` removed and the output directed to
`/tmp/vc-augment-ranking/symbols/`. The stripped and symbol-bearing extensions
have identical `.text` SHA-256:
`48dca59ecd432dd1ad6c39cd8b98cd99325db5356d92f1f907d45eac94b3b793`.
To reproduce that relink and profile, after the build above:

```bash
vc_reader/.venv/bin/python -S - <<'PY'
from pathlib import Path
import shlex
import subprocess
build = Path('/tmp/vc-augment-ranking')
dest = build / 'symbols'
dest.mkdir(exist_ok=True)
args = shlex.split((build / 'CMakeFiles/_augment.dir/link.txt').read_text())
args.remove('-Wl,-s')
args[args.index('-o') + 1] = str(dest / '_augment.cpython-314-x86_64-linux-gnu.so')
subprocess.run(args, cwd=build, check=True)
PY
gprofng collect app -p hi -O /tmp/vc-augment-ranking/profile.er \
  taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/run_native.py \
  /tmp/vc-augment-ranking/symbols/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_augment.py --native-only \
  --shape 128 128 128 \
  --only Elastic,SheetCompression,MedianFilter,RicianNoise \
  --reps 10 --warmup 3 --build-dir /tmp/vc-augment-ranking \
  --output vc_reader/benchmarks/results/native-ranking-profile.json
gprofng display text -limit 25 -functions /tmp/vc-augment-ranking/profile.er \
  > vc_reader/benchmarks/results/native-ranking-profile.txt
```

The symbol-bearing extension has a different whole-file hash, so the profiling
JSON correctly reports `native_matches_build=false` against the stripped build.
The main runs also record five separate native per-transform timer calls for
each crop. Those internal timings exclude the API boundary and are diagnostic,
not directly interchangeable with the public-call latencies.

## Coverage and limitations

The 26 cases cover individual image, spatial and mask-corruption augmentations.
Exact case names and constructor arguments are in the raw JSON. CopyOnly,
CropPad, Morphology (label closing), ReplaceLabel, Regions, OneHot, MaskImage,
DeepSupervision, MedialSurface, Pseudo2DRoundTrip and the composite
IntensityPipeline were removed from the benchmark and current results.
RemoveComponents and ClipOutliers are also excluded by request. All package
implementations remain available. RandomBinaryOperator remains as a deliberate
mask perturbation.

StripGeometry, SurfacePerturb and translate_crop_box need tracing/surface inputs
not supplied by these CT crops; they are explicitly excluded. Standalone OneOf
and probability wrappers are also excluded. Dense vectors/normals, actual
annotation masks, multiple channels, float64, alternate parameter ranges,
`inplace=True`, reader throughput and concurrent DataLoader workers were not
measured. Only Linux x86-64 was exercised; no macOS/arm64 numbers are claimed.
Threshold-mask corruption timings need validation on the user's
training labels before treating them as production costs.

The full raw reports and profile are retained in the git-ignored `results/`
directory. This document retains the measurements for repository review.

## Verification

**566 tests passed** against the freshly built augmentation extension, covering
the augmentation suite, optimized kernels, B-spline elastic and the new
benchmark regression tests. The latter run all 26 cases through a tiny local
Zarr fixture with Torch/Vesuvius/SciPy imports blocked, validate multiple crops,
sample counts and ranked output, check that helper cases stay excluded, and
reject unknown case names. Additional tests preserve the original three crop
centers, verify the six-region default at both sizes and check explicit crop
overrides. The actual performance runs use the real CT crops listed above and separately check
finite outputs, exact seeded repeatability and input preservation.

The test command appends site-packages after the standard library: this
environment contains an old `argparse` backport that otherwise shadows the
stdlib under the benchmark's explicit `PYTHONPATH` workaround and breaks pytest.

```bash
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=vc_reader/src vc_reader/.venv/bin/python -S - <<'PY'
import runpy
import sys
sys.path.append('vc_reader/.venv/lib/python3.14/site-packages')
sys.argv = ['vc_reader/benchmarks/run_native.py',
            '/tmp/vc-augment-ranking/_augment.cpython-314-x86_64-linux-gnu.so',
            '-m', 'pytest', 'vc_reader/tests/test_augment.py',
            'vc_reader/tests/test_augment_optimization.py',
            'vc_reader/tests/test_elastic_bspline.py',
            'vc_reader/tests/test_augment_benchmark.py', '-q', '--tb=short']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
```
