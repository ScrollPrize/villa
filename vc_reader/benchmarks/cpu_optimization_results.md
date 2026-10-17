# CPU augmentation follow-up: paired noise and image-precision kernels

Measured October 16, 2026 (America/Chicago), against source commit
`996a4a721`. This pass targets the remaining CPU costs before considering CUDA.
The user explicitly allowed paired Box–Muller draws and float32 computation
where double has no numerical purpose. RandomBinaryOperator is unchanged.

## Changes and precision

Implementation: `cpp/augment/core.hpp` contains paired draws and typed
convolution; `intensity.hpp` dispatches by image dtype and fuses sharpening;
`median.hpp` provides the reusable exact selector. Regression coverage is in
`tests/test_augment_optimization.py`, with runtime behavior documented in
`docs/augmentation.md`.

- GaussianNoise and RicianNoise consume both outputs of each Box–Muller
  transform. Transcendental math follows image precision. This changes their
  seeded outputs, including old replay plans containing noise seeds. Parameter
  sampling and Elastic control-lattice draws continue using the original RNG
  method. Replay, shared noise and repeated seeds remain deterministic.
- GaussianBlur, LocalSmoothing, Sharpen, MedianFilter and Gamma use float32
  working buffers for float32 images. Convolution and sharpening arithmetic
  also use float32. Gamma/LocalGamma use float32 powers on ordinary normalized
  intensities. Float64 images retain double computation.
- Sharpen combines the Laplacian stencil and final update in one pass and skips
  the min/max scan when clamping is disabled. Axis subtraction order is retained.
- Continuous 3×3×3 median filtering uses a generated, pruned bitonic selection
  network over blocks of 32 output voxels. It selects exact input values without
  quantization. Quantized inputs retain the sliding histogram; other window
  sizes, 2D inputs and mixed signed-zero ties retain the existing fallback.
  Palette hashing now follows image dtype as well.

Double remains for large reductions/statistics, intensity ranges, Gaussian
coefficient normalization and local-field construction (whose differences of
nearby `erf` values can lose precision). Geometry retains double coordinates,
interpolation and Elastic's normalization/gradient bound: these are shared with
label rounding, conservative validity interpolation and invertibility checks.
Rician squared magnitudes use double to avoid overflowing on large finite
float32 inputs. Extreme filter inputs/strengths and gamma base/exponent
conversion underflow use double fallbacks, as do gamma exponents above 16
(sensitive to base rounding near one). No fast-math, approximate math
functions, architecture-specific intrinsics or internal worker threads were
introduced.

## Measurement

Intel Core Ultra 7 270K Plus, Linux x86-64, CPU affinity 0, GCC 15.2, Python
3.14.4. Both extensions are Release, `-O3 -g -DNDEBUG`, C++20. Benchmark jobs,
builds and tests ran separately. The profiling runs are excluded from timings.

The workload is the same six PHercParis4 level-2 regions in
[the ranking report](native_augmentation_results.md#crop-details): one float32
channel from uint8 CT divided by 255, with int16 threshold masks for Elastic.
The local source is:

```text
/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2
```

There are two warmups and eleven measured calls per crop, **66 observations per
case/size**, using seeds 123–133 and deterministic case shuffling. Gamma and
LocalGamma are a separate matched two-case run with the same protocol.
The noisy-input median run uses fixed Gaussian perturbations, std .01 and seed
123, one warmup and five repetitions per crop (**30 observations**).

Timings include the public NumPy API, validation, sampling, native kernels,
allocation and input-preserving ownership. They exclude reading, normalization,
hashing and output destruction. Each case starts from the original crop.
Parameters match the original suite: median size 3, noise std .05, blur sigma 1,
LocalSmoothing sigma 1/strength .7, Sharpen strength .25 with no clamping, and
Elastic sigma 3/magnitude 1. All constructor arguments are retained in JSON.

All times below are **mean / p50 / p95 milliseconds**. Speedups use p50.

### 128³

| Case | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| MedianFilter, CT + noise | 498.982 / 501.028 / 505.384 | 66.727 / 66.709 / 67.190 | 7.51× |
| Sharpen | 23.608 / 24.002 / 24.940 | 4.194 / 3.578 / 8.820 | 6.71× |
| GaussianNoise | 37.140 / 36.946 / 39.033 | 9.484 / 9.333 / 9.648 | 3.96× |
| GaussianBlur | 21.450 / 21.747 / 22.364 | 5.969 / 5.569 / 6.911 | 3.90× |
| RicianNoise | 73.063 / 72.945 / 73.312 | 19.186 / 19.075 / 19.290 | 3.82× |
| LocalSmoothing | 43.439 / 43.973 / 44.505 | 26.712 / 27.849 / 28.822 | 1.58× |
| Gamma | 17.432 / 17.402 / 17.968 | 13.315 / 13.308 / 13.421 | 1.31× |
| LocalGamma | 21.257 / 21.183 / 21.832 | 17.634 / 17.600 / 17.909 | 1.20× |
| MedianFilter, original CT | 80.568 / 79.773 / 87.586 | 77.998 / 77.210 / 80.711 | 1.03× |
| Elastic (unchanged control) | 83.467 / 83.230 / 85.456 | 82.710 / 82.505 / 84.128 | 1.01× |

Quantized median is approximately unchanged; the small difference is not a
strong speedup claim. Sharpen has a substantial p95 tail. These are pooled
observations across different crops, not confidence intervals.

### 64³

| Case | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| GaussianNoise | 4.587 / 4.581 / 4.633 | 1.143 / 1.144 / 1.173 | 4.01× |
| RicianNoise | 9.079 / 9.077 / 9.108 | 2.358 / 2.355 / 2.386 | 3.86× |
| Sharpen | 1.152 / 1.147 / 1.192 | 0.368 / 0.368 / 0.392 | 3.12× |
| GaussianBlur | 1.098 / 1.095 / 1.165 | 0.586 / 0.586 / 0.609 | 1.87× |
| LocalSmoothing | 1.908 / 1.885 / 2.049 | 1.259 / 1.255 / 1.311 | 1.50× |
| Gamma | 2.056 / 2.039 / 2.137 | 1.569 / 1.567 / 1.597 | 1.30× |
| LocalGamma | 2.590 / 2.570 / 2.669 | 2.127 / 2.125 / 2.148 | 1.21× |
| MedianFilter | 9.536 / 9.434 / 10.316 | 9.513 / 9.339 / 10.256 | 1.01× |
| Elastic (unchanged control) | 8.103 / 8.109 / 8.221 | 8.094 / 8.107 / 8.170 | 1.00× |

## Correctness

The real-crop cross-build comparison uses all six 128³ regions at seed 123.
Maximum observed absolute errors on normalized images:

| Transform | Maximum absolute error |
| --- | ---: |
| GaussianBlur | 2.98023224e-7 |
| Sharpen | 2.38418579e-7 |
| LocalSmoothing | 1.78813934e-7 |
| Gamma, LocalGamma | 5.96046448e-8 |
| MedianFilter, Elastic | 0 |

The comparison uses `atol=1e-6, rtol=1e-6`; these are observed errors on this
workload, not universal guarantees. Median and Elastic output hashes match
exactly on all six 128³ and 64³ crops, including Elastic's integer labels.
Median hashes also match exactly on all six noisy 128³ inputs. Noise outputs
intentionally differ; independent checks cover Gaussian mean/standard
deviation, pair correlation, Rician second moments and the relation between
Gaussian pairs and Rician magnitudes.

**758 augmentation tests passed.** Coverage includes float32/float64, replay/pickle, shared noise,
odd noise buffer lengths, SciPy references, median block tails and reflected
boundaries, signed-zero ties, large finite inputs, gamma statistics and extreme
exponents. Performance is measured only on Linux x86-64; macOS/arm64, worker
concurrency and end-to-end training throughput were not measured.

## Reproduction

The preserved baseline is `/tmp/vc-augment-cpu2-before`; the candidate build is
`/tmp/vc-augment-cpu2`. Both include the measured extension and recorded build
flags. No packages were installed. From the repository root:

```bash
AGENTS_AGENT_MODE=1 cmake -S vc_reader -B /tmp/vc-augment-cpu2 \
  -DVC_READER_BUILD_READER=OFF \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O3 -g -DNDEBUG' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-augment-cpu2 --target _augment -j 2

export AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=vc_reader/src:vc_reader/.venv/lib/python3.14/site-packages
bench() {
  build=$1; tag=$2; shift 2
  taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/run_native.py \
    "$build/_augment.cpython-314-x86_64-linux-gnu.so" \
    vc_reader/benchmarks/bench_augment.py --native-only \
    --only Elastic,MedianFilter,RicianNoise,GaussianNoise,GaussianBlur,LocalSmoothing,Sharpen \
    --build-dir "$build" --reps 11 --warmup 2 \
    --output "vc_reader/benchmarks/results/cpu2-$tag.json" "$@"
}
for tag in before after; do
  build=/tmp/vc-augment-cpu2
  if [ "$tag" = before ]; then build=/tmp/vc-augment-cpu2-before; fi
  bench "$build" "$tag-128"
  bench "$build" "$tag-64" --shape 64 64 64
  bench "$build" "$tag-noisy-128" --only MedianFilter --input-noise-std .01 --reps 5 --warmup 1
  bench "$build" "$tag-gamma-128" --only Gamma,LocalGamma
  bench "$build" "$tag-gamma-64" --only Gamma,LocalGamma --shape 64 64 64
done
```

`taskset` and the extension filename are host-specific. Raw `cpu2-*.json`
reports in the ignored `results/` directory contain every observation, binary
hash, input/output hashes, arguments, CPU affinity and build flags.

Tests append site-packages after the standard library to avoid this environment's
old `argparse` backport:

```bash
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=vc_reader/src vc_reader/.venv/bin/python -S - <<'PY'
import runpy
import sys
sys.path.append('vc_reader/.venv/lib/python3.14/site-packages')
sys.argv = ['vc_reader/benchmarks/run_native.py',
            '/tmp/vc-augment-cpu2/_augment.cpython-314-x86_64-linux-gnu.so',
            '-m', 'pytest', 'vc_reader/tests/test_augment.py',
            'vc_reader/tests/test_augment_optimization.py',
            'vc_reader/tests/test_elastic_bspline.py',
            'vc_reader/tests/test_augment_benchmark.py',
            'vc_reader/tests/test_sheet_compression.py', '-q', '--tb=short']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
```

Use `check_augment_build.py` through `run_native.py` with the baseline to capture
reference outputs, then the candidate to compare. The matched arguments used
here are `--only Elastic,MedianFilter,GaussianBlur,LocalSmoothing,Sharpen --seeds
123 --atol 1e-6 --rtol 1e-6`, and separately `--only Gamma,LocalGamma --seeds 123`
with the same tolerances. Reports are `cpu2-equivalence.json` and
`cpu2-gamma-equivalence.json`.

## Sampling profiles

Separate `gprofng collect app -p hi` runs use all nine cases on the same six 128³
crops, seven repetitions and two warmups. They include validation, hashing and
startup, so exclusive CPU samples are not per-transform latency fractions.
The baseline's leading identified costs are the median histogram (17.78%),
RicianNoise apply (7.84%), nearest label resampling (5.48%), image interpolation
(5.44%), Gaussian convolution (4.01%) and GaussianNoise apply (3.72%). Another
19.02% is at an unresolved libm address; it is not attributed to a particular
operation.

After the changes, the leading identified costs are the median histogram
(26.73%), nearest label resampling (8.31%), image interpolation (8.02%),
RicianNoise apply (5.23%) and LocalGamma apply (4.08%). Gaussian convolution is
2.91%. The histogram's sampled CPU time is essentially unchanged (4.797 versus
4.846 seconds); its share rises because the other operations became faster.
This leaves quantized median and Elastic resampling as the main remaining
targets. The unresolved libm address falls to 4.67%, but cannot be attributed
more specifically from this profile.

Symbol-bearing extensions are relinked from the same Release objects with only
`-Wl,-s` removed, as described in the original ranking report. The baseline
symbols in `/tmp/vc-augment-elastic/symbols` have the same `.text` SHA-256 as
the preserved baseline. Relink the candidate into its `symbols/` directory,
then run (with the environment above):

```bash
gprofng collect app -p hi -O /tmp/vc-cpu2-after-profile.er \
  taskset -c 0 vc_reader/.venv/bin/python -S vc_reader/benchmarks/run_native.py \
  /tmp/vc-augment-cpu2/symbols/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_augment.py --native-only \
  --only Elastic,MedianFilter,RicianNoise,GaussianNoise,GaussianBlur,LocalSmoothing,Sharpen,Gamma,LocalGamma \
  --build-dir /tmp/vc-augment-cpu2 --reps 7 --warmup 2 \
  --output vc_reader/benchmarks/results/cpu2-after-profile.json
gprofng display text -limit 25 -functions /tmp/vc-cpu2-after-profile.er \
  > vc_reader/benchmarks/results/cpu2-after-profile.txt
```

For the before run, substitute `/tmp/vc-augment-elastic` and `before` in the
paths above. Both `cpu2-{before,after}-profile.{json,txt}` reports are retained.
Whole-file hashes differ for the symbol-bearing relinks; latency reports use
the original stripped extensions and record matching build hashes.
