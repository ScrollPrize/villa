# CPU augmentation follow-up: paired noise and image-precision kernels

The latest pass is [High-cost kernels, binary labels and CUDA](#follow-up-high-cost-kernels-and-binary-label-dtypes)
below, measured against the preceding median optimization.

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


## Follow-up: exact byte-code median

Measured October 16, 2026, against `672df617b` (which already includes compact
uint8 pipelines). This is a new baseline; the earlier table predates compact
execution. **There is still useful CPU headroom:** the exact 3×3×3 palette
median is now about 2.8× faster through the public API at both tested sizes.
CUDA was not needed for this pass and was not benchmarked.

### Implementation and numerical behavior

`cpp/augment/median.hpp` now exposes the existing selector as a shared stencil
helper. `intensity.hpp::median_rank_filter` uses it for 3D size-3 byte codes,
so both ordinary images and compact chains benefit. Byte codes are processed
in blocks of 64, with contiguous interior loads and reflected endpoints. The
byte comparator decisions sit outside the lane loops to allow compiler
vectorization without intrinsics or CPU-specific flags. The 2 KiB byte scratch
buffer is local to each call.

Padding the 27 neighbors with five maximum codes still selects the 14th
smallest real value, including when real codes equal 255. Palette values are
decoded only after selection. No additional quantization, changed precision,
random draws, ordering of transforms, threads, dependencies or public API
changes are involved. Wider/2D palette windows keep the histogram. Continuous
inputs keep the original floating selector loop and gather strategy, and
mixed signed zeros keep their original fallback. An attempted shared loop
reorganization regressed continuous inputs; its floating-path changes were
removed before the final measurements.

### Matched measurements

Same host, six PHercParis4 level-2 crops, normalization, affinity, compiler and
Release flags as above: GCC 15.2, `-O3 -g -DNDEBUG`, C++20, one CPU worker.
Builds, tests, latency runs and sampling profiles ran separately. Each size
has two warmups and eleven repetitions per crop (66 observations per case).
The API includes validation, sampling, ownership, palette encoding, filtering
and final float32 decode; disk reads and output destruction are excluded.
Timings are **mean / p50 / p95 milliseconds**; speedups use p50.

#### 128³

| Case | Before | After | p50 speedup |
| --- | ---: | ---: | ---: |
| MedianFilter | 49.755 / 49.162 / 59.153 | 17.771 / 17.184 / 24.109 | 2.86× |
| Elastic | 83.919 / 83.835 / 85.100 | 82.247 / 82.266 / 83.371 | control |
| LocalSmoothing | 28.651 / 29.075 / 29.795 | 27.488 / 27.935 / 28.429 | control |
| RicianNoise | 19.206 / 19.046 / 19.380 | 19.197 / 19.051 / 19.270 | control |

#### 64³

| Case | Before | After | p50 speedup |
| --- | ---: | ---: | ---: |
| MedianFilter | 6.367 / 6.457 / 7.114 | 2.294 / 2.292 / 2.736 | 2.82× |
| Elastic | 8.064 / 8.072 / 8.183 | 8.065 / 8.078 / 8.178 | control |
| LocalSmoothing | 1.261 / 1.257 / 1.274 | 1.257 / 1.257 / 1.278 | control |
| RicianNoise | 2.360 / 2.360 / 2.385 | 2.359 / 2.358 / 2.390 | control |

Elastic, LocalSmoothing and RicianNoise are unchanged controls. Their small
movements are not claimed as improvements. The median p95 spread includes
variation across the six input crops; these are pooled observations, not
confidence intervals.

For continuous input (CT plus fixed Gaussian noise, std .01, seed 123), the
128³ median uses one warmup and five repetitions per crop (30 observations):

| Case | Before | After |
| --- | ---: | ---: |
| MedianFilter, CT + noise | 66.503 / 66.475 / 66.959 | 67.177 / 66.446 / 69.303 |

Its p50 is unchanged; the candidate has a somewhat higher p95 in this run.

#### Complete chains, 128³

`bench_compact_pipeline.py` uses the same six crops, one validation/warmup per
mode and eleven timed calls per crop/mode (66 observations), alternating the
ordinary/compact order. Seed is always 123. The longer chain is median3,
rotate90, transpose, flip, brightness multiply/add, contrast, two Gamma steps
with retained statistics, and invert. All parameters remain in JSON.

| Chain / execution mode | Before | After | p50 speedup |
| --- | ---: | ---: | ---: |
| Median3, ordinary | 50.794 / 50.596 / 57.665 | 18.089 / 17.889 / 22.133 | 2.83× |
| Median3, compact | 50.045 / 49.818 / 57.348 | 17.662 / 17.346 / 21.950 | 2.87× |
| Median + permutations + intensity, ordinary | 96.405 / 96.077 / 102.973 | 64.170 / 64.198 / 67.681 | 1.50× |
| Median + permutations + intensity, compact | 53.585 / 53.263 / 60.823 | 21.811 / 21.542 / 26.037 | 2.47× |
| Median + blur + Gamma, ordinary | 67.494 / 67.448 / 74.369 | 35.136 / 35.015 / 39.086 | 1.93× |
| Median + blur + Gamma, compact | 67.108 / 66.839 / 74.248 | 34.919 / 34.738 / 39.117 | 1.92× |

These are augmentation-chain measurements, not end-to-end training throughput.

### Correctness and limits

**863 augmentation tests passed.** The 40 new cases cover float32/float64,
ordinary/compact execution, widths 1/31/32/33/63/64/65/127/128/129, reflected
edges, maximum codes, nonuniform palettes, input preservation and replay
against SciPy's independent exact median. Existing tests cover continuous
inputs, quantization, 2D/wider windows, signed zeros and extreme ranges.

Cross-build comparison of Elastic and MedianFilter on all six 128³ crops at
seeds 123/124/125 has zero changed values and zero absolute error, with
`atol=rtol=0`. All input/output hashes also match between builds for the four
ranking cases at both sizes and for noisy median at 128³ (seed 123). Each
chain's ordinary and compact outputs are byte-identical on all six crops.

The implementation is portable C++20. Performance was measured only on Linux
x86-64, at the specified shapes and parameters. macOS/arm64, very narrow
volumes, concurrent workers and training throughput were not benchmarked.

### Reproduction of this pass

No packages were installed. The baseline extension, flags and headers are
preserved in `/tmp/vc-augment-cpu3-before`; the candidate is in
`/tmp/vc-augment-cpu3`. Baseline source is commit `672df617b`. Raw reports are
`benchmarks/results/cpu3-*.json` (ignored locally); ranking reports record
binary hashes, build flags, crop hashes and every observation. Commands below
run from the repository root. The explicit Python paths, extension suffix and
`taskset` are specific to this host.

```bash
AGENTS_AGENT_MODE=1 cmake -S vc_reader -B /tmp/vc-augment-cpu3 \
  -DVC_READER_BUILD_READER=OFF \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O3 -g -DNDEBUG' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-augment-cpu3 --target _augment -j 2

# Append site-packages after stdlib to avoid the old argparse backport.
cat > /tmp/vc-cpu3-run <<'SH'
#!/bin/bash
export AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=vc_reader/src
exec taskset -c 0 vc_reader/.venv/bin/python -S -c 'import sys, runpy; sys.path.append("vc_reader/.venv/lib/python3.14/site-packages"); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")' vc_reader/benchmarks/run_native.py "$@"
SH

for tag in before after; do
  build=/tmp/vc-augment-cpu3
  if [ "$tag" = before ]; then build=/tmp/vc-augment-cpu3-before; fi
  extension="$build/_augment.cpython-314-x86_64-linux-gnu.so"
  for edge in 128 64; do
    bash /tmp/vc-cpu3-run "$extension" vc_reader/benchmarks/bench_augment.py \
      --native-only --only Elastic,MedianFilter,LocalSmoothing,RicianNoise \
      --build-dir "$build" --shape "$edge" "$edge" "$edge" --reps 11 --warmup 2 \
      --output "vc_reader/benchmarks/results/cpu3-$tag-$edge.json"
  done
  bash /tmp/vc-cpu3-run "$extension" vc_reader/benchmarks/bench_compact_pipeline.py \
    --cases Median3 Median3_permutations_intensity Median3_blur_gamma --reps 11 \
    --output "vc_reader/benchmarks/results/cpu3-$tag-chains.json"
  bash /tmp/vc-cpu3-run "$extension" vc_reader/benchmarks/bench_augment.py \
    --native-only --only MedianFilter --build-dir "$build" \
    --input-noise-std .01 --reps 5 --warmup 1 \
    --output "vc_reader/benchmarks/results/cpu3-$tag-noisy-128.json"
done

bash /tmp/vc-cpu3-run /tmp/vc-augment-cpu3/_augment.cpython-314-x86_64-linux-gnu.so \
  -m pytest vc_reader/tests/test_augment.py vc_reader/tests/test_augment_optimization.py \
  vc_reader/tests/test_augment_compact.py vc_reader/tests/test_elastic_bspline.py \
  vc_reader/tests/test_augment_benchmark.py vc_reader/tests/test_sheet_compression.py \
  -q --tb=short

# Capture needs a fresh reference directory; the measured capture is retained here.
bash /tmp/vc-cpu3-run /tmp/vc-augment-cpu3-before/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/check_augment_build.py /tmp/vc-cpu3-reference \
  --capture --only Elastic,MedianFilter
bash /tmp/vc-cpu3-run /tmp/vc-augment-cpu3/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/check_augment_build.py /tmp/vc-cpu3-reference \
  --only Elastic,MedianFilter --atol 0 --rtol 0 \
  --output vc_reader/benchmarks/results/cpu3-equivalence.json
```

### Profile and remaining targets

Separate `gprofng collect app -p hi` profiles use only Elastic and MedianFilter,
the same six 128³ crops, seven repetitions and two warmups. They include
startup, validation and hashing, and are excluded from latency measurements.
Both symbol-bearing relinks have exactly the same `.text` bytes as their
corresponding measured Release extensions.

| Identified function | Before exclusive seconds (%) | After exclusive seconds (%) |
| --- | ---: | ---: |
| Palette median (wrapper + selector) | 3.170 (31.80%) | 0.781 (10.33%) |
| Nearest label resampling | 1.504 (15.09%) | 1.522 (20.14%) |
| Image interpolation | 1.487 (14.91%) | 1.499 (19.84%) |
| Float palette encoding | 0.435 (4.36%) | 0.434 (5.74%) |
| B-spline field construction | 0.402 (4.03%) | 0.411 (5.43%) |

The after median row sums the wrapper and newly extracted selector. An
unresolved libc address accounts for 1.188 seconds (15.72%) after the change;
it is not attributed to a particular operation. Percentages are shares of the
entire two-case profiling process, not of one Elastic or median call.

Elastic remains the largest measured individual operation, about 82 ms at
128³. Nearest label resampling and image interpolation together now account
for about 40% of samples in this two-case workload. A bounded integer-floor
rewrite was tested but gave no reliable end-to-end gain and was reverted.
Continuous median remains about 66 ms on noisy CT; LocalSmoothing is about
28 ms and RicianNoise about 19 ms. CPU possibilities are not exhausted, but
Elastic resampling is the clearest next target. A future CUDA comparison
should include transfers and match the current geometry, precision, labels
and validity semantics; this pass makes no CUDA performance claim.

The exact candidate profiling commands were:

```bash
mkdir -p /tmp/vc-augment-cpu3/symbols
/usr/bin/c++ -fPIC -O3 -g -DNDEBUG -shared -Wl,--gc-sections \
  -o /tmp/vc-augment-cpu3/symbols/_augment.cpython-314-x86_64-linux-gnu.so \
  /tmp/vc-augment-cpu3/CMakeFiles/_augment.dir/cpp/augment/bindings.cpp.o \
  /tmp/vc-augment-cpu3/libnanobind-static.a
AGENTS_AGENT_MODE=1 gprofng collect app -p hi -O /tmp/vc-cpu3-after-profile.er \
  bash /tmp/vc-cpu3-run \
  /tmp/vc-augment-cpu3/symbols/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_augment.py --native-only --only Elastic,MedianFilter \
  --build-dir /tmp/vc-augment-cpu3 --reps 7 --warmup 2 \
  --output vc_reader/benchmarks/results/cpu3-after-profile.json
gprofng display text -limit 25 -functions /tmp/vc-cpu3-after-profile.er \
  > vc_reader/benchmarks/results/cpu3-after-profile.txt
```

The baseline was relinked before editing and retained under
`/tmp/vc-augment-cpu3-before/symbols`; its profile substitutes that build and
`before` in the paths above. Raw function summaries are retained alongside
the latency JSON reports.

## Follow-up: high-cost kernels and binary label dtypes

Measured October 16, 2026, against `0a5094037` (the previous median pass).
There is still CPU headroom: continuous median and LocalSmoothing each improve
about 1.29× at 128³. CUDA experiments show substantially more headroom for
median and Elastic, particularly when the **dense field is built on GPU**.

### Production CPU changes

- `median.hpp`: generate an odd-even merge selection network; eliminate known
  padding and comparisons that cannot affect rank 14. Active comparators fall
  from 191 to 126, and required min/max operations from 351 to 226. Floating
  and exact palette-code paths share the schedule. No quantization is added;
  the existing mixed-signed-zero fallback remains.
- `intensity.hpp`: obtain local-field extrema from nonnegative separable axis
  extrema, preserving multiplication order, and normalize during the first
  volume write. Unsafe extrema retain the original full-volume fallback.
  LocalSmoothing blends directly into the writable float32 image instead of
  copying a channel out and back. Float64, overflow fallbacks, input ownership,
  channel selection and replay behavior are preserved.
- `elastic.hpp`: extract the existing control-lattice draws into a shared
  helper for CPU and CUDA experiments. RNG draws and order are unchanged.
  Experimental CPU interpolation rewrites had no reliable public-API gain
  and were discarded; CPU Elastic performance is effectively unchanged.

Same machine, source array and six `DEFAULT_CROPS_128` positions as above:
Intel Core Ultra 7 270K Plus, Ubuntu x86-64, GCC 15.2, Python 3.14.4;
Release `-O3 -g -DNDEBUG`, C++20, no fast-math or architecture flags. CPU 0,
one thread. Inputs are one float32 CT channel /255; Elastic additionally uses
an int16 threshold mask. The 64³ cases use centered subcrops. The continuous
median input adds fixed Gaussian noise (std .01, NumPy seed 123) before timing.

Standalone cases: two warmups and eleven repetitions per crop (66 samples),
seeds 123–133, deterministically shuffled case order. Continuous median: one
warmup and five repetitions (30 samples). Public API timing includes sampling,
validation, allocation and input-preserving execution, excluding I/O,
normalization, hashing and output destruction. Before/after builds run
sequentially. Every cell is **mean / p50 / p95 milliseconds**.

#### 128³ CPU

| Case | Before | After | p50 speedup |
| --- | ---: | ---: | ---: |
| MedianFilter, continuous CT | 66.387 / 66.366 / 66.674 | 51.513 / 51.498 / 51.670 | 1.29× |
| LocalSmoothing | 26.990 / 27.903 / 28.429 | 20.834 / 21.692 / 22.331 | 1.29× |
| MedianFilter | 18.152 / 17.771 / 22.475 | 14.603 / 14.175 / 19.028 | 1.25× |
| LocalGamma | 18.694 / 17.603 / 25.963 | 16.374 / 15.301 / 23.329 | 1.15× |
| Elastic | 82.318 / 82.361 / 82.940 | 82.162 / 82.115 / 83.020 | 1.00× |

#### 64³ CPU

| Case | Before | After | p50 speedup |
| --- | ---: | ---: | ---: |
| MedianFilter, continuous CT | 8.175 / 8.174 / 8.188 | 6.343 / 6.342 / 6.362 | 1.29× |
| LocalSmoothing | 1.258 / 1.256 / 1.278 | 0.885 / 0.883 / 0.904 | 1.42× |
| MedianFilter | 2.537 / 2.533 / 2.979 | 1.849 / 1.844 / 2.300 | 1.37× |
| LocalGamma | 2.129 / 2.127 / 2.155 | 1.826 / 1.824 / 1.846 | 1.17× |
| Elastic | 8.058 / 8.079 / 8.132 | 8.056 / 8.079 / 8.137 | 1.00× |

#### Complete CPU chain, 128³

Image only: Elastic(3,1) → MedianFilter(3) → LocalSmoothing(12, loc=.5,
strength=.7, sigma=1). Includes field sampling and all three transforms, one
warmup and eleven repetitions per crop (66 samples/mode), fixed seed 123.
Ordinary/compact ordering alternates. Compact and ordinary outputs match
byte-for-byte in both builds; deformation makes these median inputs continuous.

| Mode | Before | After |
| --- | ---: | ---: |
| ordinary | 151.055 / 150.985 / 151.859 | 120.364 / 120.222 / 121.103 |
| compact | 151.173 / 151.160 / 152.062 | 120.255 / 120.254 / 120.919 |

### CPU profile

Separate `gprofng` sampling runs use Elastic, median and LocalSmoothing on the
six noisy 128³ crops, one warmup and five repetitions. These profiles include
startup/validation/hashing and are excluded from the latency tables. Both
symbol-bearing relinks have identical `.text` bytes to their corresponding
measured Release extensions.

| Function | Before exclusive seconds (%) | After exclusive seconds (%) |
| --- | ---: | ---: |
| Continuous median selector/stencil | 3.349 (37.70%) | 2.576 (32.46%) |
| Nearest label resampling | 1.127 (12.68%) | 1.108 (13.96%) |
| Image interpolation | 1.060 (11.93%) | 1.066 (13.43%) |
| B-spline field construction | .295 (3.32%) | .301 (3.79%) |
| Gaussian blur | .215 (2.42%) | .220 (2.78%) |
| Local field construction | .062 (.70%) | .037 (.46%) |

The local-field inclusive time drops from .210 to .071 seconds; this includes
helpers such as the removed full-volume extrema scan. Percentage shares are
of the whole profiling process, not one augmentation. Continuous median and
Elastic resampling remain the leading CPU kernels.

For each `build` and `tag` from the reproduction loop below:

```bash
mkdir -p "$build/symbols"
/usr/bin/c++ -fPIC -O3 -g -DNDEBUG -shared -Wl,--gc-sections \
  -o "$build/symbols/_augment.cpython-314-x86_64-linux-gnu.so" \
  "$build/CMakeFiles/_augment.dir/cpp/augment/bindings.cpp.o" \
  "$build/libnanobind-static.a"
AGENTS_AGENT_MODE=1 gprofng collect app -p hi -O "/tmp/vc-cpu4-$tag-profile.er" \
  bash /tmp/vc-cpu3-run "$build/symbols/_augment.cpython-314-x86_64-linux-gnu.so" \
  vc_reader/benchmarks/bench_augment.py --native-only \
  --only Elastic,MedianFilter,LocalSmoothing --input-noise-std .01 \
  --build-dir "$build" --reps 5 --warmup 1 \
  --output "vc_reader/benchmarks/results/cpu4-$tag-profile.json"
gprofng display text -limit 20 -functions "/tmp/vc-cpu4-$tag-profile.er" \
  > "vc_reader/benchmarks/results/cpu4-$tag-profile.txt"
```

The retained final after experiment is `/tmp/vc-cpu4-after-final-profile.er`.
The baseline symbols were preserved from the previous build; relinking a
fresh baseline needs its object and nanobind archive as in the command above.

### Binary label dtype check

The existing label path already preserves bool and integer dtypes. Storing a
binary mask as bool/uint8 uses 2 MiB at 128³ versus 4 MiB for int16 or 16 MiB
for int64. Int8 can also represent a -1 ignore label. Choose a dtype that can
represent all labels and fill values; no implicit narrowing or bitpacking was
added. Smaller storage did **not** speed up this Elastic workload.

Six real CT-derived masks, one validation/warmup and five repetitions per crop
(30 samples/dtype/mode), shuffled order, seeds 123–127. Casts and I/O are
excluded. Every dtype matches the int16 reference, preserves its output dtype,
and leaves the input unchanged. Baseline extension used for this quick check.

| Dtype | Labels only | Image + labels |
| --- | ---: | ---: |
| bool | 55.265 / 55.244 / 55.966 | 83.299 / 83.185 / 84.241 |
| uint8 | 55.193 / 55.191 / 56.142 | 83.400 / 83.378 / 84.481 |
| int8 | 55.080 / 54.998 / 56.057 | 83.297 / 83.281 / 84.325 |
| int16 | 54.361 / 54.326 / 55.159 | 82.648 / 82.624 / 83.417 |
| int32 | 54.481 / 54.417 / 55.453 | 82.499 / 82.521 / 83.250 |
| int64 | 55.969 / 55.286 / 56.447 | 83.864 / 83.783 / 85.366 |

### CUDA feasibility

Optional benchmark (`bench_cuda.cu`, `bench_cuda.py`), **not production
pipeline dispatch**. NVIDIA RTX 5090, driver 595.84, CUDA toolkit 13.3,
`sm_120`, `-O3 --fmad=false`, no fast-math. CUDA access requires running outside
the sandbox on this host. No packages were installed. CPU and CUDA backends
run in a seeded shuffled order in one process, pinned to CPU 0. Six crops,
one warmup per case/crop and five repetitions (30 samples), seeds 123–127.

The first Elastic experiment reused the exact CPU field to isolate resampling.
The second uploads only the small CPU-drawn random control lattice and builds
**all dense B-spline expansion, centering, RMS/gradient normalization and
resampling on GPU**. At 128³ this uploads 375,000 bytes of controls instead of
50,331,648 bytes of dense field. Small B-spline weight tables are initialized
once from the shared production helper. CPU RNG/control draws remain included
in host-to-host timings. This preserves existing draws without requiring GPU
libm to reproduce CPU transcendental functions.

Median uses the production selection schedule compiled into GPU registers.
Elastic uses float64 field values, weights and interpolation accumulation,
float32 image outputs, int16 labels, zero fill and ties-to-even nearest
sampling. GPU field reductions use fixed double-precision trees: deterministic
on this tested GPU, but a different summation order from the CPU.

Host-to-host timings include host output allocation, CPU field/control draws,
pageable H2D/D2H transfers, GPU kernels and synchronization. Device buffers and
context are reused. Kernel timings use CUDA events, exclude transfers and CPU
work, and **include field expansion/normalization** in the GPU-field row.
The prototype omits public schema/finite validation and supports only one
finite float32 channel on a cubic 3D grid with these fixed parameters. Thus
these are feasibility results, not a drop-in production API speedup claim.

#### 128³ CUDA comparison

| Case | CPU public API | CUDA host-to-host | CUDA kernels |
| --- | ---: | ---: | ---: |
| MedianFilter_continuous | 52.937 / 52.741 / 55.296 | 1.174 / 1.160 / 1.203 | 0.095 / 0.098 / 0.103 |
| Elastic | 82.263 / 82.170 / 82.805 | 37.016 / 36.369 / 40.226 | 0.241 / 0.241 / 0.245 |
| Elastic_gpu_field | 82.343 / 82.234 / 83.221 | 3.297 / 3.143 / 3.801 | 0.617 / 0.618 / 0.631 |

#### 64³ CUDA comparison

| Case | CPU public API | CUDA host-to-host | CUDA kernels |
| --- | ---: | ---: | ---: |
| MedianFilter_continuous | 6.539 / 6.444 / 7.054 | 0.255 / 0.268 / 0.293 | 0.053 / 0.068 / 0.074 |
| Elastic | 8.384 / 8.156 / 10.296 | 2.857 / 2.472 / 4.444 | 0.064 / 0.081 / 0.088 |
| Elastic_gpu_field | 8.538 / 8.110 / 10.302 | 0.577 / 0.583 / 0.649 | 0.136 / 0.156 / 0.167 |

All 128³ GPU-field images and labels matched CPU **byte-for-byte** across the
30 timed calls; the maximum double-precision field difference was
1.1102230246251565e-15. The harness also compares fields at 64³ and exercises
1³, 2³ and 7³ grids, Gaussian random inputs, zeros, binary plateaus and signed
labels. Repeated GPU fields for the same seed matched byte-for-byte.
CUDA Compute Sanitizer memcheck on the 7³ synthetic and six-crop workload
reported **0 errors**. Sanitizer timings are excluded from the tables.

These observed image matches are not a guarantee for every input: altered
reduction rounding can affect an interpolation result or a nearest-neighbor
tie. The GPU-field experiment explicitly permits field error up to 5e-12 and
image error up to 2e-7, while requiring labels to be exact; it records actual
errors and unequal counts after every timed pair. The original CPU-field
experiment and median require byte equality. No CPU numerical policy changed.

GPU field construction removes the main limitation of the first experiment:
128³ Elastic falls from about 36 ms with a CPU field to about 3.1 ms with a GPU
field, including transfers. GPU work itself is about .62 ms. Continuous
median is about 1.16 ms including transfers. Keeping consecutive transforms
and training tensors on device is the next integration opportunity; trainer
throughput, other GPUs, macOS/arm64 performance and a production CUDA API have
not been measured here.

### Validation and reproduction

**963 augmentation tests passed** on the final CPU build. Added exact tests
cover local-field separable extrema against the former full-volume reference
(including extreme scales, off-image centers, singleton axes and centering),
continuous median with duplicate extrema, and LocalSmoothing channel selection,
inplace/owning behavior and replay. Cross-build comparison of Elastic,
LocalSmoothing, LocalGamma, BrightnessGradient and LocalContrast on six real
128³ crops and seeds 123–125 found maximum absolute error zero. All standalone
CPU before/after output hashes, including noisy median, match.

Builds are `/tmp/vc-augment-cpu4-before` and `/tmp/vc-augment-cpu4`. Raw JSON,
including every observation, input/output fingerprints and binary identity,
is under ignored `benchmarks/results/cpu4-*`. The 128³ CUDA run used the
candidate just before the behavior-preserving control-draw helper extraction;
final CPU benchmarks/tests and the 64³ CUDA run use the rebuilt final candidate.
Run from the repository root, using the `/tmp/vc-cpu3-run` wrapper defined
[above](#reproduction-of-this-pass), which selects a native extension and fixes
Python paths, CPU affinity and thread environment.

```bash
AGENTS_AGENT_MODE=1 cmake -S vc_reader -B /tmp/vc-augment-cpu4 \
  -DVC_READER_BUILD_READER=OFF \
  -DPython_EXECUTABLE="$PWD/vc_reader/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O3 -g -DNDEBUG' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
AGENTS_AGENT_MODE=1 cmake --build /tmp/vc-augment-cpu4 --target _augment -j 2

for tag in before after; do
  build=/tmp/vc-augment-cpu4
  if [ "$tag" = before ]; then build=/tmp/vc-augment-cpu4-before; fi
  extension="$build/_augment.cpython-314-x86_64-linux-gnu.so"
  for edge in 128 64; do
    bash /tmp/vc-cpu3-run "$extension" vc_reader/benchmarks/bench_augment.py \
      --native-only --only Elastic,MedianFilter,LocalSmoothing,LocalGamma \
      --build-dir "$build" --shape "$edge" "$edge" "$edge" --reps 11 --warmup 2 \
      --output "vc_reader/benchmarks/results/cpu4-$tag-$edge.json"
    bash /tmp/vc-cpu3-run "$extension" vc_reader/benchmarks/bench_augment.py \
      --native-only --only MedianFilter --input-noise-std .01 \
      --build-dir "$build" --shape "$edge" "$edge" "$edge" --reps 5 --warmup 1 \
      --output "vc_reader/benchmarks/results/cpu4-$tag-noisy-$edge.json"
  done
  bash /tmp/vc-cpu3-run "$extension" vc_reader/benchmarks/bench_compact_pipeline.py \
    --cases Elastic_median3_local_smoothing --reps 11 \
    --output "vc_reader/benchmarks/results/cpu4-$tag-chain.json"
done
bash /tmp/vc-cpu3-run /tmp/vc-augment-cpu4-before/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_label_dtypes.py --reps 5 \
  --output vc_reader/benchmarks/results/cpu4-label-dtypes.json

bash /tmp/vc-cpu3-run /tmp/vc-augment-cpu4/_augment.cpython-314-x86_64-linux-gnu.so \
  -m pytest vc_reader/tests/test_augment.py vc_reader/tests/test_augment_optimization.py \
  vc_reader/tests/test_augment_compact.py vc_reader/tests/test_elastic_bspline.py \
  vc_reader/tests/test_augment_benchmark.py vc_reader/tests/test_sheet_compression.py \
  -q --tb=short

# Use a fresh capture directory when reproducing.
bash /tmp/vc-cpu3-run /tmp/vc-augment-cpu4-before/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/check_augment_build.py /tmp/vc-cpu4-reference \
  --capture --only Elastic,LocalSmoothing,LocalGamma,BrightnessGradient,LocalContrast
bash /tmp/vc-cpu3-run /tmp/vc-augment-cpu4/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/check_augment_build.py /tmp/vc-cpu4-reference \
  --only Elastic,LocalSmoothing,LocalGamma,BrightnessGradient,LocalContrast \
  --atol 0 --rtol 0 --output vc_reader/benchmarks/results/cpu4-equivalence.json

# Optional CUDA experiment: sm_120 is specific to the tested RTX 5090.
AGENTS_AGENT_MODE=1 /usr/local/cuda/bin/nvcc -std=c++20 -O3 --fmad=false \
  --expt-relaxed-constexpr -arch=sm_120 -Xcompiler=-O3,-fPIC -shared \
  -Ivc_reader/cpp/augment vc_reader/benchmarks/bench_cuda.cu -o /tmp/vc-bench-cuda.so
for edge in 128 64 1 2 7; do
  bash /tmp/vc-cpu3-run /tmp/vc-augment-cpu4/_augment.cpython-314-x86_64-linux-gnu.so \
    vc_reader/benchmarks/bench_cuda.py --binary /tmp/vc-bench-cuda.so \
    --edge "$edge" --reps 5 --gpu-field \
    --output "vc_reader/benchmarks/results/cpu4-cuda-field-$edge.json"
done

AGENTS_AGENT_MODE=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  --target-processes all --error-exitcode 99 bash /tmp/vc-cpu3-run \
  /tmp/vc-augment-cpu4/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_cuda.py --binary /tmp/vc-bench-cuda.so \
  --edge 7 --reps 1 --gpu-field \
  --output vc_reader/benchmarks/results/cpu4-cuda-memcheck.json
```
