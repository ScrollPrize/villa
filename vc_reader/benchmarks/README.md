# vc_reader benchmarks

Compare `vc_reader.reader.ZarrReader` with zarr-python, and with TensorStore
when it is installed, on real Vesuvius Challenge data. Every backend's output
is hashed and compared with the first backend's; a mismatch aborts the run.
Only the reads are timed: hashing, eviction and opening are not. Backends are
interleaved inside each repetition, so they see the same disk and network
conditions. Results are written as JSON to `results/` and printed as a table.

Requirements: the `bench` extra (numpy, zarr>=3, s3fs, tensorstore). With the development venv from the package README:
`.venv/bin/python benchmarks/bench_local.py`. TensorStore is skipped if it is
not installed.

## Augmentation correctness: `validate_augment.py`

This checks a local Zarr crop through the reader and native augmentation, with
NumPy geometry references, exact replay, and input ownership checks. It does not
measure throughput. Labels are optional; provide matching image/label arrays to
also check target alignment. Inputs are read only.

```sh
PYTHONPATH=src python benchmarks/validate_augment.py /path/to/image.zarr \
  --start 1280 2688 3200 --shape 128 128 128 --output /tmp/augmentation-check.npz
```

The optional NPZ contains before/after arrays and a JSON report. See the
[augmentation guide](../docs/augmentation.md) for build requirements and tests.

## C++ augmentation ranking: `bench_augment.py --native-only`

Benchmark only the current C++ augmentation package through its public NumPy
API. Requires NumPy and the built vc_reader extensions; it does not import
Torch, SciPy, Vesuvius, or any other augmentation implementation. The case
definitions are shared with the comparison harness below. See
[the native ranking report](native_augmentation_results.md) for measured results.
The [next CPU optimization pass](cpu_optimization_results.md) includes paired
noise, float32 intermediates and continuous-input median measurements.

```sh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=vc_reader/src vc_reader/.venv/bin/python \
  vc_reader/benchmarks/bench_augment.py --native-only \
  --shape 128 128 128 \
  --reps 20 --warmup 3 --profile all \
  --output vc_reader/benchmarks/results/native-ranking-128.json
```

Each crop is read and normalized once, outside timing. Calls include native
sampling, validation, output allocation and Python/NumPy adaptation, with
default input-preserving ownership. Outputs are destroyed outside timing.
There are no internal augmentation worker threads. Each repetition uses seed
`123 + repetition`; cases are shuffled with a fixed ordering seed. All cases
are checked for exact seeded repeatability, finite outputs and preserved
inputs before timing. A failed check aborts the run.

Without `--start`, native mode uses six PHercParis4 regions: the original three
at ZYX `(8960,4480,3200)`, `(5120,3840,3840)`, `(5120,2816,5120)`, plus denser
crops at `(4608,5760,5888)`, `(4480,5760,5248)`, `(4224,5888,5120)` for 128³.
The additions have 72.8%, 57.8%, and 47.9% threshold foreground, versus
10.3–25.9% in the original regions. Other shapes keep those crop centers;
`--shape 64 64 64` uses centered subcrops. Explicit `--start` replaces the
default workload; repeat `--additional-start` to supply further custom origins.

Results are ranked by pooled p50 across all equally weighted crops. JSON stores
every per-crop observation, mean/p50/p95/min/max, input/output hashes, transform
parameters, build flags and binary identity. `--build-dir` should point at the
measured CMake build; `run_native.py` (below) selects a specific extension
without installing it. `--profile all` (or comma-separated case names) adds
five separate calls using the native pipeline's per-transform timer; these
are excluded from the latency samples. Use a sampling profiler for C++
function-level attribution.

`--only` selects transforms; `--filter-sweep` selects median sizes 3/4/5/7 and
Gaussian sigmas .5/1/2/4. `--input-noise-std .01` explicitly adds fixed noise
to the real CT before timing, exercising continuous-valued filter paths.
Inputs must be uint8 CT. Labels are derived `CT > 127.5` masks, not annotations.
The suite covers 24 individual image, spatial and mask-corruption augmentations.
Data-preparation helpers, label encoding/cleanup, target generation, copy
controls and composite pipelines are excluded. RemoveComponents, ClipOutliers,
Affine and SheetCompression are also excluded by request; their implementations
remain available. Tracing operations requiring surface
coordinates or supervised points and standalone probability wrappers are
listed as exclusions in the JSON. This mode excludes reader timing and
`inplace` calls. It measures selected parameters, not training throughput or
an average over all possible parameter settings.

To check numerical differences between two native builds on these same six
crops, use `check_augment_build.py` through `run_native.py`. Capture reference
outputs once, then compare the candidate. This is an untimed check; it records
binary hashes, input hashes, changed-value counts and maximum absolute error.
Integer outputs must match exactly. Floating-point defaults are `atol=1e-7`,
`rtol=1e-6`; override them explicitly if needed. Neither mode imports another
augmentation implementation.

```bash
export AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=vc_reader/src:vc_reader/.venv/lib/python3.14/site-packages
vc_reader/.venv/bin/python -S vc_reader/benchmarks/run_native.py \
  /tmp/vc-augment-perf/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/check_augment_build.py /tmp/vc-elastic-reference --capture
vc_reader/.venv/bin/python -S vc_reader/benchmarks/run_native.py \
  /tmp/vc-augment-elastic/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/check_augment_build.py /tmp/vc-elastic-reference \
  --output vc_reader/benchmarks/results/elastic-pass-equivalence.json
```

Defaults are Elastic, 128³, and seeds 123/124/125. `--only`, `--shape`,
`--seeds`, and `--array` must match between capture and comparison. Use a new
reference directory for a different workload. Capture refuses to overwrite an
existing manifest. Tests exercise capture/comparison with a tiny local Zarr and
verify that image and label mismatches fail the check.

## Augmentation comparison: `bench_augment.py`

[Measured results on 64³ and 128³ PHercParis4 crops](augmentation_results.md)
include all timings, numerical differences, and profiler findings.
The [median and Gaussian follow-up](filter_results.md) measures the later filter
optimizations, including wider kernels and continuous-valued inputs.
The [broader optimization follow-up](optimization_results.md) covers spatial
resampling, local intensity transforms, morphology, noise and label encoding.
The [elastic experiment](elastic/README.md) compares dense Gaussian, coarse
Gaussian, cubic B-spline, and tensor-guided fields on three Paris 4 regions,
with [parameter comparison images](../elastic_comparison/README.md).
The [production B-spline follow-up](elastic_production_results.md) documents
the replacement of Gaussian Elastic, CPU optimizations, and current timings.

Compare the native public API with the actual Python transforms in the Vesuvius
source tree on a real cached CT crop. Requires Torch, SciPy, scikit-image,
fft-conv-pytorch, SimpleITK, pandas, and acvl-utils in addition to vc_reader.
No dependencies are installed by the harness. Run from the repository root:

```sh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=vc_reader/src:vesuvius/src vc_reader/.venv/bin/python \
  vc_reader/benchmarks/bench_augment.py --reps 10 --warmup 2 \
  --profile GaussianNoise,MedianFilter,LowResolution \
  --output vc_reader/benchmarks/results/augment-128.json
```

The default is a populated 128³ PHercParis4 crop at ZYX `(1280,2688,3200)`.
`--array`, `--start`, and `--shape` override it. Inputs have one float32 channel;
label cases use an int16 threshold of the CT, not ground-truth annotations.
`--only Flip,Gamma` selects cases. Use `--build-dir` to record the relevant
CMake build flags and verify the extension matches that build.

Use `--filter-sweep` for median sizes 3/4/5/7 and Gaussian sigmas .5/1/2/4.
Add `--input-noise-std .01` to perturb the normalized CT with fixed Gaussian
noise before timing and exercise the continuous-valued median fallback. In
reader mode the same precomputed perturbation is added to each fresh crop
inside the timer. For a populated 64³ crop:

```sh
PYTHONPATH=vc_reader/src:vesuvius/src vc_reader/.venv/bin/python \
  vc_reader/benchmarks/bench_augment.py --filter-sweep --input-noise-std .01 \
  --start 1312 2752 3232 --shape 64 64 64 --reps 5 --warmup 2 \
  --output vc_reader/benchmarks/results/filters-noisy-64.json
```

Each operation is constructed once. Reads, parameter construction, RNG reset,
and output comparisons are excluded from timings. Python receives a timed clone
of its inputs because some transforms mutate them; the native API copies inside
its call. Both therefore preserve the caller's sample. Sampling and Python/Torch
adaptation are included. Flip and transpose use explicit fixed parameters in the
Python `apply` call to prevent random no-ops; this exception is recorded per row.
Python blur uses direct convolution with autotuning disabled.

Backends alternate order across repetitions, with one Torch/BLAS thread and two
warm-ups by default. JSON includes every observation, mean, p50, p95, min/max,
versions, CPU, build flags, extension hash, and per-output numerical differences.
`native_output_sha256` records the exact output bytes outside the timer, including
stochastic outputs at the fixed native seed, for before/after regression checks.
To benchmark a saved augmentation binary without installing it, pass it to
`run_native.py` before the benchmark filename:

```sh
PYTHONPATH=vc_reader/src:vesuvius/src vc_reader/.venv/bin/python \
  vc_reader/benchmarks/run_native.py /tmp/before/_augment.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/bench_augment.py --reps 11 --warmup 3 --output /tmp/before.json
```

Use the extension suffix matching your interpreter. The wrapper accepts both
reader (`_native`) and augmentation (`_augment`) builds.

Independent RNG draws are marked separately from deterministic comparisons;
timing similar stochastic workloads does not establish identical augmentation
distributions. A failed comparison remains visible as an error, not a speedup.
The comparison mode uses the same augmentation-only case list. Specialized
tracing APIs, probability/composition wrappers, data-preparation helpers and
target-generation utilities are excluded. Earlier linked comparison reports
describe their original workloads.

To avoid importing unrelated volume/download dependencies, the harness registers
the Vesuvius source package path without executing its root initializer. All
augmentation modules and their dependencies execute unchanged. `--profile`
creates separate cProfile reports after timing (three calls per backend). These
locate Python-level costs; native calls appear as one opaque extension call.

To include the reader handoff, use `--read-each-call --inplace`. Both backends
read a fresh float32 crop through `vc_reader`, divide by 255, construct any
synthetic labels, and augment it. Python uses the fresh tensor directly; native
augmentation receives `inplace=True`. Reads and preparation are timed in this
mode; construction and warmup remain outside timing. The JSON records reader
counters and the reader extension hash. This measures warm local reads, not
cold storage or network throughput. `--inplace` requires fresh reads to prevent
repeatedly augmenting already-modified data.

## Local reads: `bench_local.py`

```
python benchmarks/bench_local.py                    # default array below, 3 reps
python benchmarks/bench_local.py --array /path/to/array.zarr/0 --reps 5
python benchmarks/bench_local.py --only crops64,batch --backends zarr,vc_reader:slab
```

The default array is the local copy of PHercParis4 volume 20260411134726
(2.4 µm, 78 keV), level 2, under
`/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/`. It is 382 GB
of uncompressed 128³ uint8 chunks on an md RAID of NVMe drives, larger than
RAM. The array is only ever read.

Crops are placed uniformly at random over the whole array (keeping those whose
centre chunk is stored). The scenarios are:

| scenario | operation |
|---|---|
| `crops128`, `crops64` | 200 × 128³ / 500 × 64³ crops, one after another (latency) |
| `batch128` | 256 × 128³ crops in one request (`read_many`; a 32-thread pool for the others) |
| `chunks` | 256 whole stored chunks in one request, aligned to the chunk grid |
| `block512` | 4 × 512³ blocks (inference tiles) |
| `crops64-hot` | 500 × 64³ crops inside one 512³ neighbourhood, with vc_reader's memory cache |

Before each repetition, `-cold` scenarios drop every touched chunk file from
the page cache with `posix_fadvise(DONTNEED)`, which needs no root (Linux
only; skipped on tmpfs). `-warm` scenarios run one untimed repetition first.

Local backends: `vc_reader` reads chunk files through memory maps (the
default), `vc_reader:slab` through `pread`, and `vc_reader+memory` adds the
memory cache.

## Points and sampling: `bench_sample.py`

```
python benchmarks/bench_sample.py
python benchmarks/bench_sample.py --reps 7 --compare results/sample-YYYYMMDD-HHMMSS.json
python benchmarks/bench_sample.py --cache-state both --readers mmap,memory --reps 7
```

`read_points`, `sample(order=0)` and `sample(order=1)` on the default array
above, warm (one untimed repetition first), through memory maps and through the
memory cache. The point sets lie in one stored 1024³ region: 1M points uniform
over it (`random`), a 16 × 512 × 512 sampling grid on a tilted plane (`patch`,
an ink-detection patch), and 1M points in a 96³ box (`local`). Each scenario's
output is hashed; `--compare` prints the speedup over an earlier results file
and fails if any output differs from it, so a change can be shown to return
exactly the same values.

`--cache-state cold` closes the reader before every repetition, evicts the
files in the points' enclosing box with `posix_fadvise(DONTNEED)`, and opens a
new reader. `both` runs cold and warm scenarios; neither opening nor eviction
is timed. Cold runs require Linux and a disk-backed array. This measures cold
file data with warm filesystem metadata; it does not flush the drive's own
cache. Outputs are checked on every repetition, outside the timer.
`--readers slab` also measures the `pread` path. `--threads`, `--start Z Y X`,
`--edge`, and `--output` control the workload and results file. Use a stored
region: absent local chunks are fill values, so a mostly absent region does
not measure useful I/O. Results include all timings, mean, median and min/max.

## Testing an existing native build without installation

`run_native.py` loads the specified extension before running a benchmark or
pytest. This avoids accidentally measuring an older installed extension.
With the existing development environment and fetched CMake dependencies:

```bash
export AGENTS_AGENT_MODE=1
cmake -S . -B build/cp314-cp314-linux_x86_64 \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/cp314-cp314-linux_x86_64 -j 8
.venv/bin/python benchmarks/run_native.py build/cp314-cp314-linux_x86_64/_native*.so -m pytest -q
.venv/bin/python benchmarks/run_native.py build/cp314-cp314-linux_x86_64/_native*.so \
  benchmarks/bench_sample.py --cache-state both --reps 7 --output /tmp/sample-after.json
```

Use your environment's build directory in place of the example tag. Save the
baseline extension in a separate directory before rebuilding, then pass that
file to the same wrapper for baseline runs. Run builds, tests and benchmarks
sequentially so they do not compete for CPU or evict one another's file data.
For sampling profiles without hardware-counter privileges, `gprofng collect
app -p hi -o /tmp/reader.er` can precede the Python command. Build with
`RelWithDebInfo` (and `-DCMAKE_CXX_FLAGS_RELWITHDEBINFO='-O3 -g -DNDEBUG'` to
match Release optimization) for symbols; inspect with
`gprofng display text -limit 20 -functions /tmp/reader.er`.

## Oriented crops: `bench_affine.py`

Compare native `read_affine` with explicit NumPy coordinate construction plus
`sample`, and with sampling a prebuilt grid. `affine-out` additionally reuses
the destination. This measures raw float32 samples, without normalization;
it does not compare against the tracers' Numba or GPU implementations.

```bash
.venv/bin/python benchmarks/run_native.py /tmp/vcr-affine-before/_native*.so \
  benchmarks/bench_affine.py --reps 7 --cache-state both \
  --methods points,points-prebuilt --output /tmp/affine-before.json
.venv/bin/python benchmarks/run_native.py build/cp314-cp314-linux_x86_64/_native*.so \
  benchmarks/bench_affine.py --reps 7 --cache-state both --output /tmp/affine-after.json
```

The baseline extension is saved before rebuilding. `--array` selects another
local array; the fixed source region must be populated (the harness checks
that every enclosing chunk is present). `--only cube128` selects one workload,
and `--methods affine` selects one implementation. The JSON records the array,
origin, axes, output shapes, environment, thread count and every timing.
Each method has one warm-up and seven measured repetitions by default, with
method order alternating each repetition. Every output is checked against
`sample` outside the timer (`rtol=1e-6, atol=1e-5`). Cold runs close mappings and
evict the enclosing chunk files before each method; this requires Linux and
does not flush the drive cache. Opening and eviction are excluded from timing.

### Initial affine results, 2026-10-16

PHercParis4 `20260411134726`, 2.4 µm/78 keV volume, level 2, on the default
NVMe RAID path above. uint8, uncompressed 128³ chunks; grid origin
`(9300.25, 4000.5, 4000.75)`, rotated axes with step lengths `(0.8, 0.9, 0.73)`.
Both extensions used GCC 15, Release (`-O3 -DNDEBUG -g`), Python 3.14.4,
16 reader threads, Linux x86-64 on the 24-CPU host described below.
No dependencies were installed or source volumes modified.

Times are **mean / min / median / max milliseconds**, seven repetitions.
The baseline includes constructing the coordinates every call; affine
allocates its result through the existing output pool in both cache states.

| Crop / cache | Explicit grid + sample, before | Native affine, after | Median speedup |
|---|---|---|---|
| 64³ cold | 3.046 / 2.630 / 2.845 / 3.692 | 2.342 / 1.751 / 2.133 / 3.673 | 1.33× |
| 64³ warm | 2.481 / 2.268 / 2.491 / 2.735 | 1.669 / 1.358 / 1.725 / 1.873 | 1.44× |
| 128³ cold | 19.032 / 18.059 / 19.087 / 20.000 | 6.770 / 6.612 / 6.728 / 6.969 | 2.84× |
| 128³ warm | 19.472 / 18.013 / 19.265 / 20.853 | 7.113 / 5.987 / 6.818 / 8.695 | 2.83× |
| 1×512×512 cold | 3.328 / 3.146 / 3.313 / 3.513 | 2.508 / 2.159 / 2.371 / 3.320 | 1.40× |
| 1×512×512 warm | 2.481 / 2.299 / 2.526 / 2.601 | 1.436 / 1.123 / 1.426 / 1.726 | 1.77× |

Prebuilt-coordinate sampling is faster than native affine in these runs:
for warm 128³ crops, the same after-build measured 5.494 ms median with a
prebuilt grid versus 6.818 ms for affine. The benefit is avoiding grid
construction and storage (48 MiB of coordinates for 128³), not a universal
interpolation speedup. Reusing `out` was approximately unchanged for 128³
(6.839 ms median warm); small-crop timings are noisy. These results do not
establish performance on macOS/arm64, compressed sources or remote data.

`gprofng collect app -p hi` on a separate loop of 1,000 warm 128³ affine
reads with output reuse attributed 36.74% of exclusive CPU samples to point
classification/planning, 25.72% to run interpolation, 19.01% to `cell_of`,
and 8.72% to coordinate generation. These are CPU samples across threads,
not wall-time fractions. The profile used the same Release object files
relinked with symbols retained (without the linker's `-s`); profiling was
separate from the timings above.

The full suite passed 383 offline tests, including affine grids crossing
chunk/array boundaries, missing chunks, rotations, shears, reflected axes,
nearest ties, byte-swapped dtypes, output reuse, HTTP caches, pickle/offline
reads, and more than 2²⁰ samples per crop. Existing point-sampler numerical
tests still pass; no fast-math or reduced interpolation precision was added.

## Casting and normalization: `bench_normalization.py`

Compares block and affine crop reads plus NumPy post-processing against native
`dtype`/`normalization` options. Both use float64 population statistics;
z-score excludes zeros and preserves them in the output. Min/max includes
the full crop. The baseline uses in-place NumPy arithmetic after casting;
its zero mask and selected values are part of the measured workload.

Save the baseline extension and its matching `zarr_reader.py` before changing
or rebuilding them. The recorded baseline is `a879c21d7`:

```bash
mkdir -p /tmp/vcr-normalization-before
cp build/cp314-cp314-linux_x86_64/_native*.so /tmp/vcr-normalization-before/
cp src/vc_reader/reader/zarr_reader.py /tmp/vcr-normalization-before/zarr_reader.py

.venv/bin/python benchmarks/run_native.py /tmp/vcr-normalization-before/_native*.so \
  benchmarks/bench_normalization.py --baseline-reader /tmp/vcr-normalization-before/zarr_reader.py \
  --methods numpy --reps 7 --output /tmp/normalization-before.json

# After building the new extension:
.venv/bin/python benchmarks/run_native.py build/cp314-cp314-linux_x86_64/_native*.so \
  benchmarks/bench_normalization.py --reps 7 --output /tmp/normalization-final.json
```

The defaults use the same local PHercParis4 level-2 volume as the other
benchmarks, a 128³ crop starting at `(9300, 4000, 4000)`, and the fixed affine
matrix recorded in the script/JSON. The harness checks enclosing chunks are
stored. `--edge`, `--paths block,affine`, `--dtypes float32,float64`, and
`--modes cast,minmax,zscore` select workloads. `--cache-state cold` closes
readers and evicts file pages before each method (Linux only); the results
below use **warm mmap reads with no decoded memory cache**.

Both builds: GCC 15.2, Release (`-O3 -DNDEBUG -g`), Python 3.14.4, 16 native
reader threads, Linux x86-64 on the same 24-CPU/NVMe RAID host. One untimed
warm-up and seven measured repetitions per case, alternating method order
within the after-build run. Validation is outside the timer and uses
`rtol=3e-6, atol=3e-6`, allowing rounding differences from NumPy's separate
in-place arithmetic operations. No normalization approximations or fast-math
flags were introduced. Fixed segment reductions make native output independent
of reader thread count, but not necessarily bit-identical to NumPy.

Times below are **mean / min / median / max milliseconds** for complete reads
and post-processing, collected 2026-10-16. Speedup compares medians.

| Crop / output / operation | Before: read + NumPy | After: native | Speedup |
|---|---|---|---|
| block / float32 / cast | 1.008 / 0.788 / 0.943 / 1.566 | 0.545 / 0.421 / 0.552 / 0.611 | 1.71× |
| block / float32 / minmax | 1.615 / 1.547 / 1.583 / 1.728 | 2.183 / 1.304 / 2.374 / 3.129 | 0.67× |
| block / float32 / zscore | 7.022 / 6.960 / 7.005 / 7.129 | 2.061 / 1.670 / 2.115 / 2.658 | 3.31× |
| block / float64 / cast | 1.264 / 1.172 / 1.241 / 1.391 | 1.124 / 0.725 / 1.159 / 1.400 | 1.07× |
| block / float64 / minmax | 3.131 / 2.902 / 2.970 / 3.661 | 3.265 / 2.315 / 3.492 / 3.812 | 0.85× |
| block / float64 / zscore | 11.740 / 11.500 / 11.592 / 12.361 | 2.613 / 1.844 / 2.316 / 3.511 | 5.01× |
| affine / float32 / cast | 6.778 / 6.351 / 6.800 / 7.208 | 6.775 / 6.400 / 6.812 / 6.993 | 1.00× |
| affine / float32 / minmax | 7.948 / 7.383 / 7.986 / 8.225 | 7.391 / 6.808 / 7.255 / 8.205 | 1.10× |
| affine / float32 / zscore | 13.390 / 12.748 / 13.200 / 14.268 | 7.645 / 6.855 / 7.784 / 8.199 | 1.70× |
| affine / float64 / cast | 7.854 / 6.943 / 7.638 / 9.344 | 7.251 / 6.520 / 7.311 / 8.538 | 1.04× |
| affine / float64 / minmax | 9.057 / 8.569 / 8.968 / 9.933 | 7.461 / 7.194 / 7.396 / 8.232 | 1.21× |
| affine / float64 / zscore | 20.935 / 19.065 / 20.747 / 23.608 | 8.159 / 7.517 / 7.967 / 8.847 | 2.60× |

Min/max block normalization is **not a demonstrated speedup** here; its
thread scheduling and scalar reductions can cost more than NumPy's optimized
min/max passes. The same-build interleaved NumPy/native medians were
2.049/2.374 ms (float32) and 2.936/3.492 ms (float64), consistent with that
limitation. The final JSON includes all interleaved measurements. Gains
depend on crop size, zero density, cache state and competing loader workers.
macOS/arm64, compressed/remote-source performance and end-to-end training
throughput were not measured.

`gprofng collect app -p hi` profiled 500 separate warm 128³ block min/max
reads with output reuse, using the same Release objects relinked with symbols.
Final exclusive CPU samples: normalization output loop 33.95%, first statistics
pass 11.33%, and typed chunk conversion 9.22%. These include process startup
(42.59% in NumPy's BLAS worker startup), so they are not wall-time fractions.
The initial prototype spent 82.46% in the statistics pass; accumulating each
segment locally before publishing its result removed shared-memory updates
from that loop without changing reduction order.

The final extension passed 456 offline tests (2 live-network tests deselected).
New coverage includes independent crop statistics, zero masking, casts from
native/byte-swapped source types, 1-D through 4-D boxes, affine sampling,
output reuse, constant/empty/nonfinite/extreme-valued crops, thread-count
determinism, and HTTP/pickle/offline cache reuse. Existing default-read and
interpolation tests pass unchanged, except the old test rejecting nearest
float output now rejects an unsupported integer conversion instead.

## Remote reads: `bench_remote.py`

```
python benchmarks/bench_remote.py
python benchmarks/bench_remote.py --arrays ct --reps 5
```

The reads go to `s3://vesuvius-challenge-open-data` inside one fully populated
1024³ region of the same CT level (uncompressed) and of its level-2 surface
prediction (blosc-zstd, 192³ chunks). The scenarios are a 256×512×512 block,
12 sequential 128³ crops, the same 12 crops as one batch, and the batch through
vc_reader's disk cache, uncompressed and Delta3D-compressed: first filling it,
then served warm from disk.
zarr-python runs with its default async concurrency (10) and with concurrency
32. Remote numbers are dominated by the network path to us-east-1 and vary a
lot between runs.

## Native optimization measurements, 2026-10-16

Baseline: `3baa35d76a5dfebb145cb9370af71476074e02f4`. Both builds used GCC 15,
Release (`-O3 -DNDEBUG`, with debug information for profiling), Python 3.14.4,
16 reader threads, and the default PHercParis4 level-2 array above on ext4/NVMe
RAID. Host: Intel Core Ultra 7 270K Plus, 24 CPUs, Linux x86-64. No fast-math,
precision, interpolation order, thread-count, or workload changes were used.

The samplers used 1,000,000 random points in the stored 1024³ region starting
at `(9000, 3600, 3600)`, and a 16 × 512 × 512 tilted plane patch (4,194,304
points). All 729 chunks enclosing that region were present. Each sampler
scenario ran 11 measured repetitions; local reads ran 7. The noisy warm 64³
crop case was repeated with 21 repetitions. Builds, tests and benchmark
processes ran sequentially for these comparisons.

The native changes specialize the existing sampler templates for 3-D,
replace division by power-of-two chunk sizes with exact shifts, prefetch
coordinates and voxel rows for trilinear gathers, hint cold mapped point
reads, combine contiguous copy axes, and transfer raw file buffers directly
into the decoded memory cache. Arbitrary ranks and chunk sizes retain the
general path. Python runtime code is unchanged.

`gprofng` clock sampling of the baseline's random/patch trilinear workloads
(`--reps 150 --only random-linear,patch-linear`) attributed 40.9% of total
sampled CPU time to individual-point interpolation callbacks, 22.6% to point
classification/planning, 10.5% to run interpolation, 8.2% to `cell_of`, and
3.7% to chunk-ID lookup. These are exclusive CPU samples across threads,
including benchmark overhead, not wall-time fractions.

Representative timings below are **mean / min / median / max**, in milliseconds
per complete workload. Speedup uses the median. `mmap` is the default reader;
`memory` enables the decoded cache (cold repetitions start with it empty).

| Workload | Before ms | After ms | Speedup |
|---|---|---|---|
| Random nearest, mmap, cold | 104.70 / 102.34 / 104.34 / 107.43 | 92.61 / 89.99 / 92.66 / 94.99 | 1.13× |
| Random nearest, mmap, warm | 4.05 / 3.86 / 4.00 / 4.44 | 3.52 / 3.37 / 3.51 / 3.77 | 1.14× |
| Random trilinear, mmap, cold | 110.34 / 106.50 / 109.98 / 113.02 | 98.94 / 96.87 / 98.89 / 100.20 | 1.11× |
| Random trilinear, mmap, warm | 11.97 / 11.56 / 11.92 / 12.51 | 11.81 / 11.15 / 11.63 / 12.83 | 1.02× |
| Plane nearest, mmap, cold | 8.66 / 7.94 / 8.66 / 9.63 | 9.03 / 8.02 / 8.80 / 11.63 | 0.98× |
| Plane nearest, mmap, warm | 6.03 / 4.77 / 6.10 / 7.09 | 5.83 / 5.22 / 5.66 / 6.78 | 1.08× |
| Plane trilinear, mmap, cold | 13.67 / 12.69 / 13.45 / 15.10 | 11.56 / 10.86 / 11.54 / 13.00 | 1.17× |
| Plane trilinear, mmap, warm | 11.43 / 10.48 / 11.58 / 12.10 | 10.62 / 9.97 / 10.45 / 11.49 | 1.11× |
| Plane integer points, mmap, warm | 5.23 / 4.27 / 5.01 / 6.73 | 5.47 / 4.11 / 5.33 / 6.57 | 0.94× |
| Random nearest, memory, cold | 185.09 / 175.79 / 186.90 / 194.09 | 165.06 / 153.67 / 165.24 / 180.76 | 1.13× |
| 256 whole chunks, cold | 49.35 / 46.10 / 47.84 / 58.95 | 48.02 / 44.39 / 46.20 / 58.04 | 1.04× |
| 256 whole chunks, warm | 23.46 / 22.82 / 23.44 / 24.21 | 20.25 / 19.66 / 20.28 / 21.55 | 1.16× |
| 500 × 64³ crops, warm, 21 reps | 36.55 / 27.74 / 34.93 / 56.69 | 39.17 / 26.37 / 34.52 / 71.02 | 1.01× |

Unaligned crops and large-block reads were approximately unchanged. Warm
integer-plane reads and small crops remain noisy; there is no demonstrated
improvement for those cases. These measurements do not establish performance
on macOS/arm64 or on compressed source arrays. Cold eviction is a Linux
file-cache hint, not a drive-cache flush.

Every sampler repetition matched its warm-up hash, and all 36 before/after
sampler hashes and all local-read output hashes matched exactly. The optimized
extension passed 227 offline tests (2 network tests deselected), including
float64 interpolation order, missing/outside chunks, non-finite points,
arbitrary and unit chunk dimensions, 1-D through 4-D copies, and malformed raw
chunk rejection. Nearest ties-to-even and the trilinear sum order are unchanged.

Reproduce from `vc_reader/`, using `run_native.py` with the saved baseline
extension and then the optimized extension:

```bash
.venv/bin/python benchmarks/run_native.py /tmp/vcr-perf-before/_native*.so \
  benchmarks/bench_sample.py --reps 11 --cache-state both --output /tmp/sample-before.json
.venv/bin/python benchmarks/run_native.py build/cp314-cp314-linux_x86_64/_native*.so \
  benchmarks/bench_sample.py --reps 11 --cache-state both \
  --compare /tmp/sample-before.json --output /tmp/sample-after.json
.venv/bin/python benchmarks/run_native.py build/cp314-cp314-linux_x86_64/_native*.so \
  benchmarks/bench_local.py --reps 7 --backends vc_reader \
  --only crops64,crops128,batch128,block512,chunks
.venv/bin/python benchmarks/run_native.py build/cp314-cp314-linux_x86_64/_native*.so \
  benchmarks/bench_local.py --reps 21 --backends vc_reader --only crops64 --no-cold
```

The session's full timings and hashes are in the ignored `results/` directory:
`native-sample-before.json`, `native-sample-after.json`, local runs
`local-20261016-070224.json` / `local-20261016-070305.json`, and the 21-repetition
crop runs `local-20261016-070352.json` / `local-20261016-070356.json`.

## Native hotspot profiling

`profile_augment.cpp` runs five native calls on a raw 128³ uint8 chunk without
Python. On Linux with GNU gprof, build with `-O2 -g -pg` and run from a temporary
directory (gprof writes `gmon.out` in the working directory):

```sh
c++ -std=c++20 -O2 -g -pg -I vc_reader/cpp/augment \
  vc_reader/benchmarks/profile_augment.cpp -o /tmp/profile_augment
(cd /tmp && ./profile_augment /absolute/path/to/raw/chunk median)
gprof -b -p /tmp/profile_augment /tmp/gmon.out > /tmp/augment-profile.txt
```

Operations: `copy`, `flip`, `median`, `blur`, `affine`, `elastic`, `sheet`,
`local` (local gamma), `morphology`, `noise` (Gaussian). Morphology uses a binary
threshold of the CT, not annotated labels. Use uninstrumented Release builds
and `bench_augment.py` for reported latency; instrumentation changes timings.
The workload itself is portable C++20; use a platform profiler when gprof is not
available.
