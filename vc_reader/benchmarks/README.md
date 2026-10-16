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
