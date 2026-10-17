# Paris4 v3 sheet-strip reader benchmark

Measured 2026-10-17 on an RTX 5090 and Intel Core Ultra 7 270K Plus.
The current benchmark uses **native `A.Normalize("zscore")` on both CPU and
CUDA**, followed by native augmentation in the same pipeline.
The CUDA image pipeline improves sustained data-preparation throughput by
**1.80×** over the existing trainer with its configured **14 workers**.
The native CPU replacement measures **1.16×** in this run.

These are data-pipeline measurements, ending with the CT image ready on the
GPU. They include actual training sampling, CT frame estimation, tube geometry,
strip targets, worker IPC, pinning, and image uploads. They exclude transfer of
the other target tensors, GPU tube rendering, model forward/backward, and the
optimizer. They are **not training-step speedups**: GPU preprocessing will also
share the device with model computation in a real training run.

## Actual 14-worker loader

`sheet_strip_paris4_slim50k_v3`, batch 32, float32 CT images of shape
`(32, 1, 144, 128, 128)`, seed 0. Each mode ran in a separate process with
28 warmup batches and 56 measured batches (1,792 measured crops).
The three measured sequences have identical position/frame SHA-256:
`9f725ef8a1ad68cabf04f6fe880e294e758c9a9ed64d3203072060745b1be047`.

| Image pipeline | Mean batch ms | p50 ms | p95 ms | Crops/s | Relative throughput |
|---|---:|---:|---:|---:|---:|
| Existing mmap + Numba + NumPy/SciPy | 413.56 | 16.10 | 3,357.21 | 77.38 | 1.00× |
| vc_reader CPU + native normalization/augmentation | 356.33 | 16.00 | 3,874.64 | 89.80 | 1.16× |
| vc_reader CUDA + native normalization/augmentation | 230.24 | 229.81 | 266.16 | 138.99 | 1.80× |

CPU workers deliver bursts of prefetched batches followed by a longer wait.
Consequently, the low CPU p50 is not its sustainable batch interval. Throughput
uses total measured time, including those waits. Mean loader wait was 398.53,
341.29, and 0.16 ms respectively. CUDA workers prepare geometry and targets;
the parent performs the image work on the GPU.

The original DataLoader settings are retained: 14 workers, prefetch factor 4,
pinned memory, one Torch thread per worker, and 0.5 GiB worker CT cache
(0.375 GiB used by the CT reader). Native CPU readers use one reader thread
per worker and a 0.375 GiB cache. CUDA uses four reader threads in the parent,
a 0.375 GiB host cache, and a 4 GiB GPU source-chunk cache. CUDA records
84 GPU routes and zero CPU fallback routes across warmup and measurement;
it uploads 74.72 GiB of source chunks across those 84 batches. Cache statistics
are in the raw report.

## Isolated image stage

Eight batches captured from the actual trainer were replayed twice, following
two warmups per mode. This removes geometry/target generation and worker
competition, isolating read, interpolation, normalization, augmentation, and
the final CPU-to-GPU transfer. CPU modes use four reader/Torch threads in
the process; NumPy/Numba crop and augmentation loops remain serial.

| Pipeline | Mean / p50 / p95 ms per batch | Crops/s |
|---|---|---:|
| Existing | 1,567.54 / 1,589.77 / 1,660.55 | 20.41 |
| vc_reader CPU | 1,410.35 / 1,413.41 / 1,430.90 | 22.69 |
| vc_reader CUDA | 96.35 / 95.22 / 111.24 | 332.11 |

The CUDA cache was cleared before every replay batch, outside timing. Source
reads, destination allocation, pinned asynchronous uploads, and synchronization
are timed. This is not a GPU-resident replay. CUDA uploads about **969 MiB**
of whole uint8 source chunks per batch on average; CPU pipelines upload the
completed **288 MiB** float32 image batch. CUDA source uploads can therefore
amplify storage reads despite transferring uint8 values.

| Mean stage ms | Existing | vc_reader CPU | vc_reader CUDA |
|---|---:|---:|---:|
| Read and interpolate | 638.37 | 635.42 | 82.66 |
| Normalize and augment | 902.02 | 747.40 | 12.90 |
| Pin/upload/final synchronization | 27.08 | 27.45 | 0.72 |

The isolated 16.3× CUDA gain must not be substituted for the 1.8× loader gain.
The 14-worker experiment includes additional CPU work, contention, different
locations, and cache turnover. Filesystem caches were neither flushed nor
controlled. Runs proceeded existing → CPU → CUDA, so these are local cached
workload observations, not cold-storage or randomized-order estimates.

## Inputs and implementation

The cached published checkpoint's `run_config`, `model_cfg`, `sample_cfg`, and
`identity_sampling` provide the settings. No model weights are updated.

- Checkpoint: `/home/sean/.cache/huggingface/hub/models--scrollprize--fiber_strip_small_v1/snapshots/7bd163a22c839ff312879ae1916195d0c9729c0e/last.pt`.
- Actual matched-span index: `/home/sean/Documents/villa/vesuvius/src/vesuvius/neural_tracing/tracers/fibers/output/sheet_strip_data/index_slim50k`.
- AFV: `/home/sean/Documents/villa4/vesuvius/src/vesuvius/neural_tracing/tracers/fibers/output/paris4_fibers_ctz7000_17000_slim50k_20261012/fibers.afv`.
- CT: `/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2`, uncompressed uint8, 128³ chunks.
- Verified patches and umbilicus: `/mnt/raid_nvme/spiral_dataset_working`.

The benchmark subclasses `StripBuilder` and reuses `build_data`. The replacement
builder defers CT images with a tiny disposable placeholder, keeping existing
frame, label, tube, and sampling code. The original augmentation also runs on
that placeholder; its cost is included. Completed frame/photometric requests
then feed the selected image backend. Production trainer and reader code are
unchanged. TIFF patch mappings are forced read-only in this benchmark.

Both new backends use the native zero-preserving population z-score: statistics
exclude zero voxels, zero voxels remain zero during normalization, and constant
channels become zero. This intentionally differs from the trainer's normalization
that includes zeros and clamps the divisor to an epsilon. The native chain
receives sampled float32 CT values directly; no `/255` pass or Torch z-score is
used. Z-score is invariant to the input scale for ordinary nonconstant CT.

Unclipped contrast/brightness, blur probability/sigma, and noise variance are
retained. Gaussian blur uses reflection and the same radius as SciPy's
`truncate=4`; native `truncate` denotes full width, so the adapter converts it.
CUDA reader arrays feed `A.Pipeline([A.Normalize("zscore"), ...], backend="cuda")`
directly. Torch only wraps the finished arrays through DLPack and stacks them
for the model; there is no image download between GPU stages.

Validation checks native CPU/CUDA agreement on four full-size real crops,
including blurred and unblurred cases, plus explicit zero-containing, all-zero,
and constant normalization inputs. Noise residual means were below 0.0003 and
standard deviations matched their requested values within 1%. On these real
crops, maximum noise-free differences from the trainer were **7.06e-5** for
both native backends, with RMSE about **2.16e-6**. Differences on zero-containing
crops are intentional, so trainer parity is reported rather than required.
NumPy and native generators produce different noise realizations; inserting
native normalization also advances the native pipeline RNG sequence. Seeds are
fixed, but outputs are not bit-identical to the earlier benchmark. These checks
validate the adapter, not model convergence or training accuracy.

## Profiling

The earlier trainer-normalization cProfile run covered five full legacy batches plus the four-crop
validation. Its principal image-path costs were `augment_ct` (4.51 s,
including 0.96 s of Gaussian filtering), `sample_scalar_crop` (2.98 s), source
block reads (0.54 s), and trainer normalization (0.31 s, including validation).
These are inclusive timings and must not be summed as independent totals.

Nsight Systems traced 16 CUDA replay batches with cache retention (eight warmup,
eight measured). Across all 16, device H2D copies took **472 ms**, affine
sampling kernels **110 ms**, native normalization kernels **85.1 ms**, blur
kernels **17.6 ms**, and noise kernels **13.0 ms**. Nsight recorded 512 calls to
each native normalization kernel (32 crops × 16 batches), confirming that
normalization executes in vc_reader on CUDA. Native normalization averaged
**5.32 ms/batch** of kernel time; sampling averaged **6.87 ms/batch**. Profiled
wall time was about 113 ms/batch; CPU reading/planning/cache work and synchronization
account for additional time, and copy/kernel totals may overlap. Uploads and
source handling deserve more attention than augmentation kernels next.

## Reproduce

Existing environment: Python 3.14.4, Torch 2.12.1+cu130. Existing native build:
`/tmp/vc-reader-cuda`, CMake **Release**, CUDA enabled. No installation or rebuild
was needed. Native binary hashes and all measurements are recorded in
[fiber_strip_results.json](fiber_strip_results.json). Earlier results using trainer
normalization are retained under `previous_trainer_normalization`; they are not
the native-normalization results above.

From the repository root, the following shell function abbreviates the exact
interpreter, environment, and explicit native-extension selection used:

```bash
export AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=/tmp/fiber-strip-numba
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=vesuvius/src:vc_reader/src
strip_python() {
  /home/sean/Documents/villa4/vesuvius/.venv/bin/python "$@"
}
strip_bench() {
  strip_python vc_reader/benchmarks/run_native.py \
    /tmp/vc-reader-cuda/_native.cpython-314-x86_64-linux-gnu.so \
    vc_reader/benchmarks/run_native.py \
    /tmp/vc-reader-cuda/_augment.cpython-314-x86_64-linux-gnu.so \
    vc_reader/benchmarks/bench_fiber_strip.py "$@"
}
strip_bench capture --workers 0 --batches 8 --output /tmp/fiber-strip-capture.pt
strip_bench replay --capture /tmp/fiber-strip-capture.pt --batches 16 \
  --warmup 2 --clear-gpu --validate --output /tmp/fiber-strip-native-norm-replay.json
for mode in legacy cpu cuda; do
  strip_bench loader --modes "$mode" --workers 14 --batches 56 --warmup 28 \
    --output "/tmp/fiber-strip-native-norm-loader-$mode.json"
done
```

Profiling used the same explicit native loader after these prefixes:

```bash
# CPU: insert before the first run_native.py argument to strip_python:
# -m cProfile -o /tmp/fiber-strip-cpu-profile.pstats
# Benchmark arguments:
# replay --modes legacy --capture /tmp/fiber-strip-capture.pt --batches 4 \
#   --warmup 1 --validate --output /tmp/fiber-strip-cpu-profile.json

# CUDA: prefix the interpreter invocation with:
# DEBUGINFOD_URLS='' /usr/local/cuda/bin/nsys profile --trace=cuda \
#   --sample=none --cpuctxsw=none --resolve-symbols=false \
#   --force-overwrite=true -o /tmp/fiber-strip-native-norm-cuda-profile
# Benchmark arguments:
# replay --modes cuda --capture /tmp/fiber-strip-capture.pt --batches 8 \
#   --warmup 8 --output /tmp/fiber-strip-native-norm-cuda-profile.json
/usr/local/cuda/bin/nsys stats \
  --report cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_kern_sum --format csv \
  /tmp/fiber-strip-native-norm-cuda-profile.nsys-rep

strip_python vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_native.cpython-314-x86_64-linux-gnu.so \
  vc_reader/benchmarks/run_native.py \
  /tmp/vc-reader-cuda/_augment.cpython-314-x86_64-linux-gnu.so \
  -m pytest -q vc_reader/tests/test_augment_normalization.py
```

`--validate` is the regression/smoke harness; it checks native CPU/CUDA image
agreement, normalization edge cases, finiteness, and noise statistics. The
existing CPU/CUDA normalization regression suite also passed **all 60 tests**.
No production code was changed.
The multiprocess benchmark explicitly requires `fork` so workers inherit the
selected native extensions; it is intended for this Linux CUDA environment.
GPU access, existing trainer dependencies, checkpoint, index, CT, and patches
are required. Other machines can override `--checkpoint`, `--index`, `--afv`,
and `--array`; checkpoint-recorded frame-model and umbilicus paths must also
exist. Single-process replay avoids the worker start-method requirement.
