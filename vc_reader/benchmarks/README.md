# Benchmarks

Run commands from the `vc_reader/` package directory. Use matching Release
reader and augmentation extensions; CUDA measurements require a CUDA-enabled
build. Generated reports in `benchmarks/results/` and `docs/benchmarks/` are
local artifacts ignored by Git. Input fixtures remain versioned.

## Fiber trainer comparison

[`bench_fiber_trainer.py`](bench_fiber_trainer.py) runs real sampled batches
through the model and optimizer. The table uses `--steps 500 --warmup 20
--grad-steps 1 --lookahead 0 --timeline`; initialization and warmup are excluded.
Use `--backend existing` for the trainer pipeline, or `--backend vc-reader
--reader-device cpu|cuda --heading tensor --geometry vc-reader` for vc_reader.
The CUDA run also uses `--heading-stream separate`. Supply `--checkpoint`,
`--location-weights-dir`, and a fresh `--output` directory; `--help` lists the
environment and build-selection options.

| Input pipeline | Reading / sampling | Augmentation | Geometry targets | Heading |
| --- | --- | --- | --- | --- |
| Original trainer | Trainer CPU code | Trainer CPU code | Trainer CPU code | Learned heading model and trainer CT fallback |
| vc_reader CPU | Native CPU, loader workers | Native CPU, loader workers | Native CPU, loader workers | CPU structure tensor |
| vc_reader CUDA | Native CUDA | Native CUDA batches | Native CUDA in batch prefetch; worker plane targets on CPU | CUDA structure tensor |


Measured **2026-10-19** on an **Intel Core Ultra 7 270K Plus / RTX 5090**, with PyTorch **2.12.1+cu130**,
compiled BF16 model execution, ten loader workers and four trainer CPU threads.
Each row is one fresh 500-update run, in the order shown.

| Pipeline | it/s | Total runtime | MB/s | Mvox/s |
| --- | ---: | ---: | ---: | ---: |
| Original trainer | 3.65 | **137.173 s** | 726.67 | 181.67 |
| vc_reader CPU | 3.65 | **136.826 s** | 728.52 | 182.13 |
| vc_reader CUDA | 10.43 | **47.943 s** | 2079.15 | 519.79 |

Throughput counts each **144×104×104** single-channel crop once: **24,920,064,000 voxels**
and **99,680,256,000 bytes** across 16,000 float32 model-input crops. MB/s uses
decimal megabytes (10⁶ bytes); Mvox/s uses 10⁶ voxels.

The main-thread breakdown is shown below. For CPU inputs, image preparation
happens in workers and is exposed through loader waits; transfer to the model
is inside the update interval. CUDA image preparation is measured on the
main thread, while target preparation overlaps it in batch prefetch.

| Pipeline | Loader wait | Main input preparation | Model/optimizer | Update p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| Original trainer | 99.870 s | 0.029 s | 36.507 s | 72.99 / 1495.44 ms |
| vc_reader CPU | 101.786 s | 0.032 s | 34.310 s | 69.53 / 1773.11 ms |
| vc_reader CUDA | 0.756 s | 10.697 s | 35.805 s | 93.24 / 109.45 ms |

Update percentiles include the loader/input interval. The stage sums omit
between-update bookkeeping, which is included in the wall-time total.

| Pipeline | Crops | Distinct positions | Distinct source-chunk footprints |
| --- | ---: | ---: | ---: |
| Original trainer | 16,000 | 15,493 | 36,966 |
| vc_reader CPU | 16,000 | 15,528 | 37,037 |
| vc_reader CUDA | 16,000 | 15,495 | 36,894 |


## Reader and pipeline benchmarks


### General reader workloads

[`bench_pipeline.py`](bench_pipeline.py) compares reading/sampling or complete
sampling/augmentation pipelines. It accepts local, HTTP or S3 3-D uint8 Zarr v2
arrays. Reader comparisons include vc_reader CPU/CUDA, TensorStore and
zarr-python. The pipeline suite compares vc_reader backends and optionally
CPU reading followed by CUDA augmentation.

```bash
.venv/bin/python benchmarks/bench_pipeline.py --suite readers --backend both \
  --array /path/to/volume.zarr --level 2 --shape 128 128 128 \
  --workloads ml_batch surface slab rotated_box --surface-size 96 512 \
  --plane-size 512 512 --batch-size 8 --output /tmp/readers.json

.venv/bin/python benchmarks/bench_pipeline.py --suite pipeline --backend both \
  --array /path/to/volume.zarr --level 2 --augmentation heavy \
  --shape 128 128 128 --output /tmp/pipeline.json
```

`--native-dir` selects an existing build. `--cuda-output device` measures a GPU
consumer endpoint; host output includes downloads. `--memory-cache-mib` and
`--cuda-cache warm|clear` control decoded/device cache reuse. Reports record
configuration, versions, hashes, validation, raw samples and latency percentiles.
Reader baselines fetch enclosing regions and use the same SciPy trilinear
sampler; vc_reader samples directly from chunks. These synchronous request
measurements do not include a model or represent an overlapped trainer.

## Augmentation benchmarks

[`bench_augmentations.py`](bench_augmentations.py) measures all README recipes
on CPU, CUDA with transfers, and CUDA with resident arrays. Recipes and input
construction live in [`augmentation_cases.py`](augmentation_cases.py), shared
with the preview generator. `--update-readme` changes timing rows in the package
README without altering examples or images.

```bash
VC_BENCH_VOLUME=/path/to/PHercParis4-volume.zarr
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/bench_augmentations.py \
  --native-dir /tmp/vc-reader-cuda --array "$VC_BENCH_VOLUME" --level 2 \
  --start 9000 3600 3600 --size 128 --warmup 1 --reps 5 \
  --output /tmp/augmentations.json
```

Use `--only Elastic Gamma` for selected recipes. Preparation, validation and
initial resident upload/final download are outside timed resident calls;
public-call copies and validation remain included. Inspect raw measurements
when using a short repetition count. CPU/CUDA interpolation can differ slightly
at nearest-neighbor label boundaries; numerical differences are reported rather
than treated as model-quality results.

## PyTorch CUDA batches

[`bench_torch.py`](bench_torch.py) compares per-sample processing,
`Pipeline.batch()`, and `CudaLoader` with one prefetched batch. Routes use the
same images and per-sample seeds; a validation/warmup epoch precedes timing.
Reading, transfers, casting, augmentation and collation are included. There is
no forward/backward pass; use the fiber trainer comparison for actual training.

```bash
.venv/bin/python benchmarks/bench_torch.py \
  --native-dir /tmp/vc-reader-cuda --array /path/to/volume.zarr --level 2 \
  --batches 64 --batch-size 4 --reps 3 --output /tmp/torch-batches.json
```

Use `--help` for workload and profiling options. Correctness coverage is in `tests/test_torch.py` and
`tests/test_native_batch.py`.


## Other suites

| Workload | Harness / guide |
| --- | --- |
| Metadata discovery and navigation | [`bench_discovery.py`](bench_discovery.py), [protocol](../docs/scroll-reading-benchmarks.md) |
| Paired CT/ink/mask reads | [`bench_paired.py`](bench_paired.py), [protocol](../docs/scroll-reading-benchmarks.md) |
| TIFF and label readers | [`bench_source_readers.py`](bench_source_readers.py), [reader guide](../docs/source-reading.md#run-and-validate) |
| Storage formats and cache behavior | [`bench_source_storage.py`](bench_source_storage.py), [protocol](../docs/source-reading-benchmarks.md) |
| tifxyz layers and surface projection | [`bench_surface.py`](bench_surface.py) |
| Dataset/collation/transfer integration | [`bench_training.py`](bench_training.py), [guide](training.md), [acceptance](../docs/training-workflow-benchmarks.md) |

Use `--help` for each harness. Keep validation/profiling separate from throughput
runs, retain JSON reports with commands and build hashes, and report the input
shape, dtype, batch size, worker/thread counts and cache policy with timings.
