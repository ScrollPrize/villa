# Profiling

This workflow now emits best-effort profiling artifacts for `prepare`, each inference partition, `reduce`, workflow aggregation, and cleanup.

## What Is Instrumented

- `prepare`
  - S3 object listing
  - explicit S3 layer downloads with exact byte counters
  - local decode / deserialize timing while building the surface-volume zarr
  - zarr creation and chunk writes
- `inference`
  - model-weight lookup and explicit downloads
  - model load
  - `torch.compile` plus warmup
  - worker-side preprocessing and tile assembly
  - zarr / cache-backed tile reads
  - host-to-device, forward, device-to-host, and postprocessing timing
  - partition zarr output writes
  - sampled CPU / RAM, and sampled GPU / VRAM when NVML is available
  - optional `torch.profiler` traces for selected partitions in `profiling-level=detailed`
- `reduce`
  - partition discovery and cache copy from shared storage into local `/tmp`
  - tile-by-tile reduction / merge
  - final tiled TIFF write
  - explicit final S3 upload with exact byte counters
- `aggregate-profiling`
  - reads raw summaries from the shared `_profiling` directory and produces workflow-level rollups

## Artifact Locations

- Per-pod raw traces are written under:
  - `/partitions/<partitions-subdir>/_profiling/<segment-id>/<step>/...`
  - inference pods add `part-<id>/pod-<pod-name>/`
- Each relevant pod also writes fixed local copies for Argo artifact upload under:
  - `/tmp/profiling/profiling-summary.json`
  - `/tmp/profiling/profiling-timeseries.jsonl`
  - detailed inference only:
    - `/tmp/profiling/torch-trace.json`
    - `/tmp/profiling/torch-ops-summary.txt`
    - `/tmp/profiling/torch-memory-summary.json`
- Workflow aggregation writes:
  - `/tmp/profiling-outputs/workflow-profiling-summary.json`
  - `/tmp/profiling-outputs/workflow-profiling-summary.md`
  - `/tmp/profiling-outputs/workflow-profiling-partitions.csv`
  - `/tmp/profiling-outputs/workflow-profiling-partitions.jsonl`

## Reading The Metrics

- `*_seconds` fields are wall-clock durations. They can overlap.
- `preprocess_seconds` includes worker-side read / decode / transform work, so it is not additive with `remote_read_seconds` or `local_read_seconds`.
- `cache_fill_seconds` overlaps with local cache writes when remote zarr chunks are copied into the cache.
- `process_cpu_*`, `process_rss_*`, `process_uss_*`, and `process_pss_*` are sampled from the process tree.
- `gpu_*` and `vram_*` are sampled through NVML when available.
- `torch_cuda_max_memory_*` comes from PyTorch CUDA allocators when CUDA is present.
- The workflow summary classifies each partition heuristically as `compile-bound`, `s3/network-bound`, `efs/local-io-bound`, `gpu-bound`, `cpu-bound`, `memory-bound`, or `mixed`.

## Limitations And Caveats

- Explicit boto3 transfers have exact byte accounting. Hidden zarr / fsspec traffic is timed at the real call site, but byte attribution is only exact when the underlying buffer size is exposed.
- Local image decode is timed around the read-and-decode call. Pure kernel-level file-read time cannot be separated cleanly there, so local read timing is labeled `approximate`.
- Linux `iowait` is recorded only as a system signal. It is not treated as exact process I/O time.
- NVML, USS, PSS, cgroup metrics, or PyTorch CUDA stats may be unavailable depending on the environment. When that happens the summary stores `flag: "unavailable"` plus a reason.

## Expected Overhead

- `profiling-level=basic`: low overhead, dominated by one sampler wake-up per interval plus lightweight scoped timers.
- `profiling-level=detailed`: limited to selected inference partitions and adds `torch.profiler` overhead only to those partitions.
- `profiling-level=off`: writes only minimal stub profiling outputs so the workflow shape remains stable.
