# CPU batch preparation — 2026-09-25

The prepared-batch model benchmark excludes the data pipeline. In the live
compiled run, updates 51–100 achieved 9.50 states/s and updates 101–150 achieved
9.20 states/s, versus 20.83 states/s on a reused batch. The training log reports
a cumulative rate including compilation, so its initial displayed rate was
lower still. Four CPU data workers each used approximately one core while the
trainer frequently waited on interprocess communication and the GPU went idle.

## Profile and change

A single-worker CPU profile used 24 real training states after two warmup
microbatches, including the same fixed replay bank as production. It measured
904 ms per two-state batch (2.21 states/s). Of its 10.85 seconds:

- Fused CT sampling/history rendering: 6.86 s (63%).
- Block reads: 2.00 s (18%, including CT and presence).
- Presence resampling: 1.71 s (16%, excluding its block reads).
- Label generation was under 1%.

`fast_sample.py` previously checked every valid history point/segment for nearly
every voxel behind the current position, including voxels far from the history.
It now first computes a bounding box around valid history and any connected
origin. Locations more than 16 Gaussian sigmas outside that box have a history
value below `exp(-128)`, which already rounds to zero in the float32 sampler
output. Those locations skip the distance loop. Every segment lies within its
endpoints' box. Masked gaps stay disconnected, and the existing forward cutoff
is retained. This does not change precision, Gaussian width, sampling geometry,
input channels, batch size, label generation, or the training objective.

The optimization was benchmarked in a separate candidate module before editing
production. All experiments were CPU-only, used low-priority processes and
single-threaded worker computation, and ran alongside the user's training job.
No GPU allocation, trainer interruption, or restart was needed.

## Paired preparation results

Two passes over the same 12 captured real microbatches (48 states per variant),
alternating original/candidate order per batch. Each variant has its own 1 GiB
decoded-chunk cache; both implementations are warmed before measurement. Crop
176×96×96, microbatch 2, CT + presence + history, history sigma 0.35, connected
segments, up to 128 history points. Preparation includes reads, decompression,
CT/history sampling, presence sampling, and assembling all batch tensors.

| Two-state preparation time | Original | Optimized |
|---|---:|---:|
| Mean | 890.21 ms | 398.84 ms |
| Median | 931.40 ms | 350.13 ms |
| p95 | 1191.21 ms | 654.59 ms |
| Min–max | 458.09–1295.86 ms | 200.17–720.21 ms |

This is **55.2% less preparation time**, or **2.23× throughput** for this paired
single-worker workload. Isolated sampler time over 24 real states fell from
281.89 to 39.63 ms/state (7.11×). All output tensors were bit-identical for all
48 paired states; the sampler's float32 outputs also matched exactly on all
24 states before conversion to float16.

## Four-worker streaming results

The real DataLoader, including fresh/replay generation and transfer to the
parent process, measured the following over 64 microbatches (128 states) per
variant after eight warmup microbatches:

| Metric | Original | Optimized |
|---|---:|---:|
| Wall time for 128 states, including output hashing | 16.94 s | 8.70 s |
| Loader throughput | 7.56 states/s | 14.71 states/s |
| Mean wait for a two-state batch | 255.89 ms | 126.47 ms |
| Median wait | 38.71 ms | 0.28 ms |
| p95 wait | 1033.48 ms | 528.23 ms |

That is **1.95× loader throughput**. The complete output SHA-256 hashes match:
`3c7fb3a0240257ef746cb3a3a2e5b89dee2c33d292a01242ebb1febc5ebd1941`.
Each benchmark used four additional low-priority CPU workers with 1 GiB cache
per worker while the existing four-worker training job continued. These rates
include that resource contention; they are not an idle-machine capacity test.

These are CPU preparation measurements, not a new end-to-end training rate.
Concurrent training and storage/cache activity introduce timing variation.
Larger history widths or histories spread across the crop leave less work to
skip. Existing persistent loader workers retain their loaded implementation;
the running job gets this change only after a restart/resume.

Validation: **93 CPU tests passed, 4 CUDA tests skipped** to leave the active
trainer's GPU memory untouched. The new exhaustive-reference tests cover point
and segment rendering, four Gaussian widths, masked gaps, empty/missing history,
the origin connection, repeated endpoints, and nonzero float32 subnormal tails.
The measured candidate and production implementation have identical code ASTs.

## Reproduce

Scripts, the original and candidate samplers, captured items, cProfile output,
and JSON results are retained under `output/performance_batch_20260925/`.
They use the geometry and data paths from
`output/v11_compiled_20260925_144856/config.json`, including its fixed bank.
From `fiber_follow`, with the existing project environment:

```bash
export AGENTS_AGENT_MODE=1 PYTHONPATH=../../..
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
nice -n 10 ../../../../.venv/bin/python output/performance_batch_20260925/profile_batches.py
nice -n 10 ../../../../.venv/bin/python output/performance_batch_20260925/compare.py
nice -n 10 ../../../../.venv/bin/python output/performance_batch_20260925/loader_benchmark.py \
  --variant baseline --out output/performance_batch_20260925/loader_baseline.json
nice -n 10 ../../../../.venv/bin/python output/performance_batch_20260925/loader_benchmark.py \
  --variant candidate --out output/performance_batch_20260925/loader_candidate.json
```

The streaming benchmark uses four real DataLoader workers, eight warmup
microbatches, and 64 measured microbatches (128 states). It includes generating
fresh/replay states, reading/rendering them, and worker-to-parent transfer. It
also hashes every output tensor to verify the same batches across variants;
hashing is included in reported wall time.

CPU regression command (avoids allocating GPU memory during another run):

```bash
CUDA_VISIBLE_DEVICES='' AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/fiber-batch-pytest
```
