# v11 performance measurements — 2026-09-25

For further measurements against the resulting contiguous-layout baseline, see
[training encoder reuse](PERFORMANCE_ENCODING.md).

The change from `7fcea8c14` to `68a5e02cb` enlarged the crop from 64³ to
176×96×96 (6.19× as many voxels), increased flow draws from 16 to 64,
and expanded the denoiser. `a67f6ce96` added refinement logging, which reuses
existing rollouts. These defaults perform considerably more work.

The measured improvement changes v11 convolution weights from forced
`channels_last_3d` to contiguous layout. The shared preparation helper respects
v11's layout preference and retains channels-last for the beam/legacy models.
Architecture, checkpoint keys, crop, batch sizes, precision, sampling and loss
remain unchanged. Training configuration logs now report the actual layout.

## Results

RTX 5090, PyTorch production wheel `2.12.1+cu130`, CUDA runtime 13.0,
Linux x86-64. Training uses the existing BF16 autocast and cuDNN autotuning;
inference disables autotuning as the tracer does. CUDA is synchronized around
measurements. No competing training job was running.

Inputs are real scroll states from `/mnt/raid_nvme/spiral_dataset_working/fibers`,
with CT/presence volumes from `output/single_path_v11_preparation/seeds.json`.
The final comparison uses the existing preflight benchmark's seed-0 batch:
eight states, microbatch 2, effective batch 8, full 176×96×96 crop, 64 flow draws,
all four midpoint steps and confidence evaluation. Each run initializes from
seed 0 and performs three warmup updates followed by 20 measured updates.
Timings include optimizer/EMA updates and CPU↔GPU transfers; the same CPU batch
is reused, so dataset construction/loading is excluded. The direct dataset
iterator sets CPU threads to one in this benchmark.

| Metric | Channels-last baseline | Contiguous v11 |
|---|---:|---:|
| Mean optimizer update | 893.78 ms | 794.95 ms |
| Median optimizer update | 894.83 ms | 794.44 ms |
| p95 optimizer update | 902.40 ms | 808.36 ms |
| Min–max optimizer update | 867.99–904.87 ms | 777.65–814.46 ms |
| Throughput | 8.95 states/s | 10.06 states/s |
| Peak allocated CUDA memory | 9.195 GiB | 9.303 GiB |
| Peak reserved CUDA memory | 10.018 GiB | 10.150 GiB |

This is **12.4% higher training throughput**, or **11.1% less update time**,
with about 110 MiB more peak allocated memory. An earlier reversed-order trial
(12 measured updates per layout) independently measured 900.33 → 774.60 ms.

A separate comparison uses identical frozen weights and real input crops,
three warmups and 30 measured complete model calls, with CPU threads set to four.
Every call includes encoding, eight velocity evaluations and final confidence.

| Model-only latency | Baseline mean / median / min–max | Contiguous mean / median / min–max |
|---|---:|---:|
| One state | 43.61 / 43.32 / 42.42–45.93 ms | 38.51 / 37.96 / 37.71–41.61 ms |
| Two states | 65.31 / 65.63 / 63.68–69.53 ms | 59.48 / 59.13 / 57.06–63.01 ms |

Both final benchmarks also completed a real-volume 64-voxel trace, with 16
model decisions and `max_len` termination. Their 81.18 / 77.92 ms per decision
are smoke measurements, not a controlled tracing-speed claim: each is one
trace after its own training updates, and I/O/cache state can differ.

## Profile and rejected candidates

A PyTorch CPU/CUDA profile of a baseline optimizer update attributed 30.7% of
self CUDA time to `aten::copy_`, 18.9% to group normalization, 12.3% to forward
convolutions and 11.1% to convolution backward. Copies involving full-resolution
`[2,24,176,96,96]` tensors alone accounted for 120 ms. Large-volume layout
conversions around normalization/interpolation are costly; optimizing only the
transformer would miss much of the workload.

Candidates were injected into temporary instances before modifying production
code. The initial eight-update screening measured:

| Candidate | Mean update | Decision |
|---|---:|---|
| Baseline | 873.98 ms | Reference |
| Contiguous inputs before GroupNorm | 865.91 ms | Marginal; not applied |
| Contiguous inputs before interpolation | 890.30 ms | No gain; not applied |
| Contiguous inputs before decoder blocks | 895.51 ms | No gain; not applied |
| Contiguous convolution weights | 774.37 ms | Repeated and applied |
| Hoist float conversion outside refinement loop | 903.84 ms | No gain; not applied |

The full-resolution features are already float32 in the current autocast path,
so hoisting that conversion does not eliminate meaningful work. A follow-up
also tested contiguous encoder weights alone (849.26 ms) and decoder weights
alone (823.83 ms); changing both was faster (774.60 ms).

## Numerical checks and limitations

With identical weights, seed and two real full-size states, both layouts
produced **bit-for-bit identical** points, all refinement steps, confidence
logits/probabilities, flow loss and total loss. This held with cuDNN autotuning
both enabled and disabled.

CUDA backward is already nondeterministic. With training autotuning enabled,
relative L2 gradient differences were 0.0683% between two baseline runs and
0.0765% between baseline and contiguous; both had maximum absolute difference
0.001953125. With autotuning disabled, these were 0.0714% and 0.0750%.
Consequently independently trained weights/trajectories need not be bit-identical.
No precision reduction, altered objective or custom numerical kernel was added.

These are runtime benchmarks with seeded initial weights and the preflight's
unit flow scales, not a trained-model quality evaluation. Speed is hardware,
shape and backend dependent; remeasure on other GPUs. CPU and macOS execution
retain their previous preparation behavior.

All 39 local regressions passed, including CUDA BF16 determinism, gradient
routes, missing history, checkpoint handling, and the new layout comparison
and shared-helper fallback tests.

## Reproduce

From this directory, using the existing project environment:

```bash
export PYTHONPATH=../../..
../../../../.venv/bin/python scripts/benchmark_single_path.py \
  --manifest output/single_path_v11_preparation/seeds.json \
  --warmup 3 --updates 20 --no-compile --no-cache-training-encoding --conv-memory-format channels_last_3d \
  --out output/performance_20260925/channels_last_3d.json
../../../../.venv/bin/python scripts/benchmark_single_path.py \
  --manifest output/single_path_v11_preparation/seeds.json \
  --warmup 3 --updates 20 --no-compile --no-cache-training-encoding --conv-memory-format contiguous \
  --out output/performance_20260925/contiguous.json
```

Omit `--conv-memory-format` for the model default. The benchmark records raw
samples, mean/median/p95/min/max, warmups, seed, layout and runtime versions.
Neither command launches a persistent training job or overwrites checkpoints.

The project environment lacked pytest; tests used the already-installed sibling
environment, with the workspace source selected explicitly:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/fiber-perf-pytest
```

Local raw results, profiles, numerical comparisons and test output are retained
in `output/performance_20260925/`. Temporary exploration scripts are archived
in its `harness/` directory; they use `/tmp/fiber_perf/batch.pt`, the real
seed-0 training batch constructed by `probe.py`. The model-only comparison is
`check.py`, and `check_training.py` repeats numerical checks with autotuning.
These exploratory scripts are diagnostic artifacts, not production imports.
