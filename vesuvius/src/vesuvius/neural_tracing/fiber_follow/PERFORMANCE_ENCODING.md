# Reusing training encodings — 2026-09-25

This follow-up uses `d4deaee90` (including the contiguous-layout improvement)
as its baseline. Runtime-only candidates were timed before changing production
code. Encoder reuse is now enabled by default: it retains the
encoder and static conditioning graph from the curve-generation pass, then uses
it for flow matching and confidence backpropagation. Each microbatch is encoded
once instead of twice. Cached graphs are released as their backward passes finish.
Inference, model parameters, checkpoint keys, crop, effective/microbatch sizes,
64 flow draws, precision, refinement and loss normalization are unchanged.

## Measurements

RTX 5090, Linux x86-64, PyTorch `2.12.1+cu130`, CUDA 13.0, existing BF16 autocast,
contiguous convolutions and cuDNN training autotuning. CUDA was synchronized
around every timed update. No other training process was running.

The workload reuses eight real scroll states from
`/mnt/raid_nvme/spiral_dataset_working/fibers` and the CT/presence volume in
`output/single_path_v11_preparation/seeds.json`. It uses full 176×96×96 crops,
microbatch 2, effective batch 8, seed 0, AdamW and EMA updates. Loading the dataset
is excluded; CPU/GPU transfers are included. The final preflight comparison
uses three warmups and 20 measured updates per mode, with the implemented cache
option measured first. Its direct dataset iterator sets CPU threads to one.
Memory peaks include warmup and cuDNN autotuning.

| Metric | Previous default (recompute) | Reuse encoding |
|---|---:|---:|
| Mean update | 750.56 ms | 574.64 ms |
| Median update | 750.60 ms | 577.11 ms |
| p95 update | 754.16 ms | 586.32 ms |
| Min–max update | 747.32–756.30 ms | 550.32–589.75 ms |
| States/second | 10.66 | 13.92 |
| Peak allocated memory | 9.30 GiB | 23.30 GiB |
| Peak reserved memory | 10.15 GiB | 26.06 GiB |

This is **23.4% less update time / 30.6% higher throughput**. The initial
8-update runtime-only screen measured 762.17 → 566.39 ms; a repeated 20-update
comparison with four CPU threads measured 776.10 → 579.64 ms (25.3% less time).
Runtime and allocator state vary, but the gain repeated before application.
Both final runs also completed a real-volume 64-voxel trace with 16 decisions
and `max_len` termination (75.16 / 74.97 ms per decision). These are smoke
checks; the optimization concerns training, and no inference speedup is claimed.

A CPU/CUDA profiler attributed 21.7% of self GPU time to GroupNorm, 19.6% to
forward convolutions, 17.2% to convolution backward and 10.7% to copies.
Full-resolution GroupNorm alone took 116 ms per update. Reuse eliminates the
redundant forward encoder/conditioning work without replacing numerical kernels.

Rejected runtime-only candidates: in-place spatial SiLU (772.24 ms), in-place
FiLM addition (762.54 ms), dropping unused final-block history outputs
(757.47 ms), and alternate padding assignment (777.60 ms in the repeated trial).
These did not establish a useful speed advantage over their paired baseline.

## Correctness and limitations

On fixed weights and the real full-size batch, the implemented reuse option
produced bit-identical generated points, all refinement steps, confidence
logits, flow losses, total loss and training metrics. Full-update clipped
gradient relative L2 differences were 0.111% between repeated baseline runs
and 0.120% between baseline and reuse; maximum absolute differences were
0.0000342 and 0.0000550 respectively. CUDA backward is already nondeterministic,
so independently trained trajectories need not be bit-identical.

CPU regressions compare two complete updates, including gradients, AdamW-updated
weights, EMA, refinement metrics, missing history, censoring and departures.
They also check that encoder calls halve and preflight results match the cache
mode. Reuse rejects batch normalization because reducing its forward calls
would change running statistics. The production model uses group normalization.

All 45 local tests passed with CUDA access; the launcher also passes `bash -n`.

The memory cost is substantial, especially when a background collector shares
the same GPU. The benchmark does not exercise simultaneous
training and collection, and a successful isolated preflight does not establish
that both fit together. Use `--no-cache-training-encoding` if memory is tight; microbatch 1
still retains the entire effective batch's encoder graphs. Speed/memory figures
are specific to this GPU and workload, not a trained-model quality evaluation
or a claim about data-loading or inference speed.

## Reproduce

From `fiber_follow`, with the existing project environment:

```bash
export AGENTS_AGENT_MODE=1
export PYTHONPATH=../../..
../../../../.venv/bin/python scripts/benchmark_single_path.py \
  --manifest output/single_path_v11_preparation/seeds.json \
  --warmup 3 --updates 20 --no-compile --no-cache-training-encoding \
  --out output/performance_encoding_20260925/default.json
../../../../.venv/bin/python scripts/benchmark_single_path.py \
  --manifest output/single_path_v11_preparation/seeds.json \
  --warmup 3 --updates 20 --no-compile --cache-training-encoding \
  --out output/performance_encoding_20260925/cached.json
```

New runs enable reuse automatically (the launcher also forwards either override
to preflight):

```bash
bash scripts/launch_single_path.sh RUN_NAME
```

The option also works with `train.py --resume`; supply a benchmark JSON produced
with the same cache mode. It does not change checkpoint state.

```bash
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/fiber-perf-round2-pytest
```

Raw samples, profiles and correctness comparisons are in
`output/performance_encoding_20260925/`. Its `harness/` directory archives the
runtime-only probes; they use `/tmp/fiber_perf/batch.pt` constructed by the
previous performance experiment's `harness/probe.py`. These are diagnostic
artifacts, not production imports. The preflight commands above recreate their
own seed-0 batch directly from the real data.
