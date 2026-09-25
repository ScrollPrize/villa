# Whole-model training compilation experiment — 2026-09-25

Compiling the three model entry points used by training reduced measured update
time by **32.0%** and increased throughput by **47.0%**, with roughly half the
peak allocated GPU memory. This experiment uses the current encoder-reuse
default. CUDA production training and preflight now enable compilation by default;
use `--no-compile` to select eager training. The launcher forwards this choice,
and training requires a successful preflight result from the same mode. CPU
training remains eager.
No architecture, normalization choice, crop, batch size or precision was changed.

## Method and results

RTX 5090, Linux x86-64, Python 3.14, PyTorch `2.12.1+cu130`, CUDA 13.0,
cuDNN 9.20. Contiguous convolution layout, BF16 autocast, training autotuning,
four CPU threads. Eight fixed real states from the scroll CT/presence/fiber
training data, seed 0, crop 176×96×96, microbatch 2, effective batch 8,
64 flow draws, four midpoint steps. AdamW and EMA are included. Batch loading
is excluded; CPU/GPU transfers are included. Every update is CUDA-synchronized.
There was no competing GPU training process.

The final paired run loads the source once and runs eager then compiled training
in the same process, resetting weights and RNG to seed 0 for each. It uses three
warmup updates followed by 20 timed updates. Source SHA-256 fingerprints match.
This repetition avoids comparing across concurrent confidence-supervision edits
that occurred during the initial separate-process screening runs.

| Metric | Eager, encoder reuse | Compiled, encoder reuse |
|---|---:|---:|
| Mean update | 589.08 ms | 400.86 ms |
| Median update | 589.50 ms | 398.53 ms |
| p95 update | 596.16 ms | 416.25 ms |
| Min–max update | 576.35–605.13 ms | 391.07–421.44 ms |
| States/second | 13.58 | 19.96 |
| Peak allocated memory after warmup | 23.31 GiB | 11.95 GiB |
| Peak reserved memory after warmup | 25.10 GiB | 14.51 GiB |

The initial compiled run with empty dedicated compiler caches took **51.02 s**
for its first update, then averaged 395.10 ms over 20 measured updates. Its
steady-state peak allocation was 11.73 GiB. It used default compiled random
number generation. The final paired run preserves eager RNG draws with
`fallback_random=True`; its first compiled update took 19.36 s with partially
populated caches. This is not a second cold-start measurement.

Dynamo captured 2,580 calls into exactly three graphs, with no reported graph
breaks. The profiles show generated Triton kernels combining GroupNorm,
activation, casts and other elementwise work. The expensive native GroupNorm
row-statistics kernel disappears. Because those operations are fused, individual
GroupNorm time is no longer directly comparable to the eager operator time.
Convolution forward/backward is now the largest remaining GPU cost.

## Applying compilation correctly to this training API

The training loop calls `encode_conditioning`, `generate_training_curve` and
`training_forward` directly. Wrapping only `model = torch.compile(model)` compiles
the module's `forward` call, which this training path bypasses. The experiment
wraps each of the three entry methods, retaining the original module and state
parameter names. Compilation recursively includes their nested model work.
Production uses the shared `train.compile_training_model` helper after copying
the eager EMA and restoring any checkpoint. Compiler wrappers are runtime-only;
checkpoint state dictionaries retain their existing names. Resumed runs log the
actual compilation and encoder-cache modes.
See [PyTorch's compilation tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial).

The experiment uses the default Inductor backend/mode and allows the default
`fullgraph=False`; it does not use max-autotune, CUDA graphs or hand-written kernels.
Two settings matter for a meaningful comparison:

- `torch._functorch.config.backward_pass_autocast = 'off'` matches the existing
  training loop, whose forward runs under autocast but whose backward does not.
- `options={'fallback_random': True}` in the final comparison keeps noise/time
  draws consistent with eager mode. The initial default-RNG run is also saved.
  See the [compiler FAQ](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_faq.html).

## Numerical checks

With identical initial weights, eight real full-size states and identical RNG
draws, a complete logged update produced these differences:

| Quantity | Maximum absolute difference |
|---|---:|
| Generated points / all refinement steps | 0.001953125 voxels |
| Confidence probabilities | 0.001552582 |
| Confidence logits | 0.015625 |
| Per-microbatch flow loss | 0.000401497 |

Total loss was 8.23264509 eager versus 8.23287702 compiled: **0.00282% relative
difference**. Clipped-gradient relative L2 difference was **0.370%**, compared
with **0.113%** between repeated eager runs. All checked outputs and gradients
were finite. Confidence labels and supervision masks matched exactly, and
parameter/state-dictionary keys were unchanged. All 45 local regressions passed
with CUDA access.

After integrating the shared helper and enabling the production default, all
49 GPU regressions passed, including compiled updates with and without logging,
checkpoint save/restore, and another compiled update after resuming. The actual
production preflight passed 20 measured full-crop updates: mean **389.61 ms**,
median **390.38 ms**, p95 **409.51 ms**, or **20.53 states/s**. Its first warmup
update took 38.28 s with populated compiler caches; peak allocation including
startup was 13.20 GiB. The eager EMA tracing smoke completed eight decisions.
This integration run used one CPU thread and a freshly sampled real batch, so
the paired table above remains the controlled speedup comparison.
The normal training entry point also completed two logged full-crop updates,
saved checkpoints, and resumed for update three with finite loss. Its run config
and resume log both recorded compilation and encoder reuse as enabled.

Compilation changes floating-point rounding, so this is not bit-identical
execution. Independent training trajectories diverge over multiple updates,
even with matched random draws. These checks use seeded initial weights and
unit residual scales; they are not a trained-model tracing-quality evaluation.
The benchmark also excludes concurrent collection, data-loader stalls,
checkpoint writes and diagnostics. Other shapes, devices, or logging paths can
require additional compilation.

## Reproduce

Production preflight (records compilation/warmup times separately from measured
updates, then performs an eager EMA tracing smoke test):

```bash
export AGENTS_AGENT_MODE=1
export PYTHONPATH=../../..
../../../../.venv/bin/python scripts/benchmark_single_path.py \
  --manifest output/single_path_v11_preparation/seeds.json --warmup 3 --updates 20 \
  --out output/performance_compile_20260925/production.json
```

Add `--no-compile` for an eager comparison. New production runs need no extra
flag: `bash scripts/launch_single_path.sh RUN_NAME`. Compilation settings do not
change the checkpoint format or require a new model architecture.

Production startup/checkpoint smoke command (use a fresh run name):

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=../../.. \
TORCHINDUCTOR_CACHE_DIR=/tmp/fiber_compile_inductor \
TRITON_CACHE_DIR=/tmp/fiber_compile_triton \
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.train \
  --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
  --ct /mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr \
  --name production_smoke --out-root output/performance_compile_20260925 \
  --fixed-bank output/single_path_v11_preparation/fixed_recovery.npz \
  --manifest output/single_path_v11_preparation/seeds.json \
  --benchmark output/performance_compile_20260925/production.json \
  --steps 2 --workers 0 --flow-calibration-states 8 \
  --diag-every 0 --dagger-every 0 --log-every 1 --ckpt-every 1
```

For the resume smoke, repeat with `--steps 3` and
`--resume output/performance_compile_20260925/production_smoke/last.pt`.
The reduced calibration count and disabled diagnostics/collection are only for
this short integration check; normal production defaults are unchanged.

The experiment scripts, saved input batch, raw results and profiler tables
are retained under `output/performance_compile_20260925/`.
From `fiber_follow`, using the existing project environment:

```bash
export AGENTS_AGENT_MODE=1
export PYTHONPATH=../../..
export TORCHINDUCTOR_CACHE_DIR=/tmp/fiber_compile_inductor
export TRITON_CACHE_DIR=/tmp/fiber_compile_triton
../../../../.venv/bin/python output/performance_compile_20260925/harness/benchmark.py \
  --variant eager compiled --fallback-random --warmup 3 --updates 20 --profile \
  --batch output/performance_compile_20260925/batch.pt \
  --out output/performance_compile_20260925/paired.json
../../../../.venv/bin/python output/performance_compile_20260925/harness/check.py \
  --batch output/performance_compile_20260925/batch.pt \
  --out output/performance_compile_20260925/correctness_labels.json
```

The paired command writes separate `paired_eager.json` and
`paired_compiled.json` files, including raw timings, source fingerprints, runtime
versions, memory peaks and compiler counters. Use a fresh, separately named
compiler/Triton cache directory to measure cold-start compilation. No package
installation or persistent training job is needed.

Regression command:

```bash
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/fiber-compile-pytest
```
