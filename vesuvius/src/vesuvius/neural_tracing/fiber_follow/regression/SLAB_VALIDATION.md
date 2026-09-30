# Historical slab validation

Measured on 2026-09-30 with the existing project environment, PyTorch
2.12.1+cu130 and an NVIDIA GeForce RTX 5090. No dependencies were installed.
Local reports and the frozen learning fixture are in
`output/slab_validation_v10/`; previous output artifacts were preserved.

## Training cost

Both runs use the production patch4 token-only main encoder, hidden width 128,
two refinement steps, full 120x101x101 current crops, BF16 autocast, FP32 survival,
AdamW, compilation, seed 194, three warmup updates and ten measured updates.
Each update supervises six decisions, with four supplied candidates at each of
two decisions. Observed lengths match: 0, 40, 152, 160, 184 and 184 voxels.
The old implementation reconstructs two causal 24-observation streams; the new
implementation uses independent decisions with a mean of 3.5 valid slabs.

| Measurement | Recurrent baseline | Live slabs |
| --- | ---: | ---: |
| Parameters | 3,023,906 | 3,001,319 |
| Historical encoder/writer parameters | 253,698 | 115,016 |
| Mean update | 1267.12 ms | 313.07 ms |
| Median update | 1246.94 ms | 255.70 ms |
| p95 update | 1507.82 ms | 460.71 ms |
| Supervised decisions/second | 4.74 | 19.16 |
| Peak allocated GPU memory | 1.310 GiB | 1.021 GiB |

This workload measured **4.05x throughput** and **22.0% less peak allocated
GPU memory**. Measurements exclude volume loading and are not a prediction for
end-to-end training or an all-eight-slabs workload. The models and objectives
changed as requested; these numbers compare total training work per supervised
decision, not numerically equivalent implementations. Timing variability is
visible in the mean and p95. Raw per-update samples are saved in `baseline.json`
and `slabs.json`.

The baseline was frozen from commit `f88466ef8` before implementation in
`/tmp/fiber_slab_baseline/vesuvius/src/vesuvius/neural_tracing/fiber_follow`.
Commands run from `fiber_follow`, with `PY` set to the existing interpreter:

```bash
PY=/home/sean/Documents/villa4/vesuvius/.venv/bin/python
"$PY" -c 'import runpy; import vesuvius.neural_tracing.fiber_follow as p; p.__path__=["/tmp/fiber_slab_baseline/vesuvius/src/vesuvius/neural_tracing/fiber_follow"]; runpy.run_module("vesuvius.neural_tracing.fiber_follow.regression.benchmark_sparse_training",run_name="__main__")' \
  --encoder patch4 --token-only --length 24 --streams 2 --warmup 3 --repeats 10 \
  --out /tmp/fiber_slab_baseline/baseline_final.json
"$PY" -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_slabs \
  --warmup 3 --repeats 10 --decisions 6 \
  --out /tmp/fiber_slab_baseline/slabs_final.json
```

## Controlled learning and history ablations

The controlled set contains two matched pairs with identical current inputs.
The first pair differs in seed CT; the second requires non-seed historical CT.
Path heatmaps and metadata stay unchanged when CT is shuffled. After 250 updates
with seed 491, mean geometry error was **0.00436 tracing voxels**, all four
generated paths were accepted, and the full-history geometry and candidate
acceptance/rejection checks passed.

| History input | Geometry choice | Correct continuation accepted | Wrong continuation rejected |
| --- | ---: | ---: | ---: |
| Full | 100% | 100% | 100% |
| Seed only | 75% | 75% | 75% |
| None | 50% | 50% | 50% |
| Shuffled CT, unchanged metadata and heatmap | 0% | 0% | 0% |

```bash
"$PY" -m vesuvius.neural_tracing.fiber_follow.regression.history_learning \
  --device cuda --steps 250 --out /tmp/fiber_slab_baseline/learning4.json
```

This is an intentionally small learning-capability check with a reduced model
and FP32 training. It establishes that historical imagery can control both heads
and that rejecting everything cannot pass. It does not establish generalization
or held-out tracing quality. No production checkpoint was freshly trained and
no new held-out rollout result is claimed. Existing held-out evaluation remains
in the training loop.

## Real inputs and regression coverage

A real-data preflight loaded an independent two-decision batch from the local
fiber annotations, CT volume and 5,946-shard neighbor bank in 1.589 seconds.
This includes both current crops and historical slabs; it is a single cold batch,
not a throughput estimate. Its configuration and source paths are recorded in
`data_preflight.json`.

A second preflight enabled direction channels for the current crop and ran the
production patch4 token model's compiled CUDA training step. Its two real
decisions averaged 4.5 valid slabs. Loss was 2.38184 and historical encoder
gradient norm was 2.99099; both encoders had nonzero gradients and all gradients
were finite. The report is `cuda_preflight.json`. This check uses zero learning
rate to validate backpropagation without claiming a trained checkpoint.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" \
  -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight \
  --encoder patch4 --token-only --microbatch 2 --batches 1 \
  --out /tmp/fiber_slab_preflight
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" \
  -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight \
  --encoder patch4 --token-only --direction-inputs --device cuda --forward \
  --microbatch 2 --batches 1 --out /tmp/fiber_slab_cuda_preflight
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -m pytest tests -q \
  -o addopts='' -o cache_dir=/tmp/fiber-slab-pytest
```

The retained suite covers this model and training loop, including their data,
replay, inference and diagnostics. Tests for other model families, obsolete
samplers, standalone mining and unrelated utilities were removed on request.
The final run passed **275 tests and 10 subtests in 51.28 seconds**, with CUDA
available and the normally excluded slow learning test explicitly enabled.
The only warnings were PyTorch's existing Python 3.14 TorchScript deprecations.
The complete output is saved as `tests.log` alongside the other local reports.
Slab checks cover arclength spacing, boundary heading fits, loops, observed wrong
turns, remote CT reads, holdout rejection, causal replay and resume parity,
gradients from geometry and both confidence losses, and masked-slot invariance.
CUDA checks exercise BF16 compilation and optimizer updates as valid slab counts
change through two, zero, eight and one, including NaNs in masked inputs.
