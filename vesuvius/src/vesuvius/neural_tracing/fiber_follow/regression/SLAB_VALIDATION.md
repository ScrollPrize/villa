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

## PyTorch performance audit after the slab migration

The audit baseline is commit `36aeeccff`, with the same v10 model, inputs,
precision, loss coefficients and optimizer as the revised code. Measurements use
PyTorch 2.12.1+cu130 on the RTX 5090, four CPU threads and seed 194. No packages
were installed. Reports and the frozen baseline are in `/tmp/fiber_perf_audit/`.

Changes:

- Replace per-row current-image K/V compaction and Flash calls with one batched,
  padding-masked SDPA call per layer. Retain the measured cuDNN preference and
  fallback backends. Remove dynamic-key annotations and the global Dynamo
  dynamic-output setting.
- Project historical K/V once per head per decision, retaining gradients through
  every layer, refinement and supplied candidate. Keep the same parameter names
  and checkpoint shapes.
- Compile tensor losses and label construction with `fullgraph=True`. Scope
  BF16 cast emulation to the compiled functions instead of changing global
  Inductor configuration.
- Avoid copying already canonical training tensors. Preserve stride normalization
  for indexed singleton dimensions. Reconstruct patch4 dense features only once.
- Remove the retired stream-weight helper from candidate supervision counts.
  Set the patch launcher's accidental microbatch default of 166 to 16.

The attention microprofile uses BF16 Q=(B,4,16,32), K/V=(B,4,20409,32), a
partially masked reference tail, five warmups and twenty timed forward/backward
passes. At B=16, mean times were 9.416 ms for compacted per-row Flash, 1.347 ms
for batched cuDNN, and 20.012 ms for default masked dispatch. PyTorch Profiler
confirmed Flash, cuDNN and efficient-attention kernels respectively. This probe
includes compaction once per call; the full training benchmark amortizes K/V
preparation across attempts and is the appropriate measure of overall speed.
At B=1 the same probe favored compacted Flash (0.587 versus 0.764 ms), so both
single-row and batched training are measured.

Retained deliberately: FP32 physical coordinate sampling, final survival
normalization/projection and log-space survival accumulation; local depthwise
convolution layout; adaptive inference row selection; and eager valid-slab
selection/encoding before the fixed decision graphs. Valid slab counts vary,
including empty histories, and padded slabs must not incur convolution work.
The main encoder, decoder, history attention, candidate scoring and losses run
inside compiled operations. There is no remaining dynamic-key compiler override.

Kernel changes are not bitwise reproducible in BF16. A frozen two-decision,
three-attempt, four-candidate comparison found maximum raw coordinate difference
0.002136 voxels, maximum raw hazard-logit difference 0.007589, and maximum
candidate-logit difference 0.005026. A loss probe over all proposals changed from
2.624654 to 2.625154; the largest per-parameter relative gradient difference was
4.9% (coordinate bias). One near-tied refinement selection changed, making its
selected path differ by 0.1661 voxels despite the small raw-proposal difference.
These are implementation checks, not held-out tracing-quality measurements.

The retained backend preference is hardware-specific, and all fallback backends
remain enabled. PyTorch documents both fused-kernel input restrictions and
backend-dependent floating-point differences in its
[SDPA reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

The final CUDA-enabled run passed **282 tests and 10 subtests**. It covers the
original behavioral suite plus history-attention cache parity/gradients, fixed
input stride handling, one-time dense patch reconstruction, synthetic microbatch
supervision, and CPU/CUDA compiled-loss value and gradient parity. No dependency,
architecture identifier or checkpoint migration changed.

Commands from `fiber_follow`:

```bash
PY=/home/sean/Documents/villa4/vesuvius/.venv/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -m pytest tests -q \
  -o addopts='' -o cache_dir=/tmp/fiber_perf_audit/pytest_cuda
"$PY" -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_slabs \
  --warmup 3 --repeats 10 --decisions 6 \
  --out /tmp/fiber_perf_audit/single_after_final.json
"$PY" -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_slabs \
  --warmup 3 --repeats 10 --decisions 16 --microbatch 16 \
  --out /tmp/fiber_perf_audit/batch16_after_final.json
```

Use fresh output filenames when repeating. Baseline commands and the serial
validation driver are saved in `/tmp/fiber_perf_audit/validate.py`; the baseline
B=16 harness is `production.py`, attention probe is `attention.py`, and frozen
output/gradient comparison is `compare.py` in that directory. The old benchmark
only supported single-row microbatches; `production.py` groups the same synthetic
rows as the new `--microbatch` option, with zero candidate supervision for rows
that have no candidates.

Final steady-state training results (three warmups, ten measured updates per
configuration, synchronous wall timing, no concurrent test jobs):

| Workload | Before mean / p50 / p95 | After mean / p50 / p95 | Before / after peak allocated |
| --- | ---: | ---: | ---: |
| Six decisions, microbatch 1 | 176.53 / 176.38 / 183.38 ms | 187.60 / 187.14 / 191.76 ms | 1.021 / 1.015 GiB |
| Sixteen decisions, microbatch 16 | 206.86 / 206.68 / 213.13 ms | 198.18 / 197.02 / 202.98 ms | 14.682 / 14.132 GiB |

At the documented launcher microbatch of 16, mean update time decreased
4.2% and peak allocated memory decreased 0.549 GiB. The
single-row workload increased 6.3%; batched cuDNN does not win for every shape.
There were no new compiled graphs during any measured update. Both workloads
use full-size patch4 token-only crops with direction channels, two refinements,
and four candidates at the last two decisions. The six-decision ages are
0/40/152/160/184/184; the sixteen-decision ages span 0..184 uniformly.
These benchmarks exclude volume I/O, logging diagnostics and held-out tracing.
