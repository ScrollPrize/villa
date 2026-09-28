# Axial fiber follower

## Unified recurrent architecture

New memory training defaults to `--memory-version 3`, checkpoint architecture
`axial_fiber_unified_v1`. Launch a fresh run with:

```bash
bash scripts/launch_unified.sh
```

The unified model has one full-crop axial observation encoder. It encodes the
current crop, original seed crop, and historical crops with the same parameters,
without fiber-identity conditioning in the appearance encoder. There is no
separate patch encoder. The immutable seed retains a 27-point descriptor stencil;
a bounded recent-observation queue retains head descriptors and poses; adaptive
slots compress temporal context. Every descriptor comes from that same encoder.

One path evaluator attends to the current image, visible observed path, seed,
recent observations, and adaptive slots. It is reused for initial proposals,
each coordinate correction, final prefix confidence, supplied candidate curves,
and auxiliary per-observation identity/offset queries. Only the small output
projections differ. Each candidate samples its own features and reads memory;
scoring hypothetical curves never writes them into memory. Observed heads write
once per decision. Confidence sampling coordinates remain detached from BCE.

Cosine identity attention uses the same normalized embedding projection as
InfoNCE, with the configured identity temperature and an explicit null key for
unmatched observations. This evidence enters the shared evaluator before the
geometry/confidence split. The confidence-only eight-feature shortcut is absent.
InfoNCE supervises individual labeled original-fiber visible references and the
stored original seed, including when the seed leaves the crop. Membership labels
filter losses only, never the model's observed references. Adaptive slots are
context, not falsely treated as individual contrastive fiber descriptors.

Training re-encodes full historical crops with current weights, oldest first.
Old observations form no-gradient burn-in; recent encodings use activation
recomputation. Historical images stay on the CPU until each observation is
encoded. The current crop is read/encoded only once, and full seed/history crop
footprints are checked against the holdout before I/O. Replay uses saved observed
frames; history reconstruction is a fallback for older caches. Inference carries
only the bounded state and encodes one new full crop per decision after startup.

The launcher retains 16 effective batch size, 64 historical observations and 32
gradient-bearing history steps, using microbatches of 2, two loader workers,
activation checkpointing, and eager execution for the streamed sequence loop.
Full-crop replay is substantially more expensive than the legacy patch unroll.
Legacy checkpoint architectures remain loadable; they cannot initialize this
architecture. Use `--memory-version 2` explicitly for legacy memory training.

The remaining architecture description documents the crop-only and legacy
patch-memory models.

This model follows a marked fiber through a single heading-aligned CT crop. The
default architecture is `axial_fiber_v3`. Set `--memory-slots 16` to use
`axial_fiber_memory_v2`, described below. Existing v3 and `axial_fiber_memory_v1`
checkpoints remain loadable; v1 checkpoints cannot initialize new runs.

## Architecture

The input is CT plus presence from the existing `s1_ds2` volume. The fine crop is
120 × 101 × 101 samples, with its head at depth index 48 and spacing 0.5 trace-grid
voxels (one finest-level CT sample). Its physical extents are ±25 trace voxels
transversely and −24 through +35.5 along the forward axis.

A residual 3D CNN extracts 32-channel full-resolution features. A stride-two
stage produces 64 channels, followed by 128 channels and another residual CNN
block before compression. A learned 4 × 1 × 1 convolution, stride four along
forward, produces 128-channel tokens with effective spacing 8 × 2 × 2 CT samples.
The token grid is 15 × 51 × 51: 39,015 tokens. Compression centres are at input
coordinates `(8*z + 3, 2*y, 2*x)`. Spatial sampling and dense reconstruction use
this exact lattice, including its depth offset.

Four residual axial blocks apply full bidirectional attention along X, Y, and
forward/backward, followed by one expansion-four MLP and a depthwise 3D convolution
with pointwise channel mixing. Four heads are used. Physical positions and
rasterized observed history/seed cues condition the tokens. Ground-truth fiber
membership never enters the model inputs.

A lightweight decoder upsamples contextual features and combines them with the
full-resolution stem skip. One shared 32-channel dense feature map supplies localization,
correction, confidence, and the contrastive projection. Four path-decoder layers
predict the existing 16 future points at one trace-voxel spacing. Two bounded
corrections refine the curve. The same prefix confidence classifier scores
predictions and explicitly supplied candidate curves. The default model has
4,464,453 parameters.

The confidence head receives explicit cosine comparisons between each candidate
point and two separate references: the visible observed seed, and pooled visible
recent history. Each has point similarity, prefix mean, prefix minimum and valid
coverage. No annotation membership is used for these inputs. Missing references
have zero evidence and zero coverage; contaminated history cannot average away
the separate seed comparison. Predicted paths and supplied candidates use this
same head. The eight added inputs start with zero weights and learn through
prefix/candidate BCE, with gradients also reaching the shared embedding projection.

The default crop-only model has no coarse image branch, separate appearance
encoder, remote CT patch reader, or persistent embedding cache. Its history and seed features are sampled only
inside the current crop. References outside it have their position, orientation,
age and features masked out before they can influence the network. Seed metadata
is retained in rollout/replay records so that visibility can be determined later;
this does not expose out-of-crop seed information to the model.

## Training and supervision

The full existing annotation set, holdout split and live neighbor bank are used.
The default launcher uses `output/neighbor_samples_r0_32_l80_160_v2`; its miner can
continue publishing shards independently. All contrastive positive and negative
queries must lie inside the main crop, with the same support and presence checks.
No remote positive or negative patches are encoded.

Losses combine direct curve regression, prefix confidence, matched candidate BCE,
and auxiliary InfoNCE computed from the shared dense features. Observed history
references on the labeled fiber are pooled for InfoNCE; a visible original seed
can supply the reference when history is insufficient. Membership annotations
select supervised terms only and are never supplied as input cues.

Matched cases use identical current CT/history and different **visible** observed
seeds. Their short 4/8/12 trace-voxel tails leave both references inside the crop.
Old departure states with no visible original-fiber reference have identity and
confidence supervision masked, rather than receiving contradictory labels for
indistinguishable inputs. Geometry is already masked for confirmed departures.
The `identity_observable_fraction` metric records this coverage.

The default crop-only model learns local visual continuity. It cannot verify original-seed identity
once every distinguishing observation has left its crop. Full tracing evaluation
still measures departures from the original annotation, including this case.

## Launch and logs

### Optional learned observation memory

`--memory-slots 0` (the default) retains the existing crop-only parameters and
computation. Positive slots enable a separate checkpoint architecture with:

- A small shared CT/presence patch encoder: two strided 3D convolutions followed
  by eight spatial tokens per observation. Default patches are 17³ samples at
  spacing 0.5, covering eight trace voxels along each axis.
- A learned bank of adaptive slots. Cross-attention reads observations and a
  sigmoid gate learns which features to retain or replace in each slot.
- Eight seed tokens retained separately, with immutable observed position and
  tangent frame. An absent seed stays masked; replay without a saved original
  seed does not invent one from possibly contaminated history.
- Separate memory attention in the path decoder, before both coordinate and
  confidence prediction. Relative motion and original-seed pose are expressed
  in the current observation frame; world coordinates are not learned identifiers.

Training reconstructs `--memory-steps 32` past observations at
`--memory-stride 4` trace-voxel spacing from the existing saved observed history,
then appends the current head. It re-reads these small patches from the static
volume and backpropagates the current decision's task losses through the entire
memory unroll and patch encoder. It does **not** unroll the expensive full-crop
backbone or differentiate through the tracing policy. Missing history is padded
and masked. Fresh, drifted, matched, and replay observations retain their
existing sampling distributions. Geometry annotations never enter the writer;
they only determine targets and supervision availability.

Online tracing initializes from the original seed and any supplied history,
then reads one new head patch per decision and carries the bounded latent state
forward. Each directed trace has independent state; every new `trace()` call
resets it. Writes use observed heads, never predicted future points. The tracer
automatically enables memory when loading a memory checkpoint; there is no
inference switch to accidentally leave it disabled.

### Long observed sequences (memory v2)

The v2 memory trains its recurrence on the model's own trajectories, where it
can drift or move onto a neighbor while its history looks plausible:

- **Observed tracks.** Online collection records every decision's observed head,
  in order, with its relabeled original-fiber status. Each replay row stores the
  slice of earlier heads in its own trace (`seq_start`/`seq_end` over `track_*`
  arrays in the replay archive). Training replays exactly the writes the tracer
  made, at its real per-decision cadence, starting from the original seed.
  Caches without tracks load unchanged and fall back to the reconstruction above.
  `--dagger-after` (default 24) sets how far collected traces continue past a
  confirmed departure; longer values record long wrong continuations.
- **Burn-in.** `--memory-steps` bounds the sequence (reconstructed histories still
  supply at most `n_history/memory_stride` observations). Only the newest
  `--memory-grad-steps` writes and the head backpropagate; older writes build
  the state without gradients. Equal values reproduce the v1 behavior.
- **Per-write probe.** After each write, a small head reads the slots and seed
  anchor and predicts whether the trace is still on its original fiber and the
  offset back to it (within the recovery distance), in that observation's frame.
  Tracks label every write; other states label only the head. The partial
  bridge of a switch, burn-in writes and states whose memory cannot distinguish
  the original fiber are unlabeled. Targets are auxiliary losses only
  (`--memory-probe-weight`, default 0.5) and never enter the writer.
- **Switch sequences.** `--memory-switch-probability` replaces that fraction of
  fresh draws with an original-fiber prefix as long as the memory sequence,
  a smooth bridge, and a `--memory-switch-tail` neighbor tail from the live bank.
  The head is a confirmed departure; the original seed lies behind the prefix.
- **Identity read at initialization.** The decoder's memory read starts at zero,
  so initializing from a crop-only checkpoint leaves its predictions unchanged
  until training moves it. The probe trains the writer from the first update.

Logs add a `memory probe` line: identity loss and accuracy, departed-write
recall, offset loss and error, labeled writes per state, and the fraction of
states with a departed label. Tracked replay appears after the first
online collection completes (`dagger_states`).

All historical and seed patch read footprints are checked against the holdout
before training I/O. Confirmed departures can receive decision supervision when
an original-fiber observation is available in memory even if it is outside the
main crop. Supervision masks never filter the writer's observations. Presence
dropout and photometric augmentation also apply to memory patches.

Start a **new** memory experiment using the existing launch configuration:

```bash
RUN_NAME=axial_memory_seq_run1 bash scripts/launch_axial.sh \
  --memory-slots 16 --memory-steps 64 --memory-grad-steps 32 --memory-stride 4 \
  --memory-patch-size 17 --memory-probe-weight 0.5 \
  --memory-switch-probability 0.15 --memory-switch-tail 16 96 --dagger-after 96
```

To initialize its existing backbone/decoder/scorer from a trained crop-only
model, append:

```bash
  --init-tracer output/axial_crop_822_run2/ckpt_015000.pt
```

This copies the source EMA weights, initializes the new memory parameters, and
starts a new optimizer/schedule in the new output directory. The v2 memory read
starts at zero, so initial predictions match the source; this is not a quality guarantee.
Use the existing resume workflow with the same memory options to continue a
memory run. A memory checkpoint cannot be silently converted to crop-only.
The original run is not migrated or modified by these commands.

Input validation, with no training job started:

```bash
PYTHONPATH=../../.. ../../../../.venv/bin/python \
  -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight \
  --out output/memory_preflight --memory-slots 16 --memory-steps 64 --memory-grad-steps 32 \
  --memory-switch-probability 0.15 --microbatch 4 --batches 8
```

`--onpolicy <decisions.npz>` adds replay caches, for example ones with tracks.

Add `--forward --device cuda` when a GPU is available to check full-crop
forward/backward. The memory path has CPU gradient, streaming/unroll parity,
checkpoint, holdout, and Dynamo graph-capture tests. Production CUDA/BF16 and
Inductor performance require a separate preflight; `--no-compile` above avoids
assuming those measurements have been made.

Implementation validation used 16 slots (238,960 extra parameters; 4,703,413
total). A full-size CPU optimizer update on two real training examples initialized
from the step-15,000 EMA produced finite gradients, including gradients reaching
the earliest historical patches. Checkpoint weights reloaded exactly and predicted
points agreed within 3.6e-7. The report and reproducible smoke script are in
`output/memory_validation/`. A synthetic retention test also learns a distinction
available only in an observation 32 voxels behind the head. These checks establish
that the memory can train and persist; tracing quality has not yet been measured.

`memory_observations_mean` and `memory_anchor_fraction` are logged alongside the
existing task metrics. Sequence lengths affect CPU I/O and training activations;
the number of memory slots bounds inference state size, not the sequence-training
cost. The first version trains retention over at most the stored 128-voxel
history. Inference can run longer, but long-range accuracy is unproven. Replay
rebuilds memory from that bounded history plus the original seed, rather than
reproducing latent states from the entire original rollout. Historical head
frames are reconstructed from observed tangents; they may differ from the exact
frames used by the collecting model. Longer sequences and more natural failure
sampling remain separate experiments.

### Crop-only default

```bash
bash scripts/launch_axial.sh

tail -f output/logs/axial_crop_822_run2.log
```

The launcher starts a fresh 100,000-update experiment: effective batch 16,
microbatch 4, 12 loader workers, BF16, torch.compile, no activation checkpointing, AdamW at 3e-4, 500 warmup
updates and EMA 0.999. It refuses to overwrite a run. Set `RUN_NAME` and `BANK_PATH`
to change destinations. Extra training options can be appended.

Checkpoints and structured logs are written under `output/axial_crop_822_run2`.
The current architecture can resume its own checkpoints with the same run options
and `--resume PATH`; there are no old-model migration paths. Frozen monitor seeds,
recovery probes, online replay collection and neighbor-aware evaluation remain in
use. Compare coverage, precision, departure count, wrong-fiber length and candidate
acceptance/rejection, in addition to losses and contrastive ranking.

## Validation

No dependency installation is required. This workspace already has a training
Python environment at `../../../../.venv/bin/python` and an existing test environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
MPLCONFIGDIR=/tmp/axial-mpl PYTHONPATH=../../.. \
/home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/axial-pytest
```

Use the real-data preflight to check crop support, visible references, bank
negatives and forward/backward gradients at the production crop dimensions:

```bash
PYTHONPATH=../../.. ../../../../.venv/bin/python \
  -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight \
  --out output/axial_preflight --device cuda --forward --batches 4
```

This is a batch validation, not a reduced training experiment. The preflight saves
its first batch and a JSON report. `tests/test_axial.py` covers the physical token
lattice, bidirectional global context, within-patch forward order, reference
visibility, supervision masks, gradient flow and current-checkpoint round trips.

## Measured implementation cost

The initial and kernel-optimization measurements below describe the earlier
16-channel model. The current default's capacity comparison follows them.

On the RTX 5090, BF16, microbatch 4, torch.compile, four warmup iterations and
12 measured full forward/backward/AdamW iterations on seeded representative
tensors (no data I/O):

| Model | Mean update | Median | P95 | Peak allocated VRAM |
|---|---:|---:|---:|---:|
| Previous encoders and identity heads | 65.6 ms | 65.4 ms | 68.2 ms | 4.07 GiB |
| Axial crop-only, no activation checkpointing | 200.8 ms | 201.6 ms | 204.1 ms | 9.72 GiB |

Before kernel optimizations, the measured GPU update cost was about 3.06x,
higher than the earlier MAC estimate.
A real-data profiler identifies masked path cross-attention backward and grouped
3D convolutions as substantial costs. Activation checkpointing reduces memory
but adds recomputation; it is available as an option and disabled in the launcher.
These timings do not include CT reads, workers, monitoring or online collection.

Reproducible benchmark scripts, JSON measurements, a real-data optimizer/checkpoint/
trace integration report, and the profiler table are saved under
`output/axial_implementation_validation/`. The benchmark uses the previous run's
saved source snapshot for its baseline; no old-model loader remains in production.

### Kernel optimization measurements

The two largest avoidable costs have been reduced without changing crop size,
parameters, precision or training objectives. Depthwise convolutions now use
contiguous inputs **and weights** locally; ordinary convolutions retain their
channels-last layout. A singleton weight dimension made an input-only conversion
ineffective. Masked path-decoder cross-attention prefers cuDNN, retaining other
PyTorch backends for unsupported devices/dtypes. Axial attention is unchanged.

On the same RTX 5090/PyTorch 2.12.1, a saved real CT batch of four states, BF16,
torch.compile, complete forward/loss/backward/gradient clipping/AdamW, five warmup
iterations and 12 measured iterations, excluding data I/O:

| Axial implementation | Mean | Median | P95 |
|---|---:|---:|---:|
| Before kernel changes | 206.4 ms | 206.7 ms | 211.6 ms |
| Depthwise layout only | 174.0 ms | 174.1 ms | 177.5 ms |
| Masked attention only | 152.2 ms | 152.4 ms | 154.5 ms |
| Both | 119.3 ms | 119.1 ms | 121.5 ms |

This is 42.2% less GPU step time, or 1.73x throughput for this workload. It is not
an end-to-end data-loading or tracing throughput claim. Different BF16 reductions
are not bitwise identical: on the initial comparison batch the combined change
altered total loss by 0.000162, confidence probabilities by at most 0.000305 and
predicted coordinates by at most 0.0177 trace voxels. CPU double-precision and GPU
BF16 tests check outputs, input/parameter gradients and masked-reference invariance.

Reproduce the measurements from the fiber-follow directory with the GPU idle:

```bash
PYTHONPATH=../../.. ../../../../.venv/bin/python \
  output/axial_speed_validation/benchmark_full_step.py
```

The benchmark's fixed input, frozen baseline source, profiler tables, individual
kernel probes and numerical comparisons are recorded under
`output/axial_speed_validation/`. No extra packages or custom CUDA kernels are
needed. See [TRAINING_REVIEW.md](TRAINING_REVIEW.md) for the separate training
simplification review; those proposed training changes have not been applied.

### Current default: wider CNN with residual processing before compression

The default is now 32 → 64 → 128 CNN channels with a residual block on the
stride-two 128-channel grid immediately before learned forward compression.
The axial width, depth, crop dimensions and token spacing remain as above.

Measured with the optimized kernels on the same RTX 5090, BF16, torch.compile,
real CT microbatch of four, five warmups and twelve full measured optimizer steps:

| Configuration | Parameters | Mean | Median | P95 | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|
| Earlier 16-channel model | 2.93M | 116.5 ms | 117.3 ms | 118.1 ms | 9.72 GiB |
| Current default | 4.46M | 180.1 ms | 181.4 ms | 186.0 ms | 14.48 GiB |

The increase is 54.6% in GPU step time and 4.75 GiB in allocated memory for this
workload; data I/O, EMA and monitoring are excluded. Every measured step had a
finite loss and gradient norm. This establishes compute cost, not model quality.
The fixed input, frozen model source, script and measurements are under
`output/axial_capacity_validation/`. Reproduce with:

```bash
PYTHONPATH=../../.. ../../../../.venv/bin/python \
  output/axial_capacity_validation/benchmark_stem.py
```
