# Axial fiber follower

## Spatial history retrieval (memory version 4)

`axial_fiber_spatial_memory_v2` keeps the existing shared axial encoder and
replaces the historical descriptor bottleneck with spatial memory. It is a
separate checkpoint architecture; existing crop, patch-memory and unified
checkpoints still load through their original models.

Each observation retains its complete axial token grid (39,015 × 128 at the
production crop), its pose and age, and one fine descriptor at the observed
head. The descriptor preserves the supervised identity space and fine detail
from the encoder's full-resolution skip connection; the grid supplies spatial
context. Candidate queries likewise use one fine descriptor, one support flag,
and the point's coordinates. Geometry, corrections, confidence and historical
probes use this same point-based evaluator, with no 27-point stencil. Supported
points may reach the crop edges; the legacy `patch_radius` margin does not apply
to spatial memory, including its training evidence sampling.

The dense decoder still computes the fine descriptors; this change does not
claim an encoding speedup. Spatial v1 checkpoints cannot resume as v2 because
the query weights and observation layout changed. Start a fresh spatial run;
crop, patch-memory and unified checkpoint architectures remain unchanged.

The seed grid is immutable and always readable, with the seed origin/direction
defined by its observation frame. Current image tokens, visible observed-path
tokens with explicit ages, and the two most recent historical grids are always
readable. A learned summary of each older grid supplies an archive retrieval
key. Each candidate's seed-conditioned path queries score those keys. The
highest-scoring two observation blocks are read at full spatial resolution,
with separate attention weights for each path point. A soft read over all
archive summaries trains the routing parameters, including unselected entries.
Selection is bounded per curve, not an independent top-k for every point;
stable ties prefer newer observations. Geometry, corrections, confidence,
candidate evaluation and historical probes use the same evaluator.

Defaults retain two recent observations and eight additional older grids.
The current head occupies a separate bank slot and is not duplicated among
the recent history when decoding it. Oldest archive entries are evicted first.
Sixteen learned recurrent slots read the full observation grid and preserve
compressed context after eviction. Evicted spatial detail is not recoverable:
this is a bounded archive, not an unlimited historical store. Capacities are
explicit via `--spatial-recent`, `--spatial-archive`, `--spatial-retrieve` and
`--memory-slots`. World poses remain float32; feature storage follows the
model's parameter dtype, independent of the surrounding autocast context. The default bank plus
seed alone is about 229 MiB per trace in float32, before activations or temporary
copies (about 114 MiB in BF16).

### Training windows and feature lifetime

`--trajectory-window 4` supervises up to four consecutive decisions for states
with explicitly labeled observed tracks. The original sampled state is always
included. States without suitable track labels and matched-candidate states
retain their original single-state supervision. Extra decisions use only the
observed track prefix, the original seed, and separately constructed original-
fiber targets; target offsets/departure labels never enter model inputs. Every
extra state's crop/target footprint is checked against the holdout before I/O.

All crops are sampled once into the original batch. Window images are views
of those buffers and share its photometric augmentation. The first decision
encodes the seed and earlier history; subsequent decisions encode only their
new crop and carry differentiable memory forward. Activation checkpointing
can recompute encodings during backward. Losses average over decisions within
each window, then over originally sampled states, so longer tracked windows do
not increase their source's loss weight or dilute matched-candidate examples.
Logs include `supervised_decisions`; detailed decision diagnostic groups count
the actual decisions evaluated.

**No learned encodings survive a training optimizer update.** Every window
starts with empty carried memory. All of its encodings use the current weights;
one backward pass accumulates its losses, and memory is discarded before the
optimizer steps. The next update re-encodes both seed and history. Burn-in and
stratified encoder-gradient sampling still compute fresh forward features;
`no_grad` is not a feature cache. Checkpoints contain weights/optimizer/RNG,
not observation memories. Replay stores observed geometry and labels, not old
encoder activations. Inference carries features within a trace while its loaded
model weights stay fixed; a new trace/checkpoint starts new memory.

### Running and validation

To launch a **new** experiment after reviewing the change and completing a
production-device preflight:

```bash
bash scripts/launch_spatial_memory.sh
```

Its default destination is `output/axial_spatial_memory_run1`; it does not resume
or alter the existing unified experiment. To initialize just the shared encoder
and compatible identity projection, append `--init-encoder output/OLD/last.pt`.
Only identity-independent unified/spatial-v2 encoders with matching geometry and
encoder dimensions are accepted. Memory, path decoder and optimizer start fresh.
`--resume` is for checkpoints of the new architecture, with matching options.

Production preflight (also exercises the window optimizer with zero LR):

```bash
PYTHONPATH=../../.. python -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight \
  --out output/spatial_memory_preflight --device cuda --forward --compile \
  --microbatch 2 --batches 2 --memory-version 4 --memory-slots 16 \
  --memory-steps 64 --memory-grad-steps 32 --memory-encoder-grad-steps 4 \
  --spatial-recent 2 --spatial-archive 8 --spatial-retrieve 2 \
  --trajectory-window 4 --memory-switch-probability .15 --activation-checkpointing
```

Tests cover point descriptors and crop-edge support, remote spatial evidence, immutable seeds, archive
eviction, streaming/unroll parity, relative poses, candidate isolation,
padding/empty memories, gradients through retrieval, checkpoint loading,
window supervision/holdout/crop reuse, and re-encoding after parameter updates:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_spatial_memory.py \
  tests/test_unified.py tests/test_learned_memory.py tests/test_memory_sequences.py \
  -q -o cache_dir=/tmp/spatial-memory-pytest
```

This is an architectural implementation, not a measured accuracy or speed
improvement. Production CUDA memory, throughput, compilation and tracing quality
still require validation before a long run. The encoder/decoder compile in place;
archive routing and the CPU observation/window schedule remain eager.

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
Old observations form no-gradient burn-in. Within the gradient window, valid
history observations are divided chronologically into up to four balanced groups
per state, with one random encoding per group retaining encoder gradients.
Its gradient is multiplied by the group size to preserve the expected raw encoder
gradient; uneven groups receive their own weights. All writes within the window
remain differentiable, and the seed and current crop always retain gradients.
`--memory-encoder-grad-steps` controls this budget (default 4; 0 keeps all).
Sampling uses the checkpointed Torch RNG. Encodings with gradients use activation
recomputation. Historical images stay on the CPU until each observation is
encoded. The current crop is read/encoded only once, and full seed/history crop
footprints are checked against the holdout before I/O. Replay uses saved observed
frames; history reconstruction is a fallback for older caches. Inference carries
only the bounded state and encodes one new full crop per decision after startup.

The launcher uses effective batch size 8, microbatches of 2, 64 historical
observations, 32 gradient-bearing history writes and four sampled history encoder
gradients per state. It uses 12 loader workers
and activation checkpointing. The encoder and shared decoder are compiled;
the CPU-streamed sequence scheduling stays eager. In-place module compilation
preserves the same checkpoint parameter names and EMA updates.
Historical encodings have one whole-observation checkpoint boundary, avoiding
nested axial-block recomputation. Auxiliary queries run only for labeled writes
during training, never for burn-in or inference. Their scheduling mask does not
enter the writer or the path inputs. Sequence validity is transferred to the CPU
once for scheduling, avoiding a GPU synchronization at each historical write.
Full-crop replay is substantially more expensive than the legacy patch unroll.
Legacy checkpoint architectures remain loadable; they cannot initialize this
architecture. Use `--memory-version 2` explicitly for legacy memory training.

The writer materializes initial state tensors with recurrent-state layouts and
packs sequences in time-major order once before the write loop. Burn-in,
differentiable writes and writes with auxiliary probes use separate compiled entry
points, with scheduling outside the graphs. Non-probe graphs can discard unused
probe features. This avoids exhausting the compilation cache on initial/recurrent
strides, seed availability, gradient transitions and probe flags. Parameter names,
checkpoint format and the seed's gradient path are unchanged.

Training and preflight also pass CPU candidate masks as scheduling metadata.
A candidate curve is skipped only when every point in that candidate's entire
microbatch is unlabeled; the result retains its usual shape with zero placeholders
for skipped curves. Any labeled point retains the complete evaluation, including
points outside the loss commit window used by diagnostics. Masks do not enter the
evaluator, memory writes or primary predictions. Calls without a scheduling mask
(including inference) continue to evaluate all supplied candidates.

The component benchmark and before-change source snapshots are under
`output/unified_speed_validation/`. Reproduce from this directory with:

```bash
PYTHONPATH=../../.. TORCHINDUCTOR_COMPILE_THREADS=2 ../../../../.venv/bin/python \
  output/unified_speed_validation/benchmark.py
```

It measures compiled BF16 writer forward/backward across seed/burn-in variants,
and repeated production-size path evaluation with two wholly masked candidates.
It excludes the crop encoder, data loading, optimizer and EMA. Tests cover
unchanged supervised losses, parameter gradients, candidate metrics, checkpoint
names and stable compilation after warming all writer variants. Existing running
Python processes pick up these changes only after restart; checkpoints remain
compatible.

RTX 5090 component results with training stopped (PyTorch 2.12.1, BF16,
microbatch 2; mean / p50 / p95 milliseconds):

| Workload | Before | After |
| --- | --- | --- |
| 65-write recurrence, equal mix of four seed/burn cases, 48 measurements | 77.63 / 81.61 / 85.95 | 73.04 / 74.02 / 83.78 |
| Six evaluator calls versus four, two candidates wholly unlabeled, 12 measurements | 49.91 / 50.05 / 51.45 | 32.16 / 32.14 / 33.85 |

The writer benefit depends on the case: seedless burn-in fell from 83.16 to
63.28 ms, while already-cached seeded burn-in changed from 62.22 to 64.26 ms.
The original writer hit its recompilation limit during warmup; the revised
entry points completed all variants with full-graph compilation. CPU tests retain
the mathematical loss/gradient behavior. CUDA BF16 before/after gradients differ
by 1.19–1.45% in aggregate relative L2 in this synthetic recurrence, comparable
to the existing eager/compiled rounding differences; all tested gradients were
finite. These component savings are not whole-training speedup claims.

A complete optimizer-update comparison is recorded in
`output/unified_speed_validation/update_results.json`. It uses the cached real CT
batch `output/unified_validation/batch.pt` (33 valid observations per state, both
seeds present, no labeled candidate curves), repeated over four microbatches of
two for effective batch eight. After three warmup updates, eight measured updates
include forward, losses, backward, clipping, AdamW and EMA, but exclude loading
and diagnostics. Mean / p50 / p95 update time was **4.902 / 4.896 / 4.958 s before**
and **4.815 / 4.805 / 4.884 s after**, a 1.8% mean time reduction. This batch does
not exercise the seedless writer fallback; full-crop encoding still dominates.
Reproduce with:

```bash
PYTHONPATH=../../.. TORCHINDUCTOR_COMPILE_THREADS=2 ../../../../.venv/bin/python \
  output/unified_speed_validation/benchmark_update.py
```

Validation before stratified encoder sampling, with all history encoder gradients
in the window enabled, on the RTX 5090 (PyTorch 2.12.1, BF16 autocast, full 120×101×101
crops, microbatch 2, 64 historical writes, 32 gradient-bearing history writes):

| Execution | Forward/backward + AdamW | Peak allocated VRAM |
| --- | ---: | ---: |
| Eager after redundant-work removal, one update | 9.13 s | 12.75 GiB |
| Compiled first update, including compilation | 52.29 s | 8.73 GiB |
| Compiled next two updates, mean / median | 4.60 s | 8.76 GiB |

The two warmed compiled updates took 4.595 and 4.604 seconds. These capacity
checks repeat a real crop across a fully populated sequence and exercise all
losses; they measure computation rather than tracing quality or loader time.
Initial eager/compiled loss was 3.495929/3.493718 (compilation changes rounding
under the same BF16 autocast policy). All gradients were finite and production
checkpoint save/load preserved parameter values exactly. Real-data preflights,
CPU recurrence/gradient tests, and raw profiler summaries are also available;
run artifacts are under `output/unified_validation/`.

A compiled microbatch of 4 took 9.035/9.098 seconds and used 17.29 GiB allocated,
19.76 GiB reserved, and about 20.4 GiB total process VRAM. The launcher therefore
uses microbatch 2 with four-step gradient accumulation to respect the 20 GiB
process budget; microbatch 4 offered little throughput improvement per sample.

For a reproducible real-data check with the same compiled modules:

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=../../.. ../../../../.venv/bin/python \
  -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight \
  --out output/unified_preflight --device cuda --forward --compile \
  --memory-version 3 --memory-slots 16 --memory-steps 64 --memory-grad-steps 32 \
  --memory-encoder-grad-steps 4 \
  --activation-checkpointing --microbatch 2 --batches 2
```

The remaining architecture description documents the crop-only and legacy
patch-memory models.

This model follows a marked fiber through a single heading-aligned CT crop. The
default crop-only architecture is `axial_fiber_v3`. Set `--memory-slots 16 --memory-version 2` to use
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
