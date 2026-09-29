# Main-encoder feature memory (v4)

`--memory-version 4 --no-correction` selects the new v4 architecture. It replaces
the earlier, untrained v4 patch-memory design; it does not change v2/v3 models.

## Prediction and retained evidence

A single transformer decoder jointly predicts continuous lateral coordinates at
16 fixed forward distances. Unmasked self-attention lets every future point
influence every other point. There is no route lattice or dynamic-programming
search. By default there is no refinement head or second decoder pass.

Each current main crop is encoded once. The encoder's contextual spatial features
are pooled to a `2 x 4 x 4` grid, projected to 32 memory tokens, and used by the
existing admission-probe/gated-slot update. This grid covers the full main crop;
it is not another raw-image crop or another image encoder. It is a compressed
representation, not a lossless copy of the encoder feature map.

Every decoder layer can read:

- The current image and visible history.
- Persistent learned identity slots (16 in the launcher).
- An immutable seed representation (32 tokens).
- A bounded cache of spatial grids from observed crops (64 crops by default,
  including the current crop).

Cached tokens retain world positions and crop orientations. Reads express their
positions, orientations and ages relative to the current crop. Tokens are not
masked merely because their locations have left the crop. The oldest explicit
cache entry is evicted when the cache fills; the seed remains and learned slots
can retain older information. Probes and retained state remain FP32 while the
main encoder/decoder use the trainer's existing BF16 autocast on CUDA.

The cache contains image evidence actually observed at each decision, never
annotation coordinates or uncommitted predictions. At inference, a normal trace
initializes its seed from its first main crop. A cold start at a remote supplied
seed encodes that seed crop once, then the current crop. A cold start does not
reconstruct all historical main crops; its cache fills as tracing proceeds.

## Optional recurrent refinement

`--recurrent-refinement-steps 1` adds one pass through the same four-layer
trajectory decoder. The initial curve samples the existing 3x3x3 fine-feature
stencil and deep features; a learned fusion maps that evidence, support and
current coordinates into the retained trajectory tokens. A learned stage
embedding distinguishes each refinement pass. Every pass reads the same current
image/history, identity slots, seed and cached crop tokens, with full point
self-attention. The encoder and memory writer still run once per observed crop.

Cross-attention keys/values are projected once per decoder layer per decision
and shared across passes, with gradients attached. FP32 feature-map conversions
are also shared by query sampling, refinement, confidence and identity queries.
Neither cache survives the model forward or crosses optimizer updates.

The stage embedding and displacement readout start at zero. Initial coordinates
therefore remain unchanged by the refinement update; confidence is not guaranteed
unchanged because it uses the final decoder tokens. Updates have lateral norm at
most `--recurrent-refinement-limit` (default one trace voxel), preserve forward
planes, and enforce the existing crop and first-point recovery bounds. All
coordinate updates remain differentiable. Geometry uses 75% final-curve loss and
25% mean earlier-curve loss with the existing censoring/departure masks. The
confidence head detaches the final curve for its direct evidence sampling and
labels, as before; the updated decoder tokens remain differentiable, including
their dependence on earlier proposals through refinement.

Old checkpoints default to zero passes and load strictly. To continue an existing
run with refinement, use `regression.upgrade_refinement`, supplying a resumable
checkpoint and an audit of live process arguments against the latest
`resume_configuration` log event. It creates a separate run directory and launch
script, preserves existing model/EMA tensors, maps AdamW moments by parameter
name, and retains RNG, completed step, schedule, sample count, monitor fixture and
published replay. New parameters start without optimizer moments. This is an
architecture upgrade, not an exact continuation: worker streams restart, and the
new loss/decoder path changes future updates. It never stops or launches a trainer.

```bash
PYTHONPATH=/home/sean/Documents/villa4/vesuvius/src \
/home/sean/Documents/villa4/vesuvius/.venv/bin/python -m \
  vesuvius.neural_tracing.fiber_follow.regression.upgrade_refinement \
  --checkpoint output/OLD_RUN/last.pt \
  --runtime-audit output/recurrent_refinement_validation/live_config_audit.json \
  --name NEW_RUN --steps 1 --limit 1
bash output/NEW_RUN/launch.sh
```

The audit JSON contains `runtime_options`, `model_cfg` and
`cli_event_differences` (which must be empty), plus process identity/provenance.
The migration checks it against the checkpoint and latest resume log. The
generated `upgrade.json` records all option changes and the source SHA256.

## Streamed training

V4 uses consecutive observed states instead of rebuilding small historical
patches separately for independently sampled decisions. Saved rollout paths,
drifted synthetic histories, wrong continuations and matched identity examples
supply the observations. Annotation geometry is used for targets only.

Each sampled stream visits its actual seed, up to the newest `memory_steps`
historical observations, and the sampled endpoint. Generated original-fiber
histories retain their intended-original supervision; explicit tracks retain
their recorded membership. Legacy replay history without membership labels is
context only. Inserted seeds require generator certification for supervision.
Historical headings use an incoming chord or the supplied seed direction; an
unoriented oldest observation is omitted. Unknown bridge labels are masked, and confirmed
departures still teach rejection without trajectory geometry loss. Crop and
label holdout checks cover the whole stream before image reads. Matched endpoint
crops retain identical local inputs and their different earlier identity evidence.

`--feature-sequence-length 2` keeps gradients through two consecutive decisions.
Their losses are accumulated before backward. The resulting bounded state is
detached and carried across subsequent chunks and optimizer updates. Detaching
preserves memory values; it only ends gradient propagation into older graphs.
Features carried across optimizer updates were computed with slightly older
weights. They are not a permanent dataset-wide feature cache.

`--microbatch` counts main crops across traces and time. With microbatch 4 and
sequence length 2, each full chunk has two traces and two decisions. Matched
sampling requires an even number of concurrent traces. The trainer accumulates
chunks until it reaches the effective `--batch` crop budget; a partial ending
chunk can make this exceed the requested budget by less than one microbatch.
Losses are normalized by the actual number of states, and throughput logs use
actual counts. These correctness fixes use sampling revision 3 and retain equal
per-crop weights, optimizer updates, and the existing schedule. Stream ids include worker and group identity; ended streams are
removed, and worker interleaving does not mix their memory.

There is no independently reconstructed `trajectory_sequence` auxiliary pass.
The old `--trajectory-sequence-weight` must be zero. `memory_patch_size` and
`memory_grad_steps` remain legacy configuration fields for checkpoint tools;
v4 uses `feature_memory_grid` and `feature_sequence_length` instead.

Sampling fractions apply to stream endpoints. Since streams have different
lengths, the resulting per-state source mix can differ; logs report the actual
mix. More supervised crop decisions are not necessarily more independent
examples. The shorter gradient horizon and different sampling distribution need
held-out tracing validation, especially for long-lived fiber identity.

## Launch and initialization

```bash
bash scripts/launch_trajectory_memory.sh --direction-inputs --batch 8 --microbatch 4 --workers 8
```

This launches a new background run. The launcher does not overwrite existing run
names. The main encoder, single continuous decoder, confidence and identity heads
can initialize from a v2/v3 checkpoint:

```bash
RUN_NAME=axial_feature_memory_v4_run1 bash scripts/launch_trajectory_memory.sh \
  --init-tracer output/axial_spatial_memory_v3_run2/ckpt_010000.pt \
  --direction-inputs --batch 8 --microbatch 4 --workers 8
```

Migration retains compatible encoder, decoder, confidence and identity weights
and initializes the new feature memory. When importing v3, the continuous
coordinate output layer also starts fresh: v3 never trained it with its lattice
objective. Importing v2 retains its trained continuous coordinate layer.
This starts a new optimizer and schedule, not a behavior-preserving resume.

V4 checkpoints identify `feature_memory_revision: 1`. Checkpoints of the previous
patch-memory v4 are rejected explicitly. V2/v3 checkpoint loading remains
supported. Model, EMA, optimizer and RNG resume normally within the new v4.
As with the existing iterable loader, resuming starts new data streams: ephemeral
worker cursors and carried training states are not restored. Every new stream
starts with an explicit reset and rebuilds memory from its seed; this is not
bit-exact continuation of the pre-interruption sample sequence.

Training-batch diagnostics receive the actual carried input state rather than
silently plotting a cold prediction. Those state features were produced by the
training model; independent monitor tracing still uses the EMA model throughout.

Scheduled resumable checkpoints are saved before collection and diagnostics.
Recovery evaluation handles the nested remote-seed input recursively, converting
floating tensors to FP32 while preserving boolean masks.

The first `axial_feature_memory_v4_run1` stopped at step 1000 before these fixes.
Its surviving `dagger/source_001000.pt` contains model and EMA weights but no
optimizer/RNG state, so it cannot provide an exact training resume. To initialize
a new run from its trained v4 EMA weights (including memory and coordinates):

```bash
RUN_NAME=axial_feature_memory_v4_run2 bash scripts/launch_trajectory_memory.sh \
  --init-tracer output/axial_feature_memory_v4_run1/dagger/source_001000.pt \
  --direction-inputs --batch 8 --microbatch 4 --workers 8
```

This restarts the optimizer and schedule. Recovery validation after the fix
completed all 32 original monitor states on CUDA, with 162 model forwards and
52 nested remote-seed inputs. The report is
`output/feature_memory_v4_validation/recovery_step_001000_fixed.json`.

## Validation

Use the existing environment; no dependencies were installed:

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/sean/Documents/villa4/vesuvius/src:/home/sean/.cache/uv/archive-v0/c6JuZFIQxH2QbGds/lib/python3.14/site-packages \
/home/sean/Documents/villa4/vesuvius/.venv/bin/python -m pytest \
  tests/test_trajectory_memory.py tests/test_spatial_memory_v3.py \
  tests/test_learned_memory.py tests/test_memory_sequences.py \
  tests/test_identity_decisions.py tests/test_regression.py \
  -q -o cache_dir=/tmp/fiber-feature-pytest
```

The CPU suite passed 111 tests, skipped six CUDA tests and deselected one test.
It covers gradients into earlier main-crop features, bidirectional point-query
influence, remote-memory sensitivity, seed immutability, bounded cache eviction,
stream ownership and detachment across optimizer updates, cold/warm tracing,
matched endpoints, checkpoint migration/resume and graph capture. GPU numerical
and production-throughput validation are recorded separately below.

The same suite with CUDA enabled passed 116 tests initially; the new compiled
gradient test needed `.item()` in its assertion to compare a CUDA scalar with
pytest. After that test-only fix, its rerun passed (117 passing tests combined,
one deselected). Run the command above without `CUDA_VISIBLE_DEVICES=''` and with
`TORCHINDUCTOR_COMPILE_THREADS=4` for GPU validation. The two-decision BF16 compiled
test checks eager/compiled loss agreement, gradient cosine similarity above .99
and gradient norms within 5% for the encoder, decoder and memory writer.

A real-data preflight also passed for 16 eight-channel crop decisions, checking
that streamed inputs contain no separately reconstructed historical image crops.
Its report is `output/feature_memory_v4_validation/preflight/report.json`.

## Production training cost versus v3

Measured on the RTX 5090 with PyTorch 2.12.1+cu130, using the stopped
`axial_spatial_memory_v3_run2` recipe and checkpoint 10000. Both bounded runs used
the real s1 CT, presence/direction inputs (eight channels), 120x101x101 crops,
fixed recovery and saved on-policy banks, batch 8, microbatch 4, eight loader
workers, compilation and BF16 with FP32 memory. V3 retained its refinement and
earlier-decision auxiliary loss; v4 used two-decision feature-memory chunks.

Each run completed 80 optimizer updates; the first 30 (including compilation)
were excluded. Timings include loader waits, forward/backward, losses, gradient
clipping, AdamW and EMA. Metrics were computed every update for both models.
Background collection, periodic diagnostic tracing and checkpoint I/O are
excluded. These are steady training costs, not total production wall time.

| Measurement | V3 | New v4 |
| --- | ---: | ---: |
| Measured updates | 50 | 50 |
| Primary crop decisions | 400 | 436 |
| Total supervised crops, including auxiliary | 520 | 436 |
| Mean ms per supervised crop | 56.4 | 58.7 |
| Mean ms per primary decision, including auxiliary work | 73.3 | 58.7 |
| p50 / p95 ms per primary decision | 71.8 / 90.0 | 57.3 / 64.0 |
| Peak allocated GPU memory, GiB | 16.68 | 12.17 |

V4 costs about 4% more per supervised crop and uses about 27% less peak allocated
memory in this measurement. The larger improvement per primary decision reflects
v3's additional supervised crops; it should not be described as a 20% speedup per
supervised crop. V4 can slightly exceed batch 8 when finishing a variable-length
chunk, so the table divides by actual crop counts. Streams change the observed
source mixture and correlation between examples; this benchmark does not measure
convergence or held-out tracing quality.

The v4 profiler's largest self-CUDA categories were convolution backward (24%),
attention backward (10%), convolution forward (10%), copies (7%) and attention
forward (6%). Profiling was performed on an excluded warmup update.

Local reproducibility artifacts are in `output/feature_memory_v4_validation/`:
`production_v{3,4}_results.json` contains every update and summary;
`production_v{3,4}_argv.json` contains the complete trainer arguments;
`production_v4_profile.txt` and `.json` contain the profile. The bounded harness
loads the original run recipe and uses separate output directories:

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 \
PYTHONPATH=/home/sean/Documents/villa4/vesuvius/src \
/home/sean/Documents/villa4/vesuvius/.venv/bin/python \
  output/feature_memory_v4_validation/real_training_benchmark.py \
  --version 3 --steps 80 --warmup 30 --tag production
```

Use `--version 4 --profile` for v4 and a fresh `--tag` when repeating either run.
The original v3 trainer was stopped; its latest durable checkpoint is 10000.
No new long-running training job was launched.

## Recurrent refinement validation (2026-09-29)

The refinement upgrade of `axial_feature_memory_v4_run2/ckpt_008000.pt` preserves
all existing model/EMA tensors, 282 populated optimizer states and 69,418 observed
samples. Runtime settings were checked against the live process and its latest
resume event, including feature sampling revision 3; the original launch JSON
was not treated as the current configuration. The new run is
`axial_feature_memory_v4_refined_run1`, with one refinement pass, a one-voxel
update bound, and unchanged batch 8 / microbatch 4 / two-decision chunks.
`upgrade.json` and `migration_validation.json` in that directory record lineage
and the configuration/state checks. The original run stopped after saving step
8,000, and `launch.sh` resumes the upgraded checkpoint with the existing schedule.

The CPU regression suite passed 94 tests (four CUDA skips, one deselected); the
subsequently added directory-migration integration test also passed. Seven GPU
tests passed, including compiled/eager BF16 two-decision gradient comparison with
candidate scoring. Tests cover cached/recomputed attention gradients, neutral
initial displacement, one observation write, bounds, departure/censoring masks,
earlier-observation gradients, optimizer/EMA/RNG migration and full-graph capture.

Measured on the RTX 5090 / PyTorch 2.12.1+cu130, with fixed synthetic eight-channel
120x101x101 crops and candidate curves, production model dimensions, batch 8,
microbatch 4, compiled BF16 training and FP32 memory. Five warmup updates were
excluded, followed by 20 measured updates per variant (160 crops). Includes
forward/backward, AdamW, clipping and EMA; excludes image I/O, real sampling,
metrics, collectors and diagnostics. These results measure cost, not tracing
accuracy or full production wall time.

| Configuration | Mean ms/crop | p50 / p95 ms/crop | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Original v4 | 46.98 | 46.38 / 48.72 | 11.47 |
| One shared-decoder refinement pass | 50.89 | 50.78 / 52.59 | 11.79 |

This workload adds 8.3% training time and 0.33 GiB peak allocation. Shared
projections and FP32 maps reduce the overhead relative to recomputing each pass.
Raw samples and the harness are in `output/recurrent_refinement_validation/`.
Run this command with `--passes 0` for the baseline and `--passes 1` for refinement,
using a distinct `--out` path for each:

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
PYTHONPATH=/home/sean/Documents/villa4/vesuvius/src \
/home/sean/Documents/villa4/vesuvius/.venv/bin/python -u \
  output/recurrent_refinement_validation/benchmark.py \
  --checkpoint output/axial_feature_memory_v4_run2/ckpt_008000.pt \
  --passes 1 --out output/recurrent_refinement_validation/refined.json
```

## Feature-memory revision 2 experiment

Revision 1 remains the default and loads old checkpoints unchanged. Revision 2
adds 16 fine-detail tokens to the existing 32 pooled tokens. Detail locations
are the current head and evenly spaced visible observed history; absent history
is padded and masked. Each detail token projects a 3x3 fine-feature neighborhood
and a sampled deep feature. Positions, orientation, age and per-token validity
are retained. No extra image encoder or volume crop is used.

The writer retrieves from the incoming spatial cache, seed and slots before
judging admission. Cache entries retain the predicted admission probability as
metadata; low-confidence observations are still stored. A gated projection of
the retrieved coarse tokens is interpolated onto the deep encoder lattice for
decoding. This bounds the query cost to the 48 observation tokens instead of
querying history separately from all 39,015 image tokens. The stored image
features are extracted before this conditioning. Candidate confidence has its
own spatial queries into persistent memory; candidates no longer depend only on
the proposed trajectory's shared decoder tokens for historical comparison.

Ordinary two-decision gradient chunks are retained. At each stream's end, an
additional replay loss (weight 0.5 per endpoint, normalized by the update's
ordinary crop count) replays its chronological writes at current weights.
One historical encoder is selected uniformly within each available age band:
1–4, 5–16, and 17–oldest. These crops and the endpoint are re-encoded, with
activation checkpointing for the historical encoders. Other observations use
detached features computed earlier by the training model. This is explicitly a
selective, stale-feature approximation, not full-history or unbiased BPTT.
All intervening writer transitions remain differentiable, including after the
selected observation is evicted from the explicit cache. Replay endpoints are
processed individually after ordinary chunk backward, limiting peak VRAM.

Revision-2 streams allow 128 historical observations plus seed and endpoint.
Memory-switch generators request a correspondingly longer original prefix.
Existing replay histories may be shorter; no missing history is fabricated.
Only selected raw crops are retained on the CPU, and other observations retain
compact detached tensors. The replay archives and carried state are ephemeral:
resuming starts fresh streams, as in revision 1.

A stopped resumable revision-1 checkpoint can be migrated into a separate run:

```bash
python -m vesuvius.neural_tracing.fiber_follow.regression.upgrade_feature_memory \
  --checkpoint output/OLD_RUN/ckpt_010000.pt --out output/NEW_RUN
```

Existing model/EMA tensors, AdamW moments by parameter name, RNG and learning-rate
schedule are retained; new parameters have fresh moments. The immutable monitor
fixture and published replay index are copied. This changes behavior and the
training objective; it is not a behavior-preserving continuation. The generated
`resume_argv.json` contains the complete trainer arguments. Migration never
launches training.

Controlled full-size GPU measurements (BF16 encoder/decoder, FP32 memory,
effective batch 8, microbatch 4, two concurrent traces) use:

```bash
python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_feature_revision \
  --checkpoint output/RUN/ckpt_010000.pt --out output/ordinary.json
python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_feature_revision \
  --checkpoint output/NEW_RUN/ckpt_010000.pt --replay-length 128 --out output/replay128.json
```

The latter measures one additional endpoint replay *every update*. Report it
separately from ordinary updates; actual average cost depends on stream lengths
and endpoint frequency. Both include clipping, AdamW and EMA, and exclude data
loading. The fixed synthetic fixture includes candidate confidence but excludes
contrastive identity loss. Warm-up/compilation is excluded from measured steps.

For a bounded real-data fork, with per-step allocated/reserved VRAM and replay
counts, use `regression.benchmark_feature_training --checkpoint ... --out ...
--updates 80 --workers 2`. It preserves the source schedule, stops without
crossing a checkpoint/evaluation/collector boundary, and saves the final tested
weights/optimizer in the fresh output directory. Its sampling can differ across
revisions because revision 2 supports longer streams; it is an integration and
operational-cost check, not a matched-example accuracy experiment.
