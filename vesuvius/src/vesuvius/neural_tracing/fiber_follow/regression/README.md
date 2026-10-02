# Live historical slab regression

The optional v17 `--path-geometry-tokens` (requires v16 path tokens) adds 29 memory
tokens shared by the path decoder and survival scorer. They sample the committed
observed polyline 1–512 arclength voxels behind the head, plus its first point,
in the current crop frame without crop masking: local references are crop
supported, so only ~24 of the 128 local-history points otherwise reach either
head. Each token embeds position (multi-scale Fourier, 2–1024 voxels), unit
tangent, arclength behind the head and the first-point role. The final layer
starts at zero; masked tokens leave v16 outputs unchanged. No annotation geometry
enters them.

To fork a path-token run with these tokens, a fresh AdamW (no moments, warmup
restarting at the fork step), new source weights and a new rest-gradient clip:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.geometry_resume \
  --checkpoint output/SOURCE/ckpt_022000.pt --name NEW_RUN \
  --dataset-config configs/mixed_ct_datasets_paris50.json --rest-grad-clip 60
```

The dataset config may differ only in source weights. `configs/mixed_ct_datasets_paris50.json`
samples Paris 4 at 50% and each AFV source at 25%. As with `memory_resume`,
`migration.json` records the exact trainer command, and this command does not
stop or launch processes.

The optional v16 memory continuation adds three explicit observed-path tokens per
historical slab, retaining all 578 spatial tokens. Each token samples the encoded
CT/path features at the committed path and four lateral neighbors one tracing
voxel away. Position, tangent, age, seed role and slab pose accompany the features.
Samples are at observation arclength offsets -1, 0 and +1; samples outside the
observed prefix or physical slab are masked. No annotation geometry enters them.

`--pair-rank-weight .25` adds a paired-history logistic ranking loss to the
existing geometry and survival objectives. Only matching explicit pair IDs,
identical candidate geometry, and known opposite prefix labels contribute.
It averages eligible candidate/prefix comparisons per decision; logged comparison
counts count each pair once. Source sampling weights are unchanged. Ranking
accuracy and mean logit margin are logged separately from survival calibration.

`--live-continuation-steps 12 32 --live-continuation-stratified` cycles chain-limit
draws across 12–18, 19–25 and 26–32 decisions, uniformly within each band. A shared
counter per dataset balances draws across workers. Limits stay fixed for a chain;
stops, departures, annotation boundaries and stale feedback still end it early.
Logs include started-chain limit counts, reached-depth counts and mean travel.
This balances requested limits, not surviving depths. Worker timing still affects
the live sample stream; counters/queues restart on resume.

To fork a stopped training run using its actual checkpoint configuration:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.memory_resume \
  --checkpoint output/SOURCE/ckpt_010000.pt --name NEW_RUN \
  --pair-rank-weight .25 --live-steps 12 32 --zscore
```

The command prepares an independent run and records the exact trainer command
and intentional changes in `migration.json`. It preserves every existing model
and EMA tensor, remaps Adam moments by parameter name, and preserves the update
count, RNG and LR schedule. New path-projection parameters use seed 20261001;
their final layer starts at zero and their Adam state starts fresh. Extra tokens
change attention normalization slightly even before their new projection learns.
Primary and dataset-specific replay indexes reference existing geometry caches;
new rollouts are written inside the new run.

`--zscore` records `crop_zscore_v1` preprocessing in the checkpoint: each current
and historical CT crop uses `(image - mean) / max(population_std, 1e-6)` over all
sampled voxels, before augmentation. There is no foreground threshold, clipping,
or background sentinel. Constant crops become zero. Augmentation keeps its saved
magnitudes but applies to the entire image without masking/clipping. Inference
and collection use the checkpoint's normalization policy. The frozen recovery
fixture retains every geometry array; only its normalization provenance and hash
change. Existing v14/v15 checkpoints retain their original behavior.

The direct follower predicts continuous paths and causal survival confidence from
one current crop and up to eight live historical slabs. The main encoder, local
128-point history, 16-point forecast and adaptive refinement policy are unchanged.
See [architecture and supervision](TRAJECTORY_MEMORY.md).

## Training and tracing

Use the existing project environment from `fiber_follow`:

```bash
bash scripts/launch_patch4_memory.sh
# Or the convolutional main encoder:
bash scripts/launch_memory.sh
```

Set `RUN_NAME`, `BATCH_SIZE`, `GRAD_STEPS`, `WORKERS`, and `BANK_PATH` as needed.
The patch launcher defaults to batch 16 and one gradient accumulation step, eight
loader workers, and a fresh `axial_patch4_overlap_tokens_slabs_v11_run1` destination.
The convolutional launcher defaults to `axial_survival_slabs_v10_run1`.
Existing output directories are never overwritten.

`bash scripts/train_mixed_ct_stem_fresh.sh` starts a fresh mixed-CT model with
two axial encoder blocks at width 256 and FFN 256, six path decoder layers at
width 256 and FFN 2048, and four survival scorer layers at width 256 and FFN 1024.
The encoder and heads share width 256; the stem remains 32/64/128 and the history
encoder retains width-256 output.
The launcher sets `--axial-layers 2 --encoder-ffn 256
--hidden 256 --decoder-layers 6 --decoder-ffn 2048 --scorer-layers 4` and uses
batch 4 with three gradient accumulation steps. These dimensions are saved in
the model configuration. Its default run name is `mixed_ct_afv_stem32_fresh_run3`;
set `STEM_RUN_NAME` to select another new run.

When resuming, `--lr` may change the base learning rate while retaining AdamW
moments, EMA, and the existing warmup/cosine schedule position. Omit
`--reset-optimizer` to preserve that state; the active LR includes cosine decay.

Training reserves `--clean-fraction` of examples for GT (default .8). With
`--gt-perturb-probability 0` (the default), these samples use the annotation
position and tangent and all available history: no offsets, heading noise,
drift, wobble, jitter, history dropout or artificial shortening. Local history
retains the usual point limit; older history remains available to the slabs.
Noise, blur, contrast and brightness augmentation still apply to every source.

The mixed stem launcher uses 70% GT, 10% identity pairs, 10% wrong turns
and 10% model replay. Its options are:

```text
--clean-fraction .7 --gt-perturb-probability .25
--gt-perturb-max-offset .5 --gt-perturb-max-angle-deg 2
--decision-fraction .2 --fresh-fraction .75 --memory-switch-probability .3333333333333333
--bank-following-probability 0 --replay-continuation-fraction .8 --prefer-real-wrong-turns
--prefer-replay-for-light-gt --batch 10 --grad-steps 1
```

The source weights normalize into the 30% non-GT budget, producing 10% each.
`--prefer-real-wrong-turns` fills wrong-turn slots from source-local replay of
confirmed natural switches first, with synthetic wrong turns as fallback.
Eligible traces must contain a recorded, nonexploratory switched state. Samples
balance available pre-switch and post-switch states, then fibers; exploratory
states and prefixes preceding only a forced switch are excluded. Original seed,
history and geometry are retained. Before-switch samples teach continuation;
after-switch samples teach stopping. Missing or rejected real proposals fall
back to synthetic proposals, then GT. The separate 10% replay budget is unchanged.
`real_wrong_turn_fraction` and `real_wrong_turn_pre_switch_fraction` report actual
shares of all training; `bank_wrong_continuation_fraction` includes both real and
synthetic wrong-turn slots for compatibility with existing logs.

With `--prefer-replay-for-light-gt`, light-GT slots first try source-local correct
continuation replay, retaining its actual seed and history without perturbation.
Unavailable or rejected replay falls back to light GT. Thus the nominal 17.5%
light-GT allocation becomes additional correct replay when available: 52.5%
untouched GT, 25.5% correct replay, 2% failure replay, 10% identity and 10% wrong
turns. `light_gt_replay_fraction` reports the replaced share separately.

With `--live-continuation --live-continuation-steps 4 8`, correct-continuation
slots (including light-GT replacement slots) instead use recent training
predictions. Clean training examples seed source-local chains. The selected
prediction is detached, confidence/recovery gated, committed using the same
geometry and history resampling as inference, and relabeled against its original
fiber. Workers resolve the next CT frame with inference sign continuity and
rebuild all current/historical observations. No future vertices enter the input.
Confirmed departures/switches remain supervised failure examples and end their
chain; annotation boundaries and held-out footprints censor it. Stops do not
force progress. Chains reset after 4–8 live decisions; empty queues fall back to
clean GT and are counted explicitly. Failure and wrong-turn replay remain active.

Feedback uses bounded per-source queues and prioritizes existing chains over new
seeds. Workers reject predictions older than 64 optimizer updates when taking
feedback; ordinary loader prefetch can add a few more updates before training.
Live positions resolve after geometry lookahead, so future plans cannot reserve
old feedback. There is no extra model forward or gradient through prior steps.
The policy uses the current training weights and augmented training observation;
it is not an EMA rollout on unaugmented CT. Queue availability/order depends on
worker timing. A resume starts empty queues and the usual restarted loader RNG
streams while preserving optimizer, EMA and schedule state.

Logs distinguish `live_correct_continuation_fraction`, `live_failure_fraction`,
`live_fallback_fraction`, chain depth and policy age from saved replay. With the
mixed-run allocation above, 25.5% of samples request live continuation. Some of
those become failures or clean fallback; this is measured rather than assumed.

Validate real CT, source-local supervision and model feedback without GPU use:

```bash
../../../../.venv/bin/python scripts/validate_live_continuation.py \
  --checkpoint output/RUN/last.pt --out /tmp/live-continuation-validation.json
```

The mixed stem launcher uses batch 10, grad steps 1. Collection explicitly uses
batch 1 and at most one concurrent collector across sources. Inference checkpoint
storage stays on CPU; only the EMA inference model transfers to the requested
GPU. `scripts/stop.sh RUN_NAME` freezes the trainer and its descendants, includes
orphaned collectors identified by this run's checkpoint AND output paths, stops
the owned process trees, and verifies no matching collectors remain before a
restart proceeds. Other runs' collectors are excluded.

Without available replay, only one quarter of gt samples receive light perturbation: 52.5% of all examples
are untouched GT and 17.5% are lightly perturbed GT. Offset is uniform in a
lateral disk of radius .5 tracing voxels; heading tilt is bounded by 2 degrees.
The offset blends smoothly into only the last 32 voxels of history, preserving
the original seed and older observations, including on shorter startup prefixes.
Seed-only states remain untouched. There is no history dropout, shortening,
independent point jitter or wobble. Future labels still come from the GT fiber.

`--replay-continuation-fraction .8` reserves 80% of replay for correct committed
prefixes and 20% for recorded failures (8% and 2% of total training). It overrides
`--replay-failure-fraction`; without it, legacy replay sampling stays unchanged.
Continuation eligibility requires positive model travel, a recorded prefix, no
collector-confirmed departure, no exploration, and no hard/failure label
(including pre-failure windows). Correctness follows the collector's
original-fiber checks. Replay preserves the original seed, actual committed
prefix, pose and historical slabs without synthetic perturbation or replacement.
Supervision covers the forward segment beyond the current tip; future trace
vertices never enter the input. Missing correct replay falls back to GT, never
failed replay. Missing failures may use correct continuation. The optional
`--correct-replay-only` selects exclusively correct prefixes and cannot be
combined with `--replay-continuation-fraction`.

Allocation is stochastic in two-example units. Unavailable or rejected hard
examples fall back to GT, so realized GT share may exceed its target. Logs report
budgets in `identity_sampling.source_sampling`; `fresh_fraction` measures total
GT, while `gt_unperturbed_fraction`, `gt_perturbed_fraction` and
`replay_correct_continuation_fraction` report separate shares of all training
examples. These sampling settings can change on resume, including from older
checkpoints; workers require a restart. Optimizer state and the frozen recovery
fixture remain unchanged. Legacy heavy perturbation settings still apply only
to legacy samplers and recovery construction.

To prepare a separate continuation with more refinement stages, use a completed
training checkpoint as the source of all effective settings:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.refinement_resume \
  --checkpoint output/patch4_run3/ckpt_017000.pt \
  --name patch4_run3_refine3 --refinement-steps 3
```

This only prepares the new directory; `migration.json` records the exact trainer
command to launch. It preserves weights, EMA, RNG, update count, LR schedule and
existing AdamW moments. Added stage embeddings copy the last learned stage;
their moment rows start at zero, retaining the embedding tensor's Adam step.
The frozen recovery fixture is copied unchanged and published replay caches are
referenced from an independent replay index. The source run must be quiescent
while preparing the fork. Existing destinations, zero-stage source models, and
nonstandard optimizer layouts are rejected. As with ordinary resumes, loader
random streams restart. No source config defaults are used.

Fresh runs use the fine history encoder (`--history-encoder fine`), with architecture
identifiers `axial_fiber_slabs_v15`, `axial_patch4_overlap_fiber_slabs_v15`,
`axial_patch4_overlap_tokens_fiber_slabs_v15`, and
`axial_patch4_residual_stem_tokens_fiber_slabs_v15`. Existing v14 checkpoints
load and resume with their original history encoder; an omitted history setting
is inferred from the checkpoint. `--history-encoder legacy` also permits a fresh
v14 baseline. Changing history architecture during an ordinary resume is rejected:
the wider convolutions need new weights and optimizer state, so use a fresh run.
Patch4 uses a learned 6x6x6 convolution with stride 4 and padding 1. Adjacent
neighborhoods overlap by two voxels. The sampled crop is 120x104x104; patch4
dimensions must be multiples of four, with no extra image padding. The token
grid is 30x26x26 with centers at offset 1.5 input samples; dense reconstruction
uses 4x4x4 output cells. Crop sampling and physical token positions remain
centered laterally, including even crop widths. Convolution boundary padding
is still used. Replay v8 stores CT-normal frames and the trusted heading-history boundary. Recollect
older replay caches. Recurrent checkpoints and older replay
are rejected; there is no weight migration or memory compatibility interface.
The existing regression `train`, `collect`, and `infer` module entry points remain.

### Heading initialization and updates

Inference, rollout collection, and validation initialize every seed from raw CT
plus an H/V family. The unchanged seed locates a 65³ native CT cube. A structure
tensor (derivative sigma 1, integration sigma 4) estimates the sheet normal;
V projects world z into the sheet, and H follows its intersection with xy.
No seed snapping, presence PCA, direction zarr, or annotation tangent estimates
the seed axis. Unusable CT or ambiguous H/V orientation is reported; collection
skips those seeds. Model input channels remain controlled by the model config.

Supply `--family H` or `--family V` alongside inference seeds. One family applies
to all seeds; repeat it once per seed for mixed families. Both signs are traced:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.infer \
  --checkpoint output/run/last.pt --seed 18529.9,13044.9,51234.1 --family H --out output/traced
```

After confident commits, heading is a free-intercept linear fit to 13 equally
spaced samples spanning the last 12 trace-grid voxels of trusted committed path.
With less than 12 voxels, the current heading is retained. Failed decisions
retain the entire frame, including its rotation. Forced exploratory moves are excluded from subsequent
fits, including the connector to the first accepted recovery point; fitting
resumes after 12 voxels of accepted recovery geometry. Replay v8 preserves this
boundary when a decision is resumed. There is no alternate heading policy.

All image crops use the `ct_transverse_uv_v2` frame policy. With heading h
fixed, restrict the local CT structure tensor J to an orthonormal basis B of
h's perpendicular plane. The principal eigenvector of BᵀJB determines u;
v is h × u. Derivative sigma is 1 native voxel and integration sigma is 4,
using a 65³ native CT context. Independent frames choose a deterministic sign;
accepted trace steps preserve sign continuity against the transported previous
frame. Failed steps retain the full frame.

A reliable estimate requires transverse principal energy > 1e-12 (CT divided
by 255), at least 1e-4 of total tensor energy, and transverse eigenvalue gap
(max − min)/max >= 0.05. These numerical confidence gates do not certify
anatomical orientation. Weak evidence transports the previous observation's
frame. A first historical slab can borrow the current observation's frame;
without an established frame, a deterministic perpendicular basis supplies roll.
The heading is unchanged by every roll fallback. Initial H/V seed-heading
selection still requires an identifiable unrestricted CT sheet normal.

Fresh training perturbs annotation-based heading and position, then resolves
roll before sampling images. History, labels, and candidate coordinates rotate
together. Replay holds its recorded frame; historical slabs resolve local CT
frames in chronological order. Weak orientation does not discard training pairs.
Invalid geometry or CT context still rejects the whole batch within its
source, with a contextual error after 64 consecutive rejections. I/O errors
propagate. The log reports `ct_frame_rejected_batches` separately from roll
fallbacks.

Training logs current and historical frame counts, transported/deterministic
fallback counts, and mean transverse gap. JSON also records summed transverse
energy and gap. Batch tensors contain `ct_frame_source/energy/gap` and
`history_frame_source/energy/gap`; source codes are 0 = CT, 1 = transported,
2 = deterministic, and -1 = unmeasured or padded. Trace decision records include
`ct_frame_diagnostics`. Remote prefetch and holdout checks cover the tensor's
CT context and every possible crop roll until the final frame is resolved.

Real-CT failure examples and measurements are in
`output/ct_normal_failures_20261001/report/index.html`.
The production estimator passes all 46 saved contexts, including 14 failures;
measurements are in `output/ct_normal_failures_20261001/production_validation.json`.
This validates frame construction, not training or tracing accuracy. Frozen
recovery fixtures record the frame policy and must match the current policy.

Validation from `fiber_follow` (176 passed, 5 GPU tests skipped):

```bash
AGENTS_AGENT_MODE=1 MPLCONFIGDIR=/tmp/fiber-tests-mpl ../../../../.venv/bin/python -m pytest -q \
  -o cache_dir=/tmp/fiber-pytest-cache \
  tests/test_{heading,ct_frames,transverse_frames,trace_heading,history_slabs,ct_frame_retries,point_logging,regression,batch_diagnostic,feedback_refinement,decision_training,identity_decisions,observation_preparation,mixed_datasets,diagnostic_images}.py
```

### Rollout CPU work

`ModelTracer` runs the per-trace CPU work of each step in its thread pool: the
initial and per-step CT frames, the fine and history-slab crop sampling, each
item's slab-frame chain, and slab packing. Each row is computed exactly as in the
serial loop, so inputs are bit-identical (`tests/test_rollout_threading.py`). This
holds under `torch.inference_mode` as well. Without a pool (`pool=None`) the code
runs serially, as training loader workers do.

The gain needs several traces per step. On 24 held-out Paris 4 seeds × 600 voxels
(81k checkpoint, free RTX 5090), rollout throughput rose from 148 to 306 voxels/s
at batch 8, and from 154 to 412 at batch 24. Batch size did not matter before the
change because the step was CPU-bound. Traces were identical before and after.
Batch-1 callers (the in-trainer monitors and DAgger collection) see little change.
For held-out evaluation, use `--batch 24` or more. The 96-seed, 2000-voxel Paris 4
long-range eval then takes about 4 minutes in one process.

### Default CT normalization

Every CT input uses a per-volume background estimate and foreground-only robust
z-scores. Startup samples up to 128 available chunks (64 candidate 8x8x8 blocks
each), excluding almost-zero-filled blocks. The dominant block-median intensity
among the quietest quarter by local MAD estimates background; nearby quiet blocks
estimate its noise scale, with a minimum of two native uint8 levels. This is
unsupervised intensity calibration, not a labeled tissue segmentation. Local cache
coverage can bias the estimate; a new remote volume with an empty cache requires
initial chunk reads. The run logs each volume's estimated background and threshold.

`ct_normalization.json` in the output directory records source/level, array metadata,
sampled chunk coordinates, seed and estimates. Checkpoints embed the same document.
Resume reuses it exactly, restores a missing JSON from the checkpoint without
recalibration, and rejects a JSON/checkpoint mismatch. Mixed-source workers,
validation and replay collectors share those records. Inference writes its own
output JSON, reuses checkpoint estimates for known volumes, and calibrates each
new volume once. There is no alternate input-normalization mode; older architecture
checkpoints are not accepted by the v14 trainer.

After trilinear interpolation, CT values strictly above `background + 3*noise`
form the material mask. Each current crop and each historical CT slab estimates
its own foreground median/MAD from every fourth voxel on each axis (a full pass
handles a sparse foreground missed by that grid). Statistics use native uint8 bins;
interpolated intensities remain continuous. Divide by
`max(1.4826*MAD, 2*background_noise)` and clip to [-4,4]. Excluded voxels become -4,
the black/background value in normalized model space. Empty crops stay entirely
background. This is the reviewed **background + 3 noise scales** method, with no
spatial smoothing or connected-component mask.

Training applies contrast, brightness and noise after normalization, preserving
the background mask. Brightness/noise parameters express fractions of the full
8-unit normalized range. Optional blur averages material with material only;
presence remains in [0,1], direction moments and history heatmaps are unchanged.
The persistent volume cache continues to store original uint8 intensities.

Reproduce the real-volume CPU benchmark using the project Python:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_ct_normalization.py
```

Results are in `output/ct_normalization_validation/benchmark.json`. On this host,
40 timed repetitions after 3 warmups across nine reviewed crops gave main-crop
(120x104x104 float32) median 0.227–0.272 ms, mean 0.230–0.274 ms and p95
0.233–0.289 ms. A plain copy baseline took median 0.207–0.210 ms. Normalization
adds its full kernel time to sampling; the copy is a bandwidth reference, not
existing loader work it replaces. Historical 8x65x65 slabs took median
0.0065–0.0212 ms. Measurements exclude I/O, interpolation, allocation and JIT.
Initial calibration of all three cached volumes took 2.94 s; saved-estimate reuse
with metadata checks took 0.00063 s. GPU/end-to-end training throughput is unmeasured.

### BasicBlockD image and history encoders

To initialize a fresh model with the settings captured directly from
`mixed_ct_afv_stem32_fresh_run1/ckpt_031000.pt`, use:

```bash
bash scripts/train_mixed_ct_stem_fresh.sh
```

This script uses batch 6 and two gradient accumulation steps (12 examples per
optimizer update), the fine history encoder, model width 256, four image-transformer
blocks, two coordinate-decoder layers, and two scorer layers. All three transformer
components use feed-forward width 1024. It retains the checkpoint's remaining
training settings, including live continuation at 4–8 steps, tolerance 3,
LR 1e-4 and 5000 warmup updates. It starts at step zero with random weights and
a fresh optimizer; no checkpoint is opened at launch. Its default output name is
`mixed_ct_afv_stem32_fresh_run2`; set `STEM_RUN_NAME` to choose another name.
`FIBER_PYTHON` can select the Python executable. Settings are explicit in the
script, and its header records the source checkpoint's SHA256.

Start a fresh mixed-CT run with the replacement image stem and history encoder:

```bash
bash scripts/train_mixed_ct_stem.sh
```

The launcher creates `mixed_ct_afv_stem32_fresh_run1` with random model weights
and a fresh AdamW optimizer: 100000 total steps, LR 1e-4, 5000-step warmup,
batch 16 and grad steps 1, 10 workers and 48 remote prefetch connections. It uses the
same mixed-dataset configuration and persistent volume cache. No checkpoint or
migration is required. `STEM_RUN_NAME` selects another name, and extra trainer
flags can be appended. To resume this new run later, use the same launcher with
`--resume output/mixed_ct_afv_stem32_fresh_run1/last.pt`.

The image stem follows `PatchEmbed_deeper`, reusing Vesuvius's shared
`BasicBlockD`: one full-resolution block into 32 channels, then two downsampling
stages at 64 and 128 channels. Each stage has two residual blocks
(`--stem-blocks 2`), the first with stride two. There are five BasicBlockD blocks
in total. They use affine InstanceNorm without running statistics, ReLU, and
ResNet-D average-pool/projection skips. A final 1x1x1 convolution maps 128 channels
into the 128-wide token embedding. All stem layers use normal random initialization.

The stem output is added to the parallel patch projection before position/history
conditioning and axial attention. Both paths consume the sampled 120x104x104
image directly and output the same stride-four 30x26x26 token grid; no extra
image padding, tokens, or decoder feature stream are introduced. BasicBlockD
combines odd-kernel convolutions and average-pool skips, with different
receptive-field footprints from the parallel patch projection. Activation
checkpointing includes the stem when enabled. Full-resolution feature extraction
increases compute and activation memory; GPU capacity must be checked on the
training machine.

The history encoder is also initialized fresh. Its residual blocks use the shared
BasicBlockD with affine InstanceNorm and LeakyReLU(.01); its token layout is unchanged.

Patch4 axial self-attention uses the fixed-axis 3D RoPE implementation shared
with Dinovol (`vesuvius.models.build.pretrained_backbones.rope`). At width 128
with four heads, 30 of each head's 32 channels rotate (10 per spatial axis),
and two remain unchanged. Q and K rotate; V does not. All axes share the scale
of the longest grid dimension, with base 100 and no random coordinate shift,
jitter or rescaling. Sine/cosine and rotations use FP32 under BF16 autocast.
For each axial line, the two constant coordinates can be omitted: their shared
rotations cancel in Q/K dot products. Tests compare this directly with full-grid
3D RoPE, including gradients. Absolute image-coordinate embeddings and the
history position/pose/age/slot embeddings remain available to the decoder.
History cross-attention does not apply RoPE independently in each slab's frame.

`--fresh-fraction` applies to the budget left after matched decisions and
bank-following samples. With decision fraction .3, bank-following 0, and fresh
fraction .9, the requested mix is 63% annotation-fresh, 7% replay, and 30% matched
decisions. These are sampling targets; availability and paired batching affect
realized counts. Both sampling flags may change on resume. `--tolerance` may also change on resume;
it changes distance-based supervision and correctness reporting, while preserving
optimizer state and the frozen monitor inputs. Negative-bank data
still supports matched decisions and supervision even with bank-following disabled.

`--batch` is the number of independent decisions loaded and predicted together.
`--grad-steps` is the number of batches accumulated before one optimizer update.
For example, `--batch 12 --grad-steps 2` gives an effective batch of 24 decisions.
Matched pairs stay in the same batch, so batch size must be even when matched
decision sampling is enabled. Defaults are batch 4 and grad steps 2; the patch
and mixed-CT launchers use batch 16 and grad steps 1. Saved legacy configs are
translated automatically (old batch 24 / microbatch 12 becomes batch 12 / grad
steps 2); the trainer CLI now accepts only the new names. Every decision
gets equal weight. Geometry, generated survival and supplied-candidate survival
keep their coefficients (1, .5, 1). There are no observation-only steps, streamed
loss budgets, writer replay or cached historical main-encoder features.

The current crop is 120x104x104 at spacing .5, with CT/presence and optional
six direction moments. History reads only CT plus a rendered observed-path
heatmap: neither presence nor direction volumes are read for slabs. Slabs are
8x65x65 at spacing .5, with depth offsets -2 through +1.5 tracing voxels.
Selection uses actual observed arclength: seed, up to four uniformly spaced older
observations, then available anchors 96, 64 and 32 voxels behind the head. All
valid observations are at least 32 arclength voxels apart, including the seed.
Unavailable slots are padded and perform no image reads or convolution work.

Training samples a complete synthetic prefix before truncating the main model's
local history, retaining short/no-history draws. Matched pairs and synthetic
wrong-turn histories retain their actual constructed paths, including bridges.
Annotations affect labels only. Every selected slab is checked against the
training holdout before images are loaded; either unsafe member rejects a pair.
Replay and resumed recovery use saved committed prefixes, without connecting a
remote seed to a truncated local history.

The fine slab encoder uses three convolution stages, widths 32/64/128,
and strides (1,1,1), (2,2,2), (2,2,2). The first stage processes the full
8x65x65 crop before any downsampling. Each convolution is followed
by a shared Vesuvius BasicBlockD with affine InstanceNorm and LeakyReLU(.01).
Instance normalization uses one group per channel to support an empty valid-slab
batch with the same per-instance spatial statistics and no running statistics.
Each valid slab contributes 578 spatial tokens on a 2x17x17 grid, with 128 image
features projected to model width (128 by default). Lateral token spacing is
2 tracing voxels; the input crop and its field of view are unchanged. Up to eight
slabs contribute 4,624 tokens. Legacy v14 uses widths 8/16/32 and strides
(1,2,2), (2,2,2), (2,2,2), producing 162 tokens on a 2x9x9 grid with
4-voxel lateral spacing. The finer encoder increases activation memory and history
attention work; measure GPU memory before choosing the training batch size.
Position, pose, age,
slot and seed role are embedded inside the encoder. Generator and scorer have
separate residual history attention shared across their respective decoder
layers. The survival scorer has two transformer layers. The image transformer,
coordinate decoder, and survival scorer use feed-forward width 1024. The mixed-CT
run script uses model width 256, four image-transformer blocks, and two
coordinate-decoder layers. History features stay attached
and are reused across all attempts and
supplied candidates within a decision. Fully masked history contributes zero.

Training compiles supervised prediction, supplied-candidate scoring, and losses. Valid
slabs are gathered and encoded before those fixed-shape compiler boundaries.
Current-image attention uses batched, padding-masked SDPA with cuDNN preferred
on CUDA. It needs no per-row key compaction or dynamic key dimensions.
Current-image and historical attention K/V projections remain attached and are
reused within the decision, including supplied-candidate scoring.
CUDA uses BF16 autocast with FP32 survival and coordinate policy; the slab encoder
and other parameters have independent gradient clipping (5 and 100).
`--activation-checkpointing` still controls the main axial blocks.
FP32 feature sampling preserves physical coordinate interpolation; the final
survival projection and log-space accumulation preserve stopping precision.
Inductor's BF16 rounding setting is local to these compiled functions.

Logs include valid slabs per decision, historical age, actual slab-grid overlap
with the current crop, loader slab seconds, encoder seconds (CUDA events on GPU),
and historical encoder gradient norms. Loader seconds are summed worker time,
not end-to-end wall latency. Decision throughput and peak allocated VRAM are
reported separately. Current crops per update now equal supervised decisions.

### Measuring CPU data preparation

Use the saved run configuration to benchmark actual image, bank, target and
augmentation work without starting another model or modifying training data:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  ../../../../.venv/bin/python scripts/benchmark_loader.py \
  --config output/patch4_run3/config.json --workers 0 --warmup 2 --batches 20 \
  --hash-batches --out /tmp/loader.json
```

The report includes mean/p50/p95 latency and exact tensor hashes, excluding
timing fields. Hashing is outside the timed region. Add `--profile /tmp/loader.prof`
for cProfile hotspots, or use `--workers 10` to measure concurrent loader delivery.
It uses the config's initial replay paths, a fixed seed, and the live bank;
compare revisions with the same unchanged inputs. Worker profiling and concurrent
delivery are separate measurements. Neither measures GPU transfer or optimizer time.

CPU segment heatmaps fuse the original float32 distance arithmetic in Numba
(without fast math) and retain Torch's exponential. Bank masks compute coordinates
only at occupied cells. Bank draw weights are cached until the next published
manifest, and exact-distance queries defer the dense seed index until mining
actually requests it. These changes preserve sampling, crops, labels and model
precision; checkpoint formats and resume options are unchanged. As before,
resuming restarts loader RNG streams rather than reproducing uninterrupted draws.

The September 30, 2026 check against `3a7658e02` used `patch4_run3`'s
`neighbor_samples_r0_32_l80_160_v2` bank, S1 CT level 0 and configured fiber
annotations. Both versions ran alongside the original training process on an
Intel Core Ultra 7 270K Plus, with Python 3.14.4, PyTorch 2.12.1+cu130,
NumPy 2.4.6 and Numba 0.66.0 (normal cached JIT, `fastmath=False`). The command
above measured 20 batches of eight after two warmups, with seed 0:

| CPU preparation, seconds/batch | Before | After |
| --- | ---: | ---: |
| Mean | 3.077 | 1.746 |
| Median | 2.197 | 1.060 |
| p95 | 6.099 | 5.813 |

All 860 non-timing tensor hashes matched across those 160 decisions. These are
single-worker measurements under concurrent training load, including occasional
host-memory stalls; they are not end-to-end training throughput. Reports and
profiles are in `output/loader_speedup/`. In the initial ten-batch cProfile
comparison, unused dense seed-index construction consumed 13.0 seconds and
full-grid construction 6.2 seconds; both were removed from those training paths.
Segment rendering fell from 3.0 to 0.54 seconds before the additional draw-weight
cache. Validation: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
../../../../.venv/bin/python -m pytest tests -q -o cache_dir=/tmp/patch_loader_pytest`
reported 279 passed, 15 skipped, one deselected and ten passing subtests. GPU
tests were skipped in the sandbox; the running job provides the live GPU check.

`patch4_run3` resumed at step 1,000 with model/EMA/AdamW/RNG state intact and
the same 10 workers, batch 8, grad steps 2 and one refinement step. On the RTX
5090, the original updates 51–1,000 versus resumed updates 1,051–1,200 gave:

| Live measurement | Before | After |
| --- | ---: | ---: |
| Mean data wait / update | 786 ms | 130 ms |
| Median data wait / update | 780 ms | 106 ms |
| p95 data wait / update | 1,023 ms | 171 ms |
| Overall crops / second | 16.38 | 44.09 |

Wait statistics summarize 50-update logging windows (19 before, three after),
excluding the initial startup windows. Throughput includes the full logged wall
time. This is an 83.4% wait reduction and 2.69x throughput, with live sampling
variation and a shorter post-resume observation period. See
`output/loader_speedup/live_comparison.json` and `handover.json` for the measured
intervals and exact resume command. The original checkpoint is retained at
`output/patch4_run3/ckpt_001000.pt`.

### Second loader optimization round

The next round, measured against `a0d3a940b` with the same hardware and run
configuration, removes more repeated CPU work:

- Rotate the 65,536 possible direction-byte pairs once per crop, then interpolate
  through that lookup table without expanding the source block into six fields.
- Sample identical crop geometry once per batch. Matched decisions still receive
  independent output storage, historical slabs, augmentation and labels.
- Batch exact nearest-polyline queries, retaining the original arithmetic and
  unsorted tree traversal for identical tie breaking. Large queries are bounded
  to 256 points per batch.
- Cache resampled bank-path lengths alongside the shard geometry.

Sequential CPU-only runs of these algorithmic changes alongside training measured 40 batches of eight
after four warmups, with seed 0 and the config's initial (empty) replay list.
Both used one loader thread, normal cached Numba JIT without fast math, and no
profiler in the timed comparison:

| CPU preparation, seconds/batch | `a0d3a940b` | Optimized |
| --- | ---: | ---: |
| Mean | 1.887 | 1.139 |
| Median | 1.637 | 1.069 |
| p95 | 3.277 | 1.614 |

This is **1.66x loader throughput**, with all **1,720 non-timing tensor hashes
identical** across 320 decisions. No sampling probabilities, image precision,
augmentation or supervision rules changed. The baseline cProfile run attributed
20.2 of 55.1 seconds to direction crops, 8.0 to scalar interpolation, and 7.8 to
nearest-polyline queries. An intermediate profile reduced the latter to 1.8
seconds; use the unprofiled table above for overall timing, since profiling and
concurrent workloads affect the measurements.

From the fiber-follow directory, the CPU benchmark command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  ../../../../.venv/bin/python scripts/benchmark_loader.py \
  --config output/patch4_run3/config.json --workers 0 --warmup 4 --batches 40 \
  --hash-batches --out /tmp/loader-round2.json
```

Reports, profiles, unchanged baseline source snapshots and its benchmark harness
are retained in `output/loader_speedup_round2/`. The full suite passed 287
tests (15 skipped, one deselected, ten passing subtests). The final guard against
reusing mixed-dtype coordinates also passed all 14 direction/crop tests. Validation used
`NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
../../../../.venv/bin/python -m pytest tests -q -o cache_dir=/tmp/fiber-round2-pytest`.
GPU tests were skipped in the sandbox. These CPU measurements are specific to
this Ubuntu x86-64 host; the changes introduce no platform-specific dependencies.

The larger batch exposed additional allocation and IPC costs. Main images
and historical slabs now use the same direct shared-storage allocation as
PyTorch's default worker collator. NumPy views write into that storage, and the
original shared tensor is returned, eliminating the worker queue's full image
copy. Outside a worker, allocation remains ordinary CPU tensor storage. A real
one-worker loader also matched all 1,720 baseline hashes after this change.

`launch_memory.sh` defaults `NUMPY_MADVISE_HUGEPAGE=0` (overridable). On this
Linux host, large transient NumPy allocations were incurring huge-page compaction
stalls. This process-local setting changes allocation policy, not values or
system-wide settings. A CPU-only 20-batch transfer benchmark, after three warmups,
used the actual `(16,8,120,101,101)` float32 image shape: mean/median/p95 delivery
fell from 506/442/1,023 ms to 282/281/341 ms with direct shared storage. This measures
synthetic filling and IPC, not complete training; see `batch_transfer.json` and
`benchmark_batch_transfer.py` in the results directory. Run the latter with the
project Python and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, adding `--shared`
for direct allocation. The helper uses PyTorch's internal storage API, as its
collator does; storage-preservation and worker-delivery checks cover this boundary.

`patch4_run3` finally resumed from `ckpt_005000.pt` with model, EMA, AdamW and RNG state
restored (283 optimizer entries, no optimizer or LR reset). At the user's request,
the resumed run uses **batch 16 / grad steps 1**, with 10 workers and the other
training options retained. Both step-5,000 evaluations finished before stopping:
diagnostics took 31.98 seconds and recovery took 23.89 seconds; their reports and
images remain in the run directory. `output/loader_speedup_round2/handover_after_eval.json`
records completion evidence, the exact launch command, environment override and PID.
As with every existing resume, loader
random streams restart; changing batch grouping also changes sampled batches.
Live throughput therefore reflects both loader improvements and the requested
batch-size change; the fixed-batch CPU table isolates the loader changes.

The final live comparison uses 19 pre-work logging windows (950 updates ending
at steps 1,500–2,450, excluding the window containing the step-2,000 evaluation)
and nine post-restart windows (450 updates ending at steps 5,100–5,500). Startup,
periodic evaluation and active benchmarks are excluded. Wait percentiles describe
the 50-update window averages, not individual-update tail latencies.

| Live training measurement | Before this round | Final run |
| --- | ---: | ---: |
| Mean data wait / update | 151.8 ms | 31.0 ms |
| Median data wait / update | 146.9 ms | 29.3 ms |
| p95 data wait / update | 220.6 ms | 38.9 ms |
| Crops / second | 40.35 | 71.89 |
| Peak allocated GPU memory | 6.83 GiB | 13.30 GiB |

This is **79.6% less data wait and 1.78x live throughput**, including the requested
batch-size increase. GPU memory remains within the RTX 5090's capacity. Detailed
windows and the comparison are saved in `output/loader_speedup_round2/live_after.json`
and `live_comparison.json`. Training remained active beyond step 5,500.

## Source sampling and labels

The default requested shares remain 30% matched decisions, 20% bank following,
35% annotation-fresh, and 15% replay before fallback/rejection. Of fresh draws,
20% request covered-parent resampling and 30% request synthetic wrong turns.
75% of requested matched pairs teach recoverable geometry choices; the rest
contrast a departure with legitimate following. The retired stream-crop cap and
auxiliary historical decision budgets no longer alter these decision shares.

`--bank-hard-fraction .5` keeps half of bank proposals uniform. For the other
half, the loader chooses one of three geometry criteria uniformly and selects
the strongest of eight proposals: nearby paths with similar tangents and bending,
curved paths, or converging/diverging neighbors. Comparisons use four-voxel
arclength spacing; distances and curve similarity are independent of vertex
density and trace direction. Following draws still honor unique-path eligibility;
matched decisions and covered-parent draws use all valid relationships. This
changes exposure to the existing trusted bank; it does not change its annotations
or mine previously absent weak-signal examples.

New DAgger collections use the negative and near-negative banks to detect foreign
contact along the actual committed polyline, including between decision heads.
The first intersection with a `.75`-voxel bank-path tube outside the intended
annotation's `1.5`-voxel tube certifies a switch. Both radii are configurable
(`--bank-switch-tolerance`, `--bank-own-tolerance`). Overlapping tubes remain
ambiguous; absent bank coverage cannot establish fiber identity. Contact positions
are continuous segment/capsule intersections, not rounded voxel or head locations.
Once departed, the existing stopping-only supervision remains absorbing.

Replay stores `failure_kind`, `travelled`, `switch_pos`, `switch_distance`,
`switch_decision` (the within-trace decision that committed the contacting segment),
`switch_bank_path` (shard/path index), and `switch_bank_run`. The pre-switch window
retains geometry targets and shares the event metadata; confirmed switched states
have geometry masked. These labels never enter model inputs. Bank provenance is
saved with the collection. `--replay-failure-fraction .5` reserves half of replay
for available generic departures, bank switches, premature stops, endpoint
overshoots, and pre-switch states, equally by category and then by fiber. The other
half balances recoverable drift bands. If one side is empty, the other receives
its budget; fully empty replay falls back to annotation-fresh draws. Logs report
collected categories and realized training endpoint counts.

Replay v6 requires newly collected caches. It stores every committed vertex once per trace,
with causal prefix indices for each decision; old caches are rejected.

Each matched endpoint scores four shuffled paths: the two bank/annotation
continuations and two smooth transitions between them, with varied transition
onsets. Paired observations share candidate geometry and order. Candidate labels
use the same dense tolerance, recovery limit, foreign masks, endpoint semantics
and censoring as generated paths. They are not assigned all-positive/all-negative
labels by candidate identity. Logs count first-segment and later first failures.
The transitions are rejected-path proposals, never geometry targets or certified
bank following paths.

The loader no longer samples contrastive points or emits embedding query tensors.
Lateral resampling uses visible forward bank coverage directly. The obsolete
`--negative-near-fraction`, `--negative-near-distance` and `--negative-lateral-max`
options are removed; bank mining and crop support determine available foreign
geometry. Separate negative, following and continuation banks remain supported.

## Verification and diagnostics

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../../../../.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/fiber-slab-pytest

../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.history_learning \
  --device cuda --steps 250 --out /tmp/slab-learning.json

../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_slabs \
  --warmup 2 --repeats 10 --decisions 6 --out /tmp/slab-benchmark.json

# Match the launcher's batch, including batched attention and candidate padding:
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_slabs \
  --warmup 3 --repeats 10 --decisions 16 --microbatch 16 --out /tmp/slab-benchmark-b16.json
```

Use fresh output paths. The controlled learning check has two fixed identity
pairs with identical current crops and differing historical CT, one pair requiring
non-seed evidence. It requires correct geometry, acceptance of valid supplied
and generated paths, and rejection of wrong continuations. Rejecting everything
fails. It reports full history, seed only, no history, and shuffled CT with all
metadata and path heatmaps held fixed. This is a learning-capability check,
not evidence of held-out tracing quality.

For a frozen real paired tensor batch from `IdentityObservationBuilder`, run:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.regression.history_diagnostic \
  --checkpoint output/RUN/last.pt --fixture /tmp/paired-batch.pt \
  --out /tmp/paired-history-report.json
```

The fixture must contain complete adjacent matched pairs, candidates and labels;
it can be reused unchanged across checkpoints. Existing held-out rollout and
monitor recovery evaluations remain part of training. `identity_preflight` checks
real-data independent batches, and `benchmark_feature_training` measures a bounded
fork of a compatible run using its actual data. The synthetic benchmark excludes I/O;
compare equal hardware, precision and supervised decision counts.

See [recorded validation](SLAB_VALIDATION.md) for measured results and limits.

## Mixed Paris 4 and AFV training

Run from `fiber_follow/`:

```bash
bash scripts/train_mixed_ct.sh
# Optional CPU check of real crops, supervision and forward/backward for all sources:
/home/sean/Documents/villa4/vesuvius/.venv/bin/python scripts/check_mixed_ct.py
```

The launcher uses [mixed_ct_datasets.json](../configs/mixed_ct_datasets.json):
10% Paris 4, 45% `0175A_5mm_v1.afv`, and 45% `1447_5mm_v1.afv`.
`0175A_SMALLMORE.afv` is excluded. Weights apply to batches, preserving
adjacent matched decision pairs. Source counts and actual shares are logged.
The new run uses one CT channel, patch4, three refinement steps, batch 16 and grad steps
1, and initial LR 0.0003. Trailing command-line options override launcher defaults.
It starts a new model; CT-plus-prediction checkpoints have different input shapes.
Modes using presence/direction predictions still require those volumes.

AFV geometry is read lazily from SQLite in native L0 XYZ coordinates. Its embedded
scan identity and the file SHA256 must match the config. The CT sources are
PHerc0175A/20250521115057 and PHerc1447/20250521151220 (8.64 µm native voxels).
AFV trace coordinates use two native voxels (17.28 µm); the existing Paris 4 trace
grid is approximately 16 µm. Main crop spacing is half a trace voxel in both.

Remote CT chunks are fetched on demand into
`<cache_dir>/<scheme>/<bucket>/<volume-path>/<level>/`, using the normal URL-based
layout. The mixed dataset JSON sets `cache_dir` to `/mnt/raid_nvme/volume_cache`,
for example `/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/PHerc0175A/volumes/<scan>.zarr/0/`.
Local Zarr v2 metadata has `compressor: null` and `filters: null`, preserving the
source's chunk separator. There is no separate `uncompressed` directory.
Chunks are decoded once, written atomically without compression, then memory
mapped across workers and collectors. Reopening a cached region needs no network;
uncached regions still require access to the source. The disk cache grows with
visited regions and has no eviction policy; each worker has a bounded RAM/mapping
cache. Cache paths are relative to the dataset JSON unless absolute. Edit the
top-level `cache_dir` to relocate future writes, then restart/resume training.
Cache-only path changes are allowed on resume; existing cache files are not
moved automatically, and missing chunks are fetched again at the new location.

Optional async remote prefetch:

```bash
bash scripts/train_mixed_ct.sh --remote-prefetch-connections 8 \
  --remote-prefetch-lookahead 16 --remote-prefetch-queue-size 512 --remote-prefetch-timeout 120
```

With these options, one separate process runs up to eight concurrent async Zarr
requests. The S3 connection pool limit per client automatically matches
`--remote-prefetch-connections`; it does not create a worker per connection. It has its
own Python GIL; the training workers never execute remote reads in this mode.
Workers plan exact main-crop and historical-slab footprints before CT assembly.
`--remote-prefetch-lookahead` defaults to 16 future batches per remote source
per worker; 0 restores the shallow per-item hints. The first batch is delivered
before filling the deeper queue. Plans hold geometry, augmentation seeds and
labels, without CT tensors or dense foreign masks. Neighbor coverage feedback
advances during planning so fixed source/replay banks produce the same sample
sequence and targets. Mixed-source selection and matched pairs stay ordered.
Initial metadata is also fetched by the async service. Local Paris 4 reads
bypass this service.

The async process retains each worker/source's bounded chunk window and feeds
the smaller fetch queue as slots free, including while compilation stops batch
consumption. Updated windows replace old ones, and sources/workers are serviced
round-robin. With ten workers and two remote sources, the default plans up to
320 future batches beyond the usual DataLoader buffers. It stops fetching
when those regions are cached; it does not scan arbitrary parts of the volumes.
Replay and live neighbor-bank updates can affect newly planned batches; already
planned batches remain ordered and are not resampled. Consequently deeper
lookahead delays incorporating newly published replay by the bounded plan queue
plus the existing replay refresh interval. Geometry-only neighbor feedback can
add CPU work, although dense masks and image augmentation are deferred.

A priority heap distinguishes speculative ahead requests from chunks needed to
finish a batch now. A needed batch promotes matching queued or active fetches;
unrelated speculative requests already in flight can be cancelled to free
connections, then retried behind demand. Each IPC priority queue and the pending
heap are bounded by `--remote-prefetch-queue-size`; full queues defer requests
without blocking submission, and the readiness gate resubmits missing demands.
Plan windows have a separate bounded IPC queue and one retained window per
remote source/worker, so a full chunk heap does not discard accepted windows.
Atomic uncompressed writes and per-chunk process locks prevent partial reads and
duplicate downloads across training and collector processes.

Before assembling crops, a loader waits until all required chunks are cached,
then uses a **cache-only** reader: there is no synchronous network fallback.
Training can still stall when downloading cannot keep up. Fetches retry up to
three times; exhausted retries, a dead service, or the configured batch-readiness
timeout raise a visible error rather than skipping samples. Prefetch is disabled
by default (`--remote-prefetch-connections 0`) and its settings can change on
resume. Cumulative completion bytes, promotions, preemptions, deferred requests,
errors, retained lookahead chunk references/windows, and service status are
recorded under `remote_prefetch` in training logs.
These switches affect the training loader; standalone evaluation and replay
collectors retain their existing readers and share the same disk cache.

Verification commands (CPU only):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /home/sean/Documents/villa4/vesuvius/.venv/bin/python scripts/check_mixed_ct.py \
  --prefetch-connections 8 --out datasets/automated_fiber_volumes/training_prefetch_preflight.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /home/sean/Documents/villa4/vesuvius/.venv/bin/python scripts/check_remote_prefetch.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /home/sean/Documents/villa4/vesuvius/.venv/bin/python scripts/check_prefetch_lookahead.py \
  --out datasets/automated_fiber_volumes/lookahead_benchmark.json
```

The lookahead benchmark uses temporary caches and simulated 50 ms chunk latency,
four concurrent fetches, a two-entry chunk heap, and a 0.5 s idle-loader pause.
Across three alternating repetitions, lookahead cached all 16 future chunks
during the pause; shallow prefetch cached none. Mean subsequent demand wait was
0.00076 s versus 0.276 s, with identical voxel hashes. This isolates scheduler
behavior; it does not measure real S3 bandwidth or training throughput.

The recorded cold-cache check fetched eight 128³ uint8 chunks from each S3 scan
(32 MiB total), with three repetitions per connection count and alternating
order. Voxel hashes matched across all runs. Process and metadata startup were
excluded; remote/network cache state was not controlled.

| Concurrent requests | Mean fetch time | Median | p95 |
| --- | --- | --- | --- |
| 1 | 7.53 s | 8.51 s | 9.37 s |
| 8 | 2.58 s | 2.45 s | 3.40 s |

This is fetch throughput, not measured training throughput. Full measurements
are in `datasets/automated_fiber_volumes/prefetch_benchmark.json`. Unit tests also
exercise demand promotion/preemption, concurrent async reads, bounded queues,
multiple loader processes, retry failure, unchanged sampling, and cache-only
reads from compressed remote sources.

Validation reserves deterministic whole fiber IDs independently in each source;
training does not exclude a Z band. Each AFV source reserves exactly 406 fibers
(`validation.count`), for **1,000 held-out fibers total** across all sources. Paris 4
keeps its 124 previously reserved fibers and reserves another 10% of the remaining
641, leaving 577 training and 188 validation fibers. This preserves the existing
neighbor bank's parent compatibility. Its old `val_z` remains mining provenance
only. Validation geometry is excluded from AFV neighbor queries and filtered out
of Paris 4 mined paths, so it cannot reenter through intentional switches,
neighbor following, matched choice/departure pairs, or foreign labels.

Each source supports those neighbor tasks and its own rollout replay. Collection
cycles through all sources, running one subprocess at a time, with separate replay
caches. Only training fibers seed replay, and replay identity manifests must match
the source's training split. Monitor rollout metrics are evaluated and plotted
separately for each source. The rollout recovery fixture and optional long diagnostic
remain Paris 4 diagnostics. Held-out monitor, calibration and final groups are
distinct; seed generation is bounded to 32 fibers per group.

The CPU preflight writes the exact reserved IDs and evaluation manifests under
`datasets/automated_fiber_volumes/splits/`, plus `training_preflight.json`.
Training writes its manifests into the run directory and saves the resolved
dataset config and its digest in checkpoints. Resuming rejects a changed dataset
config or split. Editing weights or holdouts requires a separate experiment.

## Historical v9 checkpointing measurements

The following existing measurements are preserved from before the slab replacement.
They concern the retired recurrent implementation, not v10 configuration options.

A bounded RTX 5090 comparison (Torch 2.12.1, seed 194, two streams of 24
observations, two warmups and three measured updates, two refinement steps)
measured the following for the token-only model. Each update had 48 observations,
six supervised decisions, 14 historical encoder crops with gradients and 90
writer replay operations, exercising all three historical age bands.

| Historical encoder checkpointing | Peak allocated | Peak reserved | Mean update | p50 / p95 |
| --- | --- | --- | --- | --- |
| Enabled (default) | 1.310 GiB | 1.457 GiB | 913 ms | 910 / 921 ms |
| Disabled | 2.967 GiB | 3.195 GiB | 892 ms | 892 / 897 ms |

Disabling it used **1.658 GiB more live tensor memory** in this workload. The
measured time reduction was only 2.2%, from a short sample. This excludes I/O and
is not a tracing-quality comparison. The default remains enabled. Full local
results and exact commands are in `output/token_only_validation/comparison.json`.

### Current microbatch diagnostics

`--batch-diag-every 1000` writes the entire latest training microbatch, in loader
order, using EMA inference. Its row count follows the actual microbatch size.
All image types include the same examples, and all layers are captured at every
image event. Set the cadence to zero to disable these outputs.
On resume, the first completed update also writes a sheet set so the active
diagnostics can be checked immediately.

Each `output/<run>/diagnostic_images/<iteration>/` contains exactly:

- `predictions.png`: two crop-local CT views with annotation, observed history,
  initial and selected predictions, actual committed prefix, and confidence.
- `crop_orientation.png`: fixed u=0, v=0 and forward=0 CT sections plus an
  orthographic view of the real crop box and annotation. Green segments are
  clipped to the crop and to half a crop voxel on either side of each section;
  out-of-plane paths are not projected onto the CT. The blue dot is the crop
  origin. The full in-crop path and heading are shown in the 3D panel. No
  target-following reslicing is used. Heading/annotation angle is measured near
  the crop head.
- `encoder.png`: patch/stem embedding, early/middle/final encoder blocks and
  encoder output. Spatial channel-contrast RMS reveals variation even after
  LayerNorm. Scales are shared across examples within each stage.
- `decoder.png`: actual query-by-hidden-channel activations through each decoder
  stage for the selected refinement attempt. These are not spatial CT images.
- `history.png`: each historical CT/path input, convolution and token features,
  and generator/scorer history attention at the selected attempt. Padding stays
  explicit. Attention is averaged across heads/layers; it is not attribution.
- `metrics.json`: source, supervision availability, errors, confidence/hazards,
  prefix calibration, commits/refinements, frame quality, world frame/position,
  CT/activation statistics, history attention/ages, existing optimizer-update
  metrics, and measured inference/render time. Unknown values are JSON null.

The diagnostic path writes no fixtures, tensor archives, NPY or NPZ files.
Annotation coordinates are carried as small CPU metadata with the training
batch; diagnostics do not read CT again or construct extra datasets. Activations
are reduced before copying to CPU. Singleton eager inference bounds GPU memory
and preserves adaptive-refinement row identity. Direct raster contact sheets
and concurrent low-compression PNG writes avoid per-panel plotting overhead.
These current, augmented training examples are not held-out accuracy estimates.
The separate rollout/recovery evaluation cadence is unchanged.

Validation:

```bash
../../../../.venv/bin/python -m pytest -q -o cache_dir=/tmp/fiber-pytest-cache \
  tests/test_batch_diagnostic.py tests/test_observation_preparation.py \
  tests/test_training_defaults.py tests/test_point_logging.py \
  tests/test_decision_training.py tests/test_history_slabs.py
```

GPU timing on RTX 5090, the running stem32 checkpoint at update 3000, 12
full-size examples (six copies of an existing matched pair), four CPU threads,
with concurrent training: forcing all three refinement retries took 4.27 s on
the first event and 2.63 s warm, including five PNGs and JSON. Peak diagnostic
allocation was 0.50 GiB. These are measured examples, not a hard time guarantee
for arbitrary batch sizes or hardware. Run the benchmark with
`AGENTS_AGENT_MODE=1 MPLCONFIGDIR=/tmp/fiber-tests-mpl ../../../../.venv/bin/python
/tmp/benchmark_microbatch_refinements.py`; measurements and previews are in
`/tmp/microbatch_diagnostic_benchmark/`.

After checkpoint-5000 evaluation completed, the live run resumed from that
checkpoint. Its first current-microbatch diagnostic at step 5001 rendered all
12 examples and wrote all six files in 1.30 s (0.64 s inference/measurement).
