# Live historical slab regression

The direct follower predicts continuous paths and causal survival confidence from
one current crop and up to eight live historical slabs. See
[architecture and supervision](TRAJECTORY_MEMORY.md). The v16 path tokens add three
observed-path tokens per historical slab; the v17 `--path-geometry-tokens` add 29
memory tokens sampling the committed observed polyline 1–512 arclength voxels
behind the head plus its first point. No annotation geometry enters either.
Checkpoints that record `crop_zscore_v1` normalize each current and historical CT
crop by its own mean and population standard deviation before augmentation.

## Aligned training and tracing

Training, collection, replay, live continuation, evaluation and deployment share one
implementation (`plans/consolidated_training_tracing_alignment.md`). There are no
alternative modes, legacy flags, compatibility readers or migration tools; replay and
fixtures from earlier pipelines are rejected and must be recollected.

Labels, correspondence, replay and evaluation all use the load-time repaired annotations
(kinks bridged, fold-back fibers out of training; see the top-level README). `--init-weights`
therefore compares fiber identities and source holdouts with the initialization checkpoint,
not geometry hashes; `--resume` still requires identical geometry.

### State contract (`shared/state_labels.py`)

Every state from every source is labeled by `label_state` with one set of trace facts:

- **Historical events** of a directed trace: the first sustained geometric departure
  (3 voxels for 3 committed points, as in `score_trace`), the first certified
  foreign-fiber contact and the annotation-boundary crossing. Their locations persist.
- **Correspondence** on the original fiber only, searched from 8 voxels behind the
  previous match to the new travel plus 32 ahead. It keeps updating after a departure
  and records validity and ambiguity. It never matches another fiber or winding.
- **Current supervision**: `following`, `recoverable`, `terminal` or `unknown`, with a
  reason and explicit `geometry_valid` / `confidence_valid` masks. The sticky
  `offtrack` field is gone.

| State | Supervision |
| --- | --- |
| Following (head within 3 voxels), certified connection | geometry and proposal confidence |
| Displaced, certified connection | recovery geometry and proposal confidence; live chains continue |
| Confirmed committed switch, tagged physical endpoint, or first plane unreachable within the commit limit | rejection; live chains end after this example |
| No/ambiguous correspondence, unannotated continuation, no visible original-fiber evidence while displaced | censored |
| Reachable first plane but the annotated connection exceeds the limit | proposal confidence only |

A geometric departure can become following again; a confirmed switch stays terminal
for the episode. The six-voxel limit bounds the whole origin-to-first-point segment
(`recovery_allowed`), so with the first plane at one voxel the lateral reach is
`sqrt(35)`. The confidence-label tolerance (`--tolerance`) is separate from the
3-voxel departure threshold. `d <= 3`, `3 < d <= 6` and `d > 6` are descriptive
strata in diagnostics only.

**First connection.** With a neighbor raster, a proposal fails at its first point when
its origin-to-first-point segment crosses a foreign cell (sampled every 0.25 voxel),
even if its endpoint is correct; a target whose own connection crosses one teaches no
geometry (`connector_rejected_targets` is logged). Annotation and bank checks are
supervision/evaluation oracles only; deployment relies on learned confidence.

### Operating policy

`OperatingPolicy` (confidence threshold, commit count, recovery limit, refinement
steps) is saved in every checkpoint and used for live feedback, collection,
evaluation and deployment. Same-position refinement runs inside the model; a decision
that stays rejected stops the trace immediately. There is no stop patience and no
forced exploration.

### Collection (`shared/collect.py`)

Each collection traces one directed episode per distinct eligible fiber (default 64)
from a saved coverage cursor: unseen fibers come before repeats, directions alternate
per fiber, and fibers are covered uniformly regardless of training length weights.
Seeds without CT orientation are skipped with a recorded reason. Episodes are 768
trace voxels, collector batch 8 (`--dagger-forward-chunk` bounds model rows per call).
The rejected final decision and its proposal are retained. Dense decisions are kept
within 48 voxels before an excursion, during recovery, at stops and at terminal
failures (at most 64 voxels after the first terminal state); ordinary following is
thinned every 16 voxels. Each row stores its supervision, trace facts, events, the
policy's own proposal, an episode and event id, and one replay class (precedence:
terminal, premature stop, recoverable, pre-excursion, ordinary). The trainer launches a
collection every 1,000 updates, round-robin over sources, one at a time; busy launches
are skipped and logged with the achieved interval and publication age.

### Task budget (`shared/data.py`)

One budget applies within each dataset source (`--task-share NAME=SHARE` overrides):

| Task | Share |
| --- | ---: |
| fresh simulated traces (startup and excursions included) | 40% |
| live continuation | 25% |
| DAgger pre-excursion (48 voxels before) | 8% |
| DAgger recoverable | 6% |
| DAgger terminal | 8% |
| DAgger premature stop with a supported continuation | 3% |
| DAgger ordinary following, stratified by travel | 5% |
| certified synthetic terminal failures | 5% |

Replay is indexed by class, fiber, episode and event; draws pick a fiber, then an
episode, then an event, then a row. `--replay-max-age` and `--replay-event-cap` are
shared by all loader workers of a source. Missing positive replay falls back to a fresh
example of the same kind (recoverable falls back to a fresh excursion), missing terminal
replay to synthetic failures within `--terminal-fallback-cap`, and anything else to a
fresh example; every fallback is logged. Live chains have their own slots and never take
replay slots. Logs report, per source, requested and delivered shares, fallbacks,
supervision and reasons, label masks, positive/negative confidence targets, distinct
fibers/episodes/events, source age, event reuse, startup draws, realized seed ages and
replay travel strata.

**Fresh traces.** `make_sample` draws 15% seed-only starts, 17% with 1–8 and 17% with
9–32 requested voxels of history, and 51% with history uniform over the available
annotation (`--startup-shares`). The trace follows GT with commit-structured lateral
error (`trace_noise`): a chain of ~16-voxel commits from the seed, one mean-reverting
step of the offset at each join (33-voxel correlation, small per-trace bias) and a smooth
bulge within each commit, a per-trace scale log-uniform in 0.22–0.76, plus the
annotation's small-scale wiggle that the tracer does not follow (Gaussian 1.5 voxels).
It was fit to on-track 81k rollouts against the repaired annotations and matches their
lateral residuals, 1–2 voxel turning and correlation on held-out traces
(`output/trace_noise_realism_20261002/refit_repaired/REPORT.md`). 20% of established
traces add a smooth lateral excursion (amplitude uniform 3–6, rise log-uniform 16–128
voxels, matching real excursions) with the head on its departure or return. Heading, history and seed reference
come from the tracer's own functions; the seed heading is the CT H/V axis, and a seed
without CT orientation is rejected and redrawn within its task (no annotation fallback).

**Live chains** start half from recorded valid pre-excursion/recoverable prefixes
(restoring the observed prefix, seed reference, heading boundary, correspondence and
events) and half from fresh seed-only states, run 12–32 decisions in three balanced
bands, advance only through prefixes the operating policy accepts, continue through
following and recoverable states and end after delivering a terminal or partially
labeled state.

**Synthetic failures** follow a noised original prefix (the same `trace_noise`) and a short certified
neighbor tail (`--synthetic-tail`, default 4–16 voxels) and require visible
original-fiber evidence. Decision pairs, the pair-ranking loss and supplied-candidate
supervision were removed.

**Augmentation footprints.** A recorded or resolved frame takes its roll augmentation
(flip .5, N(0, 5°) clipped at ±15°) before footprints and reads are planned, so the
crop, slabs, references and labels share the final frame. Unresolved CT frames keep the
roll pending; their planned block already covers every roll about the heading.

### Running the aligned experiment

```bash
# 1. CPU tests (targeted files; see tests/test_alignment.py for the acceptance checks)
PYTHONPATH=../../.. ../../../../.venv/bin/python -m pytest tests/test_alignment.py -q
# 2. A bounded collection on the baseline, then the sampler check (nonzero correct replay)
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.tracing.collect \
  --checkpoint output/mixed_ct_afv_stem32_run3_pathgeom_paris50/ckpt_081000.pt \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers --dataset-name paris4 --out output/CHECK/paris4.npz
../../../../.venv/bin/python scripts/check_task_sampler.py --checkpoint CKPT --replay paris4=output/CHECK/paris4.npz \
  --out output/CHECK/sampler_check.json
# 3. The run: model/EMA tensors from 81k, fresh AdamW, empty replay, 40,000 updates
bash scripts/train_aligned.sh
```

`--init-weights` loads matching model and EMA tensors only; the sampler, labels,
options and optimizer are new, and source splits and CT normalization must match.

### Evaluation and export

`regression/evaluate.py` is the single protocol on the shared threaded tracer:

```bash
python -m vesuvius.neural_tracing.fiber_follow.evaluation.evaluate calibrate --checkpoint CK --out EVAL
python -m vesuvius.neural_tracing.fiber_follow.evaluation.evaluate run --checkpoint CK \
  --policy EVAL/selection.json --splits final --max-len 2000 --out EVAL/final.json
python -m vesuvius.neural_tracing.fiber_follow.evaluation.evaluate run --checkpoint CK --confidence .5 \
  --splits monitor calibration final --out EVAL/all_c0.5.json
python -m vesuvius.neural_tracing.fiber_follow.evaluation.evaluate compare BASE.json NEW.json
```

Calibration uses calibration seeds only and selects coverage among thresholds with at
least 95% scored precision; if none qualifies it says so and locks nothing. Runs report
the strict first-departure metrics and sticky wrong length together with geometric
excursions and returns (two voxels for 32), sustained distance events beyond six voxels
(returned, not returned, or ended), confirmed switches and length after them with
identity coverage, rejected unsafe proposals, stops with a supported continuation,
recovery commits crossing a certified neighbor and current geometric agreement, by
source and split. Comparisons pair identical seeds with fiber-resampled intervals.

`infer.py` exports every valid trace by default (`--min-length` is an opt-in filter) and
writes `seeds.json` with one status per seed: exported, seed-only, CT-unavailable,
duplicate or filtered, with reasons.

### Architecture identifiers

Fresh runs use the fine history encoder (`--history-encoder fine`), with architecture
identifiers `axial_fiber_slabs_v15`, `axial_patch4_overlap_fiber_slabs_v15`,
`axial_patch4_overlap_tokens_fiber_slabs_v15`, and
`axial_patch4_residual_stem_tokens_fiber_slabs_v15` (v16/v17 with path tokens).
Changing history architecture during a resume is rejected. Patch4 uses a learned
6x6x6 convolution with stride 4 and padding 1. The sampled crop is 120x104x104; patch4
dimensions must be multiples of four. The token grid is 30x26x26 with centers at
offset 1.5 input samples; dense reconstruction uses 4x4x4 output cells.

`scripts/stop.sh RUN_NAME` freezes the trainer and its descendants, including orphaned
collectors identified by this run's checkpoint and output paths.

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
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.tracing.infer \
  --checkpoint output/run/last.pt --seed 18529.9,13044.9,51234.1 --family H --out output/traced
```

After confident commits, heading is a free-intercept linear fit to 13 equally
spaced samples spanning the last 12 trace-grid voxels of trusted committed path.
With less than 12 voxels, the current heading is retained. A rejected decision ends
the trace. Replay preserves the trusted heading boundary when a decision is resumed.
There is no alternate heading policy.

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

Fresh training simulates the tracer's own heading and position, then resolves
roll before sampling images. History and labels rotate together. Replay holds its recorded frame; historical slabs resolve local CT
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

The aligned run (`scripts/train_aligned.sh`) uses this stem (32 channels, two blocks per
stage) with two axial encoder blocks at width 256 and FFN 256, six path decoder layers at
FFN 2048 and four survival scorer layers.

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

`--tolerance` may change on resume; it changes distance-based supervision and correctness
reporting while preserving optimizer state and the frozen monitor inputs. Sampling and
label options otherwise must match on resume.

`--batch` is the number of independent decisions loaded and predicted together.
`--grad-steps` is the number of batches accumulated before one optimizer update.
For example, `--batch 12 --grad-steps 2` gives an effective batch of 24 decisions.
Defaults are batch 4 and grad steps 2; the aligned launcher uses batch 4 and grad
steps 3. Saved legacy configs are
translated automatically (old batch 24 / microbatch 12 becomes batch 12 / grad
steps 2); the trainer CLI now accepts only the new names. Every decision
gets equal weight. Geometry and generated survival keep their coefficients (1, .5). There are no observation-only steps, streamed
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
local history, retaining short/no-history draws. Synthetic wrong-turn histories retain
their actual constructed paths, including bridges. Annotations affect labels only.
Every selected slab is checked against the training holdout before images are loaded.
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
and are reused across all attempts within a decision. Fully masked history contributes zero.

Training compiles supervised prediction and losses. Valid
slabs are gathered and encoded before those fixed-shape compiler boundaries.
Current-image attention uses batched, padding-masked SDPA with cuDNN preferred
on CUDA. It needs no per-row key compaction or dynamic key dimensions.
Current-image and historical attention K/V projections remain attached and are
reused within the decision.
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

## Bank geometry and switch certification

`--bank-hard-fraction .5` keeps half of bank proposals uniform. For the other
half, the loader chooses one of three geometry criteria uniformly and selects
the strongest of eight proposals: nearby paths with similar tangents and bending,
curved paths, or converging/diverging neighbors. Comparisons use four-voxel
arclength spacing; distances and curve similarity are independent of vertex
density and trace direction. This changes exposure to the existing trusted bank;
it does not change its annotations or mine previously absent weak-signal examples.

Collection, live chains and evaluation use the negative and near-negative banks to
detect foreign contact along the actual committed polyline, including between decision
heads and on the origin-to-first-point connector. The first intersection with a
`.75`-voxel bank-path tube outside the intended annotation's `1.5`-voxel tube certifies
a switch (`--bank-switch-tolerance`, `--bank-own-tolerance`). Overlapping tubes remain
ambiguous; absent bank coverage cannot establish fiber identity. Contact positions are
continuous segment/capsule intersections, not rounded voxel or head locations. A
certified switch is terminal for the rest of its episode. Replay stores `switch_pos`,
`switch_distance`, `switch_bank_path` (shard/path index) and `switch_bank_run`; these
labels never enter model inputs. Replay stores every committed vertex once per trace,
with causal prefix indices for each decision; caches without the current schema are
rejected.

The loader samples no contrastive points. Lateral resampling uses visible forward bank
coverage directly; bank mining and crop support determine available foreign geometry.
Separate negative and continuation banks remain supported.

## Verification and diagnostics

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../../../../.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/fiber-slab-pytest

../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.evaluation.benchmark_slabs \
  --warmup 2 --repeats 10 --decisions 6 --out /tmp/slab-benchmark.json

# Match the launcher's batch, including batched attention:
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.evaluation.benchmark_slabs \
  --warmup 3 --repeats 10 --decisions 16 --microbatch 16 --out /tmp/slab-benchmark-b16.json
```

Use fresh output paths. `identity_preflight` checks real-data independent batches, and
`benchmark_feature_training` measures a bounded fork of a compatible run using its actual
data. The synthetic benchmark excludes I/O; compare equal hardware, precision and
supervised decision counts. `scripts/check_task_sampler.py` is the pre-launch sampler
check (delivered task shares, replay supply, startup ages and excursion geometry).

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
`0175A_SMALLMORE.afv` is excluded. Weights apply to batches; each source then applies
the task budget. Source counts and actual shares are logged per source.
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
sequence and targets. Mixed-source selection stays ordered.
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
of Paris 4 mined paths, so it cannot reenter through synthetic failures or foreign labels.

Each source supports its own synthetic failures and rollout replay. Collection
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
