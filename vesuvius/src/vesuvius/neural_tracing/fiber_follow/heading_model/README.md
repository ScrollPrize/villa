# Crop-heading model

A tiny network that predicts the crop heading from what the tracer has at each decision: the current CT
around the head, plus the observed path (none at a seed). The target is the heading that keeps the upcoming
fiber as centered as possible in the follower's crop. It is a forward direction along the fiber, and it
maximizes the margin to the crop walls. It needs no structure tensors.

## What it predicts

The target is `targets.in_crop_heading`. Take the next `forward` voxels of annotated fiber from the actual
head; `forward` defaults to the follower crop's forward extent, 35.5 voxels for
`CropSpec(120, 104, behind=48, spacing=.5)`. The target is the axis through the head that minimizes the
largest lateral distance of those points from the axis. During tracing the head is off GT, so the target also
includes the direction back to the fiber.

The network sees two inputs, both in a frame whose +z is the tracer's prior heading:

- a CT patch of 32³ samples at 1.25 voxels, reaching 5 voxels behind and 34 ahead, ±19 lateral. It uses the
  follower's crop sampler and CT normalization.
- the last 32 voxels of the observed path at unit arclength, with a validity mask.

The prior heading is the tracer's trailing 12-voxel fit, or the held seed heading on shorter paths.

The patch is sampled with random roll during training, so inference can use any roll and no CT-derived roll
is needed. The output is a residual on the prior; an untrained network returns its prior.

The encoder halves the patch four times (32 → 16 → 8 → 4 → 2) with stride-2 convolutions whose kernels are 4
wide, padding 1. Each output covers inputs 2o−1 to 2o+2, so every grid stays centered on the patch and each input
feeds exactly two outputs per axis. The final 2×2×2 cells (64 features each) are mirror-image octants; patch
depth and width must be divisible by 16. This is architecture `crop_heading_ct_path_v2`. The model has 136k
parameters at width 8 and 473k at width 16. One CPU thread runs 24 heads in about 6 ms at width 8 and 15 ms at
width 16, without patch sampling.

v1 used 3-wide kernels, which centered every output on an even input. On each axis the low final cell then sat
on the patch edge and the high cell saw nearly the whole patch, so one cell carried most of the decision.
`scripts/convert_heading_v1.py SOURCE DEST` converts a v1 checkpoint exactly (a 3-wide kernel is a 4-wide one
with a zero last tap). It checks the outputs against the v1 computation before writing; the result evaluates
and predicts but does not resume. `output/heading_model_l0_w16/best_v2.pt` is the converted step-70000 model.

## Training data: the follower's own code

`data.py` builds every state from the follower's modules:

- sources, fiber splits and CT volumes come from the follower dataset config, through
  `data.datasets.load_primary_dataset`, `open_afv_source` and `primary_source_spec`;
- CT normalization comes from `data.ct_normalization.prepare_normalization`. Pass a follower run's
  `ct_normalization.json` to reuse its exact records;
- the simulated tracer states come from `data.data.make_sample`: the startup mix, the calibrated tracing
  error and the head offset;
- the prior is the tracer's own heading rule, `shared.heading.linear12_heading`.

A `seed_probability` share of states are seeds: an annotated point with no path. Seeds and paths shorter than
12 voxels hold a seed heading that the model must not depend on, so their prior is the true continuation
tilted by |N(0, 25°)|, capped at 75°. That forces the model to find the fiber from CT. A
`long_cone_probability` share of long paths gets the same cone prior, for robustness.
`sampling.sample_config` can override the follower's `SampleConfig` defaults. Every batch comes from one
source volume, chosen by its dataset-config weight.

## Train

```bash
cd vesuvius/src/vesuvius/neural_tracing/fiber_follow
python heading_model/train.py            # uses configs/heading_model.json
# --config PATH (relative to cwd, fiber_follow/, vesuvius/src/ or configs/), --resume continues last.pt,
# --name/--steps/--workers/--device override the config. Also runnable as
# python -m vesuvius.neural_tracing.fiber_follow.heading_model.train from anywhere.
```

`configs/heading_model.json` trains on the sources in `configs/mixed_ct_datasets_paris50.json`: Paris 4,
`0175A_5mm_v1.afv` and `1447_5mm_v1.afv` at 0.5/0.25/0.25. It reuses the CT normalization of
`mixed_ct_afv_stem32_run3_pathgeom_paris50` and runs 30k steps of batch 128.

`init_checkpoint` starts a new run from another checkpoint's weights, which must have the same architecture and
model config; the optimizer starts fresh, and `--resume` ignores it. `lr_schedule: "constant"` holds `lr` after
the warmup instead of the default cosine decay to zero. `configs/heading_model_l0_w16_centered.json` uses both:
20k steps at a constant 1.5e-4 (the step-60k rate of `heading_model_l0_w16`), starting from the converted
`best_v2.pt`.

For cosine continuation, `lr_decay_start` sets the absolute step where decay begins (defaults to the end of
warmup). The frame config now continues from 40k to 70k: `lr: 0.00015`, `lr_schedule: "cosine"`,
`lr_decay_start: 40000`, `steps: 70000`. Resume with:

```bash
OMP_NUM_THREADS=1 python heading_model/train.py --config configs/heading_model_l0_w16_centered_frame.json --resume
```

This restores model and optimizer state from `last.pt`; the first update at 40k uses 0.00015, the rate is
0.000075 at 55k, and the cosine reaches zero at 70k (the final update at 69,999 uses approximately 4.1e-13).
Keep the same decay start when resuming after an interruption so the schedule continues without restarting.

The run directory `output/<name>/` holds:

- `config.json`, with the resolved config and provenance;
- `ct_normalization.json` and `log.jsonl`;
- `last.pt`, resumable;
- `ckpt_STEP.pt` checkpoints;
- `best.pt`, the lowest held-out p90 off-axis distance over the crop extent.

`model.ct_downsample_levels: 1` (the default config, run `heading_model_v2`) samples the patch from the CT pyramid
level above the follower's: a 2x block mean, with a half-voxel shift so the patch lands where the follower's CT would.
The patch spacing is 2.5 native voxels, so level 1 loses little, and it reads 8x fewer bytes. Level-0 reads were disk
bound: each memory-mapped page fault pulled in a whole 2 MB chunk, 8-11 MB of reads per sample. Build level 1 locally
from the cached level-0 chunks once; it is bit-identical to the remote pyramid, takes about 2 minutes and needs no
download:

```bash
python scripts/downsample_ct_cache.py --dataset-config configs/mixed_ct_datasets_paris50.json
```

Chunks whose level-0 children are not all cached are fetched by the remote prefetcher. The notes below describe the
level-0 loader.

The loader is CT-read bound and the GPU is mostly idle. One worker builds a 128-state batch in about 0.5 s for
Paris 4 (local), about 1 s for 0175A and about 2.7 s for 1447 (S3 via the local chunk cache). Ten workers give
roughly 1,000 samples/s, so the default run takes about an hour. More `workers` or a larger `worker_cache_gb`
speeds it up.

## Evaluate

Validation runs on each source's held-out fibers and scores alignment only. Results are split by history:
seed, 1–11, 12–31 and 32+ voxels. For each bin it reports:

- the angle to the target, for the prior and the model;
- the share of states where the model is closer;
- the largest lateral distance of the fiber from the crop axis, over the 16-voxel forecast and over the
  crop extent, for the prior, the model and the target's floor.

```bash
python heading_model/evaluate.py --checkpoint output/heading_model_v1/best.pt --out report.json
```

## Use

```python
from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingPredictor
predictor = HeadingPredictor.load('output/heading_model_v1/best.pt', device='cpu')
headings = predictor.predict(vol, positions, prior_headings, observed_paths, pool)  # signed along the priors
```

The follower does not use it yet. To adopt it, call it wherever the tracer builds a crop frame. Training must
call it in the same place: the simulated traces, synthetic switches and pairs, and live chains. Otherwise the
follower sees crops oriented differently in training and in tracing.

## H/V-conditioned crop frame

`model.predict_frames: true` (with `predict_normals: true`) adds explicit H/V conditioning and returns a full
right-handed frame with columns `(u, v, forward)` through `model.forward_outputs(..., family)['frame']`.
Network family IDs are H=0, V=1; the predictor API accepts H/V strings. Both heads use a learned family
embedding added to the existing path features. The embedding starts at zero, preserving the source model's
heading and normal predictions exactly for either family.

For this model, normal labels use the 2/8 CT tensor **projected perpendicular to a smoothed annotated fiber
tangent**. The tangent is a least-squares line fit to 25 equally spaced samples over ±6 trace voxels around
the annotation position used for the heading target; the window clips at fiber ends. It uses annotations
only to construct labels, never as model input. Nearly parallel tensor/tangent pairs are rejected (remaining
normal length <0.1); other labels are downweighted by the squared remaining normal length. Fiber family comes
from the annotation's H/V tag. There is no extra CT read or sampling pass.

The forward-looking crop-heading target is retained. Frame construction projects the predicted normal
perpendicular to the predicted heading and obtains the other transverse axis with a cross product. This
projection is distinct from the training-label correction against the local fiber tangent. The loss keeps
heading and corrected-normal supervision and adds `roll_loss_weight: 0.25`: unsigned roll alignment with
both normals projected around the target heading. Heading error cannot reduce this roll term. Flat CT and
ill-defined target roll receive zero roll weight. Valid orthonormal frames are returned even for degenerate
predictions; inference uses the previous frame for sign continuity and a reproducible sign at a seed.

The completed step-20,000 normal model has been converted into
`output/heading_model_l0_w16_centered_frame/last.pt`, including its AdamW moments and training step. Only
the new family embedding starts without optimizer history. The new config continues for another 20,000
steps (40,000 total), using the same source mix, learning rate, batch size and CT sampling. Run:

```bash
OMP_NUM_THREADS=1 python heading_model/train.py \
  --config configs/heading_model_l0_w16_centered_frame.json --resume
```

To convert another completed run, `python scripts/convert_heading_frame.py SOURCE/last.pt NEW_RUN/last.pt`
creates an exclusive destination and verifies exact heading/normal equality. Best scores start fresh in the
new run. `best_frame.pt` selects the lowest mean held-out frame rotation error (averaged over source/history
bins), allowing the equivalent simultaneous sign flip of the two transverse axes. Logs also show isolated
roll error, corrected-normal error and the existing heading metrics. `last.pt` resumes all heads and optimizer.

```python
predictor = HeadingPredictor.load('output/heading_model_l0_w16_centered_frame/best_frame.pt', device='cuda')
frames = predictor.predict_frames(vol, positions, prior_headings, observed_paths,
                                  families=['H', 'V'], previous=previous_frames)
```

Each previous frame may be `None` for a seed. This API is ready for integration; the follower's tracing loop
still uses its existing frame generator. No predicted tangent or annotated fiber information is required at inference.

Verification: the frame, heading, tensor and crop regression suite passes (47 tests), including CPU
train/save/resume and preservation of AdamW state. Conversion also preserved both predictions bit-for-bit on
192 cached real CT inputs (64 each from Paris 4, 0175A and 1447, empty seed paths, alternating H/V).
Five warmed single-thread CPU inference repetitions measured 0.733 ms/sample before and 0.723 after
(approximately unchanged); this excludes CT sampling and training. Detailed timing is recorded in
`output/heading_frame_conversion_check.json`. Run the focused tests with:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_heading_frames.py tests/test_heading_model.py tests/test_heading_normals.py
```

## Joint heading and sheet-normal training

`model.predict_normals: true` adds a separate normal head to the shared CT/path encoder (architecture
`crop_heading_normal_ct_path_v3`). Heading supervision stays unchanged. The extra loss is
`normal_loss_weight * (1 - dot(predicted_normal, target_normal)^2)`, confidence-weighted by the tensor's
relative largest eigengap. The default loss weight is 0.25. Flat CT and eigengaps below 0.05 receive zero
normal weight but still train the heading. Normals are unsigned axes; there is no direct signed-roll target
or constraint forcing the normal perpendicular to the crop heading.

Labels are generated on demand in loader workers. One oriented CT sample encloses both the original heading
patch and the normal stencil. For the current level-0 Paris 4 and AFV volumes this is **44x34x34**, with the
head at depth 16; slicing `[12:44, 1:33, 1:33]` recovers the original **32x32x32** input. The tensor uses a head-centered
stencil inside that sample. The common grid has spacing 1.25 trace voxels = 2.5 selected-CT voxels. Derivative
and integration sigmas are **2 and 8 selected-CT voxels** (0.8 and 3.2 grid samples), with context radius at least
38 CT voxels. This is a tensor on resampled CT, not native CT. No intensity threshold is applied. Prefetch
bounds include the enlarged sample; CT is read and interpolated once. Z-score normalization happens **after**
the input is sliced, so enlarging the label context does not change input normalization. Other CT levels use
the same sigmas in that selected level's voxel units.

Convert a heading-only v2 checkpoint without changing its heading weights:

```bash
python scripts/convert_heading_normals.py output/heading_model_l0_w16_centered/best.pt \
  output/heading_model_l0_w16_centered/best_with_normals_v3.pt
python heading_model/train.py --config configs/heading_model_l0_w16_centered_normals.json
```

Conversion verifies exact heading-output equality, initializes only the normal branch, and saves the source
path/step and initialization seed. It refuses to overwrite a checkpoint. The supplied new run uses the
converted step-4000 best checkpoint, a fresh optimizer, constant learning rate 1.5e-4, batch 128, 16 workers,
and 20,000 steps. Paris 4 / 0175A / 1447 sampling remains 50% / 25% / 25%.

`best.pt` keeps its original heading-only selection criterion. **`best_normal.pt`** selects the lowest
held-out confidence-weighted unsigned normal loss, averaged over source/history bins with valid labels.
`last.pt` resumes both heads and optimizer. Logs include heading loss, normal loss, valid-label fraction,
and held-out unsigned normal angle p50/p90/p99 per source/history bin. Assess both heading and normal metrics
when choosing a joint checkpoint.

The default supervision was upgraded from sigmas 1/4 (radius 19) to 2/8 (radius 38). Stop the existing
process, then use the same config with `--resume`:

```bash
OMP_NUM_THREADS=1 python heading_model/train.py --config configs/heading_model_l0_w16_centered_normals.json --resume
```

This known policy transition preserves weights, optimizer, learning-rate schedule position, step and the
heading-best score from `last.pt`. Training workers and fixed validation labels are rebuilt with 2/8.
The normal-best score resets because it is measured against different labels. The old `best_normal.pt`
is retained as `best_normal_sigma1_4_before_step_STEP.pt`; new normal-best selection starts at the next
validation. The transition is recorded in `log.jsonl`, run config, and subsequent checkpoints. Unrelated
policy or normal-loss-weight changes still fail resume validation. Later 2/8 resumes retain their normal-best
score. Resume continues from the last saved checkpoint, not unsaved steps at interruption; `--steps` is the
total target step count and can be increased if the run has already reached it.

Inference still uses only the original patch and path. `HeadingPredictor.predict_with_normals(...)` returns
`(world_headings, world_normal_axes)`; `predict(...)` retains the existing heading-only API. No tensor or extra
context is computed at inference. Adopting predicted normals for the follower's frame remains a separate change.

## Earlier prototype (Paris 4, one model, `output/trace_sampling_20261002/`)

On held-out Paris 4, the angle to the target (p50/p90) went from 11.3/26.8° to 4.7/9.5° with 32+ voxels of
history. It went from 18/44° to 6.6/15° with 1–11 voxels. At seeds it went from 14–18° to 10–11°, with no gain
at p99. The fiber's off-axis distance over the crop extent fell from p50 6.8 to 3.4 voxels; the floor is 1.7.
