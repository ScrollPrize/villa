# Crop-heading and frame model

A small network that predicts the follower's crop frame from what the tracer has at each decision: the current CT
around the head, the observed path (none at a seed) and the fiber's H/V family. The follower uses it for every crop
it builds in training, collection and tracing (`tracing/crop_frames.py`); new runs use the frozen
`output/heading_model_l0_w16_centered_frame/ckpt_064000.pt` (`DEFAULT_FRAME_CHECKPOINT`).

## What it predicts

The heading target is `targets.in_crop_heading`. Take the next `forward` voxels of annotated fiber from the actual
head (35.5 voxels by default, the follower crop's forward extent). The target is the axis through the head that
minimizes the largest lateral distance of those points from the axis. During tracing the head is off GT, so the
target also includes the direction back to the fiber.

The network sees two inputs, both in a frame whose +z is the tracer's prior heading:

- a CT patch of 32³ samples at 1.25 voxels, reaching 5 voxels behind and 34 ahead, ±19 lateral. It uses the
  follower's crop sampler and CT normalization.
- the last 32 voxels of the observed path at unit arclength, with a validity mask.

The prior heading is the tracer's trailing 12-voxel fit, or the held seed heading on shorter paths. The patch is
sampled with random roll during training, so inference can use any roll. The heading output is a residual on the
prior; an untrained network returns its prior.

The encoder halves the patch four times (32 → 16 → 8 → 4 → 2) with stride-2 convolutions whose kernels are 4 wide,
padding 1. Each output covers inputs 2o−1 to 2o+2, so every grid stays centered on the patch and each input feeds
exactly two outputs per axis. The final 2×2×2 cells (64 features each) are mirror-image octants; patch depth and
width must be divisible by 16.

### Sheet normal and H/V-conditioned frame

`model.predict_normals: true` adds a normal head to the shared encoder, and `model.predict_frames: true` (which
requires it) adds H/V conditioning and returns a full right-handed frame with columns `(u, v, forward)` through
`model.forward_outputs(..., family)['frame']` (architecture `crop_frame_ct_path_family_v4`). Network family IDs
are H=0, V=1; the predictor API accepts H/V strings. Both heads add a learned family embedding to the path
features.

Normal labels come from a CT structure tensor with derivative and integration sigmas of **2 and 8 selected-CT
voxels** (context radius at least 38 CT voxels), computed on the oriented training sample: one CT read encloses
both the 32³ input and the tensor stencil (44×34×34 at level 0; slicing `[12:44, 1:33, 1:33]` recovers the input),
and z-score normalization happens after the input is sliced. The tensor normal is **projected perpendicular to a
smoothed annotated fiber tangent** (a least-squares line through 25 samples over ±6 trace voxels). Nearly parallel
tensor/tangent pairs are rejected (remaining normal length <0.1); other labels are weighted by the squared remaining
length. Flat CT and eigengaps below 0.05 give zero normal weight. Annotations only construct labels; they are never
inputs.

Frame construction projects the predicted normal perpendicular to the predicted heading and obtains the other
transverse axis with a cross product. The loss is the heading loss, `normal_loss_weight * (1 - dot²)` on the
unsigned normal, and `roll_loss_weight` times an unsigned roll term with both normals projected around the target
heading (heading error cannot reduce it). Valid orthonormal frames are returned even for degenerate predictions;
inference uses the previous frame for sign continuity and a reproducible sign at a seed.

## Training data: the follower's own code

`data.py` builds every state from the follower's modules:

- sources, fiber splits and CT volumes come from the follower dataset config, through
  `data.datasets.load_primary_dataset`, `open_afv_source` and `primary_source_spec`;
- CT normalization comes from `data.ct_normalization.prepare_normalization`. Pass a follower run's
  `ct_normalization.json` to reuse its exact records;
- the simulated tracer states come from `data.data.make_sample`: the startup mix, the calibrated tracing
  error and the head offset;
- the prior is the tracer's own heading rule, `tracing.heading.linear12_heading`.

A `seed_probability` share of states are seeds: an annotated point with no path. Seeds and paths shorter than
12 voxels hold a seed heading that the model must not depend on, so their prior is the true continuation
tilted by |N(0, 25°)|, capped at 75°. That forces the model to find the fiber from CT. A
`long_cone_probability` share of long paths gets the same cone prior, for robustness.
`sampling.sample_config` can override the follower's `SampleConfig` defaults. Every batch comes from one
source volume, chosen by its dataset-config weight.

## Train

```bash
cd vesuvius/src/vesuvius/neural_tracing/fiber_follow
python heading_model/train.py            # uses configs/heading_model_l0_w16_centered_frame.json
# --config PATH (relative to cwd, fiber_follow/, vesuvius/src/ or configs/), --resume continues last.pt,
# --name/--steps/--workers/--device override the config. Also runnable as
# python -m vesuvius.neural_tracing.fiber_follow.heading_model.train from anywhere.
```

`configs/heading_model_l0_w16_centered_frame.json` is the frame run's configuration: width 16, the sources of
`configs/mixed_ct_datasets_paris50.json` (Paris 4, `0175A_5mm_v1.afv` and `1447_5mm_v1.afv` at 0.5/0.25/0.25), batch
128, `lr: 0.00015` with a cosine decay from step 40,000 (`lr_decay_start`) to zero at 70,000. `init_checkpoint`
starts a new run from another checkpoint's weights (same architecture and model config; fresh optimizer; `--resume`
ignores it); `lr_schedule: "constant"` holds `lr` after the warmup. A resume requires the same model config, normal
target policy and loss weights.

The run directory `output/<name>/` holds `config.json` (the resolved config and provenance), `ct_normalization.json`,
`log.jsonl`, `last.pt` (resumable), `ckpt_STEP.pt`, and `best.pt`, `best_normal.pt` and `best_frame.pt`: the lowest
held-out p90 off-axis distance over the crop extent, confidence-weighted normal loss and mean frame rotation error.

The frame model descends from heading-only (`crop_heading_ct_path_v2`: `output/heading_model_l0_w16`,
`output/heading_model_l0_w16_centered`) and heading-and-normal (`crop_heading_normal_ct_path_v3`:
`output/heading_model_l0_w16_centered_normals`) runs; each run directory's `config.json` records how it was trained.

The loader is CT-read bound and the GPU is mostly idle; more `workers` or a larger `worker_cache_gb` speed it up.

## Evaluate

Validation runs on each source's held-out fibers. Results are split by history: seed, 1–11, 12–31 and 32+ voxels.
For each bin it reports the angle to the target for the prior and the model, the share of states where the model is
closer, the largest lateral distance of the fiber from the crop axis (over the 16-voxel forecast and the crop extent)
for the prior, the model and the target's floor, and the normal and roll errors.

```bash
python heading_model/evaluate.py --checkpoint output/heading_model_l0_w16_centered_frame/ckpt_064000.pt --out report.json
```

## Use

```python
from vesuvius.neural_tracing.fiber_follow.heading_model.model import HeadingPredictor
predictor = HeadingPredictor.load('output/heading_model_l0_w16_centered_frame/ckpt_064000.pt', device='cpu')
frames = predictor.predict_frames(vol, positions, prior_headings, observed_paths,
                                  families=['H', 'V'], previous=previous_frames)
```

Each previous frame may be `None` for a seed. `predict_with_normals(...)` returns `(world_headings,
world_normal_axes)`. No annotated fiber information is used at inference. `tests/test_heading_model.py` covers the
targets and CPU loader training, validation, checkpointing and resume.
