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
is needed. The output is a residual on the prior; an untrained network returns its prior. The model has 74k
parameters and costs about 0.7 ms per 24 heads on CPU, about 10 ms with patch sampling.

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

## Earlier prototype (Paris 4, one model, `output/trace_sampling_20261002/`)

On held-out Paris 4, the angle to the target (p50/p90) went from 11.3/26.8° to 4.7/9.5° with 32+ voxels of
history. It went from 18/44° to 6.6/15° with 1–11 voxels. At seeds it went from 14–18° to 10–11°, with no gain
at p99. The fiber's off-axis distance over the crop extent fell from p50 6.8 to 3.4 voxels; the floor is 1.7.
