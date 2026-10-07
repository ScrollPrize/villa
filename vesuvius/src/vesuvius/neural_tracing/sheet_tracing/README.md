# Sheet tracing

Architecture of the current best model (m7), with figures, detailed enough to rebuild it: [ARCHITECTURE.md](ARCHITECTURE.md).
Where the checkpoints are, how to run them, how to train on the H100, and which models are best: [OPERATIONS.md](OPERATIONS.md).

The tracer grows a surface from a seed curve on a papyrus sheet. Its *front* is an open 3D curve represented by
sampled points. At each step, one shared network examines CT near those points and jointly predicts a candidate
position and confidence for each. The predictions are combined into a new curve; successive curves form the
rows of a traced surface (`tifxyz`). The sampled positions are called *front points* throughout the code. The
number of them is configured by `data.front_points`, and their seed spacing by `data.point_spacing`.

## Main setup: Hermite curve and surface-token grid

This README describes `configs/hermite_grid_long.json`, the main training configuration. It sets
`trace.curve=hermite` and `model.surface_tokens=true`. The bare defaults in `config.py` are generic and use the
alternative smoothing mode; pass the Hermite grid config when starting a run.

The configured seed front has five control points 60 level-0 voxels apart and advances by `data.step=20` voxels per
step. CT crops come from pyramid level 1; positions and distances remain in level-0 voxels.
`trace.spline_samples=6` gives `(5 - 1) * 6 + 1 = 25` curve samples per output row. The number of control points
can change with the seed length at inference time; the network shares its weights across points.

One tracing step works as follows:

1. **Local evidence.** The current and previous fronts define a local frame at each control point: tangent
   along the front, advance direction, and sheet normal. The tracer samples a normalized, oriented 3D CT crop
   (`model.crop=[16,32,20]` in normal/tangent/advance order) at each point. A shared three-layer 3D CNN
   (`model.channels=[16,32,48]`) encodes the crops;
   each point token also receives its position and frame relative to the whole front, its place along the
   front, and its previous confidence.
2. **Curve and history.** Smaller CT crops (`model.token_crop=[8,8,8]`) are sampled at the 25 positions along
   the *current* curve. They form the current row of a token grid with up to `model.memory_rows=3` earlier rows
   of stored tokens, aligned by arc length. Each control point adds its crop embedding at its grid column.
   Two axial attention blocks mix information along each curve and along each column's history. Two more
   attention blocks let control-point tokens attend to one another and to the grid. These blocks use 96-wide
   embeddings and four attention heads. Earlier CT crops are not resampled at every step.
3. **Predictions.** A shared head predicts a 3D displacement relative to a straight 20-voxel advance in the
   local frame, a confidence logit, and a two-component tangent tilt along the advance and normal directions.
4. **New front.** Confident points move to their predictions. A low-confidence point is carried by the
   interpolated displacement of its nearest confident neighbours when available. Catmull-Rom chord tangents,
   adjusted by predicted tilts at confident points, define a cubic Hermite curve through the resulting points.
   Tangents are scaled by each gap's chord length. The 25 curve samples are the output row; the curve is then
   resampled uniformly by arc length to position the next front and its grid columns.

Confidence is trained to predict whether a *candidate point* is on the sheet containing the seed, using distance
to the ground-truth sheet below `loss.sheet_tolerance` as the label. It is not an accuracy score; the position
and curve losses handle geometric error. A low-confidence prediction is masked in the written `tifxyz` as `-1`.
In Hermite mode, a dense curve sample is kept only when both control points of its gap are confident. The masked
control point remains in the internal front and is predicted again on later steps, so it can become valid again;
evaluation calls this `rejoined`. Thus “stopped” in metric and code names means masked output, not halted motion.

The alternative `trace.curve=smoothing` uses more closely spaced front points (the generic defaults use 12 at
20 voxels). It fits a confidence-weighted smoothing B-spline to their predicted positions. Its smoothness
penalty acts on the *change* from the current front, preserving existing curvature; low-confidence points have
little influence on the fit. Its output columns are the front points rather than dense Hermite samples.

## Options added in the H100 experiments

| option | effect |
| --- | --- |
| `trace.stopped=curve` | Hermite mode: a low-confidence point with confident points on both sides is also placed on the Hermite curve through those neighbours. Carrying by displacement alone keeps any offset (e.g. a sheet away), so the point could never return to the sheet. This is used by the current best model, `big_curve_ft1`. The default `displacement` reproduces older checkpoints. |
| `model.context_crop` (+ `context_spacing`, `_channels`, `_strides`) | A coarse wide view per point (e.g. `[16,16,16]` at 6 vx, a 96 vx cube) through its own CNN, added via a zero-initialised projection. |
| `train.py --init CKPT` | Start from a checkpoint's weights with a fresh optimizer and schedule. Modules the checkpoint lacks (`context_*`) start fresh. `--resume` continues a run. |
| `train.max_minutes` | The cosine learning rate now anneals to zero at the time budget, not only at `iterations`. Resumed time counts. |
| `train.keep_every` | Also keep `ckpt_ITER.pt` periodically, for benchmarking intermediate checkpoints. |
| `train.compile_geometry` | `torch.compile` the GT projection: about 1.4× faster iterations on an H100, but not bit-identical (foot points differ by about 1e-3 vx typically; see `geometry.use_compiled_projection`). |
| `data.patches_dir` | One directory or a list (patches from all are combined). |
| `trace_benchmark.py --threshold/--volume` | Benchmark every model at one confidence threshold, or against a volume path other than the checkpoint's. |
| `train.live_chains` (+ `chain_steps`, `chain_on_sheet`, `chain_min_valid`, `chain_max_age`) | Live chains, like fiber_follow's live continuation (`live.py`). A training front is not restarted from GT after its 12-step rollout. It continues in a later iteration from exactly where the model left it (positions, curve, confidences, grid memory), on the next window of its patch, which loader workers read on request. Chains end after a random 24 to 72 steps, when under half their points end a rollout on the sheet, when the inference stop rule fires, or when the patch runs out. Survivors pool on the GPU until a full batch can be continued. Needs `data.tight_blocks`, and memory lags no longer than `data.steps`. Use with `train.free_fraction=1` so fronts are never snapped back to GT. |
| `train.ema` | Exponential moving average of the weights (e.g. 0.999 per iteration). Validation, `best.pt` and evaluation use the averaged weights. Checkpoints keep both (`model` raw, `ema` averaged), and `--init` starts from the averaged ones. `trace_benchmark.py --raw NAME` benchmarks a model's raw weights instead. |
| `hard_spots.py`, `data.hard_windows`, `data.hard_fraction` | `hard_spots.py` traces a model over many training-split seeds and records where each trace would stop or makes a real switch: a JSON of windows a few steps before the failure. With `data.hard_windows` set to that file, `hard_fraction` of training episodes start at a random recorded spot (jittered up to 2 steps along the advance and half a point spacing along the front). Validation is unaffected. Patch directories are referred to by their position in `data.patches_dir`, so spots mined on one machine work with another machine's paths. |
| `trace_benchmark.py --path-map OLD=NEW --out DIR` | Benchmark checkpoints trained on another machine: rewrite the data paths stored in their configs, and choose the output directory. GPUs differ slightly in bf16, so compare only models benchmarked on the same machine. |
| `model.residual`, `model.point_bottleneck` | Crop CNNs made of ResNet-D basic blocks (`vesuvius.models.build.resblocks.BasicBlockD`: two convs per stage, GroupNorm(4), GELU, a pooled and projected skip) instead of single convs. Crop sizes must be divisible by 2 per stride-2 stage, and channel counts by 4. `point_bottleneck` adds a 1x1 conv to that many channels before the point CNN's flattening Linear, which keeps that Linear small. |
| `model.joint_transformer` | One transformer (`layers` blocks, width `dim`, FFN `dim * mlp_ratio`) with full self-attention over the point and grid tokens together, plus learned point and line type embeddings. It replaces the axial grid blocks, the point blocks and `point_to_grid`. |
| `train.backward_per_step` | Backpropagate each rollout step's loss right after the step. This gives the same gradient as one backward pass over the summed loss, since the steps are detached, but only one step's activations are held in memory. Needs `train.detach_steps`. |
| `trace_benchmark.py --rescore DIR ...` | Recompute switch counts of existing benchmark folders from their tifxyz. Reports separate *real switches* (a row with at least 3 kept off-sheet points, or a kept point more than 9 vx off) from *any kept off-sheet point*. |
| `trace_spread.py` | Long traces (e.g. 120 steps) from held-out seeds spread across z, written as tifxyz for visual inspection. |
| `model.profile_crop` (+ `profile_spacing`, `_channels`, `_strides`) | A long, thin crop along the normal at every grid column (e.g. `[64,3,3]` at 2 vx, ±63 vx, several sheets either side) through its own CNN, added to the column's token via a zero-initialised projection. Shows the stack of neighbouring sheets that the small token crops cannot. |
| `model.memory_lags`, `model.memory_anchor` | Grid memory at these lags (e.g. `[1,2,3,6,12]`) instead of the last `memory_rows` rows, plus the trace's first row kept for good. Rows get two more geometry features (log lag, anchor flag). With `--init` from a model without them, the new inputs start at zero, so the model is unchanged while its memory holds at most `memory_rows` rows. |
| `train.history_rows` | Before each training rollout, trace 0..N GT rows up to the seed without loss (the front is put back on GT after each step), so the long memory starts out filled as it would be deep into a trace. Training episodes start N rows further back (about 6% less patch weight at N = 11). Validation episodes, benchmarks and inference are unchanged. |

## Data

- Training patches: `data.patches_dir` (verified tifxyz patches, level-0 `x, y, z`; `-1` marks invalid nodes).
- Volume: `data.volume` / `data.volume_level` (OME-Zarr v2 pyramid, read through fiber_follow's `ChunkedArray`).
  Level 1 of `s1_ds2.zarr` was decoded in place with
  `python -m vesuvius.neural_tracing.fiber_follow.data.decode_store --workers 8 --verify 256 .../s1_ds2.zarr/1`
  (uncompressed chunks are memory-mapped and shared through the page cache).
- An episode resamples one patch window onto a canonical `(u, v)` grid at `data.gt_resolution`
  (`u` is the advance direction, `v` runs along the front, both in voxels). There are eight orientations:
  advance along rows or columns, either sign, front either sign. The episode also loads a `data.roi`³ volume
  block that the whole rollout stays inside.
- Train/val split: `data.split_block`³ cells assigned by a seeded hash. A training or validation episode is used
  only if all corners of its block lie in cells of its split, so overlapping patches never straddle the split.
  Trace-benchmark seeds instead require their traced region to lie in val cells.
- Patch index: cached in `output/cache/patch_index_*.json` (rebuilt when the patch directory listing changes).

## Training

The run directory is `output/NAME`:

```bash
cd vesuvius/src/vesuvius/neural_tracing/sheet_tracing
../../../../.venv/bin/python -m vesuvius.neural_tracing.sheet_tracing.train --config hermite_grid_long.json --name RUN
# quick (~12 min) variants
../../../../.venv/bin/python -m vesuvius.neural_tracing.sheet_tracing.train --config tiny_hermite_grid.json --name RUN
# resume
../../../../.venv/bin/python -m vesuvius.neural_tracing.sheet_tracing.train --resume output/RUN/last.pt
```

What one iteration does:

- Seed the front from GT rows with noise.
- Roll it out for `data.steps` steps using the model's own predictions, adding noise between steps.
- Generate off-sheet cases, so confidence sees real negatives:
  - Snapped episodes move front points farther than `train.reset_distance` back onto the sheet.
  - Free episodes (`train.free_fraction`) are never snapped, so front points can drift onto other sheets.
  - In `train.jump_probability` of episodes, a random run of front points is pushed along the normal onto the
    brightest CT ridge between `train.jump_min` and `train.jump_max` voxels away: the neighbouring sheet, not the gap.
- Average the per-step losses:
  - Prediction error versus the GT point one step ahead, split into normal, advance and tangent components. With
    `loss.column_target` the target sits at the front point's seed column, which penalizes the front shrinking or
    drifting sideways. Only points currently on the seed's sheet get this position loss.
  - Distance of the dense curve samples to the sheet, for gaps whose two control points are on the sheet (so a
    segment bowing onto a neighbouring sheet is penalized).
  - Confidence BCE with the label "the prediction is on the seed's sheet", class-balanced with
    `loss.confidence_balance` (off-sheet points are rare).
- Patch nodes that are invalid or outside the patch carry no loss (a patch edge is not a sheet edge).
- Rollouts are detached between steps by default (`train.detach_steps=false` backpropagates through all of them).

Outputs in the run directory:

- `config.json`, `log.jsonl` and `training.png`.
- `val_ITER/`: episode figures, metric curves and `tifxyz/` (trace + GT for 8 validation episodes).
- `best.pt` (highest validation `score` = coverage − kept_off) and `last.pt`.
- After the run (`eval.after_training`), `eval/`: short free rollouts of `best.pt` and of the straight-advance
  baseline on the same held-out episodes (defaults: 32 episodes, 10 steps; a few seconds), plus figures and
  `tifxyz_eval/` / `tifxyz_baseline_straight/` (`epNN_trace`, `epNN_gt` for every episode).

`train.max_minutes` sets a time budget. Stop a run with Ctrl-C or SIGTERM. The trainer exits through Python so
the DataLoader shuts down its forkserver workers.

## Evaluation, trace benchmark and inference

```bash
python -m vesuvius.neural_tracing.sheet_tracing.evaluate --checkpoint output/RUN/best.pt [--steps 30 --roi 1024 --episodes 256]
python -m vesuvius.neural_tracing.sheet_tracing.trace_benchmark --checkpoints output/A/best.pt output/B/best.pt \
    --seeds 40 --steps 30 --tag a_vs_b
python -m vesuvius.neural_tracing.sheet_tracing.infer --checkpoint output/RUN/best.pt \
    --seed-patch /path/to/patch --line row:40 --range 100:160 --direction + --steps 40 --out output/RUN/traces/t0
```

The trace benchmark runs long held-out traces through the inference path. Every checkpoint gets identical seed
polylines, and the volume is read around the front each step. It takes about 30 s for 40 seeds × 30 steps × 3
models. Output in `output/trace_bench/TAG`: `report.json`, `per_step.png`, `seed_NN.png`, and `tifxyz/`
(`seedNN_gt` plus `seedNN_MODEL` for each model). Per trace it also reports `first_switch` (the first step with a
kept point off the sheet) and `lost` (the first step with under half the points on the sheet).

The evaluation metrics are per step, over points with GT supervision:

| metric | meaning |
| --- | --- |
| `same_sheet` | within `loss.sheet_tolerance` of the seed's sheet |
| `confident` / `kept` | above the confidence threshold |
| `coverage` | same sheet and confident (what a user would keep) |
| `precision` | same sheet among confident points (low means confidently on another sheet) |
| `caught` | not confident among points that are off the sheet (sheet switches that were flagged) |
| `kept_off` | confident and off the sheet, over all supervised points |
| `score` | coverage − kept_off (checkpoint selection) |
| `rejoined` | coverage among front point indices that were below the confidence threshold at an earlier step |
| `curve_same_sheet` | same sheet among dense curve samples whose gap has two confident control points |
| `dist_p50`, `dist_p90`, `advance_p90` | on-sheet distance and absolute advance error of kept points |

Inference: the number of front points follows from the seed length; the model doesn't depend on it. The output is a
tifxyz (rows are steps; columns are Hermite curve samples, or front points in smoothing mode; masked points are
`-1`) plus `confidence.npy` and `trace.json`.

Figures: the left panel is the sheet map (GT `u/v`; `o` = confident, `x` = masked, coloured by on-sheet
distance). The middle panel is the CT slice spanned by the normal and the advance direction through the middle
front point, with the GT sheet in green and the trace in red. The right panel is the CT slice across the final front,
with the curve.

## Scaling up

Everything is in the config (see `config.py` for every field and its default). Overrides take the form
`--set section.field=value`, with JSON values: `--set model.crop=[24,16,24]`. `configs/base.json` sketches a
larger smoothing-mode setup at level 0. Level 0 must be decoded in place first (about 349 GB uncompressed).

## Code

| module | contents |
| --- | --- |
| `config.py` | dataclass config sections, JSON load, `--set` overrides |
| `data/patches.py` | patch index, memory-mapped tifxyz, canonical GT windows |
| `data/episodes.py` | episode rejection sampling, spatial split, loader stream |
| `geometry.py` | GT grid evaluation and projection, B-spline and Hermite curves, carrying low-confidence points, respacing, frames |
| `sampling.py` | oriented GPU crops from a volume block |
| `model.py` | crop encoders, surface-token grid with axial attention, front-point attention blocks, head; baseline |
| `rollout.py` | one tracing step and front update (shared by train/eval/infer) |
| `losses.py`, `seeding.py` | supervision, metrics, seeds, training perturbation and sheet jumps |
| `train.py`, `evaluate.py`, `trace_benchmark.py`, `infer.py`, `visualize.py` | entry points and figures |

Run the tests with `python -m pytest vesuvius/src/vesuvius/neural_tracing/sheet_tracing/tests`. They use analytic
surfaces and don't need data files.
