# Sheet tracing

* The model, with figures, detailed enough to rebuild it: [ARCHITECTURE.md](ARCHITECTURE.md).
* Where the checkpoints are, how to run them and how to train on the H100: [OPERATIONS.md](OPERATIONS.md).

The tracer grows a surface from a seed curve on a papyrus sheet. Its *front* is an open 3D curve represented by
sampled points. At each step, one shared network examines the CT near those points and jointly predicts a candidate
position and confidence for each. The predictions are combined into a new curve, and successive curves form the rows
of a traced surface (`tifxyz`). The sampled positions are called *front points* throughout the code. Their seed
spacing is `data.point_spacing` (60 vx). Their number follows from the seed length at inference, and from
`data.front_points_range` in training.

## Supported models

Two models are supported, sharing one architecture:

* `configs/remote_l0_points_ft.json`: the sheet tracer;
* `configs/remote_l0_fiber_ft.json`: initialised from `points_ft`, it adds fiber tokens and a fiber-following mode.

The defaults in `config.py` are this architecture. A config or checkpoint that sets an option of a removed model
type (`config.REMOVED`, applied by `shared/retired.py`) to anything but the value these models use is rejected with
"... no longer exists (only ... is supported)".

One tracing step works as follows:

1. **Local evidence.** The current and previous fronts define a local frame at each control point: tangent along the
   front, advance direction and sheet normal.
   * At each point the tracer samples a z-scored, oriented level-0 CT crop (`model.crop=[24,48,32]` in
     normal/tangent/advance order, at 1 vx, centred 10 vx ahead).
   * A residual 3D CNN encodes each crop.
   * Each point token also receives its position and frame relative to the whole front, its place along the front
     and its previous confidence.
2. **Curve and history.** Small CT crops (`model.token_crop=[16,16,16]`) are sampled at the curve columns of the
   *current* curve, 6 per gap. They form the current row of a token grid, together with the stored tokens of the
   last `model.memory_rows=3` rows, aligned by arc length. Earlier CT crops are not resampled at every step.
3. **Joint attention.** One transformer (8 blocks, width 256, 8 heads) attends over the point tokens and the grid
   tokens together. With `model.fiber_tokens`, the fibers near the front join as a third kind of token.
4. **Predictions.** A shared head predicts:
   * a 3D displacement relative to a straight 20-voxel advance in the local frame;
   * a confidence logit;
   * a two-component tangent tilt along the advance and normal directions.
5. **New front.**
   * Confident points move to their predictions.
   * A low-confidence point is carried by the interpolated displacement of its nearest confident neighbours. If it
     has confident points on both sides, it is placed on the Hermite curve through them, so it can land back on
     the sheet.
   * Catmull-Rom chord tangents, adjusted by the predicted tilts at confident points, define a cubic Hermite curve
     through the resulting points. Its samples are the output row.
   * The curve is then resampled uniformly by arc length to position the next front and its grid columns.

**Confidence.** Confidence is trained to predict whether a *candidate point* is on the sheet containing the seed.
The label is distance to the ground-truth sheet below `loss.sheet_tolerance`. It is not an accuracy score: the
position and curve losses handle geometric error.

**Masked points.** A low-confidence prediction is written to the `tifxyz` as `-1`, and a dense curve sample is kept
only when both control points of its gap are confident. A masked control point stays in the internal front and is
predicted again on later steps, so it can become valid again; evaluation calls this `rejoined`. "Stopped" in metric
and code names therefore means masked output, not halted motion.

## Options

| option | effect |
| --- | --- |
| `data.patches_dir`, `data.patch_volumes`, `data.dir_weights` | One patch directory or a list of them, each with the CT volume its patches' coordinates are in (empty: `data.volume`; e.g. other scrolls prepared with `scripts/prepare_scroll.py`) and its share of the sampling weight. |
| `data.front_points_range`, `train.points_per_batch` | Training only: every batch draws its front size N uniformly from `[min, max]`, holding about `points_per_batch` points (`batch = points_per_batch / N`). `data.front_points` still sets validation. |
| `data.unpadded_blocks` | Keep each training episode's block unpadded on the GPU (a list), instead of padding them to the batch's largest extent, which long fronts at mixed orientations blow up. |
| `data.fiber_runs`, `fiber_fraction`, `fiber_min_valid` | Fiber-seeded episodes: this fraction of episodes starts from a fiber matched to its patch (`scripts/match_fibers_to_patches.py`). The fiber is the seed row, and the GT window is rotated to follow it. |
| `data.follow_fraction`, `model.fiber_follow_mode` | Fiber-following batches: a 3-point front across a matched fiber, with the middle point on it, advancing along it. The model gets a learned mode embedding (0 sheet tracing, 1 following), zero-initialised. |
| `data.fiber_pool`, `model.fiber_tokens`, `data.fiber_cut_at_start`, `fiber_max_instances`, `fiber_max_points` | In fiber-started episodes, the fibers around the episode (`scripts/build_fiber_pool.py`) as extra transformer tokens: the `fiber_tokens` points nearest the front each step, each with a random per-episode instance code shared by all points of one fiber. In following episodes the seed fiber is cut ahead of the start, so the model learns to extend it. |
| `train/train.py --init CKPT` | Start from a checkpoint's weights (EMA if present) with a fresh optimizer and schedule. The fiber and mode modules start fresh if the checkpoint lacks them. `--resume` continues a run. |
| `train.ema` | Exponential moving average of the weights (e.g. 0.999). Validation, `best.pt`, evaluation and `--init` use the averaged weights. Checkpoints keep both (`model` raw, `ema` averaged); `evaluation/benchmark.py --raw NAME` benchmarks the raw ones. |
| `train.max_minutes` | The cosine learning rate anneals to zero at the time budget. Resumed time counts. |
| `train.keep_every` | Also keep `ckpt_ITER.pt` periodically, for benchmarking intermediate checkpoints. |
| `train.compile_geometry` | `torch.compile` the GT projection: about 1.4x faster, but not bit-identical (foot points differ by about 1e-3 vx). |
| `train.compile_model` | `torch.compile` the model: about 1.7x faster at level 0, not bit-identical. Checkpoints are unchanged. |
| `train.live_chains` (+ `chain_steps`, `chain_on_sheet`, `chain_min_valid`, `chain_max_age`) | Live chains (`train/live.py`): a training front is not restarted from GT after its rollout, but continues in a later iteration from exactly where the model left it, on the next window of its patch. Needs `memory_rows <= data.steps`, and cannot be combined with `train.history_rows`. Use with `train.free_fraction=1`. |
| `train.history_rows` | Before each training rollout, trace 0..N GT rows up to the seed without loss (the front is put back on GT after each step), so the grid memory starts out filled. |
| `evaluation/hard_spots.py`, `data.hard_windows`, `data.hard_fraction` | `evaluation/hard_spots.py` records where a model's traces stop or switch sheets. With `data.hard_windows` set to that JSON, `hard_fraction` of training episodes start at a recorded spot (jittered). Validation is unaffected. |
| `evaluation/benchmark.py --path-map OLD=NEW --out DIR` | Benchmark checkpoints trained on another machine: rewrite the data paths stored in their configs. GPUs differ slightly in bf16, so compare only models benchmarked on the same machine. |
| `evaluation/benchmark.py --seeds-from DIR`, `--held-out-scoring`, `--seeds-anywhere` | Reuse another run's seeds (e.g. across front sizes); score only points whose GT foot point lies in a val cell while the trace runs on; draw seeds from anywhere. |
| `evaluation/benchmark.py --rescore DIR ...` | Recompute the switch counts of existing benchmark folders from their tifxyz. A *real switch* is a row with at least 3 kept off-sheet points, or a kept point more than 9 vx off. |
| `scripts/trace_from_fibers.py` | Traces seeded from VC3D fiber JSONs, advancing along the 2/8 structure-tensor sheet direction, up and/or down. `--follow` uses the fiber-following mode and writes each extended fiber as a VC3D fiber JSON; `--fiber-pool` shows the model the fibers around each trace. |
| `scripts/pick_inference.py` | Rank benchmark folders (front sizes x thresholds) by covered - 10 x off-sheet kept points. |

## Data

- **Training patches:** `data.patches_dir`. These are verified tifxyz patches with level-0 `x, y, z`; `-1` marks
  invalid nodes.
- **Volume:** `data.volume` and `data.volume_level`. This is an OME-Zarr v2 pyramid read through tracers/shared/crop_sampling's
  `ChunkedArray`; the models read level 0. Levels can be decoded in place with
  `python -m vesuvius.neural_tracing.tracers.shared.crop_sampling.decode_store --workers 8 --verify 256 .../s1_ds2.zarr/0`.
  Uncompressed chunks are memory-mapped and shared through the page cache.
- **Episodes:** an episode resamples one patch window onto a canonical `(u, v)` grid at `data.gt_resolution`. `u`
  is the advance direction and `v` runs along the front, both in voxels. There are eight orientations: advance along
  rows or columns, either sign, front either sign. Fiber episodes use a window rotated to the fiber instead.
- **Train/val split:** `data.split_block`^3 cells, assigned by a seeded hash. An episode is used only if all corners
  of its block lie in cells of its split. Trace-benchmark seeds instead require their traced region to lie in val
  cells (or, with `--held-out-scoring`, only their seed row).
- **Patch index:** cached in `output/cache/patch_index_*.json`. It is rebuilt when the patch directory listing
  changes.

## Training

The run directory is `output/NAME`:

```bash
cd vesuvius/src/vesuvius/neural_tracing/sheet_tracing
../../../../../.venv/bin/python -m vesuvius.neural_tracing.tracers.sheets.train.train --config remote_l0_fiber_ft.json \
    --name RUN --init output/remote_l0_points_ft/last.pt
# resume
../../../../../.venv/bin/python -m vesuvius.neural_tracing.tracers.sheets.train.train --resume output/RUN/last.pt
```

What one iteration does:

- Seed the front from GT rows (or a matched fiber) with noise.
- Roll it out for `data.steps` steps using the model's own predictions, adding noise between steps.
- Generate off-sheet cases, so that confidence sees real negatives:
  - Snapped episodes move front points farther than `train.reset_distance` back onto the sheet.
  - Free episodes (`train.free_fraction`) are never snapped, so front points can drift onto other sheets.
  - In `train.jump_probability` of episodes, a random run of front points is pushed along the normal onto the
    brightest CT ridge between `train.jump_min` and `train.jump_max` voxels away: the neighbouring sheet, not the
    gap.
- Average the per-step losses:
  - Position loss: the prediction error versus the GT point one step ahead, split into normal, advance and tangent
    components. The target sits at the front point's seed column, which penalizes the front shrinking or drifting
    sideways. Only points currently on the seed's sheet get this loss.
  - Curve loss: the distance of the dense curve samples to the sheet, for gaps whose two control points are on the
    sheet.
  - Confidence loss: BCE with the label "the prediction is on the seed's sheet", class-balanced (off-sheet points
    are rare).
  - Front-length loss: arc length versus the seed length (`loss.front_length`).
- Patch nodes that are invalid or outside the patch carry no loss: a patch edge is not a sheet edge.

Outputs in the run directory:

- `config.json`, `log.jsonl` and `training.png`.
- `val_ITER/`: episode figures, metric curves and `tifxyz/` (trace and GT for 8 validation episodes).
- `best.pt` (highest validation `score` = coverage - kept_off) and `last.pt`.
- After the run (`eval.after_training`), `eval/`: short free rollouts of `best.pt` on held-out episodes (defaults:
  32 episodes, 10 steps), with figures and `tifxyz_eval/` (`epNN_trace` and `epNN_gt` for every episode).

`train.max_minutes` sets a time budget. Stop a run with Ctrl-C or SIGTERM. The trainer exits through Python, so the
DataLoader shuts down its forkserver workers.

## Evaluation, trace benchmark and inference

```bash
python -m vesuvius.neural_tracing.tracers.sheets.evaluation.rollouts --checkpoint output/RUN/best.pt [--steps 30 --roi 1024 --episodes 256]
python -m vesuvius.neural_tracing.tracers.sheets.evaluation.benchmark --checkpoints output/A/last.pt output/B/last.pt \
    --seeds 40 --steps 30 --tag a_vs_b
python -m vesuvius.neural_tracing.tracers.sheets.tracing.inference --checkpoint output/RUN/last.pt \
    --seed-patch /path/to/patch --line row:40 --range 100:160 --direction + --steps 40 --out output/RUN/traces/t0
```

**Trace benchmark.** It runs long held-out traces through the inference path. Every checkpoint gets identical seed
polylines, and the volume is read around the front at each step.

- Output goes to `output/trace_bench/TAG`: `report.json`, `per_step.png`, `seed_NN.png` and `tifxyz/` (`seedNN_gt`
  plus `seedNN_MODEL` for each model).
- Per trace it also reports `first_switch` (the first step with a kept point off the sheet) and `lost` (the first
  step with under half the points on the sheet).

The evaluation metrics are per step, over points with GT supervision:

| metric | meaning |
| --- | --- |
| `same_sheet` | within `loss.sheet_tolerance` of the seed's sheet |
| `confident` / `kept` | above the confidence threshold |
| `coverage` | same sheet and confident (what a user would keep) |
| `precision` | same sheet among confident points (low means confidently on another sheet) |
| `caught` | not confident among points that are off the sheet (sheet switches that were flagged) |
| `kept_off` | confident and off the sheet, over all supervised points |
| `score` | coverage - kept_off (checkpoint selection) |
| `rejoined` | coverage among front point indices that were below the confidence threshold at an earlier step |
| `curve_same_sheet` | same sheet among dense curve samples whose gap has two confident control points |
| `dist_p50`, `dist_p90`, `advance_p90` | on-sheet distance and absolute advance error of kept points |

**Inference.** The number of front points follows from the seed length; the model does not depend on it. The output
is a tifxyz (rows are steps, columns are the Hermite curve samples, masked points are `-1`; `meta.json` has
`area_cm2`), plus `confidence.npy` and `trace.json`.

**Figures.**

- Left panel: the sheet map (GT `u/v`; `o` = confident, `x` = masked, coloured by on-sheet distance).
- Middle panel: the CT slice spanned by the normal and the advance direction through the middle front point, with
  the GT sheet in green and the trace in red.
- Right panel: the CT slice across the final front, with the curve.

## Code

| module | contents |
| --- | --- |
| `config.py` | dataclass config sections, JSON load, `--set` overrides, rejection of removed model types |
| `data/patches.py` | patch index, memory-mapped tifxyz, canonical and rotated GT windows |
| `data/episodes.py` | episode rejection sampling (patch, fiber-seeded, fiber-following), spatial split, loader streams |
| `data/fibers.py` | memory-mapped fiber pool and the fibers around an episode |
| `geometry.py` | GT grid evaluation and projection, Hermite curves, carrying low-confidence points, respacing, frames |
| `sampling.py` | oriented GPU crops from a volume block |
| `model.py` | crop encoders, joint transformer (shared `models.blocks.SelfAttentionLayer`) over point, grid and fiber tokens, head |
| `tracing/rollout.py` | one tracing step and front update (shared by train/eval/infer) |
| `train/live.py` | live chains |
| `losses.py`, `tracing/seeding.py` | supervision, metrics, seeds, training perturbation and sheet jumps |
| `train/train.py`, `evaluation/rollouts.py`, `evaluation/benchmark.py`, `evaluation/hard_spots.py`, `tracing/inference.py`, `evaluation/figures.py` | entry points and figures |
| `scripts/` | data preparation (other scrolls, fiber matching and pool, resampling), fiber-seeded tracing, benchmarks |

Run the tests with `python -m pytest vesuvius/src/vesuvius/neural_tracing/tracers/sheets/tests`. They use analytic
surfaces and don't need data files.
