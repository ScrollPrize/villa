# Fiber following

Three model types share one trainer, data pipeline, labels, replay, live continuation, tracer and commit policy:

- `regression` (../models/crop_transformer.py): a residual CNN tokenizes the crop and one transformer (the crop
  transformer) reads the crop cells, observed-path references and path-geometry tokens; path tokens read out each
  plane's lateral position and hazard logit, with recurrent refinement passes.
- `flow` (../models/crop_transformer.py, ../models/flow.py): flow matching on the same backbone; path tokens carry a
  noisy path and the flow time (adaLN-Zero) and read out the velocity; a scoring set reads out the hazards.
- `sequence` (../models/sequence.py, train/sequence.py): one transformer reads the whole trace, one history token per
  committed step, and predicts the path and its confidence; training runs whole episodes.

Each type's defaults are the settings of the run it was established with (`mixed_ct_afv_unified_v1`,
`unified_flow_v6_adalnzero`, `sequence_v1`). CT uses per-crop z-score normalization only (all values, float32
statistics, epsilon 1e-6; no clipping).

## Run configuration

A training run is described by one JSON file (train/run_config.py) and launched with it only:

```bash
../../../../../.venv/bin/python train/train.py --config configs/runs/regression.json
../../../../../.venv/bin/python train/train.py --config output/RUN/run.json --resume output/RUN/last.pt
bash scripts/launch_train.sh configs/runs/flow.json            # detached, log in output/logs/NAME.log
bash scripts/stop.sh NAME                                      # the trainer and its DAgger collectors
```

```json
{
  "name": "my_run",
  "out_root": "output",
  "init_weights": null,
  "init_exclude": [],
  "reset_optimizer": false,
  "dataset": {"version": 1, "cache_dir": "...", "sources": [...]},
  "model": {"type": "regression"},
  "training": {"steps": 30000, "lr": 1e-4},
  "runtime": {"workers": 10, "device": "cuda"}
}
```

- `dataset` is the dataset configuration (data/datasets.py): exactly one `paris4` source (fibers, CT,
  seed manifest, `val_z`) and any number of `afv` sources, each with a weight and validation split. Neighboring
  annotations of AFV sources supply foreign-path masks, synthetic wrong continuations and switch detection; Paris 4
  has no neighbor paths.
- `model` takes `type` plus any field of that type's configuration class (`RegressionConfig`, `FlowConfig`,
  `SequenceConfig`), including `fine` (the crop) and `frame_checkpoint` (the frozen heading/normal model; its path
  and SHA-256 are recorded with the model).
- `training` holds the optimizer and schedule, losses, label tolerance, operating policy (`n_commit`,
  `trace_confidence`, `gate`), sampling, task shares (`task_shares`: a full `{task: share}` dict; absent tasks get 0),
  DAgger collection and diagnostics; `runtime` the machine settings (device, workers, threads, CT prefetch).
- Omitted fields take the model type's defaults (`run_config.DEFAULTS` and the configuration classes); unknown keys
  are errors, except removed settings (`run_config.RETIRED`), which are dropped when they hold a value that matches
  the current code and refused otherwise; relative paths resolve against the configuration file's folder.
- The trainer writes the complete resolved configuration to `output/RUN/run.json` and into every checkpoint
  (`run_config`); `configs/runs/{regression,flow,sequence}.json` reproduce the three runs on this machine.

`batch` is decisions per forward pass (episodes for `sequence`); `grad_steps` accumulates those batches. Training
operations are compiled. Flow residual scales are fitted once from `flow_calibration_states` training states and saved
in the model configuration.

## Resume and initialization

- `--resume CHECKPOINT` continues a run: the model configuration comes from the checkpoint (a differing configured
  model section is reported and ignored); model, EMA, optimizer, RNG and step are restored. Everything else (data,
  weights, task shares, policy, schedule, workers) follows the run configuration, so it may change between resumes.
  A replay cache that no longer matches the data is skipped with a warning; a monitor recovery fixture that no
  longer matches is rebuilt (the old one is kept as `monitor_recovery.previous.npz`); CT normalization and frame
  checkpoint differences are reported, not refused. `reset_optimizer` restarts AdamW and the LR schedule.
- `init_weights` starts a new run from a checkpoint's model and EMA tensors, matched by name and shape
  (`init_exclude` prefixes keep their initialization); the trainer reports tensors left fresh, unused and mismatched.

Resumes may change `training.steps` (the total update endpoint) while preserving AdamW state. Increasing it changes
the cosine schedule and can raise the effective LR. For a continuous extension after warmup, use
`lr = saved_lr / lr_at(saved_step - origin, 1, warmup, new_total - origin)`, where `saved_lr` is the checkpoint
optimizer group's LR and `origin` is its `lr_restart_step`.

### Converting checkpoints from before the model cleanup

Checkpoints of the earlier model types `unified`, `unified_flow` and `sequence` are converted once (the model,
EMA and optimizer state are unchanged; dead configuration fields are dropped; the run configuration is rebuilt):

```bash
../../../../../.venv/bin/python scripts/convert_checkpoint.py output/RUN/last.pt output/RUN/last.converted.pt \
  --run-config output/RUN/run.converted.json
bash scripts/convert_running_runs.sh     # the three runs trained before the cleanup, latest checkpoint of each
../../../../../.venv/bin/python train/train.py --config output/RUN/run.converted.json --resume output/RUN/last.converted.pt
```

The converted run configuration keeps the paths of the machine that wrote the checkpoint. For `sequence_v1`, the
live and synthetic task shares (which sequence training does not support) move to `fresh`.

## Tracing

Everything a trace needs comes from the follower checkpoint: the model, its operating policy, its CT (store, level,
normalization) and its learned frame model. Coordinates are base voxels (the CT's level 0), as in VC3D fiber JSON.

`--checkpoint` is a file or a Hugging Face model repo (`ORG/NAME`, optionally `@REVISION`); by default it is
[`scrollprize/fiber_single_small`](https://huggingface.co/scrollprize/fiber_single_small), downloaded with its frame
model into the Hugging Face cache on first use. A repo holds the follower as `ckpt_final.pt` (else its only other
`.pt`, e.g. `last.pt`) and its frame model as `heading_model.pt`. Runs record the repo commit they used.

On a machine other than the one that trained it (`tracing/runtime.py`):
- **Frame model**: for a checkpoint file, put the frame checkpoint next to it, under its recorded file name or as
  `frame_model.pt` / `heading_model.pt`, or pass `--frame-checkpoint`. It must match the sha256 the follower recorded.
- **CT**: a remote store (`s3://`) is read through a local chunk cache: `--cache-dir`, else the recorded folder if it
  exists, else `~/.cache/vesuvius/volume_cache` (`$XDG_CACHE_HOME`). `--ct`, `--ct-level`, `--ct-grid-scale` and
  `--grid-scale` point a checkpoint at another copy or level of its CT.
- Crops are sampled on the GPU (`--crops gpu`, the default on CUDA; `--crops cpu` for the CPU sampler).

Trace from seed points (one H/V family for all seeds, or one per seed); each fiber is a VC3D fiber JSON in `OUT`,
with every seed's outcome in `OUT/seeds.json`:

```bash
python -m vesuvius.neural_tracing.tracers.fibers.tracing.inference --seed 12624,9936,46080 --family V --out traced/
python -m vesuvius.neural_tracing.tracers.fibers.tracing.inference --seeds-file seeds.json --family H \
  --out traced/                                             # seeds.json: [[x, y, z], ...]
python -m vesuvius.neural_tracing.tracers.fibers.tracing.inference --checkpoint my_run/ckpt_050000.pt \
  --seed 12624,9936,46080 --family V --out traced/         # another checkpoint
```

Trace every fiber of a volume (or a box) from a fiber presence prediction on the trace grid (the presence zarr level
whose voxel is `grid_scale` base voxels; the next, coarser level is used for progress if present). Each fiber is
written once; the output is `OUT/fibers.afv` (an Automated Fiber Volume for VC3D's fiber collection dock), plus
per-tile results that make a rerun resume. `--ct-z` is a z range in CT voxels of the checkpoint's CT level, `--box`
a trace-grid box `x0,y0,z0,x1,y1,z1`:

```bash
python -m vesuvius.neural_tracing.tracers.fibers.tracing.volume --presence presence.ome.zarr/3 \
  --ct-z 7000,17000 --out paris4_fibers/ | tee -a paris4_fibers.log
python -m vesuvius.neural_tracing.tracers.fibers.tracing.volume_afv paris4_fibers/   # fibers.afv from the tiles so far
```

The AFV's coordinate frame is the CT's when it is a Vesuvius open-data store (`s3://vesuvius-challenge-open-data/...`);
for any other CT, pass an `.afv` of the same scan as `--afv-template` (`volume_afv.py --template`). Status lines
(progress, decisions/s, ETA) print every minute. One process uses one GPU; the Paris4 run held about 23 GB with
`--gpu-chunks 1024 --forward-chunk 64` and 27 GB with `--gpu-chunks 3072` (each chunk slot is 2 MiB; more slots
load fewer CT chunks).

VC3D's Line Annotation can trace its Fiber model lines with this model (hamburger menu, "Fiber follow model" → "Use
for the Fiber model"; off by default) through `tracing/service.py`, which VC3D starts with the Python environment it
was launched from: the default checkpoint unless one is chosen there.

## Evaluation

```bash
../../../../../.venv/bin/python -m vesuvius.neural_tracing.tracers.fibers.train.dagger.collect --help
../../../../../.venv/bin/python -m vesuvius.neural_tracing.tracers.fibers.evaluation.protocol --help
```

New runs use the frozen `heading_model_l0_w16_centered_frame/ckpt_064000.pt` for current-crop heading and roll. CT
initializes seed headings with derivative sigma 2 and integration sigma 8, using a 77³ native CT context. H/V family
comes from the seed or fiber metadata; only CT and observed paths enter the frame model. The model runs in batches on
CPU, once per worker/process, with no training gradients. `TRACERS_PATH_MAP='OLD=NEW;...'` (formerly `FIBER_FOLLOW_PATH_MAP`) relocates absolute
paths recorded on another machine when a checkpoint is loaded for inference. Paths recorded under the package's former
location `neural_tracing/fiber_follow/` (checkpoints, run and dataset configurations) resolve to `neural_tracing/tracers/fibers/`.

DAgger subprocesses receive `runtime.dagger_threads` (default 4), independently of the trainer's `runtime.threads`.

### Pre-cleanup checkpoints (inference only)

The `flow_matching` checkpoints of the v4-extension and v5 runs and the `coordinate_regression` checkpoints of
`mixed_ct_afv_forward35_v1` and `mixed_ct_afv_v3init_nomem_b8` (patch encoder, decoder, segment scorer; no memory or
identity heads) trace through `legacy/`, a frozen copy of those models kept apart from `tracers/models/`. It takes the same
arguments as `tracing.inference` and refuses checkpoints with features it does not implement (decision memory, identity
heads, whole-crop planes); there is no training support.

```bash
../../../../../.venv/bin/python -m vesuvius.neural_tracing.tracers.fibers.legacy.infer --checkpoint CKPT.pt \
  --seed X,Y,Z --family H --out traced/
```

## Snapping fiber labels to the presence prediction

`scripts/snap_fibers.py` (`data/fiber_snapping.py`) moves fiber labels onto the centre of a fiber presence prediction:
at every point, the centroid of the upper half of the presence on a perpendicular disk (radius 2.5 voxels, averaged over
+-1 voxel along the fiber), smoothed along the fiber (sigma 2), two passes. It reads an AFV (writing a new AFV with the
same ids, names, families and annotations and a `snapping` metadata record) or VC3D fiber JSONs (writing snapped
`line_points`, controls moved onto the line, and `snapping.json`). The work is tiled over the presence volume and
batched on the GPU. On Paris4 AFV it brought labels from a median 1.46 to 0.68 trace voxels from the manual annotations
(`output/paris4_annotation_audit_20261006/`).

```bash
../../../../../.venv/bin/python scripts/snap_fibers.py datasets/automated_fiber_volumes/IN.afv OUT.afv \
  --presence /path/PHercParis4-..._presence.ome.zarr/3 --native-per-voxel 8   # level 3 = 8 native voxels per voxel
```

## Sheet strips: the sheet a fiber lies on

`sheet_strip/` fine-tunes a regression follower (by default the slim 1.26M `polar_tokens_a3_ms_slim_dagger_b32x2_50k`
checkpoint) to predict the papyrus sheet a fiber lies on, +-60 voxels (9.6 um) across it, over the whole crop depth.
The model is shown the fiber as a thin Gaussian tube (sigma one sample) in a second input channel and still predicts
its continuation:

- **Model** (`sheet_strip/model.py`): the crop CNNs take the tube channel (zero weights at initialization, so the
  model starts as the checkpoint), the crop widens to 128 samples (the path keeps the checkpoint's lateral bound,
  `path_lateral_limit`) and a set of strip tokens (path tokens of the crop transformer, one per crop slab and strip
  column, 450 by default) reads out the sheet's height `u` along the frame's sheet normal on a 36 x 25 grid of
  `(f, v)` (4 x 5 voxels), with one refinement pass that samples the CNN at the predicted sheet.
- **Targets** (`sheet_strip/targets.py`): the height field of the verified tifxyz patches the fiber lies on, from
  their triangles within reach of the crop. Points no patch covers, folds, patches disagreeing by more than 2 units
  (4 voxels), surfaces steeper than 70 degrees against the crop's `(v, f)` plane and heights outside the crop carry
  no loss.
- **Data** (`sheet_strip/data.py`): the follower's own fresh decisions (simulated traces, frame model, augmentation)
  with their heads on the spans where an AFV fiber lies on a patch. Half the decisions are shown the whole fiber
  (over its matched spans +-8 voxels; its path is then not supervised), half only their simulated trace up to the
  head.

```bash
# 1. Fibers on patches (sheets/scripts/match_fibers_to_patches.py), per family, AFV voxels = 4 patch voxels
python ../sheets/scripts/match_fibers_to_patches.py --afv FIBERS.afv --grid-scale 4 --family H --min-run 40 \
  --patches-dir /mnt/raid_nvme/spiral_dataset_working/verified_patches \
  --umbilicus /mnt/raid_nvme/spiral_dataset_working/umbilicus.json --out H_runs.jsonl
# 2. The run index; runs inside the z band 11250-12125 (patch voxels: the fiber runs' val_z) validate
python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.build_index --afv FIBERS.afv \
  --runs H_runs.jsonl V_runs.jsonl --patches-dir /mnt/raid_nvme/spiral_dataset_working/verified_patches --out INDEX
# 3. Train (detached; log in output/logs/NAME.log)
TRAINER=sheet_strip/train.py bash scripts/launch_train.sh configs/sheet_strip/paris4_slim50k.json
```

The run folder holds `log.jsonl`, `curves.png` (strip error against the frame model's plane, per tube mode, losses,
path error), `figures/` (every `figure_every` updates, validation and training decisions: CT slices along the fiber
and across it with the patch sheet, the prediction and the tube, and the error map), `best.pt` (lowest validation
strip error) and `last.pt`; `sheet_strip.train.load_strip_model` loads one. The Paris4 slim-50k AFV gave 617k training
runs on 155k fibers and 60k validation runs on 17.8k fibers (`output/sheet_strip_data/`).

`sheet_strip/infer.py` walks one AFV fiber and writes the predicted strip as a tifxyz (patch voxels; rows every 4
voxels along the fiber, 25 columns 5 voxels apart across it). The checkpoint defaults to the Hugging Face repo
`scrollprize/fiber_strip_small_v1` (`sheet_strip_paris4_slim50k_v3`, update 15000), downloaded into the Hugging Face
cache; `--checkpoint` takes another repo (`ORG/NAME[@REVISION]`) or a local file. It reads the CT recorded in the
checkpoint (Paris4 on the public S3 bucket; `--ct`, `--cache-dir` and the other volume options of
`fibers/tracing/runtime.py` override it). The frame model and the umbilicus are taken from their recorded paths when
those exist on this machine, else from `heading_model.pt` and `umbilicus.json` next to the checkpoint (in its
Hugging Face repo), else `--frame-checkpoint` and `--umbilicus` must be given; `--afv` is needed off this machine,
and scoring is skipped when the recorded index is not here. Outputs record the checkpoint as
`hf://REPO@COMMIT/last.pt`. It makes a decision every `--step` (default 8) trace voxels,
built as in training without augmentation. The fiber up to the head is the observed trace, and the tube shows the
whole fiber. Overlapping decisions are averaged per node, weighted towards each decision's head, after each
decision's in-sheet axis is turned to agree with the previous one's. When the fiber is in the run's index, the
patch sheet is scored and stitched the same way (`patch_sheet/`). `render.png` shows the CT sampled on the
predicted strip in bands along the fiber. Under each band, a bar shows the error against the patch sheet where
there is one. `--render-only` redraws it for an existing output folder. Nodes within `--trim` (default 75) patch
voxels of either fiber end are left out, as in `infer_all.py`.

```bash
python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer --afv FIBERS.afv --fiber 469373 \
  --out output/sheet_strip_infer/fiber_469373
```

On validation fiber 469373 (6100 voxels, 758 decisions, 2.5 minutes), the update-16000 checkpoint gave 1.22 voxels
mean error against 6.89 for the frame plane, with 82% of points within 2 voxels.

`sheet_strip/infer_all.py` runs the same per-fiber inference over every AFV fiber above a length, without scoring or
renders, sharded over processes. Each strip is written as a tifxyz in `OUT/strips/BUCKET/fiber_ID`, with
`expected_error.tif` when the model predicts its error. An H fiber runs along the strip's columns and a V fiber along
its rows, higher z at the top (row 0); the TIFFs are uncompressed single strips, which VC3D memory-maps as written
(it rewrites tiled LZW bands in place before mapping them). Nodes within `--trim` (default 75) patch voxels of either
fiber end, or beyond it, are left out (-1): there the strip overhangs the fiber, rests on one or two decisions, and the
annotated fiber's own ends are least reliable. Finished fibers are listed in `OUT/index/shard_K.jsonl`, and a
restarted shard skips them:

```bash
for k in $(seq 0 17); do
  python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer_all --afv FIBERS.afv --out OUT \
    --batch 4 --threads 1 --shard $k 18 &
done
```

The defaults (`--min-length-mm 2.5 --step 8 --blend 16 --trim 75`, the Hugging Face checkpoint) are the Paris4 bulk
run's. Its strips are reproduced exactly with `--batch 4 --threads 1`, as above; another batch size or thread count
(the defaults: 32 and 2) changes nodes by about 0.015 voxels on average and up to about 0.3 (GPU batch numerics and
the CPU frame model).

Throughput is CPU-bound: building each decision's crops (frame model, CT sampling and normalization) dominates. Each
process needs about 1.3–1.5 GB of GPU memory, so 18 shards fill a 32 GB card. On the 24-core machine that gives about
318 decisions/s, about 15 h for Paris4's 238k fibers over 2.5 mm (17M decisions). The sampler forms its frame products
per crop axis rather than per point, and `normalize_ct` makes fewer passes (both bitwise unchanged): 220 -> 310
decisions/s. Running the shards as clients of an NVIDIA MPS daemon (`nvidia-cuda-mps-control -d`, with the shards'
`CUDA_MPS_PIPE_DIRECTORY` set to its pipe directory) lowers GPU use from 92% to 76% at the same rate.
`--crop-threads 2` (a batch's crops sampled on two threads, same values) gave 326 decisions/s, but shards then
segfaulted inside CPython (3 of 18 within 50 minutes); keep the default 1.

A decision every 8 trace voxels (`--step 8`) gave the same stitched strips as every 4 on held-out fibers (median node
distance to the patch sheet within 0.012 voxels); every 16 was 0.01–0.03 worse. Decisions whose seed lies outside the
crop skip the CT seed heading they would never read (`infer.skip_remote_seed_heading`; 1.7x), and the others compute
it once per fiber (`infer.reuse_seed_heading`); outputs are unchanged.

Options of `SheetStripConfig` beyond the first run (all off by default; a run continues from a sheet-strip checkpoint
when `init_checkpoint` is one, its added modules starting fresh):

- `strip_full_fine`: the strip's evidence reads the fine CNN run over the whole crop cross-section, not only its
  central `multiscale_fine_width` samples (no new weights);
- `strip_orientation`: the strip tokens read the crop's axes against the scroll (`sheet_strip/data.crop_orientation`;
  needs `data.umbilicus`, and `infer.py --umbilicus` defaults to the run's);
- `strip_profile_reach`: each refinement pass also reads the raw CT and tube along `u` around every strip point;
- `strip_uncertainty`: a probe (gradient stopped) predicting each strip point's error scale, trained by a Laplace
  likelihood; `infer.py` writes it as `expected_error_vox`;
- `strip_band_half_height`: sheet-band context tokens on the full fine lattice beside the polar ones;
- `strip_refinement_steps`: more refinement passes.

`sheet_strip/evaluate.py` scores checkpoints on one fixed, cached set of validation decisions, both tube modes each:

Building a cache requires a new or empty output directory; use a new directory when changing the batch count.

```bash
python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.evaluate build --checkpoint RUN/last.pt \
  --out /mnt/raid_nvme/sheet_strip_eval/val6400 --batches 200   # 6400 decisions, 44 GB, 5 minutes
python -m vesuvius.neural_tracing.tracers.fibers.sheet_strip.evaluate run --eval /mnt/raid_nvme/sheet_strip_eval/val6400 \
  --checkpoint RUN/last.pt   # 1.5 minutes; results/NAME.npz and .json
```

On that set (October 2026), 4,000-update fine-tunes of the 40k `sheet_strip_paris4_slim50k_v1` against a control gave
these mean errors in voxels:

| Run | Full tube | Observed tube |
|---|---|---|
| control | 1.143 | 1.271 |
| `strip_full_fine` | 1.091 | 1.218 |
| `strip_orientation` | 1.117 | 1.229 |
| `strip_profile_reach=6` | 1.131 | 1.258 |
| all three | 1.058 | 1.169 |
| all three for 15k updates (`configs/sheet_strip/paris4_slim50k_v2_ft.json`) | **1.006** | **1.117** |

Two other variants did not help:
- band tokens on top of `strip_full_fine`: −0.001;
- a second refinement pass: +0.009.

The failure analysis behind them is in `output/sheet_strip_eval/REPORT.md` (local; `output/` is not tracked).

## Organization

The fiber tracer lives in `tracers/fibers/`; the parts other tracers can reuse sit beside it in `tracers/`
(`tracers/README.md`). Nothing in `tracers/models` or `tracers/shared` imports `tracers/fibers`.

| Package | Responsibility |
| --- | --- |
| `tracers/models` | The follower contract (`follower.py`), the model registry (`model.py`), checkpoints (`checkpoint.py`), the crop transformer (`regression`, `flow`), flow matching, the sequence model, the supervised-path contract (`targets.py`) |
| `tracers/shared` | Geometry, observed-path references, the commit policy, recorded-path relocation, retired settings, JSON/atomic output, batch tensors, the CT structure tensor (`ct_structure.py`), training-run pieces (`training.py`: run dirs, log, LR schedule, EMA) and compile options |
| `tracers/shared/annotations` | Fiber annotations: AFV collections, VC3D fiber JSON, the annotated-fiber type, kink repair, whole-fiber splits |
| `tracers/shared/crop_sampling` | CT crop loading: volumes (local and remote chunked readers, chunk-store decoding), intensity normalization, oriented crop sampling (CPU kernel for data workers, GPU sampler for in-step crops), remote prefetch |
| `data` | The z-band split, seeds, simulated traces and episodes, targets and state labels, the task budget, replay, the training dataset, observations, AFV sources and neighbors |
| `frames` | Crop frames: CT structure-tensor headings and frames (`ct.py`), the frozen learned frame model (`learned.py`) |
| `tracing` | The tracer and rollout (`trace.py`), the tracing CLI (`inference.py`) |
| `train` | Entry point and run configuration, the training update (`step.py`), objectives, logging, live continuation, DAgger collection (`dagger/`) |
| `evaluation` | Trace scoring (`scoring.py`), the offline calibrate/run/compare protocol (`protocol.py`), training-time monitoring (`monitor/`) |
| `heading_model` | Heading/normal trainer and the frame predictor network used by `frames/learned.py` |
| `sheet_strip` | The sheet-strip follower: model, patch targets, data, trainer and figures |
| `legacy` | Inference for pre-cleanup checkpoints |
| `scripts` | Launch/stop, label snapping, checkpoint conversion, the pre-launch sampler check |
| `configs` | Run (`runs/`), dataset (`datasets/`) and heading-model (`heading/`) configurations |

Run artifacts are in `output/` and AFV datasets in `datasets/` (both untracked). Three files in `output/` are inputs of
current runs: the Paris 4 seed manifest `single_path_v11_preparation/seeds.json`, the frame model
`heading_model_l0_w16_centered_frame/ckpt_064000.pt` and `heading_model_l0_w16_centered_normals/ct_normalization.json`.

## Verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/fiber-mpl \
  ../../../../../.venv/bin/python -m pytest tests -q -o cache_dir=/tmp/fiber-pytest
```

Tests cover the three model types, compiled graph capture, weight initialization/resume, gradient paths, supervision
masks, deterministic flow tracing, observation parity, collection and the standalone heading model. CUDA tests skip when CUDA is unavailable. Small-fixture checks
establish implementation correctness; they do not establish trained tracing accuracy or production GPU throughput.
