# Fiber following

One training pipeline supports two generators over the same CT observations:

- `coordinate_regression` (default): direct coordinates with recurrent refinement.
- `flow_matching`: velocity prediction trained with Gaussian/time draws; midpoint integration starts at lateral x=0, y=0 during tracing.

Both use the patch4 residual-stem encoder, historical CT/path slabs, path-geometry tokens, cached attention, and causal segment-survival scorer. They share sampling, labels, replay, live continuation, optimizer, EMA, and tracing policy. CT uses per-crop z-score normalization only (all values, float64 statistics, epsilon 1e-6; no clipping).

## Run

From this directory, use the existing project environment:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.train.train \
  --name coordinate_run --dataset-config configs/mixed_ct_datasets_paris50.json

../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.train.train \
  --name flow_run --model flow_matching --dataset-config configs/mixed_ct_datasets_paris50.json
```

`--batch` is decisions per forward pass; `--grad-steps` accumulates those batches. Training operations are compiled. Flow defaults to four midpoint steps and 64 training draws. Its residual scales are fitted once from 2,048 training states (`--flow-calibration-states`) and saved in the model configuration. Each state has equal loss weight; flow velocity loss replaces the coordinate regression geometry loss, while both train survival confidence using the same labels and loss.

Flow options (defaults keep the original model and old checkpoints load unchanged):
- `--flow-time-conditioning adaln`: the time embedding also modulates every decoder branch (self, image and history attention, FFN) and the output norm per draw. The modulation is zero-initialized, so a warm start is exact.
- `--flow-samples K --flow-sample-scale S`: K proposals from Gaussian starts (standard deviation S in residual-scale units) after the zero-start path. Training integrates and scores all of them, so the scorer learns sampled paths; tracing uses a later proposal only when no earlier one is accepted.
- `--flow-sigma-floor V`: lower bound in voxels for the fitted residual scales, which set the noise prior's width (default 1).
- `--flow-unknown-planes own_path`: target-less planes stay in self-attention, as in tracing, and move from noise to the model's own zero-start path. They get no loss.

Identity options with `--memory decisions` (defaults keep the original model):
- `--identity-objective verify`: a verifier reads the decision-memory tokens and labels current-crop locations on/off the original fiber (`models/identity_verifier.py`).
- `--identity-objective readout`: path-blind identity memory (`models/identity_readout.py`). Appearance keys come from the image stem, taken before the position embedding and the rendered observed path are added, through a small convolutional head with no position input. Each decision stores its keys around its head as extra entry rows; a memory value adds that point's radius from its decision's trace axis and the entry's age and seed role. Every current location is read out against all memory keys and trained with the verifier's labels and loss. Memory keys carry no gradient; the current keys, stem and readout train through the identity loss.
- `--identity-map` / `--identity-feedback`: the identity field enters decoder/scorer image tokens / scorer segments and retries (zero-initialized).
- `--init-exclude PREFIX` (with `--init-partial`, repeatable): keep the initialization of matching tensors, e.g. zero-initialized heads whose input changed meaning.

`FIBER_FOLLOW_PATH_MAP='OLD=NEW;...'` relocates absolute paths recorded on another machine (seed manifest volume, negative-bank run, initialization dataset sources) before the exact startup checks.

`--resume output/RUN/last.pt --name RUN` restores a new-format run. `--init-weights CHECKPOINT` strictly loads model and EMA weights into a fresh run, without optimizer, replay, scheduling, or normalization state. Runtime checkpoints use unversioned model types. Legacy checkpoints must be converted explicitly; old flow checkpoints are unsupported. The model type is inferred from checkpoints unless explicitly specified.

Resumes may change `--steps` (the total update endpoint) while preserving AdamW state.
Increasing it changes the cosine schedule and can raise the effective LR. For a continuous
extension after warmup, use `--lr saved_lr / lr_at(saved_step - origin, 1, warmup, new_total - origin)`,
where `saved_lr` is the checkpoint optimizer group's LR and `origin` is its `lr_restart_step`.
Keep warmup and origin unchanged and omit `--reset-optimizer`; the remaining cosine then
decays monotonically from the saved LR to approximately zero at the new endpoint.

The one-off conversion for the existing 81,000-step weights is:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.scripts.convert_v17_weights \
  output/mixed_ct_afv_stem32_run3_pathgeom_paris50/ckpt_081000.pt \
  output/mixed_ct_afv_stem32_run3_pathgeom_paris50/coordinate_regression_weights.pt
```

The converted file preserves model/EMA tensors and inference/data provenance. It supports inference and `--init-weights`, but not `--resume`. The source is untouched; the converter refuses to overwrite an existing destination. `scripts/train_coordinate_regression.sh` uses this converted file by default.

DAgger subprocesses receive `--dagger-threads` (default 4), independently of the trainer's `--threads`. Both use the common collector. Other inference thread defaults are unchanged.

New follower runs use the frozen `heading_model_l0_w16_centered_frame/ckpt_064000.pt` for current-crop
heading and roll. CT initializes seed headings with derivative sigma 2 and integration sigma 8, using a
77³ native CT context. Historical slabs retain their observed-path tangent and use the learned normal for roll.
H/V family comes from the seed or fiber metadata; only CT and observed paths enter the frame model.
The model runs in batches on CPU, once per worker/process, with no training gradients.

`--frame-checkpoint PATH` selects a different frozen frame model for training, inference, or collection.
Its absolute path and SHA-256 are saved in the follower model configuration, so subsequent tracing and
collection use the same weights. Keep that checkpoint available. Resume retains the recorded selection unless
explicitly overridden. Unknown-family observations use the 2/8 CT frame; recorded replay frames stay unchanged.
Read planning covers the learned model patch and every possible new crop heading. Training recomputes plane
crossings and supervision after a heading change. Frame diagnostics use source code 3 for learned frames.

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.tracing.infer --help
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.tracing.collect --help
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.evaluation.evaluate --help
```

`scripts/train_coordinate_regression.sh` records the fixed coordinate regression experiment settings; `scripts/launch_train.sh` launches a named run. `scripts/stop.sh NAME` stops that run and its collectors.

For a fresh 100k-step run initialized from
`output/mixed_ct_afv_stem32_run3_pathgeom_paris50/coordinate_regression_weights.pt` with learned frames:

```bash
bash scripts/train_coordinate_learned_frame.sh
```

This wraps the coordinate launcher with run name `mixed_ct_afv_coordinate_learned_frame_100k`, the tested 64k
frame checkpoint, 100k updates, and the source run's 5k warmup, 6,000-voxel collection limit and 96-voxel
post-failure window. It keeps batch 4 × 3 accumulation, LR 1e-4, 10 workers, 48 prefetch connections and Paris50
dataset weights. AdamW and replay start fresh; the follower's model and EMA weights initialize from the source.
After warmup, the follower LR follows its existing cosine schedule. `COORDINATE_RUN_NAME`, `COORDINATE_INIT`,
`FRAME_CHECKPOINT`, and trailing CLI arguments can override the run name, weight source, frame model and settings.

## Organization

| Package | Responsibility |
| --- | --- |
| `models` | Shared observation/history/scoring components and the two generators |
| `train` | Training entrypoint, objectives, checkpoints, logging, online collection, live continuation |
| `data` | Volumes, normalization, observation construction, labels, sampling, replay, neighbor banks |
| `tracing` | Inference, collection, heading/frame resolution, commit policy, rollout |
| `evaluation` | Frozen-seed evaluation, recovery, diagnostics, benchmarks |
| `visualization` | Measured interpretation reports for either model |
| `heading_model` | Heading/normal trainer and frozen frame predictor used by follower crop generation |
| `shared` | Geometry, reference handling, sampling kernels, experiment utilities |

Historical experiment documentation is under `docs/history` and `plans`. Datasets and existing run artifacts retain their locations. Old `regression.*` and `flow_matching.*` Python entrypoints have been removed.

## Verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/fiber-mpl \
  ../../../../.venv/bin/python -m pytest tests -q -o cache_dir=/tmp/fiber-pytest
```

Tests cover both generators, compiled graph capture, weight initialization/resume, gradient paths, supervision masks, deterministic flow tracing, observation parity, collection, normalization, interpretation, and the standalone heading model. CUDA tests skip when CUDA is unavailable. Small-fixture checks establish implementation correctness; they do not establish trained tracing accuracy or production GPU throughput.

Long held-out failure audits can reuse the shared tracer and scorers while recording every
policy decision and scoring the same path at 400, 1,000, and 2,000 voxels:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.evaluation.long_trace_audit \
  --checkpoint output/RUN/ckpt_079000.pt --out output/RUN/long_audit/primary --count 64 --batch 8
```

This evaluates the fixed monitor seeds plus distinct calibration fibers sampled without
replacement with probability proportional to annotated length. New seeds use CT headings,
random directions, and the first 20% of a fiber (at least 32 voxels from its starting end)
to favor long annotated continuations. Final-test fibers are excluded. The resulting
`seeds.json` is frozen before rollout and can be reused with `--seed-manifest` for paired
checkpoint or policy comparisons (`--confidence`, `--n-commit`, `--sources`, `--cohorts`).
Results describe this length-biased diagnostic population, not uniform fiber performance.
Saved NPZ files contain paths, annotations and decision inputs; JSON rows retain strict
first-failure, local geometric agreement, annotation censoring and policy-audit outcomes.
Local agreement is not independent confirmation of fiber identity.

Render CT sections around selected departures and stops, plus aggregate diagnostics:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.evaluation.summarize_long_audit \
  output/RUN/long_audit --render
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.evaluation.annotation_audit \
  output/RUN/long_audit
```

The report expects the main evaluation in `long_audit/primary`. Optional paired runs
in `confidence_030`, `confidence_080`, `commit_04`, and `checkpoint_055000` are compared on identical
seed keys with fiber-resampled 95% intervals. Annotation proximity is a review aid,
not a certified switch detector; the annotation audit uses the local Paris catalog.
