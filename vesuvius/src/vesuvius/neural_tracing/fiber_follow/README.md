# Fiber following

Three model types share one trainer, data pipeline, labels, replay, live continuation, tracer and commit policy:

- `regression` (models/crop_transformer.py): a residual CNN tokenizes the crop and one transformer (the crop
  transformer) reads the crop cells, observed-path references and path-geometry tokens; path tokens read out each
  plane's lateral position and hazard logit, with recurrent refinement passes.
- `flow` (models/crop_transformer.py, models/flow.py): flow matching on the same backbone; path tokens carry a
  noisy path and the flow time (adaLN-Zero by default) and read out the velocity; a scoring set reads out the hazards.
- `sequence` (models/sequence.py, train/sequence.py): one transformer reads the whole trace, one history token per
  committed step, and predicts the path and its confidence; training runs whole episodes.

Each type's defaults are the settings of the run it was established with (`mixed_ct_afv_unified_v1`,
`unified_flow_v6_adalnzero`, `sequence_v1`). CT uses per-crop z-score normalization only (all values, float64
statistics, epsilon 1e-6; no clipping).

## Run configuration

A training run is described by one JSON file (train/run_config.py) and launched with it only:

```bash
../../../../.venv/bin/python train/train.py --config configs/runs/regression.json
../../../../.venv/bin/python train/train.py --config output/RUN/run.json --resume output/RUN/last.pt
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

- `dataset` is the dataset configuration (data/datasets.py): exactly one `paris4` source (fibers, fiber zarrs, CT,
  seed manifest, `val_z`, negative bank) and any number of `afv` sources, each with a weight and validation split.
- `model` takes `type` plus any field of that type's configuration class (`RegressionConfig`, `FlowConfig`,
  `SequenceConfig`), including `fine` (the crop) and `frame_checkpoint` (the frozen heading/normal model; its path
  and SHA-256 are recorded with the model).
- `training` holds the optimizer and schedule, losses, label tolerance, operating policy (`n_commit`,
  `trace_confidence`, `gate`), sampling, task shares (`task_shares`: a full `{task: share}` dict; absent tasks get 0),
  DAgger collection and diagnostics; `runtime` the machine settings (device, workers, threads, CT prefetch).
- Omitted fields take the model type's defaults (`run_config.DEFAULTS` and the configuration classes); unknown keys
  are errors; relative paths resolve against the configuration file's folder.
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
../../../../.venv/bin/python scripts/convert_checkpoint.py output/RUN/last.pt output/RUN/last.converted.pt \
  --run-config output/RUN/run.converted.json
bash scripts/convert_running_runs.sh     # the three runs trained before the cleanup, latest checkpoint of each
../../../../.venv/bin/python train/train.py --config output/RUN/run.converted.json --resume output/RUN/last.converted.pt
```

The converted run configuration keeps the paths of the machine that wrote the checkpoint. For `sequence_v1`, the
live and synthetic task shares (which sequence training does not support) move to `fresh`.

## Tracing and evaluation

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.tracing.infer --help
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.tracing.collect --help
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.evaluation.evaluate --help
```

New runs use the frozen `heading_model_l0_w16_centered_frame/ckpt_064000.pt` for current-crop heading and roll. CT
initializes seed headings with derivative sigma 2 and integration sigma 8, using a 77³ native CT context. H/V family
comes from the seed or fiber metadata; only CT and observed paths enter the frame model. The model runs in batches on
CPU, once per worker/process, with no training gradients. `FIBER_FOLLOW_PATH_MAP='OLD=NEW;...'` relocates absolute
paths recorded on another machine when a checkpoint is loaded for inference.

DAgger subprocesses receive `runtime.dagger_threads` (default 4), independently of the trainer's `runtime.threads`.

### Pre-cleanup flow checkpoints (inference only)

The `flow_matching` checkpoints of the v4-extension and v5 runs (patch encoder, flow decoder, segment scorer; no
memory or identity heads) trace through `legacy/`, a frozen copy of that model kept apart from `models/`. It takes the
same arguments as `tracing.infer` and refuses checkpoints with features it does not implement (decision memory,
identity heads, whole-crop planes); there is no training support.

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.legacy.infer --checkpoint CKPT.pt \
  --seed X,Y,Z --family H --out traced/
```

## Organization

| Package | Responsibility |
| --- | --- |
| `models` | The follower contract (`model.py`), the crop transformer (`regression`, `flow`), flow matching, the sequence model |
| `train` | Entry point and run configuration, objectives, checkpoints, logging, online collection, live continuation |
| `data` | Volumes, normalization, observation construction, labels, sampling, replay, neighbor banks |
| `tracing` | Inference, collection, heading/frame resolution, commit policy, rollout |
| `evaluation` | Frozen-seed evaluation, recovery, diagnostics, benchmarks |
| `heading_model` | Heading/normal trainer and frozen frame predictor used by follower crop generation |
| `shared` | Geometry, reference handling, sampling kernels, experiment utilities |

Datasets and existing run artifacts retain their locations.

## Verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/fiber-mpl \
  ../../../../.venv/bin/python -m pytest tests -q -o cache_dir=/tmp/fiber-pytest
```

Tests cover the three model types, compiled graph capture, run configuration, checkpoint conversion, weight
initialization/resume, gradient paths, supervision masks, deterministic flow tracing, observation parity, collection,
normalization and the standalone heading model. CUDA tests skip when CUDA is unavailable. Small-fixture checks
establish implementation correctness; they do not establish trained tracing accuracy or production GPU throughput.

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
