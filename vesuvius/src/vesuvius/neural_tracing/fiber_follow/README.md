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

`--resume output/RUN/last.pt --name RUN` restores a new-format run. `--init-weights CHECKPOINT` strictly loads model and EMA weights into a fresh run, without optimizer, replay, scheduling, or normalization state. Runtime checkpoints use unversioned model types. Legacy checkpoints must be converted explicitly; old flow checkpoints are unsupported. The model type is inferred from checkpoints unless explicitly specified.

The one-off conversion for the existing 81,000-step weights is:

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.scripts.convert_v17_weights \
  output/mixed_ct_afv_stem32_run3_pathgeom_paris50/ckpt_081000.pt \
  output/mixed_ct_afv_stem32_run3_pathgeom_paris50/coordinate_regression_weights.pt
```

The converted file preserves model/EMA tensors and inference/data provenance. It supports inference and `--init-weights`, but not `--resume`. The source is untouched; the converter refuses to overwrite an existing destination. `scripts/train_coordinate_regression.sh` uses this converted file by default.

DAgger subprocesses receive `--dagger-threads` (default 4), independently of the trainer's `--threads`. Both use the common collector. Other inference thread defaults are unchanged.

```bash
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.tracing.infer --help
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.tracing.collect --help
../../../../.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.evaluation.evaluate --help
```

`scripts/train_coordinate_regression.sh` records the fixed coordinate regression experiment settings; `scripts/launch_train.sh` launches a named run. `scripts/stop.sh NAME` stops that run and its collectors.

## Organization

| Package | Responsibility |
| --- | --- |
| `models` | Shared observation/history/scoring components and the two generators |
| `train` | Training entrypoint, objectives, checkpoints, logging, online collection, live continuation |
| `data` | Volumes, normalization, observation construction, labels, sampling, replay, neighbor banks |
| `tracing` | Inference, collection, heading/frame resolution, commit policy, rollout |
| `evaluation` | Frozen-seed evaluation, recovery, diagnostics, benchmarks |
| `visualization` | Measured interpretation reports for either model |
| `heading_model` | Independent heading model and trainer, retained for future integration |
| `shared` | Geometry, reference handling, sampling kernels, experiment utilities |

Historical experiment documentation is under `docs/history` and `plans`. Datasets and existing run artifacts retain their locations. Old `regression.*` and `flow_matching.*` Python entrypoints have been removed.

## Verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/fiber-mpl \
  ../../../../.venv/bin/python -m pytest tests -q -o cache_dir=/tmp/fiber-pytest
```

Tests cover both generators, compiled graph capture, weight initialization/resume, gradient paths, supervision masks, deterministic flow tracing, observation parity, collection, normalization, interpretation, and the standalone heading model. CUDA tests skip when CUDA is unavailable. Small-fixture checks establish implementation correctness; they do not establish trained tracing accuracy or production GPU throughput.
