# Guided Dinovol Training

## Environment

From `villa/vesuvius`:

```bash
uv sync --extra models --extra tests
```

## Example Guided Training

Use the example config:

```bash
uv run --extra models python -m vesuvius.models.training.train \
  --config src/vesuvius/models/configuration/single_task/ps128_guided_dinovol_ink.yaml \
  --input /path/to/data
```

The guided config uses the local volumetric DINO checkpoint at:

```text
/home/giorgio/Projects/dino-vesuvius/dino-checkpoints/checkpoint_step_342500.pt
```

It also enables:

```yaml
model_config:
  guide_tokenbook_tokens: 256
```

This is an opt-in speed setting for the example config only. The model default remains full-grid TokenBook prototypes when the key is omitted.

## Tests

Run the guided coverage:

```bash
uv run --extra models --extra tests python -m pytest \
  tests/models/build/test_dinovol_local_backbone.py \
  tests/models/build/test_guided_network.py \
  tests/models/training/test_guided_trainer.py -q
```

## Benchmark

Profile unguided vs guided input gating on the local RTX 4090:

```bash
uv run --extra models python -m vesuvius.models.benchmarks.benchmark_guided_dinovol \
  --guide-checkpoint /home/giorgio/Projects/dino-vesuvius/dino-checkpoints/checkpoint_step_342500.pt \
  --patch-size 64,64,64 \
  --device cuda
```

For faster prototype-count sweeps without compile variants:

```bash
uv run --extra models python -m vesuvius.models.benchmarks.benchmark_guided_dinovol \
  --guide-checkpoint /home/giorgio/Projects/dino-vesuvius/dino-checkpoints/checkpoint_step_342500.pt \
  --patch-size 64,64,64 \
  --device cuda \
  --guide-tokenbook-tokens 256 \
  --skip-compile-variants \
  --skip-stage-breakdown
```

## Current Operational Guidance

- Keep the guided path in eager mode unless a local benchmark shows compile is stable for your exact setup.
- Do not expose or rely on a public `channels_last_3d` toggle; the measured gain was negligible relative to plain compile.
- Prefer capped TokenBook prototypes for large training patches; the example config uses `256` for `128^3`.
