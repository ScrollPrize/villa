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

Ps256 guided configs are also available:

```text
src/vesuvius/models/configuration/single_task/ps256_guided_medial.yaml
src/vesuvius/models/configuration/single_task/ps256_guided_dicece.yaml
```

For large guided runs, generate patch caches before training:

```bash
uv run --extra models vesuvius.find_patches \
  --config src/vesuvius/models/configuration/single_task/ps256_guided_medial.yaml
```

## Tests

Run the guided coverage:

```bash
uv run --extra models --extra tests python -m pytest \
  tests/models/configuration/test_ps256_config_compat.py \
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

- Large guided runs should precompute patch caches with `vesuvius.find_patches` before training.
- Guided training now supports `tr_config.compile_policy`:
  - `auto`: guided models compile the inner module, unguided models keep the legacy DDP-wrapper compile path
  - `module`: compile the inner module before DDP wrapping
  - `ddp_wrapper`: preserve the legacy `torch.compile(DDP(model))` path
  - `off`: eager mode
- The guided backbone path is explicitly excluded from compiler capture because full-token guided `ps256` hit an Inductor `BackendCompilerFailed` crash during the first compiled train step.
- Do not expose or rely on a public `channels_last_3d` toggle; the measured gain was negligible relative to plain compile.
- Prefer capped TokenBook prototypes for large training patches; the example config uses `256` for `128^3`.
- The trainer now logs the guide validation visualization twice when guidance is enabled:
  - embedded inside the composite `debug_image`
  - separately as `debug_guide_image`
- Set `tr_config.startup_timing: true` when debugging slow startup or first-step stalls. This logs dataset init, model build, compile, first batch fetch, first forward/backward, and first optimizer step timings.
- On the local RTX 4090, full `256^3` forward inference for both unguided and guided ps256 configs hit OOM, so practical timing comparisons should use smaller patches or larger-memory GPUs.
- The guide panel/render overhead is small relative to model runtime, so it is reasonable to keep both composite and separate guide-image logging enabled.
