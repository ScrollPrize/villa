# Task Plan: 10-Block 64-Channel ResNet Direction Model

## Scope

Change the V0 2D direction model architecture and defaults. This does not
change sampling, prefetch, augmentation, loss semantics, output encoding, runner
inspection modes, or checkpoint file layout.

## Plan

- Replace the current plain Conv/GroupNorm/SiLU stack in `model.py` with a
  residual CNN:
  - input projection from `in_channels` to `hidden_channels`;
  - `depth` residual blocks at constant hidden width;
  - each block uses two 3x3 convolutions, GroupNorm, and SiLU, with identity
    skip connection;
  - final 1x1 projection to the existing two Lasagna direction channels.
- Change default model knobs to `hidden_channels=64` and `depth=10`.
  - `FiberStripDirectionModelConfig` defaults.
  - `FiberStripTrainingConfig` defaults and raw-config fallback.
  - runner checkpoint fallback for older checkpoints missing config metadata.
  - checked-in example config.
- Keep explicit user/test config values honored, including smaller test models.
- Add focused model tests for output shape/range and residual-block count.

## Spec Update

Update `planning/specs.md` to state that the V0 model is a 10-block,
64-channel residual CNN by default and still outputs the Lasagna ambiguous
two-cos-channel direction encoding.

## Docs Updates

Update `docs/code_structure.md` to describe the ResNet model and 10/64 default
training knobs.

## Testing

- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add a changelog entry because this changes the default model architecture.
