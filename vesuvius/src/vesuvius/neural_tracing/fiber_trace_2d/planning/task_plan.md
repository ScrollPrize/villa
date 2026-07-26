# 3D Resume Optimizer Config Override Plan

## Current State

- `run_training(...)` builds AdamW from `training.learning_rate` and
  `training.weight_decay`.
- If `--resume`, `training.resume`, or top-level `resume` is set, `_load_snapshot`
  then restores the optimizer state dict from the checkpoint.
- PyTorch optimizer state dicts include param-group hyperparameters, so the
  checkpoint currently overwrites the current config LR and weight decay.
- Other trainer settings are read from the current config after resume. Model
  architecture still must be compatible with the checkpoint state dict.

## Implementation

- Add a small helper that extracts supported AdamW hyperparameters from the
  `training` mapping:
  - `lr` from `training.learning_rate`, defaulting to `1e-3`;
  - `weight_decay` from `training.weight_decay`, defaulting to `0.0`.
- Use that helper for fresh AdamW construction.
- Extend `_load_snapshot(...)` or its call site so that after optimizer
  `load_state_dict(...)`, the current config hyperparameters are written back
  into every optimizer param group.
- Preserve loaded optimizer state entries such as AdamW moments and step
  counters.
- Print effective optimizer hyperparameters at train startup and record them in
  TensorBoard config text when TensorBoard is enabled.

## Spec Update

- Update `planning/specs.md` to state that 3D resume restores optimizer state
  but current config optimizer hyperparameters override checkpoint param-group
  values after load.

## Docs Updates

- Update `docs/code_structure.md` in the 3D trainer section so users know they
  can change `training.learning_rate` and `training.weight_decay` when resuming.

## Tests

- Add a focused unit regression test:
  - create a checkpoint with AdamW state at one LR/weight decay;
  - load it into a new optimizer configured with different LR/weight decay;
  - verify moment state and step are still present;
  - verify every param group uses the current config LR/weight decay.
- Run:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run focused tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'optimizer or resume'`
- Run the full 3D test file:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check`.

## Changelog

- Add a 2026-07-26 changelog bullet noting that resumed 3D training preserves
  optimizer state while honoring current config LR/weight decay.

## Review Notes / Assumptions

- This task does not add scheduler support.
- This task does not make incompatible model architecture changes loadable; the
  current config still builds the model, and checkpoint weights must match.
- The independent-agent review step from `AGENTS.md` cannot be performed with
  the available multi-agent tool unless the user explicitly asks for delegation.
  I will perform a local plan/spec consistency review and log that constraint.
