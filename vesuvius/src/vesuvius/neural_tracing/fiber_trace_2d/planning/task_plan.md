# 3D Fiber Training Mixed Precision Plan

## Current State

- `fiber_trace_3d.train` runs model forward/loss/backward in default FP32.
- The trainer preserves BatchNorm semantics by passing the configured batch
  through the model as one batch.
- The active conditioned decoder path is memory-heavy because it keeps the full
  U-Net activations plus dense zero/random conditioned-decoder outputs.
- There is no AMP/autocast config, no GradScaler state in snapshots, and no
  precision mode in stdout/TensorBoard config summaries.

## Implementation Plan

- Add a small trainer-local mixed-precision config helper:
  - `training.mixed_precision: "off" | "bf16" | "fp16" | "auto"`;
  - booleans map to `bf16`/`off` for convenience;
  - `auto` enables BF16 on CUDA when supported, otherwise FP16 on CUDA, and
    stays off on CPU;
  - explicit `fp16` requires CUDA and uses `torch.amp.GradScaler`;
  - explicit `bf16` uses autocast without GradScaler.
- Refactor `_forward_loss` enough to separate tensor loss construction from
  backward so backward is executed outside the autocast context.
- Wrap train, dense test, benchmark, and sample-sheet model/loss forwards in
  the configured autocast context.
- Save/load GradScaler state in snapshots when an enabled scaler exists, while
  remaining compatible with old snapshots that do not contain scaler state.
- Print and log the effective mixed precision mode with the run config.
- Update the two active S1A conditioned configs to set
  `training.mixed_precision: "bf16"`.

## Spec Update

- Add 3D training precision semantics:
  - mixed precision is an autocast compute option, not a micro-batching mode;
  - configured `batch_size` remains the BatchNorm batch;
  - BF16 is the active large-run default in the S1A conditioned configs;
  - FP16 uses GradScaler and snapshots include scaler state when available.

## Docs Updates

- Update `docs/code_structure.md` training/model sections with the mixed
  precision config and BatchNorm constraint.
- Update `planning/changelog.md` with the BF16/AMP trainer support.

## Tests

- Add unit coverage for mixed precision config parsing and validation.
- Add a smoke test that training loss tensors can be computed inside the
  disabled/default autocast path on CPU.
- Run:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run focused tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'mixed_precision or conditioned'`
- Run full 3D test module if time permits:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check`.

## Review Notes / Assumptions

- This does not implement gradient accumulation, activation checkpointing, or
  decoder-loss chunking.
- BatchNorm behavior is preserved because no internal micro-batches are
  introduced.
- The independent-agent review step from the local workflow requires explicit
  authorization with the available tools, so I will do a local plan/spec
  consistency review and record that deviation.
