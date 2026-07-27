# 3D Fiber DDP And SyncBatchNorm Task Log

## Implementation Notes

- Added DDP state parsing from `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`; no
  training config keys are needed to enable distributed training.
- Added rank-local CUDA device selection and process-group initialization, with
  `nccl` for CUDA DDP and `gloo` for CPU DDP/unit-test paths.
- Added automatic `BatchNorm3d` to `SyncBatchNorm` conversion for CUDA DDP only.
  Single-process training leaves the model's normal `BatchNorm3d` modules
  unchanged.
- Added rank-partitioned deterministic batch indexing. For optimizer step
  `step`, rank `r` loads batch index `(step - 1) * WORLD_SIZE + r`.
  DataLoader worker datasets support the same partitioning through
  `batch_index_stride`.
- Wrapped the training model with `DistributedDataParallel` after resume
  loading. Rank-0 evaluation, visualization, and checkpointing use the unwrapped
  raw model to avoid DDP collectives in rank-0-only code.
- Changed conditioned decoder loss execution to go through `model.forward(...)`
  with explicit conditioned query kwargs, so DDP sees the conditioned training
  forward pass instead of the loss bypassing the wrapper through custom model
  methods.
- Averaged scalar training losses across ranks before rank-0 logging and
  best-metric comparison.
- Rank-gated TensorBoard, stdout progress, dense test evaluation, Trace2CP
  metric evaluation, sample-sheet visualization, and checkpoint writes.
- Saved snapshots from the unwrapped model so checkpoint keys keep the existing
  non-`module.` format.
- Made `--prefetch`, `--benchmark`, and `--trace2cp-vis` reject
  `WORLD_SIZE > 1` explicitly.

## Deviations / Deferred Items

- Independent plan review was not run because a separate reviewer/subagent was
  not explicitly authorized for this implementation pass.
- A real multi-process `torchrun` CUDA smoke was not run in this sandboxed
  validation pass. The implementation has CPU/unit coverage for rank parsing,
  stream partitioning, SyncBatchNorm selection, and checkpoint unwrapping, plus
  the full existing 3D CPU test file.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/model.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'distributed or ddp or sync_batchnorm or mixed_precision or resume or snapshot or batch_dataset'`
  passed: 12 passed, 118 deselected.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 128 passed, 2 skipped.
- `git diff --check` passed.
