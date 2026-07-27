# 3D Fiber DDP And SyncBatchNorm Training

Add proper multi-GPU DDP support to
`vesuvius.neural_tracing.fiber_trace_3d.train`, with automatic synchronized
BatchNorm in DDP CUDA training so the model can use effective BatchNorm
statistics across all GPU ranks.

Requirements:

- Keep normal single-process training behavior unchanged by default.
- Support launch through `torchrun --standalone --nproc_per_node=N ...`.
- Detect DDP directly from `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`; no config
  change should be required to switch from single-process training to DDP.
- Set each DDP process to its local CUDA device.
- Partition the deterministic training sample stream by rank so ranks do not
  train on duplicated batches.
- Preserve existing resume semantics: `training.max_steps` remains optimizer
  steps, and resumed DDP runs continue from `checkpoint_step + 1`.
- Save rank-0 checkpoints without DDP `module.` prefixes.
- Gate TensorBoard, sample visualization, checkpoint writing, test evaluation,
  Trace2CP metrics, and routine stdout to rank 0.
- Keep all ranks synchronized around rank-0-only evaluation/checkpoint/vis
  sections so DDP collectives cannot diverge.
- Use synchronized BatchNorm automatically for DDP CUDA training by converting
  `BatchNorm3d` modules to `SyncBatchNorm` before DDP wrapping, so BatchNorm
  statistics are computed over all ranks.
- Keep literal existing `BatchNorm3d` modules in ordinary single-process
  training; this is behavior-preserving and avoids changing model classes when
  synchronization cannot add information.
- Preserve BF16/FP16 autocast and GradScaler behavior.
- Document that configured `batch_size` is per-rank in DDP, while effective
  global batch is `batch_size * world_size`.
