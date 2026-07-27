# 3D Fiber DDP And SyncBatchNorm Plan

## Scope

Targets:

- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py`
- `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md`
- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md`
- active planning/status/task-log files

This task is for the 3D fiber training entry point only. Benchmark, prefetch,
Trace2CP visualization, and native Trace2CP CLIs should remain single-process
unless a later task explicitly distributes them.

## Current State

- `fiber_trace_3d.train` has no `torch.distributed`,
  `DistributedDataParallel`, `LOCAL_RANK`, or `WORLD_SIZE` handling.
- Training currently computes `sample_index = (step - 1) * batch_size`, so
  separately launched processes would duplicate samples.
- Checkpoints, TensorBoard, sample sheets, tests, Trace2CP metrics, and stdout
  are not rank-gated.
- The trainer uses normal `BatchNorm3d`. Mixed precision is already handled
  through the local autocast/GradScaler helpers.

## Launch/API Design

- Do not require config changes for DDP. Running under `torchrun` is the
  switch: when `WORLD_SIZE > 1`, the trainer initializes DDP from the standard
  environment variables. Without those variables, the existing single-process
  command behaves as before.
- Default the distributed backend internally: `nccl` for CUDA DDP and `gloo`
  only for CPU tests/smokes. Avoid a required backend config knob.
- Use SyncBatchNorm automatically for DDP CUDA training by converting the
  model with `torch.nn.SyncBatchNorm.convert_sync_batchnorm(...)` before DDP
  wrapping.
- Do not convert BatchNorm modules in ordinary single-process training. Outside
  distributed training SyncBatchNorm would not provide cross-rank statistics,
  and leaving `BatchNorm3d` untouched is the strict behavior-preserving path.
- Keep `training.batch_size` / top-level `batch_size` as the per-rank batch
  size. Effective global training batch is
  `batch_size * distributed_world_size`.
- Do not add gradient accumulation in this task.

## Implementation Plan

1. **Add distributed state helpers**
   - Introduce a small `_DistributedConfig` dataclass:
     `enabled`, `rank`, `local_rank`, `world_size`, `is_main`, `backend`,
     and `device`.
   - Add `_distributed_config_from_env(base_device)` to parse `RANK`,
     `LOCAL_RANK`, and `WORLD_SIZE`.
   - In DDP CUDA mode, call `torch.cuda.set_device(local_rank)` and use
     `cuda:{local_rank}` as the actual trainer device.
   - Add `_distributed_init(...)`, `_distributed_barrier(...)`,
     `_distributed_cleanup(...)`, `_is_main_process(...)`, and
     `_unwrap_model(...)`.

2. **Keep single-process behavior unchanged**
   - If distributed is disabled, return rank 0/world size 1 semantics and keep
     all existing paths identical except for going through helper wrappers.
   - Reject non-training subcommands under multi-rank launch for now, or make
     only rank 0 run them and immediately exit/barrier the others. Prefer an
     explicit error for `--benchmark`, `--prefetch`, and `--trace2cp-vis` when
     `WORLD_SIZE > 1` because they are not distributed training.

3. **Build/load/wrap the model in a safe order**
   - Build the raw model.
   - Convert raw model to `SyncBatchNorm` automatically when DDP CUDA is
     active.
   - Move raw model to the rank-local device.
   - Create optimizer on raw model parameters and load checkpoint state into
     the raw model/optimizer/scaler on every rank.
   - Wrap with `DistributedDataParallel` only after resume loading:
     - CUDA: `device_ids=[local_rank]`, `output_device=local_rank`;
     - CPU test mode: no `device_ids`.
   - Use the DDP-wrapped model only for the training forward/backward path.
   - Use the unwrapped raw model for rank-0-only evaluation, visualization, and
     checkpoint saving to avoid DDP forward collectives on only one rank.

4. **Save and load checkpoints without DDP prefixes**
   - Change `_save_snapshot(...)` to save `_unwrap_model(model).state_dict()`.
   - Keep `_load_snapshot(...)` loading into the raw/unwrapped model before DDP
     wrapping.
   - Keep optimizer hparam reapplication after resume unchanged.
   - Save `grad_scaler` state exactly as the mixed-precision task already does.

5. **Partition the deterministic sample stream by rank**
   - Interpret `step` as the global optimizer step.
   - At training step `step`, rank `r` loads local batch index:
     `global_batch_index = (step - 1) * world_size + r`.
   - Its sample index remains:
     `sample_index = global_batch_index * local_batch_size`.
   - For DataLoader workers, extend `_FiberTrace3DBatchDataset` with
     `batch_index_stride`, so rank `r` yields:
     `start_batch_index + index * world_size`, with
     `start_batch_index = start_step * world_size + r`.
   - Keep loader `stream_index` values unique across ranks. Existing
     deterministic augmentation keyed by `stream_index` then remains valid.
   - `training.max_steps` remains optimizer steps. DDP consumes
     `world_size * batch_size` samples per optimizer step.

6. **DDP loss and metric handling**
   - Keep loss tensors as local rank means for backprop. DDP gradient allreduce
     gives the average gradient across ranks when all local batches are the
     same size.
   - Add `_distributed_mean_loss_dict(...)` for logging/test scalar reporting:
     detach each scalar loss and `all_reduce(SUM) / world_size`.
   - For timing diagnostics, log rank-0 timings plus optional distributed
     average/max values if cheap. Do not block training on detailed timing
     reductions beyond loss scalars in V0.

7. **Rank-gate side effects and synchronize control flow**
   - Only rank 0 creates the TensorBoard writer, writes config text, prints
     routine progress, saves snapshots, evaluates dense test loss, runs
     Trace2CP metrics, and writes train/test sample sheets.
   - Broadcast the generated run date/name from rank 0 or otherwise create a
     shared run directory before rank gating, so all ranks agree on the same
     run metadata.
   - Add barriers around rank-0-only evaluation/checkpoint/visualization
     sections so non-main ranks do not enter the next DDP training forward
     while rank 0 is still doing side work.
   - During rank-0-only eval/vis, call the unwrapped model in `eval()` mode.
     This avoids DDP forward-time collectives and avoids SyncBatchNorm training
     collectives on a single rank.

8. **SyncBatchNorm behavior**
   - When DDP CUDA is active, call
     `torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)` before moving
     to device/wrapping.
   - Validate that automatic SyncBatchNorm conversion is used only with DDP
     CUDA single-device per process.
   - Document that SyncBatchNorm synchronizes BatchNorm mean/variance across
     ranks for each forward pass, so the effective BN batch is the global DDP
     batch, but it adds communication overhead.
   - Document that single-process training keeps the existing `BatchNorm3d`
     modules unchanged. This is equivalent to SyncBatchNorm without a process
     group for statistics, but safer because it avoids unnecessary module-class
     changes.

9. **Mixed precision compatibility**
   - Keep existing autocast context around the training forward/loss.
   - Keep FP16 GradScaler local to each rank; DDP gradients are reduced through
     normal DDP mechanics.
   - Checkpoint scaler state only from rank 0. On resume, every rank loads the
     same scaler state.

10. **Failure handling and cleanup**
    - Wrap training in `try/finally` so DataLoader cleanup, writer cleanup on
      rank 0, and `destroy_process_group()` happen reliably.
    - On rank 0 exceptions during side-effect phases, allow process failure to
      terminate the DDP job rather than leaving other ranks blocked forever.

## Spec Update

Add or update 3D training specs to state:

- DDP is supported only for `fiber_trace_3d.train` training mode.
- DDP is inferred from the standard `torchrun` environment. No config change is
  required to enable DDP.
- `batch_size` is per rank under DDP; effective global batch is
  `batch_size * world_size`.
- DDP partitions the deterministic stream by global optimizer step and rank:
  `((step - 1) * world_size + rank) * batch_size`.
- `training.max_steps` remains global optimizer steps.
- DDP CUDA training automatically converts `BatchNorm3d` to `SyncBatchNorm`
  and synchronizes BN statistics over all ranks.
- Rank 0 owns TensorBoard, checkpoints, tests, and sample visualizations.
- Saved checkpoints do not contain DDP `module.` prefixes and remain loadable
  in single-process training and DDP training when the model architecture
  otherwise matches.
- Non-training subcommands remain single-process.

## Docs Updates

Update `docs/code_structure.md` with:

- Example DDP launch:
  `PYTHONPATH=vesuvius/src:. torchrun --standalone --nproc_per_node=2 -m vesuvius.neural_tracing.fiber_trace_3d.train <config.json>`
- No config key is required for DDP. The launch command controls DDP through
  `torchrun`.
- Clarify local vs global batch size and total DataLoader worker count:
  total workers are roughly `world_size * training.loader_workers`.
- Explain rank-0-only outputs and where checkpoints/TensorBoard are written.
- Note that SyncBatchNorm is automatically used in DDP CUDA training and is
  the correct way to make BatchNorm see the global DDP batch; gradient
  accumulation does not do that.

## Tests

Unit tests in `test_fiber_trace_3d.py`:

- Distributed environment parsing:
  - no env gives disabled rank 0/world size 1;
  - `WORLD_SIZE=1` gives disabled rank 0/world size 1;
  - `WORLD_SIZE=2` with `RANK`/`LOCAL_RANK` enables DDP;
  - incomplete DDP env raises.
- Device selection:
  CUDA DDP maps rank to `cuda:LOCAL_RANK`; CPU DDP uses `gloo` for tests.
- SyncBatchNorm validation:
  - single-process mode leaves BatchNorm modules alone;
  - DDP CUDA mode calls conversion and produces `SyncBatchNorm` modules;
  - DDP CPU test mode does not request SyncBatchNorm.
- Sample partitioning:
  rank 0 and rank 1 produce non-overlapping sample indices for the same global
  optimizer steps and preserve resume offset from `start_step`.
- DataLoader dataset stride:
  `_FiberTrace3DBatchDataset` maps item indices to
  `start_batch_index + index * batch_index_stride`.
- Snapshot state:
  saving a wrapped/fake-wrapped model uses unwrapped state keys, not
  `module.*` keys.
- Rank gating:
  writer creation and snapshot/test/sample-vis helpers are only called on main
  rank in small mocked tests.
- Existing mixed-precision and resume tests still pass.

Optional integration/smoke tests:

- CPU DDP smoke with `torchrun --standalone --nproc_per_node=2` against a tiny
  fixture config with `model_3d.normalization: "none"` and `max_steps: 1`.
- CUDA DDP smoke, with user approval, on a small fixture:
  `torchrun --standalone --nproc_per_node=2 -m vesuvius.neural_tracing.fiber_trace_3d.train <small_config>`
- CUDA DDP smoke verifies automatic SyncBatchNorm conversion when a multi-GPU
  node is available and the configured model uses BatchNorm.

Validation commands after implementation:

```bash
python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'distributed or ddp or sync_batchnorm or mixed_precision or resume'
git diff --check
```

## Changelog Update

- Add a 2026-07-27 entry noting DDP training support, automatic
  SyncBatchNorm for DDP CUDA, rank-partitioned deterministic sample streams,
  rank-0 checkpoint/logging behavior, and single-process-compatible checkpoint
  state.

## Non-Goals

- No gradient accumulation.
- No distributed benchmark/prefetch/Trace2CP visualization modes.
- No change to target semantics, augmentation semantics, model output layout,
  mixed-precision mode semantics, or optimizer hparam resume behavior.
- No automatic global-batch learning-rate scaling; the config learning rate is
  used as written.

## Review Checklist

- Single-process config runs exactly as before.
- DDP ranks never duplicate training batches.
- DDP ranks cannot diverge around rank-0-only evaluation or visualization.
- SyncBatchNorm is automatic for DDP CUDA and absent from normal
  single-process training.
- Checkpoints remain prefix-free and loadable outside DDP.
- Logged loss scalars are distributed averages, not just rank-0 local losses.
