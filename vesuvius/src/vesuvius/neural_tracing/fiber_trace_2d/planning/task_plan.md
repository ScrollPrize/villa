# 3D Fiber PyTorch DataLoader Parallelism Plan

## Scope

This task fixes the missing runtime parallelism from the previous 3D loader
performance pass. The VC3D coordinate-sampling path and search-free 3D target
rasterization remain the intended loader semantics. The change here is the
training/benchmark batch-loading execution model.

## Current Problem

- `fiber_trace_3d.train` currently contains `_OrderedBatchLoadPipeline`, a
  `ThreadPoolExecutor` wrapper around complete `loader.load_batch(...)` calls.
- That does not satisfy the requirement to use PyTorch data-loader process
  parallelism and does not give real process isolation for Python-heavy loader
  work.
- The approved load-only benchmark now measures the faster VC3D coordinate
  sampling path, but still effectively measures one synchronous batch load at a
  time unless the thread queue happens to overlap work.

## Implementation Plan

### 1. Remove The Thread Pipeline

- Delete `_OrderedBatchLoadPipeline` from `fiber_trace_3d.train`.
- Remove the `ThreadPoolExecutor` / `Future` imports from `fiber_trace_3d.train`.
- Do not leave a silent thread fallback behind the same config names.
- Keep the old single-process synchronous path only for `num_workers <= 0` or
  an explicit serial/debug configuration.

### 2. Add A Batch-Level PyTorch Dataset

- Add a small map-style dataset class in `fiber_trace_3d.train`, e.g.
  `_FiberTrace3DBatchDataset`.
- Dataset items are whole training batches, not individual CP patches:
  - `__getitem__(batch_number)` computes
    `sample_index = (start_step + batch_number) * config.batch_size`;
  - it calls `worker_loader.load_batch(sample_index, sample_mode="random",
    device=worker_device)`;
  - it returns one `FiberTrace3DBatch`.
- Use `batch_size=None` on `torch.utils.data.DataLoader` so PyTorch does not
  apply default collation to the custom dataclass.
- Add a stable `__len__` for finite training/benchmark loops. For normal
  training, length is the requested number of remaining steps.

### 3. Lazy Per-Worker Loader Construction

- The dataset must store only picklable construction data:
  - config path or raw config mapping;
  - sample mode;
  - worker output device policy;
  - deterministic start batch index.
- It must not store an already-open `FiberTrace3DLoader`.
- In each worker process, lazily create a `FiberTrace3DLoader(load_config(path))`
  the first time `__getitem__` runs.
- That ensures each worker opens its own zarr/VC3D volume handles and sampler
  state inside the worker process.
- The main process may keep its own loader for metadata, test evaluation, and
  serial fallback, but worker loaders must be independent process-local objects.

### 4. Worker Device And CUDA Boundary

- Default worker output is CPU tensors. Main training moves the whole returned
  `FiberTrace3DBatch` to the training device with `batch.to(device)`.
- Add config support for worker-side coordinate device only if currently
  supported by the 3D loader API:
  - `training.loader_worker_device` or top-level `loader_cuda_device`;
  - default should be `"cpu"` initially if CUDA worker behavior is not verified;
  - if set to CUDA, DataLoader must use spawn-compatible multiprocessing and
    workers must return CPU tensors after synchronizing/copying.
- Do not pass CUDA tensors from worker processes to the main process.
- If CUDA-in-worker is not implemented in this task, log it as explicitly
  deferred; do not imply it exists.

### 5. DataLoader Construction

- Add helper `_make_batch_dataloader(...)` in `fiber_trace_3d.train`.
- Effective worker count:
  - `training.loader_workers` first;
  - fallback to top-level `loader_workers` if added;
  - `0` means serial/debug path or all logical cores only if explicitly
    documented and implemented;
  - positive values become `num_workers`.
- Use:
  - `persistent_workers=True` when `num_workers > 0`;
  - `prefetch_factor=training.pipeline_depth` or a new
    `training.loader_prefetch_factor`;
  - `multiprocessing_context="spawn"` for CUDA-worker compatibility, otherwise
    the safest available context for local Linux.
- Use an identity collate function if needed so a `FiberTrace3DBatch` passes
  through unchanged.
- Optional: add `pin_memory=True` only if `FiberTrace3DBatch` supports a
  `pin_memory()` method or if tests show PyTorch handles the custom dataclass
  correctly. Otherwise leave it false and document the reason.

### 6. Integrate Training Loop

- Replace the current direct `loader.load_batch(...)` call in `run_training`
  with an iterator over the DataLoader when worker count is positive.
- For each step:
  - get next CPU `FiberTrace3DBatch` from the DataLoader;
  - move it to `device` in the main process;
  - run forward/backward/optimizer as before.
- Preserve all existing scalar/test/snapshot/TensorBoard behavior.
- Timing:
  - `load_ms` becomes time spent waiting for the next DataLoader batch plus
    main-process device transfer;
  - add separate `batch_to_device_ms` if useful and cheap;
  - keep `timing/load_ms` for compatibility.

### 7. Integrate Benchmark / Load-Only

- `run_benchmark(..., load_only=True)` must use the same DataLoader path and
  worker settings as training unless explicitly forced serial.
- This is critical: load-only benchmark is how we validate parallel loader
  throughput without model work.
- Report rows using the same existing table shape, but clarify that `load_ms`
  is DataLoader wait plus batch transfer.
- Use the exact approved command for comparison:

  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 10`

### 8. Config Updates

- Add checked-in config keys under `training`:
  - `loader_workers`;
  - `loader_prefetch_factor`;
  - optionally `loader_worker_device`.
- Remove or deprecate the misleading thread-pipeline-only keys if they no
  longer apply:
  - `pipeline_workers`;
  - `pipeline_depth`;
  - `pipeline_enabled`.
- If keeping these names for compatibility, redefine them clearly as DataLoader
  settings and update docs/specs. Do not leave them documented as
  ThreadPool-based.
- Set S1A NML 3D defaults conservatively first:
  - `training.loader_workers`: start with a positive value such as `4` or `8`;
  - `training.loader_prefetch_factor`: `2`;
  - keep `batch_size` unchanged.

### 9. Determinism Guarantees

- Worker count must not affect data order:
  - DataLoader index `i` maps to the same training batch sample index every
    time;
  - DataLoader consumes in index order, even if worker completion is out of
    order internally;
  - augmentation seeds remain keyed by the global sample index, not by worker id
    or completion order.
- Add a regression test that compares serial and multi-worker batch
  `sample_indices`, `record_indices`, `control_point_indices`, and a small
  tensor checksum on synthetic data.

### 10. Failure Semantics

- Data-quality skips already handled inside loader paths may remain skips where
  implemented.
- Infrastructure errors in a DataLoader worker must fail the run loudly with
  the original exception context.
- No silent fallback from process DataLoader to thread loading on worker
  failure.

## Spec Update

- Add to `planning/specs.md` that 3D training runtime parallelism uses
  `torch.utils.data.DataLoader` worker processes for batch loading.
- State that each DataLoader worker owns its `FiberTrace3DLoader` and VC3D
  sampler handles.
- State that worker outputs are CPU batches and main training transfers to the
  training device.
- Clarify that the previous thread-backed ordered pipeline is not the intended
  3D loader-parallelism implementation.
- Clarify the semantics of `training.loader_workers` and
  `training.loader_prefetch_factor`.

## Docs Updates

- Update `docs/code_structure.md`:
  - replace the thread queue description for `fiber_trace_3d.train` with the
    PyTorch DataLoader worker process design;
  - document worker-local loader construction and deterministic whole-batch
    indexing;
  - document what `load_ms` means in benchmark/training output.
- Update `planning/local_development.md` only if the benchmark command changes.
  The approved 3D benchmark command should not change.

## Testing

- Run syntax check:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py`
- Run focused tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Add/adjust tests for:
  - DataLoader serial vs multi-worker deterministic order;
  - DataLoader item passes a complete `FiberTrace3DBatch` without default
    collation corruption;
  - synthetic multi-worker load-only path does not require remote VC3D.

## Performance Validation

- Rerun the exact approved load-only benchmark command.
- Compare against the current after-change baseline from the prior task:
  - median `1359.06 ms` per 192^3 patch;
  - mean excluding first warmup batch `1359.79 ms`.
- Report mean, median, min, max for the new DataLoader path.
- If DataLoader worker startup dominates the first batch, report both full
  10-batch numbers and post-warmup numbers.
- If the benchmark does not improve, inspect whether workers are actually
  active before claiming success.

## Changelog

- Add a 2026-07-15 changelog line that 3D training/benchmark loading now uses
  PyTorch DataLoader process workers rather than a thread-backed queue.

## Explicit Non-Goals

- Do not change model architecture, losses, patch size, precision, or
  augmentation semantics to fake throughput.
- Do not change the VC3D coordinate-sampling loader semantics from the previous
  patch.
- Do not implement a separate custom multiprocessing queue unless PyTorch
  DataLoader is proven unusable and that deviation is approved/logged first.
- Do not silently keep the thread-backed pipeline as the default.
