# Task Log: 3D Fiber PyTorch DataLoader Parallelism

## Implementation

- Removed `_OrderedBatchLoadPipeline` and the `ThreadPoolExecutor`/`Future`
  imports from `fiber_trace_3d.train`.
- Added `_FiberTrace3DBatchDataset`, where each `__getitem__` returns one full
  deterministic `FiberTrace3DBatch`.
- Added `_make_batch_dataloader(...)` using `torch.utils.data.DataLoader` with
  `batch_size=None`, identity collation, persistent workers, and configured
  worker prefetch.
- Each worker lazily creates its own `FiberTrace3DLoader` from the config path
  or synthetic test config. The main process moves returned CPU batches to the
  training device.
- `run_training` and `run_benchmark(... --load-only)` now use the same
  DataLoader path when `training.loader_workers > 0`.
- Replaced the S1A NML 3D config's misleading thread `pipeline_*` keys with
  `loader_workers`, `loader_prefetch_factor`, and `loader_worker_device`.
- Added benchmark-only `cpu_ms` and `cpu_x` columns sampled from `/proc` for
  the main process plus DataLoader worker PIDs during each row.

## Determinism

- DataLoader index `i` maps to batch sample index
  `(start_batch_index + i) * batch_size`.
- The DataLoader is consumed in deterministic index order with `shuffle=false`.
- Augmentation and sample-order randomness remain keyed from deterministic
  global sample indices inside `FiberTrace3DLoader`.

## Deviations / Simplifications / Deferred Items

- A multiprocessing worker unit test using the synthetic in-memory config
  blocked in this environment. The focused test suite now covers the
  whole-batch Dataset/DataLoader collation behavior with `num_workers=0`, and
  real process-worker execution is validated by the approved S1A benchmark
  command.
- CUDA worker-device mode is supported by the code path only in the conservative
  form requested by the task: workers synchronize and return CPU batches. The
  checked-in config uses CPU workers, and CUDA worker-device mode was not
  benchmarked.
- Benchmark CPU accounting is Linux `/proc` based. On platforms without
  `/proc/<pid>/stat`, `cpu_ms` and `cpu_x` print `nan`.

## Validation

- Syntax:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Focused tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  result: `18 passed in 2.64s`.
- Approved load-only benchmark command:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 10`

## Performance Results

- Previous single-batch VC3D coordinate-sampler baseline after warmup:
  median `1359.06 ms`, post-warmup mean `1359.79 ms`.
- Final DataLoader run (`loader_workers=8`, `prefetch_factor=2`,
  `multiprocessing_context=forkserver`):
  - all rows mean/median/min/max:
    `1989.75 / 169.02 / 124.12 / 17115.35 ms`
  - excluding first worker-startup row mean/median/min/max:
    `309.13 / 163.80 / 124.12 / 1131.87 ms`
- The first benchmark row includes DataLoader worker startup and worker-local
  loader construction. `load_ms` includes DataLoader wait plus main-process
  batch transfer; `to_device_ms` reports the transfer portion.
- Added per-row CPU timing and reran the same benchmark. Post-startup rows
  reported weighted `cpu_x=2.44`, median row `cpu_x=3.02`, and mean row
  `cpu_x=3.02`. This confirms the current 8-worker DataLoader path is still
  CPU-underutilized relative to a 32-logical-CPU machine.
