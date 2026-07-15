# Process-Level Startup Fiber Geometry Preload Plan

## Findings

- The current task files incorrectly describe a same-process thread-local
  handle cache. That is not the desired direction.
- Startup compact geometry must be precomputed during loader construction, not
  lazily during training.
- The useful existing pieces should stay:
  - deterministic sample ordering;
  - compact in-RAM fiber-line geometry store;
  - CP-window range filtering;
  - vectorized Lasagna normal sampling;
  - `loader_workers=0` meaning all logical CPU cores.
- The wrong pieces to remove are:
  - `_RecordOpenSpec`;
  - `self._worker_local`;
  - `_record_for_index(... use_worker_local=...)`;
  - descriptor-index plumbing whose only purpose is thread-local record handle
    lookup;
  - tests/docs that assert per-thread worker-local records.

## Goals

- Build all compact fiber-line geometry at `FiberStrip2DLoader` startup.
- Parallelize that startup build across independent worker processes.
- Each worker process opens its own base-volume/Lasagna channel handles.
- The parent process stores the returned compact geometries in original record
  order.
- Runtime batch loading uses the shared compact store exactly as before.
- Keep serial `loader_workers=1` as the reference/debug path.
- Keep `loader_workers=0` as all logical cores.

## Implementation Plan

### 1. Replace The Wrong Planning/Docs State

- Replace `planning/task.md`, `planning/task_plan.md`, `planning/status.md`,
  and `planning/task_log.md` with this process-level task.
- Do not update `specs.md` until implementation is underway; the plan below
  lists the required spec deltas.

### 2. Revert Thread-Local Loader Handle Experiment

- Remove `_RecordOpenSpec`.
- Remove `record_open_specs` from `FiberStrip2DLoader.__init__`.
- Remove `self._worker_local`.
- Restore `_load_records()` to return only `list[_Record]`.
- Restore `clone()` to reuse fiber metadata and refresh only per-clone samplers
  via `_clone_records_with_local_samplers(...)`.
- Remove `_record_open_specs_from_records`, `_open_records_from_specs`,
  `_open_record_from_spec`, and `_record_for_index`.
- Remove `descriptor_indices` and `use_worker_records` parameters from:
  - `build_strip_source`;
  - `build_top_strip_source`;
  - `_prepare_sample`;
  - `_prepare_top_sample_from_side_sample`.
- Restore parallel `load_batch` and `load_top_batch_for_batch` to use the
  normal descriptor/sample paths. Runtime CP-level threading can remain, but it
  is not the startup preload solution.

### 3. Add Picklable Process Jobs For Startup Geometry

- Add a small top-level job dataclass, e.g. `_GeometryBuildJob`, containing:
  - `record_index`;
  - `fiber`;
  - `volume_path`;
  - `volume_scale`;
  - `volume_spacing_base`;
  - `fiber_identity`;
  - `dataset_config`;
  - the loader config fields required to open volumes/channels;
  - `source_shape_hw`.
- Add a top-level worker function so `ProcessPoolExecutor` can pickle it.
- In the worker process:
  - open the dataset volume from the job config;
  - open Lasagna manifest channels from the job config;
  - create a minimal `_Record` for geometry construction;
  - build compact geometry using the same CP-window filtering and vectorized
    Lasagna normal sampling as the serial path;
  - return `(record_index, geometry, memory_bytes)`.
- Do not create a VC3D coordinate sampler in the worker unless geometry code
  actually needs it. Startup geometry should only need volume shape/channel
  data and fiber metadata.

### 4. Keep One Serial Reference Path

- For `loader_workers <= 1`, call `_build_geometry_for_record(...)` directly
  in the parent process using canonical records.
- This path remains the easiest debugger and correctness reference.

### 5. Use `ProcessPoolExecutor` For Startup

- In `_initialize_fiber_line_geometry_store()`:
  - compute `worker_count = min(loader_workers, total_records)`;
  - submit one geometry job per record when `worker_count > 1`;
  - collect futures in the parent;
  - store results by original `record_index`;
  - update progress from the parent only.
- On `KeyboardInterrupt`, cancel pending futures and shut the process pool down
  with `wait=False` where supported.
- Do not let worker processes print progress.

### 6. Preserve Determinism And Store Sharing

- Returned geometry can complete out of order, but parent insertion must use
  `record_index`.
- `by_fiber_identity`, `valid_control_points`, `skipped_control_points`, and
  `memory_bytes` must be computed in the parent from returned geometry.
- Cloned loaders must receive the already-built store by reference.
- Training/prefetch/runner sample streams must see the same CP validity and
  ordering as the serial path.

### 7. Tests

- Remove thread-local tests:
  - parallel startup uses worker-local records;
  - parallel load batch uses worker-local records;
  - clone refreshes all opened I/O handles.
- Restore clone test expectations:
  - clone shares fiber and volume/channel metadata where appropriate;
  - clone refreshes sampler;
  - clone shares compact geometry store.
- Keep tests for:
  - `loader_workers=0` maps to logical CPU count;
  - negative `loader_workers` is rejected;
  - serial vs parallel startup geometry equality.
- Add or adjust tests for process startup:
  - `loader_workers=2` startup produces the same compact geometry as
    `loader_workers=1`;
  - store ordering remains deterministic even when parallel build completes out
    of order if a local fake executor can cover that without multiprocessing
    brittleness.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

### 8. Benchmark

- Reuse the established command where possible:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile`
- Report startup wall time, CPU factor if available, compact store memory,
  valid/skipped CP counts, and load-only throughput.

## Spec Update

Update `planning/specs.md` after code changes:

- Startup compact geometry preload is process-level parallel when
  `loader_workers > 1`.
- Worker processes independently open zarr/VC3D/Lasagna handles and return only
  compact geometry.
- The parent owns the single shared compact geometry store.
- `loader_workers=1` is the serial/debug path.
- `loader_workers=0` means all logical cores.
- No thread-local record handle cache is part of the startup preload design.

## Docs Updates

- Update `docs/code_structure.md` to describe process-level startup geometry
  preload and parent-owned store assembly.
- Update `planning/local_development.md` only if benchmark/development commands
  or workflow notes change.
- Update `planning/changelog.md` with a one-line durable entry after
  implementation.

## Risks / Checks

- Multiprocessing requires picklable job data. Do not pass opened zarr arrays,
  VC3D sampler objects, locks, or executor objects to workers.
- Process startup may increase transient memory because each process opens its
  own handles. Keep cache budgets explicit and measure.
- On platforms using spawn, worker functions must be module-level.
- The process path must not silently fall back to lazy geometry construction.
