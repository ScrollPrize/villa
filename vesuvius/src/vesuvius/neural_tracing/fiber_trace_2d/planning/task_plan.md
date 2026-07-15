# 3D Fiber Loader Performance Rewrite Plan

## Baseline First

- Run the current 3D load-only benchmark before changing code:
  `PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_3d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 10`
- Record per-batch and summary timings in `planning/task_log.md`.
- If the benchmark cannot run because data/cache is unavailable in the current
  environment, record that explicitly and use the user's reported timings as
  the baseline until a local run is possible.

## Loader Architecture

- Replace the normal 3D input path built from `_read_raw_block(...)` plus torch
  `grid_sample` with explicit-coordinate sampling through the existing 2D
  `CoordinateSampler` / `Vc3dCoordinateSampler` abstraction.
- Keep 3D input semantics CP-centered and axis-aligned in output patch space.
  This is not a fiber strip: output voxels are still a regular ZYX patch around
  the CP. Only the volume read/sampling mechanism changes.
- Extend/reuse the 2D sampler factory so 3D records own a VC3D-backed sampler
  instead of a plain zarr array for production remote volumes.
- Keep a NumPy/zarr sampler fallback only for tests or explicitly synthetic
  in-memory arrays, not for the S1A production config.
- Convert 3D output-to-source coordinates to base-volume ZYX coordinates before
  calling `sample_coord_batch`, preserving `base_volume_scale` semantics.

## 3D Coordinate Maps

- Build a concrete 3D map object per sample, analogous to the 2D fused map:
  - `backward_source_zyx`: output patch voxel -> selected-level source-volume
    coordinate for VC3D image sampling;
  - `forward_map_zyx`: selected-level source-volume coordinate -> output patch
    coordinate for transformed CP/line/target generation.
- Construct both map directions directly from augmentation parameters. Do not
  derive one direction from the other with a search, nearest lookup, dense
  inverse, or iterative solver.
- Keep current supported geometry semantics: CP shift, arbitrary 3D rotation,
  isotropic scale, axis flips, and the existing explicitly paired smooth
  displacement modes.
- Generate runtime 3D augmentation coordinate maps on CUDA for speed, including
  in loader worker processes. Each worker owns its own CUDA context/stream and
  uses it only for coordinate-map construction.
- After CUDA coordinate-map generation, copy the final output-to-source
  coordinates to CPU before VC3D sampling. CUDA tensors must not be sent between
  worker processes and the main training process.
- For VC3D sampling, flatten the 3D coordinate volume into a coordinate image
  shape accepted by `sample_coord_batch`, e.g. `[D*H, W, 3]`, then reshape the
  returned image and valid mask back to `[D,H,W]`.
- Keep value-only augmentations after sampling as torch tensor operations.

## Search-Free Target Generation

- Remove the dense nearest-segment search over every patch voxel from the main
  NML target path.
- Map clipped fiber-line segments into output patch coordinates using the
  prebuilt forward map.
- Rasterize/draw the segment presence and direction target directly from those
  transformed output-space segments:
  - use a 3D line/segment rasterization or swept-sphere/capsule drawing helper
    around each transformed segment;
  - produce the positive presence mask from the configured radius;
  - write/accumulate the local segment tangent for direction supervision at
    drawn voxels;
  - keep presence negative mask as valid interior excluding the configured CP
    edge margin.
- Keep non-NML sources CP-only as currently specified, but also build their CP
  target from direct transformed CP/tangent geometry rather than dense searches.
- Add assertions/tests that target generation does not call the old
  nearest-segment full-grid search path.

## Runtime Parallelization

- Add a 3D loader-worker configuration mirroring the 2D style:
  - `loader_workers`: runtime sample-loading workers; `0` means all logical
    cores, `1` is serial/debug.
  - `loader_cuda_device`: CUDA device used by loader workers for coordinate
    map construction; default to the training CUDA device when CUDA is
    available.
  - `pipeline_enabled`, `pipeline_depth`, and `pipeline_workers` under
    `training` for overlapping future batch loading with model training.
- Use process-based parallelism for sample preparation and VC3D sampling rather
  than relying on `ThreadPoolExecutor` for Python-heavy work.
- Each loader worker process:
  - initializes its own CUDA context/stream for coordinate-map generation;
  - opens its own VC3D sampler/volume handle;
  - uses a bounded per-worker VC3D memory cache budget to avoid duplicating an
    unbounded cache;
  - returns CPU arrays/tensors only.
- Keep deterministic ordering by assigning deterministic raw sample indices to
  worker tasks and consuming completed work in sample-index order.
- Avoid duplicated large immutable metadata where practical:
  - pass compact record descriptors/config to workers;
  - let each worker open volume handles;
  - do not keep per-CP dense coordinate caches.
- Keep model execution in the main process. Worker CUDA usage is limited to
  coordinate generation, then coordinates are synchronized/copied to CPU for
  VC3D. The main process stacks worker CPU outputs and transfers the final
  batch to the configured training device.
- Prefer a custom `torch.multiprocessing`/spawned worker pool over a stock
  PyTorch `DataLoader` if DataLoader worker lifecycle makes persistent CUDA
  stream, VC3D sampler, and ordered task consumption awkward. A DataLoader
  implementation is acceptable only if it keeps the same worker-owned CUDA
  context, no-CUDA-tensor-crossing, deterministic-order contract.
- Add clear profiling columns for loader wall time, summed worker time, process
  CPU/thread factor, VC3D sample time, target draw time, value augmentation, and
  forward/backward time.

## Prefetch

- Change 3D prefetch to use the same coordinate-map construction used by
  training, then call `CoordinateSampler.chunk_requests_for_coords(...)`.
- Prefetch must request chunks for the conservative maximum augmentation
  envelope, independent of the particular random augmentation drawn for a
  training step.
- Keep `--prefetch-steps 0` as the full deterministic CP dataset sentinel.
- Preserve current S1A behavior: prefetch base-volume chunks only, not Lasagna
  manifest channels.

## Config Updates

- Add 3D config keys:
  - top level `loader_workers`;
  - top level `loader_cuda_device`;
  - optional per-worker `volume_cache_memory_mib` behavior if not already
    sufficient;
  - training `pipeline_enabled`, `pipeline_depth`, `pipeline_workers`.
- Set S1A NML 3D defaults conservatively:
  - `loader_workers: 0` for all logical cores once verified;
  - `pipeline_enabled: true`;
  - bounded `pipeline_depth` such as `2` or `3` to avoid excessive 192^3 memory.
- Keep `batch_size` as real model batch size. Do not reintroduce micro-batching.

## Testing

- Add/extend unit tests in `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`:
  - coordinate-map sampling returns the same synthetic volume values as the old
    axis-aligned no-augmentation path on an in-memory volume;
  - VC3D/coordinate-sampler flatten/reshape preserves `[D,H,W]` ordering;
  - transformed CP and line segment targets are generated without full-grid
    nearest-segment search;
  - deterministic batch order is unchanged under `loader_workers=1` vs multiple
    workers;
  - prefetch chunk requests come from coordinate dependencies, not bbox zarr
    slicing.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Performance Validation

- After implementation, rerun the same load-only benchmark command used for the
  baseline.
- Also run a short full benchmark:
  `PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_3d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --benchmark-batches 10`
- Report before/after:
  - mean, median, min, max load-only batch time;
  - full batch load vs forward/backward time;
  - samples/s;
  - worker parallelism factor and CPU utilization notes.

## Spec Update

- Replace the current 3D spec allowance for loading an oversized zarr block plus
  torch `grid_sample` with the new requirement that normal 3D training uses
  explicit coordinate maps and the VC3D blocking coordinate sampler.
- Add that 3D target generation must draw/rasterize transformed clipped
  segments directly and must not perform full-patch nearest-segment searches.
- Add 3D runtime parallelism requirements and config keys.
- Clarify that production 3D prefetch is coordinate-dependency based, matching
  training coordinate maps.

## Docs Updates

- Update `docs/code_structure.md`:
  - describe the 3D coordinate-map loader path;
  - identify VC3D sampler use and the test-only NumPy fallback;
  - document the process-worker training loader and pipeline;
  - describe search-free segment drawing for labels.
- Update `planning/local_development.md` if the 3D benchmark command or VC3D
  binding requirement differs from the current documented 2D workflow.

## Changelog

- Add a 2026-07-15 entry for the 3D loader rewrite from zarr crop/grid-sample
  to VC3D coordinate sampling, process-parallel loading, and search-free target
  drawing.

## Explicit Non-Goals

- Do not change 2D fiber training semantics.
- Do not add 3D fiber-aligned strips or 3D slicing as model input.
- Do not change model precision, patch size, or loss weights to fake a speedup.
- Do not silently skip unsupported VC3D/binding cases; fail loudly with the
  missing API or setup requirement.
