# Task Plan: Training Benchmark And Profiling Mode

## Scope

Add training-only benchmark/profiling command-line modes. This should not
change normal training semantics, sampling order, model architecture, loss,
TensorBoard logging, checkpoints, prefetch, or runner inspection modes.

## Plan

- Extend `train.py` CLI with:
  - `--benchmark`: run 100 train batches by default, skip test evaluation,
    TensorBoard, run-directory creation, and snapshots, then report patch
    samples/s.
  - `--profile`: enable stage timing and per-batch table output. It can be used
    alone or with `--benchmark`; with benchmark it profiles the benchmark run.
- Add a shared benchmark runner that:
  - uses the existing `FiberStrip2DLoader.load_batch` path;
  - performs the same image preparation, supervision building, forward,
    backward, and optimizer step as training;
  - counts CNN patches as `control_points_per_step * strip_z_offset_count`;
  - defaults to 100 batches without changing config files.
- Pass optional profile dictionaries through `load_batch` / `build_sample` so
  existing loader profile blocks collect:
  - coordinate generation from descriptor, line-window, Lasagna normal, strip
    grid, and line-coordinate timing;
  - coordinate augmentation;
  - volume sampling/Zarr read;
  - image/value augmentation.
- Time model stages in `train.py`:
  - forward plus loss;
  - backward plus optimizer step.
- Print:
  - one table header followed by per-batch rows when profiling;
  - a final summary with total patches, elapsed wall time, patches/s, and
    average ms per CNN patch by stage.

## Spec Update

Update `planning/specs.md` to document `train.py --benchmark` and
`train.py --profile`, including the fixed 100-batch default, skipped
side-effects, patch-sample throughput unit, and profiled stage meanings.

## Docs Updates

Update `docs/code_structure.md` to describe the benchmark/profile CLI modes and
the source of the profile stages.

## Testing

- Add focused tests using the existing fake local sampler:
  - loader profile collection records expected stage keys through `load_batch`;
  - benchmark mode returns a summary and does not create run directories.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add a changelog entry for the new training benchmark/profile modes.
