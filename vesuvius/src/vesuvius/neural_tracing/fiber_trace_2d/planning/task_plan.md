# Loader Serialization Bottleneck Plan

## Implementation

- Measure the current `loader_example.json --benchmark --load-only --profile`
  path before changes.
- Inspect Python and VC3D sampling boundaries for hidden serialization.
- Isolate whole-batch queueing, `_prepare_sample`, strip-coordinate cache
  loading, and `sample_coord_batch` separately before changing behavior.
- Keep CPU source/cache loading parallel, but remove per-sample CUDA work from
  loader workers:
  - workers should build deterministic sample descriptors and load/crop the
    cached strip source on CPU;
  - after the batch source list is assembled, materialize geometric
    augmentations for the whole batch in one batched GPU step;
  - convert the final augmented coordinate tensor to NumPy once for VC3D.
- Keep the grouped VC3D volume-sampling call unchanged unless measurement shows
  it is still the bottleneck after batched source materialization.
- Preserve deterministic output order and deterministic augmentation parameter
  selection by sample index and strip-z offset.
- Do not tune `loader_example.json` as a substitute for removing the measured
  serialization/contention point.

## Spec Update

- Clarify that loader workers should not run tiny per-sample CUDA coordinate
  augmentation kernels; GPU geometric augmentation should be batched over the
  collected loader batch.
- Clarify that deterministic output order is preserved even when CPU source
  loading completes out of order.

## Docs Updates

- Update `planning/task_log.md` with current-task measurements only.
- Update `planning/status.md` with this task checklist.
- Update `docs/code_structure.md` to describe CPU source loading followed by
  batched GPU coordinate materialization if the implementation changes that
  path.
- Add a changelog line if throughput improves materially.

## Testing

- `python -m py_compile` for touched Python files.
- Focused `test_fiber_trace_2d_loader.py`.
- Established load-only benchmark command before and after the change.
