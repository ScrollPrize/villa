# Merge embedding into retrain —2026-09-12

Destination: `fiber-training-retrain` at `b52438d64`.
Source: committed `fiber-training-embedding` at `1f24f17d2`.
The source already includes earlier retrain `b32f31a55`. The other checkout,
`/home/hendrik/villa`, and its uncommitted work were not modified.

## Preserved features

- Incoming dense JSON-fiber supervision and scroll-balanced continuous arc sampling.
- Incoming shared cooperative loading queues, phase admission and startup/memory calibration.
- Incoming GPU-manager responsiveness, process-service separation and recovery fixes,
  including the source branch's already-authorized communication grace behavior.
- Retrain's corrected complete-reference segment metrics and all latest28k/best11k
  benchmark artifacts, source bindings, comparisons and evaluation configurations.

## Conflicts and resolution

There were no production-code/config/test conflicts. Five active planning files
conflicted: task.md, task_plan.md, task_log.md, status.md and changelog.md.
These described different completed tasks, not competing runtime behavior.
Both task/plan/status/log histories were archived verbatim with source commits:

- [Retrain history](merge_embedding_into_retrain_ours_20260912.md).
- [Embedding history](merge_embedding_into_retrain_theirs_20260912.md).

Current planning describes this merge. Both changelog entries and all incoming
specifications were retained. An independent reviewer verified all eight archived
sections, incoming production/specification bytes and retrain benchmark/config
bytes against the two pinned commits. No feature was resolved by selecting an
entire branch wholesale.

## Test-only correction

The incoming source already had one stale test: evaluation_benchmark.py defaults
to `pipeline`, but test_evaluation_benchmark.py expected `dataloader`.
Changed only the assertion to `pipeline`; production behavior is untouched and
explicit tests still cover dataloader, pipeline and sync plus invalid input.
The independent reviewer verified this mismatch and the correction.

## Validation commands

From `/home/hendrik/villa2`, with the existing environment and no GPU allocation:

```bash
GPU_MANAGER_SCHEMA_PYTHON=/usr/bin/python3 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/hendrik/villa2/gpu_manager/src /home/hendrik/.venv_las/bin/python -m pytest --noconftest -q gpu_manager/tests

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2/vesuvius /home/hendrik/villa2' -q \
  vesuvius/tests/neural_tracing/test_arc_sampling.py \
  vesuvius/tests/neural_tracing/test_collection_training_config.py \
  vesuvius/tests/neural_tracing/test_cooperative_integration.py \
  vesuvius/tests/neural_tracing/test_cooperative_setup.py \
  vesuvius/tests/neural_tracing/test_evaluation_benchmark.py \
  vesuvius/tests/neural_tracing/test_evaluation_pipeline.py \
  vesuvius/tests/neural_tracing/test_fiber_trace_3d.py \
  vesuvius/tests/neural_tracing/test_forward_visualization.py \
  vesuvius/tests/neural_tracing/test_memory_calibration.py \
  vesuvius/tests/neural_tracing/test_queued_batches.py \
  vesuvius/tests/neural_tracing/test_sample_broker.py \
  vesuvius/tests/neural_tracing/test_startup_calibration.py \
  vesuvius/tests/neural_tracing/test_ordered_sample_pipeline.py \
  vesuvius/tests/neural_tracing/test_replay_metrics.py
```

Manager tests:141passed in36.13s. Initial fiber selection:374passed,9skipped,
1stale assertion failure in115.98s; full rerun after correction:375passed,
9skipped,2existing Pillow warnings in119.15s. No installations, training,
inference reruns, manager restarts or live process mutations were performed.
GPU-specific coverage and non-Linux platform execution are not part of this
CPU-only merge validation. Existing Pillow deprecation warnings are unrelated.
