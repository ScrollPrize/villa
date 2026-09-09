# Task log: sample-based Fiber3D progress

- User requested cumulative global samples as default progress so batch-size comparisons are meaningful.
- Independent plan review approved sample-axis writer conversion, persisted resume offset, explicit legacy override, and unchanged update-based scheduling. Updated specs to describe the new default.
- Implemented one TensorBoard boundary adapter for all scalar/image events; metadata text stays at zero. Training, dense-test, Trace2CP and test-timing console metrics show samples and update.
- All four training snapshot paths save samples_seen and effective optimizer_global_batch_size. Resume applies subsequent updates at the new batch size; old snapshots infer only a known global batch or require training.resume_samples_seen.
- Updated profile_experiment console parsing to recognize both historical step and new samples/update formats while retaining update units in tuning reports.
- Real TensorBoard event tests verify train/test/timing/image axes and text at zero; checkpoint tests cover repeated batch changes and legacy ambiguity.
- Independent implementation review approved; no blockers. Scheduling, numerical computation, data sampling and diagnostic/profile JSON units are unchanged. Historical event files remain unchanged and require separate conversion for overlay; no training process was launched or restarted.
- Focused validation completed: 10 passed, 276 deselected in 3.08 seconds. git diff --check passed.
- Broad suite invocation (same environment below, both test files, no -k) stalled after partial progress and was interrupted; do not claim a completed full-suite result. Focused validation uses:
  PYTHONPATH=/home/hendrik/villa2/vesuvius/src:/home/hendrik/villa2/vesuvius:/home/hendrik/villa2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d_sample_progress.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'sample_progress or snapshot or test_timing or linear_warmup or scalar_metrics_due'
