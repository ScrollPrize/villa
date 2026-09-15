# Refinement merge — 2026-09-15

Current state committed de6fc71e8. Merging fiber-training-refinement c3e0d21a8 into fiber-vischeck. No source-code conflicts. Five planning conflicts: changelog/status/task/task_log/task_plan. Both prior active task sets archived under merge_refinement_20260915; changelog retains both independent sets of entries. Active planning now describes this merge.

Independent integration review approved: incoming training/loader/shared-format files match the refinement branch, while native tracing, mixed inference, fixed-path fitting and embedding integration remain unchanged. Refinement defaults off; embedding and refinement geometry metadata coexist. No semantic fixes were necessary.

Validation:

- `bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests`: passed native replay/tracing tests and 232 Python tests, 8 skipped (83.19 s Python suite).
- `/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/vesuvius/src /home/hendrik/villa2/vesuvius /home/hendrik/villa2' -q vesuvius/tests/neural_tracing/test_refinement.py vesuvius/tests/neural_tracing/test_fiber_trace.py::test_parse_vc3d_fiber_v3_segment_metadata`: 17 passed, 2 skipped (10.48 s).
- Initial refinement command omitted the vesuvius test-package root, causing a `tests.zarr_utils` import/collection failure. Corrected PYTHONPATH as above; no source changes needed.
- `git diff --cached --check`: passed; no unresolved merge paths.

No training, full evaluation, or GPU validation launched for this merge. Existing native build reused because incoming changes do not modify native sources. Merge retains both feature sets and all recorded results.
