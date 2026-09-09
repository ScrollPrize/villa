# Task log: integrate Fiberlets development into Fiber3D training

## 2026-09-09

- The user committed the previously dirty Fiber3D DDP-loss work as
  `dca6d0f2b` (`Normalize Fiber3D losses across DDP batches`), leaving the
  `fiber-training` worktree clean before integration.
- Refreshed `origin/fiberlets-development-integration` from `f0b824523` to
  `b660837d9`; the remote tip already merges a newer `main`.
- Exact merge base: `b218ace879d3239a63047e1e2039f327929d283e`.
  Divergence is 17 commits unique to `fiber-training` and 206 unique to the
  integration branch.
- Selected a merge commit rather than rebasing. This preserves the identities
  of both substantial histories and avoids replaying the training series across
  206 integration commits.
- The non-mutating three-way preview reports overlapping changes in 11 files.
  Text conflicts are expected in `docs/code_structure.md`, `planning/specs.md`,
  `planning/changelog.md`, and the four transient current-task documents.
  `lasagna/tiled_predict3d.py`,
  `lasagna/tests/test_preprocess_cos_omezarr.py`, and native
  `ChunkCache.hpp/.cpp` auto-merge but require explicit semantic review.
- Independent plan review accepted the merge strategy and overlap
  classification but blocked execution until three controls were explicit:
  commit the overlapping planning edits before merge, lock the merge to the
  reviewed `b660837d9` object while verifying its remote ref has not moved, and
  incrementally rebuild merged native sources before executing cache tests.
  The plan now includes all three controls.
- Plan re-review required the native validation set to include the two direct
  local-feature owners, `test_volume_extras` and
  `test_chunked_plane_sampler_fallback`; both are now mandatory alongside the
  general cache/sampler tests.
- Independent plan re-review approved the locked-SHA merge, clean-worktree
  preflight, 11-file overlap audit, fresh native rebuild, and focused test set
  with no remaining blocker.
- Committed the approved planning preflight as `821c964a0`, leaving a clean
  worktree. Re-verified the remote-tracking ref still equaled reviewed object
  `b660837d9ef8c0d59eb7783b1dee588d6f382529`; updated divergence was 18 local
  commits and 206 incoming commits.
- Started a no-commit/no-fast-forward merge of that exact object. The actual
  result exactly matched the preview: seven textual conflicts and four
  automatically merged overlap files.
- Resolved `docs/code_structure.md`, `planning/changelog.md`, and
  `planning/specs.md` by retaining the complete sections from both parents.
  This preserves Fiber3D training, shared-inference, Fiberlet storage/tracing,
  and benchmark contracts. Retained this task's first-parent versions of the
  four transient current-task documents, per the workflow's current-task-only
  rule; both parents' previous task records remain in Git history and their
  durable results remain in the combined changelog/spec/docs.
- Audited all four source/test files changed on both sides but auto-merged by
  Git. The combined `lasagna/tiled_predict3d.py` retains training-side native
  output stride composition and incoming scheduler slot/band/deadlock controls;
  its combined test file retains both test families. The combined native
  `ChunkCache.hpp/.cpp` retains training-side decode concurrency/statistics and
  incoming opaque payloads, leases, cancellation, and cache-budget behavior.
- Found one semantic cross-feature gap in that automatic C++ merge: successful
  opaque Fiberlet decodes carried no dense byte vector, so the training-side
  `decodedChunkBytes` metric recorded zero. Changed the common measurement path
  to use `DecodedChunkPayload::residentBytes()` for opaque results and added
  generated/reopened opaque-cache assertions to `test_fiberlet_storage`.
- Python validation:
  - `py_compile` passed for the merged shared runner, its test, and Fiber3D
    trainer.
  - All 281 `test_fiber_trace_3d.py` tests passed (two Pillow deprecation
    warnings).
  - Seven targeted shared scheduler/native-output overlap tests passed.
  - The complete shared-inference file reports 54 passes and 9 failures. These
    are not introduced by the merge: both locked parents contain legacy tests
    that mock absent `_auto_download`, and both create default Zarr-v3 stores
    while the shared runner intentionally requires `.zarray` Zarr-v2 inputs.
    The first combined failure-rendering run also triggered a Python 3.14/NumPy
    pytest-repr segmentation fault; `--tb=no` completed and produced the stable
    54/9 inventory.
- Native validation:
  - Incremental CMake regeneration was attempted in the existing GCC 13 Release
    build but stopped before compilation because the incoming CMake now requires
    HiGHS and no `highsConfig.cmake`/`highs.cps` exists on this host. No install
    was performed under the repository's no-bootstrap policy.
  - Using the exact existing GCC 13 Release compile flags, the merged
    `ChunkCache.cpp` and modified `test_fiberlet_storage.cpp` both compile to
    fresh objects successfully (one benign precompiled-header flag warning on
    the test object). Fresh native linking/execution remains unavailable until
    HiGHS is configured.
- Exact validation commands and outcomes:

  ```bash
  PYTHONPYCACHEPREFIX=/tmp/fiber_training_merge_pycache /home/hendrik/.venv_las/bin/python -m py_compile lasagna/tiled_predict3d.py lasagna/tests/test_preprocess_cos_omezarr.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py
  PYTHONPATH=/home/hendrik/villa/vesuvius/src:/home/hendrik/villa/vesuvius:/home/hendrik/villa /home/hendrik/.venv_las/bin/python -m pytest -x -vv vesuvius/tests/neural_tracing/test_fiber_trace_3d.py
  PYTHONPATH=/home/hendrik/villa/vesuvius/src:/home/hendrik/villa/vesuvius:/home/hendrik/villa /home/hendrik/.venv_las/bin/python -m pytest -q --tb=short lasagna/tests/test_preprocess_cos_omezarr.py -k 'accumulation_frontier_reserves_capacity_from_later_results or shared_worker_uses_descriptor_queues_and_shared_results or shared_parallel_pipeline_matches_serial_output_exactly or shared_runner_composes_native_and_requested_output_stride or shared_parallel_tensorstore_pipeline_matches_python_zarr_exactly or shared_parallel_pipeline_profile_reports_bounded_stage_aggregates or shared_parallel_pipeline_detects_hard_worker_exit'
  PYTHONPATH=/home/hendrik/villa/vesuvius/src:/home/hendrik/villa/vesuvius:/home/hendrik/villa /home/hendrik/.venv_las/bin/python -m pytest -q --tb=no lasagna/tests/test_preprocess_cos_omezarr.py
  cmake --build volume-cartographer/build/dev-quickbuild-gcc --target vc_core test_chunk_cache test_chunk_cache_persist test_chunked_plane_sampler test_chunked_plane_sampler_more test_volume_extras test_chunked_plane_sampler_fallback -j 16
  jq -r '.[] | select(.file | endswith("/core/src/render/ChunkCache.cpp")) | .command' volume-cartographer/build/dev-quickbuild-gcc/compile_commands.json | sed 's#-o core/CMakeFiles/vc_core.dir/src/render/ChunkCache.cpp.o#-o /tmp/fibermerge_ChunkCache.o#' | bash
  jq -r '.[] | select(.file | endswith("/core/test/test_chunk_cache.cpp")) | .command' volume-cartographer/build/dev-quickbuild-gcc/compile_commands.json | sed 's#-o core/test/CMakeFiles/test_chunk_cache.dir/test_chunk_cache.cpp.o#-o /tmp/fibermerge_test_fiberlet_storage.o#; s#/home/hendrik/villa/volume-cartographer/core/test/test_chunk_cache.cpp#/home/hendrik/villa/volume-cartographer/core/test/test_fiberlet_storage.cpp#' | bash
  ```

  `py_compile`, the 281-test Fiber3D suite, the seven-test overlap subset, and
  both direct GCC object compilations passed. The complete Lasagna file produced
  the recorded 54-pass/9-parent-existing-failure inventory. The CMake build
  command stopped during regeneration because the required HiGHS package was
  absent; it did not compile or link stale targets.
- No conflict marker or unmerged index entry remains. There are no deletions in
  the merge delta. Resolution-local diff formatting is clean; the full incoming
  history retains its pre-existing generated-SVG trailing whitespace and one
  Markdown EOF blank-line warning, which were not rewritten during integration.
- Independent implementation/result review approved the combined code,
  conflict resolutions, opaque-payload accounting fix and test, staged-tree
  integrity, and reproducible validation record with no remaining blocker.
