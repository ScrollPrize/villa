# Merge Fiber 3D Extension And Adapt Multi-Dir Config Task Log

## Implementation Notes

- Merged `fiber-3d-ext` into `fiber-3d-multidir` with
  `git merge --no-commit --no-ff --autostash fiber-3d-ext`.
- The branch merge itself applied cleanly and stopped before commit.
- An initial `git stash pop` targeted an older unrelated stash entry and
  produced conflicts in:
  - `volume-cartographer/apps/VC3D/LineAnnotationController.cpp`
  - `volume-cartographer/core/test/CMakeLists.txt`
- Those two conflicts were resolved to the merge-side content. The older stash
  entry was kept, so unrelated old VC annotation work was not pulled into this
  merge and was not discarded from the stash list.
- The intended `MERGE_AUTOSTASH` was then applied explicitly. It conflicted
  only in active planning docs and was resolved by keeping this current
  merge/config-adaptation task as the active task.
- The newly added
  `fiber_trace_3d/configs/train_s1a_nml_all_64_sd2.json` now uses:
  - `model_3d.direction_branch_count: 2`
  - `model_3d.output_channels: 14`
  - `model_3d.normalization: "batch"`
  - `training.run_name: "s1a_nml_all_3d_64_2_multidir"`

## Deviations Or Deferrals

- The merge is intentionally left uncommitted.
- Native/projected Trace2CP inference still uses branch 0 for model output
  interpretation. Multi-branch inference selection remains deferred.
- C++ build validation for the merged VC3D requested-level sampler changes has
  not been run yet in this task.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/sampling.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed.
- `PYTHONPATH=vesuvius/src:. python - <<'PY' ... load_config(...) ...`
  passed for `loader_example.json`, `train_s1a_nml_all.json`, and
  `train_s1a_nml_all_64_sd2.json`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 61 passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "trace2cp_render_sampling"`
  passed: 2 passed, 271 deselected.
- `git diff --check` passed.
