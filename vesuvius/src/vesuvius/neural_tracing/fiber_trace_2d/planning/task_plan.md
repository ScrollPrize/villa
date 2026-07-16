# Merge Fiber 3D Extension And Adapt Multi-Dir Config Plan

## Merge Resolution

- Merge `fiber-3d-ext` into the active `fiber-3d-multidir` branch with
  `--no-commit` so the combined changes can be inspected before committing.
- Keep the branch's requested-level blocking coordinate sampling changes in:
  - `fiber_trace_2d/loader.py`
  - `fiber_trace_2d/sampling.py`
  - `fiber_trace_3d/trace2cp_tool.py`
  - VC3D sampler bindings and tests under `volume-cartographer/`
- Reapply the current multi-direction training changes in:
  - `fiber_trace_3d/model.py`
  - `fiber_trace_3d/train.py`
  - 3D training configs
  - `test_fiber_trace_3d.py`
- Do not include unrelated old stash changes. If an old stash conflict appears,
  keep the merge-side file and leave the stash entry intact.

## Config Adaptation

- Update the newly added
  `fiber_trace_3d/configs/train_s1a_nml_all_64_sd2.json`:
  - set `model_3d.direction_branch_count` to `2`;
  - set `model_3d.output_channels` to `14`;
  - keep BatchNorm explicit with `model_3d.normalization: "batch"`;
  - give the run a distinct multi-dir name so it cannot accidentally resume or
    overwrite old 7-channel runs.

## Spec Update

- Keep the existing multi-direction 3D training spec updates.
- Keep the merged requested-level blocking coordinate sampling spec updates.
- Add the current 64-scale S1A NML config to the multi-direction config
  wording where relevant.

## Docs Updates

- Keep `docs/code_structure.md` updates from both sides:
  - strict requested-level blocking sampler behavior;
  - branch-aware 3D model/loss/visualization behavior.
- Keep changelog entries for both the blocking sampler merge and multi-dir
  training/config adaptation.

## Testing Plan

- Python compile check for touched 3D model/train/tests and Trace2CP tooling.
- Full 3D Python regression file:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Focused 2D Trace2CP sampler/render tests from the merged branch:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "trace2cp_render_sampling"`
- JSON-load sanity check for all `fiber_trace_3d/configs/*.json`.
- Run `git diff --check`.

## Changelog

- Update the July 16 changelog entry to note that the 64-scale S1A NML training
  config was adapted to the two-branch output layout.

## Deviations Or Deferrals

- Do not commit the merge automatically; leave it staged/uncommitted unless the
  user asks for a commit.
- C++ rebuilds for the merged VC3D sampler changes may depend on the local
  CMake/Qt state. Report clearly if not run.
