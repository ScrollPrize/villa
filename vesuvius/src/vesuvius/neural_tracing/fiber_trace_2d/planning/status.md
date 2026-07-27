# Shared 3D Tiled Inference Status

- [x] Re-read backed-up predict3d task files after the main merge.
- [x] Check current `preprocess_cos_omezarr.py predict3d` state.
- [x] Check current 3D fiber model/inference-related APIs.
- [x] Update backed-up `task.md_` to make rolling predict3d behavior the
  baseline and shared fiber inference the remaining task.
- [x] Update backed-up `task_plan.md_` with the current-state-aware plan.
- [x] Promote the updated backed-up planning content into the regular active
  planning files.
- [x] Update task log.
- [ ] Implement shared tiled inference extraction.
- [ ] Port Lasagna predict3d to the shared runner without behavior changes.
- [ ] Add fiber 3D inference adapter and CLI.
- [ ] Add/update specs and docs.
- [ ] Add regression tests for shared runner, Lasagna adapter, and fiber
  adapter.
- [ ] Run validation.

## Current Plan Items

- [ ] Extract generic tiled inference/resume/chunk-writing mechanics.
- [ ] Preserve current Lasagna predict3d CLI and output semantics.
- [ ] Add product adapters for Lasagna cos/normal outputs and fiber outputs.
- [ ] Store fiber output options as coherent seven-channel bundles.
- [ ] Share common CLI arguments and resume behavior across cos and fiber
  inference.
- [ ] Verify crop-composable output and resume behavior with tests.
