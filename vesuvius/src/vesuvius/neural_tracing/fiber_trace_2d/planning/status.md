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
- [x] Implement shared tiled inference extraction.
- [x] Define shared tiled inference adapter interfaces.
- [x] Port Lasagna predict3d to the shared runner without behavior changes.
- [x] Add fiber 3D inference adapter.
- [x] Add fiber 3D inference CLI.
- [x] Define fiber inference manifest and data-level-only pyramid behavior.
- [x] Add backwards-compatibility migration tests for shared helper exports and
  predict3d CLI dispatch.
- [x] Add/update specs and docs.
- [x] Add regression tests for shared runner, Lasagna adapter, and fiber
  adapter.
- [x] Run focused predict3d extraction validation.
- [x] Run full task validation after adapters/docs are implemented.

## Current Plan Items

- [x] Extract generic tiled inference/resume/chunk-writing mechanics.
- [x] Add minimal shared output/model adapter interfaces.
- [x] Preserve current Lasagna predict3d CLI and output semantics.
- [x] Add product adapters for fiber outputs.
- [x] Store fiber output options as coherent seven-channel bundles.
- [x] Share common CLI arguments and resume behavior across cos and fiber
  inference.
- [x] Verify crop-composable output and resume behavior with tests.
- [x] Document fiber option-bundle metadata in output manifest.
- [x] Verify `preprocess_cos_omezarr.py predict3d` compatibility surface after
  the shared-module extraction.
