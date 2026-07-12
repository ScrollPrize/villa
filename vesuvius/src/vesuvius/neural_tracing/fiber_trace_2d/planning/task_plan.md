# Joint Top-View Direction And Distance-Transform Model Plan

## Scope

- Keep the existing side-strip direction/presence/optional-embedding model
  behavior unchanged.
- Add an optional jointly trained top-view model, enabled in the standard
  example config for this experiment.
- Reuse the existing VC3D-style top-strip coordinate construction used by
  Trace2CP visualization.

## Implementation

- Extend training config with:
  - `top_view_enabled`;
  - `top_view_direction_weight`;
  - `top_view_dt_weight`;
  - `top_view_dt_radius_px`, default `30.0`.
- Add loader support to construct one top-view patch per loaded CP sample:
  - same deterministic CP order as the side batch;
  - same geometric/value augmentation parameters as the center side-strip
    sample for that CP;
  - top-view coordinates sampled through the normal/top-strip grid;
  - value augmentation remains deferred to the existing GPU tensor preparation
    path.
- Add prefetch support that includes the top-view envelope coordinates when
  `training.top_view_enabled` is true.
- Add top-view supervision:
  - direction target from transformed top-strip line tangent, using the same
    ambiguous two-channel direction encoding as the side model;
  - distance-transform target along the rounded normal/cross-fiber line through
    the transformed CP, with `max(0, 1 - distance / radius)` and explicit zero
    targets beyond the radius.
- Train two `FiberStripDirectionNet` instances when top-view is enabled:
  - side model keeps its configured scalar/embedding heads;
  - top model uses direction plus one sigmoid scalar channel interpreted as DT.
- Save and resume checkpoints with optional `top_model_state_dict`.
- Add TensorBoard scalars for top-view direction/angle/DT losses and samples.
- Add TensorBoard images for top-view direction overlay and top-view DT maps.
- Update the current example config to enable the top-view auxiliary model.

## Spec Update

- Add top-view joint training semantics, output layout, distance-transform
  target definition, and prefetch inclusion to `planning/specs.md`.
- Clarify that the top-view scalar channel is a distance transform, not
  sheet/fiber presence.

## Docs Updates

- Update `docs/code_structure.md` training/loader/model descriptions.
- Update `planning/status.md`, `planning/task_log.md`, and changelog.

## Tests

- Add focused tests for:
  - top-view distance-transform supervision includes center, radius, and
    beyond-radius targets;
  - loading a top-view batch returns one top patch per CP and preserves sample
    ordering/augmentation metadata;
  - prefetch request generation can include top-view dependencies;
  - top-view-enabled model output/loss path creates a direction+DT top model.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
