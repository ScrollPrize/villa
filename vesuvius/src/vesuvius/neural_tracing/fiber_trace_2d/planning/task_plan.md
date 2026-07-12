# Top-View Loader Performance Parity Plan

## Scope

- Keep model outputs, losses, deterministic CP order, and top-view supervision
  semantics unchanged.
- Refactor only the top-view loading path and profiling attribution.
- Do not change the side-view cache key format; existing side-view cache files
  must remain usable.

## Implementation

- Add a view-specific source-cache key for top-view source geometry while
  preserving the existing side-view key exactly.
- Factor top-view source construction to use the same cache load/store payload
  as side-view source construction.
- Replace per-top-sample augmentation/line-coordinate code with the same
  batched structure used by `_prepare_sample`:
  - build unique augmentation transforms once per top sample;
  - batch source line/control-point lookup through `sample_xy_maps_bilinear`;
  - stack top coords and masks, then run batched coordinate augmentation;
  - convert torch coords/masks to NumPy once at the sampler boundary.
- Keep grouped top-view `sample_coord_batch` dispatch by sampler, but attribute
  it to the shared `volume_sample` profile stage so benchmark tables show the
  extra sampling work.
- Merge top-view profile aliases into the aggregate side-view profile columns
  where useful (`descriptor`, `strip_coord_cache`, `source`, `line`,
  `coord_augmentation`, `volume_sample`) so top-view time does not disappear
  into `outside`.

## Spec Update

- Document that top-view training uses the same cached, vectorized, batched
  source/augmentation/sampling path as side-view training, with only the grid
  builder differing.
- Document that top-view cache entries are separate from side-view entries but
  use the same cache payload semantics.

## Docs Updates

- Update `docs/code_structure.md` loader description for top-view performance
  parity.
- Update `planning/status.md`, `planning/task_log.md`, and changelog.

## Tests

- Add focused loader tests that top-view source cache serves a fresh loader
  without resampling normals.
- Add focused loader tests that top-view batch loading uses one grouped
  `sample_coord_batch` call for multiple CPs and no per-patch `sample_coords`
  call.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - existing load-only profile command for a smoke performance check if local
    data/cache access is available.
