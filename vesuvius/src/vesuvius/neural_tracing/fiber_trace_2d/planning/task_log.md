# Task log: reload replay artifacts in napari

## Findings

- Startup currently parses replay artifacts directly inside `launch_viewer`, so
  Napari `refresh()` only redraws fixed in-memory geometry and cannot reread a
  newly published replay generation.
- The volume layer owns the lazy Zarr crop separately from replay artifacts.
  Its source can remain untouched while the reference-dependent EDT is rebuilt
  and applied to that same lazy source.
- Existing control callbacks close over distance/filter state. Reload support
  needs mutable derived-state holders so the same controls act on replacement
  distances without replacing the widgets or layers.

## Deviations

- Napari is not installed in the current Python environment, so an actual GUI
  button smoke could not run without changing dependencies. The strict loader,
  compatibility, empty/nonempty topology, feature rebuilding, distance
  preparation, and commit/rollback mechanism are covered independently.

## Plan review

- Independent review required a prediction-source fingerprint in addition to
  shape/scale. Reload now requires the same fiber-manifest content hash.
- The unchanged object is the lazy Zarr source, not `volume_layer.data`, because
  reapplying a new EDT creates a new derived Dask mask graph.
- Reload now has explicit prepare/commit/rollback phases with event-blocked
  layer mutation. Replay mode creates stable typed layers even for empty
  populations, allowing empty/nonempty count transitions without changing layer
  identity or control bindings.
- Current control callbacks must read mutable derived-state containers so they
  immediately operate on reloaded distances without widget replacement.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src pytest -q
  vesuvius/tests/test_view_fiber_presence.py`: 43 passed.
- Ruff check/format check, Python compilation, and `git diff --check` pass.
- The real `/tmp/vc-fiber-replay-anchor-stages/fiber_replay.json` was read twice
  through the root loader, both visual artifact sets were strictly loaded and
  compatibility-checked, and all eight populated anchor-filter groups were
  recomputed. The fixture's empty fiberlet layer is accepted and may become
  populated on reload.
- Inspection confirms the reload callback does not call the OME-Zarr resolver,
  `zarr.open_array`, or `open_lazy_crop`; its replacement mask graph is built
  from the original `presence_source_data` object.
