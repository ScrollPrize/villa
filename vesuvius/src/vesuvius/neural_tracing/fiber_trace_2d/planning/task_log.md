# VC3D Requested-Level Blocking Coordinate Sampling Task Log

Current task log only.

## Planning

- Replaced the previous native 3D fusion task with the current requested-level
  VC3D blocking coordinate sampling task.
- Planned the fix at the VC3D coordinate sampler/binding layer so native 3D
  Trace2CP raw volume panels and prediction/presence sampling share the same
  strict requested-level behavior.
- Planned no changes to tracing score, fusion, model inference, normalization,
  or strip geometry.

## Implementation

- Added `ChunkedPlaneSampler::sampleCoordsLevelBlockingRequestedLevel(...)` for
  strict requested-level coordinate sampling.
- The strict sampler collects requested-level dependencies, queues them,
  fetches each dependency through `getChunkBlocking`, stores the returned
  `ChunkResult`s in a local pinned map, and samples only from that pinned map.
- Missing requested-level chunks render as black covered pixels. `Error` and
  post-blocking `MissQueued` statuses throw instead of falling back.
- `Volume.sample_coords(..., blocking=True)` now calls the strict sampler.
  Nonblocking `sample_coords` still uses the existing progressive
  fine-to-coarse path.
- Added sampler stats: `missing_chunks`, `fallback_levels`, and
  `requested_level_only`. `blocking_prefetch_chunks` remains a blocking-only
  count.
- `Vc3dCoordinateSampler` now rejects blocking results that do not report
  `requested_level_only` or that report nonzero fallback levels, so stale
  bindings fail loudly.
- 2D Trace2CP strip rendering and native 3D Trace2CP block sampling now reject
  chunk errors and fallback stats.
- Updated specs, code-structure docs, changelog, and task status for the
  requested-level blocking contract.
- Renamed the public stat from the misleading `full_res_only` wording to
  `requested_level_only`. The sampler still samples the configured/requested
  zarr level, e.g. `base_volume_scale: 2` stays group 2 / 4x scale.
- Updated the installed editable `volume-cartographer` package with
  `python -m pip install -e volume-cartographer --no-deps --break-system-packages`
  as documented in `planning/local_development.md`.

## Validation

- `cmake --build volume-cartographer/build/python-bindings -j 4`
  - Passed. Rebuilt `vc_core` and `vc.volume` with the strict requested-level
    sampler.
- `python -m pip install -e volume-cartographer --no-deps --break-system-packages`
  - Passed. Installed rebuilt editable `vc.volume` into the user site without
    changing zarr/numcodecs dependencies.
- `python -c "import vc.volume, zarr, numcodecs; ..."`
  - Passed. Active import is
    `/home/hendrik/.local/lib/python3.14/site-packages/vc/volume.cpython-314-x86_64-linux-gnu.so`,
    with `zarr==2.18.7` and `numcodecs==0.15.1`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k "trace2cp_render_sampling"`
  - Passed: `2 passed, 271 deselected`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d_trace2cp"`
  - Passed: `19 passed, 39 deselected`.
- `PYTHONPATH=volume-cartographer/build/python-bindings/python:vesuvius/src:. python - <<'PY' ...`
  - Passed: imported rebuilt `vc.volume`.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/sampling.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  - Passed.
- `cmake --build volume-cartographer/build/ci-coverage-clang-systemdeps --target test_chunked_plane_sampler_fallback -j 4`
  - Did not reach compilation. The existing build tree failed during CMake
    regeneration because `Qt6::Concurrent` was not found for
    `test_segmentation_lasagna_panel_ui`.

## Deviations Or Deferrals

- Persistent `.empty` marker cache corruption is identified as a separate
  possible cause of black requested-level chunks. This task will remove scale
  fallback and pin decoded requested-level chunks, but does not yet add a
  remote revalidation mode for stale `.empty` markers.
- Added C++ regression tests for strict requested-level sampling behavior, but
  could not execute that C++ test binary because the local CMake test build
  tree currently fails before compiling the touched target due to the unrelated
  Qt6::Concurrent configuration issue above.
