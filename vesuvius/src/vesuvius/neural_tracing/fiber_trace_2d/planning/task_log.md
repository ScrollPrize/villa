# Task log: dense-fiber failure replay

## Findings

- `loadFiberJson` already strictly loads VC3D line/control points and maps each
  control point to an exact dense-line index. `referenceTangentToward` already
  captures the desired first non-degenerate dense-edge direction.
- `traceFiberExtrapolation` and `traceOneWayCore` provide the regular native
  scoring and length-bounded tracing, but expose no data-dependent committed-
  step stop condition. Replay should extend that core rather than reproduce it.
- Beam width one is the existing explicit greedy mode in both native
  implementations. It avoids an ambiguous failure point across live beams.
- Existing whole-fiber metric checks only CP target-plane crossings against an
  absolute default of 20 base voxels. It restarts at CPs and therefore is not
  the requested dense continuous replay.
- Python's legacy 2D `_score_trace2cp` normalizes target-column cross-track
  error by the usable strip-edge distance and clamps it to `[0,1]`; it does not
  define a compatible continuous 3D failure threshold. The 10-percent/10-voxel
  rule is bidirectional meeting fusion, not reference replay.
- Anchor extraction already block-samples bounded selected cells plus support
  and NMS context, but its selection is currently only one rectangular cell
  range and its strict artifact assumes that rectangle.
- Fiberlet path preload currently materializes one dense rectangular bounding
  box around all candidate corridors. That is unsuitable for a long curved
  tube even if anchor extraction is sparse; replay needs a sparse union of
  actual corridor nodes.
- The napari viewer already accepts an external Zarr and separate anchor/path
  OBJs. A replay bundle can add reference/trace/failure layers and supply the
  crop without storing the Zarr location.

## Plan review

- Two independent reviews checked the task plan against the current task,
  specification, code, and viewer behavior. The revised plan now:
  - separates trace-scale and canonical prediction-scale fields and normals;
  - uses a bounded, monotone, step-initialized dense-reference match instead of
    equal-arclength or global-nearest matching;
  - defines exact postroll, reference-end, invalid-start, and hard-budget rules;
  - preserves repeated reference vertices and only skips consecutive zero edges;
  - filters refined outside-tube anchors before NMS and clips replay DP nodes
    and endpoints to the tube while preserving interpolation halo reads;
  - includes complete sparse-preload dependencies and requires dense parity;
  - uses immutable content-addressed generations plus an atomic root bundle;
  - makes bundle JSON authoritative and validates derived OBJ geometry/hashes;
  - makes replay/manual viewer modes, path containment, status layouts, and
    external-Zarr metadata checks strict.

## Deviations and limitations

- The initial reference interval is 512 base voxels in each direction because
  the user left `N` unspecified; it remains configurable through `--along`.
- Initial failure threshold is 20 base voxels because there is no matching
  Python continuous-3D default. The artifact also records normalized
  `error/threshold` so this choice can be evaluated without ambiguity.
- The bounded direct reference refinement defaults to one nominal trace step on
  either side of the predicted next arclength. This is an explicit initial
  experimental default exposed by `--match-refine`; it can be changed without
  compatibility code because the workflow has not shipped.
- Fiberlet DP scores integer stored-prediction voxels directly; it performs no
  trilinear prediction or normal interpolation. The planned interpolation-halo
  dependency expansion is therefore not applicable. Sparse preload includes
  the canonical corridor nodes and every admissible virtual endpoint attachment.
- The C++ side is the sole bundle writer. The only bundle reader is the napari
  Python path, which performs strict schema, containment, hash, status, and OBJ
  geometry validation. No unused duplicate C++ bundle reader was added.
- The napari GUI itself was not launched during automated validation. Its
  Qt-free bundle parsing, CLI conflict logic, geometry conversion, crop helpers,
  and existing layer-control helpers are covered by Python tests; the real
  bundle and external Zarr metadata were also checked without opening a window.

## Implementation

- Added shared dense-polyline geometry and a committed-step observer in the
  existing native one-way trace core. Replay forces greedy settings and returns
  typed statuses without changing ordinary tracer calls.
- Added exact tube construction, explicit sparse anchor-cell selection,
  pre-NMS refined-anchor filtering, tube-constrained DP endpoints/nodes, and a
  sorted immutable sparse scoring preload. Standalone anchor/path behavior is
  retained.
- Added `vc_fiberlets fiber-replay`, shared native trace CLI parsing used by
  both C++ tools, separate trace/canonical bindings, atomic content-addressed
  run publication, and strict replay artifacts.
- Added napari `--replay` mode with the external Zarr kept separate, strict
  metadata/artifact validation, and independent reference, trace, failure,
  anchor, and fiberlet layers.
- Added a hard presence-tube mask from a one-time displayed-level reference EDT.
  Its runtime slider only changes the lazy distance threshold and defaults to
  the extraction radius shared by replay anchors and fiberlets.
- Added explicit replay stage start/completion timing. Anchor extraction now
  exposes library-level selected-cell and NMS-context progress, and replay wires
  both that ETA stream and the existing fiberlet-search ETA stream to the CLI.
- The napari presence EDT now rasterizes the union of reference and complete
  greedy replay geometry so the same radius exposes predictions around both.
- Anchor artifacts now include every selected cell center plus center-to-anchor
  lines for retained results. Replay validates the artifact and shows point and
  displacement geometry as separately toggleable napari layers.

## Validation

- Built with 32 jobs:
  `cmake --build build --target vc_fiberlets vc_fiber_trace_metric test_fiber_trace3d test_fiber_replay test_fiber_anchors test_fiberlet_paths -j32`.
- C++ focused suites passed: `test_fiber_trace3d` 49 cases,
  `test_fiber_replay` 2 cases, `test_fiber_anchors` 33 cases, and
  `test_fiberlet_paths` 23 cases.
- Python viewer suite passed 29 cases with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/test_view_fiber_presence.py`.
  Plugin autoload is disabled locally because the installed pytest Zarr plugin
  imports the absent `zarr.testing`; this is an environment issue before test
  collection, not a project test failure.
- Ruff and Python compilation passed for the viewer and its tests.
- Real-data smoke command used the Paris `fiber_s1_002` manifest, the
  `kb_20260605T150824406_000001.json` reference, and `las_008` normals with
  `--fail 0 --after 1 --along 16 --radius 16 --threads 32`. It reported
  `failure_with_postroll`, 3 trace points, 1 match, 1 postroll step, 8 selected
  cells, 1 anchor, and 0 fiberlets. The strict reader loaded the generated
  bundle and the external presence level matched shape `(9473,4087,4087)` and
  scale `(8,8,8)`.
