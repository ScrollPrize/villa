# Robust line strips (v7)

VC3D uses the native v7 robust fit when generating line annotation strips and
constraint graph fiber strips, in both full graph and fiber review modes. Its
only inputs are the traced centerline and the Lasagna normal field. CT, SDT,
fiber-direction volumes, learned models, and accepted patches are not inputs.

`core/src/lasagna/RobustLineStrip.cpp` implements midpoint normal following with
40/80/120-voxel tangent spans, local geometry repair, and four Huber IRLS fits.
The fit keeps the traced centerline fixed, uses the less-pull ridge of 0.0025,
and guards local steps against folded triangles and more than 3x distortion.
Repaired strips also retain their half-width within 5%. Missing normals may be
filled from a small agreeing neighborhood; unsupported regions fail explicitly.
`LineStripRepair.cpp` uses the existing Ceres dependency for native L-BFGS repair.
It ports the experimental objective, but is not bit-identical to PyTorch's solver.

`LineViewBuilder` fits on an internal grid with spacing at most two voxels, then
samples the requested output grid. It keeps the finer fitting mesh if coarsening
would introduce invalid geometry. The returned dimensions and position map are
authoritative. Normal crosscuts retain their original construction. Callers that
omit `LineViewConfig::normalSampler` retain the frame-only diagnostic builder;
VC3D's production callers supply the sampler.

## Existing workspaces

Fiber nodes persist `strip_generation: 7`. Missing or older versions regenerate
from the workspace's embedded trace, using its configured normal field. Full
graphs upgrade when opened. Fiber review upgrades each fiber on first selection,
including the selected fiber at opening, to avoid blocking on an entire catalog.

`core/src/constraints/GraphFiberStrips.cpp` upgrades transactionally. Patch XYZ,
patch placements, accepted/rejected decisions, order, and pending edits remain
intact. If the output grid is unchanged, fiber UV and contact coordinates remain
bit-for-bit identical. When a finer mesh is required, the fiber chart is sampled
from the saved chart and fiber contacts are remapped through arclength/across
coordinates; patch charts are never refitted. Accepted alignments and pending
edits share the replacement fiber source. Upgraded workspaces become unsaved;
the normal Save action persists them. A failed upgrade retains saved geometry
and reports the failure, so it can be retried instead of marking it current.
Read-only winding-layout imports are unaffected.

## Build and verification

From the repository root, using the existing configured build:

```sh
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build -j 4 \
  VC3D test_robust_line_strip test_lasagna_line_view_surfaces \
  test_lasagna_normal_sampler test_line_annotation_generated_views \
  test_constraint_graph robust_line_strip_probe
QT_QPA_PLATFORM=offscreen volume-cartographer/build/bin/test_constraint_graph
QT_QPA_PLATFORM=offscreen ctest --test-dir volume-cartographer/build/core/test \
  --output-on-failure -R '^(test_robust_line_strip|test_lasagna_line_view_surfaces|test_lasagna_normal_sampler|test_line_annotation_generated_views)$'
volume-cartographer/build/bin/VC3D
```

The tests cover plane/cylinder geometry, pinned centerlines, normal-axis sign
invariance, deterministic fitting, missing fields, cancellation, finer-grid
fallback, legacy migration, preservation of placements/contacts/decisions, and
JSON roundtrips. No dependency installation is required by this change.

For local real-data validation, `robust_line_strip_probe` accepts JSON containing
`normal_manifest`, `normal_scale` (working-to-base scale), and `line` (XYZ arrays):

```sh
AGENTS_AGENT_MODE=1 OMP_NUM_THREADS=1 \
  volume-cartographer/build/bin/robust_line_strip_probe fiber.json /tmp/fiber-v7
```

It writes `/tmp/fiber-v7.bin` (row-major float32 XYZ) and `.json` (shape, column
arclengths, generation version, and elapsed seconds). The default is a two-voxel
grid, 61 rows. Optional JSON fields `output_spacing` and `output_rows` exercise
other output grids; `20` and `7` reproduce the default graph request. To isolate
the robust fit from repair, append `seed.bin rows cols [preserve-width]` to fit an
existing two-voxel float32 grid.

## Integration validation and limits

On the existing 55-fiber cohort with accepted patch associations from
`~/Desktop/fiber_linking.json`, dense native output passed all geometry and
centerline checks. Equal-fiber mean patch distance was **1.673897 voxels**, versus
**1.674583** for experimental v7 and **1.816006** for local repair + less-pull fit.
The evaluator used the same frozen validation footprint for every method.
Accepted patches were read only by the independent evaluator.

Local artifacts are under `/tmp/fiber-strip-v7-native`: `accuracy.json`,
`geometry-audit.json`, `cohort-results.json`, and rendered `*-comparison.png`
examples. The largest repair differences occur locally and need visual review;
the 000149 maximum-difference example has no accepted-patch coverage. Aggregate
patch metrics cannot establish correctness there. Coarse output grids additionally
interpolate the fitted surface, so the dense-grid metric is not a guarantee of
the same error at every display resolution.

In the Linux arm64 RelWithDebInfo build, one generation per fiber, two independent
probe processes, yielded mean 9.61 s, p50 4.90 s, p95 40.39 s, range 1.42–51.22 s.
These include dataset opening and fitting but exclude TIFF export. This is an
observed runtime, not a controlled speedup measurement or an instant-generation
guarantee. Graph generation and migration run in cancellable workers; annotation
view materialization still runs synchronously. No new platform-specific code,
precision reductions, or dependencies were introduced; macOS was not run locally.
