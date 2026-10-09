# Changelog

## 2026-10-09

- Added cached native per-fiber axial alignment at extraction/backfill, then
  whole-fiber H/V+rejection stage using fixed normal-stage exclusions. Shared
  manager launcher, degree-aware penalties and late ternary entropy;8full-data
  GPU trials. Recommended inspection candidate4.5329%total rejected,
  0.05826%retained relation contradictions,95.2522%active arc retained.
  143native/15Python tests and independent all-result audit; see hv_stage.md and
  hv_stage_results.md. No claim of anatomical error rate; no sourceDB mutation.

## 2026-10-08

- Committed prior native/solver/visualization state (aa21090c2,64a467462).
  Added optional frame-checked orange umbilicus tube/centerline; exported1000longest
  retained fibers,25444outward arrows,0broken to whole_scroll_1000_long_retained/.
  29.376MiB;11tests and artifact checks passed; previous previews unchanged.

- Canonical saved signs now inward; display arrows outward, with explicit
  conventions and legacy conversion. Long-preview2463arrows overwritten at5x
  length/radius (640/20); original OBJ unchanged, historical solution preserved.

- Longer whole-scroll preview:100 spatially selected from1000longest retained
  fibers,0broken,2463normal arrows; length min/median/max7546/8285/11358basevoxels.
  Output whole_scroll_100_long_retained/ (2.49MiB); previous preview preserved.

- Whole-scroll preview:100 spatially distributed retained fibers,0broken,
  295arrows,887vertices,787OBJedges. Four exporter tests passed. Output:
  data/sign_resolution_v2/whole_scroll_100_retained/ (372KiB).

- Fixed MeshLab fiber OBJ compatibility:explicit two-vertex line elements for
  every segment, replacing long multi-index polyline records. Existing preview
  repaired programmatically with geometry-range checks; signed-normal PLY unchanged.
  Also fixed pre-existing raw AFV reader evaluation order dropping first vertices;
  regenerated45full curves/720vertices/675edges and updated index ranges.

- Added CPU signed-normal visualization:raw AFV OBJ polylines, colored PLY arrow
  meshes, broken-only OBJ overlay and per-fiber/event/arc metadata. Reuses raw AFV
  reader with optional ID selection; no geometry repair/inference. Preview45fibers,
  349arrows around brokenfiber454634;7tests. See sign_resolution.md.

- Implemented whole-fiber GPU sign/broken resolution from perpendicular normal
  dots and weak catalogue-umbilicus radial prior. Reuses native normal sweep and
  frame-checked Umbilicus; source DB fingerprints prevent stale-prior pairing.
  Full512,698fibers/1,320,610events,10000Adamsteps~22s/penalty on managed H100;
  penalty0.1retains98.4322%events with0.001069%weighted sign opposition and0.4685%
  broken fibers.142native/5Python tests and independent saved-solution validation.
  No anatomy/optimality claim; source DB untouched. See sign_resolution_results.md.

- Added native second-pass fiber-local normal orientation over existing constraint
  DB, regular16base-voxel sampling plus exact endpoints, independent+1fiber gauge.
  Schema2 stores only signed endpoint normals; source extraction/measurements
  preserved. Full512,698fibers/7,368,335events processed in222.89s wall;25.7667%
  endpoint flips, no invalid endpoints (5invalid intermediate samples skipped).
  142native tests and full-field/endpoint-axis validation pass. See
  normal_orientation_results.md and schema documentation.

- Added non-mutating initial-relation outcome diagnostics. Fullraw audit:
  parallel29.5660%rejected/perp10.3820%;seed failures unclassified. All7.368M
  measurements exactly match prior DB,141tests pass. Tables and denominators:
  relation_rejection_results.md.

- AFV extraction now stores continuous winding evidence without variation,
  category or1.5cutoff processing filters; preserves median parallel estimator.
  Geometry-only adaptive refinement avoids winding rounding; nonfinite probes
  still rejected. New raw_winding_v1DB:7,368,335events,11.27GiB;141tests,
  exactlegacy pilot and DB integrity/measurement checks pass. Original filtered
  artifact retained. Details: raw_winding_results.md.

- Corrected perpendicular local-event refinement to recenter projection windows
  on current points. Full AFV reciprocal rejection22.5143%→0.000416%(4iters),
  zero at16/64. Moving16now default; historical fixed4configs preserved.
  Parallel path unchanged.140tests and exactlegacy pilot passed; artifact DB
  unchanged. See moving_projection_results.md for event consolidation, timings,
  reproducible commands and finite/local-convergence limitations.

- Disabled parallel central/probe reciprocal filtering by default, retaining an
  explicit legacy option. Full-volume4/16/64perpendicular-iteration comparison:
  extra iterations do not help;22.5143%of checked perpendicular attempts still
  rejected,98.84%at seed-window boundaries. Accepted events rise2.334Mto4.215M
  from parallel removal.139native tests and exact legacy pilot passed. Details
  and unchanged historical database caveat: parallel_reciprocal_results.md.

- Added unsigned, uncut local-event AFV constraint extraction using shared native
  BP/Lasagna machinery; slab performance benchmark documented separately.
- Added versioned SQLite constraint export with all source fibers, full event
  measurements, endpoint sheet normals, identity/provenance and validation.
- Completed full Paris4_v1 extraction:512,698fibers,2,334,188events,5m12s total,
  reusable3.96GiB database. See full_constraint_results.md.
