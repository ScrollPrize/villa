# Changelog

## 2026-10-08

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
