# Task log: hard split continuity and aligned winding signs

## Findings

- Existing `hardContinuity` edges used finite pair costs and allowed an
  active/Defect boundary with default piece-break cost zero. Independent
  nodewise marginal MAP decoding could also publish different H/V or winding
  states across one active source-fiber continuation edge.
- The required invariant is edge-local, not a whole-source chain state. A
  continuation edge is neutral when either endpoint is Defect. When both
  endpoints are active, they must have identical H/V and integer winding. A
  Defect gap may therefore separate independently labeled active runs.
- Existing sign configuration was global: finite `--winding-sign-cost 44`
  made every enabled sign finite, while `hard` made every enabled sign exact.
  Extraction already retained the raw absolute aligned-normal agreement needed
  for a local reliability gate.
- The requested 30-degree gate is the inclusive raw-alignment comparison
  `abs(dot(connector, aligned_normal)) >= cos(30 degrees)`.
- Exact pair potentials alone do not guarantee a feasible published assignment
  after independent nodewise marginal MAP decoding. Deterministic final
  projection must therefore enforce the same edge-local invariant.

## Implementation

- Added hard split continuity, enabled by default, to orientation BP and both
  H/V-aware winding solvers. Defect neutralizes a continuation edge; two active
  endpoints must share H/V and integer winding.
- Deterministic final projection preserves the orientation seed or winding
  gauge, otherwise disables the lower-confidence endpoint, and uses the larger
  node index as the exact tie-break. It never copies a state through an entire
  source chain.
- Added `--split-continuity hard|finite`. `finite` retains the previous
  pairwise behavior and makes `--piece-break-cost` effective; `hard` is the
  default.
- Added `--winding-hard-sign-angle DEG|off`, default 30 degrees. An admitted,
  enabled, nonzero dominant perpendicular or parallel sign is exact when its
  raw normal alignment reaches the threshold. Weaker signs retain the existing
  finite confidence-weighted cost. Global `--winding-sign-cost hard` remains an
  unconditional override.
- Applied identical sign promotion in solver preparation, factor diagnostics,
  reference observations, and final hard-sign feasibility projection. Added
  promotion flags to the factor CSV.
- Added `fiber winding constraint agreement` output. It reports prepared,
  active/evaluated, Defect-neutralized, infringed, and
  `infringed/evaluated` percentage for continuity, perpendicular `0.5`/`1.5+`,
  parallel `0`/`1`/`2+`, and the sum.

## Superseded run

An initial 2048 comparison accidentally implemented whole-chain propagation:
one Defect forced every connected source piece to Defect. Its quality numbers
are invalid for the requested behavior and are intentionally not retained as a
result. The corrected attribution matrix separates continuity mode and hard
sign promotion on both the 1024 and 2048 crops.

## Validation

- Optimized build:
  `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiber_trace_winding_bp test_fiberlet_crop_trace -j 16`
- `volume-cartographer/build/bin/test_fiber_trace_winding_bp`: 63 cases passed.
- `volume-cartographer/build/bin/test_fiberlet_crop_trace`: 81 cases passed.
- `git diff --check`: clean.

## Crop comparisons

All runs used the Release build, quality fraction `0.25`, 512-base-voxel
pieces, fixed phase `0.5`, fixed scale `0.822`, fixed orientation, 500 maximum
winding messages, parallel cutoff `0.5`, and the current default winding
weights, temperatures, Defect cost, and finite sign cost. Only continuation
mode and the hard-sign alignment gate changed. Runner wall time includes input,
normal alignment, constraint extraction, solve, diagnostics, and artifact
output; solve time is joint-grid winding only.

| Crop | Continuity/sign | Status | Messages | Solve | Active | Defect | Continuity infringed | All infringed | Exact refs | Ref accuracy | Wall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | finite/off | message limit | 500 | 13.6 s | 1,360 | 0 (0.00%) | 128/861 (14.87%) | 32,389/69,172 (46.82%) | 5/8 | 890/2,177 (40.88%) | 19 s |
| 1024 | hard/off | converged | 115 | 3.0 s | 1,357 | 3 (0.22%) | 0/855 (0.00%) | 23,380/68,855 (33.96%) | 8/8 | 1,313/2,175 (60.37%) | 9 s |
| 1024 | finite/30 deg | converged | 196 | 4.9 s | 1,354 | 6 (0.44%) | 92/853 (10.79%) | 22,734/68,531 (33.17%) | 8/8 | 1,340/2,148 (62.38%) | 11 s |
| 1024 | hard/30 deg | converged | 221 | 5.5 s | 1,294 | 66 (4.85%) | 0/809 (0.00%) | 24,075/62,615 (38.45%) | 7/8 | 1,175/2,050 (57.32%) | 11 s |
| 2048 | finite/off | message limit | 500 | 9.6 s | 2,523 | 0 (0.00%) | 533/2,028 (26.28%) | 16,373/46,600 (35.14%) | 5/8 | 411/953 (43.13%) | 82 s |
| 2048 | hard/off | converged | 398 | 7.8 s | 2,488 | 35 (1.39%) | 0/1,963 (0.00%) | 10,865/45,363 (23.95%) | 6/8 | 551/952 (57.88%) | 79 s |
| 2048 | finite/30 deg | message limit | 500 | 9.5 s | 2,474 | 49 (1.94%) | 423/1,966 (21.52%) | 14,220/44,591 (31.89%) | 5/8 | 413/934 (44.22%) | 81 s |
| 2048 | hard/30 deg | message limit | 500 | 9.4 s | 2,469 | 54 (2.14%) | 0/1,942 (0.00%) | 11,130/44,460 (25.03%) | 5/8 | 521/953 (54.67%) | 80 s |

Hard continuation alone is the strongest tested setting on both crops. It
eliminates all active continuation mismatches while disabling only 0.22% of
1024 pieces and 1.39% of 2048 pieces. The 30-degree hard-sign gate accounts for
most additional Defects and lowers exact-reference and aggregate reference
accuracy relative to hard/off on both crops. The requested default remains
hard/30 degrees; those final artifacts overwrite
`data/workdir3/fiber-crop-{1024,2048}/fibers*`. Complete logs and comparison
artifacts are under `/tmp/hard-continuity-matrix`.
