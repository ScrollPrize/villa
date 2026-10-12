# Changelog

## 2026-10-12

- Exact diagnostics acceleration with bounded ordered CPU snapshots; worker-count
  sweep selected4. Added reference-width initialization (radial prior unchanged)
  and offset-aligned mean width factors. Completed low-LR100/1000pass fresh and
  reference-width-started joint solves, retained all10pass histories and TIFF/
  vector comparison configurations. Results:uv_joint_lr_width_results.md.

## 2026-10-11

- Added opt-in combined winding-coordinate mapping with explicit deterministic
  tiled width-gradient reduction. H100 exclusive updates1.63x/2.80x faster at
  batches4096/16384; matched4032-update quality remains comparable. Validated
  ordinary/precise division; ordinary selected for new path per user, reference
  backend still default. Derivation/reproduction:uv_coordinate_results.md.

- Corrected V constraints to equality; removed invented CT-Z endpoint displacement
  target from cache producer and all consumers. Factored weak per-fiber V≈Z prior
  into shared independently weighted helper, no local Z monotonicity condition.
  Historical runs were not rerun with this changed objective.

- Removed redundant pairwise wrapped-phase cost from all UV distance objectives;
  winding-distance phase correction, individual priors and vertical alignment
  remain unchanged. Squared selected by user as working candidate. Earlier
  four-way results predate this objective change; no new solve implied.

- Added opt-in expected squared/Huber and hybrid global-distance objectives,
  preserving sign and broken-probability losses. Four matched fused random-batch
  runs with non-reference initialization on GPUs0–3;49tests and real GPU gradient
  equivalence checks passed. See uv_distance_objectives_results.md.

- Added opt-in fused shared winding-pair objective/backward; real H100 benchmarks
  show2.38× pair and1.82× minibatch-update speedup, with lower intermediate memory.
  Preserved batch size, FP32 and optimizer schedule. Numerical trajectory caveats
  and reproduction recorded in uv_pair_fusion_results.md; eager remains default.

- Added explicit categorical winding-count expansion, preserving old physical
  geometry and broken probability while enabling additional hypotheses. Both
  directions scheduled for ten151-candidate sweeps; geometry remains frozen.
- Reference maps now use saturated golden-angle colors interpolated in OKLab,
  opaque2px stable overdraw and1000px raster height (LZW TIFF).
- Both151-candidate continuations completed and audited: retained length fell
  to25.52%/25.44% with zero sign violations through rejection; added hypotheses
  did not win. Not promoted. Both old/new comparison sets rendered;35tests pass.

- Corrected reference detail crops to LZW TIFF, including migration of existing
  PNG crops after exact pixel verification. Full maps and projections unchanged.

- Added reference winding comparison: rectangle-overlap half-winding landmarks,
  continuous shared random palette, original-source weighted median discrete
  reference assignment and chosen-winding curve projection. Read-only saved100
  CCW/CW maps, TIFF/SVG, match metadata and input-hashed projection cache.
  No correspondence optimization or solver changes. See reference_comparison.md.

## 2026-10-09

- Added100-sweep H/V density diagnostics (length/UV area plus counts), enabled
  versus all saved/conditionally re-placed disabled fibers. CPU O(events*K)
  placement reuses canonical costs against fixed enabled MAP neighbors; no
  global solve or inference. Common-area heatmaps/profiles and full numeric
  exports, provenance and unsupported/conflicted-placement diagnostics.

- Added H/V-separated rejection diagnostics: hard disabled length/counts and soft
  probabilities, V-H and V/H,10kUprofiles/3kUVheatmaps. Exact piece/cell counting,
  MAP integer and posterior-weighted variants. Both20-sweep solutions complete;
  globally V less rejected than H. Results:hv_rejection_results.md.

- Added saved-solution kollesis diagnostics: posterior-weighted crowding, Lasagna
  discrepancy, rejection and winding entropy, full-height U profiles and U/V
  heatmaps/support maps. Both20-sweep conventions completed; no clear common
  full-height joint bands established. Results/commands:kollesis_results.md.

- Exported complete retained CCW/CW U/V layouts as zoomable SVG plus PNG previews,
  confidence opacity/histograms and source tooltips. Correctly aligned sweep-cost
  progression, missing final energies evaluated without optimization. Independent
  artifact audit and39tests pass; see uv_visualization_results.md.

- Added full-neighborhood categorical cost targets with unsuppressed p-target
  logit updates; shared geometry/costs and frozen-input auditing. Damped pilot
  repair227→0sign violations,69.4125→71.9722%arc retained; aggressive updates
  oscillate. Continuous geometry is fixed and continuation failures remain;
  no claim of completed global layout.65regression tests pass.

- Resumed opt-in global layout with all accepted direction/exclusion outputs:
  unwrapped shared anchors, centered per-bucket offsets and strong quarter-turn
  hinge, all-pair compatibility likelihood with separate sign/rejection penalties.
  Both H conventions configured; legacy results/mode preserved.49tests pass.
  Initial pilot revealed weak offset scaling and disproportionate continuation
  normalization; revised pilot and full trials use stronger offsets and per-event
  normalization. Results are experimental, not promoted anatomical assignments.
  Full6.6061Mevent/489663piece trials:CCW6.4448%sign violations/91.4355%arc,
  CW6.4855%/91.3679%; ~17min/sharedH100each.54%ambiguous and poor continuation;
  both saved-state audits pass, findings in uv_anchor_results.md. No promoted default.

- Committed H/current state as3f5b443e8 with hash-pinned selected solution.
  Generalized H pipeline to shared family-specific direction preparation,
  sign/broken optimization and visualization; no duplicated V implementation.
  V prior follows local positive-Z umbilicus.237800V/1842020events; baseline528
  conflicts; penalties.1/.07/.05/.03 all77newrejections/0conflicts,99.953354%Varc
  retained. H arrays/labels frozen.41tests pass and independent all-event/raw
  tangent audit. V/global layout coupling still paused.

- Added optional stage3 whole-H breakage with fixed evidence normalization and
  joint direction/broken entropy. Full confidence-preserving sweep .1/.07/.05/.03;
  selected.05rejects184H(0.073115%), eliminates all1721direction conflicts,
  retains99.912734%Harc and99.767440%parallel events. Earlier exclusions fixed;
  U/V paused. Initial softened-confidence trials archived as diagnostic only.
 25tests; independent audit confirms none of the184can individually be restored
  in either direction without conflict. Existing renderer reused for survivor
  normal/curve views, with new all-disabled and same-subset tangent-arrow OBJ.

- Added third-stage whole-H tangent-direction alignment after frozen normal/HV
  stages.251658Hfibers/3603373parallel events; weighted disagreement0.698138%
  (umbilicus-only) to0.047735%, no new rejections,25.796smanagedH100. ExactCW/CCW
  opposite outputs, shared tangent/sign-loss helpers,22tests and independent
  full-event/raw-geometry audit. h_direction.md/results/configs. Global U/V paused.
- Preserved preliminary U/V implementation and two completed full trials in
  data/uv_global_v1; notes uv_layout_task_log.md. Not promoted: substantial
  ambiguity/rejection and continuation issues. Awaiting new global constraints.

- Committed H/V stage and recommended params(3f579c7c2,6949e82fc). Added classified
  visualization layers from recommended result:400H/400V/200broken,24319arrows,
  orange umbilicus. Broken sample66stage1/134stage2;44.419MiB. Shared exporter,
  source binding and independent layer indices;16Python tests and artifact audit.

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
# H-anchored surface prototype

Added full20-sweep CCW/CW tifxyz construction at100base-vx spacing, strict joint
confidence>50%, bounded1000vxH/V brackets and V normal-curvature transfer. Shared
native normal export validates existing baked gauge. Updated20-sweep fiber plots.
Initial surfaces are fragmented with large jumps; see surface_results.md.
# 2026-10-11 Joint full passes

Added optional common physical-Z mapping for reference/fitted comparison images,
including remapped reference coverage and matching cached-triangle projected Z.

Follow-up: remove broken-gradient warmup from joint optimization; use fixed
per-piece expected rejection cost by default, with matching sign calibration.
Explicit visualization winding offset supports reference-metric-aligned colors.

Added exhaustive joint event/node traversal and evaluation-only source-fiber
reference winding L1 / within-two metrics after global median offset correction.
# 2026-10-11: learning-rate batch scaling

- Document square-root state-LR scaling from B4096/LR0.03, supported by the
  B16384/LR0.06 trial. Keep width LR2; other batch rates are unvalidated starting
  points. No automatic scaling or default changes.

# 2026-10-11: quadratic window/batch experiment

- Added opt-in local MAP windows for quadratic distance/V, retaining exact
  full-distribution sign hinge with sorted prefix sums and unchanged broken gate.
- Added complete-update CUDA phase timings and optional Torch profiler trace.
- Added exclusive multi-GPU experiment lanes under one manager lease; reused
  solver subprocesses, cancellation and per-lane logs. Shared pilot results
  are excluded from final comparisons. Defaults unchanged.
# 2026-10-12

- Validated combined-coordinate gradients and larger batches; true1/2/3/6GPU
  joint solver with SparseAdam touched-row semantics,32768/GPU chosen. GPU
  throughput3.95Mconstraints/s on6GPUs vs0.879M onone.
- Exact median fastpath/shared readout and bounded asynchronous diagnostics cut
  ten-pass walltime152.74s→34.85s; all12 checkpoint metrics and final arrays exact.
  CPU/GPUgradient preflights and97 targeted tests pass. Full results/commands in
  uv_coordinate_results.md,uv_distributed_results.md,uv_diagnostics_results.md.

# 2026-10-12: reference-selected regional experiment

Added exact Z-clipped reference-selected induced graphs with provenance and
geometry/constraint audits. Reused existing6GPU joint solver on windings10–19,
Z30k–50k;100pass fresh/reference results recorded in uv_reference_subset_results.md.
Optional regional width diagnostics leave reference initialization/losses unchanged.
50targeted tests passed plus production CUDA/distributed preflights.

Follow-up: six100pass solves on shifted reference windows13–19/14–20/15–21,
sameZ30k–50k,fullcheckpoint reports in uv_reference_windows_results.md.
ActualXYZ validpoints are completely covered by winding_bounds; none precedeID10.
