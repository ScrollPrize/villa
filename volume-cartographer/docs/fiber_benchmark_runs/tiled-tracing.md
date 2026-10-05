# Tiled beam tracing and local deduplication

Default geometry:1024base-voxel cores,64base-voxel halo on each side (1152side
for complete interior tiles;128shared overlap). Partial edge cores are supported.
Trace and deduplicate the expanded tile independently, then clip retained lines
to half-open cores. No global overlap graph or global suppression is performed.
The existing raw40500 beam8/full-prefix512/iso20 settings and local presence-ranked
ellipse10/20,f0 deduplication are retained. Each reverse-filter piece is independent.

## Execution

Approved entrypoint (no arguments):

```sh
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```

`scripts/training_failure_mining_control.json` selects
`{"action":"iso20_tiled","pilot":true}` for the2-tile2048x256x256seam pilot,
or `{"action":"iso20_tiled"}` for the2048cubed8-tile test. Both reuse current
prediction/anchor caches. This runner does not launch inference; missing cached
data is an error, not fill-zero. The existing coverage/inference pipeline remains
unchanged.128native threads total, at most4concurrent tile jobs; each job moves
straight from tracing to local deduplication, with no batch barriers. Saved native
records provide per-stage times, hashes and resume validation. Changed binaries,
inputs or configurations require a new output namespace; no stale artifact overwrite.

Anchor preparation runs once with four queued owner chunks and32fitting threads
each before tile jobs. It reuses the existing cache prefetch queue and preserves
collection order. Per-owner sample cap is2GiB (up to8GiB across four owners, plus
decoded caches/scoring scratch); prediction I/O concurrency is not reduced.
Tile tracers use
`--cached-anchors-only`, sharing existing anchor chunks without contending for the
exclusive generation lock. Missing anchor chunks fail explicitly. The initial
pilot exposed this locking conflict; it was stopped, preserving cached data.

Output namespaces are `/ephemeral/hendrik/fiber_crop_bench/20261004_iso20_tiled_pilot_v4`
and `20261004_iso20_tiled2048_v2`. `progress.json` records completed tiles;
`results.json` reports counts/timings. `assembled.zarr` stores clipped trace geometry.

## Native interface and topology

`vc_fiber_trace_chunk assemble-tiles tiles.json --normal-manifest PATH --output OUT
--threads 128` consumes locally deduplicated trace artifacts. The manifest has
`contract:1`, global `minimum`/`maximum`, `tiles` with `index_xyz`, `core_min/max`,
`halo_min/max`, `artifact`, and a `seam` configuration.

`processing.preprocessing.tiled_trace` contains versioned tile descriptions,
per-fragment source tile/line/run/arc intervals and artificial-cut endpoint flags,
plus candidate links keyed by output fragment ordinal and endpoint0(start)/1(end).
Original vertices remain unchanged except inserted clipping intersections and an
explicit first-segment midpoint seed required by storage. Reentry
runs remain separate. Stable tile-index/source/run order is independent of job
completion. No connecting segment is invented between offset endpoints.

Matching checks26neighbor tile positions, shared halo only. It reuses the native
spatial overlap index, compares only opposite tiles, requires80%coverage in BOTH
directions and32base voxels of matched support,10/20normal/tangential ellipse,
45degree tangent agreement and opposing outward endpoint directions. These are
initial configurable seam gates, not tuned results. Multiple candidates are kept.
Only artificial core cuts can link; actual reverse gaps/terminals cannot.

New `--capture-step-costs` persists per-segment tracer costs. Clipping partitions
each original step cost linearly by retained arc fraction (not a fresh integration
of the predictor); no whole-parent cost duplication or whole-line-average estimate.
Assembly rejects missing captured costs. Local dedup retains captured arrays.

## Downstream support and explicit limitation

Reciprocal unique endpoint matches can be carried into BP as active-state equality
constraints, without artificial tile-boundary gap penalties. No geometric stitching.
The standard BP-only direction-ablation consumer currently rejects unresolved
alternatives: existing pairwise continuity factors cannot represent an OR/choice
among continuations. Candidate topology is preserved for a future choice-aware
consumer, never silently dropped or turned into simultaneous hard equalities.
Other constraint consumers/global refiltering are guarded against invalidating it.
Consequently an ambiguous assembled result does NOT yet have a valid global BP
benchmark score. Resolving this requires an explicit alternative-choice model.
Unmatched internal cuts also block BP rather than silently introducing breaks.
`endpoint_counts` reports internal/external cuts, unique/ambiguous matches and
unmatched internal/external cuts separately.

Preparation currently calls the shared anchor-box materialization API, so it also
constructs scoring views and sorts anchors before discarding that list. This is
avoidable warm-start work; a payload-only preparation API is a future optimization.
It does not repeat anchor fitting when canonical cache chunks already exist.

## Validation

Approved `build` action uses the existing workspace RelWithDebInfo build and native
regression suite. Python tests: `scripts/test_iso20_tiled.py`. Native tests cover
core clipping, multiple reentries, costs, reverse orientation, unique/ambiguous
seams, real-terminal exclusion, order independence and artifact cost roundtrip.
The same approved runner with control action`iso20_tiled_tests` runs the focused
tiled/post-overlap/2048 configuration tests (9passed). Action`iso20_tiled_compare_pilots`
performs native v3/v4 artifact readback and compares exported geometry/quality files.

## Pilot result (2026-10-04)

Pilot v3: one cached-data run, two cores spanning2048×256×256base voxels, RelWithDebInfo;
2concurrent jobs×64native threads. Same frozen raw40500 prediction and tracing
configuration as above. Total55.32s: cached anchor preparation25.07s, tile tracing
24.53/28.06s, local deduplication0.686/0.692s, assembly0.240s. First cold halo-anchor
preparation in the preceding attempt took189.69s; excluded from this warm timing.
Local outputs172/160lines become153clipped fragments. All6internal cut endpoints
matched as3unique continuations;0ambiguous and0unmatched internal endpoints.
294external cuts are expected at the outer pilot crop. These are implementation
and seam-coverage results, not a BP accuracy score or a full-volume speedup claim.

Repeat v4 with bounded owner prefetch:54.20s total,25.50s preparation,
21.91/26.83s tracing,0.611/0.512s deduplication,0.185s assembly. Native artifact
readback succeeds and every exported geometry/quality file is byte-identical to
v3. Preparation concurrency does not change this cached pilot's output.

## Completed 2048 run and handoff (2026-10-05)

Saved run: `/ephemeral/hendrik/fiber_crop_bench/20261004_iso20_tiled2048_v2`.
All eight tiles completed tracing and local deduplication. The 7,322 retained
tile lines became 3,722 core-clipped fragments. Total wall time was 4,775.68s,
including 2,347.49s anchor preparation and 9.40s assembly. These are single-run
measurements, not a repeated throughput benchmark.

| Topology statistic | Count |
|---|---:|
| Accepted candidate links | 739 |
| Mutually unique links | 715 |
| Ambiguous links | 24 |
| Internal cut endpoints | 2,439 |
| External cut endpoints | 2,314 |
| Unique matched endpoints (including 4 external) | 1,430 |
| Ambiguous endpoints | 28 |
| Unmatched internal endpoints | 985 |
| Unmatched external endpoints | 2,310 |

BP was not run: the current consumer rejects ambiguity and unmatched internal
cuts. The 985 unmatched endpoints are NOT a rejected-candidate count; no
per-gate rejection counters exist yet.

The requested next change is to remove the 80%-of-both-supports requirement and
use the absolute 32-base-voxel overlap criterion, with explicit rejection counts.
For example, a 40-voxel segment fully overlapping a 128-voxel segment currently
fails the percentage gate despite sufficient absolute support. Endpoint ellipse
and direction gates can independently reject pairs as well.

**This change is planned, not implemented in this checkpoint.** The current
matcher and BP guard remain unchanged. Independent review confirms ambiguous
candidates must be alternatives, not simultaneous hard or independent soft
equalities. A choice-aware consumer or explicit topology selection is needed.
Reuse `tiles.json` and the saved `tileNNN-dedup.zarr` artifacts to audit matching;
no retracing, inference or deduplication is required for that audit.
