# Single-case merge/split benchmark

Root: `/ephemeral/hendrik/fiber3d_train2/evaluations/merge_split_case000_20260916_v1`.
Frozen paired annotations and diagnostic: `pair/`; regular vis/exports indexed by
root `README.md`. Primary scored case: corrected mini5v3 scroll00/case0,
PHerc0139/20260102150214@L0, original short200 piece000_08e44ead4019b9400317.

## What the supplied companion represents

User-supplied `corrected_fibers/evaltrace-fiberlet_20260916T154925562_000000.json`
is an edited vc3d annotation (generation5, multiple control points, extrapolated
tails), not simply the original exported failure segment. It no longer retains
the original export provenance. Raw SHA256:
`b1a078370d3ff51ae591d7deaafafa4dec1245c4f6c91bb8d0ab54b3ee4d5a4b`.

Native CP-bounded eligibility gives1324.2702base voxels for the companion and
1485.5264 for the primary. Extrapolated tails excluded; one eligible run each.
Existing accelerated nearest_polyline_points provides point-to-segment Euclidean
distances in base XYZ. No distances over artificial restart/gap connectors.

| Arc along companion (base vx) | Distance to primary (vx) | Distance to failed trace (vx) |
|---:|---:|---:|
|0.0|45.31|37.61|
|463.5|11.36|12.04|
|993.2|28.59|5.82|
|1191.8|54.63|3.97|
|1324.3|101.66|0.00|

Intermediate table values interpolate the stored-vertex distance profile.
Smallest sampled companion→primary distance10.987vx at companion arc447.964;
primary→companion10.991vx at primary arc543.737. At inference scale8 this is
approximately1.37prediction voxels. Distances are not ellipse error measures.

This supports a nearby path that approaches the primary and separates, with
the failed trace following the companion after separation. The source rendering
shows the terminal side-view divergence and broad prediction in the shared
region. It does NOT independently prove two physical fibers or quantify that
one has weaker presence. In particular the companion endpoint exactly equals
a point of the failed trace (trace arc1433.585), so that endpoint is not an
independent validation. Companion remains a user-supplied hypothesis, not GT.

## Reproduced baseline

Same bestEMA999u22000 model67edd3a1…, weighted4 fit,128base fiberlet chunks,
beam16/lookahead384/step48, ellipse40normal/80tangent,160grace,80restart grouping.
Native RelWithDebInfo. Existing scientific inference/construction cache identities
retained; no copy of full volumes or model change.

One case, one restart,3.56377776mm eligible distance; pooled mean1.78188888mm.
First segment failure at reference arc1464.3755base voxels. Replay object exactly
matches mini5v3 case0. Runtime8.33185s for one run including vis; optional pair
post-analysis follows this timer and is not included. No speedup claim.

Files:

- `pair/manifest.json`: both frozen annotations, hashes, model and coordinate identity.
- `pair/reference_plan.json`: native eligible geometry for both paths.
- `pair/geometry_profiles.json`: bidirectional GT/companion and trace/companion,
  plus trace/GT profiles, coordinates and nearest projected points.
- `scrolls/00/cases/0/rendered/strips_0.jpg`: regular CT/presence/reference/trace vis.
- `evaluation.json` and `provenance.json`: full settings and cache bindings.

Command:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_merge_split_case000_20260916_v1.json
```

## Next experimental objective (not implemented)

Jointly explain prediction support with multiple continuous paths; allow shared
support/multiple occupancy in the merged region, preserve entry-to-exit identity,
and penalize leaving a weaker branch unexplained. Simple exclusive voxel assignment
would incorrectly prohibit real overlapping evidence. Evaluate identity continuity
and support coverage in addition to existing GT restart metrics. This single
selected case is a development regression, not an unbiased generalization test.
