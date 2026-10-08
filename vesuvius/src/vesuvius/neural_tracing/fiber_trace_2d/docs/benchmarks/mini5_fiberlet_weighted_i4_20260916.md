# Five-failure weighted-anchor fiberlet pilot

Completed 2026-09-16. Native fiberlet backend, same best EMA999 u22000 model
(`67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`).
Five fixed directed pieces,14.1571mm, selected before the new run: first two
failing manifest paths from PHerc0139, first from PHerc1299, first two Paris4.
This is a failure-selected pilot, not a representative quality estimate.

## Method

Four weighted-fit iterations, presence exponent1, direction exponent2, fixed
sigma2/local sigma1.5 prediction voxels, damping0.5. One anchor per owned cell,
no NMS. These are actually refined positions, not the center-only control.
All downstream fiberlet construction/search uses the shared native implementation.
Storage chunks128base, beam16, lookahead384base, step48base,224CPUworkers.
Unchanged40/80base ellipse,160base excursion grace,80base restart episodes.

Reused existing inference; logs show zero new tiles and published chunks.
Verified resolved four-iteration weighted processing metadata for each source,
distinct construction cache from legacy, identical inference manifest, and
fiberlet geometry backend in all five trace artifacts. Runner verified annotation,
checkpoint/normal hashes, five completed directed cases and total reference length.
No executable changes; previously validated build and native/Python tests reused.

## Results

| Metric | Historical legacy anchors | Weighted four iterations |
|---|---:|---:|
| Restarts | 5 | 5 |
| Failure-free pieces | 0/5 | 0/5 |
| Credited reference coverage | 100% | 100% |
| Reported mean distance, mm | 2.359517 | 2.359517 |
| Segment median, mm | 1.085337 | 1.089352 |
| Segment P05, mm | 0.092498 | 0.183807 |
| Segment P95, mm | 3.559222 | 3.488193 |
| New benchmark wall time | Not rerun | 43.9985 seconds |

Reported mean is combined directed length/(pooled restarts+1), not the arithmetic
mean of all segment lengths. Quantiles pool complete segments including terminal
ones. Their changes do not imply fewer failures; earlier failures can lengthen
terminal segments. All failures were distance_above_threshold; no seed misses or
graph exhaustion in either subset. Coverage includes successful continuation
after restarts, not failure-free completion.

| Short200 piece | Scroll | Restarts legacy → weighted | First failure arc legacy → weighted, base vx | Change, base vx |
|---|---|---:|---:|---:|
| 000 | PHerc0139 | 1 → 1 | 799.569 → 799.571 | +0.002 |
| 001 | PHerc0139 | 1 → 1 | 813.624 → 802.615 | -11.008 |
| 019 | PHerc1299 | 1 → 1 | 994.745 → 991.399 | -3.346 |
| 025 | PHercParis4 | 1 → 1 | 1714.985 → 1730.097 | +15.113 |
| 026 | PHercParis4 | 1 → 1 | 1199.484 → 1115.245 | -84.240 |

Positive arc change means the first failure occurred later along the same GT
piece. No failures were solved. Piece026 failed about0.202mm earlier;025 about
0.036mm later. Keep weighted fitting experimental; no default promotion.

## Artifacts and reproduction

### Saved regular beam comparison

Exact same directed pieces from completed short200_classic_20260916_v1 and
short200_best_u22000_20260916_v1; no new inference/evaluation for this comparison.
All have100% credited coverage including continuation after restart.

| Piece | Classic restarts | Best-u22000 beam restarts | Weighted fiberlet restarts | Classic first failure arc | Best first failure arc |
|---|---:|---:|---:|---:|---:|
| 000 | 1 | 1 | 1 | 878.62 | 831.59 |
| 001 | 1 | 1 | 1 | 839.32 | 785.18 |
| 019 | 1 | 1 | 1 | 963.51 | 1000.51 |
| 025 | 0 | 1 | 1 | — | 1712.19 |
| 026 | 0 | 1 | 1 | — | 1117.76 |

Arcs are directional base voxels from the piece start. All five also fail regular
best-snapshot beam tracing, so they are not fiberlet-only regressions. Classic
succeeds on both Paris4 pieces; these offer contrasting model/tracer outcomes.

### Regular inline visualization

Configuration:eval_mini5_fiberlet_weighted_i4_vis_20260916_v1.json, same approved
benchmark-full command. Separate output root:
`/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_fiberlet_weighted_i4_vis_20260916_v1`

Regular native eval renderer (CT/presence, reference/traced top/side views,
restart labels and aligned cross-sections), not a new custom rendering.
Images under scrolls/00/cases/0,1;scrolls/01/cases/0;scrolls/02/cases/0,1,
each in rendered/strips_0.jpg with render.json metadata.

Completed in25.688608seconds. Five images/metadata verified; all replay and
reference geometry exactly equal non-rendered pilot. Regular regression suite:
125native+263Python passed,8skipped. This faster cached rerun is not a comparison
of anchor algorithms. Cost-oracle implementation paused pending visual inspection.

Historical root:
`/ephemeral/hendrik/fiber3d_train2/evaluations/short200_fiberlet_u22000_128_scroll_20260916_v2`

New root:
`/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_fiberlet_weighted_i4_20260916_v1`

New cases: scrolls/00/cases/0,1 = pieces000,001; scrolls/01/cases/0 =019;
scrolls/02/cases/0,1 =025,026. Each has trace.json; no rendering requested.
metrics.json contains full new aggregate/scroll metrics. Locked subset manifest
`fiber_trace_3d/configs/mini5_fiberlet_failures_20260916_v1.json` records exact
annotation hashes, source mapping and historical trace paths.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_weighted_i4_20260916_v1.json
```

Completed output is immutable; for another run use a new output_root in the
configuration. Historical subset metrics were reconstructed with the existing
reference_segments/segment_statistics helpers, selecting cases by exact
backend_config.fibers[source_index] path. No matched legacy runtime measurement;
44seconds includes setup, cached input access, construction and tracing.
