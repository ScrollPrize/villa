# Task Log: diagnose persistent wide-radius fiberlet replay failures

## Prior result carried into this task

The matched-terminal lookahead sweep is preserved in
`volume-cartographer/docs/fiberlets.md`. Its best complete radius-768 result was
the 384-base-voxel horizon, 192-base-voxel delay, terminal weight 0.25, and
profile blend 0.75: two failures at approximately 41,744 and 42,747 base-voxel
reference arc. Larger horizons did not reach the zero-or-one-failure target.

## Execution notes

- Independent review rejected treating radius 64 as ground truth and comparing
  radius-specific array indices. The experiment now requires stable endpoint-
  based directed identities, explicit topology classification, bounded error
  checks, and common decision context before making a causal cost claim.
- Radius is part of cache identity. A radius-64 run may require its own cold
  cache generation; it cannot be described as reusing the radius-768 graph.
- Offline reranking is limited to a fixed collected frontier. Raw subsegment
  profiles and checkpoint/search context are required, and any setting that
  changes candidate generation or future beam history still requires replay.

## Diagnostic scope implementation

- Added repeatable `fiberlet-replay --stats --decision-window BEGIN,END` filters.
  They do not change search, matching, restart state, or cache identity; they
  prevent retained-route materialization outside the requested reference-arc
  windows.
- The motivating unrestricted radius-64 diagnostic produced a 4.2 GB bundle,
  used 21,313,108 KiB peak RSS, and spent 59.47 seconds publishing. This is not
  a viable collection format for full fibers.
- The first radius-768 windowed run was stopped after three minutes at its first
  retained decision. Decision geometry was still reconstructed from the segment
  seed, making late-fiber diagnostics prefix-length dependent. Decision route
  points now cover only the checkpoint-to-lookahead suffix and record that
  suffix's path-length origin.

## Matched runs

- All comparisons used the Paris4 `fiber_s1_002` manifest, David reference
  fiber `dj_20260805T025256484_000003.json`, and `las_008` normals. The replay
  objective was beam 16, exact search, checkpoint 48 base voxels, lookahead
  384, integration step 16, profile blend 0.75, delay 192, and
  `W=0.99280572049126892` (terminal weight 0.25).
- The planned radius-64 baseline was rejected: it retained the radius-768
  failure at 41,744 and also failed at 48,159. Its unrestricted `--stats`
  artifact was 4.2 GB, used 21,313,108 KiB peak RSS, and spent 59.47 seconds
  publishing, which motivated the bounded decision windows.
- Radius 32 followed the reference through both target windows. Its only
  fiberlet event was below-threshold graph exhaustion at 50,753 near the
  endpoint. The full run took 29.11 seconds wall time, 472.07 seconds user,
  11.93 seconds system, and 927,040 KiB peak RSS. This is a constrained
  reference-following route, not ground truth for radius-768 graph topology.
- The windowed full-history radius-768 replay reproduced failures at
  41,744.2399792966 and 42,747.2975921847. It took 23.12 seconds wall time and
  2,001,892 KiB peak RSS. Its 40,000-43,300 diagnostic contained 67 decisions
  and occupied 196 MB. The matched radius-32 window contained 68 decisions and
  occupied 331 MB.

## Focused common-start diagnostics

- Added `fiberlet-replay --arc BEGIN` for both greedy and graph replay. Focused
  replays retain full-corridor cache identity and containment while scheduling
  the requested interval. This closes the pre-existing gap where the CLI help
  advertised `--arc` but only `quantization-benchmark` accepted it.
- Focus 1 used `[40500,42100]`: radius 32 had no fiberlet failure and radius
  768 reproduced 41,744. At reference arc 41,289 the radius-32-like
  continuation was wide-frontier rank 7. Selected versus constrained loss density was
  0.283476 versus 0.284211. The constrained alternative had better weighted
  edge loss (12.2647 versus 12.3026) and worse join loss (1.5219 versus
  1.3826). With diagnostic geometry clipped to the common checkpoint/lookahead
  interval, that alternative was 0.28 base voxels from the radius-32 route and
  the selected route was 7.08 base voxels away. A close candidate remained at
  the next checkpoints, but the closest top-16 candidate was 32.2 base voxels
  away by checkpoint 108. The selected and rank-7 routes have the same
  committed prefix at this checkpoint and differ only in lookahead, so this
  comparison does not identify the causal committed decision that later
  produces the failure.
- Focus 2 used `[41744.239979296603,43144.239979296603]`: radius 32 had no
  fiberlet failure and radius 768 reproduced 42,747. Both selected identical
  geometry through reference arc 42,404. At checkpoint 66 the selected wide
  route was 13.52 base voxels from the constrained continuation, and no closer
  route survived in the top 16.
- Repeating focus 2 with beam 128 found a marginally closer 13.11-base-voxel
  alternative only at rank 99. Its loss density was 0.235793 versus 0.231951
  selected. Its weighted edge loss was lower (7.4989 versus 7.5524), while its
  join loss was 0.7827 versus 0.2075. Beam 128 shifted the failure to 42,781
  but did not fix it.
- These comparisons are descriptive, not causal. Radius 32 changes extraction,
  NMS, and graph topology, and the first compared routes have not yet committed
  different prefixes. The second constrained proxy is absent from the top 128,
  but that does not establish that it is the minimum-cost admissible route in
  the radius-768 graph. The measured join-cost gaps therefore do not by
  themselves justify changing join weights. Fiberlet interiors and joins
  already use the same local direction, Lasagna-normal, tangential-turn, and
  normal-turn metric; a future continuation term would have to encode longer
  history rather than duplicate the local join metric.

## Threshold interpretation and deferred experiments

- Both recorded events are failures under the current ellipsoidal comparison,
  even though each component is below its individual radius. The first has
  normal/tangential errors `19.134/38.976` base voxels and ratio `1.0736`; the
  second has `10.618/68.473` and ratio `1.0072`. Replay tests
  `sqrt((normal/20)^2 + (tangential/80)^2) > 1`, not the independent box test
  `normal > 20 || tangential > 80`.
- The earlier 7.08- and 13.11-base-voxel values compare alternative replay
  routes with the radius-32 proxy. They are not route-to-reference errors and
  therefore do not determine failure status.
- The next rigorous comparison should find the minimum-cost route in the same
  radius-768 graph while constraining every evaluated point to the configured
  reference ellipsoid. That supplies a genuine admissible objective baseline
  without changing graph population or using radius as a proxy.
- A later evaluation may tolerate a temporary threshold excursion if the route
  rejoins within a configured base-arc distance. It should retain peak ratio,
  excursion length, integrated excess, and rejoin location so imperfect ground
  truth can be distinguished from an actual persistent fiber switch.

## Offline tuning assessment

- A reusable local collection should store stable directed logical IDs, raw
  decoded subsegment density profiles and offsets, edge/join components,
  checkpoint/prefix state, route geometry, rank/cutoff, and cache/profile
  fingerprints. Positive labels should be based on reference distance inside
  the focused interval; hard negatives are the best-scoring routes outside the
  correctness tube, including alternatives collected by excluding the positive
  branch.
- Such a collection supports fast rescoring of a fixed frontier for objective
  parameters. Radius, generation settings, beam/checkpoint/lookahead policy,
  state limits, and parameters that alter future beam history still require a
  focused replay. The focused hot-cache runs cost roughly 0.2 seconds at radius
  32 and 1.3-1.5 seconds at radius 768, so automated search can use focused
  replay without rerunning the complete fiber.
- The two diagnosed windows are training/tuning cases only. Generalization
  requires held-out failure windows or fibers.

## Deviation

- The plan named radius 64. It was replaced by radius 32 only after the
  measured radius-64 run failed in a target window. The replacement and its
  constrained-oracle limitation are reported above rather than silently
  treating the smaller graph as ground truth.

## Validation

- `cmake --build volume-cartographer/build --target vc_fiberlets test_fiber_trace3d test_fiberlet_paths test_fiber_replay test_fiberlet_storage -j32`
  succeeded.
- `test_fiber_trace3d`: 55 cases passed, including the selected reference-begin
  regression.
- `test_fiber_replay`: 12 cases passed.
- `test_fiberlet_storage`: 17 cases passed.
- `test_fiberlet_paths` still reports the same 298 pre-existing checks at
  lines 414 and 1026-1028; no failure references the new decision-window test.
- Independent review found that the first diagnostic implementation retained
  whole boundary fiberlets while labeling their geometry as checkpoint-to-
  lookahead. Route diagnostics now interpolate both interval boundaries. The
  regression checks those endpoints and confirms diagnostic filtering leaves
  selected points and failure counts unchanged. The focused measurements above
  were rerun after this correction.
- A final hot-cache focused radius-768 replay of `[40500,42100]` reproduced the
  41,744 failure in 1.26 seconds trace time and emitted no chunk-generation
  events, confirming that focused scheduling reopened the full-corridor cache
  without rewriting it.
- `git diff --check` passes.
- Decision-frontier diagnostics remain opt-in. Ordinary replay leaves
  `recordDecisionDiagnostics=false`; the CLI sets it only for explicit
  `--stats`, and `--decision-window` is rejected without `--stats`.
