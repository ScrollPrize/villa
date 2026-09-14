# Multi-anchor appearance experiments

Goal: improve the69-restart EMA512 reference result. Same frozen current model
EMA99 update6500/832000 patches, SHA256
`0a2651b711e7e5ee6d445ac8c68ff482c1951798ced2f44896c25602e1c50417`.
Same whole PHerc1299 held-out fiber `lt_20260825T075316065_000069.json`
(dataset aliasPHerc1229), both directions. No GT in reference updates; GT only
measures failure/restart using the unchanged ellipse. Training remains running.

## Implementation

### Completed independent refinements

All13 families completed,270 replays total, process exit0. Every family winner
and the overall winner reproduced its exact replay hash. Overall64 =29+35
directional restarts. Finite coordinate searches, not proof of global optima.

| Family | Restarts |
|---|---:|
| Novelty nearest-LRU |71|
| Whole-path nearest |72|
| Whole-path support-softmin |71|
| Plain nearest-LRU |66|
| Plain mean-LRU |68|
| Novelty mean-LRU |**64**|
| Windowed representative anchors |67|
| EMA |69|
| Gated EMA |75|
| EMA + seed |78|
| Gated EMA + seed |73|
| Best history |74|
| Mean history |70|

Further seed-only/delayedEMA/hit-only replacement experiments and multi-start
boundary refinements follow; the following paragraphs record discoveries made
during this completed round.

Confirmed global best of this round at `refine-0118`: **64 restarts**, novelty-gated mean-LRU
with2 anchors,512-base-voxel spacing,weight.10,novelty.10,noEMA. Wall36.82s;
replay hash `013d255be05b3b6fb295a14b4a9c92fffef2558b724179d00d8b8903ed1b81a0`.
This family's finite local search and exact repeat are complete. It improves on
EMA69 by5 restarts (7.2%) on this same tuning fiber; not a generalization result.

`embedding_ema99_all_refine_20260913_v1/report.json` records the finite axes and
all replays. At `refine-0072`, plain nearest-LRU improved to **66 restarts
(28+38)** versus EMA512/.25's69 (28+41). Settings:2 anchors,512-base-voxel
admission spacing,weight.10,noEMA mix,novelty0. Runtime41.98s, same frozen input
and QuickBuildO3 binary as above. This is a tuning result on one fiber, not a
generalization claim; this family's finite-axis refinement is complete, while
the final exact-parameter repeat remains open.
The exact replay hash is
`68d07241e4ac50456fc1f1ff52263c9374ecfb4e7dde451097037dfa783fe172`;
the equivalent-scoring half-life128 variant reproduced it exactly (EMA mix0).

The66-restart trace has68 segments,median112/mean469.41base voxels.53 have
traced length below512, but54.49% of total traced distance occurs beyond
that point. Sparse anchors therefore are not equivalent to seed-only by traced
distance; a separate seed-only control is still needed. These are path-length
statistics, not instrumentation of internal beam lookahead/admission events.

Completed families so far:novelty-LRU71,whole-path nearest72,support-softmin71,
plain nearest-LRU66 (three coordinate cycles).
Plain mean-LRU completed68 (K4/spacing512/.30/noEMA,two coordinate cycles).
Novelty mean-LRU completed64 at the settings above.
Other families and further mechanism tests are still running/planned. Do not
interpret this partial section as the final refinement report.

Frozen native build for this refinement:QuickBuild,optimization3;
`vc_fiber_trace_metric` SHA256
`4cadee53019df7a8cf2d96f640aba8b69b84adeb93b40c21529afb5e4f369470`,
shared `libvc_fiber_tracer.so` SHA256
`20197fb52e6a413ed82a6a6df296f217586caed3b2e9cf0164150990655bc96c`.
Pending seed/delayed-EMA source changes are not in this running binary.

Shared `FiberEmbedding.hpp` greedily chooses observed medoids to reduce summed
nearest cosine distance over the bounded observation bank. First anchor is most
central; subsequent anchors have maximum coverage gain, stable earliest ties,
stopping when no gain remains. This approximates the facility-location objective;
not globally optimal and not immune to outliers. At most64 bank samples/8anchors.
Parent bank scores candidate before selected-child updates; expiry on update gives
at most one-step window grace. Recompute only when membership changes. Each
beam owns independent history/indices/EMA; reset from valid seed at each restart.

Cost `(1-mix)*min_anchor(1-cosine) + mix*(1-cosine_to_EMA)`, then weight.25.
Model EMA99 is distinct from the reference EMA512. Defaults remain unchanged.

## First grid

| Bank / spacing / window | Anchors | Anchor only | Half EMA |
|---|---:|---:|---:|
|32 /128 /4096|2|76|83|
|32 /128 /4096|4|84|78|
|32 /128 /4096|8|76|84|
|64 /256 /16384|2|87|78|
|64 /256 /16384|4|78|82|
|64 /256 /16384|8|82|74|

Controls: off87, original mean82, EMA51269. All controls exact trajectories;
EMA repeat identical hash
`6e590851a9a00555dbc14edc3aa6f1fdccfc57a47b4a8c3c85a22ba900050484`.
None of these12 anchor settings beats EMA69. Best anchor configuration74.
The rolling-window restriction was not requested and does not fulfill the
whole-path appearance-model goal. These experiments are retained for comparison,
not presented as the final implementation.

## Refinement

EMA half-life256/384/640/768/1024:74/72/87/84/73 restarts.
Short-bank K2 mix.75/.9:91/76; K8 mix.75/.9:80/88.
Long-bank K8 mix.25/.75/.9:85/83/73. EMA512 remains69 with exact repeat.
Proceed to LRU, then no-expiry streaming appearance summaries.

LRU validation:2native suites passed1.92s,113Python tests44.66s;
logs `cooperative-tests/20260913_214250_448282`. Fixed capacity and admission
cadence, nearest touch, deterministic eviction and optional mean/EMA costs.

## Initial LRU grid

No age expiry. Weight.25; columns compare nearest anchor, equal-anchor mean,
and nearest mixed equally with EMA512. All12 combinations completed.

| Capacity | Admission spacing | Nearest | Mean | Nearest + half EMA |
|---|---:|---:|---:|---:|
|4|128|75|88|77|
|4|512|73|73|81|
|8|128|76|87|79|
|8|512|75|80|84|

Off87/original82/EMA69 exact; repeated EMA69 exact. Best LRU73, not an
improvement. Keep both nearest and mean seeds for refinement. Report:
`embedding_ema99_lru_20260913_v1/report.json` under evaluations root below.

## Initial whole-path grid

Spherical summaries preserve whole-path mass, no expiry. Weight.25; nearest
centroid alone or half EMA512. All12 variants complete, controls/repeat exact.

| Capacity | Admission spacing | Nearest | Half EMA |
|---|---:|---:|---:|
|2|128|81|78|
|2|512|72|81|
|4|128|78|83|
|4|512|75|84|
|8|128|84|84|
|8|512|76|84|

Best72 vs EMA69. Report `embedding_ema99_appearance_20260913_v1/report.json`.
Next: novelty-gated LRU and support-softmin appearance, then refine each family.
The initial whole-path implementation passed2native tests1.95s and115Python
tests45.24s, log `cooperative-tests/20260913_215750_020249`.

## Admission diagnostic

The best initial whole-path trace (72 failures,74 segments) has actual traced
lengths mean432.11, median144.00, max6088.02base voxels. At spacing512,61/74
segments end before the first post-seed observation; mean available observations
including seed1.55. At spacing128:36/74,mean3.84; at spacing32:1/74,mean14.07.
Computed by summing Euclidean edges in `trace_points_base_xyz`, not GT length.
Thus sparse initial admission grids barely exercise multi-anchor behavior on
most restarts. Dense8/32/64 admissions are prioritized for refinement.

## Mechanism grid

Same weight.25 and admission512. All24 replays completed, including controls
and exact EMA69 repeat. These are algorithmic changes, not just weight sweeps.

| Variant | K | Aggregation | Novelty | Temperature | Restarts | Wall s |
|---|---:|---|---:|---:|---:|---:|
|anchors-lru|4|best|0.02|—|74|40.38|
|anchors-lru|4|best|0.1|—|71|45.53|
|anchors-lru|4|best|0.3|—|75|42.63|
|anchors-lru|4|mean|0.02|—|79|38.25|
|anchors-lru|4|mean|0.1|—|75|44.64|
|anchors-lru|4|mean|0.3|—|81|44.14|
|anchors-lru|8|best|0.02|—|72|37.93|
|anchors-lru|8|best|0.1|—|71|45.86|
|anchors-lru|8|best|0.3|—|75|42.99|
|anchors-lru|8|mean|0.02|—|75|38.82|
|anchors-lru|8|mean|0.1|—|75|43.83|
|anchors-lru|8|mean|0.3|—|81|44.87|
|appearance|2|support-softmin|—|0.03|71|45.84|
|appearance|2|support-softmin|—|0.1|75|52.01|
|appearance|2|support-softmin|—|0.3|71|43.64|
|appearance|4|support-softmin|—|0.03|74|41.29|
|appearance|4|support-softmin|—|0.1|74|45.34|
|appearance|4|support-softmin|—|0.3|72|43.83|

24-replay wall mean40.57s,
min/median/p95/max3.54/42.81/45.86/52.01s.
Walls include variable path length and cache extension, not isolated method overhead.
Best novelty71 vs plainLRU73; supported appearance71 vs nearest72.
Report `embedding_ema99_mechanisms_20260913_v1/report.json`.

Independent13-family refinement follows, with8/32/64 admissions and per-family
weight/capacity/mix/history/temperature/novelty axes. Coarse coordinate cycles
also check .05 weight neighbors, strict improvements only; this establishes a
local tested result, not a global optimum. Each family is repeated separately.

## Reproduction

Approved commands (settings in `vesuvius/scripts/fiber3d_experiment.json`):

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-trace-sweep
```

Build: native GCC QuickBuildO3, standard C++ portable math, no precision changes.
Both native suites pass1.57s,113Python tests pass38.69s;
logs `/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260913_211352_827829`.
Independent plan/implementation review. GPU runs manager nonexclusive20GiB/16CPU;
cached inference reused, missing coverage generated under grant. Same
tile256/border32/overlap96/BF16 model/FP32 embeddings and other tracing parameters.

Reports under `/ephemeral/hendrik/fiber3d_train2/evaluations/`:

- `embedding_ema99_anchors_20260913_v1/report.json`: complete first grid and repeat.
- `embedding_ema99_refinement_20260913_v1/report.json`: finer mixtures/half-lives.

Each row retains exact arguments, trajectory hash, full per-direction summaries,
wall time and inferred-tile count. Walls include manager-co-running training,
startup and variable path length/cache extension; not isolated selection overhead.
Single-fiber in-sample tuning, not evidence of cross-scroll generalization.
