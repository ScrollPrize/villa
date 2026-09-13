# Multi-anchor appearance experiments

Goal: improve the69-restart EMA512 reference result. Same frozen current model
EMA99 update6500/832000 patches, SHA256
`0a2651b711e7e5ee6d445ac8c68ff482c1951798ced2f44896c25602e1c50417`.
Same whole PHerc1299 held-out fiber `lt_20260825T075316065_000069.json`
(dataset aliasPHerc1229), both directions. No GT in reference updates; GT only
measures failure/restart using the unchanged ellipse. Training remains running.

## Implementation

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

## Reproduction / validation

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

## Reproduction commands

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

## Commands and environment

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
