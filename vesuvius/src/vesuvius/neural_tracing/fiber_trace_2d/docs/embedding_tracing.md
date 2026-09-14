# Embedding-guided live tracing

Follow-up: [cached GT/error-route and short/long-range analysis](embedding_route_analysis_20260913.md).

Opt-in native beam cost; ordinary presence, direction and normal scoring stays
unchanged. The live evaluator reads `--embedding-weight` in `trace_args` and
requests embeddings from the same model forward when positive. Weight zero
retains the existing prediction products and makes no embedding reads.

## Scoring and storage

Experimental LRU aggregation `normalized-mean` uses cosine distance to the
normalized sum of anchors, rather than mean distance to individual anchors.
Ordinary mean equals `1-r*cos(query,mean_direction)`, where r is the length of
the mean vector; normalization removes this disagreement-based shrinkage.
Exact cancellation costs1. The cached sum is refreshed only on bank changes.
Trace profiling records candidate-weighted r deciles (last bucket1, numerical
overshoot clipped for binning). This may amplify near-cancelling directions;
native validation passed. Initial same-setting A/B worsened64/68/66 to75
restarts each on the frozen tuning fiber; independent refinement remains in
progress. No default change. See embedding_multi_anchor_20260913.md.

Experimental LRU `--embedding-anchor-touch hit` only refreshes an old anchor
when an observation exactly matches or has cosine distance below the novelty
threshold. Novel observations evict actual LRU without first touching their
nearest anchor. Default `nearest` retains the original forced-nearest touch.
This isolates replacement semantics without changing costs or memory. Native
tests pass; empirical comparison is running on the frozen model/cache.

`--embedding-reference-mode anchors` builds a compact observed appearance model.
The regular bounded history is its observation bank (<=64); greedy medoid
selection minimizes summed nearest cosine distance, taking up to
`--embedding-anchor-count` (1–8, default4). It favors frequently represented
appearances and different modes with coverage gains, not merely distant points.
This is greedy approximation, not an exact global optimum. Selection reruns only
when bank membership changes. Nearest-anchor cosine cost is mixed with EMA cost
using `--embedding-anchor-ema-mix` (0=anchors only,1=EMA only,default.5).
Bank, chosen indices and EMA are private to each beam and reset on restart.
Arclength sample count/spacing/window bound CPU work and memory; no GT is used.
Scoring uses the parent's bank, with expiry on child update (at most one-step
window grace), unlike candidate-time filtering of mean/best history modes.

Opt-in `--embedding-reference-mode` selects `mean` (unchanged default), `best`,
`ema`, `ema-seed`, `ema-gated`, or `ema-seed-gated`. `best` takes minimum
`1-cosine` over the bounded history and requires uniform falloff. EMA modes use
`--embedding-ema-half-life-base-voxels` (default512): for step distance D,
`d=2^(-D/half_life)`, new reference is `normalize(d*old+(1-d)*candidate)`.
Score uses the parent reference before update. Exact cancellation keeps old.
Seed modes average EMA and fixed restart-seed costs equally. Gated modes update
only if selected presence>=0.5 and cosine to parent reference>=0.5; scoring and
tracing still proceed when the gate rejects. Rejected distance never accumulates
into a later update. Memory is copied per beam and reset at every restart.
History settings do not affect EMA modes. These tracer-reference EMAs are
independent of the EMA99 model snapshot. Defaults and other costs are unchanged.

- Extra cost per candidate: `weight * mean(1 - cosine(candidate, reference))`.
  This uses the same per-step cost units as existing alignment/smoothness, not
  another step-length multiplier. Cosine is clamped to [-1,1]. The cost is
  nonnegative, preserving the beam search's lower-bound pruning.
- References belong to each beam hypothesis, begin with its inferred seed
  embedding, and are sampled along traced arclength. Defaults: spacing32,
  window256 base voxels, maximum8 references. Restarts begin a new history.
  GT is not consulted for scoring or history updates.
- Flags: `--embedding-weight`, `--embedding-history-spacing-base-voxels`,
  `--embedding-history-window-base-voxels`, `--embedding-history-count`.
- `FiberTrace3DPredictAdapter(emit_embeddings=True)` exposes signed float32
  `embedding_NNN` scalar channels, alongside unchanged uint8 predictions.
  Raw vectors use the existing overlap accumulator and coherent demand-cache
  publication. The cache identity includes embedding output format.
- Native `ChannelSampler` uses the existing decoded chunk cache and trilinear
  component interpolation, then unit-normalizes the interpolated vector.
  Direction/normal interpolation retains its existing encoding-aware fast path.
  Embeddings must share the exact prediction lattice and chunk layout, so the
  same demand-readiness gate covers their sibling chunks. Missing configuration
  or incompatible channels errors; zero/nonfinite vectors reject candidates
  (and error at the seed).

Float32 is intentional for this first validation. Compact float16 or normalized
int8 storage is a follow-up, requiring measured cosine/trace comparisons before
changing precision. This adds no new public Atlas artifact type.

## Manager and pilot

The existing command lease now optionally accepts a non-exclusive fixed CPU/VRAM
grant. Evaluator `gpu_manager` settings require explicit `cpu_cores` and
`vram_bytes_per_gpu` when `exclusive:false`. The command is nonpausable and
nonadaptive; it waits for its full grant, caps the Torch allocator with a 2GiB
context reserve, and releases ownership only after child cleanup. Protocol and
default exclusive behavior are unchanged. Adaptive training may briefly unload
and restart to resize; non-exclusive does not mean zero transition overhead.

Edit `embedding_trace_pilot` in `vesuvius/scripts/fiber3d_experiment.json`, then:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-trace-pilot
```

The pilot freezes current raw after admission, selects the shortest whole held-out
fiber of the configured source, and compares weight0 and0.25 bidirectionally.
An optional `fiber` path selects exactly one fiber already listed in that source's
prepared config, allowing difficult-case checks without changing the evaluator.
Both variants share the frozen checkpoint and settings but have fresh prediction
caches. Source caches are not purged. `resume:true` verifies the frozen hash,
keeps completed variants, and uses fresh paths for failed attempts. Set a new
output and `resume:false` for a new current snapshot. `report.json` records full
metrics and provenance; each variant also has `inference_summary.json`.

The pilot uses one GPU with a 20GiB grant, 16 logical CPUs, tile256, border32,
overlap96, batch1, bf16 forward and float32 accumulation. It is a functionality
and initial quality check, not a corpus-wide result or isolated speed benchmark.

## Validation

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests
```

Control suites: `embedding-native-build`, `embedding-native-tests`,
`embedding-tracer`. The native build is this checkout's
`volume-cartographer/build/dev-quickbuild-gcc`, not the villa2 build in `/tmp`.
Tests cover signed disk roundtrip, unchanged ordinary forward output, invalid
embeddings, interpolation-before-normalization, history isolation and cost,
weight-zero compatibility, shared demand-cache contracts and manager grants.

## Current-snapshot pilot, 2026-09-13

Snapshot: `combined_5scroll_s1a_contrastive_128_sd1_16x8_ema_20260913_121417`,
**current raw update5400 /691200 samples**, frozen after manager admission.
SHA256: `7ea0f9c89e4bd4ce8c48dbae920b7ed24c810c178ab34576767e30b930057b35`.
Fiber: PHerc0139 `lt_20260909T071741261_000094.json` from the held-out collection;
7.329 mm, run in both directions. This is not the older consistency snapshot.

| Measure | Embeddings off | Weight0.25 |
|---|---:|---:|
| Failures (two directions) | 1 | 1 |
| Mean segment length (mm) | 4.88577 | 4.88577 |
| Failures/directed mm | 0.06823 | 0.06823 |
| End-to-end variant wall time (s) | 32.01 | 37.91 |
| Inferred tiles / published chunks | 78 /20 | 82 /22 |
| GPU worker time per tile, mean (ms) | 208.59 | 212.28 |
| GPU worker time per tile, min/median/max (ms) | 205.64 /206.04 /376.43 | 208.60 /209.01 /396.89 |
| Peak Torch allocated /reserved (GiB, one GPU) | 4.74 /7.10 | 4.74 /7.10 |

The output trajectories differ; the failure-count-derived metric is unchanged.
No quality improvement is established by this one-fiber pilot. Timings are one
run each, include startup and unpurged source caches, and use a shared manager
grant rather than isolated GPUs. More tiles were requested by the changed path;
the 18% wall-time difference is not purely embedding compute overhead.

Full evidence:
`/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_trace_20260913_v2/report.json`.
Baseline: `weight-0`; enabled: `weight-0p25-retry-1789309548141295533`.
Each directory contains per-direction trace JSON/OBJ, native logs, summary and
inference timing/memory data. Failed attempts remain for inspection.

Final validation: 97 Python tests passed (31.80s), both native suites passed
(1.60s); logs in
`/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260913_142632_123641`.
No training/model/loss modifications and no full50-fiber sweep in this task.

## Hard-fiber follow-up, 2026-09-13

Selected `lt_20260825T075316065_000069.json`, canonical **PHerc1299** (the
prepared source/dataset label is `PHerc1229`). Historical best11k evaluation
had107 bidirectional failures. The prepared held-out geometry matches that
historical fiber exactly. The shorter35.725mm fiber was chosen over much longer
high-failure cases before seeing embedding results. No fallback case was needed.

Same frozen raw5400/691200 checkpoint and SHA as above, unchanged settings,
one non-exclusive20GiB GPU/16CPU grant for both runs, fresh prediction namespaces.
Historical107 only motivated selection; the newly measured off result is the
controlled baseline.

| Measure | Off | Weight0.25 |
|---|---:|---:|
| Forward failures | 40 | 56 |
| Reverse failures | 56 | 53 |
| Total failures | 96 | 109 |
| Mean segment length (mm) | 0.72909 | 0.64370 |
| Failures/directed mm | 1.34359 | 1.52553 |
| Variant wall time (s) | 93.21 | 143.30 |
| Inferred tiles | 335 | 349 |
| Worker time/tile, mean (ms, one GPU) | 223.09 | 299.19 |
| Worker time/tile, min/median/max (ms) | 205.57 /206.94 /432.81 | 209.66 /295.46 /667.32 |
| Peak Torch allocated/reserved (GiB, one GPU) | 4.74 /7.10 | 4.74 /7.10 |

**This setting worsened the selected hard case:**13.5% more failures and11.7%
shorter mean segment length. Reverse improved slightly, forward worsened.
This does not establish that embeddings cannot help: weight/history were not
tuned, and this is one deliberately selected hard fiber, not a corpus sweep.
No settings were adjusted after seeing the result. Source/OS caches were not
purged and training shared the node; timings are observed, not isolated overhead.

Evidence: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_trace_hard_20260913_v1/report.json`;
per-direction artifacts under `weight-0/cases` and `weight-0p25/cases`.
Command: the same approved `run_fiber3d_experiment embedding-trace-pilot` above.
Native QuickBuild unchanged. Explicit selection regression plus existing suites:
98 Python passed39.35s, two native suites passed1.88s, logs under
`cooperative-tests/20260913_143916_847400`. No install or training changes.
# History parameter sweep

Experimental `--embedding-reference-mode seed` scores cosine distance to the
fixed initial embedding only, with no update or expiry. It resets per trace.
This is a control for sparse histories that rarely admit a second observation,
and is distinct from `ema-seed`, which mixes an adaptive EMA with the seed.

Experimental `--embedding-reference-mode ema-delayed` uses the same normalized
EMA, but delays admission by `--embedding-ema-delay-base-voxels` (default0,
maximum2048). The seed is active immediately. Candidates score their parent's
reference; selected-child observations enter a private FIFO and update the EMA
only once the traced arclength exceeds their observation arclength by the delay.
Each update retains its original step distance, not the time/distance of release.
Zero delay calls the ordinary EMA update exactly. No GT or future observation
is consulted. This tests resistance to immediate wrong-turn reinforcement, at
the cost of slower adaptation to correct appearance changes; it is not a
confidence guarantee. Pending memory is capped at4096 observations per beam;
delay/step capacity is checked before tracing and unexpected overflow fails
loudly. Ordinary step8/delay32–256 needs4–32 pending vectors. The preflight
half-step estimate is conservative, not a guarantee at arbitrary float coordinate
magnitudes. No full-path queue or cross-beam sharing is introduced.

Experimental `--embedding-reference-mode appearance` corrects the windowed
medoid restriction: it maintains up to `--embedding-anchor-count` whole-path
spherical clusters, with FP64 vector sums and integer support. Every spaced
admission is included; at K+1 clusters, merge the pair with smallest increase
in spherical reconstruction error, `||a||+||b||-||a+b||`. Ties use stable order.
Normalized centroids (not observed medoids) are the anchors. True sums retain
all admitted contributions; no distance expiry or rolling bank. Exact cancellation
keeps the prior valid representative for scoring without changing the true sum.
Nearest-centroid cosine cost optionally mixes with EMA. Cluster support affects
compression, not score weighting. This is approximate streaming clustering;
bounded capacity can merge rare modes. State remains private per beam/restart.

Optional `--embedding-anchor-aggregation support-softmin` uses cluster support
fractions p and temperature T (`--embedding-anchor-temperature`, positive):
`min(c)-T*log(sum p*exp(-(c-min(c))/T))`. Unlike nearest matching, a rare matching
cluster carries less influence. Small T approaches nearest; large T approaches
support-weighted mean cost. Stable min-shift and expm1/log1p handle the limits.
This is an alternative hypothesis, not an assumption that rare appearances are
incorrect. EMA mixing still applies after this aggregation.


Experimental `--embedding-reference-mode anchors-lru` keeps up to
`--embedding-anchor-count` (1–8) observed vectors with no distance expiry.
`--embedding-history-spacing-base-voxels` limits admission frequency. At an
admitted selected point, touch its closest existing anchor, evict the least
recently used if full, then insert the point. Exact duplicates only touch and
advance the admission clock. Ties use insertion order. Rejected candidates do
not touch state; each beam owns a copy, reset at a new seed.
`--embedding-anchor-aggregation best|mean` chooses nearest or average cosine
distance, optionally mixed with EMA using `--embedding-anchor-ema-mix`.
The parent state scores candidates before admission. LRU ignores history window
and history count; its capacity is anchor count. Closest does not imply a
confident match. This mode is distinct from windowed medoid selection.

For LRU, `--embedding-anchor-novelty` (default0, cosine distance[0,2]) optionally
rejects insertion if the observation is already close to an anchor. It still
touches that closest anchor and advances the observation cadence. Existing
anchor vectors do not move. This reduces replacement churn; zero preserves
the ungated LRU policy.


See [the full history-spacing/falloff sweep](embedding_history_sweep_20260913.md)
for all36 combinations, controls and exact-repeat checks. Defaults remain unchanged.
