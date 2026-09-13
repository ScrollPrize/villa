# Cached embedding analysis: GT versus baseline error routes

## Scope and reproduction

Canonical PHerc1299 (dataset label PHerc1229), held-out fiber
`lt_20260825T075316065_000069.json`; baseline96 failures (40 forward/56 reverse).
Same frozen raw update5400 /691200 samples as the off/on pilot.
SHA256: `7ea0f9c89e4bd4ce8c48dbae920b7ed24c810c178ab34576767e30b930057b35`.

Used **existing enabled-run cache** for all baseline and GT coordinates. No
inference, GPU allocation or training interruption. 3280 distinct
native queries, all covered and finite/nonzero; no missing/invalid queries.
Native query + analysis stage: 0.792s (not total command startup).
Final shared-helper verification repeated this in0.850s with bit-identical query
results (SHA256 4b48b3022ac41439e3463da74df99c8a9837bea729bbbf7b71e7bd8fc34da9f4).
Freshly decoded local signed float32 channels are component-interpolated then
normalized through the same C++ ChannelSampler used by tracing.

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment embedding-route-analysis
```

Settings: `embedding_route_analysis` in scripts/fiber3d_experiment.json.
Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/embedding_route_analysis_20260913_v2/`.
Final verification copy: `embedding_route_analysis_20260913_v3/` in the same parent.
`report.json` contains distributions/per-failure records;
`points.json` + `queries.json` retain every coordinate, vector and presence.
The native CPU-only command is
`vc_fiber_trace_metric sample-embeddings MANIFEST POINTS_JSON`.
It rejects missing interpolation-corner files before reads; no remote fetching.
Use a new output path for reruns. v1 remains an intermediate diagnostic; v2
corrects oracle history seed initialization and adds failure-distance contrasts.

## Short and long distance on the same GT fiber

Anchor grid32 base voxels; endpoints are queried at the exact requested offset
(including8/16, not rounded to the grid). All eligible pairs were covered.
Base voxel2.399um; input level1 halves these distances in input voxels.

| Separation (base voxels) | Pairs | Mean cosine | P10 | Median | P90 |
|---|---:|---:|---:|---:|---:|
| 8 | 466 | 0.944 | 0.867 | 0.981 | 0.998 |
| 16 | 465 | 0.948 | 0.866 | 0.979 | 0.997 |
| 32 | 465 | 0.926 | 0.806 | 0.967 | 0.993 |
| 64 | 464 | 0.878 | 0.695 | 0.943 | 0.986 |
| 128 | 462 | 0.795 | 0.478 | 0.882 | 0.979 |
| 256 | 458 | 0.691 | 0.268 | 0.801 | 0.968 |
| 512 | 450 | 0.651 | 0.214 | 0.788 | 0.965 |
| 1024 | 434 | 0.585 | 0.026 | 0.674 | 0.954 |
| 2048 | 402 | 0.512 | -0.004 | 0.587 | 0.921 |
| 4096 | 338 | 0.338 | -0.135 | 0.325 | 0.872 |

The representation is locally smoother than globally consistent. Even the
current256-base history window spans appreciable same-fiber embedding drift.
These are along-line distances, not Euclidean distances; one fiber only.
Differences across lags also include changing eligible endpoint sets.

## At baseline failure points: true versus taken location

Direct GT/taken cosine: mean0.657,
median0.755, P10/P90
0.199/0.950;
24.0% exceed0.9.

Each comparison below uses **identical references for both candidates**.
Gap = cosine(GT, reference) − cosine(taken, reference): positive favors GT.
Histories exclude the candidate/future and reset at each replay segment.
GT history is seed-initialized at32-base spacing, window256, at most8 refs.
Actual-route history reconstructs that spacing policy from the saved path;
it is not a replay of all alternative beams or a bit-exact capture of their
internal floating-point arclength state.

| Reference | Cases | GT similarity mean | Taken similarity mean | Mean gap | GT preferred |
|---|---:|---:|---:|---:|---:|
| gt_history | 95 | 0.804 | 0.680 | 0.124 | 71.6% |
| route_history | 96 | 0.738 | 0.755 | -0.017 | 54.2% |
| seed | 96 | 0.784 | 0.669 | 0.116 | 68.8% |

One event has no earlier GT arclength inside its reset segment, hence95 GT
histories versus96 route histories. All requested references were covered.
The GT/seed histories are oracle diagnostics; the deployed tracer does not know
GT after seeding. Its own route history offers little reliable separation here.

## Before the failure threshold

One observation per failure/direction/offset. Offsets follow **traced arclength**;
use the last saved point at/before that offset (up to roughly one8-base step
earlier). Points before the segment starts are omitted, not clamped or repeated.

| Distance before failure (base voxels) | Events | Direct GT/taken cosine mean | Actual-route history gap mean | GT preferred by route history |
|---|---:|---:|---:|---:|
| 128 | 30 | 0.890 | 0.028 | 70.0% |
| 64 | 52 | 0.824 | 0.026 | 55.8% |
| 32 | 79 | 0.748 | 0.001 | 49.4% |
| 0 | 96 | 0.657 | -0.017 | 54.2% |

This is not a matched longitudinal cohort across rows: short reset segments
drop out at larger offsets. Events/directions are correlated, not independent
statistical trials. Pre-failure path points may still be on the correct fiber.

## Failure discrimination versus reference distance

One GT reference exactly d base voxels behind the failure's matched GT location,
compared with both the GT and taken location. This deliberately follows GT even
across earlier replay resets: **oracle diagnostic**, not a usable path history.

| GT reference lag | Cases | GT cosine | Taken cosine | Mean gap | GT preferred |
|---|---:|---:|---:|---:|---:|
| 8 | 96 | 0.863 | 0.675 | 0.188 | 82.3% |
| 16 | 96 | 0.875 | 0.700 | 0.174 | 77.1% |
| 32 | 96 | 0.831 | 0.683 | 0.148 | 72.9% |
| 64 | 96 | 0.741 | 0.651 | 0.090 | 59.4% |
| 128 | 96 | 0.605 | 0.590 | 0.014 | 56.3% |
| 256 | 91 | 0.474 | 0.498 | -0.024 | 44.0% |
| 512 | 82 | 0.512 | 0.455 | 0.057 | 68.3% |
| 1024 | 80 | 0.383 | 0.435 | -0.051 | 38.8% |
| 2048 | 77 | 0.281 | 0.307 | -0.026 | 41.6% |

Short GT references discriminate better here; far references are unreliable.
Near references are also spatially closer to GT, so this alone does not prove
learned fiber identity beyond generic spatial embedding smoothness.

## Presence-conditioned failures

At the96 failure locations, mean presence is GT0.303
versus taken0.528; GT median
0.109. 70/96 GT points have
presence<0.5. Consequently these are not all clean high-confidence competing fibers.

| Subset (threshold0.5) | Events | Direct GT/taken cosine mean | Median | Fraction >0.9 | Route-history gap mean |
|---|---:|---:|---:|---:|---:|
| both_high | 22 | 0.860 | 0.931 | 59.1% | 0.025 |
| gt_low | 70 | 0.584 | 0.665 | 12.9% | -0.033 |
| taken_high | 55 | 0.591 | 0.658 | 25.5% | -0.038 |

For the22 both-high-presence cases,59% still have cosine>0.9: local competing
locations often remain similar. The baseline evaluator supplies its existing
normal-aware ellipse error ratios; no new Euclidean exclusion criterion is
substituted. A wrong route is not independently verified to be another physical
fiber. This analysis cannot assign the failure to annotation, localization,
presence or identity errors without inspecting imagery.

## Interpretation and limits

The cache contains a useful short-range continuity signal, not a stable global
fiber identifier. Current history can drift with the wrong route, while long
references can penalize valid GT changes. Together with weak GT presence and
remaining high-presence ambiguity, this is consistent with the failed off/on
pilot; it does not prove a unique cause or justify simply increasing the weight.

No tracer/training settings changed. Any shorter-history/anchor/contrastive
scoring experiment should be evaluated separately. Selected hard fiber, no
corpus-wide conclusion. No cache coverage bias in this specific query set.

Validation:100 Python tests passed42.41s and both native suites passed1.87s;
logs `cooperative-tests/20260913_150242_110872`. Native QuickBuild GCC13/O3/g1,
no installation. Tests cover causal history/common references, signed native
interpolation, missing chunks/out-of-bounds and present-but-zero vectors.
