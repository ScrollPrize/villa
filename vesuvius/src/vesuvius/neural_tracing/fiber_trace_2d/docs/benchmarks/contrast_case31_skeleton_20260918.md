# Skeleton least-squares prototype on the explicit contrast choices (2026-09-18)

First measured result of the skeleton-topology tracing backend. Method and
design: [skeleton least-squares tracing](../skeleton_lsq_tracing.md).

This is a **diagnostic on one and then two hand-picked cases**, scoped to a
tube around the benchmark fibers. It is not a population metric and is not
comparable to unscoped full-cohort results.

## Shared bindings

All rows use the full-resolution best EMA999 prediction cache and checkpoint
SHA-256 `5ec278f71b15b27187b55805310176fe2fa3b00c5264fc528b95bc6d0690206e`,
through `cached_evaluation`
`/ephemeral/hendrik/fiber3d_train2/evaluations/fullres_refinement_213753_best_ema999_vis_20260917_v1/full`.
No inference tiles were computed: every run reported zero published tiles, and
every crop reported zero missing voxels, so the prediction cache already
covered both scoped regions. The stored prediction grid is at two base voxels
per prediction voxel. Evaluation is the unchanged protocol: normal radius 40,
tangential factor 2, excursion grace 160, restart episodes separated by 80
reference voxels.

## Case31 against the existing baselines

Piece `053_5de02b9d2fd3c9919f58` (PHerc0139, reverse), 3200 base voxels and
7.6768 mm.

| tracer | failures | failure reason | credited coverage |
|---|---:|---|---:|
| native beam | 1 | `distance_above_threshold` at reference arc 3093.6 | not reported for this case |
| bounded fiberlet width 64 | 0 | none | complete |
| scoped fiberlet radius 32 | 0 | none | complete |
| **skeleton, threshold 224** | **1** | `skeleton_terminal` | **100.0 %, 7.6768 mm** |

The distinction that matters is the failure *kind*. The beam failure is a
wrong-route error: it left the acceptance ellipsoid and followed a competing
route. The skeleton backend produced no `distance_above_threshold` event at
all on this case; its single failure is a coverage break, where the thresholded
skeleton ends and the route has nowhere to continue. Credited coverage is the
full eligible length.

Result roots:

- Scoped: `/ephemeral/hendrik/fiber3d_train2/evaluations/contrast_case31_skeleton_20260918_v2`
- Control: `/ephemeral/hendrik/fiber3d_train2/evaluations/contrast_case31_skeleton_unrestricted_20260918_v2`

The control uses `manifest_unrestricted.json`, so the crop is a tube around the
reference only, with no competing trace. It produced the same outcome, one
`skeleton_terminal` failure and 100.0 % credited coverage, from a smaller
network: 16 regions, 75 ports and 165 lines against 21, 91 and 194. The result
on this case is therefore not an artifact of admitting the competing route into
the scope.

## Threshold sweep

The presence threshold is one ordinary configuration parameter, swept through
configs rather than searched at runtime. All rows are case31 with the scoped
manifest and a port limit of 12.

| threshold | failures | credited coverage | skeleton components | regions | port-limit regions | model build |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8 | 82.8 % | 112 | 66 | 4 | 22.6 s |
| 96 | 6 | 90.6 % | 133 | 57 | 1 | 15.9 s |
| 128 | 5 | 91.1 % | 140 | 46 | 5 | 14.9 s |
| 160 | 1 | 97.8 % | 149 | 34 | 1 | 13.7 s |
| 192 | 3 | 97.0 % | 150 | 35 | 0 | 12.0 s |
| 208 | 3 | 97.0 % | 154 | 30 | 2 | 10.4 s |
| 224 | 1 | 100.0 % | 147 | 21 | 0 | 8.8 s |

A low threshold merges neighbouring fibers into thick blobs, which thinning
resolves into many junctions; the extra ambiguous regions are where the routing
loses the fiber. A high threshold keeps only fiber cores, leaving fewer and
cleaner junctions. Component count barely moves across the sweep, so the gain
comes from junction structure, not from connectivity. Threshold 224 is the
promoted default for these configs; it is tuned on one case and is not a
general recommendation.

## Two-case run

Config `eval_explicit_choices_skeleton_20260918_v1.json` over the two-case
manifest `explicit_contrast_choice_tests_20260918_v2`, which adds the
merge/split case000 piece `000_08e44ead4019b9400317` (1485.5 base voxels,
3.5638 mm) with the user-supplied companion fiber as its explicit choice.

Combined: 3 failures over 11.2406 mm eligible, 11.1254 mm traced, 98.98 %
credited coverage. Case31 contributed its single `skeleton_terminal` failure;
case000 contributed one `distance_above_threshold` and one
`skeleton_terminal`. For comparison, the earlier fiberlet merge/split run on
case000 recorded one restart.

Case000 is much more expensive: its crop covers 27.5 million scoped voxels
against 4.7 million for case31, because the companion fiber separates widely
from the reference and the tube must contain both. Model build took 252.7 s,
of which 146.8 s was graph construction, against 10.3 s total for case31. The
whole two-case run took 267.5 s.

Result root:
`/ephemeral/hendrik/fiber3d_train2/evaluations/explicit_choices_skeleton_20260918_v1`.

## Stage cost

Case31, threshold 224, one worker for the solver:

| stage | seconds |
|---|---:|
| threshold | 2.0 |
| thinning | 0.4 |
| graph and reduction | 3.1 |
| port optimization | 2.1 |
| line refinement | 0.8 |
| region choice | 0.5 |
| model total | 8.7 |
| run wall time | 11.7 |

Graph construction dominates and scales poorly with crop size, which is the
first thing to address if this method is taken beyond single cases.

## What this does not show

One and then two cases cannot separate method quality from case selection.
The threshold was chosen on the same case it is reported on. The tube scope is
defined by the ground-truth reference, so the topology the method sees is
ground-truth relative; the unrestricted control removes the competing trace
from that scope but not the reference itself. No short200 or full-cohort
number exists for this backend.
