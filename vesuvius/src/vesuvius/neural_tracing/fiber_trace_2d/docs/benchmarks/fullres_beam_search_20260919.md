# Beam-search parameter search on the full-resolution model (2026-09-19)

Step-by-step tuning of the native beam tracer on the five-scroll, 50-fiber
cohort (180 directed cases, 6,867.627253 mm) with the full-resolution best
EMA999 checkpoint (SHA-256
`5ec278f71b15b27187b55805310176fe2fa3b00c5264fc528b95bc6d0690206e`). The
search optimizes directly on this cohort by decision; there is no held-out
split, so the numbers are a tuning result, not an unbiased estimate. Ranking is
by restart episodes (`restart-episodes-80-base-voxels-v1`), never raw failures.

## Result

| Rank | Variant | Overrides on the baseline | Episodes | Raw failures | Improved / worsened directions | Mean mm | Median mm |
|---:|---|---|---:|---:|---|---:|---:|
| 1 | **cone35_beam16_step8** | cone 35°, beam 16, step 8 | **113** | 128 | 24 / 14 | 60.24 | 10.22 |
| 2 | triple_beam24 | cone 35°, beam 24, step 8 | 114 | 127 | 26 / 13 | 59.72 | 9.38 |
| 3 | triple_cone45 | cone 45°, beam 16, step 8 | 116 | 128 | 27 / 17 | 58.70 | 10.32 |
| 4 | step8_beam16 | beam 16, step 8 | 118 | 130 | 22 / 16 | 57.71 | 9.57 |
| 5 | cone35 | cone 35° | 121 | 165 | 10 / 5 | 56.29 | 8.92 |
| 5 | cone45 | cone 45° | 121 | 155 | 11 / 6 | 56.29 | 9.30 |
| 7 | beam16 | beam 16 | 123 | 169 | 9 / 6 | 55.38 | 9.34 |
| 8 | step8 | step 8 | 124 | 149 | 19 / 14 | 54.94 | 10.17 |
| 8 | cone35_beam16 | cone 35°, beam 16 | 124 | 159 | 12 / 9 | 54.94 | 8.90 |
| 8 | triple_step6 | cone 35°, beam 16, step 6 | 124 | 144 | 16 / 14 | 54.94 | 9.34 |
| 11 | cone15 | cone 15° | 125 | 171 | 14 / 15 | 54.50 | 8.85 |
| 12 | normal1 | normal weight 1 | 126 | 167 | 13 / 13 | 54.08 | 8.92 |
| 12 | beam32 | beam 32 | 126 | 172 | 9 / 8 | 54.08 | 9.10 |
| 12 | cone35_step8 | cone 35°, step 8 | 126 | 141 | 20 / 18 | 54.08 | 10.28 |
| 15 | **baseline** | beam 8, lookahead 2, step 4, cone 25°, normal 0.1, tangent 10 | **127** | 170 | — | 53.65 | 8.92 |
| 16 | pool_max_r1 | sample pooling max, radius 1 | 128 | 177 | 15 / 17 | 53.24 | 8.91 |
| 17 | cumulative0 | cumulative tangent weight 0 | 129 | 174 | 12 / 12 | 52.83 | 8.68 |
| 17 | pool_mean_r1 | sample pooling mean, radius 1 | 129 | 180 | 10 / 11 | 52.83 | 8.92 |
| 17 | pool_max_r2 | sample pooling max, radius 2 | 129 | 163 | 18 / 15 | 52.83 | 9.07 |
| 20 | normal0.01 | normal weight 0.01 | 130 | 159 | 9 / 11 | 52.42 | 8.92 |
| 21 | pool_mean_r2 | sample pooling mean, radius 2 | 131 | 163 | 14 / 18 | 52.03 | 8.68 |
| 21 | triple_step10 | cone 35°, beam 16, step 10 | 131 | 145 | 21 / 24 | 52.03 | 9.70 |
| 23 | tangent3 | tangent weight 3 | 136 | 172 | 10 / 19 | 50.13 | 8.52 |
| 23 | normal3 | normal weight 3 | 136 | 174 | 15 / 20 | 50.13 | 8.91 |
| 25 | cumulative8 | cumulative tangent weight 8 | 137 | 182 | 15 / 21 | 49.77 | 8.90 |
| 26 | beam4 | beam 4 | 138 | 189 | 11 / 20 | 49.41 | 8.87 |
| 27 | step12 | step 12 | 140 | 156 | 24 / 31 | 48.71 | 9.98 |
| 28 | step2 | step 2 | 167 | 225 | 16 / 39 | 40.88 | 8.15 |

Three earlier tangent-weight variants ran in a superseded root
(`fullres_beam_sweep_stage1_20260919_v2`, cache-only and then demand mode on
the recorded namespace): tangent 30 gave 136, tangent 100 gave 201 and tangent
300 gave 420 episodes. Raising the in-plane smoothness cost is monotonically
harmful; lowering it to 3 is also worse (136).

Per scroll, the winner improves or holds every scroll against the baseline:

| Scroll | Baseline | cone35_beam16_step8 |
|---|---:|---:|
| PHerc0139 | 64 | 55 |
| PHerc0332 | 4 | 2 |
| PHerc1229 | 9 | 6 |
| PHerc1451 | 0 | 0 |
| PHercParis4 | 50 | 50 |

## Reading the landscape

- Step, width and cone each help alone, but the pairs are not additive: cone
  35 with beam 16 gives 124 and cone 35 with step 8 gives 126, both worse than
  either alone, while all three together give 113. Beam 16 with step 8 (118)
  is the one pair that stacks.
- Step 8 is a sharp optimum inside the triple: 6 gives 124, 8 gives 113, 10
  gives 131. Alone, 2 gives 167, 4 gives 127, 8 gives 124, 12 gives 140.
- Width and cone are flat near the optimum: beam 24 (114) and cone 45 (116)
  in the triple are within noise of beam 16 and cone 35. Beam 32 alone is
  worse than 16; cone 45 alone equals cone 35.
- The smoothness weights (tangent, normal, cumulative) have no useful range
  away from the defaults; the normal weight is flat between 0.1 and 1.
- Sample pooling (a new tracer option, see below) is neutral to slightly
  harmful at radius 1 and 2 with max or mean pooling on a cross stencil. It
  does not reproduce the older quarter-resolution model's advantage.
- Lookahead 4 was dropped as infeasible: this tracer does not prune
  intermediate lookahead generations, so depth 4 expands roughly 19² more
  states per step than depth 2, reached 236 GB of anonymous memory and traced
  one direction per minute.

Against the recorded history, 113 closes half the gap between the full-res
baseline (127) and the best quarter-resolution model (refinement 212757 u22000,
99 episodes).

## Provenance

- Output root: `/ephemeral/hendrik/fiber3d_train2/evaluations/fullres_beam_search_20260919_v2`
  (`matrix.md`, `matrix.json`, `ranking.json`, `sweep_provenance.json`, one
  `attempt_NNN` per variant with `metrics.json` and `direction_episodes.json`).
- Config: `fiber_trace_3d/configs/eval_fullres_beam_search_20260919_v2.json`,
  run with `run_classic_fiber_evaluation.py cached-full`, resumed after each
  edit of the variant list.
- Cached evaluation: `fullres_refinement_213753_best_ema999_vis_20260917_v1/full`
  (its stale rendering/inference markers were waived explicitly).
- Prediction cache: an independent copy of the recorded namespace,
  `fullres_refinement_213753_best_ema999_20260917_v2_chunk64_beamsearch_copy_20260919`,
  extended by demand inference for every variant (a few hundred to about a
  thousand tiles each). The recorded namespace was not touched.
- Native binary: this checkout's `volume-cartographer/build/fiber-evaluation`
  (RelWithDebInfo, GCC 13). Rows up to `step12` ran with SHA-256
  `fbaf69b1cd5c6b489c31b3cab00f53e719fd957740fa81e79fe131034e9801db`; the
  pooling option was added mid-search and later rows ran with
  `f7a19434f2b46520…` (full hash in each variant's `result.json`). With
  pooling disabled the scoring path is unchanged; the baseline reproduced 127
  with both binaries.
- Runtime only, no effect on results: 32 replay workers, 64 GiB decoded
  cache, jemalloc preloaded into the tracer from the `cone35` row on (halves
  replay time), GPU lease idle release 3600 s.

Wall times per variant (5 to 13 minutes) include the GPU grant wait and are
not throughput measurements.

## Tracer changes made for this search

- `--sample-pool-radius-voxels`, `--sample-pool-mode`, `--sample-pool-stencil`
  in the native tracer (pooling of the prediction sample over a stencil).
- Bounded background prefetch in the live evaluator
  (optional `inference.background_chunk_cap`, unlimited by default), after a
  whole-tube prefetch stalled the accumulator.
- `gpu_manager.idle_release_seconds` for the demand GPU lease.
- Runtime-only `native_environment` for the native evaluator.
- The `beam_sweep` runner itself (`beam_parameter_sweep.py`).
