# Fiber-tracing benchmark runbook

## Crop winding: independent reference assignment

Use `--reference-independent` with the native joint-grid mixed BP evaluator to rank each reference from the frozen, reference-free initial solve. It rejects reference conditioning/pruning and non-joint-grid solvers. This is distinct from the historical conditioned-oracle/pruned-solve scores.

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_independent_reference_2048_20260923.json
```

Replace `2048` with `1024` for the smaller control. Saved source traces are reused; initial BP and reference cross-constraints are recomputed. No neural inference or source tracing. Outputs include CSV-formatted `independent_reference_summary` and `independent_reference_candidates` blocks in the native log. Candidate ordering minimizes hard-sign errors first, then configured weighted finite loss. Positive minimum errors means no hard-feasible candidate; alternative candidates are diagnostic violations, not a relaxation of the BP solve. Ties and unbounded tails are reported explicitly. Candidate grid is half-windings without a GT-parity filter. GT is used only for global sign/per-gauge offset alignment and evaluation, never to update/prune initial BP. Retain raw windings when comparing gauges.

This is the central index for the current fiber-tracing benchmarks and their
approved runner commands. Dated reports remain the authority for measured
numbers; this file identifies which evidence is current and how each workload is
run.

The commands below use the existing environment directly. Do not activate or
recreate a virtual environment, and do not prepend `AGENTS_AGENT_MODE=1` for a
manual run.

## Which benchmark to use

The Paris4 1024³ crop benchmark compares direct greedy replay, fiberlet replay,
and BP winding optimization on the same crop/new prediction. Its reproducible
configs and results are in
[`2026-09-23-crop-new-predictions.md`](../../../../../../volume-cartographer/docs/fiber_benchmark_runs/2026-09-23-crop-new-predictions.md).
Use the approved evaluation runner's `crop-benchmark --config ...` mode. BP
reference-conditioned diagnostics are not held-out tracing scores.

| Family | Status | Purpose | Current evidence |
|---|---|---|---|
| Five-scroll, 50-fiber evaluation | **Canonical scientific evaluation** | Broad model/tracer comparison on the held-out cohort | [full-resolution best EMA999](benchmarks/fullres_refinement_213753_best_ema999_20260917.md) |
| Beam parameter search on the cohort | **Tuning result, optimized on the cohort itself** | Step-by-step search of native beam parameters for the full-resolution model | [search result](benchmarks/fullres_beam_search_20260919.md) |
| Frozen short200 v3 | **Canonical targeted regression set** | Reproduce difficult restart regions cheaply; it is deliberately failure-enriched and is not an unbiased model-quality estimate | [dataset definition](short_failure_benchmark.md) |
| Short200 beam vs bounded fiberlets | **Canonical short-set comparison** | Compare tracing backends on exactly the same 200 directed pieces | [beam result](benchmarks/short200_fullres_refinement_213753_best_ema999_vis_20260918.md), [fiberlet comparison](benchmarks/short200_fullres_fiberlet_vs_beam_20260918.md) |
| Anchor-only short200 | **Current diagnostic** | Measure anchor placement/gaps without path construction or tracing | [method](anchor_coverage.md), [result](benchmarks/short200_anchor_coverage_u22000_20260916.md) |
| Corrected mini5 | **Fast diagnostic** | Iterate on anchor/fiberlet construction and search before short200 | [current corrected result](benchmarks/mini5_corrected_weighted_i4_20260916.md) |
| Explicit contrast cases | **Oracle/diagnostic only** | Compare a small number of known route choices in a narrow corridor | Case31 configs described below |
| Skeleton least-squares prototype | **Diagnostic only, tube-scoped** | Exercise the skeleton/NLLS/discrete-choice tracer on selected cases | [method](skeleton_lsq_tracing.md), [case31 result](benchmarks/contrast_case31_skeleton_20260918.md), [geometry](benchmarks/contrast_case31_skeleton_geometry_20260919.md) |
| Live-inference/cache profiles | **Performance diagnostics** | Isolate input, inference, publication, readiness and cached tracing throughput | [live evaluation architecture](live_inference_trace_evaluation.md), [cached trace profile](cached_trace_profile_20260911.md) |
| Older snapshot, embedding and fixed-path sweeps | **Historical** | Preserve model-selection and experimental history; do not use as the current headline baseline | Dated files under [`benchmarks/`](benchmarks/) |

## Training-integrated reviewed-failure regression

Training uses a locked 32-piece subset of the September 30 reviewed cohort for
both its ordinary dense test metrics and its native tracing regression. A piece
is included when either direction restarted in any of the canonical `lowres`,
`fullres`, `gaussian`, `halfsigma_raw`, or `halfsigma_ema99` evaluations. The
membership is 11 PHerc0139, 6 PHerc1299, and 15 PHercParis4 pieces (64 directed
traces). This is deliberately failure-enriched: it is useful for regression and
model selection, but it is not an unbiased estimate of generalization.

The persistent split is
`/home/hendrik/data/2026-10-01_reviewed_failure_union_v1`. Its manifest fixes
the exact filenames and SHA-256 values; training fails at startup if a file is
missing, added, or changed. Recreate a new version (never mutate this one) with
`vesuvius/scripts/materialize_reviewed_failure_test.py` and explicit canonical
`full/summary.json` paths.

The training configuration's `training.reviewed_trace_test` runs after the
ordinary test and current snapshot at its configured interval. It inherits the
training task's manager allocation rather than registering a nested task. All
DDP ranks poll a collective while the existing optimized live-inference/native
runner executes, preventing the NCCL watchdog from mistaking a long trace test
for a dead rank. Each step retains its workload, bindings, inference summary,
native summary, runner log, and compact metrics under
`RUN/reviewed_trace_test/step_NNNNNNNN/`.

TensorBoard group `reviewed_trace/` records actual restart episodes separately
for forward and reverse directions and in combination. The implied mean is the
corresponding total directed reference length divided by `(restarts + 1)`, so
zero combined restarts yields the sum of all directed reference pieces and one
restart yields half that sum. An incomplete direction, infrastructure failure,
or changed cohort aborts the test; there is no fallback to a different set.

The current benchmark model checkpoint is SHA-256
`5ec278f71b15b27187b55805310176fe2fa3b00c5264fc528b95bc6d0690206e`
for the full-resolution 20260916-213753 best EMA999 snapshot. The anchor and
mini5 diagnostics intentionally retain the older u22000 checkpoint SHA-256
`67edd3a1bde162c7105ee37d849a0f0b60a9b840dd37fc7ff6e17e0c3a2e9742`.

## Reproducibility and immutable evidence

Committed configs pin parameters and hashes, but many also name completed
`/ephemeral` or `/tmp` results, frozen checkpoints and cache namespaces on this
host. A repository checkout alone is therefore not sufficient to reproduce the
large-data runs elsewhere.

- Never delete, overwrite or repurpose a completed result root.
- Reissuing an exact command below is appropriate only when the runner mode is
  explicitly resumable/auditing (`benchmark-render`, `benchmark-review`, or a
  documented interrupted run).
- For a genuinely fresh measurement, copy the committed config to a new,
  versioned filename and assign a fresh `output_root`. Give inference a fresh
  namespace when measuring cold inference. Preserve the checkpoint SHA,
  benchmark manifest, scientific parameters and pilot contract.
- Reusing a prediction cache is valid for tracing-quality comparisons, but its
  wall time is not a cold inference benchmark. Reports must say which caches
  were reused.
- Manager-backed execution is the default encoded by the configs. Use
  `--unmanaged` only for an intentional unmanaged diagnostic.

The immutable short benchmark is
`/ephemeral/hendrik/fiber3d_train2/evaluations/short_failures_200_20260916_v3/manifest.json`:
200 directed pieces and 662.089848 mm. The later
`short_failures_200_gt_review_20260918_v1` directory is a **versioned annotation
audit draft** containing visually rejected cases. It does not alter or replace
the frozen v3 manifest or any result measured on it. Promote corrections only
as a new manifest, config and result series.

## Build and focused validation

Use the reusable native build and replay-test entrypoints:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests
```

The Python runner is:

```text
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py MODE --config ABSOLUTE_CONFIG
```

## 1. Canonical five-scroll/50-fiber evaluation

This is the workload for scientific model-quality claims. It contains 50 source
fibers across five scrolls/six collections and, after CP/Lasagna eligibility
splits, 180 directed cases covering 6,867.627253 mm. The current result has 127
filtered restart episodes.

- Config: [`eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json`](../../fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json)
- Recorded result: `/tmp/fullres_refinement_213753_best_ema999_off_20260917_v22_chunk64_file_results/full`
- Persistent prediction namespace: `fullres_refinement_213753_best_ema999_20260917_v2_chunk64`
- Detailed provenance/results: [fullres report](benchmarks/fullres_refinement_213753_best_ema999_20260917.md)

For a new ordinary full-cohort config, the lifecycle is pilot preparation,
validation and execution followed by full preparation, validation and execution:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare-pilot --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-pilot --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py pilot --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_off_20260917_v1.json
```

Those exact paths already contain evidence. For a fresh run, first copy the
config and change `output_root`; do not try to recreate or overwrite the existing
prepared files. A deliberately reused, successful pilot may be named by
`pilot_result`, but the full config must continue to pin that pilot contract.

An ordinary live run is complete only when `full/summary.json` says completed,
`full/inference_summary.json` exists, and neither `full/incomplete.json` nor
`full/inference_incomplete.json` exists. Counts and eligible distance must match
the prepared bindings. `summary.json` alone is not proof of inference completion.

For a result configured for rendering, resume saved visualization without
replaying completed traces via `resume-visualization`. The current visualization
config is
[`eval_fullres_refinement_213753_best_ema999_vis_20260917_v1.json`](../../fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_vis_20260917_v1.json):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py resume-visualization --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_refinement_213753_best_ema999_vis_20260917_v1.json
```

### Beam parameter sweep on the cached full-resolution cohort

The full-resolution best EMA999 prediction cache is immutable, and native beam
replay against it takes about two minutes per variant, so beam-search
parameters are tuned directly on the 180-direction cohort. A sweep config lists
`trace_args` overrides under `beam_sweep.variants`; the first variant must be
the unmodified baseline. Variants run through the `cached-full` runner mode.
With `beam_sweep.inference` set to `"demand"` each variant opens its own
manager-backed demand-inference session on the bound prediction namespace, as
the short benchmark does, because a variant that steps differently from the
baseline reaches chunks the demand-filled cache never held; the writer lock is
held only for that variant and a busy writer is waited for. With `"cached"` the
replay is CPU-only and a missing chunk fails the variant. Ranking uses restart
episodes, never raw failure counts, and each row records which directed cases
gained or lost an episode against the baseline.

- Config: [`eval_fullres_beam_search_20260919_v2.json`](../../fiber_trace_3d/configs/eval_fullres_beam_search_20260919_v2.json)
- Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/fullres_beam_search_20260919_v2`
  (`matrix.md`, `matrix.json`, `ranking.json`, one `attempt_NNN` per variant)
- Result: [search report](benchmarks/fullres_beam_search_20260919.md). Best
  setting cone 35°, beam 16, step 8: 113 episodes against the baseline's 127.
- The bound cached evaluation still carries stale rendering/inference markers
  from an interrupted visualization pass; the sweep waives them explicitly via
  `accept_incomplete_markers` because cached-only replay cannot pass a missing
  chunk silently.
- The search extends an independent copy of the prediction namespace,
  `fullres_refinement_213753_best_ema999_20260917_v2_chunk64_beamsearch_copy_20260919`,
  selected by `beam_sweep.cache_remap`, so that the recorded namespace stays
  free for other benchmarks (each demand session takes the cache writer lock).
  The copy holds the published channels, identity and model manifests only;
  legacy `.live_tmp` scratch and fiberlet construction data were not copied,
  so it is not usable for fiberlet replay.
- Each variant needs some fresh inference (a few hundred to about a thousand
  tiles) because it steps where the demand-filled cache never went. The
  session's GPU lease is kept for the whole variant with
  `gpu_manager.idle_release_seconds` = 3600; the default 30 s made a variant
  re-queue behind the training several times and stretched it to 40 minutes.
- `beam_sweep.reference_tube_base_voxels` can prefetch the whole acceptance
  tube around every reference in one session so later variants are CPU-only.
  It is off in this config: on this cohort the 250-voxel tube is about 164k
  chunks and 50k tiles, roughly two GPU hours, far more than the search's
  variants read in total. Set `inference.background_chunk_cap` (optional, unlimited by default) to
  feed the prefetch in bounded waves, because requesting the whole tube at
  once filled the accumulator hot arena and stalled inference.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa/vesuvius/scripts/run_classic_fiber_evaluation.py cached-full --config /home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_fullres_beam_search_20260919_v2.json
```

Rerunning the same command resumes: completed variants are validated against
their saved metrics and skipped, failed ones are retried. The variant list is
outside the sweep identity on purpose, so the search proceeds step by step:
edit the list after each result, rerun, and only the new variants execute.
The cache, checkpoint, trace protocol and inference mode are pinned by the
identity; the native binary hash is recorded per variant instead, because the
tracer gained options during the search (see below) and each row must name the
binary that produced it. Worker count, decoded-cache budget, GPU manager
settings and the native environment are runtime-only and also outside the
identity.

The tracer's `--sample-pool-radius-voxels R` (with `--sample-pool-mode`
max|mean and `--sample-pool-stencil` cross|cube) was added for this search: it
samples each candidate at a stencil around it and pools the prediction, which
mimics tracing a blurred or lower-resolution prediction without building a new
cache. It disables the fused corner-scoring path, so it costs roughly the
stencil size in extra samples per candidate.

## 2. Frozen short200 regression benchmark

The extraction is reproducible from locked historical evaluations with:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py short-failures --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short_failures_200_20260916_v3.json
```

See [the extraction protocol](short_failure_benchmark.md). Do not rerun extraction
to compare models: compare them against the same frozen manifest.

### Beam baseline

- Config: [`eval_short200_fullres_refinement_213753_best_ema999_vis_20260918_v1.json`](../../fiber_trace_3d/configs/eval_short200_fullres_refinement_213753_best_ema999_vis_20260918_v1.json)
- Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/short200_fullres_refinement_213753_best_ema999_vis_20260918_v1`
- Expected: 200 pieces, 662.089848 mm, 200 traces and 200 galleries

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_fullres_refinement_213753_best_ema999_vis_20260918_v1.json
```

### Bounded-fiberlet comparison

- Config: [`eval_short200_fullres_fiberlet_bounded_w64_213753_best_ema999_20260918_v4.json`](../../fiber_trace_3d/configs/eval_short200_fullres_fiberlet_bounded_w64_213753_best_ema999_20260918_v4.json)
- Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/short200_fullres_fiberlet_bounded_w64_213753_best_ema999_20260918_v4`
- Search: weighted four-iteration anchors; beam 16; step 48; lookahead 384;
  width 64; continue and report route-state-limit events

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_fullres_fiberlet_bounded_w64_213753_best_ema999_20260918_v4.json
```

Rendering and indexing saved short-benchmark traces are separate resumable steps;
they reuse trace/inference artifacts and do not rerun tracing:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-render --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_fullres_fiberlet_bounded_w64_213753_best_ema999_20260918_v4.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-review --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_fullres_fiberlet_bounded_w64_213753_best_ema999_20260918_v4.json
```

A short run is complete when all per-scroll summaries and aggregate
`full/summary.json`/`metrics.json` are complete, 200 unique manifest pieces are
represented, and no incomplete marker remains. A rendered run additionally
needs every declared page, `README.md` and `review_index.json`. Fiberlet reports
must include credited reference coverage and route-state-limit counts; a missing
seed or limit event must not be counted as a successful trace.

## 3. Anchor-only coverage

This diagnostic runs anchor extraction in the GT tube without fiberlet
construction or replay. It is not a tracer-quality score.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_anchor_coverage_u22000_20260916_v2.json
```

- Config: [`eval_short200_anchor_coverage_u22000_20260916_v2.json`](../../fiber_trace_3d/configs/eval_short200_anchor_coverage_u22000_20260916_v2.json)
- Output: `/ephemeral/hendrik/fiber3d_train2/evaluations/short200_anchor_coverage_u22000_20260916_v2`
- Expected: all 200 pieces; report accepted anchors, cell-normalized interior and
  endpoint gaps, empty bins/pieces and direction-qualified coverage.

The weighted-anchor parameter matrix is a separate, unpromoted experiment:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_anchor_weighted_fit_20260916_v1.json
```

See [weighted-fit method](anchor_weighted_fit.md) and
[matrix result](benchmarks/short200_anchor_weighted_fit_20260916.md).

## 4. Corrected mini5 and tuning sweeps

The corrected mini5 is the fast five-piece fiberlet diagnostic. It is useful for
rejecting parameter ideas, not for promoting model quality.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_corrected_weighted_i4_vis_20260916_v3.json
```

The current output is
`/ephemeral/hendrik/fiber3d_train2/evaluations/mini5_corrected_weighted_i4_vis_20260916_v3`.
All five cases and galleries must exist.

The graph-search sweep uses, in order:

- [`eval_mini5_fiberlet_graph_stage1_20260916_v1.json`](../../fiber_trace_3d/configs/eval_mini5_fiberlet_graph_stage1_20260916_v1.json)
- [`eval_mini5_fiberlet_graph_stage2_20260917_v1.json`](../../fiber_trace_3d/configs/eval_mini5_fiberlet_graph_stage2_20260917_v1.json)
- [`eval_mini5_fiberlet_graph_stage3_20260917_v1.json`](../../fiber_trace_3d/configs/eval_mini5_fiberlet_graph_stage3_20260917_v1.json)

The construction/DP sweep uses:

- [`eval_mini5_fiberlet_path_lattice_stage1_20260917_v1.json`](../../fiber_trace_3d/configs/eval_mini5_fiberlet_path_lattice_stage1_20260917_v1.json)
- [`eval_mini5_fiberlet_path_lattice_stage2_20260917_v1.json`](../../fiber_trace_3d/configs/eval_mini5_fiberlet_path_lattice_stage2_20260917_v1.json)
- [`eval_mini5_fiberlet_path_reach_stage3_20260917_v1.json`](../../fiber_trace_3d/configs/eval_mini5_fiberlet_path_reach_stage3_20260917_v1.json)
- [`eval_mini5_fiberlet_path_cost_stage4_20260917_v1.json`](../../fiber_trace_3d/configs/eval_mini5_fiberlet_path_cost_stage4_20260917_v1.json)

Each is invoked with the same `benchmark-full --config ABSOLUTE_PATH` form.
None displaced the production parameters; see the
[graph](benchmarks/fiberlet_graph_retuning_u22000_20260917.md) and
[construction/DP](benchmarks/fiberlet_construction_dp_retuning_u22000_20260917.md)
reports.

Exact sweep commands are:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_graph_stage1_20260916_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_graph_stage2_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_graph_stage3_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_path_lattice_stage1_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_path_lattice_stage2_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_path_reach_stage3_20260917_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_mini5_fiberlet_path_cost_stage4_20260917_v1.json
```

## 5. Explicit contrast and pair diagnostics

Case31 packages a GT route and a competing beam route for cheap controlled
fiberlet tests. Package it with:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py contrast-prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/contrast_prepare_case31_20260918_v1.json
```

Then use `benchmark-full` with one of these configs:

- [`eval_contrast_case31_fiberlet_scoped_20260918_v1.json`](../../fiber_trace_3d/configs/eval_contrast_case31_fiberlet_scoped_20260918_v1.json): corridor-scoped diagnostic.
- [`eval_contrast_case31_fiberlet_bounded_w64_20260918_v1.json`](../../fiber_trace_3d/configs/eval_contrast_case31_fiberlet_bounded_w64_20260918_v1.json): bounded production-style search.
- [`eval_contrast_case31_fiberlet_unrestricted_20260918_v1.json`](../../fiber_trace_3d/configs/eval_contrast_case31_fiberlet_unrestricted_20260918_v1.json): route-state-limit stress diagnostic.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_contrast_case31_fiberlet_scoped_20260918_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_contrast_case31_fiberlet_bounded_w64_20260918_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_contrast_case31_fiberlet_unrestricted_20260918_v1.json
```

The skeleton backend runs on the same packaged material. It is scoped to a
tube around the reference and any explicit choice, so its numbers are
diagnostic only and are not comparable to unscoped results. Its configs are
[`eval_contrast_case31_skeleton_20260918_v1.json`](../../fiber_trace_3d/configs/eval_contrast_case31_skeleton_20260918_v1.json),
its reference-only control
[`eval_contrast_case31_skeleton_unrestricted_20260918_v1.json`](../../fiber_trace_3d/configs/eval_contrast_case31_skeleton_unrestricted_20260918_v1.json),
and the two-case workload
[`eval_explicit_choices_skeleton_20260918_v1.json`](../../fiber_trace_3d/configs/eval_explicit_choices_skeleton_20260918_v1.json):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_contrast_case31_skeleton_20260918_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_contrast_case31_skeleton_unrestricted_20260918_v1.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_explicit_choices_skeleton_20260918_v1.json
```

To measure geometry rather than restarts, set `dump_crop_presence` in the
`skeleton` block. The run then writes `crop_presence.u8` beside the other case
artifacts, raw bytes in the crop's Z, Y, X order at the shape and origin
`skeleton.json` reports, and the fibre's cross-section can be measured offline
against `reference_refined.obj`. It costs about 200 MB per case, so it is left
off in the committed configs. The
[geometry report](benchmarks/contrast_case31_skeleton_geometry_20260919.md)
records what that measurement showed.

The two-case manifest is packaged by
[`contrast_prepare_explicit_choices_20260918_v2.json`](../../fiber_trace_3d/configs/contrast_prepare_explicit_choices_20260918_v2.json),
which adds merge/split case000 and its user-supplied companion fiber to case31:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py contrast-prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/contrast_prepare_explicit_choices_20260918_v2.json
```

These results answer whether the intended route is available/cheaper; they are
not population metrics. The older two-fiber merge/split topology diagnostic uses:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py pair-connectivity --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/merge_split_case000_20260916_v1.json
```

See [merge/split report](benchmarks/merge_split_case000_20260916.md).

`benchmark-remove-case` is a curation tool that creates versioned review
material. It is deliberately not listed as a benchmark run command.

## 6. Performance profiles

Use `profile`, not a scientific `full` result, to investigate throughput. The
current isolated configs are:

- [`eval_classic_collection_cached_batched_profile.json`](../../fiber_trace_3d/configs/eval_classic_collection_cached_batched_profile.json): batched cached-input/inference pipeline.
- [`eval_classic_collection_cached_trace_profile.json`](../../fiber_trace_3d/configs/eval_classic_collection_cached_trace_profile.json): cached native tracing.
- [`eval_classic_collection_readiness_profile.json`](../../fiber_trace_3d/configs/eval_classic_collection_readiness_profile.json): inference-readiness waits.

Example exact invocation:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection_cached_batched_profile.json
```

Record command, dataset, build type, cold/warm cache state, tile/batch counts,
wall time and mean plus p50/p95 (or min/median/max). Relevant historical profile
reports include [dynamic GPU batching](dynamic_inference_batching_20260911.md),
[full-set batching](full_dynamic_inference_batching_20260911.md),
[chunk-major accumulation](live_inference_chunk_major_20260910.md) and
[tracer waits](tracer_wait_profile_20260911.md).

## Helper modes, not benchmark families

## Full hard-region failure union

`failure-union` extracts a quota-free, deduplicated corpus from completed
evaluations.  The checked-in configuration combines the full-resolution,
Classic and best output-stride-4 reports, clips each failure to its eligible
reference span, adds 500 base-voxels of context, and audits the existing
full-resolution OME-Zarr cache within a 64-base-voxel corridor:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py failure-union --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/failure_union_full_20260921_v1.json
```

The output directory contains reference-only annotations under `fibers/`,
`manifest.json`, `cache_coverage.json`, and `vc3d_channels/` symlinks to the
published presence and normal channels.  A missing chunk is reported in the
coverage manifest rather than interpreted as an all-zero prediction.
The completed 2026-09-21 run is in
`/ephemeral/hendrik/fiber3d_train2/evaluations/full_failure_union_20260921_v2`
(190 pieces; 573.7638 mm; 8,421 of 11,124 required channel chunks present).

- `observe --pid PID` and `report-progress` inspect an active/saved run.
- `report-metrics` regenerates metric comparisons from saved traces.
- `reference-plan` audits eligible CP/non-Lasagna spans without evaluation.
- `freeze-snapshot` freezes and hashes a training snapshot.
- `cached-full` reruns against already published predictions with CUDA disabled.
- `fixed-path-extract`, `fixed-path-fit`, `fixed-path-report`,
  `integration-fit`, `integration-audit` and `integration-report` belong to the
  historical embedding-calibration work.

These modes may produce reports, but they do not define additional benchmark
populations. Historical snapshot/embedding reports under [`benchmarks/`](benchmarks/)
remain useful provenance and should not be deleted or relabeled as current.
